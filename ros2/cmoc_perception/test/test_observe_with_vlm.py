"""ROS action integration tests with a local HTTP fixture; no VLM credentials needed.

Run after building/sourcing: python3 -m unittest discover -s ros2/cmoc_perception/test -v
"""
from urllib.error import HTTPError, URLError
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from cmoc_interfaces.action import ObserveWithVLM

from cmoc_perception import observe_with_vlm_server as module


MUG_SCHEMA = {
    'type': 'object',
    'properties': {'found': {'type': 'boolean'}, 'description': {'type': 'string'}},
    'required': ['found', 'description'], 'additionalProperties': False,
}
MUG_RESPONSE = {'found': False, 'description': 'No coffee mug is visible.'}


class ObserveIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.status = 200
        cls.reply = None
        cls.delay = 0.0
        cls.received = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                cls.requests.append((self.path, body, self.headers.get('Authorization')))
                cls.received.set()
                time.sleep(cls.delay)
                self.send_response(cls.status)
                self.end_headers()
                reply = cls.reply if cls.reply is not None else (
                    {'message': {'content': json.dumps(MUG_RESPONSE)}} if self.path == '/api/chat'
                    else {'choices': [{'message': {'content': json.dumps(MUG_RESPONSE)}}]})
                try:
                    self.wfile.write(json.dumps(reply).encode())
                except BrokenPipeError:
                    pass  # Expected when testing client timeouts.

            def log_message(self, *_):
                pass

        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.http_thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.http_thread.start()
        cls.url = f'http://127.0.0.1:{cls.http.server_port}'
        rclpy.init()
        cls.server = module.ObserveWithVLMActionServer()
        cls.client_node = Node('observe_test_client')
        cls.client = ActionClient(cls.client_node, ObserveWithVLM, '/observe_with_vlm')
        cls.publisher = cls.client_node.create_publisher(Image, '/tiago/camera/color/image_raw', 10)
        cls.executor = MultiThreadedExecutor(num_threads=4)
        cls.executor.add_node(cls.server)
        cls.executor.add_node(cls.client_node)
        cls.thread = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.thread.start()
        assert cls.client.wait_for_server(timeout_sec=5)

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown()
        cls.thread.join()
        cls.client.destroy()
        cls.server._server.destroy()
        cls.server.destroy_node()
        cls.client_node.destroy_node()
        rclpy.shutdown()
        cls.http.shutdown()
        cls.http.server_close()
        cls.http_thread.join()

    def setUp(self):
        type(self).status = 200
        type(self).reply = None
        type(self).delay = 0.0
        self.received.clear()
        self.requests.clear()
        self.server.set_parameters([
            Parameter('backend', value='ollama'),
            Parameter('ollama_url', value=self.url),
            Parameter('nebula_url', value=self.url + '/api/chat/completions'),
            Parameter('nebula_api_key_env', value='OBSERVE_TEST_KEY'),
            Parameter('request_timeout_sec', value=2.0),
        ])
        with self.server._image_lock:
            self.server._latest_image = None

    def frame(self):
        image = Image(height=2, width=3, encoding='bgra8', step=12,
                      data=bytes([0, 0, 255, 255] * 6))
        deadline = time.monotonic() + 5
        while self.server._latest_image is None and time.monotonic() < deadline:
            self.publisher.publish(image)
            time.sleep(.05)
        self.assertIsNotNone(self.server._latest_image)

    def wait(self, future):
        deadline = time.monotonic() + 8
        while not future.done() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(future.done(), 'Action did not terminate')
        return future.result()

    def ask(self, prompt='Describe what you see.', schema=json.dumps(MUG_SCHEMA)):
        feedback = []
        goal = self.wait(self.client.send_goal_async(
            ObserveWithVLM.Goal(prompt=prompt, json_schema=schema),
            feedback_callback=lambda m: feedback.append(m.feedback.state)))
        self.assertTrue(goal.accepted)
        result = self.wait(goal.get_result_async())
        self.assertEqual(result.status, 4 if result.result.success else 6)
        return result.result, feedback

    def test_no_frame(self):
        result, _ = self.ask()
        self.assertFalse(result.success)
        self.assertEqual(result.response, 'No camera frame available.')
        self.assertFalse(self.requests)

    def test_ollama_and_nebula_payloads(self):
        self.frame()
        for backend in ('ollama', 'nebula'):
            self.server.set_parameters([Parameter('backend', value=backend)])
            with patch.dict(os.environ, {'OBSERVE_TEST_KEY': 'fixture-key'}):
                result, feedback = self.ask('Do you see a coffee mug?')
            self.assertTrue(result.success, result.response)
            self.assertEqual(json.loads(result.perception_provenance),
                             {'backend': backend, 'model': self.server.get_parameter(backend + '_model').value,
                              'fallback_used': False})
            self.assertEqual(json.loads(result.response), MUG_RESPONSE)
            self.assertIn('running_vlm', feedback)
            path, body, auth = self.requests[-1]
            self.assertFalse(body['stream'])
            self.assertEqual(len(body['messages']), 1)
            message = body['messages'][0]
            if backend == 'ollama':
                self.assertEqual(path, '/api/chat')
                self.assertEqual(body['format'], MUG_SCHEMA)
                self.assertEqual(body['options']['temperature'], 0)
                self.assertEqual(message['content'], 'Do you see a coffee mug?')
                self.assertEqual(len(message['images']), 1)
                image = message['images'][0]
                self.assertIsNone(auth)
            else:
                self.assertEqual(auth, 'Bearer fixture-key')
                self.assertEqual(body['response_format'], {'type': 'json_schema',
                    'json_schema': {'name': 'observation', 'strict': True, 'schema': MUG_SCHEMA}})
                self.assertEqual(message['content'][0]['text'], 'Do you see a coffee mug?')
                image = message['content'][1]['image_url']['url'].split(',', 1)[1]
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape, (2, 3, 3))
            self.assertGreater(int(decoded[0, 0, 2]), 250)

    def test_unavailable_backends(self):
        self.frame()
        for backend in ('ollama', 'nebula'):
            self.server.set_parameters([Parameter('backend', value=backend),
                                       Parameter(backend + '_url', value='http://127.0.0.1:1')])
            with patch.dict(os.environ, {'OBSERVE_TEST_KEY': 'fixture-key'}):
                result, _ = self.ask()
            self.assertFalse(result.success)
            self.assertIn('unavailable', result.response)

    def test_missing_key(self):
        self.frame()
        self.server.set_parameters([Parameter('backend', value='nebula')])
        with patch.dict(os.environ, {}, clear=True):
            result, _ = self.ask()
        self.assertTrue(result.success, result.response)
        self.assertEqual(json.loads(result.perception_provenance),
                         {'backend': 'ollama', 'model': 'qwen3-vl:2b', 'fallback_used': True})
        self.assertEqual(self.requests[0][0], '/api/chat')

    def test_http_and_invalid_response(self):
        self.frame()
        type(self).status = 503
        result, _ = self.ask()
        self.assertFalse(result.success)
        self.assertIn('HTTP 503', result.response)
        type(self).status = 200
        for reply in ({}, {'message': {'content': ''}}):
            type(self).reply = reply
            result, _ = self.ask()
            self.assertFalse(result.success)

    def test_invalid_backend(self):
        self.frame()
        self.server.set_parameters([Parameter('backend', value='invalid')])
        result, _ = self.ask()
        self.assertFalse(result.success)
        self.assertIn('Unsupported backend', result.response)

    def test_snapshot_and_continuous_subscription(self):
        self.frame()
        type(self).delay = .5
        goal = self.wait(self.client.send_goal_async(ObserveWithVLM.Goal(prompt='snapshot', json_schema=json.dumps(MUG_SCHEMA))))
        self.assertTrue(self.received.wait(5))
        other = self.wait(self.client.send_goal_async(ObserveWithVLM.Goal(prompt='busy')))
        self.assertFalse(other.accepted)
        with self.server._image_lock:
            old = self.server._latest_image
        # A new frame arrives during HTTP, but the request retains the red snapshot.
        blue = Image(height=2, width=3, encoding='bgra8', step=12,
                     data=bytes([255, 0, 0, 255] * 6))
        deadline = time.monotonic() + 3
        while self.server._latest_image is old and time.monotonic() < deadline:
            self.publisher.publish(blue)
            time.sleep(.02)
        self.assertIsNot(self.server._latest_image, old)
        result = self.wait(goal.get_result_async())
        self.assertTrue(result.result.success)
        self.assertEqual(len(self.requests), 1)
        encoded = self.requests[0][1]['messages'][0]['images'][0]
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(int(decoded[0, 0, 2]), 250)

    def test_timeout(self):
        self.frame()
        type(self).delay = .5
        self.server.set_parameters([Parameter('request_timeout_sec', value=.1)])
        result, _ = self.ask()
        self.assertFalse(result.success)
        self.assertIn('timed out', result.response)

    def test_nebula_and_ollama_read_timeouts_each_try_twice(self):
        self.frame()
        type(self).delay = .5
        self.server.set_parameters([Parameter('backend', value='nebula'),
                                   Parameter('request_timeout_sec', value=.1)])
        with patch.dict(os.environ, {'OBSERVE_TEST_KEY': 'fixture-key'}):
            result, _ = self.ask()
        self.assertFalse(result.success)
        self.assertEqual([request[0] for request in self.requests], ['/api/chat/completions', '/api/chat/completions', '/api/chat', '/api/chat'])
        self.assertIn('Both perception backends failed', result.response)

    def test_observation_client_cleanup_after_perception_failure(self):
        from observation import observe_scene_with_vlm

        self.frame()
        type(self).status = 400
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
                observe_scene_with_vlm(timeout=5)
        # Subsequent application spins must not execute callbacks for the
        # destroyed observation clients on the global executor.
        probe = Node('observation_cleanup_probe')
        try:
            rclpy.spin_once(probe, timeout_sec=0.01)
        finally:
            probe.destroy_node()

    def test_fallback_reuses_frame_schema_and_returns_local_result(self):
        for failure in (RuntimeError('read timeout'), RuntimeError('connection refused'),
                        RuntimeError('HTTP 503'), ValueError('API key is missing')):
            with self.subTest(failure=str(failure)), \
                    patch.object(self.server, '_ask_nebula', side_effect=failure) as remote, \
                    patch.object(self.server, '_ask_ollama', return_value='local observation') as local:
                result, provenance = self.server._observe('nebula', 'prompt', 'same-image', MUG_SCHEMA)
                self.assertEqual(result, 'local observation')
                self.assertEqual(provenance, {'backend': 'ollama', 'model': 'qwen3-vl:2b', 'fallback_used': True})
                remote.assert_called_once_with('prompt', 'same-image', MUG_SCHEMA)
                local.assert_called_once_with('prompt', 'same-image', MUG_SCHEMA)
        with patch.object(self.server, '_ask_nebula', return_value='remote') as remote, \
                patch.object(self.server, '_ask_ollama') as local:
            result, provenance = self.server._observe('nebula', 'prompt', 'image', MUG_SCHEMA)
            self.assertEqual(result, 'remote')
            self.assertFalse(provenance['fallback_used'])
            local.assert_not_called()

    def test_exact_two_attempts_per_backend_success_or_failure(self):
        def failure(cause):
            error = RuntimeError('transport failure')
            error.__cause__ = cause
            return error

        causes = [TimeoutError('read timeout'), URLError(ConnectionRefusedError('refused')),
                  HTTPError('http://local', 503, 'unavailable', {}, None)]
        for cause in causes:
            for local_succeeds in (True, False):
                with self.subTest(cause=type(cause).__name__, local_succeeds=local_succeeds):
                    order = []
                    def remote(*args):
                        order.append('nebula')
                        raise failure(cause)
                    def local(*args):
                        order.append('ollama')
                        if local_succeeds and order.count('ollama') == 2:
                            return 'local result'
                        raise failure(cause)
                    with patch.object(self.server, '_ask_nebula', side_effect=remote), \
                            patch.object(self.server, '_ask_ollama', side_effect=local):
                        if local_succeeds:
                            result, provenance = self.server._observe('nebula', 'prompt', 'image', MUG_SCHEMA)
                            self.assertEqual(result, 'local result')
                            self.assertEqual(provenance, {'backend': 'ollama', 'model': 'qwen3-vl:2b', 'fallback_used': True})
                        else:
                            with self.assertRaisesRegex(RuntimeError, 'Both perception backends failed'):
                                self.server._observe('nebula', 'prompt', 'image', MUG_SCHEMA)
                    self.assertEqual(order, ['nebula', 'nebula', 'ollama', 'ollama'])

    def test_nontransient_http_errors_are_not_retried(self):
        error = RuntimeError('HTTP 401')
        error.__cause__ = HTTPError('http://local', 401, 'unauthorized', {}, None)
        with patch.object(self.server, '_ask_nebula', side_effect=error) as remote, \
                patch.object(self.server, '_ask_ollama', side_effect=error) as local:
            with self.assertRaisesRegex(RuntimeError, 'Both perception backends failed'):
                self.server._observe('nebula', 'prompt', 'image', MUG_SCHEMA)
        remote.assert_called_once()
        local.assert_called_once()

    def test_client_provenance_envelope_after_fallback(self):
        from observation import observe_scene_with_vlm
        self.frame()
        self.server.set_parameters([Parameter('backend', value='nebula')])
        scene = {'objects': [], 'relations': []}
        type(self).reply = {'message': {'content': json.dumps(scene)}}
        with patch.dict(os.environ, {}, clear=True):
            result = observe_scene_with_vlm(with_provenance=True, timeout=5)
        self.assertEqual(result['scene_graph'], scene)
        self.assertEqual(result['perception_provenance'],
                         {'backend': 'ollama', 'model': 'qwen3-vl:2b', 'fallback_used': True})

    def test_invalid_requested_schema_before_camera_or_http(self):
        for schema in ('{', '{"type": "not_a_type"}', '{"required": "found"}',
                       'null', '{"minimum": NaN}', '{"$schema": "unknown-dialect"}'):
            with self.subTest(schema=schema):
                result, _ = self.ask(schema=schema)
                self.assertFalse(result.success)
                self.assertIn('Invalid requested JSON schema:', result.response)
                self.assertFalse(self.requests)

    def test_malformed_output_is_not_repaired(self):
        self.frame()
        for backend in ('ollama', 'nebula'):
            self.server.set_parameters([Parameter('backend', value=backend)])
            for output in ('not JSON', '```json\n{"found": false}\n```',
                           '{"found": false,}', '{"found": NaN}', 'null trailing'):
                type(self).reply = ({'message': {'content': output}} if backend == 'ollama'
                                    else {'choices': [{'message': {'content': output}}]})
                with patch.dict(os.environ, {'OBSERVE_TEST_KEY': 'fixture-key'}):
                    result, _ = self.ask()
                self.assertFalse(result.success)
                self.assertIn('VLM returned invalid JSON:', result.response)

    def test_schema_mismatch(self):
        self.frame()
        for output in ({'found': 'false', 'description': 'test'}, {'found': False},
                       dict(MUG_RESPONSE, extra=1), ['wrong type']):
            type(self).reply = {'message': {'content': json.dumps(output)}}
            result, _ = self.ask()
            self.assertFalse(result.success)
            self.assertIn('VLM response failed schema validation', result.response)

    def test_different_caller_schemas(self):
        self.frame()
        for schema, output in (
            ({'type': 'array', 'items': {'type': 'integer'}}, [1, 2]),
            ({'$schema': 'http://json-schema.org/draft-07/schema#',
              'type': 'string', 'enum': ['room']}, 'room'),
            (True, {'anything': 'allowed'}),
            ({'$defs': {'value': {'type': 'integer'}}, '$ref': '#/$defs/value'}, 3),
        ):
            type(self).reply = {'message': {'content': json.dumps(output)}}
            result, _ = self.ask(schema=json.dumps(schema))
            self.assertTrue(result.success, result.response)
            self.assertEqual(json.loads(result.response), output)
            self.assertEqual(self.requests[-1][1]['format'], schema)
