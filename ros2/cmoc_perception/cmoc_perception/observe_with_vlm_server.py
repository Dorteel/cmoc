#!/usr/bin/env python3
"""Ask a selected VLM about one snapshot of TIAGo's latest RGB frame."""

import base64
import json
import os
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
from jsonschema import Draft202012Validator, SchemaError, ValidationError
from jsonschema.validators import validator_for
from cv_bridge import CvBridge
import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cmoc_interfaces.action import ObserveWithVLM


def _strict_json(text):
    # Python otherwise accepts NaN/Infinity, which are not valid JSON values.
    def reject_constant(value):
        raise ValueError(f'{value} is not a valid JSON value')
    return json.loads(text, parse_constant=reject_constant)


class ObserveWithVLMActionServer(Node):
    def __init__(self):
        super().__init__('observe_with_vlm_server')
        for name, default in {
            'backend': 'ollama',
            'camera_topic': '/tiago/camera/color/image_raw',
            'ollama_url': 'http://localhost:11434',
            'ollama_model': 'qwen3-vl:2b',
            'nebula_url': 'https://nebula.cs.vu.nl/api/chat/completions',
            'nebula_model': 'SURF.Qwen3.5 122B A10B NVFP4',
            'nebula_api_key_env': 'NEBULA_API_KEY',
            'request_timeout_sec': 120.0,
        }.items():
            self.declare_parameter(name, default)
        self._latest_image = None
        self._image_lock = Lock()
        self._goal_lock = Lock()
        self._bridge = CvBridge()
        # Keep receiving frames while the action's separate callback group waits
        # for HTTP. Only the latest message is retained; no per-goal subscription.
        self._camera_callbacks = MutuallyExclusiveCallbackGroup()
        self._subscription = self.create_subscription(
            Image, self.get_parameter('camera_topic').value, self._cache_image,
            qos_profile_sensor_data, callback_group=self._camera_callbacks)
        self._server = ActionServer(
            self, ObserveWithVLM, '/observe_with_vlm', self.execute,
            goal_callback=self._accept_goal)

    def _accept_goal(self, _request):
        # One inference at a time keeps an executor thread free for the camera.
        if self._goal_lock.acquire(blocking=False):
            return GoalResponse.ACCEPT
        return GoalResponse.REJECT

    def _cache_image(self, image):
        with self._image_lock:
            self._latest_image = image

    def _post(self, url, payload, api_key=None):
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        timeout = float(self.get_parameter('request_timeout_sec').value)
        if timeout <= 0:
            raise ValueError('request_timeout_sec must be positive.')
        request = Request(url, data=json.dumps(payload).encode('utf-8'),
                          headers=headers, method='POST')
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            # Do not echo remote bodies or request headers: they may contain secrets.
            raise RuntimeError(f'VLM endpoint returned HTTP {error.code}.') from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError('VLM endpoint unavailable or request timed out.') from None

    def _ask_ollama(self, prompt, image_b64, schema):
        result = self._post(
            self.get_parameter('ollama_url').value.rstrip('/') + '/api/chat',
            {'model': self.get_parameter('ollama_model').value,
             'format': schema,
             'options': {'temperature': 0},
             'stream': False,
             'messages': [{'role': 'user', 'content': prompt, 'images': [image_b64]}]})
        return result['message']['content']

    def _ask_nebula(self, prompt, image_b64, schema):
        api_key = os.environ.get(self.get_parameter('nebula_api_key_env').value)
        if not api_key:
            raise ValueError('Nebula API key is missing; set the environment variable '
                             'named by nebula_api_key_env before starting the server.')
        result = self._post(
            self.get_parameter('nebula_url').value,
            {'model': self.get_parameter('nebula_model').value,
             # Native schema support verified on the configured Nebula model.
             # Always validate locally too; backend guarantees are not assumed.
             'response_format': {'type': 'json_schema', 'json_schema': {
                 'name': 'observation', 'strict': True, 'schema': schema}},
             'stream': False,
             'messages': [{'role': 'user', 'content': [
                 {'type': 'text', 'text': prompt},
                 {'type': 'image_url', 'image_url': {
                     'url': 'data:image/jpeg;base64,' + image_b64}},
             ]}]}, api_key=api_key)
        return result['choices'][0]['message']['content']

    def execute(self, goal_handle):
        result = ObserveWithVLM.Result()
        logger = self.get_logger()
        logger.info('Observation requested')
        try:
            backend = self.get_parameter('backend').value
            logger.info(f'Backend: {backend}')
            logger.info('Parsing requested JSON schema')
            try:
                schema = _strict_json(goal_handle.request.json_schema)
                validator_class = validator_for(schema, default=None)
                if validator_class is None:
                    if isinstance(schema, dict) and '$schema' in schema:
                        raise ValueError('Unsupported JSON Schema dialect: ' + str(schema['$schema']))
                    validator_class = Draft202012Validator
                validator_class.check_schema(schema)
                validator = validator_class(schema)
            except (ValueError, TypeError, SchemaError) as error:
                detail = error.message if isinstance(error, SchemaError) else str(error)
                raise ValueError(f'Invalid requested JSON schema: {detail}') from None

            # Snapshot once after schema validation. Camera callbacks replace,
            # never mutate, this message while inference runs.
            with self._image_lock:
                image = self._latest_image
            if image is None:
                raise ValueError('No camera frame available.')
            if backend not in ('ollama', 'nebula'):
                raise ValueError('Unsupported backend; expected ollama or nebula.')
            goal_handle.publish_feedback(ObserveWithVLM.Feedback(state='running_vlm'))
            # ROS color image -> BGR pixels -> JPEG bytes -> base64.
            pixels = self._bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            ok, jpeg = cv2.imencode('.jpg', pixels)
            if not ok:
                raise ValueError('Could not encode camera frame as JPEG.')
            image_b64 = base64.b64encode(jpeg.tobytes()).decode('ascii')
            ask = self._ask_ollama if backend == 'ollama' else self._ask_nebula
            logger.info('Sending frame to VLM')
            response = ask(goal_handle.request.prompt, image_b64, schema)
            logger.info('Structured response received')
            try:
                if not isinstance(response, str):
                    raise ValueError('Expected a JSON string from the VLM.')
                response_json = _strict_json(response)
            except ValueError as error:
                raise ValueError(f'VLM returned invalid JSON: {error}') from None
            try:
                validator.validate(response_json)
            except ValidationError as error:
                raise ValueError(
                    f'VLM response failed schema validation at {error.json_path}: '
                    f'{error.message}') from None
            # success describes the pipeline, not the meaning of any JSON field.
            result.response = json.dumps(response_json, allow_nan=False)
            logger.info('Response validated successfully')
        except (ValueError, RuntimeError) as error:
            result.response = str(error)
        except (KeyError, IndexError, TypeError):
            result.response = 'VLM returned an invalid response format.'
        except Exception as error:
            result.response = f'Observation failed ({type(error).__name__}).'
        else:
            result.success = True
            goal_handle.succeed()
            return result
        finally:
            self._goal_lock.release()
        logger.error(result.response)
        goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ObserveWithVLMActionServer()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node._server.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
