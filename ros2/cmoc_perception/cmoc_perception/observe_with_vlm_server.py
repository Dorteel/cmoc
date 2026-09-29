#!/usr/bin/env python3
"""Ask a selected VLM about one snapshot of TIAGo's latest RGB frame."""

import hashlib
import base64
import json
import os
from threading import Condition, Lock
from urllib.error import HTTPError, URLError
from .vlm_cache import VLMResponseCache
from .vlm_http import NEBULA_URL, NEBULA_MODEL, ask_nebula, post_json, encode_nebula_image, safe_detail

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


CAMERA_STARTUP_TIMEOUT_SEC = 5.0


def _strict_json(text):
    # Python otherwise accepts NaN/Infinity, which are not valid JSON values.
    def reject_constant(value):
        raise ValueError(f'{value} is not a valid JSON value')
    return json.loads(text, parse_constant=reject_constant)


def _validated_response(response, validator):
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
    return response_json


class ObserveWithVLMActionServer(Node):
    def __init__(self):
        super().__init__('observe_with_vlm_server')
        for name, default in {
            'backend': 'ollama',
            'vlm_cache_dir': '',
            'fresh_camera_frames': False,
            'camera_topic': '/tiago/camera/color/image_raw',
            'ollama_url': 'http://localhost:11434',
            'ollama_model': 'qwen3-vl:2b',
            'nebula_url': NEBULA_URL,
            'nebula_model': NEBULA_MODEL,
            'nebula_api_key_env': 'NEBULA_API_KEY',
            'request_timeout_sec': 120.0,
            'nebula_image_max_dimension': 1024,
            'nebula_jpeg_quality': 85,
        }.items():
            self.declare_parameter(name, default)
        self._latest_image = None
        self._image_lock = Lock()
        self._image_ready = Condition(self._image_lock)
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
            self._image_ready.notify_all()

    def _post(self, url, payload, api_key=None, *, backend=None):
        return post_json(url, payload, api_key,
                         timeout=float(self.get_parameter('request_timeout_sec').value),
                         backend=backend or self.get_parameter('backend').value,
                         log_error=self.get_logger().error)

    def _ask_ollama(self, prompt, image_b64, schema):
        result = self._post(
            self.get_parameter('ollama_url').value.rstrip('/') + '/api/chat',
            {'model': self.get_parameter('ollama_model').value,
             'format': schema,
             'options': {'temperature': 0},
             'stream': False,
             'messages': [{'role': 'user', 'content': prompt, 'images': [image_b64]}]},
            backend='ollama')
        return result['message']['content']

    def _ask_nebula(self, prompt, image_b64, schema):
        return ask_nebula(
            prompt, image_b64, schema,
            api_key=os.environ.get(self.get_parameter('nebula_api_key_env').value),
            url=self.get_parameter('nebula_url').value,
            model=self.get_parameter('nebula_model').value, post=self._post,
            log_info=self.get_logger().info, retry_read_timeout=False)

    def _observe(self, backend, prompt, image_b64, schema, *, cache=None, cache_requests=None, validate=None):
        """At most two transient-error attempts per backend, then fallback/fail."""
        def request(name):
            if cache is not None:
                cached = cache.load(cache_requests[name], validate)
                if cached is not None:
                    return cached
            ask = self._ask_nebula if name == 'nebula' else self._ask_ollama
            for attempt in (1, 2):
                self.get_logger().info(f'{name} attempt {attempt}/2')
                try:
                    return ask(prompt, image_b64, schema)
                except (RuntimeError, ValueError) as error:
                    cause = error.__cause__ or error
                    transient = (500 <= cause.code < 600 if isinstance(cause, HTTPError)
                                 else isinstance(cause, (URLError, TimeoutError, ConnectionError)))
                    if attempt == 2 or not transient:
                        raise
                    key = os.environ.get(self.get_parameter('nebula_api_key_env').value)
                    self.get_logger().warning(f'{name} transient failure; retrying once: ' +
                                              safe_detail(error, key))
        fallback_used = False
        if backend == 'nebula':
            try:
                response = request('nebula')
            except (RuntimeError, ValueError) as error:
                key = os.environ.get(self.get_parameter('nebula_api_key_env').value)
                self.get_logger().warning('Nebula unavailable; using Ollama fallback: ' +
                                          safe_detail(error, key))
                backend = 'ollama'
                fallback_used = True
                try:
                    response = request('ollama')
                except (RuntimeError, ValueError, KeyError, IndexError, TypeError) as local_error:
                    raise RuntimeError('Both perception backends failed: Nebula: ' +
                                       safe_detail(error, key) + '; Ollama: ' +
                                       safe_detail(local_error, key)) from local_error
        else:
            response = request('ollama')
        return response, {'backend': backend,
                          'model': self.get_parameter(backend + '_model').value,
                          'fallback_used': fallback_used}

    def execute(self, goal_handle):
        result = ObserveWithVLM.Result()
        logger = self.get_logger()
        logger.info('Observation requested')
        fresh = self.get_parameter('fresh_camera_frames').value
        boundary = self.get_clock().now().nanoseconds if fresh else None
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
                if fresh:
                    logger.info('[VLM] Waiting for post-request camera frame...')
                    def current_frame():
                        image = self._latest_image
                        return (image is not None and
                                image.header.stamp.sec * 1000000000 + image.header.stamp.nanosec > boundary)
                    if not self._image_ready.wait_for(current_frame, timeout=CAMERA_STARTUP_TIMEOUT_SEC):
                        raise ValueError('No fresh camera frame available after gaze')
                if self._latest_image is None and self.get_parameter('vlm_cache_dir').value:
                    logger.info('[VLM CACHE] Waiting for camera frame...')
                    # Condition releases the lock while the separate camera
                    # callback group receives frames; no polling or stale-cache fallback.
                    if not self._image_ready.wait_for(
                            lambda: self._latest_image is not None,
                            timeout=CAMERA_STARTUP_TIMEOUT_SEC):
                        raise ValueError(f'No camera frame available after {CAMERA_STARTUP_TIMEOUT_SEC:.1f} s')
                    logger.info('[VLM CACHE] Camera frame ready')
                image = self._latest_image
            if image is None:
                raise ValueError('No camera frame available.')
            if backend not in ('ollama', 'nebula'):
                raise ValueError('Unsupported backend; expected ollama or nebula.')
            goal_handle.publish_feedback(ObserveWithVLM.Feedback(state='running_vlm'))
            # ROS color image -> BGR pixels -> JPEG bytes -> base64.
            pixels = self._bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
            if backend == 'nebula':
                image_b64 = encode_nebula_image(
                    pixels,
                    max_dimension=self.get_parameter('nebula_image_max_dimension').value,
                    jpeg_quality=self.get_parameter('nebula_jpeg_quality').value,
                    source_encoding=image.encoding, log_info=logger.info)
            else:
                ok, jpeg = cv2.imencode('.jpg', pixels)
                if not ok:
                    raise ValueError('Could not encode camera frame as JPEG.')
                image_b64 = base64.b64encode(jpeg.tobytes()).decode('ascii')
            logger.info('Processing frame for VLM')
            cache = None
            cache_requests = {}
            cache_directory = self.get_parameter('vlm_cache_dir').value
            if cache_directory:
                cache = VLMResponseCache(cache_directory, logger.info)
                # Hash original bytes too: lossy JPEG/resizing must never merge
                # distinct camera frames. Timestamps are not visual content.
                common = {
                    'version': 1, 'image_sha256': hashlib.sha256(bytes(image.data)).hexdigest(),
                    'image_encoding': image.encoding, 'height': image.height, 'width': image.width,
                    'step': image.step, 'is_bigendian': image.is_bigendian,
                    'encoded_image_sha256': hashlib.sha256(base64.b64decode(image_b64)).hexdigest(),
                    'prompt': goal_handle.request.prompt, 'system_prompt': None, 'schema': schema,
                    'stream': False,
                }
                for name in ('nebula', 'ollama'):
                    cache_requests[name] = {
                        **common, 'backend': name, 'model': self.get_parameter(name + '_model').value,
                        'url': self.get_parameter(name + '_url').value,
                        'options': {'temperature': 0} if name == 'ollama' else {},
                        'response_format': 'ollama-schema' if name == 'ollama' else 'strict-json-schema',
                    }
            response, provenance = self._observe(
                backend, goal_handle.request.prompt, image_b64, schema,
                **({'cache': cache, 'cache_requests': cache_requests,
                    'validate': lambda text: _validated_response(text, validator)} if cache else {}))
            logger.info('Structured response received')
            response_json = _validated_response(response, validator)
            if cache is not None:
                cache.save(cache_requests[provenance['backend']], response_json)
            # success describes the pipeline, not the meaning of any JSON field.
            result.response = json.dumps(response_json, allow_nan=False)
            result.perception_provenance = json.dumps(provenance)
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
