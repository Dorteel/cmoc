"""Shared transport diagnostics; no ROS, VLM server, or external API calls."""

import io
import json
import socket
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from ros2.cmoc_perception.cmoc_perception import vlm_http


class VLMHTTPTests(unittest.TestCase):
    @patch.object(vlm_http, 'urlopen')
    def test_underlying_exception_is_preserved_and_safely_logged(self, urlopen):
        error = URLError(socket.gaierror(-2, 'Name or service not known'))
        urlopen.side_effect = error
        log = Mock()
        with self.assertRaisesRegex(RuntimeError, 'endpoint unavailable') as caught:
            vlm_http.post_json('https://example.invalid', {}, log_error=log)
        self.assertIs(caught.exception.__cause__, error)
        detail = log.call_args.args[0]
        self.assertIn('URLError', detail)
        self.assertIn('gaierror', detail)
        self.assertIn('Name or service not known', detail)
        self.assertEqual(detail, caught.exception.detail)

    @patch.object(vlm_http, 'urlopen')
    def test_http_status_body_redaction_and_truncation(self, urlopen):
        key = 'test-secret-key'
        body = ('bad request ' + key + ' Bearer other-token ' + 'x' * 6000).encode()
        urlopen.side_effect = HTTPError('https://example.invalid', 400, 'bad request', {}, io.BytesIO(body))
        log = Mock()
        with self.assertRaisesRegex(RuntimeError, 'HTTP 400') as caught:
            vlm_http.post_json('https://example.invalid', {}, api_key=key, log_error=log)
        detail = log.call_args.args[0]
        self.assertIn('HTTPError', detail)
        self.assertIn('HTTP 400', detail)
        self.assertIn('body=bad request', detail)
        self.assertNotIn(key, detail)
        self.assertNotIn('other-token', detail)
        self.assertLessEqual(len(detail), 2048)
        self.assertNotIn(key, str(caught.exception))

    @patch.object(vlm_http, 'urlopen')
    def test_read_body_timeout_identified(self, urlopen):
        error = TimeoutError('timed out')
        urlopen.return_value.__enter__.return_value.read.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            vlm_http.post_json('https://example.invalid', {})
        self.assertIn('TimeoutError', caught.exception.detail)
        self.assertIn('phase=read response body', caught.exception.detail)
        self.assertIn('timeout=120s', caught.exception.detail)

    @patch.object(vlm_http, 'urlopen')
    def test_request_headers_and_timeout_unchanged(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = b'{"ok": true}'
        self.assertEqual(vlm_http.post_json('https://example.invalid', {'model': 'm'}, 'secret'), {'ok': True})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret')
        self.assertEqual(request.get_header('Content-type'), 'application/json')
        self.assertEqual(json.loads(request.data), {'model': 'm'})
        self.assertEqual(urlopen.call_args.kwargs['timeout'], 120)

    def test_same_nebula_image_and_schema_format(self):
        post = Mock(return_value={'choices': [{'message': {'content': '{}'}}]})
        schema = {'type': 'object'}
        vlm_http.ask_nebula('prompt', 'jpeg-data', schema, api_key='secret', post=post)
        url, payload = post.call_args.args
        self.assertEqual(url, vlm_http.NEBULA_URL)
        self.assertEqual(payload['model'], vlm_http.NEBULA_MODEL)
        self.assertEqual(payload['messages'][0]['content'], [
            {'type': 'text', 'text': 'prompt'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,jpeg-data'}}])
        self.assertEqual(payload['response_format'], {'type': 'json_schema', 'json_schema': {
            'name': 'observation', 'strict': True, 'schema': schema}})
        vlm_http.ask_nebula('hello', None, None, api_key='secret', post=post)
        self.assertNotIn('response_format', post.call_args.args[1])
        self.assertEqual(len(post.call_args.args[1]['messages'][0]['content']), 1)


class NebulaHardeningTests(unittest.TestCase):
    def test_normalization_and_image_metrics(self):
        import base64
        import cv2
        import numpy as np
        for shape, maximum, expected in (((720, 1280, 3), 1024, (576, 1024)),
                                          ((1280, 720, 3), 512, (512, 288)),
                                          ((20, 30, 3), 1024, (20, 30))):
            with self.subTest(shape=shape):
                log = Mock()
                encoded = vlm_http.encode_nebula_image(
                    np.zeros(shape, dtype=np.uint8), max_dimension=maximum,
                    jpeg_quality=85, source_encoding='bgra8', log_info=log)
                jpeg = base64.b64decode(encoded)
                decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape[:2], expected)
                messages = ' '.join(call.args[0] for call in log.call_args_list)
                self.assertIn('bgra8', messages)
                self.assertIn(f'{len(jpeg)} bytes', messages)
                self.assertIn(f'base64={len(encoded)} bytes', messages)
                self.assertNotIn(encoded, messages)

    @patch.object(vlm_http.time, 'sleep')
    @patch.object(vlm_http, 'urlopen')
    def test_one_read_timeout_retry_and_payload_metrics(self, urlopen, sleep):
        first, second = Mock(), Mock()
        first.read.side_effect = TimeoutError('read timed out')
        second.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        urlopen.return_value.__enter__.side_effect = [first, second]
        log = Mock()
        self.assertEqual(vlm_http.ask_nebula('hello', None, None, api_key='secret', log_info=log), 'OK')
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1.0)
        messages = ' '.join(call.args[0] for call in log.call_args_list)
        self.assertIn(f'Payload: {len(urlopen.call_args.args[0].data)} bytes', messages)
        self.assertIn('Attempt 2/2', messages)
        self.assertIn('Response received in', messages)
        self.assertIn('read timeout', messages)
        self.assertNotIn('secret', messages)

    @patch.object(vlm_http.time, 'sleep')
    @patch.object(vlm_http, 'urlopen')
    def test_second_timeout_propagates(self, urlopen, sleep):
        urlopen.return_value.__enter__.return_value.read.side_effect = TimeoutError('timed out')
        with self.assertRaises(RuntimeError):
            vlm_http.ask_nebula('hello', None, None, api_key='secret')
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    @patch.object(vlm_http.time, 'sleep')
    @patch.object(vlm_http, 'urlopen')
    def test_http_and_unclassified_connect_timeouts_never_retry(self, urlopen, sleep):
        for error in (HTTPError('https://example.invalid', 401, 'unauthorized', {}, io.BytesIO(b'bad key')),
                      HTTPError('https://example.invalid', 400, 'schema', {}, io.BytesIO(b'bad schema')),
                      HTTPError('https://example.invalid', 503, 'unavailable', {}, io.BytesIO(b'busy')),
                      URLError(TimeoutError('connect timed out'))):
            urlopen.reset_mock()
            urlopen.side_effect = error
            with self.assertRaises(RuntimeError):
                vlm_http.ask_nebula('hello', None, None, api_key='secret')
            urlopen.assert_called_once()
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
