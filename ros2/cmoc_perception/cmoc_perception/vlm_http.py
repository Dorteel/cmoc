"""Shared ROS-independent VLM transport and existing Nebula payload format."""

import base64
import json
import re
import time
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

NEBULA_URL = 'https://nebula.cs.vu.nl/api/chat/completions'
NEBULA_MODEL = 'SURF.Qwen3.5 122B A10B NVFP4'



def encode_nebula_image(pixels, *, max_dimension=1024, jpeg_quality=85,
                        source_encoding='bgr8', log_info=None):
    """Normalize BGR pixels identically for camera and standalone diagnostics."""
    import cv2

    if max_dimension < 1 or not 1 <= jpeg_quality <= 100:
        raise ValueError('Image maximum dimension must be positive and JPEG quality 1..100.')
    height, width = pixels.shape[:2]
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1:
        pixels = cv2.resize(pixels, (max(1, round(width * scale)),
                                    max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError('Could not encode image as JPEG.')
    data = jpeg.tobytes()
    encoded = base64.b64encode(data).decode('ascii')
    if log_info:
        h, w = pixels.shape[:2]
        log_info(f'Nebula source image: {width}x{height}, {source_encoding}')
        log_info(f'Nebula image: {w}x{h}, JPEG, {len(data)} bytes '
                 f'({len(data) / 1024:.1f} KiB); base64={len(encoded)} bytes')
    return encoded

def safe_detail(text, api_key=None):
    text = str(text)
    if api_key:
        for secret in (api_key, json.dumps(api_key)[1:-1], quote(api_key, safe='')):
            text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)Bearer\s+[^\s"\'<>]+', 'Bearer [REDACTED]', text)
    return text[:2048]


def post_json(url, payload, api_key=None, *, timeout=120, backend='nebula', log_error=None):
    if timeout <= 0:
        raise ValueError('request_timeout_sec must be positive.')
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    request = Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
    started = time.monotonic()
    phase = 'connect / response headers (urllib may not distinguish)'
    try:
        with urlopen(request, timeout=timeout) as response:
            phase = 'read response body'
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        elapsed = time.monotonic() - started
        reason = getattr(error, 'reason', error)
        frames = traceback.extract_tb(error.__traceback__)
        if isinstance(reason, BaseException):
            frames += traceback.extract_tb(reason.__traceback__)
        names = {frame.name for frame in frames}
        if '_read_status' in names:
            phase = 'read response headers'
        elif 'do_handshake' in names:
            phase = 'TLS handshake'
        elif 'create_connection' in names:
            phase = 'connect'
        detail = (f'{backend} request failed: {type(error).__name__}; '
                  f'cause={type(reason).__name__}: {reason}; phase={phase}; '
                  f'elapsed={elapsed:.2f}s; timeout={timeout}s')
        if isinstance(error, HTTPError):
            try:
                body = error.read(4096 + len(api_key or '')).decode('utf-8', errors='replace')
            except (OSError, ValueError) as body_error:
                body = f'<response body unavailable: {type(body_error).__name__}>'
            finally:
                error.close()
            detail += f'; HTTP {error.code}; body={body}'
            message = f'{backend} VLM endpoint returned HTTP {error.code}.'
        else:
            message = f'{backend} VLM endpoint unavailable or request timed out.'
        detail = safe_detail(detail, api_key)
        if log_error is not None:
            log_error(detail)
        failure = RuntimeError(message)
        failure.detail = detail
        # Retry only proven reads, never ambiguous connection/TLS timeouts.
        failure.read_timeout = (not isinstance(error, HTTPError)
                                and isinstance(reason, TimeoutError)
                                and phase.startswith('read response'))
        raise failure from error


def ask_nebula(prompt, image_b64, schema, *, api_key, url=NEBULA_URL,
               model=NEBULA_MODEL, post=post_json, log_info=None, retry_delay=1.0):
    if not api_key:
        raise ValueError('Nebula API key is missing; set the environment variable '
                         'named by nebula_api_key_env before starting the server.')
    content = [{'type': 'text', 'text': prompt}]
    if image_b64 is not None:
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + image_b64}})
    payload = {'model': model, 'stream': False, 'messages': [{'role': 'user', 'content': content}]}
    if schema is not None:
        payload['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'observation', 'strict': True, 'schema': schema}}
    def report(message):
        if log_info is not None:
            log_info(safe_detail(message, api_key))

    report(f'Nebula model: {model}')
    size = len(json.dumps(payload).encode('utf-8'))
    report(f'Payload: {size} bytes ({size / 1024:.1f} KiB)')
    for attempt in (1, 2):
        report(f'Attempt {attempt}/2')
        started = time.monotonic()
        try:
            result = post(url, payload, api_key=api_key)
        except RuntimeError as error:
            report(getattr(error, 'detail', str(error)))
            report(f'Attempt failed after {time.monotonic() - started:.1f} s')
            if attempt == 2 or not getattr(error, 'read_timeout', False):
                raise
            report(f'Nebula read timeout; retrying once in {retry_delay:g} s')
            time.sleep(retry_delay)
        else:
            report(f'Response received in {time.monotonic() - started:.1f} s')
            return result['choices'][0]['message']['content']
