"""Run the real perception execute path with mocked HTTP/camera boundaries."""
import json
from threading import Condition, Event, Lock, Thread
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from cmoc_perception.observe_with_vlm_server import ObserveWithVLMActionServer as Server

SCHEMA = {'type': 'object', 'properties': {'found': {'type': 'boolean'}},
          'required': ['found'], 'additionalProperties': False}


@pytest.fixture
def server(tmp_path):
    params = {'backend': 'ollama', 'vlm_cache_dir': str(tmp_path), 'fresh_camera_frames': False,
              'ollama_model': 'local', 'nebula_model': 'remote',
              'ollama_url': 'http://local', 'nebula_url': 'http://remote',
              'nebula_api_key_env': 'CACHE_TEST_KEY',
              'nebula_image_max_dimension': 1024, 'nebula_jpeg_quality': 85}
    s = NS()
    s.params = params
    s.get_parameter = lambda name: NS(value=params[name])
    s.logger = Mock()
    s.get_logger = lambda: s.logger
    s._image_lock = Lock()
    s._image_ready = Condition(s._image_lock)
    s._goal_lock = Lock()
    s._latest_image = NS(data=b'original pixels', encoding='bgr8', height=2, width=2,
                         step=6, is_bigendian=0)
    s._bridge = Mock()
    s._bridge.imgmsg_to_cv2.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    s._ask_ollama = Mock(return_value='{"found": false}')
    s._ask_nebula = Mock(return_value='{"found": true}')
    s._observe = MethodType(Server._observe, s)
    return s


def observe(server, prompt='visible objects', schema=None):
    server._goal_lock.acquire()
    goal = Mock()
    goal.request = NS(prompt=prompt, json_schema=json.dumps(schema or SCHEMA))
    return Server.execute(server, goal)


def test_same_request_persists_and_hits(server):
    assert observe(server).success
    assert observe(server).success
    server._ask_ollama.assert_called_once()
    logs = ' '.join(str(c.args[0]) for c in server.logger.info.call_args_list)
    assert all('[VLM CACHE] ' + event in logs for event in ('MISS', 'SAVED', 'HIT'))
    from pathlib import Path
    entry = json.loads(next(Path(server.params['vlm_cache_dir']).glob('*.json')).read_text())
    assert entry['response'] == {'found': False}
    assert entry['backend'] == 'ollama'
    assert len(entry['image_sha256']) == 64


@pytest.mark.parametrize('change', ['image', 'prompt', 'schema', 'model', 'backend', 'url'])
def test_request_changes_miss(server, change):
    observe(server)
    kwargs = {}
    if change == 'image':
        # Encoded JPEG is identical here; raw camera-byte change must still miss.
        server._latest_image.data = b'changed post-look pixels'
    elif change == 'prompt':
        kwargs['prompt'] = 'search elsewhere'
    elif change == 'schema':
        kwargs['schema'] = {**SCHEMA, 'description': 'new schema'}
    elif change == 'model':
        server.params['ollama_model'] = 'other'
    elif change == 'backend':
        server.params['backend'] = 'nebula'
    else:
        server.params['ollama_url'] = 'http://other'
    assert observe(server, **kwargs).success
    assert server._ask_ollama.call_count + server._ask_nebula.call_count == 2


@pytest.mark.parametrize('damage', ['broken-json', 'schema-invalid', 'metadata-invalid'])
def test_corrupted_entry_recovers(server, damage):
    from pathlib import Path
    observe(server)
    path = next(Path(server.params['vlm_cache_dir']).glob('*.json'))
    entry = json.loads(path.read_text())
    if damage == 'broken-json':
        path.write_text('{broken')
    else:
        if damage == 'schema-invalid':
            entry['response'] = {'found': 'not boolean'}
        else:
            entry['key'] = 'wrong'
        path.write_text(json.dumps(entry))
    assert observe(server).success
    assert server._ask_ollama.call_count == 2


def test_disabled_and_invalid_network_responses_never_cache(server, tmp_path):
    server.params['vlm_cache_dir'] = ''
    observe(server)
    observe(server)
    assert server._ask_ollama.call_count == 2
    assert not list(tmp_path.iterdir())
    server.params['vlm_cache_dir'] = str(tmp_path)
    server._ask_ollama.return_value = '{"found": "invalid"}'
    assert not observe(server).success
    assert not list(tmp_path.iterdir())


def test_changed_post_look_frame_requires_new_response(server):
    assert json.loads(observe(server).response)['found'] is False
    assert json.loads(observe(server).response)['found'] is False
    server._latest_image.data = b'new view after look-at'
    server._ask_ollama.return_value = '{"found": true}'
    assert json.loads(observe(server).response)['found'] is True
    assert server._ask_ollama.call_count == 2


def test_fallback_entry_is_not_a_nebula_hit(server):
    server.params['backend'] = 'nebula'
    server._ask_nebula.side_effect = ValueError('unavailable')
    first = observe(server)
    assert json.loads(first.perception_provenance)['backend'] == 'ollama'
    second = observe(server)
    assert second.success
    assert server._ask_nebula.call_count == 2  # primary still attempted, not replaced by local cache
    assert server._ask_ollama.call_count == 1
    server._ask_nebula.side_effect = None
    third = observe(server)
    assert json.loads(third.response)['found'] is True
    assert json.loads(third.perception_provenance)['backend'] == 'nebula'


@pytest.mark.parametrize('cached,changed', [(False, False), (True, False), (True, True)])
def test_request_waits_for_camera_before_cache_lookup(server, cached, changed):
    from copy import deepcopy
    if cached:
        assert observe(server).success
    frame = deepcopy(server._latest_image)
    if changed:
        frame.data = b'changed post-look frame'
    server._latest_image = None
    server._ask_ollama.reset_mock()
    waiting = Event()
    server.logger.info.side_effect = lambda message: waiting.set() if 'Waiting for camera frame' in message else None
    results = []
    worker = Thread(target=lambda: results.append(observe(server)))
    worker.start()
    try:
        assert waiting.wait(2)
        assert worker.is_alive()
        server._ask_ollama.assert_not_called()
        Server._cache_image(server, frame)
    finally:
        if worker.is_alive():
            Server._cache_image(server, frame)
        worker.join(2)
    assert not worker.is_alive()
    assert results[0].success
    assert server._ask_ollama.call_count == (0 if cached and not changed else 1)
    messages = [c.args[0] for c in server.logger.info.call_args_list]
    assert '[VLM CACHE] Camera frame ready' in messages
    expected = '[VLM CACHE] HIT' if cached and not changed else '[VLM CACHE] MISS'
    ready = messages.index('[VLM CACHE] Camera frame ready')
    assert any(m.startswith(expected) for m in messages[ready + 1:])


def test_camera_wait_timeout_fails_without_cache_or_backend(server, monkeypatch):
    from cmoc_perception import observe_with_vlm_server as module
    # Even an existing cache entry cannot be used without a camera frame.
    assert observe(server).success
    server._ask_ollama.reset_mock()
    server._latest_image = None
    monkeypatch.setattr(module, 'CAMERA_STARTUP_TIMEOUT_SEC', .1)
    result = observe(server)
    assert not result.success
    assert result.response == 'No camera frame available after 0.1 s'
    server._ask_ollama.assert_not_called()
    assert not server._goal_lock.locked()


def test_no_camera_cache_disabled_keeps_immediate_failure(server):
    server.params['vlm_cache_dir'] = ''
    server._latest_image = None
    server._image_ready = Mock()
    result = observe(server)
    assert not result.success
    assert result.response == 'No camera frame available.'
    server._image_ready.wait_for.assert_not_called()
    server._ask_ollama.assert_not_called()


@pytest.mark.parametrize('changed', [False, True])
def test_search_waits_past_request_boundary_before_cache(server, changed):
    from copy import deepcopy
    assert observe(server).success
    server._ask_ollama.reset_mock()
    server.params['fresh_camera_frames'] = True
    server.get_clock = lambda: NS(now=lambda:NS(nanoseconds=10))
    # Even a matching cached image with the same timestamp is too old.
    server._latest_image.header = NS(stamp=NS(sec=0, nanosec=10))
    frame = deepcopy(server._latest_image)
    frame.header.stamp.nanosec = 11
    if changed:
        frame.data = b'new post-gaze pixels'
    waiting = Event()
    server.logger.info.side_effect = lambda message: waiting.set() if 'post-request' in message else None
    results = []
    worker = Thread(target=lambda:results.append(observe(server)))
    worker.start()
    try:
        assert waiting.wait(2)
        assert worker.is_alive()
        server._ask_ollama.assert_not_called()
        Server._cache_image(server, frame)
    finally:
        Server._cache_image(server, frame)
        worker.join(2)
    assert not worker.is_alive() and results[0].success
    assert server._ask_ollama.call_count == int(changed)


def test_search_stale_frame_timeout_does_not_use_cached_initial_view(server, monkeypatch):
    assert observe(server).success
    server._ask_ollama.reset_mock()
    server.params['fresh_camera_frames'] = True
    server.get_clock = lambda: NS(now=lambda:NS(nanoseconds=10))
    server._latest_image.header = NS(stamp=NS(sec=0, nanosec=9))
    monkeypatch.setattr('cmoc_perception.observe_with_vlm_server.CAMERA_STARTUP_TIMEOUT_SEC', 0.01)
    result = observe(server)
    assert not result.success
    assert result.response == 'No fresh camera frame available after gaze'
    server._ask_ollama.assert_not_called()
