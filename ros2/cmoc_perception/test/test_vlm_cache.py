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
    params = {'backend': 'ollama', 'vlm_cache_dir': str(tmp_path), 'fresh_camera_frames': False, 'vlm_cache_save': False,
              'ollama_model': 'local', 'nebula_model': 'remote',
              'ollama_url': 'http://local', 'nebula_url': 'http://remote',
              'nebula_api_key_env': 'CACHE_TEST_KEY',
              'nebula_image_max_dimension': 1024, 'nebula_jpeg_quality': 85}
    s = NS()
    s._first_observation = True
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



def saved_graph(server, graph=None):
    from pathlib import Path
    path = Path(server.params['vlm_cache_dir']) / 'scene_graph.json'
    path.write_text(json.dumps(graph or {'found': False}))
    return path


def test_startup_replay_then_head_turn_and_third_request_are_live(server):
    path = saved_graph(server)
    server._latest_image = None
    first = observe(server)
    assert first.success and json.loads(first.response) == {'found': False}
    server._ask_ollama.assert_not_called()
    server._bridge.imgmsg_to_cv2.assert_not_called()
    assert not server._goal_lock.locked()
    server._latest_image = NS(encoding='bgr8', data=b'head turned')
    server._ask_ollama.return_value = '{"found": true}'
    for count in (1, 2):
        result = observe(server)
        assert result.success and json.loads(result.response) == {'found': True}
        assert server._ask_ollama.call_count == count
    assert json.loads(path.read_text()) == {'found': False}
    server.logger.info.assert_any_call(f'[VLM CACHE] HIT {path}')


def test_replay_bypasses_fresh_camera_wait(server):
    saved_graph(server)
    server.params['fresh_camera_frames'] = True
    server._latest_image = None
    assert observe(server).success
    server._ask_ollama.assert_not_called()


@pytest.mark.parametrize('existing', [False, True])
def test_record_first_live_graph_only(server, existing):
    if existing:
        saved_graph(server, {'found': True})
    server.params['vlm_cache_save'] = True
    assert observe(server).success
    from pathlib import Path
    path = Path(server.params['vlm_cache_dir']) / 'scene_graph.json'
    assert json.loads(path.read_text()) == {'found': False}
    server._ask_ollama.assert_called_once()
    server._ask_ollama.return_value = '{"found": true}'
    assert json.loads(observe(server).response) == {'found': True}
    assert json.loads(path.read_text()) == {'found': False}
    server.logger.info.assert_any_call(f'[VLM CACHE] SAVED {path}')


@pytest.mark.parametrize('contents', ['{broken', '{"found": "wrong"}'])
def test_corrupt_cache_fails_clearly(server, contents):
    path = saved_graph(server)
    path.write_text(contents)
    result = observe(server)
    assert not result.success and 'Invalid VLM cache' in result.response
    assert path.read_text() == contents
    server._ask_ollama.assert_not_called()


def test_missing_cache_requires_explicit_recording(server):
    result = observe(server)
    assert not result.success and '--vlm-cache-save' in result.response
    server._ask_ollama.assert_not_called()


def test_disabled_calls_backend_for_every_request(server, tmp_path):
    server.params['vlm_cache_dir'] = ''
    for _ in range(3):
        assert observe(server).success
    assert server._ask_ollama.call_count == 3
    assert not list(tmp_path.iterdir())


def test_invalid_live_graph_does_not_overwrite_saved_graph(server):
    path = saved_graph(server)
    server.params['vlm_cache_save'] = True
    server._ask_ollama.return_value = '{"found": "wrong"}'
    assert not observe(server).success
    assert json.loads(path.read_text()) == {'found': False}


def test_recording_waits_for_first_frame(server):
    frame = server._latest_image
    server._latest_image = None
    server.params['vlm_cache_save'] = True
    waiting = Event()
    server.logger.info.side_effect = lambda m: waiting.set() if 'Waiting for camera' in m else None
    results = []
    worker = Thread(target=lambda: results.append(observe(server)))
    worker.start()
    try:
        assert waiting.wait(2)
        server._ask_ollama.assert_not_called()
    finally:
        Server._cache_image(server, frame)
        worker.join(2)
    assert not worker.is_alive() and results[0].success
    server._ask_ollama.assert_called_once()


def test_recording_camera_timeout(server, monkeypatch):
    server.params['vlm_cache_save'] = True
    server._latest_image = None
    monkeypatch.setattr('cmoc_perception.observe_with_vlm_server.CAMERA_STARTUP_TIMEOUT_SEC', .01)
    result = observe(server)
    assert not result.success and 'No camera frame available after' in result.response
    server._ask_ollama.assert_not_called()

@pytest.mark.parametrize('changed', [False, True])
def test_search_waits_past_request_boundary_without_cache(server, changed):
    from copy import deepcopy
    server.params['vlm_cache_dir'] = ''
    assert observe(server).success
    server._ask_ollama.reset_mock()
    server.params['fresh_camera_frames'] = True
    server.get_clock = lambda: NS(now=lambda:NS(nanoseconds=10))
    # A frame at the request timestamp is too old.
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
    assert server._ask_ollama.call_count == 1


def test_search_stale_frame_timeout_does_not_use_cached_initial_view(server, monkeypatch):
    server.params['vlm_cache_dir'] = ''
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
