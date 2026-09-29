"""Standalone viewer contract: files are sufficient and are never rewritten."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from utils import view_kg

ROOT = Path(view_kg.__file__).resolve().parents[1]


def test_arbitrary_relative_json_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'my_scene_graph.json'
    path.write_text(json.dumps({'objects': [{'id': 'fork', 'type': 'Fork'}]}))
    with patch('sys.argv', ['view_kg.py', path.name]), patch.object(view_kg, 'run_server') as server:
        assert view_kg.main() == 0
    server.assert_called_once_with(path, None)
    assert view_kg.graph_to_data(view_kg.load_graph(path))['nodes']


@pytest.mark.parametrize('content', [None, '{broken', 'null', '[]', '{}', '{"objects": [null]}'])
def test_clean_cli_errors(tmp_path, content):
    path = tmp_path / 'bad.json'
    if content is not None:
        path.write_text(content)
    result = subprocess.run([sys.executable, str(ROOT / 'utils/view_kg.py'), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    expected = 'Scene graph file not found:' if content is None else 'Invalid scene graph JSON:'
    assert expected in result.stdout
    assert str(path) in result.stdout
    assert 'Traceback' not in result.stdout + result.stderr


def test_committed_graphs_render_with_runtime_imports_blocked(tmp_path):
    paths = [ROOT / 'episodic_memory' / name for name in view_kg.SNAPSHOTS.values()]
    before = {path: path.read_bytes() for path in paths}
    script = '''
import importlib.abc
import runpy
import sys
from pathlib import Path
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if (root in {'rclpy', 'rospy', 'controller', 'demo', 'semantic_memory',
                     'scene_graph_interface', 'knowledge_interface', 'core',
                     'simulator_launcher', 'observation', 'navigation'}
                or root.endswith(('_msgs', '_srvs')) or 'webots' in root):
            raise AssertionError('Forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
viewer = runpy.run_path(sys.argv[1])
for filename in sys.argv[2:]:
    graph = viewer['load_graph'](Path(filename))
    assert viewer['graph_to_data'](graph)['nodes']
assert '<svg' in viewer['build_html']()
'''
    env = {key: value for key, value in os.environ.items()
           if key != 'PYTHONPATH' and not key.startswith(('ROS_', 'AMENT_', 'COLCON_'))}
    result = subprocess.run([sys.executable, '-I', '-c', script,
                             str(ROOT / 'utils/view_kg.py'), *map(str, paths)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path: path.read_bytes() for path in paths} == before
