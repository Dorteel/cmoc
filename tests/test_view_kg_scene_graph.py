"""Canonical scene visualization uses the existing read-only viewer adapter."""
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from utils import view_kg


@pytest.fixture
def scene():
    return {'objects': [
        {'id': 'fork(1)', 'type': 'ForkConnector', 'qualities': {'location': [1, 2, 3]}},
        {'id': 'worktop(1)', 'type': 'Worktop', 'qualities': {}},
        {'id': 'KITCHEN', 'type': 'Location', 'qualities': {}}],
        'relations': [
            {'subject': 'fork(1)', 'predicate': 'on_top', 'object': 'worktop(1)'},
            {'subject': 'worktop(1)', 'predicate': 'in', 'object': 'KITCHEN'},
            {'subject': 'fork(1)', 'predicate': 'next_to', 'object': 'worktop(1)'}]}


def test_scene_shortcut_is_canonical_file_not_spa_snapshot():
    assert view_kg.resolve_path('scene_graph') == (
        Path(view_kg.__file__).resolve().parents[1] / 'episodic_memory/scene_graph.json')
    assert view_kg.SNAPSHOTS == {'g1': 'g1_sense.json', 'g2': 'g2_plan.json', 'g3': 'g3_action.json'}
    for stage, name in view_kg.SNAPSHOTS.items():
        assert view_kg.resolve_path(stage) == view_kg.ARTIFACT_DIRECTORY / name


def test_scene_entities_types_properties_and_spatial_edges(tmp_path, scene):
    path = tmp_path / 'scene_graph.json'
    original = json.dumps(scene).encode()
    path.write_bytes(original)
    rendered = view_kg.graph_to_data(view_kg.load_graph(path))
    nodes = {node['label']: node for node in rendered['nodes']}
    assert {'fork(1)', 'worktop(1)', 'KITCHEN'} <= nodes.keys()
    assert nodes['fork(1)']['types'] == ['urn:cmoc:type:ForkConnector']
    assert '[1, 2, 3]' in nodes
    labels = {node['id']: node['label'] for node in rendered['nodes']}
    edges = {(labels[e['source']], e['label'], labels[e['target']]) for e in rendered['edges']}
    for relation in scene['relations']:
        assert (relation['subject'], relation['predicate'], relation['object']) in edges
    assert ('fork(1)', 'location', '[1, 2, 3]') in edges
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('watch', [False, True])
def test_missing_scene_message_and_watch_start(tmp_path, capsys, watch):
    argv = ['view_kg.py', 'scene_graph'] + (['--watch'] if watch else [])
    with patch.object(view_kg, 'ARTIFACT_DIRECTORY', tmp_path), \
         patch('sys.argv', argv), patch.object(view_kg, 'run_server') as server:
        view_kg.main()
    assert capsys.readouterr().out.strip() == f'Scene graph file not found: {tmp_path / "scene_graph.json"}'
    if watch:
        server.assert_called_once_with(tmp_path / 'scene_graph.json', None)
    else:
        server.assert_not_called()


def test_scene_command_passes_direct_file_to_existing_server(tmp_path, scene):
    path = tmp_path / 'scene_graph.json'
    path.write_text(json.dumps(scene))
    with patch.object(view_kg, 'ARTIFACT_DIRECTORY', tmp_path), \
         patch('sys.argv', ['view_kg.py', 'scene_graph']), patch.object(view_kg, 'run_server') as server:
        view_kg.main()
    server.assert_called_once_with(path, None)


def test_scene_watch_reloads_replacement(tmp_path, scene):
    path = tmp_path / 'scene_graph.json'
    path.write_text(json.dumps(scene))
    state = {'revision': 0, 'data': {}, 'error': None}
    class TwoPolls:
        count = 0
        def is_set(self):
            return self.count >= 2
        def wait(self, seconds):
            self.count += 1
            if self.count == 1:
                replacement = tmp_path / 'replacement.json'
                updated = {'objects': [{'id': 'new entity', 'type': 'Box', 'qualities': {}}], 'relations': []}
                replacement.write_text(json.dumps(updated))
                replacement.replace(path)
    view_kg.watch_graph(path, None, state, threading.Lock(), TwoPolls())
    assert state['revision'] == 2
    assert state['error'] is None
    assert 'new entity' in {node['label'] for node in state['data']['nodes']}
    assert 'fork(1)' not in {node['label'] for node in state['data']['nodes']}
