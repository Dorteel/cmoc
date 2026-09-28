"""Atomic runtime snapshots and existing viewer adaptation; no ROS/browser."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

from rdflib import Graph, URIRef

import demo
import graph_snapshots
from scene_graph_interface import KnowledgeInterface
from utils import view_kg


def test_canonical_episodic_path():
    assert demo.EPISODIC_GRAPH == Path(demo.__file__).parent / 'episodic_memory/scene_graph.json'
    assert demo.create_episodic('empty').snapshot() == {'objects': [], 'relations': []}
    assert demo.create_episodic('existing').snapshot()['objects']


def test_atomic_overwrite_and_failure_preserves_previous_snapshot():
    destination = graph_snapshots.write_graph_snapshot('g1', {'instruction': 'first'})
    assert destination == graph_snapshots.ARTIFACT_DIRECTORY / 'g1_sense.json'
    real_replace = graph_snapshots.os.replace
    def replace(source, target):
        assert json.loads(destination.read_text()) == {'instruction': 'first'}
        assert json.loads(Path(source).read_text()) == {'instruction': 'second'}
        real_replace(source, target)
    with patch.object(graph_snapshots.os, 'replace', side_effect=replace):
        graph_snapshots.write_graph_snapshot('g1', {'instruction': 'second'})
    assert json.loads(destination.read_text()) == {'instruction': 'second'}
    try:
        graph_snapshots.write_graph_snapshot('g1', {'bad': float('nan')})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid JSON should fail')
    assert json.loads(destination.read_text()) == {'instruction': 'second'}
    assert list(destination.parent.iterdir()) == [destination]


def test_spa_writes_raw_g1_and_linked_g2():
    raw = {'objects': [{'id': 'mannequin', 'type': 'mannequin', 'qualities': {}}], 'relations': []}
    kg = Mock()
    kg.resolve_concept.return_value = []
    with patch('builtins.input', return_value=''), patch('demo.observe_scene_with_vlm', return_value=raw):
        result = demo.spa_loop(KnowledgeInterface(), kg, Mock(), Mock())
    directory = graph_snapshots.ARTIFACT_DIRECTORY
    g1 = json.loads((directory / 'g1_sense.json').read_text())
    g2 = json.loads((directory / 'g2_plan.json').read_text())
    assert set(g1) == {'instruction', 'scene_graph'}
    assert g1['scene_graph'] == raw
    assert g2 == result['planning_graph']
    assert g2['scene_graph']['objects'][0]['id'] == 'user'
    assert {'entity_links', 'frame', 'bindings', 'type', 'issues'} <= g2.keys()
    assert json.loads((directory / 'g3_action.json').read_text()) is None
    assert json.loads((directory / 'scene_graph.json').read_text())['objects'][0]['id'] == 'mannequin'


def test_viewer_shortcuts_and_missing_message(capsys):
    for stage, name in view_kg.SNAPSHOTS.items():
        assert view_kg.resolve_path(stage) == Path(demo.__file__).parent / 'episodic_memory' / name
    with patch.object(view_kg, 'ARTIFACT_DIRECTORY', graph_snapshots.ARTIFACT_DIRECTORY), \
            patch('sys.argv', ['view_kg.py', 'g2']), patch.object(view_kg, 'run_server') as server:
        view_kg.main()
        server.assert_not_called()
    assert 'G2 snapshot not available yet' in capsys.readouterr().out


def test_viewer_semantics_and_rdf_preservation(tmp_path):
    snapshot = {'instruction': 'Bring me a fork', 'scene_graph': {'objects': [
        {'id': 'user', 'type': 'person', 'qualities': {}},
        {'id': 'FORK_1', 'type': 'fork', 'qualities': {}}], 'relations': []},
        'entity_links': {'FORK_1': 'fork.n.01'},
        'frame': {'Agent': 'robot', 'Theme': 'fork', 'Source': 'KITCHEN', 'Destination': 'user'},
        'bindings': {'Agent': 'robot', 'Theme': 'FORK_1', 'Source': 'KITCHEN', 'Destination': 'user'},
        'type': 'bring'}
    path = graph_snapshots.write_graph_snapshot('g2', snapshot)
    data = view_kg.graph_to_data(view_kg.load_graph(path))
    assert {'linkedTo', 'Theme', 'boundTheme', 'boundDestination'} <= {e['label'] for e in data['edges']}
    assert {'Bringing', 'FORK_1', 'fork.n.01', 'user'} <= {n['label'] for n in data['nodes']}
    rdf_path = tmp_path / 'old.ttl'
    rdf_path.write_text('<urn:a> <urn:b> <urn:c> .')
    assert (URIRef('urn:a'), URIRef('urn:b'), URIRef('urn:c')) in view_kg.load_graph(rdf_path)


def test_watch_missing_snapshot_still_starts(capsys):
    with patch.object(view_kg, 'ARTIFACT_DIRECTORY', graph_snapshots.ARTIFACT_DIRECTORY), \
            patch('sys.argv', ['view_kg.py', 'g2', '--watch']), patch.object(view_kg, 'run_server') as server:
        view_kg.main()
        server.assert_called_once()


def test_existing_watcher_reloads_atomic_json_replacement():
    import threading
    path = graph_snapshots.write_graph_snapshot('g1', {'instruction': 'first',
                                                     'scene_graph': {'objects': [], 'relations': []}})
    state = {'revision': 0, 'data': {}, 'error': None}
    class TwoPolls:
        count = 0
        def is_set(self):
            return self.count >= 2
        def wait(self, seconds):
            self.count += 1
            if self.count == 1:
                graph_snapshots.write_graph_snapshot('g1', {'instruction': 'second',
                    'scene_graph': {'objects': [], 'relations': []}})
    view_kg.watch_graph(path, None, state, threading.Lock(), TwoPolls())
    assert state['revision'] == 2
    assert state['error'] is None
    assert 'second' in {node['label'] for node in state['data']['nodes']}
