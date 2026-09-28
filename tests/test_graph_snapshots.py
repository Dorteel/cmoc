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
    assert set(g1) == {'instruction', 'scene_graph', 'context_graph'}
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


def test_episode_identity_aliases_and_explicit_frame_elements():
    from copy import deepcopy
    from perceived_entity_linking import perceived_entity_linking

    raw = {'objects': [
        {'id': identifier, 'type': kind, 'qualities': {}}
        for identifier, kind in [('person_1', 'person'), ('radiator_1', 'radiator'),
                                 ('FORK_1', 'fork'), ('KITCHEN', 'Location')]],
        'relations': [{'subject': 'radiator_1', 'predicate': 'next_to', 'object': 'person_1'},
                      {'subject': 'FORK_1', 'predicate': 'in', 'object': 'KITCHEN'}]}
    original = deepcopy(raw)
    memory = KnowledgeInterface()
    memory.merge_observation({'objects': [
        {'id': identifier, 'type': kind, 'qualities': {}}
        for identifier, kind in [('robot', 'robot'), ('user', 'person'),
                                 ('pedestrian_1', 'pedestrian'), ('mannequin', 'mannequin')]],
        'relations': []})
    kg = Mock()
    kg.resolve_concept.side_effect = lambda term: [{'id': 'fork.n.01'}] if term == 'fork' else []
    with patch('builtins.input', return_value=''), patch('demo.observe_scene_with_vlm', return_value=raw):
        result = demo.spa_loop(memory, kg, Mock(), Mock())
    g1, g2 = result['sense_graph'], result['planning_graph']
    assert raw == original == g1['scene_graph']
    def edges(snapshot):
        return {(r['subject'], r['predicate'], r['object']) for r in snapshot['context_graph']['relations']}
    assert {('episode_1', 'hasInstruction', 'instruction_1'),
            ('episode_1', 'hasObservation', 'observation_1'),
            ('observation_1', 'observedBy', 'robot'),
            ('observation_1', 'observes', 'person_1')} <= edges(g1)
    assert ('observation_1', 'observes', 'user') in edges(g2)
    assert ('episode_1', 'hasObservation', 'observation_1') in edges(g2)
    assert g2['aliases']['user'] == ['mannequin', 'pedestrian_1', 'person_1']
    assert g2['scene_graph']['relations'][0]['object'] == 'user'
    linked_memory = perceived_entity_linking(memory.snapshot(), kg)
    ids = [obj['id'] for obj in linked_memory['scene_graph']['objects']]
    assert ids.count('user') == 1
    assert not {'person_1', 'pedestrian_1', 'mannequin'} & set(ids)
    assert g2['frame']['Theme'] == 'fork'
    assert g2['bindings']['Theme'] == 'FORK_1'
    assert g2['frame'] is not g2['bindings']
    assert ('episode_1', 'hasFrame', 'BringingFrame_1') in edges(g2)
    for role, value in g2['bindings'].items():
        assert ('BringingFrame_1', 'hasFrameElement', role + '_FE') in edges(g2)
        assert (role + '_FE', 'bindsTo', value) in edges(g2)
    rendered = view_kg.graph_to_data(view_kg.snapshot_to_graph(g2))
    assert {'observedBy', 'hasObservation', 'hasInstruction', 'hasFrameElement', 'bindsTo', 'aliases'} <= {
        e['label'] for e in rendered['edges']}
    labels = {n['label'] for n in rendered['nodes']}
    assert {'episode_1', 'observation_1', 'instruction_1', 'BringingFrame_1', 'Theme_FE', 'user'} <= labels
    assert not {'person_1', 'pedestrian_1', 'mannequin'} & labels
    rendered_g1 = view_kg.graph_to_data(view_kg.snapshot_to_graph(g1))
    assert {'person_1', 'robot', 'episode_1'} <= {n['label'] for n in rendered_g1['nodes']}


def test_unresolved_frame_elements_are_visible_without_bindings():
    state = graph_snapshots.episode_snapshot({'instruction': 'Bring me a fork',
                                              'scene_graph': {'objects': [], 'relations': []}})
    state.update(frame={'Agent': 'robot', 'Theme': 'fork', 'Source': None, 'Destination': 'user'},
                 bindings=dict.fromkeys(('Agent', 'Theme', 'Source', 'Destination')))
    graph_snapshots.add_frame_graph(state)
    assert len([n for n in state['context_graph']['nodes'] if n['type'] == 'FrameElement']) == 4
    assert not any(r['predicate'] == 'bindsTo' for r in state['context_graph']['relations'])
    rendered = view_kg.graph_to_data(view_kg.snapshot_to_graph(state))
    assert {'Agent_FE', 'Theme_FE', 'Source_FE', 'Destination_FE'} <= {n['label'] for n in rendered['nodes']}
