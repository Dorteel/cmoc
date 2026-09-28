"""Presentation-only hierarchy, deterministic layout, and existing viewer dispatch."""
from copy import deepcopy
import json
import threading
from unittest.mock import patch

import pytest
from utils import view_kg


def tree(objects, relations=()):
    return view_kg.scene_tree_data(view_kg.snapshot_to_graph({'objects': objects, 'relations': list(relations)}))


def obj(identifier, kind='Object', **qualities):
    return {'id': identifier, 'type': kind, 'qualities': qualities}


def rel(subject, predicate, target):
    return {'subject': subject, 'predicate': predicate, 'object': target}


def parents(data):
    return {n['label']: n.get('parent') for n in data['nodes']}


def test_priority_and_room_fallback_preserve_all_relations():
    objects = [obj('room', 'Location'), obj('box', 'Container'), obj('table', 'Table'), obj('fork')]
    relations = [rel('box', 'in', 'room'), rel('table', 'in', 'room'),
                 rel('fork', 'in', 'room'), rel('fork', 'on_top', 'table'),
                 rel('fork', 'in', 'box'), rel('fork', 'next_to', 'table')]
    result = tree(objects, relations)
    assert parents(result)['fork'] == 'entity:box'
    relations.remove(rel('fork', 'in', 'box'))
    result = tree(objects, relations)
    assert parents(result)['fork'] == 'entity:table'
    assert parents(result)['table'] == 'entity:room'
    assert parents(result)['room'] == 'display:World'
    preserved = [edge['relation'] for edge in result['edges'] if edge.get('relation')]
    assert sorted(preserved, key=str) == sorted(relations, key=str)
    assert rel('fork', 'in', 'room') in [e['relation'] for e in result['edges'] if e['kind'] == 'secondary']


@pytest.mark.parametrize('stronger,weaker', [('in', 'on_top'), ('on_top', 'attached'),
    ('on', 'attached_to'), ('attached', 'draped'), ('draped', 'under')])
def test_nonroom_relation_priority(stronger, weaker):
    result = tree([obj('item'), obj('a'), obj('z')],
                  [rel('item', weaker, 'a'), rel('item', stronger, 'z')])
    assert parents(result)['item'] == 'entity:z'


def test_every_instance_once_cycles_ties_dangling_and_unassigned():
    objects = [obj('a'), obj('b'), obj('loose'), obj('World'), obj('Unassigned'), obj('room', 'Room')]
    relations = [rel('a', 'in', 'b'), rel('b', 'in', 'a'), rel('loose', 'touching', 'missing')]
    result = tree(objects, relations)
    ids = [n['id'] for n in result['nodes']]
    assert len(ids) == len(set(ids))
    assert {n['label'] for n in result['nodes'] if n['id'].startswith('entity:')} == {
        'a', 'b', 'loose', 'World', 'Unassigned', 'room', 'missing'}
    links = {n['id']: n.get('parent') for n in result['nodes']}
    for identifier in links:
        visited = set()
        while identifier is not None:
            assert identifier not in visited
            visited.add(identifier)
            identifier = links[identifier]
    assert parents(result)['loose'] == 'display:Unassigned'
    assert tree(list(reversed(objects)), list(reversed(relations))) == result
    assert tree([], [])['nodes'][0]['label'] == 'World'


def test_family_colors_properties_and_no_overlap():
    objects = [obj('room', 'Location'), obj('chair1', 'WoodenChair'), obj('chair2', 'Chair'),
               obj('fork', 'ForkConnector', location=[1, 2, 3], orientation=[0, 0, 1, 0], color='silver'),
               obj('robot', 'Tiago++'), obj('user', 'Pedestrian'), obj('box', 'Cabinet'), obj('table', 'Worktop')]
    original = deepcopy(objects)
    result = tree(objects, [rel(o['id'], 'in', 'room') for o in objects[1:]])
    nodes = {n['label']: n for n in result['nodes']}
    assert nodes['chair1']['family'] == nodes['chair2']['family'] == 'Furniture'
    assert nodes['chair1']['color'].split(',')[:2] == nodes['chair2']['color'].split(',')[:2]
    assert view_kg.family_color('Furniture', 'chair1') == nodes['chair1']['color']
    assert len({view_kg.family_color('Furniture', str(i)) for i in range(20)}) > 1
    assert {n['family'] for n in result['nodes'] if 'family' in n} == set(view_kg.FAMILY_HUES)
    assert nodes['fork']['badges'] == ['type: ForkConnector', 'position: [1, 2, 3]']
    assert nodes['fork']['details']['qualities']['color'] == 'silver'
    assert 'graspability' not in nodes['fork']['details']['qualities']
    assert objects == original
    for i, a in enumerate(result['nodes']):
        for b in result['nodes'][i+1:]:
            assert abs(a['x']-b['x']) >= 260 or abs(a['y']-b['y']) >= 76


def test_real_scene_tree_is_read_only_and_complete():
    path = view_kg.resolve_path('scene_graph')
    original = path.read_bytes()
    scene = json.loads(original)
    result = view_kg.scene_tree_data(view_kg.load_graph(path))
    entity_ids = [n['id'] for n in result['nodes'] if n['id'].startswith('entity:')]
    assert len(entity_ids) == len(set(entity_ids))
    assert {'entity:'+o['id'] for o in scene['objects']} <= set(entity_ids)
    assert len([e for e in result['edges'] if e.get('relation')]) == len(scene['relations'])
    assert path.read_bytes() == original


def test_tree_cli_and_watcher(tmp_path):
    path = tmp_path / 'scene_graph.json'
    path.write_text(json.dumps({'objects': [obj('a')], 'relations': []}))
    with patch.object(view_kg, 'ARTIFACT_DIRECTORY', tmp_path), \
         patch('sys.argv', ['view_kg.py', 'scene_graph', '--tree', '--watch']), \
         patch.object(view_kg, 'run_server') as server:
        view_kg.main()
    server.assert_called_once_with(path, None, tree=True)
    state = {'revision': 0}
    class Polls:
        count = 0
        def is_set(self):
            return self.count == 2
        def wait(self, seconds):
            self.count += 1
            if self.count == 1:
                path.write_text(json.dumps({'objects': [obj('updated')], 'relations': []}))
    view_kg.watch_graph(path, None, state, threading.Lock(), Polls(), tree=True)
    assert state['revision'] == 2
    assert state['data']['layout'] == 'tree'
    assert 'updated' in {n['label'] for n in state['data']['nodes']}


@pytest.mark.parametrize('stage', ['g1', 'g2', 'g3'])
def test_tree_flag_does_not_change_spa_shortcuts(stage):
    with patch('sys.argv', ['view_kg.py', stage, '--tree']), pytest.raises(SystemExit) as error:
        view_kg.main()
    assert error.value.code == 2
