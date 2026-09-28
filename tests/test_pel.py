"""Deterministic Perceived Entity Linking (PEL); no ROS or VLM calls."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from perceived_entity_linking import perceived_entity_linking, bind_task
from scene_graph_interface import KnowledgeInterface
import demo


def obj(identifier, kind, position=None):
    return {'id': identifier, 'type': kind, 'qualities': {} if position is None else {'location': position}}


def graph():
    return {'objects': [obj('robot', 'robot', [0, 0, 0]), obj('mannequin', 'mannequin'),
                        obj('KITCHEN', 'Location'), obj('LIVING_ROOM_1', 'Location'),
                        obj('FORK_1', 'ForkConnector', [3, 0, 0]), obj('FORK_2', 'ForkConnector', [1, 0, 0])],
            'relations': [{'subject': 'FORK_1', 'predicate': 'in', 'object': 'KITCHEN'},
                          {'subject': 'FORK_2', 'predicate': 'in', 'object': 'LIVING_ROOM_1'},
                          {'subject': 'mannequin', 'predicate': 'next_to', 'object': 'FORK_1'},
                          {'subject': 'FORK_2', 'predicate': 'next_to', 'object': 'mannequin'}]}


class PELTests(unittest.TestCase):
    def setUp(self):
        self.kg = Mock()
        self.kg.resolve_concept.side_effect = lambda term: {
            'fork': [{'id': 'fork.n.01'}], 'chair': [{'id': 'chair.n.01'}],
            'coffee_mug': [{'id': 'mug.n.04'}],
            'table': [{'id': 'table.n.01'}, {'id': 'table.n.02'}],
        }.get(term, [])
        self.frame = {'Agent': 'robot', 'Theme': 'fork', 'Source': None, 'Destination': 'user'}

    def ground(self, value):
        return bind_task(self.frame, perceived_entity_linking(value, self.kg), self.kg)

    def test_alias_endpoints_and_raw_graph_unchanged(self):
        raw = graph()
        before = deepcopy(raw)
        result = perceived_entity_linking(raw, self.kg)
        self.assertEqual(raw, before)
        self.assertIn('user', [o['id'] for o in result['scene_graph']['objects']])
        self.assertEqual(result['scene_graph']['relations'][-2]['subject'], 'user')
        self.assertEqual(result['scene_graph']['relations'][-1]['object'], 'user')

    def test_unambiguous_ambiguous_and_lexical_fallback(self):
        value = {'objects': [obj('chair', ' Chair '), obj('table', 'table'),
                             obj('cup1', 'coffee mug')], 'relations': []}
        result = perceived_entity_linking(value, self.kg)
        self.assertEqual(result['entity_links'], {'chair': 'chair.n.01', 'cup1': 'mug.n.04'})
        self.assertNotIn('table', result['entity_links'])

    def test_nearest_remembered_instance_and_its_source(self):
        result = self.ground(graph())
        self.assertEqual(result['type'], 'bring')
        self.assertEqual(result['theme_concept'], 'fork.n.01')
        self.assertEqual(result['bindings'], {'Agent': 'robot', 'Theme': 'FORK_2',
                         'Source': 'LIVING_ROOM_1', 'Destination': 'user'})
        self.assertEqual(result['frame']['Theme'], 'fork')
        self.assertEqual(result['frame']['Source'], 'LIVING_ROOM_1')
        self.assertIsNone(self.frame['Source'])

    def test_tie_uses_id_not_input_order(self):
        value = graph()
        value['objects'][-1]['qualities']['location'] = [-3, 0, 0]
        value['objects'].reverse()
        self.assertEqual(self.ground(value)['bindings']['Theme'], 'FORK_1')

    def test_insufficient_positions_do_not_guess(self):
        for identifier in ('robot', 'FORK_1'):
            value = graph()
            next(o for o in value['objects'] if o['id'] == identifier)['qualities'] = {}
            result = self.ground(value)
            self.assertEqual(result['type'], 'incomplete')
            self.assertIsNone(result['bindings']['Theme'])

    def test_single_match_needs_no_distance_but_requires_source(self):
        value = graph()
        value['objects'] = [o for o in value['objects'] if o['id'] != 'FORK_2']
        next(o for o in value['objects'] if o['id'] == 'FORK_1')['qualities'] = {}
        self.assertEqual(self.ground(value)['bindings']['Theme'], 'FORK_1')
        value['relations'] = []
        self.assertEqual(self.ground(value)['type'], 'incomplete')

    @patch('builtins.input', return_value='Bring me a fork')
    def test_spa_g1_g2_and_old_episodic_links_on_demand(self, ask):
        memory = KnowledgeInterface()
        memory.merge_observation(graph())
        # merge keeps explicit room updates and discards outdated coordinates;
        # reinsert measured geometry so the remembered instances can be ranked.
        memory.merge_observation({'objects': [obj('FORK_1', 'ForkConnector', [3, 0, 0]),
                                              obj('FORK_2', 'ForkConnector', [1, 0, 0])],
                                 'relations': graph()['relations'][:2]})
        sensed = {'objects': [obj('mannequin', 'mannequin')], 'relations': []}
        with patch('demo.observe_scene_with_vlm', return_value={'scene_graph': sensed}):
            result = demo.spa_loop(memory, self.kg, Mock(), Mock())
        self.assertEqual(result['sense_graph']['scene_graph'], sensed)
        self.assertEqual(result['planning_graph']['scene_graph']['objects'][0]['id'], 'user')
        self.assertEqual(result['planning_graph']['entity_links']['FORK_2'], 'fork.n.01')
        self.assertEqual(result['bindings']['Theme'], 'FORK_2')
        self.assertEqual(result['type'], 'bring')
        self.assertEqual(memory.snapshot()['objects'][1]['id'], 'mannequin')

    def test_known_aliases_merge_with_existing_user(self):
        value = graph()
        value['objects'].append(obj('user', 'person'))
        self.assertEqual(self.ground(value)['type'], 'bring')


if __name__ == '__main__':
    unittest.main()


def test_other_people_are_not_demo_user():
    kg = Mock()
    kg.resolve_concept.return_value = []
    raw = {'objects': [obj('person_2', 'person'), obj('pedestrian_2', 'pedestrian')], 'relations': []}
    result = perceived_entity_linking(raw, kg)
    assert result['scene_graph'] == raw
    assert result['aliases'] == {}


def test_all_frame_semantic_values_ground_independently():
    from perceived_entity_linking import ground_frame_elements
    kg = Mock()
    kg.resolve_concept.side_effect = lambda value: {
        'robot': [{'id': 'robot.resolved'}], 'fork': [{'id': 'fork.a'}, {'id': 'fork.b'}],
        'kitchen': [{'id': 'kitchen.resolved'}], 'user': [],
    }[value]
    frame = {'Agent': 'robot', 'Theme': 'fork', 'Source': 'KITCHEN', 'Destination': 'user'}
    links, issues = ground_frame_elements(frame, kg)
    assert [c.args[0] for c in kg.resolve_concept.call_args_list] == ['robot', 'fork', 'kitchen', 'user']
    assert links == {'Agent': 'robot.resolved', 'Theme': None, 'Source': 'kitchen.resolved', 'Destination': None}
    assert len(issues) == 2
    assert frame['Source'] == 'KITCHEN'


def test_frame_grounding_reuses_pel_without_resolving_user_or_fork_again():
    from perceived_entity_linking import ground_frame_elements
    kg = Mock()
    kg.resolve_concept.return_value = []
    pel = {'entity_links': {'user': 'pedestrian.n.01', 'tablefork1': 'fork.n.01'},
           'scene_graph': {'objects': [obj('user', 'pedestrian'), obj('tablefork1', 'ForkConnector')]}}
    frame = {'Agent': None, 'Theme': 'fork', 'Source': None, 'Destination': 'user'}
    links, _ = ground_frame_elements(frame, kg, pel)
    assert links['Theme'] == 'fork.n.01'
    assert links['Destination'] == 'pedestrian.n.01'
    kg.resolve_concept.assert_not_called()


def test_concrete_tiago_and_nearest_fork_from_list_positions():
    from knowledge_interface import KnowledgeInterface as RoboKGNet
    kg = RoboKGNet()
    frame = {'Agent': 'robot', 'Theme': 'fork', 'Source': None, 'Destination': 'user'}
    raw = {'objects': [obj('TIAGo', 'Tiago', [-2.69, -2.49, 0.095]),
                       obj('robot', 'robot'),  # Perceived semantic ID must not hide TIAGo's position.
                       obj('user', 'pedestrian'), obj('KITCHEN', 'Location'),
                       obj('tablefork1', 'ForkConnector', [-3.08202, -1.94102, 0.895599]),
                       obj('tablefork2', 'ForkConnector', [-3.30326, -1.85667, 0.915005])],
           'relations': [{'subject': fork, 'predicate': 'in', 'object': 'KITCHEN'}
                         for fork in ('tablefork1', 'tablefork2')]}
    before = deepcopy(raw)
    linked = perceived_entity_linking(raw, kg)
    result = bind_task(frame, linked, kg)
    assert result['bindings'] == {'Agent': 'TIAGo', 'Theme': 'tablefork1',
                                  'Source': 'KITCHEN', 'Destination': 'user'}
    assert result['type'] == 'bring'
    assert result['issues'] == []
    assert frame['Agent'] == result['frame']['Agent'] == 'robot'
    assert raw == before


def test_nearest_selection_ignores_height_and_uses_selected_source():
    kg = Mock()
    kg.resolve_concept.side_effect = lambda term: [{'id': 'fork.n.01'}] if term == 'fork' else []
    frame = {'Agent': 'robot', 'Theme': 'fork', 'Source': None, 'Destination': 'user'}
    raw = {'objects': [obj('TIAGo', 'Tiago', [0, 0, 0]), obj('user', 'person'),
                       obj('KITCHEN', 'Location'), obj('OTHER_ROOM', 'Location'),
                       obj('near', 'ForkConnector', [1, 0, 100]),
                       obj('far', 'ForkConnector', [2, 0, 0])],
           'relations': [{'subject': 'near', 'predicate': 'in', 'object': 'KITCHEN'},
                         {'subject': 'far', 'predicate': 'in', 'object': 'OTHER_ROOM'}]}
    result = bind_task(frame, perceived_entity_linking(raw, kg), kg)
    assert result['bindings']['Theme'] == 'near'
    assert result['bindings']['Source'] == 'KITCHEN'


def test_user_prefers_positioned_simulator_room_without_mutating_aliases():
    from procedural_memory.planning.bringing_plan import room_of, plan_bring
    kg = Mock()
    kg.resolve_concept.return_value = []
    for room in ('KITCHEN', 'LIVING_ROOM_1'):
        raw = {'objects': [obj('pedestrian_1', 'Pedestrian', [-0.674, -2.53, 1.24564]),
                           obj('mannequin_1', 'mannequin'), obj('person_1', 'person'),
                           obj('user', 'person', [99, 99, 0]),
                           obj(room, 'Location'), obj('room_1', 'Location'),
                           obj('TIAGo', 'Tiago'), obj('tablefork1', 'ForkConnector')],
               'relations': [{'subject': identifier, 'predicate': 'in', 'object': target}
                             for identifier, target in [('pedestrian_1', room), ('mannequin_1', 'room_1'),
                                                       ('person_1', 'room_1'), ('user', 'room_1'),
                                                       ('TIAGo', room), ('tablefork1', room)]]}
        original = deepcopy(raw)
        pel = perceived_entity_linking(raw, kg)
        assert raw == original
        canonical = pel['scene_graph']
        assert [r for r in canonical['relations'] if r['subject'] == 'user' and r['predicate'] == 'in'] == [
            {'subject': 'user', 'predicate': 'in', 'object': room}]
        assert next(o for o in canonical['objects'] if o['id'] == 'user')['qualities']['location'] == [-0.674, -2.53, 1.24564]
        assert room_of('user', canonical) == room
        assert pel['aliases']['user'] == ['mannequin_1', 'pedestrian_1', 'person_1']
        # Reaching the solver proves that destination-room validation succeeded.
        planner = Mock()
        planner.solve.return_value = None
        plan_bring({'Agent': 'TIAGo', 'Theme': 'tablefork1', 'Source': room, 'Destination': 'user'}, canonical, planner)
        planner.solve.assert_called_once()
    # Without a numeric simulator position, do not arbitrarily resolve the conflict.
    raw['objects'][0]['qualities'] = {}
    canonical = perceived_entity_linking(raw, kg)['scene_graph']
    assert {r['object'] for r in canonical['relations'] if r['subject'] == 'user' and r['predicate'] == 'in'} == {room, 'room_1'}
