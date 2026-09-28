"""Sense/planning-consolidation tests with mocked VLM calls; no ROS processes or Webots."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import demo
from observation import observe_scene_with_vlm
from scene_graph_interface import KnowledgeInterface


def observation(room='KITCHEN', object_id='fork1', object_type='ForkConnector'):
    return {'objects': [
        {'id': room, 'type': 'Location', 'qualities': {}},
        {'id': 'robot', 'type': 'robot', 'qualities': {}},
        {'id': 'user', 'type': 'person', 'qualities': {}},
        {'id': object_id, 'type': object_type, 'qualities': {'color': 'silver'}},
    ], 'relations': [{'subject': object_id, 'predicate': 'in', 'object': room}]}


class SpaTests(unittest.TestCase):
    def setUp(self):
        self.robokg = Mock()
        self.robokg.resolve_concept.side_effect = lambda term: [{'id': 'fork.n.01'}] if term == 'fork' else []
        self.robokg.get_locations.return_value = []
        self.semantic = Mock()
        self.semantic.rank_locations.return_value = []
        self.navigator = Mock()

    def run_spa(self, episodic=None, **kwargs):
        if episodic is None:
            episodic = demo.create_episodic(kwargs.get('scenario', 'existing'))
        return demo.spa_loop(episodic, self.robokg, self.semantic, self.navigator, **kwargs)

    def tearDown(self):
        self.assertEqual(self.navigator.mock_calls, [])

    @patch('demo.observe_scene_with_vlm')
    @patch('builtins.input', return_value='')
    def test_default_instruction_and_enriched_state(self, ask, observe):
        graph = observation()
        observe.return_value = {'scene_graph': graph}
        episodic = KnowledgeInterface()
        state = self.run_spa(episodic)
        ask.assert_called_once_with('Instruction [Bring me a fork]: ')
        observe.assert_called_once_with(schema_path='schemas/objects.json', with_provenance=True)
        expected = {'instruction': 'Bring me a fork', 'scene_graph': graph, 'type': 'bring',
                    'frame': {'Agent': 'robot', 'Theme': 'fork',
                              'Source': 'KITCHEN', 'Destination': 'user'}}
        self.assertEqual({key: state['planning_graph'][key] for key in expected}, expected)
        self.assertEqual(state['planning_graph']['entity_links']['fork1'], 'fork.n.01')
        self.assertEqual(state['bindings']['Theme'], 'fork1')
        self.assertEqual({key: state[key] for key in expected}, expected)
        self.assertEqual({key: state['sense_graph'][key] for key in ('instruction', 'scene_graph')}, {'instruction': 'Bring me a fork',
                                                'scene_graph': graph})
        self.assertNotIn('frame', state['sense_graph'])
        self.assertEqual(state['action_graph']['episode_id'], 'episode_1')
        self.assertEqual(state['action_graph']['executed'], [])
        state['frame']['Source'] = 'changed after planning'
        self.assertEqual(state['planning_graph']['frame']['Source'], 'KITCHEN')
        state['scene_graph']['objects'].clear()
        self.assertEqual(state['sense_graph']['scene_graph'], graph)
        self.assertEqual(state['planning_graph']['scene_graph'], graph)
        self.assertEqual(episodic.query_theme_location('fork'), 'KITCHEN')

    @patch('builtins.input', return_value='')
    def test_planning_consolidates_before_parsing_and_binding(self, ask):
        episodic = KnowledgeInterface()
        calls = Mock()
        with patch.object(episodic, 'merge_observation', wraps=episodic.merge_observation) as merge, \
                patch('demo.NaiveFrameFiller', wraps=demo.NaiveFrameFiller) as parser, \
                patch.object(episodic, 'query_theme_location', wraps=episodic.query_theme_location) as query:
            calls.attach_mock(merge, 'merge')
            calls.attach_mock(parser, 'parse')
            calls.attach_mock(query, 'bind')

            def observe(**kwargs):
                # Sensing must finish before any memory or frame work starts.
                self.assertEqual(calls.mock_calls, [])
                return {'scene_graph': observation()}

            with patch('demo.observe_scene_with_vlm', side_effect=observe):
                state = self.run_spa(episodic)
        self.assertEqual([call[0] for call in calls.mock_calls], ['merge', 'parse'])
        self.assertEqual(set(state['sense_graph']), {'instruction', 'scene_graph', 'context_graph'})
        self.assertEqual(state['planning_graph']['frame']['Source'], 'KITCHEN')

    @patch('demo.observe_scene_with_vlm', return_value={'scene_graph': {'objects': [], 'relations': []}})
    @patch('builtins.input', return_value='')
    def test_existing_memory_fills_source_without_polluting_current_view(self, ask, observe):
        episodic = KnowledgeInterface()
        episodic.merge_observation(observation())
        state = self.run_spa(episodic)
        self.assertEqual(state['frame']['Source'], 'KITCHEN')
        self.assertEqual(state['scene_graph'], {'objects': [], 'relations': []})

    @patch('demo.observe_scene_with_vlm')
    @patch('builtins.input', return_value='Bring me a spoon')
    def test_explicit_instruction_and_unknown_theme(self, ask, observe):
        observe.return_value = {'scene_graph': observation()}
        state = self.run_spa(scenario='empty')
        self.assertEqual(state['instruction'], 'Bring me a spoon')
        self.assertEqual(state['frame']['Theme'], 'spoon')
        self.assertIsNone(state['frame']['Source'])

    @patch('demo.observe_scene_with_vlm')
    @patch('builtins.input', return_value='')
    def test_fresh_observations_update_memory_and_prompt_only_once(self, ask, observe):
        first, second = observation(), observation('LIVING_ROOM_1')
        observe.side_effect = [{'scene_graph': first}, {'scene_graph': second}]
        episodic = KnowledgeInterface()
        state = self.run_spa(episodic, observation_count=2)
        ask.assert_called_once()
        self.assertEqual(observe.call_count, 2)
        self.assertEqual(state['frame']['Source'], 'LIVING_ROOM_1')
        self.assertEqual(state['scene_graph'], second)
        self.assertEqual(first, observation())  # Memory updates never mutate sensed state.
        self.assertEqual(len(episodic.query_theme('fork1')), 1)
        self.assertEqual(len(episodic.query_locations()), 2)  # Unseen rooms retained.

    @patch('demo.observe_scene_with_vlm', return_value={'scene_graph': {'objects': [], 'relations': []}})
    @patch('builtins.input', return_value='')
    def test_scenarios_choose_initial_memory(self, ask, observe):
        for scenario in ('existing', 'empty', 'human-moves'):
            with patch('demo.EpisodicMemory', wraps=KnowledgeInterface) as constructor:
                state = self.run_spa(scenario=scenario)
                expected = None if scenario == 'empty' else Path(demo.__file__).parent / 'episodic_memory/scene_graph.json'
                constructor.assert_called_once_with(expected)
                if scenario == 'empty':
                    self.assertIsNone(state['frame']['Source'])
        self.assertEqual(KnowledgeInterface().query_theme(''), [])
        self.assertEqual(KnowledgeInterface().query_locations(), [])


    @patch('demo.observe_scene_with_vlm', return_value={'scene_graph': {'objects': [], 'relations': []}})
    @patch('builtins.input', return_value='')
    def test_missing_grounding_exhausts_search_without_inventing_locations(self, ask, observe):
        state = self.run_spa(KnowledgeInterface(), search=True)
        self.assertEqual(state['type'], 'incomplete')
        self.assertTrue(state['issues'])
        self.assertIsNone(state['bindings']['Theme'])
        self.robokg.get_locations.assert_called_once_with('fork.n.01')
        self.semantic.rank_locations.assert_called_once_with('fork', [])



class EpisodicUpdatesTests(unittest.TestCase):
    def test_geometry_update_removes_stale_room_and_preserves_unseen_entities(self):
        memory = KnowledgeInterface()
        initial = {'objects': [
            {'id': 'KITCHEN', 'type': 'Location', 'qualities': {'location': [0, 0, 0], 'size': [4, 4]}},
            {'id': 'LIVING_ROOM_1', 'type': 'Location', 'qualities': {'location': [10, 0, 0], 'size': [4, 4]}},
            {'id': 'person1', 'type': 'person', 'qualities': {'location': [0, 0, 0], 'color': 'blue'}},
        ], 'relations': [{'subject': 'person1', 'predicate': 'in', 'object': 'KITCHEN'}]}
        memory.merge_observation(initial)
        self.assertEqual(memory.query_theme_location('person'), 'KITCHEN')
        memory.merge_observation({'objects': [
            {'id': 'person1', 'type': 'person', 'qualities': {'location': [10, 0, 0]}}
        ], 'relations': []})
        self.assertEqual(memory.query_theme_location('person'), 'LIVING_ROOM_1')
        self.assertEqual(memory.query_theme('person')[0]['qualities']['color'], 'blue')
        self.assertEqual(len(memory.query_locations()), 2)
        # A symbolic update must not retain old coordinates that would re-ground
        # the object into the previous room on a later observation.
        memory.merge_observation(observation('KITCHEN', 'person1', 'person'))
        self.assertNotIn('location', memory.query_theme('person')[0]['qualities'])
        self.assertEqual(memory.query_theme_location('person'), 'KITCHEN')

    def test_merge_does_not_overwrite_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.json'
            original = json.dumps(observation())
            path.write_text(original)
            memory = KnowledgeInterface(path)
            memory.merge_observation(observation('LIVING_ROOM_1'))
            self.assertEqual(path.read_text(), original)
            self.assertEqual(memory.query_theme_location('fork'), 'LIVING_ROOM_1')


class ObservationClientTests(unittest.TestCase):
    def setUp(self):
        self.executor = Mock()
        self.ros = Mock()
        self.ros.ok.return_value = True
        self.client = Mock()
        self.handle = Mock(accepted=True)
        self.action_type = Mock()
        self.action_type.Goal.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        self.response = SimpleNamespace(status=4, result=SimpleNamespace(
            success=True, response=json.dumps(observation())))
        self.client.send_goal_async.return_value.result.return_value = self.handle
        self.handle.get_result_async.return_value.result.return_value = self.response
        self.factory = Mock(return_value=self.client)
        patcher = patch.dict('sys.modules', {
            'rclpy': self.ros,
            'rclpy.action': SimpleNamespace(ActionClient=self.factory),
            'rclpy.executors': SimpleNamespace(SingleThreadedExecutor=Mock(return_value=self.executor)),
            'cmoc_interfaces.action': SimpleNamespace(ObserveWithVLM=self.action_type),
            'action_msgs.msg': SimpleNamespace(GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4)),
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_action_uses_existing_object_and_relation_schemas(self):
        self.assertEqual(observe_scene_with_vlm(), observation())
        goal = self.client.send_goal_async.call_args.args[0]
        schema = json.loads(goal.json_schema)
        root = Path(demo.__file__).parent
        self.assertEqual(schema['properties']['objects']['items'],
                         json.loads((root / 'schemas/objects.json').read_text()))
        self.assertEqual(schema['properties']['relations'], json.loads(
            (root / 'schemas/perception/create_scene_graph.json').read_text())['properties']['relations'])
        self.assertEqual(self.factory.call_args.args[2], '/observe_with_vlm')
        self.client.destroy.assert_called_once()
        self.ros.create_node.return_value.destroy_node.assert_called_once()
        self.ros.init.assert_not_called()
        self.ros.shutdown.assert_not_called()

    def test_failure_is_not_consolidated_as_an_observation(self):
        self.response.result.success = False
        self.response.result.response = 'No camera frame available.'
        with self.assertRaisesRegex(RuntimeError, 'No camera frame available'):
            observe_scene_with_vlm()
        self.client.destroy.assert_called_once()
        self.ros.create_node.return_value.destroy_node.assert_called_once()

    def test_timeout_cancels_future_before_destroying_client(self):
        events = []
        future = self.handle.get_result_async.return_value
        future.done.return_value = False
        future.cancel.side_effect = lambda: events.append('cancel')
        self.executor.shutdown.side_effect = lambda: events.append('shutdown executor')
        self.client.destroy.side_effect = lambda: events.append('destroy client')
        with self.assertRaisesRegex(TimeoutError, 'action timed out'):
            observe_scene_with_vlm()
        self.assertEqual(events, ['shutdown executor', 'cancel', 'destroy client'])
        self.assertEqual(self.executor.spin_until_future_complete.call_args.kwargs['timeout_sec'], 510)
        self.ros.shutdown.assert_not_called()

    def test_rejected_action_is_reported(self):
        self.handle.accepted = False
        with self.assertRaisesRegex(RuntimeError, 'goal rejected'):
            observe_scene_with_vlm()


if __name__ == '__main__':
    unittest.main()
