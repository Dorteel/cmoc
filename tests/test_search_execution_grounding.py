"""Execution-only simulator grounding; no Webots or ROS processes."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from navigation import MapAlignment, RoomNavigator, entity_approach_pose
from plan_execution import execute_step, execution_target_position
from scene_graph_interface import KnowledgeInterface
from search_strategy import search_frame, search_result
from external.webots_ros2_simulation.controllers.fallback_action_supervisor import world_utils

CONTROLLER = Path(world_utils.__file__).parent
CLIENT = 'external.webots_ros2_simulation.controllers.fallback_action_supervisor.action_cli.send_action'


def node(identifier, name='', definition='', position=(3, 2, 0)):
    result = Mock()
    result.getTypeName.return_value = 'Solid'
    result.getId.return_value = identifier
    result.getDef.return_value = definition
    result.getField.return_value = SimpleNamespace(getSFString=lambda: name)
    result.getNumberOfFields.return_value = 0
    result.getPosition.return_value = list(position)
    return result


def supervisor(*children):
    root = node(0)
    root.getNumberOfFields.return_value = 1
    root.getFieldByIndex.return_value = SimpleNamespace(
        getTypeName=lambda: 'MFNode', getCount=lambda: len(children),
        getMFNode=lambda index: children[index])
    return SimpleNamespace(getRoot=lambda: root)


@pytest.mark.parametrize('name,definition', [('table_1', ''), ('other', 'table_1'), ('table_1', 'table_1')])
def test_named_pose_uses_current_world_position(name, definition):
    target = node(1, name, definition)
    world = supervisor(target)
    assert world_utils.get_object_pose(world, 'table_1') == {'position': [3.0, 2.0, 0.0]}
    target.getPosition.return_value = [4, 5, 1]
    assert world_utils.get_object_pose(world, 'table_1') == {'position': [4.0, 5.0, 1.0]}


def test_unknown_and_ambiguous_targets_fail():
    with pytest.raises(ValueError, match='not found: table_1'):
        world_utils.get_object_pose(supervisor(), 'table_1')
    with pytest.raises(ValueError, match='ambiguous: table_1'):
        world_utils.get_object_pose(supervisor(node(1, 'table_1'), node(2, 'table_1')), 'table_1')
    with pytest.raises(ValueError, match='ambiguous'):
        world_utils.get_object_pose(supervisor(node(1, 'table_1'), node(2, definition='table_1')), 'table_1')


@pytest.mark.parametrize('name,definition', [
    ('table(1)', ''), (' TABLE---1 ', ''), ('other', 'TABLE__1'),
    ('table(1)', 'TABLE_1'),
])
def test_unique_normalized_identifier_resolves(name, definition):
    world = supervisor(node(1, name, definition))
    assert world_utils.get_object_pose(world, 'table_1') == {'position': [3.0, 2.0, 0.0]}


def test_exact_match_wins_over_normalized_alternatives():
    exact = node(1, 'table_1', position=(9, 8, 0))
    world = supervisor(exact, node(2, 'table(1)'), node(3, 'TABLE--1'))
    assert world_utils.get_object_pose(world, 'table_1') == {'position': [9.0, 8.0, 0.0]}


def test_ambiguous_normalized_match_fails():
    world = supervisor(node(1, 'table(1)'), node(2, definition='TABLE__1'))
    with pytest.raises(ValueError, match='ambiguous: table_1'):
        world_utils.get_object_pose(world, 'table_1')


def test_unrelated_names_do_not_resolve():
    with pytest.raises(ValueError, match='not found: table_1'):
        world_utils.get_object_pose(supervisor(node(1, 'table(2)')), 'table_1')


def test_existing_supervisor_dispatch_exposes_only_requested_pose():
    spec = importlib.util.spec_from_file_location('pose_actions', CONTROLLER / 'actions.py')
    actions = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'world_utils': world_utils, 'head_gaze': Mock()}):
        spec.loader.exec_module(actions)
    assert actions.execute_action(supervisor(node(1, 'table_1')), 'get_object_pose',
                                  {'target': 'table_1'}) == {'position': [3.0, 2.0, 0.0]}
    with pytest.raises(ValueError, match='unexpected fields'):
        actions.execute_action(supervisor(), 'get_object_pose', {'target': 'table_1', 'theme': 'book'})


def test_look_at_uses_service_pose_without_changing_evidence():
    memory = KnowledgeInterface()
    memory.merge_observation({'objects': [{'id': 'table_1', 'type': 'table', 'qualities': {}}],
                              'relations': []})
    evidence = deepcopy(memory.observed_snapshot())
    supplied = {'table_1': [99, 99]}
    navigator = RoomNavigator.__new__(RoomNavigator)
    navigator.goals = SimpleNamespace(alignment=MapAlignment(0, 0, 0))
    navigator.current_map_position = lambda: (0, 0)
    navigator._reachable_approach = Mock(side_effect=lambda name, target, robot: entity_approach_pose(robot, target))
    navigator.navigate_to = Mock(return_value=True)
    with patch(CLIENT, return_value={'ok': True, 'result': {'position': [3, 2, 0]}}) as query:
        assert execute_step({'action': 'look-at', 'args': ['robot', 'table_1']},
                            navigator, {}, supplied, execution_oracle=Mock(resolve=Mock(return_value="table(2)"), optical_targets={"table_1":[3,2,1]}, debug=False))
    query.assert_called_once_with('gaze', {'action': 'look-at', 'optical_target': [3, 2, 1]})
    navigator._reachable_approach.assert_not_called()
    navigator.navigate_to.assert_not_called()
    assert supplied == {'table_1': [99, 99]}
    assert memory.observed_snapshot() == evidence
    frame = search_frame({'Agent': 'robot', 'Theme': 'book'}, 'table_1')
    kg = Mock()
    kg.resolve_concept.return_value = []
    assert search_result(frame, evidence, kg)['Success'] is False
    fresh = {'objects': [{'id': 'book1', 'type': 'book'}], 'relations': []}
    assert search_result(frame, fresh, kg)['Success'] is True


@pytest.mark.parametrize('response', [
    {'ok': False, 'error': 'execution target not found: table_1'},
    {'ok': True, 'result': {'position': [float('nan'), 0, 0]}},
    {'ok': True, 'result': {}},
])
def test_failed_lookup_stops_before_movement(response):
    navigator = Mock()
    with patch(CLIENT, return_value=response), pytest.raises(RuntimeError, match='Cannot resolve execution target table_1'):
        execution_target_position('table_1', identity_source='simulator')
    navigator.approach_entity.assert_not_called()


def test_unavailable_simulator_is_clear_failure():
    with patch(CLIENT, side_effect=ConnectionRefusedError('not running')), pytest.raises(RuntimeError, match='table_1'):
        execution_target_position('table_1', identity_source='simulator')
