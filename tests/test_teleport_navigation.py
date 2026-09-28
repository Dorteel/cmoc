"""Optional teleport dispatch; no ROS or simulator processes."""
import copy
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch, call

import pytest
import demo
from navigation import MapAlignment, RoomNavigator
from teleport_navigation import TeleportNavigator
from plan_execution import execute_plan


@pytest.fixture
def navigator():
    nav = TeleportNavigator.__new__(TeleportNavigator)
    nav.robot_height = .095
    nav.goals = NS(alignment=MapAlignment(10, -3, math.pi / 2))
    nav.client = Mock()
    nav.current_map_position = Mock(return_value=(10, -3))
    nav._reachable_approach = Mock(return_value=(10, -1.5, math.pi / 2))
    return nav


def test_exact_resolved_pose_converted_to_world_without_nav2_movement(navigator):
    with patch('teleport_navigation.send_action', return_value={'ok': True}) as send:
        assert navigator.approach_entity('book(3)', (2, 0))
    navigator._reachable_approach.assert_called_once_with('book(3)', (10, -1), (10, -3))
    action, params = send.call_args.args
    assert action == 'move'
    assert params['coordinates'] == pytest.approx([1.5, 0, .095])
    assert params['rotation'] == pytest.approx([0, 0, 1, 0])
    navigator.client.send_goal_async.assert_not_called()


def test_room_pose_is_shared(navigator):
    navigator.sample_room_goal = Mock(return_value=(10, -1))
    with patch('teleport_navigation.send_action', return_value={'ok': True}) as send:
        assert navigator.go_to_room('KITCHEN', seed=4)
    navigator.sample_room_goal.assert_called_once_with('KITCHEN', 4)
    assert send.call_args.args[1]['coordinates'] == pytest.approx([2, 0, .095])


@pytest.mark.parametrize('failure', [ConnectionRefusedError('offline'), {'ok': False, 'error': 'robot missing'}])
def test_teleport_failure_never_falls_back(navigator, failure):
    kwargs = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': failure}
    with patch('teleport_navigation.send_action', **kwargs), pytest.raises(RuntimeError, match='Webots teleport'):
        navigator.navigate_to(10, -1)
    navigator.client.send_goal_async.assert_not_called()


def test_symbolic_plan_and_pick_place_unchanged(navigator):
    planning = {'status': 'planned', 'navigation_rooms': {'KITCHEN': 'KITCHEN', 'book(3)': 'KITCHEN', 'user': 'KITCHEN'},
        'plan': [
            {'action': 'navigate', 'args': ['TIAGo', 'KITCHEN']},
            {'action': 'pick', 'args': ['TIAGo', 'book(3)']},
            {'action': 'navigate', 'args': ['TIAGo', 'user']},
            {'action': 'place', 'args': ['TIAGo', 'book(3)', 'user']}]}
    before = copy.deepcopy(planning)
    with patch('teleport_navigation.send_action', return_value={'ok': True}) as send, \
            patch('plan_execution.manipulation', return_value=True) as manipulate:
        result = execute_plan(planning, navigator, execute=True,
            entity_positions={'book(3)': (2, 0), 'user': (4, 0)})
    assert result['status'] == 'success'
    assert planning == before
    assert send.call_count == 2
    assert [c.args[0] for c in navigator._reachable_approach.call_args_list] == ['book(3)', 'user']
    assert manipulate.call_args_list == [call('pick', ['TIAGo', 'book(3)']), call('place', ['TIAGo', 'book(3)', 'user'])]
    navigator.client.send_goal_async.assert_not_called()


def test_flag_requires_execution(capsys):
    with patch('sys.argv', ['demo.py', '--teleport']), patch('demo.rclpy') as ros:
        with pytest.raises(SystemExit) as error:
            demo.main()
    assert error.value.code == 2
    assert '--teleport requires --execute' in capsys.readouterr().err
    ros.init.assert_not_called()


@pytest.mark.parametrize('teleport', [False, True])
def test_demo_selects_backend_only_when_requested(teleport):
    args = ['demo.py', '--execute', '--no-simulator'] + (['--teleport'] if teleport else [])
    with patch('sys.argv', args), patch('demo.rclpy'), patch('demo.SimulatorLauncher'), \
            patch('demo.RoomNavigator') as normal, patch('demo.TeleportNavigator') as tele, \
            patch('demo.PerceptionLauncher'), patch('demo.create_episodic'), \
            patch('demo.RoboKGNet'), patch('demo.SemanticMemory'), patch('demo.spa_loop') as spa:
        demo.main()
    selected, other = (tele, normal) if teleport else (normal, tele)
    selected.assert_called_once_with()
    other.assert_not_called()
    assert spa.call_args.args[3] is selected.return_value
    selected.return_value.close.assert_called_once()
