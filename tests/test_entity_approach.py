"""Entity approach geometry and dispatch; no ROS navigation goals sent."""
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import pytest

from navigation import entity_approach_pose, RoomNavigator, MapAlignment
from plan_execution import execute_step


@pytest.mark.parametrize('robot,target', [((1, 2), (4, 6)), ((4, 6), (1, 2)), ((0, 0), (-2, 3))])
def test_standoff_line_and_facing(robot, target):
    x, y, yaw = entity_approach_pose(robot, target)
    assert math.dist((x, y), target) == pytest.approx(0.5)
    dx, dy = target[0] - robot[0], target[1] - robot[1]
    assert (x - robot[0]) * dy - (y - robot[1]) * dx == pytest.approx(0)
    assert math.dist(robot, (x, y)) == pytest.approx(math.dist(robot, target) - 0.5)
    assert yaw == pytest.approx(math.atan2(target[1] - y, target[0] - x))


@pytest.mark.parametrize('target', [(0.1, 0.2), (0.5, 0), (0, 0)])
def test_close_or_coincident_target_keeps_position(target):
    x, y, yaw = entity_approach_pose((0, 0), target)
    assert (x, y) == (0, 0)
    assert math.isfinite(yaw)
    assert yaw == pytest.approx(math.atan2(target[1], target[0]))


def test_existing_navigator_transforms_target_and_uses_current_robot_pose(capsys):
    navigator = RoomNavigator.__new__(RoomNavigator)
    navigator.goals = NS(alignment=MapAlignment(10, -3, math.pi / 2))
    navigator.current_map_position = Mock(return_value=(10, -3))
    navigator.navigate_to = Mock(return_value=True)
    navigator._reachable_approach = Mock(side_effect=lambda name, target, robot: entity_approach_pose(robot, target))
    assert navigator.approach_entity('person42', [2, 0])
    x, y, yaw = navigator.navigate_to.call_args.args
    assert (x, y, yaw) == pytest.approx((10, -1.5, math.pi / 2))
    navigator.current_map_position.assert_called_once()
    assert 'target=person42' in capsys.readouterr().out


def test_generic_entity_dispatch_and_room_fallback():
    navigator = Mock()
    rooms = {'parcel': 'KITCHEN', 'KITCHEN': 'KITCHEN'}
    entity = {'action': 'navigate', 'args': ['TIAGo', 'parcel']}
    execute_step(entity, navigator, rooms, {'parcel': [1, 2]})
    navigator.approach_entity.assert_called_once_with('parcel', [1, 2])
    navigator.go_to_room.assert_not_called()
    execute_step(entity, navigator, rooms, {})
    navigator.go_to_room.assert_called_once_with('KITCHEN')
    navigator.reset_mock()
    execute_step({'action': 'navigate', 'args': ['TIAGo', 'KITCHEN']}, navigator, rooms,
                 {'KITCHEN': [1, 2]})
    navigator.go_to_room.assert_called_once_with('KITCHEN')
    navigator.approach_entity.assert_not_called()


def test_live_tf_lookup_and_cleanup():
    navigator = RoomNavigator.__new__(RoomNavigator)
    navigator.node = Mock()
    navigator._wait = Mock(return_value=NS(transform=NS(translation=NS(x=3, y=4))))
    buffer, listener = Mock(), Mock()
    with patch.dict('sys.modules', {'rclpy.time': NS(Time=lambda: 0),
                                   'tf2_ros': NS(Buffer=lambda: buffer, TransformListener=Mock(return_value=listener))}):
        assert navigator.current_map_position() == (3, 4)
        buffer.wait_for_transform_async.assert_called_once_with('map', 'base_link', 0)
        listener.unregister.assert_called_once()
        navigator._wait.side_effect = TimeoutError('no transform')
        buffer.wait_for_transform_async.return_value.done.return_value = False
        with pytest.raises(RuntimeError, match='Current robot map pose unavailable'):
            navigator.current_map_position()
        buffer.wait_for_transform_async.return_value.cancel.assert_called_once()


def test_positioned_entity_needs_no_room_fallback_entry():
    navigator = Mock()
    execute_step({'action': 'navigate', 'args': ['TIAGo', 'box1']}, navigator, {}, {'box1': [1, 2]})
    navigator.approach_entity.assert_called_once_with('box1', [1, 2])
    navigator.go_to_room.assert_not_called()


def test_spa_passes_canonical_target_position_only_to_execution():
    import demo
    from copy import deepcopy
    world = {'objects': [
        {'id': 'TIAGo', 'type': 'Tiago', 'qualities': {'location': [0, 0, 0]}},
        {'id': 'pedestrian_1', 'type': 'Pedestrian', 'qualities': {'location': [3, 4, 1]}},
        {'id': 'fork1', 'type': 'ForkConnector', 'qualities': {'location': [1, 0, 1]}},
        {'id': 'KITCHEN', 'type': 'Location', 'qualities': {}}],
        'relations': [{'subject': identifier, 'predicate': 'in', 'object': 'KITCHEN'}
                      for identifier in ('TIAGo', 'pedestrian_1', 'fork1')]}
    memory = Mock()
    memory.snapshot.return_value = deepcopy(world)
    kg = Mock()
    kg.resolve_concept.side_effect = lambda term: [{'id': 'fork.n.01'}] if term == 'fork' else []
    planning = {'status': 'planned', 'plan': [
        {'action': 'navigate', 'args': ['TIAGo', 'KITCHEN']},
        {'action': 'pick', 'args': ['TIAGo', 'fork1']},
        {'action': 'navigate', 'args': ['TIAGo', 'user']}],
                'navigation_rooms': {'KITCHEN': 'KITCHEN', 'user': 'KITCHEN'}}
    navigator = Mock()
    with patch('builtins.input', return_value=''), \
            patch('demo.observe_scene_with_vlm', return_value={'scene_graph': {'objects': [], 'relations': []}}), \
            patch('demo.plan_bring', return_value=planning), \
            patch('plan_execution.manipulation', return_value=True):
        result = demo.spa_loop(memory, kg, Mock(), navigator, execute=True)
    assert [call.args for call in navigator.approach_entity.call_args_list] == [('fork1', [1, 0]), ('user', [3, 4])]
    assert result['bindings']['Theme'] == 'fork1'
    assert result['bindings']['Source'] == result['frame']['Source'] == 'KITCHEN'
    assert result['planning']['plan'][0]['args'][1] == 'KITCHEN'
    navigator.go_to_room.assert_not_called()
    assert 'entity_positions' not in result['planning_graph']
    assert 'entity_positions' not in result['action_graph']


def test_pre_pick_refinement_reuses_geometry_and_preserves_symbolic_plan(capsys):
    from copy import deepcopy
    from plan_execution import execute_plan
    planning = {'status': 'planned', 'plan': [
        {'action': 'navigate', 'args': ['TIAGo', 'LIVING_ROOM_1']},
        {'action': 'pick', 'args': ['TIAGo', 'book(3)']},
        {'action': 'navigate', 'args': ['TIAGo', 'user']},
        {'action': 'place', 'args': ['TIAGo', 'book(3)', 'user']}],
        'navigation_rooms': {'LIVING_ROOM_1': 'LIVING_ROOM_1', 'user': 'KITCHEN'}}
    before = deepcopy(planning)
    navigator = RoomNavigator.__new__(RoomNavigator)
    navigator.goals = NS(alignment=MapAlignment(0, 0, 0))
    navigator.current_map_position = Mock(side_effect=[(0, 0), (2.7, 3.6)])
    navigator.navigate_to = Mock(return_value=True)
    navigator._reachable_approach = Mock(side_effect=lambda name, target, robot: entity_approach_pose(robot, target))
    navigator.go_to_room = Mock()
    with patch('plan_execution.manipulation', return_value=True) as pick:
        result = execute_plan(planning, navigator, execute=True,
                              entity_positions={'book(2)': [0.1, 0], 'book(3)': [3, 4], 'user': [6, 4]})
    assert result['status'] == 'success'
    assert planning == before and result['plan'] == before['plan']
    navigator.go_to_room.assert_not_called()
    x, y, yaw = navigator.navigate_to.call_args_list[0].args
    assert math.dist((x, y), (3, 4)) == pytest.approx(.5)
    assert yaw == pytest.approx(math.atan2(4 - y, 3 - x))
    assert navigator.navigate_to.call_count == 2  # Destination still uses entity approach.
    assert pick.call_args_list[0].args == ('pick', ['TIAGo', 'book(3)'])
    output = capsys.readouterr().out
    assert 'symbolic target: LIVING_ROOM_1' in output
    assert 'concrete target: book(3)' in output
    assert 'approach_position=' in output


@pytest.mark.parametrize('position', [None, [], [float('nan'), 1], ['x', 2]])
def test_pre_pick_missing_or_invalid_position_falls_back(position, capsys):
    from plan_execution import execute_plan
    navigator = Mock()
    planning = {'status': 'planned', 'plan': [
        {'action': 'navigate', 'args': ['TIAGo', 'KITCHEN']},
        {'action': 'pick', 'args': ['TIAGo', 'tablefork1']}], 'navigation_rooms': {'KITCHEN': 'KITCHEN'}}
    with patch('plan_execution.manipulation', return_value=True):
        result = execute_plan(planning, navigator, execute=True, entity_positions={'tablefork1': position})
    assert result['status'] == 'success'
    navigator.go_to_room.assert_called_once_with('KITCHEN')
    navigator.approach_entity.assert_not_called()
    assert 'Theme position unavailable; falling back to room navigation: KITCHEN' in capsys.readouterr().out


def test_room_step_without_following_pick_is_not_refined():
    from plan_execution import execute_plan
    navigator = Mock()
    planning = {'status': 'planned', 'plan': [{'action': 'navigate', 'args': ['TIAGo', 'KITCHEN']}],
                'navigation_rooms': {'KITCHEN': 'KITCHEN'}}
    execute_plan(planning, navigator, execute=True, entity_positions={'book(3)': [3, 4]})
    navigator.go_to_room.assert_called_once_with('KITCHEN')
    navigator.approach_entity.assert_not_called()


def make_costmap(value=0):
    return NS(header=NS(frame_id='map'), metadata=NS(
        resolution=.05, size_x=100, size_y=100,
        origin=NS(position=NS(x=0, y=0), orientation=NS(z=0, w=1))),
        data=[value] * 10000)


@pytest.mark.parametrize('value,expected', [(0, True), (1, False), (253, False), (254, False), (255, False)])
def test_nav2_costmap_rejects_inflated_occupied_unknown_and_bounds(value, expected):
    from navigation import costmap_pose_is_free
    costmap = make_costmap(value)
    assert costmap_pose_is_free(costmap, 2, 2) is expected
    assert not costmap_pose_is_free(costmap, -1, 2)
    assert not costmap_pose_is_free(costmap, 5, 2)


@pytest.fixture
def path_validator(monkeypatch):
    import sys
    costmap = make_costmap()
    service = Mock()
    service.call_async.return_value = NS(map=costmap)
    planner = Mock()
    outcomes = []

    def send(goal):
        point = goal.goal.pose.position
        outcome = outcomes.pop(0) if outcomes else 'ok'
        endpoint = NS(x=point.x + (.3 if outcome == 'tolerance' else 0), y=point.y)
        response = NS(status=4, result=NS(error_code=0 if outcome != 'blocked' else 208,
            path=NS(header=NS(frame_id='map'), poses=[NS(pose=NS(position=endpoint))])))
        return NS(accepted=True, get_result_async=lambda: response)

    planner.send_goal_async.side_effect = send
    def goal():
        return NS(goal=NS(header=NS(), pose=NS(position=NS(), orientation=NS())))
    monkeypatch.setitem(sys.modules, 'rclpy.action', NS(ActionClient=Mock(return_value=planner)))
    monkeypatch.setitem(sys.modules, 'nav2_msgs.action', NS(ComputePathToPose=NS(Goal=goal)))
    monkeypatch.setitem(sys.modules, 'nav2_msgs.srv', NS(GetCostmap=NS(Request=lambda: NS())))
    monkeypatch.setitem(sys.modules, 'action_msgs.msg', NS(GoalStatus=NS(STATUS_SUCCEEDED=4)))
    navigator = RoomNavigator.__new__(RoomNavigator)
    navigator.node = Mock()
    navigator.node.create_client.return_value = service
    navigator._wait = lambda future, timeout: future
    navigator.goals = NS(alignment=MapAlignment(0, 0, 0))
    navigator.current_map_position = lambda: (1, 2)
    navigator.navigate_to = Mock(return_value=True)
    return navigator, costmap, planner, outcomes


def test_preferred_pose_validated_before_movement(path_validator, capsys):
    navigator, _, planner, _ = path_validator
    assert navigator.approach_entity('book(3)', (3, 2))
    assert navigator.navigate_to.call_args.args == pytest.approx((2.5, 2, 0))
    planner.send_goal_async.assert_called_once()
    planner.destroy.assert_called_once()
    navigator.node.destroy_client.assert_called_once()
    assert 'Desired pose navigable: True' in capsys.readouterr().out


@pytest.mark.parametrize('failure', ['costmap', 'blocked', 'tolerance'])
def test_invalid_preferred_uses_reachable_radial_candidate(path_validator, failure):
    navigator, costmap, planner, outcomes = path_validator
    if failure == 'costmap':
        costmap.data[40 * 100 + 50] = 254
    else:
        outcomes.append(failure)
    navigator.approach_entity('book(3)', (3, 2))
    x, y, yaw = navigator.navigate_to.call_args.args
    assert (x, y) != (2.5, 2)
    assert .5 <= math.dist((x, y), (3, 2)) <= 1
    assert yaw == pytest.approx(math.atan2(2-y, 3-x))


def test_all_blocked_fails_without_movement(path_validator):
    navigator, costmap, planner, _ = path_validator
    costmap.data[:] = [254] * len(costmap.data)
    with pytest.raises(RuntimeError, match='No reachable approach pose found for book'):
        navigator.approach_entity('book(3)', (3, 2))
    navigator.navigate_to.assert_not_called()
    planner.send_goal_async.assert_not_called()
    planner.destroy.assert_called_once()
    navigator.node.destroy_client.assert_called_once()


def test_all_radial_candidates_bounded_and_face_target():
    from navigation import approach_candidates
    poses = list(approach_candidates((0, 0), (3, 2)))
    for x, y, yaw in poses:
        assert .5 - 1e-9 <= math.dist((x, y), (3, 2)) <= 1 + 1e-9
        assert yaw == pytest.approx(math.atan2(2-y, 3-x))
