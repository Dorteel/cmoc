"""Grounded Bringing planning/dispatch tests; no robot or external server."""
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import pytest

from procedural_memory.planning.bringing_plan import plan_bring
from plan_execution import execute_plan, execute_step, manipulation
from graph_snapshots import action_snapshot
from utils.view_kg import snapshot_to_graph

BINDINGS = dict(Agent='robot', Theme='tablefork1', Source='KITCHEN', Destination='user')
STEPS = [{'action': 'navigate', 'args': ['robot', 'KITCHEN']},
         {'action': 'pick', 'args': ['robot', 'tablefork1']},
         {'action': 'navigate', 'args': ['robot', 'user']},
         {'action': 'place', 'args': ['robot', 'tablefork1', 'user']}]
PLANNING = {'status': 'planned', 'plan': STEPS,
            'navigation_rooms': {'KITCHEN': 'KITCHEN', 'user': 'LIVING_ROOM_1'}}


def world():
    return {'objects': [{'id': value, 'type': 'Location'} for value in ('START', 'KITCHEN', 'LIVING_ROOM_1')],
            'relations': [{'subject': subject, 'predicate': 'in', 'object': room}
                          for subject, room in [('robot', 'START'), ('tablefork1', 'KITCHEN'), ('user', 'LIVING_ROOM_1')]]}


def solution():
    def action(name, *tokens):
        return NS(action=NS(name=name), actual_parameters=[NS(object=lambda token=t: NS(name=token)) for t in tokens])
    return NS(actions=[action('move', 'agent', 'room0', 'room1'), action('pick', 'agent', 'theme', 'room1'),
                       action('move', 'agent', 'room1', 'destination'), action('place', 'agent', 'theme', 'destination')])


def test_planner_called_with_existing_domain_and_concrete_initial_state():
    planner = Mock()
    def solve(domain, problem):
        assert domain.as_posix().endswith('bringing/chatgpt/domain.pddl')
        text = problem.read_text()
        assert '(robot_at agent room0)' in text and '(at theme room1)' in text
        assert 'look_for' not in text
        return solution()
    planner.solve.side_effect = solve
    result = plan_bring(BINDINGS, world(), planner)
    assert result == PLANNING
    planner.solve.assert_called_once()


@pytest.mark.parametrize('bindings', [{}, {**BINDINGS, 'Theme': None}])
def test_incomplete_does_not_call_planner(bindings):
    planner = Mock()
    assert plan_bring(bindings, world(), planner)['status'] == 'incomplete'
    planner.solve.assert_not_called()


def test_unknown_initial_location_and_planner_failure():
    planner = Mock()
    assert plan_bring(BINDINGS, {'objects': [], 'relations': []}, planner)['status'] == 'incomplete'
    planner.solve.assert_not_called()
    planner.solve.return_value = None
    assert plan_bring(BINDINGS, world(), planner)['status'] == 'failed'
    planner.solve.side_effect = RuntimeError('no solution')
    assert 'no solution' in plan_bring(BINDINGS, world(), planner)['reason']


def test_dry_run_and_incomplete_send_no_commands():
    with patch('plan_execution.execute_step') as dispatch:
        result = execute_plan(PLANNING)
        assert result['status'] == 'dry_run' and result['executed'] == []
        assert execute_plan({'status': 'incomplete', 'plan': []}, execute=True)['status'] == 'incomplete'
        dispatch.assert_not_called()


def test_dispatch_order_success_and_failed_step_stops():
    with patch('plan_execution.execute_step', return_value=True) as dispatch:
        result = execute_plan(PLANNING, Mock(), execute=True)
        assert result['status'] == 'success'
        assert [call.args[0] for call in dispatch.call_args_list] == STEPS
        assert [attempt['step'] for attempt in result['executed']] == [1, 2, 3, 4]
    for failure in (False, TimeoutError('timeout'), RuntimeError('rejected')):
        with patch('plan_execution.execute_step', side_effect=[True, failure]) as dispatch:
            result = execute_plan(PLANNING, Mock(), execute=True)
            assert result['status'] == 'failed'
            assert result['failed_step'] == STEPS[1]
            assert len(result['executed']) == dispatch.call_count == 2


def test_room_navigation_and_manipulation_mapping():
    navigator = Mock()
    with patch('plan_execution.manipulation', return_value=True) as send:
        for step in STEPS:
            execute_step(step, navigator, PLANNING['navigation_rooms'])
        assert [call.args for call in navigator.go_to_room.call_args_list] == [('KITCHEN',), ('LIVING_ROOM_1',)]
        assert [call.args for call in send.call_args_list] == [('pick', STEPS[1]['args']), ('place', STEPS[3]['args'])]


def test_step_confirmation_can_stop_without_sending_next_goal():
    with patch('builtins.input', side_effect=['', 'q']), patch('plan_execution.execute_step', return_value=True) as dispatch:
        result = execute_plan(PLANNING, execute=True, step_by_step=True)
        assert result['status'] == 'cancelled'
        assert dispatch.call_count == 1


@pytest.mark.parametrize('action,args,endpoint,expected', [
    ('pick', ['robot', 'tablefork1'], '/pick', {'robot': 'TIAGo', 'object': 'tablefork1'}),
    ('place', ['robot', 'tablefork1', 'user'], '/place_next_to',
     {'robot': 'TIAGo', 'object': 'tablefork1', 'target': 'pedestrian'})])
@pytest.mark.parametrize('failure', [None, 'unavailable', 'rejected', 'result', 'timeout'])
def test_ros_fields_waits_failures_and_cleanup(action, args, endpoint, expected, failure):
    ros, executor, client = Mock(), Mock(), Mock()
    ros.ok.return_value = True
    client.wait_for_server.return_value = failure != 'unavailable'
    handle = Mock(accepted=failure != 'rejected')
    goal_future = client.send_goal_async.return_value
    goal_future.result.return_value = handle
    goal_future.done.return_value = True
    result_future = handle.get_result_async.return_value
    result_future.done.return_value = failure != 'timeout'
    result_future.result.return_value = NS(status=4, result=NS(success=failure != 'result', message='failed'))
    factory = Mock(return_value=client)
    action_type = NS(Goal=lambda **kwargs: NS(**kwargs))
    with patch.dict('sys.modules', {'rclpy': ros, 'rclpy.action': NS(ActionClient=factory),
            'rclpy.executors': NS(SingleThreadedExecutor=lambda: executor),
            'simulation_actions.action': NS(Pick=action_type, PlaceNextTo=action_type),
            'action_msgs.msg': NS(GoalStatus=NS(STATUS_SUCCEEDED=4))}):
        if failure:
            with pytest.raises((RuntimeError, TimeoutError)):
                manipulation(action, args)
        else:
            assert manipulation(action, args)
        assert factory.call_args.args[2] == endpoint
        if failure != 'unavailable':
            assert vars(client.send_goal_async.call_args.args[0]) == expected
        client.destroy.assert_called_once()
        executor.shutdown.assert_called_once()
        ros.create_node.return_value.destroy_node.assert_called_once()
        ros.shutdown.assert_not_called()


def test_g3_order_and_size_are_task_only():
    with patch('plan_execution.execute_step', return_value=True):
        result = execute_plan(PLANNING, execute=True)
    snapshot = action_snapshot(result)
    assert snapshot['episode_id'] == 'episode_1'
    assert snapshot['plan'] == STEPS
    assert [n['order'] for n in snapshot['context_graph']['nodes'] if n['type'] == 'Action'] == [1, 2, 3, 4]
    assert len(snapshot['context_graph']['nodes']) == 6
    assert 'remembered_entities' not in snapshot and 'scene_graph' not in snapshot
    assert len(snapshot_to_graph(snapshot)) < 70


def test_world_size_cannot_expand_plan_or_g3():
    planner = Mock()
    planner.solve.return_value = solution()
    small = plan_bring(BINDINGS, world(), planner)
    large_world = world()
    large_world['objects'] += [{'id': f'unrelated_{i}', 'type': 'chair'} for i in range(1000)]
    large = plan_bring(BINDINGS, large_world, planner)
    assert small == large
    assert action_snapshot(execute_plan(small)) == action_snapshot(execute_plan(large))


def test_spa_writes_planned_g3_and_stops_on_failed_execution():
    import demo
    from scene_graph_interface import KnowledgeInterface
    import graph_snapshots
    import json
    scene = world()
    scene['objects'] += [{'id': identifier, 'type': kind, 'qualities': {}}
                         for identifier, kind in [('robot', 'robot'), ('user', 'person'), ('tablefork1', 'ForkConnector')]]
    kg = Mock()
    kg.resolve_concept.side_effect = lambda term: [{'id': 'fork.n.01'}] if term == 'fork' else []
    with patch('builtins.input', return_value=''), \
            patch('demo.observe_scene_with_vlm', return_value={'scene_graph': scene}) as observe, \
            patch('demo.plan_bring', return_value=PLANNING) as planner, \
            patch('plan_execution.execute_step', side_effect=[True, False]) as dispatch:
        result = demo.spa_loop(KnowledgeInterface(), kg, Mock(), Mock(), execute=True, observation_count=3)
    observe.assert_called_once()  # Legacy execution has no Search visibility gate.
    assert planner.call_args.args[0] == BINDINGS
    assert dispatch.call_count == 2
    g3 = json.loads((graph_snapshots.ARTIFACT_DIRECTORY / 'g3_action.json').read_text())
    assert result['action_graph'] == g3
    assert g3['episode_id'] == 'episode_1' and g3['status'] == 'failed'
    assert len(g3['executed']) == 2 and len(g3['plan']) == 4
    assert len(g3['context_graph']['nodes']) == 6
    assert 'scene_graph' not in g3
