"""Oracle correspondence stays outside planning and observation evidence."""
from copy import deepcopy
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from execution_grounding import ExecutionGroundingOracle
from navigation import MapAlignment, RoomNavigator, entity_approach_pose
from plan_execution import execute_plan, execution_target_position
from search_strategy import search_frame, search_result
from test_search_execution_grounding import node, supervisor, world_utils, CLIENT


def table(identifier, name, position=(3, 2, 0)):
    obj = node(identifier, name, position=position)
    obj.getTypeName.return_value = 'Table'
    return obj


def knowledge():
    kg = Mock()
    kg.resolve_concept.side_effect = lambda term: [{'id': 'table.n.02'}] if term == 'table' else []
    return kg


def observed(qualities=None):
    return {'objects': [{'id': 'object_5', 'type': 'table', 'qualities': qualities or {'color': 'brown'}}],
            'relations': []}


def test_unique_type_and_ambiguous_instances():
    one = supervisor(table(1, 'table(1)'))
    assert world_utils.resolve_execution_instance(one, 'object_5', 'table', 'table.n.02')['target'] == 'table(1)'
    two = supervisor(table(1, 'table(1)'), table(2, 'table(2)', (7, 2, 0)))
    with pytest.raises(ValueError, match='ambiguous') as error:
        world_utils.resolve_execution_instance(two, 'object_5', 'table', qualities={'color': 'brown'})
    assert 'table(1)' in str(error.value) and 'table(2)' in str(error.value)
    result = world_utils.resolve_execution_instance(two, 'object_5', 'table', qualities={'location': [7, 2, 0]})
    assert result['target'] == 'table(2)'
    result = world_utils.resolve_execution_instance(two, 'object_5', 'table', qualities={'name': 'table(2)'})
    assert result['target'] == 'table(2)'
    with pytest.raises(ValueError, match='no compatible'):
        world_utils.resolve_execution_instance(one, 'object_5', 'unknown')


def test_direct_target_skips_oracle_and_ambiguous_direct_does_not_guess():
    oracle = Mock()
    with patch(CLIENT, return_value={'ok': True, 'result': {'position': [3, 2, 0]}}):
        assert execution_target_position('table(1)', oracle, identity_source='simulator') == [3, 2]
    oracle.resolve.assert_not_called()
    with patch(CLIENT, return_value={'ok': False, 'error': 'execution target is ambiguous: table_1'}):
        with pytest.raises(RuntimeError, match='ambiguous'):
            execution_target_position('table_1', oracle, identity_source='simulator')
    oracle.resolve.assert_not_called()


def test_look_at_oracle_round_trip_keeps_symbolic_plan_and_evidence(capsys):
    scene = observed()
    before = deepcopy(scene)
    kg = knowledge()
    oracle = ExecutionGroundingOracle(scene, kg, debug=True, current_observed_ids=["object_5"])
    world = supervisor(table(1, 'table(1)', (0, 0, -2)))
    calls = []

    def service(action, parameters):
        calls.append((action, deepcopy(parameters)))
        try:
            if action == 'get_object_pose':
                assert parameters == {'target':'TIAGo'}
                result = {'position':[0,0,0]}
            elif action == 'gaze':
                result = {}
            else:
                assert action == 'resolve_execution_instance'
                result = world_utils.resolve_execution_instance(world, **parameters)
            return {'ok': True, 'result': result}
        except ValueError as error:
            return {'ok': False, 'error': str(error)}

    nav = RoomNavigator.__new__(RoomNavigator)
    nav.goals = SimpleNamespace(alignment=MapAlignment(0, 0, 0))
    nav.current_map_position = lambda: (0, 0)
    nav._reachable_approach = Mock(side_effect=lambda name, target, robot: entity_approach_pose(robot, target))
    nav.navigate_to = Mock(return_value=True)
    plan = {'status': 'planned', 'plan': [{'action': 'look-at', 'args': ['robot', 'object_5']}],
            'navigation_rooms': {}}
    original = deepcopy(plan)
    from test_camera_grounding import camera, GEOMETRY, CONTEXT
    with patch(CLIENT, side_effect=service), patch('execution_grounding.send_action', side_effect=service), patch.object(world_utils, 'get_node', return_value=Mock()), patch('execution_grounding.acquire_camera_geometry', return_value=CONTEXT):
        result = execute_plan(plan, nav, execute=True, execution_oracle=oracle)
    assert result['status'] == 'success'
    assert [a for a, _ in calls] == ['get_object_pose', 'resolve_execution_instance', 'gaze']
    assert calls[1][1]['concept'] == 'table.n.02'
    assert calls[0][1] == {'target': 'TIAGo'}
    assert oracle.execution_bindings == {'object_5': 'table(1)'}
    assert scene == before and plan == original
    assert 'table(1)' not in repr(result)
    nav.navigate_to.assert_not_called()
    nav._reachable_approach.assert_not_called()
    assert calls[-1][1] == {'action': 'look-at', 'optical_target': [0.0, 0.0, 2.0]}
    frame = search_frame({'Agent': 'robot', 'Theme': 'book'}, 'object_5')
    assert not search_result(frame, scene, kg)['Success']
    assert search_result(frame, {'objects': [{'id': 'book_1', 'type': 'book'}]}, kg)['Success']
    kg.update_location.assert_not_called()
    assert 'resolved execution instance=table(1)' in capsys.readouterr().out


def test_oracle_ambiguity_reports_failure_without_mapping(capsys):
    oracle = ExecutionGroundingOracle(observed(), knowledge(), debug=True)
    with patch('execution_grounding.send_action', return_value={
        'ok': False, 'error': "ambiguous: simulator candidates=['table(1)', 'table(2)']"}):
        with pytest.raises(RuntimeError, match='Cannot execution-ground object_5.*ambiguous'):
            oracle.resolve('object_5')
    assert oracle.execution_bindings == {}
    assert 'execution grounding failed: ambiguous' in capsys.readouterr().out


def test_dry_run_does_not_query_oracle():
    oracle = Mock()
    plan = {'status': 'planned', 'plan': [{'action': 'look-at', 'args': ['robot', 'object_5']}],
            'navigation_rooms': {}}
    assert execute_plan(plan, Mock(), execution_oracle=oracle)['status'] == 'dry_run'
    oracle.resolve.assert_not_called()


@pytest.mark.parametrize('label', ['dining table', 'dining_table', 'DiningTable'])
def test_dining_table_uses_real_kg_superclass_without_changing_observation(label, capsys):
    from knowledge_interface import KnowledgeInterface
    scene = {'objects': [{'id': 'dining_table_1', 'type': label, 'qualities': {}}], 'relations': []}
    before = deepcopy(scene)
    oracle = ExecutionGroundingOracle(scene, KnowledgeInterface(), debug=True)
    world = supervisor(table(1, 'table(1)'))

    def service(action, parameters):
        assert action == 'resolve_execution_instance'
        assert parameters['concept'] == 'dining_table.n.01'
        assert parameters['compatible_types'] == ['dining_table', 'table']
        return {'ok': True, 'result': world_utils.resolve_execution_instance(world, **parameters)}

    with patch('execution_grounding.send_action', side_effect=service):
        assert oracle.resolve('dining_table_1') == 'table(1)'
    assert scene == before
    assert oracle.execution_bindings == {'dining_table_1': 'table(1)'}
    output = capsys.readouterr().out
    assert 'dining_table.n.01' in output
    assert "compatible simulator instances=['table(1)']" in output


def test_dining_table_compatibility_preserves_ambiguity_and_disambiguation():
    from knowledge_interface import KnowledgeInterface
    world = supervisor(table(1, 'table(1)'), table(2, 'table(2)', (7, 2, 0)))

    def service(action, parameters):
        try:
            return {'ok': True, 'result': world_utils.resolve_execution_instance(world, **parameters)}
        except ValueError as error:
            return {'ok': False, 'error': str(error)}

    for qualities, expected in [({}, None), ({'name': 'table(2)'}, 'table(2)')]:
        scene = {'objects': [{'id': 'dining_table_1', 'type': 'dining table', 'qualities': qualities}]}
        oracle = ExecutionGroundingOracle(scene, KnowledgeInterface())
        with patch('execution_grounding.send_action', side_effect=service):
            if expected is None:
                with pytest.raises(RuntimeError, match='ambiguous'):
                    oracle.resolve('dining_table_1')
                assert oracle.execution_bindings == {}
            else:
                assert oracle.resolve('dining_table_1') == expected

    # The new taxonomy expansion must not make an unrelated concept a table.
    scene = {'objects': [{'id': 'chair_1', 'type': 'chair', 'qualities': {}}]}
    with patch('execution_grounding.send_action', side_effect=service):
        with pytest.raises(RuntimeError, match='no compatible'):
            ExecutionGroundingOracle(scene, KnowledgeInterface()).resolve('chair_1')
