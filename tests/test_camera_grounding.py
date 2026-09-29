"""Actual camera transform math; no live Webots or ROS."""
from math import pi
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from external.webots_ros2_simulation.controllers.fallback_action_supervisor import camera_grounding as camera
from execution_grounding import ExecutionGroundingOracle
from test_execution_oracle import table, supervisor, world_utils, knowledge, observed

GEOMETRY = ([0, 0, 0], [1, 0, 0, 0, -1, 0, 0, 0, -1], pi/2, pi/2)


CONTEXT = {'camera': GEOMETRY, 'scene_to_map': [[0,0,0],[1,0,0,0,1,0,0,0,1]]}


def candidates(*positions):
    return [(f'table({i})', table(i, f'table({i})', pos))
            for i, pos in enumerate(positions, 1)]


@pytest.mark.parametrize('other', [(0, 0, 3), (5, 0, -2), (0, 5, -2)])
def test_one_inside_fov_rejects_behind_and_outside(other):
    matches, diagnostics = camera.select_camera_candidate(candidates((0, 0, -2), other), GEOMETRY)
    assert matches[0][0] == 'table(1)'
    assert 'visible=no' in diagnostics[1]


def test_closest_visible_candidate():
    matches, _ = camera.select_camera_candidate(candidates((.1, 0, -2), (1, 0, -2)), GEOMETRY)
    assert matches[0][0] == 'table(1)'


@pytest.mark.parametrize('positions', [
    ((-.1, 0, -2), (.1, 0, -2)),
    ((0, 0, -2), (0, 0, -2.04)),
])
def test_near_equal_distances_remain_ambiguous(positions):
    with pytest.raises(ValueError, match='ambiguous'):
        camera.select_camera_candidate(candidates(*positions), GEOMETRY)


def test_no_visible_candidate_fails():
    with pytest.raises(ValueError, match='no candidates inside FOV'):
        camera.select_camera_candidate(candidates((0, 0, 2), (8, 0, -2)), GEOMETRY)


def test_camera_rotation_and_translation_not_robot_base_heading():
    # Rotate camera around Y: its optical +Z axis is world -X.
    geometry = ([10, 5, 1], [0, 0, -1, 0, 1, 0, 1, 0, 0], pi/2, pi/2)
    matches, details = camera.select_camera_candidate(candidates((8, 5, 1), (12, 5, 1)), geometry)
    assert matches[0][0] == 'table(1)'
    assert 'horizontal_angle=0.0deg' in details[0]


def test_supervisor_camera_runs_even_after_metric_filtering():
    world = supervisor(table(1, 'table(1)', (0, 0, -2)),
                       table(2, 'table(2)', (2, 0, -2)))
    with patch.object(world_utils, 'get_node', return_value=Mock()),          patch.object(camera, 'select_camera_candidate', wraps=camera.select_camera_candidate) as geometry:
        result = world_utils.resolve_execution_instance(world, 'object_5', 'table', use_camera=True, camera_context=CONTEXT)
        assert result['target'] == 'table(1)'
        assert result['camera_diagnostics']
        geometry.reset_mock()
        result = world_utils.resolve_execution_instance(world, 'object_5', 'table',
                                                        qualities={'location': [2, 0, -2]}, use_camera=True, camera_context=CONTEXT)
        assert result['target'] == 'table(2)'
        geometry.assert_called_once()


@pytest.mark.parametrize('current', [False, True])
def test_oracle_only_requests_geometry_for_currently_observed_id(current):
    scene = observed()
    oracle = ExecutionGroundingOracle(scene, knowledge(),
                                      current_observed_ids=['object_5'] if current else [])
    with patch('execution_grounding.send_action', return_value={
        'ok': True, 'result': {'position':[0,0,0], 'optical_target':[0,0,2], 'target': 'table(1)', 'candidates': ['table(1)']}
    }) as query, patch('execution_grounding.acquire_camera_geometry', return_value=CONTEXT):
        oracle.resolve('object_5')
    assert query.call_args.args[1]['use_camera'] is current
    assert scene == observed()


def test_missing_tf_geometry_has_no_scene_tree_fallback():
    world = supervisor(table(1, 'table(1)'), table(2, 'table(2)'))
    with patch.object(world_utils, 'get_node') as get_node:
        with pytest.raises(ValueError, match='Camera TF geometry unavailable'):
            world_utils.resolve_execution_instance(world, 'object_5', 'table', use_camera=True)
    get_node.assert_not_called()


def test_geometry_ambiguity_reports_attempt_pose_and_margin():
    world = supervisor(table(1, 'table(1)', (0, 0, -2)), table(2, 'table(2)', (0, 0, -2.02)))
    with patch.object(world_utils, 'get_node', return_value=Mock()),          patch.object(camera, 'select_camera_candidate', wraps=camera.select_camera_candidate):
        with pytest.raises(ValueError) as error:
            world_utils.resolve_execution_instance(world, 'object_5', 'table', use_camera=True, camera_context=CONTEXT)
    for text in ('attempting camera-relative', 'camera pose acquired: yes',
                 'distance advantage=0.020m', 'required advantage>0.050m',
                 'table(1):', 'table(2):'):
        assert text in str(error.value)


def test_nonposed_node_excluded_but_bare_physical_table_retained():
    bad = table(2, 'semantic_table')
    bad.getPosition.side_effect = RuntimeError('no pose')
    world = supervisor(table(1, 'table', (0, 0, -2)), bad)
    with patch.object(world_utils, 'get_node', return_value=Mock()), patch.object(camera, 'select_camera_candidate', wraps=camera.select_camera_candidate) as geometry:
        result = world_utils.resolve_execution_instance(world, 'object_5', 'table', use_camera=True, camera_context=CONTEXT)
    assert result['target'] == 'table'
    assert result['candidates'] == ['table']
    assert any('excluded non-executable' in line for line in result['camera_diagnostics'])
    geometry.assert_called_once()


def test_retained_anchor_logs_reason_and_never_calls_camera(capsys):
    oracle = ExecutionGroundingOracle(observed(), knowledge(), debug=True)
    world = supervisor(table(1, 'table(1)'), table(2, 'table(2)'))
    def service(action, parameters):
        try:
            return {'ok': True, 'result': world_utils.resolve_execution_instance(world, **parameters)}
        except ValueError as error:
            return {'ok': False, 'error': str(error)}
    with patch('execution_grounding.send_action', side_effect=service),          patch('execution_grounding.acquire_camera_geometry') as geometry:
        with pytest.raises(RuntimeError, match='not currently observed'):
            oracle.resolve('object_5')
    geometry.assert_not_called()
    assert 'currently observed anchor: no' in capsys.readouterr().out


@pytest.mark.parametrize('current', [False, True])
def test_demo_passes_current_g1_ids_not_retained_memory(current):
    import demo
    from scene_graph_interface import KnowledgeInterface
    from test_demo_modes import world, knowledge as kg
    memory = KnowledgeInterface()
    memory.merge_observation(world(False))
    fresh = world(False)
    if not current:
        fresh['objects'] = [o for o in fresh['objects'] if o['id'] != 'worktop(1)']
        fresh['relations'] = [r for r in fresh['relations'] if r['subject'] != 'worktop(1)']
    with patch('builtins.input', return_value=''),          patch('demo.observe_scene_with_vlm', return_value={'scene_graph': fresh}),          patch('demo.execute_plan', return_value={'status': 'dry_run', 'plan': [], 'executed': []}) as execute:
        result = demo.spa_loop(memory, kg(), Mock(choose_gaze_action=Mock(return_value={"action": "look-at", "target": "worktop(1)" if current else "drawer1"})), None, search=True)
    assert result['frame']['Location'] == ('worktop(1)' if current else 'drawer1')
    oracle = execute.call_args.kwargs['execution_oracle']
    assert ('worktop(1)' in oracle._current_observed_ids) is current
    assert 'worktop(1)' in oracle._objects


def test_current_oracle_reaches_camera_and_prints_per_candidate_diagnostics(capsys):
    world = supervisor(table(1, 'table(1)', (0, 0, -2)), table(2, 'table(2)', (3, 0, -2)))
    oracle = ExecutionGroundingOracle(observed(), knowledge(), debug=True,
                                      current_observed_ids=['object_5'])
    def service(action, parameters):
        if action == 'get_object_pose':
            return {'ok':True,'result':{'position':[0,0,0]}}
        return {'ok': True, 'result': world_utils.resolve_execution_instance(world, **parameters)}
    with patch('execution_grounding.acquire_camera_geometry', return_value=CONTEXT), patch('execution_grounding.send_action', side_effect=service),          patch.object(world_utils, 'get_node', return_value=Mock()),          patch.object(camera, 'select_camera_candidate', wraps=camera.select_camera_candidate):
        assert oracle.resolve('object_5') == 'table(1)'
    output = capsys.readouterr().out
    for text in ('currently observed anchor: yes', 'camera grounding enabled: yes',
                 'attempting camera-relative disambiguation', 'camera pose acquired: yes',
                 'table(1):', 'table(2):', 'visible=no', 'camera-grounded execution instance=table(1)'):
        assert text in output


def test_distance_beats_angular_centering_and_rejects_closer_outside():
    matches, _ = camera.select_camera_candidate(
        candidates((0,0,-4), (.5,0,-2), (2,0,-.1)), GEOMETRY)
    assert matches[0][0] == 'table(2)'


def test_real_kg_subtype_visible_compatibility_and_memory_boundary(capsys):
    from copy import deepcopy
    from knowledge_interface import KnowledgeInterface
    scene = observed()
    before = deepcopy(scene)
    kg = KnowledgeInterface()
    oracle = ExecutionGroundingOracle(scene, kg, debug=True, current_observed_ids=['object_5'])
    desk = table(1, 'desk', (.5,0,-2))
    desk.getTypeName.return_value = 'Desk'
    chair = table(2, 'chair', (0,0,-1))
    chair.getTypeName.return_value = 'Chair'
    world = supervisor(desk, chair, table(3, 'table_far', (0,0,-4)),
                       table(4, 'table_outside', (1,0,-.1)))
    calls = []
    def service(action, parameters):
        calls.append(action)
        if action == 'get_object_pose':
            return {'ok':True, 'result':{'position':[0,0,0]}}
        return {'ok':True, 'result':world_utils.resolve_execution_instance(world, **parameters)}
    with patch('execution_grounding.send_action', side_effect=service), patch('execution_grounding.acquire_camera_geometry', return_value=CONTEXT):
        assert oracle.resolve('object_5') == 'desk'
    assert scene == before
    assert oracle._objects == {o['id']:o for o in before['objects']}
    assert calls == ['get_object_pose', 'resolve_execution_instance']
    out = capsys.readouterr().out
    assert 'table.n.02 -> desk.n.01' in out
    assert 'chair (Chair): rejected' in out
    assert 'Selected closest visible instance: desk' in out
    assert 'Visible simulator objects' in out


def test_declared_model_is_semantic_evidence_not_instance_name():
    from types import SimpleNamespace
    obj = table(1, 'arbitrary_id', (0,0,-2))
    obj.getTypeName.return_value = 'Solid'
    fields = obj.getField.side_effect
    obj.getField.side_effect = lambda name: (SimpleNamespace(getSFString=lambda: {'name':'arbitrary_id','model':'table'}[name])
                                            if name in ('name','model') else fields(name))
    result = world_utils.resolve_execution_instance(supervisor(obj), 'table_1', 'table', use_camera=True, camera_context=CONTEXT)
    assert result['target'] == 'arbitrary_id'


def test_no_invented_worktop_mapping_and_incompatible_visibility_logged():
    obj = table(1, 'worktop(1)', (0,0,-2))
    obj.getTypeName.return_value = 'Worktop'
    with pytest.raises(ValueError, match='no visible compatible candidate') as error:
        world_utils.resolve_execution_instance(supervisor(obj), 'table_1', 'table', use_camera=True, camera_context=CONTEXT)
    assert 'worktop(1) (Worktop): rejected' in str(error.value)


def test_geometry_changes_execution_only_not_search_ranking_success_or_memory():
    from copy import deepcopy
    from scene_graph_interface import KnowledgeInterface as Episodic
    from knowledge_interface import KnowledgeInterface as RoboKG
    from search_strategy import select_candidate, search_frame, search_result
    from test_active_search import setup
    current, retained, search_kg, llm = setup()
    memory = Episodic()
    memory.merge_observation(retained)
    before = deepcopy(memory.observed_snapshot())
    frame = {'Agent':'robot', 'Theme':'fork'}
    ranking = select_candidate(frame, search_kg, llm, before, current=current)
    oracle = ExecutionGroundingOracle(current, RoboKG(), current_observed_ids=['table_1'])
    a, b = table(1, 'physical_a', (0,0,-2)), table(2, 'physical_b', (0,0,-4))
    world = supervisor(a,b)
    def service(action, parameters):
        if action == 'get_object_pose':
            return {'ok':True, 'result':{'position':[0,0,0]}}
        return {'ok':True, 'result':world_utils.resolve_execution_instance(world, **parameters)}
    with patch('execution_grounding.send_action', side_effect=service), patch('execution_grounding.acquire_camera_geometry', return_value=CONTEXT):
        assert oracle.resolve('table_1') == 'physical_a'
        b.getPosition.return_value = [0,0,-1]
        assert oracle.resolve('table_1') == 'physical_b'
    assert memory.observed_snapshot() == before
    assert current == setup()[0]
    assert select_candidate(frame, search_kg, llm, memory.observed_snapshot(), current=current) == ranking
    assert not search_result(search_frame(frame, 'table_1'), current, search_kg)['Success']
