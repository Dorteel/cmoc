"""Perceived IDs never bypass camera correspondence through name collisions."""
from copy import deepcopy
from unittest.mock import Mock, patch
import pytest
from execution_grounding import ExecutionGroundingOracle
from plan_execution import execute_step, execution_target_position
from test_execution_oracle import table, knowledge
from test_search_execution_grounding import supervisor, world_utils, CLIENT
from test_camera_grounding import camera, GEOMETRY, CONTEXT


@pytest.mark.parametrize('ambiguous', [False, True])
def test_same_name_cannot_bypass_camera_or_fallback_on_ambiguity(ambiguous, capsys):
    scene = {'objects':[{'id':'table_1', 'type':'table', 'qualities':{'name':'table_1'}}], 'relations':[]}
    before = deepcopy(scene)
    world = supervisor(table(1,'table_1', (0.1,0,-2) if ambiguous else (20,0,-2)),
                       table(2,'table(2)',(-0.1,0,-2) if ambiguous else (0,0,-2)))
    oracle = ExecutionGroundingOracle(scene, knowledge(), current_observed_ids=['table_1'], debug=True)
    calls = []
    def service(action, parameters):
        calls.append((action,deepcopy(parameters)))
        try:
            if action == 'resolve_execution_instance':
                result = world_utils.resolve_execution_instance(world,**parameters)
            elif action == 'get_object_pose':
                assert parameters == {'target':'TIAGo'}
                result = {'position':[0,0,0]}
            else:
                assert action == 'gaze'
                result = {'head_positions':{'head_1_joint':0.2,'head_2_joint':0.1}}
            return {'ok':True,'result':result}
        except ValueError as error:
            return {'ok':False,'error':str(error)}
    nav = Mock()
    with patch(CLIENT,side_effect=service), patch('execution_grounding.send_action',side_effect=service), patch.object(world_utils,'get_node',return_value=Mock()), patch('execution_grounding.acquire_camera_geometry',return_value=CONTEXT) as geometry:
        if ambiguous:
            with pytest.raises(RuntimeError,match='camera grounding ambiguous'):
                execute_step({'action':'look-at','args':['robot','table_1']},nav,{},execution_oracle=oracle)
            assert [a for a,_ in calls] == ['get_object_pose','resolve_execution_instance']
            assert oracle.execution_bindings == {}
        else:
            assert execute_step({'action':'look-at','args':['robot','table_1']},nav,{},execution_oracle=oracle)
            assert [a for a,_ in calls] == ['get_object_pose','resolve_execution_instance','gaze']
            assert calls[0][1] == {'target':'TIAGo'}
            assert calls[2][1] == {'action':'look-at','optical_target':[0,0,2]}
            assert oracle.execution_bindings == {'table_1':'table(2)'}
        geometry.assert_called_once()
    assert not nav.mock_calls
    assert scene == before
    logs = capsys.readouterr().out
    assert 'Same-name simulator lookup: ignored' in logs
    assert 'Identity source: current VLM observation' in logs
    if not ambiguous:
        assert 'Head target: pan=0.2, tilt=0.1; Base motion: none' in logs


def test_single_same_name_still_requires_camera_visibility():
    world = supervisor(table(1,'table_1',(20,0,-2)))
    with patch.object(world_utils,'get_node',return_value=Mock()), patch.object(camera,'select_camera_candidate',wraps=camera.select_camera_candidate) as geometry:
        with pytest.raises(ValueError,match='no candidates inside FOV'):
            world_utils.resolve_execution_instance(world,'table_1','table',use_camera=True,camera_context=CONTEXT)
        geometry.assert_called_once()


def test_direct_lookup_requires_explicit_simulator_provenance():
    with patch(CLIENT,return_value={'ok':True,'result':{'position':[1,2,3]}}) as send:
        with pytest.raises(RuntimeError,match='perceived identity requires'):
            execution_target_position('table_1')
        send.assert_not_called()
        assert execution_target_position('table_1',identity_source='simulator') == [1,2]
        send.assert_called_once_with('get_object_pose',{'target':'table_1'})


def test_retained_only_cannot_be_executed_as_current_gaze():
    oracle = ExecutionGroundingOracle({'objects':[{'id':'table_1','type':'table'}]},knowledge())
    with patch(CLIENT) as send, patch('execution_grounding.send_action') as resolve:
        with pytest.raises(RuntimeError,match='not currently observed'):
            execution_target_position('table_1',oracle)
    send.assert_not_called()
    resolve.assert_not_called()
