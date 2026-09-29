"""LLM gaze decisions, head-only execution, and active Search lifecycle."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import math
import pytest
from jsonschema import validate
from search_strategy import choose_gaze, gaze_key, search_frame
from procedural_memory.planning.search_plan import plan_search
from plan_execution import execute_step

CURRENT = {'objects': [{'id': 'table_1', 'type': 'table'}, {'id': 'chair_1', 'type': 'chair'}], 'relations': []}
ACTIONS = [{'action': 'look-at', 'target': 'table_1'}, {'action': 'look-left'}, {'action': 'look-right'}]

@pytest.mark.parametrize('choice', ACTIONS)
def test_choices_current_only_and_directions_always_offered(choice):
    llm = Mock(choose_gaze_action=Mock(return_value=choice))
    before = deepcopy(CURRENT)
    assert choose_gaze({'Theme': 'fork'}, CURRENT, llm) == choice
    options = llm.choose_gaze_action.call_args.args[2]
    assert options == [ACTIONS[0], {'action': 'look-at', 'target': 'chair_1'}, *ACTIONS[1:]]
    assert CURRENT == before

@pytest.mark.parametrize('choice', [{'action':'navigate'}, {'action':'look-at','target':'retained_1'},
                                    {'action':'look-left','target':'table_1'}, {}, None])
def test_invalid_llm_choices_rejected(choice):
    with pytest.raises(ValueError, match='Invalid LLM'):
        choose_gaze({'Theme':'fork'}, CURRENT, Mock(choose_gaze_action=Mock(return_value=choice)))

@pytest.mark.parametrize('choice', ACTIONS)
def test_checked_action_rejected(choice):
    with pytest.raises(ValueError, match='repeated checked'):
        choose_gaze({'Theme':'fork'}, CURRENT, Mock(choose_gaze_action=Mock(return_value=choice)), {gaze_key(choice)})

@pytest.mark.parametrize('choice', ACTIONS)
def test_real_planner_and_schemas(choice):
    frame = search_frame({'Agent':'robot', 'Theme':'fork'}, choice.get('target'))
    if choice['action'] != 'look-at':
        frame['GazeAction'] = choice['action']
    validate(frame, json.loads(Path('schemas/task_frames/search.json').read_text()))
    result = plan_search(frame)
    assert result['status'] == 'planned'
    assert result['plan'] == [{'action':choice['action'], 'args':['robot'] + ([choice['target']] if 'target' in choice else [])}]
    assert 'Success' not in frame

@pytest.mark.parametrize('action', ['look-left','look-right'])
def test_direction_execution_has_no_navigation_or_ground_truth_lookup(action):
    nav, oracle = Mock(), Mock()
    with patch('external.webots_ros2_simulation.controllers.fallback_action_supervisor.action_cli.send_action', return_value={'ok':True}) as send:
        assert execute_step({'action':action,'args':['robot']},nav,{},execution_oracle=oracle)
    send.assert_called_once_with('gaze', {'action':action})
    assert not nav.mock_calls and not oracle.mock_calls

@pytest.fixture
def head():
    path = Path('external/webots_ros2_simulation/controllers/fallback_action_supervisor/head_gaze.py')
    spec = importlib.util.spec_from_file_location('head_gaze_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'world_utils':Mock(), 'camera_grounding':Mock()}):
        spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('action,expected', [('look-left',math.pi/6),('look-right',-math.pi/6)])
def test_head_relative_pan_only_and_clamps(head, action, expected):
    position = NS(value=0.0)
    field = Mock(getSFFloat=lambda:position.value)
    joint = Mock()
    joint.setJointPosition.side_effect = lambda value,index:setattr(position,'value',value)
    tilt = Mock()
    head.head_joints = Mock(return_value={'head_1_joint':(joint,field,-0.7,0.7), 'head_2_joint':(tilt,Mock(),-0.5,0.5)})
    supervisor = Mock()
    head.gaze(supervisor,action)
    assert position.value == pytest.approx(expected)
    head.gaze(supervisor,action)
    assert position.value == pytest.approx(0.7 if expected>0 else -0.7)
    with pytest.raises(ValueError,match='limit reached'):
        head.gaze(supervisor,action)
    assert not tilt.mock_calls and not supervisor.mock_calls
    head.get_node.return_value.setPosition.assert_not_called()
    head.get_node.return_value.setOrientation.assert_not_called()

@pytest.mark.parametrize('choice', ACTIONS)
def test_fresh_sense_after_each_gaze_no_fabricated_relations(choice):
    import demo
    from scene_graph_interface import KnowledgeInterface
    from test_search_recovery import scene, kg
    memory = KnowledgeInterface()
    initial = scene()
    initial['objects'].append({'id':'table_1','type':'table','qualities':{}})
    found = scene(True)
    events=[]
    views=iter([initial, initial, found])
    llm=Mock()
    llm.choose_gaze_action.side_effect=[choice, {'action':'look-at','target':'drawer1'}]
    def sense(**kwargs):
        events.append('sense')
        return {'scene_graph':deepcopy(next(views))}
    def execute(planning,*args,**kwargs):
        events.append(planning['plan'][0]['action'])
        return {'status':'success','plan':planning['plan'],'executed':[]}
    with patch('builtins.input',return_value=''), patch('demo.observe_scene_with_vlm',side_effect=sense), patch('demo.execute_plan',side_effect=execute), patch('demo.plan_bring',return_value={'status':'planned','plan':[{'action':'pick','args':['robot','fork1']}]}):
        result=demo.spa_loop(memory,kg(),llm,None,search=True,execute=True)
    assert events == ['sense',choice['action'],'sense','look-at','sense','pick']
    assert [o['Success'] for o in result['search_outcomes']] == [False,True]
    assert gaze_key(choice) in llm.choose_gaze_action.call_args.kwargs['checked']
    assert [r for r in memory.observed_snapshot()['relations'] if r['subject']=='fork1'] == found['relations'][-1:]
    assert result['search_outcomes'][0]['Location'] == choice.get('target')
    for outcome in result['search_outcomes']:
        validate(outcome, json.loads(Path('schemas/task_frames/search_result.json').read_text()))


def test_look_at_orients_head_toward_camera_target_without_base_motion(head):
    positions = {}
    joints = {}
    for name in ('head_1_joint', 'head_2_joint'):
        positions[name] = 0.0
        field = Mock(getSFFloat=lambda n=name:positions[n])
        joint = Mock()
        joint.setJointPosition.side_effect = lambda value,index,n=name:positions.__setitem__(n,value)
        joints[name] = (joint,field,-1.0,1.0)
    head.head_joints = Mock(return_value=joints)
    head.validate_coordinates = lambda p:p
    supervisor = Mock()
    head.gaze(supervisor,'look-at',[1,-1,3])
    assert positions['head_1_joint'] == pytest.approx(-math.atan2(1,3))
    assert positions['head_2_joint'] == pytest.approx(math.atan2(1,math.sqrt(10)))
    assert not supervisor.mock_calls
    head.get_node.return_value.setPosition.assert_not_called()
    head.get_node.return_value.setOrientation.assert_not_called()


def test_llm_prompt_is_one_gaze_decision():
    from semantic_memory import SemanticMemory
    llm = SemanticMemory.__new__(SemanticMemory)
    llm.url, llm.model = 'http://test', 'test'
    response = Mock()
    response.json.return_value = {'message':{'content':'{"action":"look-left"}'}}
    with patch('semantic_memory.requests.post',return_value=response) as post:
        assert llm.choose_gaze_action('fork',CURRENT,ACTIONS,checked={'look-right'}) == {'action':'look-left'}
    prompt = post.call_args.kwargs['json']['messages'][0]['content']
    assert 'Theme: fork' in prompt and 'Already checked: ["look-right"]' in prompt
    assert 'gain new visual evidence' in prompt
