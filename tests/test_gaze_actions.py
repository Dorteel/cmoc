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

def test_checked_objects_filtered_but_directions_always_available():
    current = {'objects': [{'id': 'table', 'type': 'table'}, {'id': 'chair', 'type': 'chair'}]}
    remaining = [{'action': 'look-at', 'target': 'chair'}, {'action': 'look-left'}, {'action': 'look-right'}]
    llm = Mock(choose_gaze_action=Mock(return_value=remaining[0]))
    checked = {'look-at(table)'}
    assert choose_gaze({'Theme': 'fork'}, current, llm, checked) == remaining[0]
    assert llm.choose_gaze_action.call_args.args[2] == remaining
    llm.choose_gaze_action.return_value = {'action': 'look-left'}
    choose_gaze({'Theme': 'fork'}, current, llm, checked | {gaze_key(option) for option in remaining})
    assert llm.choose_gaze_action.call_args.args[2] == remaining[1:]

@pytest.mark.parametrize('choice', [{'action':'navigate'}, {'action':'look-at','target':'retained_1'},
                                    {'action':'look-left','target':'table_1'}, {}, None])
def test_invalid_llm_choices_rejected(choice):
    with pytest.raises(ValueError, match='Invalid LLM'):
        choose_gaze({'Theme':'fork'}, CURRENT, Mock(choose_gaze_action=Mock(return_value=choice)))

@pytest.mark.parametrize('choice', ACTIONS[:1])
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

@pytest.mark.parametrize('kind', ['floor', 'FloorConnector', 'ground surface'])
def test_floor_region_bypasses_instance_grounding(kind, capsys):
    from execution_grounding import ExecutionGroundingOracle
    scene = {'objects': [{'id': 'region_1', 'type': kind}], 'relations': []}
    before = deepcopy(scene)
    kg = Mock()
    kg.resolve_concept.side_effect = lambda term: [{'id': 'floor.n.01'}] if term in ('floor', 'ground surface') else []
    oracle = ExecutionGroundingOracle(scene, kg, current_observed_ids=['region_1'])
    nav = Mock()
    with patch.object(oracle, 'resolve') as resolve, \
         patch('execution_grounding.send_action') as geometry, \
         patch('external.webots_ros2_simulation.controllers.fallback_action_supervisor.action_cli.send_action',
               return_value={'ok': True}) as send:
        assert execute_step({'action': 'look-at', 'args': ['robot', 'region_1']}, nav, {}, execution_oracle=oracle)
    send.assert_called_once_with('gaze', {'action': 'look-down'})
    resolve.assert_not_called()
    geometry.assert_not_called()
    assert not nav.mock_calls
    assert oracle.execution_bindings == oracle.optical_targets == {}
    assert scene == before
    assert 'classified as gaze region: floor' in capsys.readouterr().out


def test_floor_region_requires_current_type_evidence():
    from execution_grounding import ExecutionGroundingOracle
    kg = Mock(resolve_concept=Mock(return_value=[]))
    oracle = ExecutionGroundingOracle({'objects': [{'id': 'floor_1', 'type': 'table'},
                                                   {'id': 'old_floor', 'type': 'floor'}]},
                                     kg, current_observed_ids=['floor_1'])
    assert oracle.gaze_region('floor_1') is None
    assert oracle.gaze_region('old_floor') is None
    assert oracle.gaze_region('missing') is None


@pytest.fixture
def head():
    path = Path('external/webots_ros2_simulation/controllers/fallback_action_supervisor/head_gaze.py')
    spec = importlib.util.spec_from_file_location('head_gaze_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'world_utils':Mock(), 'camera_grounding':Mock()}):
        spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('action,expected', [('look-left', math.pi/3), ('look-right', -math.pi/3)])
def test_direction_rotates_base_without_translation_and_centers_head(head, action, expected):
    from test_search_execution_grounding import world_utils
    yaw = NS(value=0.4)
    positions = {'head_1_joint': 0.3, 'head_2_joint': -0.5}
    robot = head.get_node.return_value
    robot.getOrientation.side_effect = lambda: [math.cos(yaw.value), -math.sin(yaw.value), 0,
                                                math.sin(yaw.value), math.cos(yaw.value), 0, 0, 0, 1]
    rotation, translation = Mock(), Mock()
    robot.getField.side_effect = lambda name: {'rotation': rotation, 'translation': translation}[name]
    rotation.setSFRotation.side_effect = lambda value: setattr(yaw, 'value', value[3])
    head.set_yaw = world_utils.set_yaw
    joints = {}
    for name in positions:
        field = Mock(getSFFloat=lambda n=name: positions[n])
        joint = Mock()
        joint.setJointPosition.side_effect = lambda value, index, n=name: positions.__setitem__(n, value)
        joints[name] = (joint, field, -0.7, 0.7)
    head.head_joints = Mock(return_value=joints)
    supervisor = Mock(getBasicTimeStep=Mock(return_value=20))
    for _ in range(3):
        previous = yaw.value
        result = head.gaze(supervisor, action)
        delta = yaw.value - previous
        assert math.atan2(math.sin(delta), math.cos(delta)) == pytest.approx(expected)
        assert result['base_yaw'] == yaw.value
        assert positions == {'head_1_joint': 0, 'head_2_joint': 0}
    assert supervisor.step.call_count == 3
    assert not translation.mock_calls
    assert all(call.args == ('rotation',) for call in robot.getField.call_args_list)
    robot.setPosition.assert_not_called()

def test_floor_head_tilt_preserves_pan_and_clamps(head):
    positions = {'head_1_joint': 0.3, 'head_2_joint': 0.0}
    joints = {}
    for name, limits in [('head_1_joint', (-1.24, 1.24)), ('head_2_joint', (-.98, .79))]:
        field = Mock(getSFFloat=lambda n=name: positions[n])
        joint = Mock()
        joint.setJointPosition.side_effect = lambda value, index, n=name: positions.__setitem__(n, value)
        joints[name] = (joint, field, *limits)
    head.head_joints = Mock(return_value=joints)
    supervisor = Mock(getBasicTimeStep=Mock(return_value=20))
    head.gaze(supervisor, 'look-down')
    assert positions['head_2_joint'] == pytest.approx(-math.pi / 6)
    head.gaze(supervisor, 'look-down')
    assert positions['head_2_joint'] == -.98
    with pytest.raises(ValueError, match='Head tilt limit reached'):
        head.gaze(supervisor, 'look-down')
    assert positions['head_1_joint'] == 0.3
    joints['head_1_joint'][0].setJointPosition.assert_not_called()
    head.validate_coordinates.assert_not_called()
    head.get_node.return_value.setPosition.assert_not_called()
    head.get_node.return_value.setOrientation.assert_not_called()
    assert supervisor.step.call_count == 2


@pytest.mark.parametrize('choice', [*ACTIONS, {'action': 'look-at', 'target': 'floor_1'}])
def test_fresh_sense_after_each_gaze_no_fabricated_relations(choice, capsys):
    import demo
    from scene_graph_interface import KnowledgeInterface
    from test_search_recovery import scene, kg
    memory = KnowledgeInterface()
    initial = scene()
    initial['objects'].append({'id':'table_1','type':'table','qualities':{}})
    initial['objects'].append({'id':'floor_1','type':'floor','qualities':{}})
    found = scene(True)
    events=[]
    views=iter([initial, initial, found])
    llm=Mock()
    llm.choose_gaze_action.side_effect=[{**choice, 'reason':'Inspecting this view may reveal the fork.'}, {'action':'look-at','target':'drawer1', 'reason':'The drawer may provide new evidence.'}]
    def sense(**kwargs):
        events.append('sense')
        return {'scene_graph':deepcopy(next(views))}
    def execute(planning,*args,**kwargs):
        events.append(planning['plan'][0]['action'])
        return {'status':'success','plan':planning['plan'],'executed':[]}
    with patch('builtins.input',return_value=''), patch('demo.observe_scene_with_vlm',side_effect=sense), patch('demo.execute_plan',side_effect=execute), patch('demo.plan_bring',return_value={'status':'planned','plan':[{'action':'pick','args':['robot','fork1']}]}):
        result=demo.spa_loop(memory,kg(),llm,None,search=True,execute=True)
    assert events == ['sense',choice['action'],'sense','look-at','sense','pick']
    output = capsys.readouterr().out
    stages = ['SPA LOOP 1', '[SENSE] Starting perception', '[SENSE] Observation received',
              '[PLAN] Building task frame', "[SENSE] Theme 'fork' not observed",
              '[SEARCH] Available gaze actions:', '[SEARCH] LLM selected:',
              '[SEARCH] Reason:', '[PLAN] Generating Search plan',
              'SPA LOOP 2', '[SENSE] Starting fresh perception']
    offsets = [output.index(stage) for stage in stages]
    assert offsets == sorted(offsets)
    assert 'PEL:' not in output  # Detailed graph diagnostics require debug mode.

    assert [o['Success'] for o in result['search_outcomes']] == [False,True]
    assert llm.choose_gaze_action.call_args.kwargs['checked'] == ()
    if choice['action'] == 'look-at':
        assert gaze_key(choice) not in {gaze_key(option) for option in llm.choose_gaze_action.call_args.args[2]}
    else:
        assert choice in llm.choose_gaze_action.call_args.args[2]
    assert [r for r in memory.observed_snapshot()['relations'] if r['subject']=='fork1'] == found['relations'][-1:]
    assert result['search_outcomes'][0]['Location'] == choice.get('target')
    assert 'reason' not in json.dumps(memory.snapshot())
    assert 'reason' not in json.dumps(result['search_outcomes'])
    assert 'Inspecting this view' not in json.dumps(result)
    assert 'The drawer may provide' not in json.dumps(result)
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
    supervisor = Mock(getBasicTimeStep=Mock(return_value=20))
    head.gaze(supervisor,'look-at',[1,-1,3])
    assert positions['head_1_joint'] == pytest.approx(-math.atan2(1,3))
    assert positions['head_2_joint'] == pytest.approx(math.atan2(1,math.sqrt(10)))
    head.head_joints.assert_called_once_with(head.get_node.return_value)
    supervisor.step.assert_called_once()
    head.get_node.return_value.setPosition.assert_not_called()
    head.get_node.return_value.setOrientation.assert_not_called()


def test_llm_prompt_is_one_gaze_decision():
    from semantic_memory import SemanticMemory
    llm = SemanticMemory.__new__(SemanticMemory)
    llm.url, llm.model = 'http://test', 'test'
    response = Mock()
    response.json.return_value = {'message':{'content':'{"action":"look-left","reason":"Looking left may reveal new objects."}'}}
    with patch('semantic_memory.requests.post',return_value=response) as post:
        assert llm.choose_gaze_action('fork',CURRENT,ACTIONS,checked={'look-right'}) == {'action':'look-left', 'reason':'Looking left may reveal new objects.'}
    prompt = post.call_args.kwargs['json']['messages'][0]['content']
    assert 'Theme: fork' in prompt and 'checked' not in prompt
    assert 'gain new visual evidence' in prompt


def configured_head_tree(missing=None):
    """Webots PROTO internals have ID -1; exposed/expanded fields can overlap."""
    def node(kind, fields=None, proto=False, node_id=-1):
        fields = fields or {}
        values = list(fields.values())
        return Mock(getId=Mock(return_value=node_id), getTypeName=lambda: kind,
                    isProto=lambda: proto, getField=lambda key: fields.get(key),
                    getNumberOfFields=lambda: len(values),
                    getFieldByIndex=lambda i: values[i],
                    getNumberOfBaseNodeFields=lambda: len(values),
                    getBaseNodeFieldByIndex=lambda i: values[i])

    def nodes(children):
        return Mock(getTypeName=lambda: 'MFNode', getCount=lambda: len(children),
                    getMFNode=lambda i: children[i])

    def single(child):
        return Mock(getTypeName=lambda: 'SFNode', getSFNode=lambda: child)

    joints = {}
    for name, limits in [('head_1_joint', (-1.24, 1.24)), ('head_2_joint', (-.98, .79))]:
        if name == missing:
            continue
        motor = node('RotationalMotor', {
            'name': Mock(getTypeName=lambda: 'SFString', getSFString=lambda n=name: n),
            'minPosition': Mock(getTypeName=lambda: 'SFFloat', getSFFloat=lambda v=limits[0]: v),
            'maxPosition': Mock(getTypeName=lambda: 'SFFloat', getSFFloat=lambda v=limits[1]: v)})
        position = Mock(getTypeName=lambda: 'SFFloat', getSFFloat=lambda: .2)
        params = node('HingeJointParameters', {'position': position})
        joints[name] = node('HingeJoint', {'device': nodes([motor]),
                                         'jointParameters': single(params)})
    # Traversal visits Physics first, reproducing the live failure.
    robot = node('Tiago++', {'children': nodes([*joints.values(), node('Physics')])},
                 proto=True, node_id=2413)
    return robot, joints


def test_discovers_configured_joints_with_shared_internal_minus_one_id(head):
    import xml.etree.ElementTree as ET
    robot, expected = configured_head_tree()
    found = head.head_joints(robot)
    config = ET.parse('external/webots_ros2_simulation/config/tiago_webots_wheels.urdf')
    configured = {j.attrib['name'] for j in config.findall('.//ros2_control/joint')
                  if j.attrib['name'].startswith('head_')}
    assert set(found) == configured == {head.PAN_JOINT, head.TILT_JOINT}
    robot.getId.assert_not_called()
    for name, joint in expected.items():
        assert found[name][0] is joint
        joint.getId.assert_not_called()
    assert found[head.PAN_JOINT][2:] == (-1.24, 1.24)
    assert found[head.TILT_JOINT][2:] == (-.98, .79)


@pytest.mark.parametrize('missing,role', [('head_1_joint', 'pan'), ('head_2_joint', 'tilt')])
def test_missing_head_joint_reports_actual_interface(head, missing, role):
    robot, _ = configured_head_tree(missing)
    with pytest.raises(ValueError, match=f"Head {role} joint '{missing}'.*Webots"):
        head.head_joints(robot)


@pytest.mark.parametrize('direction', ['left', 'right'])
def test_sweep_is_only_a_preference_and_directions_remain_repeatable(direction):
    llm = Mock(choose_gaze_action=Mock(return_value={'action': f'look-{direction}'}))
    checked = {'look-at(table_1)', 'look-left', 'look-right'}
    for _ in range(3):
        assert choose_gaze({'Theme': 'fork'}, CURRENT, llm, checked, sweep_direction=direction) == {
            'action': f'look-{direction}'}
        assert llm.choose_gaze_action.call_args.args[2] == [
            {'action': 'look-at', 'target': 'chair_1'}, {'action': 'look-left'}, {'action': 'look-right'}]
        assert llm.choose_gaze_action.call_args.kwargs['sweep_direction'] == direction


def test_spa_sweep_survives_object_gaze_and_resets_for_new_search():
    import demo
    from scene_graph_interface import KnowledgeInterface
    from test_search_recovery import scene, kg
    events, offered, preferences = [], [], []
    views = iter([scene(), scene(), scene(), scene(), scene(True), scene(), scene(True)])
    choices = iter([{'action': 'look-left'}, {'action': 'look-at', 'target': 'worktop(1)'},
                    {'action': 'look-left'}, {'action': 'look-left'}, {'action': 'look-right'}])
    llm = Mock()
    def choose(theme, current, options, checked, sweep_direction=None):
        offered.append(deepcopy(options))
        preferences.append(sweep_direction)
        return next(choices)
    llm.choose_gaze_action.side_effect = choose
    def sense(**kwargs):
        events.append('sense')
        return {'scene_graph': next(views)}
    bring_count = 0
    def execute(planning, *args, **kwargs):
        nonlocal bring_count
        action = planning['plan'][0]['action']
        events.append(action)
        if action == 'pick':
            bring_count += 1
        return {'status': 'search_required' if action == 'pick' and bring_count == 1 else 'success',
                'plan': planning['plan'], 'executed': []}
    with patch('builtins.input', return_value='Bring me a fork'), \
         patch('demo.observe_scene_with_vlm', side_effect=sense), \
         patch('demo.execute_plan', side_effect=execute), \
         patch('demo.plan_bring', return_value={'status': 'planned', 'plan': [{'action': 'pick', 'args': ['robot', 'fork1']}]}):
        result = demo.spa_loop(KnowledgeInterface(), kg(), llm, None, execute=True, search=True)
    left, right = {'action': 'look-left'}, {'action': 'look-right'}
    assert left in offered[0] and right in offered[0]
    for options in offered[1:4]:
        assert left in options and right in options
        assert {'action': 'look-at', 'target': 'drawer1'} in options
    assert {'action': 'look-at', 'target': 'worktop(1)'} not in offered[2]
    assert left in offered[4] and right in offered[4]
    assert preferences == [None, 'left', 'left', 'left', None]
    assert {'action': 'look-at', 'target': 'worktop(1)'} in offered[4]
    assert events == ['sense', 'look-left', 'sense', 'look-at', 'sense', 'look-left',
                      'sense', 'look-left', 'sense', 'pick', 'sense', 'look-right', 'sense', 'pick']
    assert result['action_graph']['status'] == 'success'
