"""Fresh Theme visibility ends Search and resumes the original Bring pipeline."""
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

import demo
from scene_graph_interface import KnowledgeInterface
from test_search_recovery import BRING, kg, scene


def test_visible_theme_with_unresolved_binding_calls_bring_not_gaze():
    fresh = scene(True)
    # Visibility alone cannot manufacture a graspable execution instance.
    fresh['objects'][-1]['type'] = 'fork'
    with patch('builtins.input', return_value='Bring me a fork'), \
         patch('demo.observe_scene_with_vlm', return_value={'scene_graph': fresh}), \
         patch('demo.choose_gaze') as gaze, patch('demo.plan_search') as search_plan, \
         patch('demo.perceived_entity_linking', wraps=demo.perceived_entity_linking) as pel, \
         patch('demo.plan_bring', wraps=demo.plan_bring) as bring:
        result = demo.spa_loop(KnowledgeInterface(), kg(), Mock(), None, search=True)
    assert result['bindings']['Theme'] is None
    assert result['bring_intention'] == BRING
    assert result['sense_graph']['scene_graph'] == fresh
    assert pel.call_args_list[0].args[0] == fresh
    assert fresh['objects'][-1] in bring.call_args.args[1]['objects']
    bring.assert_called_once()
    gaze.assert_not_called()
    search_plan.assert_not_called()


def test_search_success_binds_fresh_fork_and_executes_bring_in_same_spa_run():
    memory, llm = KnowledgeInterface(), Mock()
    missing, found = scene(), scene(True)
    llm.choose_gaze_action.return_value = {'action': 'look-left'}
    views = iter([missing, found, found])  # Sense, post-turn Sense, before-pick verification.
    events = []
    def observe(**kwargs):
        events.append('sense')
        return {'scene_graph': deepcopy(next(views))}
    def dispatch(step, *args, **kwargs):
        events.append(step['action'])
        return True
    with patch('builtins.input', return_value='Bring me a fork') as instruction, \
         patch('demo.observe_scene_with_vlm', side_effect=observe), \
         patch('demo.plan_search', wraps=demo.plan_search) as search_plan, \
         patch('demo.plan_bring', wraps=demo.plan_bring) as bring, \
         patch('plan_execution.execute_step', side_effect=dispatch) as execute:
        result = demo.spa_loop(memory, kg(), llm, Mock(), search=True, execute=True)
    instruction.assert_called_once()
    llm.choose_gaze_action.assert_called_once()
    search_plan.assert_called_once()
    bring.assert_called_once()
    assert result['bring_intention'] == BRING
    assert result['bindings']['Theme'] == 'fork1'
    assert result['bindings']['Source'] == 'KITCHEN'
    assert result['bindings']['Destination'] == 'user'
    assert bring.call_args.args[0] == result['bindings']
    assert found['objects'][-1] in bring.call_args.args[1]['objects']
    assert result['search_outcomes'][0]['Success'] is True
    assert result['planning']['status'] == 'planned'
    assert result['planning']['plan']
    assert result['action_graph']['status'] == 'success'
    actions = [call.args[0] for call in execute.call_args_list]
    assert actions[0]['action'] == 'look-left'
    assert [step['action'] for step in actions[1:]] == ['pick', 'navigate', 'place']
    assert all('fork1' in step['args'] for step in actions if step['action'] in ('pick', 'place'))
    assert events[:3] == ['sense', 'look-left', 'sense']
    assert events[-1] == 'place'


@pytest.mark.parametrize('retained', [False, True])
def test_absent_current_theme_continues_search_even_if_retained(retained):
    memory = KnowledgeInterface()
    if retained:
        memory.merge_observation(scene(True))
    llm = Mock(choose_gaze_action=Mock(return_value={'action': 'look-right'}))
    with patch('builtins.input', return_value='Bring me a fork'), \
         patch('demo.observe_scene_with_vlm', return_value={'scene_graph': scene()}), \
         patch('demo.plan_bring') as bring:
        result = demo.spa_loop(memory, kg(), llm, None, search=True)
    llm.choose_gaze_action.assert_called_once()
    bring.assert_not_called()
    assert result['type'] == 'search'
    assert result['frame']['GazeAction'] == 'look-right'


@pytest.mark.parametrize('visible_recovery', [False, True])
def test_visible_recovery_resumes_bring_and_search_state_resets(visible_recovery):
    memory, llm = KnowledgeInterface(), Mock()
    # Begin a sweep, inspect an object, find the fork, then lose it before pick.
    # A visible next Sense must resume Bring despite recovering=True, without gaze.
    views = iter([scene(), scene(), scene(), scene(True), scene(),
                  *([scene()] if not visible_recovery else []), scene(True), scene(True)])
    llm.choose_gaze_action.side_effect = [
        {'action': 'look-left'}, {'action': 'look-at', 'target': 'worktop(1)'}, {'action': 'look-left'},
        *([{'action': 'look-right'}] if not visible_recovery else [])]
    gaze_states = []
    original_gaze = demo.choose_gaze
    def choose(*args, **kwargs):
        gaze_states.append((set(args[3]), kwargs['sweep_direction']))
        return original_gaze(*args, **kwargs)
    with patch('builtins.input', return_value='Bring me a fork'), \
         patch('demo.observe_scene_with_vlm', side_effect=lambda **kwargs: {'scene_graph': deepcopy(next(views))}), \
         patch('demo.choose_gaze', side_effect=choose), \
         patch('demo.plan_bring', wraps=demo.plan_bring) as bring, \
         patch('plan_execution.execute_step', return_value=True):
        result = demo.spa_loop(memory, kg(), llm, Mock(), search=True, execute=True)
    assert gaze_states == [(set(), None), (set(), 'left'), ({'look-at(worktop(1))'}, 'left'),
                           *([(set(), None)] if not visible_recovery else [])]
    assert llm.choose_gaze_action.call_count == (3 if visible_recovery else 4)
    assert bring.call_count == 2
    assert result['action_graph']['status'] == 'success'
    assert result['bindings']['Theme'] == 'fork1'
    assert result['bring_intention'] == BRING
