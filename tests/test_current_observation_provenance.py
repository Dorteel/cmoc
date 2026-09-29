"""Current provenance comes from actual PEL correspondence, never stored aliases."""
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from execution_grounding import ExecutionGroundingOracle
from perceived_entity_linking import perceived_entity_linking
from scene_graph_interface import KnowledgeInterface
from search_strategy import select_candidate, search_frame, search_result


def kg():
    result = Mock()
    result.resolve_concept.return_value = []
    return result


@pytest.mark.parametrize('raw,canonical', [('table_1', 'table_1'), ('mannequin_1', 'user')])
def test_current_pel_identity_enables_camera(raw, canonical, capsys):
    graph = {'objects': [{'id': raw, 'type': 'table' if raw == 'table_1' else 'mannequin'}], 'relations': []}
    before = deepcopy(graph)
    knowledge = kg()
    pel = perceived_entity_linking(graph, knowledge)
    assert pel['identity_mapping'] == {raw: canonical}
    oracle = ExecutionGroundingOracle(pel['scene_graph'], knowledge, debug=True,
                                      current_observed_ids=[raw],
                                      current_identity_mapping=pel['identity_mapping'])
    with patch('execution_grounding.acquire_camera_geometry', return_value={}), patch('execution_grounding.send_action', return_value={
        'ok': True, 'result': {'position':[0,0,0], 'optical_target':[0,0,2], 'target': 'physical', 'candidates': ['physical']}}) as query:
        oracle.resolve(canonical)
    assert query.call_args.args[1]['use_camera'] is True
    assert graph == before
    assert f'raw entity {raw} -> retained {canonical}' in capsys.readouterr().out


def test_retained_alias_and_old_mapping_do_not_establish_current_visibility():
    graph = {'objects': [{'id': 'dining_table_1', 'type': 'dining table', 'aliases': ['table_1']}],
             'relations': []}
    oracle = ExecutionGroundingOracle(graph, kg(), current_observed_ids=['new_table'],
                                      current_identity_mapping={'table_1': 'dining_table_1'})
    with patch('execution_grounding.acquire_camera_geometry', return_value={}), patch('execution_grounding.send_action', return_value={
        'ok': True, 'result': {'position':[0,0,0], 'optical_target':[0,0,2], 'target': 'table(1)', 'candidates': ['table(1)']}}) as query:
        oracle.resolve('dining_table_1')
    assert query.call_args.args[1]['use_camera'] is False


def test_saved_style_table_ids_have_no_invented_correspondence_or_knowledge_effect():
    memory = KnowledgeInterface()
    memory.merge_observation({'objects': [{'id': 'dining_table_1', 'type': 'dining table'}], 'relations': []})
    current = {'objects': [{'id': 'table_1', 'type': 'table'}], 'relations': []}
    memory.merge_observation(current)
    before = memory.observed_snapshot()
    knowledge = kg()
    pel = perceived_entity_linking(current, knowledge)
    assert pel['identity_mapping'] == {'table_1': 'table_1'}
    semantic = Mock()
    semantic.rank_gaze_targets.return_value = [{'location': 'dining_table_1', 'score': 1}]
    frame = {'Agent': 'robot', 'Theme': 'book'}
    selection = select_candidate(frame, knowledge, semantic, before)
    oracle = ExecutionGroundingOracle(before, knowledge, current_observed_ids=['table_1'],
                                      current_identity_mapping=pel['identity_mapping'])
    assert 'dining_table_1' not in oracle._current_observed_ids
    assert memory.observed_snapshot() == before
    assert select_candidate(frame, knowledge, semantic, before) == selection
    assert not search_result(search_frame(frame, 'dining_table_1'), current, knowledge)['Success']
    assert 'identity_mapping' not in memory.snapshot()


@pytest.mark.parametrize('case', ['same', 'unmatched', 'ambiguous', 'fresh_conflict', 'history_conflict', 'two_new_ids'])
def test_spa_reuses_only_unique_confirmed_webots_identity(case, capsys):
    import demo
    memory = KnowledgeInterface()
    first = {'objects': [{'id':'table', 'type':'table'}, {'id':'box', 'type':'box'}],
             'relations': [{'subject':'box', 'predicate':'on', 'object':'table'}]}
    second = deepcopy(first)
    second['objects'][0]['id'] = 'table_1'
    second['relations'][0]['object'] = 'table_1'
    if case == 'fresh_conflict':
        second['objects'].append({'id':'table', 'type':'table'})
    if case == 'history_conflict':
        first['objects'].append({'id':'other_table', 'type':'table'})
    if case == 'two_new_ids':
        second['objects'].append({'id':'table_2', 'type':'table'})
    original = deepcopy((first, second))
    views, oracles, choices = iter([first, second]), [], []
    llm = Mock()

    def choose(theme, current, options, checked, sweep_direction=None):
        choices.append((deepcopy(current), set(checked), deepcopy(options)))
        return {'action':'look-at', 'target':'table'} if len(choices) == 1 else {'action':'look-left'}

    def service(action, parameters):
        if action == 'get_object_pose':
            return {'ok':True, 'result':{'position':[0,0,0]}}
        identifier = parameters['perceived_id']
        if identifier == 'table_1' and case == 'ambiguous':
            return {'ok':False, 'error':'camera grounding ambiguous'}
        target = 'another table' if identifier == 'table_1' and case == 'unmatched' else 'round table'
        return {'ok':True, 'result':{'target':target, 'candidates':[target], 'optical_target':[0,0,2]}}

    def execute(planning, *args, execution_oracle, **kwargs):
        oracles.append(execution_oracle)
        if len(oracles) == 1:
            assert execution_oracle.resolve('table', require_current=True) == 'round table'
            if case == 'history_conflict':
                execution_oracle.resolve('other_table', require_current=True)
        return {'status':'success' if len(oracles) == 1 else 'dry_run',
                'plan':planning['plan'], 'executed':[]}

    llm.choose_gaze_action.side_effect = choose
    with patch('builtins.input', return_value='Bring me a fork'), \
         patch('demo.observe_scene_with_vlm', side_effect=lambda **kwargs:{'scene_graph':next(views)}), \
         patch('demo.execute_plan', side_effect=execute), \
         patch('execution_grounding.acquire_camera_geometry', return_value={}), \
         patch('execution_grounding.send_action', side_effect=service):
        result = demo.spa_loop(memory, kg(), llm, None, search=True, execute=True, debug_search=True)
    assert oracles[0] is oracles[1]
    expected = 'table' if case == 'same' else 'table_1'
    assert choices[1][0]['objects'][0]['id'] == expected
    assert choices[1][0]['relations'][0]['object'] == expected
    assert choices[1][1] == set()
    assert {'action': 'look-at', 'target': 'table'} not in choices[1][2]
    assert oracles[1]._current_identity_mapping['table_1'] == expected
    assert expected in oracles[1]._current_observed_ids
    assert oracles[1].execution_bindings == ({'table':'round table', 'other_table':'round table'}
                                            if case == 'history_conflict' else {'table':'round table'})
    assert ('table_1' in {o['id'] for o in memory.observed_snapshot()['objects']}) == (case != 'same')
    assert result['sense_graph']['scene_graph'] == second
    assert (first, second) == original
    assert ('[IDENTITY] table_1 -> round table -> table' in capsys.readouterr().out) == (case == 'same')
