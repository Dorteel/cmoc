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
