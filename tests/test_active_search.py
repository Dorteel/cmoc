"""Active Search ranks current inspection targets without inventing locations."""
from copy import deepcopy
import json
from unittest.mock import Mock, patch

from search_strategy import select_candidate, search_frame, search_result
from semantic_memory import SemanticMemory


def setup():
    current = {'objects': [{'id': 'table_1', 'type': 'table'}, {'id': 'chair_1', 'type': 'chair'},
                           {'id': 'cabinet_1', 'type': 'cabinet'}], 'relations': []}
    retained = {'objects': current['objects'] + [{'id': 'dining_table_1', 'type': 'dining table'}], 'relations': []}
    kg, llm = Mock(), Mock()
    kg.resolve_concept.side_effect = lambda term: [{'id': 'fork.n.01'}] if term == 'fork' else []
    kg.get_locations.return_value = []
    llm.rank_gaze_targets.return_value = [{'location': 'dining_table_1', 'score': 9},
                                        {'location': 'table_1', 'score': 1}]
    return current, retained, kg, llm


def test_llm_receives_current_unchecked_only_and_cannot_select_retained_id():
    current, retained, kg, llm = setup()
    before = deepcopy((current, retained))
    selected = select_candidate({'Theme': 'fork'}, kg, llm, retained, {'cabinet_1'}, current=current)
    assert selected['location'] == 'table_1'
    assert [o['id'] for o in llm.rank_gaze_targets.call_args.args[1]] == ['table_1', 'chair_1']
    assert llm.rank_gaze_targets.call_args.kwargs['checked'] == {'cabinet_1'}
    assert (current, retained) == before
    assert not search_result(search_frame({'Agent': 'robot', 'Theme': 'fork'}, 'table_1'), current, kg)['Success']


def test_kg_prior_grounds_to_current_table_without_llm():
    current, retained, kg, llm = setup()
    kg.get_locations.return_value = ['table']
    assert select_candidate({'Theme': 'fork'}, kg, llm, retained, current=current)['location'] == 'table_1'
    llm.rank_gaze_targets.assert_not_called()


def test_direct_retained_theme_evidence_is_explicit_priority_exception():
    current, retained, kg, llm = setup()
    retained['objects'].append({'id': 'fork1', 'type': 'fork'})
    retained['relations'].append({'subject': 'fork1', 'predicate': 'on', 'object': 'dining_table_1'})
    assert select_candidate({'Theme': 'fork'}, kg, llm, retained, current=current)['location'] == 'dining_table_1'
    llm.rank_gaze_targets.assert_not_called()


def test_empty_current_view_does_not_expand_retained_pool():
    _, retained, kg, llm = setup()
    kg.get_locations.return_value = ['table']
    assert select_candidate({'Theme': 'fork'}, kg, llm, retained,
                            current={'objects': [], 'relations': []})['location'] is None
    llm.rank_gaze_targets.assert_not_called()


def test_inspection_prompt_and_result_make_no_location_hypothesis():
    current, _, _, _ = setup()
    response = Mock()
    response.json.return_value = {'message': {'content': json.dumps({
        'locations': [{'location': 'table_1', 'score': 1}]})}}
    with patch('semantic_memory.requests.post', return_value=response) as post:
        result = SemanticMemory().rank_gaze_targets('fork', current['objects'], checked={'cabinet_1'})
    prompt = post.call_args.kwargs['json']['messages'][0]['content']
    assert 'inspect next to gain evidence' in prompt
    assert 'table_1' in prompt and 'dining_table_1' not in prompt
    assert result == [{'location': 'table_1', 'score': 1}]
    assert 'locatedAt' not in repr(result)
