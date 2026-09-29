"""Fallback-only pruning; no simulator or LLM calls."""
from unittest.mock import Mock, patch
import json

from semantic_fallback import fallback_inputs, prune_fallback
from semantic_memory import SemanticMemory
from search_strategy import select_candidate


def obj(identifier, kind='table', **extra):
    return {'id': identifier, 'type': kind, **extra}


def test_filters_nonanchors_and_retains_surfaces_containers_rooms():
    objects = [obj('book_1', 'book'), obj('robot', 'Tiago'), obj('user', 'person'),
               obj('mannequin_1', 'mannequin'), obj('pants_1', 'pants'),
               obj('wall_1', 'wall'), obj('unknown', 'unknown'),
               obj('table_1'), obj('box_1', 'box'), obj('room_1', 'Location'),
               obj('platform_1', 'platform')]
    graph = {'objects': objects, 'relations': [
        {'subject': 'book_1', 'predicate': 'on', 'object': 'platform_1'}]}
    assert [o['id'] for o in fallback_inputs('book', graph, (), lambda _: None)] == [
        'platform_1', 'room_1']


def test_positive_scores_and_cap():
    objects = [obj(f'table_{i}') for i in range(9)]
    scores = [{'location': o['id'], 'score': i} for i, o in enumerate(objects)]
    scores += [{'location': 'table_0', 'score': -1}, {'location': 'table_0'},
               {'location': 'table_0', 'score': float('nan')}]
    kept = prune_fallback(scores, objects, lambda _: None)
    assert [r['location'] for r in kept] == ['table_8', 'table_7', 'table_6', 'table_5', 'table_4']


def test_alias_family_prefers_concrete_but_distinct_instances_survive():
    objects = [obj('table'), obj('table_1'), obj('table_001'),
               obj('dining_table_1', aliases=['table_1']), obj('object_5')]
    ranked = [{'location': o['id'], 'score': 1} for o in objects]
    kept = prune_fallback(ranked, objects, lambda _: None)
    assert [r['location'] for r in kept] == ['table_1', 'object_5']
    # Same semantic category is not evidence that two numbered tables are one.
    distinct = [obj('table_1'), obj('table_2'), obj('dining_table_1')]
    assert len(prune_fallback([{'location': o['id'], 'score': 1} for o in distinct],
                              distinct, lambda _: None)) == 3


def test_zero_score_cannot_be_selected_and_is_logged():
    kg, semantic = Mock(), Mock()
    kg.resolve_concept.return_value = []
    semantic.rank_gaze_targets.return_value = [
        {'location': 'table_1', 'score': 0}, {'location': 'shelf_1', 'score': .8}]
    graph = {'objects': [obj('table_1'), obj('shelf_1', 'shelf')], 'relations': [
        {'subject': 'table_1', 'predicate': 'next_to', 'object': 'shelf_1'},
        {'subject': 'shelf_1', 'predicate': 'next_to', 'object': 'table_1'}]}
    assert select_candidate({'Theme': 'book'}, kg, semantic, graph)['location'] == 'shelf_1'


def test_llm_ranks_observed_types_not_generic_instance_ids():
    response = Mock()
    response.json.return_value = {'message': {'content': json.dumps({
        'locations': [{'location': 'table', 'score': 1.0}]})}}
    with patch('semantic_memory.requests.post', return_value=response) as post:
        ranked = SemanticMemory().rank_locations('book', [obj('object_5')])
    assert ranked[0]['location'] == 'object_5' and ranked[0]['score'] == 1
    prompt = post.call_args.kwargs['json']['messages'][0]['content']
    assert "Possible locations: ['table']" in prompt


def test_current_entities_are_passed_to_inspection_llm():
    kg, llm = Mock(), Mock()
    kg.resolve_concept.return_value = []
    objects = [obj('cup', 'cup'), obj('table_1'), obj('cabinet_1', 'cabinet'),
               obj('room_1', 'Location'), obj('unused_table'), obj('mannequin_1', 'mannequin'),
               obj('shirt_1', 'shirt'), obj('window_1', 'window')]
    graph = {'objects': objects, 'relations': [
        {'subject': 'cup', 'predicate': 'on', 'object': 'table_1'},
        {'subject': 'cup', 'predicate': 'in', 'object': 'cabinet_1'},
        {'subject': 'shirt_1', 'predicate': 'on', 'object': 'table_1'},
    ]}
    llm.rank_gaze_targets.return_value = [{'location': 'table_1', 'score': 1}]
    assert select_candidate({'Theme': 'fork'}, kg, llm, graph)['location'] == 'table_1'
    assert [o['id'] for o in llm.rank_gaze_targets.call_args.args[1]] == ['cup', 'table_1', 'cabinet_1', 'room_1', 'unused_table', 'shirt_1', 'window_1']


def test_schema_spatial_vocabulary_drives_anchors_without_type_restrictions():
    from semantic_fallback import SPATIAL_RELATIONS
    for relation in SPATIAL_RELATIONS:
        graph = {'objects': [obj('source', 'cup'), obj('window_1', 'window')],
                 'relations': [{'subject': 'source', 'predicate': relation, 'object': 'window_1'}]}
        assert [o['id'] for o in fallback_inputs('fork', graph, (), lambda _: None)] == ['window_1']


def test_current_entities_need_no_spatial_relation_to_be_inspected():
    kg, llm = Mock(), Mock()
    kg.resolve_concept.return_value = []
    graph = {'objects': [obj('table_1'), obj('window_1', 'window')], 'relations': []}
    llm.rank_gaze_targets.return_value = [{'location': 'table_1', 'score': 1}]
    assert select_candidate({'Theme': 'fork'}, kg, llm, graph)['location'] == 'table_1'
    llm.rank_gaze_targets.assert_called_once()


def test_stronger_sources_bypass_structural_fallback():
    # An unconnected table remains eligible for KG/VLM suggestions, as before.
    kg, llm = Mock(), Mock()
    graph = {'objects': [obj('table_1'), obj('fork_1', 'fork')], 'relations': []}
    kg.resolve_concept.return_value = [{'id': 'fork.n.01'}]
    kg.get_locations.return_value = [('table_1', 1)]
    assert select_candidate({'Theme': 'fork'}, kg, llm, graph)['location'] == 'table_1'
    kg.get_locations.return_value = []
    assert select_candidate({'Theme': 'fork'}, kg, llm, graph, suggestions=['table_1'])['location'] == 'table_1'
    graph['relations'] = [{'subject': 'fork_1', 'predicate': 'on', 'object': 'table_1'}]
    assert select_candidate({'Theme': 'fork'}, kg, llm, graph)['location'] == 'table_1'
    llm.rank_gaze_targets.assert_not_called()
