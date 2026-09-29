"""Tracing must explain existing selection without changing it."""
from unittest.mock import Mock

from search_strategy import select_candidate


def test_object_5_relation_path_and_ranking_are_visible(capsys):
    observed = {'objects': [
        {'id': 'object_6', 'type': 'book'},
        {'id': 'object_5', 'type': 'table'},
        {'id': 'table_1', 'type': 'table'},
    ], 'relations': [
        {'subject': 'object_6', 'predicate': 'on', 'object': 'object_5'},
        {'subject': 'object_6', 'predicate': 'on', 'object': 'table_1'},
    ]}
    kg, semantic = Mock(), Mock()
    kg.resolve_concept.return_value = []
    plain = select_candidate({'Theme': 'book'}, kg, semantic, observed)
    assert capsys.readouterr().out == ''
    traced = select_candidate({'Theme': 'book'}, kg, semantic, observed, debug=True)
    assert traced == plain == {'location': 'object_5', 'ungrounded': []}
    output = capsys.readouterr().out
    for text in ('object_6 --on--> object_5', 'method=case-insensitive exact ID',
                 '1. object_5 source=observed evidence', '2. table_1 source=observed evidence',
                 'Selected gaze target: object_5', 'Semantic fallback: not called'):
        assert text in output
    semantic.rank_gaze_targets.assert_not_called()


def test_grounding_rejections_and_multi_matches_preserve_semantics(capsys):
    observed = {'objects': [{'id': 'object_5', 'type': 'table'},
                            {'id': 'table_1', 'type': 'table'}], 'relations': []}
    kg, semantic = Mock(), Mock()
    kg.resolve_concept.return_value = [{'id': 'book.n.01'}]
    kg.get_locations.return_value = [None, 'missing', 'table', 'table_1']
    options = dict(checked={'object_5'}, suggestions=['other'])
    plain = select_candidate({'Theme': 'book'}, kg, semantic, observed, **options)
    traced = select_candidate({'Theme': 'book'}, kg, semantic, observed, debug=True, **options)
    assert plain == traced == {'location': 'table_1', 'ungrounded': ['missing']}
    output = capsys.readouterr().out
    for text in ('invalid candidate form', 'could not ground', 'Task-local checked anchors:', 'duplicate retained', 'method=normalized type/ID',
                 'Final ranked candidates:', 'source=RoboKGNet'):
        assert text in output


def test_semantic_fallback_inputs_and_returned_scores_are_logged(capsys):
    kg, semantic = Mock(), Mock()
    kg.resolve_concept.return_value = []
    semantic.rank_gaze_targets.return_value = [{'location': 'object_5', 'score': .7}]
    result = select_candidate({'Theme': 'book'}, kg, semantic,
                              {'objects': [{'id': 'object_5', 'type': 'table'}, {'id': 'cup', 'type': 'cup'}],
                               'relations': [{'subject': 'cup', 'predicate': 'on', 'object': 'object_5'}]},
                              suggestions=['missing'], debug=True)
    assert result['location'] == 'object_5'
    output = capsys.readouterr().out
    for text in ("VLM suggestions: ['missing']", 'could not ground', 'Semantic fallback retained: 1',
                 'score=0.7', 'source=semantic fallback'):
        assert text in output
