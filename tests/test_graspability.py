"""Simulator affordance contract, separate from lexical matching; no ROS."""
from copy import deepcopy
from unittest.mock import Mock, patch
import pytest

from perceived_entity_linking import is_graspable, normalize_type, perceived_entity_linking, bind_task
import demo


@pytest.mark.parametrize('kind', ['ForkConnector', 'BookConnector'])
def test_real_passive_connector_protos_are_graspable(kind):
    assert is_graspable({'type': kind})


@pytest.mark.parametrize('kind', ['book', 'fork', 'ImaginaryConnector', 'TiagoGripperConnector'])
def test_suffix_or_visual_type_alone_is_not_grasp_support(kind):
    assert not is_graspable({'type': kind})


@pytest.mark.parametrize('kind,label', [('ForkConnector', 'fork'), ('BookConnector', 'book'),
                                       ('WoodenSpoonConnector', 'wooden spoon')])
def test_connector_normalization(kind, label):
    assert normalize_type(kind) == label


def objects():
    return {'objects': [{'id': name, 'type': kind, 'qualities': {'location': position}}
                        for name, kind, position in [('TIAGo', 'Tiago', [0, 0, 0]),
                        ('user', 'person', [4, 0, 0]), ('KITCHEN', 'Location', [0, 0, 0]),
                        ('book_visual_1', 'book', [0.1, 0, 0]),
                        ('book(2)', 'BookConnector', [1, 0, 0]),
                        ('book(3)', 'BookConnector', [2, 0, 0])]],
            'relations': [{'subject': name, 'predicate': 'in', 'object': 'KITCHEN'}
                          for name in ('TIAGo', 'user', 'book_visual_1', 'book(2)', 'book(3)')]}


def knowledge():
    kg = Mock()
    # Selection is tested independently of WordNet ambiguity: provide one match.
    kg.resolve_concept.side_effect = lambda term: [{'id': 'book.n.02'}] if term == 'book' else []
    return kg


def test_nearest_selection_filters_non_graspable_semantic_match():
    raw = objects()
    before = deepcopy(raw)
    kg = knowledge()
    pel = perceived_entity_linking(raw, kg)
    assert pel['entity_links']['book_visual_1'] == pel['entity_links']['book(2)'] == 'book.n.02'
    result = bind_task(dict(Agent='robot', Theme='book', Source=None, Destination='user'), pel, kg)
    assert result['bindings'] == dict(Agent='TIAGo', Theme='book(2)', Source='KITCHEN', Destination='user')
    assert result['type'] == 'bring'
    assert raw == before


def test_ungraspable_theme_is_incomplete_and_never_dispatches_pick():
    raw = objects()
    raw['objects'] = [obj for obj in raw['objects'] if obj['type'] != 'BookConnector']
    memory = Mock()
    memory.snapshot.return_value = raw
    with patch('builtins.input', return_value='Bring me a book'), \
            patch('demo.observe_scene_with_vlm', return_value={'scene_graph': raw}), \
            patch('plan_execution.manipulation') as pick, \
            patch('procedural_memory.planning.bringing_plan.Planner') as planner:
        result = demo.spa_loop(memory, knowledge(), Mock(), Mock(), execute=True)
    assert result['bindings']['Theme'] is None
    assert 'No graspable concrete Theme found for book' in result['issues']
    assert result['action_graph']['status'] == 'incomplete'
    assert result['action_graph']['executed'] == []
    pick.assert_not_called()
    planner.assert_not_called()


def test_real_book_type_binds_without_guessing_a_wordnet_sense():
    from knowledge_interface import KnowledgeInterface
    kg = KnowledgeInterface()
    assert len(kg.resolve_concept('book')) > 1
    pel = perceived_entity_linking(objects(), kg)
    result = bind_task(dict(Agent='robot', Theme='book', Source=None, Destination='user'), pel, kg)
    assert result['bindings']['Theme'] == 'book(2)'
    assert result['bindings']['Source'] == 'KITCHEN'
    assert result['theme_concept'] is None
    assert 'book(2)' not in pel['entity_links']
    assert result['type'] == 'bring'
