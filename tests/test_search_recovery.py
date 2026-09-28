"""Search epistemic boundary and SPA recovery, without ROS or simulator processes."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from jsonschema import validate, ValidationError

import demo
from scene_graph_interface import KnowledgeInterface
from search_strategy import select_candidate, search_frame, search_result
from procedural_memory.planning.search_plan import plan_search
from plan_execution import execute_plan

BRING = {'Agent': 'robot', 'Theme': 'fork', 'Source': None, 'Destination': 'user'}


def scene(found=False):
    objects = [{'id': name, 'type': kind, 'qualities': {}} for name, kind in
               [('KITCHEN', 'Location'), ('worktop(1)', 'worktop'), ('drawer1', 'drawer'),
                ('robot', 'robot'), ('user', 'person')]]
    relations = [{'subject': name, 'predicate': 'in', 'object': 'KITCHEN'}
                 for name in ('robot', 'user', 'worktop(1)', 'drawer1')]
    if found:
        objects.append({'id': 'fork1', 'type': 'ForkConnector', 'qualities': {}})
        relations.append({'subject': 'fork1', 'predicate': 'in', 'object': 'KITCHEN'})
    return {'objects': objects, 'relations': relations}


def kg():
    result = Mock()
    result.resolve_concept.side_effect = lambda term: [{'id': 'fork.n.01'}] if term == 'fork' else []
    result.get_locations.return_value = [('KITCHEN', 99), ('worktop(1)', 1)]
    return result


def semantic():
    result = Mock()
    result.rank_locations.return_value = []
    return result


def test_exact_before_room_and_exclusion_is_exact_and_task_local():
    graph, knowledge, llm = scene(), kg(), semantic()
    assert select_candidate(BRING, knowledge, llm, graph)['location'] == 'worktop(1)'
    assert select_candidate(BRING, knowledge, llm, graph, {'worktop(1)'})['location'] == 'KITCHEN'
    assert select_candidate(BRING, knowledge, llm, graph, {'worktop(1)', 'KITCHEN'})['location'] is None
    assert select_candidate(BRING, knowledge, llm, graph)['location'] == 'worktop(1)'


def test_llm_fallback_and_explicit_ungrounded_suggestion():
    knowledge, llm = kg(), semantic()
    knowledge.get_locations.return_value = []
    llm.rank_locations.return_value = [{'location': 'drawer1'}]
    result = select_candidate(BRING, knowledge, llm, scene(), suggestions=['behind the bowl'])
    assert result == {'location': 'drawer1', 'ungrounded': ['behind the bowl']}
    llm.rank_locations.assert_called_once()


@pytest.mark.parametrize('success', [True, False])
def test_frame_and_result_schemas(success):
    frame = search_frame(BRING, 'worktop(1)')
    assert frame == {'verb': 'search', 'Agent': 'robot', 'Theme': 'fork', 'Location': 'worktop(1)', 'Success': False}
    frame['Success'] = success
    schema = json.loads(Path('schemas/task_frames/search.json').read_text())
    validate(frame, schema)
    frame['Success'] = str(success).lower()
    with pytest.raises(ValidationError):
        validate(frame, schema)
    validate({'theme': 'fork', 'location': 'KITCHEN', 'success': success},
             json.loads(Path('schemas/task_frames/search_result.json').read_text()))


def test_real_search_pddl_uses_opaque_candidate_token():
    result = plan_search(search_frame(BRING, 'worktop(1)'))
    assert result['status'] == 'planned'
    assert result['plan'] == [{'action': 'look-at', 'args': ['robot', 'worktop(1)']}]


@pytest.mark.parametrize('hidden_location', ['worktop(1)', 'drawer1', 'KITCHEN'])
def test_hidden_ground_truth_cannot_change_candidate(tmp_path, hidden_location):
    seed = scene(True)
    seed['relations'] = [r for r in seed['relations'] if r['subject'] != 'fork1']
    seed['relations'].append({'subject': 'fork1', 'predicate': 'on', 'object': hidden_location})
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(seed))
    memory = KnowledgeInterface(path)
    memory.merge_observation(scene())
    assert select_candidate(BRING, kg(), semantic(), memory.observed_snapshot())['location'] == 'worktop(1)'
    assert not search_result(search_frame(BRING, hidden_location), scene(), kg())['success']
    # Only a real observation can promote the alternate location to evidence.
    memory.merge_observation({'objects': [seed['objects'][-1]], 'relations': [seed['relations'][-1]]})
    assert select_candidate(BRING, kg(), semantic(), memory.observed_snapshot())['location'] == hidden_location


def test_failed_then_successful_search_replans_bring_and_persists_evidence():
    memory, knowledge = KnowledgeInterface(), kg()
    views = [{'scene_graph': scene()}, {'scene_graph': scene()},
             {'scene_graph': scene(), 'search_candidates': ['drawer1']},
             {'scene_graph': scene(True)}]
    events = []
    def observe(**kwargs):
        events.append('alternatives' if 'search_theme' in kwargs else 'sense')
        return deepcopy(views.pop(0))
    def execute(planning, *args, **kwargs):
        action = planning['plan'][0]['action']
        events.append(action)
        return {'status': 'success', 'plan': planning['plan'], 'executed': [], 'failed_step': None}
    bring_plan = {'status': 'planned', 'plan': [{'action': 'pick', 'args': ['robot', 'fork1']}], 'navigation_rooms': {}}
    with patch('builtins.input', return_value='Bring me a fork'), \
         patch('demo.observe_scene_with_vlm', side_effect=observe), \
         patch('demo.execute_plan', side_effect=execute), \
         patch('demo.plan_bring', return_value=bring_plan) as bring:
        result = demo.spa_loop(memory, knowledge, semantic(), Mock(), execute=True, search=True)
    assert events == ['sense', 'look-at', 'sense', 'alternatives', 'look-at', 'sense', 'pick']
    assert [(r['location'], r['success']) for r in result['search_outcomes']] == [('worktop(1)', False), ('KITCHEN', True)]
    assert result['bring_intention'] == BRING
    bring.assert_called_once()
    assert bring.call_args.args[0]['Theme'] == 'fork1'
    assert any(r['subject'] == 'fork1' and r['object'] == 'KITCHEN' for r in memory.observed_snapshot()['relations'])
    knowledge.update_location.assert_not_called()  # No histogram/confidence learning.
    import graph_snapshots
    persisted = json.loads((graph_snapshots.ARTIFACT_DIRECTORY / 'scene_graph.json').read_text())
    assert KnowledgeInterface(graph_snapshots.ARTIFACT_DIRECTORY / 'scene_graph.json').observed_snapshot() == persisted['observed_evidence']


def test_compact_search_g2_independent_of_memory_size():
    sizes = []
    for count in (0, 1000):
        memory = KnowledgeInterface()
        graph = scene()
        graph['objects'].extend({'id': f'unrelated{i}', 'type': 'chair', 'qualities': {}} for i in range(count))
        memory.merge_observation(graph)
        with patch('builtins.input', return_value=''), patch('demo.observe_scene_with_vlm', return_value={'scene_graph': scene()}):
            result = demo.spa_loop(memory, kg(), semantic(), None, search=True)
        sizes.append(json.dumps(result['planning_graph'], sort_keys=True))
    assert sizes[0] == sizes[1]
    assert 'Success' in sizes[0]


def test_execution_error_does_not_become_search_failure():
    planning = plan_search(search_frame(BRING, 'worktop(1)'))
    with patch('plan_execution.execute_step', side_effect=RuntimeError('camera motor unavailable')):
        result = execute_plan(planning, Mock(), execute=True)
    assert result['status'] == 'failed'
    assert result['reason'] == 'camera motor unavailable'


def test_missing_before_pick_requests_search_without_executing_pick():
    planning = {'status': 'planned', 'plan': [{'action': 'pick', 'args': ['robot', 'fork1']}], 'navigation_rooms': {}}
    with patch('plan_execution.execute_step') as dispatch:
        result = execute_plan(planning, execute=True, before_pick=lambda step: False)
    assert result['status'] == 'search_required'
    dispatch.assert_not_called()


def test_missing_pick_discards_old_bring_plan_and_replans_after_search():
    memory = KnowledgeInterface()
    old = {'status': 'planned', 'navigation_rooms': {}, 'plan': [
        {'action': 'pick', 'args': ['robot', 'fork1']},
        {'action': 'place', 'args': ['robot', 'fork1', 'stale_destination']}]}
    new = deepcopy(old)
    new['plan'][1]['args'][2] = 'user'
    observations = [scene(True), scene(), scene(), scene(True), scene(True)]
    dispatched = []
    def dispatch(step, *args):
        dispatched.append(deepcopy(step))
        return True
    with patch('builtins.input', return_value=''), \
         patch('demo.observe_scene_with_vlm', side_effect=[{'scene_graph': s} for s in observations]), \
         patch('demo.plan_bring', side_effect=[old, new]) as bring, \
         patch('plan_execution.execute_step', side_effect=dispatch):
        result = demo.spa_loop(memory, kg(), semantic(), Mock(), execute=True, search=True)
    assert bring.call_count == 2
    assert [s['action'] for s in dispatched] == ['look-at', 'pick', 'place']
    assert dispatched[-1]['args'][-1] == 'user'
    assert result['action_graph']['status'] == 'success'
    assert result['bring_intention'] == BRING


@pytest.mark.parametrize('hidden_location', ['worktop(1)', 'drawer1'])
def test_spa_hidden_seed_never_binds_theme_or_selects_candidate(tmp_path, hidden_location):
    seed = scene(True)
    seed['relations'] = [{'subject': 'fork1', 'predicate': 'on', 'object': hidden_location}]
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(seed))
    with patch('builtins.input', return_value=''), \
         patch('demo.observe_scene_with_vlm', return_value={'scene_graph': scene()}), \
         patch('demo.plan_bring') as bring:
        result = demo.spa_loop(KnowledgeInterface(path), kg(), semantic(), None, search=True)
    assert result['frame']['Location'] == 'worktop(1)'
    assert result['frame']['Success'] is False
    assert result['type'] == 'search'
    bring.assert_not_called()
