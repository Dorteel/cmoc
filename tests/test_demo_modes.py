"""Legacy is the default; Search knowledge and recovery are explicitly opt-in."""
from copy import deepcopy
import json
from unittest.mock import Mock, patch

import pytest

import demo
from scene_graph_interface import KnowledgeInterface


def world(found=True):
    objects = [{'id': name, 'type': kind, 'qualities': {}} for name, kind in
               [('robot', 'robot'), ('user', 'person'), ('KITCHEN', 'Location'),
                ('worktop(1)', 'worktop'), ('drawer1', 'drawer')]]
    if found:
        objects.append({'id': 'fork1', 'type': 'ForkConnector', 'qualities': {}})
    return {'objects': objects, 'relations': [
        {'subject': obj['id'], 'predicate': 'in', 'object': 'KITCHEN'}
        for obj in objects if obj['type'] != 'Location']}


def knowledge():
    kg = Mock()
    kg.resolve_concept.side_effect = lambda name: [{'id': 'fork.n.01'}] if name == 'fork' else []
    kg.get_locations.return_value = [('worktop(1)', 1), ('KITCHEN', 1)]
    return kg


def test_default_grounds_hidden_seed_and_never_calls_search(tmp_path):
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(world()))
    with patch('builtins.input', return_value=''), \
         patch('demo.observe_scene_with_vlm', return_value={'scene_graph': world(False)}) as observe, \
         patch('demo.search_frame') as frame, patch('demo.select_candidate') as select, \
         patch('demo.plan_search') as search_planner, \
         patch('plan_execution.execute_step', return_value=True):
        result = demo.spa_loop(KnowledgeInterface(path), knowledge(), Mock(), Mock(), execute=True)
    assert result['bindings']['Theme'] == 'fork1'
    assert result['bindings']['Source'] == 'KITCHEN'
    assert result['action_graph']['status'] == 'success'
    assert result['search_outcomes'] == []
    observe.assert_called_once()
    frame.assert_not_called()
    select.assert_not_called()
    search_planner.assert_not_called()


def test_default_missing_theme_does_not_enable_recovery():
    with patch('builtins.input', return_value=''), \
         patch('demo.observe_scene_with_vlm', return_value={'scene_graph': world(False)}), \
         patch('demo.search_frame') as frame, patch('demo.select_candidate') as select:
        result = demo.spa_loop(KnowledgeInterface(), knowledge(), Mock(), None)
    assert result['planning']['status'] == 'incomplete'
    frame.assert_not_called()
    select.assert_not_called()


@pytest.mark.parametrize('hidden_location', ['worktop(1)', 'drawer1'])
def test_search_ignores_hidden_seed_locations(tmp_path, hidden_location):
    seed = world()
    seed['relations'] = [r for r in seed['relations'] if r['subject'] != 'fork1']
    seed['relations'].append({'subject': 'fork1', 'predicate': 'on', 'object': hidden_location})
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(seed))
    with patch('builtins.input', return_value=''), \
         patch('demo.observe_scene_with_vlm', return_value={'scene_graph': world(False)}):
        result = demo.spa_loop(KnowledgeInterface(path), knowledge(), Mock(choose_gaze_action=Mock(return_value={"action": "look-at", "target": "worktop(1)"})), None, search=True)
    assert result['type'] == 'search'
    assert result['frame']['Location'] == 'worktop(1)'
    assert 'Success' not in result['frame']


def test_visible_theme_produces_same_bring_in_both_modes():
    results = []
    for search in (False, True):
        with patch('builtins.input', return_value=''), \
             patch('demo.observe_scene_with_vlm', return_value={'scene_graph': world()}) as observe, \
             patch('demo.plan_search') as search_planner, \
             patch('plan_execution.execute_step', return_value=True) as dispatch:
            result = demo.spa_loop(KnowledgeInterface(), knowledge(), Mock(), Mock(),
                                   execute=True, search=search)
        search_planner.assert_not_called()
        assert result['action_graph']['status'] == 'success'
        assert observe.call_count == (2 if search else 1)
        results.append((result['frame'], result['bindings'], result['planning'],
                        [deepcopy(call.args[0]) for call in dispatch.call_args_list]))
    assert results[0] == results[1]


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('backend', ['dry_run', 'nav2', 'teleport'])
def test_cli_mode_and_shared_backend_dispatch(search, backend, capsys):
    argv = ['demo.py', '--no-simulator']
    if search:
        argv.extend(['--search', '--vlm-cache'])
    if backend != 'dry_run':
        argv.append('--execute')
    if backend == 'teleport':
        argv.append('--teleport')
    with patch('sys.argv', argv), patch('demo.rclpy'), patch('demo.SimulatorLauncher'), \
         patch('demo.PerceptionLauncher') as perception, patch('demo.create_episodic'), \
         patch('demo.RoboKGNet'), patch('demo.SemanticMemory'), \
         patch('demo.RoomNavigator') as nav, patch('demo.TeleportNavigator') as teleport, \
         patch('demo.spa_loop') as spa:
        demo.main()
    perception.assert_called_once_with(backend='nebula', **({'vlm_cache': True, 'fresh_frames': True} if search else {}))
    assert spa.call_args.kwargs['search'] is search
    assert spa.call_args.kwargs['execute'] is (backend != 'dry_run')
    if backend == 'dry_run':
        assert spa.call_args.args[3] is None
        nav.assert_not_called()
        teleport.assert_not_called()
    else:
        selected, unused = (teleport, nav) if backend == 'teleport' else (nav, teleport)
        selected.assert_called_once()
        selected.return_value.wait_until_ready.assert_called_once()
        selected.return_value.close.assert_called_once()
        unused.assert_not_called()
        assert spa.call_args.args[3] is selected.return_value
    expected = 'SEARCH / PARTIAL OBSERVABILITY' if search else 'LEGACY SCENE GRAPH'
    assert f'Demo mode: {expected}' in capsys.readouterr().out


def test_search_does_not_fill_visible_theme_location_from_seed(tmp_path):
    path = tmp_path / 'seed.json'
    path.write_text(json.dumps(world()))
    observed = world()
    observed['relations'] = [r for r in observed['relations'] if r['subject'] != 'fork1']
    sources = []
    for search in (False, True):
        with patch('builtins.input', return_value=''), \
             patch('demo.observe_scene_with_vlm', return_value={'scene_graph': deepcopy(observed)}):
            result = demo.spa_loop(KnowledgeInterface(path), knowledge(), Mock(), None, search=search)
        sources.append(result['bindings']['Source'])
    assert sources == ['KITCHEN', None]


@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('visible_ungraspable', [False, True])
def test_missing_self_and_unresolved_book_search_dispatch(search, visible_ungraspable):
    from perceived_entity_linking import bind_task, perceived_entity_linking
    observed = world(False)
    observed['objects'] = [o for o in observed['objects'] if o['id'] != 'robot']
    observed['relations'] = [r for r in observed['relations'] if r['subject'] != 'robot']
    if visible_ungraspable:
        observed['objects'].append({'id': 'book1', 'type': 'book', 'qualities': {}})
    kg = knowledge()
    llm = Mock()
    llm.choose_gaze_action.return_value = {"action": "look-at", "target": "worktop(1)"}
    llm.rank_gaze_targets.return_value = [{'location': 'worktop(1)', 'score': 1.0}]
    observed['relations'].append({'subject': 'user', 'predicate': 'next_to', 'object': 'worktop(1)'})
    frame = {'Agent': 'robot', 'Theme': 'book', 'Source': None, 'Destination': 'user'}
    before = deepcopy(observed)
    bound = bind_task(frame, perceived_entity_linking(observed, kg), kg,
                      **({'known_self': 'TIAGo'} if search else {}))
    assert bound['bindings']['Agent'] == ('TIAGo' if search else None)
    assert bound['bindings']['Theme'] is None
    assert bound['bindings']['Source'] is None
    assert observed == before
    memory = KnowledgeInterface()
    with patch('builtins.input', return_value='Bring me a book'),          patch('demo.observe_scene_with_vlm', return_value={'scene_graph': observed}),          patch('demo.plan_bring', wraps=demo.plan_bring) as bring:
        result = demo.spa_loop(memory, kg, llm, None, search=search)
    if search:
        bring.assert_not_called()
        assert result['planning']['plan'] == [
            {'action': 'look-at', 'args': ['robot', 'worktop(1)']}]
        assert result['frame']['Theme'] == 'book'
    else:
        assert result['planning']['status'] == 'incomplete'
    assert not any(o['id'] == 'TIAGo' for o in memory.observed_snapshot()['objects'])
