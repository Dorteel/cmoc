"""Gaze request bounds, explanations, validation and terminal narration."""
import json
from unittest.mock import Mock, patch
import pytest
import requests

from semantic_memory import SemanticMemory, GAZE_TIMEOUT_SECONDS
from schemas import build_gaze_choice_schema
from search_strategy import choose_gaze, search_frame
from procedural_memory.planning.search_plan import plan_search
from plan_execution import execute_plan

CURRENT = {'objects': [{'id': 'table_1', 'type': 'table'}], 'relations': []}
REASON = 'table_1 is a plausible place to inspect when searching for a fork.'
CHOICE = {'action': 'look-at', 'target': 'table_1', 'reason': REASON}


def reply(content):
    response = Mock()
    response.json.return_value = {'message': {'content': json.dumps(content)}}
    return response


def test_success_reason_is_display_only_and_stages_are_ordered(capsys):
    llm = SemanticMemory()
    with patch('semantic_memory.requests.post', return_value=reply(CHOICE)) as post:
        choice = choose_gaze({'Theme': 'fork'}, CURRENT, llm)
    assert choice == CHOICE
    assert post.call_args.kwargs['timeout'] == GAZE_TIMEOUT_SECONDS == 30
    assert post.call_args.kwargs['json']['think'] is False
    assert post.call_args.kwargs['json']['format'] == build_gaze_choice_schema([
        {'action': 'look-at', 'target': 'table_1'}, {'action': 'look-left'}, {'action': 'look-right'}])
    frame = search_frame({'Agent': 'robot', 'Theme': 'fork'}, choice['target'])
    assert 'reason' not in frame
    planning = plan_search(frame)
    assert 'reason' not in json.dumps(planning)
    assert planning['plan'] == [{'action': 'look-at', 'args': ['robot', 'table_1']}]
    with patch('plan_execution.execute_step', return_value=True):
        result = execute_plan(planning, execute=True)
    assert result['status'] == 'success'
    output = capsys.readouterr().out
    stages = ['Available gaze actions:', 'Calling gaze LLM', 'Gaze LLM backend: Ollama',
              'Gaze LLM model: qwen3:1.7b', 'Gaze LLM response received in',
              'LLM selected: look-at(table_1)', 'Reason: ' + REASON,
              '[PLAN] Generated plan:', '[ACT] Executing step 1/1', '[ACT] Action complete']
    indices = [output.index(stage) for stage in stages]
    assert indices == sorted(indices)


def test_timeout_fails_once_without_action(capsys):
    with patch('semantic_memory.requests.post', side_effect=requests.Timeout('read timed out')) as post:
        with pytest.raises(TimeoutError, match='Gaze LLM timed out.*30 s'):
            choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory())
    post.assert_called_once()
    output = capsys.readouterr().out
    assert '[SEARCH] Gaze LLM failed after' in output
    assert 'LLM selected:' not in output


@pytest.mark.parametrize('content', [
    [], {}, {**CHOICE, 'reason': ''}, {**CHOICE, 'reason': 'x' * 241},
    {**CHOICE, 'reason': None},
])
def test_malformed_explanation_rejected(content):
    with patch('semantic_memory.requests.post', return_value=reply(content)):
        with pytest.raises(ValueError, match='Invalid gaze LLM response'):
            choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory())


@pytest.mark.parametrize('change', [
    {'action': 'navigate'}, {'target': 'unobserved_table'},
    {'action': 'look-left', 'target': 'table_1'}, {'extra': 'not permitted'},
])
def test_action_validation_still_strict(change):
    with patch('semantic_memory.requests.post', return_value=reply({**CHOICE, **change})):
        with pytest.raises(ValueError, match='Invalid LLM gaze action'):
            choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory())


def test_direction_omits_target():
    choice = {'action': 'look-left', 'reason': 'Looking left may reveal unseen objects.'}
    with patch('semantic_memory.requests.post', return_value=reply(choice)):
        assert choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory()) == choice


def test_invalid_json_is_clear(capsys):
    response = reply(CHOICE)
    response.json.return_value['message']['content'] = '{broken'
    with patch('semantic_memory.requests.post', return_value=response):
        with pytest.raises(ValueError, match='Invalid gaze LLM response'):
            choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory())
    assert 'Gaze LLM failed after' in capsys.readouterr().out


def test_real_unresponsive_http_server_is_bounded(monkeypatch):
    """Exercise actual socket timeout, not only a mock raising Timeout."""
    import socket
    from threading import Event, Thread
    from time import monotonic
    stopped = Event()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        listener.settimeout(2)
        def serve():
            connection, _ = listener.accept()
            with connection:
                stopped.wait(2)  # Accept the request but deliberately never reply.
        worker = Thread(target=serve)
        worker.start()
        monkeypatch.setattr('semantic_memory.GAZE_TIMEOUT_SECONDS', .05)
        started = monotonic()
        try:
            with pytest.raises(TimeoutError, match='Gaze LLM timed out'):
                choose_gaze({'Theme': 'fork'}, CURRENT,
                            SemanticMemory(url=f'http://127.0.0.1:{listener.getsockname()[1]}/api/chat'))
            assert monotonic() - started < 1
        finally:
            stopped.set()
            worker.join(3)
        assert not worker.is_alive()


def test_next_gaze_context_contains_only_current_entities_and_options():
    llm = SemanticMemory()
    floor = {'action': 'look-at', 'target': 'floor_1', 'reason': 'floor_1 may reveal the fork.'}
    current = {'objects': [{'id': 'floor_1', 'type': 'floor', 'reason': REASON}],
               'relations': [{'subject': 'old_table', 'predicate': 'near', 'object': 'floor_1'}],
               'previous_plan': CHOICE}
    with patch('semantic_memory.requests.post', side_effect=[reply(CHOICE), reply(floor)]) as post:
        choose_gaze({'Theme': 'fork'}, CURRENT, llm)
        assert choose_gaze({'Theme': 'fork'}, current, llm, {'look-at(table_1)', 'look-right'}) == floor
    messages = post.call_args.kwargs['json']['messages']
    assert len(messages) == 1 and messages[0]['role'] == 'user'
    prompt = messages[0]['content']
    assert 'table' not in prompt and REASON not in prompt
    assert 'checked' not in prompt and 'previous_plan' not in prompt
    assert '"action": "look-right"' in prompt
    assert 'Current visible entities:' not in prompt
    assert '"type":' not in prompt
    assert 'Allowed gaze actions (choose only from this list): ' + json.dumps([
        {'action': 'look-at', 'target': 'floor_1'}, {'action': 'look-left'}, {'action': 'look-right'}]) in prompt


@pytest.mark.parametrize('invalid', [
    {'action': 'look-at', 'target': 'table_1', 'reason': REASON},
    {'action': 'look-at', 'target': 'stale_target', 'reason': 'stale_target may reveal the fork.'},
])
def test_invalid_or_repeated_response_retries_once_with_identical_clean_context(invalid, capsys):
    floor = {'action': 'look-at', 'target': 'floor_1', 'reason': 'floor_1 may reveal the fork.'}
    current = {'objects': [*CURRENT['objects'], {'id': 'floor_1', 'type': 'floor'}]}
    with patch('semantic_memory.requests.post', side_effect=[reply(invalid), reply(floor)]) as post:
        assert choose_gaze({'Theme': 'fork'}, current, SemanticMemory(), {'look-at(table_1)'}) == floor
    assert post.call_count == 2
    assert post.call_args_list[0] == post.call_args_list[1]
    prompt = post.call_args.kwargs['json']['messages'][0]['content']
    assert '"action": "look-at", "target": "table_1"' not in prompt
    assert 'checked' not in prompt and REASON not in prompt
    output = capsys.readouterr().out
    assert '[SEARCH] Gaze LLM context:' in output
    assert '[SEARCH]   Theme: fork' in output
    assert '[SEARCH]   Current options:' in output
    assert '[SEARCH] Raw response:' in output


def test_second_invalid_response_fails_search_gracefully(capsys):
    import demo
    from scene_graph_interface import KnowledgeInterface
    from test_search_recovery import scene, kg
    memory = KnowledgeInterface()
    nav = Mock()
    with patch('builtins.input', return_value='Bring me a fork'), \
         patch('demo.observe_scene_with_vlm', return_value={'scene_graph': scene()}), \
         patch('semantic_memory.requests.post', return_value=reply(CHOICE)) as post, \
         patch('plan_execution.execute_step') as execute:
        result = demo.spa_loop(memory, kg(), SemanticMemory(), nav, execute=True, search=True)
    assert post.call_count == 2
    assert result['planning']['status'] == result['action_graph']['status'] == 'failed'
    assert result['planning']['plan'] == []
    assert 'Invalid LLM gaze action' in result['action_graph']['reason']
    execute.assert_not_called()
    assert not nav.mock_calls
    output = capsys.readouterr().out
    assert '[SEARCH] Search failed:' in output
    assert 'Traceback' not in output


def test_structured_output_failure_retries_same_schema_and_context():
    with patch('semantic_memory.requests.post', side_effect=[requests.HTTPError('schema rejected'), reply(CHOICE)]) as post:
        assert choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory()) == CHOICE
    assert post.call_count == 2
    assert post.call_args_list[0] == post.call_args_list[1]
    assert isinstance(post.call_args.kwargs['json']['format'], dict)


@pytest.mark.parametrize('direction', ['look-left', 'look-right'])
def test_direction_with_null_target_retries_then_rejects(direction):
    choice = {'action': direction, 'target': None, 'reason': f"Look {direction.removeprefix('look-')} for the fork."}
    with patch('semantic_memory.requests.post', return_value=reply(choice)) as post:
        with pytest.raises(ValueError, match='Invalid LLM gaze action'):
            choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory())
    assert post.call_count == 2


@pytest.mark.parametrize('reason', ['The floor is made of marshmallows.', 'Inspect a completely different cabinet',
                                  'Two sentences. No semantic validation.', 'One line.\nAnother.'])
def test_reason_is_explanatory_only(reason):
    choice = {**CHOICE, 'reason': reason}
    with patch('semantic_memory.requests.post', return_value=reply(choice)) as post:
        assert choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory()) == choice
    post.assert_called_once()


@pytest.mark.parametrize('direction', ['left', 'right'])
def test_sweep_bias_keeps_both_directions_in_schema_and_accepts_reversal(direction):
    opposite = 'right' if direction == 'left' else 'left'
    choice = {'action': f'look-{opposite}', 'reason': 'The other direction is more promising.'}
    with patch('semantic_memory.requests.post', return_value=reply(choice)) as post:
        assert choose_gaze({'Theme': 'fork'}, CURRENT, SemanticMemory(),
                           {'look-at(table_1)', 'look-left', 'look-right'}, sweep_direction=direction) == choice
    payload = post.call_args.kwargs['json']
    assert payload['format'] == build_gaze_choice_schema([{'action': 'look-left'}, {'action': 'look-right'}])
    prompt = payload['messages'][0]['content']
    assert f'Strongly prefer continuing look-{direction}' in prompt
    assert 'reverse direction only with a good reason' in prompt
    assert 'table_1' not in prompt and 'checked' not in prompt
    post.assert_called_once()
