"""Gaze request bounds, explanations, validation and terminal narration."""
import json
from unittest.mock import Mock, patch
import pytest
import requests

from semantic_memory import SemanticMemory, GAZE_TIMEOUT_SECONDS
from search_strategy import choose_gaze, search_frame
from procedural_memory.planning.search_plan import plan_search
from plan_execution import execute_plan

CURRENT = {'objects': [{'id': 'table_1', 'type': 'table'}], 'relations': []}
REASON = 'A table is a plausible place to inspect when searching for a fork.'
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
    [], {}, {**CHOICE, 'reason': ''}, {**CHOICE, 'reason': 'Two sentences. Not allowed.'},
    {**CHOICE, 'reason': 'No punctuation'}, {**CHOICE, 'reason': 'x' * 241 + '.'},
    {**CHOICE, 'reason': 'One line.\nAnother.'},
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


def test_direction_accepts_null_target():
    choice = {'action': 'look-left', 'target': None, 'reason': 'Looking left may reveal unseen objects.'}
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
