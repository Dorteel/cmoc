"""Gaze structured output is restricted to this cycle's filtered options."""
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator, ValidationError, validate

from schemas import build_gaze_choice_schema

OPTIONS = [{'action': 'look-at', 'target': 'chair_1'},
           {'action': 'look-left'}, {'action': 'look-right'}]


def test_schema_contains_exactly_current_options_without_mutating_them():
    before = deepcopy(OPTIONS)
    schema = build_gaze_choice_schema(OPTIONS)
    Draft202012Validator.check_schema(schema)
    assert len(schema['oneOf']) == len(OPTIONS)
    for option, branch in zip(OPTIONS, schema['oneOf']):
        assert {key: value['const'] for key, value in branch['properties'].items()
                if key != 'reason'} == option
        assert branch['additionalProperties'] is False
        assert set(branch['required']) == {*option, 'reason'}
        validate({**option, 'reason': 'Inspecting this action may reveal the Theme.'}, schema)
    assert OPTIONS == before


@pytest.mark.parametrize('choice', [
    {'action': 'look-at', 'target': 'old_table', 'reason': 'Inspect old_table.'},
    {'action': 'look-at', 'reason': 'Inspect chair_1.'},
    {'action': 'look-left', 'target': None, 'reason': 'Look left.'},
    {'action': 'look-right', 'target': 'chair_1', 'reason': 'Look right.'},
    {'action': 'look-left'},
    {'reason': 'Look left.'},
    {'action': 'look-left', 'reason': ''},
    {'action': 'look-left', 'reason': 42},
    {'action': 'look-left', 'reason': 'x' * 241},
    {'action': 'look-left', 'reason': 'Look left.', 'extra': True},
    [{'action': 'look-left', 'reason': 'Look left.'}],
])
def test_schema_rejects_noncurrent_or_malformed_choices(choice):
    with pytest.raises(ValidationError):
        validate(choice, build_gaze_choice_schema(OPTIONS))


def test_schema_changes_with_filtered_options():
    fresh = [{'action': 'look-at', 'target': 'floor_1'}, {'action': 'look-left'}, {'action': 'look-right'}]
    schema = build_gaze_choice_schema(fresh)
    assert schema != build_gaze_choice_schema(OPTIONS)
    validate({**fresh[0], 'reason': 'Inspect floor_1.'}, schema)
    with pytest.raises(ValidationError):
        validate({**OPTIONS[0], 'reason': 'Inspect the target.'}, schema)
    for direction in fresh[1:]:
        validate({**direction, 'reason': 'Continue exploring.'}, schema)


def test_empty_options_cannot_produce_a_choice_schema():
    with pytest.raises(ValueError, match='without options'):
        build_gaze_choice_schema([])
