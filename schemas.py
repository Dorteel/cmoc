"""Dynamic structured-output schemas for LLM decisions."""


def build_gaze_choice_schema(options):
    """Allow exactly one of the already-filtered current gaze options."""
    if not options:
        raise ValueError('Cannot build a gaze choice schema without options')
    branches = []
    for option in options:
        properties = {
            'action': {'const': option['action']},
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240},
        }
        required = ['action', 'reason']
        if option['action'] == 'look-at':
            properties['target'] = {'const': option['target']}
            required.append('target')
        branches.append({'type': 'object', 'properties': properties,
                         'required': required, 'additionalProperties': False})
    return {'oneOf': branches}
