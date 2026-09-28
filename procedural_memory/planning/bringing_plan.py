"""Ground the existing Bringing PDDL domain; no search or recovery actions."""
from pathlib import Path
from tempfile import TemporaryDirectory

from .planner import Planner

DOMAIN = Path(__file__).parent / 'bringing/chatgpt/domain.pddl'


def room_of(identifier, world):
    rooms = {obj['id'] for obj in world['objects'] if obj['type'] == 'Location'}
    if identifier in rooms:
        return identifier
    candidates = {r['object'] for r in world.get('relations', [])
                  if r['subject'] == identifier and r['predicate'] == 'in' and r['object'] in rooms}
    if len(candidates) != 1:
        raise ValueError(f'Known unique room required for {identifier} by the Bringing planner')
    return next(iter(candidates))


def plan_bring(bindings, world, planner=None):
    """Return a small ordered UP plan. Demo starts with an empty gripper."""
    result = {'status': 'incomplete', 'plan': []}
    if any(not bindings.get(role) for role in ('Agent', 'Theme', 'Source', 'Destination')):
        return {**result, 'reason': 'Incomplete concrete Bringing bindings'}
    try:
        # robot is a task identity; TIAGo is the existing simulator memory ID.
        agent = bindings['Agent']
        robot_ids = {agent, 'TIAGo'} if agent == 'robot' else {agent}
        starts = set()
        for identifier in robot_ids:
            try:
                starts.add(room_of(identifier, world))
            except ValueError:
                pass
        if len(starts) != 1:
            raise ValueError('Robot initial room is unknown or ambiguous')
        start = starts.pop()
        source = room_of(bindings['Theme'], world)
        if source != bindings['Source']:
            raise ValueError('Theme location does not agree with the Source binding')
        destination_room = room_of(bindings['Destination'], world)
    except ValueError as error:
        return {**result, 'reason': str(error)}

    # Opaque safe PDDL tokens preserve exact external IDs (case/punctuation).
    # Destination is a delivery waypoint in its known room, not fabricated XYZ.
    values = {'agent': agent, 'theme': bindings['Theme'], 'destination': bindings['Destination']}
    locations = list(dict.fromkeys([start, source]))
    values.update({f'room{i}': value for i, value in enumerate(locations)})
    tokens = {value: key for key, value in values.items()}
    problem = f'''(define (problem grounded_bring) (:domain bringing)
      (:objects agent - robot theme - object {' '.join(f'room{i}' for i in range(len(locations)))} destination - location)
      (:init (robot_at agent {tokens[start]}) (at theme {tokens[source]}) (hand_empty agent))
      (:goal (at theme destination)))'''
    try:
        with TemporaryDirectory(prefix='cmoc-bring-') as directory:
            path = Path(directory) / 'problem.pddl'
            path.write_text(problem)
            solved = (planner or Planner()).solve(DOMAIN, path)
        if solved is None:
            raise RuntimeError('Planner returned no plan')
        steps = []
        for action in solved.actions:
            name = action.action.name
            args = [values[param.object().name] for param in action.actual_parameters]
            if name == 'move':
                steps.append({'action': 'navigate', 'args': [args[0], args[2]]})
            elif name in ('pick', 'place'):
                steps.append({'action': name, 'args': args[:2] if name == 'pick' else args})
            else:
                raise RuntimeError(f'Unsupported planner action: {name}')
        if not steps:
            raise RuntimeError('Planner returned an empty plan')
        return {'status': 'planned', 'plan': steps,
                'navigation_rooms': {source: source, bindings['Destination']: destination_room}}
    except Exception as error:
        return {'status': 'failed', 'plan': [], 'reason': f'Planning failed: {error}'}
