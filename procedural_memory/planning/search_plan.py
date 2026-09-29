"""One selected gaze action; success is deliberately absent from PDDL."""
from pathlib import Path
from tempfile import TemporaryDirectory
from .planner import Planner

DOMAIN = Path(__file__).parent / 'search/domain.pddl'


def plan_search(frame, planner=None):
    action = frame.get('GazeAction', 'look-at')
    if action not in ('look-at', 'look-left', 'look-right'):
        return {'status': 'failed', 'plan': [], 'reason': 'Invalid gaze action'}
    goal = '(checked candidate0)' if action == 'look-at' else f"(looked-{action[5:]} agent0)"
    problem = f"""(define (problem candidate_search) (:domain searching)
      (:objects agent0 - agent candidate0 - location)
      (:init) (:goal {goal}))"""
    try:
        with TemporaryDirectory(prefix='cmoc-search-') as directory:
            path = Path(directory) / 'problem.pddl'
            path.write_text(problem)
            solved = (planner or Planner()).solve(DOMAIN, path)
        if solved is None or len(solved.actions) != 1 or solved.actions[0].action.name != action:
            raise RuntimeError('Expected one selected gaze action')
        return {'status': 'planned', 'plan': [{'action': action,
                'args': [frame['Agent']] + ([frame['Location']] if action == 'look-at' else [])}], 'navigation_rooms': {}}
    except Exception as error:
        return {'status': 'failed', 'plan': [], 'reason': f'Search planning failed: {error}'}
