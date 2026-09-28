"""One candidate, one look-at; success is deliberately absent from PDDL."""
from pathlib import Path
from tempfile import TemporaryDirectory
from .planner import Planner

DOMAIN = Path(__file__).parent / 'search/domain.pddl'


def plan_search(frame, planner=None):
    problem = """(define (problem candidate_search) (:domain searching)
      (:objects agent0 - agent candidate0 - location)
      (:init) (:goal (checked candidate0)))"""
    try:
        with TemporaryDirectory(prefix='cmoc-search-') as directory:
            path = Path(directory) / 'problem.pddl'
            path.write_text(problem)
            solved = (planner or Planner()).solve(DOMAIN, path)
        if solved is None or len(solved.actions) != 1 or solved.actions[0].action.name != 'look-at':
            raise RuntimeError('Expected one look-at action')
        return {'status': 'planned', 'plan': [{'action': 'look-at',
                'args': [frame['Agent'], frame['Location']]}], 'navigation_rooms': {}}
    except Exception as error:
        return {'status': 'failed', 'plan': [], 'reason': f'Search planning failed: {error}'}
