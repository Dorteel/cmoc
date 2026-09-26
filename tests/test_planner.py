from pathlib import Path

import pytest
from unified_planning.plans import SequentialPlan

from procedural_memory.planning.planner import Planner


@pytest.mark.parametrize("planner_name", [None, "fast-downward"])
def test_bringing_plan(planner_name):
    # Parse and solve the actual Bringing files through the public utility.
    test_dir = (
        Path(__file__).resolve().parents[1]
        / "procedural_memory" / "planning" / "bringing" / "test"
    )
    plan = Planner(planner_name).solve(
        test_dir / "domain.pddl", test_dir / "problem.pddl"
    )

    # Search and movement may occur in either order; do not fix a full plan.
    assert isinstance(plan, SequentialPlan)
    assert plan.actions
    assert any(action.action.name == "look_for" for action in plan.actions)
