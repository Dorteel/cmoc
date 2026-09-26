"""Read PDDL and solve it through Unified Planning."""

from pathlib import Path

from unified_planning.engines.results import POSITIVE_OUTCOMES
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner


class Planner:
    def __init__(self, planner_name="fast-downward-opt"):
        # Remember an optional engine name; otherwise select one automatically.
        self.planner_name = planner_name

    def solve(self, domain_path, problem_path):
        # Parse both PDDL files into Unified Planning's problem model.
        problem = PDDLReader().parse_problem(str(domain_path), str(problem_path))

        # Use the requested engine, or select one supporting the problem features.
        options = (
            {"name": self.planner_name}
            if self.planner_name is not None
            else {"problem_kind": problem.kind}
        )
        with OneshotPlanner(**options) as planner:
            result = planner.solve(problem)

        # Return a successful plan; report the solver status otherwise.
        if result.status in POSITIVE_OUTCOMES and result.plan is not None:
            return result.plan
        raise RuntimeError(f"Planning failed: {result.status.name}")


if __name__ == "__main__":
    # Locate the Bringing example relative to this file, then print its plan.
    test_dir = Path(__file__).resolve().parent / "bringing" / "test"
    print(Planner().solve(test_dir / "domain.pddl", test_dir / "problem.pddl"))
