from listener import NaiveFrameFiller
from scene_graph_interface import KnowledgeInterface as EpisodicMemory
from knowledge_interface import KnowledgeInterface as RoboKGNet
from semantic_memory import SemanticMemory
from search_strategy import resolve_search

from pathlib import Path
from copy import deepcopy
import argparse
import time

import rclpy
from rclpy.signals import SignalHandlerOptions

from simulator_launcher import SimulatorLauncher
from perception_launcher import PerceptionLauncher
from navigation import RoomNavigator
from observation import observe_scene_with_vlm
from graph_snapshots import EPISODIC_GRAPH, write_graph_snapshot
from perceived_entity_linking import perceived_entity_linking, bind_task


def create_episodic(scenario):
    if scenario not in ("existing", "empty", "human-moves"):
        raise ValueError(f"Unknown scenario: {scenario}")
    path = None if scenario == "empty" else EPISODIC_GRAPH
    return EpisodicMemory(path)


def candidate_locations(frame, episodic, robokg, semantic_memory):
    """Prefer stored locations; preserve each backend's candidate ordering."""
    theme = frame["Theme"]
    concepts = robokg.resolve_concept(theme)
    if not concepts:
        concepts = robokg.resolve_concept(theme.replace(" ", "_"))
    locations = robokg.get_locations(concepts[0]["id"]) if len(concepts) == 1 else []
    if not locations:
        locations = resolve_search(frame, episodic, semantic_memory)["locations"]
    return locations


def spa_loop(episodic, robokg, semantic_memory, navigator, *, scenario="existing", observation_count=1):
    """Consolidate, perform Perceived Entity Linking (PEL), and ground Bring; do not execute."""
    if scenario not in ("existing", "empty", "human-moves"):
        raise ValueError(f"Unknown scenario: {scenario}")
    if observation_count < 1:
        raise ValueError("observation_count must be positive")
    instruction = None

    def sense(previous_result=None):
        nonlocal instruction
        if instruction is None:
            instruction = input("Instruction [Bring me a fork]: ").strip() or "Bring me a fork"
        scene_graph = observe_scene_with_vlm(schema_path="schemas/objects.json")
        return {"instruction": instruction, "scene_graph": scene_graph}

    def consolidate_knowledge(state):
        episodic.merge_observation(state["scene_graph"])
        frame = NaiveFrameFiller(state["instruction"]).fill()
        if frame is None:
            raise ValueError("Instruction is not supported by NaiveFrameFiller")
        state["frame"] = frame
        return state

    def plan(state):
        # G1 remains sensed information only; consolidation starts planning.
        sense_graph = deepcopy(state)
        state = consolidate_knowledge(deepcopy(state))
        # PEL is planning-only: normalize perceived IDs and link canonical concepts.
        perceived = perceived_entity_linking(state["scene_graph"], robokg)
        remembered = perceived_entity_linking(episodic.snapshot(), robokg)
        state["scene_graph"] = perceived["scene_graph"]
        state["entity_links"] = remembered["entity_links"]
        state.update(bind_task(state["frame"], remembered, robokg))
        return {
            **state,
            "sense_graph": sense_graph,
            "planning_graph": deepcopy(state),  # G2: entities and frame bindings.
            "action_graph": None,  # G3 is unavailable until Act is implemented.
        }

    def act(plan_):
        # Decision preview only: navigator is shared for future execution.
        print("Sensed state:", plan_["sense_graph"])
        print("Frame:", plan_["frame"])
        print("Selected plan:", plan_["type"])
        print("Bindings:", plan_["bindings"])
        if plan_["type"] == "incomplete":
            print("Incomplete:", "; ".join(plan_["issues"]))
        return None

    task_complete = False
    result = None
    observations = 0
    # Stop the preview after a bounded number of observations, without pretending
    # the task is complete or repeatedly calling a VLM before task execution exists.
    while not task_complete and observations < observation_count:
        state = sense(result)
        write_graph_snapshot("g1", state)
        plan_ = plan(state)
        write_graph_snapshot("episodic", episodic.snapshot())
        write_graph_snapshot("g2", plan_["planning_graph"])
        result = act(plan_)
        plan_["action_graph"] = deepcopy(result)
        write_graph_snapshot("g3", plan_["action_graph"])
        observations += 1
    return plan_


def run_demo(navigator=None):
    instruction = "Bring me the coffee mug"

    frame = NaiveFrameFiller(instruction).fill()

    episodic = EpisodicMemory(EPISODIC_GRAPH)
    robokg = RoboKGNet()
    semantic_memory = SemanticMemory(model="qwen3:1.7b")

    # MOVEMENT TEST: deliberately separate from LOOK_FOR.
    if navigator is not None:
        for room in ('KITCHEN', 'LIVING_ROOM_1'):
            if not navigator.go_to_room(room, seed=42):
                raise RuntimeError(f'Movement test failed for {room}')

    # ATTEMPT: first try episodic memory.
    frame["Source"] = episodic.query_theme_location(frame["Theme"])

    if frame["Source"]:
        search_locations = [frame["Source"]]

    else:
        search_locations = candidate_locations(frame, episodic, robokg, semantic_memory)

    print("Frame:", frame)
    print("Search order:")

    for candidate in search_locations:
        print(candidate)


def main():
    parser = argparse.ArgumentParser(description="CMOC demo with the TIAGo apartment")
    parser.add_argument('--no-simulator', action='store_true',
                        help='Use an existing simulator; do not start or stop it')
    parser.add_argument('--test-navigation', action='store_true',
                        help='Visit KITCHEN and LIVING_ROOM_1 before the memory demo')
    parser.add_argument('--scenario', choices=('existing', 'empty', 'human-moves'),
                        default='existing', help='Initial episodic memory for the SPA preview')
    args = parser.parse_args()
    # Keep Ctrl+C as KeyboardInterrupt so cancellation runs before ROS shutdown.
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    simulator = None
    navigator = None
    perception = None
    try:
        simulator = SimulatorLauncher()
        # SETUP: TIAGo must be ready before initializing CMOC memories.
        if not args.no_simulator:
            simulator.start()
            simulator.wait_until_ready()
            print('TIAGo ready: received /wheel/odom.', flush=True)
        if not args.no_simulator or args.test_navigation:
            navigator = RoomNavigator()
            navigator.wait_until_ready()
        if args.test_navigation:
            run_demo(navigator)
        else:
            perception = PerceptionLauncher(backend="nebula")
            perception.start()
            perception.wait_for_observe_with_vlm()
            episodic = create_episodic(args.scenario)
            robokg = RoboKGNet()
            semantic_memory = SemanticMemory(model="qwen3:1.7b")
            print("\n==============================\nCMOC READY\nPerception backend: Nebula\n==============================", flush=True)
            spa_loop(episodic, robokg, semantic_memory, navigator, scenario=args.scenario)
        if not args.no_simulator:
            print('Demo complete. Simulator stays open; press Ctrl+C to stop.', flush=True)
            while simulator.is_running():
                time.sleep(0.2)
            raise RuntimeError('Simulator launch exited; see launch output above')
    except KeyboardInterrupt:
        print('Stopping demo...')
    finally:
        try:
            if navigator is not None:
                navigator.close()
        finally:
            try:
                if perception is not None:
                    perception.stop()
            finally:
                try:
                    if simulator is not None:
                        simulator.stop()
                finally:
                    if rclpy.ok():
                        rclpy.shutdown()


if __name__ == '__main__':
    main()
