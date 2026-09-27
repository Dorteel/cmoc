from listener import NaiveFrameFiller
from scene_graph_interface import KnowledgeInterface as EpisodicMemory
from knowledge_interface import KnowledgeInterface as RoboKGNet
from semantic_memory import SemanticMemory
from search_strategy import resolve_search

from pathlib import Path
import argparse
import time

import rclpy
from rclpy.signals import SignalHandlerOptions

from simulator_launcher import SimulatorLauncher
from navigation import RoomNavigator


def run_demo(navigator=None):
    instruction = "Bring me the coffee mug"

    frame = NaiveFrameFiller(instruction).fill()

    episodic = EpisodicMemory(Path(__file__).resolve().parent / "scene_graph.json")
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
        # LOOK_FOR: select candidate locations only; no robot execution.
        # 2. Try RoboKGNet semantic knowledge.
        theme = frame["Theme"]
        concepts = robokg.resolve_concept(theme)

        # Try WordNet-style lexical spelling too: "coffee mug" -> "coffee_mug".
        if not concepts:
            concepts = robokg.resolve_concept(theme.replace(" ", "_"))

        if len(concepts) == 1:
            concept_id = concepts[0]["id"]
            search_locations = robokg.get_locations(concept_id)
        else:
            search_locations = []

        # 3. Fall back to generative semantic memory if RoboKGNet
        #    cannot resolve the concept OR knows no locations.
        if not search_locations:
            search = resolve_search(frame, episodic, semantic_memory)
            search_locations = search["locations"]

    print("Frame:", frame)
    print("Search order:")

    for candidate in search_locations:
        print(candidate)


def main():
    parser = argparse.ArgumentParser(description="CMOC demo with the TIAGo apartment")
    parser.add_argument('--no-simulator', action='store_true',
                        help='Run memory demo only; leave any existing simulator alone')
    parser.add_argument('--test-navigation', action='store_true',
                        help='Visit KITCHEN and LIVING_ROOM_1 before the memory demo')
    args = parser.parse_args()
    # Keep Ctrl+C as KeyboardInterrupt so cancellation runs before ROS shutdown.
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    simulator = None
    navigator = None
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
            run_demo()
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
                if simulator is not None:
                    simulator.stop()
            finally:
                if rclpy.ok():
                    rclpy.shutdown()


if __name__ == '__main__':
    main()
