from listener import NaiveFrameFiller
from scene_graph_interface import KnowledgeInterface as EpisodicMemory
from knowledge_interface import KnowledgeInterface as RoboKGNet
from semantic_memory import SemanticMemory
from search_strategy import select_candidate, search_frame, search_result
from procedural_memory.planning.search_plan import plan_search

from pathlib import Path
from copy import deepcopy
import argparse
import time

import rclpy
from rclpy.signals import SignalHandlerOptions

from simulator_launcher import SimulatorLauncher
from perception_launcher import PerceptionLauncher
from navigation import RoomNavigator
from teleport_navigation import TeleportNavigator
from observation import observe_scene_with_vlm
from graph_snapshots import EPISODIC_GRAPH, write_graph_snapshot, episode_snapshot, add_frame_graph, action_snapshot
from procedural_memory.planning.bringing_plan import plan_bring
from plan_execution import execute_plan
from perceived_entity_linking import perceived_entity_linking, bind_task, ground_frame_elements, _position


def create_episodic(scenario):
    if scenario not in ("existing", "empty", "human-moves"):
        raise ValueError(f"Unknown scenario: {scenario}")
    path = None if scenario == "empty" else EPISODIC_GRAPH
    return EpisodicMemory(path)


def candidate_locations(frame, episodic, robokg, semantic_memory):
    """Select from semantic priors and interaction evidence, never simulator seeds."""
    selected = select_candidate(frame, robokg, semantic_memory, episodic.observed_snapshot())
    return [selected['location']] if selected['location'] else []


def spa_loop(episodic, robokg, semantic_memory, navigator, *, scenario="existing", observation_count=1, execute=False, step_by_step=False, search=False):
    """Ground and plan Bring, then preview or execute strictly in planner order."""
    if scenario not in ("existing", "empty", "human-moves"):
        raise ValueError(f"Unknown scenario: {scenario}")
    if observation_count < 1:
        raise ValueError("observation_count must be positive")
    # Keep seed geometry available only to Act; observation merges may invalidate it.
    execution_world = episodic.snapshot() if search else None
    instruction = None
    intention = None
    pending_search = None
    checked = set()
    outcomes = []
    suggestions = []
    recovery_candidate = None
    recovering = False

    def sense(previous_result=None):
        nonlocal instruction
        if instruction is None:
            instruction = input("Instruction [Bring me a fork]: ").strip() or "Bring me a fork"
        observation = observe_scene_with_vlm(schema_path="schemas/objects.json", with_provenance=True)
        return {"instruction": instruction, **observation}

    def consolidate_knowledge(state):
        nonlocal intention, pending_search, suggestions, recovering, recovery_candidate
        if pending_search is not None:
            outcome = search_result(pending_search, state['scene_graph'], robokg)
            outcomes.append(outcome)
            state['search_result'] = deepcopy(outcome)
            if outcome['Success']:
                recovering = False
                recovery_candidate = None
                # The attended candidate supplies spatial context for the new sighting.
                room_ids = {o['id'] for o in episodic.observed_snapshot()['objects'] if o['type'] == 'Location'}
                location = next((o for o in episodic.observed_snapshot()['objects']
                                 if o['id'] == outcome['Location']), {})
                surface = location.get('type', '').casefold() in ('worktop', 'counter', 'table', 'shelf')
                predicate = 'on' if surface and outcome['Location'] not in room_ids else 'in'
                for identifier in outcome['observed_ids']:
                    relation = {'subject': identifier, 'predicate': predicate, 'object': outcome['Location']}
                    if relation not in state['scene_graph']['relations']:
                        state['scene_graph']['relations'].append(relation)
                suggestions = []
            else:
                checked.add(outcome['Location'])
                alternatives = observe_scene_with_vlm(
                    schema_path="schemas/objects.json", with_provenance=True,
                    search_theme=pending_search['Theme'])
                episodic.merge_observation(alternatives['scene_graph'])
                suggestions = alternatives.get('search_candidates', [])
            pending_search = None
        episodic.merge_observation(state["scene_graph"])
        if intention is None:
            intention = NaiveFrameFiller(state["instruction"]).fill()
        frame = deepcopy(intention)
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
        world = episodic.observed_snapshot() if search else episodic.snapshot()
        remembered = perceived_entity_linking(world, robokg)
        theme_visible = (search_result(search_frame(intention, ''), state['scene_graph'], robokg)['Success']
                         if search else False)
        state["scene_graph"] = perceived["scene_graph"]
        state["aliases"] = remembered["aliases"]
        # Context follows planning identities; the G1 context is a separate copy.
        aliases = {alias: canonical for canonical, names in state['aliases'].items() for alias in names}
        for relation in state['context_graph']['relations']:
            for endpoint in ('subject', 'object'):
                relation[endpoint] = aliases.get(relation[endpoint], relation[endpoint])
        state.update(bind_task(state["frame"], remembered, robokg))
        if search and not theme_visible and (state['bindings']['Theme'] is None or recovering):
            selection = select_candidate(intention, robokg, semantic_memory,
                                         episodic.observed_snapshot(), checked, suggestions)
            if recovery_candidate is not None and recovery_candidate not in checked:
                selection['location'] = recovery_candidate
            if selection['location'] is not None:
                frame = search_frame(intention, selection['location'])
                planning = plan_search(frame)
                compact = {'type': 'search', 'frame': frame, 'candidate': frame['Location'],
                           'planning': planning}
                return {**compact, 'bindings': {}, 'entity_links': {}, 'aliases': {}, 'issues': [],
                        'entity_positions': {}, 'sense_graph': sense_graph,
                        'planning_graph': deepcopy(compact), 'action_graph': None}
            state['bindings']['Theme'] = None
            state['type'] = 'incomplete'
            state['issues'].append('Search candidates exhausted or ungrounded: ' + repr(selection['ungrounded']))
        state['frame_element_links'], state['frame_element_issues'] = ground_frame_elements(state['frame'], robokg, remembered)
        # These diagnostic issues do not change concrete Bring selection.
        state['issues'].extend(state['frame_element_issues'])
        # G2 links are task-only; raw scene data remains separate.
        relevant_ids = {'robot', 'user'} | {value for value in state['bindings'].values() if value}
        state['entity_links'] = {identifier: concept for identifier, concept in remembered['entity_links'].items()
                                 if identifier in relevant_ids}
        state['aliases'] = {identifier: names for identifier, names in state['aliases'].items()
                            if identifier in relevant_ids}
        add_frame_graph(state)
        state["planning"] = plan_bring(state["bindings"], remembered["scene_graph"])
        return {
            **state,
            # Execution-only coordinates for entity navigation; no world dump in G2.
            "entity_positions": {obj['id']: _position(obj) for obj in remembered['scene_graph']['objects']
                                 if obj['type'] != 'Location' and _position(obj) is not None
                                 and any(step['action'] in ('navigate', 'pick') and step['args'][1] == obj['id']
                                         for step in state['planning'].get('plan', []))},
            "sense_graph": sense_graph,
            "planning_graph": deepcopy(state),  # G2: entities and frame bindings.
            "action_graph": None,  # Populated after Act with task-only results.
        }

    def act(plan_):
        nonlocal pending_search, recovery_candidate
        if plan_.get('type') == 'search':
            # Coordinates are resolved only after candidate selection, for execution.
            world = episodic.snapshot()
            candidate = plan_['frame']['Location']
            target = next((obj for obj in world['objects'] if obj['id'] == candidate), None)
            if target is None or _position(target) is None:
                target = next((obj for obj in execution_world['objects'] if obj['id'] == candidate), None)
            positions = {candidate: _position(target)} if target and _position(target) is not None else {}
            result = execute_plan(plan_['planning'], navigator, execute=execute,
                                  step_by_step=step_by_step, entity_positions=positions)
            if result['status'] == 'success':
                pending_search = deepcopy(plan_['frame'])
                result['status'] = 'search_observation_required'
            return result
        def verify_theme(step):
            nonlocal recovery_candidate, recovering
            fresh = observe_scene_with_vlm(schema_path="schemas/objects.json", with_provenance=True)
            episodic.merge_observation(fresh['scene_graph'])
            seen = perceived_entity_linking(fresh['scene_graph'], robokg)['scene_graph']
            present = step['args'][1] in {obj['id'] for obj in seen['objects']}
            if not present:
                recovering = True
                trusted = perceived_entity_linking(episodic.observed_snapshot(), robokg)
                recovery_candidate = bind_task(intention, trusted, robokg)['bindings']['Source']
            return present

        return execute_plan(plan_['planning'], navigator, execute=execute, step_by_step=step_by_step,
                            entity_positions=plan_['entity_positions'],
                            **({'before_pick': verify_theme} if search else {}))

    task_complete = False
    result = None
    observations = 0
    # Recovery uses the same SPA loop; system failures still stop execution.
    while not task_complete and (observations < observation_count or pending_search is not None
                                or (result and result["status"] == "search_required")):
        state = episode_snapshot(sense(result), observations + 1)
        write_graph_snapshot("g1", state)
        plan_ = plan(state)
        write_graph_snapshot("episodic", {**episodic.snapshot(), "observed_evidence": episodic.observed_snapshot()})
        g2_path = write_graph_snapshot("g2", plan_["planning_graph"])
        print("Frame:", plan_["frame"], flush=True)
        print("PEL:", plan_["entity_links"], flush=True)
        print("Aliases:", plan_["aliases"], flush=True)
        print("Bindings:", plan_["bindings"], flush=True)
        print("Issues:", plan_["issues"], flush=True)
        print("G2 saved:", g2_path, flush=True)
        if plan_["planning"]["status"] == "planned":
            print("Plan:", plan_["planning"]["plan"], flush=True)
        result = act(plan_)
        plan_["action_graph"] = action_snapshot(result)
        task_complete = result["status"] == "success"
        write_graph_snapshot("g3", plan_["action_graph"])
        if result.get("reason") == "Interrupted":
            raise KeyboardInterrupt
        if execute and result["status"] not in ("success", "search_observation_required", "search_required"):
            break  # No automatic replanning or recovery after execution failure.
        observations += 1
    plan_["search_outcomes"] = deepcopy(outcomes)
    plan_["bring_intention"] = deepcopy(intention)
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
                        default='existing', help='Initial episodic memory for SPA')
    parser.add_argument('--execute', action='store_true', help='Send the generated plan to ROS; default is dry-run')
    parser.add_argument('--search', action='store_true', default=False,
                        help='Enable observation-backed Search recovery; default uses legacy scene-graph grounding')
    parser.add_argument('--step', action='store_true', help='With --execute, confirm each step before sending it')
    parser.add_argument('--teleport', action='store_true', help='With --execute, teleport to resolved navigation poses in Webots')
    args = parser.parse_args()
    if args.step and not args.execute:
        parser.error('--step requires --execute')
    if args.teleport and not args.execute:
        parser.error('--teleport requires --execute')
    print('Demo mode: ' + ('SEARCH / PARTIAL OBSERVABILITY' if args.search else 'LEGACY SCENE GRAPH'), flush=True)
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
        if not args.no_simulator or args.test_navigation or args.execute:
            navigator = TeleportNavigator() if args.teleport else RoomNavigator()
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
            spa_loop(episodic, robokg, semantic_memory, navigator, scenario=args.scenario,
                     execute=args.execute, step_by_step=args.step, search=args.search)
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
