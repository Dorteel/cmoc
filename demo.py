from listener import NaiveFrameFiller
from scene_graph_interface import KnowledgeInterface as EpisodicMemory
from knowledge_interface import KnowledgeInterface as RoboKGNet
from semantic_memory import SemanticMemory
from search_strategy import select_candidate, search_frame, search_result, search_trace, choose_gaze, gaze_key
from procedural_memory.planning.search_plan import plan_search
from execution_grounding import ExecutionGroundingOracle

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
from graph_snapshots import EPISODIC_GRAPH, write_graph_snapshot, episode_snapshot, add_frame_graph, action_snapshot, interpretation_snapshot
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


def spa_loop(episodic, robokg, semantic_memory, navigator, *, scenario="existing", observation_count=1, execute=False, step_by_step=False, search=False, debug_search=False):
    """Ground and plan Bring, then preview or execute strictly in planner order."""
    if scenario not in ("existing", "empty", "human-moves"):
        raise ValueError(f"Unknown scenario: {scenario}")
    if observation_count < 1:
        raise ValueError("observation_count must be positive")
    instruction = None
    intention = None
    pending_search = None
    checked = set()
    sweep_direction = None
    outcomes = []
    recovering = False
    current_identity_mapping = {}
    execution_oracle = (ExecutionGroundingOracle(episodic.observed_snapshot(), robokg, debug=debug_search)
                        if search else None)

    def sense(previous_result=None):
        nonlocal instruction
        if instruction is None:
            instruction = input("Instruction [Bring me a fork]: ").strip() or "Bring me a fork"
        print('[SENSE] Starting ' + ('fresh perception' if previous_result else 'perception'), flush=True)
        observation = observe_scene_with_vlm(schema_path="schemas/objects.json", with_provenance=True)
        print('[SENSE] Observation received', flush=True)
        return {"instruction": instruction, **observation}

    def consolidate_knowledge(state):
        nonlocal intention, pending_search, recovering, sweep_direction
        if pending_search is not None:
            outcome = search_result(pending_search, state['scene_graph'], robokg)
            outcomes.append(outcome)
            state['search_result'] = deepcopy(outcome)
            if outcome['Success']:
                recovering = False
                sweep_direction = None
                checked.clear()
            elif pending_search.get('GazeAction', 'look-at') == 'look-at':
                checked.add(gaze_key({'action': 'look-at', 'target': pending_search['Location']}))
            # A relative base turn exposes a new sector; it is not a globally checked target.
            search_trace(True, f"Fresh observation Theme found: {outcome['Success']}")
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
        nonlocal current_identity_mapping, sweep_direction, recovering
        # G1 remains sensed information only; consolidation starts planning.
        sense_graph = deepcopy(state)
        print('[PLAN] Building task frame', flush=True)
        confirmed_aliases = {}
        if search:
            fresh = {obj['id']: obj for obj in state['scene_graph']['objects']}
            execution_oracle._objects.update(deepcopy(fresh))
            execution_oracle._current_observed_ids = frozenset(fresh)
            execution_oracle._current_identity_mapping = {}
            execution_oracle.optical_targets.clear()
            if execute and execution_oracle.execution_bindings:
                for identifier, obj in fresh.items():
                    if identifier in execution_oracle.execution_bindings or not any(
                            execution_oracle._objects[old]['type'] == obj['type']
                            for old in execution_oracle.execution_bindings):
                        continue
                    try:
                        instance = execution_oracle.resolve(identifier, require_current=True)
                    except RuntimeError:
                        continue  # No identity evidence: preserve the normal fresh ID.
                    finally:
                        # A reconciliation probe must not become execution history.
                        execution_oracle.execution_bindings.pop(identifier, None)
                        execution_oracle.optical_targets.pop(identifier, None)
                    previous = [old for old, target in execution_oracle.execution_bindings.items()
                                if target == instance]
                    if len(previous) == 1 and previous[0] not in fresh:
                        confirmed_aliases[identifier] = previous[0]
                confirmed_aliases = {raw: old for raw, old in confirmed_aliases.items()
                                     if list(confirmed_aliases.values()).count(old) == 1}
                for raw, old in confirmed_aliases.items():
                    search_trace(debug_search, f"[IDENTITY] {raw} -> "
                                 f"{execution_oracle.execution_bindings[old]} -> {old}")
        # PEL owns object and relation renaming; G1/VLM output stays unchanged.
        perceived = perceived_entity_linking(state["scene_graph"], robokg,
                                              confirmed_aliases=confirmed_aliases)
        # Preserve raw perception while recording only its own lexical grounding.
        sense_graph['entity_links'] = {
            raw: perceived['entity_links'][canonical]
            for raw, canonical in perceived['identity_mapping'].items()
            if canonical in perceived['entity_links']}
        state = deepcopy(state)
        if confirmed_aliases:
            state['scene_graph'] = perceived['scene_graph']
        state = consolidate_knowledge(state)
        current_identity_mapping = dict(perceived["identity_mapping"])
        world = episodic.observed_snapshot() if search else episodic.snapshot()
        remembered = perceived_entity_linking(world, robokg)
        theme_visible = (search_result(search_frame(intention, ''), state['scene_graph'], robokg)['Success']
                         if search else False)
        if search:
            print(f"[SENSE] Theme '{intention['Theme']}' " + ('observed' if theme_visible else 'not observed'), flush=True)
        state["scene_graph"] = perceived["scene_graph"]
        state["aliases"] = remembered["aliases"]
        # Context follows planning identities; the G1 context is a separate copy.
        aliases = {alias: canonical for canonical, names in state['aliases'].items() for alias in names}
        aliases.update(confirmed_aliases)
        for relation in state['context_graph']['relations']:
            for endpoint in ('subject', 'object'):
                relation[endpoint] = aliases.get(relation[endpoint], relation[endpoint])
        state.update(bind_task(state["frame"], remembered, robokg,
                               **({"known_self": "TIAGo"} if search else {})))
        if search and not theme_visible:
            search_trace(debug_search, f"Search triggered: Theme={intention['Theme']!r}, "
                         f"binding={state['bindings']['Theme']!r}, visible={theme_visible}, recovering={recovering}")
            try:
                chosen = choose_gaze(intention, perceived['scene_graph'] if confirmed_aliases else sense_graph['scene_graph'], semantic_memory,
                                     checked, debug=debug_search, sweep_direction=sweep_direction)
                frame = search_frame(intention, chosen.get('target'))
                if chosen['action'] != 'look-at':
                    sweep_direction = chosen['action'].removeprefix('look-')
                    frame['GazeAction'] = chosen['action']
                print('[PLAN] Generating Search plan', flush=True)
                planning = plan_search(frame)
            except (ValueError, TimeoutError) as error:
                sweep_direction = None
                print(f'[SEARCH] Search failed: {error}', flush=True)
                frame = search_frame(intention, None)
                planning = {'status': 'failed', 'plan': [], 'reason': str(error)}
            compact = {'type': 'search', 'frame': frame, 'candidate': frame['Location'],
                       'planning': planning}
            return {**compact, 'bindings': {}, 'entity_links': {}, 'aliases': {}, 'issues': [],
                    'entity_positions': {}, 'sense_graph': sense_graph,
                    'planning_graph': deepcopy(compact), 'action_graph': None}
        sweep_direction = None
        checked.clear()
        if search:
            recovering = False
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
        print('[PLAN] Generating Bring plan', flush=True)
        state["planning"] = plan_bring(state["bindings"], remembered["scene_graph"])
        return {
            **state,
            # Execution-only coordinates for entity navigation; no world dump in G2.
            "entity_positions": {obj['id']: _position(obj) for obj in remembered['scene_graph']['objects']
                                 if obj['type'] != 'Location' and _position(obj) is not None
                                 and any(step['action'] in ('navigate', 'pick') and step['args'][1] == obj['id']
                                         for step in state['planning'].get('plan', []))},
            "sense_graph": sense_graph,
            "planning_graph": interpretation_snapshot(state, sense_graph),
            "action_graph": None,  # Resolved frame plus plan and execution results.
        }

    def act(plan_):
        nonlocal pending_search
        if plan_.get('type') == 'search':
            search_trace(debug_search, f"Executing gaze action: {plan_['planning']['plan']}")
            execution_oracle._objects.update({o['id']: deepcopy(o) for o in episodic.observed_snapshot()['objects']})
            execution_oracle._current_identity_mapping = current_identity_mapping
            execution_oracle._current_observed_ids |= frozenset(current_identity_mapping.values())
            result = execute_plan(plan_['planning'], navigator, execute=execute,
                                  step_by_step=step_by_step,
                                  execution_oracle=execution_oracle)
            if result['status'] == 'success':
                search_trace(debug_search, 'Fresh perception requested')
                pending_search = deepcopy(plan_['frame'])
                result['status'] = 'search_observation_required'
            return result
        def verify_theme(step):
            nonlocal recovering
            print('[SENSE] Verifying Theme before pick with fresh perception', flush=True)
            fresh = observe_scene_with_vlm(schema_path="schemas/objects.json", with_provenance=True)
            episodic.merge_observation(fresh['scene_graph'])
            seen = perceived_entity_linking(fresh['scene_graph'], robokg)['scene_graph']
            present = step['args'][1] in {obj['id'] for obj in seen['objects']}
            if not present:
                recovering = True
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
        print(f'\n================ SPA LOOP {observations + 1} ================', flush=True)
        state = episode_snapshot(sense(result), observations + 1)
        write_graph_snapshot("g1", state)
        plan_ = plan(state)
        write_graph_snapshot("episodic", {**episodic.snapshot(), "observed_evidence": episodic.observed_snapshot()})
        write_graph_snapshot("g1", plan_["sense_graph"])
        g2_path = write_graph_snapshot("g2", plan_["planning_graph"])
        if plan_.get('type') != 'search':
            # Publish the actual plan before any navigation/manipulation can fail.
            preview = {'status': plan_['planning']['status'],
                       'plan': plan_['planning'].get('plan', []),
                       'executed': [], 'failed_step': None}
            write_graph_snapshot("g3", action_snapshot(preview, plan_))
        if debug_search:
            print("Frame:", plan_["frame"], flush=True)
            print("PEL:", plan_["entity_links"], flush=True)
            print("Aliases:", plan_["aliases"], flush=True)
            print("Bindings:", plan_["bindings"], flush=True)
            print("Issues:", plan_["issues"], flush=True)
            print("G2 saved:", g2_path, flush=True)
            if plan_["planning"]["status"] == "planned":
                print("Plan:", plan_["planning"]["plan"], flush=True)
        result = act(plan_)
        plan_["action_graph"] = action_snapshot(result, plan_)
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
    cache_flags = parser.add_mutually_exclusive_group()
    cache_flags.add_argument('--vlm-cache', action='store_true', help='Replay saved graph for the first observation only')
    cache_flags.add_argument('--vlm-cache-save', action='store_true', help='Save the first live observation to .cache/vlm/scene_graph.json')
    parser.add_argument('--debug-search', action='store_true', help='Trace Search gaze choices, execution grounding, and fresh observations')
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
            perception = PerceptionLauncher(backend="nebula", **({"vlm_cache": True} if args.vlm_cache else {}),
                                            **({"vlm_cache_save": True} if args.vlm_cache_save else {}),
                                            **({"fresh_frames": True} if args.search else {}))
            perception.start()
            perception.wait_for_observe_with_vlm()
            episodic = create_episodic(args.scenario)
            robokg = RoboKGNet()
            semantic_memory = SemanticMemory(model="qwen3:1.7b")
            print("\n==============================\nCMOC READY\nPerception backend: Nebula\n==============================", flush=True)
            spa_loop(episodic, robokg, semantic_memory, navigator, scenario=args.scenario,
                     execute=args.execute, step_by_step=args.step, search=args.search, debug_search=args.debug_search)
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
