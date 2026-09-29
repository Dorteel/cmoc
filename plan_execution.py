"""Sequential Bringing dispatcher using the existing navigator and ROS actions."""

import math

# Verified names in the demo world / its upstream Pedestrian PROTO.
SIMULATOR_NAMES = {'robot': 'TIAGo', 'user': 'pedestrian'}


def manipulation(action, args, timeout=60):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.executors import SingleThreadedExecutor
    from action_msgs.msg import GoalStatus
    from simulation_actions.action import Pick, PlaceNextTo

    if not rclpy.ok():
        raise RuntimeError('ROS context is not active')
    action_type, endpoint = (Pick, '/pick') if action == 'pick' else (PlaceNextTo, '/place_next_to')
    fields = {'robot': SIMULATOR_NAMES.get(args[0], args[0]), 'object': args[1]}
    if action == 'place':
        fields['target'] = SIMULATOR_NAMES.get(args[2], args[2])
    node = rclpy.create_node('cmoc_bring_action')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    client = None
    handle = None
    futures = []

    def wait(future, seconds):
        futures.append(future)
        executor.spin_until_future_complete(future, timeout_sec=seconds)
        if not future.done():
            raise TimeoutError(f'{endpoint} timed out')
        return future.result()

    try:
        client = ActionClient(node, action_type, endpoint)
        if not client.wait_for_server(timeout_sec=10):
            raise RuntimeError(f'{endpoint} action server unavailable')
        handle = wait(client.send_goal_async(action_type.Goal(**fields)), 10)
        if not handle.accepted:
            raise RuntimeError(f'{endpoint} goal rejected')
        response = wait(handle.get_result_async(), timeout)
        if response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.success:
            raise RuntimeError(f'{endpoint} failed: {response.result.message}')
        return True
    except (TimeoutError, KeyboardInterrupt):
        if handle is not None and handle.accepted:
            try:
                wait(handle.cancel_goal_async(), 3)
            except (Exception, KeyboardInterrupt):
                pass  # Cancellation is best effort; never continue the plan.
        raise
    finally:
        executor.shutdown()
        for future in futures:
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()
        if client is not None:
            client.destroy()
        node.destroy_node()


def execution_target_position(target, oracle=None, *, full_pose=False, identity_source="perceived"):
    # Only identity_source="simulator" permits direct lookup. Perceived IDs
    # must first establish current camera correspondence through the Oracle.
    # Simulator pose lookup provides execution grounding only. On a physical
    # robot this would come from perception, object localization, semantic
    # mapping, or another grounding system. Never treat these coordinates as
    # observations or Search evidence.
    from external.webots_ros2_simulation.controllers.fallback_action_supervisor.action_cli import send_action

    try:
        if identity_source == 'perceived':
            if oracle is None:
                raise RuntimeError('perceived identity requires an execution grounding Oracle')
            execution_target = oracle.resolve(target, require_current=True)
        elif identity_source == 'simulator':
            execution_target = target
        else:
            raise ValueError(f'Unknown identity source: {identity_source}')
        response = send_action('get_object_pose', {'target': execution_target})
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'unknown supervisor error'))
        position = response['result']['position']
        if (not isinstance(position, list) or len(position) != 3
                or not all(type(v) in (int, float) and math.isfinite(v) for v in position)):
            raise ValueError('expected three finite world coordinates')
        return position if full_pose else position[:2]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise RuntimeError(f'Cannot resolve execution target {target}: {error}') from error


def execute_step(step, navigator, navigation_rooms, entity_positions=None, *, execution_oracle=None):
    action, args = step['action'], step['args']
    if action == 'navigate':
        if navigator is None:
            raise RuntimeError('RoomNavigator is unavailable')
        target = args[1]
        room = navigation_rooms.get(target)
        if target != room and entity_positions and target in entity_positions:
            return navigator.approach_entity(target, entity_positions[target])
        if room is None:
            raise RuntimeError(f'No target coordinates or known room for {target}')
        if target != room:
            print(f'Navigating to {target}\'s known room {room}; target coordinates unavailable; using room fallback', flush=True)
        return navigator.go_to_room(room)
    if action in ('look-at', 'look-left', 'look-right'):
        from external.webots_ros2_simulation.controllers.fallback_action_supervisor.action_cli import send_action
        parameters = {'action': action}
        if action == 'look-at':
            if execution_oracle is None:
                raise RuntimeError('Search gaze requires an execution grounding Oracle')
            if execution_oracle.gaze_region(args[1]) == 'floor':
                print(f'[GROUNDING] Perceived entity {args[1]} classified as gaze region: floor', flush=True)
                print('[ACT] Gaze region floor -> tilt head downward', flush=True)
                print('[ACT] Base motion: none', flush=True)
                parameters['action'] = 'look-down'
            else:
                print(f'[GROUNDING] Resolving perceived entity {args[1]}', flush=True)
                resolved = execution_oracle.resolve(args[1], require_current=True)
                print(f'[GROUNDING] Camera-grounded instance: {resolved}', flush=True)
                parameters['optical_target'] = execution_oracle.optical_targets[args[1]]
        if action == 'look-at':
            print(f'[ACT] Moving head toward {args[1]}', flush=True)
        else:
            print(f"[ACT] Rotating base {'+60' if action == 'look-left' else '-60'} degrees; translation: none", flush=True)
        response = send_action('gaze', parameters)
        if not response.get('ok'):
            raise RuntimeError(f"Head gaze failed: {response.get('error', 'unknown error')}")
        if execution_oracle is not None and execution_oracle.debug:
            positions = response.get('result', {}).get('head_positions', {})
            print(f"[SEARCH] Head target: pan={positions.get('head_1_joint')}, "
                  f"tilt={positions.get('head_2_joint')}; Base motion: {'none' if action == 'look-at' else 'rotation only'}", flush=True)
        return True
    if action in ('pick', 'place'):
        return manipulation(action, args)
    raise ValueError(f'Unsupported action: {action}')


def execute_plan(planning, navigator=None, *, execute=False, step_by_step=False, entity_positions=None, before_pick=None, execution_oracle=None):
    plan = planning.get('plan', [])
    result = {'status': planning['status'], 'plan': plan, 'executed': [], 'failed_step': None}
    if planning['status'] != 'planned':
        print(f"Plan {planning['status']}: {planning.get('reason', 'No executable plan')}", flush=True)
        return {**result, 'reason': planning.get('reason', 'No executable plan')}
    print('[PLAN] Generated plan:', flush=True)
    for index, step in enumerate(plan, 1):
        print(f"[PLAN]   {index}. {step['action']} {' '.join(step['args'])}", flush=True)
    if not execute:
        return {**result, 'status': 'dry_run'}
    for index, step in enumerate(plan, 1):
        try:
            if step_by_step and input(f'Execute step {index}? [Enter=yes, q=stop]: ').strip().lower() == 'q':
                return {**result, 'status': 'cancelled', 'reason': 'Stopped before next step'}
            if step['action'] == 'pick' and before_pick is not None and not before_pick(step):
                return {**result, 'status': 'search_required', 'reason': 'Theme not perceived before pick'}
            print(f"[ACT] Executing step {index}/{len(plan)}: {step['action']} {' '.join(step['args'])}", flush=True)
            attempt = {'step': index, **step, 'status': 'attempted'}
            result['executed'].append(attempt)
            execution_step = step
            following = plan[index] if index < len(plan) else None
            if (step['action'] == 'navigate' and following and following['action'] == 'pick'
                    and step['args'][0] == following['args'][0]):
                # The pick already names the grounded Theme; never select again
                # or mutate the symbolic room target / Source binding.
                theme = following['args'][1]
                position = (entity_positions or {}).get(theme)
                if (isinstance(position, (list, tuple)) and len(position) == 2
                        and all(type(value) in (int, float) and math.isfinite(value) for value in position)):
                    print(f"Refining navigation:\n  symbolic target: {step['args'][1]}\n"
                          f"  concrete target: {theme}\n  target position (scene): {tuple(position)}\n"
                          "  requested distance: 0.5 m", flush=True)
                    execution_step = {'action': 'navigate', 'args': [step['args'][0], theme]}
                else:
                    print('Theme position unavailable; falling back to room navigation: '
                          + step['args'][1], flush=True)
            if not execute_step(execution_step, navigator, planning['navigation_rooms'], entity_positions,
                                **({'execution_oracle': execution_oracle} if execution_oracle is not None else {})):
                raise RuntimeError(f"{step['action']} failed")
            attempt['status'] = 'success'
            print('[ACT] Action complete', flush=True)
        except (Exception, KeyboardInterrupt) as error:
            print(f'[ACT] Execution stopped at step {index}: {str(error) or "Interrupted"}', flush=True)
            if result['executed'] and result['executed'][-1]['step'] == index:
                result['executed'][-1]['status'] = 'failed'
            return {**result, 'status': 'failed', 'failed_step': step,
                    'reason': str(error) or 'Interrupted'}
    return {**result, 'status': 'success'}
