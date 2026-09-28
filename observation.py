"""Thin client for CMOC's existing current-camera VLM action."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def observe_scene_with_vlm(schema_path=ROOT / 'schemas/objects.json', timeout=510, *, with_provenance=False):
    """Request a new observation of the server's latest TIAGo RGB frame.

    The application owns rclpy. The existing server owns camera capture, VLM
    backend selection and schema validation; this helper only calls its action.
    with_provenance returns scene_graph plus server-produced perception_provenance;
    the default preserves the scene-only API for existing callers.
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.action import ActionClient
    from action_msgs.msg import GoalStatus
    from cmoc_interfaces.action import ObserveWithVLM

    if not rclpy.ok():
        raise RuntimeError('ROS context is not active')
    schema_path = Path(schema_path)
    if not schema_path.is_absolute():
        schema_path = ROOT / schema_path
    schema = json.loads((ROOT / 'schemas/perception/create_scene_graph.json').read_text())
    schema['properties']['objects']['items'] = json.loads(schema_path.read_text())
    prompt = (
        'Describe only entities and relations visible in the current camera frame. '
        'Use the supplied object schema inside the scene graph. Use descriptive '
        'object IDs consistently across observations, and object types such as fork '
        'or person. Use type Location for a visually identifiable room. '
        'Do not guess metric world coordinates, dimensions, or hidden entities; '
        'omit unknown qualities. Return only JSON matching the supplied schema.'
    )
    node = rclpy.create_node('cmoc_observation_client')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    futures = []
    client = None
    handle = None

    def wait(future):
        futures.append(future)
        executor.spin_until_future_complete(future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError('VLM observation action timed out')
        return future.result()

    try:
        client = ActionClient(node, ObserveWithVLM, '/observe_with_vlm')
        if not client.wait_for_server(timeout_sec=10):
            raise RuntimeError('Start the cmoc_perception /observe_with_vlm action server first')
        goal = ObserveWithVLM.Goal(prompt=prompt, json_schema=json.dumps(schema))
        handle = wait(client.send_goal_async(goal))
        if not handle.accepted:
            raise RuntimeError('VLM observation goal rejected')
        result = wait(handle.get_result_async())
        if result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.success:
            raise RuntimeError(f'VLM observation failed: {result.result.response}')
        scene = json.loads(result.result.response)
        if with_provenance:
            return {'scene_graph': scene,
                    'perception_provenance': json.loads(result.result.perception_provenance)}
        return scene
    finally:
        # The existing server has no cancellation support; its HTTP timeout bounds
        # inference if this caller is interrupted. Never shut down application ROS.
        # Keep queued action callbacks off the application's global executor.
        # Stop callbacks before destroying their waitable/node, including failures.
        executor.shutdown()
        for future in futures:
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()  # Mark any terminal exception as retrieved.
        if client is not None:
            client.destroy()
        node.destroy_node()
