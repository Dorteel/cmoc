"""Room goal sampling and a small synchronous Nav2 client."""

import json
import math
from pathlib import Path
import random
import time

from PIL import Image
import yaml

from external.webots_ros2_simulation.scripts.map_alignment import MapAlignment
from spatial_grounder import get_location_bounds

ROOT = Path(__file__).resolve().parent
SIMULATOR = ROOT / 'external/webots_ros2_simulation'
MAP_DIRECTORY = SIMULATOR / 'maps/apartment_room_aligned'


def entity_approach_pose(robot, target):
    """Map-frame goal 0.5 m before target, or rotate in place when closer."""
    rx, ry = robot
    tx, ty = target
    if not all(math.isfinite(value) for value in (rx, ry, tx, ty)):
        raise ValueError('Entity approach requires finite coordinates')
    dx, dy = tx - rx, ty - ry
    distance = math.hypot(dx, dy)
    if distance <= 0.5:
        x, y = rx, ry
    else:
        x, y = tx - 0.5 * dx / distance, ty - 0.5 * dy / distance
    return x, y, math.atan2(ty - y, tx - x)


class OccupancyMap:
    def __init__(self, path):
        path = Path(path)
        data = yaml.safe_load(path.read_text())
        self.resolution = float(data['resolution'])
        self.origin = data['origin']
        self.free_thresh = float(data['free_thresh'])
        self.occupied_thresh = float(data['occupied_thresh'])
        self.negate = bool(data['negate'])
        if self.resolution <= 0 or not 0 <= self.free_thresh < self.occupied_thresh <= 1:
            raise ValueError('Invalid occupancy resolution/thresholds')
        if data.get('mode', 'trinary') != 'trinary':
            raise ValueError('Only trinary occupancy maps are supported')
        # The repository owns this image; ignore any obsolete absolute prefix.
        image_path = Path(data['image'])
        if image_path.is_absolute():
            image_path = Path(image_path.name)
        with Image.open(path.parent / image_path) as image:
            self.image = image.convert('L')
        self.width, self.height = self.image.size

    def pixel(self, x, y):
        """Inverse map origin rotation, floor cells, then flip PGM's Y axis."""
        ox, oy, yaw = self.origin
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = x - ox, y - oy
        col = math.floor((c * dx + s * dy) / self.resolution)
        row = self.height - 1 - math.floor((-s * dx + c * dy) / self.resolution)
        return col, row

    def cell_state(self, col, row):
        if not (0 <= col < self.width and 0 <= row < self.height):
            return 'unknown'
        shade = self.image.getpixel((col, row)) / 255
        occupancy = shade if self.negate else 1 - shade
        if occupancy > self.occupied_thresh:
            return 'occupied'
        return 'free' if occupancy < self.free_thresh else 'unknown'

    def is_free(self, x, y, clearance=0):
        col, row = self.pixel(x, y)
        radius = math.ceil(clearance / self.resolution)
        # Conservative square clearance around the goal; this is not a planner
        # or a substitute for Nav2's live costmap/footprint collision checks.
        return all(self.cell_state(col + dx, row + dy) == 'free'
                   for dx in range(-radius, radius + 1)
                   for dy in range(-radius, radius + 1))


class RoomGoals:
    def __init__(self, directory=MAP_DIRECTORY, scene_path=SIMULATOR / 'scene_graph.json'):
        directory = Path(directory)
        self.alignment = MapAlignment.load(directory / 'alignment.yaml')
        metadata = yaml.safe_load((directory / 'alignment.yaml').read_text())
        self.map = OccupancyMap(directory / 'map.yaml')
        locations = {o['id']: o for o in json.loads(Path(scene_path).read_text())['objects']
                     if o.get('type') == 'Location'}
        self.bounds = {}
        for room in metadata['room_bounds']['rooms']:
            location = locations[room['id']]
            if any(abs(a - b) > 1e-5 for a, b in zip(
                    location['qualities']['location'], room['raw_scene_position'])):
                raise ValueError(f"Scene/alignment mismatch for {room['id']}")
            bounds = get_location_bounds(location)
            if bounds is None:
                raise ValueError(f"Missing room geometry: {room['id']}")
            self.bounds[room['id']] = bounds
        params = yaml.safe_load((SIMULATOR / 'nav2_params_jazzy.yaml').read_text())
        costmap = params['global_costmap']['global_costmap']['ros__parameters']
        self.clearance = max(costmap['robot_radius'], costmap['inflation_layer']['inflation_radius'])

    def sample_room_goal(self, room_id, seed=None):
        bounds = self.bounds[room_id]
        rng = random.Random(seed)
        for _ in range(10000):
            x = rng.uniform(bounds['min_x'], bounds['max_x'])
            y = rng.uniform(bounds['min_y'], bounds['max_y'])
            goal = self.alignment.scene_to_map(x, y)
            if self.map.is_free(*goal, clearance=self.clearance):
                return goal
        raise RuntimeError(f'No free goal sampled in {room_id}; map coverage/clearance may be insufficient')


class RoomNavigator:
    def __init__(self):
        # ROS is only needed when a client is actually constructed.
        import rclpy
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose
        self.rclpy = rclpy
        self.action_type = NavigateToPose
        self.goals = RoomGoals()
        # The application owns the default context used by the global executor.
        self.node = rclpy.create_node('cmoc_room_navigator')
        self.client = ActionClient(self.node, NavigateToPose, '/navigate_to_pose')
        self.active_goal = None

    def wait_until_ready(self, timeout=120):
        self.wait_for_nav2_active(timeout)

    def wait_for_nav2_active(self, timeout=120):
        """Gate navigation on lifecycle activation, not action discovery."""
        from lifecycle_msgs.msg import State
        from lifecycle_msgs.srv import GetState

        if timeout <= 0:
            raise ValueError('timeout must be positive')
        names = ('bt_navigator', 'planner_server', 'controller_server')
        states = dict.fromkeys(names, 'service unavailable')
        clients = {}
        deadline = time.monotonic() + timeout
        previous = None
        print('Waiting for Nav2...', flush=True)
        try:
            for name in names:
                clients[name] = self.node.create_client(GetState, f'/{name}/get_state')
            while time.monotonic() < deadline:
                if not self.rclpy.ok():
                    raise RuntimeError('ROS context is not active')
                all_active = True
                for name, client in clients.items():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        all_active = False
                        break
                    if not client.service_is_ready():
                        states[name] = 'service unavailable'
                        all_active = False
                        continue
                    future = client.call_async(GetState.Request())
                    try:
                        response = self._wait(future, min(0.5, remaining))
                    except TimeoutError:
                        client.remove_pending_request(future)
                        states[name] = 'response timeout'
                        all_active = False
                        continue
                    state = response.current_state
                    states[name] = state.label or str(state.id)
                    all_active &= state.id == State.PRIMARY_STATE_ACTIVE
                summary = ', '.join(f'{name}={state}' for name, state in states.items())
                if summary != previous:
                    print(summary, flush=True)
                    previous = summary
                if all_active:
                    print('Nav2 ready.', flush=True)
                    return
                time.sleep(min(0.5, max(0, deadline - time.monotonic())))
            summary = ', '.join(f'{name}={state}' for name, state in states.items())
            raise RuntimeError(f'Nav2 did not become active within {timeout}s: {summary}')
        finally:
            for client in clients.values():
                self.node.destroy_client(client)

    def _wait(self, future, timeout):
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        if not future.done():
            raise TimeoutError('Nav2 action timed out')
        return future.result()

    def navigate_to(self, x, y, yaw=0.0):
        if not self.rclpy.ok():
            raise RuntimeError("ROS context is not active")
        from action_msgs.msg import GoalStatus
        goal = self.action_type.Goal()
        goal.pose.header.frame_id = 'map'
        # Zero timestamp requests the latest available transform.
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)
        print(f'Goal: ({x:.2f}, {y:.2f})', flush=True)
        self.active_goal = self._wait(self.client.send_goal_async(goal), 30)
        if not self.active_goal.accepted:
            self.active_goal = None
            print('Navigation failed: goal rejected', flush=True)
            return False
        try:
            result = self._wait(self.active_goal.get_result_async(), 300)
            success = result.status == GoalStatus.STATUS_SUCCEEDED
        except TimeoutError:
            self._wait(self.active_goal.cancel_goal_async(), 5)
            success = False
        self.active_goal = None
        print('Navigation succeeded' if success else 'Navigation failed', flush=True)
        return success

    def current_map_position(self, timeout=5):
        """Read the current Nav2 robot pose, not its pre-plan episodic position."""
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener

        buffer = Buffer()
        listener = TransformListener(buffer, self.node, spin_thread=False)
        future = None
        try:
            future = buffer.wait_for_transform_async('map', 'base_link', Time())
            transform = self._wait(future, timeout)
            position = transform.transform.translation
            return position.x, position.y
        except Exception as error:
            raise RuntimeError(f'Current robot map pose unavailable: {error}') from error
        finally:
            if future is not None and not future.done():
                future.cancel()
            listener.unregister()

    def approach_entity(self, target, scene_position):
        target_position = self.goals.alignment.scene_to_map(*scene_position)
        x, y, yaw = entity_approach_pose(self.current_map_position(), target_position)
        print(f'Entity approach (map frame):\n  target={target}\n'
              f'  target_position={target_position}\n'
              f'  approach_position=({x:.3f}, {y:.3f})\n  yaw={yaw:.3f}', flush=True)
        return self.navigate_to(x, y, yaw)

    def sample_room_goal(self, room_id, seed=None):
        return self.goals.sample_room_goal(room_id, seed)

    def go_to_room(self, room_id, seed=None):
        print(f'Navigating to {room_id}', flush=True)
        return self.navigate_to(*self.sample_room_goal(room_id, seed))

    def close(self):
        try:
            if self.active_goal is not None and self.active_goal.accepted:
                self._wait(self.active_goal.cancel_goal_async(), 5)
        finally:
            self.client.destroy()
            self.node.destroy_node()
