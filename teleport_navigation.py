"""Optional Webots movement backend; all metric goal resolution stays in RoomNavigator."""

import json
import math

from navigation import RoomNavigator, SIMULATOR
from external.webots_ros2_simulation.controllers.fallback_action_supervisor.action_cli import send_action


class TeleportNavigator(RoomNavigator):
    def __init__(self):
        # Keep TIAGo at its configured world height; navigation goals only specify XY.
        scene = json.loads((SIMULATOR / 'scene_graph.json').read_text())
        robot = next(obj for obj in scene['objects'] if obj['id'] == 'TIAGo')
        self.robot_height = float(robot['qualities']['location'][2])
        if not math.isfinite(self.robot_height):
            raise ValueError('TIAGo requires a finite configured world height')
        super().__init__()
        print('Navigation mode: TELEPORT', flush=True)

    def navigate_to(self, x, y, yaw=0.0):
        # Reuse exactly the resolved map pose, converting back to Webots ENU.
        alignment = self.goals.alignment
        world_x, world_y = alignment.map_to_scene(x, y)
        world_yaw = math.atan2(math.sin(yaw - alignment.yaw), math.cos(yaw - alignment.yaw))
        print(f'Teleporting TIAGo to: ({world_x:.3f}, {world_y:.3f}, {world_yaw:.3f}) [Webots world]', flush=True)
        try:
            response = send_action('move', {
                'robot': 'TIAGo',
                'coordinates': [world_x, world_y, self.robot_height],
                'rotation': [0.0, 0.0, 1.0, world_yaw],
            })
        except (OSError, ValueError) as error:
            raise RuntimeError(f'Webots teleport facility unavailable or failed: {error}') from error
        if not response.get('ok'):
            raise RuntimeError(f"Webots teleport failed: {response.get('error', 'unknown supervisor error')}")
        return True
