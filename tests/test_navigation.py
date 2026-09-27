"""ROS-free map/alignment tests, including the real room-aligned apartment map."""

import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from navigation import MAP_DIRECTORY, MapAlignment, OccupancyMap, RoomGoals


class NavigationGeometryTests(unittest.TestCase):
    def test_real_alignment_loads(self):
        alignment = MapAlignment.load(MAP_DIRECTORY / 'alignment.yaml')
        x, y = alignment.scene_to_map(-1.94, -3.3)
        self.assertAlmostEqual(x, -1.94)
        self.assertAlmostEqual(y, -3.3)

    def test_nonidentity_alignment_loaded_from_yaml(self):
        data = yaml.safe_load((MAP_DIRECTORY / 'alignment.yaml').read_text())
        data['transform'] = {'translation': {'x': 3, 'y': -2}, 'yaw': math.pi / 2}
        data.pop('matrix')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alignment.yaml'
            path.write_text(yaml.safe_dump(data))
            alignment = MapAlignment.load(path)
        x, y = alignment.scene_to_map(2, 1)
        self.assertAlmostEqual(x, 2)
        self.assertAlmostEqual(y, 0)
        sx, sy = alignment.map_to_scene(x, y)
        self.assertAlmostEqual(sx, 2)
        self.assertAlmostEqual(sy, 1)

    def test_pixels_thresholds_negate_and_rotated_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image = Image.new('L', (3, 2))
            image.putdata([255, 0, 205, 255, 255, 255])
            image.save(directory / 'map.pgm')
            data = dict(image='map.pgm', resolution=0.5, origin=[10, 20, math.pi / 2],
                        occupied_thresh=0.65, free_thresh=0.196, negate=0)
            path = directory / 'map.yaml'
            path.write_text(yaml.safe_dump(data))
            occupancy = OccupancyMap(path)
            self.assertEqual(occupancy.pixel(9.75, 20.25), (0, 1))
            self.assertEqual(occupancy.pixel(9.25, 20.75), (1, 0))
            self.assertTrue(occupancy.is_free(9.75, 20.25))
            self.assertFalse(occupancy.is_free(9.25, 20.75))
            self.assertEqual(occupancy.cell_state(2, 0), 'unknown')
            self.assertEqual(occupancy.cell_state(1, 0), 'occupied')
            self.assertFalse(occupancy.is_free(0, 0))
            self.assertFalse(occupancy.is_free(9.75, 20.25, clearance=0.5))
            data['negate'] = 1
            path.write_text(yaml.safe_dump(data))
            occupancy = OccupancyMap(path)
            self.assertEqual(occupancy.cell_state(1, 0), 'free')
            self.assertEqual(occupancy.cell_state(0, 0), 'occupied')

    def test_real_sampled_goals_inside_room_and_free(self):
        rooms = RoomGoals()
        for room in ('KITCHEN', 'LIVING_ROOM_1', 'BATHROOM_1'):
            for seed in (0, 7, 42):
                with self.subTest(room=room, seed=seed):
                    goal = rooms.sample_room_goal(room, seed=seed)
                    self.assertEqual(goal, rooms.sample_room_goal(room, seed=seed))
                    x, y = rooms.alignment.map_to_scene(*goal)
                    bounds = rooms.bounds[room]
                    self.assertLess(bounds['min_x'], x)
                    self.assertLess(x, bounds['max_x'])
                    self.assertLess(bounds['min_y'], y)
                    self.assertLess(y, bounds['max_y'])
                    self.assertTrue(rooms.map.is_free(*goal, clearance=rooms.clearance))
        with self.assertRaises(KeyError):
            rooms.sample_room_goal('NOT_A_ROOM')


class NavigationActionTests(unittest.TestCase):
    def navigator(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from navigation import RoomNavigator
        navigator = RoomNavigator.__new__(RoomNavigator)
        navigator.rclpy = Mock()
        navigator.rclpy.ok.return_value = True
        navigator.client = Mock()
        navigator.active_goal = None
        navigator.action_type = Mock()
        navigator.action_type.Goal.return_value = SimpleNamespace(
            pose=SimpleNamespace(header=SimpleNamespace(frame_id=''),
                                 pose=SimpleNamespace(position=SimpleNamespace(),
                                                      orientation=SimpleNamespace())))
        return navigator

    def test_action_success_and_rejection(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        status_module = SimpleNamespace(GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4))
        with patch.dict('sys.modules', {'action_msgs.msg': status_module}):
            navigator = self.navigator()
            handle = Mock(accepted=True)
            navigator._wait = Mock(side_effect=[handle, SimpleNamespace(status=4)])
            self.assertTrue(navigator.navigate_to(1, 2, math.pi))
            goal = navigator.client.send_goal_async.call_args.args[0]
            self.assertEqual(goal.pose.header.frame_id, 'map')
            self.assertEqual(goal.pose.pose.position.x, 1)
            self.assertAlmostEqual(goal.pose.pose.orientation.z, 1)
            navigator._wait = Mock(return_value=Mock(accepted=False))
            self.assertFalse(navigator.navigate_to(1, 2))

    def test_result_timeout_cancels(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        with patch.dict('sys.modules', {'action_msgs.msg': SimpleNamespace(
                GoalStatus=SimpleNamespace(STATUS_SUCCEEDED=4))}):
            navigator = self.navigator()
            handle = Mock(accepted=True)
            navigator._wait = Mock(side_effect=[handle, TimeoutError(), None])
            self.assertFalse(navigator.navigate_to(1, 2))
            handle.cancel_goal_async.assert_called_once()

    def test_inactive_context_fails_before_sending_goal(self):
        navigator = self.navigator()
        navigator.rclpy.ok.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'ROS context is not active'):
            navigator.navigate_to(1, 2)
        navigator.client.send_goal_async.assert_not_called()

    def test_close_leaves_application_context_alive(self):
        navigator = self.navigator()
        from unittest.mock import Mock
        navigator.node = Mock()
        navigator.close()
        navigator.node.destroy_node.assert_called_once()
        navigator.rclpy.init.assert_not_called()
        navigator.rclpy.shutdown.assert_not_called()
        navigator.rclpy.try_shutdown.assert_not_called()



class Nav2ReadinessTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from navigation import RoomNavigator
        self.now = 0.0
        self.services = {name: Mock() for name in
                         ('bt_navigator', 'planner_server', 'controller_server')}
        self.navigator = RoomNavigator.__new__(RoomNavigator)
        self.navigator.node = Mock()
        self.navigator.node.create_client.side_effect = (
            lambda service, path: self.services[path.split('/')[1]])
        self.navigator.rclpy = Mock()
        self.navigator.rclpy.ok.return_value = True
        self.navigator._wait = Mock(side_effect=lambda future, timeout: future.result())
        modules = {
            'lifecycle_msgs.msg': SimpleNamespace(State=SimpleNamespace(PRIMARY_STATE_ACTIVE=3)),
            'lifecycle_msgs.srv': SimpleNamespace(GetState=Mock()),
        }
        for patcher in (patch.dict('sys.modules', modules),
                        patch('navigation.time.monotonic', side_effect=lambda: self.now),
                        patch('navigation.time.sleep', side_effect=self.advance)):
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in self.services:
            self.states(name, [3])

    def advance(self, seconds):
        self.now += seconds

    def states(self, name, values):
        from types import SimpleNamespace
        from unittest.mock import Mock
        values = iter(values)
        last = [3]

        def response(request):
            last[0] = next(values, last[0])
            state = SimpleNamespace(id=last[0], label={2: 'inactive', 3: 'active'}[last[0]])
            return Mock(result=Mock(return_value=SimpleNamespace(current_state=state)))

        self.services[name].call_async.side_effect = response
        self.services[name].service_is_ready.return_value = True

    def test_inactive_then_active_requires_another_poll(self):
        self.states('bt_navigator', [2, 3])
        self.navigator.wait_until_ready(timeout=2)
        self.assertEqual(self.services['bt_navigator'].call_async.call_count, 2)
        self.assertEqual(self.now, 0.5)
        self.navigator.node.destroy_client.assert_any_call(self.services['bt_navigator'])
        self.navigator.rclpy.shutdown.assert_not_called()

    def test_all_active_returns_immediately(self):
        self.navigator.wait_until_ready()
        self.assertEqual(self.now, 0)
        for name, client in self.services.items():
            client.call_async.assert_called_once()
            self.assertIn('/' + name + '/get_state',
                          [call.args[1] for call in self.navigator.node.create_client.call_args_list])
        self.assertEqual(self.navigator.node.destroy_client.call_count, 3)

    def test_timeout_reports_last_observed_states(self):
        self.states('bt_navigator', [2])
        with self.assertRaisesRegex(RuntimeError,
                'within 1s: bt_navigator=inactive, planner_server=active, controller_server=active'):
            self.navigator.wait_until_ready(timeout=1)
        self.assertEqual(self.navigator.node.destroy_client.call_count, 3)

    def test_missing_service_does_not_count_as_ready(self):
        client = self.services['planner_server']
        client.service_is_ready.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'planner_server=service unavailable'):
            self.navigator.wait_until_ready(timeout=1)
        client.call_async.assert_not_called()

    def test_service_response_timeout_is_bounded_and_cleaned_up(self):
        self.navigator._wait.side_effect = TimeoutError()
        with self.assertRaisesRegex(RuntimeError, 'bt_navigator=response timeout'):
            self.navigator.wait_until_ready(timeout=1)
        self.assertTrue(self.services['bt_navigator'].remove_pending_request.called)


if __name__ == '__main__':
    unittest.main()
