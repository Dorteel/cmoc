"""Verify demo setup and cleanup without starting Webots or memory backends."""

import unittest
from unittest.mock import Mock, patch

import demo


class DemoSetupTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('demo.rclpy')
        self.ros = patcher.start()
        self.addCleanup(patcher.stop)
        self.ros.ok.return_value = True

    def tearDown(self):
        self.ros.init.assert_called_once_with(
            args=[], signal_handler_options=demo.SignalHandlerOptions.NO)
        self.ros.shutdown.assert_called_once_with()

    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py'])
    def test_setup_precedes_memory_and_ctrl_c_stops_simulator(self, factory, run, navigator):
        calls = Mock()
        calls.attach_mock(self.ros, 'ros')
        calls.attach_mock(factory.return_value, 'simulator')
        calls.attach_mock(run, 'memory')
        calls.attach_mock(navigator.return_value, 'navigator')
        factory.return_value.is_running.side_effect = KeyboardInterrupt
        demo.main()
        self.assertEqual([call[0] for call in calls.mock_calls],
                         ['ros.init', 'simulator.start', 'simulator.wait_until_ready',
                          'navigator.wait_until_ready', 'memory',
                          'simulator.is_running', 'navigator.close', 'simulator.stop',
                          'ros.ok', 'ros.shutdown'])

    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py'])
    def test_readiness_failure_stops_before_initializing_memory(self, factory, run, navigator):
        factory.return_value.wait_until_ready.side_effect = TimeoutError('not ready')
        with self.assertRaises(TimeoutError):
            demo.main()
        run.assert_not_called()
        factory.return_value.stop.assert_called_once()

    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--no-simulator'])
    def test_no_simulator_runs_existing_demo(self, factory, run, navigator):
        demo.main()
        factory.return_value.start.assert_not_called()
        factory.return_value.wait_until_ready.assert_not_called()
        run.assert_called_once()


    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--test-navigation'])
    def test_movement_flag_passes_ready_navigator(self, factory, run, navigator):
        factory.return_value.is_running.side_effect = KeyboardInterrupt
        demo.main()
        navigator.return_value.wait_until_ready.assert_called_once()
        run.assert_called_once_with(navigator.return_value)
        navigator.return_value.close.assert_called_once()
        factory.return_value.stop.assert_called_once()



    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--test-navigation'])
    def test_cleanup_failure_still_shuts_down_ros(self, factory, run, navigator):
        factory.return_value.is_running.side_effect = KeyboardInterrupt
        navigator.return_value.close.side_effect = RuntimeError('cleanup failed')
        with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
            demo.main()
        factory.return_value.stop.assert_called_once()

    @patch('demo.SimulatorLauncher', side_effect=RuntimeError('construction failed'))
    @patch('sys.argv', ['demo.py'])
    def test_construction_failure_still_shuts_down_ros(self, factory):
        with self.assertRaisesRegex(RuntimeError, 'construction failed'):
            demo.main()



    @patch('demo.RoomNavigator')
    @patch('demo.run_demo')
    @patch('demo.SimulatorLauncher')
    @patch('sys.argv', ['demo.py', '--test-navigation'])
    def test_inactive_nav2_blocks_movement_and_cleans_up(self, factory, run, navigator):
        navigator.return_value.wait_until_ready.side_effect = RuntimeError('bt_navigator=inactive')
        with self.assertRaisesRegex(RuntimeError, 'bt_navigator=inactive'):
            demo.main()
        run.assert_not_called()
        navigator.return_value.go_to_room.assert_not_called()
        navigator.return_value.close.assert_called_once()
        factory.return_value.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
