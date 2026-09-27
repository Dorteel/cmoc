"""Mocked process tests: never start ROS or Webots."""

import signal
import subprocess
import unittest
from unittest.mock import Mock, patch

from simulator_launcher import SIMULATOR, SimulatorLauncher


class SimulatorLauncherTests(unittest.TestCase):
    @patch('simulator_launcher.subprocess.Popen')
    def test_start_command_and_repeated_start(self, popen):
        launcher = SimulatorLauncher()
        popen.return_value.poll.return_value = None
        launcher.start()
        command = popen.call_args.args[0]
        self.assertEqual(command[:3],
                         ['ros2', 'launch', str(SIMULATOR / 'launch/navigation.launch.py')])
        self.assertTrue(any(arg.endswith('apartment_room_aligned/map.yaml') for arg in command))
        self.assertTrue(any(arg.startswith('map_to_odom:=') for arg in command))
        self.assertTrue(any(arg.endswith('config/navigation_doors.json') for arg in command))
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertTrue(launcher.is_running())
        with self.assertRaises(RuntimeError):
            launcher.start()
        popen.return_value.poll.return_value = 1
        self.assertFalse(launcher.is_running())

    @patch.object(SimulatorLauncher, '_stop_process')
    @patch('simulator_launcher.subprocess.Popen')
    def test_readiness_sample(self, popen, stop):
        launcher = SimulatorLauncher()
        launcher._process = Mock()
        launcher._process.poll.return_value = None
        popen.return_value.wait.return_value = 0
        launcher.wait_until_ready()
        command = popen.call_args.args[0]
        self.assertEqual(command[:5], ['ros2', 'topic', 'echo', '/wheel/odom',
                                     'nav_msgs/msg/Odometry'])
        self.assertIn('--once', command)
        stop.assert_called_once_with(popen.return_value)

    @patch.object(SimulatorLauncher, '_stop_process')
    @patch('simulator_launcher.subprocess.Popen')
    @patch('simulator_launcher.time.monotonic', side_effect=[0, 0, 2])
    def test_readiness_timeout_cleans_probe(self, clock, popen, stop):
        launcher = SimulatorLauncher()
        launcher._process = Mock()
        launcher._process.poll.return_value = None
        popen.return_value.wait.side_effect = subprocess.TimeoutExpired('ros2', 0.2)
        with self.assertRaisesRegex(TimeoutError, '/wheel/odom'):
            launcher.wait_until_ready(timeout=1)
        stop.assert_called_once_with(popen.return_value)

    @patch.object(SimulatorLauncher, '_stop_process')
    @patch('simulator_launcher.subprocess.Popen')
    def test_early_exit_and_probe_failure(self, popen, stop):
        launcher = SimulatorLauncher()
        launcher._process = Mock()
        launcher._process.poll.side_effect = [None, 1]
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            launcher.wait_until_ready()
        stop.assert_called_once()
        launcher._process.poll.side_effect = None
        launcher._process.poll.return_value = None
        popen.return_value.wait.return_value = 2
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            launcher.wait_until_ready()

    @patch('simulator_launcher.os.killpg', side_effect=[None, ProcessLookupError])
    def test_stop_signals_owned_group_and_is_idempotent(self, killpg):
        launcher = SimulatorLauncher()
        process = launcher._process = Mock(pid=123)
        launcher.stop()
        self.assertEqual(killpg.call_args_list[0].args, (123, signal.SIGINT))
        process.wait.assert_called_once()
        launcher.stop()
        self.assertFalse(launcher.is_running())

    @patch('simulator_launcher.time.monotonic', side_effect=[0, 11, 11, 17, 17, 19])
    @patch('simulator_launcher.os.killpg')
    def test_stop_escalates_for_stubborn_children(self, killpg, clock):
        process = Mock(pid=123)
        SimulatorLauncher._stop_process(process)
        self.assertEqual([call.args[1] for call in killpg.call_args_list],
                         [signal.SIGINT, signal.SIGTERM, signal.SIGKILL])
        process.wait.assert_called_once()


if __name__ == '__main__':
    unittest.main()
