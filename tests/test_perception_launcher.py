"""Mocked perception startup/readiness; never launches the real server."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from perception_launcher import PerceptionLauncher


class PerceptionLauncherTests(unittest.TestCase):
    def setUp(self):
        self.ros = Mock()
        self.ros.ok.return_value = True
        self.client = Mock()
        self.action_type = Mock()
        self.factory = Mock(return_value=self.client)
        modules = {
            'rclpy': self.ros,
            'rclpy.action': SimpleNamespace(ActionClient=self.factory),
            'cmoc_interfaces.action': SimpleNamespace(ObserveWithVLM=self.action_type),
        }
        patcher = patch.dict('sys.modules', modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.launcher = PerceptionLauncher()

    @patch('perception_launcher.subprocess.Popen')
    @patch('perception_launcher.SimulatorLauncher._stop_process')
    def test_start_command_once_and_stop_owned_process(self, stop, popen):
        self.launcher.start()
        popen.assert_called_once_with(
            ['ros2', 'run', 'cmoc_perception', 'observe_with_vlm_server',
             '--ros-args', '-p', 'backend:=ollama'],
            start_new_session=True)
        with self.assertRaisesRegex(RuntimeError, 'already started'):
            self.launcher.start()
        self.launcher.stop()
        self.launcher.stop()
        stop.assert_called_once_with(popen.return_value)

    def test_ready_uses_action_api_and_keeps_ros_alive(self):
        self.launcher._process = Mock()
        self.launcher._process.poll.return_value = None
        self.launcher.wait_for_observe_with_vlm()
        self.factory.assert_called_once_with(self.ros.create_node.return_value,
                                             self.action_type, '/observe_with_vlm')
        self.client.wait_for_server.assert_called_once()
        self.client.destroy.assert_called_once()
        self.ros.create_node.return_value.destroy_node.assert_called_once()
        self.ros.init.assert_not_called()
        self.ros.shutdown.assert_not_called()

    @patch('perception_launcher.time.monotonic', side_effect=[0, 0, 60])
    def test_timeout_reports_action_and_deadline(self, clock):
        self.launcher._process = Mock()
        self.launcher._process.poll.return_value = None
        self.client.wait_for_server.return_value = False
        with self.assertRaisesRegex(RuntimeError,
                '/observe_with_vlm action server did not become ready within 60s'):
            self.launcher.wait_for_observe_with_vlm()
        self.client.destroy.assert_called_once()
        self.ros.create_node.return_value.destroy_node.assert_called_once()

    def test_early_process_exit_fails_clearly(self):
        self.launcher._process = Mock()
        self.launcher._process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'process exited'):
            self.launcher.wait_for_observe_with_vlm()
        self.client.wait_for_server.assert_not_called()
        self.client.destroy.assert_called_once()



    @patch.dict('os.environ', {'NEBULA_API_KEY': 'test-placeholder'})
    @patch('perception_launcher.subprocess.Popen')
    def test_nebula_uses_ros_parameters_and_inherited_key(self, popen):
        launcher = PerceptionLauncher(backend='nebula')
        launcher.start()
        command = popen.call_args.args[0]
        self.assertEqual(command[-3:], ['--ros-args', '-p', 'backend:=nebula'])
        self.assertNotIn('test-placeholder', ' '.join(command))
        self.assertNotIn('env', popen.call_args.kwargs)
        self.assertFalse(any('model:=' in arg for arg in command))  # Server's existing model.

    @patch.dict('os.environ', {}, clear=True)
    @patch('perception_launcher.subprocess.Popen')
    def test_missing_nebula_key_allows_server_fallback(self, popen):
        PerceptionLauncher(backend='nebula').start()
        popen.assert_called_once()

    @patch.dict('os.environ', {}, clear=True)
    @patch('perception_launcher.subprocess.Popen')
    def test_ollama_remains_available_without_nebula_key(self, popen):
        PerceptionLauncher(backend='ollama', model='test-model').start()
        self.assertEqual(popen.call_args.args[0][-2:], ['-p', 'ollama_model:=test-model'])



    @patch.dict('os.environ', {'NEBULA_API_KEY': 'inheritance-test-placeholder'})
    def test_actual_child_inherits_key_without_env_override(self):
        import subprocess
        import sys
        real_popen = subprocess.Popen
        children = []

        def child_probe(command, **kwargs):
            self.assertNotIn('env', kwargs)
            child = real_popen([sys.executable, '-c',
                               'import os; print(bool(os.environ.get("NEBULA_API_KEY")))'],
                              stdout=subprocess.PIPE, text=True, **kwargs)
            children.append(child)
            return child

        with patch('perception_launcher.subprocess.Popen', side_effect=child_probe):
            PerceptionLauncher(backend='nebula').start()
        output, _ = children[0].communicate(timeout=5)
        self.assertEqual(output.strip(), 'True')
        self.assertEqual(children[0].returncode, 0)


if __name__ == '__main__':
    unittest.main()


def test_cache_flag_sets_absolute_runtime_directory():
    from pathlib import Path
    with patch('perception_launcher.subprocess.Popen') as popen:
        PerceptionLauncher(backend='nebula', vlm_cache=True).start()
    command = popen.call_args.args[0]
    directory = Path(__file__).resolve().parents[1] / '.cache/vlm'
    assert f'vlm_cache_dir:={directory}' in command


def test_search_fresh_frames_use_simulator_clock_and_legacy_stays_default():
    for fresh in (False, True):
        with patch('perception_launcher.subprocess.Popen') as popen:
            PerceptionLauncher(fresh_frames=fresh).start()
        command = popen.call_args.args[0]
        assert ('fresh_camera_frames:=true' in command) is fresh
        assert ('use_sim_time:=true' in command) is fresh


def test_record_flag_sets_single_cache_directory_and_save_parameter():
    with patch('perception_launcher.subprocess.Popen') as popen:
        PerceptionLauncher(vlm_cache_save=True).start()
    command = popen.call_args.args[0]
    assert 'vlm_cache_save:=true' in command
    assert any(arg.startswith('vlm_cache_dir:=') for arg in command)
