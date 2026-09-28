"""Socket/ownership regression checks; no Webots, ROS, or VLM processes."""

import errno
import os
import select
import signal
import socket
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from simulator_launcher import SimulatorLauncher, SIMULATOR
from external.webots_ros2_simulation.controllers.fallback_action_supervisor import command_server


class SupervisorLifecycleTests(unittest.TestCase):
    def test_single_supervisor_path_in_demo_world_and_launch(self):
        world = (SIMULATOR / 'worlds/setting_the_table_complete_apartment_tiago_ros2.wbt').read_text()
        self.assertEqual(world.count('controller "fallback_action_supervisor"'), 1)
        apartment = (SIMULATOR / 'launch/tiago_apartment_ros2.launch.py').read_text()
        navigation = (SIMULATOR / 'launch/navigation.launch.py').read_text()
        self.assertEqual(navigation.count("str(project / 'launch' / 'tiago_apartment_ros2.launch.py')"), 1)
        self.assertEqual(apartment.count("'/usr/local/webots/webots'"), 1)
        self.assertNotIn('fallback_action_supervisor', apartment + navigation)

    @patch('simulator_launcher.subprocess.Popen')
    @patch('simulator_launcher.check_port_available', side_effect=RuntimeError('occupied'))
    def test_occupied_port_prevents_second_launch(self, check, popen):
        launcher = SimulatorLauncher()
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            launcher.start()
        popen.assert_not_called()
        self.assertIsNone(launcher._process)

    @patch.object(command_server.subprocess, 'run')
    @patch.object(command_server.socket, 'socket')
    def test_occupied_port_reports_address_and_owner_and_closes_socket(self, factory, run):
        factory.return_value.bind.side_effect = OSError(errno.EADDRINUSE, 'Address already in use')
        run.return_value = Mock(returncode=0, stdout='LISTEN 127.0.0.1:8765 users:(("python3",pid=123,fd=4))')
        with self.assertRaisesRegex(RuntimeError, '127.0.0.1:8765.*pid=123'):
            command_server.open_server()
        factory.return_value.close.assert_called_once()

    @patch.object(command_server.subprocess, 'run', side_effect=FileNotFoundError())
    @patch.object(command_server.socket, 'socket')
    def test_owner_lookup_failure_still_reports_clear_error(self, factory, run):
        factory.return_value.bind.side_effect = OSError(errno.EADDRINUSE, 'in use')
        with self.assertRaisesRegex(RuntimeError, '127.0.0.1:8765.*owner unavailable'):
            command_server.check_port_available()

    def test_stop_terminates_owned_supervisor_listener_only(self):
        # An ordinary child inherits the launch group, just like the inspected
        # Webots supervisor. Ephemeral ports here isolate the test from live demos.
        child_code = '''
import os, signal, socket, sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
s = socket.socket()
s.bind(('127.0.0.1', 0))
s.listen()
print(os.getpid(), os.getpgrp(), s.getsockname()[1], flush=True)
signal.pause()
'''
        parent_code = '''
import signal, subprocess, sys
signal.signal(signal.SIGINT, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', sys.argv[1]])
child.wait()
'''
        launcher = SimulatorLauncher()
        process = subprocess.Popen([sys.executable, '-c', parent_code, child_code],
                                   start_new_session=True, stdout=subprocess.PIPE, text=True)
        launcher._process = process
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0], 'child did not start')
            pid, group, port = map(int, process.stdout.readline().split())
            self.assertEqual(group, process.pid)
            with socket.socket() as unrelated:
                unrelated.bind(('127.0.0.1', 0))
                unrelated.listen()
                launcher.stop()
                self.assertIsNotNone(process.poll())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                with socket.socket() as reused:
                    reused.bind(('127.0.0.1', port))
                    reused.listen()  # Immediate restart is possible.
                with socket.create_connection(unrelated.getsockname(), timeout=1):
                    pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()
