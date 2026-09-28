"""Own the existing Webots ROS launch process and its readiness probe (Linux)."""

import os
from pathlib import Path
import signal
import subprocess
import time

from external.webots_ros2_simulation.controllers.fallback_action_supervisor.command_server import check_port_available


SIMULATOR = Path(__file__).resolve().parent / 'external' / 'webots_ros2_simulation'


class SimulatorLauncher:
    def __init__(self):
        self._process = None

    def start(self):
        if self._process is not None:
            raise RuntimeError('Simulator already started; call stop() before restarting.')
        launch = SIMULATOR / 'launch' / 'navigation.launch.py'
        if not launch.is_file():
            raise FileNotFoundError('Simulator missing: run git submodule update --init --recursive')
        check_port_available()
        from navigation import MAP_DIRECTORY, MapAlignment
        alignment = MapAlignment.load(MAP_DIRECTORY / 'alignment.yaml')
        command = ['ros2', 'launch', str(launch),
                   f'map:={MAP_DIRECTORY / "map.yaml"}',
                   f'map_to_odom:={alignment.x},{alignment.y},{alignment.yaw}',
                   f'doors_config:={Path(__file__).resolve().parent / "config/navigation_doors.json"}']
        try:
            self._process = subprocess.Popen(
                command, cwd=SIMULATOR,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError('ros2 not found; source the ROS workspace first.') from exc

    def is_running(self):
        return self._process is not None and self._process.poll() is None

    def wait_until_ready(self, timeout=120):
        """Wait for an actual sample from TIAGo's active diff-drive controller.

        Run only one apartment in this ROS domain; an existing publisher cannot
        be distinguished from this launch by its topic name alone.
        """
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        if not self.is_running():
            raise RuntimeError('Simulator is not running')
        deadline = time.monotonic() + timeout
        probe = subprocess.Popen(
            ['ros2', 'topic', 'echo', '/wheel/odom', 'nav_msgs/msg/Odometry',
             '--once', '--field', 'header', '--qos-reliability', 'best_effort'],
            stdout=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            while True:
                if not self.is_running():
                    raise RuntimeError('Simulator launch exited before TIAGo became ready')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'TIAGo not ready after {timeout}s: no /wheel/odom sample')
                try:
                    code = probe.wait(timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    continue
                if code != 0:
                    raise RuntimeError(f'ROS readiness probe failed (exit {code}); check ROS environment')
                return
        finally:
            self._stop_process(probe)

    @staticmethod
    def _stop_process(process):
        # ROS launch handles SIGINT first; escalate for children that fail to exit.
        # Keep checking the group even if its launch parent has already exited.
        for sig, grace in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 1)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                process.poll()  # Reap the parent when it exits.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                continue
            break
        process.wait(timeout=1)

    def stop(self):
        if self._process is not None:
            self._stop_process(self._process)
            self._process = None
