"""Lifecycle wrapper for the existing CMOC perception executable."""

import os
import subprocess
import time

from simulator_launcher import SimulatorLauncher


class PerceptionLauncher:
    def __init__(self, backend="ollama", model=None):
        if backend not in ("ollama", "nebula"):
            raise ValueError("Unsupported perception backend; expected ollama or nebula")
        self.backend = backend
        self.model = model
        self._process = None

    def start(self):
        if self._process is not None:
            raise RuntimeError('Perception already started; call stop() before restarting')
        if self.backend == "nebula" and not os.environ.get("NEBULA_API_KEY", "").strip():
            raise RuntimeError("NEBULA_API_KEY is not set")
        command = ['ros2', 'run', 'cmoc_perception', 'observe_with_vlm_server',
                   '--ros-args', '-p', f'backend:={self.backend}']
        if self.model is not None:
            command.extend(['-p', f'{self.backend}_model:={self.model}'])
        try:
            self._process = subprocess.Popen(
                command,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError('ros2 not found; source the ROS workspace first') from exc

    def wait_for_observe_with_vlm(self, timeout=60):
        import rclpy
        from rclpy.action import ActionClient
        from cmoc_interfaces.action import ObserveWithVLM

        if timeout <= 0:
            raise ValueError('timeout must be positive')
        if not rclpy.ok():
            raise RuntimeError('ROS context is not active')
        deadline = time.monotonic() + timeout
        node = rclpy.create_node('cmoc_perception_readiness')
        client = None
        try:
            client = ActionClient(node, ObserveWithVLM, '/observe_with_vlm')
            while True:
                if self._process is None or self._process.poll() is not None:
                    raise RuntimeError('CMOC perception process exited before /observe_with_vlm became ready; '
                                       'check that cmoc_perception is built and the workspace is sourced')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f'/observe_with_vlm action server did not become ready within {timeout}s')
                if client.wait_for_server(timeout_sec=min(0.5, remaining)):
                    print('/observe_with_vlm ready', flush=True)
                    return
        finally:
            if client is not None:
                client.destroy()
            node.destroy_node()

    def stop(self):
        if self._process is not None:
            SimulatorLauncher._stop_process(self._process)
            self._process = None
