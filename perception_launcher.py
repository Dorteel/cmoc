"""Lifecycle wrapper for the existing CMOC perception executable."""

from pathlib import Path
import subprocess
import time

from simulator_launcher import SimulatorLauncher


class PerceptionLauncher:
    def __init__(self, backend="ollama", model=None, *, vlm_cache=False, fresh_frames=False):
        if backend not in ("ollama", "nebula"):
            raise ValueError("Unsupported perception backend; expected ollama or nebula")
        self.backend = backend
        self.model = model
        self.vlm_cache = vlm_cache
        self.fresh_frames = fresh_frames
        self._process = None

    def start(self):
        if self._process is not None:
            raise RuntimeError('Perception already started; call stop() before restarting')
        command = ['ros2', 'run', 'cmoc_perception', 'observe_with_vlm_server',
                   '--ros-args', '-p', f'backend:={self.backend}']
        if self.fresh_frames:
            command.extend(['-p', 'fresh_camera_frames:=true', '-p', 'use_sim_time:=true'])
        if self.vlm_cache:
            directory = Path(__file__).resolve().parent / '.cache/vlm'
            command.extend(['-p', f'vlm_cache_dir:={directory}'])
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
