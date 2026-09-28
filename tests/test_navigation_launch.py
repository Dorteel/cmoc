"""Inspect real launch actions without starting Webots or a second Nav2 stack."""
import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess, EmitEvent
from launch.events import Shutdown
from launch.events.process import ProcessExited
from launch_ros.actions import Node

from simulator_launcher import SIMULATOR


def load_launch():
    spec = importlib.util.spec_from_file_location('cmoc_navigation_launch', SIMULATOR / 'launch/navigation.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('doors', ['', str(Path(__file__).resolve().parents[1] / 'config/navigation_doors.json')])
def test_nav2_is_included_once_without_door_wait_gate(doors):
    context = LaunchContext()
    context.launch_configurations.update({
        'params_file': str(SIMULATOR / 'nav2_params_jazzy.yaml'),
        'map': str(SIMULATOR / 'maps/kitchen.yaml'),
        'map_to_odom': '0,0,0', 'doors_config': doors})
    actions = load_launch()._launch(context)
    handlers = [a for a in actions if isinstance(a, RegisterEventHandler)]
    probes = [a for a in actions if isinstance(a, ExecuteProcess) and not isinstance(a, Node)]
    assert len(handlers) == len(probes) == 1
    probe = probes[0]
    command = [''.join(context.perform_substitution(s) for s in part) for part in probe.cmd]
    assert '/wheel/odom' in command
    assert '--once' in command
    assert not any('wait_for_mapping_doors' in part for part in command)
    handler = handlers[0].event_handler
    event = ProcessExited(action=probe, returncode=0, name='readiness', cmd=command, cwd=None, env=None, pid=1)
    assert handler.matches(event)
    released = list(handler.handle(event, context))
    assert not any(isinstance(a, IncludeLaunchDescription) and
                   'nav2_bringup' in str(a.launch_description_source.location)
                   for a in actions)
    for code in (1, 124):
        failure = ProcessExited(action=probe, returncode=code, name='readiness',
                                cmd=command, cwd=None, env=None, pid=1)
        failed = list(handler.handle(failure, context))
        assert len(failed) == 1
        assert isinstance(failed[0], EmitEvent)
        assert isinstance(failed[0].event, Shutdown)
    actions = actions + released
    includes = [action for action in actions if isinstance(action, IncludeLaunchDescription)]
    for action in includes:
        action.launch_description_source.get_launch_description(context)
    paths = [action.launch_description_source.location for action in includes]
    assert len(paths) == 2
    assert sum(path.endswith('/nav2_bringup/launch/navigation_launch.py') for path in paths) == 1
    assert sum(path.endswith('/tiago_apartment_ros2.launch.py') for path in paths) == 1
    # Door status does not participate in releasing Nav2.
    navigation = includes[next(i for i, path in enumerate(paths) if path.endswith('/navigation_launch.py'))]
    assert dict(navigation.launch_arguments)['autostart'] == 'True'
    executables = [action.node_executable
                   for action in actions if isinstance(action, Node)]
    assert executables.count('map_server') == 1
    assert executables.count('lifecycle_manager') == 1
    assert executables.count('scan_to_scan_filter_chain') == 1
    apartment = includes[next(i for i, path in enumerate(paths) if path.endswith('/tiago_apartment_ros2.launch.py'))]
    assert dict(apartment.launch_arguments)['mapping_doors_config'] == doors
