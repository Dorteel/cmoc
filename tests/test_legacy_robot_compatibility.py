"""The robot bootstrap and wheel configuration are independent of Search mode."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

from launch import LaunchContext
import yaml

import demo
from simulator_launcher import SIMULATOR, SimulatorLauncher


def test_launch_preserves_historical_wheel_configuration_pair():
    path = SIMULATOR / 'launch/tiago_apartment_ros2.launch.py'
    spec = importlib.util.spec_from_file_location('apartment_compatibility', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    urdf = SIMULATOR / 'config/tiago_webots_wheels.urdf'
    control = SIMULATOR / 'config/ros2_control_wheel_odom.yml'
    context = LaunchContext()
    context.launch_configurations.update({
        'robot_urdf': str(urdf), 'world': str(SIMULATOR / 'worlds/setting_the_table_complete_apartment_tiago_ros2.wbt'),
        'mapping_doors_config': '', 'mapping_doors_status_topic': '', 'scan_topic': '/scan'})
    with patch.object(module, 'Node') as node, patch.object(module, 'WebotsController') as driver, \
         patch.object(module, 'Ros2SupervisorLauncher'), patch.object(module, 'WaitForControllerConnection'):
        module._start_apartment(context)
    params = driver.call_args.kwargs['parameters']
    assert params[0]['robot_description'] == str(urdf)
    assert params[0]['set_robot_state_publisher'] is True
    assert params[1] == str(control)
    expected = {'wheel_left_joint', 'wheel_right_joint'}
    assert {j.attrib['name'] for j in ET.parse(urdf).findall('ros2_control/joint')} == expected
    controller = yaml.safe_load(control.read_text())['diffdrive_controller']['ros__parameters']
    assert set(controller['left_wheel_names'] + controller['right_wheel_names']) == expected


def test_legacy_and_search_use_identical_launcher_without_robot_overrides():
    files = [SIMULATOR / 'config/tiago_webots_wheels.urdf', SIMULATOR / 'config/ros2_control_wheel_odom.yml']
    original = [p.read_bytes() for p in files]
    commands = []
    for search in (False, True):
        argv = ['demo.py', '--execute', '--step'] + (['--search'] if search else [])
        with patch('sys.argv', argv), patch('demo.rclpy'), patch('demo.RoomNavigator'), \
             patch('demo.PerceptionLauncher'), patch('demo.create_episodic'), patch('demo.RoboKGNet'), \
             patch('demo.SemanticMemory'), patch('demo.spa_loop') as spa, \
             patch('simulator_launcher.check_port_available'), patch('simulator_launcher.subprocess.Popen') as popen, \
             patch.object(SimulatorLauncher, 'wait_until_ready'), patch.object(SimulatorLauncher, 'stop'), \
             patch.object(SimulatorLauncher, 'is_running', side_effect=KeyboardInterrupt):
            demo.main()
        assert spa.call_args.kwargs['search'] is search
        commands.append(popen.call_args.args[0])
    assert commands[0] == commands[1]
    assert commands[0][:3] == ['ros2', 'launch', str(SIMULATOR / 'launch/navigation.launch.py')]
    assert not any('robot_description' in arg or 'robot_urdf' in arg for arg in commands[0])
    assert [p.read_bytes() for p in files] == original
