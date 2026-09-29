"""TF/CameraInfo boundaries and pure ROS optical geometry."""
from math import atan, pi, sin, cos
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET
import pytest
import yaml
from ros_camera_geometry import geometry_from_tf, acquire_camera_geometry, CAMERA_INFO_TOPIC
from external.webots_ros2_simulation.controllers.fallback_action_supervisor.camera_grounding import select_camera_candidate


def transform(x=0,y=0,z=0,yaw=0):
    return NS(transform=NS(translation=NS(x=x,y=y,z=z),rotation=NS(x=0,y=0,z=sin(yaw/2),w=cos(yaw/2))))


def info():
    return NS(header=NS(frame_id='Astra_rgb',stamp=NS(sec=2,nanosec=0)),width=640,height=480,k=[320,0,320,0,400,240,0,0,1])


def test_tf_camera_intrinsics_and_nonidentity_map_alignment():
    buffer = Mock()
    # scene robot (1,2,0) maps to (8,21,0) after +90deg, translation (10,20).
    buffer.lookup_transform.side_effect = [transform(8,21,1),transform(10,20,yaw=pi/2),transform(8,21)]
    log = Mock()
    result = geometry_from_tf(buffer,info(),'stamp',[1,2,0],log)
    assert result['camera'][0] == [8,21,1]
    assert result['camera'][2:] == pytest.approx([pi/2,2*atan(480/800)])
    assert [call.args for call in buffer.lookup_transform.call_args_list] == [
        ('map','Astra_rgb','stamp'),('map','odom','stamp'),('map','base_link','stamp')]
    assert 'difference=0.000m' in log.call_args_list[4].args[0]


def test_coordinate_disagreement_stops_grounding():
    buffer = Mock()
    buffer.lookup_transform.return_value = transform()
    with pytest.raises(ValueError,match='coordinate mismatch'):
        geometry_from_tf(buffer,info(),None,[2,0,0],Mock())


@pytest.mark.parametrize('field,value', [('frame_id',''),('fx',0)])
def test_empty_frame_or_invalid_intrinsics_fail(field,value):
    message = info()
    if field == 'frame_id': message.header.frame_id = value
    else: message.k[0] = value
    buffer = Mock()
    expected = 'CameraInfo received but header.frame_id is empty' if field == 'frame_id' else 'Invalid RGB CameraInfo intrinsics'
    with pytest.raises(ValueError, match=expected):
        geometry_from_tf(buffer,message,None,[0,0,0],Mock())
    buffer.lookup_transform.assert_not_called()


@pytest.mark.parametrize('other', [(0,0,-2),(3,0,2),(0,3,2)])
def test_optical_positive_z_and_frustum_rejection(other):
    from test_search_execution_grounding import node
    matches = [('center',node(1,'center',position=[0,0,2])), ('other',node(2,'other',position=other))]
    selected,details = select_camera_candidate(matches,([0,0,0],[1,0,0,0,1,0,0,0,1],pi/2,pi/2))
    assert selected[0][0] == 'center'
    assert 'depth=2.00m' in details[0] and 'visible=no' in details[1]


def test_scene_to_map_applied_before_projection():
    from test_search_execution_grounding import node
    selected,_ = select_camera_candidate([('target',node(1,'target',position=[1,2,3]))],
        ([10,21,0],[1,0,0,0,1,0,0,0,1],pi/2,pi/2),
        ([12,20,0],[0,-1,0,1,0,0,0,0,1]))
    assert selected[0][0] == 'target'


def test_real_head_states_replace_zero_placeholders_without_commands():
    directory=Path('external/webots_ros2_simulation/config')
    tree=ET.parse(directory/'tiago_webots_wheels.urdf')
    config=yaml.safe_load((directory/'ros2_control_wheel_odom.yml').read_text())
    for name in ('head_1_joint','head_2_joint'):
        joint=tree.find(f"./ros2_control/joint[@name='{name}']")
        assert joint.find("state_interface[@name='position']") is not None
        assert joint.find('command_interface') is None
        assert name not in config['joint_state_broadcaster']['ros__parameters']['extra_joints']


@pytest.mark.parametrize('failure', [False,True])
def test_runtime_tf_acquisition_is_bounded_and_has_no_fallback(failure):
    import rclpy
    import tf2_ros
    from sensor_msgs.msg import CameraInfo
    from tf2_ros import TransformException
    node, listener, buffer = Mock(),Mock(),Mock()
    message=CameraInfo()
    message.header.frame_id='reported_camera_optical_frame'
    message.header.stamp.sec=2
    message.width,message.height=640,480
    message.k=[320.,0.,320.,0.,400.,240.,0.,0.,1.]
    callback=[]
    node.create_subscription.side_effect=lambda typ,topic,cb,qos:callback.append(cb)
    def spin(*args,**kwargs):
        callback[0](message)
    buffer.lookup_transform.side_effect=TransformException('missing camera TF') if failure else None
    buffer.lookup_transform.return_value=transform()
    log=Mock()
    with patch.object(rclpy,'ok',return_value=True), patch.object(rclpy,'create_node',return_value=node), patch.object(rclpy,'spin_once',side_effect=spin), patch.object(tf2_ros,'Buffer',return_value=buffer), patch.object(tf2_ros,'TransformListener',return_value=listener), patch('ros_camera_geometry.time.monotonic',side_effect=[0,0,0,6]):
        if failure:
            with pytest.raises(RuntimeError,match='after 5.0s: missing camera TF'):
                acquire_camera_geometry([0,0,0],log)
            assert 'TF acquired=no' in log.call_args.args[0]
        else:
            result=acquire_camera_geometry([0,0,0],log)
            assert result['camera'][0]==[0,0,0]
    camera_calls = [call.args for call in buffer.lookup_transform.call_args_list if call.args[1] not in ('odom', 'base_link')]
    assert camera_calls
    assert all(args[1] == message.header.frame_id for args in camera_calls)
    assert all(call.args[1] != 'Astra_rgb' for call in buffer.lookup_transform.call_args_list)
    assert node.create_subscription.call_args.args[1]==CAMERA_INFO_TOPIC
    listener.unregister.assert_called_once()
    node.destroy_node.assert_called_once()


def test_reported_frame_and_diagnostics_precede_tf_lookup():
    message = info()
    message.header.frame_id = 'another_camera_optical_frame'
    log = Mock()
    buffer = Mock()

    def lookup(target, source, stamp):
        if source == message.header.frame_id:
            assert [call.args[0] for call in log.call_args_list] == [
                f'[GROUNDING] CameraInfo topic: {CAMERA_INFO_TOPIC}',
                f'[GROUNDING] CameraInfo frame_id: {message.header.frame_id}',
                f'[GROUNDING] TF lookup: map -> {message.header.frame_id}',
                '[GROUNDING] Requested camera time: 2.000',
            ]
        assert source != 'Astra_rgb'
        return transform()

    buffer.lookup_transform.side_effect = lookup
    geometry_from_tf(buffer, message, 'stamp', [0,0,0], log)
    assert buffer.lookup_transform.call_args_list[0].args == (
        'map', message.header.frame_id, 'stamp')


@pytest.mark.parametrize('offset_ns', [0, 60_000_000, -60_000_000, 100_000_000,
                                      -100_000_000, 100_000_001, -100_000_001, 2_000_000_000])
def test_camera_tf_timestamp_skew_is_strictly_bounded(offset_ns):
    from geometry_msgs.msg import TransformStamped
    from rclpy.time import Time
    from tf2_ros import Buffer, ExtrapolationException

    # Use real tf2 interpolation/extrapolation, not mocked error messages.
    buffer = Buffer()
    message = info()
    message.header.frame_id = 'Astra rgb'
    requested = Time(seconds=33, nanoseconds=140_000_000)
    message.header.stamp = requested.to_msg()
    available = Time(nanoseconds=requested.nanoseconds + offset_ns)
    for child in ('Astra rgb', 'odom', 'base_link'):
        tf = TransformStamped()
        tf.header.frame_id = 'map'
        tf.child_frame_id = child
        tf.header.stamp = available.to_msg()
        tf.transform.rotation.w = 1.0
        buffer.set_transform(tf, 'test')
    log = Mock()
    with patch.object(buffer, 'lookup_transform', wraps=buffer.lookup_transform) as lookup:
        if abs(offset_ns) > 100_000_000:
            with pytest.raises(ExtrapolationException):
                geometry_from_tf(buffer, message, requested, [0,0,0], log)
            assert lookup.call_count == 2
        else:
            result = geometry_from_tf(buffer, message, requested, [0,0,0], log)
            assert result['camera'][0] == [0,0,0]
            assert lookup.call_count == (3 if offset_ns == 0 else 4)
            # Alignment and robot checks must share the camera's selected time.
            assert all(call.args[2].nanoseconds == available.nanoseconds
                       for call in lookup.call_args_list[-2:])
        assert lookup.call_args_list[0].args == ('map', 'Astra rgb', requested)
    logs = [call.args[0] for call in log.call_args_list]
    assert '[GROUNDING] Requested camera time: 33.140' in logs
    used = [line for line in logs if 'Using nearest available TF' in line]
    assert bool(used) == (0 < abs(offset_ns) <= 100_000_000)
    if offset_ns == 60_000_000:
        assert '[GROUNDING] Exact TF unavailable; skew=0.060s' in logs
        assert '[GROUNDING] Using nearest available TF at 33.200' in logs


@pytest.mark.parametrize('error_name', ['LookupException', 'ConnectivityException', 'TransformException'])
def test_unrelated_tf_errors_never_request_latest(error_name):
    import tf2_ros
    from rclpy.time import Time
    error = getattr(tf2_ros, error_name)('unrelated TF failure')
    buffer = Mock()
    buffer.lookup_transform.side_effect = error
    requested = Time(seconds=2)
    with pytest.raises(type(error), match='unrelated TF failure'):
        geometry_from_tf(buffer, info(), requested, [0,0,0], Mock())
    buffer.lookup_transform.assert_called_once_with('map', info().header.frame_id, requested)
