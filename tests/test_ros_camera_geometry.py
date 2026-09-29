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
    assert 'difference=0.000m' in log.call_args_list[0].args[0]


def test_coordinate_disagreement_stops_grounding():
    buffer = Mock()
    buffer.lookup_transform.return_value = transform()
    with pytest.raises(ValueError,match='coordinate mismatch'):
        geometry_from_tf(buffer,info(),None,[2,0,0],Mock())


@pytest.mark.parametrize('field,value', [('frame_id','guessed_frame'),('fx',0)])
def test_unknown_frame_or_invalid_intrinsics_fail(field,value):
    message = info()
    if field == 'frame_id': message.header.frame_id = value
    else: message.k[0] = value
    with pytest.raises(ValueError):
        geometry_from_tf(Mock(),message,None,[0,0,0],Mock())


@pytest.mark.parametrize('other', [(0,0,-2),(3,0,2),(0,3,2)])
def test_optical_positive_z_and_frustum_rejection(other):
    matches = [('center',Mock(getPosition=lambda:[0,0,2])), ('other',Mock(getPosition=lambda:other))]
    selected,details = select_camera_candidate(matches,([0,0,0],[1,0,0,0,1,0,0,0,1],pi/2,pi/2))
    assert selected[0][0] == 'center'
    assert 'depth=2.00m' in details[0] and 'visible=no' in details[1]


def test_scene_to_map_applied_before_projection():
    selected,_ = select_camera_candidate([('target',Mock(getPosition=lambda:[1,2,3]))],
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
    message.header.frame_id='Astra_rgb'
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
    assert node.create_subscription.call_args.args[1]==CAMERA_INFO_TOPIC
    listener.unregister.assert_called_once()
    node.destroy_node.assert_called_once()
