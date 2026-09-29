"""Execution-only RGB geometry from ROS TF and the native CameraInfo stream."""
from math import atan, hypot, isfinite, sqrt
import time

CAMERA_INFO_TOPIC = '/tiago/camera/color/camera_info'
# tiago_webots_wheels.urdf frameName; Webots WbCamera::urdfRotation exports RDF.
RGB_OPTICAL_FRAME = 'Astra_rgb'
TF_TIMEOUT_SEC = 5.0
ROBOT_XY_TOLERANCE = 0.15


def pose(transform):
    t, q = transform.transform.translation, transform.transform.rotation
    if not all(isfinite(v) for v in (t.x,t.y,t.z)):
        raise ValueError('Invalid TF translation')
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = sqrt(x*x+y*y+z*z+w*w)
    if not isfinite(norm) or norm < 1e-9:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = (v/norm for v in (x,y,z,w))
    rotation = [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    return [[t.x,t.y,t.z], rotation]


def transform_point(point, transform):
    translation, rotation = transform
    return [translation[r]+sum(rotation[r*3+c]*point[c] for c in range(3)) for r in range(3)]


def geometry_from_tf(buffer, info, stamp, simulator_robot, log):
    if info.header.frame_id != RGB_OPTICAL_FRAME:
        raise ValueError(f'Unexpected RGB CameraInfo frame: {info.header.frame_id!r}; expected {RGB_OPTICAL_FRAME}')
    fx, fy = info.k[0], info.k[4]
    if not all(isfinite(v) and v > 0 for v in (fx,fy,info.width,info.height)):
        raise ValueError('Invalid RGB CameraInfo intrinsics')
    camera = pose(buffer.lookup_transform('map', info.header.frame_id, stamp))
    # GroundTruthOdom publishes raw ENU simulator pose in odom; navigation's
    # existing map -> odom transform supplies the required world alignment.
    alignment = pose(buffer.lookup_transform('map', 'odom', stamp))
    robot = pose(buffer.lookup_transform('map', 'base_link', stamp))[0]
    aligned = transform_point(simulator_robot, alignment)
    difference = hypot(aligned[0]-robot[0],aligned[1]-robot[1])
    log(f'Coordinate-frame check: TF robot={robot[:2]}, simulator robot={simulator_robot[:2]}, '
        f'aligned robot={aligned[:2]}, difference={difference:.3f}m')
    if not isfinite(difference) or difference > ROBOT_XY_TOLERANCE:
        raise ValueError(f'Simulator/map coordinate mismatch: {difference:.3f}m > {ROBOT_XY_TOLERANCE}m')
    hfov, vfov = 2*atan(info.width/(2*fx)), 2*atan(info.height/(2*fy))
    log(f'Camera geometry: reference frame=map; optical frame={info.header.frame_id}; '
        f'TF acquired=yes; horizontal_fov={hfov:.4f}; vertical_fov={vfov:.4f}')
    return {'camera': [*camera,hfov,vfov], 'scene_to_map': alignment}


def acquire_camera_geometry(simulator_robot, log=lambda message: None):
    import rclpy
    from rclpy.time import Time
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo
    from tf2_ros import Buffer, TransformListener, TransformException
    if not rclpy.ok():
        raise RuntimeError('Camera TF unavailable: ROS context is not active')
    node = rclpy.create_node('cmoc_execution_camera', parameter_overrides=[Parameter('use_sim_time', value=True)])
    listener = None
    try:
        buffer = Buffer()
        listener = TransformListener(buffer,node)
        received = []
        # Hold the first new sample while its timestamp catches up in TF.
        node.create_subscription(CameraInfo,CAMERA_INFO_TOPIC,
                                 lambda msg: received.append(msg) if not received else None,
                                 qos_profile_sensor_data)
        deadline = time.monotonic()+TF_TIMEOUT_SEC
        reason = f'No CameraInfo on {CAMERA_INFO_TOPIC}'
        while time.monotonic() < deadline:
            rclpy.spin_once(node,timeout_sec=min(0.05,max(0,deadline-time.monotonic())))
            if not received:
                continue
            info = received[0]
            try:
                return geometry_from_tf(buffer,info,Time.from_msg(info.header.stamp),simulator_robot,log)
            except TransformException as error:
                reason = str(error)
        raise RuntimeError(f'Camera TF/CameraInfo unavailable after {TF_TIMEOUT_SEC:.1f}s: {reason}')
    except Exception as error:
        log(f'Camera geometry: reference frame=map; optical frame={RGB_OPTICAL_FRAME}; TF acquired=no; reason={error}')
        raise
    finally:
        if listener is not None:
            listener.unregister()
        node.destroy_node()
