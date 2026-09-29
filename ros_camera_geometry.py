"""Execution-only RGB geometry from ROS TF and the native CameraInfo stream."""
from math import atan, hypot, isfinite, sqrt
import time

CAMERA_INFO_TOPIC = '/tiago/camera/color/camera_info'
TF_TIMEOUT_SEC = 5.0
TF_SKEW_TOLERANCE_NS = 100_000_000  # At most 0.1 s of simulator publication skew.
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
    camera_frame = info.header.frame_id
    if not camera_frame:
        raise ValueError('CameraInfo received but header.frame_id is empty')
    fx, fy = info.k[0], info.k[4]
    if not all(isfinite(v) and v > 0 for v in (fx,fy,info.width,info.height)):
        raise ValueError('Invalid RGB CameraInfo intrinsics')
    log(f'[GROUNDING] CameraInfo topic: {CAMERA_INFO_TOPIC}')
    log(f'[GROUNDING] CameraInfo frame_id: {camera_frame}')
    log(f'[GROUNDING] TF lookup: map -> {camera_frame}')
    requested_ns = info.header.stamp.sec * 1_000_000_000 + info.header.stamp.nanosec
    log(f'[GROUNDING] Requested camera time: {requested_ns / 1e9:.3f}')
    from rclpy.time import Time
    from tf2_ros import ExtrapolationException, TransformException
    try:
        camera_tf = buffer.lookup_transform('map', camera_frame, stamp)
    except ExtrapolationException as exact_error:
        # Latest is only a candidate: validate its actual timestamp before use.
        # Never turn missing frames/connectivity errors into a temporal fallback.
        try:
            camera_tf = buffer.lookup_transform('map', camera_frame, Time(clock_type=stamp.clock_type))
        except TransformException:
            raise exact_error
        available = Time.from_msg(camera_tf.header.stamp, clock_type=stamp.clock_type)
        skew_ns = abs(available.nanoseconds - requested_ns)
        log(f'[GROUNDING] Exact TF unavailable; skew={skew_ns / 1e9:.3f}s')
        if available.nanoseconds == 0 or skew_ns > TF_SKEW_TOLERANCE_NS:
            raise exact_error
        log(f'[GROUNDING] Using nearest available TF at {available.nanoseconds / 1e9:.3f}')
        # Keep camera, world alignment, and robot checks at one common time.
        stamp = available
    camera = pose(camera_tf)
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
    received = []
    try:
        buffer = Buffer()
        listener = TransformListener(buffer,node)
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
        log(f'Camera geometry: reference frame=map; optical frame={received[0].header.frame_id if received else None}; TF acquired=no; reason={error}')
        raise
    finally:
        if listener is not None:
            listener.unregister()
        node.destroy_node()
