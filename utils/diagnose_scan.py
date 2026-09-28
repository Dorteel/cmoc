#!/usr/bin/env python3
"""Read-only scan/TF/footprint diagnosis. Run with the robot stationary in free space."""
import argparse
import json
import math
from pathlib import Path
import time


def inside(point, polygon):
    x, y = point
    result = False
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        if (ay > y) != (by > y) and x < (bx-ax) * (y-ay) / (by-ay) + ax:
            result = not result
    return result


def boundary_distance(point, polygon):
    x, y = point
    distances = []
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = bx-ax, by-ay
        t = max(0, min(1, ((x-ax)*dx + (y-ay)*dy) / (dx*dx+dy*dy))) if dx or dy else 0
        distances.append(math.hypot(x-ax-t*dx, y-ay-t*dy))
    return min(distances)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--topic', default='/scan', help='LaserScan topic (default: /scan)')
    args = parser.parse_args()
    import yaml
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import PolygonStamped, PointStamped
    from tf2_ros import Buffer, TransformListener, TransformException
    from tf2_geometry_msgs import do_transform_point

    params = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'external/webots_ros2_simulation/nav2_params_jazzy.yaml').read_text())
    radius = params['local_costmap']['local_costmap']['ros__parameters']['robot_radius']
    slow = json.loads(params['collision_monitor']['ros__parameters']['PolygonSlow']['points'])
    rclpy.init()
    node = rclpy.create_node('cmoc_scan_diagnostic')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    latest = {}
    node.create_subscription(LaserScan, args.topic, lambda msg: latest.update(scan=msg), qos_profile_sensor_data)
    node.create_subscription(PolygonStamped, '/local_costmap/published_footprint',
                             lambda msg: latest.update(footprint=msg), qos_profile_sensor_data)
    count = 0
    seen = None
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            scan = latest.get('scan')
            if scan is None or scan is seen:
                continue
            stamp = Time.from_msg(scan.header.stamp)
            try:
                transforms = {frame: buffer.lookup_transform(frame, scan.header.frame_id, stamp)
                              for frame in ('base_link', 'base_footprint')}
                def transform(x, y, z, tf):
                    point = PointStamped()
                    point.point.x, point.point.y, point.point.z = float(x), float(y), float(z)
                    return do_transform_point(point, tf).point
                polygon = None
                footprint = latest.get('footprint')
                if footprint:
                    tf = buffer.lookup_transform('base_footprint', footprint.header.frame_id,
                                                 Time.from_msg(footprint.header.stamp))
                    polygon = [(p.x, p.y) for p in
                        (transform(p.x, p.y, p.z, tf) for p in footprint.polygon.points)]
                    if len(polygon) < 3:
                        polygon = None
            except TransformException as error:
                latest['tf_error'] = str(error)
                continue
            seen = scan
            finite = [r for r in scan.ranges if math.isfinite(r)]
            angles, in_body, near_boundary, in_slow = [], 0, 0, 0
            for i, r in enumerate(scan.ranges):
                if not math.isfinite(r) or not scan.range_min <= r <= scan.range_max:
                    continue
                angle = scan.angle_min + i * scan.angle_increment
                p = transform(r*math.cos(angle), r*math.sin(angle), 0, transforms['base_footprint'])
                xy = (p.x, p.y)
                hit = inside(xy, polygon) if polygon else math.hypot(*xy) < radius
                edge = boundary_distance(xy, polygon) if polygon else abs(math.hypot(*xy)-radius)
                in_body += hit
                near_boundary += edge <= .03
                in_slow += inside(xy, slow)
                if r < .4:
                    angles.append(round(math.degrees(angle), 1))
            if count == 0:
                print(json.dumps({'frame_id': scan.header.frame_id, 'angle_min': scan.angle_min,
                    'angle_max': scan.angle_max, 'range_min': scan.range_min, 'range_max': scan.range_max,
                    'TF_scan_to_base': {name: {'xyz': [tf.transform.translation.x, tf.transform.translation.y,
                        tf.transform.translation.z], 'quaternion_xyzw': [tf.transform.rotation.x,
                        tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]}
                        for name, tf in transforms.items()}}))
            count += 1
            print(json.dumps({'scan': count, 'minimum_finite': min(finite) if finite else None,
                'counts_under_m': {str(t): sum(r < t for r in finite) for t in (.15, .2, .3, .4)},
                'footprint': 'published' if polygon else f'configured radius {radius} m',
                'valid_inside_footprint': in_body, 'valid_within_3cm_of_boundary': near_boundary,
                'valid_inside_PolygonSlow': in_slow,
                'close_angles_deg_first_20': angles[:20],
                'close_angle_span_deg': [min(angles), max(angles)] if angles else None}), flush=True)
        if not count:
            raise SystemExit('No scan with usable TF received. ' + latest.get('tf_error',
                'Start the simulator; leave the robot stationary in open space.'))
    finally:
        listener.unregister()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
