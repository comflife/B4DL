import json
import os
import re
import math
from bisect import bisect_left


def extract_timestamp_from_lidar_path(path: str) -> int:
    """Extract timestamp from nuScenes LiDAR filename.
    e.g. ...__LIDAR_TOP__1531883530449377.pcd.bin -> 1531883530449377
    """
    match = re.search(r'__LIDAR_TOP__(\d+)\.pcd\.bin', path)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract timestamp from {path}")


def load_can_bus_poses(can_bus_dir: str, scene_name: str):
    """Load pose.json for a given scene from can_bus."""
    pose_path = os.path.join(can_bus_dir, f"{scene_name}_pose.json")
    with open(pose_path, 'r') as f:
        return json.load(f)


def find_nearest_pose(pose_list, timestamp: int):
    """Find pose with utime closest to given timestamp."""
    utimes = [p['utime'] for p in pose_list]
    idx = bisect_left(utimes, timestamp)
    if idx == 0:
        return pose_list[0]
    if idx == len(utimes):
        return pose_list[-1]
    if abs(utimes[idx] - timestamp) < abs(utimes[idx - 1] - timestamp):
        return pose_list[idx]
    return pose_list[idx - 1]


def quaternion_to_yaw(qw, qx, qy, qz):
    """Convert quaternion to yaw angle (rotation around Z-axis) in degrees."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw)


def normalize_angle_diff(angle_diff):
    """Normalize angle difference to [-180, 180]."""
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
    return angle_diff


def describe_speed_change(v_start, v_end):
    """Describe speed change in natural language."""
    delta_v = v_end - v_start
    if v_end < 0.5:
        return "came to a stop"
    if abs(delta_v) < 0.5:
        return f"maintained a nearly constant speed of approximately {v_start:.1f} m/s"
    elif delta_v > 0:
        return f"accelerated from approximately {v_start:.1f} m/s to {v_end:.1f} m/s"
    else:
        return f"decelerated from approximately {v_start:.1f} m/s to {v_end:.1f} m/s"


def describe_heading_change(yaw_start, yaw_end):
    """Describe heading change in natural language."""
    yaw_diff = normalize_angle_diff(yaw_end - yaw_start)
    if abs(yaw_diff) < 5:
        return "maintained a relatively straight heading"
    elif yaw_diff > 0:
        return f"turned left by approximately {abs(yaw_diff):.1f} degrees"
    else:
        return f"turned right by approximately {abs(yaw_diff):.1f} degrees"


def describe_position_change(pos_start, pos_end, yaw_start):
    """Describe relative position change in natural language."""
    dx = pos_end[0] - pos_start[0]
    dy = pos_end[1] - pos_start[1]
    
    # Forward direction vector from yaw
    forward_x = math.cos(math.radians(yaw_start))
    forward_y = math.sin(math.radians(yaw_start))
    
    # Lateral (left) direction vector
    left_x = -forward_y
    left_y = forward_x
    
    forward_dist = dx * forward_x + dy * forward_y
    lateral_dist = dx * left_x + dy * left_y
    
    parts = []
    
    if abs(forward_dist) > 0.5:
        if forward_dist > 0:
            parts.append(f"moved forward by approximately {abs(forward_dist):.1f} meters")
        else:
            parts.append(f"moved backward by approximately {abs(forward_dist):.1f} meters")
    
    if abs(lateral_dist) > 0.5:
        if lateral_dist > 0:
            parts.append(f"shifted to the left by approximately {abs(lateral_dist):.1f} meters")
        else:
            parts.append(f"shifted to the right by approximately {abs(lateral_dist):.1f} meters")
    
    if not parts:
        return "remained approximately in the same position"
    
    return " and ".join(parts)


def describe_single_pose(pose):
    """Describe a single pose in absolute natural language."""
    # Speed
    v = math.sqrt(sum(v_i**2 for v_i in pose['vel']))
    if v < 0.5:
        speed_desc = "was stationary"
    else:
        speed_desc = f"was traveling at approximately {v:.1f} m/s"
    
    # Heading
    q = pose['orientation']
    yaw = quaternion_to_yaw(q[0], q[1], q[2], q[3])
    heading_desc = f"facing approximately {abs(yaw):.1f} degrees"
    
    # Position
    pos = pose['pos']
    pos_desc = f"positioned at coordinates ({pos[0]:.1f}, {pos[1]:.1f})"
    
    # Acceleration
    a = math.sqrt(sum(a_i**2 for a_i in pose['accel']))
    if a > 10.5:
        accel_desc = "was experiencing significant acceleration"
    elif a > 9.9:
        accel_desc = "was experiencing gentle acceleration"
    else:
        accel_desc = "maintained smooth motion"
    
    return f"The ego vehicle {speed_desc}, {pos_desc}, {heading_desc}, and {accel_desc}."


def compute_meta_description(start_pose, end_pose):
    """Convert start/end poses to natural language meta description."""
    first_desc = describe_single_pose(start_pose)
    last_desc = describe_single_pose(end_pose)
    return f"The metadata of the first frame is '{first_desc}' and the metadata of the last frame is '{last_desc}'"


def build_scene_token_to_can_bus_mapping(can_bus_dir: str, scene_metadata_path: str, cache_path: str = None):
    """Build mapping from scene_token to can_bus scene name by matching first LiDAR timestamp."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    with open(scene_metadata_path, 'r') as f:
        scene_metadata = json.load(f)
    
    # Load all can_bus pose files and their first utime
    can_bus_utimes = {}
    for fname in sorted(os.listdir(can_bus_dir)):
        if fname.endswith('_pose.json'):
            scene_name = fname.replace('_pose.json', '')
            pose_path = os.path.join(can_bus_dir, fname)
            with open(pose_path, 'r') as f:
                poses = json.load(f)
            if poses:
                can_bus_utimes[scene_name] = poses[0]['utime']
    
    mapping = {}
    unmatched = []
    for scene in scene_metadata:
        scene_token = scene['scene_token']
        lidar_paths = scene.get('paths', {}).get('PATH_LIDAR_TOP', {})
        if not lidar_paths:
            unmatched.append(scene_token)
            continue
        first_path = lidar_paths.get('PATH_000', '')
        if not first_path:
            unmatched.append(scene_token)
            continue
        try:
            lidar_ts = extract_timestamp_from_lidar_path(first_path)
        except ValueError:
            unmatched.append(scene_token)
            continue
        
        # Find closest can_bus scene
        best_scene_name = None
        best_diff = float('inf')
        for cb_name, cb_utime in can_bus_utimes.items():
            diff = abs(cb_utime - lidar_ts)
            if diff < best_diff:
                best_diff = diff
                best_scene_name = cb_name
        
        if best_scene_name and best_diff < 1_000_000:  # 1 second threshold
            mapping[scene_token] = best_scene_name
            # Remove matched scene to prevent duplicate assignment
            del can_bus_utimes[best_scene_name]
        else:
            unmatched.append(scene_token)
    
    print(f"Mapped {len(mapping)} scenes. Unmatched: {len(unmatched)}")
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(mapping, f, indent=2)
    
    return mapping
