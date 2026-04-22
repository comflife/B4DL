import argparse
import json
import os

from meta_token import (
    build_scene_token_to_can_bus_mapping,
    load_can_bus_poses,
    extract_timestamp_from_lidar_path,
    find_nearest_pose,
    compute_meta_description,
)


def add_meta_to_dataset(
    input_json_path: str,
    output_json_path: str,
    scene_metadata_path: str,
    can_bus_dir: str,
    mapping_cache_path: str = None,
):
    with open(input_json_path, 'r') as f:
        dataset = json.load(f)
    
    with open(scene_metadata_path, 'r') as f:
        scene_metadata = json.load(f)
    
    # Build scene_token -> scene metadata lookup
    scene_lookup = {s['scene_token']: s for s in scene_metadata}
    
    # Build scene_token -> can_bus scene name mapping
    mapping = build_scene_token_to_can_bus_mapping(
        can_bus_dir, scene_metadata_path, mapping_cache_path
    )
    
    # Pre-compute meta descriptions per scene_token
    meta_cache = {}
    unmatched_scenes = set()
    
    for item in dataset:
        scene_token = item.get('scene_token')
        if not scene_token:
            continue
        
        if scene_token in meta_cache:
            continue
        
        scene_name = mapping.get(scene_token)
        if not scene_name:
            unmatched_scenes.add(scene_token)
            meta_cache[scene_token] = ""
            continue
        
        scene = scene_lookup.get(scene_token)
        if not scene:
            unmatched_scenes.add(scene_token)
            meta_cache[scene_token] = ""
            continue
        
        # Get first and last LiDAR paths
        lidar_paths = scene.get('paths', {}).get('PATH_LIDAR_TOP', {})
        num_frames = scene.get('num_frames', 0)
        if num_frames == 0:
            meta_cache[scene_token] = ""
            continue
        
        first_path = lidar_paths.get(f'PATH_{0:03d}', '')
        last_path = lidar_paths.get(f'PATH_{num_frames-1:03d}', '')
        
        if not first_path or not last_path:
            meta_cache[scene_token] = ""
            continue
        
        try:
            start_ts = extract_timestamp_from_lidar_path(first_path)
            end_ts = extract_timestamp_from_lidar_path(last_path)
        except ValueError as e:
            print(f"Warning: {e}")
            meta_cache[scene_token] = ""
            continue
        
        # Load poses
        try:
            poses = load_can_bus_poses(can_bus_dir, scene_name)
        except Exception as e:
            print(f"Warning: failed to load poses for {scene_name}: {e}")
            meta_cache[scene_token] = ""
            continue
        
        start_pose = find_nearest_pose(poses, start_ts)
        end_pose = find_nearest_pose(poses, end_ts)
        
        meta_desc = compute_meta_description(start_pose, end_pose)
        meta_cache[scene_token] = meta_desc
    
    if unmatched_scenes:
        print(f"Warning: {len(unmatched_scenes)} scenes without can_bus mapping:")
        for st in sorted(unmatched_scenes)[:5]:
            print(f"  - {st}")
    
    # Insert <meta> token into conversations
    modified_count = 0
    for item in dataset:
        scene_token = item.get('scene_token')
        meta_desc = meta_cache.get(scene_token, '')
        if not meta_desc:
            continue
        
        conversations = item.get('conversations', [])
        if not conversations:
            continue
        
        # Insert <meta> at the beginning of the first human message
        first_conv = conversations[0]
        if first_conv.get('from') == 'human':
            original_value = first_conv['value']
            new_value = f"<meta>\n{meta_desc}\n\n{original_value}"
            first_conv['value'] = new_value
            modified_count += 1
    
    with open(output_json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nSaved to {output_json_path}")
    print(f"Total items: {len(dataset)}")
    print(f"Items with <meta>: {modified_count}")
    print(f"Unique scenes with meta: {sum(1 for v in meta_cache.values() if v)}")


def main():
    parser = argparse.ArgumentParser(description="Add <meta> token to B4DL dataset")
    parser.add_argument('--input', type=str, default='./b4dl_dataset/stage2_combined.json',
                        help='Input JSON path')
    parser.add_argument('--output', type=str, default='./b4dl_dataset/stage2_combined_meta.json',
                        help='Output JSON path')
    parser.add_argument('--scene-metadata', type=str, default='./b4dl_dataset/metadata/scene_metadata.json',
                        help='Scene metadata JSON path')
    parser.add_argument('--can-bus-dir', type=str, default='./nuscenes/can_bus',
                        help='nuScenes can_bus directory')
    parser.add_argument('--mapping-cache', type=str, default='./b4dl_dataset/metadata/scene_token_to_can_bus.json',
                        help='Cache path for scene_token to can_bus mapping')
    
    args = parser.parse_args()
    add_meta_to_dataset(
        args.input, args.output, args.scene_metadata,
        args.can_bus_dir, args.mapping_cache
    )


if __name__ == '__main__':
    main()
