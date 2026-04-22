import json
import re
import copy

def create_stage2_v2(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Time Grounding interval pattern: 'from frame XXX to frame YYY'
    tg_pattern = re.compile(r'from frame (\d+) to frame (\d+)', re.IGNORECASE)
    
    new_data = []
    tg_count = 0
    
    for item in data:
        new_item = copy.deepcopy(item)
        ans = new_item['conversations'][1]['value']
        
        match = tg_pattern.search(ans)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            
            # Add <tg> token at the beginning of the answer
            # Original answer: "from frame 006 to frame 014."
            # New answer: "<tg>\nfrom frame 006 to frame 014."
            new_item['conversations'][1]['value'] = f"<tg>\n{ans}"
            
            # Add temporal grounding metadata
            new_item['start_frame'] = start_frame
            new_item['end_frame'] = end_frame
            new_item['is_time_grounding'] = True
            tg_count += 1
        else:
            new_item['is_time_grounding'] = False
        
        new_data.append(new_item)
    
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Created {output_path}")
    print(f"Total items: {len(new_data)}")
    print(f"Time Grounding items: {tg_count}")

if __name__ == '__main__':
    create_stage2_v2(
        '/home/byounggun/B4DL/b4dl_dataset/stage2_combined_meta.json',
        '/home/byounggun/B4DL/b4dl_dataset/stage2_combined_meta_v2.json'
    )
