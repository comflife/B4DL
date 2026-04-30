#!/bin/bash

python3 generate_description.py \
    --start_index 10 \
    --end_index 20 \
    --api_key {your openai api key} \
    --nuscenes_root /home/byounggun/B4DL/nuscenes \
    --dataroot ./data