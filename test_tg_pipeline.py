import sys
import json

# 1. Test JSON data
print("=== 1. JSON Data Test ===")
with open('/home/byounggun/B4DL/b4dl_dataset/stage2_combined_meta_v2.json') as f:
    data = json.load(f)

tg_items = [d for d in data if d.get('is_time_grounding', False)]
print(f"Total items: {len(data)}")
print(f"TG items: {len(tg_items)}")

for i in range(min(3, len(tg_items))):
    item = tg_items[i]
    print(f"\nTG Sample {i}:")
    print(f"  scene_id: {item['scene_id']}")
    print(f"  start_frame: {item['start_frame']}, end_frame: {item['end_frame']}")
    print(f"  answer: {item['conversations'][1]['value']}")

# 2. Test tokenizer (without importing model)
print("\n=== 2. Tokenizer Test ===")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    '/home/byounggun/B4DL/base_model/vicuna-v1-5-7b',
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token

num_new = tokenizer.add_special_tokens({"additional_special_tokens": ["<meta>", "<4DLiDAR>", "<tg>"]})
print(f"Added {num_new} special tokens")
tg_id = tokenizer.convert_tokens_to_ids('<tg>')
print(f"<tg> token id: {tg_id}")

# Test tokenizing a TG answer
sample_ans = tg_items[0]['conversations'][1]['value']
print(f"\nSample answer: {sample_ans}")
tokens = tokenizer(sample_ans, return_tensors='pt')
print(f"Tokenized length: {tokens['input_ids'].shape[1]}")
tg_positions = (tokens['input_ids'] == tg_id).nonzero(as_tuple=True)[1]
print(f"<tg> token positions: {tg_positions.tolist()}")

# 3. Test dataset preprocessing (without loading features)
print("\n=== 3. Dataset Preprocessing Test ===")
import copy
from vtimellm import conversation as conversation_lib
from vtimellm.mm_utils import tokenizer_image_token
from vtimellm.constants import IGNORE_INDEX

# Setup conversation
conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]

# Manually preprocess a TG sample
source = copy.deepcopy(tg_items[0]['conversations'])
print(f"Original conversations: {source}")

conv = conversation_lib.default_conversation.copy()
roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
conv.messages = []
for j, sentence in enumerate(source):
    role = roles[sentence["from"]]
    conv.append_message(role, sentence["value"])
prompt = conv.get_prompt()
print(f"\nPrompt: {prompt[:200]}...")

input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
print(f"input_ids shape: {input_ids.shape}")
tg_positions = (input_ids == tg_id).nonzero(as_tuple=True)[1]
print(f"<tg> token positions in prompt: {tg_positions.tolist()}")

print("\n=== All basic tests passed! ===")
