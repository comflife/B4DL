"""Save the current mm_projector from a partially-trained stage1a run."""
import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

NEW_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW_CODE_DIR))

from qwen_mm import MMQwen, BOX_END, BOX_START, IMAGE_PLACEHOLDER, quantizer as q3d


def main():
    ap = argparse.ArgumentParser()
    _data_root = os.environ.get("DATA_ROOT", "/data1/byounggun/3davs_b4dl")
    ap.add_argument("--output_dir", default=f"{_data_root}/checkpoints/qwen_stage1a_vxnxt")
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen3.5-9B")
    args = ap.parse_args()

    print("[save_proj] loading tokenizer + model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = MMQwen.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Add special tokens exactly like train_qwen_sft.py does
    coord_names = q3d.coord_token_names()
    extras = [IMAGE_PLACEHOLDER, BOX_START, BOX_END] + coord_names
    needed = [t for t in extras if t not in tokenizer.get_vocab()]
    if needed:
        added = tokenizer.add_special_tokens({"additional_special_tokens": needed})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        with torch.no_grad():
            in_emb = model.get_input_embeddings().weight
            out_emb = model.get_output_embeddings().weight
            mean_in = in_emb[:-added].mean(dim=0, keepdim=True)
            mean_out = out_emb[:-added].mean(dim=0, keepdim=True)
            in_emb[-added:] = mean_in
            out_emb[-added:] = mean_out

    model.set_image_token_id(tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER))

    # Re-init projector clean (same fix as train_qwen_sft.py)
    model.mm_projector = torch.nn.Linear(128, model.config.hidden_size).to(torch.float32)

    # Load any existing projector checkpoint if present
    proj_path = Path(args.output_dir) / "mm_projector.bin"
    if proj_path.exists():
        sd = torch.load(proj_path, map_location="cpu")
        feat_sd = {}
        for k, v in sd.items():
            if k.startswith("mm_projector."):
                feat_sd[k.split("mm_projector.", 1)[1]] = v
        if feat_sd:
            model.mm_projector.load_state_dict(feat_sd, strict=True)
            print(f"[save_proj] loaded existing projector from {proj_path}")
    else:
        print(f"[save_proj] no existing projector found — saving fresh init to {proj_path}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = {}
    for k, v in model.mm_projector.state_dict().items():
        out[f"mm_projector.{k}"] = v
    torch.save(out, proj_path)
    print(f"[save_proj] saved -> {proj_path}")


if __name__ == "__main__":
    main()
