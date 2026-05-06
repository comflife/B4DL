"""End-to-end smoke test for the VoxelNeXt -> MMQwen pipeline.

Builds a synthetic VoxelNeXt-style feature blob (no real LiDAR or
detector required), saves it, and runs `MMQwen` forward + generate to
make sure:

  * `qwen_mm.data._load_lidar_feat` correctly returns a dict.
  * `MMQwen._project_images` routes to the dict path and produces
    K_real tokens after applying the mask.
  * The feat + xyz projectors both have grads.
  * `model.generate(..., images=[blob])` produces text without
    crashing on the new K=256 splice.

Run from the qwen_mm conda env (NOT the voxelnext extraction env).
This file does NOT touch /data1/...; everything is on /tmp.
"""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoTokenizer

NEW_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW_CODE_DIR))

from qwen_mm import MMQwen, IMAGE_PLACEHOLDER, BOX_START, BOX_END  # noqa: E402
from qwen_mm.data import _load_lidar_feat, SMAPDataset, Collator  # noqa: E402
from train_qwen_sft import add_special_tokens_and_resize  # noqa: E402


K = 256
FEAT_DIM = 128


def make_voxelnext_blob(scene_id: str, k_real: int = 200, save_dir: Path = None):
    """Make a synthetic (feat, xyz, mask) blob and save under save_dir."""
    feat = torch.randn(K, FEAT_DIM, dtype=torch.float16)
    # Cluster xyz into a strip 5–30 m in front of the ego, like a typical scene
    xyz = torch.zeros(K, 3, dtype=torch.float16)
    xyz[:, 0] = torch.empty(K).uniform_(-25.0, 25.0).half()
    xyz[:, 1] = torch.empty(K).uniform_( 5.0, 50.0).half()
    xyz[:, 2] = torch.empty(K).uniform_(-1.5, 0.5).half()
    score = torch.full((K,), 0.0, dtype=torch.float16)
    score[:k_real] = torch.linspace(0.9, 0.05, k_real).half()
    cls = torch.zeros(K, dtype=torch.int8)
    mask = torch.zeros(K, dtype=torch.bool)
    mask[:k_real] = True
    blob = {"feat": feat, "xyz": xyz, "score": score, "cls": cls, "mask": mask}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(blob, save_dir / f"{scene_id}.pt")
    return blob


def main():
    print("[smoke] building synthetic VoxelNeXt blob")
    feat_dir = Path(tempfile.mkdtemp(prefix="vxnxt_feat_"))
    blob = make_voxelnext_blob("dummy_scene", k_real=200, save_dir=feat_dir)
    print(f"[smoke] saved to {feat_dir}/dummy_scene.pt — feat={tuple(blob['feat'].shape)} "
          f"xyz={tuple(blob['xyz'].shape)} mask_real={int(blob['mask'].sum())}")

    print("[smoke] _load_lidar_feat -> dict path")
    loaded = _load_lidar_feat(feat_dir, "dummy_scene")
    assert isinstance(loaded, dict), "expected dict for VoxelNeXt blob"
    for k in ("feat", "xyz", "mask"):
        assert k in loaded, k
    print(f"[smoke]   loaded keys = {sorted(loaded.keys())}")

    print("[smoke] building MMQwen with mm_input_dim=128, mm_pos_dim=3")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = MMQwen.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
    # Rebuild projectors for new dims (this is exactly what train_qwen_sft.py does).
    model.mm_input_dim = 128
    model.config.mm_input_dim = 128
    model.mm_projector = torch.nn.Linear(128, model.config.hidden_size)
    model.mm_pos_dim = 3
    model.config.mm_pos_dim = 3
    model.pos_projector = torch.nn.Linear(3, model.config.hidden_size)

    add_special_tokens_and_resize(tok, model)
    model.coord_aux_weight = 0.5
    print(f"[smoke] image_id={model.image_token_id} coord_range=[{model.coord_token_min},{model.coord_token_max}]")

    # Build a tiny dataset entry.
    sample = {
        "scene_id": "dummy_scene",
        "task": "nugrounding",
        "template_type": "det_area",
        "conversations": [
            {"from": "human", "value": "<image>\n[front view] Where is the car?"},
            {"from": "gpt", "value": "The car is at <|box_start|><coord_400><coord_500><coord_500><coord_500><coord_500><coord_500><coord_500><coord_500><|box_end|>."},
        ],
    }
    data_file = feat_dir / "data.json"
    data_file.write_text(json.dumps([sample]))

    ds = SMAPDataset(str(data_file), tokenizer=tok, feat_folder=str(feat_dir), max_length=512)
    item = ds[0]
    print(f"[smoke] sample input_ids={tuple(item['input_ids'].shape)} "
          f"image is dict? {isinstance(item['image'], dict)} "
          f"feat={tuple(item['image']['feat'].shape)}")

    coll = Collator(pad_token_id=tok.pad_token_id)
    batch = coll([item])
    print(f"[smoke] batch ids={tuple(batch['input_ids'].shape)} "
          f"images[0] type={type(batch['images'][0]).__name__}")

    # Forward + check loss + grad on projectors.
    model.train()
    out = model(
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        attention_mask=batch["attention_mask"],
        images=batch["images"],
    )
    loss = out.loss
    loss.backward()
    g_feat = model.mm_projector.weight.grad
    g_pos = model.pos_projector.weight.grad
    print(f"[smoke] forward+backward ok | loss={loss.item():.4f} "
          f"feat_grad_norm={g_feat.norm().item():.4f} "
          f"pos_grad_norm={g_pos.norm().item():.4f}")
    assert g_feat is not None and g_feat.norm().item() > 0
    assert g_pos is not None and g_pos.norm().item() > 0

    # Generate (no labels; should not crash).
    model.eval()
    msgs = [{"role": "user", "content": sample["conversations"][0]["value"]}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        gen = model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            images=batch["images"],
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    print(f"[smoke] generate ok | n_new={gen.shape[1]}")
    print(f"[smoke] decoded: {tok.decode(gen[0], skip_special_tokens=False)[:120]}")

    print("[smoke] ALL OK")


if __name__ == "__main__":
    main()
