"""Custom GRPO fine-tuner for the Qwen3-0.6B + SMAP MM model on nuGrounding.

Design (matches the earlier Llama version, ported to MMQwen):

  * Policy = stage-1b SFT model + a fresh LoRA adapter (trainable).
  * Reference = the same backbone with `disable_adapter()` (no second copy).
  * For each prompt: K rollouts via model.generate(images=...).
  * Group-normalised advantages (GRPO baseline) + KL to reference.
  * Per-token NLL is computed by re-running model(forward) over
    [prompt | rollout] tokens; gradients flow only through the rollout span.

Reward is the LiDAR-aware composite from rewards_lidar.compute_reward, with
`template_type` propagated from each sample so det_object falls back to
class-name matching while det_area focuses on bbox geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

NEW_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW_CODE_DIR))

from qwen_mm import (  # noqa: E402
    BOX_END,
    BOX_START,
    IMAGE_PLACEHOLDER,
    MMQwen,
)
from qwen_mm.data import _load_smap_feat  # noqa: E402
from rewards_lidar import compute_reward  # noqa: E402


# ---------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    output_dir: str
    base_model: str  # original Qwen3 path, used only for tokenizer if no SFT
    sft_dir: str  # Stage-1b SFT output (model + tokenizer + projector)
    feat_folder: str
    data_path: str
    sample_ratio: float = 0.25
    num_rollouts: int = 4
    gradient_accumulation_steps: int = 4
    max_prompt_len: int = 512
    max_new_tokens: int = 96
    temperature: float = 1.0
    top_p: float = 0.9
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    epochs: int = 1
    seed: int = 0
    lora_r: int = 32
    lora_alpha: int = 64
    log_every: int = 10
    save_every: int = 1000


# ---------------------------------------------------------------------------
def build_prompt_ids(tokenizer, question_text: str, max_len: int) -> torch.Tensor:
    """Apply Qwen ChatML template to a single user turn and return input_ids."""
    # Make sure <image> is present so the splicer has a target.
    if IMAGE_PLACEHOLDER not in question_text:
        question_text = IMAGE_PLACEHOLDER + "\n" + question_text
    messages = [{"role": "user", "content": question_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).input_ids
    return ids  # (1, T)


def compute_response_logp(
    model: MMQwen,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
    image_feats: torch.Tensor,
) -> torch.Tensor:
    """sum_{t in response} log pi(t | prefix). Returns shape (1,)."""
    full_ids = torch.cat([input_ids, response_ids], dim=1)
    out = model(
        input_ids=full_ids,
        images=[image_feats.to(full_ids.device)],
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits  # (1, T_eff, V)
    R = response_ids.shape[1]
    response_logits = logits[:, -R - 1 : -1, :]
    logp = F.log_softmax(response_logits.float(), dim=-1)
    token_logp = logp.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return token_logp.sum(dim=1)  # (1,)


# ---------------------------------------------------------------------------
class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        # Load tokenizer + model from the stage-1b SFT directory if present;
        # otherwise fall back to the base + the projector path (mostly for
        # smoke testing).
        load_path = cfg.sft_dir if os.path.isdir(cfg.sft_dir) else cfg.base_model
        print(f"[grpo] loading tokenizer + model from {load_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = MMQwen.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        # Make sure tokenizer/model agree on the image token id.
        if IMAGE_PLACEHOLDER not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [IMAGE_PLACEHOLDER, BOX_START, BOX_END]}
            )
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)
        self.model.set_image_token_id(
            self.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        )

        # If sft_dir didn't already include a saved mm_projector, try the
        # standalone path inside sft_dir.
        proj_path = Path(cfg.sft_dir) / "mm_projector.bin"
        if proj_path.exists():
            sd = torch.load(proj_path, map_location="cpu")
            if any(k.startswith("mm_projector.") for k in sd):
                sd = {k.split("mm_projector.")[1]: v for k, v in sd.items() if k.startswith("mm_projector.")}
            self.model.mm_projector.load_state_dict(sd, strict=True)
            print(f"[grpo] loaded mm_projector from {proj_path}")

        # LoRA on attention projections; mm_projector stays frozen so we don't
        # compete with the SFT optimum.
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        self.model.cuda()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=0.0
        )

        with open(cfg.data_path) as f:
            data = json.load(f)
        rng = random.Random(cfg.seed)
        rng.shuffle(data)
        n = max(1, int(len(data) * cfg.sample_ratio))
        self.data = data[:n]
        print(f"[grpo] sampled {len(self.data)} of {n} (ratio={cfg.sample_ratio})")

        self.feat_folder = Path(cfg.feat_folder)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _rollout(self, input_ids, image_feat) -> List[torch.Tensor]:
        rollouts = []
        for _ in range(self.cfg.num_rollouts):
            out = self.model.generate(
                input_ids=input_ids,
                images=[image_feat.cuda()],
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
            # When inputs_embeds are used, generate returns ONLY the newly
            # generated tokens (no prefix). Treat the whole thing as response.
            response = out.detach()
            # Truncate at first eos to keep loss tight.
            eos = self.tokenizer.eos_token_id
            if eos is not None:
                eos_pos = (response[0] == eos).nonzero(as_tuple=True)[0]
                if eos_pos.numel():
                    response = response[:, : int(eos_pos[0]) + 1]
            rollouts.append(response)
        return rollouts

    # ------------------------------------------------------------------
    def _step(self, sample: dict):
        scene_id = sample.get("scene_id") or sample.get("sample_token")
        try:
            feat = _load_smap_feat(self.feat_folder, scene_id)
        except Exception as e:
            return None, {"skip_reason": f"feat_load:{e}"}

        question = sample["conversations"][0]["value"]
        gt_answer = sample["conversations"][1]["value"]
        template_type = sample.get("template_type")

        input_ids = build_prompt_ids(
            self.tokenizer, question, self.cfg.max_prompt_len
        ).cuda()

        # Rollouts (no grad).
        self.model.eval()
        rollouts = self._rollout(input_ids, feat)
        decoded = [
            self.tokenizer.decode(r[0], skip_special_tokens=False) for r in rollouts
        ]

        rewards, infos = [], []
        for d in decoded:
            r, inf = compute_reward(d, gt_answer, template_type=template_type)
            rewards.append(r)
            infos.append(inf)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        # Dr.GRPO advantage: subtract group mean, no std normalisation. The
        # std-normalised form blows up tiny reward differences when rollouts
        # converge on a reward "plateau" (e.g. all hitting a fallback floor),
        # producing huge advantages that drive the policy off the manifold.
        # See NORD / "Understanding R1-Zero-like Training" for the rationale.
        advantages = rewards_t - rewards_t.mean()

        # Policy + reference log-probs.
        self.model.train()
        logp_pol = []
        for r in rollouts:
            lp = compute_response_logp(self.model, input_ids, r, feat)
            logp_pol.append(lp)
        logp_pol = torch.stack(logp_pol).squeeze(-1)  # (K,)

        with torch.no_grad():
            with self.model.disable_adapter():
                logp_ref = []
                for r in rollouts:
                    lp = compute_response_logp(self.model, input_ids, r, feat)
                    logp_ref.append(lp)
                logp_ref = torch.stack(logp_ref).squeeze(-1)  # (K,)

        adv = advantages.to(logp_pol.device).to(logp_pol.dtype)
        kl = (logp_pol - logp_ref).clamp(-10.0, 10.0)
        loss = -(adv * logp_pol).mean() + self.cfg.kl_coef * kl.mean()

        log = {
            "reward_mean": float(rewards_t.mean()),
            "reward_std": float(rewards_t.std()),
            "reward_max": float(rewards_t.max()),
            "kl": float(kl.mean()),
            "loss": float(loss.detach()),
            "n_pred_mean": float(np.mean([i["n_pred"] for i in infos])),
            "bev_iou_mean": float(np.mean([i["bev_iou"] for i in infos])),
            "template": template_type or "?",
            "class_ok_rate": float(np.mean([float(i.get("class_ok", False)) for i in infos])),
        }
        return loss, log

    # ------------------------------------------------------------------
    def train(self):
        from collections import deque

        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_f = open(out / "train_log.jsonl", "a")

        # Rolling window of per-prompt info dicts, sized so each log line
        # averages over `gradient_accumulation_steps * log_every` prompts
        # (e.g. 4 * 10 = 40). Smooths out single-prompt noise.
        window_size = self.cfg.gradient_accumulation_steps * self.cfg.log_every
        win = deque(maxlen=window_size)

        # Per-template-type bookkeeping for diagnosis.
        per_template = {}

        global_step = 0
        accum = 0
        t0 = time.time()
        for epoch in range(self.cfg.epochs):
            for i, sample in enumerate(self.data):
                loss, info = self._step(sample)
                if loss is None:
                    continue
                (loss / self.cfg.gradient_accumulation_steps).backward()
                accum += 1

                # Track per-prompt stats.
                win.append(info)
                tt = info.get("template", "?")
                pt = per_template.setdefault(tt, [0, 0.0])
                pt[0] += 1
                pt[1] += info["reward_mean"]

                if accum >= self.cfg.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    global_step += 1

                    if global_step % self.cfg.log_every == 0:
                        # Average numeric fields across the rolling window.
                        keys = ["reward_mean", "reward_std", "reward_max",
                                "kl", "loss", "n_pred_mean", "bev_iou_mean"]
                        avg = {k: float(np.mean([w[k] for w in win])) for k in keys}
                        avg["step"] = global_step
                        avg["epoch"] = epoch
                        avg["wall"] = time.time() - t0
                        avg["window"] = len(win)
                        # Per-template summary.
                        avg["per_template"] = {
                            t: {"n": v[0], "r_mean": v[1] / max(1, v[0])}
                            for t, v in per_template.items()
                        }
                        tmpl_str = " ".join(
                            f"{t}={v['r_mean']:+.2f}(n={v['n']})"
                            for t, v in avg["per_template"].items()
                        )
                        print(
                            f"[step {global_step:>6}] "
                            f"loss={avg['loss']:+.4f} "
                            f"r_mean={avg['reward_mean']:+.3f} "
                            f"r_max={avg['reward_max']:+.3f} "
                            f"bev={avg['bev_iou_mean']:.3f} "
                            f"kl={avg['kl']:+.3f} "
                            f"win={avg['window']} | {tmpl_str}"
                        )
                        log_f.write(json.dumps(avg) + "\n")
                        log_f.flush()

                    if global_step % self.cfg.save_every == 0:
                        ck = out / f"step_{global_step}"
                        ck.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(ck)
                        print(f"[grpo] saved adapter -> {ck}")

        self.model.save_pretrained(out / "final")
        log_f.close()


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--sft_dir",
        default="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b",
    )
    ap.add_argument(
        "--feat_folder",
        default="/data1/byounggun/3davs_b4dl/features/smap_lidar12",
    )
    ap.add_argument(
        "--data_path",
        default="/home/byounggun/B4DL/3dtesting_dataset/train.json",
    )
    ap.add_argument(
        "--output_dir",
        default="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage2_grpo",
    )
    ap.add_argument("--sample_ratio", type=float, default=0.25)
    ap.add_argument("--num_rollouts", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1000)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = GRPOConfig(
        base_model=args.base_model,
        sft_dir=args.sft_dir,
        feat_folder=args.feat_folder,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_ratio=args.sample_ratio,
        num_rollouts=args.num_rollouts,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        epochs=args.epochs,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
    )
    GRPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
