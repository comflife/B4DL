"""Custom GRPO fine-tuner for the SMAP-LLM model on nuGrounding.

Why custom: the upstream B4DL codebase pins transformers 4.31 / peft 0.4 and
has no TRL. Pulling TRL's GRPOTrainer would force an SDK upgrade *and* it
doesn't natively handle our LiDAR-feature visual prompt anyway. So we run a
small, focused GRPO loop here:

  * Policy = Stage-1 full-FT model + a freshly added LoRA adapter (trainable).
  * Reference = same backbone with `disable_adapter()` (no extra GPU copy).
  * For each prompt, K rollouts via model.generate(images=...).
  * Group-normalised advantages (GRPO baseline) + KL to reference.
  * Per-token NLL is computed by re-running model(forward) over
    [prompt + rollout] tokens — gradients only on the rollout span.

This keeps memory feasible on a single A6000 (≈40GB peak with bf16 + LoRA).

Reward is the LiDAR-aware composite from rewards_lidar.compute_reward.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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

# Bring upstream B4DL/mllm and our patches into the path.
NEW_CODE_DIR = Path(__file__).resolve().parent
B4DL_MLLM = Path("/home/byounggun/B4DL/mllm").resolve()
sys.path.insert(0, str(NEW_CODE_DIR))
sys.path.insert(0, str(B4DL_MLLM))

# Importing this also installs the projector / dataset monkey patches we
# defined for the SFT side (input_dim=512, .pt feature loader).
import train_smap  # noqa: F401  pylint: disable=unused-import

import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from vtimellm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from vtimellm.model import VTimeLLMLlamaForCausalLM  # noqa: E402
from vtimellm import conversation as conversation_lib  # noqa: E402
from vtimellm.mm_utils import tokenizer_image_token  # noqa: E402

from rewards_lidar import compute_reward, RewardWeights  # noqa: E402


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_smap_feature(feat_dir: str, scene_id: str) -> torch.Tensor:
    blob = torch.load(Path(feat_dir) / f"{scene_id}.pt", map_location="cpu")
    feat = blob["output_smap"]
    if feat.dim() == 3:
        feat = feat.squeeze(0)
    return feat.float()


def build_prompt(question_text: str) -> str:
    """Wrap a single-turn question with the v1 conversation template.

    `question_text` should already contain the <image> placeholder where the
    LiDAR tokens go.
    """
    conv = conversation_lib.conv_templates["v1"].copy()
    # Replace the placeholder so the tokenizer routine in vtimellm picks it up.
    q = question_text.replace("<image>", DEFAULT_IMAGE_TOKEN)
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ---------------------------------------------------------------------------
# Per-rollout log-prob computation
# ---------------------------------------------------------------------------
def compute_response_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
    image_feats: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sum_logp, len) for the `response_ids` span only.

    We feed [prompt | response] through the model with `images=` so the
    multimodal projector inserts the LiDAR tokens; we read logits at positions
    that predict each response token.
    """
    full_ids = torch.cat([input_ids, response_ids], dim=1)  # (1, T)
    out = model(
        input_ids=full_ids,
        images=image_feats[None].to(full_ids.device),
        use_cache=False,
        return_dict=True,
    )
    logits = out.logits  # (1, T, V)

    # The multimodal pipeline expands the single <image> token into N
    # LiDAR tokens, so the position of the first response logit is shifted.
    # We use a robust scheme: locate response by its known length and use the
    # *last* `len(response_ids)` positions of logits (causal LM shifts by 1).
    R = response_ids.shape[1]
    # logits at positions [-R-1 : -1] predict response tokens [0 : R].
    response_logits = logits[:, -R - 1 : -1, :]  # (1, R, V)
    target = response_ids  # (1, R)
    logp = F.log_softmax(response_logits.float(), dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, R)
    return token_logp.sum(dim=1), torch.tensor([R], device=token_logp.device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    output_dir: str
    base_model: str
    stage1_dir: str
    feat_folder: str
    data_path: str
    sample_ratio: float = 0.25
    num_rollouts: int = 4
    micro_batch: int = 1  # prompts per optimizer step (rollouts multiply this)
    gradient_accumulation_steps: int = 4
    max_prompt_len: int = 512
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    epochs: int = 1
    seed: int = 0
    lora_r: int = 64
    lora_alpha: int = 128
    log_every: int = 10
    save_every: int = 1000


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        # ---------------- model + tokenizer ----------------
        print(f"[grpo] loading tokenizer from {cfg.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            model_max_length=cfg.max_prompt_len + cfg.max_new_tokens + 64,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        print(f"[grpo] loading stage1 weights from {cfg.stage1_dir}")
        self.model = VTimeLLMLlamaForCausalLM.from_pretrained(
            cfg.stage1_dir,
            torch_dtype=torch.bfloat16,
        )
        # Apply our SMAP projector dims (in case stage1 dir is just the LM
        # weights without projector — load it explicitly if the file exists).
        proj_path = Path(cfg.stage1_dir) / "mm_projector.bin"
        if proj_path.exists():
            sd = torch.load(proj_path, map_location="cpu")
            sd = {k.split("mm_projector.")[1]: v for k, v in sd.items() if "mm_projector" in k}
            self.model.get_model().mm_projector.load_state_dict(sd)
            print(f"[grpo] loaded projector from {proj_path}")

        # Match the special-token set Stage-1 registered (only our bbox
        # span markers; the B4DL <meta>/<4DLiDAR>/<tg> tokens are intentionally
        # NOT used in this pipeline).
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|box_start|>", "<|box_end|>"]}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # LoRA wrap (policy = base + adapter; reference = base with adapter
        # disabled).
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

        # ---------------- optimizer ----------------
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=0.0
        )

        # ---------------- data ----------------
        with open(cfg.data_path) as f:
            data = json.load(f)
        rng = random.Random(cfg.seed)
        rng.shuffle(data)
        n = max(1, int(len(data) * cfg.sample_ratio))
        self.data = data[:n]
        print(f"[grpo] sampled {len(self.data)} from {cfg.data_path}")

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_rollouts(
        self, input_ids: torch.Tensor, image_feats: torch.Tensor
    ) -> List[torch.Tensor]:
        """Returns a list of length K of (1, R_i) response token tensors."""
        rollouts = []
        for _ in range(self.cfg.num_rollouts):
            out = self.model.generate(
                input_ids,
                images=image_feats[None].cuda(),
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
            response = out[:, input_ids.shape[1]:]
            rollouts.append(response.detach())
        return rollouts

    # ------------------------------------------------------------------
    # One example -> loss
    # ------------------------------------------------------------------
    def step_one(self, sample: dict) -> Tuple[torch.Tensor, dict]:
        scene_id = sample.get("scene_id") or sample.get("sample_token")
        try:
            feats = load_smap_feature(self.cfg.feat_folder, scene_id)
        except Exception as e:
            return None, {"skip_reason": f"feat_load:{e}"}

        question = sample["conversations"][0]["value"]
        gt_answer = sample["conversations"][1]["value"]

        prompt = build_prompt(question)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # 1) rollouts (no grad)
        self.model.eval()
        rollouts = self.generate_rollouts(input_ids, feats)
        # Decode for reward.
        decoded = [
            self.tokenizer.decode(r[0], skip_special_tokens=True) for r in rollouts
        ]
        template_type = sample.get("template_type")
        rewards = []
        infos = []
        for d in decoded:
            r, inf = compute_reward(d, gt_answer, template_type=template_type)
            rewards.append(r)
            infos.append(inf)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        if rewards_t.std() < 1e-6:
            advantages = torch.zeros_like(rewards_t)
        else:
            advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

        # 2) policy log-probs (with grad) and reference log-probs (no grad).
        self.model.train()
        logp_policy = []
        for r in rollouts:
            lp_sum, _ = compute_response_logp(self.model, input_ids, r, feats)
            logp_policy.append(lp_sum)
        logp_policy = torch.stack(logp_policy).squeeze(-1)  # (K,)

        with torch.no_grad():
            with self.model.disable_adapter():
                logp_ref = []
                for r in rollouts:
                    lp_sum, _ = compute_response_logp(
                        self.model, input_ids, r, feats
                    )
                    logp_ref.append(lp_sum)
                logp_ref = torch.stack(logp_ref).squeeze(-1)  # (K,)

        # 3) GRPO objective. ratio is 1 in the first iteration; we use the
        #    log-prob form for stability (no importance weighting since we
        #    do a single update per rollout batch).
        adv = advantages.to(logp_policy.device).to(logp_policy.dtype)
        kl = (logp_policy - logp_ref).clamp(-10.0, 10.0)
        loss = -(adv * logp_policy).mean() + self.cfg.kl_coef * kl.mean()

        log = {
            "reward_mean": float(rewards_t.mean()),
            "reward_std": float(rewards_t.std()),
            "reward_max": float(rewards_t.max()),
            "kl": float(kl.mean()),
            "loss": float(loss.detach()),
            "n_pred_mean": float(np.mean([i["n_pred"] for i in infos])),
            "bev_iou_mean": float(np.mean([i["bev_iou"] for i in infos])),
        }
        return loss, log

    # ------------------------------------------------------------------
    def train(self):
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "train_log.jsonl"
        log_f = open(log_path, "a")

        global_step = 0
        accum = 0
        t0 = time.time()
        for epoch in range(self.cfg.epochs):
            for i, sample in enumerate(self.data):
                loss, info = self.step_one(sample)
                if loss is None:
                    continue
                (loss / self.cfg.gradient_accumulation_steps).backward()
                accum += 1
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
                        info["step"] = global_step
                        info["epoch"] = epoch
                        info["wall"] = time.time() - t0
                        print(
                            f"[step {global_step:>6}] "
                            f"loss={info['loss']:.4f} "
                            f"r_mean={info['reward_mean']:+.3f} "
                            f"r_max={info['reward_max']:+.3f} "
                            f"bev={info['bev_iou_mean']:.3f} "
                            f"kl={info['kl']:.3f}"
                        )
                        log_f.write(json.dumps(info) + "\n")
                        log_f.flush()

                    if global_step % self.cfg.save_every == 0:
                        ck = out_dir / f"step_{global_step}"
                        ck.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(ck)
                        print(f"[grpo] saved adapter -> {ck}")

        # Final save.
        self.model.save_pretrained(out_dir / "final")
        log_f.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="/home/byounggun/B4DL/base_model/vicuna-v1-5-7b")
    ap.add_argument(
        "--stage1_dir",
        default="/data1/byounggun/3davs_b4dl/checkpoints/stage1_full_combined",
        help="Directory with the Stage-1 SFT checkpoint (LM weights + mm_projector.bin).",
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
        default="/data1/byounggun/3davs_b4dl/checkpoints/stage2_grpo",
    )
    ap.add_argument("--sample_ratio", type=float, default=0.25)
    ap.add_argument("--num_rollouts", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=1000)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = GRPOConfig(
        base_model=args.base_model,
        stage1_dir=args.stage1_dir,
        feat_folder=args.feat_folder,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_ratio=args.sample_ratio,
        num_rollouts=args.num_rollouts,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
    trainer = GRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
