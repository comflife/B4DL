"""Joint LidarCLIP (SST encoder) + VTimeLLM stage2 training on Nu-Grounding (3dtesting).

Differences vs. train.py:
- Loads raw .pcd.bin via JointLidarDataset (no pre-extracted features).
- Builds LidarEncoderSST, loads its pretrained weights, attaches it as a
  submodule to the (LoRA-wrapped) LLM so its parameters train jointly.
- compute_loss runs:  pc -> SST -> (B, 1, 768) features  ->  LLM(images=...).
- Saves: LoRA adapter (PEFT) + lidar_encoder.bin + optional mm_projector.bin.
"""
import os
# Must be set before importing torch.utils.tensorboard (via accelerate).
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
import distutils.version  # noqa: F401  -- force submodule registration

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.append(root_dir)
sys.path.append("/home/byounggun/B4DL/encoders/lidarclip")

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch

# torch<1.10/1.12 compat shims for accelerate>=1.0 calling newer torch APIs.
# Must run BEFORE importing transformers/accelerate.
if not hasattr(torch.cuda, "is_bf16_supported"):
    def _is_bf16_supported(*args, **kwargs):
        if not torch.cuda.is_available():
            return False
        try:
            major, _ = torch.cuda.get_device_capability()
            return major >= 8  # Ampere+
        except Exception:
            return False
    torch.cuda.is_bf16_supported = _is_bf16_supported

if not hasattr(torch.backends, "mps"):
    class _MpsShim:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def is_built(): return False
    torch.backends.mps = _MpsShim()

if not hasattr(torch, "mps"):
    class _MpsModule:
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def synchronize(): pass
    torch.mps = _MpsModule()

if not hasattr(torch, "xpu"):
    class _XpuShim:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def empty_cache(): pass
        @staticmethod
        def synchronize(): pass
    torch.xpu = _XpuShim()

import torch.nn as nn
import transformers
from transformers import Trainer

from vtimellm import conversation as conversation_lib
from vtimellm.model import VTimeLLMLlamaForCausalLM
from vtimellm.train.dataset_3dtesting import JointDataArguments, make_joint_data_module
from vtimellm.train.train import (
    ModelArguments,
    TrainingArguments,
    find_all_linear_names,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    smart_tokenizer_and_embedding_resize,
)

from lidarclip.model.sst import LidarEncoderSST


def rank0_print(*a, **kw):
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(*a, **kw)


def build_lidar_encoder(sst_config_path: str, ckpt_path: Optional[str], clip_dim: int = 768):
    enc = LidarEncoderSST(sst_config_path, clip_embedding_dim=clip_dim)
    if ckpt_path and os.path.isfile(ckpt_path):
        rank0_print(f"[lidar] loading ckpt: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck.get("state_dict", ck)
        # LightningModule wrapped weights as 'lidar_encoder.<name>'
        stripped = {}
        for k, v in sd.items():
            if k.startswith("lidar_encoder."):
                stripped[k[len("lidar_encoder."):]] = v
        if not stripped:
            stripped = sd  # already raw encoder dict
        missing, unexpected = enc.load_state_dict(stripped, strict=False)
        rank0_print(f"[lidar] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        rank0_print(f"[lidar] no ckpt provided / not found ({ckpt_path}); using random init.")
    return enc


@dataclass
class LidarArguments:
    lidar_sst_config: str = field(
        default="/home/byounggun/B4DL/encoders/lidarclip/lidarclip/model/sst_encoder_only_config.py")
    lidar_ckpt: Optional[str] = field(
        default="/home/byounggun/B4DL/encoders/lidarclip/ckpt/lidarclip_mm/epochepoch=04.ckpt")
    freeze_lidar: bool = field(default=False)
    lidar_lr: Optional[float] = field(default=None,
                                      metadata={"help": "If set, use a separate LR for SST encoder."})


class JointTrainer(Trainer):
    """Trainer that runs the lidar encoder before the LLM forward."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pcs = inputs.pop("point_clouds")
        device = next(model.parameters()).device
        # SST expects a list of (N, 4) tensors on the same device
        pcs = [p.to(device=device, dtype=torch.float32) for p in pcs]
        # Reach the lidar encoder regardless of PEFT/DDP wrapping.
        encoder = self._unwrap_lidar(model)
        feats, _ = encoder(pcs)              # (B, 768)
        feats = feats.to(dtype=self._llm_dtype(model)).unsqueeze(1)  # (B, 1, 768)
        inputs["images"] = feats
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    @staticmethod
    def _unwrap_lidar(model):
        m = model
        while hasattr(m, "module") and not hasattr(m, "lidar_encoder"):
            m = m.module
        if hasattr(m, "lidar_encoder"):
            return m.lidar_encoder
        # PEFT wraps original model under .base_model.model
        return m.base_model.model.lidar_encoder

    @staticmethod
    def _llm_dtype(model):
        for p in model.parameters():
            if p.is_floating_point():
                return p.dtype
        return torch.float32

    def _save(self, output_dir=None, state_dict=None):
        # Standard PEFT/HF save (LoRA adapter + config), then add lidar_encoder + mm_projector.
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        rank0_print(f"[save] {output_dir}")

        # 1. LoRA adapter weights
        peft_state = get_peft_state_maybe_zero_3(
            self.model.named_parameters(), self.args.lora_bias)
        non_lora_state = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters())
        # Drop lidar_encoder.* and mm_projector from non_lora (saved separately)
        non_lora_state = {k: v for k, v in non_lora_state.items()
                          if "lidar_encoder." not in k and "mm_projector" not in k}
        if self.args.should_save:
            self.model.config.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir, state_dict=peft_state)
            torch.save(non_lora_state,
                       os.path.join(output_dir, "non_lora_trainables.bin"))

            # 2. lidar encoder
            lidar = self._unwrap_lidar(self.model)
            torch.save({k: v.detach().cpu() for k, v in lidar.state_dict().items()},
                       os.path.join(output_dir, "lidar_encoder.bin"))

            # 3. mm_projector (only if it's trainable; else stage1's is unchanged)
            if not self.args.freeze_mm_mlp_adapter:
                mm = self._unwrap_mm_projector(self.model)
                torch.save({k: v.detach().cpu() for k, v in mm.state_dict().items()},
                           os.path.join(output_dir, "mm_projector.bin"))

    @staticmethod
    def _unwrap_mm_projector(model):
        m = model
        while hasattr(m, "module") and not hasattr(m, "get_model"):
            m = m.module
        if hasattr(m, "base_model"):  # PEFT
            m = m.base_model.model
        return m.get_model().mm_projector


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, JointDataArguments, TrainingArguments, LidarArguments))
    model_args, data_args, training_args, lidar_args = parser.parse_args_into_dataclasses()

    compute_dtype = (torch.float16 if training_args.fp16
                     else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # Base LLM
    model = VTimeLLMLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    smart_tokenizer_and_embedding_resize(
        {"additional_special_tokens": ["<meta>", "<4DLiDAR>", "<tg>"]},
        tokenizer, model)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # mm_projector init (loads stage1 weights)
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    model.get_model().initialize_vision_modules(model_args=model_args)
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    # LoRA on LLM
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bf16:
            model.to(torch.bfloat16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # mm_projector kept in compute_dtype for embedding-merge consistency.
    # (After LoRA, base weights are bf16; mm_projector follows from .to() above.)

    # LiDAR encoder — kept in fp32 (bf16 SparseConv is unstable).
    lidar_encoder = build_lidar_encoder(
        lidar_args.lidar_sst_config, lidar_args.lidar_ckpt, clip_dim=768)
    lidar_encoder = lidar_encoder.to(torch.float32)
    if lidar_args.freeze_lidar:
        for p in lidar_encoder.parameters():
            p.requires_grad = False
        lidar_encoder.eval()

    # Attach as submodule so optimizer covers its params and DDP/Trainer treat it uniformly.
    if hasattr(model, "base_model"):
        model.base_model.model.lidar_encoder = lidar_encoder
    else:
        model.lidar_encoder = lidar_encoder

    # Sanity print
    n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lidar_encoder" not in n)
    n_lidar = sum(p.numel() for p in lidar_encoder.parameters() if p.requires_grad)
    rank0_print(f"trainable: LoRA+misc={n_lora:,}  lidar_encoder={n_lidar:,}")

    data_module = make_joint_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = JointTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer._save(training_args.output_dir)


if __name__ == "__main__":
    train()
