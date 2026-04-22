import os
import torch
import torch.nn.functional as F

from transformers import Trainer
from typing import Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class VTimeLLMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Setup Number Token Loss (NTL) if enabled
        if getattr(self.args, 'use_ntl_loss', False):
            vocab_size = len(self.tokenizer)
            digit_values = torch.zeros(vocab_size, dtype=torch.float32)
            digit_mask = torch.zeros(vocab_size, dtype=torch.bool)
            
            for tid in range(vocab_size):
                try:
                    token_str = self.tokenizer.decode([tid], skip_special_tokens=True).strip()
                    # Support both single digit and multi-digit tokens
                    if token_str.isdigit():
                        digit_values[tid] = float(token_str)
                        digit_mask[tid] = True
                except Exception:
                    pass
            
            self.register_buffer('ntl_digit_values', digit_values)
            self.register_buffer('ntl_digit_mask', digit_mask)
            print(f"[NTL] Initialized. Found {digit_mask.sum().item()} digit tokens out of {vocab_size}")
    
    def compute_ntl_loss(self, logits, labels):
        """
        Number Token Loss (NTL) - computes regression-like loss on digit tokens.
        logits: [B, L, V]
        labels: [B, L]
        """
        if not hasattr(self, 'ntl_digit_mask'):
            return torch.tensor(0.0, device=logits.device)
        
        # Create mask for valid digit positions (exclude IGNORE_INDEX)
        valid_positions = (labels != -100)
        
        # Check which labels are digit tokens
        is_digit = self.ntl_digit_mask[labels.clamp(min=0)]  # clamp to handle -100
        valid_mask = valid_positions & is_digit
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device)
        
        # Gather valid logits and labels
        valid_logits = logits[valid_mask]  # [N, V]
        valid_labels = labels[valid_mask]  # [N]
        
        # Softmax over vocabulary
        probs = F.softmax(valid_logits, dim=-1)  # [N, V]
        
        # Expected numerical value: E[X] = Σ p(x) * value(x)
        expected = (probs * self.ntl_digit_values.unsqueeze(0)).sum(dim=-1)  # [N]
        
        # Ground truth numerical values
        gt_values = self.ntl_digit_values[valid_labels]  # [N]
        
        # Smooth L1 loss (Huber loss) between expected and GT
        loss = F.smooth_l1_loss(expected, gt_values)
        return loss
    
    def compute_tg_regression_loss(self, model, inputs, outputs):
        """Compute temporal grounding regression loss from <tg> token hidden states."""
        is_tg = inputs.get('is_time_grounding')
        if is_tg is None or not is_tg.any():
            return torch.tensor(0.0, device=outputs.logits.device)
        
        tg_token_id = self.tokenizer.convert_tokens_to_ids("<tg>")
        input_ids = inputs.get('input_ids')
        
        tg_mask = (input_ids == tg_token_id)
        tg_indices = is_tg.nonzero(as_tuple=True)[0]
        
        if tg_indices.numel() == 0:
            return torch.tensor(0.0, device=outputs.logits.device)
        
        hidden_states = outputs.hidden_states[-1]
        tg_hidden_list = []
        gt_intervals = []
        
        for idx in tg_indices:
            sample_tg_mask = tg_mask[idx]
            if sample_tg_mask.any():
                tg_pos = sample_tg_mask.nonzero(as_tuple=True)[0][0]
                tg_hidden_list.append(hidden_states[idx, tg_pos])
                gt_intervals.append(torch.tensor([
                    inputs['start_frame'][idx].item(),
                    inputs['end_frame'][idx].item()
                ], dtype=torch.float32, device=hidden_states.device))
        
        if len(tg_hidden_list) == 0:
            return torch.tensor(0.0, device=outputs.logits.device)
        
        tg_hiddens = torch.stack(tg_hidden_list, dim=0)
        gt_intervals = torch.stack(gt_intervals, dim=0)
        pred_intervals = model.regression_head(tg_hiddens)
        reg_loss = F.smooth_l1_loss(pred_intervals, gt_intervals)
        return reg_loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Combined loss: CE + optional NTL + optional TG Regression.
        """
        model_inputs = dict(inputs)
        for key in ("is_time_grounding", "start_frame", "end_frame"):
            model_inputs.pop(key, None)

        outputs = model(**model_inputs)
        ce_loss = outputs.loss
        
        device = outputs.logits.device
        total_loss = ce_loss
        
        logits = outputs.logits  # [B, L, V]
        labels = inputs['labels']  # [B, L]
        
        # 1. Number Token Loss (NTL)
        if getattr(self.args, 'use_ntl_loss', False):
            # Shift logits and labels to align with next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ntl_loss = self.compute_ntl_loss(shift_logits, shift_labels)
            total_loss = total_loss + self.args.lambda_ntl * ntl_loss
        
        # 2. Temporal Grounding Regression Loss
        lambda_reg = getattr(self.args, 'lambda_reg', 0.0)
        if lambda_reg > 0:
            reg_loss = self.compute_tg_regression_loss(model, inputs, outputs)
            total_loss = total_loss + lambda_reg * reg_loss
        
        if return_outputs:
            return (total_loss, outputs)
        return total_loss

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(VTimeLLMTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(VTimeLLMTrainer, self)._save(output_dir, state_dict)
