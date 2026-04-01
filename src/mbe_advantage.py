# @Ksgk : MBE-Scaled Advantage GRPO Trainer
# Scales GRPO advantages by per-sequence MBE: higher MBE amplifies advantage,
# lower MBE dampens it. Unlike IBRL (which adds MBE as a loss term), this
# modulates the learning signal strength per rollout.
#
# scaled_advantage = advantage * (1 + alpha * (mbe - mean_mbe) / (std_mbe + eps))
#
# alpha=0 → pure GRPO (no MBE effect)
# alpha>0 → MBE amplifies/dampens advantages
# -----------------------------------------------------------------------------

import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import selective_log_softmax, entropy_from_logits

from src.mbe import mbe_reverse_gram


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class MBEAdvantageConfig(GRPOConfig):
    r"""
    GRPO config with MBE advantage scaling.

    New fields:
        mbe_alpha: Strength of MBE advantage scaling. 0 = pure GRPO.
        mbe_layer: Which hidden layer(s) for MBE. None = all layers (averaged).
    """
    mbe_alpha: float = field(
        default=1.0,
        metadata={"help": "MBE advantage scaling strength. 0 disables MBE scaling."},
    )
    mbe_layer: int | list[int] | None = field(
        default=None,
        metadata={"help": "Hidden layer(s) for MBE. None = average all layers."},
    )


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class MBEAdvantageTrainer(GRPOTrainer):
    """
    GRPO with MBE-scaled advantages.

    For each batch in _compute_loss:
        1. Forward pass with output_hidden_states=True
        2. Compute per-sequence MBE on completion tokens
        3. Scale advantages: adv *= 1 + alpha * z_mbe  (z_mbe = z-scored MBE)
        4. Proceed with standard GRPO loss

    This makes the model learn more from rollouts with diverse internal
    representations (high MBE) and less from rollouts with uniform/collapsed
    representations (low MBE).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mbe_alpha = getattr(self.args, "mbe_alpha", 1.0)
        self.mbe_layer = getattr(self.args, "mbe_layer", None)

    # -----------------------------------------------------------------
    # Per-sequence MBE from hidden states
    # -----------------------------------------------------------------
    def _compute_per_sequence_mbe(
        self, hidden_states: list[torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sequence MBE on completion tokens.

        Args:
            hidden_states: list of (B, T, D) tensors, one per layer
            mask: (B, T) completion mask

        Returns:
            (B,) per-sequence MBE values (averaged across selected layers)
        """
        B = mask.shape[0]
        device = mask.device

        # Select layers
        if self.mbe_layer is None:
            layers = hidden_states  # all layers
        elif isinstance(self.mbe_layer, list):
            layers = [hidden_states[i] for i in self.mbe_layer]
        else:
            layers = [hidden_states[self.mbe_layer]]

        # Accumulate per-sequence MBE across layers
        mbe_accum = torch.zeros(B, device=device)
        n_layers = 0

        for h in layers:
            # h: (B, T, D)
            per_seq_mbe = torch.zeros(B, device=device)
            for i in range(B):
                # Extract unmasked completion tokens for this sequence
                seq_mask = mask[i].bool()
                h_seq = h[i][seq_mask]  # (L_i, D)
                if h_seq.shape[0] < 2:
                    per_seq_mbe[i] = 0.0
                    continue
                h_seq = h_seq.unsqueeze(0)  # (1, L_i, D)
                mbe_val = mbe_reverse_gram(h_seq)  # (1,)
                per_seq_mbe[i] = mbe_val.squeeze()
            mbe_accum += per_seq_mbe
            n_layers += 1

        return mbe_accum / max(n_layers, 1)  # (B,)

    # -----------------------------------------------------------------
    # Override _compute_loss
    # -----------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        if self.mbe_alpha == 0:
            return super()._compute_loss(model, inputs)

        # --- Reconstruct inputs (same as parent) ---
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]

        # --- Forward pass WITH hidden states ---
        batch_size = input_ids.size(0)
        all_logps = []
        all_entropies = []
        all_hidden_per_layer = None

        for start in range(0, batch_size, batch_size):
            end = min(start + batch_size, input_ids.size(0))
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "output_hidden_states": True,
                "use_cache": False,
            }
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            outputs = model(**model_inputs)
            logits = outputs.logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature
            completion_token_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_token_ids)
            all_logps.append(logps)

            with torch.no_grad():
                all_entropies.append(entropy_from_logits(logits))

            # Collect hidden states (completion tokens only), skip embedding layer [0]
            if outputs.hidden_states is not None:
                batch_hidden = [
                    layer_h[:, -logits_to_keep:, :] for layer_h in outputs.hidden_states[1:]
                ]
                if all_hidden_per_layer is None:
                    all_hidden_per_layer = [[h] for h in batch_hidden]
                else:
                    for li, h in enumerate(batch_hidden):
                        all_hidden_per_layer[li].append(h)

        per_token_logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0)

        # Concatenate hidden states across sub-batches
        hidden_states = None
        if all_hidden_per_layer is not None:
            hidden_states = [torch.cat(layer_list, dim=0) for layer_list in all_hidden_per_layer]

        # --- Compute per-sequence MBE and scale advantages ---
        advantages = inputs["advantages"]

        if hidden_states is not None:
            with torch.no_grad():
                per_seq_mbe = self._compute_per_sequence_mbe(hidden_states, mask)
                # Z-score MBE within batch
                mbe_mean = per_seq_mbe.mean()
                mbe_std = per_seq_mbe.std()
                z_mbe = (per_seq_mbe - mbe_mean) / (mbe_std + 1e-6)
                # Scale factor: 1 + alpha * z_mbe
                mbe_scale = 1.0 + self.mbe_alpha * z_mbe  # (B,)
                # Clamp to prevent sign flips or extreme values
                mbe_scale = mbe_scale.clamp(min=0.1, max=3.0)

            advantages = advantages * mbe_scale

            # Log MBE metrics
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["mbe/mean"].append(
                self.accelerator.gather(mbe_mean.detach()).nanmean().item()
            )
            self._metrics[mode]["mbe/std"].append(
                self.accelerator.gather(mbe_std.detach()).nanmean().item()
            )
            self._metrics[mode]["mbe/scale_mean"].append(
                self.accelerator.gather(mbe_scale.mean().detach()).nanmean().item()
            )
            self._metrics[mode]["mbe/scale_std"].append(
                self.accelerator.gather(mbe_scale.std().detach()).nanmean().item()
            )

        # --- Standard GRPO loss from here (mirrors parent exactly) ---
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        completion_token_count = mask.sum().clamp(min=1.0)

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo", "luspo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # --- Metrics (mirrors parent) ---
        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(gathered_low_clip.min().item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(gathered_high_clip.max().item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss
