"""attn-reward Dr.GRPO (user Idea 2, Jul 21): scale each CoT token's
advantage by how much the ANSWER tokens attend to it — used tokens earn
their reward, redundant tokens don't, and the model is left to do its own
compression.

Mechanics: after generation, an eager-attention clone (weight-synced from
the live policy each step) re-forwards the rollouts; layer-L answer-span ->
completion attention gives per-token scores; weights = score / rollout-mean
(mean-1 => total credit conserved), clamped to [w_min, w_max]; answer-span
and post-span tokens keep weight 1. Advantages (B,T) *= W, exactly the
GradientMaskGRPOTrainer channel. Use with loss_type=dr_grpo.

Graft-study grounding: answer-attention is the only selection rule whose
positions causally carry the computation (columns 90% vs lens/random at
floor) — the same signal, used here as credit assignment.
"""
from __future__ import annotations

import re

import torch

from trl import GRPOTrainer

# _probe is a checkout ckpt from current policy, used to produce attention score on eager mode
#        since online switching attention mode is not supported

# Request 1. We need to ablate on layer choice for the attention score extraction here
#            we are running on a thin margin 68% - 70% here, perhaps a better layer would close it

# Concern 1. We allocate attn reward ONLY when #### is present, this is not the case for our Dr.GRPO pipelines
#            we detect the last number as answer when #### is not present, so the fair game is to find the last 
#            number in the sequence, when #### is not present. Moreover, we use all trailing tokens after ####'s 
#            backward attention to compute CoT attn reward, this is unjustified, in case that model generate verbose
#            trailing responses, this effectively dillute the attention score. We need to re-validate if our attn filter
#            method uses the same logic. Ideally we detect the answer number (if #### is present it is the following number, 
#            if #### is not present it is simply the last number, and we need to check attention from -1 shifted positions
#            because that position predict the answer with its last layer rep)

# Modification 1. 
#            Currently, we normalize the attention score within each sequence, I add attn normalization within a group 
#            and a batch for ablation purpose, it is true that attn score is dilluted in longer CoT, but we might actually want it
#            experiment results will tell us which one is better. What if we have a CoT that is extremely useless and the answer 
#            simply don't attend to it at all? The current in-sequence normalization scheme will still force out some CoT tokens
#            that enjoy positive reward, simply because they are relatively more attended towards compared to their peers.
#            This is one motivation for the normalization over bigger range ideas, too. 

# Idea 1. 
#           Token level attn reward injects instability into the training process, so we can add back the 
#           KL anchor (but keep the Dr.GRPO fixes), I suspect this can lift up stability of the training process, but we should
#           be expecting to see a CoT length compression nonethless here
#           GRPOConfig(loss_type="dr_grpo", beta=0.01) suffice to enable this config (perhaps we have a better beta value, ablate here)

# Concern 2. 
#          Make sure we are using the attn reward by checking the logs, the try except logic is concerning

# Concern 3. 
#         the pos_only is not ablated with, we need to justify why we even need it



# ===================== Irrelevant Consideration to this script =========================
# Q5. Two things contradict each other in my mind, Dr.GRPO do not need KL regularization, my ablation shows that the 
#     removal of length normalization on the sequence level loss is the sole rescuer that eliminates need for KL 
#     regularization, in another word, it eliminates the KL divergence. Now the funny issue is that on-policy logit
#     distillation pipeline for us necessitate a KL regularization to work, even if we are just doing logit imitation
# Thought. For off-policy logit distillation, we should ablte on the effect of KL anchor, we observe a more significant 
#          degradation for on-policy logit distillation when we remove the KL anchor (which is very very odd, if we are
#          already on-policy, why would the logit distillation be misleading? what is so un-trustworthy for logit-
#          distillation?) 
# Q6. For on-policy logit distillation, did we normalize the gradient in a sequence, which creates similar bias that
#     dillute gradient signal for tokens in a long sequence? No for OPD we actually do the reverse: we average loss across
#     all token positions, this means we actually allocate MORE gradient to the longer rollout, since longer rollout also 
#     tends to be the wrong rollouts, we are basically learning more on failed rollouts than on correct rollouts. That is
#     OPD has an innate bias towards learning on failed rollouts. For off-policy logit distillation, this might be 
#     catastrohpic: naturally we expect the model to learn off-policy corect & short rollout, but instead we focus more 
#     on the wrong, lengthy rollout, this is not the purpose of off-policy learning. A hypothesis is on-policy learning is 
#     about trimming errors, whilst off-policy learning is about learning success, so, we might want to include sequence
#     level loss normalization for off-policy logit distillation, and see if it makes any difference.

# Hypothesis 1. Whilst we distill from the top-K logit, there is no supervision force for the rest of the logit at all. 
#               This makes KL-anchor free distillation unstable. 
# Validation 1. KL anchor ONLY on non-top-K logit, so that we put control to the 'scale' of the probability mass, as well
#               as the shape of its distribution. 
# =========================================================================================


class AttnRewardGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, attn_layer: int = 21, w_min: float = 0.2,
                 w_max: float = 5.0, pos_only: bool = False,
                 sync_every: int = 1, attn_norm: str = "sequence", **kwargs):
        if attn_norm not in ("sequence", "batch", "group"):
            raise ValueError(f"unknown attn_norm={attn_norm}")
        self.attn_layer = attn_layer
        self.w_min, self.w_max = w_min, w_max
        self.pos_only = pos_only
        self.sync_every = sync_every
        self.attn_norm = attn_norm
        self._probe = None
        self._probe_step = -1
        super().__init__(*args, **kwargs)

    def _get_probe(self):
        from transformers import AutoModelForCausalLM
        base = self.accelerator.unwrap_model(self.model)
        if self._probe is None:
            self._probe = AutoModelForCausalLM.from_config(
                base.config, attn_implementation="eager")
            self._probe = self._probe.to(torch.bfloat16).to(
                self.accelerator.device)
            self._probe.eval()
            [p.requires_grad_(False) for p in self._probe.parameters()]
            self._probe_step = -1
        if self._probe_step != self.state.global_step or \
                self._probe_step < 0:
            sd = {k: v.detach().to(torch.bfloat16)
                  for k, v in base.state_dict().items()}
            self._probe.load_state_dict(sd, strict=True)
            self._probe_step = self.state.global_step
        return self._probe

    def _answer_start_tok(self, comp_ids_row, mask_row):
        """Token index where the answer span ('####' onwards) starts;
        binary search over incremental decodes. None if no marker."""
        toks = comp_ids_row[mask_row.bool()].tolist()
        text = self.processing_class.decode(toks, skip_special_tokens=True)
        m = re.search(r"####", text)
        if not m:
            return None
        target = m.start()
        lo, hi = 0, len(toks)
        while lo < hi:
            mid = (lo + hi) // 2
            if len(self.processing_class.decode(
                    toks[:mid], skip_special_tokens=True)) < target:
                lo = mid + 1
            else:
                hi = mid
        return lo
    
    @torch.no_grad()
    def _attn_weights(self, output):
        probe = self._get_probe()
        p_ids, p_mask = output["prompt_ids"], output["prompt_mask"]
        c_ids, c_mask = output["completion_ids"], output["completion_mask"]
        ids = torch.cat([p_ids, c_ids], 1)
        attn = torch.cat([p_mask, c_mask], 1)
        B, T = c_ids.shape
        P = p_ids.size(1)
        W = torch.ones((B, T), device=c_ids.device, dtype=torch.float32)
        scores = torch.zeros_like(W)
        cot_mask = torch.zeros_like(W, dtype=torch.bool)
        for c0 in range(0, B, 2):
            rr = slice(c0, min(c0 + 2, B))
            out = probe(input_ids=ids[rr], attention_mask=attn[rr],
                        output_attentions=True) 
            # Here we actually call the probe deliberately, what is 'rr' again? And probe is the thing that expose 
            # attention scores? 
            A = out.attentions[self.attn_layer]        # (b, H, L, L)
            del out
            Ah = A.mean(1)                             # head-avg (b, L, L) | avg attention over all heads
            del A
            for bi_l, bi_g in enumerate(range(rr.start, rr.stop)): # looping over all sequences in the batch, compute attn reward separately
                a0 = self._answer_start_tok(c_ids[bi_g], c_mask[bi_g])
                n_valid = int(c_mask[bi_g].sum())
                if a0 is None or a0 < 8 or a0 >= n_valid:
                    continue
                q_rows = Ah[bi_l, P + a0:P + n_valid, :]     # ans -> all | P is end of prompt ids, P + a0 is '####' answer formatting token idx, P + n_valid is end of completion idx, here we get attention from all trailing tokens after ####'s attention score back to CoT tokens 
                score = q_rows[:, P:P + a0].mean(0)          # per CoT tok avg. attn from all trailing completion tokens after '####' | P + a0 is the indx of '####' answer formatting tokens
                scores[bi_g, :a0] = score
                cot_mask[bi_g, :a0] = True
            del Ah
        if self.attn_norm == "sequence":
            for row in range(B):
                mask = cot_mask[row]
                if mask.any():
                    W[row, mask] = scores[row, mask] / scores[row, mask].mean().clamp(min=1e-8)
        elif self.attn_norm == "batch":
            if cot_mask.any():
                W[cot_mask] = scores[cot_mask] / scores[cot_mask].mean().clamp(min=1e-8)
        else:
            for start in range(0, B, self.num_generations):
                group_mask = cot_mask[start:start + self.num_generations]
                if group_mask.any():
                    group_scores = scores[start:start + self.num_generations]
                    W[start:start + self.num_generations][group_mask] = (
                        group_scores[group_mask] / group_scores[group_mask].mean().clamp(min=1e-8)
                    )
        W[cot_mask] = W[cot_mask].clamp(self.w_min, self.w_max)
        return W

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        try:
            W = self._attn_weights(output)
            adv = output["advantages"]
            if adv.dim() == 1:
                adv = adv.unsqueeze(-1).expand(-1, W.size(1)).contiguous()
            if self.pos_only:
                W = torch.where(adv > 0, W, torch.ones_like(W))
            output["advantages"] = adv * W.to(adv.dtype)
            if self.state.global_step % 10 == 0:
                cw = W[output["completion_mask"].bool()]
                print(f"[attnrw] W mean {cw.mean():.2f} "
                      f"frac<0.5 {(cw < 0.5).float().mean():.2f} "
                      f"frac>2 {(cw > 2).float().mean():.2f}", flush=True)
        except Exception as e:  # never kill training on probe hiccups
            print(f"[attnrw] skipped step ({type(e).__name__}: {e})",
                  flush=True)
        return output
