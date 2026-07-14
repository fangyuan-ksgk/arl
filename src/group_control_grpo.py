"""
Greedy anchoring: 咔咔乱杀 （+2pt）
Skip：跟基线同等 （说明成功率低的group没什么价值）
Pos only：纯奖励无惩罚导致end repetition，长度爆炸
Neg only：减弱-2pt，稳定性较差，这说明GRPO主要靠negative gradient起作用


  "skip"    [Idea 1]  If a group has at least one correct rollout (proxy for
            "greedily solvable") but a wrong majority, zero the whole group's
            advantages — the step becomes a no-op for that query.

  "resample" [Idea 2]  For such groups, generate G additional rollouts and
            splice their correct completions over wrong slots until the group
            is majority-right, then recompute advantages. Extra compute, same
            objective shape.

  "greedy"  [Idea 3]  Always generate one extra greedy (T=0) rollout per query
            and splice it over the lowest-reward slot. The incumbent greedy
            chain is then always *in* the group: if it is correct while
            samples are wrong, its positive advantage reinforces the incumbent
            (dominance) instead of only erasing the wrong modes.

  "pos_only" [Ablation 1]  Mute the negative-gradient arm: advantages are
            clamped at 0 from below, so wrong rollouts contribute nothing —
            pure reinforcement. The pairwise probe showed reinforce-updates
            are benign; if churn collapses under pos_only, the unlikelihood
            arm was the driver.

  "neg_only" [Ablation 2]  Mute the positive-gradient arm: advantages clamped
            at 0 from above — pure unlikelihood on wrong rollouts. Isolates
            the suspected destructive arm (expect margin erosion and elevated
            churn without accuracy gains).

All three post-process the output of ``_generate_and_score_completions`` and
recompute advantages locally as ``r - mean_group(r)`` (Dr.GRPO: no std
scaling), so use with ``loss_type=dr_grpo``, ``scale_rewards=none``.

STATUS: inspection sketch — compiles, mirrors the TRL-1.7 API patterns already
validated in src/complement_grpo.py (tokenize/generate/splice), but has NOT
had a live smoke run yet. Smoke with 3 steps before a real run.

Wiring (script/grpo.py):
    trainer_cls = type("T", (FastEvalGRPOTrainer, GroupControlGRPOTrainer), {})
    trainer = trainer_cls(..., group_control="skip",
                          correctness_fn=replay_correctness)
"""
from __future__ import annotations

import torch

from trl import GRPOTrainer


class GroupControlGRPOTrainer(GRPOTrainer):

    def __init__(self, *args, group_control: str | None = None,
                 correctness_fn=None, format_fn=None, **kwargs):
        if group_control not in (None, "skip", "resample", "greedy",
                                 "pos_only", "neg_only"):
            raise ValueError(f"unknown group_control={group_control}")
        self.group_control = group_control
        # In mixin compositions ComplementGRPOTrainer sets self.correctness_fn
        # before this __init__ runs — don't clobber it with our None default.
        self.correctness_fn = correctness_fn or getattr(self, "correctness_fn", None)
        self.format_fn = format_fn                # optional, adds 0.5 like format_reward
        super().__init__(*args, **kwargs)
        # validate the RESOLVED attribute — in mixin compositions the kwarg is
        # consumed by ComplementGRPOTrainer, which sets self.correctness_fn
        # before this __init__ runs in the super() chain.
        if group_control in ("skip", "resample", "greedy") \
                and self.correctness_fn is None:
            raise ValueError(f"group_control={group_control} requires correctness_fn")

    # ---------------------------------------------------------------- helpers
    def _decode(self, completion_ids):
        return [self.processing_class.decode([t for t in ids if t != self.pad_token_id],
                                             skip_special_tokens=True)
                for ids in completion_ids.tolist()]

    @property
    def pad_token_id(self):
        pid = self.processing_class.pad_token_id
        return pid if pid is not None else self.processing_class.eos_token_id

    def _rewards(self, texts, examples):
        """Correctness (1.0) + optional format (0.5) — the run's reward shape."""
        r = torch.tensor([float(self.correctness_fn(t, e))
                          for t, e in zip(texts, examples)])
        if self.format_fn is not None:
            r += torch.tensor([0.5 * float(self.format_fn(t)) for t in texts])
        return r

    def _dr_grpo_advantages(self, rewards, G):
        grp = rewards.view(-1, G)
        return (grp - grp.mean(dim=1, keepdim=True)).reshape(-1)

    def _generate_extra(self, examples, *, greedy=False):
        """One completion per example via the colocated vLLM engine."""
        prompts = [e["prompt"] for e in examples]
        tokenized, images, _ = self._tokenize_prompts(prompts)
        vg = self.vllm_generation
        old_t = vg.temperature
        if greedy:
            vg.temperature = 0.0
        try:
            _, completion_ids, *_ = vg.generate(
                prompts=tokenized, images=images, num_generations=1)
        finally:
            vg.temperature = old_t
        return completion_ids                      # list[list[int]]

    def _splice(self, output, row, token_ids):
        """Overwrite completion row `row` in the padded batch tensors."""
        ids = list(token_ids)
        eos = self.processing_class.eos_token_id
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(eos)
        width = output["completion_ids"].shape[1]
        ids = ids[:width]
        device = output["completion_ids"].device
        output["completion_ids"][row] = torch.full((width,), self.pad_token_id,
                                                   device=device)
        output["completion_ids"][row, :len(ids)] = torch.tensor(ids, device=device)
        output["completion_mask"][row] = 0
        output["completion_mask"][row, :len(ids)] = 1
        # spliced tokens are off-policy: importance weights would be wrong, so
        # drop cached per-token logps for that row if the trainer kept them
        for k in ("old_per_token_logps", "sampling_per_token_logps"):
            if output.get(k) is not None:
                output[k][row] = 0.0

    # ---------------------------------------------------------------- hook
    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if not (self.model.training and self.group_control):
            return output

        # Sign-mask ablations: pure advantage clamps, no correctness needed.
        if self.group_control in ("pos_only", "neg_only"):
            adv = output["advantages"]
            masked = (adv < 0) if self.group_control == "pos_only" else (adv > 0)
            output["advantages"] = (adv.clamp(min=0)
                                    if self.group_control == "pos_only"
                                    else adv.clamp(max=0))
            self._metrics["train"][f"control/{self.group_control}_masked_frac"].append(
                float(masked.float().mean()))
            return output

        G = self.num_generations
        examples = list(inputs)                         # one per completion row
        texts = self._decode(output["completion_ids"])
        correct = torch.tensor([float(self.correctness_fn(t, e))
                                for t, e in zip(texts, examples)])
        frac = correct.view(-1, G).mean(dim=1)          # per group
        # "greedily solvable" proxy at train time: group contains >=1 correct
        bad = (frac > 0) & (frac < 0.5)                 # majority-wrong, salvageable

        if self.group_control == "skip":
            mask = bad.repeat_interleave(G)
            output["advantages"] = torch.where(
                mask.to(output["advantages"].device),
                torch.zeros_like(output["advantages"]),
                output["advantages"])
            self._metrics["train"]["control/skipped_groups"].append(
                float(bad.float().sum()))
            return output

        if self.group_control == "resample":
            changed = False
            for g in torch.nonzero(bad).flatten().tolist():
                rows = range(g * G, (g + 1) * G)
                example = examples[g * G]
                need = int(G // 2 + 1 - correct[list(rows)].sum())
                extra = self._generate_extra([example] * G)     # G more samples
                good = [ids for ids in extra
                        if self.correctness_fn(
                            self.processing_class.decode(ids, skip_special_tokens=True),
                            example)][:need]
                wrong_rows = [r for r in rows if not correct[r]][:len(good)]
                for r, ids in zip(wrong_rows, good):
                    self._splice(output, r, ids)
                    correct[r] = 1.0
                    changed = True
            if changed:
                texts = self._decode(output["completion_ids"])
                output["advantages"] = self._dr_grpo_advantages(
                    self._rewards(texts, examples), G).to(output["advantages"].device)
            self._metrics["train"]["control/resampled_groups"].append(
                float(bad.float().sum()))
            return output

        if self.group_control == "greedy":
            uniq_examples = examples[::G]
            greedy_ids = self._generate_extra(uniq_examples, greedy=True)
            adv = output["advantages"].view(-1, G)
            for g, ids in enumerate(greedy_ids):
                worst = int(adv[g].argmin())            # replace lowest-advantage slot
                self._splice(output, g * G + worst, ids)
            texts = self._decode(output["completion_ids"])
            output["advantages"] = self._dr_grpo_advantages(
                self._rewards(texts, examples), G).to(output["advantages"].device)
            self._metrics["train"]["control/greedy_injected"].append(
                float(len(greedy_ids)))
            return output

        return output