"""GRPO complemented by shortest-correct self-distillation.

At each GRPO generation step, independently sample queries from the replay
index set, generate ``num_generations`` vLLM responses for them, retain the
shortest correct completion seen so far, and add completion-masked SFT loss:

    loss = GRPO loss + sft_weight * SFT(shortest correct replay rollout)

The replay SFT batch is separate from the batch used by the GRPO objective.
The target cache improves online whenever a shorter correct rollout appears.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from accelerate.utils import gather_object

from trl import GRPOTrainer


CorrectnessFn = Callable[[str, dict[str, Any]], bool]


def _prompt_key(prompt: Any) -> str:
    return json.dumps(prompt, sort_keys=True, ensure_ascii=False)


class ComplementGRPOTrainer(GRPOTrainer):
    """GRPO with an independent shortest-correct replay SFT objective."""

    def __init__(
        self,
        *args,
        correctness_fn: CorrectnessFn,
        sft_replay_indices: Iterable[int] | None,
        sft_weight: float = 0.1,
        replay_sft_batch_size: int = 1,
        **kwargs,
    ):
        if sft_weight < 0:
            raise ValueError("sft_weight must be non-negative")
        if replay_sft_batch_size < 1:
            raise ValueError("replay_sft_batch_size must be positive")
        self.correctness_fn = correctness_fn
        self.sft_replay_indices = tuple(sft_replay_indices or ())
        self.sft_weight = sft_weight
        self.replay_sft_batch_size = replay_sft_batch_size
        self.correct_rollouts: dict[str, tuple[list[int], list[int]]] = {}
        self._current_sft_batch = None
        super().__init__(*args, **kwargs)

        if self.sft_weight and not self.sft_replay_indices:
            raise ValueError("sft_replay_indices is required when sft_weight > 0")
        if self.sft_weight and replay_sft_batch_size > len(self.sft_replay_indices):
            raise ValueError("replay_sft_batch_size exceeds the replay set size")
        if self.sft_weight and not self.use_vllm:
            raise ValueError("complementary rollout collection requires vLLM")

        self._replay_order = list(self.sft_replay_indices)
        self._replay_rng = random.Random(self.args.seed)
        self._replay_rng.shuffle(self._replay_order)
        self._replay_cursor = 0
        temperature = float(getattr(self, "temperature", 1.0))
        if self.sft_weight and self.sft_replay_indices and abs(temperature - 1.0) > 1e-6:
            raise ValueError(
                f"shortest-correct target collection requires temperature=1.0, got {temperature}"
            )

    def _sample_replay_examples(self) -> list[dict[str, Any]]:
        indices = []
        while len(indices) < self.replay_sft_batch_size:
            if self._replay_cursor == len(self._replay_order):
                self._replay_rng.shuffle(self._replay_order)
                self._replay_cursor = 0
            idx = self._replay_order[self._replay_cursor]
            self._replay_cursor += 1
            if idx not in indices:
                indices.append(idx)
        return [self.train_dataset[i] for i in indices]

    def _generate_replay_rollouts(self, examples) -> None:
        """Sample G completions per replay query and update the global cache."""
        global_examples = [
            example
            for example in examples
            for _ in range(self.num_generations)
        ]
        world_size = self.accelerator.num_processes
        if len(global_examples) % world_size:
            raise ValueError(
                "replay_sft_batch_size * num_generations must be divisible "
                "by the number of processes"
            )
        local_size = len(global_examples) // world_size
        start = self.accelerator.process_index * local_size
        local_examples = global_examples[start : start + local_size]
        local_prompts = [copy.deepcopy(example["prompt"]) for example in local_examples]

        prompt_ids, completion_ids, *_ = self.vllm_generation.generate(
            prompts=local_prompts,
            num_generations=self.num_generations,
        )
        candidates = []
        for example, prompt, completion in zip(
            local_examples,
            prompt_ids,
            completion_ids,
            strict=True,
        ):
            tokens = list(completion)
            text = self.processing_class.decode(tokens, skip_special_tokens=True)
            if tokens and self.correctness_fn(text, example):
                if self.eos_token_id is not None and tokens[-1] != self.eos_token_id:
                    tokens.append(self.eos_token_id)
                candidates.append((_prompt_key(example["prompt"]), list(prompt), tokens))

        # Every DDP rank receives the globally shortest candidate among all G
        # generations, so each rank trains against the same cached target.
        for key, prompt, completion in gather_object(candidates):
            current = self.correct_rollouts.get(key)
            if current is None or len(completion) < len(current[1]):
                self.correct_rollouts[key] = (prompt, completion)

    def _build_sft_batch(self, examples, device) -> dict[str, torch.Tensor] | None:
        rows = [
            self.correct_rollouts[key]
            for example in examples
            if (key := _prompt_key(example["prompt"])) in self.correct_rollouts
        ]
        if not rows:
            return None

        prompt_width = max(len(prompt) for prompt, _ in rows)
        completion_width = max(len(completion) for _, completion in rows)
        prompt_ids = torch.full(
            (len(rows), prompt_width),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        prompt_mask = torch.zeros_like(prompt_ids)
        completion_ids = torch.full(
            (len(rows), completion_width),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        completion_mask = torch.zeros_like(completion_ids)
        for i, (prompt, completion) in enumerate(rows):
            prompt_ids[i, -len(prompt) :] = torch.tensor(prompt, device=device)
            prompt_mask[i, -len(prompt) :] = 1
            completion_ids[i, : len(completion)] = torch.tensor(completion, device=device)
            completion_mask[i, : len(completion)] = 1
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.model.training and self.sft_weight and self.sft_replay_indices:
            examples = self._sample_replay_examples()
            self._generate_replay_rollouts(examples)
            self._current_sft_batch = self._build_sft_batch(
                examples,
                output["completion_ids"].device,
            )
            self._metrics["train"]["sft/cache_size"].append(len(self.correct_rollouts))
            self._metrics["train"]["sft/batch_size"].append(
                0 if self._current_sft_batch is None
                else len(self._current_sft_batch["prompt_ids"])
            )
        return output

    def _sft_loss(self, model) -> torch.Tensor | None:
        if self._current_sft_batch is None:
            return None
        prompt_ids = self._current_sft_batch["prompt_ids"]
        prompt_mask = self._current_sft_batch["prompt_mask"]
        target_ids = self._current_sft_batch["completion_ids"]
        target_mask = self._current_sft_batch["completion_mask"]
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, target_mask], dim=1)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = target_ids.size(1) + 1
        logits = model(**model_inputs).logits[:, :-1]
        logits = logits[:, -target_ids.size(1) :]
        token_loss = F.cross_entropy(
            logits.transpose(1, 2).float(),
            target_ids,
            reduction="none",
        )
        return (token_loss * target_mask).sum() / target_mask.sum()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        grpo_loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            **kwargs,
        )
        sft_loss = self._sft_loss(model)
        if sft_loss is None:
            return grpo_loss

        mode = "train" if model.training else "eval"
        self._metrics[mode]["sft/loss"].append(float(sft_loss.detach()))
        normalizer = getattr(self, "current_gradient_accumulation_steps", 1)
        return grpo_loss + self.sft_weight * sft_loss / normalizer
