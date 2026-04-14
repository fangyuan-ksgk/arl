"""Prefix-Conditioned Rollout Exploration (PCRE) for ARL.

Motivation
----------
Standard GRPO penalises every token of an incorrect rollout, including
prefixes that may be perfectly reasonable—they just needed a better suffix.
Conversely, it only reinforces the exact winning trace, making the model
brittle to alternative reasoning paths.

Idea: randomly truncate previous rollouts (correct, incorrect, or both) and
have the model complete from the truncated prefix.  This forces the model to
find the correct answer regardless of which reasoning state it starts from,
discouraging single-pattern memorisation.

Algorithm
---------
1. A `PrefixRolloutCollector` (zero-reward reward fn) intercepts every batch
   of completions during GRPO training and stores them in a
   `PrefixRolloutBuffer`.
2. A `PrefixAugmentedDataset` wraps the base dataset.  On each `__getitem__`,
   with probability `augment_prob`, it samples a rollout from the buffer,
   slices it to a random prefix, and returns a dataset item whose `prompt`
   is a pre-formatted raw string ending mid-assistant-turn.
3. Because the prompt is a raw string (not a list-of-dicts), TRL's
   GRPOTrainer takes its non-conversational tokenisation path and will NOT
   re-apply the chat template—the model simply continues generating from the
   prefix position.

Tokenisation note
-----------------
`tokenizer.apply_chat_template(..., continue_final_message=True)` is used to
produce prompts that end without `<|im_end|>`, so generation naturally
continues the partial assistant turn.  This requires transformers >= 4.44.
"""

import random
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class PrefixRolloutBuffer:
    """
    Circular FIFO buffer of (question, completion_text, is_correct) triples.

    Args:
        max_size: Maximum number of rollouts to retain.
        min_prefix_frac: Minimum fraction of completion length to keep as prefix.
        max_prefix_frac: Maximum fraction of completion length to keep as prefix.
    """

    def __init__(
        self,
        max_size: int = 500,
        min_prefix_frac: float = 0.15,
        max_prefix_frac: float = 0.75,
    ):
        self.max_size = max_size
        self.min_prefix_frac = min_prefix_frac
        self.max_prefix_frac = max_prefix_frac
        self._buf: list[tuple[str, str, bool]] = []  # (question, completion, is_correct)

    def __len__(self) -> int:
        return len(self._buf)

    # ------------------------------------------------------------------
    def add(self, question: str, completion_text: str, is_correct: bool) -> None:
        """Add a single rollout (FIFO eviction when full)."""
        if len(self._buf) >= self.max_size:
            self._buf.pop(0)
        self._buf.append((question, completion_text, is_correct))

    def add_batch(
        self,
        prompts: list,
        completions: list,
        correctness: list[bool],
    ) -> None:
        """
        Add a batch of rollouts.

        Args:
            prompts:     List of prompts as message-dicts or strings.
            completions: List of completion message-dicts (TRL format:
                         [[{"role": "assistant", "content": "..."}], ...]).
            correctness: List of booleans, one per rollout.
        """
        for prompt, comp, ok in zip(prompts, completions, correctness):
            question = _extract_question(prompt)
            comp_text = _extract_completion_text(comp)
            if question and comp_text:
                self.add(question, comp_text, bool(ok))

    # ------------------------------------------------------------------
    def sample_prefix_augmented_item(
        self,
        base_item: dict,
        tokenizer,
        from_correct: Optional[bool] = None,
    ) -> Optional[dict]:
        """
        Sample a rollout from the buffer, truncate to a random prefix,
        and return a modified dataset item ready for TRL.

        The returned item's `prompt` is a **raw string** (pre-formatted with
        the partial assistant turn) rather than a list of message dicts.
        TRL will tokenise it without re-applying the chat template.

        Args:
            base_item:    Original dataset item (must contain 'gold_answer').
            tokenizer:    HuggingFace tokenizer with chat-template support.
            from_correct: True  → only sample correct rollouts.
                          False → only sample incorrect rollouts.
                          None  → sample from all rollouts.

        Returns:
            Modified item dict, or None if no suitable rollout found.
        """
        if not self._buf:
            return None

        candidates = self._buf
        if from_correct is True:
            candidates = [r for r in self._buf if r[2]]
        elif from_correct is False:
            candidates = [r for r in self._buf if not r[2]]

        if not candidates:
            return None

        question, completion_text, _ = random.choice(candidates)

        # Random character-level truncation
        L = len(completion_text)
        if L < 20:
            return None
        lo = max(1, int(L * self.min_prefix_frac))
        hi = max(lo + 1, int(L * self.max_prefix_frac))
        cut = random.randint(lo, hi)
        prefix_text = completion_text[:cut]

        # Build the pre-formatted prompt string.
        # `continue_final_message=True` omits the closing <|im_end|> so
        # the model continues from the end of prefix_text.
        try:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": prefix_text},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        except Exception:
            return None

        return {**base_item, "prompt": prompt_str}

    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        n_correct = sum(1 for _, _, ok in self._buf if ok)
        return {
            "buffer_size": len(self._buf),
            "n_correct": n_correct,
            "n_incorrect": len(self._buf) - n_correct,
        }


# ---------------------------------------------------------------------------
# Collector (side-effect reward function)
# ---------------------------------------------------------------------------

class PrefixRolloutCollector:
    """
    A **zero-reward** reward function that fills a PrefixRolloutBuffer as
    a side-effect of GRPO's reward evaluation phase.

    Drop it into GRPOTrainer's `reward_funcs` list after the correctness
    reward so it can piggyback on the correctness labels::

        buf = PrefixRolloutBuffer()
        collector = PrefixRolloutCollector(buf, correctness_fn=correctness_reward)
        trainer = GRPOTrainer(
            reward_funcs=[correctness_reward, format_reward, collector],
            ...
        )

    The collector itself contributes zero reward to the training signal.
    """

    def __init__(
        self,
        buffer: PrefixRolloutBuffer,
        correctness_fn=None,
    ):
        self.__name__ = "prefix_rollout_collector"
        self.buffer = buffer
        self.correctness_fn = correctness_fn

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        if self.correctness_fn is not None:
            try:
                correct_flags = self.correctness_fn(
                    prompts=prompts, completions=completions, **kwargs
                )
            except TypeError:
                correct_flags = self.correctness_fn(completions, **kwargs)
            correctness = [float(r) >= 0.5 for r in correct_flags]
        else:
            correctness = [False] * len(completions)

        self.buffer.add_batch(prompts, completions, correctness)
        return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class PrefixAugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps a base HuggingFace dataset and, for some fraction of examples,
    replaces the prompt with a prefix-augmented version sampled from a
    PrefixRolloutBuffer.

    Args:
        base_dataset:  Original HF dataset (must have 'prompt' & 'gold_answer').
        buffer:        PrefixRolloutBuffer to sample prefixes from.
        tokenizer:     Tokenizer used to format prefix-injected prompts.
        augment_prob:  Fraction of examples to augment (default 0.3).
        from_correct:  True → prefixes from correct rollouts only;
                       False → incorrect only;
                       None  → all rollouts (default).
        min_buffer:    Minimum buffer size before augmentation starts.
    """

    def __init__(
        self,
        base_dataset,
        buffer: PrefixRolloutBuffer,
        tokenizer,
        augment_prob: float = 0.3,
        from_correct: Optional[bool] = None,
        min_buffer: int = 16,
    ):
        self.base = base_dataset
        self.buffer = buffer
        self.tokenizer = tokenizer
        self.augment_prob = augment_prob
        self.from_correct = from_correct
        self.min_buffer = min_buffer

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base[idx])
        if (
            random.random() < self.augment_prob
            and len(self.buffer) >= self.min_buffer
        ):
            augmented = self.buffer.sample_prefix_augmented_item(
                item, self.tokenizer, from_correct=self.from_correct
            )
            if augmented is not None:
                return augmented
        return item

    # TRL compatibility: expose column_names / features from base dataset
    @property
    def column_names(self):
        return list(self.base.column_names) if hasattr(self.base, "column_names") else None

    @property
    def features(self):
        return self.base.features if hasattr(self.base, "features") else None

    def __repr__(self) -> str:
        stats = self.buffer.get_stats()
        return (
            f"PrefixAugmentedDataset(n={len(self)}, augment_prob={self.augment_prob}, "
            f"from_correct={self.from_correct}, buffer={stats})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_question(prompt) -> str:
    """Extract the question string from a prompt (message-list or string)."""
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    if isinstance(prompt, str):
        return prompt
    return ""


def _extract_completion_text(completion) -> str:
    """Extract raw text from a TRL completion (list-of-one-dict or string)."""
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return first.get("content", "")
    if isinstance(completion, str):
        return completion
    return ""
