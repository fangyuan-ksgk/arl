"""Interleaved GRPO experiments.

Original requests:
  1. With every GRPO objective, add a forgettable query's cached-rollout
     policy gradient (composite objective).
  2. With every GRPO group, add queries from the forgettable-query set.
  3. With every GRPO group, add forgettable queries whose representations are
     closest to the current in-group query.

Insight: compression -> interference
  * Similar representations requiring different continuations cause bad churn.
  * Similar representations with similar continuations enable generalization.

Experimental motivation:
  * XOR training shows that similar inputs with different targets cause churn.
  * Interleaving conflicting samples is critical for avoiding interference.
  * GRPO/SFT may forget because representation-similar queries are not
    continually interleaved during training.

Current status: intervention (2) is implemented below as compute-matched replay.
Each generation batch replaces a fixed fraction of unique prompts with known
forgettable prompts while preserving complete ``num_generations`` groups.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Sized

from trl import GRPOTrainer


def load_forgettable_indices(path: str | Path) -> list[int]:
    """Load dataset indices or infer them from per-query correctness histories.

    Accepted formats:
      * JSON list of integer indices.
      * JSON object with ``indices`` or ``forgettable_indices``.
      * JSONL records containing ``idx`` plus either an explicit
        ``forgettable``/``class`` label or a time-ordered ``correct`` value.

    A correctness trajectory is forgettable iff it is correct at some point
    and incorrect at a later point.
    """
    path = Path(path)
    text = path.read_text().strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else [payload]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    if records and all(isinstance(item, int) for item in records):
        return sorted(set(records))

    if len(records) == 1 and isinstance(records[0], dict):
        container = records[0]
        for key in ("forgettable_indices", "indices"):
            if key in container:
                return sorted({int(i) for i in container[key]})
        if isinstance(container.get("records"), list):
            records = container["records"]

    explicit: set[int] = set()
    histories: dict[int, list[tuple[tuple[int, int], bool]]] = defaultdict(list)
    for order, record in enumerate(records):
        if not isinstance(record, dict) or "idx" not in record:
            continue
        idx = int(record["idx"])
        label = str(record.get("class", "")).upper()
        if record.get("forgettable") is True or label in {"REGRESSED", "TRANSIENT"}:
            explicit.add(idx)
        if "correct" in record:
            step = int(record.get("global_step", record.get("step", order)))
            histories[idx].append(((step, order), bool(record["correct"])))

    for idx, history in histories.items():
        seen_correct = False
        for _, correct in sorted(history):
            if correct:
                seen_correct = True
            elif seen_correct:
                explicit.add(idx)
                break
    return sorted(explicit)

# Given precomputed replay indices, ensure every GRPO generation batch
# contains a fixed fraction of replay prompt groups.


class InterleavedSampler:
    """RepeatSampler-compatible sampler with replay in every unique-prompt batch."""

    def __init__(
        self,
        data_source: Sized,
        replay_indices: Iterable[int],
        *,
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int = 1,
        replay_fraction: float = 0.25,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not 0 < replay_fraction < 1:
            raise ValueError("replay_fraction must be between 0 and 1")
        if batch_size < 2:
            raise ValueError("interleaving requires at least two unique prompts per generation batch")

        self.data_source = data_source
        self.num_samples = len(data_source)
        self.replay_indices = sorted({int(i) for i in replay_indices})
        invalid = [i for i in self.replay_indices if not 0 <= i < self.num_samples]
        if invalid:
            raise IndexError(f"replay indices outside dataset: {invalid[:5]}")
        if not self.replay_indices:
            raise ValueError("replay_indices is empty")

        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.replay_count = max(1, min(batch_size - 1, round(batch_size * replay_fraction)))
        self.regular_count = batch_size - self.replay_count
        replay_set = set(self.replay_indices)
        self.regular_indices = [i for i in range(self.num_samples) if i not in replay_set]
        if len(self.regular_indices) < self.regular_count:
            raise ValueError("not enough non-replay examples to form one generation batch")
        if len(self.replay_indices) < self.replay_count:
            raise ValueError("not enough replay examples to form one generation batch")

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    @property
    def num_batches(self) -> int:
        # Match the baseline RepeatSampler's epoch length exactly.
        return self.num_samples // self.batch_size

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        def cycling(pool: list[int]) -> Iterator[int]:
            while True:
                values = pool.copy()
                if self.shuffle:
                    rng.shuffle(values)
                yield from values

        def take_unique(stream: Iterator[int], count: int) -> list[int]:
            values: list[int] = []
            seen: set[int] = set()
            while len(values) < count:
                value = next(stream)
                if value not in seen:
                    seen.add(value)
                    values.append(value)
            return values

        regular = cycling(self.regular_indices)
        replay = cycling(self.replay_indices)
        for _ in range(self.num_batches):
            batch = take_unique(regular, self.regular_count)
            batch += take_unique(replay, self.replay_count)
            if self.shuffle:
                rng.shuffle(batch)
            for _ in range(self.repeat_count):
                for index in batch:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (
            self.num_batches
            * self.batch_size
            * self.mini_repeat_count
            * self.repeat_count
        )


class InterleavedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer using :class:`InterleavedSampler` for training only."""

    def __init__(
        self,
        *args,
        replay_indices: Iterable[int] | None = None,
        replay_fraction: float = 0.25,
        **kwargs,
    ):
        self.replay_indices = tuple(replay_indices or ())
        self.replay_fraction = replay_fraction
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, dataset=None):
        if not self.replay_indices:
            return super()._get_train_sampler(dataset)

        dataset = self.train_dataset if dataset is None else dataset
        unique_batch_size = self.args.generation_batch_size // self.num_generations
        return InterleavedSampler(
            data_source=dataset,
            replay_indices=self.replay_indices,
            mini_repeat_count=self.num_generations,
            batch_size=unique_batch_size,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            replay_fraction=self.replay_fraction,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )
