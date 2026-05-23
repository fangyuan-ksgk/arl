"""Online answer buffer for the velocity-based per-token reward.

Stores, per *query*, the set of accepted (externally-correct) answer
strings along with the number of times each has been observed. Used by
:class:`src.velocity.VelocityRewardComputer` to assign a reference answer
``a`` to incorrect rollouts (which lack one of their own) so that the
decoding-velocity reward is well-defined for every rollout.

Sampling is uniform **over unique answers** (counts are recorded but not
used for sampling weight) — per the algorithm spec.

Capacity bound
--------------
Each per-query bucket is capped at ``capacity_per_query`` entries. On
overflow, the lowest-count entry is evicted (LFU). Counts are kept so
the eviction policy is stable; they could later drive a sampling-weight
toggle if desired.

Multi-process note
------------------
This buffer is process-local. Under DDP / FSDP each rank maintains its
own buffer; they will diverge unless explicitly all-reduced. For first
wiring we accept the divergence (it's a noisy reference signal anyway,
and each rank still sees a self-consistent buffer). Cross-rank
synchronization can be added later via a periodic gather + merge.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np


__all__ = ["OnlineBuffer"]


class OnlineBuffer:
    """Per-query bag of accepted answer strings with appearance counts.

    Examples
    --------
    >>> buf = OnlineBuffer(capacity_per_query=4)
    >>> buf.add("q1", "(8-4)*6")
    >>> buf.add("q1", "6*(8-4)")
    >>> buf.add("q1", "(8-4)*6")    # count -> 2
    >>> buf.has("q1")
    True
    >>> buf.sample("q1", rng=np.random.default_rng(0)) in {"(8-4)*6", "6*(8-4)"}
    True
    """

    def __init__(self, capacity_per_query: int = 256):
        if capacity_per_query < 1:
            raise ValueError("capacity_per_query must be >= 1")
        self.capacity_per_query = capacity_per_query
        # query_key -> Counter[answer_str -> count]
        self._store: Dict[Hashable, Counter] = defaultdict(Counter)

    # ------------------------------------------------------------------ writes
    def add(self, query_key: Hashable, answer: Optional[str], count: int = 1) -> bool:
        """Record an accepted ``answer`` for ``query_key``.

        Returns ``True`` iff a new unique answer was added (vs incrementing
        an existing entry). No-ops on empty / whitespace / ``None`` answers.
        """
        if answer is None:
            return False
        answer = answer.strip()
        if not answer:
            return False
        bucket = self._store[query_key]
        is_new = answer not in bucket
        bucket[answer] += count
        if len(bucket) > self.capacity_per_query:
            # LFU eviction — drop the lowest-count entry. Tie-break is
            # arbitrary (Counter preserves insertion order; min picks the
            # first matching key).
            victim = min(bucket, key=bucket.__getitem__)
            del bucket[victim]
        return is_new

    def seed_from(self, items: Iterable[Tuple[Hashable, str]]) -> int:
        """Pre-populate from an iterable of ``(query_key, answer)`` pairs.

        Useful for SFT-supervised datasets where every query already has
        at least one known-correct answer. Returns the number of unique
        ``(query_key, answer)`` entries inserted.
        """
        n_new = 0
        for item in items:
            if len(item) == 2:
                qk, a = item
                count = 1
            elif len(item) == 3:
                qk, a, count = item
            else:
                raise ValueError(f"seed_from item must be (qk, a) or (qk, a, count), got {item!r}")
            n_new += int(self.add(qk, a, count=count))
        return n_new

    # ------------------------------------------------------------------ reads
    def has(self, query_key: Hashable) -> bool:
        """Whether at least one accepted answer is known for ``query_key``."""
        bucket = self._store.get(query_key)
        return bool(bucket)

    def sample(
        self,
        query_key: Hashable,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[str]:
        """Sample one answer **uniformly over unique entries** (not by count).

        Returns ``None`` if the bucket is empty.
        """
        bucket = self._store.get(query_key)
        if not bucket:
            return None
        if rng is None:
            rng = np.random.default_rng()
        keys = list(bucket.keys())
        return keys[int(rng.integers(0, len(keys)))]

    def keys_for(self, query_key: Hashable) -> List[str]:
        """Snapshot of unique answers for ``query_key`` (order arbitrary)."""
        bucket = self._store.get(query_key)
        return list(bucket.keys()) if bucket else []

    # ------------------------------------------------------------------ stats
    def num_queries(self) -> int:
        return sum(1 for b in self._store.values() if b)

    def __len__(self) -> int:
        """Total number of unique (query_key, answer) entries across all queries."""
        return sum(len(b) for b in self._store.values())

    def stats(self) -> Dict[str, float]:
        """Aggregate counts; convenient for logging."""
        sizes = [len(b) for b in self._store.values() if b]
        return {
            "n_queries":      float(len(sizes)),
            "n_entries":      float(sum(sizes)),
            "median_per_q":   float(np.median(sizes)) if sizes else 0.0,
            "max_per_q":      float(max(sizes)) if sizes else 0.0,
        }
