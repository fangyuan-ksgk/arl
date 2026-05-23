"""Prefix injection at rollout time for TRL GRPO (vLLM colocate mode).

The :class:`PrefixInjector` is a ``rollout_func`` for
:class:`trl.GRPOTrainer` — pass it via ``rollout_func=...`` and it will:

1. Per group (default) or per rollout, decide whether to inject a prefix.
2. Sample an accepted CoT from a :class:`PrefixBuffer`, optionally truncate
   it, and **prepend it to the prompt token IDs** sent to vLLM. vLLM
   conditions normally on ``prompt + prefix`` and generates fresh tokens.
3. After generation, the prefix tokens are folded into ``completion_ids``
   (left of the freshly-generated tokens), so the trainer treats them as
   if the policy produced them. The reported ``logprobs`` for prefix
   tokens come from vLLM's ``prompt_logprobs`` (the current policy's own
   logprobs), making the initial importance ratio exactly 1.

Scope
-----
This implementation targets ``vllm_mode="colocate"`` only — the path used
by the Game-of-24 notebook. The trainer's distributed / server / tensor-
parallel branches are NOT exercised here.

Per-rollout ``max_tokens``
--------------------------
``SamplingParams.max_tokens`` is held **constant** across rollouts
(= ``max_completion_length``) so every rollout gets the same fresh-
generation budget regardless of prefix length. The total per-request
length is ``len(prompt) + len(prefix) + max_completion_length`` — make
sure ``vllm_max_model_length`` is set generously.
"""

from __future__ import annotations

from typing import Callable, Hashable, Optional

import numpy as np


__all__ = ["PrefixInjector"]


class PrefixInjector:
    """Rollout-time prefix injector (colocate-mode TRL rollout_func).

    Parameters
    ----------
    buffer
        :class:`~src.online_buffer.OnlineAnswerBuffer` storing accepted CoT
        strings (one bag per query). Can be the same object as the answer
        buffer if your task's "answer" and "CoT-prefix-candidate" coincide;
        usually you want a separate buffer of full CoT traces.
    query_key_fn
        ``(prompt_str) -> Hashable`` mapping the chat-templated prompt
        string to a buffer key. Must agree with the seed / online-update
        side of the buffer.
    p_inject
        Probability of injecting a prefix into a given group (or rollout
        if ``share_within_group=False``).
    truncate
        ``"none"``    — use the sampled CoT in full.
        ``"uniform"`` — sample a truncation length uniformly in
                        ``[1, len(cot_toks)]``.
    share_within_group
        ``True`` (recommended) — one prefix decision per group of
        ``num_generations`` rollouts. Keeps the GRPO within-group baseline
        comparing apples to apples.
        ``False`` — independent prefix decision per rollout; noisier
        baseline but more diversity per step.
    rng
        ``np.random.Generator``. Default: fresh non-reproducible.
    """

    def __init__(
        self,
        buffer,
        *,
        query_key_fn: Callable[[str], Hashable],
        p_inject: float = 0.5,
        truncate: str = "uniform",
        share_within_group: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        if truncate not in ("none", "uniform"):
            raise ValueError(f"unknown truncate: {truncate!r}")
        if not 0.0 <= p_inject <= 1.0:
            raise ValueError(f"p_inject must be in [0, 1], got {p_inject}")
        self.buffer = buffer
        self.query_key_fn = query_key_fn
        self.p_inject = p_inject
        self.truncate = truncate
        self.share_within_group = share_within_group
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ helpers
    def _sample_prefix_ids(self, qk, tokenizer):
        cot = self.buffer.sample(qk, rng=self.rng)
        if not cot:
            return []
        ids = tokenizer(cot, add_special_tokens=False).input_ids
        if not ids:
            return []
        if self.truncate == "uniform":
            L = int(self.rng.integers(1, len(ids) + 1))
            ids = ids[:L]
        return list(ids)

    # ------------------------------------------------------------------ main
    def __call__(self, prompts, trainer):
        """The rollout_func entry point. ``prompts`` are already
        chat-templated strings (TRL handles the conversion in colocate
        mode before calling us).
        """
        from vllm import SamplingParams, TokensPrompt

        if trainer.args.vllm_mode != "colocate":
            raise NotImplementedError(
                "PrefixInjector currently supports vllm_mode='colocate' only"
            )
        llm = trainer.vllm_generation.llm
        tokenizer = trainer.processing_class
        G = trainer.num_generations

        base_sp_kw = dict(
            n=1,
            temperature=trainer.temperature,
            top_p=trainer.top_p,
            top_k=trainer.top_k if trainer.top_k is not None else -1,
            min_p=0.0 if trainer.min_p is None else trainer.min_p,
            repetition_penalty=trainer.repetition_penalty,
            max_tokens=trainer.max_completion_length,
            logprobs=0,           # logprob of the sampled token
            prompt_logprobs=0,    # logprob of each prompt token under the policy
        )

        # Build per-rollout (token_prompt, sampling_params).
        flat_token_prompts: list = []
        flat_sps: list = []
        flat_prefix_lens: list = []
        flat_bare_lens: list = []

        for prompt_str in prompts:
            p_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
            p_ids = list(p_ids)
            qk = self.query_key_fn(prompt_str)

            if self.share_within_group:
                inject = (self.rng.random() < self.p_inject) and self.buffer.has(qk)
                shared = self._sample_prefix_ids(qk, tokenizer) if inject else []
                prefix_choices = [shared] * G
            else:
                prefix_choices = [
                    self._sample_prefix_ids(qk, tokenizer)
                    if (self.rng.random() < self.p_inject and self.buffer.has(qk))
                    else []
                    for _ in range(G)
                ]

            for prefix in prefix_choices:
                flat_token_prompts.append(TokensPrompt(prompt_token_ids=p_ids + prefix))
                flat_sps.append(SamplingParams(**base_sp_kw))
                flat_prefix_lens.append(len(prefix))
                flat_bare_lens.append(len(p_ids))

        # vLLM generate. Accepts list of SamplingParams aligned with prompts.
        outputs = llm.generate(flat_token_prompts, sampling_params=flat_sps, use_tqdm=False)

        prompt_ids_out: list = []
        completion_ids_out: list = []
        logprobs_out: list = []

        for out, plen, blen in zip(outputs, flat_prefix_lens, flat_bare_lens):
            gen = out.outputs[0]
            gen_ids = list(gen.token_ids)
            full_prompt = list(out.prompt_token_ids)
            bare_ids = full_prompt[:blen]
            prefix_ids = full_prompt[blen:blen + plen]
            completion = prefix_ids + gen_ids

            # logprobs for prefix tokens: from vLLM's prompt_logprobs (first
            # entry is None — the BOS / first prompt token under-defined).
            # Take only the prefix slice.
            prefix_lps: list = []
            if plen > 0 and out.prompt_logprobs is not None:
                prompt_lp_seq = out.prompt_logprobs[blen:blen + plen]
                for entry in prompt_lp_seq:
                    if entry is None:
                        prefix_lps.append(0.0)
                    else:
                        # Pick the entry for the actual token id at this position.
                        # vLLM returns dict {token_id: Logprob}; we want the one
                        # corresponding to the prompt's own token. Since
                        # prompt_logprobs=0 only returns the sampled (= actual)
                        # token, the dict has one entry — take it.
                        prefix_lps.append(float(next(iter(entry.values())).logprob))
            else:
                prefix_lps = [0.0] * plen

            gen_lps: list = []
            if gen.logprobs is not None:
                for tok_lps in gen.logprobs:
                    if tok_lps:
                        gen_lps.append(float(next(iter(tok_lps.values())).logprob))
                    else:
                        gen_lps.append(0.0)
            if len(gen_lps) != len(gen_ids):
                gen_lps = [0.0] * len(gen_ids)

            prompt_ids_out.append(bare_ids)
            completion_ids_out.append(completion)
            logprobs_out.append(prefix_lps + gen_lps)

        return {
            "prompt_ids": prompt_ids_out,
            "completion_ids": completion_ids_out,
            "logprobs": logprobs_out,
        }
