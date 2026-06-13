#!/usr/bin/env python3
"""
Train Game-of-24 with **TreeTrainer** (Optimistic Prefix Advantage) and,
optionally, **Tree Sampling** — the breadth-first block sampler from
``tree_sample.ipynb`` — wired into TRL's vLLM rollout path.

Trainers
--------
``--trainer tree``  (default)
    ``src.tree_trainer.TreeTrainer``: vanilla GRPO pipeline, but the scalar
    group advantage ``a_i = (r_i - mean)/(std+eps)`` is re-credited per token
    via a prefix trie — every token inherits ``A*(prefix) = max a_j`` over the
    rollouts still reachable from it. ``--use-global-tree`` keeps the trie
    persistent across batches.

``--trainer grpo``  (baseline)
    Plain ``trl.GRPOTrainer`` — flat per-rollout advantage. Use this to A/B the
    OPA credit assignment.

Tree sampling
-------------
``--tree-sampling`` overrides the rollout generator so each prompt is expanded
breadth-first: every active leaf spawns ``--tree-branch`` continuations of
``--tree-block-size`` tokens, identical continuations collapse, and leaves that
emit EOS are carried forward (never re-expanded). After ``--tree-steps`` steps
the leaves are reconciled to exactly ``num_generations`` completions/prompt so
GRPO's group structure is preserved. Backend-agnostic: it drives the **colocate
vLLM engine** (``self.vllm_generation.llm.generate`` with token-id prompts) or,
in **server** mode, the vLLM HTTP client (``self.vllm_generation.vllm_client``,
text prompts) — no second model, no HF ``generate``. Works with either
``--vllm-mode colocate`` or ``--vllm-mode server``.

Examples
--------
::

    # OPA + tree sampling (the full TRIPO setup)
    python script/tripo_game24.py --tree-sampling

    # OPA, standard vLLM rollouts
    python script/tripo_game24.py

    # GRPO baseline
    python script/tripo_game24.py --trainer grpo

    # persistent global prefix trie
    python script/tripo_game24.py --use-global-tree
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


# ---------------------------------------------------------------------------
# Tree-sampling mixin: override TRL's outer `_generate` to do breadth-first
# block sampling on the colocate vLLM engine.
# ---------------------------------------------------------------------------
class TreeSamplingMixin:
    """Drop-in override of ``GRPOTrainer._generate`` implementing breadth-first
    block (tree) sampling on the **colocate vLLM engine**.

    Configure via attributes set on the instance after construction:
        ``tree_branch``      continuations spawned per active leaf per step.
        ``tree_steps``       number of breadth-first expansion steps.
        ``tree_block_size``  max new tokens per step (default: max_completion //
                             tree_steps).

    Returns the exact 7-tuple TRL's ``_generate_and_score_completions`` expects
    ``(prompt_ids, completion_ids, tool_mask, completions,
       total_completion_tokens, logprobs, extra_fields)`` so the downstream
    reward / advantage / loss pipeline (including OPA) is untouched. We return
    ``logprobs=None`` so TRL recomputes per-token logprobs from the live policy
    (correct gradients; tree sampling only shapes *which* sequences exist).
    """

    tree_branch: int = 2
    tree_steps: int = 3
    tree_block_size: Optional[int] = None

    # ------------------------------------------------------------------
    def _tree_sampling_params(self, max_tokens: int):
        from vllm import SamplingParams

        a = self.args
        kwargs = dict(
            n=self.tree_branch,
            repetition_penalty=getattr(a, "repetition_penalty", 1.0),
            temperature=getattr(a, "temperature", 1.0),
            top_p=getattr(a, "top_p", 1.0),
            top_k=getattr(a, "top_k", 0),
            min_p=getattr(a, "min_p", None) or 0.0,
            max_tokens=max_tokens,
        )
        return SamplingParams(**kwargs)

    @staticmethod
    def _as_token_prompts(prefixes: Sequence[Sequence[int]]):
        """Wrap token-id lists for `LLM.generate`, tolerant of vLLM versions."""
        try:
            from vllm import TokensPrompt
        except Exception:  # pragma: no cover - older/newer layout
            from vllm.inputs import TokensPrompt  # type: ignore
        return [TokensPrompt(prompt_token_ids=list(p)) for p in prefixes]

    def _tree_server_sampling_kwargs(self, max_tokens: int) -> dict:
        """SamplingParams (as kwargs) for the vLLM HTTP server client."""
        a = self.args
        return dict(
            n=self.tree_branch,
            repetition_penalty=getattr(a, "repetition_penalty", 1.0),
            temperature=getattr(a, "temperature", 1.0),
            top_p=getattr(a, "top_p", 1.0),
            top_k=getattr(a, "top_k", 0),
            min_p=getattr(a, "min_p", None) or 0.0,
            max_tokens=max_tokens,
            logprobs=0,
        )

    # ------------------------------------------------------------------
    # Backend-agnostic block expansion: generate `tree_branch` continuations
    # (<= block_size tokens) for each token-id prefix. Works against either the
    # in-process colocate vLLM engine OR the vLLM HTTP server client.
    # ------------------------------------------------------------------
    def _expand_prefixes(self, prefixes, block_size):
        """Return, per prefix, a list of ``(continuation_token_ids, done)``."""
        if getattr(self, "vllm_mode", None) == "server":
            return self._expand_prefixes_server(prefixes, block_size)
        return self._expand_prefixes_colocate(prefixes, block_size)

    def _expand_prefixes_colocate(self, prefixes, block_size):
        llm = self.vllm_generation.llm
        sp = self._tree_sampling_params(block_size)
        outs = llm.generate(self._as_token_prompts(prefixes),
                            sampling_params=sp, use_tqdm=False)
        # finish_reason == "length" => hit block cap (not done);
        # "stop"/"abort"/None-with-eos => terminated.
        return [[(list(o.token_ids), o.finish_reason != "length")
                 for o in out.outputs]
                for out in outs]

    def _expand_prefixes_server(self, prefixes, block_size):
        # The TRL server's /generate/ endpoint accepts TEXT prompts only, so we
        # decode each token-id prefix (special tokens kept) and re-send. Leaves
        # are extended with our own tracked ids + the server's completion ids,
        # so the stored token sequence stays self-consistent.
        client = self.vllm_generation.vllm_client
        tok = self.processing_class
        texts = [tok.decode(p, skip_special_tokens=False) for p in prefixes]
        out = client.generate(prompts=texts,
                              **self._tree_server_sampling_kwargs(block_size))
        comp = out["completion_ids"]            # prompt-major, `n` contiguous each
        n = self.tree_branch
        eos = tok.eos_token_id
        results = []
        for i in range(len(prefixes)):
            per = []
            for j in range(n):
                cont = list(comp[i * n + j])
                # No finish_reason over HTTP: a continuation that stopped before
                # the block cap (or ends in EOS) is treated as terminated.
                done = (len(cont) < block_size) or (bool(cont) and cont[-1] == eos)
                per.append((cont, done))
            results.append(per)
        return results

    def _tree_sample_one(self, prompt_token_ids: List[int], block_size: int):
        """Breadth-first tree sample from a single tokenized prompt.

        Returns a list of completion token-id lists (EXCLUDING the prompt), one
        per surviving leaf. Mirrors `tree_sample` in tree_sample.ipynb: dedup
        identical continuations, carry EOS-terminated leaves forward unchanged.
        """
        leaves: List[Tuple[List[int], bool]] = [([], False)]  # (continuation, done)
        for _ in range(self.tree_steps):
            nxt: List[Tuple[List[int], bool]] = []
            seen = set()

            # Carry finished leaves forward (don't regenerate past EOS).
            active: List[List[int]] = []
            for leaf, done in leaves:
                if done:
                    key = tuple(leaf)
                    if key not in seen:
                        seen.add(key)
                        nxt.append((leaf, True))
                else:
                    active.append(leaf)

            if active:
                prefixes = [prompt_token_ids + leaf for leaf in active]
                expanded = self._expand_prefixes(prefixes, block_size)
                for leaf, conts in zip(active, expanded):
                    for cont, child_done in conts:
                        child = leaf + cont
                        key = tuple(child)
                        if key not in seen:
                            seen.add(key)
                            nxt.append((child, child_done))

            leaves = nxt
            if all(done for _, done in leaves):
                break

        return [leaf for leaf, _ in leaves] or [[]]

    @staticmethod
    def _reconcile(leaves: List[List[int]], n: int, rng: random.Random):
        """Force exactly ``n`` completions per prompt to keep GRPO's groups.

        Fewer unique leaves than ``n`` -> pad by resampling existing leaves;
        more -> keep the first ``n`` (breadth-first order is roughly diversity
        ordered since identical continuations were deduped)."""
        if not leaves:
            return [[]] * n
        if len(leaves) >= n:
            return leaves[:n]
        pad = [rng.choice(leaves) for _ in range(n - len(leaves))]
        return leaves + pad

    # ------------------------------------------------------------------
    def _generate(self, prompts: list):
        # Greedy eval pass (pass@1): bypass tree sampling entirely and defer to
        # the base trainer's deterministic single-sample generation.
        if getattr(self, "_greedy_pass", False):
            return super()._generate(prompts)
        # Tree sampling needs a vLLM backend; it is backend-agnostic via
        # `_expand_prefixes` (colocate engine handle OR HTTP server client).
        if not getattr(self, "use_vllm", False):
            raise RuntimeError(
                "tree sampling requires vLLM (use_vllm=True; "
                "vllm_mode 'colocate' or 'server')."
            )

        import torch

        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        num_gen = self.num_generations if mode == "train" else self.num_generations_eval

        # Match TRL: refresh vLLM weights when the optimizer step advanced.
        if self.state.global_step != self._last_loaded_step:
            self.vllm_generation.sync_weights()
            self._last_loaded_step = self.state.global_step

        block_size = self.tree_block_size or max(1, self.max_completion_length // self.tree_steps)
        rng = random.Random(self.state.global_step * 1_000_003 + (0 if mode == "train" else 7))

        # The sampler repeats each unique prompt `num_gen` times consecutively
        # (TRL slices `prompts[::num_generations]` for the unique set too).
        unique_prompts = prompts[::num_gen]
        tok = self.processing_class

        prompt_ids: List[List[int]] = []
        completion_ids: List[List[int]] = []
        for u in unique_prompts:
            templated = tok.apply_chat_template(
                u, tools=getattr(self, "tools", None),
                add_generation_prompt=True, tokenize=True,
                **getattr(self, "chat_template_kwargs", {}),
            )
            leaves = self._tree_sample_one(templated, block_size)
            leaves = self._reconcile(leaves, num_gen, rng)
            for comp in leaves:
                # Defensive cap: never exceed the configured completion budget.
                comp = comp[: self.max_completion_length]
                prompt_ids.append(list(templated))
                completion_ids.append(comp)

        # Decode completions in TRL's expected shape.
        contents = tok.batch_decode(completion_ids, skip_special_tokens=True)
        completions = [[{"role": "assistant", "content": c}] for c in contents]

        completion_lengths = torch.tensor(
            [len(c) for c in completion_ids], device=device
        )
        total_completion_tokens = self.accelerator.gather(completion_lengths).sum()

        # Lightweight metrics (defaultdict(list) keys; safe if absent elsewhere).
        if completion_lengths.numel():
            self._metrics[mode]["completions/mean_length"].append(
                completion_lengths.float().mean().item()
            )
            self._metrics[mode]["completions/max_length"].append(
                float(completion_lengths.max().item())
            )

        tool_mask = None
        logprobs = None      # -> TRL recomputes sampling logprobs from the policy
        extra_fields: dict = {}
        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
            None,   # images      (TRL 1.5.1 expects 9-tuple)
            None,   # tool_images
        )


# ---------------------------------------------------------------------------
# Fast dual-pass eval: sampled (T=eval) pass@K + greedy (T=0) pass@1.
# Ported from script/grpo_gsm8k.py (FastEvalGRPOTrainer), adapted to the
# Game-of-24 rollout-logger signature (keyed on `numbers`).
# ---------------------------------------------------------------------------
class FastEvalMixin:
    """Override the eval path so each eval call generates twice per prompt:

      1. **sampled pass@K** — T = eval temperature, ``num_generations_eval``
         rollouts/prompt (drives the offline ``pass@k`` estimator, incl.
         t=1 pass@8). Routed through the full reward pipeline so every reward
         fn (and the rollout logger) sees it.
      2. **greedy pass@1** — T = 0, one deterministic rollout per *unique*
         prompt. Logged directly via the rollout logger.

    Skips the local loss forward entirely (eval cost is vLLM generation, not the
    loss). When ``eval_dataset`` is a dict of validation splits, ``evaluate``
    stamps the split name onto the rollout logger so each rollout is tagged.
    """

    @contextlib.contextmanager
    def _greedy_eval(self):
        """Force greedy decoding (T=0, 1 generation/prompt) within the block."""
        old_neval = getattr(self, "num_generations_eval", self.num_generations)
        self.num_generations_eval = 1
        self._greedy_pass = True               # tree sampler falls back to base
        vg = getattr(self, "vllm_generation", None)
        gc = getattr(self, "generation_config", None)
        old_vg_t = getattr(vg, "temperature", None) if vg is not None else None
        old_gc = (gc.temperature, gc.do_sample) if gc is not None else None
        old_self_t = getattr(self, "temperature", None)
        # TRL builds the vLLM SamplingParams from self.temperature (the server
        # VLLMClient path reads THIS); vg.temperature only reaches the colocate
        # engine and vLLM ignores do_sample. Set all three so greedy (T=0) really
        # applies in BOTH colocate and server mode (else server "greedy" samples
        # at T=1 -> long rambling completions, wrong greedy@1, slower eval).
        self.temperature = 0.0
        if vg is not None:
            vg.temperature = 0.0
        if gc is not None:
            gc.do_sample = False
            gc.temperature = 0.0
        try:
            yield
        finally:
            self.num_generations_eval = old_neval
            self._greedy_pass = False
            if old_self_t is not None:
                self.temperature = old_self_t
            if vg is not None:
                vg.temperature = old_vg_t
            if gc is not None:
                gc.temperature, gc.do_sample = old_gc

    def _current_temperature(self):
        vg = getattr(self, "vllm_generation", None)
        if vg is not None and getattr(vg, "temperature", None) is not None:
            return float(vg.temperature)
        gc = getattr(self, "generation_config", None)
        if gc is not None and getattr(gc, "temperature", None) is not None:
            return float(gc.temperature)
        return getattr(getattr(self, "args", None), "temperature", None)

    def _eval_generate(self, prompts, inputs, *, decoding):
        logger = getattr(self, "_rollout_logger", None)
        if logger is not None:
            logger.decoding = decoding
            logger.temperature = self._current_temperature()
        _t = time.time()
        result = self._generate(prompts)
        completion_ids_list = result[1]
        completions = result[3]
        dt = time.time() - _t
        if decoding == "greedy":
            if logger is not None:
                logger(completions=completions,
                       numbers=[x["numbers"] for x in inputs])
        else:
            self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        return dt, len(completions)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        import torch
        prompts = [x["prompt"] for x in inputs]

        # Route every rollout here to the EVAL log. `prediction_step` runs only
        # during evaluation, so this is the robust signal (the EvalFlagCallback's
        # on_prediction_step fires *after* this method, one batch too late).
        logger = getattr(self, "_rollout_logger", None)
        if logger is not None:
            logger.in_eval = True
            logger.global_step = self.state.global_step

        # Pass 1: sampled pass@K (T=eval, num_generations_eval rollouts/prompt).
        dt, n_gen = self._eval_generate(prompts, inputs, decoding="sample")

        # Pass 2: greedy pass@1 (T=0, one rollout per unique prompt). The eval
        # sampler repeats each prompt G times; slice back to uniques.
        G = max(1, getattr(self, "num_generations_eval", self.num_generations))
        uniq_inputs = inputs[::G]
        uniq_prompts = prompts[::G]
        with self._greedy_eval():
            dt_g, _ = self._eval_generate(uniq_prompts, uniq_inputs, decoding="greedy")

        split = getattr(getattr(self, "_rollout_logger", None), "eval_dataset_name", "eval")
        print(f"[eval:{split}] {n_gen} sample + {len(uniq_prompts)} greedy gens "
              f"in {dt:.1f}s+{dt_g:.1f}s", flush=True)

        loss = torch.zeros((), device=self.accelerator.device)
        return loss, None, None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # When eval_dataset is a dict, transformers recurses with
        # metric_key_prefix=f"eval_{name}"; capture that to tag the split.
        logger = getattr(self, "_rollout_logger", None)
        if logger is not None:
            name = metric_key_prefix
            if name.startswith("eval_"):
                name = name[len("eval_"):]
            logger.eval_dataset_name = name or "eval"
        return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                metric_key_prefix=metric_key_prefix)


def build_trainer_class(base, tree_sampling: bool):
    """Compose the trainer: FastEvalMixin (dual-pass eval) always on top, then
    optional TreeSamplingMixin, then the base (TreeTrainer or GRPOTrainer)."""
    mixins = (FastEvalMixin,)
    if tree_sampling:
        mixins = mixins + (TreeSamplingMixin,)
    return type("TriPOTrainer", mixins + (base,), {})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from src.tree_trainer import TreeTrainer
    from src.game24utils import (
        build_puzzle_pool, split_train_eval, make_dataset,
        correctness_reward, format_reward, RolloutLogger,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rollout_log = out_dir / "rollouts.jsonl"
    eval_rollout_log = out_dir / "eval_rollout.jsonl"
    train_rollout_log.write_text("")
    eval_rollout_log.write_text("")

    # Reproducible puzzle split.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    puzzles = build_puzzle_pool(max_n=args.max_n)
    train_puzzles, eval_puzzles = split_train_eval(
        puzzles, eval_frac=args.eval_frac, eval_min=args.eval_min,
        rng=random.Random(0xE7A15),   # FIXED split seed -> IDENTICAL eval set across all run
                                       # seeds (was: global rng seeded by args.seed -> eval set
                                       # differed per seed). Training data ORDER still varies by
                                       # args.seed (set above), so seed variation is preserved.
    )
    train_ds, eval_ds = make_dataset(train_puzzles), make_dataset(eval_puzzles)
    if getattr(args, "no_think", False):
        def _nt(ex):
            pp=[dict(m) for m in ex["prompt"]]; pp[-1]["content"]=pp[-1]["content"]+" /no_think"; return {"prompt": pp}
        train_ds=train_ds.map(_nt); eval_ds=eval_ds.map(_nt)
        print("[no_think] appended /no_think to all prompts", flush=True)
    print(f"[data] total={len(puzzles)} train={len(train_puzzles)} "
          f"eval={len(eval_puzzles)}", flush=True)

    # Single in-distribution validation split. Passing a dict lets the
    # FastEvalMixin tag every rollout with the split name.
    eval_datasets = {"eval": eval_ds}

    # Eval group size for the sampled pass@K (defaults to --num-generations).
    num_generations_eval = args.num_generations_eval or args.num_generations

    # GRPO requires the train *global* batch to be a multiple of num_generations.
    train_global = args.per_device_batch_size * args.grad_accum
    if train_global % args.num_generations != 0:
        raise ValueError(
            f"train global batch (pdbs*grad_accum = {args.per_device_batch_size}*"
            f"{args.grad_accum} = {train_global}) must be a multiple of "
            f"--num-generations ({args.num_generations})."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rollout_logger = RolloutLogger(train_rollout_log, eval_rollout_log, tokenizer)

    class EvalFlagCallback(TrainerCallback):
        """Routes RolloutLogger writes to the train/eval JSONL around eval loops."""
        def __init__(self, logger):
            self.logger = logger

        def on_prediction_step(self, a, state, control, **kw):
            self.logger.in_eval = True
            self.logger.global_step = state.global_step

        def on_step_begin(self, a, state, control, **kw):
            self.logger.in_eval = False
            self.logger.global_step = state.global_step

        def on_evaluate(self, a, state, control, **kw):
            self.logger.in_eval = False

    # ------- Config ------------------------------------------------------
    config_kwargs = dict(
        output_dir=str(out_dir / "trl"),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_batch_size,
        num_generations_eval=num_generations_eval,
        per_device_eval_batch_size=num_generations_eval,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        temperature=args.temperature,
        bf16=torch.cuda.is_available(),
        save_strategy="no",
        report_to="none",
        seed=args.seed,   # reshuffles data order per seed (lottery runs)
    )
    if not args.no_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = args.vllm_mode
        if args.vllm_mode == "colocate":
            config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
        else:
            config_kwargs["vllm_server_host"] = args.vllm_server_host
            config_kwargs["vllm_server_port"] = args.vllm_server_port

    if args.eval_steps > 0:
        config_kwargs["eval_strategy"] = "steps"
        config_kwargs["eval_steps"] = args.eval_steps
        config_kwargs["eval_on_start"] = True

    config_kwargs["vllm_importance_sampling_correction"] = False
    config_kwargs["beta"] = args.beta            # KL coef vs reference; >0 anchors policy (anti-collapse)
    config_kwargs["scale_rewards"] = args.scale_rewards   # group(default,÷std) | none(Dr.GRPO) | batch
    # ---- 4B memory knobs (all opt-in; defaults = unchanged behavior) --------------------------
    # bf16(=AMP) above only casts COMPUTE; weights/grads/optimizer stay fp32 unless we load bf16.
    # For 4B full-FT in 80GB (server mode): --model-dtype bfloat16 (weights 16->8GB, Adam states too)
    # + --optim paged_adamw_8bit (Adam states 32->8GB). gradient-checkpointing OFF by default (it
    # recomputes activations in backward -> ~20-30%% slower; rarely needed once bf16 + 8-bit Adam).
    if getattr(args, "model_dtype", None):
        config_kwargs["model_init_kwargs"] = {"dtype": args.model_dtype}
    config_kwargs["optim"] = args.optim
    if getattr(args, "gradient_checkpointing", False):
        config_kwargs["gradient_checkpointing"] = True
        config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    config = GRPOConfig(**config_kwargs)

    if args.tree_sampling and args.no_vllm:
        raise ValueError("--tree-sampling requires vLLM (drop --no_vllm; "
                         "either --vllm-mode colocate or server works).")

    # ------- Trainer selection ------------------------------------------
    base_cls = TreeTrainer if args.trainer == "tree" else GRPOTrainer
    trainer_cls = build_trainer_class(base_cls, args.tree_sampling)

    reward_funcs = ([format_reward] if args.no_correctness_reward
                    else [correctness_reward, format_reward])
    predictive_velo_obj = None
    if args.predictive_velocity_reward:
        from src.mbe_reward import PredictiveVeloReward
        predictive_velo_obj = PredictiveVeloReward(
            tokenizer,
            scale=args.predictive_velocity_scale,
            clip=args.predictive_velocity_clip,
            norm_mode=args.predictive_norm_mode,
            answer_source=args.predictive_answer_source,
        )
        reward_funcs.append(predictive_velo_obj)
        print(f"[reward] predictive velocity enabled: scale={args.predictive_velocity_scale}, "
              f"clip=±{args.predictive_velocity_clip}, norm_mode='{args.predictive_norm_mode}', "
              f"answer_source='{args.predictive_answer_source}'",
              flush=True)
    if args.no_correctness_reward:
        print("[reward] correctness reward DISABLED for training "
              "(eval accuracy still logged by RolloutLogger)", flush=True)
    reward_funcs.append(rollout_logger)

    trainer_kwargs = dict(
        model=args.model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=train_ds,
        eval_dataset=(eval_datasets if args.eval_steps > 0 else None),
        processing_class=tokenizer,
        callbacks=[EvalFlagCallback(rollout_logger)],
    )
    if args.trainer == "tree":
        trainer_kwargs["use_global_tree"] = args.use_global_tree
        trainer_kwargs["credit_mode"] = args.credit_mode
        if args.virtual_rollout != "none":
            trainer_kwargs["virtual_rollout"] = args.virtual_rollout
            trainer_kwargs["virtual_max_reward"] = args.virtual_max_reward
        if args.shaped_reward:
            trainer_kwargs["shaped_reward"] = True
            trainer_kwargs["shaped_kwargs"] = dict(
                pos_scale=args.shaped_pos_scale,
                neg_scale=args.shaped_neg_scale,
            )
        # Buffer tricks (src/tree_trainer.py; all default OFF -> vanilla GRPO
        # advantages). Compose by ordering: inject/resample first (tokens +
        # rewards + LOCAL group z-score), buffered baseline last.
        trainer_kwargs["buffered_baseline"] = args.buffered_baseline
        trainer_kwargs["inject_rollout"] = args.inject_rollout
        trainer_kwargs["inject_incorrect"] = args.inject_incorrect
        trainer_kwargs["resample_prefix"] = args.resample_prefix
        trainer_kwargs["resample_train_prefix"] = args.resample_train_prefix
        trainer_kwargs["resample_inject"] = args.resample_inject
        if args.tree_persist_path:
            trainer_kwargs["tree_persist_path"] = args.tree_persist_path

    trainer = trainer_cls(**trainer_kwargs)

    # Direct handle for the fast-eval greedy pass (records pass@1 rollouts).
    trainer._rollout_logger = rollout_logger

    if predictive_velo_obj is not None:
        predictive_velo_obj.set_model(trainer.model)

    # Wire tree-sampling knobs onto the live trainer (read at generation time).
    if args.tree_sampling:
        trainer.tree_branch = args.tree_branch
        trainer.tree_steps = args.tree_steps
        trainer.tree_block_size = args.tree_block_size

    print(f"[trainer] {base_cls.__name__}"
          f"{' + TreeSampling' if args.tree_sampling else ''}"
          f"{' (global trie)' if (args.trainer == 'tree' and args.use_global_tree) else ''}"
          f"{f' credit={args.credit_mode}' if args.trainer == 'tree' else ''}"
          f"{' shaped' if (args.trainer == 'tree' and args.shaped_reward) else ''}"
          + (''.join(f' +{f}' for f in ('buffered_baseline', 'inject_rollout',
                                        'inject_incorrect', 'resample_prefix',
                                        'resample_train_prefix', 'resample_inject')
                     if args.trainer == 'tree' and getattr(args, f))) +
          f" | num_generations={args.num_generations}"
          f" eval(sample@{num_generations_eval} t={args.temperature} + greedy@1 t=0)"
          f" splits={list(eval_datasets) if args.eval_steps > 0 else []}", flush=True)
    if args.tree_sampling:
        bs = args.tree_block_size or max(1, args.max_completion_length // args.tree_steps)
        print(f"[tree-sampling] branch={args.tree_branch} steps={args.tree_steps} "
              f"block_size={bs} (<= {args.tree_branch ** args.tree_steps} leaves/prompt "
              f"reconciled to {args.num_generations})", flush=True)

    t0 = time.time()
    if args.trainer == "tree" and args.absorb_steps > 0:
        trainer.absorb_buffer(args.absorb_steps, args.absorb_groups_per_query,
                              n_pos=args.absorb_n_pos, n_neg=args.absorb_n_neg)
    trainer.train()
    if args.trainer == "tree":
        trainer.save_tries()   # persist trie even with save_strategy='no'
    print(f"[done] training in {time.time() - t0:.0f}s | "
          f"rollouts -> {train_rollout_log}\n"
          f"          eval -> {eval_rollout_log}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="output/tripo_game24")
    p.add_argument("--trainer", choices=["tree", "grpo"], default="tree",
                   help="'tree' = TreeTrainer (OPA per-token credit); "
                        "'grpo' = vanilla GRPO baseline.")
    # Tree-credit (OPA) options
    p.add_argument("--use-global-tree", action="store_true",
                   help="(tree trainer) persist the prefix trie across batches.")
    p.add_argument("--credit-mode", choices=["base", "max", "min"], default="base",
                   help="(tree trainer) per-prefix advantage backup: 'base' = "
                        "vanilla GRPO trajectory-level advantage (default, no "
                        "redistribution); 'max' = Optimistic Prefix Advantage "
                        "(best reachable continuation); 'min' = pessimistic "
                        "(worst reachable).")
    # Tree sampling options
    p.add_argument("--tree-sampling", action="store_true",
                   help="Breadth-first block sampling on the colocate vLLM engine.")
    p.add_argument("--tree-branch", type=int, default=2,
                   help="Continuations spawned per active leaf per step.")
    p.add_argument("--tree-steps", type=int, default=3,
                   help="Number of breadth-first expansion steps.")
    p.add_argument("--tree-block-size", type=int, default=None,
                   help="Max new tokens per step (default: max_completion//tree_steps).")
    # Training / generation
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--num-generations-eval", type=int, default=None,
                   help="Rollouts/prompt for the sampled (t=1) eval pass@K "
                        "(default: --num-generations, i.e. pass@8). The greedy "
                        "pass@1 always uses 1 deterministic rollout/prompt.")
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL coefficient vs reference policy. 0=TRL default (no anchor, unstable on "
                        "format-reward); >0 (e.g. 0.04) anchors the policy and prevents collapse.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--no-think", action="store_true", help="append /no_think to prompts (Qwen3)")
    p.add_argument("--scale-rewards", choices=["group","none","batch"], default="group",
                   help="advantage scaling: group=÷group-std (TRL default, 1/std blow-up); "
                        "none=Dr.GRPO/no std-normalize; batch=÷batch-std")
    # Predictive velocity reward (src/mbe_reward.py:PredictiveVeloReward).
    # Game24 uses the same '####' marker convention as GSM8K, so defaults apply.
    p.add_argument("--predictive-velocity-reward", action="store_true",
                   help="Add length-normalised predictive velocity reward: "
                        "clip((log p(a|q,o) − log p(a|q)) / denom, ±clip) / scale")
    p.add_argument("--predictive-velocity-scale", type=float, default=4.0)
    p.add_argument("--predictive-velocity-clip", type=float, default=1.0)
    p.add_argument("--predictive-norm-mode", type=str, default="log_total",
                   choices=["log_total", "cot_len"])
    p.add_argument("--predictive-answer-source", type=str, default="rollout",
                   choices=["rollout", "gold"],
                   help="v1 'rollout' = score the model's own answer a; "
                        "v2 'gold' = score the GT answer a* (first `solutions` entry) — "
                        "works before the model ever finds a correct answer.")
    p.add_argument("--no-correctness-reward", action="store_true",
                   help="Drop the correctness reward from training (e.g. to replace it with "
                        "--predictive-velocity-reward). Eval accuracy is unaffected: "
                        "RolloutLogger computes correctness independently.")
    # Buffer tricks (src/tree_trainer.py; tree trainer only, all default off).
    # 1.  --buffered-baseline : advantages rescaled by the BUFFERED reward std
    #     (stable across batches), re-centered to zero-sum per group.
    # 2.  --inject-rollout    : degenerate group (all_wrong / format_only /
    #     reward_hacking) -> swap ONE slot for a buffered CORRECT rollout
    #     (leaf-recorded reward), local group z-score recomputed.
    # 2.1 --inject-incorrect  : additionally break ALL-CORRECT groups with a
    #     buffered WRONG rollout (requires --inject-rollout).
    # 3.  --resample-prefix   : REPLACE each all-correct group with fresh
    #     continuations of an under-explored buffered prefix (same step).
    # 3.1 --resample-train-prefix : train the forced prefix tokens too
    #     (default: prefix is attended context only, no gradient via tool_mask).
    # 3b  --resample-inject   : reserve one slot of the resampled group for the
    #     buffered correct rollout THROUGH that prefix (guaranteed contrast).
    p.add_argument("--buffered-baseline", action="store_true",
                   help="(tree) zero-sum group advantages with the buffered-std denominator.")
    p.add_argument("--inject-rollout", action="store_true",
                   help="(tree) repair degenerate groups with a buffered correct rollout.")
    p.add_argument("--inject-incorrect", action="store_true",
                   help="(tree) also break all-correct groups with a buffered wrong rollout.")
    p.add_argument("--resample-prefix", action="store_true",
                   help="(tree) same-step resample of all-correct groups from an "
                        "under-explored buffered prefix.")
    p.add_argument("--resample-train-prefix", action="store_true",
                   help="(tree) give the forced prefix tokens gradient (default: context only).")
    p.add_argument("--resample-inject", action="store_true",
                   help="(tree) reserve one resampled slot for the buffered correct rollout "
                        "through the sampled prefix.")
    p.add_argument("--tree-persist-path", type=str, default=None,
                   help="(tree) JSON path to persist the global tries across runs.")
    p.add_argument("--absorb-steps", type=int, default=0,
                   help="(tree) >0: before train(), absorb ALL buffered healthy groups "
                        "(>=1 correct + >=1 incorrect rollout per prompt trie) in exactly "
                        "this many gradient updates (accumulation derived). Requires a "
                        "pre-grown --tree-persist-path buffer from the seed-lottery runs.")
    p.add_argument("--absorb-groups-per-query", type=int, default=1,
                   help="(tree) healthy groups stitched per healthy query for the absorb phase.")
    p.add_argument("--absorb-n-pos", type=int, default=1,
                   help="(tree) guaranteed correct rollouts per stitched group "
                        "(rest of the group is uniform over all buffered leaves).")
    p.add_argument("--absorb-n-neg", type=int, default=1,
                   help="(tree) guaranteed incorrect rollouts per stitched group. "
                        "Requires n_pos + n_neg <= num-generations.")
    # Confident-failure / rare-success advantage shaping (src/arsenal.py; tree trainer only)
    p.add_argument("--shaped-reward", action="store_true",
                   help="Replace the scalar GRPO advantage with the confident-failure/rare-success "
                        "shaped reward (src/arsenal.py) BEFORE the OPA trie backup; off = TRL default adv.")
    p.add_argument("--shaped-pos-scale", type=float, default=1.0,
                   help="success-term scale for --shaped-reward (rewards rare/hard wins more).")
    p.add_argument("--shaped-neg-scale", type=float, default=1.0,
                   help="failure-term scale for --shaped-reward (penalizes confident failures more).")
    # No-gradient "virtual rollout" reward insertion to revive dead GRPO groups
    # (src/arsenal.py:virtual_rollout_advantages; tree trainer only). Patches the
    # reward->advantage step: appends one virtual reward per group before the z-score.
    p.add_argument("--virtual-rollout",
                   choices=["none", "insert_max", "insert_min", "insert_max_min",
                            "insert_max_all_incorrect", "insert_max_mixed"],
                   default="none",
                   help="insert_max=append a MAX-reward virtual rollout to every group; "
                        "insert_min=append MIN (0.0) only to all-correct groups; "
                        "insert_max_min=append MIN reward when group is all-correct else MAX; "
                        "insert_max_all_incorrect=append MAX only to all-incorrect groups; "
                        "insert_max_mixed=append MAX only to mixed groups; "
                        "none=off (default).")
    p.add_argument("--virtual-max-reward", type=float, default=1.2,
                   help="reward value of the MAX virtual rollout (default 1.2 = correctness 1.0 + format 0.2).")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--optim", default="adamw_torch",
                   help="HF optimizer name. 'paged_adamw_8bit' cuts Adam states ~4x "
                        "(4B: 32GB->8GB) so full-FT 4B fits in 80GB server mode.")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="recompute activations in backward (saves memory, ~20-30%% slower). "
                        "OFF by default; usually unneeded once weights are bf16 + 8-bit Adam.")
    p.add_argument("--model-dtype", default=None,
                   help="model load dtype, e.g. 'bfloat16' (weights bf16 -> halves model/grad/Adam "
                        "memory; needed for 4B). Default None = HF default (fp32 weights + bf16 AMP).")
    # Data
    p.add_argument("--max-n", type=int, default=13,
                   help="Puzzle numbers drawn from {1..max_n} (13 = full deck; "
                        "~1362 solvable, enough for the 400 validation floor).")
    p.add_argument("--eval-frac", type=float, default=0.40,
                   help="Validation fraction; actual size = max(eval_frac*N, eval_min).")
    p.add_argument("--eval-min", type=int, default=400,
                   help="Validation floor: at least this many puzzles (capped at N).")
    p.add_argument("--eval-steps", type=int, default=50,
                   help="Eval every N steps (0 to disable).")
    # vLLM
    p.add_argument("--no_vllm", action="store_true",
                   help="Disable vLLM (TRL native generation). Tree sampling unsupported.")
    p.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--vllm-server-host", default="0.0.0.0")
    p.add_argument("--vllm-server-port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main()