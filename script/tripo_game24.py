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
GRPO's group structure is preserved. This reuses TRL's **colocate vLLM engine**
(``self.vllm_generation.llm``) — no second model, no HF ``generate``.
Requires ``--vllm-mode colocate``.

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

    def _tree_sample_one(self, prompt_token_ids: List[int], block_size: int):
        """Breadth-first tree sample from a single tokenized prompt.

        Returns a list of completion token-id lists (EXCLUDING the prompt), one
        per surviving leaf. Mirrors `tree_sample` in tree_sample.ipynb: dedup
        identical continuations, carry EOS-terminated leaves forward unchanged.
        """
        llm = self.vllm_generation.llm
        sp = self._tree_sampling_params(block_size)

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
                outs = llm.generate(
                    self._as_token_prompts(prefixes),
                    sampling_params=sp,
                    use_tqdm=False,
                )
                for leaf, out in zip(active, outs):
                    for o in out.outputs:
                        cont = list(o.token_ids)
                        # finish_reason == "length" => hit block cap, not done;
                        # "stop"/"abort"/None-with-eos => terminated.
                        child_done = o.finish_reason != "length"
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
        # Tree sampling is implemented on the colocate vLLM engine only.
        if not (getattr(self, "use_vllm", False)
                and getattr(self, "vllm_mode", None) == "colocate"):
            raise RuntimeError(
                "tree sampling requires vLLM colocate mode "
                "(use_vllm=True, vllm_mode='colocate')."
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
        )


def build_trainer_class(base, tree_sampling: bool):
    if not tree_sampling:
        return base
    return type("TreeSamplingTrainer", (TreeSamplingMixin, base), {})


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
        build_puzzle_pool, bucket_by_difficulty, make_splits, build_datasets,
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
    easy, medium, hard = bucket_by_difficulty(puzzles, easy_min=8, hard_max=2)
    train_puzzles, eval_puzzles, hard_probe = make_splits(
        easy, medium, hard, eval_frac=args.eval_frac, probe_frac=0.40,
    )
    train_ds, eval_ds, _probe_ds = build_datasets(train_puzzles, eval_puzzles, hard_probe)
    print(f"[data] train={len(train_puzzles)} eval={len(eval_puzzles)} "
          f"probe={len(hard_probe)}", flush=True)

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
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        temperature=args.temperature,
        bf16=torch.cuda.is_available(),
        save_strategy="no",
        report_to="none",
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

    config = GRPOConfig(**config_kwargs)

    if args.tree_sampling and (args.no_vllm or args.vllm_mode != "colocate"):
        raise ValueError("--tree-sampling requires vLLM colocate mode "
                         "(drop --no_vllm and use --vllm-mode colocate).")

    # ------- Trainer selection ------------------------------------------
    base_cls = TreeTrainer if args.trainer == "tree" else GRPOTrainer
    trainer_cls = build_trainer_class(base_cls, args.tree_sampling)

    trainer_kwargs = dict(
        model=args.model,
        reward_funcs=[correctness_reward, format_reward, rollout_logger],
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds if args.eval_steps > 0 else None,
        processing_class=tokenizer,
        callbacks=[EvalFlagCallback(rollout_logger)],
    )
    if args.trainer == "tree":
        trainer_kwargs["use_global_tree"] = args.use_global_tree

    trainer = trainer_cls(**trainer_kwargs)

    # Wire tree-sampling knobs onto the live trainer (read at generation time).
    if args.tree_sampling:
        trainer.tree_branch = args.tree_branch
        trainer.tree_steps = args.tree_steps
        trainer.tree_block_size = args.tree_block_size

    print(f"[trainer] {base_cls.__name__}"
          f"{' + TreeSampling' if args.tree_sampling else ''}"
          f"{' (global trie)' if (args.trainer == 'tree' and args.use_global_tree) else ''}"
          f" | num_generations={args.num_generations}", flush=True)
    if args.tree_sampling:
        bs = args.tree_block_size or max(1, args.max_completion_length // args.tree_steps)
        print(f"[tree-sampling] branch={args.tree_branch} steps={args.tree_steps} "
              f"block_size={bs} (<= {args.tree_branch ** args.tree_steps} leaves/prompt "
              f"reconciled to {args.num_generations})", flush=True)

    t0 = time.time()
    trainer.train()
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
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    # Data
    p.add_argument("--max-n", type=int, default=9,
                   help="Puzzle numbers drawn from {1..max_n}.")
    p.add_argument("--eval-frac", type=float, default=0.10)
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
