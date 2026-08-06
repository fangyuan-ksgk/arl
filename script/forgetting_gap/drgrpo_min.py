"""Minimal single-file Dr.GRPO baseline + forgetting-gap harness (GSM8K,
Qwen3-0.6B). This is the reference run every OPD script in this directory is
measured against, and the source of the teacher checkpoints they distill from.

Rebuilt (Jul 25) from script/grpo.py (rewards / prompt / Dr.GRPO defaults) with
the OPD-family batch geometry of script/opd/attnrtg_min.py so the baseline is
apples-to-apples with the OPD runs: lr 5e-6, bs 4 x ga 16 (=64 rollouts/step),
G=8, 1024 completion tokens, T=1.0, loss_type=dr_grpo, scale_rewards=none,
beta=0.01, 300 steps. Only difference vs attnrtg_min: no advantage shaping.

Expected (2 x A100, 4 seeds): ~70.5 final greedy accuracy, ~440-500 mean CoT
tokens, per-run forgetting gap (union over ckpts - final) ~20 pt, cross-seed
lottery gap (union over seeds - mean seed) ~20 pt. The two gaps coincide: the
lottery across seeds IS the temporal churn inside one run (claude.md MILESTONE).

  train     one seed, ckpt every --save_steps (20)
  eval      greedy vLLM pass over every ckpt: per-query correctness jsonl
            (+ acc and CoT length per ckpt)
  annotate  same, on train rows -> train_forgetting_annotations.jsonl
            (train_idx -> correct_at{step: bool}), the teacher table the OPD
            scripts consume
  report    aggregate seeds: per-seed acc/CoT-length, forgetting gap,
            cross-seed lottery gap

    # one seed per GPU
    CUDA_VISIBLE_DEVICES=0 python script/opd/drgrpo_min.py train \
        --out output/forgetting_drgrpo/seed0 --seed 0
    CUDA_VISIBLE_DEVICES=0 python script/opd/drgrpo_min.py eval \
        --out output/forgetting_drgrpo/seed0
    CUDA_VISIBLE_DEVICES=0 python script/opd/drgrpo_min.py annotate \
        --out output/forgetting_drgrpo/seed0 --train_subset 2000
    python script/opd/drgrpo_min.py report \
        --runs output/forgetting_drgrpo/seed0,output/forgetting_drgrpo/seed1
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import domains  # noqa: E402  gsm8k | math | mbpp
import peftcfg  # noqa: E402  LoRA (the >=8B path)

INSTR = ("Solve the problem step by step inside <think>...</think>. "
         "After </think>, give a brief final explanation and end your "
         "response with a line of the exact form:\n#### <number>\n"
         "where <number> is the final numeric answer with no units, no "
         "commas, and no extra text.")


def extract(text):
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"-?[\d,]*\.?\d+", text)
    return m[-1].replace(",", "") if m else None


def equal(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def prompt_of(question):
    return [{"role": "user", "content": f"{question}\n\n{INSTR} /think"}]


def _text(c):
    return c[0]["content"] if isinstance(c, list) else c


def _rows(completions, kw):
    """Rebuild per-example dataset rows from TRL's reward-fn kwargs.

    TRL forwards every dataset column as a list AND extra objects such as
    trainer_state, so filter to list-likes of the right length."""
    n = len(completions)
    cols = {k: v for k, v in kw.items()
            if isinstance(v, (list, tuple)) and len(v) == n}
    return ([dict(zip(cols, vals)) for vals in zip(*cols.values())]
            if cols else [{} for _ in range(n)])


# --------------------------------------------------------------------- train
def train(a):
    from trl import GRPOConfig, GRPOTrainer

    D = domains.get(a.dataset)
    ds, _ = D["load"]()

    if a.teachable_from:
        # DATA-MATCHED CONTROL. The OPD scripts train only on prompts some
        # teacher checkpoint solves (opd_min.make_dataset filters
        # best_ckpt > 0). That fraction tracks domain difficulty:
        # GSM8K 94.5% (1890/2000), MATH 60.9% (609/1000), MBPP 36.6%
        # (320/874). So on MBPP an unrestricted baseline trains on 2.7x more
        # prompts than the OPD arm — and on the harder ones the teacher never
        # solved. Comparing the two measures the data filter as much as the
        # loss. This flag restricts the baseline to the same prompt set.
        import opd_min
        opd_min.DATASET[0] = a.dataset
        steps = opd_min.ckpt_steps_of(a.teachable_from)
        table = opd_min.build_teacher_table(
            Path(a.teachable_from) / "train_forgetting_annotations.jsonl",
            steps, pick="best")
        keep = {i for i, v in table.items() if v}
        n0 = len(ds)
        ds = ds.filter(lambda ex, i: i in keep, with_indices=True)
        print(f"[drgrpo] teachable-subset control: {len(ds)} of {n0} prompts "
              f"({100 * len(ds) / n0:.1f}%) from {a.teachable_from}", flush=True)

    def correctness(completions, **kw):
        return [float(D["correct"](_text(c), r))
                for c, r in zip(completions, _rows(completions, kw))]

    def format_reward(completions, **kw):
        return [0.5 * bool(re.search(D["format_re"], _text(c)))
                for c in completions]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    roll_path = out / "train_rollouts.jsonl"
    roll_path.write_text("")

    class LenLogger:
        """reward-fn shim (returns 0.0): per-rollout CoT length trace."""
        __name__ = "len_logger"

        def __init__(self):
            self.tok = None
            self.calls = 0

        def __call__(self, completions, **kw):
            if self.tok is None:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(a.model)
            rows = _rows(completions, kw)
            with roll_path.open("a") as f:
                for c, r_ in zip(completions, rows):
                    t = _text(c)
                    e = D["answer_end"](t)
                    cot = t[:e] if e is not None else t
                    f.write(json.dumps({
                        "call": self.calls,
                        "correct": bool(D["correct"](t, r_)),
                        "has_marker": e is not None,
                        "n_tokens": len(self.tok.encode(
                            t, add_special_tokens=False)),
                        "n_cot_tokens": len(self.tok.encode(
                            cot, add_special_tokens=False))}) + "\n")
            self.calls += 1
            return [0.0] * len(completions)

    cfg = GRPOConfig(
        output_dir=a.out, seed=a.seed, max_steps=a.max_steps,
        learning_rate=a.lr, per_device_train_batch_size=a.per_device_bs,
        gradient_accumulation_steps=a.ga, num_generations=a.num_generations,
        gradient_checkpointing=bool(a.grad_ckpt), optim=a.optim,
        max_completion_length=(a.max_completion_length
                               or D["max_completion_length"]),
        temperature=1.0,
        loss_type="dr_grpo", scale_rewards="none", beta=a.beta,
        use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.gpu_util,
        model_init_kwargs=({"dtype": a.model_dtype}
                           if a.model_dtype != "auto" else None),
        # cap the engine context to prompt+completion+margin: the model's
        # native 40k window demands ~5.6GB of KV cache, which does not fit
        # beside a 4B policy + probe on one card (claude.md's truncation
        # warning does not apply — 2048 > ~150 prompt + 1024 completion)
        vllm_max_model_length=a.vllm_max_len,
        save_strategy="steps", save_steps=a.save_steps,
        save_only_model=True, save_total_limit=None,
        logging_steps=10, report_to=[], bf16=True)
    trainer = GRPOTrainer(
        model=a.model, reward_funcs=[correctness, format_reward, LenLogger()],
        args=cfg, train_dataset=ds, peft_config=peftcfg.lora_config(a))
    trainer.train()
    trainer.save_model(a.out)
    print("Training complete (drgrpo_min)", flush=True)


# ---------------------------------------------------------------------- eval
def _ckpts(out, include_base, base_model):
    dirs = sorted(Path(out).glob("checkpoint-*"),
                  key=lambda p: int(p.name.split("-")[1]))
    steps = [(0, base_model)] if include_base else []
    return steps + [(int(d.name.split("-")[1]), str(d)) for d in dirs]


def _greedy_pass(a, rows, tag):
    """Greedy vLLM decode of `rows` [(idx, question, gold)] for every ckpt.

    Writes <out>/<tag>_step{S}.jsonl per ckpt (one row per query) and
    <out>/<tag>_summary.json (acc / mean total tokens / mean CoT tokens).
    Skips ckpts whose jsonl already exists so it can be resumed.
    """
    import gc
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(a.out)
    ckpts = _ckpts(out, a.include_base, a.model)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {out}")
    summary = {}
    sfile = out / f"{tag}_summary.json"
    if sfile.exists():
        summary = json.loads(sfile.read_text())
    for step, path in ckpts:
        rec = out / f"{tag}_step{step}.jsonl"
        if rec.exists() and str(step) in summary and not a.overwrite:
            print(f"[{tag}] step {step}: cached "
                  f"acc {summary[str(step)]['acc']:.4f}", flush=True)
            continue
        tok = AutoTokenizer.from_pretrained(path)
        prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                           add_generation_prompt=True)
                   for r in rows]
        llm = LLM(model=path, dtype="bfloat16",
                  gpu_memory_utilization=a.gpu_util,
                  disable_log_stats=True, enforce_eager=False)
        outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                    max_tokens=a.max_tokens))
        oks, ntok, ncot = [], [], []
        with rec.open("w") as f:
            for r, o in zip(rows, outs):
                idx, gold = r["idx"], r["row"].get("gold", "")
                text = o.outputs[0].text
                D = domains.get(a.dataset)
                pred = ""
                ok = bool(D["correct"](text, r["row"]))
                e = D["answer_end"](text)
                cot = text[:e] if e is not None else text
                m = e is not None
                n_c = len(tok.encode(cot, add_special_tokens=False))
                oks.append(ok)
                ntok.append(len(o.outputs[0].token_ids))
                ncot.append(n_c)
                f.write(json.dumps({"idx": int(idx), "step": int(step),
                                    "n_markers": (0 if e is None else 1),
                                    "correct": ok, "predicted": pred,
                                    "gold": gold,
                                    "n_tokens": len(o.outputs[0].token_ids),
                                    "n_cot_tokens": n_c,
                                    "has_marker": bool(m)}) + "\n")
        summary[str(step)] = dict(acc=float(np.mean(oks)),
                                  n=len(oks),
                                  mean_tokens=float(np.mean(ntok)),
                                  mean_cot_tokens=float(np.mean(ncot)))
        sfile.write_text(json.dumps(summary, indent=1))
        print(f"[{tag}] step {step}: acc {np.mean(oks):.4f}  "
              f"len {np.mean(ntok):.0f}  cot {np.mean(ncot):.0f}", flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return summary


def evaluate(a):
    import numpy as np
    from datasets import load_dataset

    _, test = domains.get(a.dataset)["load"]()
    if a.eval_samples:
        test = test.select(range(min(a.eval_samples, len(test))))
    rows = [{"idx": i, "prompt": test[i]["prompt"], "row": test[i]}
            for i in range(len(test))]
    summary = _greedy_pass(a, rows, "test")
    steps = sorted(int(s) for s in summary)
    M = np.array([[json.loads(l)["correct"] for l in
                   (Path(a.out) / f"test_step{s}.jsonl").open()]
                  for s in steps])
    final, union = M[-1].mean(), M.any(0).mean()
    print(f"\nsteps {steps}")
    print(f"final {100 * final:.1f} | best "
          f"{100 * max(summary[str(s)]['acc'] for s in steps):.1f} | "
          f"union-over-ckpts {100 * union:.1f} | FORGETTING GAP "
          f"{100 * (union - final):.1f} | final cot "
          f"{summary[str(steps[-1])]['mean_cot_tokens']:.0f} tok")


def annotate(a):
    """Per-ckpt greedy correctness on TRAIN rows -> teacher table for OPD."""
    from datasets import load_dataset

    train_ds, _ = domains.get(a.dataset)["load"]()
    n = min(a.train_subset, len(train_ds)) if a.train_subset else len(train_ds)
    rows = [{"idx": i, "prompt": train_ds[i]["prompt"], "row": train_ds[i]}
            for i in range(n)]
    summary = _greedy_pass(a, rows, "train")
    steps = sorted(int(s) for s in summary if int(s) > 0)
    per_q = {}
    for s in steps:
        for line in (Path(a.out) / f"train_step{s}.jsonl").open():
            r = json.loads(line)
            per_q.setdefault(r["idx"], {})[str(s)] = bool(r["correct"])
    ann = Path(a.out) / "train_forgetting_annotations.jsonl"
    with ann.open("w") as f:
        for idx in sorted(per_q):
            f.write(json.dumps({"train_idx": int(idx),
                                "correct_at": per_q[idx]}) + "\n")
    solved = sum(any(v.values()) for v in per_q.values())
    final = sum(v[str(steps[-1])] for v in per_q.values())
    print(f"\n{ann}: {len(per_q)} train queries | ever-solved "
          f"{100 * solved / len(per_q):.1f} | final "
          f"{100 * final / len(per_q):.1f} | train forgetting gap "
          f"{100 * (solved - final) / len(per_q):.1f}", flush=True)


# -------------------------------------------------------------------- report
def report(a):
    import numpy as np

    runs = [Path(r) for r in a.runs.split(",") if r.strip()]
    finals, rows_out = {}, []
    for run in runs:
        summary = json.loads((run / "test_summary.json").read_text())
        steps = sorted(int(s) for s in summary)
        rec = [[json.loads(l) for l in (run / f"test_step{s}.jsonl").open()]
               for s in steps]
        M = np.array([[r["correct"] for r in step] for step in rec])
        # strict = ####-gated (the lenient extract() falls back to the last
        # number, which hides a format collapse — seed 2 dropped the marker at
        # step ~140 with NO accuracy loss, see the marker column)
        K = np.array([[r["correct"] and r["has_marker"] for r in step]
                      for step in rec])
        mk = np.array([[r["has_marker"] for r in step] for step in rec])
        finals[run.name] = M[-1]
        rows_out.append(dict(
            run=run.name, steps=steps,
            acc=[round(100 * summary[str(s)]["acc"], 2) for s in steps],
            cot=[round(summary[str(s)]["mean_cot_tokens"], 1) for s in steps],
            marker=[round(100 * float(x), 1) for x in mk.mean(1)],
            final=100 * float(M[-1].mean()),
            best=100 * float(max(summary[str(s)]["acc"] for s in steps)),
            mean_over_ckpts=100 * float(M.mean(1).mean()),
            union_ckpts=100 * float(M.any(0).mean()),
            forgetting_gap=100 * float(M.any(0).mean() - M[-1].mean()),
            forgetting_gap_vs_mean=100 * float(M.any(0).mean()
                                               - M.mean(1).mean()),
            strict_final=100 * float(K[-1].mean()),
            marker_final=100 * float(mk[-1].mean()),
            final_cot=summary[str(steps[-1])]["mean_cot_tokens"],
            final_tokens=summary[str(steps[-1])]["mean_tokens"]))
    print(f"{'run':<8} {'final':>6} {'mean_ck':>8} {'union':>6} "
          f"{'un-mean':>8} {'un-fin':>7} {'strict':>7} {'mark%':>6} "
          f"{'cot':>5} {'len':>5}")
    for r in rows_out:
        print(f"{r['run']:<8} {r['final']:>6.2f} {r['mean_over_ckpts']:>8.2f} "
              f"{r['union_ckpts']:>6.2f} {r['forgetting_gap_vs_mean']:>8.2f} "
              f"{r['forgetting_gap']:>7.2f} {r['strict_final']:>7.2f} "
              f"{r['marker_final']:>6.1f} {r['final_cot']:>5.0f} "
              f"{r['final_tokens']:>5.0f}")
    F = np.stack([finals[r["run"]] for r in rows_out])
    mean_seed = float(F.mean(1).mean())
    union_seed = float(F.any(0).mean())
    agg = dict(
        n_seeds=len(rows_out), runs=rows_out,
        mean_final_acc=100 * mean_seed,
        std_final_acc=100 * float(F.mean(1).std()),
        union_over_seeds=100 * union_seed,
        lottery_gap=100 * (union_seed - mean_seed),
        mean_forgetting_gap=float(np.mean([r["forgetting_gap"]
                                           for r in rows_out])),
        mean_forgetting_gap_vs_mean=float(np.mean(
            [r["forgetting_gap_vs_mean"] for r in rows_out])),
        mean_union_ckpts=float(np.mean([r["union_ckpts"] for r in rows_out])),
        mean_final_cot=float(np.mean([r["final_cot"] for r in rows_out])))
    print(f"\navg final acc {agg['mean_final_acc']:.2f} "
          f"+/- {agg['std_final_acc']:.2f} over {agg['n_seeds']} seeds "
          f"| mean CoT {agg['mean_final_cot']:.0f} tok")
    print(f"union over seeds {agg['union_over_seeds']:.2f} "
          f"-> LOTTERY GAP {agg['lottery_gap']:.2f} pt")
    print(f"mean union-over-ckpts {agg['mean_union_ckpts']:.2f} | per-run "
          f"FORGETTING GAP {agg['mean_forgetting_gap_vs_mean']:.2f} pt "
          f"(vs mean-over-ckpts) / {agg['mean_forgetting_gap']:.2f} pt "
          f"(vs final)")
    if a.report_out:
        Path(a.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report_out).write_text(json.dumps(agg, indent=1))
        print(f"-> {a.report_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "annotate", "report"])
    ap.add_argument("--out", default="output/forgetting_drgrpo/seed0")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--per_device_bs", type=int, default=4)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--optim", default="adamw_torch",
                    help="adamw_bnb_8bit shrinks optimizer state 4x — needed "
                         "for 4B full-FT on a single 80GB card")
    ap.add_argument("--dataset", default="gsm8k",
                    choices=["gsm8k", "math", "mbpp"])
    ap.add_argument("--max_completion_length", type=int, default=0,
                    help="0 = the dataset's default (gsm8k 1024, math 2048)")
    ap.add_argument("--model_dtype", default="auto",
                    help="bfloat16 for >=4B: TRL otherwise "
                         "upcasts to fp32, doubling weights "
                         "and grads (32GB at 4B)")
    ap.add_argument("--vllm_max_len", type=int, default=2048)
    ap.add_argument("--grad_ckpt", type=int, default=0,
                    help="1 for >=1.7B: trades ~30% speed for activation memory")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--save_steps", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--gpu_util", type=float, default=0.30)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--eval_samples", type=int, default=None)
    ap.add_argument("--train_subset", type=int, default=2000)
    ap.add_argument("--teachable_from", default="",
                    help="restrict training to the prompts an OPD arm would see (those some ckpt of this teacher run solves) — the data-matched control for any OPD comparison")
    ap.add_argument("--include_base", action="store_true",
                    help="also eval the untrained model as step 0")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--runs", default="", help="report: comma-separated dirs")
    ap.add_argument("--report_out", default="")
    peftcfg.add_lora_args(ap)
    args = ap.parse_args()
    {"train": train, "eval": evaluate, "annotate": annotate,
     "report": report}[args.cmd](args)
