"""One evaluation path for every arm. Built after four separate eval bugs.

WHAT WENT WRONG BEFORE, AND WHAT STOPS IT HERE

  1. A MATH model was scored on GSM8K because the eval recorded no dataset.
     -> every result carries {dataset, n, cap} AND a fingerprint of the exact
        question set; `table` REFUSES to compare files whose fingerprints
        differ, rather than printing a plausible wrong delta.

  2. Two scripts spelled the same knob differently (--max_tokens vs
     --max_completion_length, --gpu_util vs --eval_util), so a driver silently
     died with argparse exit 2 and an arm went unscored for a day.
     -> one entry point, one spelling: --cap, --n, --util.

  3. Two scripts wrote two schemas (eval_summary.json vs test_summary.json),
     so readers found nothing and "no result" looked like "not run".
     -> one schema, one location: output/evalx/<slug>.{json,jsonl}.

  4. Every number was right-censored by its own cap, so "0% lost at 4096" only
     meant nothing was ever allowed past 3072, and accuracy looked like a
     property of the model when it was a property of the budget.
     -> evaluate ONCE at a generous cap and derive every smaller budget.
        Greedy decoding is deterministic, so capping at C < the observed
        length truncates the SAME prefix: the answer is lost, never changed.
        Hence acc@C = mean(correct AND n_tokens <= C), exactly, for any
        C <= cap. One pass gives the whole curve, and no arm can ever again be
        compared to another at a different budget by accident.

KNOWN LIMIT, MEASURED NOT ASSUMED. Deriving budgets from one pass assumes
greedy decoding is reproducible. It is not across vLLM configurations: between
a cap-3072 pass and a cap-8192 pass of the SAME checkpoint, only 37 of 370
questions that both finished early had identical lengths. Different
max_model_len changes batching and kernels, one token diverges, and the
continuation diverges with it. So derived budgets describe the trajectory the
model produces UNDER THIS PASS, and cross-pass comparisons at the same budget
can differ by several points. Compare arms WITHIN one evalx pass; treat a
number from a differently-capped run as a different measurement.

    python script/opd/evalx.py run --cap 8192 --n 1000 \
        --models Qwen/Qwen3-1.7B output/math3072/drgrpo_s0/checkpoint-150
    python script/opd/evalx.py run --cap 8192 --n 1000 --run output/math3072/opd_min_s0
    python script/opd/evalx.py table --budgets 1024 2048 3072 8192
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import domains  # noqa: E402

OUT = ROOT / "output" / "evalx"


def slug(model):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(model).strip("/"))
    return s[-90:].strip("_")


def fingerprint(test):
    """Hash the actual prompts, so two files can be PROVEN to be the same
    questions in the same order."""
    h = hashlib.sha256()
    for i in range(len(test)):
        h.update(json.dumps(test[i]["prompt"], sort_keys=True).encode())
    return h.hexdigest()[:16]


def load_test(dataset, n):
    D = domains.get(dataset)
    _, test = D["load"]()
    if n:
        test = test.select(range(min(n, len(test))))
    return D, test


def cmd_run(a):
    import gc

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    D, test = load_test(a.dataset, a.n)
    fp = fingerprint(test)
    models = list(a.models)
    if a.run:
        models += [str(p) for p in sorted(
            Path(a.run).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]))]
    if not models:
        raise SystemExit("nothing to evaluate")
    OUT.mkdir(parents=True, exist_ok=True)

    for m in models:
        tag = slug(m)
        man = OUT / f"{tag}.json"
        if man.exists() and not a.overwrite:
            old = json.loads(man.read_text())
            if (old["fingerprint"] == fp and old["cap"] >= a.cap
                    and old["dataset"] == a.dataset):
                print(f"[evalx] {tag}: already have cap {old['cap']} on the "
                      f"same questions, skip", flush=True)
                continue
        tok = AutoTokenizer.from_pretrained(m)
        prompts = [tok.apply_chat_template(test[i]["prompt"], tokenize=False,
                                           add_generation_prompt=True)
                   + a.force_prefix for i in range(len(test))]
        llm = LLM(model=m, dtype="bfloat16", gpu_memory_utilization=a.util,
                  max_model_len=a.cap + a.prompt_room,
                  disable_log_stats=True)
        outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                    max_tokens=a.cap))
        rows = []
        for i, o in enumerate(outs):
            text = o.outputs[0].text
            e = D["answer_end"](text)
            rows.append({
                "idx": i,
                "correct": bool(D["correct"](text, test[i])),
                "n_tokens": len(o.outputs[0].token_ids),
                "n_cot_tokens": len(tok.encode(
                    text[:e] if e is not None else text,
                    add_special_tokens=False))})
        (OUT / f"{tag}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        hit = sum(1 for r in rows if r["n_tokens"] >= a.cap)
        man.write_text(json.dumps({
            "model": str(m), "dataset": a.dataset, "n": len(rows),
            "cap": a.cap, "fingerprint": fp,
            "force_prefix": a.force_prefix,
            "acc": sum(r["correct"] for r in rows) / len(rows),
            "hit_cap": hit,
            "mean_tokens": sum(r["n_tokens"] for r in rows) / len(rows)},
            indent=1))
        print(f"[evalx] {tag}: acc {100*sum(r['correct'] for r in rows)/len(rows):.1f} "
              f"| mean {sum(r['n_tokens'] for r in rows)/len(rows):.0f} tok "
              f"| still at cap {hit}/{len(rows)}", flush=True)
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def cmd_table(a):
    mans = sorted(OUT.glob("*.json"))
    recs = []
    for f in mans:
        m = json.loads(f.read_text())
        rows = [json.loads(l) for l in open(f.with_suffix(".jsonl"))]
        m["rows"] = rows
        recs.append(m)
    if a.only:
        recs = [r for r in recs
                if any(o in r["model"] for o in a.only)]
    if not recs:
        raise SystemExit(f"no results in {OUT}")

    fps = {r["fingerprint"] for r in recs}
    if len(fps) > 1:
        for r in recs:
            print(f"  {r['fingerprint']}  n={r['n']:5d}  {r['model']}")
        raise SystemExit(
            "REFUSING to tabulate: these were scored on DIFFERENT question "
            "sets. Re-run the odd ones out with the same --dataset/--n.")

    budgets = [b for b in a.budgets]
    print(f"dataset={recs[0]['dataset']}  n={recs[0]['n']}  "
          f"questions={recs[0]['fingerprint']}\n")
    w = max(len(Path(r['model']).as_posix()[-46:]) for r in recs)
    head = f"{'model':{w}s} {'mean tok':>9s} {'cap':>6s} " + \
        " ".join(f"{('@'+str(b)):>8s}" for b in budgets)
    print(head)
    print("-" * len(head))
    for r in sorted(recs, key=lambda x: -x["acc"]):
        rows, cap = r["rows"], r["cap"]
        cells = []
        for b in budgets:
            if b > cap:
                cells.append(f"{'--':>8s}")     # censored: never observed
            else:
                # by ANSWER POSITION, not total length. A truncated answer
                # can still be scored correct off an intermediate \boxed{},
                # which inflated the direct capped evals by ~4 pts -- and
                # inflated verbose arms MORE, since they truncate more.
                v = 100 * sum(1 for x in rows
                              if x["correct"]
                              and x["n_cot_tokens"] <= b) / len(rows)
                cells.append(f"{v:8.1f}")
        name = Path(r['model']).as_posix()[-46:]
        print(f"{name:{w}s} {r['mean_tokens']:9.0f} {cap:6d} "
              + " ".join(cells))
    print("\n-- = that budget exceeds the cap the pass was generated under, so\n"
          "     the answer is censored and the number would be a lower bound.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--run", default="", help="evaluate every ckpt of a run")
    ap.add_argument("--dataset", default="math")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--prompt_room", type=int, default=1024)
    ap.add_argument("--force_prefix", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[1024, 1536, 2048, 3072, 4096, 8192])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    {"run": cmd_run, "table": cmd_table}[a.cmd](a)
