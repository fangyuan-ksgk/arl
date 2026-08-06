"""The one table: Dr.GRPO per-ckpt, union, forgetting gap, and OPD — at two
evaluation budgets, from ONE set of generous passes.

Everything is read from output/evalx/, so every cell is the same 1000 MATH
questions (fingerprint-checked) generated in one 8192-token pass per model.
The @3072 column is derived from that same pass by ANSWER POSITION
(n_cot_tokens <= 3072), i.e. "would this answer have been finished inside a
3072 budget". It is NOT a re-run at cap 3072 — greedy decoding is not
reproducible across vLLM configurations, so a re-run differs by several
points and would not be comparable cell-to-cell.

union      = fraction of questions solved by AT LEAST ONE checkpoint
forgetting = union - final. It is the quantity OPD exists to harvest, and the
             point of showing it at two budgets is that a gap which only
             appears at the tight budget is checkpoints truncating DIFFERENT
             questions, not knowledge learned and then lost.

    python script/opd/gap_table.py --run output/math3072/drgrpo_s0 \
        --opd output/math3072/opd_min_s0
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script" / "opd"))
EV = ROOT / "output" / "evalx"


def slug(p):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(p).strip("/"))[-90:].strip("_")


def rows_for(model):
    f = EV / f"{slug(model)}.jsonl"
    if not f.exists():
        return None
    return {r["idx"]: r for r in map(json.loads, open(f))}


def acc(rows, budget=None):
    if budget is None:
        return 100 * sum(r["correct"] for r in rows.values()) / len(rows)
    return 100 * sum(1 for r in rows.values()
                     if r["correct"] and r["n_cot_tokens"] <= budget) / len(rows)


def solved(rows, budget=None):
    if budget is None:
        return {i for i, r in rows.items() if r["correct"]}
    return {i for i, r in rows.items()
            if r["correct"] and r["n_cot_tokens"] <= budget}


def ckpts(run):
    return sorted(Path(run).glob("checkpoint-*"),
                  key=lambda p: int(p.name.split("-")[1]))


def audit(a):
    """Health of the teacher table BEFORE spending a day training on it.

    Every one of these has silently invalidated a run at least once.
    """
    import domains
    import opd_min
    run = Path(a.run)
    ann = run / "train_forgetting_annotations.jsonl"
    if not ann.exists():
        raise SystemExit(f"no annotation at {ann}")
    tr, te = domains.get(a.dataset)["load"]()
    rows = [json.loads(l) for l in open(ann)]
    steps = sorted(int(s) for s in rows[0]["correct_at"])
    solv = [[s for s in steps if r["correct_at"][str(s)]] for r in rows]
    teach = [x for x in solv if x]
    forgotten = [x for x in teach if steps[-1] not in x]

    print(f"\n--- teacher-table audit: {run} ---")
    print(f"  full train split      : {len(tr)}")
    print(f"  annotated             : {len(rows)}"
          f"   {'<-- NOT the full split' if len(rows) < len(tr) else 'OK (full split)'}")
    print(f"  teachable (>=1 solver): {len(teach)}"
          f"   = {100*len(teach)/len(tr):.1f}% of the full split")
    print(f"  forgotten (some ckpt solves, FINAL does not): {len(forgotten)}"
          f"  = {100*len(forgotten)/len(rows):.1f}%")

    # 16 prompts per optimizer step at the pipeline's geometry
    for st in (75, 150, 300):
        ep = 16 * st / max(len(teach), 1)
        flag = "  <-- REPEATS DATA" if ep > 1.0 else ""
        print(f"  at {st:3d} steps OPD sees {16*st:5d} draws = {ep:.2f} epochs{flag}")

    print("  teacher concentration (share of prompts from ONE checkpoint):")
    for pick in ("best", "latest", "shortest"):
        try:
            t = opd_min.build_teacher_table(str(ann), steps, pick=pick)
        except SystemExit:
            continue
        c = collections.Counter(v for v in t.values() if v)
        if not c:
            continue
        top = c.most_common(1)[0]
        share = 100 * top[1] / sum(c.values())
        flag = "  <-- effectively single-teacher" if share > 70 else ""
        print(f"    pick={pick:9s} {share:5.1f}% from ckpt-{top[0]}{flag}")


def main(a):
    if a.audit:
        return audit(a)
    B = a.budgets
    print(f"MATH, n=1000, trained @{a.train_budget} tokens. "
          f"All cells from one 8192-token pass per model.\n")
    hdr = f"{'checkpoint':>12s} " + "".join(
        f"{('acc@' + str(b)):>11s}" for b in B)
    print(f"--- Dr.GRPO (the teacher run) ---")
    print(hdr)
    sets = {b: [] for b in B}
    have, miss = [], []
    for c in ckpts(a.run):
        r = rows_for(c)
        if r is None:
            miss.append(c.name)
            print(f"{c.name:>12s} " + "".join(f"{'pending':>11s}" for _ in B))
            continue
        have.append(c.name)
        cells = []
        for b in B:
            sets[b].append(solved(r, b))
            cells.append(f"{acc(r, b):11.1f}")
        print(f"{c.name:>12s} " + "".join(cells))

    if have:
        final = rows_for(ckpts(a.run)[-1])
        print("-" * len(hdr))
        for name, fn in (("union", lambda b: 100 * len(set().union(*sets[b])) / a.n),
                         ("final", lambda b: acc(final, b) if final else float("nan")),
                         ("FORGETTING GAP",
                          lambda b: (100 * len(set().union(*sets[b])) / a.n
                                     - (acc(final, b) if final else 0))),):
            print(f"{name:>12s} " + "".join(f"{fn(b):11.1f}" for b in B))
        if miss:
            print(f"\n  union is over {len(have)}/{len(have)+len(miss)} "
                  f"checkpoints ({', '.join(miss)} pending) -> LOWER BOUND")

    if a.opd:
        print(f"\n--- OPD distilled from those checkpoints ---")
        print(hdr)
        for c in ckpts(a.opd):
            r = rows_for(c)
            if r is None:
                print(f"{c.name:>12s} " + "".join(f"{'pending':>11s}" for _ in B))
                continue
            print(f"{c.name:>12s} "
                  + "".join(f"{acc(r, b):11.1f}" for b in B))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="output/math3072/drgrpo_s0")
    ap.add_argument("--opd", default="output/math3072/opd_min_s0")
    ap.add_argument("--budgets", type=int, nargs="+", default=[3072, 8192])
    ap.add_argument("--train_budget", type=int, default=3072)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--dataset", default="math")
    ap.add_argument("--audit", action="store_true",
                    help="check the teacher table for the failure modes that "
                         "have silently invalidated runs: partial annotation, "
                         "data starvation, single-teacher collapse")
    main(ap.parse_args())
