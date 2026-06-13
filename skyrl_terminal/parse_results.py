"""
Parse SkyRL console logs + dumped eval results into tidy DataFrames for plotting.

Two sources:
  * The console training log (LOGGER=console pprints a metrics dict each step).
  * ``{export_path}/dumped_evals/global_step_*_evals/`` (per-task eval rollouts).

Used by the demonstration notebook to draw the GRPO learning curves and the
per-task pass heatmap.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# A "key: value," line inside a pprinted metrics dict (possibly quoted float).
_KV = re.compile(r"['\"]([\w/\.@]+)['\"]\s*:\s*'?([-\d.eE]+)'?[,}]")
_STEP = re.compile(r"global_step['\"]?\s*[:=]\s*'?(\d+)")


def parse_console_log(log_path: str) -> pd.DataFrame:
    """Pull per-step scalar metrics from a SkyRL console log.

    Returns a long-form DataFrame: columns [step, metric, value].
    Strategy: scan line-by-line; whenever we see a global_step marker, update the
    current step; collect every 'key': value float we see and attach the most
    recent step. Train and eval metric dicts both get captured.
    """
    if not Path(log_path).exists():
        return pd.DataFrame(columns=["step", "metric", "value"])
    text = Path(log_path).read_text(errors="replace")
    rows: List[Dict] = []
    cur_step = 0
    for line in text.splitlines():
        m = _STEP.search(line)
        if m:
            cur_step = int(m.group(1))
        for k, v in _KV.findall(line):
            try:
                rows.append({"step": cur_step, "metric": k, "value": float(v)})
            except ValueError:
                pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["step", "metric"], keep="last")
    return df


def load_eval_dumps(export_path: str) -> pd.DataFrame:
    """Per-task scores across all dumped eval checkpoints.

    Returns columns [step, task_name, difficulty, category, score].
    The dumped ``score`` is a per-token reward list; the trajectory reward is its
    sum (terminal reward lands on the last token).
    """
    base = Path(export_path) / "dumped_evals"
    rows: List[Dict] = []
    for d in sorted(base.glob("global_step_*_evals")):
        m = re.search(r"global_step_(\d+)_evals", d.name)
        step = int(m.group(1)) if m else -1
        jsonl = d / "terminal_bench.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            ex = r["env_extras"]
            ex = ast.literal_eval(ex) if isinstance(ex, str) else ex
            info = ex.get("extra_info", {})
            sc = r["score"]
            sc = ast.literal_eval(sc) if isinstance(sc, str) else sc
            val = float(sum(sc)) if isinstance(sc, list) else float(sc)
            rows.append({
                "step": step,
                "task_name": info.get("task_name", "?"),
                "difficulty": info.get("difficulty", "?"),
                "category": info.get("category", "?"),
                "score": val,
            })
    return pd.DataFrame(rows)


def eval_curve(console_df: pd.DataFrame) -> pd.DataFrame:
    """Wide table of the headline eval metrics over steps."""
    keys = [
        "eval/all/avg_score",
        "eval/all/pass_at_1",
        "eval/all/environment/all_passed",
        "eval/all/environment/frac_passed",
    ]
    sub = console_df[console_df.metric.isin(keys)]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="step", columns="metric", values="value").reset_index()


if __name__ == "__main__":
    import sys
    log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/terminal_grpo.log"
    exp = sys.argv[2] if len(sys.argv) > 2 else "/home/claudeuser/exports"
    cdf = parse_console_log(log)
    print("console metrics found:", sorted(cdf.metric.unique())[:40])
    print(eval_curve(cdf))
    edf = load_eval_dumps(exp)
    if not edf.empty:
        print(edf.groupby("step").score.mean())
