"""
Multi-model results sweep via the miniGRPO eval engine (HF generate + sandbox,
no Ray/vLLM → cheap, low-memory). Writes model_results.json which make_tutorial.py
folds into tutorial.html.

    /home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/model_sweep.py

Run AFTER the heavy SkyRL terminal job concludes (one A100). Each model is loaded,
evaluated, then freed before the next.
"""

import json
import os
import sys

sys.path.insert(0, "/home/claudeuser/arl/skyrl_terminal/minigrpo")
import minigrpo as mg  # noqa: E402

OUT = "/home/claudeuser/arl/skyrl_terminal/model_results.json"

# Small, fast instruct/coder models that fit comfortably on one A100.
TOYBOX_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
]
TERMINAL_MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
]

N_SAMPLES = 4


def sweep(models, env_class, n_samples, max_new_tokens, limit=None):
    import torch
    rows = []
    for m in models:
        print(f"\n=== {env_class}: {m} ===", flush=True)
        try:
            model, tok = mg.load_model(m)
            r = mg.evaluate(model, tok, n_samples=n_samples, env_class=env_class,
                            max_new_tokens=max_new_tokens, limit=limit)
            rows.append({"model": m.split("/")[-1], "pass@1": r["pass@1"],
                         "passk": r[f"pass@{n_samples}"], "mean_reward": r["mean_reward"]})
            print(f"  -> {r}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  !! {m} failed: {e}", flush=True)
            rows.append({"model": m.split("/")[-1], "pass@1": 0.0, "passk": 0.0,
                         "mean_reward": 0.0, "error": str(e)[:120]})
    return rows


def main():
    results = {}
    # merge with any prior results so partial reruns don't lose data
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT))
        except Exception:
            results = {}

    results[f"ToyBox (miniGRPO eval, n={N_SAMPLES})"] = sweep(
        TOYBOX_MODELS, "toybox", N_SAMPLES, max_new_tokens=512)
    json.dump(results, open(OUT, "w"), indent=2)  # checkpoint after toybox

    results[f"Terminal-Bench (miniGRPO eval, n={N_SAMPLES})"] = sweep(
        TERMINAL_MODELS, "terminal", N_SAMPLES, max_new_tokens=768)
    json.dump(results, open(OUT, "w"), indent=2)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
