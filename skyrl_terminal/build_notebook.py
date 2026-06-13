"""
Assemble the demonstration notebook: SkyRL_Terminal_and_Vision_GRPO.ipynb

Figures-first walkthrough of both pipelines, with code folded into the scripts in
this directory. Run make_figures.py first (or let cell 2 do it), then this.
"""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE.parent / "SkyRL_Terminal_and_Vision_GRPO.ipynb"


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


# Pull live headline numbers so the notebook prose matches the run.
def headline():
    import sys
    sys.path.insert(0, str(HERE))
    import parse_results as pr
    out = {}
    try:
        cur = pr.eval_curve(pr.parse_console_log("/tmp/terminal_grpo.log"))
        if not cur.empty:
            base = cur.iloc[0]
            best = cur.sort_values("eval/all/avg_score").iloc[-1]
            out["t_base_score"] = round(float(base.get("eval/all/avg_score", 0)), 3)
            out["t_best_score"] = round(float(best.get("eval/all/avg_score", 0)), 3)
            out["t_base_solved"] = round(float(base.get("eval/all/environment/all_passed", 0)), 3)
            out["t_best_solved"] = round(float(best.get("eval/all/environment/all_passed", 0)), 3)
    except Exception as e:
        out["err"] = str(e)
    try:
        gc = pr.parse_console_log("/tmp/geo3k_grpo.log")
        gs = gc[gc.metric == "eval/all/avg_score"]
        if not gs.empty:
            out["g_base"] = round(float(gs.sort_values("step").iloc[0].value), 3)
            out["g_best"] = round(float(gs.value.max()), 3)
    except Exception:
        pass
    return out


def build():
    h = headline()
    cells = []

    cells.append(md(
        "# Terminal-Bench & Vision GRPO with SkyRL — on one A100, no Docker\n\n"
        "This notebook is a **figures-first** tour of two RL pipelines built on "
        "[SkyRL](https://github.com/novasky-ai/SkyRL), both trained with **GRPO** on a single "
        "A100-80GB:\n\n"
        "| Goal | Task | Model | Reward | Result |\n"
        "|---|---|---|---|---|\n"
        f"| **1** | **Terminal-Bench** (real tasks) | Qwen2.5-Coder-3B | task's own `pytest` | "
        f"baseline **{h.get('t_base_score','?')}** avg / **{h.get('t_base_solved','?')}** solved → see curve |\n"
        f"| **2** | **Geometry-3K** (visual math) | Qwen3.5-4B (vision) + LoRA | `\\boxed{{}}` exact-match | "
        f"baseline **{h.get('g_base','see run')}** → see curve |\n\n"
        "All code lives in `arl/skyrl_terminal/` and the SkyRL-gym env in "
        "`SkyRL/skyrl-gym/skyrl_gym/envs/terminal/`. The notebook only orchestrates and plots."
    ))

    cells.append(md(
        "## 0. The core obstacle, and the local trick\n\n"
        "Terminal-Bench normally runs each task **inside a Docker container**: the agent works in "
        "`/app`, and a `pytest` verifier checks the resulting filesystem. **But this machine is an "
        "unprivileged container** — no `CAP_SYS_ADMIN`, user namespaces blocked — so Docker-in-Docker, "
        "rootless Docker, and bubblewrap all fail.\n\n"
        "**Trick:** `proot` is a *userspace* ptrace chroot that needs **zero privilege**. We bind a "
        "per-rollout temp dir to the guest path `/app`:\n\n"
        "```bash\nproot -b $SANDBOX:/app -w /app  bash -c \"<agent script>\"\n```\n\n"
        "Inside, `/app` *is* the sandbox dir, so the agent's writes and the verifier's absolute "
        "`/app/...` assertions resolve to the same isolated directory. The task's `tests/` stay on the "
        "host (never copied in), so the model can't peek at them."
    ))
    cells.append(md("![pipeline](skyrl_terminal/figures/pipeline_architecture.png)"))

    cells.append(md(
        "## 1. Goal 1 — Terminal-Bench GRPO\n\n"
        "### 1a. Curate tasks that are *faithfully* reproducible locally\n"
        "A task is kept only if running its **own `solution.sh`** in the proot sandbox makes its "
        "`pytest` fully pass — guaranteeing the RL reward is correct (canonical solution → 1.0). "
        "This filtered the 241 original tasks down to **32** that need only base tools.\n\n"
        "```bash\npython skyrl_terminal/curate_tasks.py --workers 48   # -> local_tasks.json\n```"
    ))
    cells.append(code(
        "import sys; sys.path.insert(0, 'skyrl_terminal')\n"
        "import json, pandas as pd\n"
        "kept = json.load(open('skyrl_terminal/local_tasks.json'))['kept']\n"
        "df = pd.DataFrame(kept)[['name','difficulty','category']]\n"
        "print(f'{len(df)} local-compatible tasks'); df.head(12)"
    ))

    cells.append(md(
        "### 1b. The SkyRL-gym `terminal` env (the only training code we wrote)\n"
        "`step()` extracts the model's ```bash``` block, runs it in the sandbox, and rewards the "
        "fraction of the task's pytest assertions that pass. Full code: "
        "`SkyRL/skyrl-gym/skyrl_gym/envs/terminal/{sandbox.py,env.py}`."
    ))
    cells.append(code(
        "from pathlib import Path\n"
        "src = Path('/home/claudeuser/SkyRL/skyrl-gym/skyrl_gym/envs/terminal/env.py').read_text()\n"
        "print(src.split('def step')[1][:900])"
    ))

    cells.append(md(
        "### 1c. Train\n"
        "Single-GPU colocated GRPO (vLLM + FSDP share the A100, sleep/wake). 32 prompts × 8 samples.\n\n"
        "```bash\nbash skyrl_terminal/run_terminal_grpo.sh\n```\n\n"
        "### 1d. Results"
    ))
    cells.append(code(
        "import make_figures as mf\n"
        "mf.terminal_curve('/tmp/terminal_grpo.log','/home/claudeuser/exports/terminal_v2','skyrl_terminal/figures/terminal_curve.png')\n"
        "mf.terminal_heatmap('/home/claudeuser/exports/terminal_v2','skyrl_terminal/figures/terminal_heatmap.png')\n"
        "from IPython.display import Image, display\n"
        "display(Image('skyrl_terminal/figures/terminal_curve.png'))\n"
        "display(Image('skyrl_terminal/figures/terminal_heatmap.png'))"
    ))

    cells.append(md(
        "## 2. Goal 2 — Vision GRPO (Qwen3.5-4B, LoRA)\n\n"
        "Multi-turn GRPO on **Geometry-3K** with **Qwen3.5-4B** — the natively vision-multimodal "
        "Qwen3.5 model (`Qwen3_5ForConditionalGeneration`, `vision_config`). The model sees a geometry "
        "diagram + question, may call a `calc_score` tool over up to 3 turns, and commits a `\\boxed{}` "
        "answer (binary reward). LoRA (rank 32) keeps the VLM + vLLM within 80 GB on one GPU.\n\n"
        "> **Sizing note / insight:** the upstream recipe uses an **8B** VLM on 8×H100. On a single "
        "A100-80GB, colocating an 8B VLM's vLLM engine *and* its FSDP training copy OOM-killed the vLLM "
        "engine-core at init (silent SIGKILL → `Failed core proc(s): {}`). The **4B** model plus "
        "`max_model_len=16384` and `enforce_eager=true` fits the colocated setup. The env's "
        "transformers 5.8.0 + vLLM 0.20.2 both already register the brand-new `Qwen3_5` architecture.\n\n"
        "```bash\nbash skyrl_terminal/run_geo3k_1gpu.sh\n```"
    ))
    cells.append(code(
        "import make_figures as mf\n"
        "mf.geo3k_curve('/tmp/geo3k_grpo.log','skyrl_terminal/figures/geo3k_curve.png')\n"
        "from IPython.display import Image\n"
        "try: display(Image('skyrl_terminal/figures/geo3k_curve.png'))\n"
        "except Exception as e: print('geo3k run not finished:', e)"
    ))

    cells.append(md(
        "## 3. Reproduce end-to-end\n\n"
        "```bash\n"
        "# 0. setup (once): clones + uv sync are in skyrl_terminal/SETUP.md\n"
        "# 1. curate + build dataset\n"
        "python skyrl_terminal/curate_tasks.py --workers 48\n"
        "/home/claudeuser/SkyRL/.venv/bin/python skyrl_terminal/build_dataset.py\n"
        "# 2. Goal 1: terminal-bench GRPO\n"
        "bash skyrl_terminal/run_terminal_grpo.sh\n"
        "# 3. Goal 2: vision GRPO\n"
        "bash skyrl_terminal/run_geo3k_1gpu.sh\n"
        "# 4. regenerate all figures\n"
        "python skyrl_terminal/make_figures.py\n"
        "```"
    ))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}}
    nbf.write(nb, str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    build()
