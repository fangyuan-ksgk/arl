"""
Build a self-contained teaching notebook for the ToyBox env, with REAL outputs
baked in (executed against skyrl_gym here, so it reads correctly without a kernel).

    /home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/build_toybox_notebook.py
"""

import base64
import contextlib
import io
import json
import sys

sys.path.insert(0, "/home/claudeuser/SkyRL/skyrl-gym")

OUT = "/home/claudeuser/arl/ToyBox_intro.ipynb"
FIG = "/home/claudeuser/arl/skyrl_terminal/figures/learn/trick6_toybox_loop.png"

_b64 = base64.b64encode(open(FIG, "rb").read()).decode()

# ── Cells: ("md", text) or ("code", source). Code cells exec in a shared ns and
#    their stdout is captured + baked as the cell output. ─────────────────────
CELLS = [
    ("md", f"""# ToyBox 🎮 — a tiny agentic RL environment

ToyBox is a **self-contained, multi-turn environment** where a small language model acts like a
coding agent: it runs `python` / `bash` snippets in a throwaway sandbox, reads the output, iterates,
and finishes by submitting an `<answer>` or writing `TASK_COMPLETE`. Reward is the **mean of
self-verifying checks** (partial credit) — no reward model.

<img src="data:image/png;base64,{_b64}" width="760">

This notebook walks through, in order:
1. **The task data format** — what a task and a dataset row look like
2. **The sandbox** — how the agent's code actually runs
3. **How a model interacts** — the `make → init → step → reward` loop, single- and multi-turn
"""),

    ("md", """## 1 · The task data format

Every puzzle is a plain Python `dict` in `skyrl_gym/envs/toybox/tasks.py`: a `prompt` the model sees,
plus a list of hidden `checks` that score the result. Let's look at one."""),

    ("code", '''import sys, json
sys.path.insert(0, "/home/claudeuser/SkyRL/skyrl-gym")
from skyrl_gym.envs.toybox import tasks as T

print(f"{len(T.TASKS)} tasks:", ", ".join(t["id"] for t in T.TASKS))
print("\\n--- one task (prime_oracle) ---")
print(json.dumps(T.get_task("prime_oracle"), indent=2))'''),

    ("md", """A **check** has a `kind` that says how to score the final state. The full vocabulary:

| kind | passes when |
|---|---|
| `answer_equals` / `answer_numeric` | the submitted `<answer>` matches |
| `stdout_equals` / `stdout_contains` | the last run's stdout matches/contains |
| `file_exists` / `file_equals` / `file_contains` | a file in the sandbox matches |
| `pyfunc` | a function the agent wrote passes test cases |

**Reward = mean of the checks**, so a partially-correct solution gets partial credit."""),

    ("md", """### The dataset row (parquet)

For training, tasks are serialized to parquet. Each row is one RL prompt: a shared **system** message
(teaching the action protocol) + the task's **user** message. The hidden answer key never appears in
the prompt — only the `task_id`, which the env uses to look up the checks."""),

    ("code", '''import pyarrow.parquet as pq

row = pq.read_table("/home/claudeuser/data/toybox/train.parquet").slice(0, 1).to_pylist()[0]
print("env_class :", row["env_class"])
print("task_id   :", row["extra_info"]["task_id"])
print("reward_spec:", row["reward_spec"])
print("\\nprompt messages:")
for m in row["prompt"]:
    print(f"  [{m['role']:6}] {m['content'][:88]}...")'''),

    ("md", """## 2 · The sandbox

Each rollout gets a fresh temp directory. The agent's `python`/`bash` run there as subprocesses with a
scrubbed env and a hard timeout. Relative writes (`output.txt`) land in that dir, where the checkers
read them back. Let's drive the sandbox directly."""),

    ("code", '''from skyrl_gym.envs.toybox import sandbox as sb

# seed a file, then run python and bash against it
box = sb.prepare_sandbox({"poem.txt": "hello toy box world"})
print("python →", sb.run_python(box, "print(open('poem.txt').read().upper())").stdout.strip())
print("bash   →", sb.run_bash(box, "wc -w poem.txt").stdout.strip())
sb.cleanup_sandbox(box)'''),

    ("md", """## 3 · How a model interacts

The agent loop is the standard gym contract: `make(env) → init(prompt) → step(action) → (obs, reward, done)`.
An **action** is just the model's text; the env parses it for:

- ```` ```python ```` / ```` ```bash ```` blocks → run in the sandbox, output returned as the next observation
- `<answer>...</answer>` → submit a final answer (ends the episode)
- `TASK_COMPLETE` → declare done (for file/stdout tasks)

Here's a tiny helper that runs one action and a **single-turn** solve of `prime_oracle` — note we pass
no model; the "action" is just text, exactly what a model would emit."""),

    ("code", '''import skyrl_gym

def make_env(task_id, max_turns=3):
    return skyrl_gym.make("toybox", env_config={},
                          extras={"extra_info": {"task_id": task_id}, "max_turns": max_turns})

action = """Here is the function:
```python
def is_prime(n):
    return n > 1 and all(n % i for i in range(2, int(n**0.5) + 1))
```
TASK_COMPLETE"""

env = make_env("prime_oracle")
env.init([{"role": "user", "content": ""}])
out = env.step(action)
env.close()
print("reward :", out["reward"])
print("done   :", out["done"])
print("checks :", out["metadata"]["n_passed"], "/", out["metadata"]["n_total"], "passed")'''),

    ("md", """### Multi-turn: explore, observe, then answer

`treasure_hunt` hides a codeword in a dot-folder. A real agent would **look around first** (turn 1),
read the observation, then **answer** (turn 2). Watch the env return an observation when the episode
isn't done yet, then a reward when it is."""),

    ("code", '''env = make_env("treasure_hunt", max_turns=3)
env.init([{"role": "user", "content": ""}])

# turn 1: explore with bash — episode continues, env hands back what it saw
obs = env.step("```bash\\nls -la && cat .vault/treasure.txt\\n```")
print("turn 1 done? ", obs["done"])
print("observation:\\n", obs["observations"][0]["content"][:220])

# turn 2: submit the codeword we found
out = env.step("Found it. <answer>moonstone</answer>")
print("\\nturn 2 done? ", out["done"], "| reward:", out["reward"])
env.close()'''),

    ("md", """### Partial credit (why the reward is a *mean*)

A buggy `is_prime` that's right on some inputs and wrong on others should land **between 0 and 1** —
a denser learning signal than all-or-nothing. Here `return n % 2 == 1` passes 6/9 cases."""),

    ("code", '''buggy = """```python
def is_prime(n):
    return n % 2 == 1
```
TASK_COMPLETE"""

env = make_env("prime_oracle")
env.init([{"role": "user", "content": ""}])
out = env.step(buggy)
env.close()
print(f"reward: {out['reward']:.3f}  ({out['metadata']['n_passed']}/{out['metadata']['n_total']} checks)")'''),

    ("md", """## 4 · Plugging in a real model

Everything above used hand-written text as the "action." To drive it with an actual model, you sample
the action from `model.generate(prompt)` and feed it to `env.step` — exactly what the **miniGRPO** helper
does. The cell below needs a GPU, so it's left unexecuted; uncomment to run:

```python
import sys; sys.path.insert(0, "/home/claudeuser/arl/skyrl_terminal/minigrpo")
import minigrpo as mg
model, tok = mg.load_model("Qwen/Qwen2.5-0.5B-Instruct")
print(mg.evaluate(model, tok, n_samples=4, env_class="toybox"))   # {'pass@1': ..., 'pass@4': ..., 'mean_reward': ...}
```

That's the whole arc: **task dict → dataset row → sandbox → step loop → reward**. To add your own puzzle,
append a dict (a `prompt` + a list of `checks`) to `tasks.py` and re-run `build_toybox_dataset.py`."""),
]


def build():
    ns: dict = {}
    cells = []
    ec = 0
    for ci, (kind, src) in enumerate(CELLS):
        if kind == "md":
            cells.append({"cell_type": "markdown", "id": f"cell{ci}", "metadata": {},
                          "source": src.splitlines(keepends=True)})
        else:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(src, "<cell>", "exec"), ns)
                text = buf.getvalue()
            except Exception as e:  # bake the error so the notebook is honest
                text = buf.getvalue() + f"\n[error] {type(e).__name__}: {e}"
            ec += 1
            outputs = ([{"output_type": "stream", "name": "stdout",
                         "text": text.splitlines(keepends=True)}] if text else [])
            cells.append({"cell_type": "code", "id": f"cell{ci}", "metadata": {}, "execution_count": ec,
                          "outputs": outputs, "source": src.splitlines(keepends=True)})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (SkyRL .venv)", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {OUT} — {len(cells)} cells, {ec} executed")


if __name__ == "__main__":
    build()
