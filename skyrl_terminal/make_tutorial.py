"""
Build a single, self-contained tutorial.html (figures embedded as base64 — one
portable file you can open anywhere). Re-run after model sweeps land to inject
the results tables.

    /home/claudeuser/tbench-venv/bin/python arl/skyrl_terminal/make_tutorial.py
"""

import base64
import json
import os

HERE = "/home/claudeuser/arl/skyrl_terminal"
FIG = f"{HERE}/figures/learn"
OUT = f"{HERE}/tutorial.html"
RESULTS_JSON = f"{HERE}/model_results.json"  # written by the sweep; optional


def img(name, alt):
    p = f"{FIG}/{name}"
    if not os.path.exists(p):
        return f'<p><em>[missing figure: {name}]</em></p>'
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def trick(n, title, fig, bullets):
    lis = "\n".join(f"<li>{b}</li>" for b in bullets)
    return f"""
    <section class="trick">
      <h3>{title}</h3>
      {img(fig, title)}
      <ul>{lis}</ul>
    </section>"""


def results_table():
    """Render the model-sweep tables if model_results.json exists, else a placeholder."""
    if not os.path.exists(RESULTS_JSON):
        return ('<p class="todo">⏳ Model sweeps run after the current terminal-bench job '
                'finishes (held back to keep one A100 free). This section auto-fills then.</p>')
    data = json.load(open(RESULTS_JSON))
    html = ""
    for env_name, rows in data.items():
        body = ""
        for r in rows:
            body += ("<tr><td>{model}</td><td>{pass1:.3f}</td><td>{passk:.3f}</td>"
                     "<td>{mean:.3f}</td></tr>").format(
                model=r["model"], pass1=r.get("pass@1", 0), passk=r.get("passk", 0),
                mean=r.get("mean_reward", 0))
        html += f"""
        <h3>{env_name}</h3>
        <table>
          <tr><th>model</th><th>pass@1</th><th>pass@k</th><th>mean reward</th></tr>
          {body}
        </table>"""
    return html


CSS = """
:root{--fg:#222;--muted:#666;--accent:#1a6;--bg:#fafafa;--card:#fff;--code:#f4f4f6;}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
  background:var(--bg);max-width:920px;margin:0 auto;padding:32px 20px 80px;line-height:1.6}
h1{font-size:2rem;margin:.2em 0}
h2{margin-top:1.8em;border-bottom:2px solid #e3e3e3;padding-bottom:.3em}
h3{margin-top:1.4em}
.sub{color:var(--muted);font-size:1.05rem}
img{max-width:100%;border:1px solid #e0e0e0;border-radius:8px;margin:.6em 0;background:#fff}
.trick{background:var(--card);border:1px solid #e8e8e8;border-radius:12px;padding:18px 22px;margin:18px 0;
  box-shadow:0 1px 3px rgba(0,0,0,.04)}
ul{margin:.4em 0}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);border-radius:6px}
code{padding:1px 5px;font-size:.92em}
pre{padding:14px 16px;overflow-x:auto;border:1px solid #e6e6e6;font-size:.86em;line-height:1.45}
table{border-collapse:collapse;width:100%;margin:1em 0;background:#fff}
th,td{border:1px solid #e2e2e2;padding:7px 11px;text-align:left;font-size:.93rem}
th{background:#f0f4f0}
.toc{background:#fff;border:1px solid #e8e8e8;border-radius:10px;padding:10px 22px}
.toc a{color:var(--accent);text-decoration:none}
.todo{background:#fff7e6;border:1px solid #ffd98a;border-radius:8px;padding:12px 16px;color:#7a5a00}
.pill{display:inline-block;background:#eef7ee;border:1px solid #cce6cc;border-radius:999px;
  padding:2px 10px;font-size:.8rem;color:#1a6;margin-right:6px}
.lead{background:#eef3ff;border:1px solid #cdddff;border-radius:10px;padding:14px 18px}
footer{margin-top:40px;color:var(--muted);font-size:.85rem;border-top:1px solid #e3e3e3;padding-top:14px}
"""

TRICKS = [
    ("Trick 1 — Container-free sandboxing with proot", "trick1_proot_sandbox.png", [
        "No <code>CAP_SYS_ADMIN</code>, user-ns blocked → Docker/rootless/bubblewrap all fail.",
        "<code>proot -b $TMP:/app -w /app</code> is a userspace ptrace chroot — zero privilege.",
        "Agent writes <code>/app/…</code> and the pytest verifier's absolute <code>/app/…</code> checks hit the <b>same</b> temp dir; tests stay host-side."]),
    ("Trick 2 — Multi-turn rollout + loss masking", "trick2_multiturn_masking.png", [
        "Re-render the full conversation each turn → generate → <code>env.step</code> → append → loop.",
        "Train only on <b>generated</b> tokens (mask=1); observations & image tokens are masked out (mask=0).",
        "Deferred-offset trick reads obs tokens from the next render — one fewer call per turn."]),
    ("Trick 3 — GRPO: group-relative advantage, no critic", "trick3_grpo.png", [
        "Sample a group of G completions, score each, advantage <code>Aᵢ=(rᵢ−mean)/std</code>.",
        "The group mean <b>is</b> the baseline — that's the whole simplification vs PPO.",
        "<code>pass@8 ≫ pass@1</code> ⇒ the model can solve it but unreliably; GRPO sharpens it. Partial credit = denser gradient."]),
    ("Trick 4 — The multimodal parquet trap (a real bug we fixed)", "trick4_parquet_bug.png", [
        "Mixed image+text content lists can't share one Arrow struct → <code>to_parquet</code> JSON-encodes each part to a string.",
        "Lenient local <code>apply_chat_template</code> passes; strict vLLM <code>/render</code> rejects strings → crash.",
        "Fix = decode strings back to dicts before render. <b>Fix the data, not the model.</b>"]),
    ("Trick 5 — One A100, two ceilings (the OOM that bit us all day)", "trick5_one_gpu_two_ooms.png", [
        "<code>colocate_all</code> time-shares 80 GB VRAM between vLLM and FSDP; tune <code>gpu_mem_util</code>, <code>enforce_eager</code>, <code>max_model_len</code>.",
        "Ray sees the container <b>cgroup (~116 GB)</b>, not the 1007 GB host. A full batch's FSDP forward_backward (~58 GB) blew past the 95% kill line.",
        "Bonus gotcha: a leaked 74 GB runaway from a crashed rollout squatting in the shared cgroup OOM-killed a fresh run at <i>init</i>. Reap orphans; watch both meters."]),
    ("Trick 6 — ToyBox: self-verifying agentic loop + partial credit", "trick6_toybox_loop.png", [
        "Model emits <code>```python</code>/<code>```bash</code>, runs in a sandbox, reads output, iterates, ends with <code>&lt;answer&gt;</code> or <code>TASK_COMPLETE</code>.",
        "Hidden <code>checks[]</code> (answer/file/stdout/pyfunc) → <b>reward = mean(checks) ∈ [0,1]</b>.",
        "A buggy <code>is_prime</code> passing 6/9 cases scores 0.67, not 0 — denser signal than all-or-nothing."]),
]

TOYBOX_ROWS = [
    ("🏰", "fizzbuzz_dungeon", "print FizzBuzz 1–15", "stdout_equals"),
    ("🔮", "prime_oracle", "write is_prime", "pyfunc"),
    ("🗝️", "caesar_crypt", "decode ROT13", "answer_equals"),
    ("💰", "base64_treasure", "decode base64", "answer_equals"),
    ("🧭", "treasure_hunt", "find hidden .vault/ file", "answer (bash)"),
    ("🔥", "fibonacci_forge", "write fib", "pyfunc"),
    ("📜", "word_count_wizard", "count words → file", "file_equals"),
    ("🪞", "reverse_runes", "reverse a string", "answer_equals"),
    ("🗿", "gcd_golem", "write gcd", "pyfunc"),
    ("🪙", "sum_the_loot", "sum a file of ints", "answer_numeric"),
    ("🃏", "json_jester", "extract a JSON key → file", "file_equals"),
    ("⚖️", "anagram_altar", "write is_anagram", "pyfunc"),
]


def toybox_table():
    body = "".join(f"<tr><td>{e}</td><td><code>{i}</code></td><td>{s}</td><td>{c}</td></tr>"
                   for e, i, s, c in TOYBOX_ROWS)
    return f"""<table><tr><th></th><th>task</th><th>skill</th><th>check</th></tr>{body}</table>"""


HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SkyRL GRPO — visual field guide</title><style>{CSS}</style></head><body>

<h1>SkyRL GRPO — a visual field guide</h1>
<p class="sub">RL-on-LLMs tricks, a fun agentic env, and a GRPO you can hack — built on a single A100, no Docker.</p>
<p><span class="pill">read time ~10 min</span><span class="pill">figures-first</span><span class="pill">copy-paste commands</span></p>

<div class="lead"><b>What you'll get:</b> six diagrams that each teach one trick, the ToyBox puzzle env,
and <code>miniGRPO</code> — a 250-line GRPO whose <i>advantage function</i> is a single thing you edit
(the same seam your <code>tree_trainer.py</code> overrides). Everything below maps to real files in
<code>arl/skyrl_terminal/</code>.</div>

<div class="toc"><b>Contents</b>
<ol>
<li><a href="#tricks">The six tricks (diagrams)</a></li>
<li><a href="#toybox">ToyBox: the puzzle env</a></li>
<li><a href="#grpo">miniGRPO: hack the algorithm</a></li>
<li><a href="#results">Model results</a></li>
<li><a href="#run">Run everything (cheatsheet)</a></li>
</ol></div>

<h2 id="tricks">1 · The six tricks</h2>
{''.join(trick(i+1, t, f, b) for i,(t,f,b) in enumerate(TRICKS))}

<h2 id="toybox">2 · ToyBox — the puzzle env</h2>
<p>A fun, self-contained multi-turn env (<code>id="toybox"</code>): a small model acts as a coding agent on
bite-sized puzzles, scored by self-verifying checks with partial credit. 12 puzzles spanning the
Python interpreter, the shell, and pure reasoning:</p>
{toybox_table()}
<p>Add a puzzle by appending a dict (a <code>prompt</code> + a list of <code>checks</code>) to
<code>SkyRL/skyrl-gym/skyrl_gym/envs/toybox/tasks.py</code>. Sanity-check with
<code>toybox_smoke_test.py</code> (drives all 12 with perfect solutions — no GPU).</p>

<h2 id="grpo">3 · miniGRPO — hack the algorithm</h2>
<p>One GRPO step, in plain PyTorch (no Ray/vLLM/FSDP). The <b>advantage is a swappable function</b> —
exactly the seam <code>tree_trainer.TreeTrainer._compute_loss</code> overrides:</p>
<pre>prompts ─&gt; sample G completions (sandbox) ─&gt; rewards
                                              │
   ┌──── THE SEAM (edit me) ◀─────────────────┘
   │  group_zscore:  aᵢ = (rᵢ − mean_g)/(std_g+ε)     # scalar
   │  to_per_token:  flat | opa | &lt;your idea&gt;          # scalar -&gt; per token
   └──────────────────────┬───────────────────────────┘
                          ▼
   loss = −mean( advantageₜ · logp(tokenₜ) )           # == GRPO</pre>
<p>Built-in modes in <code>ADVANTAGE_MODES</code>: <code>flat</code> (vanilla GRPO) and <code>opa</code>
(your Optimistic Prefix Advantage — imported straight from <code>arl/src/tree_trainer.py</code>, builds a
per-group prefix trie, credits each token with the best reachable continuation). Add your own:</p>
<pre>def adv_myidea(rewards, group_ids, token_seqs):
    scal = group_zscore(rewards, group_ids)   # reuse the scalar step
    return [...]                              # per-token list[list[float]]
ADVANTAGE_MODES["myidea"] = adv_myidea
# python minigrpo.py --train --mode myidea</pre>
<p><b>Scaling to SkyRL</b> (Ray/FSDP/vLLM): SkyRL has a plug-in registry — no core edits. Register
<code>@register_advantage_estimator("opa")</code> (see
<code>examples/train/algorithms/custom_advantage_estimator/</code>) and run with
<code>trainer.algorithm.advantage_estimator=opa</code>. ⚠️ One wrinkle: the estimator only receives
<code>(rewards, response_mask, index)</code>, <b>not</b> the response token ids OPA needs — thread them in
from <code>data["responses"]</code> in <code>Trainer.compute_advantages_and_returns</code>
(<code>skyrl/train/trainer.py:960</code>). Prototype in miniGRPO (token ids are right there in each
<code>Rollout</code>), then port the validated function. Full detail in
<code>minigrpo/README.md</code>.</p>

<h2 id="results">4 · Model results</h2>
{results_table()}

<h2 id="run">5 · Run everything (cheatsheet)</h2>
<pre>PY=/home/claudeuser/SkyRL/.venv/bin/python
TB=/home/claudeuser/tbench-venv/bin/python

# ToyBox env sanity (no GPU)
$PY arl/skyrl_terminal/toybox_smoke_test.py

# miniGRPO: hack the advantage (no GPU for the seam test)
$PY arl/skyrl_terminal/minigrpo/minigrpo.py selftest
$PY arl/skyrl_terminal/minigrpo/minigrpo.py --eval --model Qwen/Qwen2.5-1.5B-Instruct
$PY arl/skyrl_terminal/minigrpo/minigrpo.py --train --mode opa --steps 30

# Full SkyRL runs (1 A100, memory-safe)
bash arl/skyrl_terminal/run_terminal_bench.sh        # terminal-bench GRPO
bash arl/skyrl_terminal/run_toybox_grpo.sh           # toybox GRPO

# Regenerate the figures / this page
$TB arl/skyrl_terminal/make_learning_figures.py
$TB arl/skyrl_terminal/make_tutorial.py</pre>

<footer>Generated by <code>make_tutorial.py</code>. Figures embedded as base64 — this file is fully
self-contained and portable. Companion docs: <code>LEARN_THE_TRICKS.md</code>, <code>TOYBOX.md</code>,
<code>minigrpo/README.md</code>, <code>HANDOFF.md</code>.</footer>
</body></html>"""


if __name__ == "__main__":
    with open(OUT, "w") as f:
        f.write(HTML)
    kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT} ({kb:.0f} KB, self-contained)")
