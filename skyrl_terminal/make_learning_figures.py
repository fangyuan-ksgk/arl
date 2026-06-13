"""
Generate figures-first teaching diagrams for the SkyRL GRPO project. 🎨

One clean PNG per "trick" -> arl/skyrl_terminal/figures/learn/. These are hand-laid
matplotlib diagrams (no GPU) meant to be skimmed top-to-bottom.

    /home/claudeuser/tbench-venv/bin/python arl/skyrl_terminal/make_learning_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401 (kept for reference)

OUT = "/home/claudeuser/arl/skyrl_terminal/figures/learn"
os.makedirs(OUT, exist_ok=True)

# palette
RED = "#ffd9d9"
GREEN = "#cdeccd"
BLUE = "#d6e4ff"
YELL = "#fff1bf"
GRAY = "#ececec"
PURP = "#e7d9ff"
EDGE = "#444444"


def _new(title, w=12, h=6.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 96, title, ha="center", va="top", fontsize=15, fontweight="bold")
    return fig, ax


def box(ax, x, y, w, h, text, fc=BLUE, fontsize=10.5, weight="normal", tc="black", ec=EDGE, ls="-"):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=1.6, linestyle=ls,
                               zorder=2, joinstyle="round"))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=tc,
            fontweight=weight, zorder=3, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, text=None, color=EDGE, lw=2.2, style="-|>", tcolor=None, dx=0, dy=2.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=4, shrinkB=4), zorder=1)
    if text:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", va="center",
                fontsize=9, color=tcolor or color, style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def caption(ax, text):
    ax.text(50, 4.5, text, ha="center", va="center", fontsize=10.5, style="italic",
            color="#222", bbox=dict(boxstyle="round,pad=0.5", fc="#fafafa", ec="#bbb"))


# ---------------------------------------------------------------------------
def fig_proot():
    fig, ax = _new("Trick 1 — Container-free sandboxing with proot  (no Docker, no root)")
    box(ax, 22, 80, 38, 14,
        "PROBLEM\nUnprivileged container:\nno CAP_SYS_ADMIN · user-ns blocked\n→ Docker / rootless / bwrap all FAIL",
        fc=RED, fontsize=10)
    box(ax, 74, 80, 44, 12,
        "THE TRICK  (userspace ptrace chroot)\nproot -b $TMP:/app -w /app  <cmd>",
        fc=GREEN, fontsize=11, weight="bold")
    arrow(ax, 41, 80, 52, 80)

    # host
    box(ax, 50, 50, 90, 30, "", fc="#f6f6f6", ec="#999")
    ax.text(9, 62, "HOST filesystem", fontsize=10, fontweight="bold", color="#666")
    box(ax, 28, 50, 32, 12, "throwaway dir\n/tmp/tb_app_XXXX/", fc=YELL, fontsize=10)
    box(ax, 72, 50, 32, 12, "task  tests/\n(stays host-side)", fc=GRAY, fontsize=10)

    box(ax, 50, 24, 26, 10, "guest path  /app", fc=BLUE, fontsize=11, weight="bold")
    arrow(ax, 28, 44, 44, 27, "bind :/app", color="#1a6")
    arrow(ax, 50, 19, 50, 12, "", color="#999")

    box(ax, 22, 9.5, 30, 8, "agent writes\n/app/out.txt", fc=GREEN, fontsize=9.5)
    box(ax, 78, 9.5, 34, 8, "verifier reads\n/app/out.txt (pytest)", fc=GREEN, fontsize=9.5)
    arrow(ax, 30, 21, 26, 13.5, color="#1a6")
    arrow(ax, 70, 21, 74, 13.5, color="#1a6")
    # tests reach verifier from host
    arrow(ax, 72, 44, 80, 13.5, "tests never copied in", color="#888", style="-|>", dx=10)

    ax.text(50, 33, "Agent's writes and the verifier's absolute /app/… checks resolve to the SAME dir.",
            ha="center", fontsize=9.5, style="italic", color="#333")
    plt.savefig(f"{OUT}/trick1_proot_sandbox.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_masking():
    fig, ax = _new("Trick 2 — Multi-turn rollout: conversation is the source of truth + loss masking")
    # loop on the left
    cx = 23
    box(ax, cx, 84, 34, 8, "1 ▸ render FULL conversation → token_ids", fc=BLUE, fontsize=9.5)
    box(ax, cx, 72, 34, 8, "2 ▸ generate (assistant tokens)", fc=GREEN, fontsize=9.5)
    box(ax, cx, 60, 34, 8, "3 ▸ env.step(action) → reward, obs", fc=YELL, fontsize=9.5)
    box(ax, cx, 48, 34, 8, "4 ▸ append assistant + obs msgs", fc=GRAY, fontsize=9.5)
    for y1, y2 in [(80, 76), (68, 64), (56, 52)]:
        arrow(ax, cx, y1, cx, y2)
    arrow(ax, cx + 18, 48, cx + 18, 84, color="#a33")
    ax.text(cx + 18.8, 66, "loop ≤\nmax_turns", color="#a33", fontsize=8.5, style="italic", ha="left", va="center")

    # token tape on the right
    ax.text(73, 88, "resulting token tape  +  loss_mask", fontsize=11, fontweight="bold", ha="center")
    segs = [("prompt", GRAY, "0"), ("gen\nturn1", GREEN, "1"), ("obs\nturn1", RED, "0"),
            ("gen\nturn2", GREEN, "1"), ("obs\nturn2", RED, "0"), ("gen\nturn3", GREEN, "1")]
    x = 49.5
    w = 7.3
    for name, c, m in segs:
        box(ax, x + w / 2, 70, w, 9, name, fc=c, fontsize=8.2)
        ax.text(x + w / 2, 62.8, f"m={m}", ha="center", fontsize=8.5,
                color="#1a6" if m == "1" else "#a33", fontweight="bold")
        x += w + 0.55
    box(ax, 73, 46, 48, 11,
        "Only GENERATED tokens train (mask=1).\nObservations — env feedback & image tokens —\nare masked out (mask=0).",
        fc="#eef7ee", fontsize=10)
    caption(ax, "Deferred-offset trick: record where obs starts, read its tokens from the NEXT turn's "
                "render — saves one render call/turn.")
    plt.savefig(f"{OUT}/trick2_multiturn_masking.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_grpo():
    fig, ax = _new("Trick 3 — GRPO: group-relative advantage, no value network")
    box(ax, 16, 70, 22, 12, "one prompt\nq", fc=BLUE, fontsize=11, weight="bold")
    ys = [86, 74, 62, 50]
    rs = ["r=1.0", "r=0.0", "r=1.0", "r=0.0"]
    for i, (y, r) in enumerate(zip(ys, rs)):
        box(ax, 47, y, 26, 8, f"completion o{i+1}", fc=GREEN if "1.0" in r else GRAY, fontsize=9)
        arrow(ax, 27, 70, 34, y)
        box(ax, 70, y, 12, 8, r, fc=YELL, fontsize=9.5)
        arrow(ax, 60, y, 64, y)
    ax.text(47, 42, "sample a GROUP of G=8 completions", ha="center", fontsize=9.5, style="italic", color="#555")

    box(ax, 86, 70, 22, 26,
        "advantage\nAᵢ = (rᵢ − mean) / std\n\n(baseline = the\ngroup's own mean)",
        fc=PURP, fontsize=10, weight="bold")
    arrow(ax, 76, 68, 78, 70)

    box(ax, 30, 22, 52, 12,
        "PPO needs a separate CRITIC network for the baseline.\nGRPO drops it — the group mean IS the baseline. Cheaper, simpler.",
        fc="#eef2ff", fontsize=10)
    box(ax, 80, 22, 34, 12,
        "INSIGHT\npass@8 ≫ pass@1  ⇒  the model\nCAN solve it, just not reliably.\nGRPO sharpens pass@8 → pass@1.",
        fc="#fff3f0", fontsize=9.3)
    caption(ax, "Reward can be binary (0/1) or fractional (partial credit) — fractional gives a denser gradient.")
    plt.savefig(f"{OUT}/trick3_grpo.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_parquet():
    fig, ax = _new("Trick 4 — The multimodal parquet trap  (lenient local vs strict server)")
    box(ax, 20, 80, 34, 12,
        "content = [\n  {type:image_url, …},\n  {type:text, …}  ]   ← real dicts",
        fc=GREEN, fontsize=9.5)
    arrow(ax, 37, 80, 50, 80, "datasets.to_parquet", color="#a60", dx=0, dy=3)
    box(ax, 70, 80, 40, 13,
        "heterogeneous dicts can't share ONE\nArrow struct → each item JSON-encoded\n[ '{\"type\":\"image_url\"…}', '…' ]  ← STRINGS!",
        fc=RED, fontsize=9.3)

    arrow(ax, 70, 73, 38, 58, color="#777")
    arrow(ax, 70, 73, 80, 58, color="#777")
    box(ax, 30, 50, 40, 13,
        "branch A: local prompt-length filter\napply_chat_template (LENIENT)\n✓ accepts strings → build passes",
        fc="#eef7ee", fontsize=9.3)
    box(ax, 78, 50, 40, 13,
        "branch B: rollout renders via vLLM\n/v1/chat/completions/render (STRICT Pydantic)\n✗ 'Input should be a valid dictionary' → CRASH",
        fc="#ffecec", fontsize=9.3)

    arrow(ax, 78, 43, 50, 30, color="#1a6")
    box(ax, 50, 22, 70, 13,
        "THE FIX — _normalize_mm_content():  json.loads() each string content part back into a dict\n"
        "(once, right after env.init, before any render). Idempotent; plain-text turns untouched.",
        fc=GREEN, fontsize=10, weight="bold")
    caption(ax, "Why Qwen3-VL 'worked' but Qwen3.5 didn't: same bad data, but their /render validators differ. "
                "Fix the data, not the model.")
    plt.savefig(f"{OUT}/trick4_parquet_bug.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_oom():
    fig, ax = _new("Trick 5 — One A100, TWO ceilings:  GPU VRAM  vs  host-RAM cgroup")
    # GPU column
    box(ax, 27, 86, 40, 7, "GPU — 80 GB VRAM  (colocate_all=true)", fc=BLUE, fontsize=10, weight="bold")
    ax.add_patch(plt.Rectangle((12, 24), 30, 54, facecolor="white", edgecolor=EDGE, lw=1.8))
    box(ax, 27, 66, 26, 12, "vLLM engine\nKV cache\ngpu_mem_util≈0.6", fc=GREEN, fontsize=9)
    box(ax, 27, 46, 26, 14, "FSDP policy+ref\n(time-shared\nwith vLLM)", fc=PURP, fontsize=9)
    box(ax, 27, 31, 26, 8, "enforce_eager=1\nmax_model_len=16k", fc=YELL, fontsize=8.5)
    ax.text(27, 20, "4B fits · 8B-class OOM-kills\nthe engine core at init", ha="center", fontsize=8.8, color="#a33")

    # HOST column
    box(ax, 73, 86, 44, 7, "HOST RAM — Ray sees the CGROUP, ~116 GB", fc=RED, fontsize=10, weight="bold")
    ax.add_patch(plt.Rectangle((58, 24), 30, 54, facecolor="white", edgecolor=EDGE, lw=1.8))
    # fill bar to ~95%
    ax.add_patch(plt.Rectangle((58, 24), 30, 54 * 0.95, facecolor="#ffd0d0", edgecolor="none"))
    box(ax, 73, 50, 26, 14, "FSDP\nforward_backward\n≈ 58 GB", fc="#ff9d9d", fontsize=9, weight="bold")
    box(ax, 73, 33, 26, 9, "256-traj batch\n(32 prompts × 8)", fc="#ffc2c2", fontsize=8.8)
    ax.plot([58, 88], [24 + 54 * 0.95, 24 + 54 * 0.95], color="#c00", lw=2, ls="--")
    ax.text(73, 71, "95% kill threshold", ha="center", fontsize=8.8, color="#c00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.text(73, 20, "host is really 1007 GB —\nbut the cgroup is the wall", ha="center", fontsize=8.8, color="#a33")

    caption(ax, "The terminal run died here (host-RAM OOM at full batch), NOT on the GPU. "
                "Fix: smaller batch (TRAIN_BS=16 N_SAMPLES=6). Watch BOTH meters.")
    plt.savefig(f"{OUT}/trick5_one_gpu_two_ooms.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_toybox():
    fig, ax = _new("Trick 6 — ToyBox: a self-verifying agentic loop with partial-credit reward")
    # the loop (clockwise)
    box(ax, 20, 74, 30, 12, "TASK prompt\n(themed puzzle)", fc=BLUE, fontsize=10, weight="bold")
    box(ax, 55, 74, 34, 12, "model emits\n```python  /  ```bash", fc=PURP, fontsize=10)
    box(ax, 88, 74, 20, 12, "sandbox\nruns it", fc=YELL, fontsize=10)
    box(ax, 88, 50, 20, 10, "stdout →\nobservation", fc=GRAY, fontsize=9.5)
    box(ax, 55, 50, 34, 10, "model iterates\n(loop ≤ max_turns)", fc=GRAY, fontsize=9.5)
    arrow(ax, 35, 74, 38, 74)
    arrow(ax, 72, 74, 78, 74)
    arrow(ax, 88, 68, 88, 55)
    arrow(ax, 78, 50, 72, 50)
    arrow(ax, 55, 55, 55, 68, color="#a33")
    arrow(ax, 45, 50, 38, 40, "<answer> / TASK_COMPLETE", color="#1a6", dx=-2, dy=-2)

    box(ax, 22, 33, 34, 10, "checks[] run (hidden)\nanswer / file / stdout / pyfunc", fc="#eef7ee", fontsize=9.3)
    arrow(ax, 39, 33, 50, 33)
    box(ax, 70, 33, 40, 12, "reward = mean(checks) ∈ [0,1]\nPARTIAL CREDIT", fc=GREEN, fontsize=11, weight="bold")

    box(ax, 50, 15, 86, 9,
        "e.g. a buggy is_prime passes 6/9 cases → reward 0.67  (not 0) — a denser GRPO signal than all-or-nothing.",
        fc="#fff7e6", fontsize=10)
    plt.savefig(f"{OUT}/trick6_toybox_loop.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_proot()
    fig_masking()
    fig_grpo()
    fig_parquet()
    fig_oom()
    fig_toybox()
    print("wrote 6 figures to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
