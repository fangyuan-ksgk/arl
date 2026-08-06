# Forgetting-gap verification + OPD

Everything needed to run the experiment end to end, and the configuration
constants that make the two arms comparable. Each constant below was set by a
measurement; several were set by a measurement that first came out **wrong**,
and those are flagged.

```bash
bash script/forgetting_gap/pipeline.sh all           # control -> annotate -> opd -> eval -> report
GPU=1 SEED=1 bash script/forgetting_gap/pipeline.sh all
PICK=latest bash script/forgetting_gap/pipeline.sh opd
python script/forgetting_gap/report.py --audit --run output/fgap/drgrpo_s0
```

## Contents — self-contained

| file | what it is |
|---|---|
| `pipeline.sh` | the five stages, with every config constant and why it has that value |
| `report.py` | the results table, and `--audit` for pre-flight checks on the teacher table |
| `drgrpo_min.py` | Dr.GRPO training, the annotation pass, and the legacy eval |
| `opd_min.py` | on-policy distillation from the per-query teacher checkpoint |
| `evalx.py` | the single evaluation path (fingerprinted, one generous pass) |
| `domains.py` | dataset registry: loaders, correctness, answer spans |
| `peftcfg.py` | optional LoRA plumbing (unused by default; imported by both trainers) |

The only outside dependency is `../grpo.py`, which supplies the MATH loader and
`math_equal`; `domains.py` puts `<repo>/script` on `sys.path` to reach it. Every
module resolves siblings first, so this folder runs against its own copies.

**These `.py` files are a pinned copy of `../opd/` as of 2026-08-06.** The
originals stay where the rest of the repo imports them. If you change one here,
it does not propagate — re-copy deliberately in whichever direction you mean:

```bash
diff -q script/forgetting_gap/opd_min.py script/opd/opd_min.py
```

---

## The claim under test

A Dr.GRPO run solves queries at some checkpoint and stops solving them later.
**union over checkpoints − final accuracy** is the *forgetting gap*, and OPD
exists to harvest it by distilling each query from a checkpoint that solved it.

Both halves have to be measured under a budget that does not itself create the
gap. That is the whole difficulty.

---

## Training config — identical for both arms

| knob | value | why |
|---|---|---|
| model | `Qwen/Qwen3-1.7B` | |
| data | MATH, **7500** train / 5000 test | |
| **max new tokens (train)** | **3072** | see below |
| lr | **1e-5** | worth **18 points** at this scale (an OPD sweep gave 48.6 at 1e-6 vs 66.6 at 1e-5). A mismatched lr swamps every method effect. |
| rollouts | G=8, `per_device_bs 2 × ga 64` = 128/step → **16 prompts/step** | |
| steps | **150** for both arms | equal prompt draws (2400) |
| Dr.GRPO | `--beta 0.01`, `loss_type dr_grpo`, `scale_rewards none` | |
| OPD | `--pick best --upsample 1 --interleave rr`, fp32 teacher, bf16 anchor, α=1.0, topK=64 | |

### Why 3072, and why it is not a free parameter

At **1536** the score was ~90% a truncation count. Re-scoring the *same*
checkpoints 1536 → 3072 moved seed 0 by **+10.4** and the other seeds by <1,
and the across-seed sd collapsed from **8.1 → 2.1**. The apparent "bimodal
seed effect" was the window, not the model.

But raising the budget also changes *what RL learns*. At 1536 one seed
discovered it could emit an **empty think block** and answer directly —
563 tokens instead of 3009, at the same accuracy. At 3072 no seed does this;
both keep writing ~2500 tokens with ~50% of rollouts still hitting the cap.
A tight cap teaches brevity, a loose one teaches capability. So 3072 is a
**hyperparameter of the experiment**, and any result must state it.

---

## Annotation — the full train split, at the training budget

```
--train_subset 0      # 0 = all 7500. NOT a sample.
--max_tokens 3072     # MUST equal the training budget
--overwrite           # cached train_step<S>.jsonl hold the previous pass
```

Two failure modes, both of which have already happened here:

1. **Partial annotation starves OPD.** OPD trains only on prompts some
   checkpoint solves. A 1000-prompt annotation left it **894 prompts = 11.9%**
   of what the control saw. At 75 steps that is **epoch 1.42** while the
   control was at 0.32 — the arms then differ in data, not just in objective,
   and the comparison is void.
2. **Annotating at a different budget than training mislabels the teachers.**
   The table records *which checkpoint solves which query*; answering that
   under a different truncation regime than the students train under assigns
   the wrong teacher.

Run `report.py --audit` before training. It prints the annotated fraction, the
teachable count, the forgotten count, the epochs OPD will reach, and the
teacher concentration.

### Watch the teacher concentration

On the 1000-prompt annotation:

```
pick=best       92.1% from ckpt-100   <-- effectively single-teacher
pick=latest     87.1% from ckpt-150   <-- effectively single-teacher
pick=shortest   24.9% from ckpt-50
```

Under `best` or `latest`, ~90% of the distillation comes from **one**
checkpoint — multi-teacher OPD in name only, with the round-robin machinery
acting on ~10% of the data. If you are testing the multi-teacher hypothesis,
this number is the experiment.

---

## Evaluation — one generous pass, both budgets derived

```
evalx run --cap 8192 --n 1000       # generate ONCE, nothing censored
```

then read two columns:

- **acc@8192** — capability with no truncation
- **acc@3072** — *train-test matched*: was the boxed answer emitted within the
  training budget? (`n_cot_tokens ≤ 3072`)

`evalx` records `{dataset, n, cap}` plus a **SHA of the actual prompts**, and
`report.py` refuses to tabulate results whose fingerprints differ.

### The capped-eval trap — do not re-generate at 3072

Running a second generation with `--cap 3072` gives a **different and inflated**
number. Two reasons, both measured:

1. When a model is cut off, the scorer takes the last `\boxed{}` present —
   some intermediate step — and if it matches the gold it scores **correct**.
   That is **73 of 233** truncated answers for the control. It inflates
   verbose arms more, because they truncate more, biasing exactly the
   comparison of interest. It also violates the stated convention that
   *truncated = wrong*, which the derived column implements correctly.
2. Greedy decoding is **not reproducible across vLLM configurations**. Of 370
   questions that finished early in both a cap-3072 and a cap-8192 pass of the
   same checkpoint, only **37** had identical lengths. Different
   `max_model_len` changes batching and kernels; one token diverges and the
   continuation follows.

So: compare arms **within one pass**, and treat any number from a
differently-capped run as a different measurement.

---

## Reading the result

```
  checkpoint    acc@3072   acc@8192
checkpoint-100      76.6       89.0
checkpoint-150      70.9       88.9
       union        83.9       92.8
FORGETTING GAP      13.0        3.9
```

The gap at the training budget is **13.0**; with nothing truncated it is
**3.9**. Two thirds of it was checkpoints running out of room on *different*
questions, not knowledge learned and lost. A real gap survives — the run does
genuinely peak mid-training and decay — but it is a third of the size the
method was designed around, and that is the number OPD has to beat.

**Statistical floor:** at n=1000 and p≈0.89 a paired (McNemar) comparison
resolves about **1.5 points**. Differences below that are not measurable on
this test set; use the full 5000-question split for a headline claim.
