"""Domains for the OPD family (Jul 29): GSM8K, MATH, MBPP.

Every script in script/opd/ was GSM8K-hardcoded. This module factors out the
four things a domain has to supply so `--dataset {gsm8k,math,mbpp}` works
everywhere:

    load()        -> (train_ds, test_ds) with a `prompt` chat list per row
    correct(text, row) -> bool          verification (answer match / tests pass)
    answer_end(text)   -> int|None      char offset where the ANSWER begins,
                                        used by bridge_attn's attention anchor
                                        and by the CoT-length metric
    defaults      max_completion_length, per_device_bs

MATH re-uses the boxed-answer logic of script/grpo.py; MBPP re-uses the
subprocess test-execution reward of script/grpo_code.py (untrusted model code
runs in a separate process with a timeout — see that file's notes).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "script"))

GSM8K_INSTR = (
    "Solve the problem step by step inside <think>...</think>. "
    "After </think>, give a brief final explanation and end your response "
    "with a line of the exact form:\n#### <number>\n"
    "where <number> is the final numeric answer with no units, no commas, "
    "and no extra text.")
MATH_INSTR = ("Solve the following math problem step by step. "
              "Put your final answer in \\boxed{}.")
CODE_INSTR = ("Write a Python function that solves the task. "
              "Return the complete function in a ```python code block.")


# --------------------------------------------------------------------- gsm8k
def _gsm8k_extract(text):
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m[-1].replace(",", "").rstrip(".")
    m = re.findall(r"-?[\d,]*\.?\d+", text)
    return m[-1].replace(",", "") if m else None


def _num_eq(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def _gsm8k_load():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")

    def fmt(ex):
        return {"prompt": [{"role": "user",
                            "content": f"{ex['question']}\n\n{GSM8K_INSTR} "
                                       f"/think"}],
                "gold": _gsm8k_extract(ex["answer"])}
    return ds["train"].map(fmt), ds["test"].map(fmt)


def _gsm8k_answer_end(text):
    ms = list(re.finditer(r"####", text))
    return ms[-1].start() if ms else None


def _gsm8k_answer_span(text):
    ms = list(re.finditer(r"####\s*(-?[\d,\.]+)", text))
    return (ms[-1].start(1), ms[-1].end(1)) if ms else None


# ---------------------------------------------------------------------- math
def _math_load():
    from grpo import extract_boxed, load_math          # script/grpo.py
    train, test = load_math()

    def fmt(ex):
        return {"prompt": [{"role": "user",
                            "content": f"{ex['problem']}\n\n{MATH_INSTR} "
                                       f"/think"}],
                "gold": extract_boxed(ex["solution"])}
    return train.map(fmt), test.map(fmt)


def _math_correct(text, gold):
    from grpo import math_equal, math_extract
    return bool(math_equal(math_extract(text), gold))


def _math_answer_end(text):
    ms = list(re.finditer(r"\\boxed\{", text))
    return ms[-1].start() if ms else None


def _math_answer_span(text):
    ms = list(re.finditer(r"\\boxed\{([^}]*)\}", text))
    return (ms[-1].start(1), ms[-1].end(1)) if ms else None


# ---------------------------------------------------------------------- mbpp
def _mbpp_load():
    from datasets import concatenate_datasets, load_dataset
    # the "full" config splits 974 tasks across train/test/validation/prompt;
    # concatenate them all and re-partition by task_id (the canonical split),
    # otherwise only 374 rows are visible and RL sees ~274 prompts.
    d = load_dataset("google-research-datasets/mbpp", "full")
    full = concatenate_datasets([d[k] for k in d])
    train_ids = set(range(1, 601)) | set(range(701, 975))
    test_ids = set(range(601, 701))

    def fmt(ex):
        tests = "\n".join(ex["test_list"][:1])
        return {"prompt": [{"role": "user",
                            "content": f"{ex['text']}\n\nYour function must "
                                       f"satisfy:\n{tests}\n\n{CODE_INSTR} "
                                       f"/think"}],
                "gold": ""}
    tr = full.filter(lambda x: x["task_id"] in train_ids).map(fmt)
    te = full.filter(lambda x: x["task_id"] in test_ids).map(fmt)
    return tr, te


def _mbpp_correct(text, row):
    from grpo_code import execute_code_with_tests, extract_code
    tests = row["test_list"] if isinstance(row, dict) else []
    if not tests:
        return False
    return bool(execute_code_with_tests(extract_code(text), list(tests)))


def _code_answer_span(text):
    """the final code block's body — what the tests actually execute"""
    ms = list(re.finditer(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL))
    return (ms[-1].start(1), ms[-1].end(1)) if ms else None


def _code_answer_end(text):
    ms = list(re.finditer(r"```", text))
    return ms[-2].start() if len(ms) >= 2 else (ms[0].start() if ms else None)


DOMAINS = {
    "gsm8k": dict(
        load=_gsm8k_load,
        correct=lambda text, row: _num_eq(_gsm8k_extract(text),
                                          row["gold"] if isinstance(row, dict)
                                          else row),
        answer_end=_gsm8k_answer_end,
        answer_span=_gsm8k_answer_span,
        format_re=r"####\s*[\d,\.\-]+",
        max_completion_length=1024, per_device_bs=4),
    "math": dict(
        load=_math_load,
        correct=lambda text, row: _math_correct(
            text, row["gold"] if isinstance(row, dict) else row),
        answer_end=_math_answer_end,
        answer_span=_math_answer_span,
        format_re=r"\\boxed\{",
        max_completion_length=2048, per_device_bs=2),
    "mbpp": dict(
        load=_mbpp_load,
        correct=lambda text, row: _mbpp_correct(text, row),
        answer_end=_code_answer_end,
        answer_span=_code_answer_span,
        format_re=r"```",
        # 1024 was measured to be WRONG at >=1.7B: at that cap 73% of
        # completions are truncated and only 31% contain a closed ```python
        # block, so MBPP scores truncation rather than capability. Same model,
        # same weights, cap 1024 -> 2048 -> 3072 gives 30.0 -> 45.0 -> 47.0.
        max_completion_length=3072, per_device_bs=2),
}


def get(name):
    if name not in DOMAINS:
        raise SystemExit(f"unknown dataset {name}; have {sorted(DOMAINS)}")
    return DOMAINS[name]
