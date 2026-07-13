"""qa_em reward for SearchR1 — exact match after SQuAD normalization.

Port of SkyRL-gym `skyrl_gym/envs/search/utils.py` (itself adapted from
Search-R1 `verl/utils/reward_score/qa_em.py`). Reward = 1.0 iff the text inside
the LAST <answer>...</answer> tag, after normalization (lowercase, strip
punctuation/articles/extra whitespace), exactly matches any accepted answer in
ground_truth["target"].

Also exposes `qa_em_reward`, a TRL GRPOTrainer-compatible reward function that
reads the `ground_truth` dataset column.
"""

import re
import string

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_check(prediction: str, golden_answers) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(str(golden_answer)) == normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str: str):
    """Return the content of the LAST <answer>...</answer> tag, or None."""
    matches = ANSWER_RE.findall(solution_str)
    return matches[-1].strip() if matches else None


def compute_score_em(solution_str: str, ground_truth, format_score: float = 0.0, score: float = 1.0) -> float:
    """ground_truth is {"target": [accepted answers]} (SearchR1 schema) or a plain list/str."""
    if isinstance(ground_truth, dict):
        targets = ground_truth.get("target", [])
    else:
        targets = ground_truth
    # parquet/pandas round-trips can hand back numpy arrays
    if not isinstance(targets, (list, str)):
        targets = list(targets)
    answer = extract_solution(solution_str)
    if answer is None:
        return 0.0
    return float(score) if em_check(answer, targets) else float(format_score)


def _completion_text(completion) -> str:
    """TRL passes conversational completions as [{"role": ..., "content": ...}, ...]."""
    if isinstance(completion, str):
        return completion
    return "".join(m.get("content", "") or "" for m in completion if isinstance(m, dict))


def qa_em_reward(prompts=None, completions=None, completion_ids=None, ground_truth=None, **kwargs):
    """TRL reward function. `ground_truth` is the dataset column (one entry per completion)."""
    rewards = [
        compute_score_em(_completion_text(completion), gt)
        for completion, gt in zip(completions, ground_truth)
    ]
    # Optional visibility: log rollout stats forwarded by the rollout_func via extra fields.
    log_metric = kwargs.get("log_metric")
    if log_metric is not None:
        num_search_calls = kwargs.get("num_search_calls")
        if num_search_calls:
            log_metric("rollout/mean_search_calls", sum(num_search_calls) / len(num_search_calls))
        search_failures = kwargs.get("search_failures")
        if search_failures:
            log_metric("rollout/mean_search_failures", sum(search_failures) / len(search_failures))
    return rewards
