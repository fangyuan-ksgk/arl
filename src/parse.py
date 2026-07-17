"""Split a solved reasoning prefix from immediate backtracking."""

import re

_SENTENCE = re.compile(r"\S.*?(?:[.!?](?=\s|$)|\n+|$)", re.DOTALL)
_NUMBER = re.compile(r"-?[\d,]+\.?\d*")
_LEADING_MARKUP = re.compile(r"^[\s<>{}\[\]*_`#—-]+")
_BACKTRACK = re.compile(
    r"^(?:wait\b|but\s+wait\b|hold\s+on\b|however\b|actually\b|"
    r"on\s+second\s+thought\b|no[,!:\s]|oh[,!\s]|"
    r"i\s+(?:made|have\s+made|may\s+have\s+made|might\s+have\s+made)\s+"
    r"(?:a\s+)?mistake\b|(?:that|this)\s+(?:contradicts|cannot|can’t|can't|"
    r"doesn’t|doesn't|seems\s+wrong)\b)",
    re.IGNORECASE,
)


def _number(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _is_answer(match: re.Match, sentence: str, target: float) -> bool:
    left = match.start() == 0 or sentence[match.start() - 1].isspace()
    right = match.end() == len(sentence) or sentence[match.end()].isspace()
    return (
        left
        and right
        and _number(match.group()) == target
        and ("=" not in sentence or match.start() > sentence.rfind("="))
    )


def split_backtracking(text: str, gold: str) -> tuple[str, str] | None:
    """Return ``(correct_prefix, backtracking_suffix)`` for the first match."""
    target = _number(str(gold))
    if target is None:
        return None

    sentences = list(_SENTENCE.finditer(text))
    for answer, backtrack in zip(sentences, sentences[1:]):
        sentence = answer.group()
        solved = any(
            _is_answer(match, sentence, target) for match in _NUMBER.finditer(sentence)
        )
        start = _LEADING_MARKUP.sub("", backtrack.group()).strip()
        if solved and _BACKTRACK.match(start):
            return text[: answer.end()].strip(), text[answer.end() :].lstrip()
    return None
