"""Minimal Qwen3 chain-of-thought intervention helpers."""

from collections.abc import Sequence


def make_ablated_prefix_ids(
    tokenizer,
    messages: Sequence[dict[str, str]],
    ablated_cot: str,
) -> list[int]:
    """Return a chat prefix ending after a complete ablated thinking block."""
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    block_ids = tokenizer.encode(
        f"<think>\n{ablated_cot}\n</think>\n\n",
        add_special_tokens=False,
    )
    return prompt_ids + block_ids


def split_qwen3_response(response: str) -> tuple[str, str | None]:
    """Split a Qwen3 response into CoT and answer."""
    response = response.removeprefix("<think>").lstrip("\n")
    if "</think>" not in response:
        return response.strip(), None
    cot, answer = response.split("</think>", 1)
    return cot.rstrip(), answer.strip()
