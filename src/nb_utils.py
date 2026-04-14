"""
Notebook utilities for prefix_rollout.ipynb.
Kept in a .py file so the notebook cells stay clean.
"""
import re, html as _html, torch
from IPython.display import display, HTML


# ---------------------------------------------------------------------------
# CSS (assembled without literal </...> to avoid XML issues when this
# file is rendered in tool output)
# ---------------------------------------------------------------------------
_C = "</"          # avoid writing literal </ in XML-passed source

_STYLE = (
    "<style>\n"
    "  .seq-vis { font-family: 'Courier New', monospace; font-size: 13px;\n"
    "             line-height: 1.9; padding: 14px 18px; background:#fafafa;\n"
    "             border:1px solid #ddd; border-radius:8px; white-space:pre-wrap;\n"
    "             word-break:break-all; }\n"
    "  .no-grad  { background:#ffe0e0; color:#7a0000; padding:2px 3px;\n"
    "              border-radius:3px; }\n"
    "  .grad     { background:#d4f7d4; color:#004d00; padding:2px 3px;\n"
    "              border-radius:3px; text-decoration:underline wavy #22aa22; }\n"
    "  .legend   { font-family:sans-serif; font-size:12px; margin-bottom:6px; }\n"
    f"{_C}style>\n"
)


def _span(text: str, cls: str) -> str:
    return f'<span class="{cls}">{_html.escape(text)}{_C}span>'


def _legend_item(color_bg: str, color_fg: str, label: str, detail: str) -> str:
    underline = "text-decoration:underline wavy #22aa22;" if "grad" in label.lower() and "no" not in label.lower() else ""
    return (
        f'<span style="background:{color_bg};padding:2px 8px;border-radius:3px;'
        f'color:{color_fg};{underline}">'
        f'&#9632; {label} &mdash; <b>{detail}{_C}b>{_C}span>'
    )


def render_grad_html(
    no_grad_text: str,
    grad_text: str,
    n_no_grad: int,
    n_grad: int,
    n_saved: int | None = None,
):
    """Render a two-segment sequence with red (no-grad) and green (grad) highlighting."""
    saved_note = f", saving {n_saved} vs full" if n_saved is not None else ""
    legend = (
        '<div class="legend">'
        + _legend_item("#ffe0e0", "#7a0000", "No gradient", f"{n_no_grad} tokens")
        + "&nbsp;&nbsp;"
        + _legend_item("#d4f7d4", "#004d00", "Gradient ✓", f"{n_grad} tokens{saved_note}")
        + f'{_C}div>'
    )
    body = (
        '<div class="seq-vis">'
        + _span(no_grad_text, "no-grad")
        + _span(grad_text, "grad")
        + f'{_C}div>'
    )
    display(HTML(_STYLE + legend + body))


# ---------------------------------------------------------------------------
# Rollout generation
# ---------------------------------------------------------------------------
_IM_END = "<|im_end|>"       # safe: | is not an XML namechar

def extract_answer(text: str) -> str:
    m = re.search(r"####\s*([\d,\.\-]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def is_correct(completion_text: str, gold: str) -> bool:
    try:
        return float(extract_answer(completion_text)) == float(gold)
    except (ValueError, TypeError):
        return False


def generate_rollout(model, tokenizer, question: str, max_new_tokens: int = 400):
    """Return (prompt_str, completion_text) for a single sampled rollout."""
    msgs = [{"role": "user", "content": question}]
    prompt_str = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    # Keep <think>/</think> (Qwen3 special tokens); strip structural chat tokens.
    comp = tokenizer.decode(new_ids, skip_special_tokens=False)
    comp = comp.replace(_IM_END, "").strip()
    return prompt_str, comp
