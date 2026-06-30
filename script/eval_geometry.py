"""R3 eval: Geometry3K held-out test accuracy (boxed answer match) via HF generation.
Run in the Qwen3.5 venv. vLLM 0.11's Qwen3-VL multimodal path asserts, so we use HF generate.
    /home/claudeuser/venv_qwen35/bin/python script/eval_geometry.py --model_path <dir> --limit 100
"""
import argparse, json, re, torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText


def extract_answer(text):
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""


def ans_eq(p, g):
    p, g = str(p).strip().rstrip("."), str(g).strip().rstrip(".")
    if p.lower() == g.lower():
        return True
    try:
        return abs(float(p) - float(g)) < 1e-3
    except (ValueError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-VL-4B-Instruct", help="for LoRA adapter dirs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import os
    is_adapter = os.path.exists(os.path.join(a.model_path, "adapter_config.json"))
    src = a.base_model if is_adapter else a.model_path
    proc = AutoProcessor.from_pretrained(src)
    model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda:0")
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.model_path)
    model.eval()

    ds = load_dataset("hiyouga/geometry3k")[a.split]
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    correct = 0
    for ex in ds:
        q = re.sub(r"<image>", "", ex["problem"]).strip() + " Reason step by step, then give the final answer in \\boxed{}."
        img = ex["images"][0].convert("RGB").resize((448, 448))   # match training (uniform grid → mrope-safe)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        txt = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += ans_eq(extract_answer(txt), str(ex["answer"]))
    acc = correct / len(ds)
    print(f"[geo-eval] {a.model_path} split={a.split} acc={acc:.4f} ({correct}/{len(ds)})")
    if a.out:
        json.dump({"model": a.model_path, "split": a.split, "n": len(ds), "acc": acc}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
