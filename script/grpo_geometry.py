"""R3: GRPO on Geometry3K (+ GeoQA) for a Qwen3.5-VL model (multimodal). vLLM 0.11 can't generate
qwen3_5, so rollouts use HF generation (use_vllm=False). RUN IN THE QWEN3.5 VENV (transformers 5.12.1):
    /home/claudeuser/venv_qwen35/bin/python script/grpo_geometry.py --model Qwen/Qwen3.5-4B ...
"""
import argparse, re, torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import GRPOTrainer as _GRPOTrainer, GRPOConfig
from peft import LoraConfig


class GRPOTrainer(_GRPOTrainer):
    """Propagate mm_token_type_ids for Qwen3.5-VL 3D-mrope (TRL doesn't). Ported from vlm_train_grpo.py."""
    def _build_mm_token_type_ids(self, input_ids):
        tid = getattr(self.processing_class, "image_token_id", None)
        if tid is None:
            return None
        m = torch.zeros_like(input_ids); m[input_ids == tid] = 1
        vid = getattr(self.processing_class, "video_token_id", None)
        if vid is not None:
            m[input_ids == vid] = 2
        return m

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if "pixel_values" in out and "mm_token_type_ids" not in out:
            ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
            mm = self._build_mm_token_type_ids(ids)
            if mm is not None:
                out["mm_token_type_ids"] = mm
        return out

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep,
                                           batch_size=None, compute_entropy=False, **kwargs):
        if kwargs.get("pixel_values") is not None and "mm_token_type_ids" not in kwargs:
            mm = self._build_mm_token_type_ids(input_ids)
            if mm is not None:
                kwargs["mm_token_type_ids"] = mm
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep,
            batch_size=batch_size, compute_entropy=compute_entropy, **kwargs)


SUFFIX = " Reason step by step, then give the final answer in \\boxed{}."


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


def correctness_reward(completions, gold_answer, **kwargs):
    res = []
    for comp, gold in zip(completions, gold_answer):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        res.append(1.0 if ans_eq(extract_answer(text), gold) else 0.0)
    return res


def _fixed_image(img, side=448):
    """Resize to a fixed square so every example yields a uniform image-token grid — sidesteps the
    qwen3_5 get_rope_index shape-mismatch that variable image sizes trigger in the training forward."""
    return img.convert("RGB").resize((side, side))


def load_geo(use_geoqa, n_train):
    def fmt(problem, image, answer):
        return {"prompt": [{"role": "user", "content": [{"type": "image"},
                {"type": "text", "text": re.sub(r"<image>", "", problem).strip() + SUFFIX}]}],
                "image": _fixed_image(image), "gold_answer": str(answer)}
    g = load_dataset("hiyouga/geometry3k")["train"]
    g = g.map(lambda ex: fmt(ex["problem"], ex["images"][0], ex["answer"]),
              remove_columns=g.column_names)
    parts = [g]
    if use_geoqa:
        q = load_dataset("leonardPKU/GEOQA_R1V_Train_8K")["train"]
        cols = q.column_names
        ic = "image" if "image" in cols else ("images" if "images" in cols else None)
        qc = "problem" if "problem" in cols else ("question" if "question" in cols else None)
        ac = "answer" if "answer" in cols else ("solution" if "solution" in cols else None)
        if ic and qc and ac:
            def fmtq(ex):
                img = ex[ic][0] if isinstance(ex[ic], list) else ex[ic]
                return fmt(ex[qc], img, ex[ac])
            parts.append(q.map(fmtq, remove_columns=cols))
    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    ds = ds.shuffle(seed=0)
    if n_train:
        ds = ds.select(range(min(n_train, len(ds))))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--output_dir", default="output/geo_qwen35_4b")
    ap.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=True,
                    help="vLLM colocate rollouts (vLLM 0.24 has NATIVE qwen3_5 support).")
    ap.add_argument("--vllm_model_impl", default="vllm", choices=["vllm", "transformers"],
                    help="vLLM 0.24 native qwen3_5 path; 'transformers' is the belt-and-suspenders fallback.")
    ap.add_argument("--vllm_gpu_mem", type=float, default=0.40)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_completion_length", type=int, default=1024)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--no_geoqa", action="store_true")
    a = ap.parse_args()

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16)
    ds = load_geo(not a.no_geoqa, a.n_train or None)
    print(f"[geo] train={len(ds)} | sample gold={ds[0]['gold_answer']}", flush=True)
    lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    cfg = GRPOConfig(
        output_dir=a.output_dir, num_generations=a.num_generations,
        max_completion_length=a.max_completion_length, per_device_train_batch_size=a.num_generations,
        gradient_accumulation_steps=a.grad_accum, max_steps=a.max_steps, learning_rate=a.learning_rate,
        logging_steps=5, save_strategy="no", loss_type="dr_grpo", scale_rewards="none", beta=0.0,
        use_vllm=a.use_vllm, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_gpu_mem,
        vllm_model_impl=a.vllm_model_impl,            # native qwen3_5 path (vLLM 0.24)
        bf16=True, gradient_checkpointing=True, report_to="none")
    trainer = GRPOTrainer(model=model, reward_funcs=[correctness_reward], args=cfg,
                          train_dataset=ds, processing_class=proc, peft_config=lora)
    trainer.train()
    trainer.save_model(a.output_dir)
    print(f"[geo] done -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
