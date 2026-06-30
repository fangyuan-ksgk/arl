"""R4: Dr.GRPO on textarena GuessTheNumber-v0 — a self-contained multi-turn agentic benchmark (no API).
Runs in the MAIN env (Qwen3-1.7B, vLLM colocate — fast). Training states are produced by playing games
with a random-in-range agent (target + feasible range are read from env.state); the reward verifies the
model's guess against the target and the feedback-implied range. Eval win-rate via eval_textarena.py.

    python script/grpo_guessnumber.py --model Qwen/Qwen3-1.7B --max_steps 120
"""
import argparse, random, re
import torch
import textarena as ta
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

INSTR = ("\n\nReason briefly, then respond with ONLY your guess inside brackets on the last line "
         "(e.g. [7]). Use the 'higher'/'lower' hints to narrow the range. /no_think")


def feasible_range(history, lo=1, hi=20):
    for g, fb in history:
        if "higher" in fb:
            lo = max(lo, g + 1)
        elif "lower" in fb:
            hi = min(hi, g - 1)
    return lo, hi


def gen_states(n_games, seed0):
    rows = []
    for gi in range(n_games):
        env = ta.make("GuessTheNumber-v0")
        env.reset(num_players=1, seed=seed0 + gi)
        done, turns = False, 0
        while not done and turns < 8:
            _pid, obs = env.get_observation()
            gs = env.state.game_state
            lo, hi = feasible_range(gs["guess_history"])
            rows.append({"prompt": [{"role": "user", "content": str(obs) + INSTR}],
                         "target": int(gs["game_number"]), "lo": int(lo), "hi": int(hi)})
            guess = random.randint(lo, hi)            # random valid agent to advance the game
            try:
                done, _ = env.step(action=f"[{guess}]")
            except Exception:
                done = True
            turns += 1
    random.shuffle(rows)
    return Dataset.from_list(rows)


def extract_guess(text):
    m = re.findall(r"\[(-?\d+)\]", text)
    return int(m[-1]) if m else None


def correctness_reward(completions, target, lo, hi, **kwargs):
    out = []
    for comp, t, l, h in zip(completions, target, lo, hi):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        g = extract_guess(text)
        if g is None:
            out.append(-1.0)              # no valid bracketed guess
        elif g == t:
            out.append(1.0)               # win
        elif l <= g <= h:
            out.append(0.4)               # consistent with feedback (uses the hints)
        else:
            out.append(-0.5)              # contradicts the feedback
    return out


class VirtualRolloutGRPOTrainer(GRPOTrainer):
    """GRPOTrainer + virtual-rollout advantage shaping (same gadget as grpo_math/grpo_code; complements
    MBE velocity). No-op when virtual_rollout_mode is None. See src/arsenal.virtual_rollout_advantages."""
    def _calculate_rewards(self, *args, **kwargs):
        rpf = super()._calculate_rewards(*args, **kwargs)
        self._last_rewards_per_func = rpf
        return rpf

    def _local_rewards_per_func(self, out):
        rpf = getattr(self, "_last_rewards_per_func", None)
        adv = out.get("advantages")
        if rpf is None or adv is None:
            return None
        Bp = adv.shape[0]
        lo = self.accelerator.process_index * Bp
        return rpf[lo:lo + Bp]

    def _virtual_rollout_advantages(self, out, local):
        from src.arsenal import virtual_rollout_advantages
        adv = out.get("advantages")
        names = self.reward_func_names
        rewards = local.sum(dim=1)
        if "correctness_reward" in names:
            corrects = (local[:, names.index("correctness_reward")] == 1.0)
        else:
            corrects = torch.zeros_like(rewards, dtype=torch.bool)
        return virtual_rollout_advantages(
            rewards, corrects, self.num_generations,
            max_reward=getattr(self, "virtual_max_reward", 1.2),
            mode=self.virtual_rollout_mode).to(adv)

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        if getattr(self, "virtual_rollout_mode", None) and self.model.training:
            local = self._local_rewards_per_func(out)
            if local is not None and out.get("advantages") is not None:
                out["advantages"] = self._virtual_rollout_advantages(out, local)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--output_dir", default="output/guess_qwen3_1p7b")
    ap.add_argument("--n_games", type=int, default=400)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=512)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--vllm_gpu_mem", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    # reward gadgets — same interface as grpo_gsm8k / grpo_math / grpo_code (so local_sgd_grpo passes them uniformly)
    ap.add_argument("--mbe_velocity_reward", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mbe_velocity_scale", type=float, default=5.0)
    ap.add_argument("--mbe_velocity_clip", type=float, default=1.0)
    ap.add_argument("--mbe_velocity_stride", type=int, default=8)
    ap.add_argument("--mbe_velocity_layers", type=str, default="-1")
    ap.add_argument("--mbe_velocity_mode", type=str, default="trajectory", choices=["trajectory", "rollercoaster"])
    ap.add_argument("--virtual_rollout", type=str, default="none",
                    choices=["none", "insert_max", "insert_min", "insert_max_min",
                             "insert_max_all_incorrect", "insert_max_mixed"])
    ap.add_argument("--virtual_max_reward", type=float, default=1.2)
    a = ap.parse_args()

    random.seed(a.seed)
    ds = gen_states(a.n_games, seed0=1)
    print(f"[guess] train states={len(ds)} | e.g. target={ds[0]['target']} range=[{ds[0]['lo']},{ds[0]['hi']}]", flush=True)
    from transformers import AutoConfig as _AC
    _mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
    _g4 = "gemma4" in (getattr(_AC.from_pretrained(a.model), "model_type", "") or "")
    _tm = [f"{m}.linear" for m in _mods] if _g4 else _mods  # gemma4: inner nn.Linear of Gemma4ClippableLinear
    lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none",
                      target_modules=_tm, task_type="CAUSAL_LM")
    reward_funcs = [correctness_reward]
    mbe_velo_reward_obj = None
    if a.mbe_velocity_reward:
        from src.mbe_reward import MBEVeloReward
        velo_layers = [int(x) for x in a.mbe_velocity_layers.split(",") if x.strip()]
        mbe_velo_reward_obj = MBEVeloReward(
            AutoTokenizer.from_pretrained(a.model),
            layers=velo_layers, stride=a.mbe_velocity_stride,
            scale=a.mbe_velocity_scale, clip=a.mbe_velocity_clip, mode=a.mbe_velocity_mode)
        reward_funcs.append(mbe_velo_reward_obj)
        print(f"MBE velocity reward enabled: scale={a.mbe_velocity_scale}, clip=±{a.mbe_velocity_clip}")
    cfg = GRPOConfig(
        output_dir=a.output_dir, num_generations=a.num_generations,
        max_completion_length=a.max_completion_length, per_device_train_batch_size=a.num_generations,
        gradient_accumulation_steps=a.grad_accum, max_steps=a.max_steps, learning_rate=a.learning_rate,
        lr_scheduler_type="linear", warmup_steps=10, logging_steps=5, save_strategy="no",
        loss_type="dr_grpo", scale_rewards="none", beta=0.0, bf16=True, gradient_checkpointing=True,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_gpu_mem,
        seed=a.seed, report_to="none")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="bfloat16")
    trainer = VirtualRolloutGRPOTrainer(model=model, reward_funcs=reward_funcs, args=cfg,
                                        train_dataset=ds, peft_config=lora)
    trainer.virtual_rollout_mode = None if a.virtual_rollout == "none" else a.virtual_rollout
    trainer.virtual_max_reward = a.virtual_max_reward
    if trainer.virtual_rollout_mode:
        print(f"Virtual-rollout advantage shaping: mode={trainer.virtual_rollout_mode}, max_reward={trainer.virtual_max_reward}")
    if mbe_velo_reward_obj is not None:
        mbe_velo_reward_obj.set_model(trainer.model)
    trainer.train()
    trainer.save_model(a.output_dir)
    print(f"[guess] done -> {a.output_dir}", flush=True)


if __name__ == "__main__":
    main()
