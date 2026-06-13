# ToyBox 🎮 — a fun agentic GRPO playground

A small, self-contained multi-turn RL environment where a tiny model acts as a
coding agent solving themed puzzles. It's the "toyish" cousin of the
Terminal-Bench env: same agentic loop (act → observe → iterate), but the tasks
are bite-sized, the rewards are instant and self-verifying, and there are no
Docker/proot dependencies.

## How it works
Each turn the model can emit fenced code blocks that run in a throwaway sandbox
dir; their output is fed back as the next observation:
- ```` ```python ... ``` ```` — runs a Python snippet (latest saved as `solution.py`)
- ```` ```bash ... ``` ```` — runs a shell command

It finishes a task by submitting `<answer>...</answer>` or writing `TASK_COMPLETE`
(or when `max_turns` is hit). Reward = **mean of the task's checks in [0, 1]**
(partial credit → denser GRPO signal than all-or-nothing).

## The task pack (12 puzzles)
| theme | id | skill | check kind |
|---|---|---|---|
| 🏰 | fizzbuzz_dungeon | print FizzBuzz 1–15 | stdout_equals |
| 🔮 | prime_oracle | write `is_prime` | pyfunc |
| 🗝️ | caesar_crypt | decode ROT13 | answer_equals |
| 💰 | base64_treasure | decode base64 | answer_equals |
| 🧭 | treasure_hunt | find hidden `.vault/` file | answer_equals (bash) |
| 🔥 | fibonacci_forge | write `fib` | pyfunc |
| 📜 | word_count_wizard | count words → `count.txt` | file_equals |
| 🪞 | reverse_runes | reverse a string | answer_equals |
| 🗿 | gcd_golem | write `gcd` | pyfunc |
| 🪙 | sum_the_loot | sum a file of ints | answer_numeric |
| 🃏 | json_jester | extract a JSON key → file | file_equals |
| ⚖️ | anagram_altar | write `is_anagram` | pyfunc |

Tasks live in `SkyRL/skyrl-gym/skyrl_gym/envs/toybox/tasks.py` — add one by
appending a dict with a `prompt` and a list of `checks`. Supported check kinds:
`answer_equals`, `answer_numeric`, `stdout_equals`, `stdout_contains`,
`file_exists`, `file_equals`, `file_contains`, `pyfunc`.

## Files
- Env: `SkyRL/skyrl-gym/skyrl_gym/envs/toybox/{env.py,tasks.py,sandbox.py,__init__.py}`,
  registered as `id="toybox"`.
- Dataset builder: `build_toybox_dataset.py` → `~/data/toybox/{train,validation}.parquet`.
- Run script: `run_toybox_grpo.sh` (1 GPU, defaults to Qwen2.5-1.5B-Instruct).
- Offline smoke test: `toybox_smoke_test.py` (drives all 12 tasks with perfect
  solutions + negative/partial controls — no GPU/model needed).

## Quick start
```bash
# 1. Sanity-check the env + checkers (no GPU):
/home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/toybox_smoke_test.py

# 2. Build the dataset (4x repeat of the 12-task pack = 48 train rows):
/home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/build_toybox_dataset.py \
    --output_dir ~/data/toybox --repeat 4

# 3. Train (single A100; ~multi-turn GRPO):
bash arl/skyrl_terminal/run_toybox_grpo.sh
#    knobs: MODEL_PATH= EPOCHS= TRAIN_BS= N_SAMPLES= MAX_TURNS= LR=
```

> Note: the sandbox runs agent code as the host user (like the terminal env) — a
> toy execution boundary (throwaway CWD, scrubbed env, hard timeout), not a
> security sandbox. Fine for a trusted single-user research box.
