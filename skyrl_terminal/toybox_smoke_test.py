"""
No-GPU smoke test for the ToyBox env: drive every task with a 'perfect agent'
action and assert reward == 1.0, plus a couple of negative cases. Proves the
env + sandbox + checkers pipeline end-to-end without a model or GPU.

    /home/claudeuser/SkyRL/.venv/bin/python arl/skyrl_terminal/toybox_smoke_test.py
"""

import skyrl_gym
from skyrl_gym.envs.toybox import tasks as toytasks

# A correct action for each task id (single-turn solutions).
SOLUTIONS = {
    "fizzbuzz_dungeon": (
        "Here's the incantation:\n```python\n"
        "for i in range(1, 16):\n"
        "    if i % 15 == 0: print('FizzBuzz')\n"
        "    elif i % 3 == 0: print('Fizz')\n"
        "    elif i % 5 == 0: print('Buzz')\n"
        "    else: print(i)\n```\nTASK_COMPLETE"
    ),
    "prime_oracle": (
        "```python\n"
        "def is_prime(n):\n"
        "    if n < 2: return False\n"
        "    i = 2\n"
        "    while i * i <= n:\n"
        "        if n % i == 0: return False\n"
        "        i += 1\n"
        "    return True\n```\nTASK_COMPLETE"
    ),
    "caesar_crypt": "ROT13 of 'qentba' is <answer>dragon</answer>",
    "base64_treasure": (
        "```python\nimport base64\nprint(base64.b64decode('Z3JpZmZpbg==').decode())\n```\n"
        "So the word is <answer>griffin</answer>"
    ),
    "treasure_hunt": (
        "Let me look around:\n```bash\nls -la && cat .vault/treasure.txt\n```\n"
        "Found it! <answer>moonstone</answer>"
    ),
    "fibonacci_forge": (
        "```python\n"
        "def fib(n):\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n): a, b = b, a + b\n"
        "    return a\n```\nTASK_COMPLETE"
    ),
    "word_count_wizard": (
        "```python\n"
        "n = len(open('poem.txt').read().split())\n"
        "open('count.txt', 'w').write(str(n))\n"
        "print('wrote', n)\n```\nTASK_COMPLETE"
    ),
    "reverse_runes": "Reversed: <answer>xobyot</answer>",
    "gcd_golem": (
        "```python\n"
        "def gcd(a, b):\n"
        "    while b: a, b = b, a % b\n"
        "    return a\n```\nTASK_COMPLETE"
    ),
    "sum_the_loot": (
        "```bash\npaste -sd+ loot.txt | bc\n```\nThe total is <answer>200</answer>"
    ),
    "json_jester": (
        "```python\n"
        "import json\n"
        "d = json.load(open('clue.json'))\n"
        "open('answer.txt', 'w').write(d['secret'])\n"
        "print('done')\n```\nTASK_COMPLETE"
    ),
    "anagram_altar": (
        "```python\n"
        "def is_anagram(a, b):\n"
        "    norm = lambda s: sorted(s.replace(' ', '').lower())\n"
        "    return norm(a) == norm(b)\n```\nTASK_COMPLETE"
    ),
}


def make_env(task_id):
    return skyrl_gym.make("toybox", env_config={}, extras={"extra_info": {"task_id": task_id}, "max_turns": 4})


def run_solution(task_id, action):
    env = make_env(task_id)
    try:
        env.init([{"role": "user", "content": "go"}])
        out = env.step(action)
        return out["reward"], out["metadata"]
    finally:
        env.close()


def main():
    ids = [t["id"] for t in toytasks.TASKS]
    missing = [i for i in ids if i not in SOLUTIONS]
    assert not missing, f"no solution provided for: {missing}"

    print(f"ToyBox smoke test — {len(ids)} tasks\n" + "=" * 48)
    failures = []
    for tid in ids:
        reward, meta = run_solution(tid, SOLUTIONS[tid])
        ok = reward >= 1.0 and meta["all_passed"]
        mark = "✅" if ok else "❌"
        print(f"{mark} {tid:18s} reward={reward:.2f}  {meta['n_passed']}/{meta['n_total']} checks")
        if not ok:
            failures.append((tid, reward, meta))

    print("-" * 48)
    # Negative controls: wrong/empty answers must NOT score 1.0.
    neg_r, _ = run_solution("caesar_crypt", "<answer>wrongword</answer>")
    print(f"{'✅' if neg_r < 1.0 else '❌'} negative caesar_crypt reward={neg_r:.2f} (expect <1)")
    empty_r, _ = run_solution("prime_oracle", "I am not sure how to do this.")
    print(f"{'✅' if empty_r < 1.0 else '❌'} negative prime_oracle(no code) reward={empty_r:.2f} (expect <1)")
    # Partial-credit control: a buggy is_prime should land strictly between 0 and 1.
    buggy = "```python\ndef is_prime(n):\n    return n % 2 == 1\n```\nTASK_COMPLETE"  # wrong for 1, 9, 2 ...
    part_r, part_m = run_solution("prime_oracle", buggy)
    print(f"{'✅' if 0 < part_r < 1.0 else '❌'} partial prime_oracle(buggy) reward={part_r:.2f} "
          f"({part_m['n_passed']}/{part_m['n_total']}) (expect strictly 0<r<1)")

    if failures or not (neg_r < 1.0 and empty_r < 1.0 and 0 < part_r < 1.0):
        print("\nSMOKE TEST FAILED")
        raise SystemExit(1)
    print("\nALL TASKS SOLVED + CONTROLS PASSED 🎉")


if __name__ == "__main__":
    main()
