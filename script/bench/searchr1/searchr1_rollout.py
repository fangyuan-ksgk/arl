"""SearchR1 multi-turn rollout for TRL 1.7 GRPOTrainer via the `rollout_func` API.

Why rollout_func (not the `tools` API): the SearchR1 protocol is raw-text tags
(<search>q</search> -> <information>docs</information> -> <answer>a</answer>)
inside a SINGLE assistant turn, exactly as trained in Search-R1/SkyRL
(use_conversation_multi_turn=false). TRL's `tools` path instead requires native
chat-template tool calls (parse_response + response_schema) and injects results
as role="tool" messages, which changes the protocol. rollout_func gives us the
token-level control we need and still supports loss masking: the returned
`env_mask` (1 = model-generated token, 0 = injected environment token) is
treated by GRPOTrainer exactly like its internal tool_mask — applied to the
loss (loss_mask = completion_mask * tool_mask), to the vLLM importance-sampling
correction, and to completion-length metrics.

Contract (grpo_trainer.py `_generate`): rollout_func(prompts, trainer) must
return {"prompt_ids", "completion_ids", "logprobs"}; extra keys (env_mask,
num_search_calls, ...) are forwarded to reward functions. The trainer syncs
vLLM weights BEFORE calling rollout_func, so we can use
trainer.vllm_generation directly (colocate mode).

Loop (max_turns generation rounds, SkyRL parity):
  gen -> if <answer>...</answer> or no <search> tag: done
      -> else retrieve, append "\n<information>Doc 1: ...</information>\n"
         as masked tokens (env_mask=0, logprob=0.0) and generate again.
Stop strings ["</search>", "</answer>"] must be configured with
include_stop_str_in_output=True (GRPOConfig.generation_kwargs for vLLM; the HF
fallback here passes stop_strings itself).
"""

import re

import torch

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
STOP_STRINGS = ["</search>", "</answer>"]
INFORMATION_TEMPLATE = "\n<information>{obs}</information>\n"
# Don't inject an information block unless at least this many tokens of the
# completion budget remain for the model to keep generating afterwards.
MIN_TOKENS_AFTER_INFO = 16


def parse_search_query(text: str):
    matches = SEARCH_RE.findall(text)
    return matches[-1].strip() if matches else None


def has_answer(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text


def _generate_turn(trainer, prompt_ids_list, max_new_tokens):
    """One generation round for a list of token-id prompts.

    Returns (completion_ids: list[list[int]], logprobs: list[list[float]] | None).
    Uses the trainer's colocated vLLM engine when available, else falls back to
    HF `model.generate` (CPU smoke tests / debugging).
    """
    if getattr(trainer, "use_vllm", False):
        vg = trainer.vllm_generation
        orig_max = vg.max_completion_length
        try:
            # VLLMGeneration reads self.max_completion_length at call time; override
            # per turn so one turn can't eat the whole completion budget.
            vg.max_completion_length = max_new_tokens
            _, completion_ids, logprobs, _ = vg.generate(
                prompts=prompt_ids_list, images=None, num_generations=1
            )
        finally:
            vg.max_completion_length = orig_max
        if logprobs is not None:
            # per-token top-k -> keep top-1 (the sampled token), as grpo_trainer does
            logprobs = [[lp[0] if lp[0] is not None else 0.0 for lp in seq] for seq in logprobs]
        return [list(ids) for ids in completion_ids], logprobs

    # ---- HF fallback (no vLLM: CPU smoke test) ----
    tok = trainer.processing_class
    model = trainer.accelerator.unwrap_model(trainer.model) if hasattr(trainer, "accelerator") else trainer.model
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    max_len = max(len(ids) for ids in prompt_ids_list)
    input_ids = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompt_ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(prompt_ids_list):  # left-pad
        input_ids[i, max_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - len(ids):] = 1

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=trainer.args.temperature,
            top_p=trainer.args.top_p,
            pad_token_id=pad_id,
            stop_strings=list(STOP_STRINGS),
            tokenizer=tok,
        )
    if was_training:
        model.train()

    completion_ids = []
    for row in out[:, max_len:].tolist():
        ids = []
        for t in row:
            ids.append(t)
            if t == tok.eos_token_id:
                break
        # strip right-padding of rows that finished early (e.g. via stop string)
        while ids and ids[-1] == pad_id and ids[-1] != tok.eos_token_id:
            ids.pop()
        completion_ids.append(ids)
    return completion_ids, None  # no sampling logprobs on the HF path (only needed for vLLM IS correction)


def make_rollout_func(retrieve_fn, max_turns: int = 4, per_turn_max_tokens: int = 500):
    """Build a GRPOTrainer rollout_func.

    Args:
        retrieve_fn: callable(query: str) -> formatted passages string
            ("Doc 1: ...\n..."). Exceptions are caught and surfaced to the
            model inside the <information> block.
        max_turns: max generation rounds (assistant actions), SkyRL parity = 4.
        per_turn_max_tokens: vLLM max_tokens per round (SkyRL max_generate_length).
    """

    def rollout_func(prompts, trainer):
        tok = trainer.processing_class
        # config lives on trainer.args (GRPOConfig) — the bare-trainer getattr
        # silently returned {} and left thinking ON in this custom path (07-08)
        ct_kwargs = (getattr(trainer, "chat_template_kwargs", None)
                     or getattr(getattr(trainer, "args", None), "chat_template_kwargs", None)
                     or {})
        max_completion = trainer.max_completion_length or 2048

        prompt_ids = [
            tok.apply_chat_template(
                p, add_generation_prompt=True, tokenize=True, return_dict=False, **ct_kwargs
            )
            for p in prompts
        ]
        n = len(prompt_ids)
        completion_ids = [[] for _ in range(n)]
        env_mask = [[] for _ in range(n)]
        logprobs = [[] for _ in range(n)]
        num_search_calls = [0] * n
        search_failures = [0] * n
        got_logprobs = False

        active = list(range(n))
        for turn in range(max_turns):
            if not active:
                break
            budgets = {i: max_completion - len(completion_ids[i]) for i in active}
            active = [i for i in active if budgets[i] > 0]
            if not active:
                break
            max_new = min(per_turn_max_tokens, max(budgets[i] for i in active))

            gen_prompts = [prompt_ids[i] + completion_ids[i] for i in active]
            new_ids, new_lps = _generate_turn(trainer, gen_prompts, max_new)
            if new_lps is not None:
                got_logprobs = True

            next_active = []
            for j, i in enumerate(active):
                ids = new_ids[j]
                lps = new_lps[j] if new_lps is not None else [0.0] * len(ids)
                if len(ids) > budgets[i]:  # enforce the per-sample completion budget
                    ids, lps = ids[: budgets[i]], lps[: budgets[i]]
                completion_ids[i] += ids
                env_mask[i] += [1] * len(ids)
                logprobs[i] += lps

                text = tok.decode(ids, skip_special_tokens=True)
                if has_answer(text):
                    continue  # final answer given
                query = parse_search_query(text)
                if query is None:
                    continue  # neither answer nor search -> episode ends
                if turn == max_turns - 1:
                    continue  # no generation round left to consume the docs

                num_search_calls[i] += 1
                try:
                    obs = retrieve_fn(query)
                except Exception as e:  # noqa: BLE001 - surface retrieval errors to the model
                    search_failures[i] += 1
                    obs = f"Search error: {e}"
                info_text = INFORMATION_TEMPLATE.format(obs=obs)
                info_ids = tok(info_text, add_special_tokens=False)["input_ids"]

                remaining = max_completion - len(completion_ids[i])
                if len(info_ids) + MIN_TOKENS_AFTER_INFO > remaining:
                    continue  # docs don't fit -> stop here (mirrors TRL's overlong rollback)

                # Inject retrieved docs as ENVIRONMENT tokens: masked out of the
                # loss (env_mask=0) and given placeholder logprob 0.0, exactly
                # like grpo_trainer._tool_call_loop does for tool results.
                completion_ids[i] += info_ids
                env_mask[i] += [0] * len(info_ids)
                logprobs[i] += [0.0] * len(info_ids)
                next_active.append(i)
            active = next_active

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            # Only meaningful on the vLLM path (used for importance-sampling correction).
            "logprobs": logprobs if got_logprobs else None,
            "env_mask": env_mask,
            "num_search_calls": num_search_calls,
            "search_failures": search_failures,
        }

    return rollout_func
