# Setup — SkyRL Terminal-Bench & Vision GRPO (1× A100-80GB, no Docker)

From-scratch reproduction of the environment used in
`../SkyRL_Terminal_and_Vision_GRPO.ipynb`.

## 0. Host facts this setup assumes
- Unprivileged container (no `CAP_SYS_ADMIN`, user namespaces blocked → **no Docker / rootless / bubblewrap**).
- 1× A100-80GB, CUDA 12.8 driver, Python 3.12, `uv` available.

## 1. Clone SkyRL and the Terminal-Bench task source
```bash
cd /home/claudeuser
git clone https://github.com/novasky-ai/SkyRL.git
git clone https://github.com/laude-institute/terminal-bench.git   # 241 tasks in original-tasks/
```

## 2. Install SkyRL (FSDP + vLLM backend)
```bash
cd /home/claudeuser/SkyRL
uv sync --extra fsdp          # torch 2.11+cu128, vllm 0.20.2, transformers, ray, flash-attn
.venv/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"
```

## 3. The local sandbox toolchain (the Docker replacement)
```bash
sudo apt-get install -y proot            # userspace ptrace chroot — needs zero privilege
python3 -m venv /home/claudeuser/tbench-venv
/home/claudeuser/tbench-venv/bin/pip install pytest pyyaml      # verifier interpreter
```

## 4. Register the custom `terminal` SkyRL-gym env
Already in this repo's companion paths:
- `SkyRL/skyrl-gym/skyrl_gym/envs/terminal/{sandbox.py,env.py,__init__.py}`
- one line added to `SkyRL/skyrl-gym/skyrl_gym/envs/__init__.py` registering `id="terminal"`.

## 5. Curate tasks + build the dataset
```bash
cd /home/claudeuser/arl/skyrl_terminal
/home/claudeuser/tbench-venv/bin/python curate_tasks.py --workers 48        # -> local_tasks.json (32 kept)
/home/claudeuser/SkyRL/.venv/bin/python build_dataset.py                    # -> ~/data/terminal_bench/*.parquet
```

## 6. Train
```bash
bash run_terminal_grpo.sh          # Goal 1: Terminal-Bench GRPO (Qwen2.5-Coder-3B)
bash run_geo3k_1gpu.sh             # Goal 2: Vision GRPO (Qwen3-VL-8B + LoRA), dataset auto-generates
```

## Notes / knobs
- Sandbox timeouts: `TBENCH_EXEC_TIMEOUT` / `TBENCH_VERIFY_TIMEOUT` env vars override the dataset values
  (kills runaway agent scripts fast → shorter GRPO steps).
- `TBENCH_VERIFIER_PYTHON` points the verifier at the pytest venv (default `/home/claudeuser/tbench-venv/bin/python`).
- Single-GPU colocated GRPO: vLLM and FSDP share the A100 via sleep/wake (`trainer.placement.colocate_all=true`).
