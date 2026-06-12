#!/usr/bin/env bash
# One-command reproduction of the scaled-down SearchR1 result on 1 A100.
#   bash reproduce_searchr1.sh
# Does: (1) build SQuAD->SearchR1 dataset + 2.3k-passage corpus, (2) start the
# faiss-free e5 retrieval server on :8000 and wait for it, (3) run 1-GPU multi-turn
# search-agent GRPO. Expect pass@1 ~0.26 (baseline) -> ~0.31 after ~31 steps.
set -uo pipefail

# always run as claudeuser (root hits the /workspace uv-cache quota on this box)
if [ "$(id -u)" = "0" ]; then exec su claudeuser -c "bash $(printf '%q' "$0") $(printf '%q ' "$@")"; fi
export HOME=/home/claudeuser
PY=/home/claudeuser/SkyRL/.venv/bin/python
HERE=/home/claudeuser/arl/skyrl_terminal
DATA=/home/claudeuser/data/searchR1_mini

# (1) dataset + corpus (skip if already built)
if [ ! -f "$DATA/train.parquet" ]; then
  echo ">> [1/3] building SQuAD->SearchR1 dataset + corpus ..."
  "$PY" "$HERE/build_searchr1_mini.py" --n_train 2000 --n_val 300
fi

# (2) retrieval server on :8000 (start if not already listening)
if ! (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
  echo ">> [2/3] starting e5 retrieval server on :8000 ..."
  setsid "$PY" "$HERE/mini_retrieval_server.py" --corpus "$DATA/corpus.jsonl" --port 8000 \
    > /tmp/mini_retriever.log 2>&1 < /dev/null &
  echo "   (waiting for it to embed the corpus + bind; first run ~1 min, then cached)"
  until grep -q "ready: serving" /tmp/mini_retriever.log 2>/dev/null; do sleep 3; done
fi
echo ">> retriever ready on :8000"

# (3) train
echo ">> [3/3] launching 1-GPU SearchR1 GRPO (Qwen2.5-3B + LoRA, multi-turn) ..."
exec bash "$HERE/run_search_1gpu.sh" "$@"
