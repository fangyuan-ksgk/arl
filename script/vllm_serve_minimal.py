"""Minimal vLLM HTTP server with NCCL-based weight updates.

No TRL dependency. Exposes:
    GET  /health/
    POST /init_communicator/   {host, port, world_size, client_device_uuid}
    POST /update_named_param/  {name, dtype, shape}
    POST /close_communicator/
    POST /generate/            {prompts, max_tokens, temperature, top_p, n}

Run as:
    CUDA_VISIBLE_DEVICES=0 NCCL_CUMEM_ENABLE=0 \
        python script/vllm_serve_minimal.py \
            --model Qwen/Qwen3-0.6B --port 8000 --enforce-eager
"""
from __future__ import annotations

import argparse
import os
import sys

# Required so vLLM workers can be spawned with CUDA initialized in the parent.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Make sure the directory containing THIS file is importable both in the
# parent process and in any spawned EngineCore subprocesses, so that
# `worker_extension_cls="vllm_serve_minimal.WeightSyncExt"` resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ["PYTHONPATH"] = (
    _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")
).rstrip(os.pathsep)

from typing import List, Optional

import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# vLLM worker extension. One instance per worker process; `self.device` and
# `self.model_runner` are injected by vLLM at construction time.
# ---------------------------------------------------------------------------
class WeightSyncExt:
    communicator = None
    client_rank = None

    def init_communicator(
        self,
        host: str,
        port: int,
        world_size: int,
        client_device_uuid: str,
    ) -> None:
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup

        my_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        if client_device_uuid == my_uuid:
            raise RuntimeError(
                f"Client and vLLM worker share the same GPU (uuid={my_uuid}). "
                "Place them on different devices via CUDA_VISIBLE_DEVICES."
            )

        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size,
        )
        self.communicator = PyNcclCommunicator(pg, device=self.device)
        # The client (the trainer) is the highest rank in the group.
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape) -> None:
        import torch

        if self.communicator is None:
            raise RuntimeError("init_communicator must be called first")
        dt = getattr(torch, dtype.split(".")[-1])
        weight = torch.empty(shape, dtype=dt, device=self.device)
        # NCCL broadcast from client into the worker.
        self.communicator.broadcast(weight, src=self.client_rank)
        self.communicator.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        if self.communicator is not None:
            del self.communicator
            self.communicator = None
            self.client_rank = None


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
def build_app(llm: LLM) -> FastAPI:
    app = FastAPI()

    class InitReq(BaseModel):
        host: str
        port: int
        world_size: int
        client_device_uuid: str

    class UpdateReq(BaseModel):
        name: str
        dtype: str
        shape: List[int]

    class GenReq(BaseModel):
        prompts: List[str]
        max_tokens: int = 32
        temperature: float = 0.0
        top_p: float = 1.0
        n: int = 1

    @app.get("/health/")
    def health():
        return {"status": "ok"}

    @app.post("/init_communicator/")
    def init(req: InitReq = Body(...)):
        llm.collective_rpc(
            "init_communicator",
            args=(req.host, req.port, req.world_size, req.client_device_uuid),
        )
        return {"status": "ok"}

    @app.post("/update_named_param/")
    def update(req: UpdateReq = Body(...)):
        llm.collective_rpc(
            "update_named_param",
            args=(req.name, req.dtype, tuple(req.shape)),
        )
        return {"status": "ok"}

    @app.post("/close_communicator/")
    def close():
        llm.collective_rpc("close_communicator")
        return {"status": "ok"}

    @app.post("/generate/")
    def generate(req: GenReq = Body(...)):
        sp = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            n=req.n,
        )
        outs = llm.generate(req.prompts, sp)
        return {
            "prompt_ids": [list(o.prompt_token_ids) for o in outs],
            "completion_ids": [
                [list(c.token_ids) for c in o.outputs] for o in outs
            ],
            "completion_text": [
                [c.text for c in o.outputs] for o in outs
            ],
        }

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        # NOTE: the dotted path must be importable from the worker process.
        # When invoking via `python script/vllm_serve_minimal.py ...` from the
        # repo root, the `script` package is on PYTHONPATH automatically.
        worker_extension_cls="vllm_serve_minimal.WeightSyncExt",
    )

    uvicorn.run(build_app(llm), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
