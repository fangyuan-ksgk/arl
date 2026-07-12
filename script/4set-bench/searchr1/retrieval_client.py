"""Thin client for mini_retrieval_server.py (arl/skyrl_terminal/).

The server answers POST /retrieve {"query", "topk", "return_scores"} with
{"result": [[{"document": {"id", "contents"}, "score": float}, ...]]} — a list
of PER-QUERY doc lists (result[0] = docs for our single query).

`retrieve()` returns the docs formatted exactly like SkyRL's search tool
(`_passages2string`): "Doc 1: <contents>\nDoc 2: ...\n".
"""

import requests

DEFAULT_URL = "http://127.0.0.1:8000/retrieve"

_session = requests.Session()
_session.mount("http://", requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16))


def passages_to_string(docs) -> str:
    out = ""
    for idx, doc_item in enumerate(docs):
        content = doc_item["document"]["contents"].strip()
        out += f"Doc {idx + 1}: {content}\n"
    return out


def retrieve(query: str, url: str = DEFAULT_URL, topk: int = 3, timeout: int = 30) -> str:
    resp = _session.post(
        url,
        json={"query": query, "topk": topk, "return_scores": True},
        timeout=timeout,
    )
    resp.raise_for_status()
    result = resp.json()["result"]  # list of per-query doc lists
    return passages_to_string(result[0])


def make_retrieve_fn(url: str = DEFAULT_URL, topk: int = 3, timeout: int = 30):
    """Returns retrieve_fn(query) -> formatted passage string, for the rollout loop."""

    def retrieve_fn(query: str) -> str:
        return retrieve(query, url=url, topk=topk, timeout=timeout)

    return retrieve_fn
