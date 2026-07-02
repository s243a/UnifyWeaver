#!/usr/bin/env python3
"""llm_reranker.py — LLM RERANKER component (distinct from the filing agent).

Takes a query + an ordered candidate list (e.g. from the μ-ranker) and returns the candidates REORDERED
by an LLM's judgement of fit — a pure reranking, no filing decision/action. A filing agent can USE this as a
stage before it decides.

It also reports how far the LLM MOVED the input order (`displacement`, `top1_changed`): the eval signal you
flagged — the less the LLM has to move things, the better the upstream (μ) ranker is doing.

    from llm_reranker import rerank
    out = rerank("quantum entanglement", [{"title": "Physics"}, {"title": "Cooking"}], provider="claude", model="haiku")
    out["reranked"]      # candidates in the LLM's order
    out["displacement"]  # mean |Δrank| normalised to [0,1]; 0 = LLM kept the μ order

Backends via llm_cli (claude=Haiku default / gemini / agy / codex / …). Falls back to the input order if the
LLM is unavailable or its output can't be parsed (displacement=0, parsed=False) — so it never breaks the pipeline.
"""
import re
import sys
from typing import List, Dict, Optional

from llm_cli import call_llm

_PROMPT = """A web bookmark needs to be filed into ONE of the candidate folders below. Rank the candidates from \
BEST to WORST fit for the bookmark.

Bookmark: {query}

Candidate folders:
{numbered}

Reply with ONLY the ranking as a comma-separated list of candidate numbers, best first (e.g. "3,1,2,..."). \
Include every number exactly once. No other text."""


def _parse_order(text: str, n: int) -> Optional[List[int]]:
    """Parse the LLM's '3,1,2' reply → 0-based index order. Tolerant: takes the numbers in order, dedups,
    appends any missing. Returns None if nothing parseable."""
    nums = [int(x) for x in re.findall(r"\d+", text or "")]
    seen, order = set(), []
    for x in nums:
        i = x - 1                                              # candidates are presented 1-based
        if 0 <= i < n and i not in seen:
            seen.add(i); order.append(i)
    if not order:
        return None
    for i in range(n):                                        # append any the LLM dropped, in original order
        if i not in seen:
            order.append(i)
    return order


def rerank(query: str, candidates: List[Dict], provider: str = "claude", model: str = "haiku",
           timeout: int = 60, max_candidates: int = 20) -> Dict:
    """Rerank `candidates` (list of dicts with a 'title') for `query`. → dict:
       reranked (list, LLM order), order (0-based indices), displacement (mean |Δrank|/max, [0,1]),
       top1_changed (bool), parsed (bool), raw (LLM text)."""
    cand = candidates[:max_candidates]
    n = len(cand)
    if n <= 1:
        return {"reranked": list(cand), "order": list(range(n)), "displacement": 0.0,
                "top1_changed": False, "parsed": True, "raw": ""}
    numbered = "\n".join("%d. %s" % (i + 1, c.get("title", c.get("tree_id", "?"))) for i, c in enumerate(cand))
    raw = call_llm(_PROMPT.format(query=query, numbered=numbered), provider=provider, model=model, timeout=timeout)
    order = _parse_order(raw, n)
    if order is None:                                         # unavailable / unparseable → keep μ order
        return {"reranked": list(cand), "order": list(range(n)), "displacement": 0.0,
                "top1_changed": False, "parsed": False, "raw": raw or ""}
    reranked = [cand[i] for i in order]
    # displacement: mean |new_rank − old_rank| normalised by the worst case (~n/2)
    disp = sum(abs(new - old) for new, old in enumerate(order)) / n
    disp_norm = min(1.0, disp / max(1, n / 2))
    return {"reranked": reranked, "order": order, "displacement": round(disp_norm, 3),
            "top1_changed": order[0] != 0, "parsed": True, "raw": raw}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="LLM reranker smoke test / CLI")
    ap.add_argument("--query", required=True)
    ap.add_argument("--candidates", required=True, help="comma-separated folder titles (μ order)")
    ap.add_argument("--provider", default="claude"); ap.add_argument("--model", default="haiku")
    ap.add_argument("--dry", action="store_true", help="print the prompt only; no LLM call")
    a = ap.parse_args()
    cands = [{"title": t.strip()} for t in a.candidates.split(",")]
    if a.dry:
        numbered = "\n".join("%d. %s" % (i + 1, c["title"]) for i, c in enumerate(cands))
        print(_PROMPT.format(query=a.query, numbered=numbered)); sys.exit(0)
    out = rerank(a.query, cands, provider=a.provider, model=a.model)
    print("parsed=%s displacement=%s top1_changed=%s" % (out["parsed"], out["displacement"], out["top1_changed"]))
    print("reranked:", [c["title"] for c in out["reranked"]])
