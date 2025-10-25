# file_selector.py
from typing import List, Dict, Any
from collections import defaultdict

def select_top_files(
    results: List[Dict[str, Any]],
    top_m: int = 5,
    score_agg: str = "max",   # "max" | "mean" | "sum"
    min_chunks_per_file: int = 1,
) -> List[str]:
    buckets = defaultdict(list)
    for r in results:
        fp = r.get("file") or r.get("file_path")
        if not fp:
            continue
        buckets[fp].append(float(r.get("score", 0.0)))

    scored: List[tuple] = []
    for fp, scores in buckets.items():
        if len(scores) < min_chunks_per_file:
            continue
        if not scores:
            continue
        if score_agg == "mean":
            agg = sum(scores) / max(len(scores), 1)
        elif score_agg == "sum":
            agg = sum(scores)
        else:
            agg = max(scores)
        scored.append((fp, float(agg)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [fp for fp, _ in scored[:top_m]]
