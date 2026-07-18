#!/usr/bin/env python3
"""Assemble the sonnet-subagent batch JSON files into a `sonnet` scored TSV (score_with_codex format).

Sonnet-5 judged the 300-row overlap in batches (launched as subagents; each wrote a strict JSON
array keyed by GLOBAL overlap pair id to /tmp/mu_data/sonnet_batches/out100_*.json). This reuses
score_inferred_tail.ingest, which maps objects by id onto the overlap pairs file. Sonnet is the
HELD-OUT judge (not a Kalman fusion channel) — the meta-judge's calibration reference.

  python3 assemble_sonnet_scores.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_inferred_tail import ingest

BATCH_DIR = "/tmp/mu_data/sonnet_batches"
OVERLAP = "/tmp/mu_data/pt_campaign_all_score_in_overlap.tsv"
RESPONSES = "/tmp/mu_data/pt_campaign_sonnet_responses.txt"
OUT = "/tmp/mu_data/pt_campaign_scored_sonnet.tsv"


def main():
    files = sorted(glob.glob(os.path.join(BATCH_DIR, "out100_*.json")))
    if not files:
        raise SystemExit(f"no sonnet batch outputs in {BATCH_DIR} (out100_*.json)")
    all_objs, seen_ids = [], set()
    for fp in files:
        txt = open(fp, encoding="utf-8").read().strip()
        txt = txt.replace("```json", " ").replace("```", " ").strip()
        arr = json.loads(txt)
        if not isinstance(arr, list):
            raise SystemExit(f"{fp}: not a JSON list")
        for o in arr:
            if isinstance(o, dict) and "id" in o:
                if o["id"] in seen_ids:
                    continue
                seen_ids.add(o["id"])
                all_objs.append(o)
        print(f"{os.path.basename(fp)}: {len(arr)} objects")
    with open(RESPONSES, "w", encoding="utf-8") as f:
        json.dump(all_objs, f)
    ingest(OVERLAP, RESPONSES, OUT, judge="sonnet")
    # sanity
    n = sum(1 for _ in open(OUT)) - 1
    print(f"\nassembled {len(all_objs)} sonnet judgments; ids {min(seen_ids)}..{max(seen_ids)}; "
          f"scored TSV rows {n} -> {OUT}")
    if len(all_objs) < 300:
        print(f"WARNING: only {len(all_objs)}/300 overlap pairs judged — check batch coverage")


if __name__ == "__main__":
    main()
