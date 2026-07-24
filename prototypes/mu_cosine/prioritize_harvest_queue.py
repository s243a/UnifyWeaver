#!/usr/bin/env python3
"""Strategically reorder the Pearltrees harvest queue toward filing-failure topics.

The augmented harvest queue (build_pearltrees_dag.py) lists missing trees in arbitrary order. At
the harvester's account-safe pacing the full drain takes weeks, so ORDER IS THE LEVER: this script
scores every queued tree by max e5 similarity to the bookmarks the filing eval currently FAILS
(rank > --fail-rank on the standing manifest) and writes a priority-sorted queue. Harvesting
completes the DAG first exactly where the eval says our graph has gaps (the `missing` stratum).

Output queue keeps the input schema (+ per-row `priority`), so batch_repair.py consumes it
directly via --queue-file. PRIVATE paths only; nothing committed.

  python3 prioritize_harvest_queue.py            # writes harvest_queue_prioritized.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PT_API = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", ".local", "data", "pearltrees_api")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.path.join(PT_API, "harvest_queue_augmented.json"))
    ap.add_argument("--out", default=os.path.join(PT_API, "harvest_queue_prioritized.json"))
    ap.add_argument("--fail-rank", type=int, default=5,
                    help="a bookmark counts as a failure topic if its true folder ranks below this")
    ap.add_argument("--min-bm", type=int, default=3)
    ap.add_argument("--max-queries", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--e5-cache", default="/tmp/mu_data/queue_prior_e5.pt")
    a = ap.parse_args(argv)

    import routed_queries as rq
    from mu_attention import build_e5_tables
    q_titles, f_titles, truepos, cos, ranks, margin, order, qman = rq.build(a)
    fail = [q_titles[b] for b in range(len(q_titles)) if ranks[b] > a.fail_rank]
    print(f"failure bookmarks (rank>{a.fail_rank}): {len(fail)} (manifest {qman[:16]})")

    q = json.load(open(a.queue))
    rows = q["maps"]
    titles = [r.get("title") or r["tree_id"] for r in rows]
    names = sorted(set(titles) | set(fail))
    qtbl, ptbl, idx = build_e5_tables(names, cache_path=a.e5_cache, batch_size=128)
    tv = np.stack([ptbl.numpy()[idx[t]] for t in titles])
    fv = np.stack([qtbl.numpy()[idx[t]] for t in fail])
    prio = (fv @ tv.T).max(axis=0)
    oq = np.argsort(-prio)
    out = dict(q)
    out["maps"] = [dict(rows[i], priority=float(prio[i])) for i in oq]
    out["prioritized"] = f"max e5 cos to rank>{a.fail_rank} filing-failure bookmarks"
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=0)
    print(f"prioritized queue -> {a.out}")
    for i in oq[:5]:
        print(f"  {prio[i]:.3f}  {titles[i][:60]}")


if __name__ == "__main__":
    main()
