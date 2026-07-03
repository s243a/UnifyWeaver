#!/usr/bin/env python3
"""Batch-score concept pairs with the codex judge (gpt-5.5, low thinking) → §14 JSON → ingest.
Reuses score_inferred_tail.build_prompts + ingest. Logs per-batch timing so usage thresholds can be estimated.
Judge tag = 'gpt-5.5-low' (its own learned calibration row; see JUDGES in mu_attention.py).

  source ~/.nvm/nvm.sh && nvm use 22            # codex needs node 22
  python3 score_with_codex.py --pairs inferred_tail_pilot.tsv --batch 10 --out /tmp/mu_data/pilot_scored_gpt55low.tsv
"""
import argparse, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_inferred_tail import build_prompts, ingest

JUDGE = "gpt-5.5-low"
CODEX = ["codex", "exec", "-s", "read-only", "-m", "gpt-5.5", "-c", "model_reasoning_effort=low"]


def load_pairs(path):
    return [ln.rstrip("\n").split("\t") for ln in open(path, encoding="utf-8") if not ln.startswith("#")]


def call_codex(prompt, timeout):
    try:
        p = subprocess.run(CODEX, input=prompt, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return ""


def first_json_array(txt):
    # codex double-prints (echo + response); take the FIRST complete [ {...} ] array
    m = re.search(r'\[\s*\{.*?\}\s*\]', txt, re.S)
    return m.group(0) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", required=True)
    ap.add_argument("--responses", default="/tmp/mu_data/codex_responses.txt")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.responses) or ".", exist_ok=True)

    pairs = load_pairs(a.pairs)
    if a.limit:
        pairs = pairs[:a.limit]
    prompts = build_prompts(pairs, a.batch)
    print(f"scoring {len(pairs)} pairs in {len(prompts)} batches of {a.batch} via codex {JUDGE}", flush=True)

    arrays, ok, t0 = [], 0, time.time()
    for i, pr in enumerate(prompts):
        t = time.time()
        arr = first_json_array(call_codex(pr, a.timeout))
        dt = time.time() - t
        if arr:
            arrays.append(arr); ok += 1
            status = "ok"
        else:
            status = "FAIL(no-json/timeout)"
        rate = (ok * a.batch) / max(1e-9, time.time() - t0)
        print(f"  batch {i+1:3d}/{len(prompts)}  {dt:5.1f}s  {status}  | cum {time.time()-t0:5.0f}s  ~{rate:.2f} pairs/s", flush=True)

    with open(a.responses, "w", encoding="utf-8") as f:
        f.write("\n".join(arrays))
    ingest(a.pairs if not a.limit else a.pairs, a.responses, a.out, judge=JUDGE)
    print(f"DONE: {ok}/{len(prompts)} batches scored ({time.time()-t0:.0f}s total) → {a.out}  (judge={JUDGE})", flush=True)


if __name__ == "__main__":
    main()
