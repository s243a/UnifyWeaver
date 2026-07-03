#!/usr/bin/env python3
"""Batch-score concept pairs with the codex judge (gpt-5.5, low thinking) → §14 JSON → ingest.
Reuses score_inferred_tail.build_prompts + ingest. Logs per-batch timing so usage thresholds can be estimated.
Judge tag = 'gpt-5.5-low' (its own learned calibration row; see JUDGES in mu_attention.py).

  source ~/.nvm/nvm.sh && nvm use 22            # codex needs node 22
  python3 score_with_codex.py --pairs inferred_tail_pilot.tsv --batch 10 --out /tmp/mu_data/pilot_scored_gpt55low.tsv
"""
import argparse, json, os, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_inferred_tail import build_prompts, ingest

JUDGE = "gpt-5.5-low"


def load_pair_lines(path):
    """Return (data_lines_without_newline, header_comment_lines) preserving order; data = non-# lines."""
    data, header = [], []
    for ln in open(path, encoding="utf-8"):
        (header if ln.startswith("#") else data).append(ln.rstrip("\n"))
    return data, header


def call_codex(prompt, model, effort, sandbox, timeout):
    try:
        p = subprocess.run(
            ["codex", "exec", "-s", sandbox, "-m", model, "-c", "model_reasoning_effort=%s" % effort],
            input=prompt, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return ""


def first_json_array(txt):
    """Robustly extract the FIRST complete top-level JSON array (codex double-prints; objects nest, so a
    regex is unsafe — review #1). Scan for '[' and let json.raw_decode consume a full, balanced value."""
    dec = json.JSONDecoder()
    i = 0
    while True:
        j = txt.find("[", i)
        if j < 0:
            return None
        try:
            val, end = dec.raw_decode(txt, j)
            if isinstance(val, list):
                return txt[j:end]
        except json.JSONDecodeError:
            pass
        i = j + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low", help="model_reasoning_effort (minimal blocked via codex image_gen)")
    ap.add_argument("--sandbox", default="read-only", help="codex -s sandbox mode (review #10)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--responses", default="/tmp/mu_data/codex_responses.txt")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.responses) or ".", exist_ok=True)

    data, header = load_pair_lines(a.pairs)
    if a.limit:
        data = data[:a.limit]
    pairs = [ln.split("\t") for ln in data]
    prompts = build_prompts(pairs, a.batch)
    print("scoring %d pairs in %d batches of %d via codex %s (effort=%s, sandbox=%s)"
          % (len(pairs), len(prompts), a.batch, a.model, a.effort, a.sandbox), flush=True)

    arrays, ok, fail, t0 = [], 0, 0, time.time()
    for i, pr in enumerate(prompts):
        t = time.time()
        arr = first_json_array(call_codex(pr, a.model, a.effort, a.sandbox, a.timeout))
        dt = time.time() - t
        if arr:
            arrays.append(arr); ok += 1; status = "ok"
        else:
            fail += 1; status = "FAIL(no-json/timeout)"
        done = ok + fail
        rate = (ok * a.batch) / max(1e-9, time.time() - t0)     # scored pairs/s (successful only, by design)
        print("  batch %3d/%d  %5.1fs  %-20s | cum %5.0fs  fail %d/%d (%.0f%%)  ~%.2f pairs/s"
              % (i + 1, len(prompts), dt, status, time.time() - t0, fail, done, 100.0 * fail / done, rate),
              flush=True)

    with open(a.responses, "w", encoding="utf-8") as f:
        f.write("\n".join(arrays))

    # ingest against the SAME (possibly limited) pair set — write a temp pairs file so ids align 1:1 (review #2)
    if a.limit:
        fd, ing_path = tempfile.mkstemp(suffix="_limited.tsv", dir=os.path.dirname(a.responses) or ".")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(header + data) + "\n")
    else:
        ing_path = a.pairs
    ingest(ing_path, a.responses, a.out, judge=JUDGE)
    print("DONE: %d/%d batches scored, %d failed (%.0f%%), %.0fs total → %s  (judge=%s)"
          % (ok, len(prompts), fail, 100.0 * fail / max(1, len(prompts)), time.time() - t0, a.out, JUDGE),
          flush=True)


if __name__ == "__main__":
    main()
