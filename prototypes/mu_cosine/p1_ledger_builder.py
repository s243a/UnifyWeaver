#!/usr/bin/env python3
"""P1 ledger builder — consumes an eligibility report + v2 bundles, emits the frozen row ledger.

Position in the lane split (2026-07-24): sol's eligibility VERIFIER produces the report (it alone
re-derives bundles and decides eligibility); THIS builder consumes the report fail-closed and
constructs the pre-training ledger per PROCESS_EXPRESSION_P1_PREREG.json. Built and tested
against SYNTHETIC and EMPTY inputs only, per the coordination note — no real labels exist yet.

Contract enforced here (prereg `ledger` section + PROTOCOL §2–4):
  - report status != eligible → exit 2 with `blocked_no_eligible_v2_labels`; NO ledger file.
  - every bundle's task/pick bytes must match the report's sha256 (fail closed) — the builder
    never re-derives eligibility, but it never trusts paths without byte identity either.
  - exact qid join per bundle (every task qid exactly one pick row; no strays).
  - record unit = query-process-target; process identity = FULL 64-hex digest
    (process_expression_p1_protocol._full_process_digest; compact ast_sha is never identity).
  - the training ledger contains NO recorded destination (labels live in a separate artifact).
  - null picks stay in the ledger flagged `excluded_from_primary_loss` (no primary ranking loss).
  - per-process weights normalize to EQUAL total training mass per process.
  - split: all process rows of a query share a side (query-block, seeded, recorded per row).

  python3 p1_ledger_builder.py --eligibility-report R.json --out-dir OUT [--split-seed 0]
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_expression_p1_protocol import _full_process_digest

REPORT_SCHEMA = "unifyweaver.process-expression-p1-eligibility.v1"
LEDGER_SCHEMA = "unifyweaver.process-expression-p1-ledger.v1"
BLOCKED = "blocked_no_eligible_v2_labels"


class LedgerError(ValueError):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_jsonl(path):
    with open(path, "rb") as f:
        raw = f.read()
    rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    return raw, rows


def build(report_path, out_dir, split_seed=0, holdout_frac=0.30):
    rep = json.load(open(report_path))
    if rep.get("schema") != REPORT_SCHEMA:
        raise LedgerError(f"unrecognized eligibility report schema: {rep.get('schema')!r}")
    if rep.get("status") != "eligible":
        print(f"{BLOCKED}: eligibility report status = {rep.get('status')!r}; no ledger written")
        return 2
    bundles = rep.get("bundles") or []
    if not bundles:
        print(f"{BLOCKED}: eligible report lists zero bundles; no ledger written")
        return 2

    rows = []
    for b in bundles:
        for k in ("tier_id", "task_path", "picks_path", "task_sha256", "picks_sha256",
                  "process_expression"):
            if not b.get(k):
                raise LedgerError(f"bundle missing {k}")
        t_raw, t_rows = read_jsonl(b["task_path"])
        p_raw, p_rows = read_jsonl(b["picks_path"])
        if sha_bytes(t_raw) != b["task_sha256"]:
            raise LedgerError(f"task bytes differ from report hash ({b['tier_id']})")
        if sha_bytes(p_raw) != b["picks_sha256"]:
            raise LedgerError(f"picks bytes differ from report hash ({b['tier_id']})")
        digest = _full_process_digest(b["process_expression"])
        tasks = {r["qid"]: r for r in t_rows if "qid" in r and "menu" in r}
        picks = {}
        for r in p_rows:
            if "qid" not in r:
                continue
            if r["qid"] in picks:
                raise LedgerError(f"duplicate pick qid {r['qid']} ({b['tier_id']})")
            picks[r["qid"]] = r.get("pick")
        if set(tasks) != set(picks):
            raise LedgerError(f"qid join mismatch ({b['tier_id']}): "
                              f"{len(tasks)} tasks vs {len(picks)} picks")
        for qid, t in sorted(tasks.items()):
            pick = picks[qid]
            menu_titles = [m["title"] for m in t["menu"]]
            rows.append({
                "row_id": f"{digest[:12]}:{qid}",
                "query": t["bookmark"],
                "menu_titles": menu_titles,             # candidates only — never the recorded destination
                "pick": pick,
                "excluded_from_primary_loss": pick is None,
                "tier_id": b["tier_id"],
                "process_expression": b["process_expression"],
                "process_digest": digest,
            })

    # equal total training mass per process
    from collections import Counter
    per_proc = Counter(r["process_digest"] for r in rows)
    for r in rows:
        r["weight"] = 1.0 / (len(per_proc) * per_proc[r["process_digest"]])

    # query-block split: every process row of a query shares a side (seeded, deterministic)
    for r in rows:
        h = hashlib.sha256(f"{split_seed}|{r['query']}".encode()).hexdigest()
        r["split"] = "held" if (int(h[:8], 16) / 0xFFFFFFFF) < holdout_frac else "train"

    os.makedirs(out_dir, exist_ok=True)
    ledger = {"schema": LEDGER_SCHEMA, "eligibility_report_sha256":
              sha_bytes(open(report_path, "rb").read()), "split_seed": split_seed,
              "holdout_frac": holdout_frac, "n_rows": len(rows),
              "n_null": sum(r["excluded_from_primary_loss"] for r in rows),
              "processes": {d: c for d, c in sorted(per_proc.items())}, "rows": rows}
    payload = json.dumps(ledger, ensure_ascii=False, sort_keys=True).encode()
    out = os.path.join(out_dir, "p1_ledger.json")
    with open(out, "wb") as f:
        f.write(payload)
    print(f"ledger -> {out} (rows {len(rows)}, null {ledger['n_null']}, "
          f"processes {len(per_proc)}, sha {sha_bytes(payload)[:16]})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligibility-report", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.30)
    a = ap.parse_args(argv)
    return build(a.eligibility_report, a.out_dir, a.split_seed, a.holdout_frac)


if __name__ == "__main__":
    sys.exit(main())
