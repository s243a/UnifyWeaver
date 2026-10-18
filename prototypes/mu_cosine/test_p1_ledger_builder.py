"""Synthetic + empty-input tests for p1_ledger_builder (per the coordination note: no real labels)."""
import hashlib
import json
import os

import pytest

import p1_ledger_builder as lb


def write_bundle(tmp, name, qids, expr, null_qids=()):
    task = "\n".join(json.dumps(
        {"qid": q, "bookmark": f"bm{q}", "menu": [{"pos": p, "title": f"f{q}_{p}"}
                                                  for p in range(3)]}) for q in qids) + "\n"
    picks = "\n".join(json.dumps(
        {"qid": q, "pick": (None if q in null_qids else 1)}) for q in qids) + "\n"
    tp, pp = tmp / f"{name}_task.jsonl", tmp / f"{name}_picks.jsonl"
    tp.write_text(task)
    pp.write_text(picks)
    return {"tier_id": name, "task_path": str(tp), "picks_path": str(pp),
            "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
            "picks_sha256": hashlib.sha256(picks.encode()).hexdigest(),
            "process_expression": expr}


def report(tmp, status, bundles):
    p = tmp / "report.json"
    p.write_text(json.dumps({"schema": lb.REPORT_SCHEMA, "status": status, "bundles": bundles}))
    return str(p)


def test_blocked_and_empty_exit_2_no_ledger(tmp_path):
    assert lb.build(report(tmp_path, "blocked_no_eligible_v2_labels", []), tmp_path / "o") == 2
    assert lb.build(report(tmp_path, "eligible", []), tmp_path / "o2") == 2
    assert not (tmp_path / "o" / "p1_ledger.json").exists()
    assert not (tmp_path / "o2" / "p1_ledger.json").exists()


def test_eligible_builds_conformant_ledger(tmp_path):
    b1 = write_bundle(tmp_path, "t1", [1, 2, 3], "e5(routing(e5,haiku,menus=[10],t=[0.02]))",
                      null_qids=(3,))
    b2 = write_bundle(tmp_path, "t2", [1, 2], "e5(routing(e5,sonnet.lineage,menus=[10],t=[0.02]))")
    assert lb.build(report(tmp_path, "eligible", [b1, b2]), tmp_path / "out") == 0
    led = json.load(open(tmp_path / "out" / "p1_ledger.json"))
    assert led["n_rows"] == 5 and led["n_null"] == 1
    # full 64-hex process identity; two distinct processes
    assert all(len(d) == 64 for d in led["processes"]) and len(led["processes"]) == 2
    # no recorded destination anywhere in the training ledger (key-level check)
    banned = {"dest", "destination", "true_folder", "truepos", "label"}
    for r in led["rows"]:
        assert not (set(r) & banned), set(r) & banned
    # equal total mass per process
    mass = {}
    for r in led["rows"]:
        mass[r["process_digest"]] = mass.get(r["process_digest"], 0) + r["weight"]
    vals = list(mass.values())
    assert abs(vals[0] - vals[1]) < 1e-12
    # process-complete: same query on the same side across processes
    side = {}
    for r in led["rows"]:
        assert side.setdefault(r["query"], r["split"]) == r["split"]
    # null rows flagged out of the primary loss
    assert [r["excluded_from_primary_loss"] for r in led["rows"] if r["pick"] is None] == [True]


def test_fail_closed_on_tamper_and_join(tmp_path):
    b = write_bundle(tmp_path, "t1", [1, 2], "e5(routing(e5,haiku,menus=[10],t=[0.02]))")
    open(b["task_path"], "a").write(" ")                       # byte tamper
    with pytest.raises(lb.LedgerError, match="bytes differ"):
        lb.build(report(tmp_path, "eligible", [b]), tmp_path / "o")
    b2 = write_bundle(tmp_path, "t2", [1, 2], "e5(routing(e5,haiku,menus=[10],t=[0.02]))")
    picks = open(b2["picks_path"]).read().splitlines()[:1]     # drop a pick → join mismatch
    open(b2["picks_path"], "w").write("\n".join(picks) + "\n")
    b2["picks_sha256"] = hashlib.sha256(open(b2["picks_path"], "rb").read()).hexdigest()
    with pytest.raises(lb.LedgerError, match="join mismatch"):
        lb.build(report(tmp_path, "eligible", [b2]), tmp_path / "o")


def test_deterministic_output(tmp_path):
    b = write_bundle(tmp_path, "t1", [1, 2, 3], "e5(routing(e5,haiku,menus=[10],t=[0.02]))")
    lb.build(report(tmp_path, "eligible", [b]), tmp_path / "o1")
    lb.build(report(tmp_path, "eligible", [b]), tmp_path / "o2")
    assert (open(tmp_path / "o1" / "p1_ledger.json", "rb").read()
            == open(tmp_path / "o2" / "p1_ledger.json", "rb").read())
