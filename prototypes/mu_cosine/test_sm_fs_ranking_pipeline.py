"""CAND-2/3/6 replacement tests: gate, capability separation, transactions."""
import json
import os

import pytest

import sm_fs_ranking_pipeline as pl


def test_install_private_no_replace_and_verification(tmp_path):
    p = str(tmp_path / "a.json")
    pl.install_private(p, b"one\n")
    with pytest.raises(pl.PipelineError, match="no-replace"):
        pl.install_private(p, b"two\n")
    st = os.lstat(p)
    assert oct(st.st_mode & 0o777) == "0o600" and st.st_nlink == 1
    assert not list(tmp_path.glob(".stage-*"))            # rollback/cleanup ran


def test_read_bound_rejects_tamper_and_symlink(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"data")
    good = pl.sha_bytes(b"data")
    assert pl.read_bound(str(p), expect_sha=good) == b"data"
    with pytest.raises(pl.PipelineError, match="sha"):
        pl.read_bound(str(p), expect_sha="0" * 64)
    ln = tmp_path / "link"
    ln.symlink_to(p)
    with pytest.raises(pl.PipelineError, match="unavailable"):
        pl.read_bound(str(ln))                            # O_NOFOLLOW refuses symlinks


def test_fitting_gate_rejects_hand_written_lock(tmp_path):
    # live prereg says model_fitting_authorized=false -> gate refuses BEFORE looking at the lock
    lock = tmp_path / "lock.json"
    lock.write_bytes(b"{}")
    rcpt = tmp_path / "r.json"
    rcpt.write_bytes(b"{}")
    with pytest.raises(pl.PipelineError, match="does not authorize model fitting"):
        pl.fitting_allowed(str(lock), str(rcpt))


def test_projection_excludes_held_associations(tmp_path, monkeypatch):
    pairs = [
        {"query": "q1", "candidate": "c", "class": "positive_parent", "fold": 0, "target": 1.0},
        {"query": "q2", "candidate": "c", "class": "positive_parent", "fold": 1, "target": 1.0},
        {"query": "q2", "candidate": "d", "class": "structural_nonancestor", "fold": 1,
         "hardness": "easy", "target": 0.02},
    ]
    monkeypatch.setattr(pl, "load_pairs_verified", lambda: ({}, pairs))
    monkeypatch.setattr(pl, "RUN_DIR", str(tmp_path))
    class A: fold = 0
    pl.cmd_project(A)
    rows = [json.loads(l) for l in open(tmp_path / "fold0" / "train_projection.jsonl")]
    assert all(r["query"] != "q1" for r in rows)          # held query absent entirely
    assert len(rows) == 2
    meta = json.loads(open(tmp_path / "fold0" / "train_projection.meta.json").read())
    assert meta["held_queries_excluded"] == 1
