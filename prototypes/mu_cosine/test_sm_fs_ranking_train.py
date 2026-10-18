"""Gate + schedule tests for sm_fs_ranking_train (no fitting exists to test — by design)."""
import json

import pytest

import sm_fs_ranking_train as rt


def test_fit_gate_fails_closed(tmp_path):
    ok, why = rt.fitting_gate(None)
    assert not ok and "no final" in why
    bad = tmp_path / "lock.json"
    bad.write_text(json.dumps({"schema": "wrong"}))
    ok, why = rt.fitting_gate(str(bad))
    assert not ok and "not final" in why
    bad.write_text(json.dumps({
        "schema": "unifyweaver.sm-fs-ranking-execution-lock.final.v1",
        "fitting_authorized": False}))
    ok, why = rt.fitting_gate(str(bad))
    assert not ok and "does not authorize" in why
    bad.write_text(json.dumps({
        "schema": "unifyweaver.sm-fs-ranking-execution-lock.final.v1",
        "fitting_authorized": True, "prereg_id": "deadbeef",
        "prereg_id_derived": "deadbeef", "independently_verified": True}))
    ok, why = rt.fitting_gate(str(bad))
    assert not ok and "differs from live document" in why


def test_sampler_bucket_slot_matches_kav():
    # negative-bucket KAV: n=6 (3+2+1 slots, all buckets nonempty) — replicate slot decode
    from sm_fs_protocols import sampler_index
    idx, digest, retry = sampler_index(
        6, fold=4, seed=3997003, step=799, draw=23, role="negative-bucket", query_id="Q")
    assert digest == "4dd9521b16106af60d590b85a66c65d2ade1a9b30e5954f5bc2d27f651eb2d32"
    assert idx == 2 and retry == 0
    acc, chosen = 0, None
    for b, w in rt.BUCKET_SLOTS:
        acc += w
        if idx < acc:
            chosen = b
            break
    assert chosen == "hard"                      # idx 2 falls in hard's 3 slots (0-2)


def test_common_slot_identical_across_arms():
    pos = {"Q": [10, 11, 12, 13, 14, 15, 16]}
    buckets = {"Q": {"hard": [1], "medium": [2], "easy": [3]}}
    c1, _ = rt.sample_step(2, 3997002, 17, ["Q"], pos, buckets, "positive_only")
    c2, _ = rt.sample_step(2, 3997002, 17, ["Q"], pos, buckets, "graded_negative")
    assert c1 == c2                              # common-positive rows shared verbatim
    # KAV: common-positive fold=2 seed=3997002 step=17 draw=5 n=7 -> index 2
    assert c1[5] == pos["Q"][2]
