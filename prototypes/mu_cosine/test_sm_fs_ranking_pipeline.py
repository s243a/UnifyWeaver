"""CAND2-8 executable tests: real loader+optimizer+one fit step, scoring, receipts,
bootstrap KAVs, decision path, transactions."""
import json
import math
import os

import pytest

import sm_fs_ranking_pipeline as pl


def test_install_private_no_replace_rollback_and_parent_checks(tmp_path):
    base = tmp_path / "run"
    p = str(base / "a.json")
    pl.install_private(p, b"one\n")
    with pytest.raises(pl.PipelineError, match="no-replace"):
        pl.install_private(p, b"two\n")
    st = os.lstat(p)
    assert oct(st.st_mode & 0o777) == "0o600" and st.st_nlink == 1
    assert not list(base.glob(".stage-*"))
    os.chmod(base, 0o755)                                   # parent must be 0700
    with pytest.raises(pl.PipelineError, match="0700"):
        pl.install_private(str(base / "b.json"), b"x")
    os.chmod(base, 0o700)


def test_read_bound_private_mode_and_tamper(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"data")
    os.chmod(p, 0o644)
    with pytest.raises(pl.PipelineError, match="0600"):
        pl.read_bound(str(p), private=True)
    os.chmod(p, 0o600)
    assert pl.read_bound(str(p), private=True) == b"data"
    with pytest.raises(pl.PipelineError, match="sha"):
        pl.read_bound(str(p), expect_sha="0" * 64)


def test_score_catalog_frozen_tie_rule_and_nan_fail_closed():
    assert pl.score_catalog([0.5, 0.9, 0.9, 0.1], ["a", "b", "c", "d"], 2) == 2
    assert pl.score_catalog([0.5, 0.9, 0.9, 0.1], ["a", "b", "c", "d"], 1) == 1
    with pytest.raises(pl.PipelineError, match="nonfinite"):
        pl.score_catalog([0.5, float("nan"), 0.2], ["a", "b", "c"], 0)
    with pytest.raises(pl.PipelineError, match="nonfinite"):
        pl.score_catalog([0.5, 0.2, float("inf")], ["a", "b", "c"], 2)


def test_bootstrap_known_answers_and_gate():
    from sm_fs_bootstrap import BootstrapError, N_BLOCKS, block_draw, decide
    assert block_draw(3997999, 0, 0) == (60, 0)
    assert block_draw(3997999, 0, 1) == (57, 0)
    assert block_draw(3997999, 4999, 41) == (14, 0)
    assert block_draw(3997999, 9998, 81) == (63, 0)
    with pytest.raises(BootstrapError, match="82"):
        decide({"b0": [0.1]})                              # wrong observed block count
    blocks = {f"b{i:02d}": [] for i in range(N_BLOCKS)}
    for i in range(19):
        blocks[f"b{i:02d}"] = [0.02]
    with pytest.raises(BootstrapError, match="nonempty"):
        decide(blocks)                                     # <20 nonempty observed blocks
    for i in range(30):
        blocks[f"b{i:02d}"] = [float("nan")]
    with pytest.raises(BootstrapError, match="nonfinite"):
        decide(blocks)


def test_fit_receipt_validation_rejects_substitution():
    plan = {"projections": {"0": {"projection_sha256": "p" * 64}}}
    good = {"schema": pl.FIT_SCHEMA, "fold": 0, "arm": "positive_only", "seed": 3997001,
            "init_sha256": pl.INIT_SHA[3997001], "projection_sha256": "p" * 64,
            "trainable_names_shapes": [["n", [1]]] * 18,
            "environment": {"invariant": {"dtype": "float32"}},
            "checkpoint_sha256": "c" * 64}
    pl.validate_fit_receipt(dict(good), 0, "positive_only", 3997001, plan)
    for key, val, err in (
        ("schema", "wrong", "schema"),
        ("seed", 3997002, "identity"),
        ("init_sha256", "0" * 64, "countersigned"),
        ("projection_sha256", "0" * 64, "plan-bound"),
        ("trainable_names_shapes", [], "inventory"),
    ):
        bad = dict(good)
        bad[key] = val
        with pytest.raises(pl.PipelineError, match=err):
            pl.validate_fit_receipt(bad, 0, "positive_only", 3997001, plan)


def test_real_checkpoint_loads_optimizer_constructs_and_one_step_runs():
    """CAND2-1/8: the countersigned checkpoint, the established loader, a real optimizer,
    and one executed fit step on synthetic rows (not decision-bearing; outputs discarded)."""
    torch = pytest.importorskip("torch")
    import copy

    import numpy as np

    from mu_attention import E5_REVISION, Tokenizer, build_e5_tables
    ckpt_bytes = pl.read_bound(
        os.path.join(pl.RANK_DIR, "init_seed3997001.pt"),
        expect_sha=pl.INIT_SHA[3997001], private=True)
    model, cfg = pl.load_checkpoint_bytes(ckpt_bytes, "cpu")   # legacy cfg loads (CAND2-1)
    names, shapes, tensors = pl.resolve_allowlist(model)
    assert len(shapes) == 18
    opt = torch.optim.Adam(tensors, lr=pl.ADAM["lr"])
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    titles = {"q/path": "Query Title", "c/path": "Candidate Title"}
    qtbl, ptbl, idx = build_e5_tables(sorted(titles), cache_path=None, texts=titles,
                                      model_revision=E5_REVISION)
    tok = Tokenizer(qtbl, ptbl, idx, {}, {})
    rows = [{"query": "q/path", "candidate": "c/path", "target": 1.0},
            {"query": "q/path", "candidate": "c/path", "target": 0.02}]
    before = tensors[0].detach().clone()
    loss = pl.fit_one_step(model, ref, tensors, opt, tok, rows, [0], [1],
                           np.random.default_rng(1), "cpu")
    assert math.isfinite(loss)
    assert not torch.equal(before, tensors[0].detach())        # the step really updated
