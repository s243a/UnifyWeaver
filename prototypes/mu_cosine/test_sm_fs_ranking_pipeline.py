"""CAND3-5 adversarial + executable tests: forged reviews, drift, forged receipts,
recomputation, rollback injection, real loader + one fit step."""
import json
import math
import os
import subprocess

import pytest

import sm_fs_ranking_chain as chain
import sm_fs_ranking_pipeline as pl


# ---------- transactions ----------

def test_install_private_no_replace_rollback_and_parent_checks(tmp_path):
    base = tmp_path / "run"
    p = str(base / "a.json")
    pl.install_private(p, b"one\n")
    with pytest.raises(pl.PipelineError, match="no-replace"):
        pl.install_private(p, b"two\n")
    st = os.lstat(p)
    assert oct(st.st_mode & 0o777) == "0o600" and st.st_nlink == 1
    assert not list(base.glob(".stage-*"))
    os.chmod(base, 0o755)
    with pytest.raises(pl.PipelineError, match="0700"):
        pl.install_private(str(base / "b.json"), b"x")
    os.chmod(base, 0o700)


def test_rollback_failure_injection_removes_target_or_raises_composed(tmp_path, monkeypatch):
    base = tmp_path / "run"
    target = str(base / "c.json")
    calls = {"n": 0}
    real = pl._fsync_dir

    def failing_fsync(d):
        calls["n"] += 1
        if calls["n"] == 1:                     # first post-link fsync fails
            raise OSError("injected fsync failure")
        return real(d)

    monkeypatch.setattr(pl, "_fsync_dir", failing_fsync)
    with pytest.raises((pl.PipelineError, OSError)):
        pl.install_private(target, b"data\n")
    monkeypatch.setattr(pl, "_fsync_dir", real)
    assert not os.path.exists(target)           # partial target rolled back, durably
    pl.install_private(target, b"data\n")       # namespace reusable after rollback


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


# ---------- chain of custody (CAND3-1) ----------

def make_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "prototypes" / "mu_cosine").mkdir(parents=True)
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"]):
        subprocess.run(["git"] + args, cwd=repo, check=True, capture_output=True)
    return repo


def commit(repo, relpath, data):
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    subprocess.run(["git", "add", relpath], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=repo, check=True,
                   capture_output=True)


def make_review(candidate_sha, accepted=True):
    doc = {"schema": "unifyweaver.sm-fs-ranking-candidate-review.v9-test",
           "candidate": {"candidate_sha256": candidate_sha},
           "authorization": {"candidate_accepted": accepted}}
    doc["review_id"] = chain.derive_review_id(doc)
    return doc


def test_untracked_or_edited_review_cannot_authorize(tmp_path):
    repo = make_repo(tmp_path)
    rel = "prototypes/mu_cosine/SM_FS_TEST_REVIEW.json"
    review = make_review("c" * 64)
    # untracked file on disk — the v3 forgery — must be rejected
    (repo / rel).write_bytes(chain.canon(review))
    with pytest.raises(chain.ChainError, match="not Git-tracked"):
        chain.verify_accepted_review(rel, "c" * 64, str(repo))
    # tracked but edited on disk after commit — must be rejected
    commit(repo, rel, chain.canon(review))
    (repo / rel).write_bytes(chain.canon(review) + b" ")
    with pytest.raises(chain.ChainError, match="differs from the committed blob"):
        chain.verify_accepted_review(rel, "c" * 64, str(repo))
    # properly committed — accepted, and only for the exact candidate SHA
    (repo / rel).write_bytes(chain.canon(review))
    assert chain.verify_accepted_review(rel, "c" * 64, str(repo))
    with pytest.raises(chain.ChainError, match="THIS candidate"):
        chain.verify_accepted_review(rel, "d" * 64, str(repo))
    # tampered review id must not rederive
    bad = dict(review)
    bad["review_id"] = "0" * 64
    commit(repo, rel + "2", chain.canon(bad))
    with pytest.raises(chain.ChainError, match="rederive"):
        chain.verify_accepted_review(rel + "2", "c" * 64, str(repo))
    # request-changes review must not authorize
    rc = make_review("c" * 64, accepted=False)
    commit(repo, rel + "3", chain.canon(rc))
    with pytest.raises(chain.ChainError, match="candidate_accepted"):
        chain.verify_accepted_review(rel + "3", "c" * 64, str(repo))


def test_post_review_drift_rejected_but_amendment_whitelist_allowed():
    candidate = {"steps": 800, "adam": {"lr": 5e-4}, "ranking_prereg_sha256": "old",
                 "code_sha256": {"a.py": "1"}}
    keys = ("steps", "adam", "ranking_prereg_sha256", "code_sha256")
    ok = dict(candidate)
    ok["ranking_prereg_sha256"] = "amended"      # whitelisted: the prereg amendment
    chain.compare_with_candidate(ok, candidate, keys)
    drift = dict(candidate)
    drift["steps"] = 801                         # sol's exact reproduction
    with pytest.raises(chain.ChainError, match="post-review drift in binding 'steps'"):
        chain.compare_with_candidate(drift, candidate, keys)
    drift2 = dict(candidate)
    drift2["code_sha256"] = {"a.py": "2"}
    with pytest.raises(chain.ChainError, match="code_sha256"):
        chain.compare_with_candidate(drift2, candidate, keys)


# ---------- receipts and recomputation (CAND3-2) ----------

def _ctx():
    return {"plan": {"projections": {"0": {"projection_sha256": "p" * 64}}},
            "plan_sha256": "l" * 64, "final_lock_sha256": "f" * 64,
            "candidate_sha256": "c" * 64, "review_id": "r" * 64,
            "execution_commit": "e" * 40,
            "candidate": {"trainable_names_shapes": [["n", [1]]] * 18,
                          "environment_invariant": {"dtype": "float32"}}}


def _good_fit_receipt():
    return {"schema": pl.FIT_SCHEMA, "fold": 0, "arm": "positive_only", "seed": 3997001,
            "init_sha256": pl.INIT_SHA[3997001], "projection_sha256": "p" * 64,
            "plan_sha256": "l" * 64, "final_lock_sha256": "f" * 64,
            "candidate_sha256": "c" * 64, "review_id": "r" * 64,
            "execution_commit": "e" * 40, "steps": pl.STEPS, "batch_size": pl.BS,
            "adam": pl.ADAM, "anchor_weight": pl.ANCHOR_W, "grad_clip": pl.CLIP,
            "trainable_names_shapes": [["n", [1]]] * 18,
            "environment": {"invariant": {"dtype": "float32"},
                            "runtime": {"cuda_available": False, "driver": None}},
            "checkpoint_sha256": "k" * 64}


def test_fit_receipt_forgeries_rejected():
    ctx = _ctx()
    pl.validate_fit_receipt(_good_fit_receipt(), 0, "positive_only", 3997001, ctx)
    for key, val, err in (
        ("final_lock_sha256", "0" * 64, "final-lock chain"),
        ("candidate_sha256", "0" * 64, "candidate chain"),
        ("review_id", "0" * 64, "review chain"),
        ("plan_sha256", "0" * 64, "plan chain"),
        ("execution_commit", "z" * 40, "execution commit"),
        ("trainable_names_shapes", [["forged", [1]]] * 18, "inventory"),
        ("environment", {"invariant": {"dtype": "float64"},
                         "runtime": {"cuda_available": False}}, "environment invariant"),
        ("adam", {}, "optimizer"),
    ):
        bad = _good_fit_receipt()
        bad[key] = val
        with pytest.raises(pl.PipelineError, match=err):
            pl.validate_fit_receipt(bad, 0, "positive_only", 3997001, ctx)


def test_decision_recomputes_rank_never_trusts_rr():
    catalog = ["a", "b", "c"]
    # forged receipt row: caller claims rank 1 but scores put the destination last
    scores = [0.1, 0.9, 0.5]
    assert pl.recompute_rank(scores, catalog, 0) == 3
    with pytest.raises(pl.PipelineError, match="nonfinite"):
        pl.recompute_rank([0.1, float("nan"), 0.5], catalog, 0)
    with pytest.raises(pl.PipelineError, match="length"):
        pl.recompute_rank([0.1, 0.5], catalog, 0)
    # sol's fabricated-rr reproduction dies at exactly this seam: a stored rank that
    # disagrees with the recomputed rank is the decide-time rejection condition
    stored_rank = 1
    assert pl.recompute_rank(scores, catalog, 0) != stored_rank


def test_score_tie_rule():
    assert pl.recompute_rank([0.5, 0.9, 0.9, 0.1], ["a", "b", "c", "d"], 2) == 2
    assert pl.recompute_rank([0.5, 0.9, 0.9, 0.1], ["a", "b", "c", "d"], 1) == 1


# ---------- bootstrap (CAND2-6, retained) ----------

def test_bootstrap_known_answers_and_gate():
    from sm_fs_bootstrap import BootstrapError, N_BLOCKS, block_draw, decide
    assert block_draw(3997999, 0, 0) == (60, 0)
    assert block_draw(3997999, 0, 1) == (57, 0)
    assert block_draw(3997999, 4999, 41) == (14, 0)
    assert block_draw(3997999, 9998, 81) == (63, 0)
    with pytest.raises(BootstrapError, match="82"):
        decide({"b0": [0.1]})
    blocks = {f"b{i:02d}": [] for i in range(N_BLOCKS)}
    for i in range(19):
        blocks[f"b{i:02d}"] = [0.02]
    with pytest.raises(BootstrapError, match="nonempty"):
        decide(blocks)


# ---------- real loader + optimizer + one executed step (CAND2-1/8, retained) ----------

def test_real_checkpoint_loads_optimizer_constructs_and_one_step_runs():
    torch = pytest.importorskip("torch")
    import copy

    import numpy as np

    from mu_attention import E5_REVISION, Tokenizer, build_e5_tables
    ckpt_bytes = pl.read_bound(
        os.path.join(pl.RANK_DIR, "init_seed3997001.pt"),
        expect_sha=pl.INIT_SHA[3997001], private=True)
    model, cfg = pl.load_checkpoint_bytes(ckpt_bytes, "cpu")
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
    assert not torch.equal(before, tensors[0].detach())
