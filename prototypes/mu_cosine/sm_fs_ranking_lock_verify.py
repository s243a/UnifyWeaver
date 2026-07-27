#!/usr/bin/env python3
"""Candidate/final lock emitter + verifier v3 (CAND2-3/5 replacement).

Finalization chain (CAND2-3): a final lock is valid only when
  1. the LIVE preregistration authorizes fitting;
  2. the lock binds the exact reviewed CANDIDATE bytes (candidate_sha256) and the rigor lane's
     REVIEW ID, and the named review artifact — a file COMMITTED by the rigor lane's own PR —
     records verdict "accepted" for that exact candidate SHA and review ID. Caller-authored
     JSON cannot substitute for the committed review artifact;
  3. every hash the candidate bound still recomputes from disk (the only permitted intervening
     changes are the preregistration amendments the review authorizes).

All-fields verification (CAND2-5): _verify_common enforces EVERY bound field — code hashes
(ID-free set; sm_fs_protocols is deliberately excluded per CAND2-2), ranking manifest AND the
manifest-bound pairs/fold bytes, titles, plan, initialized checkpoints BY READING THE BYTES,
the exact 18 trainable names+shapes (obtained by really loading a countersigned checkpoint
through the established loader), environment INVARIANT fields (versions/flags — enforced), with
runtime fields (CUDA device/driver) recorded separately as descriptive so sandboxed
verification neither falsely passes nor falsely fails.

  python3 sm_fs_ranking_lock_verify.py emit
  python3 sm_fs_ranking_lock_verify.py verify --lock PATH
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sm_fs_ranking_pipeline import (ADAM, ANCHOR_W, ARMS, BS, CLIP, DRAWS, INIT_SHA,
                                    RANK_DIR, ROOT, RUN_DIR, SEEDS, STEPS, PipelineError,
                                    canon, enforce_environment, git_head,
                                    install_private, lane_clean, load_checkpoint_bytes,
                                    need, read_bound, resolve_allowlist, sha_bytes)

CODE_FILES = ("sm_fs_ranking_pipeline.py", "sm_fs_ranking_lock_verify.py",
              "sm_fs_sampler.py", "sm_fs_bootstrap.py", "sm_fs_ranking_construct.py",
              "fine_tune_channel_heads.py", "mu_attention.py", "judge_cards.py")
CAND_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.candidate.v3"
FINAL_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.final.v2"
RECEIPT_SCHEMA = "unifyweaver.sm-fs-ranking-final-verification-receipt.v2"


def _trainable_inventory():
    """Load a real countersigned checkpoint (cpu) and resolve the exact 18 names+shapes."""
    ckpt_bytes = read_bound(os.path.join(RANK_DIR, f"init_seed{SEEDS[0]}.pt"),
                            expect_sha=INIT_SHA[SEEDS[0]], private=True,
                            description="initialized checkpoint")
    model, _ = load_checkpoint_bytes(ckpt_bytes, "cpu")
    names, shapes, tensors = resolve_allowlist(model)
    return shapes


def _bindings():
    need(lane_clean(), "tracked prototypes/mu_cosine files dirty — land the commit first "
                       "(scope: tracked lane files only)")
    man_bytes = read_bound(os.path.join(RANK_DIR, "manifest.json"))
    man = json.loads(man_bytes)
    # manifest-bound data files verified by bytes, not presence (CAND2-5)
    read_bound(os.path.join(RANK_DIR, "pairs.jsonl"),
               expect_sha=man["outputs"]["pairs.jsonl"], description="pairs")
    read_bound(os.path.join(RANK_DIR, "fold_assignment.tsv"),
               expect_sha=man["outputs"]["fold_assignment.tsv"], description="folds")
    for s in SEEDS:                                     # init bytes actually read (CAND2-5)
        read_bound(os.path.join(RANK_DIR, f"init_seed{s}.pt"), expect_sha=INIT_SHA[s],
                   private=True, description=f"init seed {s}")
    env = enforce_environment()
    return {
        "git_commit": git_head(),
        "cleanliness_scope": "tracked prototypes/mu_cosine files only",
        "ranking_bundle_manifest_sha256": sha_bytes(man_bytes),
        "initialized_checkpoints": {str(s): INIT_SHA[s] for s in SEEDS},
        "title_table_sha256": sha_bytes(read_bound(
            os.path.join(RUN_DIR, "titles.json"), private=True, description="title table")),
        "training_plan_sha256": sha_bytes(read_bound(
            os.path.join(RUN_DIR, "training_plan.json"), private=True,
            description="training plan")),
        "e5_revision": __import__("mu_attention").E5_REVISION,
        "tokenizer_structure": "mu_attention.Tokenizer(qtbl, ptbl, idx, {}, {}); "
                               "e5 query/passage prefixes; titles-only text",
        "code_sha256": {n: sha_bytes(read_bound(os.path.join(ROOT, n))) for n in CODE_FILES},
        "adam": ADAM, "steps": STEPS, "batch_size": BS, "query_draws": DRAWS,
        "anchor_weight": ANCHOR_W, "grad_clip": CLIP, "early_stopping": False,
        "trainable_names_shapes": _trainable_inventory(),
        "frozen_reference": "copy.deepcopy of exact loaded init, eval, grads off, "
                            "before optimizer creation",
        "augmentation": {"train_rng": "numpy default_rng(seed+1), consumed in row order",
                         "anchor_rows_augmented": False},
        "tie_rule": "ascending-frozen-catalog-column",
        "environment_invariant": env["invariant"],
    }


BOUND_KEYS = ("cleanliness_scope", "ranking_bundle_manifest_sha256",
              "initialized_checkpoints", "title_table_sha256", "training_plan_sha256",
              "e5_revision", "tokenizer_structure", "code_sha256", "adam", "steps",
              "batch_size", "query_draws", "anchor_weight", "grad_clip", "early_stopping",
              "trainable_names_shapes", "frozen_reference", "augmentation", "tie_rule",
              "environment_invariant")


def emit_candidate():
    lock = dict(_bindings())
    lock["schema"] = CAND_SCHEMA
    lock["status"] = "CANDIDATE for independent review; fitting stays blocked"
    lock["fitting_authorized"] = False
    lock["environment_runtime_descriptive"] = enforce_environment()["runtime"]
    data = canon(lock)
    out = os.path.join(RUN_DIR, "candidate_lock_v3.json")
    install_private(out, data)
    print(f"candidate v3 -> {out} (sha {sha_bytes(data)[:16]}, "
          f"commit {lock['git_commit'][:12]})")
    return out


def _verify_common(lock):
    live = _bindings()
    for key in BOUND_KEYS:
        need(lock.get(key) == live[key],
             f"lock binding {key!r} does not match recomputed state")


def verify_candidate_lock(lock):
    need(lock.get("schema") == CAND_SCHEMA, "not a v3 candidate lock")
    need(lock.get("fitting_authorized") is False, "candidate must not authorize fitting")
    _verify_common(lock)
    need(lock.get("git_commit") == git_head(),
         "candidate was generated from a different commit than the current tree")
    return True


def verify_final_lock(lock):
    need(lock.get("schema") == FINAL_SCHEMA, "not a final.v2 lock")
    need(lock.get("fitting_authorized") is True, "final lock must authorize fitting")
    cand_path = os.path.join(RUN_DIR, "candidate_lock_v3.json")
    cand_bytes = read_bound(cand_path, private=True, description="reviewed candidate")
    need(lock.get("candidate_sha256") == sha_bytes(cand_bytes),
         "final lock does not bind the reviewed candidate bytes")
    review_name = lock.get("review_artifact")
    need(isinstance(review_name, str) and review_name.startswith("SM_FS_")
         and "/" not in review_name, "final lock must name a committed review artifact")
    review = json.loads(read_bound(os.path.join(ROOT, review_name),
                                   description="committed review artifact"))
    need(review.get("verdict") == "accepted", "committed review verdict is not 'accepted'")
    need(review.get("candidate_sha256") == lock.get("candidate_sha256"),
         "committed review does not accept this candidate")
    need(review.get("review_id") == lock.get("review_id"),
         "review id mismatch between lock and committed review")
    _verify_common(lock)                      # only prereg amendments may have intervened
    return True


def fitting_allowed(final_lock_path, receipt_path):
    doc = json.loads(read_bound(os.path.join(ROOT, "SM_FS_LINEAGE_RANKING_PREREG.json"),
                                description="live preregistration"))
    need(doc.get("model_fitting_authorized") is True,
         "live preregistration does not authorize model fitting")
    lock_bytes = read_bound(final_lock_path, description="final lock")
    lock = json.loads(lock_bytes)
    receipt = json.loads(read_bound(receipt_path, description="verification receipt"))
    need(receipt.get("schema") == RECEIPT_SCHEMA, "receipt schema mismatch")
    need(receipt.get("final_lock_sha256") == sha_bytes(lock_bytes),
         "receipt does not bind these exact final-lock bytes")
    need(receipt.get("review_id") == lock.get("review_id"),
         "receipt review id mismatch")
    derived = sha_bytes(canon({k: v for k, v in doc.items() if k != "prereg_id"}))
    need(lock.get("prereg_id") == doc.get("prereg_id") == receipt.get("prereg_id"),
         "prereg id mismatch across lock/receipt/live document")
    need(lock.get("prereg_id_derived") == derived, "lock's derived prereg id is stale")
    verify_final_lock(lock)
    return lock, receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("emit")
    v = sub.add_parser("verify")
    v.add_argument("--lock", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "emit":
        emit_candidate()
    else:
        lock = json.loads(read_bound(a.lock, description="lock"))
        (verify_candidate_lock if lock.get("schema") == CAND_SCHEMA
         else verify_final_lock)(lock)
        print("lock verifies against recomputed state")


if __name__ == "__main__":
    main()
