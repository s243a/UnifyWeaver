#!/usr/bin/env python3
"""Candidate/final execution-lock emitter + independent verifier (CAND-4/5 replacement).

One authority for the lock schemas: `emit_candidate()` writes the v2-namespace candidate with
every §4 binding as an enforced FIELD; `verify_candidate_lock`/`verify_final_lock` recompute
every bound hash from disk and reject tampering. The final verifier is what the pipeline's
fitting gate calls — locks never self-assert (CAND-2).

  python3 sm_fs_ranking_lock_verify.py emit      # fresh candidate in the v2 namespace
  python3 sm_fs_ranking_lock_verify.py verify --lock PATH
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sm_fs_ranking_pipeline import (ADAM, ANCHOR_W, ARMS, BOOT, BS, CLIP, DRAWS, INIT_SHA,
                                    RANK_DIR, ROOT, RUN_DIR, SEEDS, STEPS, PipelineError,
                                    canon, enforce_environment, git_head, install_private,
                                    lane_clean, need, read_bound, sha_bytes)

CODE_FILES = ("sm_fs_ranking_pipeline.py", "sm_fs_ranking_lock_verify.py",
              "sm_fs_ranking_construct.py", "sm_fs_protocols.py",
              "fine_tune_channel_heads.py", "mu_attention.py", "judge_cards.py")
CAND_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.candidate.v2"
FINAL_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.final.v1"


def _bindings():
    need(lane_clean(), "tracked prototypes/mu_cosine files dirty — land the commit first "
                       "(scope: tracked lane files only; untracked and other paths excluded)")
    man_sha = sha_bytes(read_bound(os.path.join(RANK_DIR, "manifest.json")))
    titles_path = os.path.join(RUN_DIR, "titles.json")
    return {
        "git_commit": git_head(),
        "cleanliness_scope": "tracked prototypes/mu_cosine files only",
        "ranking_bundle_manifest_sha256": man_sha,
        "initialized_checkpoints": {str(s): INIT_SHA[s] for s in SEEDS},
        "title_table_sha256": sha_bytes(read_bound(titles_path, description="title table")),
        "e5_revision": __import__("mu_attention").E5_REVISION,
        "tokenizer_structure": "mu_attention.Tokenizer(qtbl, ptbl, idx, {}, {}); "
                               "e5 query/passage prefixes; titles-only text",
        "code_sha256": {n: sha_bytes(read_bound(os.path.join(ROOT, n))) for n in CODE_FILES},
        "adam": ADAM, "steps": STEPS, "batch_size": BS, "query_draws": DRAWS,
        "anchor_weight": ANCHOR_W, "grad_clip": CLIP, "early_stopping": False,
        "trainable_contract": {"tensor_count": 18, "param_count": 1195782,
                               "assertion": "resolve_allowlist before optimizer construction"},
        "frozen_reference": "copy.deepcopy of exact loaded init, eval, grads off, "
                            "before optimizer creation",
        "augmentation": {"train_rng": "numpy default_rng(seed+1), consumed in row order",
                         "anchor_rows_augmented": False},
        "tie_rule": "ascending-frozen-catalog-column",
        "bootstrap": BOOT,
        "schemas": ["unifyweaver.sm-fs-ranking-train-projection.v1",
                    "unifyweaver.sm-fs-ranking-fit-receipt.v1",
                    "unifyweaver.sm-fs-ranking-eval-receipt.v1",
                    "unifyweaver.sm-fs-ranking-decision.v1"],
        "environment": enforce_environment(),
    }


def emit_candidate():
    lock = dict(_bindings())
    lock["schema"] = CAND_SCHEMA
    lock["status"] = "CANDIDATE for independent review; fitting stays blocked"
    lock["fitting_authorized"] = False
    data = canon(lock)
    out = os.path.join(RUN_DIR, "candidate_lock_v2.json")
    install_private(out, data)
    print(f"candidate v2 -> {out} (sha {sha_bytes(data)[:16]}, "
          f"commit {lock['git_commit'][:12]})")
    return out


def _verify_common(lock):
    live = _bindings()
    for key in ("ranking_bundle_manifest_sha256", "initialized_checkpoints",
                "title_table_sha256", "e5_revision", "code_sha256", "adam", "steps",
                "batch_size", "query_draws", "anchor_weight", "grad_clip",
                "trainable_contract", "tie_rule", "bootstrap", "schemas"):
        need(lock.get(key) == live[key],
             f"lock binding {key!r} does not match recomputed state")
    need(lock.get("git_commit") == live["git_commit"],
         "lock was generated from a different commit than the current tree")


def verify_candidate_lock(lock, **_):
    need(lock.get("schema") == CAND_SCHEMA, "not a v2 candidate lock")
    need(lock.get("fitting_authorized") is False, "candidate must not authorize fitting")
    _verify_common(lock)
    return True


def verify_final_lock(lock, **_):
    need(lock.get("schema") == FINAL_SCHEMA, "not a final lock")
    need(lock.get("fitting_authorized") is True, "final lock must authorize fitting")
    _verify_common(lock)
    return True


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
