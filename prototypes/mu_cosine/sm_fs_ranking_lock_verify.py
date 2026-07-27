#!/usr/bin/env python3
"""Candidate lock emitter + verifier v4. Finalization itself lives in sm_fs_ranking_chain.

The candidate binds the STARTING preregistration bytes (proven Git-tracked at HEAD), the chain/
sampler/bootstrap module hashes, the plan, the exact trainable inventory (obtained by really
loading a countersigned checkpoint), and the environment invariant. The final lock (chain
module, schema final.v3) may later differ ONLY in the amendment whitelist; everything else is
compared field-by-field against this candidate.

  python3 sm_fs_ranking_lock_verify.py emit
  python3 sm_fs_ranking_lock_verify.py verify --lock PATH
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sm_fs_ranking_chain as chain
from sm_fs_ranking_pipeline import (ADAM, ANCHOR_W, BS, CLIP, DRAWS, INIT_SHA, RANK_DIR,
                                    REPO_ROOT, ROOT, RUN_DIR, SEEDS, STEPS,
                                    enforce_environment, git_head, install_private,
                                    lane_clean, load_checkpoint_bytes, need, read_bound,
                                    resolve_allowlist, sha_bytes)

CODE_FILES = ("sm_fs_ranking_pipeline.py", "sm_fs_ranking_lock_verify.py",
              "sm_fs_ranking_chain.py", "sm_fs_sampler.py", "sm_fs_bootstrap.py",
              "sm_fs_ranking_construct.py", "fine_tune_channel_heads.py",
              "mu_attention.py", "judge_cards.py")
CAND_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.candidate.v4"


def _trainable_inventory():
    ckpt_bytes = read_bound(os.path.join(RANK_DIR, f"init_seed{SEEDS[0]}.pt"),
                            expect_sha=INIT_SHA[SEEDS[0]], private=True,
                            description="initialized checkpoint")
    model, _ = load_checkpoint_bytes(ckpt_bytes, "cpu")
    _, shapes, _ = resolve_allowlist(model)
    return shapes


def _bindings():
    need(lane_clean(), "tracked prototypes/mu_cosine files dirty — land the commit first "
                       "(scope: tracked lane files only)")
    man_bytes = read_bound(os.path.join(RANK_DIR, "manifest.json"))
    man = json.loads(man_bytes)
    read_bound(os.path.join(RANK_DIR, "pairs.jsonl"),
               expect_sha=man["outputs"]["pairs.jsonl"], description="pairs")
    read_bound(os.path.join(RANK_DIR, "fold_assignment.tsv"),
               expect_sha=man["outputs"]["fold_assignment.tsv"], description="folds")
    for s in SEEDS:
        read_bound(os.path.join(RANK_DIR, f"init_seed{s}.pt"), expect_sha=INIT_SHA[s],
                   private=True, description=f"init seed {s}")
    rank_prereg = chain.git_tracked_bytes(
        "prototypes/mu_cosine/SM_FS_LINEAGE_RANKING_PREREG.json", REPO_ROOT)
    ret_prereg = chain.git_tracked_bytes(
        "prototypes/mu_cosine/SM_FS_RETENTION_TRANSFER_PREREG.json", REPO_ROOT)
    env = enforce_environment()
    return {
        "cleanliness_scope": "tracked prototypes/mu_cosine files only",
        "ranking_bundle_manifest_sha256": sha_bytes(man_bytes),
        "initialized_checkpoints": {str(s): INIT_SHA[s] for s in SEEDS},
        "title_table_sha256": sha_bytes(read_bound(
            os.path.join(RUN_DIR, "titles.json"), private=True, description="title table")),
        "training_plan_sha256": sha_bytes(read_bound(
            os.path.join(RUN_DIR, "training_plan.json"), private=True,
            description="training plan")),
        "ranking_prereg_sha256": sha_bytes(rank_prereg),
        "retention_prereg_sha256": sha_bytes(ret_prereg),
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
              "ranking_prereg_sha256", "retention_prereg_sha256",
              "e5_revision", "tokenizer_structure", "code_sha256", "adam", "steps",
              "batch_size", "query_draws", "anchor_weight", "grad_clip", "early_stopping",
              "trainable_names_shapes", "frozen_reference", "augmentation", "tie_rule",
              "environment_invariant")


def emit_candidate():
    lock = dict(_bindings())
    lock["schema"] = CAND_SCHEMA
    lock["status"] = "CANDIDATE for independent review; fitting stays blocked"
    lock["fitting_authorized"] = False
    lock["git_commit"] = git_head()
    lock["environment_runtime_descriptive"] = enforce_environment()["runtime"]
    data = chain.canon(lock)
    out = os.path.join(RUN_DIR, "candidate_lock_v4.json")
    install_private(out, data)
    print(f"candidate v4 -> {out} (sha {sha_bytes(data)[:16]}, "
          f"commit {lock['git_commit'][:12]})")
    return out


def verify_candidate_lock(lock):
    need(lock.get("schema") == CAND_SCHEMA, "not a v4 candidate lock")
    need(lock.get("fitting_authorized") is False, "candidate must not authorize fitting")
    live = _bindings()
    for key in BOUND_KEYS:
        need(lock.get(key) == live[key],
             f"lock binding {key!r} does not match recomputed state")
    need(lock.get("git_commit") == git_head(),
         "candidate was generated from a different commit than the current tree")
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
        verify_candidate_lock(json.loads(read_bound(a.lock, description="lock")))
        print("candidate verifies against recomputed state")


if __name__ == "__main__":
    main()
