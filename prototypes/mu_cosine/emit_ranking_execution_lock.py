#!/usr/bin/env python3
"""Emit the CANDIDATE execution lock for the SM-FS ranking runs (reseal sequence step 2).

Regenerated per REVIEW_sm_fs_ranking_execution_lock.md §3.2–3.4 from a clean landed commit:
- initialized checkpoints are NOT regrown — the countersigned bytes are authoritative; this
  emitter asserts their hashes against sol's review table and fails closed on drift;
- the source targets TSV is hash-verified against the frozen projection before any read;
- outputs install via O_EXCL no-replace into the 0700 run dir (no in-place truncation);
- environment values are OBSERVED after enforcement (sm_fs_ranking_train.enforce_determinism),
  not desired booleans;
- provenance = the current commit (which contains constructor, emitter, trainer, verifier) with
  a dirty-tracked-tree gate; transitive growth provenance recorded as DESCRIPTIVE.

Schema: …execution-lock.candidate.v1 — this is step 2's proposal for independent review (step 3).
The FINAL lock (….final.v1, binding the amended prereg ID) is generated at step 5 only.

  python3 emit_ranking_execution_lock.py
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
RANK_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_v1")
RUN_DIR = os.path.expanduser("~/mu_data/sm_fs_ranking_run_v1")
BASE_SHA = "9bb60915e9bcf8d4a89293c538284fd5ac631d9729f355917e128db7353bc8ef"
TARGETS_SHA = "c3b298d5335ee901111f4985bbf5f7c5feb017c503a8c81db60dba1b947ac051"
INIT_SHA = {  # countersigned in REVIEW §1 — authoritative bytes, never regrown
    "3997001": "a3bf4c0588cc3e4cf1ad335b66440c113e7ee11fd116bd7307a3cee57447098e",
    "3997002": "fb353e693951819683793641464155e144caad73c553039fc64f1c6e253ad796",
    "3997003": "f42bdea0071a64dca0dad5312178127906b96215ba6890f6a37ad19d87ffdd5a",
}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    # gate scope = the lane this lock binds (a standing sci-repl submodule pointer is unrelated)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no",
                            "--", "prototypes/mu_cosine"],
                           cwd=os.path.join(ROOT, "..", ".."),
                           capture_output=True, text=True).stdout.strip()
    assert not dirty, f"tracked tree dirty — candidate lock must come from a landed commit:\n{dirty}"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    assert sha_file(os.path.join(ROOT, "model_prod_namecond_full.pt")) == BASE_SHA
    targets = os.path.join(os.path.expanduser("~/mu_data/sm_fs_v3"), "lineage_fs_targets.tsv")
    assert sha_file(targets) == TARGETS_SHA, "source targets projection drifted"
    for seed, want in INIT_SHA.items():
        got = sha_file(os.path.join(RANK_DIR, f"init_seed{seed}.pt"))
        assert got == want, f"initialized checkpoint seed {seed} drifted: {got[:16]}"

    from sm_fs_ranking_train import enforce_determinism
    observed_env = enforce_determinism()

    code = {name: sha_file(os.path.join(ROOT, name)) for name in (
        "sm_fs_ranking_construct.py", "sm_fs_ranking_train.py", "sm_fs_protocols.py",
        "sm_fs_ranking_execution_review.py", "emit_ranking_execution_lock.py",
        "fine_tune_pearltrees_filing.py", "mu_attention.py",
        "fine_tune_channel_heads.py", "judge_cards.py")}
    lock = {
        "schema": "unifyweaver.sm-fs-ranking-execution-lock.candidate.v1",
        "status": "CANDIDATE — for independent review (sequence step 3); fitting stays blocked",
        "git_commit": commit, "tracked_tree_clean": True,
        "warm_start": {"path": "model_prod_namecond_full.pt", "sha256": BASE_SHA,
                       "countersigned": "REVIEW_sm_fs_ranking_execution_lock.md §1"},
        "initialized_checkpoints": {s: {"sha256": v, "authoritative": True}
                                    for s, v in INIT_SHA.items()},
        "source_targets_sha256": TARGETS_SHA,
        "ranking_bundle_manifest_sha256": sha_file(os.path.join(RANK_DIR, "manifest.json")),
        "training_plan_sha256": sha_file(os.path.join(RUN_DIR, "training_plan.json")),
        "code_sha256": code,
        "growth_provenance_note": "initialized bytes authoritative; growth code hashes above are "
                                  "DESCRIPTIVE (incl. transitive fine_tune_channel_heads/"
                                  "judge_cards per REVIEW §3.2)",
        "observed_environment": observed_env,
        "artifact_modes": {"run_dirs": "0700", "files": "0600",
                           "install": "O_EXCL no-replace + fsync"},
        "fitting_authorized": False,
    }
    data = (json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode()
    out = os.path.join(RUN_DIR, "candidate_lock.json")
    os.makedirs(RUN_DIR, mode=0o700, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)   # no-replace
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    print(f"candidate lock -> {out} (sha {hashlib.sha256(data).hexdigest()[:16]}, "
          f"commit {commit[:12]}, fitting_authorized=false)")


if __name__ == "__main__":
    main()
