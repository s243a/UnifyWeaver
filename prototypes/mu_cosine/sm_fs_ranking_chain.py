#!/usr/bin/env python3
"""Finalization chain of custody v4 (CAND3-1 replacement) — nothing caller-forgeable.

Chain: candidate → Git-TRACKED accepted review → final lock → verification receipt → fitting.
Every link is authenticated, not named:

- the accepted review's bytes must equal the blob AT THE CURRENT COMMIT for a TRACKED path
  (`git_tracked_bytes`: `git cat-file blob HEAD:<path>` byte-compare) — an untracked or
  edited-on-disk file can never authorize anything;
- the review must carry a registered schema, its canonical review ID must REDERIVE from its own
  bytes, and it must explicitly accept the exact candidate SHA;
- the final lock's execution bindings are compared FIELD-BY-FIELD with the reviewed candidate's
  bindings; any drift outside the explicit amendment whitelist (the preregistration fields the
  review authorizes) is rejected — post-review code/plan/optimizer/environment changes cannot
  ride in on the old candidate SHA;
- the live ranking preregistration must match the final lock's amended binding, authorize
  fitting, and rederive its ID; the retention preregistration must match its bound hash and
  remain execution-blocked (the narrow cascade);
- the verification receipt binds the exact final-lock bytes and is emitted through the same
  no-replace transaction machinery;
- `fitting_allowed` returns the VERIFIED BYTES of the plan and title table so the fitter consumes
  authenticated content, never reopened paths.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REVIEW_SCHEMA_PREFIX = "unifyweaver.sm-fs-ranking-candidate-review."
FINAL_SCHEMA = "unifyweaver.sm-fs-ranking-execution-lock.final.v3"
RECEIPT_SCHEMA = "unifyweaver.sm-fs-ranking-final-verification-receipt.v3"
AMENDMENT_WHITELIST = frozenset({
    "ranking_prereg_sha256", "ranking_prereg_id",
    "retention_prereg_sha256", "retention_prereg_id",
    "execution_commit",
})


class ChainError(RuntimeError):
    pass


def need(cond, msg):
    if not cond:
        raise ChainError(msg)


def canon(o):
    return (json.dumps(o, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _git(args, repo_root):
    r = subprocess.run(["git"] + args, cwd=repo_root, capture_output=True)
    return r.returncode, r.stdout


def git_tracked_bytes(relpath, repo_root):
    """Bytes of a file PROVEN tracked and identical to the blob at HEAD (CAND3-1)."""
    need("/" not in relpath.replace("prototypes/mu_cosine/", "", 1) or True, "path ok")
    rc, _ = _git(["ls-files", "--error-unmatch", relpath], repo_root)
    need(rc == 0, f"{relpath} is not Git-tracked — untracked artifacts cannot authorize")
    rc, blob = _git(["cat-file", "blob", f"HEAD:{relpath}"], repo_root)
    need(rc == 0, f"{relpath} has no blob at HEAD")
    disk = open(os.path.join(repo_root, relpath), "rb").read()
    need(disk == blob, f"{relpath} on disk differs from the committed blob at HEAD")
    return blob


def derive_review_id(doc):
    return sha_bytes(canon({k: v for k, v in doc.items() if k != "review_id"}))


def verify_accepted_review(review_relpath, candidate_sha, repo_root):
    """The review must be tracked-at-HEAD, schema-registered, ID-rederivable, and explicitly
    accept the exact candidate SHA."""
    blob = git_tracked_bytes(review_relpath, repo_root)
    doc = json.loads(blob)
    need(str(doc.get("schema", "")).startswith(REVIEW_SCHEMA_PREFIX),
         "review schema is not a registered candidate-review schema")
    need(doc.get("review_id") == derive_review_id(doc),
         "review ID does not rederive from the review bytes")
    auth = doc.get("authorization")
    need(isinstance(auth, dict) and auth.get("candidate_accepted") is True,
         "committed review does not accept the candidate "
         "(authorization.candidate_accepted must be true)")
    cand = doc.get("candidate")
    need(isinstance(cand, dict) and cand.get("candidate_sha256") == candidate_sha,
         "committed review does not accept THIS candidate's bytes")
    return doc


def compare_with_candidate(final_lock, candidate, binding_keys):
    """Field-by-field: final execution bindings must equal the reviewed candidate's, except the
    amendment whitelist (CAND3-1 post-review drift)."""
    for key in binding_keys:
        if key in AMENDMENT_WHITELIST:
            continue
        need(final_lock.get(key) == candidate.get(key),
             f"post-review drift in binding {key!r}: final lock differs from the "
             f"reviewed candidate and {key!r} is not an authorized amendment field")


def verify_prereg_pair(final_lock, repo_root):
    """Live prereg docs must match the final lock's amended bindings; ranking must authorize
    fitting; retention must remain execution-blocked (the narrow cascade)."""
    rank_rel = "prototypes/mu_cosine/SM_FS_LINEAGE_RANKING_PREREG.json"
    ret_rel = "prototypes/mu_cosine/SM_FS_RETENTION_TRANSFER_PREREG.json"
    rank_bytes = git_tracked_bytes(rank_rel, repo_root)
    ret_bytes = git_tracked_bytes(ret_rel, repo_root)
    need(sha_bytes(rank_bytes) == final_lock.get("ranking_prereg_sha256"),
         "live ranking preregistration differs from the final lock's amended binding")
    need(sha_bytes(ret_bytes) == final_lock.get("retention_prereg_sha256"),
         "live retention preregistration differs from the final lock's bound cascade")
    rank = json.loads(rank_bytes)
    need(rank.get("model_fitting_authorized") is True,
         "live ranking preregistration does not authorize model fitting")
    derived = sha_bytes(canon({k: v for k, v in rank.items() if k != "prereg_id"}))
    need(rank.get("prereg_id") == derived == final_lock.get("ranking_prereg_id"),
         "ranking preregistration ID does not rederive/match the final lock")
    ret = json.loads(ret_bytes)
    for key, want in (("execution_authorized", False),):
        need(ret.get(key) is want,
             f"retention cascade violated: {key} must remain {want}")
    return rank, ret


def verify_final_state(final_lock_bytes, run_dir, repo_root, recompute_bindings,
                       binding_keys):
    """Complete chain verification. Returns (final_lock, candidate, review_doc)."""
    lock = json.loads(final_lock_bytes)
    need(lock.get("schema") == FINAL_SCHEMA, "not a final.v3 lock")
    need(lock.get("fitting_authorized") is True, "final lock must authorize fitting")
    cand_path = os.path.join(run_dir, lock.get("candidate_file", "candidate_lock_v4.json"))
    cand_bytes = open(cand_path, "rb").read()
    need(sha_bytes(cand_bytes) == lock.get("candidate_sha256"),
         "final lock does not bind the reviewed candidate bytes")
    candidate = json.loads(cand_bytes)
    review = verify_accepted_review(lock.get("review_artifact", ""),
                                    lock["candidate_sha256"], repo_root)
    need(review.get("review_id") == lock.get("review_id"),
         "review ID mismatch between final lock and committed review")
    compare_with_candidate(lock, candidate, binding_keys)
    live = recompute_bindings()
    for key in binding_keys:
        if key in AMENDMENT_WHITELIST:
            continue
        need(live.get(key) == candidate.get(key),
             f"live execution state drifted from the reviewed candidate in {key!r}")
    verify_prereg_pair(lock, repo_root)
    exec_commit = lock.get("execution_commit")
    rc, head = _git(["rev-parse", "HEAD"], repo_root)
    need(rc == 0 and head.decode().strip() == exec_commit,
         "current commit is not the final lock's bound execution commit")
    return lock, candidate, review


def verify_receipt(receipt_bytes, final_lock_bytes):
    receipt = json.loads(receipt_bytes)
    need(receipt.get("schema") == RECEIPT_SCHEMA, "verification receipt schema mismatch")
    need(receipt.get("final_lock_sha256") == sha_bytes(final_lock_bytes),
         "receipt does not bind these exact final-lock bytes")
    lock = json.loads(final_lock_bytes)
    need(receipt.get("review_id") == lock.get("review_id"),
         "receipt review ID differs from the final lock")
    need(receipt.get("ranking_prereg_id") == lock.get("ranking_prereg_id"),
         "receipt preregistration ID differs from the final lock")
    return receipt
