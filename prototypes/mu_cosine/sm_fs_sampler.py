#!/usr/bin/env python3
"""ID-free training-sampler seam (CAND2-2).

Byte-identical re-export of the frozen counter-based ranking sampler, importable WITHOUT binding
any preregistration ID: candidate locks bind THIS file's hash, so a step-4 prereg amendment that
changes sm_fs_protocols.py's hard-coded document IDs cannot invalidate the reviewed sampler.
The algorithm, key schema, sampler ID, and KAVs are unchanged from sm_fs_protocols
(`sm-fs-ranking-sampler-v1`, `unifyweaver.sm-fs-ranking-sampler-key.v1`).
"""
import hashlib
import json

SAMPLER_ID = "sm-fs-ranking-sampler-v1"
KEY_SCHEMA = "unifyweaver.sm-fs-ranking-sampler-key.v1"
ROLES = ("query", "common-positive", "contrast-positive",
         "negative-bucket", "negative-candidate")


class SamplerError(ValueError):
    pass


def _need(c, m):
    if not c:
        raise SamplerError(m)


def canonical_json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sampler_key_bytes(*, fold, seed, step, draw, role, query_id="", bucket="", retry=0):
    for v, n in ((fold, "fold"), (seed, "seed"), (step, "step"),
                 (draw, "draw"), (retry, "retry")):
        _need(type(v) is int and v >= 0, f"sampler {n} must be a nonnegative integer")
    _need(role in ROLES, "unknown sampler role")
    _need(isinstance(query_id, str) and isinstance(bucket, str), "key fields must be strings")
    _need(role != "query" or (query_id == "" and bucket == ""),
          "query draw must not bind a selected query or bucket")
    _need(role == "query" or query_id != "", "post-query draw must bind the selected query")
    _need(role == "negative-candidate" or bucket == "",
          "only a negative-candidate draw may bind a bucket")
    _need(role != "negative-candidate" or bucket in {"hard", "medium", "easy"},
          "negative-candidate draw must bind its bucket")
    return canonical_json_bytes({
        "bucket": bucket, "draw": draw, "fold": fold, "query_id": query_id,
        "retry": retry, "role": role, "sampler_id": SAMPLER_ID,
        "schema": KEY_SCHEMA, "seed": seed, "step": step,
    })


def sampler_index(n, *, fold, seed, step, draw, role, query_id="", bucket=""):
    _need(type(n) is int and n > 0, "sampler population must be positive")
    modulus = 1 << 256
    limit = (modulus // n) * n
    retry = 0
    while True:
        digest = hashlib.sha256(sampler_key_bytes(
            fold=fold, seed=seed, step=step, draw=draw, role=role,
            query_id=query_id, bucket=bucket, retry=retry)).digest()
        value = int.from_bytes(digest, "big", signed=False)
        if value < limit:
            return value % n, digest.hex(), retry
        retry += 1
