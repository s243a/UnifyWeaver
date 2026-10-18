#!/usr/bin/env python3
"""Resumable, checkpointed gpt-5.5-low scoring for campaign pair files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from sample_product_kalman_enwiki_campaign import write_json_atomic
from score_inferred_tail import build_prompts, ingest
from score_with_codex import JUDGE, call_codex, first_json_array, load_pair_lines


SCHEMA_VERSION = 1
PAIR_ID = re.compile(r"^\s*(\d+)\. NODE:", re.MULTILINE)
RESPONSE_FIELDS = {
    "element_of": ("mu_fwd", "mu_rev", "applies"),
    "subcategory": ("mu_fwd", "mu_rev", "applies"),
    "subtopic": ("mu_fwd", "mu_rev", "applies"),
    "super_category": ("mu_fwd", "mu_rev", "applies"),
    "see_also": ("mu", "applies"),
    "assoc": ("mu", "applies"),
    "none": ("applies",),
    "unknown": ("mu_fwd", "mu_rev", "applies"),
}


class CampaignScoringError(RuntimeError):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def prompt_ids(prompt):
    return [int(value) for value in PAIR_ID.findall(prompt)]


def validate_response(response, expected_ids):
    if not isinstance(response, list):
        raise CampaignScoringError("judge response is not a JSON list")
    if not all(isinstance(item, dict) and "id" in item for item in response):
        raise CampaignScoringError("judge response contains a non-object or missing id")
    if any(isinstance(item["id"], bool) or not isinstance(item["id"], int) for item in response):
        raise CampaignScoringError("judge response contains a non-integer id")
    ids = [item["id"] for item in response]
    if sorted(ids) != sorted(expected_ids) or len(ids) != len(set(ids)):
        raise CampaignScoringError(
            f"judge response ids {sorted(ids)} do not exactly match expected {sorted(expected_ids)}"
        )
    for item in response:
        item_id = int(item["id"])
        for relation, fields in RESPONSE_FIELDS.items():
            values = item.get(relation)
            if not isinstance(values, dict):
                raise CampaignScoringError(f"response {item_id} is missing object {relation!r}")
            for field in fields:
                value = values.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise CampaignScoringError(
                        f"response {item_id} has non-numeric {relation}.{field}: {value!r}"
                    )
                if not 0.0 <= float(value) <= 1.0:
                    raise CampaignScoringError(
                        f"response {item_id} has out-of-range {relation}.{field}: {value!r}"
                    )


def checkpoint_contract(source_hash, prompt, batch_index, batch_size, model, effort, sandbox, judge):
    return {
        "schema_version": SCHEMA_VERSION,
        "source_pairs_sha256": source_hash,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "batch_index": batch_index,
        "batch_size": batch_size,
        "model": model,
        "effort": effort,
        "sandbox": sandbox,
        "judge": judge,
    }


def load_checkpoint(path, contract, expected_ids):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for key, expected in contract.items():
        if data.get(key) != expected:
            raise CampaignScoringError(
                f"checkpoint {path} contract mismatch for {key}: "
                f"expected {expected!r}, found {data.get(key)!r}"
            )
    validate_response(data.get("response"), expected_ids)
    return data


def count_scored_rows(path):
    return sum(1 for line in open(path, encoding="utf-8") if not line.startswith("#"))


def run(args, caller=call_codex):
    if args.judge == JUDGE and (args.model != "gpt-5.5" or args.effort != "low"):
        raise CampaignScoringError(
            f"judge tag {JUDGE} requires model=gpt-5.5 and effort=low; "
            f"got {args.model}/{args.effort}"
        )
    data, _header = load_pair_lines(args.pairs)
    if args.limit:
        data = data[:args.limit]
    if not data:
        raise CampaignScoringError("pair file contains no data rows")
    pairs = [line.split("\t") for line in data]
    prompts = build_prompts(pairs, args.batch)
    source_hash = sha256_path(args.pairs)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    responses = []
    batch_records = []
    t0 = time.time()

    print(
        f"scoring {len(pairs)} pairs in {len(prompts)} checkpointed batches via "
        f"codex {args.model} (effort={args.effort}, judge={args.judge})",
        flush=True,
    )
    for batch_index, prompt in enumerate(prompts):
        expected_ids = prompt_ids(prompt)
        contract = checkpoint_contract(
            source_hash, prompt, batch_index, args.batch,
            args.model, args.effort, args.sandbox, args.judge,
        )
        checkpoint = checkpoint_dir / f"batch_{batch_index:04d}.json"
        if checkpoint.exists():
            record = load_checkpoint(checkpoint, contract, expected_ids)
            status = "resume"
        else:
            record = None
            errors = []
            for attempt in range(1, args.retries + 2):
                started = time.time()
                text = caller(prompt, args.model, args.effort, args.sandbox, args.timeout)
                array_text = first_json_array(text)
                try:
                    if not array_text:
                        raise CampaignScoringError("no JSON array found")
                    response = json.loads(array_text)
                    validate_response(response, expected_ids)
                except (json.JSONDecodeError, CampaignScoringError) as exc:
                    errors.append(f"attempt {attempt}: {exc}")
                    print(
                        f"  batch {batch_index + 1}/{len(prompts)} attempt {attempt} failed: {exc}",
                        flush=True,
                    )
                    continue
                record = dict(contract)
                record.update({
                    "expected_ids": expected_ids,
                    "elapsed_seconds": time.time() - started,
                    "response": response,
                })
                write_json_atomic(checkpoint, record)
                break
            if record is None:
                raise CampaignScoringError(
                    f"batch {batch_index} failed after {args.retries + 1} attempts: {'; '.join(errors)}"
                )
            status = "scored"
        responses.append(record["response"])
        batch_records.append({
            "batch_index": batch_index,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_path(checkpoint),
            "row_count": len(record["response"]),
        })
        print(
            f"  batch {batch_index + 1:3d}/{len(prompts)} {status:6s} "
            f"({len(record['response'])} rows; elapsed {time.time() - t0:.0f}s)",
            flush=True,
        )

    response_text = "\n".join(json.dumps(items, separators=(",", ":"), sort_keys=True) for items in responses) + "\n"
    atomic_text(args.responses, response_text)
    ingest(args.pairs, args.responses, args.out, judge=args.judge)
    scored_rows = count_scored_rows(args.out)
    if scored_rows != len(pairs):
        raise CampaignScoringError(f"ingested {scored_rows}/{len(pairs)} rows")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_pairs": str(args.pairs),
        "source_pairs_sha256": source_hash,
        "row_count": len(pairs),
        "batch_size": args.batch,
        "batch_count": len(prompts),
        "model": args.model,
        "effort": args.effort,
        "sandbox": args.sandbox,
        "judge": args.judge,
        "responses": str(args.responses),
        "responses_sha256": sha256_path(args.responses),
        "scored": str(args.out),
        "scored_sha256": sha256_path(args.out),
        "checkpoint_dir": str(checkpoint_dir),
        "batches": batch_records,
    }
    write_json_atomic(args.manifest, manifest)
    print(f"DONE: {scored_rows} rows -> {args.out}", flush=True)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--sandbox", default="read-only")
    ap.add_argument("--judge", default=JUDGE)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    return ap


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
