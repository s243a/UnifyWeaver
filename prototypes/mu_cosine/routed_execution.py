#!/usr/bin/env python3
"""Content-bound chunking, repeated draws, retries, and aggregation for routed filing.

This module performs no hosted calls.  It turns one certified-public parent
``routed-task.v2`` into exact logical requests, seals local copies of provider
responses, and derives one parent-compatible ``routed-picks.v2`` artifact.

Execution integrity is not statistical independence.  All outputs remain
exploratory until chunk, draw, provider-run/session, and presentation dependence
are represented by the result inference procedure; see
``DESIGN_routed_execution_bundle.md``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import ctypes
from datetime import datetime
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routed_policy import (
    RoutedPolicyError,
    build_pick_envelope,
    build_task_envelope,
    canonical_json_bytes,
    content_record_bytes,
    file_content_record,
    read_pick_file,
    read_policy_file,
    read_raw_picks_bytes,
    read_task_file,
    sha256_bytes,
    strict_json_loads,
    write_pick_file,
    write_task_file,
)


PLAN_SCHEMA = "unifyweaver.routed-execution-plan.v1"
ATTEMPT_SCHEMA = "unifyweaver.routed-execution-attempt.v1"
BUNDLE_SCHEMA = "unifyweaver.routed-execution-bundle.v1"
REQUEST_PREAMBLE = b"UNIFYWEAVER ROUTED JUDGE REQUEST v1\n\n"
DRAW_COUNT = 3
PACKING_ID = "contiguous-parent-row-order-v1"
ROTATION_ID = "sha256-qid-base-plus-cyclic-draw-v1"
REQUEST_RENDERING_ID = "preamble-plus-prompt-plus-blank-line-plus-task-jsonl-v1"
ATTEMPT_RULE_ID = "contiguous-zero-based-one-terminal-v1"
AGGREGATION_ID = "strict-majority-folder-id-v1"
INFERENCE_STATUS = "integrity-only-execution-dependence-unmodeled"
PLAN_FILENAME = "execution-plan.json"
PARENT_TASK_FILENAME = "parent-task.jsonl"
PROMPT_FILENAME = "judge-prompt.md"
POLICY_FILENAME = "routing-policy.json"
DERIVED_DIRNAME = "derived"
AGGREGATE_FILENAME = "aggregate-picks.jsonl"
BUNDLE_FILENAME = "execution-bundle.jsonl"
EXECUTION_POLICY_SCHEMA = "unifyweaver.routed-execution-policy.v1"
AGGREGATE_PROVENANCE_SCHEMA = "unifyweaver.routed-execution-aggregate.v1"


def _id_for(core: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(core)))


def _require_int(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoutedPolicyError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise RoutedPolicyError(f"{label} must be at least {minimum}")
    return value


def _require_string(value, label, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise RoutedPolicyError(f"{label} must be {qualifier}")
    return value


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RoutedPolicyError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _require_mapping(value, label):
    if not isinstance(value, Mapping):
        raise RoutedPolicyError(f"{label} must be an object")
    return dict(value)


def _qid_record(qids: Sequence[int]) -> dict[str, Any]:
    values = []
    for qid in qids:
        values.append(_require_int(qid, "qid", minimum=0))
    payload = b"".join(canonical_json_bytes(qid) for qid in values)
    return {"count": len(values), "sha256": sha256_bytes(payload)}


def _rows_record(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    return {"count": len(rows), "sha256": sha256_bytes(payload)}


def _relative_path(value, label):
    value = _require_string(value, label)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise RoutedPolicyError(f"{label} must be a safe relative POSIX path")
    return value


def _resolved_relative(root: Path, value, label):
    relative = _relative_path(value, label)
    candidate = (root / relative).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise RoutedPolicyError(f"{label} escapes the execution directory")
    return candidate


def _strict_json_file(path):
    value = strict_json_loads(Path(path).read_bytes(), str(path))
    if not isinstance(value, Mapping):
        raise RoutedPolicyError(f"{path} must contain one JSON object")
    return dict(value)


def _strict_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_bytes().splitlines(), 1):
        if line.strip():
            rows.append(strict_json_loads(line, f"{path}:{line_number}"))
    if not rows:
        raise RoutedPolicyError(f"empty JSONL file: {path}")
    return rows


def _atomic_write_bytes_no_clobber(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, target)
        except FileExistsError as exc:
            raise RoutedPolicyError(f"refusing to overwrite existing artifact: {target}") from exc
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return file_content_record(target)


def _atomic_write_json_no_clobber(path, value):
    return _atomic_write_bytes_no_clobber(path, canonical_json_bytes(value))


def _atomic_write_jsonl_no_clobber(path, header, rows):
    data = canonical_json_bytes(dict(header)) + b"".join(
        canonical_json_bytes(dict(row)) for row in rows
    )
    return _atomic_write_bytes_no_clobber(path, data)


def _rename_directory_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RoutedPolicyError("atomic no-replace directory install is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    ) == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise RoutedPolicyError(f"refusing to overwrite existing artifact: {target}")
    raise RoutedPolicyError("atomic no-replace directory install failed") from OSError(
        error, os.strerror(error), str(target)
    )


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _PlanWriterLock:
    def __init__(self, root):
        self.path = Path(root) / ".routed-execution-writer.lock"
        self.descriptor = None

    def __enter__(self):
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise RoutedPolicyError(
                f"cannot open plan writer lock without following links: {self.path}"
            ) from exc
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RoutedPolicyError("plan writer lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        self.descriptor = descriptor
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


def _implementation_record():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            check=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    files = []
    for name in (
        "routed_execution.py",
        "routed_policy.py",
        "routed_queries.py",
        "filing_privacy.py",
    ):
        data = (root / name).read_bytes()
        files.append({"name": name, "sha256": sha256_bytes(data)})
    return {
        "git_commit": commit,
        "files_sha256": _rows_record(files)["sha256"],
    }


def _render_request(prompt_bytes: bytes, task_bytes: bytes) -> bytes:
    prompt = bytes(prompt_bytes)
    if not prompt.endswith(b"\n"):
        prompt += b"\n"
    return REQUEST_PREAMBLE + prompt + b"\n" + bytes(task_bytes)


def _rotation_base(schedule_id, qid, menu_size):
    digest = sha256_bytes(
        canonical_json_bytes(
            {
                "algorithm": ROTATION_ID,
                "schedule_id": schedule_id,
                "qid": qid,
            }
        )
    )
    return int(digest[:16], 16) % menu_size


def _presented_row(parent_row, schedule_id, draw_index):
    row = copy.deepcopy(dict(parent_row))
    menu = row["menu"]
    if not menu:
        raise RoutedPolicyError(f"qid {row['qid']} has an empty menu")
    base = _rotation_base(schedule_id, row["qid"], len(menu))
    offset = (base + draw_index) % len(menu)
    order = list(range(offset, len(menu))) + list(range(0, offset))
    presented = []
    for position, canonical_position in enumerate(order):
        item = copy.deepcopy(menu[canonical_position])
        item["pos"] = position
        presented.append(item)
    row["menu"] = presented
    presentation = {
        "qid": row["qid"],
        "rotation_offset": offset,
        "presented_to_canonical": order,
    }
    return row, presentation


def _child_relative_path(draw_index, chunk_index, suffix):
    return f"draw-{draw_index:03d}/chunk-{chunk_index:03d}.{suffix}"


def _schedule_core(
    parent_header,
    parent_record,
    parent_core,
    prompt_record,
    policy_envelope,
    policy_record,
    *,
    namespace,
    chunk_size,
    certification,
    implementation=None,
):
    namespace = _require_string(namespace, "execution namespace")
    chunk_size = _require_int(chunk_size, "chunk size", minimum=1)
    parent_row_count = parent_core["task_rows"]["count"]
    chunk_count = math.ceil(parent_row_count / chunk_size) if parent_row_count else 0
    required_judge = parent_core.get("selection", {}).get("required_judge")
    if not isinstance(required_judge, Mapping):
        raise RoutedPolicyError("parent task does not bind a required judge")
    judge_contract = parent_core.get("judge_contract")
    if (
        not isinstance(judge_contract, Mapping)
        or judge_contract.get("required_judge") != required_judge
    ):
        raise RoutedPolicyError("parent selection and judge contract differ")
    execution_contract = parent_core.get("execution_contract")
    if (
        not isinstance(execution_contract, Mapping)
        or execution_contract.get("chunked_provider_calls")
        != "routed-execution-plan-v1-required"
        or execution_contract.get("repeated_draw_aggregation")
        != AGGREGATION_ID
        or execution_contract.get("draw_count") != DRAW_COUNT
        or execution_contract.get("missing_or_terminal_failed_call")
        != "fail-closed-no-imputation"
        or execution_contract.get("confirmatory_inference")
        != "not-authorized-until-execution-dependence-is-modeled"
    ):
        raise RoutedPolicyError("parent task does not bind the execution-v1 contract")
    policy_id = parent_core.get("policy_provenance", {}).get("policy_id")
    if policy_id != policy_envelope.get("policy_id"):
        raise RoutedPolicyError("parent task and routing policy IDs differ")
    prompt_sha256 = required_judge.get("prompt_sha256")
    if prompt_sha256 != prompt_record["sha256"]:
        raise RoutedPolicyError("prompt bytes do not match the parent judge contract")
    privacy = parent_core.get("privacy")
    if not isinstance(privacy, Mapping):
        raise RoutedPolicyError("parent task privacy receipt is missing")
    source = parent_core.get("source")
    if not isinstance(source, Mapping):
        raise RoutedPolicyError("parent task source receipt is missing")
    return {
        "parent_task": {
            "relative_path": PARENT_TASK_FILENAME,
            "task_id": parent_header["task_id"],
            **dict(parent_record),
        },
        "routing_policy": {
            "relative_path": POLICY_FILENAME,
            "policy_id": policy_id,
            **dict(policy_record),
        },
        "prompt": {
            "relative_path": PROMPT_FILENAME,
            **dict(prompt_record),
        },
        "privacy": dict(privacy),
        "source": dict(source),
        "required_judge": dict(required_judge),
        "execution_namespace": namespace,
        "draw_count": DRAW_COUNT,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "packing_id": PACKING_ID,
        "rotation_id": ROTATION_ID,
        "request_rendering_id": REQUEST_RENDERING_ID,
        "attempt_rule_id": ATTEMPT_RULE_ID,
        "aggregation": {
            "aggregation_id": AGGREGATION_ID,
            "denominator": "all-three-scheduled-draws",
            "strict_majority_threshold": 2,
            "null_is_vote": True,
            "no_consensus_maps_to_parent_null": True,
        },
        "inference_status": INFERENCE_STATUS,
        "certification": dict(certification),
        "implementation": dict(
            _implementation_record() if implementation is None else implementation
        ),
    }


def _validate_certification(certification, parent_header, parent_core):
    if not isinstance(certification, Mapping):
        raise RoutedPolicyError("current-source certification is required")
    if certification.get("current_source_rebuild_verified") is not True:
        raise RoutedPolicyError("parent task was not rebuilt from the current certified source")
    if certification.get("parent_task_id") != parent_header["task_id"]:
        raise RoutedPolicyError("certification is bound to a different parent task")
    if (
        certification.get("privacy_manifest_sha256")
        != parent_core.get("privacy", {}).get("manifest_sha256")
    ):
        raise RoutedPolicyError("certification privacy manifest differs from the parent task")
    if (
        certification.get("source_members_sha256")
        != parent_core.get("source", {}).get("members_sha256")
    ):
        raise RoutedPolicyError("certification source inventory differs from the parent task")
    if not isinstance(certification.get("implementation"), Mapping):
        raise RoutedPolicyError("certification implementation receipt is missing")
    return dict(certification)


def certify_parent_task_current(parent_task_path):
    """Rebuild a parent task from the current public source before dispatch."""
    import routed_queries

    parent_header, _, _ = read_task_file(parent_task_path)
    core = parent_header["task_core"]
    arguments = SimpleNamespace(
        min_bm=core["catalog"]["min_bm"],
        max_queries=core["population"]["max_queries"],
        seed=core["population"]["seed"],
        top_k=core["ranker"]["top_k"],
    )
    state = routed_queries.build(arguments)
    rebuilt_header, _, _, selection = routed_queries._verify_task_rebuild(
        state, parent_task_path
    )
    if rebuilt_header != parent_header:
        raise RoutedPolicyError("current-source parent rebuild changed the task envelope")
    policy = routed_queries._frozen_policy()
    if (
        parent_header["task_core"].get("policy_provenance", {}).get("policy_id")
        != policy["policy_id"]
    ):
        raise RoutedPolicyError("parent task is not bound to the current frozen policy")
    routed_queries._validate_selection_against_tier(policy, selection)
    return {
        "current_source_rebuild_verified": True,
        "parent_task_id": parent_header["task_id"],
        "privacy_manifest_sha256": core["privacy"]["manifest_sha256"],
        "source_members_sha256": core["source"]["members_sha256"],
        "implementation": routed_queries._implementation_record(),
    }


def _task_file_bytes(header, rows):
    return canonical_json_bytes(dict(header)) + b"".join(
        canonical_json_bytes(dict(row)) for row in rows
    )


def _build_plan_envelope(plan_core):
    core = dict(plan_core)
    return {
        "schema": PLAN_SCHEMA,
        "record_type": "execution_plan",
        "plan_id": _id_for(core),
        "plan_core": core,
    }


def _child_task(
    parent_header,
    parent_rows,
    schedule,
    schedule_id,
    draw_index,
    chunk_index,
):
    chunk_size = schedule["chunk_size"]
    start = chunk_index * chunk_size
    stop = min(start + chunk_size, len(parent_rows))
    source_rows = parent_rows[start:stop]
    presented_rows = []
    presentations = []
    for parent_row in source_rows:
        presented, presentation = _presented_row(
            parent_row, schedule_id, draw_index
        )
        presented_rows.append(presented)
        presentations.append(presentation)
    qids = [row["qid"] for row in source_rows]
    parent_core = parent_header["task_core"]
    child_core = {
        key: copy.deepcopy(value)
        for key, value in parent_core.items()
        if key != "task_rows"
    }
    child_core["execution_fragment"] = {
        "schedule_id": schedule_id,
        "parent_task_id": parent_header["task_id"],
        "draw_index": draw_index,
        "draw_count": schedule["draw_count"],
        "chunk_index": chunk_index,
        "chunk_count": schedule["chunk_count"],
        "ordered_qids": _qid_record(qids),
        "presentation_rows": _rows_record(presentations),
        "packing_id": schedule["packing_id"],
        "rotation_id": schedule["rotation_id"],
    }
    child_header = build_task_envelope(child_core, presented_rows)
    child_bytes = _task_file_bytes(child_header, presented_rows)
    entry = {
        "draw_index": draw_index,
        "chunk_index": chunk_index,
        "ordered_qids": _qid_record(qids),
        "presentations": presentations,
        "task": {
            "relative_path": _child_relative_path(
                draw_index, chunk_index, "task.jsonl"
            ),
            "task_id": child_header["task_id"],
            **content_record_bytes(child_bytes),
        },
        "request": {
            "relative_path": _child_relative_path(
                draw_index, chunk_index, "request.txt"
            ),
        },
    }
    return entry, child_header, presented_rows, child_bytes


def _derive_children(parent_header, parent_rows, prompt_bytes, schedule, schedule_id):
    if schedule.get("draw_count") != DRAW_COUNT:
        raise RoutedPolicyError(f"execution v1 requires exactly {DRAW_COUNT} draws")
    chunk_count = _require_int(
        schedule.get("chunk_count"), "chunk_count", minimum=0
    )
    expected_chunks = (
        math.ceil(len(parent_rows) / schedule["chunk_size"]) if parent_rows else 0
    )
    if chunk_count != expected_chunks:
        raise RoutedPolicyError("schedule chunk_count does not match parent rows")
    parent_qids = [row["qid"] for row in parent_rows]
    if len(parent_qids) != len(set(parent_qids)):
        raise RoutedPolicyError("parent task qids are not unique")
    for row in parent_rows:
        folder_ids = [item["folder_id"] for item in row["menu"]]
        if len(folder_ids) != len(set(folder_ids)):
            raise RoutedPolicyError(
                f"qid {row['qid']} has duplicate canonical folder IDs"
            )

    children = []
    payloads = {}
    for draw_index in range(DRAW_COUNT):
        draw_qids = []
        for chunk_index in range(chunk_count):
            entry, child_header, child_rows, child_bytes = _child_task(
                parent_header,
                parent_rows,
                schedule,
                schedule_id,
                draw_index,
                chunk_index,
            )
            request_bytes = _render_request(prompt_bytes, child_bytes)
            entry["request"] = {
                "relative_path": _child_relative_path(
                    draw_index, chunk_index, "request.txt"
                ),
                **content_record_bytes(request_bytes),
            }
            children.append(entry)
            payloads[(draw_index, chunk_index)] = {
                "entry": entry,
                "task_header": child_header,
                "task_rows": child_rows,
                "task_bytes": child_bytes,
                "request_bytes": request_bytes,
            }
            draw_qids.extend(row["qid"] for row in child_rows)
        if draw_qids != parent_qids or len(draw_qids) != len(set(draw_qids)):
            raise RoutedPolicyError(
                f"draw {draw_index} is not an exact ordered partition of parent qids"
            )

    per_qid_positions = defaultdict(lambda: defaultdict(Counter))
    for child in children:
        for presentation in child["presentations"]:
            for presented_position, canonical_position in enumerate(
                presentation["presented_to_canonical"]
            ):
                per_qid_positions[presentation["qid"]][canonical_position][
                    presented_position
                ] += 1
    for qid, candidates in per_qid_positions.items():
        menu_size = len(
            next(row["menu"] for row in parent_rows if row["qid"] == qid)
        )
        for canonical_position, counts in candidates.items():
            all_counts = [counts[position] for position in range(menu_size)]
            if max(all_counts) - min(all_counts) > 1:
                raise RoutedPolicyError(
                    f"qid {qid} candidate {canonical_position} is not position-balanced"
                )
    return children, payloads


def _load_plan_envelope(path):
    envelope = _strict_json_file(path)
    if (
        envelope.get("schema") != PLAN_SCHEMA
        or envelope.get("record_type") != "execution_plan"
        or not isinstance(envelope.get("plan_core"), Mapping)
    ):
        raise RoutedPolicyError("malformed routed execution plan")
    rebuilt = _build_plan_envelope(envelope["plan_core"])
    if envelope != rebuilt:
        raise RoutedPolicyError("execution plan envelope/content hash mismatch")
    return envelope


def _default_prompt_path(policy_envelope, policy_path):
    prompt = policy_envelope["policy_core"]["judge_prompt"]
    declared = Path(prompt["path"])
    if declared.is_absolute():
        return declared
    candidates = (
        Path(policy_path).resolve().parent / declared,
        Path(__file__).resolve().parent / declared,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RoutedPolicyError(f"frozen judge prompt not found: {declared}")


def create_execution_plan(
    parent_task_path,
    out_dir,
    *,
    chunk_size,
    namespace,
    prompt_path=None,
    policy_path=None,
):
    """Create one immutable, self-contained no-spend execution plan."""
    parent_task_path = Path(parent_task_path)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise RoutedPolicyError(f"refusing to overwrite execution directory: {out_dir}")
    parent_header, parent_rows, parent_source_record = read_task_file(parent_task_path)
    parent_core = parent_header["task_core"]
    certification = certify_parent_task_current(parent_task_path)
    certification = _validate_certification(
        certification, parent_header, parent_core
    )
    if policy_path is None:
        policy_path = Path(__file__).resolve().parent / "ROUTED_POLICY_three_tier_v1.json"
    policy_path = Path(policy_path)
    policy_envelope = read_policy_file(policy_path)
    if prompt_path is None:
        prompt_path = _default_prompt_path(policy_envelope, policy_path)
    prompt_path = Path(prompt_path)
    parent_bytes = parent_task_path.read_bytes()
    policy_bytes = policy_path.read_bytes()
    prompt_bytes = prompt_path.read_bytes()
    parent_record = content_record_bytes(parent_bytes)
    if parent_record != parent_source_record:
        raise RoutedPolicyError("parent task changed while planning")
    policy_record = content_record_bytes(policy_bytes)
    prompt_record = content_record_bytes(prompt_bytes)
    schedule = _schedule_core(
        parent_header,
        parent_record,
        parent_core,
        prompt_record,
        policy_envelope,
        policy_record,
        namespace=namespace,
        chunk_size=chunk_size,
        certification=certification,
    )
    schedule_id = _id_for(schedule)
    children, payloads = _derive_children(
        parent_header, parent_rows, prompt_bytes, schedule, schedule_id
    )
    plan_core = {
        "schedule_id": schedule_id,
        "schedule": schedule,
        "children": children,
    }
    envelope = _build_plan_envelope(plan_core)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=str(out_dir.parent))
    )
    try:
        _atomic_write_bytes_no_clobber(staging / PARENT_TASK_FILENAME, parent_bytes)
        _atomic_write_bytes_no_clobber(staging / POLICY_FILENAME, policy_bytes)
        _atomic_write_bytes_no_clobber(staging / PROMPT_FILENAME, prompt_bytes)
        for key, payload in payloads.items():
            entry = payload["entry"]
            _atomic_write_bytes_no_clobber(
                staging / entry["task"]["relative_path"], payload["task_bytes"]
            )
            _atomic_write_bytes_no_clobber(
                staging / entry["request"]["relative_path"], payload["request_bytes"]
            )
        _atomic_write_json_no_clobber(staging / PLAN_FILENAME, envelope)
        staged_envelope, _ = verify_execution_plan(staging / PLAN_FILENAME)
        if staged_envelope != envelope:
            raise RoutedPolicyError("staged plan did not verify byte-for-byte")
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, out_dir)
        _fsync_directory(out_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verified, _ = verify_execution_plan(out_dir / PLAN_FILENAME)
    if verified != envelope:
        raise RoutedPolicyError("installed plan did not verify byte-for-byte")
    return envelope


def verify_execution_plan(plan_path):
    """Re-derive all child task and request bytes from the copied parent."""
    plan_path = Path(plan_path)
    envelope = _load_plan_envelope(plan_path)
    core = envelope["plan_core"]
    schedule = core.get("schedule")
    if not isinstance(schedule, Mapping):
        raise RoutedPolicyError("execution plan schedule is missing")
    schedule = dict(schedule)
    root = plan_path.resolve().parent
    parent_receipt = _require_mapping(
        schedule.get("parent_task"), "parent task receipt"
    )
    policy_receipt = _require_mapping(
        schedule.get("routing_policy"), "routing policy receipt"
    )
    prompt_receipt = _require_mapping(schedule.get("prompt"), "prompt receipt")
    parent_path = _resolved_relative(
        root, parent_receipt.get("relative_path"), "parent task path"
    )
    policy_path = _resolved_relative(
        root,
        policy_receipt.get("relative_path"),
        "routing policy path",
    )
    prompt_path = _resolved_relative(
        root, prompt_receipt.get("relative_path"), "prompt path"
    )
    parent_header, parent_rows, parent_record = read_task_file(parent_path)
    policy_envelope = read_policy_file(policy_path)
    policy_record = file_content_record(policy_path)
    prompt_bytes = prompt_path.read_bytes()
    prompt_record = content_record_bytes(prompt_bytes)
    if {
        "relative_path": PARENT_TASK_FILENAME,
        "task_id": parent_header["task_id"],
        **parent_record,
    } != schedule.get("parent_task"):
        raise RoutedPolicyError("plan parent-task receipt mismatch")
    if {
        "relative_path": POLICY_FILENAME,
        "policy_id": policy_envelope["policy_id"],
        **policy_record,
    } != schedule.get("routing_policy"):
        raise RoutedPolicyError("plan routing-policy receipt mismatch")
    if {
        "relative_path": PROMPT_FILENAME,
        **prompt_record,
    } != schedule.get("prompt"):
        raise RoutedPolicyError("plan prompt receipt mismatch")
    certification = _validate_certification(
        schedule.get("certification"), parent_header, parent_header["task_core"]
    )
    expected_schedule = _schedule_core(
        parent_header,
        parent_record,
        parent_header["task_core"],
        prompt_record,
        policy_envelope,
        policy_record,
        namespace=schedule.get("execution_namespace"),
        chunk_size=schedule.get("chunk_size"),
        certification=certification,
        implementation=schedule.get("implementation"),
    )
    if schedule != expected_schedule:
        raise RoutedPolicyError("execution schedule does not re-derive")
    schedule_id = _id_for(schedule)
    if core.get("schedule_id") != schedule_id:
        raise RoutedPolicyError("execution schedule ID mismatch")
    children, payloads = _derive_children(
        parent_header, parent_rows, prompt_bytes, schedule, schedule_id
    )
    if core.get("children") != children:
        raise RoutedPolicyError("execution child manifest does not re-derive")
    for key, payload in payloads.items():
        entry = payload["entry"]
        task_path = _resolved_relative(
            root, entry["task"]["relative_path"], "child task path"
        )
        request_path = _resolved_relative(
            root, entry["request"]["relative_path"], "child request path"
        )
        if task_path.read_bytes() != payload["task_bytes"]:
            raise RoutedPolicyError(f"child task bytes differ for {key}")
        if request_path.read_bytes() != payload["request_bytes"]:
            raise RoutedPolicyError(f"child request bytes differ for {key}")
    context = {
        "root": root,
        "parent_path": parent_path,
        "parent_header": parent_header,
        "parent_rows": parent_rows,
        "policy_envelope": policy_envelope,
        "prompt_bytes": prompt_bytes,
        "schedule": schedule,
        "children": payloads,
    }
    return envelope, context


def _require_utc_timestamp(value, label):
    value = _require_string(value, label)
    if not value.endswith("Z"):
        raise RoutedPolicyError(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise RoutedPolicyError(f"{label} is not an ISO-8601 UTC timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RoutedPolicyError(f"{label} must be UTC")
    return value, parsed


def _attempt_relative_dir(draw_index, chunk_index, attempt_index):
    return (
        f"attempts/draw-{draw_index:03d}/chunk-{chunk_index:03d}/"
        f"attempt-{attempt_index:03d}"
    )


def _build_attempt_envelope(attempt_core):
    core = dict(attempt_core)
    return {
        "schema": ATTEMPT_SCHEMA,
        "record_type": "execution_attempt",
        "attempt_id": _id_for(core),
        "attempt_core": core,
    }


def _validate_attempt_fields(
    *,
    status,
    attempt_index,
    provider_run_id,
    provider_request_id,
    provider_response_id,
    started_at_utc,
    completed_at_utc,
    error_type,
):
    if status not in ("retryable_failure", "success", "terminal_failure"):
        raise RoutedPolicyError(f"unsupported attempt status: {status!r}")
    _require_int(attempt_index, "attempt_index", minimum=0)
    provider_run_id = _require_string(provider_run_id, "provider_run_id")
    provider_request_id = _require_string(
        provider_request_id, "provider_request_id"
    )
    provider_response_id = _require_string(
        provider_response_id, "provider_response_id", allow_empty=True
    )
    started_at_utc, started = _require_utc_timestamp(
        started_at_utc, "started_at_utc"
    )
    completed_at_utc, completed = _require_utc_timestamp(
        completed_at_utc, "completed_at_utc"
    )
    if completed < started:
        raise RoutedPolicyError("attempt completion precedes its start")
    error_type = _require_string(error_type, "error_type", allow_empty=True)
    if status == "success":
        if not provider_response_id:
            raise RoutedPolicyError("successful attempts require provider_response_id")
        if error_type:
            raise RoutedPolicyError("successful attempts cannot declare error_type")
    elif not error_type:
        raise RoutedPolicyError("failed attempts require a nonempty error_type")
    return {
        "provider_run_id": provider_run_id,
        "provider_request_id": provider_request_id,
        "provider_response_id": provider_response_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "error_type": error_type,
    }


def _attempt_core(
    plan_path,
    plan_envelope,
    plan_context,
    child,
    *,
    draw_index,
    chunk_index,
    attempt_index,
    status,
    provider_fields,
    raw_relative_path,
    raw_record,
    pick_rows,
):
    entry = child["entry"]
    return {
        "plan": {
            "relative_path": PLAN_FILENAME,
            "plan_id": plan_envelope["plan_id"],
            **file_content_record(plan_path),
        },
        "schedule_id": plan_envelope["plan_core"]["schedule_id"],
        "parent_task_id": plan_context["parent_header"]["task_id"],
        "coordinate": {
            "draw_index": draw_index,
            "chunk_index": chunk_index,
            "attempt_index": attempt_index,
        },
        "child_task": dict(entry["task"]),
        "request": dict(entry["request"]),
        "attempt_rule_id": ATTEMPT_RULE_ID,
        "status": status,
        "required_judge": dict(plan_context["schedule"]["required_judge"]),
        "provider": {
            "run_id": provider_fields["provider_run_id"],
            "request_id": provider_fields["provider_request_id"],
            "response_id": provider_fields["provider_response_id"],
            "provenance_status": "declared-content-bound-not-provider-authenticated",
        },
        "timing": {
            "started_at_utc": provider_fields["started_at_utc"],
            "completed_at_utc": provider_fields["completed_at_utc"],
        },
        "error_type": provider_fields["error_type"],
        "raw_response": {
            "relative_path": raw_relative_path,
            **dict(raw_record),
        },
        "normalized_pick_rows": _rows_record(pick_rows),
    }


_ATTEMPT_FILE_RE = re.compile(
    r"^attempts/draw-(\d+)/chunk-(\d+)/attempt-(\d+)/"
    r"(attempt\.jsonl|response\.raw)$"
)


def _scan_attempt_artifacts(root):
    attempt_root = Path(root) / "attempts"
    if not attempt_root.exists():
        return {}
    if attempt_root.is_symlink():
        raise RoutedPolicyError("attempt root may not be a symlink")
    artifacts = defaultdict(dict)
    for path in sorted(attempt_root.rglob("*")):
        if path.is_symlink():
            raise RoutedPolicyError(f"attempt artifact may not be a symlink: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        match = _ATTEMPT_FILE_RE.fullmatch(relative)
        if match is None:
            raise RoutedPolicyError(f"unexpected file in attempt tree: {relative}")
        coordinate = tuple(int(match.group(index)) for index in (1, 2, 3))
        kind = match.group(4)
        expected_relative = (
            f"{_attempt_relative_dir(*coordinate)}/{kind}"
        )
        if relative != expected_relative:
            raise RoutedPolicyError(f"noncanonical attempt path: {relative}")
        if kind in artifacts[coordinate]:
            raise RoutedPolicyError(f"duplicate attempt artifact: {relative}")
        artifacts[coordinate][kind] = path
    for coordinate, files in artifacts.items():
        if set(files) != {"attempt.jsonl", "response.raw"}:
            raise RoutedPolicyError(
                f"incomplete attempt artifact pair at coordinate {coordinate}"
            )
    return dict(artifacts)


def _child_for_coordinate(plan_context, draw_index, chunk_index):
    child = plan_context["children"].get((draw_index, chunk_index))
    if child is None:
        raise RoutedPolicyError(
            f"attempt coordinate is not in the plan: draw={draw_index}, "
            f"chunk={chunk_index}"
        )
    return child


def _read_attempt_file(path):
    records = _strict_jsonl(path)
    header, rows = records[0], records[1:]
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != ATTEMPT_SCHEMA
        or header.get("record_type") != "execution_attempt"
        or not isinstance(header.get("attempt_core"), Mapping)
    ):
        raise RoutedPolicyError("malformed routed execution attempt")
    return dict(header), [dict(row) for row in rows]


def _verify_execution_attempt_with_context(
    attempt_path, plan_path, plan_envelope, plan_context
):
    attempt_path = Path(attempt_path)
    header, stored_rows = _read_attempt_file(attempt_path)
    core = header["attempt_core"]
    coordinate = core.get("coordinate")
    if not isinstance(coordinate, Mapping):
        raise RoutedPolicyError("attempt coordinate is missing")
    draw_index = _require_int(
        coordinate.get("draw_index"), "draw_index", minimum=0
    )
    chunk_index = _require_int(
        coordinate.get("chunk_index"), "chunk_index", minimum=0
    )
    attempt_index = _require_int(
        coordinate.get("attempt_index"), "attempt_index", minimum=0
    )
    expected_dir = (
        plan_context["root"]
        / _attempt_relative_dir(draw_index, chunk_index, attempt_index)
    ).resolve()
    if attempt_path.resolve() != expected_dir / "attempt.jsonl":
        raise RoutedPolicyError("attempt file is not at its content-bound path")
    child = _child_for_coordinate(plan_context, draw_index, chunk_index)
    status = core.get("status")
    provider = core.get("provider")
    timing = core.get("timing")
    if not isinstance(provider, Mapping) or not isinstance(timing, Mapping):
        raise RoutedPolicyError("attempt provider/timing provenance is missing")
    if provider.get("provenance_status") != (
        "declared-content-bound-not-provider-authenticated"
    ):
        raise RoutedPolicyError("unsupported provider provenance status")
    provider_fields = _validate_attempt_fields(
        status=status,
        attempt_index=attempt_index,
        provider_run_id=provider.get("run_id"),
        provider_request_id=provider.get("request_id"),
        provider_response_id=provider.get("response_id", ""),
        started_at_utc=timing.get("started_at_utc"),
        completed_at_utc=timing.get("completed_at_utc"),
        error_type=core.get("error_type", ""),
    )
    raw_relative = (
        f"{_attempt_relative_dir(draw_index, chunk_index, attempt_index)}/"
        "response.raw"
    )
    raw_path = _resolved_relative(
        plan_context["root"], raw_relative, "raw response path"
    )
    if raw_path != expected_dir / "response.raw":
        raise RoutedPolicyError("raw response is not at its content-bound path")
    raw_bytes = raw_path.read_bytes()
    raw_record = content_record_bytes(raw_bytes)
    child_path = _resolved_relative(
        plan_context["root"],
        child["entry"]["task"]["relative_path"],
        "child task path",
    )
    if status == "success":
        pick_rows = read_raw_picks_bytes(
            raw_bytes,
            child["task_header"],
            child["task_rows"],
            str(raw_path),
        )
    else:
        pick_rows = []
    if stored_rows != pick_rows:
        raise RoutedPolicyError("attempt normalized picks do not re-derive")
    expected_core = _attempt_core(
        Path(plan_path),
        plan_envelope,
        plan_context,
        child,
        draw_index=draw_index,
        chunk_index=chunk_index,
        attempt_index=attempt_index,
        status=status,
        provider_fields=provider_fields,
        raw_relative_path=raw_relative,
        raw_record=raw_record,
        pick_rows=pick_rows,
    )
    expected_header = _build_attempt_envelope(expected_core)
    if header != expected_header:
        raise RoutedPolicyError("attempt envelope/content hash mismatch")
    if core.get("child_task")["task_id"] != read_task_file(child_path)[0]["task_id"]:
        raise RoutedPolicyError("attempt child task drifted")
    return header, pick_rows


def verify_execution_attempt(attempt_path, plan_path):
    """Verify one copied provider response and its normalized rows."""
    plan_envelope, plan_context = verify_execution_plan(plan_path)
    return _verify_execution_attempt_with_context(
        attempt_path, plan_path, plan_envelope, plan_context
    )


def _verified_attempts(plan_path, plan_context):
    plan_envelope = _load_plan_envelope(plan_path)
    artifacts = _scan_attempt_artifacts(plan_context["root"])
    verified = {}
    for coordinate, files in sorted(artifacts.items()):
        draw_index, chunk_index, attempt_index = coordinate
        _child_for_coordinate(plan_context, draw_index, chunk_index)
        header, rows = _verify_execution_attempt_with_context(
            files["attempt.jsonl"], plan_path, plan_envelope, plan_context
        )
        verified[coordinate] = {
            "header": header,
            "rows": rows,
            "path": files["attempt.jsonl"],
        }
    return verified


def _seal_execution_attempt_unlocked(
    plan_path,
    *,
    draw_index,
    chunk_index,
    attempt_index,
    status,
    raw_response_path,
    provider_run_id,
    provider_request_id,
    provider_response_id,
    started_at_utc,
    completed_at_utc,
    error_type="",
):
    """Copy and seal one provider attempt; no hosted request is performed."""
    draw_index = _require_int(draw_index, "draw_index", minimum=0)
    chunk_index = _require_int(chunk_index, "chunk_index", minimum=0)
    attempt_index = _require_int(attempt_index, "attempt_index", minimum=0)
    provider_fields = _validate_attempt_fields(
        status=status,
        attempt_index=attempt_index,
        provider_run_id=provider_run_id,
        provider_request_id=provider_request_id,
        provider_response_id=provider_response_id,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
        error_type=error_type,
    )
    plan_envelope, plan_context = verify_execution_plan(plan_path)
    child = _child_for_coordinate(plan_context, draw_index, chunk_index)
    existing = _verified_attempts(plan_path, plan_context)
    coordinate_attempts = [
        item
        for coordinate, item in existing.items()
        if coordinate[:2] == (draw_index, chunk_index)
    ]
    coordinate_attempts.sort(
        key=lambda item: item["header"]["attempt_core"]["coordinate"]["attempt_index"]
    )
    indices = [
        item["header"]["attempt_core"]["coordinate"]["attempt_index"]
        for item in coordinate_attempts
    ]
    if indices != list(range(len(indices))) or attempt_index != len(indices):
        raise RoutedPolicyError(
            "attempts must be sealed contiguously from zero with no gaps"
        )
    if coordinate_attempts and coordinate_attempts[-1]["header"]["attempt_core"][
        "status"
    ] != "retryable_failure":
        raise RoutedPolicyError("no attempt may follow a terminal outcome")
    existing_request_ids = {
        item["header"]["attempt_core"]["provider"]["request_id"]
        for item in existing.values()
    }
    existing_response_ids = {
        item["header"]["attempt_core"]["provider"]["response_id"]
        for item in existing.values()
        if item["header"]["attempt_core"]["provider"]["response_id"]
    }
    if provider_fields["provider_request_id"] in existing_request_ids:
        raise RoutedPolicyError("provider request ID was already used in this plan")
    if (
        provider_fields["provider_response_id"]
        and provider_fields["provider_response_id"] in existing_response_ids
    ):
        raise RoutedPolicyError("provider response ID was already used in this plan")

    raw_bytes = Path(raw_response_path).read_bytes()
    raw_record = content_record_bytes(raw_bytes)
    if status == "success":
        pick_rows = read_raw_picks_bytes(
            raw_bytes,
            child["task_header"],
            child["task_rows"],
            str(raw_response_path),
        )
    else:
        pick_rows = []
    relative_dir = _attempt_relative_dir(draw_index, chunk_index, attempt_index)
    raw_relative = f"{relative_dir}/response.raw"
    core = _attempt_core(
        Path(plan_path),
        plan_envelope,
        plan_context,
        child,
        draw_index=draw_index,
        chunk_index=chunk_index,
        attempt_index=attempt_index,
        status=status,
        provider_fields=provider_fields,
        raw_relative_path=raw_relative,
        raw_record=raw_record,
        pick_rows=pick_rows,
    )
    envelope = _build_attempt_envelope(core)
    final_dir = plan_context["root"] / relative_dir
    if final_dir.exists():
        raise RoutedPolicyError(f"refusing to overwrite attempt: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    for ancestor in (
        plan_context["root"] / "attempts",
        plan_context["root"] / "attempts" / f"draw-{draw_index:03d}",
        final_dir.parent,
    ):
        if ancestor.is_symlink():
            raise RoutedPolicyError(
                f"attempt path component may not be a symlink: {ancestor}"
            )
    staging_root = plan_context["root"] / ".staging-attempts"
    staging_root.mkdir(parents=False, exist_ok=True)
    if staging_root.is_symlink():
        raise RoutedPolicyError("attempt staging root may not be a symlink")
    staging = Path(
        tempfile.mkdtemp(prefix=f"{final_dir.name}.", dir=str(staging_root))
    )
    try:
        _atomic_write_bytes_no_clobber(staging / "response.raw", raw_bytes)
        _atomic_write_jsonl_no_clobber(
            staging / "attempt.jsonl", envelope, pick_rows
        )
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, final_dir)
        _fsync_directory(final_dir.parent)
        try:
            staging_root.rmdir()
        except OSError:
            pass
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verified, _ = verify_execution_attempt(final_dir / "attempt.jsonl", plan_path)
    if verified != envelope:
        raise RoutedPolicyError("installed attempt did not verify byte-for-byte")
    return envelope


def seal_execution_attempt(plan_path, **attempt):
    """Serialize, copy, and seal one attempt without making a hosted call."""
    root = Path(plan_path).resolve().parent
    with _PlanWriterLock(root):
        return _seal_execution_attempt_unlocked(plan_path, **attempt)


def _build_bundle_envelope(bundle_core):
    core = dict(bundle_core)
    return {
        "schema": BUNDLE_SCHEMA,
        "record_type": "execution_bundle",
        "bundle_id": _id_for(core),
        "bundle_core": core,
    }


def _attempt_reference(root, item):
    core = item["header"]["attempt_core"]
    coordinate = core["coordinate"]
    path = item["path"]
    return {
        "draw_index": coordinate["draw_index"],
        "chunk_index": coordinate["chunk_index"],
        "attempt_index": coordinate["attempt_index"],
        "attempt_id": item["header"]["attempt_id"],
        "status": core["status"],
        "provider_run_id": core["provider"]["run_id"],
        "provider_request_id": core["provider"]["request_id"],
        "provider_response_id": core["provider"]["response_id"],
        "relative_path": path.relative_to(root).as_posix(),
        **file_content_record(path),
    }


def _complete_attempt_state(plan_path, plan_envelope, plan_context):
    verified = _verified_attempts(plan_path, plan_context)
    request_ids = set()
    response_ids = set()
    attempt_references = []
    successful = {}
    for coordinate, item in sorted(verified.items()):
        core = item["header"]["attempt_core"]
        request_id = core["provider"]["request_id"]
        response_id = core["provider"]["response_id"]
        if request_id in request_ids:
            raise RoutedPolicyError(f"provider request ID reused: {request_id}")
        request_ids.add(request_id)
        if response_id:
            if response_id in response_ids:
                raise RoutedPolicyError(f"provider response ID reused: {response_id}")
            response_ids.add(response_id)
        attempt_references.append(_attempt_reference(plan_context["root"], item))

    for draw_index, chunk_index in sorted(plan_context["children"]):
        items = [
            item
            for coordinate, item in verified.items()
            if coordinate[:2] == (draw_index, chunk_index)
        ]
        items.sort(
            key=lambda item: item["header"]["attempt_core"]["coordinate"][
                "attempt_index"
            ]
        )
        indices = [
            item["header"]["attempt_core"]["coordinate"]["attempt_index"]
            for item in items
        ]
        if indices != list(range(len(indices))):
            raise RoutedPolicyError(
                f"attempt sequence has a gap for draw={draw_index}, chunk={chunk_index}"
            )
        if not items:
            raise RoutedPolicyError(
                f"missing provider call for draw={draw_index}, chunk={chunk_index}"
            )
        statuses = [item["header"]["attempt_core"]["status"] for item in items]
        if any(status != "retryable_failure" for status in statuses[:-1]):
            raise RoutedPolicyError(
                f"attempt sequence has an early terminal outcome for "
                f"draw={draw_index}, chunk={chunk_index}"
            )
        if statuses[-1] == "retryable_failure":
            raise RoutedPolicyError(
                f"provider call is incomplete for draw={draw_index}, chunk={chunk_index}"
            )
        if statuses[-1] == "terminal_failure":
            raise RoutedPolicyError(
                f"provider call terminally failed for draw={draw_index}, "
                f"chunk={chunk_index}; no imputation is allowed"
            )
        successful[(draw_index, chunk_index)] = items[-1]
    if len(successful) != len(plan_context["children"]):
        raise RoutedPolicyError("execution has an unexpected successful-call count")
    attempt_references.sort(
        key=lambda row: (
            row["draw_index"],
            row["chunk_index"],
            row["attempt_index"],
        )
    )
    return successful, attempt_references


def _derive_votes(plan_context, successful):
    votes = defaultdict(list)
    for (draw_index, chunk_index), child in sorted(
        plan_context["children"].items()
    ):
        attempt = successful[(draw_index, chunk_index)]
        pick_by_qid = {row["qid"]: row["pick"] for row in attempt["rows"]}
        if len(pick_by_qid) != len(attempt["rows"]):
            raise RoutedPolicyError("successful attempt contains duplicate qids")
        for row in child["task_rows"]:
            pick = pick_by_qid[row["qid"]]
            folder_id = (
                None if pick is None else row["menu"][pick]["folder_id"]
            )
            votes[row["qid"]].append(
                {
                    "draw_index": draw_index,
                    "chunk_index": chunk_index,
                    "attempt_id": attempt["header"]["attempt_id"],
                    "presented_pick": pick,
                    "folder_id": folder_id,
                }
            )

    vote_rows = []
    pick_rows = []
    outcome_counts = Counter()
    for parent_row in plan_context["parent_rows"]:
        qid = parent_row["qid"]
        qid_votes = sorted(votes.get(qid, []), key=lambda row: row["draw_index"])
        if len(qid_votes) != DRAW_COUNT or {
            row["draw_index"] for row in qid_votes
        } != set(range(DRAW_COUNT)):
            raise RoutedPolicyError(f"qid {qid} does not have all three scheduled votes")
        counts = Counter(row["folder_id"] for row in qid_votes)
        winner, winner_count = counts.most_common(1)[0]
        if winner_count >= 2:
            if winner is None:
                outcome = "null_majority"
                parent_pick = None
                result_folder_id = None
            else:
                outcome = "folder_majority"
                matches = [
                    item["pos"]
                    for item in parent_row["menu"]
                    if item["folder_id"] == winner
                ]
                if len(matches) != 1:
                    raise RoutedPolicyError(
                        f"majority folder {winner!r} is not unique in parent qid {qid}"
                    )
                parent_pick = matches[0]
                result_folder_id = winner
        else:
            outcome = "no_consensus"
            parent_pick = None
            result_folder_id = None
        outcome_counts[outcome] += 1
        vote_rows.append(
            {
                "record_type": "execution_vote",
                "qid": qid,
                "votes": qid_votes,
                "vote_counts": [
                    {"folder_id": folder_id, "count": count}
                    for folder_id, count in sorted(
                        counts.items(),
                        key=lambda pair: (
                            pair[0] is not None,
                            "" if pair[0] is None else pair[0],
                        ),
                    )
                ],
                "outcome": outcome,
                "result_folder_id": result_folder_id,
                "parent_pick": parent_pick,
            }
        )
        pick_rows.append(
            {
                "record_type": "pick",
                "qid": qid,
                "pick": parent_pick,
            }
        )
    if set(votes) != {row["qid"] for row in plan_context["parent_rows"]}:
        raise RoutedPolicyError("execution votes contain qids outside the parent task")
    summary = {
        "qid_count": len(vote_rows),
        "folder_majority_count": outcome_counts["folder_majority"],
        "null_majority_count": outcome_counts["null_majority"],
        "no_consensus_count": outcome_counts["no_consensus"],
    }
    return vote_rows, pick_rows, summary


def _execution_policy_core(plan_envelope, plan_context):
    return {
        "schema": EXECUTION_POLICY_SCHEMA,
        "routing_policy_id": plan_context["policy_envelope"]["policy_id"],
        "plan_id": plan_envelope["plan_id"],
        "aggregation_id": AGGREGATION_ID,
        "draw_count": DRAW_COUNT,
    }


def _aggregate_judge_provenance(
    plan_envelope,
    plan_context,
    attempt_references,
    vote_rows,
):
    attempt_set_sha256 = _rows_record(attempt_references)["sha256"]
    vote_rows_sha256 = _rows_record(vote_rows)["sha256"]
    execution_policy_core = _execution_policy_core(plan_envelope, plan_context)
    execution_policy_id = _id_for(execution_policy_core)
    execution_aggregate = {
        "schema": AGGREGATE_PROVENANCE_SCHEMA,
        "plan_id": plan_envelope["plan_id"],
        "execution_policy_id": execution_policy_id,
        "routing_policy_id": plan_context["policy_envelope"]["policy_id"],
        "aggregation_id": AGGREGATION_ID,
        "draw_count": DRAW_COUNT,
        "attempt_set_sha256": attempt_set_sha256,
        "vote_rows_sha256": vote_rows_sha256,
        "inference_status": INFERENCE_STATUS,
    }
    return {
        **dict(plan_context["schedule"]["required_judge"]),
        "run_id": execution_policy_id,
        "provenance_status": "declared-content-bound-not-provider-authenticated",
        "execution_aggregate": execution_aggregate,
    }


def _derive_bundle_state(plan_path):
    plan_envelope, plan_context = verify_execution_plan(plan_path)
    successful, attempt_references = _complete_attempt_state(
        plan_path, plan_envelope, plan_context
    )
    vote_rows, pick_rows, summary = _derive_votes(plan_context, successful)
    judge = _aggregate_judge_provenance(
        plan_envelope, plan_context, attempt_references, vote_rows
    )
    parent_record = file_content_record(plan_context["parent_path"])
    aggregate_header = build_pick_envelope(
        plan_context["parent_header"],
        parent_record,
        plan_context["parent_rows"],
        pick_rows,
        judge,
    )
    aggregate_bytes = _task_file_bytes(aggregate_header, pick_rows)
    execution = judge["execution_aggregate"]
    bundle_core = {
        "plan": {
            "relative_path": PLAN_FILENAME,
            "plan_id": plan_envelope["plan_id"],
            **file_content_record(plan_path),
        },
        "schedule_id": plan_envelope["plan_core"]["schedule_id"],
        "parent_task_id": plan_context["parent_header"]["task_id"],
        "routing_policy_id": plan_context["policy_envelope"]["policy_id"],
        "execution_policy_id": execution["execution_policy_id"],
        "aggregation_id": AGGREGATION_ID,
        "draw_count": DRAW_COUNT,
        "attempt_rule_id": ATTEMPT_RULE_ID,
        "attempts": attempt_references,
        "attempt_set_sha256": execution["attempt_set_sha256"],
        "vote_rows": _rows_record(vote_rows),
        "aggregate": {
            "relative_path": f"{DERIVED_DIRNAME}/{AGGREGATE_FILENAME}",
            "pick_id": aggregate_header["pick_id"],
            **content_record_bytes(aggregate_bytes),
        },
        "summary": summary,
        "inference_status": INFERENCE_STATUS,
        "confirmatory_inference_authorized": False,
    }
    bundle_header = _build_bundle_envelope(bundle_core)
    bundle_bytes = _task_file_bytes(bundle_header, vote_rows)
    return {
        "plan_envelope": plan_envelope,
        "plan_context": plan_context,
        "attempt_references": attempt_references,
        "vote_rows": vote_rows,
        "pick_rows": pick_rows,
        "judge": judge,
        "aggregate_header": aggregate_header,
        "aggregate_bytes": aggregate_bytes,
        "bundle_header": bundle_header,
        "bundle_bytes": bundle_bytes,
    }


def _build_execution_bundle_unlocked(plan_path):
    """Derive an aggregate only after every logical call terminally succeeds."""
    state = _derive_bundle_state(plan_path)
    root = state["plan_context"]["root"]
    final_dir = root / DERIVED_DIRNAME
    if final_dir.exists():
        raise RoutedPolicyError(f"refusing to overwrite derived bundle: {final_dir}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{DERIVED_DIRNAME}.", dir=str(root))
    )
    try:
        aggregate_path = staging / AGGREGATE_FILENAME
        aggregate_header, aggregate_record = write_pick_file(
            state["plan_context"]["parent_path"],
            aggregate_path,
            state["pick_rows"],
            state["judge"],
        )
        if (
            aggregate_header != state["aggregate_header"]
            or aggregate_path.read_bytes() != state["aggregate_bytes"]
            or aggregate_record
            != content_record_bytes(state["aggregate_bytes"])
        ):
            raise RoutedPolicyError("aggregate pick artifact did not derive exactly")
        _atomic_write_jsonl_no_clobber(
            staging / BUNDLE_FILENAME,
            state["bundle_header"],
            state["vote_rows"],
        )
        if (staging / BUNDLE_FILENAME).read_bytes() != state["bundle_bytes"]:
            raise RoutedPolicyError("bundle artifact did not derive exactly")
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, final_dir)
        _fsync_directory(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verified, _ = verify_execution_bundle(final_dir / BUNDLE_FILENAME)
    if verified != state["bundle_header"]:
        raise RoutedPolicyError("installed bundle did not verify byte-for-byte")
    return state["bundle_header"], state["aggregate_header"]


def build_execution_bundle(plan_path):
    """Serialize final derivation against the immutable attempt prefix."""
    root = Path(plan_path).resolve().parent
    with _PlanWriterLock(root):
        return _build_execution_bundle_unlocked(plan_path)


def verify_execution_bundle(bundle_path):
    """Re-derive the full attempt set, votes, aggregate, and bundle."""
    bundle_path = Path(bundle_path)
    if bundle_path.name != BUNDLE_FILENAME or bundle_path.parent.name != DERIVED_DIRNAME:
        raise RoutedPolicyError("bundle is not at its fixed derived path")
    root = bundle_path.resolve().parent.parent
    expected_bundle_path = root / DERIVED_DIRNAME / BUNDLE_FILENAME
    if bundle_path.resolve() != expected_bundle_path:
        raise RoutedPolicyError("bundle path does not resolve inside its execution root")
    records = _strict_jsonl(bundle_path)
    stored_header, stored_vote_rows = records[0], records[1:]
    if (
        not isinstance(stored_header, Mapping)
        or stored_header.get("schema") != BUNDLE_SCHEMA
        or stored_header.get("record_type") != "execution_bundle"
        or not isinstance(stored_header.get("bundle_core"), Mapping)
    ):
        raise RoutedPolicyError("malformed routed execution bundle")
    if stored_header != _build_bundle_envelope(stored_header["bundle_core"]):
        raise RoutedPolicyError("bundle envelope/content hash mismatch")
    plan_path = root / PLAN_FILENAME
    state = _derive_bundle_state(plan_path)
    if stored_vote_rows != state["vote_rows"]:
        raise RoutedPolicyError("bundle vote rows do not re-derive")
    if stored_header != state["bundle_header"]:
        raise RoutedPolicyError("bundle header does not re-derive")
    aggregate_path = root / DERIVED_DIRNAME / AGGREGATE_FILENAME
    aggregate_header, aggregate_rows, _ = read_pick_file(
        aggregate_path, state["plan_context"]["parent_path"]
    )
    if (
        aggregate_header != state["aggregate_header"]
        or aggregate_rows != state["pick_rows"]
        or aggregate_path.read_bytes() != state["aggregate_bytes"]
    ):
        raise RoutedPolicyError("bundle aggregate does not re-derive")
    return stored_header, state


def _print_identifier(label, envelope):
    identifier = next(
        envelope[key]
        for key in ("plan_id", "attempt_id", "bundle_id", "pick_id")
        if key in envelope
    )
    print(f"{label}: {identifier}")


def run_plan(arguments):
    envelope = create_execution_plan(
        arguments.parent_task,
        arguments.out_dir,
        chunk_size=arguments.chunk_size,
        namespace=arguments.namespace,
        prompt_path=arguments.prompt,
        policy_path=arguments.policy,
    )
    _print_identifier("execution plan", envelope)
    print(
        f"wrote exact local-only requests under {arguments.out_dir}; "
        "this command made no hosted calls"
    )


def run_verify_plan(arguments):
    envelope, _ = verify_execution_plan(arguments.plan)
    _print_identifier("verified execution plan", envelope)


def run_seal_attempt(arguments):
    envelope = seal_execution_attempt(
        arguments.plan,
        draw_index=arguments.draw,
        chunk_index=arguments.chunk,
        attempt_index=arguments.attempt,
        status=arguments.status,
        raw_response_path=arguments.raw_response,
        provider_run_id=arguments.provider_run_id,
        provider_request_id=arguments.provider_request_id,
        provider_response_id=arguments.provider_response_id,
        started_at_utc=arguments.started_at,
        completed_at_utc=arguments.completed_at,
        error_type=arguments.error_type,
    )
    _print_identifier("sealed attempt", envelope)


def run_verify_attempt(arguments):
    envelope, _ = verify_execution_attempt(arguments.attempt_file, arguments.plan)
    _print_identifier("verified attempt", envelope)


def run_build_bundle(arguments):
    bundle, aggregate = build_execution_bundle(arguments.plan)
    _print_identifier("execution bundle", bundle)
    _print_identifier("parent-compatible aggregate", aggregate)
    print(
        "INFERENCE STATUS: integrity/provenance only; execution dependence is "
        "unmodeled and confirmatory inference is unauthorized"
    )


def run_verify_bundle(arguments):
    envelope, _ = verify_execution_bundle(arguments.bundle)
    _print_identifier("verified execution bundle", envelope)
    print(
        "INFERENCE STATUS: integrity/provenance only; execution dependence is "
        "unmodeled and confirmatory inference is unauthorized"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build and verify local routed-judge execution artifacts. "
            "No subcommand calls a hosted provider."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--parent-task", required=True)
    plan.add_argument("--out-dir", required=True)
    plan.add_argument("--chunk-size", type=int, required=True)
    plan.add_argument("--namespace", required=True)
    plan.add_argument("--prompt")
    plan.add_argument("--policy")

    verify_plan = subparsers.add_parser("verify-plan")
    verify_plan.add_argument("--plan", required=True)

    seal_attempt = subparsers.add_parser("seal-attempt")
    seal_attempt.add_argument("--plan", required=True)
    seal_attempt.add_argument("--draw", type=int, required=True)
    seal_attempt.add_argument("--chunk", type=int, required=True)
    seal_attempt.add_argument("--attempt", type=int, required=True)
    seal_attempt.add_argument(
        "--status",
        choices=("retryable_failure", "success", "terminal_failure"),
        required=True,
    )
    seal_attempt.add_argument("--raw-response", required=True)
    seal_attempt.add_argument("--provider-run-id", required=True)
    seal_attempt.add_argument("--provider-request-id", required=True)
    seal_attempt.add_argument("--provider-response-id", default="")
    seal_attempt.add_argument("--started-at", required=True)
    seal_attempt.add_argument("--completed-at", required=True)
    seal_attempt.add_argument("--error-type", default="")

    verify_attempt = subparsers.add_parser("verify-attempt")
    verify_attempt.add_argument("--plan", required=True)
    verify_attempt.add_argument("--attempt-file", required=True)

    bundle = subparsers.add_parser("build-bundle")
    bundle.add_argument("--plan", required=True)

    verify_bundle = subparsers.add_parser("verify-bundle")
    verify_bundle.add_argument("--bundle", required=True)

    arguments = parser.parse_args(argv)
    actions = {
        "plan": run_plan,
        "verify-plan": run_verify_plan,
        "seal-attempt": run_seal_attempt,
        "verify-attempt": run_verify_attempt,
        "build-bundle": run_build_bundle,
        "verify-bundle": run_verify_bundle,
    }
    try:
        actions[arguments.mode](arguments)
    except (OSError, RoutedPolicyError, ValueError) as exc:
        parser.error(f"FAIL-CLOSED: {exc}")


if __name__ == "__main__":
    main()
