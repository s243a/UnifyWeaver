#!/usr/bin/env python3
"""Integrity primitives for public-only routed filing tasks and policies.

The hashes below provide content binding, not provider authentication.  Judge
identity remains declared provenance, but changing a task, menu, pick, policy,
or provenance field necessarily changes the corresponding identifier.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from filing_privacy import (
    POLICY_ID as PRIVACY_POLICY_ID,
    PUBLIC_CATALOG_POLICY_ID,
)


TASK_SCHEMA = "unifyweaver.routed-task.v2"
RAW_PICKS_SCHEMA = "unifyweaver.raw-routed-picks.v1"
PICKS_SCHEMA = "unifyweaver.routed-picks.v2"
POLICY_SCHEMA = "unifyweaver.routed-policy.v1"


class RoutedPolicyError(ValueError):
    """A routed task/pick/policy artifact failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_record_bytes(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": sha256_bytes(data)}


def file_content_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    return content_record_bytes(Path(path).read_bytes())


def _strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise RoutedPolicyError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _bad_constant(value):
    raise RoutedPolicyError(f"non-finite JSON number: {value}")


def strict_json_loads(data: bytes | str, source="<json>"):
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_bad_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RoutedPolicyError(f"malformed JSON in {source}: {exc}") from exc


def _require_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RoutedPolicyError(f"{label} must be an integer")
    return value


def _ordered_qid_record(qids: Sequence[int]) -> dict[str, Any]:
    for qid in qids:
        _require_int(qid, "qid")
    data = b"".join(canonical_json_bytes(qid) for qid in qids)
    return {"count": len(qids), "sha256": sha256_bytes(data)}


def _rows_record(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    data = b"".join(canonical_json_bytes(dict(row)) for row in rows)
    return {
        "count": len(rows),
        "qid_sha256": _ordered_qid_record([row["qid"] for row in rows])["sha256"],
        "sha256": sha256_bytes(data),
    }


def _id_for(core: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(core)))


def float64_hex(value: float | None) -> str | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise RoutedPolicyError("band boundary must be finite")
    return value.hex()


def parse_float64_hex(value: str | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RoutedPolicyError("band boundary must be a float.hex string or null")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise RoutedPolicyError(f"invalid float.hex boundary: {value!r}") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise RoutedPolicyError(f"noncanonical float.hex boundary: {value!r}")
    return parsed


def make_band(lower: float | None, upper: float | None) -> dict[str, Any]:
    low = None if lower is None else float(lower)
    high = None if upper is None else float(upper)
    if low is not None and high is not None and not low < high:
        raise RoutedPolicyError("band lower boundary must be smaller than upper")
    return {
        "lower_float64_hex": float64_hex(low),
        "upper_float64_hex": float64_hex(high),
        "semantics": "lower-inclusive_upper-exclusive",
    }


def parse_band(band: Mapping[str, Any]) -> tuple[float | None, float | None]:
    if not isinstance(band, Mapping):
        raise RoutedPolicyError("band must be an object")
    if band.get("semantics") != "lower-inclusive_upper-exclusive":
        raise RoutedPolicyError("unsupported band semantics")
    low = parse_float64_hex(band.get("lower_float64_hex"))
    high = parse_float64_hex(band.get("upper_float64_hex"))
    if low is not None and high is not None and not low < high:
        raise RoutedPolicyError("band is empty or reversed")
    return low, high


def in_band(value: float, band: Mapping[str, Any]) -> bool:
    value = float(value)
    if not math.isfinite(value):
        raise RoutedPolicyError("margin must be finite")
    low, high = parse_band(band)
    return (low is None or value >= low) and (high is None or value < high)


def band_qids(margins: Sequence[float], band: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(qid for qid, margin in enumerate(margins) if in_band(float(margin), band))


def _validate_task_row(row: Mapping[str, Any]):
    if not isinstance(row, Mapping) or row.get("record_type") != "task":
        raise RoutedPolicyError("task row has wrong record_type")
    _require_int(row.get("qid"), "task qid")
    if not isinstance(row.get("bookmark"), str) or not row["bookmark"]:
        raise RoutedPolicyError("task bookmark must be a nonempty string")
    menu = row.get("menu")
    if not isinstance(menu, list) or not menu:
        raise RoutedPolicyError("task menu must be a nonempty list")
    for expected, item in enumerate(menu):
        if not isinstance(item, Mapping):
            raise RoutedPolicyError("menu item must be an object")
        if _require_int(item.get("pos"), "menu position") != expected:
            raise RoutedPolicyError("menu positions must be contiguous and zero based")
        if not isinstance(item.get("title"), str) or not item["title"]:
            raise RoutedPolicyError("menu title must be a nonempty string")
        if not isinstance(item.get("folder_id"), str) or not item["folder_id"]:
            raise RoutedPolicyError("menu folder_id must be a nonempty string")
        if "path" in item and not isinstance(item["path"], str):
            raise RoutedPolicyError("menu path must be a string")


def build_task_envelope(
    task_core: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    seen = set()
    for row in rows:
        _validate_task_row(row)
        if row["qid"] in seen:
            raise RoutedPolicyError(f"duplicate task qid: {row['qid']}")
        seen.add(row["qid"])
    core = dict(task_core)
    if "task_rows" in core:
        raise RoutedPolicyError("caller must not prepopulate task_rows")
    selection = core.get("selection")
    if not isinstance(selection, Mapping):
        raise RoutedPolicyError("task selection is missing")
    expected_menu_size = _require_int(selection.get("menu_size"), "menu_size")
    if expected_menu_size <= 0:
        raise RoutedPolicyError("menu_size must be positive")
    for row in rows:
        if len(row["menu"]) != expected_menu_size:
            raise RoutedPolicyError(
                f"qid {row['qid']} has {len(row['menu'])} menu items, "
                f"expected {expected_menu_size}"
            )
    core["task_rows"] = _rows_record(rows)
    return {
        "schema": TASK_SCHEMA,
        "record_type": "task_envelope",
        "task_id": _id_for(core),
        "task_core": core,
    }


def _atomic_write_jsonl(
    path: str | os.PathLike[str],
    header: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(dict(header)))
            for row in rows:
                handle.write(canonical_json_bytes(dict(row)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return file_content_record(target)


def write_task_file(path, task_core, rows):
    header = build_task_envelope(task_core, rows)
    record = _atomic_write_jsonl(path, header, rows)
    return header, record


def _read_jsonl(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        rows.append(strict_json_loads(line, f"{path}:{line_no}"))
    if not rows:
        raise RoutedPolicyError(f"empty JSONL file: {path}")
    return rows


def read_task_file(path):
    records = _read_jsonl(path)
    header, rows = records[0], records[1:]
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != TASK_SCHEMA
        or header.get("record_type") != "task_envelope"
    ):
        raise RoutedPolicyError("legacy or malformed task file; v2 envelope required")
    core = header.get("task_core")
    if not isinstance(core, Mapping):
        raise RoutedPolicyError("task_core is missing")
    rebuilt = build_task_envelope(
        {key: value for key, value in core.items() if key != "task_rows"}, rows
    )
    if dict(header) != rebuilt:
        raise RoutedPolicyError("task envelope/content hash mismatch")
    return dict(header), [dict(row) for row in rows], file_content_record(path)


def _validate_pick_rows(task_rows, pick_rows):
    menu_len = {row["qid"]: len(row["menu"]) for row in task_rows}
    expected = tuple(row["qid"] for row in task_rows)
    picks = {}
    for row in pick_rows:
        if not isinstance(row, Mapping) or row.get("record_type") != "pick":
            raise RoutedPolicyError("pick row has wrong record_type")
        qid = _require_int(row.get("qid"), "pick qid")
        if qid in picks:
            raise RoutedPolicyError(f"duplicate pick qid: {qid}")
        pick = row.get("pick")
        if pick is not None:
            pick = _require_int(pick, f"pick for qid {qid}")
        picks[qid] = pick
    actual = tuple(row["qid"] for row in pick_rows)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extras = sorted(set(actual) - set(expected))
        raise RoutedPolicyError(
            f"pick qids do not exactly match task; missing={missing[:5]}, extras={extras[:5]}"
        )
    for qid, pick in picks.items():
        if pick is not None and not 0 <= pick < menu_len[qid]:
            raise RoutedPolicyError(f"pick out of range for qid {qid}: {pick}")
    return picks


def read_raw_picks(path, task_header, task_rows):
    records = _read_jsonl(path)
    raw_header, records = records[0], records[1:]
    if (
        not isinstance(raw_header, Mapping)
        or raw_header.get("schema") != RAW_PICKS_SCHEMA
        or raw_header.get("record_type") != "raw_pick_header"
        or raw_header.get("task_id") != task_header["task_id"]
        or set(raw_header) != {"schema", "record_type", "task_id"}
    ):
        raise RoutedPolicyError(
            "raw picks require an exact task_id header; unbound legacy rows cannot be sealed"
        )
    rows = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"qid", "pick"}:
            raise RoutedPolicyError(
                "raw picks must contain exactly {qid,pick}; legacy headers require an explicit migration"
            )
        rows.append(
            {
                "record_type": "pick",
                "qid": record["qid"],
                "pick": record["pick"],
            }
        )
    _validate_pick_rows(task_rows, rows)
    return rows


def build_pick_envelope(
    task_header,
    task_file_record,
    task_rows,
    pick_rows,
    judge_provenance,
):
    _validate_pick_rows(task_rows, pick_rows)
    expected = _ordered_qid_record([row["qid"] for row in task_rows])
    core = {
        "task_id": task_header["task_id"],
        "task_file": dict(task_file_record),
        "expected_qids": expected,
        "judge": dict(judge_provenance),
        "pick_rows": _rows_record(pick_rows),
    }
    return {
        "schema": PICKS_SCHEMA,
        "record_type": "pick_envelope",
        "pick_id": _id_for(core),
        "pick_core": core,
    }


def seal_pick_file(task_path, raw_picks_path, out_path, judge_provenance):
    task_header, task_rows, task_file_record = read_task_file(task_path)
    pick_rows = read_raw_picks(raw_picks_path, task_header, task_rows)
    header = build_pick_envelope(
        task_header,
        task_file_record,
        task_rows,
        pick_rows,
        judge_provenance,
    )
    record = _atomic_write_jsonl(out_path, header, pick_rows)
    return header, record


def read_pick_file(path, task_path):
    task_header, task_rows, task_file_record = read_task_file(task_path)
    records = _read_jsonl(path)
    header, pick_rows = records[0], records[1:]
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != PICKS_SCHEMA
        or header.get("record_type") != "pick_envelope"
    ):
        raise RoutedPolicyError("legacy or malformed picks; v2 envelope required")
    core = header.get("pick_core")
    if not isinstance(core, Mapping):
        raise RoutedPolicyError("pick_core is missing")
    rebuilt = build_pick_envelope(
        task_header,
        task_file_record,
        task_rows,
        pick_rows,
        core.get("judge", {}),
    )
    if dict(header) != rebuilt:
        raise RoutedPolicyError("pick envelope/content hash mismatch")
    return dict(header), [dict(row) for row in pick_rows], dict(
        _validate_pick_rows(task_rows, pick_rows)
    )


def build_policy_envelope(policy_core):
    core = dict(policy_core)
    return {
        "schema": POLICY_SCHEMA,
        "record_type": "policy_envelope",
        "policy_id": _id_for(core),
        "policy_core": core,
    }


def read_policy_file(path):
    envelope = strict_json_loads(Path(path).read_bytes(), str(path))
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("schema") != POLICY_SCHEMA
        or envelope.get("record_type") != "policy_envelope"
    ):
        raise RoutedPolicyError("malformed routed policy")
    core = envelope.get("policy_core")
    if not isinstance(core, Mapping) or dict(envelope) != build_policy_envelope(core):
        raise RoutedPolicyError("policy envelope/content hash mismatch")
    validate_policy_core(core)
    return dict(envelope)


def validate_policy_core(core):
    if core.get("evidence_status") != "exploratory_transductive":
        raise RoutedPolicyError("current routed policy must remain exploratory_transductive")
    if core.get("primary_grading") != "exact_destination_id":
        raise RoutedPolicyError("primary grading must be exact_destination_id")
    if core.get("privacy_policy_id") != PRIVACY_POLICY_ID:
        raise RoutedPolicyError("policy must bind the public-only privacy policy")
    if core.get("catalog_policy_id") != PUBLIC_CATALOG_POLICY_ID:
        raise RoutedPolicyError("policy must bind the public catalog eligibility rule")
    prompt = core.get("judge_prompt")
    if (
        not isinstance(prompt, Mapping)
        or not isinstance(prompt.get("path"), str)
        or not isinstance(prompt.get("sha256"), str)
        or len(prompt["sha256"]) != 64
    ):
        raise RoutedPolicyError("policy must bind the frozen judge prompt")
    tiers = core.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise RoutedPolicyError("policy tiers are missing")
    seen_ids = set()
    previous_upper = None
    for index, tier in enumerate(tiers):
        if not isinstance(tier, Mapping):
            raise RoutedPolicyError("policy tier must be an object")
        tier_id = tier.get("tier_id")
        if not isinstance(tier_id, str) or not tier_id or tier_id in seen_ids:
            raise RoutedPolicyError("policy tier ids must be unique nonempty strings")
        seen_ids.add(tier_id)
        low, high = parse_band(tier.get("band", {}))
        if index == 0 and low is not None:
            raise RoutedPolicyError("first policy tier must start at -infinity")
        if index > 0 and low is None:
            raise RoutedPolicyError("only the first policy tier may start at -infinity")
        if index < len(tiers) - 1 and high is None:
            raise RoutedPolicyError("only the final policy tier may end at +infinity")
        if index and low != previous_upper:
            raise RoutedPolicyError("policy tiers overlap or leave a gap")
        action = tier.get("action")
        if action not in ("judge", "auto_top1"):
            raise RoutedPolicyError(f"unsupported tier action: {action!r}")
        if action == "judge":
            if _require_int(tier.get("menu_size"), "menu_size") <= 0:
                raise RoutedPolicyError("judge menu_size must be positive")
            if not tier.get("lineage") or not isinstance(
                tier.get("required_judge_family"), str
            ):
                raise RoutedPolicyError("judge tier must freeze lineage and required_judge")
        previous_upper = high
    if previous_upper is not None:
        raise RoutedPolicyError("last policy tier must end at +infinity")
    bootstrap = core.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise RoutedPolicyError("policy bootstrap contract is missing")
    if bootstrap.get("unit") != "bookmark-folder-connected-component":
        raise RoutedPolicyError("unsupported bootstrap unit")
    if _require_int(bootstrap.get("replicates"), "bootstrap replicates") <= 0:
        raise RoutedPolicyError("bootstrap replicates must be positive")
    _require_int(bootstrap.get("seed"), "bootstrap seed")
    interval = bootstrap.get("interval")
    if interval != [0.025, 0.975]:
        raise RoutedPolicyError("policy interval must be the frozen [0.025,0.975]")


def policy_tier_for_margin(core, margin):
    matched = [tier for tier in core["tiers"] if in_band(margin, tier["band"])]
    if len(matched) != 1:
        raise RoutedPolicyError(f"margin matched {len(matched)} policy tiers")
    return matched[0]


def paired_node_block_bootstrap(
    policy_correct: Sequence[bool],
    baseline_correct: Sequence[bool],
    bookmark_nodes: Sequence[str],
    folder_nodes: Sequence[str],
    *,
    replicates=9999,
    seed=3867001,
):
    """Paired bootstrap over bookmark/folder connected components.

    Rows sharing either a bookmark-title node or destination-folder node remain
    together, so repeated destinations and duplicate bookmark titles cannot be
    split across resampled blocks.
    """
    policy = np.asarray(policy_correct, dtype=float)
    baseline = np.asarray(baseline_correct, dtype=float)
    if not (
        len(policy)
        == len(baseline)
        == len(bookmark_nodes)
        == len(folder_nodes)
        and len(policy) > 0
    ):
        raise RoutedPolicyError("bootstrap inputs must have equal nonzero length")
    if replicates <= 0:
        raise RoutedPolicyError("bootstrap replicates must be positive")

    parent = list(range(len(policy)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first = {}
    for index, (bookmark, folder) in enumerate(zip(bookmark_nodes, folder_nodes)):
        for node in (f"B:{bookmark}", f"F:{folder}"):
            if node in first:
                union(index, first[node])
            else:
                first[node] = index
    blocks = defaultdict(list)
    for index in range(len(policy)):
        blocks[find(index)].append(index)
    block_rows = [np.asarray(rows, dtype=int) for _, rows in sorted(blocks.items())]
    delta = policy - baseline
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for draw in range(replicates):
        chosen = rng.integers(0, len(block_rows), len(block_rows))
        rows = np.concatenate([block_rows[index] for index in chosen])
        samples[draw] = float(delta[rows].mean())
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "point": float(delta.mean()),
        "lower_0.025": float(lo),
        "upper_0.975": float(hi),
        "bootstrap_mean": float(samples.mean()),
        "replicates": int(replicates),
        "seed": int(seed),
        "block_count": len(block_rows),
        "unit": "bookmark-folder-connected-component",
    }
