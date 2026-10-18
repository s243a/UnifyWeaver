#!/usr/bin/env python3
"""Prepare a deterministic, local-only Pearltrees STEM harvest work order.

The planner consumes a verified diffusion snapshot, an explicit set of public
STEM roots, and the raw API response cache.  It never calls Pearltrees, an LLM,
or an embedding model.  Known-public gaps and visibility-revalidation work are
emitted as separate queues so an authentication failure cannot silently promote
private or unknown data into the public acquisition lane.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import ctypes
from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import sys
import tempfile


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prepare_pearltrees_diffusion_snapshot as diffusion_snapshot  # noqa: E402
from privacy import is_private_title  # noqa: E402


ALGORITHM = "pearltrees-stem-harvest-plan-v1"
SEED_SCHEMA = "pearltrees-stem-harvest-seeds-v1"
MANIFEST_SCHEMA = "pearltrees-stem-harvest-plan-manifest-v1"
QUEUE_SCHEMA = "pearltrees-stem-harvest-queue-v1"
MARKER_NAME = "LOCAL_ONLY_DO_NOT_PUBLISH"
MARKER_BYTES = b"LOCAL ONLY - DO NOT PUBLISH PEARLTREES HARVEST WORK ORDERS\n"
ARTIFACT_NAMES = (
    "public_harvest_queue.json",
    "visibility_revalidation_queue.json",
)
ALL_FILES = frozenset((*ARTIFACT_NAMES, "manifest.json", MARKER_NAME))
TRAVERSAL_RELATIONS = frozenset(("collection", "ref", "path"))
NODE_ID_RE = re.compile(r"pt:([1-9][0-9]*)")
LOCAL_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?"
)
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024
CACHE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
COUNT_NAMES = frozenset(
    (
        "already_fetched_public",
        "private_or_excluded_seen",
        "public_candidates_total",
        "public_queue",
        "reachable_total",
        "revalidation_candidates_total",
        "revalidation_queue",
        "scope_excluded",
        "traversable_public",
    )
)
POLICY = {
    "known_private_nodes_emitted": False,
    "llm_or_embedding_used": False,
    "public_queue_requires_explicit_public_snapshot_state": True,
    "revalidation_queue_is_local_only": True,
    "unknown_nodes_traversed": False,
}


class HarvestPlanError(ValueError):
    """Fail-closed input, planning, or installation error."""


def _canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise HarvestPlanError("value is not canonical JSON") from exc
    return (encoded + "\n").encode("utf-8")


def _duplicate_checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise HarvestPlanError("JSON object contains a duplicate key")
        value[key] = item
    return value


def _reject_nonfinite(_token: str) -> object:
    raise HarvestPlanError("JSON contains a non-finite number")


def _strict_json(data: bytes, label: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonfinite,
        )
    except HarvestPlanError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise HarvestPlanError(f"{label} is not strict UTF-8 JSON") from exc


def _read_bounded(path: Path, label: str, ceiling: int = MAX_INPUT_BYTES) -> bytes:
    try:
        record = path.lstat()
    except OSError as exc:
        raise HarvestPlanError(f"{label} cannot be inspected") from exc
    if not stat.S_ISREG(record.st_mode) or path.is_symlink():
        raise HarvestPlanError(f"{label} must be a non-symlink regular file")
    if record.st_size > ceiling:
        raise HarvestPlanError(f"{label} exceeds its read ceiling")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise HarvestPlanError(f"{label} cannot be read") from exc
    if len(data) != record.st_size:
        raise HarvestPlanError(f"{label} changed while being read")
    return data


def _hash_regular_file(path: Path, label: str, ceiling: int) -> str:
    """Hash a large regular file without retaining a second in-memory copy."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise HarvestPlanError(f"{label} cannot be opened") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > ceiling:
            raise HarvestPlanError(f"{label} is not a bounded regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > ceiling:
                raise HarvestPlanError(f"{label} exceeds its read ceiling")
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise HarvestPlanError(f"{label} cannot be read") from exc
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if total != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise HarvestPlanError(f"{label} changed while being read")
    return digest.hexdigest()


def _sqlite_header_uses_wal(path: Path) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            header = os.read(descriptor, 20)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise HarvestPlanError("API response cache header cannot be read") from exc
    return (
        len(header) >= 20
        and header[:16] == b"SQLite format 3\x00"
        and (header[18] == 2 or header[19] == 2)
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _node_id(value: object, label: str = "node ID") -> str:
    if not isinstance(value, str) or NODE_ID_RE.fullmatch(value) is None:
        raise HarvestPlanError(f"{label} is not canonical")
    return value


def _node_number(node_id: str) -> str:
    return NODE_ID_RE.fullmatch(node_id).group(1)  # type: ignore[union-attr]


def _node_key(node_id: str) -> int:
    return int(_node_number(node_id))


def _load_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    data = _read_bounded(path, label)
    if data and not data.endswith(b"\n"):
        raise HarvestPlanError(f"{label} is not newline terminated")
    rows: list[dict[str, object]] = []
    for line in data.splitlines():
        value = _strict_json(line, label)
        if not isinstance(value, dict):
            raise HarvestPlanError(f"{label} row must be an object")
        rows.append(value)
    return rows


def _load_seeds(
    path: Path, expected_snapshot_fingerprint: str
) -> tuple[list[str], list[str], str, str]:
    data = _read_bounded(path, "STEM seed manifest", 1_000_000)
    value = _strict_json(data, "STEM seed manifest")
    if not isinstance(value, dict) or set(value) != {
        "exclude_roots",
        "roots",
        "schema",
        "snapshot_fingerprint",
        "public_cache_not_before",
    }:
        raise HarvestPlanError("STEM seed manifest fields mismatch")
    if (
        value.get("schema") != SEED_SCHEMA
        or value.get("snapshot_fingerprint") != expected_snapshot_fingerprint
        or not isinstance(value.get("roots"), list)
        or not isinstance(value.get("exclude_roots"), list)
    ):
        raise HarvestPlanError("STEM seed manifest schema mismatch")
    roots = [_node_id(item, "STEM root ID") for item in value["roots"]]
    exclude_roots = [
        _node_id(item, "excluded scope root ID") for item in value["exclude_roots"]
    ]
    if not roots or len(roots) != len(set(roots)):
        raise HarvestPlanError("STEM roots must be nonempty and unique")
    if len(exclude_roots) != len(set(exclude_roots)) or set(roots) & set(exclude_roots):
        raise HarvestPlanError("scope exclusions must be unique and disjoint from roots")
    roots = sorted(roots, key=_node_key)
    exclude_roots = sorted(exclude_roots, key=_node_key)
    public_cache_not_before = value["public_cache_not_before"]
    _local_timestamp(public_cache_not_before, "public cache cutoff")
    semantic_sha = _sha256(
        _canonical_json(
            {
                "exclude_roots": exclude_roots,
                "roots": roots,
                "schema": SEED_SCHEMA,
                "snapshot_fingerprint": expected_snapshot_fingerprint,
                "public_cache_not_before": public_cache_not_before,
            }
        )
    )
    return roots, exclude_roots, semantic_sha, public_cache_not_before


def _load_snapshot(
    snapshot_dir: Path,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], str, set[str]]:
    try:
        manifest = diffusion_snapshot.verify_snapshot(snapshot_dir)
    except Exception as exc:
        raise HarvestPlanError("diffusion snapshot verification failed") from exc
    fingerprint = manifest.get("snapshot_fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise HarvestPlanError("diffusion snapshot fingerprint is malformed")
    nodes_list = _load_jsonl(snapshot_dir / "nodes.jsonl", "snapshot nodes")
    evidence = _load_jsonl(snapshot_dir / "edge_evidence.jsonl", "snapshot edge evidence")
    nodes: dict[str, dict[str, object]] = {}
    for row in nodes_list:
        node = _node_id(row.get("node_id"))
        if node in nodes:
            raise HarvestPlanError("snapshot contains a duplicate node")
        if row.get("visibility") not in {"public", "private", "unknown"}:
            raise HarvestPlanError("snapshot node visibility is malformed")
        if not isinstance(row.get("excluded"), bool):
            raise HarvestPlanError("snapshot node exclusion is malformed")
        nodes[node] = row
    source_records = manifest.get("fingerprint_core", {}).get("source_records")
    if not isinstance(source_records, list):
        raise HarvestPlanError("diffusion snapshot source records are malformed")
    api_sqlite_hashes: set[str] = set()
    for source in source_records:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("kind"), str)
            or not isinstance(source.get("content_records"), list)
        ):
            raise HarvestPlanError("diffusion snapshot source records are malformed")
        for record in source["content_records"]:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            ):
                raise HarvestPlanError("diffusion snapshot source content record is malformed")
            if source["kind"] == "api_sqlite":
                api_sqlite_hashes.add(record["sha256"])
    return nodes, evidence, fingerprint, api_sqlite_hashes


def _canonical_cache_id(value: object) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise HarvestPlanError("API cache tree ID is not canonical")
    if value == "0" or (len(value) > 1 and value.startswith("0")):
        raise HarvestPlanError("API cache tree ID is not canonical")
    return f"pt:{value}"


def _cache_response_status(tree_id: str, raw: object) -> str:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return "malformed"
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonfinite,
        )
    except (HarvestPlanError, json.JSONDecodeError, TypeError, ValueError):
        return "malformed"
    if not isinstance(payload, dict):
        return "malformed"
    wrappers = [key for key in ("api_response", "response") if key in payload]
    if len(wrappers) > 1:
        return "malformed"
    response = payload[wrappers[0]] if wrappers else payload
    if not isinstance(response, dict):
        return "malformed"
    info = response.get("info", {})
    tree = response.get("tree", {})
    if info is None:
        info = {}
    if tree is None:
        tree = {}
    if not isinstance(info, dict) or not isinstance(tree, dict):
        return "malformed"

    candidate_ids = [
        value
        for value in (
            info.get("id"),
            tree.get("id"),
            payload.get("tree_id"),
            response.get("treeId"),
        )
        if value is not None
    ]
    parsed_ids: set[str] = set()
    for value in candidate_ids:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            return "malformed"
        try:
            parsed_ids.add(_canonical_cache_id(str(value)))
        except HarvestPlanError:
            return "malformed"
    if parsed_ids and parsed_ids != {tree_id}:
        return "malformed"

    public_claim = False
    private_claim = False
    masked_claim = False
    for mapping in (info, tree):
        title = mapping.get("title")
        if title is not None and not isinstance(title, str):
            return "malformed"
        visibility = mapping.get("visibility")
        visibility_value: int | None
        if visibility is None or (isinstance(visibility, str) and not visibility.strip()):
            visibility_value = None
        elif isinstance(visibility, bool) or isinstance(visibility, float):
            return "malformed"
        else:
            raw_visibility = str(visibility)
            if (
                not raw_visibility.isascii()
                or not raw_visibility.isdigit()
                or (len(raw_visibility) > 1 and raw_visibility.startswith("0"))
            ):
                return "malformed"
            visibility_value = int(raw_visibility)
        normalized_title = (title or "").strip().casefold()
        exact_mask = visibility_value == 2 and normalized_title in {
            "*private*",
            "private",
        }
        if visibility_value == 0:
            public_claim = True
        if exact_mask:
            masked_claim = True
        elif visibility_value not in (None, 0) or is_private_title(title):
            private_claim = True
    if private_claim:
        return "private_or_restricted"
    if masked_claim:
        return "masked_auth"
    return "public" if public_claim else "malformed"


def _local_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or LOCAL_TIMESTAMP_RE.fullmatch(value) is None:
        raise HarvestPlanError(f"{label} is not a canonical offset-free ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HarvestPlanError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is not None:
        raise HarvestPlanError(f"{label} must use the harvester's offset-free clock domain")
    return parsed


def _load_api_cache(
    path: Path, public_cache_not_before: str
) -> tuple[dict[str, str], str]:
    if any(Path(f"{path}{suffix}").exists() for suffix in CACHE_SIDECAR_SUFFIXES):
        raise HarvestPlanError("API response cache must be checkpointed without sidecars")
    digest = _hash_regular_file(path, "API response cache", MAX_CACHE_BYTES)
    if _sqlite_header_uses_wal(path):
        raise HarvestPlanError("WAL-mode API response caches are not accepted")
    cutoff = _local_timestamp(public_cache_not_before, "public cache cutoff")
    uri = f"{path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
        if not journal_mode or str(journal_mode[0]).casefold() == "wal":
            raise HarvestPlanError("WAL-mode API response caches are not accepted")
        if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise HarvestPlanError("API response cache fails SQLite quick_check")
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(api_responses)").fetchall()
        }
        if not {"tree_id", "fetched_at", "response_json"}.issubset(columns):
            raise HarvestPlanError("API response cache schema mismatch")
        statuses: dict[str, str] = {}
        for raw_id, fetched_at, response_json in connection.execute(
            "SELECT tree_id, fetched_at, response_json FROM api_responses ORDER BY tree_id"
        ):
            tree_id = _canonical_cache_id(raw_id)
            if tree_id in statuses:
                raise HarvestPlanError("API response cache has duplicate tree IDs")
            status = _cache_response_status(tree_id, response_json)
            if status == "public":
                try:
                    fetched = _local_timestamp(fetched_at, "API cache fetched_at")
                except HarvestPlanError:
                    status = "malformed"
                else:
                    if fetched < cutoff:
                        status = "stale_public"
            statuses[tree_id] = status
    except HarvestPlanError:
        raise
    except sqlite3.Error as exc:
        raise HarvestPlanError("API response cache cannot be queried") from exc
    finally:
        if "connection" in locals():
            connection.close()
    if any(Path(f"{path}{suffix}").exists() for suffix in CACHE_SIDECAR_SUFFIXES) or (
        _hash_regular_file(path, "API response cache", MAX_CACHE_BYTES) != digest
    ):
        raise HarvestPlanError("API response cache changed during planning")
    return statuses, digest


def _priority(distance: int, frontier_links: int, root_count: int) -> tuple[int, str]:
    if distance == 0:
        return 0, "seed_gap"
    if frontier_links >= 2 or root_count >= 2:
        return 1, "shared_containment_frontier"
    return 2, "direct_containment_frontier"


def _queue_row(
    node_id: str,
    *,
    distance: int,
    frontier_links: int,
    root_count: int,
    lane: str,
    cache_status: str,
) -> dict[str, object]:
    tier, reason = _priority(distance, frontier_links, root_count)
    return {
        "cache_status": cache_status,
        "distance_hops": distance,
        "frontier_links": frontier_links,
        "priority_tier": tier,
        "queued_by": ALGORITHM,
        "reason": f"{lane}_{reason}",
        "stem_root_count": root_count,
        "tree_id": _node_number(node_id),
    }


def derive_work_orders(
    nodes: dict[str, dict[str, object]],
    evidence: list[dict[str, object]],
    roots: list[str],
    exclude_roots: list[str],
    cache_status: dict[str, str],
    *,
    max_hops: int,
    batch_limit: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Derive public and revalidation work without statistical weights."""

    if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
        raise HarvestPlanError("max_hops must be a positive integer")
    if isinstance(batch_limit, bool) or not isinstance(batch_limit, int) or batch_limit < 1:
        raise HarvestPlanError("batch_limit must be a positive integer")
    for root in roots:
        node = nodes.get(root)
        if node is None or node.get("visibility") != "public" or node.get("excluded") is not False:
            raise HarvestPlanError("every STEM root must be explicitly public and retained")
        if cache_status.get(root, "missing") not in {"public", "missing", "stale_public"}:
            raise HarvestPlanError("a STEM root conflicts with API privacy evidence")

    children: dict[str, set[str]] = defaultdict(set)
    for row in evidence:
        if row.get("relation") not in TRAVERSAL_RELATIONS:
            continue
        child = _node_id(row.get("child"), "edge child")
        parent = _node_id(row.get("parent"), "edge parent")
        if row.get("privacy_endpoint_excluded") is not False:
            continue
        if child not in nodes or parent not in nodes:
            raise HarvestPlanError("edge evidence cites an unknown node")
        children[parent].add(child)

    scope_excluded: set[str] = set()
    scope_queue: deque[str] = deque()
    for root in exclude_roots:
        if root not in nodes:
            raise HarvestPlanError("excluded scope root is absent from the snapshot")
        scope_excluded.add(root)
        scope_queue.append(root)
    while scope_queue:
        parent = scope_queue.popleft()
        for child in sorted(children.get(parent, ()), key=_node_key):
            if child not in scope_excluded:
                scope_excluded.add(child)
                scope_queue.append(child)
    if set(roots) & scope_excluded:
        raise HarvestPlanError("a STEM root lies below an explicit scope exclusion")

    distance_by_node: dict[str, int] = {}
    roots_by_node: dict[str, set[str]] = defaultdict(set)
    traversable_public: set[str] = set()
    for root in roots:
        queue: deque[tuple[str, int]] = deque([(root, 0)])
        seen = {root}
        while queue:
            node_id, distance = queue.popleft()
            node = nodes[node_id]
            if node_id in scope_excluded:
                continue
            roots_by_node[node_id].add(root)
            distance_by_node[node_id] = min(distance_by_node.get(node_id, distance), distance)
            if (
                node.get("visibility") != "public"
                or node.get("excluded") is not False
                or cache_status.get(node_id, "missing") != "public"
            ):
                # This is the first unresolved frontier for this branch.  Do not
                # traverse through it until a fresh snapshot resolves the node.
                continue
            traversable_public.add(node_id)
            if distance == max_hops:
                continue
            for child in sorted(children.get(node_id, ()), key=_node_key):
                if child in scope_excluded:
                    continue
                child_row = nodes[child]
                candidate_distance = distance + 1
                roots_by_node[child].add(root)
                distance_by_node[child] = min(
                    distance_by_node.get(child, candidate_distance), candidate_distance
                )
                if (
                    child_row.get("visibility") == "public"
                    and child_row.get("excluded") is False
                    and cache_status.get(child, "missing") == "public"
                    and child not in seen
                ):
                    seen.add(child)
                    queue.append((child, candidate_distance))

    frontier_links: dict[str, int] = defaultdict(int)
    for parent in traversable_public:
        for child in children.get(parent, ()):
            if child in distance_by_node:
                frontier_links[child] += 1

    public_queue: list[dict[str, object]] = []
    revalidation_queue: list[dict[str, object]] = []
    counts: defaultdict[str, int] = defaultdict(int, {name: 0 for name in COUNT_NAMES})
    for node_id in sorted(distance_by_node, key=_node_key):
        row = nodes[node_id]
        visibility = row.get("visibility")
        excluded = row.get("excluded")
        status = cache_status.get(node_id, "missing")
        distance = distance_by_node[node_id]
        links = frontier_links[node_id]
        root_count = len(roots_by_node[node_id])
        if excluded is True or visibility == "private":
            counts["private_or_excluded_seen"] += 1
            continue
        if visibility == "public":
            if status == "public":
                counts["already_fetched_public"] += 1
            elif status == "missing":
                public_queue.append(
                    _queue_row(
                        node_id,
                        distance=distance,
                        frontier_links=links,
                        root_count=root_count,
                        lane="public",
                        cache_status=status,
                    )
                )
            elif status == "stale_public":
                revalidation_queue.append(
                    _queue_row(
                        node_id,
                        distance=distance,
                        frontier_links=links,
                        root_count=root_count,
                        lane="public_cache_revalidation",
                        cache_status=status,
                    )
                )
            else:
                raise HarvestPlanError("snapshot and API cache privacy states disagree")
        elif visibility == "unknown":
            if status == "public":
                raise HarvestPlanError("snapshot omitted a public API visibility claim")
            elif status in {"missing", "stale_public"}:
                revalidation_queue.append(
                    _queue_row(
                        node_id,
                        distance=distance,
                        frontier_links=links,
                        root_count=root_count,
                        lane="unknown_visibility_revalidation",
                        cache_status=status,
                    )
                )
            else:
                raise HarvestPlanError("snapshot and API cache privacy states disagree")
        else:
            raise HarvestPlanError("reachable node has invalid visibility")

    sort_key = lambda item: (
        item["priority_tier"],
        item["distance_hops"],
        -item["frontier_links"],
        -item["stem_root_count"],
        int(item["tree_id"]),
    )
    public_queue.sort(key=sort_key)
    revalidation_queue.sort(key=sort_key)
    public_total = len(public_queue)
    revalidation_total = len(revalidation_queue)
    public_queue = public_queue[:batch_limit]
    revalidation_queue = revalidation_queue[:batch_limit]
    public_queue = [dict(row, rank=index) for index, row in enumerate(public_queue, 1)]
    revalidation_queue = [
        dict(row, rank=index) for index, row in enumerate(revalidation_queue, 1)
    ]
    counts.update(
        {
            "public_queue": len(public_queue),
            "public_candidates_total": public_total,
            "revalidation_queue": len(revalidation_queue),
            "revalidation_candidates_total": revalidation_total,
            "traversable_public": len(traversable_public),
            "reachable_total": len(distance_by_node),
            "scope_excluded": len(scope_excluded),
        }
    )
    return public_queue, revalidation_queue, dict(sorted(counts.items()))


def _artifact_record(data: bytes) -> dict[str, object]:
    return {"bytes": len(data), "sha256": _sha256(data)}


def _queue_payload(lane: str, rows: list[dict[str, object]]) -> bytes:
    return _canonical_json(
        {
            "count": len(rows),
            "lane": lane,
            "maps": rows,
            "schema": QUEUE_SCHEMA,
        }
    )


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_git_worktree_marker(path: Path) -> bool:
    try:
        if path.is_file():
            return path.read_bytes()[:64].lstrip().startswith(b"gitdir:")
        return path.is_dir() and (path / "HEAD").is_file()
    except OSError:
        return True


def _safe_output(output_dir: Path, local_root: Path) -> tuple[Path, Path]:
    output = Path(os.path.abspath(output_dir))
    root = Path(os.path.abspath(local_root))
    try:
        root_mode = root.lstat().st_mode
        parent_mode = output.parent.lstat().st_mode
    except OSError as exc:
        raise HarvestPlanError("local root or output parent is unavailable") from exc
    if not stat.S_ISDIR(root_mode) or not stat.S_ISDIR(parent_mode):
        raise HarvestPlanError("local root and output parent must be directories")
    root = root.resolve(strict=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if output.exists() or output.is_symlink():
        raise HarvestPlanError("output directory already exists")
    if not _path_within(output, root) or output == root:
        raise HarvestPlanError("output directory must be below the explicit local root")
    if _path_within(output, REPO_ROOT.resolve()) or any(
        _is_git_worktree_marker(ancestor / ".git")
        for ancestor in (parent, *parent.parents)
    ):
        raise HarvestPlanError("local-only output cannot be installed in the Git worktree")
    return output, parent


def _reject_input_output_overlap(output: Path, inputs: list[Path]) -> None:
    for raw in inputs:
        try:
            source = raw.resolve(strict=True)
        except OSError as exc:
            raise HarvestPlanError("planner input cannot be resolved") from exc
        if source == output or _path_within(source, output) or _path_within(output, source):
            raise HarvestPlanError("planner inputs and output cannot overlap")


def _write_private(path: Path, data: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise HarvestPlanError("local-only artifact could not be written") from exc


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise HarvestPlanError("atomic no-replace installation is unavailable") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1) == 0:
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise HarvestPlanError("output appeared during atomic installation")
    raise HarvestPlanError("atomic no-replace installation failed") from OSError(
        number, os.strerror(number)
    )


def _validate_queue_rows(rows: object, lane: str) -> list[dict[str, object]]:
    if not isinstance(rows, list):
        raise HarvestPlanError("harvest queue maps must be a list")
    expected_fields = {
        "cache_status",
        "distance_hops",
        "frontier_links",
        "priority_tier",
        "queued_by",
        "rank",
        "reason",
        "stem_root_count",
        "tree_id",
    }
    validated: list[dict[str, object]] = []
    for expected_rank, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise HarvestPlanError("harvest queue row fields mismatch")
        for key in ("distance_hops", "frontier_links", "priority_tier", "rank", "stem_root_count"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HarvestPlanError("harvest queue numeric field is malformed")
        if row["rank"] != expected_rank or row["stem_root_count"] < 1:
            raise HarvestPlanError("harvest queue rank or root count is malformed")
        tree_id = _canonical_cache_id(row["tree_id"])
        tier, suffix = _priority(
            row["distance_hops"], row["frontier_links"], row["stem_root_count"]
        )
        if tier != row["priority_tier"] or row["queued_by"] != ALGORITHM:
            raise HarvestPlanError("harvest queue priority provenance mismatch")
        if lane == "known_public":
            allowed_reasons = {f"public_{suffix}"}
            allowed_status = {"missing"}
        else:
            allowed_reasons = {
                f"public_cache_revalidation_{suffix}",
                f"unknown_visibility_revalidation_{suffix}",
            }
            allowed_status = {"missing", "stale_public"}
        if row["reason"] not in allowed_reasons or row["cache_status"] not in allowed_status:
            raise HarvestPlanError("harvest queue lane semantics mismatch")
        if _node_number(tree_id) != row["tree_id"]:
            raise HarvestPlanError("harvest queue tree ID is noncanonical")
        validated.append(row)
    sort_key = lambda item: (
        item["priority_tier"],
        item["distance_hops"],
        -item["frontier_links"],
        -item["stem_root_count"],
        int(item["tree_id"]),
    )
    if validated != sorted(validated, key=sort_key):
        raise HarvestPlanError("harvest queue order is noncanonical")
    if len({row["tree_id"] for row in validated}) != len(validated):
        raise HarvestPlanError("harvest queue contains duplicate tree IDs")
    return validated


def _verify_artifacts(run_dir: Path) -> dict[str, object]:
    try:
        directory_record = run_dir.lstat()
        names = {entry.name for entry in run_dir.iterdir()}
    except OSError as exc:
        raise HarvestPlanError("harvest plan cannot be inspected") from exc
    if not stat.S_ISDIR(directory_record.st_mode) or stat.S_IMODE(directory_record.st_mode) != 0o700:
        raise HarvestPlanError("harvest plan directory mode must be 0700")
    if names != ALL_FILES:
        raise HarvestPlanError("harvest plan file set mismatch")
    payloads: dict[str, bytes] = {}
    for name in ALL_FILES:
        path = run_dir / name
        record = path.lstat()
        if (
            not stat.S_ISREG(record.st_mode)
            or record.st_nlink != 1
            or stat.S_IMODE(record.st_mode) != 0o600
        ):
            raise HarvestPlanError("harvest plan artifact envelope is invalid")
        payloads[name] = _read_bounded(path, f"harvest plan {name}")
    if payloads[MARKER_NAME] != MARKER_BYTES:
        raise HarvestPlanError("local-only marker mismatch")
    manifest = _strict_json(payloads["manifest.json"], "harvest plan manifest")
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {"artifacts", "counts", "fingerprint_core", "plan_fingerprint", "policy", "schema"}
        or manifest.get("schema") != MANIFEST_SCHEMA
        or _canonical_json(manifest) != payloads["manifest.json"]
    ):
        raise HarvestPlanError("harvest plan manifest schema mismatch")
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or set(records) != set(ARTIFACT_NAMES):
        raise HarvestPlanError("harvest plan artifact records mismatch")
    for name in ARTIFACT_NAMES:
        if records[name] != _artifact_record(payloads[name]):
            raise HarvestPlanError("harvest plan artifact hash mismatch")
    core = manifest.get("fingerprint_core")
    fingerprint = manifest.get("plan_fingerprint")
    if isinstance(core, dict):
        _local_timestamp(core.get("public_cache_not_before"), "public cache cutoff")
    if (
        not isinstance(core, dict)
        or set(core)
        != {
            "algorithm",
            "api_cache_sha256",
            "artifacts",
            "batch_limit",
            "counts",
            "max_hops",
            "policy",
            "public_cache_not_before",
            "root_set_sha256",
            "snapshot_fingerprint",
            "traversal_relations",
        }
        or core.get("algorithm") != ALGORITHM
        or core.get("artifacts") != records
        or core.get("counts") != manifest.get("counts")
        or core.get("policy") != manifest.get("policy")
        or core.get("traversal_relations") != sorted(TRAVERSAL_RELATIONS)
        or any(
            not isinstance(core.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", core[key]) is None
            for key in ("api_cache_sha256", "root_set_sha256", "snapshot_fingerprint")
        )
        or any(
            isinstance(core.get(key), bool)
            or not isinstance(core.get(key), int)
            or core[key] < 1
            for key in ("batch_limit", "max_hops")
        )
        or fingerprint != _sha256(_canonical_json(core))
    ):
        raise HarvestPlanError("harvest plan fingerprint mismatch")
    if manifest.get("policy") != POLICY:
        raise HarvestPlanError("harvest plan policy mismatch")
    queue_rows: dict[str, list[dict[str, object]]] = {}
    for name, lane in (
        ("public_harvest_queue.json", "known_public"),
        ("visibility_revalidation_queue.json", "local_visibility_revalidation"),
    ):
        queue = _strict_json(payloads[name], name)
        if (
            not isinstance(queue, dict)
            or set(queue) != {"count", "lane", "maps", "schema"}
            or queue.get("schema") != QUEUE_SCHEMA
            or queue.get("lane") != lane
            or not isinstance(queue.get("maps"), list)
            or queue.get("count") != len(queue["maps"])
            or _canonical_json(queue) != payloads[name]
        ):
            raise HarvestPlanError("harvest queue schema mismatch")
        queue_rows[lane] = _validate_queue_rows(queue["maps"], lane)
    all_ids = [
        row["tree_id"]
        for rows in queue_rows.values()
        for row in rows
    ]
    if len(all_ids) != len(set(all_ids)):
        raise HarvestPlanError("harvest work lanes overlap")
    counts = manifest.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != COUNT_NAMES
        or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in counts.items()
        )
    ):
        raise HarvestPlanError("harvest plan counts are malformed")
    public = _strict_json(payloads["public_harvest_queue.json"], "public queue")
    revalidation = _strict_json(
        payloads["visibility_revalidation_queue.json"], "revalidation queue"
    )
    if (
        counts.get("public_queue") != public["count"]
        or counts.get("revalidation_queue") != revalidation["count"]
        or counts.get("public_candidates_total") < public["count"]
        or counts.get("revalidation_candidates_total") < revalidation["count"]
    ):
        raise HarvestPlanError("harvest plan counts disagree with artifacts")
    return manifest


def verify_plan(
    run_dir: str | os.PathLike[str],
    snapshot_dir: str | os.PathLike[str],
    api_cache_db: str | os.PathLike[str],
    seed_manifest: str | os.PathLike[str],
) -> dict[str, object]:
    """Re-derive a plan from its bound inputs and compare exact artifact bytes."""

    run = Path(run_dir)
    manifest = _verify_artifacts(run)
    core = manifest["fingerprint_core"]
    nodes, evidence, snapshot_fingerprint, api_sqlite_hashes = _load_snapshot(
        Path(snapshot_dir)
    )
    roots, exclude_roots, root_set_sha, public_cache_not_before = _load_seeds(
        Path(seed_manifest), snapshot_fingerprint
    )
    cache_status, cache_sha = _load_api_cache(
        Path(api_cache_db), public_cache_not_before
    )
    if cache_sha not in api_sqlite_hashes:
        raise HarvestPlanError("API cache is not an exact source of the verified snapshot")
    if (
        snapshot_fingerprint != core["snapshot_fingerprint"]
        or root_set_sha != core["root_set_sha256"]
        or cache_sha != core["api_cache_sha256"]
        or public_cache_not_before != core["public_cache_not_before"]
    ):
        raise HarvestPlanError("harvest plan input binding mismatch")
    public, revalidation, counts = derive_work_orders(
        nodes,
        evidence,
        roots,
        exclude_roots,
        cache_status,
        max_hops=core["max_hops"],
        batch_limit=core["batch_limit"],
    )
    expected = {
        "public_harvest_queue.json": _queue_payload("known_public", public),
        "visibility_revalidation_queue.json": _queue_payload(
            "local_visibility_revalidation", revalidation
        ),
    }
    for name, data in expected.items():
        if _read_bounded(run / name, f"harvest plan {name}") != data:
            raise HarvestPlanError("harvest plan does not reproduce from bound inputs")
    if manifest["counts"] != counts:
        raise HarvestPlanError("harvest plan counts do not reproduce from bound inputs")
    return manifest


def prepare_plan(
    snapshot_dir: str | os.PathLike[str],
    api_cache_db: str | os.PathLike[str],
    seed_manifest: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    local_root: str | os.PathLike[str],
    *,
    max_hops: int,
    batch_limit: int,
) -> dict[str, object]:
    output, parent = _safe_output(Path(output_dir), Path(local_root))
    _reject_input_output_overlap(
        output, [Path(snapshot_dir), Path(api_cache_db), Path(seed_manifest)]
    )
    nodes, evidence, snapshot_fingerprint, api_sqlite_hashes = _load_snapshot(
        Path(snapshot_dir)
    )
    roots, exclude_roots, root_set_sha, public_cache_not_before = _load_seeds(
        Path(seed_manifest), snapshot_fingerprint
    )
    cache_status, cache_sha = _load_api_cache(
        Path(api_cache_db), public_cache_not_before
    )
    if cache_sha not in api_sqlite_hashes:
        raise HarvestPlanError("API cache is not an exact source of the verified snapshot")
    public, revalidation, counts = derive_work_orders(
        nodes,
        evidence,
        roots,
        exclude_roots,
        cache_status,
        max_hops=max_hops,
        batch_limit=batch_limit,
    )
    payloads = {
        "public_harvest_queue.json": _queue_payload("known_public", public),
        "visibility_revalidation_queue.json": _queue_payload(
            "local_visibility_revalidation", revalidation
        ),
    }
    artifact_records = {name: _artifact_record(data) for name, data in payloads.items()}
    core: dict[str, object] = {
        "algorithm": ALGORITHM,
        "api_cache_sha256": cache_sha,
        "artifacts": artifact_records,
        "batch_limit": batch_limit,
        "counts": counts,
        "max_hops": max_hops,
        "policy": POLICY,
        "public_cache_not_before": public_cache_not_before,
        "root_set_sha256": root_set_sha,
        "snapshot_fingerprint": snapshot_fingerprint,
        "traversal_relations": sorted(TRAVERSAL_RELATIONS),
    }
    manifest: dict[str, object] = {
        "artifacts": artifact_records,
        "counts": counts,
        "fingerprint_core": core,
        "plan_fingerprint": _sha256(_canonical_json(core)),
        "policy": POLICY,
        "schema": MANIFEST_SCHEMA,
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    os.chmod(temporary, 0o700)
    try:
        for name, data in payloads.items():
            _write_private(temporary / name, data)
        _write_private(temporary / "manifest.json", _canonical_json(manifest))
        _write_private(temporary / MARKER_NAME, MARKER_BYTES)
        descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _verify_artifacts(temporary)
        _rename_noreplace(temporary, output)
        parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return verify_plan(output, snapshot_dir, api_cache_db, seed_manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--snapshot-dir", required=True)
    prepare.add_argument("--api-cache-db", required=True)
    prepare.add_argument("--seed-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--local-root", required=True)
    prepare.add_argument("--local-only", action="store_true", required=True)
    prepare.add_argument("--max-hops", type=int, default=8)
    prepare.add_argument("--batch-limit", type=int, default=128)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--snapshot-dir", required=True)
    verify.add_argument("--api-cache-db", required=True)
    verify.add_argument("--seed-manifest", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        verification = "source_reproduced"
        if args.command == "prepare":
            manifest = prepare_plan(
                args.snapshot_dir,
                args.api_cache_db,
                args.seed_manifest,
                args.output_dir,
                args.local_root,
                max_hops=args.max_hops,
                batch_limit=args.batch_limit,
            )
        elif args.command == "verify":
            manifest = verify_plan(
                args.run_dir,
                args.snapshot_dir,
                args.api_cache_db,
                args.seed_manifest,
            )
        else:
            manifest = _verify_artifacts(Path(args.run_dir))
            verification = "structural_integrity_only"
        print(
            json.dumps(
                {
                    "counts": manifest["counts"],
                    "plan_fingerprint": manifest["plan_fingerprint"],
                    "verification": verification,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception:
        print(json.dumps({"error": "STEM harvest plan failed closed"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
