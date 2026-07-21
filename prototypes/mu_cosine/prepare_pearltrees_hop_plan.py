#!/usr/bin/env python3
"""Freeze the outcome-blind, no-solve Pearltrees HOP fidelity plan.

This tool consumes only the verified structural artifacts from canonical
snapshot attempt A.  It selects anchors, protected sets, strict HOP prefixes,
exact cut ledgers, references, and calibration shells.  It never imports a
numerical array package, builds a precision matrix, calibrates leakage, or
executes a solve.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
from functools import lru_cache
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

import declare_pearltrees_diffusion_sources as declaration
import prepare_pearltrees_diffusion_consensus as consensus


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PROTOCOL_PATH = HERE / "PROTOCOL_bounded_diffusion_fidelity.md"
DESIGN_PATH = HERE / "DESIGN_pearltrees_hop_plan.md"

SCHEMA = "pearltrees-hop-fidelity-plan-v2"
ALGORITHM = "outcome-blind-nested-hop-plan-v2"
MARKER_NAME = "LOCAL_ONLY_DO_NOT_PUBLISH"
MARKER_BYTES = b"LOCAL ONLY - DO NOT PUBLISH HOP PLAN NODE ARTIFACTS\n"
MANIFEST_NAME = "manifest.json"
ARTIFACT_NAMES = (
    "quartiles.jsonl",
    "selected_anchors.jsonl",
    "batches.jsonl",
    "anchor_traversals.jsonl",
    "domains.jsonl",
    "boundaries.jsonl",
    "calibration_shells.jsonl",
    "bootstrap_multiplicities.jsonl",
)
ALL_PLAN_FILES = frozenset(ARTIFACT_NAMES + (MANIFEST_NAME, MARKER_NAME))

SELECTION_SEED = 3_882_001
BOOTSTRAP_SEED = 3_882_002
QUARTILE_COUNT = 4
ANCHORS_PER_QUARTILE = 32
CALIBRATION_PER_QUARTILE = 8
AUDIT_PER_QUARTILE = 24
ANCHORS_PER_BATCH = 4
PROTECTED_NONANCHORS = 16
CANDIDATE_BUDGETS = (256, 512, 1024)
REFERENCE_BUDGET = 4096
CALIBRATION_SHELL_RADIUS = 3
CALIBRATION_TARGET = "exp(-1)"
DENSE_ARRAY_COUNT = 4
FLOAT64_BYTES = 8
MINIMUM_RECIPROCAL_CONDITION_HEX = "0x1.0000000000000p-26"
MAXIMUM_PLAN_MANIFEST_BYTES = 4 * 1024 * 1024
MAXIMUM_CONSENSUS_RECEIPT_BYTES = 4 * 1024 * 1024
MAXIMUM_ATTEMPT_MANIFEST_BYTES = 16 * 1024 * 1024
MAXIMUM_SOURCE_SPEC_BYTES = 16 * 1024 * 1024
MAXIMUM_PRIVATE_BUNDLE_FILES = 256

HASH_KEY_ENCODING = "sha256-canonical-json-line-[3882001,purpose,typed_id]-v1"
TYPED_ID_ORDER = "namespace-then-positive-decimal-integer-v1"
REFERENCE_RULE = "continue-anchor-multisource-hop-order-through-4096-v1"


class HopPlanError(ValueError):
    """Fail-closed plan integrity or operational error."""


@dataclass
class _TouchBudget:
    ceiling: int
    observed: int = 0

    def add(self, amount):
        self.observed += int(amount)
        if self.observed > self.ceiling:
            raise HopPlanError("planner edge-touch ceiling exceeded")


def _duplicate_checked_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise HopPlanError("duplicate JSON key")
        value[key] = item
    return value


def _reject_nonfinite(_value):
    raise HopPlanError("non-finite JSON constant")


def _canonical_json(value):
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HopPlanError("value is not canonical finite JSON") from exc
    return (text + "\n").encode("utf-8")


def _strict_json_bytes(data, label):
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HopPlanError(f"invalid canonical JSON in {label}") from exc
    if _canonical_json(value) != data:
        raise HopPlanError(f"noncanonical JSON in {label}")
    return value


def _jsonl_bytes(records):
    return b"".join(_canonical_json(record) for record in records)


def _strict_jsonl_bytes(data, label):
    if data and not data.endswith(b"\n"):
        raise HopPlanError(f"noncanonical JSONL in {label}")
    records = []
    for line in data.splitlines(keepends=True):
        records.append(_strict_json_bytes(line, label))
    return records


def _content_record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _content_record_is_valid(value):
    return (
        isinstance(value, dict)
        and set(value) == {"sha256", "size_bytes"}
        and isinstance(value["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
        and isinstance(value["size_bytes"], int)
        and not isinstance(value["size_bytes"], bool)
        and value["size_bytes"] >= 0
    )


def _file_record(path):
    try:
        return _content_record(Path(path).read_bytes())
    except OSError as exc:
        raise HopPlanError("required implementation artifact is unavailable") from exc


def _typed_id_key(value):
    if not isinstance(value, str):
        raise HopPlanError("structural artifact contains an invalid typed node ID")
    match = re.fullmatch(r"([a-z][a-z0-9_-]*):([1-9][0-9]*)", value)
    if match is None:
        raise HopPlanError("structural artifact contains an invalid typed node ID")
    return match.group(1), int(match.group(2))


def _selection_key(purpose, node_id):
    if purpose not in {"select", "split", "batch"}:
        raise HopPlanError("unknown selection-key purpose")
    digest = hashlib.sha256(
        _canonical_json([SELECTION_SEED, purpose, node_id])
    ).hexdigest()
    return digest, _typed_id_key(node_id)


def _path_is_within(path, root):
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def _is_git_worktree_marker(path):
    try:
        path = Path(path)
        if path.is_symlink():
            return True
        if path.is_file():
            with open(path, "rb") as stream:
                return stream.read(256).lstrip().startswith(b"gitdir:")
        return path.is_dir() and (path / "HEAD").is_file()
    except OSError:
        return True


def _has_git_ancestor(path):
    resolved = Path(path).resolve()
    return any(
        _is_git_worktree_marker(ancestor / ".git")
        for ancestor in (resolved, *resolved.parents)
    )


def _has_symlink_ancestor(path):
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        try:
            if candidate.is_symlink():
                return True
        except OSError:
            return True
    return False


@dataclass
class _BoundDirectory:
    path: Path
    fd: int
    device: int
    inode: int

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _bind_directory(path, label):
    raw = Path(path)
    if _has_symlink_ancestor(raw):
        raise HopPlanError(f"{label} cannot contain a symlink")
    resolved = raw.resolve()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(resolved, flags)
        observed = os.fstat(fd)
        current = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        try:
            os.close(fd)
        except (OSError, UnboundLocalError):
            pass
        raise HopPlanError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(observed.st_mode) or (
        observed.st_dev,
        observed.st_ino,
    ) != (current.st_dev, current.st_ino):
        os.close(fd)
        raise HopPlanError(f"{label} identity mismatch")
    return _BoundDirectory(resolved, fd, observed.st_dev, observed.st_ino)


def _assert_bound_directory(binding, label):
    if binding.fd < 0:
        raise HopPlanError(f"{label} binding is closed")
    try:
        observed = os.fstat(binding.fd)
        current = os.stat(binding.path, follow_symlinks=False)
    except OSError as exc:
        raise HopPlanError(f"{label} changed during planning") from exc
    if (
        observed.st_dev,
        observed.st_ino,
        current.st_dev,
        current.st_ino,
    ) != (binding.device, binding.inode, binding.device, binding.inode):
        raise HopPlanError(f"{label} changed during planning")


def _bind_child_directory(parent_binding, leaf, label):
    if (
        not isinstance(leaf, str)
        or leaf in {"", ".", ".."}
        or Path(leaf).name != leaf
    ):
        raise HopPlanError(f"{label} leaf is malformed")
    _assert_bound_directory(parent_binding, "plan output parent")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(leaf, flags, dir_fd=parent_binding.fd)
        observed = os.fstat(fd)
        current = os.stat(leaf, dir_fd=parent_binding.fd, follow_symlinks=False)
    except OSError as exc:
        try:
            os.close(fd)
        except (OSError, UnboundLocalError):
            pass
        raise HopPlanError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != (current.st_dev, current.st_ino)
    ):
        os.close(fd)
        raise HopPlanError(f"{label} identity mismatch")
    return _BoundDirectory(
        parent_binding.path / leaf, fd, observed.st_dev, observed.st_ino
    )


def _create_bound_staging(parent_binding, target_leaf):
    _assert_bound_directory(parent_binding, "plan output parent")
    for _attempt in range(100):
        leaf = f".{target_leaf}.staging-{os.urandom(16).hex()}"
        try:
            os.mkdir(leaf, 0o700, dir_fd=parent_binding.fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise HopPlanError("HOP plan staging directory could not be created") from exc
        binding = _bind_child_directory(
            parent_binding, leaf, "HOP plan staging directory"
        )
        try:
            os.fchmod(binding.fd, 0o700)
            if stat.S_IMODE(os.fstat(binding.fd).st_mode) != 0o700:
                raise HopPlanError("HOP plan staging mode could not be fixed")
            _assert_bound_directory(binding, "HOP plan staging directory")
            return leaf, binding
        except Exception:
            binding.close()
            raise
    raise HopPlanError("HOP plan staging name allocation failed")


def _read_at_most(fd, maximum_size, label):
    if (
        not isinstance(maximum_size, int)
        or isinstance(maximum_size, bool)
        or maximum_size < 0
    ):
        raise HopPlanError(f"{label} read ceiling is malformed")
    chunks = []
    remaining = maximum_size + 1
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) > maximum_size:
        raise HopPlanError(f"{label} artifact exceeds its bound")
    return data


def _read_bound_file(
    binding, name, expected_record, label, *, maximum_size=None
):
    if expected_record is not None and not _content_record_is_valid(expected_record):
        raise HopPlanError(f"{label} expected content record is malformed")
    if expected_record is None and maximum_size is None:
        raise HopPlanError(f"{label} requires a read ceiling")
    read_ceiling = (
        expected_record["size_bytes"]
        if expected_record is not None
        else maximum_size
    )
    if maximum_size is not None:
        read_ceiling = min(read_ceiling, maximum_size)
    _assert_bound_directory(binding, label)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        fd = os.open(name, flags, dir_fd=binding.fd)
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise HopPlanError(f"{label} artifact contract mismatch")
        if expected_record is not None and before.st_size != expected_record["size_bytes"]:
            raise HopPlanError(f"{label} artifact size mismatch")
        if before.st_size > read_ceiling:
            raise HopPlanError(f"{label} artifact exceeds its bound")
        data = _read_at_most(fd, read_ceiling, label)
        after = os.fstat(fd)
    except OSError as exc:
        raise HopPlanError(f"{label} artifact is unavailable") from exc
    finally:
        try:
            os.close(fd)
        except (OSError, UnboundLocalError):
            pass
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or after.st_nlink != 1
        or stat.S_IMODE(after.st_mode) != 0o600
    ):
        raise HopPlanError(f"{label} artifact changed during read")
    if expected_record is not None and _content_record(data) != expected_record:
        raise HopPlanError(f"{label} artifact content mismatch")
    _assert_bound_directory(binding, label)
    return data


def _read_regular_path(path, expected_record, label):
    if not _content_record_is_valid(expected_record):
        raise HopPlanError(f"{label} expected content record is malformed")
    raw = Path(path)
    if _has_symlink_ancestor(raw):
        raise HopPlanError(f"{label} cannot contain a symlink")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        fd = os.open(raw, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise HopPlanError(f"{label} must be a unique regular file")
        if before.st_size != expected_record["size_bytes"]:
            raise HopPlanError(f"{label} size mismatch")
        data = _read_at_most(fd, expected_record["size_bytes"], label)
        after = os.fstat(fd)
    except OSError as exc:
        raise HopPlanError(f"{label} is unavailable") from exc
    finally:
        try:
            os.close(fd)
        except (OSError, UnboundLocalError):
            pass
    if (
        (before.st_dev, before.st_ino, before.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
        or after.st_nlink != 1
    ):
        raise HopPlanError(f"{label} changed during read")
    if _content_record(data) != expected_record:
        raise HopPlanError(f"{label} content mismatch")
    return data


def _bounded_directory_names(binding, maximum_count, label):
    if (
        not isinstance(maximum_count, int)
        or isinstance(maximum_count, bool)
        or maximum_count < 0
    ):
        raise HopPlanError(f"{label} inventory ceiling is malformed")
    _assert_bound_directory(binding, label)
    names = []
    try:
        with os.scandir(binding.fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > maximum_count:
                    raise HopPlanError(f"{label} artifact inventory exceeds its bound")
    except OSError as exc:
        raise HopPlanError(f"{label} artifact inventory is unavailable") from exc
    _assert_bound_directory(binding, label)
    return tuple(names)


def _assert_private_directory_files(
    binding, label, *, maximum_count=MAXIMUM_PRIVATE_BUNDLE_FILES
):
    """Recheck the already-verified private bundle's leaf envelope by dirfd."""

    _assert_bound_directory(binding, label)
    try:
        names = _bounded_directory_names(binding, maximum_count, label)
        for name in names:
            fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=binding.fd,
            )
            try:
                observed = os.fstat(fd)
            finally:
                os.close(fd)
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or stat.S_IMODE(observed.st_mode) != 0o600
            ):
                raise HopPlanError(f"{label} artifact contract mismatch")
    except OSError as exc:
        raise HopPlanError(f"{label} artifact inventory is unavailable") from exc
    _assert_bound_directory(binding, label)


def _git_commit():
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HopPlanError("repository commit is unavailable") from exc
    commit = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise HopPlanError("repository commit is malformed")
    return commit


def _hop_order(anchors, adjacency, maximum_nodes=None, *, touch_budget=None):
    anchors = tuple(sorted(set(anchors), key=_typed_id_key))
    if not anchors:
        raise HopPlanError("HOP traversal requires anchors")
    if any(anchor not in adjacency for anchor in anchors):
        raise HopPlanError("HOP traversal anchor is absent from adjacency")
    limit = len(adjacency) if maximum_nodes is None else int(maximum_nodes)
    if isinstance(maximum_nodes, bool) or limit < len(anchors):
        raise HopPlanError("HOP traversal limit is invalid")
    order = list(anchors)
    distance = {anchor: 0 for anchor in anchors}
    seen = set(anchors)
    layer = list(anchors)
    truncated_tie_count = 0
    while layer and len(order) < limit:
        candidates = set()
        for node in layer:
            if touch_budget is not None:
                touch_budget.add(len(adjacency[node]))
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    candidates.add(neighbor)
        ordered = tuple(sorted(candidates, key=_typed_id_key))
        if not ordered:
            break
        remaining = limit - len(order)
        chosen = ordered[:remaining]
        if len(chosen) < len(ordered):
            truncated_tie_count = len(ordered) - len(chosen)
        next_distance = distance[layer[0]] + 1
        order.extend(chosen)
        for node in chosen:
            distance[node] = next_distance
        seen.update(chosen)
        layer = list(chosen)
        if truncated_tie_count:
            break
    return tuple(order), distance, truncated_tie_count


def _boundary_record(
    batch_id, role, requested_nodes, nodes, adjacency, *, touch_budget=None
):
    retained = set(nodes)
    cut_edges = []
    beta = {}
    boundary_nodes = []
    for node in sorted(retained, key=_typed_id_key):
        if touch_budget is not None:
            touch_budget.add(len(adjacency[node]))
        omitted = [
            neighbor for neighbor in adjacency[node] if neighbor not in retained
        ]
        if omitted:
            omitted.sort(key=_typed_id_key)
            boundary_nodes.append(node)
            beta[node] = len(omitted)
            cut_edges.extend([[node, neighbor] for neighbor in omitted])
    cut_edges.sort(key=lambda edge: (_typed_id_key(edge[0]), _typed_id_key(edge[1])))
    return {
        "batch_id": batch_id,
        "beta": [
            {"cut_conductance": beta[node], "node_id": node}
            for node in boundary_nodes
        ],
        "boundary_node_count": len(boundary_nodes),
        "closure_policy": "exact_dirichlet_no_closure",
        "cut_edge_count": len(cut_edges),
        "cut_edges": cut_edges,
        "cut_mass": len(cut_edges),
        "requested_nodes": requested_nodes,
        "role": role,
        "unit_conductance": 1,
    }


def _quartile_slices(size):
    base, remainder = divmod(size, QUARTILE_COUNT)
    result = []
    start = 0
    for index in range(QUARTILE_COUNT):
        width = base + (1 if index < remainder else 0)
        result.append((start, start + width))
        start += width
    if start != size:
        raise HopPlanError("quartile allocation failed")
    return tuple(result)


def _load_graph(manifest_data, adjacency_data, eligibility_data):
    manifest = _strict_json_bytes(manifest_data, "attempt manifest")
    if not isinstance(manifest, dict):
        raise HopPlanError("attempt manifest is malformed")
    adjacency_rows = _strict_jsonl_bytes(adjacency_data, "adjacency")
    eligibility_rows = _strict_jsonl_bytes(eligibility_data, "anchor eligibility")
    adjacency = {}
    for row in adjacency_rows:
        if not isinstance(row, dict) or set(row) != {"neighbors", "node_id"}:
            raise HopPlanError("adjacency row is malformed")
        node = row["node_id"]
        _typed_id_key(node)
        if node in adjacency or not isinstance(row["neighbors"], list):
            raise HopPlanError("adjacency row is malformed")
        neighbors = tuple(row["neighbors"])
        if len(neighbors) != len(set(neighbors)):
            raise HopPlanError("adjacency row contains duplicate neighbors")
        if node in neighbors:
            raise HopPlanError("adjacency contains a self edge")
        for neighbor in neighbors:
            _typed_id_key(neighbor)
        adjacency[node] = tuple(sorted(neighbors, key=_typed_id_key))
    for node, neighbors in adjacency.items():
        if any(neighbor not in adjacency or node not in adjacency[neighbor] for neighbor in neighbors):
            raise HopPlanError("adjacency is incomplete or asymmetric")
    eligible = []
    seen_eligibility = set()
    for row in eligibility_rows:
        if not isinstance(row, dict) or "node_id" not in row or "eligible" not in row:
            raise HopPlanError("anchor eligibility row is malformed")
        node = row["node_id"]
        _typed_id_key(node)
        if node in seen_eligibility:
            raise HopPlanError("anchor eligibility row is malformed")
        seen_eligibility.add(node)
        if node not in adjacency:
            if (
                set(row) != {"eligible", "node_id", "reason"}
                or row["eligible"] is not False
                or row["reason"] not in {"direct_private", "private_descendant"}
            ):
                raise HopPlanError("excluded-private eligibility row is malformed")
            continue
        if set(row) != {
            "component_id",
            "degree",
            "eligible",
            "node_id",
            "reason",
        }:
            raise HopPlanError("retained eligibility row is malformed")
        if row["eligible"] is True:
            if row.get("reason") != "eligible" or row.get("degree") != len(adjacency[node]):
                raise HopPlanError("eligible-anchor degree contract mismatch")
            eligible.append((node, len(adjacency[node])))
        elif row["eligible"] is not False:
            raise HopPlanError("anchor eligibility flag is malformed")
    if not set(adjacency).issubset(seen_eligibility):
        raise HopPlanError("anchor eligibility does not cover adjacency")
    eligible.sort(key=lambda item: (item[1], _typed_id_key(item[0])))
    return manifest, adjacency, tuple(eligible)


def _freeze_anchor_design(eligible):
    quartile_records = []
    selected_records = []
    selected_by_split = {"calibration": [[] for _ in range(4)], "audit": [[] for _ in range(4)]}
    block_reasons = []
    slices = _quartile_slices(len(eligible))
    for quartile_index, (start, stop) in enumerate(slices):
        quartile_id = f"q{quartile_index + 1}"
        members = eligible[start:stop]
        selection_order = sorted(
            members, key=lambda item: _selection_key("select", item[0])
        )
        quartile_records.append(
            {
                "degree_rank_members": [
                    {"degree": degree, "node_id": node}
                    for node, degree in members
                ],
                "degree_rank_start": start,
                "degree_rank_stop_exclusive": stop,
                "maximum_degree": members[-1][1] if members else None,
                "member_count": len(members),
                "minimum_degree": members[0][1] if members else None,
                "quartile_id": quartile_id,
                "remainder_allocation": "first-r-quartiles-receive-one-v1",
                "selection_order": [node for node, _degree in selection_order],
            }
        )
        if len(selection_order) < ANCHORS_PER_QUARTILE:
            block_reasons.append("quartile_coverage_inadequate")
            continue
        chosen = selection_order[:ANCHORS_PER_QUARTILE]
        split_order = sorted(
            chosen, key=lambda item: _selection_key("split", item[0])
        )
        split_parts = {
            "calibration": split_order[:CALIBRATION_PER_QUARTILE],
            "audit": split_order[CALIBRATION_PER_QUARTILE:],
        }
        for split, values in split_parts.items():
            batch_order = sorted(
                values, key=lambda item: _selection_key("batch", item[0])
            )
            selected_by_split[split][quartile_index] = [node for node, _ in batch_order]
            selection_rank = {node: rank for rank, (node, _degree) in enumerate(selection_order)}
            split_rank = {node: rank for rank, (node, _degree) in enumerate(split_order)}
            batch_rank = {node: rank for rank, (node, _degree) in enumerate(batch_order)}
            for node, degree in batch_order:
                selected_records.append(
                    {
                        "batch_key_sha256": _selection_key("batch", node)[0],
                        "batch_rank_within_split_quartile": batch_rank[node],
                        "degree": degree,
                        "node_id": node,
                        "quartile_id": quartile_id,
                        "selection_key_sha256": _selection_key("select", node)[0],
                        "selection_rank_within_quartile": selection_rank[node],
                        "split": split,
                        "split_key_sha256": _selection_key("split", node)[0],
                        "split_rank_within_quartile": split_rank[node],
                    }
                )
    if block_reasons:
        return quartile_records, [], [], sorted(set(block_reasons))

    selected_records.sort(
        key=lambda row: (
            0 if row["split"] == "calibration" else 1,
            row["batch_rank_within_split_quartile"],
            row["quartile_id"],
        )
    )
    selected_ids = [row["node_id"] for row in selected_records]
    if len(selected_ids) != 128 or len(set(selected_ids)) != 128:
        raise HopPlanError("selected-anchor assignment is not unique")

    batches = []
    for split, batch_count in (
        ("calibration", CALIBRATION_PER_QUARTILE),
        ("audit", AUDIT_PER_QUARTILE),
    ):
        for batch_index in range(batch_count):
            anchors = [
                selected_by_split[split][quartile_index][batch_index]
                for quartile_index in range(QUARTILE_COUNT)
            ]
            if len(set(anchors)) != ANCHORS_PER_BATCH:
                raise HopPlanError("balanced batch contains duplicate anchors")
            batches.append(
                {
                    "anchors_by_quartile": [
                        {"node_id": node, "quartile_id": f"q{index + 1}"}
                        for index, node in enumerate(anchors)
                    ],
                    "batch_id": f"{split}-{batch_index:02d}",
                    "batch_index": batch_index,
                    "split": split,
                }
            )
    return quartile_records, selected_records, batches, []


def _hop_order_fingerprint(order, distance):
    digest = hashlib.sha256()
    for node in order:
        digest.update(_canonical_json({"distance": distance[node], "node_id": node}))
    return digest.hexdigest()


def _freeze_anchor_traversals(selected_records, adjacency, touch_budget):
    selected_ids = {row["node_id"] for row in selected_records}
    records = []
    internal = {}
    block_reasons = []
    for selected in sorted(selected_records, key=lambda row: _typed_id_key(row["node_id"])):
        node = selected["node_id"]
        order, distance, truncated = _hop_order(
            (node,), adjacency, touch_budget=touch_budget
        )
        if truncated != 0:
            raise HopPlanError("full anchor traversal was unexpectedly truncated")
        protected = order[: PROTECTED_NONANCHORS + 1]
        if len(protected) != PROTECTED_NONANCHORS + 1:
            block_reasons.append("protected_coverage_inadequate")
        pair_distances = {
            other: distance[other]
            for other in selected_ids
            if other in distance
        }
        shell = tuple(
            candidate
            for candidate in order
            if distance[candidate] == CALIBRATION_SHELL_RADIUS
        )
        record = {
            "full_hop_order_sha256": _hop_order_fingerprint(order, distance),
            "node_id": node,
            "protected_nodes": [
                {"hop_distance": distance[candidate], "node_id": candidate}
                for candidate in protected
            ],
            "reachable_node_count": len(order),
            "split": selected["split"],
        }
        records.append(record)
        internal[node] = {
            "pair_distances": pair_distances,
            "protected": tuple(protected),
            "shell": shell,
        }
    return records, internal, sorted(set(block_reasons))


def _freeze_domains(batches, anchor_internal, adjacency, touch_budget):
    batch_records = []
    domain_records = []
    boundary_records = []
    shell_records = []
    block_reasons = []
    maximum_projected_bytes = 0
    for batch in batches:
        batch_id = batch["batch_id"]
        anchors = tuple(row["node_id"] for row in batch["anchors_by_quartile"])
        protected = tuple(
            sorted(
                {
                    node
                    for anchor in anchors
                    for node in anchor_internal[anchor]["protected"]
                },
                key=_typed_id_key,
            )
        )
        pair_rows = []
        for source in anchors:
            distances = anchor_internal[source]["pair_distances"]
            if any(target not in distances for target in anchors):
                raise HopPlanError("batch anchors are not mutually reachable")
            pair_rows.append(
                {
                    "distances_by_quartile": [distances[target] for target in anchors],
                    "source_node_id": source,
                }
            )
        full_order, full_distance, reference_truncated = _hop_order(
            anchors,
            adjacency,
            REFERENCE_BUDGET,
            touch_budget=touch_budget,
        )
        if tuple(full_order[: len(anchors)]) != tuple(sorted(anchors, key=_typed_id_key)):
            raise HopPlanError("multi-source HOP order does not begin with its anchors")
        batch_record = dict(batch)
        batch_record.update(
            {
                "protected_node_count": len(protected),
                "protected_nodes": list(protected),
                "source_to_source_hop_distances": pair_rows,
            }
        )
        batch_records.append(batch_record)

        boundary_by_role = {}
        roles = [(f"S_{budget}", budget) for budget in CANDIDATE_BUDGETS]
        roles.append(("R_top", REFERENCE_BUDGET))
        previous_nodes = set()
        for role, requested in roles:
            realized = min(requested, len(full_order))
            nodes = full_order[:realized]
            node_set = set(nodes)
            if previous_nodes and not previous_nodes < node_set:
                # A component with fewer than the requested nodes is exact, but then
                # later nominal budgets may be equal.  Equality is recorded and is
                # not misrepresented as strict finite-budget nesting.
                if previous_nodes != node_set:
                    raise HopPlanError("HOP prefixes are not nested")
            previous_nodes = node_set
            coverage = set(protected).issubset(node_set)
            if not coverage:
                block_reasons.append("protected_coverage_inadequate")
            cutoff_distance = full_distance[nodes[-1]]
            trailing_same_shell = sum(
                1
                for candidate in full_order[realized:]
                if full_distance[candidate] == cutoff_distance
            )
            if (
                reference_truncated
                and cutoff_distance == full_distance[full_order[-1]]
            ):
                trailing_same_shell += reference_truncated
            node_rows = [
                {"hop_distance": full_distance[node], "node_id": node}
                for node in nodes
            ]
            projected = DENSE_ARRAY_COUNT * FLOAT64_BYTES * realized * realized
            maximum_projected_bytes = max(maximum_projected_bytes, projected)
            domain_records.append(
                {
                    "batch_id": batch_id,
                    "component_exhausted": False,
                    "cutoff_distance": cutoff_distance,
                    "dense_preworkspace_projection_bytes": projected,
                    "hop_order_sha256": hashlib.sha256(
                        _jsonl_bytes(node_rows)
                    ).hexdigest(),
                    "nodes": node_rows,
                    "protected_coverage": coverage,
                    "realized_nodes": realized,
                    "requested_nodes": requested,
                    "role": role,
                    "truncated_final_shell_nodes": trailing_same_shell,
                }
            )
            boundary = _boundary_record(
                batch_id,
                role,
                requested,
                nodes,
                adjacency,
                touch_budget=touch_budget,
            )
            boundary_by_role[role] = boundary
            boundary_records.append(boundary)
            domain_records[-1]["component_exhausted"] = boundary["cut_edge_count"] == 0

        if batch["split"] == "calibration":
            reference_nodes = {
                row["node_id"]
                for row in domain_records[-1]["nodes"]
            }
            reference_beta = {
                row["node_id"]: row["cut_conductance"]
                for row in boundary_by_role["R_top"]["beta"]
            }
            for anchor in anchors:
                shell = anchor_internal[anchor]["shell"]
                reasons = []
                if not shell:
                    reasons.append("empty_radius_3_shell")
                if any(node not in reference_nodes for node in shell):
                    reasons.append("shell_outside_reference")
                if any(reference_beta.get(node, 0) != 0 for node in shell):
                    reasons.append("shell_not_strictly_interior")
                if reasons:
                    block_reasons.append("calibration_shell_inadequate")
                shell_records.append(
                    {
                        "anchor_node_id": anchor,
                        "batch_id": batch_id,
                        "hop_radius": CALIBRATION_SHELL_RADIUS,
                        "reasons": sorted(set(reasons)),
                        "shell_nodes": list(shell),
                        "strictly_interior_pass": not reasons,
                        "target_attenuation": CALIBRATION_TARGET,
                    }
                )
    return (
        batch_records,
        domain_records,
        boundary_records,
        shell_records,
        maximum_projected_bytes,
        sorted(set(block_reasons)),
    )


def _positive_integer(name, value):
    if isinstance(value, bool):
        raise HopPlanError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HopPlanError(f"{name} must be a positive integer") from exc
    if parsed != value or parsed < 1:
        raise HopPlanError(f"{name} must be a positive integer")
    return parsed


def _validated_config(config):
    if not isinstance(config, dict) or set(config) != {
        "effective_resistance_arm",
        "planner_edge_touch_ceiling",
        "planner_input_ceiling_bytes",
        "study_peak_rss_ceiling_bytes",
    }:
        raise HopPlanError("planning configuration fields mismatch")
    resistance = config["effective_resistance_arm"]
    if resistance not in {"enabled", "omitted"}:
        raise HopPlanError("effective-resistance arm must be enabled or omitted")
    return {
        "effective_resistance_arm": resistance,
        "planner_edge_touch_ceiling": _positive_integer(
            "planner_edge_touch_ceiling", config["planner_edge_touch_ceiling"]
        ),
        "planner_input_ceiling_bytes": _positive_integer(
            "planner_input_ceiling_bytes", config["planner_input_ceiling_bytes"]
        ),
        "study_peak_rss_ceiling_bytes": _positive_integer(
            "study_peak_rss_ceiling_bytes", config["study_peak_rss_ceiling_bytes"]
        ),
    }


def _capture_verified_inputs(
    receipt,
    receipt_binding,
    attempt_a_binding,
    attempt_b_binding,
    source_spec,
    relation_policy,
    *,
    planner_input_ceiling_bytes,
):
    receipt_expected = _content_record(_canonical_json(receipt))
    receipt_data = _read_bound_file(
        receipt_binding,
        consensus.RECEIPT_NAME,
        receipt_expected,
        "consensus receipt",
    )
    if _strict_json_bytes(receipt_data, "consensus receipt") != receipt:
        raise HopPlanError("consensus receipt changed after verification")
    receipt_record = _content_record(receipt_data)
    expected_a = receipt["attempts"][0]["manifest_record"]
    expected_b = receipt["attempts"][1]["manifest_record"]
    manifest_a_data = _read_bound_file(
        attempt_a_binding, "manifest.json", expected_a, "canonical attempt A"
    )
    manifest_b_data = _read_bound_file(
        attempt_b_binding, "manifest.json", expected_b, "attempt B"
    )
    manifest_a = _strict_json_bytes(manifest_a_data, "attempt A manifest")
    _strict_json_bytes(manifest_b_data, "attempt B manifest")
    try:
        artifact_records = manifest_a["artifact_records"]
        adjacency_record = artifact_records["adjacency.jsonl"]
        eligibility_record = artifact_records["anchor_eligibility.jsonl"]
        planning_artifact_records = {
            name: artifact_records[name]
            for name in (
                "adjacency.jsonl",
                "anchor_eligibility.jsonl",
                "components.jsonl",
                "physical_edges.tsv",
            )
        }
    except (KeyError, TypeError) as exc:
        raise HopPlanError("attempt A manifest lacks planning artifacts") from exc
    if any(not _content_record_is_valid(value) for value in planning_artifact_records.values()):
        raise HopPlanError("planning artifact content record is malformed")
    source_record = receipt["input_records"]["source_spec"]
    policy_record = receipt["input_records"]["relation_policy"]
    planned_input_bytes = (
        len(receipt_data)
        + len(manifest_a_data)
        + len(manifest_b_data)
        + adjacency_record["size_bytes"]
        + eligibility_record["size_bytes"]
        + source_record["size_bytes"]
        + policy_record["size_bytes"]
    )
    if planned_input_bytes > planner_input_ceiling_bytes:
        raise HopPlanError("planner input ceiling exceeded")
    adjacency_data = _read_bound_file(
        attempt_a_binding,
        "adjacency.jsonl",
        adjacency_record,
        "canonical attempt A",
    )
    eligibility_data = _read_bound_file(
        attempt_a_binding,
        "anchor_eligibility.jsonl",
        eligibility_record,
        "canonical attempt A",
    )
    _read_regular_path(source_spec, source_record, "source specification")
    _read_regular_path(relation_policy, policy_record, "relation policy")
    return {
        "adjacency_data": adjacency_data,
        "eligibility_data": eligibility_data,
        "input_bindings": {
            "canonical_attempt_a_manifest": expected_a,
            "consensus_receipt": receipt_record,
            "planning_graph_artifacts": planning_artifact_records,
            "relation_policy": policy_record,
            "source_spec": source_record,
        },
        "manifest_a_data": manifest_a_data,
        "planned_input_bytes": planned_input_bytes,
    }


def _numeric_backend_contract():
    try:
        numpy_version = metadata.version("numpy")
    except metadata.PackageNotFoundError as exc:
        raise HopPlanError("frozen NumPy backend is unavailable") from exc
    return {
        "actual_blas_identity": (
            "required_nonempty_in_calibration_lock_without_library_paths"
        ),
        "actual_blas_identity_absolute_paths_prohibited": True,
        "actual_blas_identity_nonempty_lock_requirement": True,
        "alpha_calibration_backend": "numpy.linalg.eigh",
        "blas_threads_observed_lock_requirement": 1,
        "blas_threads_requested": 1,
        "cholesky_reconstruction_atol_hex": float(1e-12).hex(),
        "cholesky_reconstruction_rtol_hex": float(1e-11).hex(),
        "condition_estimation_backend": "numpy.linalg.eigvalsh",
        "decision_dtype": "float64",
        "decision_factorization_backend": "numpy.linalg.cholesky",
        "device_class": "cpu",
        "hidden_jitter": False,
        "m_matrix_off_diagonal_tolerance_hex": float(1e-12).hex(),
        "maximum_principle_relative_tolerance_hex": float(1e-10).hex(),
        "minimum_reciprocal_condition_hex": MINIMUM_RECIPROCAL_CONDITION_HEX,
        "numpy_version": numpy_version,
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "solve_residual_relative_tolerance_hex": float(1e-10).hex(),
        "symmetry_absolute_tolerance_hex": float(1e-12).hex(),
        "triangular_solve_backend": "numpy.linalg.solve",
    }


def _statistical_contract(effective_resistance_arm):
    absolute_adequacy = {
        "boundary_harmonic_q90_max": 0.10,
        "maximum_h_absolute_error_q90_max": 0.025,
        "rank_inversion_fraction_q90_max": 0.05,
        "raw_relative_l2_error_q90_max": 0.05,
        "top8_overlap_q10_min": 0.90,
    }
    noninferiority_margins = {
        "boundary_harmonic_absolute_harm": 0.01,
        "maximum_h_absolute_error_harm": 0.01,
        "primary_log_error_ratio": "log(1.10)",
        "rank_inversion_absolute_harm": 0.01,
        "source_diagonal_relative_error_harm": 0.01,
        "top8_overlap_loss": 0.01,
    }
    active_noninferiority_endpoints = [
        "boundary_harmonic_absolute_harm",
        "maximum_h_absolute_error_harm",
        "primary_log_error_ratio",
        "rank_inversion_absolute_harm",
        "source_diagonal_relative_error_harm",
        "top8_overlap_loss",
    ]
    omitted_endpoints = []
    if effective_resistance_arm == "enabled":
        absolute_adequacy["effective_resistance_relative_error_q90_max"] = 0.05
        noninferiority_margins["effective_resistance_relative_error_harm"] = 0.01
        active_noninferiority_endpoints.append(
            "effective_resistance_relative_error_harm"
        )
        effective_resistance_endpoint = "active"
    else:
        omitted_endpoints.append("effective_resistance_relative_error")
        effective_resistance_endpoint = "predeclared_omitted"
    return {
        "absolute_adequacy": absolute_adequacy,
        "active_noninferiority_endpoints": sorted(active_noninferiority_endpoints),
        "allowed_calibration_lock_modes": [
            "absolute_only",
            "blocked",
            "finite_contrast",
            "right_censored_diagnostics",
        ],
        "audit_authorization_by_lock_mode": {
            "absolute_only": True,
            "blocked": False,
            "finite_contrast": True,
            "right_censored_diagnostics": True,
        },
        "bootstrap": {
            "audit_batch_resamples": 9999,
            "draws_per_resample": AUDIT_PER_QUARTILE,
            "endpoint_rule": {
                "lower_0.05": "lower-observed-order-statistic",
                "upper_0.95": "higher-observed-order-statistic",
            },
            "identical_multiplicities_across_endpoints": True,
            "multiplicity_schedule_artifact": "bootstrap_multiplicities.jsonl",
            "sampler": (
                "sha256-rejection-canonical-json-line-"
                "[3882002,paired_audit_batch_bootstrap,replicate,draw,nonce]-mod-24-v1"
            ),
            "seed": BOOTSTRAP_SEED,
            "unit": "balanced_four-anchor_batch",
        },
        "calibration_lock_mode_rules": {
            "absolute_only": (
                "reference-adequate and K_low=1024 with R_top high endpoint; "
                "no efficacy or resource-contrast claim"
            ),
            "blocked": (
                "reference-inadequate or calibration/numerical contract failure; "
                "no audit solve"
            ),
            "finite_contrast": (
                "reference-adequate, smallest adequate K_low in {256,512}, and "
                "next larger finite endpoint has a distinct node-content hash"
            ),
            "right_censored_diagnostics": (
                "reference-adequate but no candidate budget adequate; audit is "
                "diagnostic only and cannot make convergence, efficacy, or resource claims"
            ),
        },
        "calibration_selection": {
            "candidate_order": list(CANDIDATE_BUDGETS),
            "if_k_low_1024": (
                "lock_mode=absolute_only; K_high=R_top; report absolute adequacy only"
            ),
            "k_high_rule": "next-larger-candidate-after-smallest-adequate-k-low",
            "k_low_rule": "smallest-calibration-adequate-candidate",
            "no_adequate_candidate": "lock_mode=right_censored_diagnostics",
            "node_identical_exhausted_endpoints": (
                "lock_mode=blocked in phase one; a gauge-aware extension requires "
                "a prospective amendment"
            ),
            "reference_inadequate": "lock_mode=blocked; audit_solve_authorized=false",
        },
        "effective_resistance_endpoint": effective_resistance_endpoint,
        "efficacy_log_ratio_threshold": "log(0.9)",
        "estimand": "equal-degree-quartile-macro-mean",
        "extended_real_zero_handling": True,
        "minimum_complete_audit_batches": 18,
        "noninferiority_intersection_margins": noninferiority_margins,
        "omitted_endpoints": omitted_endpoints,
        "reference_adequacy": {
            "maximum_h_absolute_error_q90_max": 0.005,
            "raw_relative_l2_error_q90_max": 0.01,
            "top8_overlap_q10_min": 0.98,
        },
        "tail_rule": "observed-order-statistic-conservative-no-interpolation",
    }


@lru_cache(maxsize=1)
def _bootstrap_multiplicity_records():
    """Freeze the paired 24-batch bootstrap without a library PRNG."""

    modulus = AUDIT_PER_QUARTILE
    rejection_limit = (1 << 256) - ((1 << 256) % modulus)
    records = []
    for replicate in range(9999):
        counts = [0] * modulus
        for draw in range(modulus):
            nonce = 0
            while True:
                digest = hashlib.sha256(
                    _canonical_json(
                        [
                            BOOTSTRAP_SEED,
                            "paired_audit_batch_bootstrap",
                            replicate,
                            draw,
                            nonce,
                        ]
                    )
                ).digest()
                value = int.from_bytes(digest, "big")
                if value < rejection_limit:
                    counts[value % modulus] += 1
                    break
                nonce += 1
        records.append(
            {"multiplicities": counts, "replicate_index": replicate}
        )
    return tuple(records)


def _operator_contract():
    return {
        "boundary": "exact_dirichlet",
        "closure": "disabled",
        "conductance": "unit_on_policy_admitted_reciprocal_edges",
        "embeddings": "prohibited",
        "operator": "topology_only_hop",
        "resistance_selector": "prohibited",
        "skeleton_selector": "prohibited",
    }


def _selection_contract():
    return {
        "anchor_selection_seed": SELECTION_SEED,
        "anchors_per_batch": ANCHORS_PER_BATCH,
        "anchors_per_quartile": ANCHORS_PER_QUARTILE,
        "audit_per_quartile": AUDIT_PER_QUARTILE,
        "batch_key_purpose": "batch",
        "calibration_per_quartile": CALIBRATION_PER_QUARTILE,
        "hash_key_encoding": HASH_KEY_ENCODING,
        "protected_nonanchors_per_anchor": PROTECTED_NONANCHORS,
        "quartile_count": QUARTILE_COUNT,
        "quartile_remainder_allocation": "first-r-quartiles-receive-one-v1",
        "selection_key_purpose": "select",
        "split_key_purpose": "split",
        "typed_id_order": TYPED_ID_ORDER,
    }


def _derive_plan(capture, receipt, config):
    config = _validated_config(config)
    manifest_a, adjacency, eligible = _load_graph(
        capture["manifest_a_data"],
        capture["adjacency_data"],
        capture["eligibility_data"],
    )
    quartiles, selected, batches, block_reasons = _freeze_anchor_design(eligible)
    touch_budget = _TouchBudget(config["planner_edge_touch_ceiling"])
    anchor_traversals = []
    batch_records = []
    domain_records = []
    boundary_records = []
    shell_records = []
    maximum_projected_bytes = 0
    if not block_reasons:
        anchor_traversals, anchor_internal, traversal_blocks = _freeze_anchor_traversals(
            selected, adjacency, touch_budget
        )
        block_reasons.extend(traversal_blocks)
        if not traversal_blocks:
            (
                batch_records,
                domain_records,
                boundary_records,
                shell_records,
                maximum_projected_bytes,
                domain_blocks,
            ) = _freeze_domains(batches, anchor_internal, adjacency, touch_budget)
            block_reasons.extend(domain_blocks)
    if maximum_projected_bytes > config["study_peak_rss_ceiling_bytes"]:
        block_reasons.append("study_resource_inadequate")
    block_reasons = sorted(set(block_reasons))
    priority = (
        "quartile_coverage_inadequate",
        "protected_coverage_inadequate",
        "calibration_shell_inadequate",
        "study_resource_inadequate",
    )
    reason = "hop_plan_frozen"
    if block_reasons:
        reason = next(item for item in priority if item in block_reasons)
    accepted = not block_reasons

    payloads = {
        "quartiles.jsonl": _jsonl_bytes(quartiles),
        "selected_anchors.jsonl": _jsonl_bytes(selected),
        "batches.jsonl": _jsonl_bytes(batch_records),
        "anchor_traversals.jsonl": _jsonl_bytes(anchor_traversals),
        "domains.jsonl": _jsonl_bytes(domain_records),
        "boundaries.jsonl": _jsonl_bytes(boundary_records),
        "calibration_shells.jsonl": _jsonl_bytes(shell_records),
        "bootstrap_multiplicities.jsonl": _jsonl_bytes(
            _bootstrap_multiplicity_records()
        ),
    }
    artifact_records = {name: _content_record(data) for name, data in payloads.items()}
    numeric_contract = _numeric_backend_contract()
    implementation_records = {
        "design": _file_record(DESIGN_PATH),
        "planner": _file_record(Path(__file__).resolve()),
        "protocol": _file_record(PROTOCOL_PATH),
    }
    resource_contract = {
        "automatic_resource_downgrade": False,
        "cache_key_contract": "plan_fingerprint+calibration_lock+batch_id+role+backend",
        "cache_policy": "content_addressed_local_only_no_cross_plan_reuse",
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "dense_array_count_before_workspace": DENSE_ARRAY_COUNT,
        "dense_projection_formula": "4*8*n*n",
        "effective_resistance_arm": config["effective_resistance_arm"],
        "float64_bytes": FLOAT64_BYTES,
        "maximum_projected_dense_preworkspace_bytes": maximum_projected_bytes,
        "maximum_reference_nodes": REFERENCE_BUDGET,
        "planner_edge_touch_ceiling": config["planner_edge_touch_ceiling"],
        "planner_edge_touches_observed": touch_budget.observed,
        "planner_input_ceiling_bytes": config["planner_input_ceiling_bytes"],
        "planner_input_bytes_observed": capture["planned_input_bytes"],
        "study_peak_rss_ceiling_bytes": config["study_peak_rss_ceiling_bytes"],
        "per_batch_elapsed_ceiling_seconds": 3600,
    }
    calibration_contract = {
        "alpha_zero_evaluation_required": True,
        "alpha_zero_numerical_admissibility_required": True,
        "alpha_status": "unfrozen",
        "base_intrinsic_leakage_conductance_hex": float(0.0).hex(),
        "bath_temperature_hex": float(0.0).hex(),
        "bisection_relative_tolerance_hex": float(1e-8).hex(),
        "bracket_seed_hop_radius": CALIBRATION_SHELL_RADIUS,
        "calibration_anchor_count": QUARTILE_COUNT * CALIBRATION_PER_QUARTILE,
        "calibration_batch_count": CALIBRATION_PER_QUARTILE,
        "calibration_split_only": True,
        "finite_result_required": True,
        "global_alpha_rule": (
            "maximum-of-eight-four-anchor-batch-maxima-equivalently-all-32-anchors"
        ),
        "hidden_floor_or_jitter": False,
        "hidden_maximum_alpha_cap": False,
        "lock_verification_contract": (
            "full-chain-and-content-record-verification-without-numerical-"
            "recomputation-or-authentication"
        ),
        "maximum_function_evaluations_per_anchor": 80,
        "maximum_leakage_conductance": None,
        "nonfinite_unbracketed_or_evaluation_exhaustion": "lock_mode=blocked",
        "per_batch_alpha_rule": "maximum-of-four-anchor-required-added-leakages",
        "radius": CALIBRATION_SHELL_RADIUS,
        "required_numerical_minimum_added_leakage_hex": float(0.0).hex(),
        "target_attenuation": CALIBRATION_TARGET,
        "zero_alpha_allowed": True,
    }
    reference_contract = {
        "candidate_union": "S_1024",
        "maximum_nodes": REFERENCE_BUDGET,
        "reference_rule": REFERENCE_RULE,
        "whole_component_when_exhausted": True,
    }
    receipt_common = receipt["common_records"]
    fingerprint_core = {
        "algorithm": ALGORITHM,
        "artifact_records": artifact_records,
        "calibration_contract": calibration_contract,
        "implementation_records": implementation_records,
        "input_bindings": capture["input_bindings"],
        "numeric_backend_contract": numeric_contract,
        "operator_contract": _operator_contract(),
        "reference_contract": reference_contract,
        "repository_commit": _git_commit(),
        "repository_commit_policy": (
            "actual-plan-generated-only-at-final-calibration-lock-implementation-commit"
        ),
        "resource_contract": resource_contract,
        "schema": SCHEMA,
        "selection_contract": _selection_contract(),
        "snapshot_common_records": receipt_common,
        "statistical_contract": _statistical_contract(
            config["effective_resistance_arm"]
        ),
    }
    plan_fingerprint = hashlib.sha256(_canonical_json(fingerprint_core)).hexdigest()
    manifest = {
        "accepted": accepted,
        "aggregate": {
            "audit_batches": sum(row["split"] == "audit" for row in batch_records),
            "calibration_batches": sum(
                row["split"] == "calibration" for row in batch_records
            ),
            "eligible_anchor_count": len(eligible),
            "publishable": False,
            "selected_anchor_count": len(selected),
        },
        "algorithm": ALGORITHM,
        "audit_solve_authorized": False,
        "block_reasons": block_reasons,
        "calibration_solve_authorized": accepted,
        "diffusion_or_fidelity_metrics_computed": False,
        "fingerprint_core": fingerprint_core,
        "no_solve": True,
        "plan_fingerprint": plan_fingerprint,
        "reason": reason,
        "schema": SCHEMA,
        "solves_executed": 0,
        "structural_metrics_computed": True,
    }
    manifest["manifest_integrity_sha256"] = hashlib.sha256(
        _canonical_json(manifest)
    ).hexdigest()
    return payloads, manifest


def _declared_source_inputs(
    source_spec, *, maximum_size=MAXIMUM_SOURCE_SPEC_BYTES
):
    maximum_size = min(
        MAXIMUM_SOURCE_SPEC_BYTES,
        _positive_integer("source_specification_read_ceiling", maximum_size),
    )
    source_input = Path(source_spec)
    if _has_symlink_ancestor(source_input):
        raise HopPlanError("source declaration bundle cannot contain a symlink")
    declaration_dir = source_input.resolve().parent
    declaration_binding = _bind_directory(
        declaration_dir, "source declaration bundle"
    )
    try:
        if stat.S_IMODE(os.fstat(declaration_binding.fd).st_mode) != 0o700:
            raise HopPlanError("source declaration bundle mode mismatch")
        names = set(
            _bounded_directory_names(
                declaration_binding, 2, "source declaration bundle"
            )
        )
        if names != {declaration.SPEC_FILENAME, declaration.LOCAL_ONLY_MARKER}:
            raise HopPlanError("source declaration bundle inventory mismatch")
        _assert_private_directory_files(
            declaration_binding, "source declaration bundle", maximum_count=2
        )
        source_data = _read_bound_file(
            declaration_binding,
            declaration.SPEC_FILENAME,
            None,
            "source declaration bundle",
            maximum_size=maximum_size,
        )
        marker_data = _read_bound_file(
            declaration_binding,
            declaration.LOCAL_ONLY_MARKER,
            _content_record(declaration.LOCAL_ONLY_MARKER_PAYLOAD),
            "source declaration bundle",
        )
        if marker_data != declaration.LOCAL_ONLY_MARKER_PAYLOAD:
            raise HopPlanError("source declaration marker mismatch")
        try:
            verified = declaration._validate_canonical_spec(
                declaration._strict_canonical_json(source_data)
            )
        except declaration.DeclarationError as exc:
            raise HopPlanError("source declaration verification failed") from exc
    finally:
        declaration_binding.close()
    paths = {declaration_dir}
    for entry in verified["sources"]:
        paths.add(Path(entry["path"]).resolve())
    legacy = verified.get("legacy_check")
    if isinstance(legacy, dict):
        paths.add(Path(legacy["dag_path"]).resolve())
    return tuple(sorted(paths, key=os.fspath))


def _assert_no_output_input_overlap(
    target,
    input_dirs,
    input_files,
    source_spec,
    *,
    source_spec_maximum_size=MAXIMUM_SOURCE_SPEC_BYTES,
):
    inputs = {
        *(Path(value).resolve() for value in input_dirs),
        *(Path(value).resolve() for value in input_files),
        *_declared_source_inputs(
            source_spec, maximum_size=source_spec_maximum_size
        ),
    }
    for item in inputs:
        if (
            item == target
            or _path_is_within(item, target)
            or _path_is_within(target, item)
        ):
            raise HopPlanError("plan output overlaps a verified input")


def _validate_output_paths(
    plan_dir,
    local_root,
    input_dirs,
    input_files,
    source_spec,
    source_spec_maximum_size,
):
    raw_plan = Path(plan_dir)
    raw_root = Path(local_root)
    if _has_symlink_ancestor(raw_plan) or _has_symlink_ancestor(raw_root):
        raise HopPlanError("local-only output paths cannot contain symlinks")
    root = raw_root.resolve()
    target = raw_plan.resolve()
    if not root.is_dir():
        raise HopPlanError("local root must be an existing directory")
    if target == root or not _path_is_within(target, root):
        raise HopPlanError("plan directory must be a child of the local root")
    if _path_is_within(target, REPO_ROOT.resolve()) or _has_git_ancestor(target.parent):
        raise HopPlanError("plan directory cannot be inside a Git worktree")
    if target.exists() or not target.parent.is_dir():
        raise HopPlanError("plan directory must be fresh with an existing parent")
    _assert_no_output_input_overlap(
        target,
        input_dirs,
        input_files,
        source_spec,
        source_spec_maximum_size=source_spec_maximum_size,
    )
    parent_binding = _bind_directory(target.parent, "plan output parent")
    return target, parent_binding


def _write_bound_bytes(binding, name, data):
    if (
        not isinstance(name, str)
        or name in {"", ".", ".."}
        or Path(name).name != name
    ):
        raise HopPlanError("local-only plan artifact name is malformed")
    _assert_bound_directory(binding, "HOP plan staging directory")
    try:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=binding.fd,
        )
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
        observed = os.fstat(fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or stat.S_IMODE(observed.st_mode) != 0o600
            or observed.st_size != len(data)
        ):
            raise HopPlanError("local-only plan artifact contract mismatch")
    except OSError as exc:
        raise HopPlanError("local-only plan artifact could not be written") from exc
    finally:
        try:
            os.close(fd)
        except (OSError, UnboundLocalError):
            pass
    _assert_bound_directory(binding, "HOP plan staging directory")


def _rename_directory_noreplace(parent_binding, source_leaf, target_leaf):
    for leaf in (source_leaf, target_leaf):
        if (
            not isinstance(leaf, str)
            or leaf in {"", ".", ".."}
            or Path(leaf).name != leaf
        ):
            raise HopPlanError("atomic installation leaf is malformed")
    _assert_bound_directory(parent_binding, "plan output parent")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise HopPlanError("atomic no-replace installation is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_binding.fd,
        os.fsencode(source_leaf),
        parent_binding.fd,
        os.fsencode(target_leaf),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise HopPlanError("plan directory appeared during atomic installation")
    raise HopPlanError("atomic no-replace installation failed") from OSError(
        error_number, os.strerror(error_number)
    )


def _cleanup_bound_staging(parent_binding, staging_binding, staging_leaf):
    """Remove only our known, still-bound staging leaves; otherwise leak safely."""

    _assert_bound_directory(parent_binding, "plan output parent")
    _assert_bound_directory(staging_binding, "HOP plan staging directory")
    entries = set(
        _bounded_directory_names(
            staging_binding, len(ALL_PLAN_FILES), "HOP plan staging directory"
        )
    )
    if not entries.issubset(ALL_PLAN_FILES):
        raise HopPlanError("HOP plan staging inventory changed during cleanup")
    for name in sorted(entries):
        try:
            observed = os.stat(name, dir_fd=staging_binding.fd, follow_symlinks=False)
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                raise HopPlanError("HOP plan staging artifact changed during cleanup")
            os.unlink(name, dir_fd=staging_binding.fd)
        except OSError as exc:
            raise HopPlanError("HOP plan staging cleanup failed") from exc
    os.fsync(staging_binding.fd)
    _assert_bound_directory(staging_binding, "HOP plan staging directory")
    _assert_bound_directory(parent_binding, "plan output parent")
    try:
        os.rmdir(staging_leaf, dir_fd=parent_binding.fd)
        os.fsync(parent_binding.fd)
    except OSError as exc:
        raise HopPlanError("HOP plan staging cleanup failed") from exc
    _assert_bound_directory(parent_binding, "plan output parent")


def _plan_manifest_from_binding(plan_binding):
    _assert_bound_directory(plan_binding, "HOP plan")
    try:
        observed = os.fstat(plan_binding.fd)
    except OSError as exc:
        raise HopPlanError("HOP plan directory is unavailable") from exc
    entries = set(
        _bounded_directory_names(
            plan_binding, len(ALL_PLAN_FILES), "HOP plan"
        )
    )
    if stat.S_IMODE(observed.st_mode) != 0o700 or entries != ALL_PLAN_FILES:
        raise HopPlanError("HOP plan directory contract mismatch")
    marker = _read_bound_file(
        plan_binding,
        MARKER_NAME,
        _content_record(MARKER_BYTES),
        "HOP plan",
    )
    if marker != MARKER_BYTES:
        raise HopPlanError("HOP plan local-only marker mismatch")
    manifest_data = _read_bound_file(
        plan_binding,
        MANIFEST_NAME,
        None,
        "HOP plan",
        maximum_size=MAXIMUM_PLAN_MANIFEST_BYTES,
    )
    manifest = _strict_json_bytes(manifest_data, "HOP plan manifest")
    expected_keys = {
        "accepted",
        "aggregate",
        "algorithm",
        "audit_solve_authorized",
        "block_reasons",
        "calibration_solve_authorized",
        "diffusion_or_fidelity_metrics_computed",
        "fingerprint_core",
        "manifest_integrity_sha256",
        "no_solve",
        "plan_fingerprint",
        "reason",
        "schema",
        "solves_executed",
        "structural_metrics_computed",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise HopPlanError("HOP plan manifest fields mismatch")
    if (
        manifest["schema"] != SCHEMA
        or manifest["algorithm"] != ALGORITHM
        or manifest["no_solve"] is not True
        or manifest["solves_executed"] != 0
        or manifest["structural_metrics_computed"] is not True
        or manifest["diffusion_or_fidelity_metrics_computed"] is not False
        or manifest["audit_solve_authorized"] is not False
        or manifest["calibration_solve_authorized"] is not manifest["accepted"]
    ):
        raise HopPlanError("HOP plan interpretation contract mismatch")
    unsealed = dict(manifest)
    integrity = unsealed.pop("manifest_integrity_sha256")
    if (
        not isinstance(integrity, str)
        or re.fullmatch(r"[0-9a-f]{64}", integrity) is None
        or hashlib.sha256(_canonical_json(unsealed)).hexdigest() != integrity
    ):
        raise HopPlanError("HOP plan manifest integrity mismatch")
    core = manifest["fingerprint_core"]
    if not isinstance(core, dict) or hashlib.sha256(_canonical_json(core)).hexdigest() != manifest["plan_fingerprint"]:
        raise HopPlanError("HOP plan fingerprint mismatch")
    return manifest


def _config_from_manifest(manifest):
    try:
        resource = manifest["fingerprint_core"]["resource_contract"]
        return _validated_config(
            {
                "effective_resistance_arm": resource["effective_resistance_arm"],
                "planner_edge_touch_ceiling": resource["planner_edge_touch_ceiling"],
                "planner_input_ceiling_bytes": resource["planner_input_ceiling_bytes"],
                "study_peak_rss_ceiling_bytes": resource["study_peak_rss_ceiling_bytes"],
            }
        )
    except (KeyError, TypeError) as exc:
        raise HopPlanError("HOP plan resource contract is malformed") from exc


def _verify_or_prepare_inputs(
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
    config,
):
    receipt_binding = _bind_directory(receipt_dir, "consensus receipt")
    attempt_a_binding = _bind_directory(attempt_a_dir, "canonical attempt A")
    attempt_b_binding = _bind_directory(attempt_b_dir, "attempt B")
    bindings = (receipt_binding, attempt_a_binding, attempt_b_binding)
    try:
        if len({(item.device, item.inode) for item in bindings}) != 3:
            raise HopPlanError("consensus directories must be distinct")
        for binding, label in zip(
            bindings, ("consensus receipt", "canonical attempt A", "attempt B")
        ):
            _assert_private_directory_files(binding, label)
        receipt_preflight_data = _read_bound_file(
            receipt_binding,
            consensus.RECEIPT_NAME,
            None,
            "consensus receipt",
            maximum_size=min(
                MAXIMUM_CONSENSUS_RECEIPT_BYTES,
                config["planner_input_ceiling_bytes"],
            ),
        )
        receipt_preflight = _strict_json_bytes(
            receipt_preflight_data, "consensus receipt"
        )
        try:
            source_record = receipt_preflight["input_records"]["source_spec"]
            policy_record = receipt_preflight["input_records"]["relation_policy"]
        except (KeyError, TypeError) as exc:
            raise HopPlanError("consensus receipt lacks input records") from exc
        if not _content_record_is_valid(source_record) or not _content_record_is_valid(
            policy_record
        ):
            raise HopPlanError("consensus receipt input record is malformed")
        manifest_preflight_bytes = 0
        for binding, label in (
            (attempt_a_binding, "canonical attempt A"),
            (attempt_b_binding, "attempt B"),
        ):
            manifest_preflight_bytes += len(
                _read_bound_file(
                    binding,
                    "manifest.json",
                    None,
                    label,
                    maximum_size=min(
                        MAXIMUM_ATTEMPT_MANIFEST_BYTES,
                        config["planner_input_ceiling_bytes"],
                    ),
                )
            )
        preliminary_bytes = (
            len(receipt_preflight_data)
            + manifest_preflight_bytes
            + source_record["size_bytes"]
            + policy_record["size_bytes"]
        )
        if preliminary_bytes > config["planner_input_ceiling_bytes"]:
            raise HopPlanError("planner input ceiling exceeded")
        _read_regular_path(source_spec, source_record, "source specification")
        _read_regular_path(relation_policy, policy_record, "relation policy")
        try:
            receipt = consensus.verify_consensus_receipt(
                receipt_dir,
                attempt_a_dir=attempt_a_dir,
                attempt_b_dir=attempt_b_dir,
                source_spec=source_spec,
                relation_policy=relation_policy,
            )
        except consensus.ConsensusError as exc:
            raise HopPlanError("full snapshot consensus verification failed") from exc
        if (
            receipt.get("accepted") is not True
            or receipt.get("graph_gate_pass") is not True
            or receipt.get("repeatability_verified") is not True
            or receipt.get("reason") != "exact_consensus_ready"
            or receipt.get("canonical_attempt") != "attempt_a"
            or receipt.get("attempt_count") != 2
            or receipt.get("graph_observation_count") != 1
            or receipt.get("no_pooling") is not True
        ):
            raise HopPlanError("snapshot consensus does not authorize HOP planning")
        for binding, label in zip(
            bindings, ("consensus receipt", "canonical attempt A", "attempt B")
        ):
            _assert_bound_directory(binding, label)
            _assert_private_directory_files(binding, label)
        capture = _capture_verified_inputs(
            receipt,
            receipt_binding,
            attempt_a_binding,
            attempt_b_binding,
            source_spec,
            relation_policy,
            planner_input_ceiling_bytes=config["planner_input_ceiling_bytes"],
        )
        for binding, label in zip(
            bindings, ("consensus receipt", "canonical attempt A", "attempt B")
        ):
            _assert_private_directory_files(binding, label)
        return receipt, capture, bindings
    except Exception:
        for binding in bindings:
            binding.close()
        raise


def verify_plan(
    plan_dir,
    *,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    plan_input = Path(plan_dir)
    if _has_symlink_ancestor(plan_input):
        raise HopPlanError("HOP plan path cannot contain a symlink")
    resolved = plan_input.resolve()
    if _path_is_within(resolved, REPO_ROOT.resolve()) or _has_git_ancestor(resolved.parent):
        raise HopPlanError("HOP plan cannot be inside a Git worktree")
    plan_binding = _bind_directory(resolved, "HOP plan")
    bindings = ()
    try:
        manifest = _plan_manifest_from_binding(plan_binding)
        config = _config_from_manifest(manifest)
        _assert_no_output_input_overlap(
            resolved,
            (receipt_dir, attempt_a_dir, attempt_b_dir),
            (source_spec, relation_policy),
            source_spec,
            source_spec_maximum_size=config["planner_input_ceiling_bytes"],
        )
        receipt, capture, bindings = _verify_or_prepare_inputs(
            receipt_dir,
            attempt_a_dir,
            attempt_b_dir,
            source_spec,
            relation_policy,
            config,
        )
        expected_payloads, expected_manifest = _derive_plan(capture, receipt, config)
        if manifest != expected_manifest:
            raise HopPlanError("HOP plan manifest does not match re-derived plan")
        artifact_records = manifest["fingerprint_core"]["artifact_records"]
        if set(artifact_records) != set(ARTIFACT_NAMES):
            raise HopPlanError("HOP plan artifact inventory mismatch")
        for name in ARTIFACT_NAMES:
            expected_data = expected_payloads[name]
            observed_data = _read_bound_file(
                plan_binding, name, artifact_records[name], "HOP plan"
            )
            if observed_data != expected_data:
                raise HopPlanError("HOP plan artifact does not match re-derived plan")
        final_receipt = consensus.verify_consensus_receipt(
            receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if final_receipt != receipt:
            raise HopPlanError("consensus changed during HOP plan verification")
        final_capture = _capture_verified_inputs(
            receipt,
            bindings[0],
            bindings[1],
            bindings[2],
            source_spec,
            relation_policy,
            planner_input_ceiling_bytes=config["planner_input_ceiling_bytes"],
        )
        if final_capture != capture:
            raise HopPlanError("planning inputs changed during HOP plan verification")
        if _plan_manifest_from_binding(plan_binding) != manifest:
            raise HopPlanError("HOP plan manifest changed during verification")
        for name in ARTIFACT_NAMES:
            if _read_bound_file(
                plan_binding, name, artifact_records[name], "HOP plan"
            ) != expected_payloads[name]:
                raise HopPlanError("HOP plan artifact changed during verification")
        _assert_private_directory_files(plan_binding, "HOP plan")
        for binding, label in zip(
            bindings, ("consensus receipt", "canonical attempt A", "attempt B")
        ):
            _assert_private_directory_files(binding, label)
        return manifest
    finally:
        plan_binding.close()
        for binding in bindings:
            binding.close()


def prepare_plan(
    *,
    receipt_dir,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
    plan_dir,
    local_root,
    planner_input_ceiling_bytes,
    planner_edge_touch_ceiling,
    study_peak_rss_ceiling_bytes,
    effective_resistance_arm,
):
    config = _validated_config(
        {
            "effective_resistance_arm": effective_resistance_arm,
            "planner_edge_touch_ceiling": planner_edge_touch_ceiling,
            "planner_input_ceiling_bytes": planner_input_ceiling_bytes,
            "study_peak_rss_ceiling_bytes": study_peak_rss_ceiling_bytes,
        }
    )
    target, parent_binding = _validate_output_paths(
        plan_dir,
        local_root,
        (receipt_dir, attempt_a_dir, attempt_b_dir),
        (source_spec, relation_policy),
        source_spec,
        config["planner_input_ceiling_bytes"],
    )
    bindings = ()
    staging = None
    staging_binding = None
    staging_leaf = None
    committed = False
    try:
        receipt, capture, bindings = _verify_or_prepare_inputs(
            receipt_dir,
            attempt_a_dir,
            attempt_b_dir,
            source_spec,
            relation_policy,
            config,
        )
        payloads, manifest = _derive_plan(capture, receipt, config)

        # Re-run the complete verifier and recapture bound leaf bytes before the
        # transaction is installed.  This closes the ordinary verify-then-reopen
        # gap; downstream verification repeats the same derivation again.
        receipt_again = consensus.verify_consensus_receipt(
            receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if receipt_again != receipt:
            raise HopPlanError("consensus changed during HOP planning")
        capture_again = _capture_verified_inputs(
            receipt,
            bindings[0],
            bindings[1],
            bindings[2],
            source_spec,
            relation_policy,
            planner_input_ceiling_bytes=config["planner_input_ceiling_bytes"],
        )
        if capture_again != capture:
            raise HopPlanError("verified planning inputs changed during HOP planning")

        _assert_bound_directory(parent_binding, "plan output parent")
        staging_leaf, staging_binding = _create_bound_staging(
            parent_binding, target.name
        )
        staging = staging_binding.path
        for name in ARTIFACT_NAMES:
            _write_bound_bytes(staging_binding, name, payloads[name])
        _write_bound_bytes(staging_binding, MARKER_NAME, MARKER_BYTES)
        _write_bound_bytes(staging_binding, MANIFEST_NAME, _canonical_json(manifest))
        os.fsync(staging_binding.fd)

        verified_staging = verify_plan(
            staging,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if verified_staging != manifest:
            raise HopPlanError("staged HOP plan changed during verification")
        _assert_bound_directory(parent_binding, "plan output parent")
        _assert_bound_directory(staging_binding, "HOP plan staging directory")
        _rename_directory_noreplace(parent_binding, staging_leaf, target.name)
        staging_binding.path = target
        staging = target
        staging_leaf = target.name
        _assert_bound_directory(staging_binding, "installed HOP plan")
        _assert_bound_directory(parent_binding, "plan output parent")
        os.fsync(parent_binding.fd)
        _assert_bound_directory(parent_binding, "plan output parent")
        verified = verify_plan(
            target,
            receipt_dir=receipt_dir,
            attempt_a_dir=attempt_a_dir,
            attempt_b_dir=attempt_b_dir,
            source_spec=source_spec,
            relation_policy=relation_policy,
        )
        if verified != manifest:
            raise HopPlanError("installed HOP plan changed during verification")
        _assert_bound_directory(staging_binding, "installed HOP plan")
        _assert_bound_directory(parent_binding, "plan output parent")
        os.fsync(parent_binding.fd)
        _assert_bound_directory(parent_binding, "plan output parent")
        committed = True
        return manifest, (0 if manifest["accepted"] else 2)
    finally:
        try:
            if not committed and staging_binding is not None:
                _cleanup_bound_staging(parent_binding, staging_binding, staging_leaf)
        finally:
            if staging_binding is not None:
                staging_binding.close()
            for binding in bindings:
                binding.close()
            parent_binding.close()


def _arg_positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--receipt-dir", required=True)
    prepare.add_argument("--attempt-a-dir", required=True)
    prepare.add_argument("--attempt-b-dir", required=True)
    prepare.add_argument("--source-spec", required=True)
    prepare.add_argument("--relation-policy", required=True)
    prepare.add_argument("--plan-dir", required=True)
    prepare.add_argument("--local-root", required=True)
    prepare.add_argument("--local-only", action="store_true", required=True)
    prepare.add_argument("--planner-input-ceiling-bytes", type=_arg_positive_int, required=True)
    prepare.add_argument("--planner-edge-touch-ceiling", type=_arg_positive_int, required=True)
    prepare.add_argument("--study-peak-rss-ceiling-bytes", type=_arg_positive_int, required=True)
    prepare.add_argument(
        "--effective-resistance-arm", choices=("enabled", "omitted"), required=True
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt-dir", required=True)
    verify.add_argument("--attempt-a-dir", required=True)
    verify.add_argument("--attempt-b-dir", required=True)
    verify.add_argument("--source-spec", required=True)
    verify.add_argument("--relation-policy", required=True)
    verify.add_argument("--plan-dir", required=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest, exit_code = prepare_plan(
                receipt_dir=args.receipt_dir,
                attempt_a_dir=args.attempt_a_dir,
                attempt_b_dir=args.attempt_b_dir,
                source_spec=args.source_spec,
                relation_policy=args.relation_policy,
                plan_dir=args.plan_dir,
                local_root=args.local_root,
                planner_input_ceiling_bytes=args.planner_input_ceiling_bytes,
                planner_edge_touch_ceiling=args.planner_edge_touch_ceiling,
                study_peak_rss_ceiling_bytes=args.study_peak_rss_ceiling_bytes,
                effective_resistance_arm=args.effective_resistance_arm,
            )
            output = {
                "accepted": manifest["accepted"],
                "audit_batches": manifest["aggregate"]["audit_batches"],
                "calibration_batches": manifest["aggregate"]["calibration_batches"],
                "no_solve": True,
                "reason": manifest["reason"],
            }
        else:
            manifest = verify_plan(
                args.plan_dir,
                receipt_dir=args.receipt_dir,
                attempt_a_dir=args.attempt_a_dir,
                attempt_b_dir=args.attempt_b_dir,
                source_spec=args.source_spec,
                relation_policy=args.relation_policy,
            )
            exit_code = 0 if manifest["accepted"] else 2
            output = {
                "accepted": manifest["accepted"],
                "no_solve": True,
                "reason": manifest["reason"],
                "verified": True,
            }
        print(json.dumps(output, sort_keys=True))
        return exit_code
    except Exception:
        print(
            json.dumps({"error": "HOP plan failed closed"}, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
