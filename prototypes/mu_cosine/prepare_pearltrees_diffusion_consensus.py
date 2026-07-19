#!/usr/bin/env python3
"""Require exact two-process consensus before using a Pearltrees snapshot.

The two snapshot compilations are repeatability evidence for one frozen graph,
not two independent graph observations.  This gate never pools artifacts and
never retries a disagreement.  Detailed artifacts and the consensus receipt
are local-only.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unicodedata
import xml.parsers.expat

import declare_pearltrees_diffusion_sources as declaration


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SNAPSHOT_PREPARER = HERE / "prepare_pearltrees_diffusion_snapshot.py"
PRIVACY_RULE = HERE / "privacy.py"
SCHEMA = "pearltrees-diffusion-consensus-v1"
ALGORITHM = "two-fresh-process-exact-consensus-v1"
RECEIPT_NAME = "consensus_receipt.json"
MARKER_NAME = "LOCAL_ONLY_DO_NOT_PUBLISH"
MARKER_BYTES = b"LOCAL ONLY - DO NOT PUBLISH SNAPSHOT CONSENSUS ARTIFACTS\n"
RECEIPT_FILES = frozenset((RECEIPT_NAME, MARKER_NAME))
EVIDENCE_INTERPRETATION = (
    "same-code, same-runtime repeatability for one frozen graph; not independent replication"
)
DOWNSTREAM_REQUIREMENT = (
    "re-verify canonical attempt_a and bind its manifest record before any solve"
)
LEGACY_PARITY_ROLE = "diagnostic_only_not_readiness"
CRITICAL_COMPARISON_CHECKS = frozenset(
    (
        "aggregate_nonlegacy_equal",
        "fingerprint_core_equal",
        "implementation_records_equal",
        "nonlegacy_manifest_equal",
        "policy_records_equal",
        "population_records_equal",
        "readiness_equal",
        "resource_records_equal",
        "scientific_artifact_records_equal",
        "snapshot_fingerprint_equal",
        "snapshot_label_hash_equal",
        "source_records_equal",
    )
)


class ConsensusError(ValueError):
    """Fail-closed operational or contract error in consensus preparation."""


def _duplicate_checked_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ConsensusError("duplicate JSON key")
        value[key] = item
    return value


def _reject_nonfinite(_value):
    raise ConsensusError("non-finite JSON constant")


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
        raise ConsensusError("value is not canonical finite JSON") from exc
    return (text + "\n").encode("utf-8")


def _strict_json_bytes(data, label):
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConsensusError(f"invalid UTF-8 JSON in {label}") from exc
    if _canonical_json(value) != data:
        raise ConsensusError(f"{label} is not canonical JSON")
    return value


def _content_record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _hex_string_is_valid(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _content_record_is_valid(value):
    return (
        isinstance(value, dict)
        and set(value) == {"sha256", "size_bytes"}
        and _hex_string_is_valid(value["sha256"], 64)
        and isinstance(value["size_bytes"], int)
        and not isinstance(value["size_bytes"], bool)
        and value["size_bytes"] >= 0
    )


def _regular_file_record(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ConsensusError(f"{label} must be a regular non-symlink file")
    try:
        return _content_record(path.read_bytes())
    except OSError as exc:
        raise ConsensusError(f"{label} could not be read") from exc


def _path_is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _has_symlink_ancestor(path):
    absolute = Path(os.path.abspath(os.fspath(path)))
    return any(item.is_symlink() for item in (absolute, *absolute.parents))


def _is_git_worktree_marker(path):
    """Recognize an actual Git marker, not an unrelated empty `.git` path."""

    try:
        if path.is_symlink():
            return True
        if path.is_file():
            with open(path, "rb") as stream:
                return stream.read(256).lstrip().startswith(b"gitdir:")
        return path.is_dir() and (path / "HEAD").is_file()
    except OSError:
        # An unreadable candidate marker is conservatively treated as Git.
        return True


def _has_git_ancestor(path):
    return any(
        _is_git_worktree_marker(ancestor / ".git") for ancestor in (path, *path.parents)
    )


def _bind_directory_identities(paths):
    bindings = []
    for path in sorted({Path(item).resolve() for item in paths}, key=str):
        if _has_symlink_ancestor(path) or not path.is_dir():
            raise ConsensusError("bound output parent must be a non-symlink directory")
        try:
            record = path.stat()
        except OSError as exc:
            raise ConsensusError("output parent identity could not be bound") from exc
        bindings.append((path, record.st_dev, record.st_ino))
    return tuple(bindings)


def _assert_directory_identities(bindings):
    for path, expected_device, expected_inode in bindings:
        if _has_symlink_ancestor(path):
            raise ConsensusError("output parent path changed to a symlink")
        try:
            record = path.stat()
        except OSError as exc:
            raise ConsensusError("output parent identity became unavailable") from exc
        if (
            not stat.S_ISDIR(record.st_mode)
            or record.st_dev != expected_device
            or record.st_ino != expected_inode
        ):
            raise ConsensusError("output parent identity changed during consensus")


def _bind_existing_directory(path, label):
    path_input = Path(path)
    if _has_symlink_ancestor(path_input):
        raise ConsensusError(f"{label} path cannot contain a symlink")
    try:
        resolved = path_input.resolve(strict=True)
        record = resolved.stat()
    except OSError as exc:
        raise ConsensusError(f"{label} identity could not be bound") from exc
    if not stat.S_ISDIR(record.st_mode):
        raise ConsensusError(f"{label} must be a directory")
    return resolved, record.st_dev, record.st_ino


def _assert_bound_directory(binding, label):
    path, expected_device, expected_inode = binding
    if _has_symlink_ancestor(path):
        raise ConsensusError(f"{label} path changed to a symlink")
    try:
        record = path.stat()
    except OSError as exc:
        raise ConsensusError(f"{label} identity became unavailable") from exc
    if (
        not stat.S_ISDIR(record.st_mode)
        or record.st_dev != expected_device
        or record.st_ino != expected_inode
    ):
        raise ConsensusError(f"{label} identity changed during verification")
    return path


def _validate_output_paths(
    local_root,
    attempt_a_dir,
    attempt_b_dir,
    receipt_dir,
    source_spec,
    relation_policy,
):
    root_input = Path(local_root)
    raw_outputs = [Path(attempt_a_dir), Path(attempt_b_dir), Path(receipt_dir)]
    if _has_symlink_ancestor(root_input) or any(
        _has_symlink_ancestor(path) for path in raw_outputs
    ):
        raise ConsensusError("local-only output paths cannot be symlinks")
    root = root_input.resolve()
    if not root.is_dir():
        raise ConsensusError("local root must be an existing non-symlink directory")
    outputs = [path.resolve() for path in raw_outputs]
    if len(set(outputs)) != len(outputs):
        raise ConsensusError("attempt and receipt directories must not alias")
    for output in outputs:
        if output == root or not _path_is_within(output, root):
            raise ConsensusError("all output directories must be children of the local root")
        if _path_is_within(output, REPO_ROOT.resolve()) or _has_git_ancestor(output.parent):
            raise ConsensusError("local-only output cannot be inside a Git worktree")
        if output.exists():
            raise ConsensusError("attempt and receipt directories must be fresh")
        if not output.parent.is_dir():
            raise ConsensusError("output-directory parents must already exist")
    for index, first in enumerate(outputs):
        for second in outputs[index + 1 :]:
            if _path_is_within(first, second) or _path_is_within(second, first):
                raise ConsensusError("attempt and receipt directories cannot be nested")

    inputs = [Path(source_spec), Path(relation_policy)]
    for item in inputs:
        if _has_symlink_ancestor(item) or not item.is_file():
            raise ConsensusError("consensus inputs must be regular non-symlink files")
        resolved = item.resolve()
        try:
            if (
                resolved.stat().st_dev == root.stat().st_dev
                and resolved.stat().st_ino == root.stat().st_ino
            ):
                raise ConsensusError("local root aliases an input")
        except OSError as exc:
            raise ConsensusError("consensus input could not be validated") from exc
        for output in outputs:
            if (
                resolved == output
                or _path_is_within(resolved, output)
                or _path_is_within(output, resolved)
            ):
                raise ConsensusError("input and output paths overlap")
    return root, tuple(outputs)


def _positive_int(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _run_snapshot_cli(arguments):
    try:
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(SNAPSHOT_PREPARER),
                *map(str, arguments),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        ).returncode
    except OSError as exc:
        raise ConsensusError("snapshot compiler subprocess could not be started") from exc


def _assert_inputs_unchanged(expected, source_spec, relation_policy):
    observed = {
        "relation_policy": _regular_file_record(relation_policy, "relation policy"),
        "source_spec": _regular_file_record(source_spec, "source spec"),
    }
    if observed != expected:
        raise ConsensusError("source spec or relation policy changed across attempts")


def _run_attempt(
    run_dir,
    *,
    source_spec,
    relation_policy,
    local_root,
    minimum_anchors,
    resource_ceiling_bytes,
    expected_inputs,
    output_parent_bindings,
):
    _assert_directory_identities(output_parent_bindings)
    prepare_code = _run_snapshot_cli(
        (
            "prepare",
            "--source-spec",
            source_spec,
            "--relation-policy",
            relation_policy,
            "--run-dir",
            run_dir,
            "--local-root",
            local_root,
            "--local-only",
            "--minimum-anchors",
            minimum_anchors,
            "--resource-ceiling-bytes",
            resource_ceiling_bytes,
        )
    )
    _assert_inputs_unchanged(expected_inputs, source_spec, relation_policy)
    if prepare_code not in {0, 2}:
        raise ConsensusError("snapshot preparation had an operational failure")
    if not Path(run_dir).is_dir():
        raise ConsensusError("snapshot preparation did not install its run directory")

    attempt_binding = _bind_existing_directory(run_dir, "snapshot attempt")
    _assert_directory_identities(output_parent_bindings)
    verified_run_dir = _assert_bound_directory(attempt_binding, "snapshot attempt")
    verify_code = _run_snapshot_cli(("verify", "--run-dir", verified_run_dir))
    _assert_inputs_unchanged(expected_inputs, source_spec, relation_policy)
    if verify_code != 0:
        raise ConsensusError("snapshot verification had an operational failure")
    _assert_directory_identities(output_parent_bindings)
    verified_run_dir = _assert_bound_directory(attempt_binding, "snapshot attempt")
    manifest_path = verified_run_dir / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ConsensusError("verified snapshot manifest could not be read") from exc
    _assert_bound_directory(attempt_binding, "snapshot attempt")
    manifest = _strict_json_bytes(manifest_bytes, "verified snapshot manifest")
    if not isinstance(manifest, dict):
        raise ConsensusError("verified snapshot manifest is not an object")
    return {
        "manifest": manifest,
        "manifest_record": _content_record(manifest_bytes),
        "prepare_exit_code": prepare_code,
        "verify_exit_code": verify_code,
    }


def _scientific_artifact_records(manifest):
    records = manifest.get("artifact_records")
    if not isinstance(records, dict):
        raise ConsensusError("verified manifest artifact records are malformed")
    return {name: record for name, record in records.items() if name != "legacy_parity.json"}


def _without_legacy_parity(manifest):
    normalized = dict(manifest)
    artifacts = dict(normalized.get("artifact_records", {}))
    artifacts.pop("legacy_parity.json", None)
    normalized["artifact_records"] = artifacts
    aggregate = dict(normalized.get("aggregate", {}))
    aggregate.pop("legacy_parity_status", None)
    normalized["aggregate"] = aggregate
    return normalized


def _named_comparisons(first, second):
    try:
        first_core = first["fingerprint_core"]
        second_core = second["fingerprint_core"]
        first_readiness = {
            "graph_asset_ready": first["graph_asset_ready"],
            "minimum_anchor_count": first["aggregate"]["minimum_anchor_count"],
            "minimum_anchor_coverage_pass": first["aggregate"]["minimum_anchor_coverage_pass"],
            "privacy_certified": first["aggregate"]["privacy_certified"],
        }
        second_readiness = {
            "graph_asset_ready": second["graph_asset_ready"],
            "minimum_anchor_count": second["aggregate"]["minimum_anchor_count"],
            "minimum_anchor_coverage_pass": second["aggregate"]["minimum_anchor_coverage_pass"],
            "privacy_certified": second["aggregate"]["privacy_certified"],
        }
        first_population = {
            "largest_component_sha256": first_core["largest_component_sha256"],
            "study_universe_sha256": first_core["study_universe_sha256"],
        }
        second_population = {
            "largest_component_sha256": second_core["largest_component_sha256"],
            "study_universe_sha256": second_core["study_universe_sha256"],
        }
        first_resources = {
            "observed_contract_bytes": first_core["observed_contract_bytes"],
            "resource_ceiling_bytes": first_core["resource_ceiling_bytes"],
        }
        second_resources = {
            "observed_contract_bytes": second_core["observed_contract_bytes"],
            "resource_ceiling_bytes": second_core["resource_ceiling_bytes"],
        }
    except (KeyError, TypeError) as exc:
        raise ConsensusError("verified snapshot manifest is missing consensus fields") from exc
    first_legacy_record = first.get("artifact_records", {}).get("legacy_parity.json")
    second_legacy_record = second.get("artifact_records", {}).get("legacy_parity.json")
    first_nonlegacy_aggregate = dict(first.get("aggregate", {}))
    second_nonlegacy_aggregate = dict(second.get("aggregate", {}))
    first_legacy_status = first_nonlegacy_aggregate.pop("legacy_parity_status", None)
    second_legacy_status = second_nonlegacy_aggregate.pop("legacy_parity_status", None)
    return {
        "aggregate_nonlegacy_equal": first_nonlegacy_aggregate == second_nonlegacy_aggregate,
        "fingerprint_core_equal": first_core == second_core,
        "full_manifest_equal": first == second,
        "implementation_records_equal": first_core.get("implementation_records")
        == second_core.get("implementation_records"),
        "legacy_parity_artifact_record_equal": first_legacy_record == second_legacy_record,
        "legacy_parity_status_equal": first_legacy_status == second_legacy_status,
        "nonlegacy_manifest_equal": _without_legacy_parity(first)
        == _without_legacy_parity(second),
        "policy_records_equal": {
            "privacy_policy": first.get("privacy_policy"),
            "relation_policy": first.get("relation_policy"),
        }
        == {
            "privacy_policy": second.get("privacy_policy"),
            "relation_policy": second.get("relation_policy"),
        },
        "population_records_equal": first_population == second_population,
        "readiness_equal": first_readiness == second_readiness,
        "resource_records_equal": first_resources == second_resources,
        "scientific_artifact_records_equal": _scientific_artifact_records(first)
        == _scientific_artifact_records(second),
        "snapshot_fingerprint_equal": first.get("snapshot_fingerprint")
        == second.get("snapshot_fingerprint"),
        "snapshot_label_hash_equal": first.get("snapshot_label_hash")
        == second.get("snapshot_label_hash"),
        "source_records_equal": first_core.get("source_records")
        == second_core.get("source_records"),
    }


def _critical_comparisons_pass(checks):
    if set(checks) != CRITICAL_COMPARISON_CHECKS | {
        "full_manifest_equal",
        "legacy_parity_artifact_record_equal",
        "legacy_parity_status_equal",
    }:
        raise ConsensusError("consensus comparison inventory mismatch")
    return all(checks[name] for name in CRITICAL_COMPARISON_CHECKS)


def _validate_invocation_contract(attempt, minimum_anchors, resource_ceiling_bytes):
    manifest = attempt["manifest"]
    try:
        aggregate = manifest["aggregate"]
        core = manifest["fingerprint_core"]
        graph_ready = manifest["graph_asset_ready"]
        privacy_certified = aggregate["privacy_certified"]
        coverage = aggregate["minimum_anchor_coverage_pass"]
    except (KeyError, TypeError) as exc:
        raise ConsensusError("verified manifest is missing invocation fields") from exc
    if aggregate.get("minimum_anchor_count") != minimum_anchors:
        raise ConsensusError("snapshot compiler did not bind the requested minimum anchors")
    if core.get("resource_ceiling_bytes") != resource_ceiling_bytes:
        raise ConsensusError("snapshot compiler did not bind the requested resource ceiling")
    if any(not isinstance(value, bool) for value in (graph_ready, privacy_certified, coverage)):
        raise ConsensusError("snapshot readiness fields are not Boolean")
    expected_prepare_code = 0 if graph_ready else 2
    if attempt["prepare_exit_code"] != expected_prepare_code:
        raise ConsensusError("prepare exit code disagrees with verified readiness")


def _validate_attempt_input_records(attempt, expected_inputs):
    try:
        observed = attempt["manifest"]["input_records"]
    except (KeyError, TypeError) as exc:
        raise ConsensusError("snapshot manifest is missing exact input records") from exc
    if observed != expected_inputs:
        raise ConsensusError("snapshot attempt does not bind the supplied input records")


def _readiness_from_manifest(manifest):
    return {
        "graph_asset_ready": manifest["graph_asset_ready"],
        "minimum_anchor_count": manifest["aggregate"]["minimum_anchor_count"],
        "minimum_anchor_coverage_pass": manifest["aggregate"][
            "minimum_anchor_coverage_pass"
        ],
        "privacy_certified": manifest["aggregate"]["privacy_certified"],
        "resource_ceiling_bytes": manifest["fingerprint_core"]["resource_ceiling_bytes"],
    }


def _summary_from_manifest(manifest):
    aggregate = manifest["aggregate"]
    return {
        "eligible_anchor_count": aggregate["eligible_anchor_count"],
        "largest_component_node_count": aggregate["largest_component_node_count"],
        "physical_edge_count": aggregate["physical_edge_count"],
        "retained_node_count": aggregate["retained_node_count"],
    }


def _common_records_from_manifest(manifest):
    core = manifest["fingerprint_core"]
    return {
        "authoritative_artifact_set_sha256": core[
            "authoritative_artifact_set_sha256"
        ],
        "compiler_implementation_records": core["implementation_records"],
        "fingerprint_core_sha256": hashlib.sha256(_canonical_json(core)).hexdigest(),
        "largest_component_sha256": core["largest_component_sha256"],
        "numeric_contract": core["numeric_contract"],
        "repository_commit": core["repository_commit"],
        "study_universe_sha256": core["study_universe_sha256"],
    }


def _attempt_receipt_record(label, attempt):
    manifest = attempt["manifest"]
    return {
        "attempt_label": label,
        "legacy_parity_artifact_record": manifest.get("artifact_records", {}).get(
            "legacy_parity.json"
        ),
        "legacy_parity_status": manifest.get("aggregate", {}).get(
            "legacy_parity_status"
        ),
        "manifest_record": attempt["manifest_record"],
        "observed_contract_bytes": manifest["fingerprint_core"][
            "observed_contract_bytes"
        ],
        "prepare_exit_code": attempt["prepare_exit_code"],
        "readiness": _readiness_from_manifest(manifest),
        "snapshot_fingerprint": manifest["snapshot_fingerprint"],
        "verify_exit_code": attempt["verify_exit_code"],
    }


def _derive_consensus(manifest_a, manifest_b):
    checks = _named_comparisons(manifest_a, manifest_b)
    exact = _critical_comparisons_pass(checks)
    legacy_warning = not (
        checks["legacy_parity_artifact_record_equal"]
        and checks["legacy_parity_status_equal"]
    )
    if exact:
        common_fingerprint = manifest_a["snapshot_fingerprint"]
        common_records = _common_records_from_manifest(manifest_a)
        readiness = _readiness_from_manifest(manifest_a)
        aggregate_summary = _summary_from_manifest(manifest_a)
        accepted = readiness["privacy_certified"] and readiness["graph_asset_ready"]
        if accepted:
            reason = "exact_consensus_ready"
        elif not readiness["privacy_certified"]:
            reason = "privacy_not_certified"
        else:
            reason = "graph_asset_not_ready"
    else:
        common_fingerprint = None
        common_records = None
        readiness = None
        aggregate_summary = None
        accepted = False
        reason = "exact_consensus_mismatch"
    return {
        "accepted": accepted,
        "aggregate_summary": aggregate_summary,
        "common_records": common_records,
        "common_snapshot_fingerprint": common_fingerprint,
        "comparison_checks": checks,
        "readiness": readiness,
        "reason": reason,
        "repeatability_verified": exact,
        "warnings": ["legacy_parity_disagreement"] if legacy_warning else [],
    }


def _seal_receipt(receipt):
    sealed = dict(receipt)
    sealed["repeatability_contract_sha256"] = hashlib.sha256(
        _canonical_json(receipt)
    ).hexdigest()
    return sealed


def _write_bytes(path, data):
    with open(path, "xb") as stream:
        os.chmod(path, 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _rename_directory_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ConsensusError("atomic no-replace rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ConsensusError("receipt directory appeared during atomic installation")
    raise ConsensusError("atomic no-replace receipt installation failed") from OSError(
        error_number, os.strerror(error_number)
    )


def _install_receipt(receipt_dir, receipt):
    receipt_dir = Path(receipt_dir)
    temporary = Path(tempfile.mkdtemp(prefix=f".{receipt_dir.name}.", dir=receipt_dir.parent))
    os.chmod(temporary, 0o700)
    installed = False
    try:
        _write_bytes(temporary / RECEIPT_NAME, _canonical_json(receipt))
        _write_bytes(temporary / MARKER_NAME, MARKER_BYTES)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _rename_directory_noreplace(temporary, receipt_dir)
        installed = True
        parent_fd = os.open(receipt_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)


def _validate_consensus_receipt_structure(receipt_dir):
    receipt_input = Path(receipt_dir)
    receipt_binding = _bind_existing_directory(receipt_input, "consensus receipt")
    receipt_dir = _assert_bound_directory(receipt_binding, "consensus receipt")
    if _path_is_within(receipt_dir, REPO_ROOT.resolve()) or _has_git_ancestor(
        receipt_dir.parent
    ):
        raise ConsensusError("consensus receipt directory cannot be inside a Git worktree")
    if not receipt_dir.is_dir() or stat.S_IMODE(receipt_dir.stat().st_mode) != 0o700:
        raise ConsensusError("consensus receipt directory is unavailable or not mode 0700")
    entries = list(receipt_dir.iterdir())
    if any(item.is_symlink() or not item.is_file() for item in entries):
        raise ConsensusError("consensus receipt contains a nonregular entry")
    if {item.name for item in entries} != RECEIPT_FILES:
        raise ConsensusError("consensus receipt file set mismatch")
    if any(stat.S_IMODE(item.stat().st_mode) != 0o600 for item in entries):
        raise ConsensusError("consensus receipt artifact mode is not 0600")
    if (receipt_dir / MARKER_NAME).read_bytes() != MARKER_BYTES:
        raise ConsensusError("consensus local-only marker mismatch")
    data = (receipt_dir / RECEIPT_NAME).read_bytes()
    _assert_bound_directory(receipt_binding, "consensus receipt")
    receipt = _strict_json_bytes(data, "consensus receipt")
    expected_keys = {
        "accepted",
        "aggregate_summary",
        "algorithm",
        "attempt_count",
        "attempts",
        "canonical_attempt",
        "common_records",
        "common_snapshot_fingerprint",
        "comparison_checks",
        "downstream_requirement",
        "evidence_interpretation",
        "graph_observation_count",
        "graph_gate_pass",
        "implementation_records",
        "input_records",
        "legacy_parity_role",
        "no_pooling",
        "readiness",
        "reason",
        "repeatability_contract_sha256",
        "repeatability_verified",
        "schema",
        "warnings",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise ConsensusError("consensus receipt fields mismatch")
    if receipt.get("schema") != SCHEMA:
        raise ConsensusError("consensus receipt schema mismatch")
    if receipt.get("algorithm") != ALGORITHM:
        raise ConsensusError("consensus receipt algorithm mismatch")
    expected_repeatability_hash = receipt["repeatability_contract_sha256"]
    unsealed = dict(receipt)
    del unsealed["repeatability_contract_sha256"]
    if (
        not isinstance(expected_repeatability_hash, str)
        or len(expected_repeatability_hash) != 64
        or hashlib.sha256(_canonical_json(unsealed)).hexdigest()
        != expected_repeatability_hash
    ):
        raise ConsensusError("consensus repeatability-contract hash mismatch")
    if (
        receipt["attempt_count"] != 2
        or receipt["graph_observation_count"] != 1
        or receipt["no_pooling"] is not True
        or receipt["canonical_attempt"] != "attempt_a"
        or receipt["evidence_interpretation"] != EVIDENCE_INTERPRETATION
        or receipt["downstream_requirement"] != DOWNSTREAM_REQUIREMENT
        or receipt["legacy_parity_role"] != LEGACY_PARITY_ROLE
    ):
        raise ConsensusError("consensus interpretation contract mismatch")

    input_records = receipt["input_records"]
    if (
        not isinstance(input_records, dict)
        or set(input_records) != {"relation_policy", "source_spec"}
        or any(not _content_record_is_valid(value) for value in input_records.values())
    ):
        raise ConsensusError("consensus input records are malformed")
    expected_runtime = {
        "expat_version": xml.parsers.expat.EXPAT_VERSION,
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "sqlite_version": sqlite3.sqlite_version,
        "unicode_data_version": unicodedata.unidata_version,
    }
    implementation = receipt["implementation_records"]
    if (
        not isinstance(implementation, dict)
        or set(implementation)
        != {
            "consensus_preparer",
            "runtime_identity",
            "snapshot_preparer_entrypoint",
            "source_declaration_validator",
        }
        or not _content_record_is_valid(implementation["consensus_preparer"])
        or not _content_record_is_valid(implementation["snapshot_preparer_entrypoint"])
        or not _content_record_is_valid(implementation["source_declaration_validator"])
        or implementation["consensus_preparer"]
        != _regular_file_record(Path(__file__).resolve(), "consensus preparer")
        or implementation["snapshot_preparer_entrypoint"]
        != _regular_file_record(SNAPSHOT_PREPARER, "snapshot preparer entrypoint")
        or implementation["source_declaration_validator"]
        != _regular_file_record(
            Path(declaration.__file__).resolve(), "source declaration validator"
        )
        or implementation["runtime_identity"] != expected_runtime
    ):
        raise ConsensusError("consensus implementation provenance mismatch")

    attempts = receipt["attempts"]
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise ConsensusError("consensus attempts must contain exactly two records")
    for expected_label, attempt in zip(("attempt_a", "attempt_b"), attempts):
        if (
            not isinstance(attempt, dict)
            or set(attempt)
            != {
                "attempt_label",
                "legacy_parity_artifact_record",
                "legacy_parity_status",
                "manifest_record",
                "observed_contract_bytes",
                "prepare_exit_code",
                "readiness",
                "snapshot_fingerprint",
                "verify_exit_code",
            }
            or attempt["attempt_label"] != expected_label
            or not _content_record_is_valid(
                attempt["legacy_parity_artifact_record"]
            )
            or not isinstance(attempt["legacy_parity_status"], str)
            or not attempt["legacy_parity_status"]
            or not _content_record_is_valid(attempt["manifest_record"])
            or not isinstance(attempt["observed_contract_bytes"], int)
            or isinstance(attempt["observed_contract_bytes"], bool)
            or attempt["observed_contract_bytes"] < 0
            or attempt["prepare_exit_code"] not in {0, 2}
            or not _hex_string_is_valid(attempt["snapshot_fingerprint"], 64)
            or attempt["verify_exit_code"] != 0
        ):
            raise ConsensusError("consensus attempt record is malformed")
        readiness = attempt["readiness"]
        if (
            not isinstance(readiness, dict)
            or set(readiness)
            != {
                "graph_asset_ready",
                "minimum_anchor_count",
                "minimum_anchor_coverage_pass",
                "privacy_certified",
                "resource_ceiling_bytes",
            }
            or any(
                not isinstance(readiness[key], bool)
                for key in (
                    "graph_asset_ready",
                    "minimum_anchor_coverage_pass",
                    "privacy_certified",
                )
            )
            or not isinstance(readiness["minimum_anchor_count"], int)
            or isinstance(readiness["minimum_anchor_count"], bool)
            or readiness["minimum_anchor_count"] < 1
            or not isinstance(readiness["resource_ceiling_bytes"], int)
            or isinstance(readiness["resource_ceiling_bytes"], bool)
            or readiness["resource_ceiling_bytes"] < 1
            or readiness["graph_asset_ready"]
            is not (
                readiness["privacy_certified"]
                and readiness["minimum_anchor_coverage_pass"]
            )
            or attempt["prepare_exit_code"]
            != (0 if readiness["graph_asset_ready"] else 2)
        ):
            raise ConsensusError("consensus attempt readiness is malformed")

    checks = receipt["comparison_checks"]
    expected_check_keys = CRITICAL_COMPARISON_CHECKS | {
        "full_manifest_equal",
        "legacy_parity_artifact_record_equal",
        "legacy_parity_status_equal",
    }
    if (
        not isinstance(checks, dict)
        or set(checks) != expected_check_keys
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise ConsensusError("consensus comparison checks are malformed")
    scientific_exact = all(checks[name] for name in CRITICAL_COMPARISON_CHECKS)
    if receipt["repeatability_verified"] is not scientific_exact:
        raise ConsensusError("repeatability decision disagrees with scientific checks")
    legacy_warning = not (
        checks["legacy_parity_artifact_record_equal"]
        and checks["legacy_parity_status_equal"]
    )
    expected_warnings = ["legacy_parity_disagreement"] if legacy_warning else []
    if receipt["warnings"] != expected_warnings:
        raise ConsensusError("consensus warning state mismatch")
    if scientific_exact and not legacy_warning and not checks["full_manifest_equal"]:
        raise ConsensusError("full manifest equality disagrees with component checks")
    if checks["full_manifest_equal"] and legacy_warning:
        raise ConsensusError("legacy warning contradicts full manifest equality")
    if checks["full_manifest_equal"] is not (
        attempts[0]["manifest_record"] == attempts[1]["manifest_record"]
    ):
        raise ConsensusError("full manifest equality disagrees with content records")
    if checks["legacy_parity_artifact_record_equal"] is not (
        attempts[0]["legacy_parity_artifact_record"]
        == attempts[1]["legacy_parity_artifact_record"]
    ):
        raise ConsensusError("legacy artifact equality disagrees with attempt records")
    if checks["legacy_parity_status_equal"] is not (
        attempts[0]["legacy_parity_status"]
        == attempts[1]["legacy_parity_status"]
    ):
        raise ConsensusError("legacy status equality disagrees with attempt records")

    if scientific_exact:
        readiness = receipt["readiness"]
        if readiness != attempts[0]["readiness"] or readiness != attempts[1]["readiness"]:
            raise ConsensusError("common readiness disagrees with attempts")
        fingerprint = receipt["common_snapshot_fingerprint"]
        if not _hex_string_is_valid(fingerprint, 64):
            raise ConsensusError("common snapshot fingerprint is malformed")
        if any(attempt["snapshot_fingerprint"] != fingerprint for attempt in attempts):
            raise ConsensusError("attempt fingerprints disagree with common fingerprint")
        if attempts[0]["observed_contract_bytes"] != attempts[1]["observed_contract_bytes"]:
            raise ConsensusError("repeatable attempts disagree on observed contract bytes")
        common_records = receipt["common_records"]
        if (
            not isinstance(common_records, dict)
            or set(common_records)
            != {
                "authoritative_artifact_set_sha256",
                "compiler_implementation_records",
                "fingerprint_core_sha256",
                "largest_component_sha256",
                "numeric_contract",
                "repository_commit",
                "study_universe_sha256",
            }
            or any(
                not _hex_string_is_valid(common_records[key], 64)
                for key in (
                    "authoritative_artifact_set_sha256",
                    "fingerprint_core_sha256",
                    "largest_component_sha256",
                    "study_universe_sha256",
                )
            )
            or common_records["fingerprint_core_sha256"] != fingerprint
            or not _hex_string_is_valid(common_records["repository_commit"], 40)
            or common_records["numeric_contract"]
            != {
                "downstream_decision_dtype": "float64",
                "preparer_arithmetic": "exact_integer_graph",
                "preparer_threads": 1,
            }
            or not isinstance(common_records["compiler_implementation_records"], dict)
            or set(common_records["compiler_implementation_records"])
            != {"preparer", "privacy_rule"}
            or any(
                not _content_record_is_valid(value)
                for value in common_records["compiler_implementation_records"].values()
            )
            or common_records["compiler_implementation_records"]["preparer"]
            != implementation["snapshot_preparer_entrypoint"]
            or common_records["compiler_implementation_records"]["privacy_rule"]
            != _regular_file_record(PRIVACY_RULE, "snapshot privacy rule")
        ):
            raise ConsensusError("common repeatability records are malformed")
        summary = receipt["aggregate_summary"]
        if (
            not isinstance(summary, dict)
            or set(summary)
            != {
                "eligible_anchor_count",
                "largest_component_node_count",
                "physical_edge_count",
                "retained_node_count",
            }
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in summary.values()
            )
        ):
            raise ConsensusError("consensus aggregate summary is malformed")
        expected_accepted = readiness["privacy_certified"] and readiness["graph_asset_ready"]
        if expected_accepted:
            expected_reason = "exact_consensus_ready"
        elif not readiness["privacy_certified"]:
            expected_reason = "privacy_not_certified"
        else:
            expected_reason = "graph_asset_not_ready"
    else:
        expected_accepted = False
        expected_reason = "exact_consensus_mismatch"
        if (
            receipt["readiness"] is not None
            or receipt["aggregate_summary"] is not None
            or receipt["common_records"] is not None
            or receipt["common_snapshot_fingerprint"] is not None
        ):
            raise ConsensusError("mismatched consensus exposes invalid common fields")
    if (
        receipt["accepted"] is not expected_accepted
        or receipt["graph_gate_pass"] is not expected_accepted
        or receipt["reason"] != expected_reason
    ):
        raise ConsensusError("consensus decision does not follow verified checks")
    return receipt


def _load_reverified_attempt(run_dir, prepare_exit_code, *, binding=None):
    run_input = Path(run_dir)
    attempt_binding = binding or _bind_existing_directory(
        run_input, "snapshot attempt"
    )
    run_resolved = _assert_bound_directory(attempt_binding, "snapshot attempt")
    if _path_is_within(run_resolved, REPO_ROOT.resolve()) or _has_git_ancestor(
        run_resolved.parent
    ):
        raise ConsensusError("snapshot attempt cannot be inside a Git worktree")
    verify_code = _run_snapshot_cli(("verify", "--run-dir", run_resolved))
    if verify_code != 0:
        raise ConsensusError("snapshot attempt failed independent verification")
    run_resolved = _assert_bound_directory(attempt_binding, "snapshot attempt")
    manifest_path = run_resolved / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ConsensusError("reverified snapshot manifest could not be read") from exc
    _assert_bound_directory(attempt_binding, "snapshot attempt")
    manifest = _strict_json_bytes(manifest_bytes, "reverified snapshot manifest")
    if not isinstance(manifest, dict):
        raise ConsensusError("reverified snapshot manifest is not an object")
    return {
        "manifest": manifest,
        "manifest_record": _content_record(manifest_bytes),
        "prepare_exit_code": prepare_exit_code,
        "verify_exit_code": verify_code,
    }


def verify_consensus_receipt(
    receipt_dir,
    *,
    attempt_a_dir,
    attempt_b_dir,
    source_spec,
    relation_policy,
):
    """Verify a receipt against both installed attempts and frozen inputs."""

    try:
        declaration.verify_installed_source_spec(source_spec)
    except declaration.DeclarationError as exc:
        raise ConsensusError("source declaration bundle failed verification") from exc
    if _has_symlink_ancestor(Path(relation_policy)):
        raise ConsensusError("relation policy path cannot contain a symlink")
    receipt = _validate_consensus_receipt_structure(receipt_dir)
    observed_inputs = {
        "relation_policy": _regular_file_record(relation_policy, "relation policy"),
        "source_spec": _regular_file_record(source_spec, "source spec"),
    }
    if receipt["input_records"] != observed_inputs:
        raise ConsensusError("consensus receipt input records do not match supplied inputs")

    recorded_attempts = receipt["attempts"]
    attempt_a_binding = _bind_existing_directory(attempt_a_dir, "attempt A")
    attempt_b_binding = _bind_existing_directory(attempt_b_dir, "attempt B")
    attempt_a_path = attempt_a_binding[0]
    attempt_b_path = attempt_b_binding[0]
    if (
        attempt_a_path == attempt_b_path
        or attempt_a_binding[1:] == attempt_b_binding[1:]
        or _path_is_within(attempt_a_path, attempt_b_path)
        or _path_is_within(attempt_b_path, attempt_a_path)
    ):
        raise ConsensusError("consensus requires two distinct nonnested attempt directories")
    actual_a = _load_reverified_attempt(
        attempt_a_path,
        recorded_attempts[0]["prepare_exit_code"],
        binding=attempt_a_binding,
    )
    actual_b = _load_reverified_attempt(
        attempt_b_path,
        recorded_attempts[1]["prepare_exit_code"],
        binding=attempt_b_binding,
    )
    _validate_attempt_input_records(actual_a, observed_inputs)
    _validate_attempt_input_records(actual_b, observed_inputs)
    _validate_invocation_contract(
        actual_a,
        recorded_attempts[0]["readiness"]["minimum_anchor_count"],
        recorded_attempts[0]["readiness"]["resource_ceiling_bytes"],
    )
    _validate_invocation_contract(
        actual_b,
        recorded_attempts[1]["readiness"]["minimum_anchor_count"],
        recorded_attempts[1]["readiness"]["resource_ceiling_bytes"],
    )
    actual_attempt_records = [
        _attempt_receipt_record("attempt_a", actual_a),
        _attempt_receipt_record("attempt_b", actual_b),
    ]
    if recorded_attempts != actual_attempt_records:
        raise ConsensusError("consensus attempt records do not match installed attempts")

    derived = _derive_consensus(actual_a["manifest"], actual_b["manifest"])
    for key, value in derived.items():
        if receipt[key] != value:
            raise ConsensusError(f"consensus receipt {key} is not derived from attempts")
    if receipt["graph_gate_pass"] is not derived["accepted"]:
        raise ConsensusError("consensus graph gate is not derived from attempts")
    return receipt


def prepare_consensus(
    source_spec,
    relation_policy,
    attempt_a_dir,
    attempt_b_dir,
    receipt_dir,
    local_root,
    *,
    minimum_anchors=128,
    resource_ceiling_bytes,
):
    if isinstance(minimum_anchors, bool) or minimum_anchors < 1:
        raise ConsensusError("minimum_anchors must be positive")
    if isinstance(resource_ceiling_bytes, bool) or resource_ceiling_bytes < 1:
        raise ConsensusError("resource ceiling must be positive")
    try:
        declaration.verify_installed_source_spec(source_spec)
    except declaration.DeclarationError as exc:
        raise ConsensusError("source declaration bundle failed verification") from exc
    local_root, outputs = _validate_output_paths(
        local_root,
        attempt_a_dir,
        attempt_b_dir,
        receipt_dir,
        source_spec,
        relation_policy,
    )
    attempt_a_dir, attempt_b_dir, receipt_dir = outputs
    output_parent_bindings = _bind_directory_identities(
        (local_root, *(path.parent for path in outputs))
    )
    declaration_dir = Path(source_spec).resolve().parent
    if declaration_dir == local_root or not _path_is_within(
        declaration_dir, local_root
    ):
        raise ConsensusError("source declaration bundle must be below the local root")
    input_records = {
        "relation_policy": _regular_file_record(relation_policy, "relation policy"),
        "source_spec": _regular_file_record(source_spec, "source spec"),
    }
    implementation_records = {
        "consensus_preparer": _regular_file_record(Path(__file__).resolve(), "consensus preparer"),
        "runtime_identity": {
            "expat_version": xml.parsers.expat.EXPAT_VERSION,
            "python_implementation": sys.implementation.name,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "sqlite_version": sqlite3.sqlite_version,
            "unicode_data_version": unicodedata.unidata_version,
        },
        "snapshot_preparer_entrypoint": _regular_file_record(
            SNAPSHOT_PREPARER, "snapshot preparer entrypoint"
        ),
        "source_declaration_validator": _regular_file_record(
            Path(declaration.__file__).resolve(), "source declaration validator"
        ),
    }

    common = {
        "source_spec": Path(source_spec).resolve(),
        "relation_policy": Path(relation_policy).resolve(),
        "local_root": local_root,
        "minimum_anchors": minimum_anchors,
        "resource_ceiling_bytes": resource_ceiling_bytes,
        "expected_inputs": input_records,
        "output_parent_bindings": output_parent_bindings,
    }
    attempt_a = _run_attempt(attempt_a_dir, **common)
    try:
        declaration.verify_installed_source_spec(source_spec)
    except declaration.DeclarationError as exc:
        raise ConsensusError("source declaration changed after attempt A") from exc
    attempt_b = _run_attempt(attempt_b_dir, **common)
    try:
        declaration.verify_installed_source_spec(source_spec)
    except declaration.DeclarationError as exc:
        raise ConsensusError("source declaration changed after attempt B") from exc
    _assert_inputs_unchanged(input_records, source_spec, relation_policy)
    _validate_invocation_contract(attempt_a, minimum_anchors, resource_ceiling_bytes)
    _validate_invocation_contract(attempt_b, minimum_anchors, resource_ceiling_bytes)
    _validate_attempt_input_records(attempt_a, input_records)
    _validate_attempt_input_records(attempt_b, input_records)

    manifest_a = attempt_a["manifest"]
    manifest_b = attempt_b["manifest"]
    derived = _derive_consensus(manifest_a, manifest_b)
    accepted = derived["accepted"]

    receipt = {
        "accepted": accepted,
        "aggregate_summary": derived["aggregate_summary"],
        "algorithm": ALGORITHM,
        "attempt_count": 2,
        "attempts": [
            _attempt_receipt_record("attempt_a", attempt_a),
            _attempt_receipt_record("attempt_b", attempt_b),
        ],
        "canonical_attempt": "attempt_a",
        "common_records": derived["common_records"],
        "common_snapshot_fingerprint": derived["common_snapshot_fingerprint"],
        "comparison_checks": derived["comparison_checks"],
        "downstream_requirement": DOWNSTREAM_REQUIREMENT,
        "evidence_interpretation": EVIDENCE_INTERPRETATION,
        "graph_observation_count": 1,
        "graph_gate_pass": accepted,
        "implementation_records": implementation_records,
        "input_records": input_records,
        "legacy_parity_role": LEGACY_PARITY_ROLE,
        "no_pooling": True,
        "readiness": derived["readiness"],
        "reason": derived["reason"],
        "repeatability_verified": derived["repeatability_verified"],
        "schema": SCHEMA,
        "warnings": derived["warnings"],
    }
    receipt = _seal_receipt(receipt)
    _assert_directory_identities(output_parent_bindings)
    _install_receipt(receipt_dir, receipt)
    _assert_directory_identities(output_parent_bindings)
    verified = verify_consensus_receipt(
        receipt_dir,
        attempt_a_dir=attempt_a_dir,
        attempt_b_dir=attempt_b_dir,
        source_spec=source_spec,
        relation_policy=relation_policy,
    )
    if verified != receipt:
        raise ConsensusError("installed consensus receipt changed during verification")
    return receipt, (0 if accepted else 2)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-spec", required=True)
    prepare.add_argument("--relation-policy", required=True)
    prepare.add_argument("--attempt-a-dir", required=True)
    prepare.add_argument("--attempt-b-dir", required=True)
    prepare.add_argument("--receipt-dir", required=True)
    prepare.add_argument("--local-root", required=True)
    prepare.add_argument("--local-only", action="store_true", required=True)
    prepare.add_argument("--minimum-anchors", type=_positive_int, default=128)
    prepare.add_argument("--resource-ceiling-bytes", type=_positive_int, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt-dir", required=True)
    verify.add_argument("--attempt-a-dir", required=True)
    verify.add_argument("--attempt-b-dir", required=True)
    verify.add_argument("--source-spec", required=True)
    verify.add_argument("--relation-policy", required=True)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            receipt, exit_code = prepare_consensus(
                args.source_spec,
                args.relation_policy,
                args.attempt_a_dir,
                args.attempt_b_dir,
                args.receipt_dir,
                args.local_root,
                minimum_anchors=args.minimum_anchors,
                resource_ceiling_bytes=args.resource_ceiling_bytes,
            )
            output = {
                "accepted": receipt["accepted"],
                "graph_asset_ready": (
                    receipt["readiness"]["graph_asset_ready"]
                    if receipt["readiness"] is not None
                    else False
                ),
                "reason": receipt["reason"],
            }
        else:
            receipt = verify_consensus_receipt(
                args.receipt_dir,
                attempt_a_dir=args.attempt_a_dir,
                attempt_b_dir=args.attempt_b_dir,
                source_spec=args.source_spec,
                relation_policy=args.relation_policy,
            )
            exit_code = 0 if receipt["accepted"] else 2
            output = {
                "accepted": receipt["accepted"],
                "reason": receipt["reason"],
                "verified": True,
            }
        print(json.dumps(output, sort_keys=True))
        return exit_code
    except Exception:
        print(
            json.dumps({"error": "snapshot consensus failed closed"}, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
