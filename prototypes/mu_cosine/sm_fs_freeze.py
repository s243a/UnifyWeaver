#!/usr/bin/env python3
"""SM-FS corpus freezer v2 — repairs sm_fs_filing.py per gpt-5.6-sol's review before any training.

Fixes, point by point (sol 2026-07-24):
  1. CATALOG WITHOUT PLACEMENT COUNTS: each privacy-specific training view retains every admitted
     exact destination plus its ancestors, with no dirs-with-≥N-maps threshold. This catalog is
     still placement-derived/transductive and is not an evaluation catalog. Exact PATH identity
     is primary; duplicate leaf titles stay distinct rows (title-equivalence is a sensitivity).
  2. DESTINATION-DISJOINT SPLIT: split by destination directory (all maps filed in a directory
     travel together); training additionally EXCLUDES any row whose destination is an ancestor
     or descendant of a reserved destination (lineage-family isolation). Ancestor components
     shared at higher levels are recorded as a documented transductive limitation.
  3. REPRODUCIBLE MEMBERSHIP: the ledger stores every row individually (path id, file sha256,
     split, classification) plus tree-snapshot and code/E5-revision fingerprints — membership is
     independently reproducible, not an aggregate digest.
  4. PRIVACY: consumes and source-verifies the content-addressed JSONL index from
     sm_fs_privacy.py
     (~/mu_data/sm_fs_privacy_index.jsonl). Public and private training views are emitted as
     separate policy-filtered bundles; unknown rows remain review-only. No unaudited fallback is
     accepted. Content addressing provides integrity and reproducibility, not external
     authentication.
  5. DURABLE CACHE + PINNED REVISION: everything under ~/mu_data; E5_REVISION recorded.

Also emits the lineage(fs) TRAINING TARGETS from the exploration partition only: rows
(map_title, ancestor_dir_title, hop) with target decay^(hop-1), process expression
"lineage(fs,decay=0.85)" in the header (P4 provenance convention).

  python3 sm_fs_freeze.py            # freeze + emit ledger and training targets (no scoring)
"""
import argparse
from collections import Counter
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mu_attention
import sm_fs_privacy
from mu_attention import E5_REVISION
from sm_fs_privacy import (
    load_index,
    verify_index_source,
)

DEFAULT_ROOT = "/mnt/c/Users/johnc/Dropbox/root"
DECAY = 0.85
EXPR = f"lineage(fs,decay={DECAY})"
BUNDLE_SCHEMA = "unifyweaver.sm-fs-training-bundle.v1"
BUNDLE_FILES = frozenset(
    {"ledger.json", "lineage_fs_targets.tsv", "manifest.json"}
)


def sha_file(p, cap=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(cap), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _artifact_record(data):
    return {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _valid_artifact_record(value):
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        return False
    byte_count = value["bytes"]
    digest = value["sha256"]
    return (
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count >= 0
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def _absolute_path(path):
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _descriptor_snapshot(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_private_regular(metadata, description):
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{description} is not a regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError(f"{description} is not mode 0600")
    if metadata.st_uid != os.geteuid():
        raise ValueError(f"{description} is not owned by the current user")
    if metadata.st_nlink != 1:
        raise ValueError(f"{description} must have exactly one hard link")


def _read_validated_descriptor(descriptor, description, path_stat):
    """Read bytes from one inode and prove the named path stayed on that inode."""

    descriptor_before = os.fstat(descriptor)
    _validate_private_regular(descriptor_before, description)
    try:
        path_before = path_stat()
    except OSError as exc:
        raise ValueError(f"{description} path is unavailable") from exc
    if _descriptor_snapshot(path_before) != _descriptor_snapshot(
        descriptor_before
    ):
        raise ValueError(f"{description} path/descriptor identity mismatch")

    chunks = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)

    descriptor_after = os.fstat(descriptor)
    try:
        path_after = path_stat()
    except OSError as exc:
        raise ValueError(f"{description} path changed while being read") from exc
    before = _descriptor_snapshot(descriptor_before)
    if (
        _descriptor_snapshot(descriptor_after) != before
        or _descriptor_snapshot(path_after) != before
    ):
        raise ValueError(f"{description} changed while being read")
    return b"".join(chunks)


def _read_bound_bytes(path, description):
    """Read one regular file once and bind all later parsing to those bytes."""

    resolved = _absolute_path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(str(resolved), flags)
    except OSError as exc:
        raise ValueError(f"{description} is unavailable: {resolved}") from exc
    try:
        data = _read_validated_descriptor(
            descriptor,
            description,
            lambda: resolved.lstat(),
        )
    finally:
        os.close(descriptor)
    return resolved, data, _artifact_record(data)


def _parse_privacy_index_bytes(data):
    """Parse exactly ``data`` through sm_fs_privacy's validating API."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".sm-fs-privacy-bound.", suffix=".jsonl"
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
        return load_index(Path(temporary_name))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except OSError:
            pass


def _load_bound_privacy_index(path):
    resolved, data, artifact = _read_bound_bytes(path, "privacy index")
    header, index = _parse_privacy_index_bytes(data)
    return resolved, artifact, header, index


def _code_records():
    return {
        "mu_attention.py": {
            "path": "prototypes/mu_cosine/mu_attention.py",
            "sha256": sha_file(Path(mu_attention.__file__)),
        },
        "sm_fs_freeze.py": {
            "path": "prototypes/mu_cosine/sm_fs_freeze.py",
            "sha256": sha_file(Path(__file__)),
        },
        "sm_fs_privacy.py": {
            "path": "prototypes/mu_cosine/sm_fs_privacy.py",
            "sha256": sha_file(Path(sm_fs_privacy.__file__)),
        },
    }


def catalog_for_rows(rows):
    """Return the exact policy view: included destinations and all ancestors."""

    return sorted(
        {
            "/".join(parts[:depth])
            for row in rows
            for parts in (row["dest"].split("/"),)
            for depth in range(1, len(parts) + 1)
        }
    )


def assert_catalog_complete(rows, catalog):
    catalog_set = set(catalog)
    for row in rows:
        if row["dest"] not in catalog_set:
            raise ValueError(
                f"policy catalog omitted included destination: {row['dest']}"
            )
        parts = row["dest"].split("/")
        missing = [
            "/".join(parts[:depth])
            for depth in range(1, len(parts) + 1)
            if "/".join(parts[:depth]) not in catalog_set
        ]
        if missing:
            raise ValueError(
                "policy catalog omitted included destination ancestor(s): "
                + ", ".join(missing)
            )


def classify(rel_parts, fname, index):
    key = "/".join(rel_parts + [fname])
    record = index.get(key)
    if record is None:
        raise ValueError(f"privacy index omitted current map: {key}")
    return (
        record["classification"],
        list(record.get("reasons", ())),
        "parse_error_type" not in record,
    )


def training_privacy_allows(classification, policy):
    if policy == "public-only":
        return classification == "public"
    if policy == "private-only":
        return classification == "private"
    raise ValueError(f"unknown training privacy policy: {policy}")


def _smmx_paths(corpus_root):
    corpus_root = _absolute_path(corpus_root)
    try:
        with sm_fs_privacy.open_corpus_root(corpus_root) as (root, root_fd):
            return [
                root / PurePosixPath(binding.relative)
                for binding in sm_fs_privacy.smmx_member_bindings(root_fd)
            ]
    except sm_fs_privacy.SmFsPrivacyError as exc:
        raise ValueError(str(exc)) from exc


def _iter_verified_source_maps(corpus_root, index):
    """Yield current maps only after their exact bytes match their index row."""

    corpus_root = _absolute_path(corpus_root)
    seen = set()
    try:
        root_context = sm_fs_privacy.open_corpus_root(corpus_root)
        with root_context as (root, root_fd):
            bindings = sm_fs_privacy.smmx_member_bindings(root_fd)
            for binding in bindings:
                relative = binding.relative
                if relative in seen:
                    raise ValueError(
                        f"duplicate current map identity: {relative}"
                    )
                indexed = index.get(relative)
                if indexed is None:
                    raise ValueError(
                        "privacy index is stale: current map is not indexed: "
                        f"{relative}"
                    )
                try:
                    data = sm_fs_privacy.read_corpus_member(
                        root_fd,
                        relative,
                        binding=binding,
                    )
                except sm_fs_privacy.SmFsPrivacyError as exc:
                    raise ValueError(
                        "privacy index is stale: map changed while being read: "
                        f"{relative}"
                    ) from exc
                digest = hashlib.sha256(data).hexdigest()
                if indexed.get("file_sha256") != digest:
                    raise ValueError(
                        "privacy index is stale: map contents changed: "
                        f"{relative}"
                    )
                seen.add(relative)
                # The caller consumes this digest directly into the ledger; it
                # must not reopen the source map between comparison/inclusion.
                yield relative, root / PurePosixPath(relative), digest, indexed
    except sm_fs_privacy.SmFsPrivacyError as exc:
        raise ValueError(str(exc)) from exc
    missing = set(index) - seen
    extra = seen - set(index)
    if missing or extra:
        detail = sorted(missing or extra, key=str.casefold)[0]
        raise ValueError(
            "privacy index is stale: map set changed "
            f"(first mismatched member: {detail})"
        )


def _expected_ledger_privacy_fields(relative, digest, indexed):
    parts = relative.split("/")
    return {
        "map_path": relative,
        "dest": "/".join(parts[:-1]),
        "title": parts[-1][:-5].strip(),
        "sha256": digest,
        "classification": indexed["classification"],
        "evidence": "content_addressed_index",
        "privacy_reasons": list(indexed.get("reasons", ())),
        "training_usable": "parse_error_type" not in indexed,
    }


def _verify_ledger_against_privacy_source(
    ledger, corpus_root, privacy_header, index
):
    """Rederive policy membership and privacy-bound row fields from live inputs."""

    verify_index_source(corpus_root, privacy_header, index)
    policy = ledger.get("training_privacy_policy")
    sealed_exclusions = ledger.get("owner_excluded_maps") or []
    if not (isinstance(sealed_exclusions, list)
            and all(isinstance(x, str) for x in sealed_exclusions)
            and sealed_exclusions == sorted(set(sealed_exclusions))):
        raise ValueError("training bundle owner exclusions are malformed")
    expected = {}
    observed_privacy = Counter()
    for relative, _path, digest, indexed in _iter_verified_source_maps(
        corpus_root, index
    ):
        if relative in sealed_exclusions:
            observed_privacy["owner_excluded"] += 1
            continue
        classification = indexed["classification"]
        observed_privacy[classification] += 1
        if not training_privacy_allows(classification, policy):
            continue
        fields = _expected_ledger_privacy_fields(relative, digest, indexed)
        if not fields["dest"]:
            raise ValueError(
                "policy-eligible privacy-index map has no destination: "
                f"{relative}"
            )
        expected[relative] = fields

    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("training bundle ledger rows are malformed")
    observed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("training bundle ledger row is malformed")
        relative = row.get("map_path")
        if not isinstance(relative, str) or relative in observed:
            raise ValueError("training bundle ledger map identity is invalid")
        observed[relative] = row
    if set(observed) != set(expected):
        raise ValueError(
            "training bundle policy-eligible privacy membership mismatch"
        )
    for relative, fields in expected.items():
        row = observed[relative]
        if any(row.get(key) != value for key, value in fields.items()):
            raise ValueError(
                "training bundle ledger privacy/source derivation mismatch: "
                f"{relative}"
            )
    expected_counts = dict(sorted(observed_privacy.items()))
    if (
        ledger.get("privacy_index", {}).get("observed_counts")
        != expected_counts
    ):
        raise ValueError("training bundle observed privacy counts mismatch")


def _validate_holdout_fraction(holdout_frac, *, allow_degenerate=False):
    if not isinstance(allow_degenerate, bool):
        raise ValueError("degenerate-holdout authorization must be boolean")
    if (
        isinstance(holdout_frac, bool)
        or not isinstance(holdout_frac, (int, float))
        or not math.isfinite(holdout_frac)
        or holdout_frac < 0
        or holdout_frac > 1
    ):
        raise ValueError("--holdout-frac must be a finite number in [0, 1]")
    if holdout_frac in (0, 1) and not allow_degenerate:
        raise ValueError(
            "holdout fractions 0 and 1 are diagnostic-only; pass "
            "--allow-degenerate-holdout to record that waiver"
        )
    if allow_degenerate and holdout_frac not in (0, 1):
        raise ValueError(
            "--allow-degenerate-holdout is valid only with a 0 or 1 "
            "diagnostic holdout fraction"
        )
    return float(holdout_frac)


def derive_split(
    rows,
    holdout_frac,
    split_seed,
    *,
    allow_degenerate=False,
):
    """Purely derive each row's split plus counts from immutable row fields."""

    holdout_frac = _validate_holdout_fraction(
        holdout_frac,
        allow_degenerate=allow_degenerate,
    )
    block_depth = 3
    cap = max(1, int(0.08 * len(rows)))

    def assign_blocks(deepen):
        assigned = []
        for row in rows:
            parts = row["dest"].split("/")
            depth = block_depth
            while True:
                block = "/".join(parts[:depth])
                if block not in deepen or depth >= len(parts):
                    break
                depth += 1
            assigned.append(block)
        return assigned

    deepen = set()
    while True:
        block_by_row = assign_blocks(deepen)
        weight = Counter(block_by_row)
        newly_oversized = {
            block for block, count in weight.items() if count > cap
        } - deepen
        if not newly_oversized:
            break
        deepen |= newly_oversized

    blocks = sorted(set(block_by_row))
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(blocks))
    held_blocks = set()
    held_rows = 0
    target = holdout_frac * len(rows)
    for index in order:
        if held_rows >= target:
            break
        block = blocks[index]
        held_blocks.add(block)
        held_rows += weight[block]

    preliminary = [
        "reserved" if block in held_blocks else "explore"
        for block in block_by_row
    ]
    reserved = {
        row["dest"]
        for row, split in zip(rows, preliminary)
        if split == "reserved"
    }
    reserved_ancestors = {
        "/".join(parts[:depth])
        for destination in reserved
        for parts in (destination.split("/"),)
        for depth in range(1, len(parts) + 1)
    }
    splits = []
    for row, split in zip(rows, preliminary):
        if split != "explore":
            splits.append(split)
            continue
        parts = row["dest"].split("/")
        explore_ancestors = {
            "/".join(parts[:depth]) for depth in range(1, len(parts) + 1)
        }
        if row["dest"] in reserved_ancestors or reserved.intersection(
            explore_ancestors
        ):
            splits.append("cross_lineage_excluded")
        else:
            splits.append("explore")

    counts = {
        split: splits.count(split)
        for split in ("explore", "reserved", "cross_lineage_excluded")
    }
    if not allow_degenerate and (
        counts["explore"] == 0 or counts["reserved"] == 0
    ):
        raise ValueError(
            "realized split has no explore or reserved rows; use more data "
            "(diagnostic bundles must explicitly request a 0 or 1 holdout)"
        )
    metadata = {
        "base_depth": block_depth,
        "cap": cap,
        "blocks": len(blocks),
        "held_blocks": len(held_blocks),
    }
    return splits, counts, metadata


def _validate_training_target_count(target_rows, *, allow_degenerate):
    if (
        isinstance(target_rows, bool)
        or not isinstance(target_rows, int)
        or target_rows < 0
    ):
        raise ValueError("training target-row count is invalid")
    if target_rows == 0 and not allow_degenerate:
        raise ValueError(
            "realized explore split has no usable training targets; use more "
            "data (diagnostic bundles must request a 0 or 1 holdout)"
        )


def _verify_split_derivation(ledger, parameters, manifest):
    rows = ledger.get("rows")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("training bundle ledger rows are malformed")
    expected_splits, expected_counts, _metadata = derive_split(
        rows,
        parameters.get("holdout_frac"),
        parameters.get("split_seed"),
        allow_degenerate=parameters.get("allow_degenerate_holdout", False),
    )
    observed_splits = [row.get("split") for row in rows]
    if canonical_json_bytes(observed_splits) != canonical_json_bytes(
        expected_splits
    ):
        raise ValueError("training bundle deterministic split mismatch")
    expected_count_bytes = canonical_json_bytes(expected_counts)
    if (
        canonical_json_bytes(ledger.get("counts")) != expected_count_bytes
        or canonical_json_bytes(manifest.get("counts"))
        != expected_count_bytes
    ):
        raise ValueError("training bundle deterministic split counts mismatch")


def exclude_cross_lineage(rows):
    """Move explore rows related to a reserved destination out of training."""

    reserved = {row["dest"] for row in rows if row["split"] == "reserved"}
    reserved_ancestors = {
        "/".join(parts[:depth])
        for destination in reserved
        for parts in (destination.split("/"),)
        for depth in range(1, len(parts) + 1)
    }
    excluded = 0
    for row in rows:
        if row["split"] != "explore":
            continue
        parts = row["dest"].split("/")
        explore_ancestors = {
            "/".join(parts[:depth]) for depth in range(1, len(parts) + 1)
        }
        if row["dest"] in reserved_ancestors or reserved.intersection(
            explore_ancestors
        ):
            row["split"] = "cross_lineage_excluded"
            excluded += 1
    return excluded


def _write_private(path, data):
    fd = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(fd, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path):
    directory_fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _rename_directory_noreplace(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
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
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to replace frozen output: {target}")
    raise RuntimeError("atomic no-replace bundle installation failed") from OSError(
        error_number, os.strerror(error_number)
    )


def _privacy_provenance(header):
    keys = (
        "schema",
        "policy_id",
        "index_sha256",
        "source_snapshot",
        "classifier_sha256",
        "parser_sha256",
        "counts",
    )
    return {key: header[key] for key in keys if key in header}


def build_bundle_manifest(
    *,
    ledger,
    ledger_bytes,
    targets_bytes,
    target_rows,
    fs_root,
    privacy_index,
    privacy_index_artifact,
    privacy_header,
    holdout_frac,
    split_seed,
    training_privacy_policy,
    allow_unresolved_private_targets,
    allow_degenerate_holdout,
    owner_excluded_maps=(),
):
    privacy_index = _absolute_path(privacy_index)
    if not _valid_artifact_record(privacy_index_artifact):
        raise ValueError("privacy-index artifact record is invalid")
    core = {
        "schema": BUNDLE_SCHEMA,
        "parameters": {
            "allow_unresolved_private_targets": bool(
                allow_unresolved_private_targets
            ),
            "allow_degenerate_holdout": bool(allow_degenerate_holdout),
            "owner_excluded_maps": sorted(owner_excluded_maps),
            "decay": DECAY,
            "fs_root": str(_absolute_path(fs_root)),
            "holdout_frac": holdout_frac,
            "process_expression": EXPR,
            "split_seed": split_seed,
            "training_privacy_policy": training_privacy_policy,
        },
        "inputs": {
            "privacy_index": {
                "artifact": dict(privacy_index_artifact),
                "path": str(privacy_index),
                "provenance": _privacy_provenance(privacy_header),
            }
        },
        "code": _code_records(),
        "e5_revision": E5_REVISION,
        "tree_snapshot": ledger["tree_snapshot"],
        "counts": ledger["counts"],
        "target_rows": target_rows,
        "outputs": {
            "ledger.json": _artifact_record(ledger_bytes),
            "lineage_fs_targets.tsv": _artifact_record(targets_bytes),
        },
    }
    manifest = dict(core)
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(core)
    ).hexdigest()
    return manifest


def _validate_private_directory(metadata):
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("training bundle is not a directory")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("training bundle directory is not mode 0700")
    if metadata.st_uid != os.geteuid():
        raise ValueError("training bundle directory is not owned by the current user")


def _read_bundle_files(bundle_dir):
    bundle = _absolute_path(bundle_dir)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(str(bundle), directory_flags)
    except OSError as exc:
        raise ValueError("training bundle directory is unavailable") from exc
    try:
        directory_before = os.fstat(directory_fd)
        _validate_private_directory(directory_before)
        try:
            path_before = bundle.lstat()
        except OSError as exc:
            raise ValueError(
                "training bundle directory path is unavailable"
            ) from exc
        if _descriptor_snapshot(path_before) != _descriptor_snapshot(
            directory_before
        ):
            raise ValueError(
                "training bundle directory path/descriptor identity mismatch"
            )

        names = os.listdir(directory_fd)
        if set(names) != BUNDLE_FILES or len(names) != len(BUNDLE_FILES):
            raise ValueError("training bundle file set mismatch")
        values = {}
        artifact_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        for name in sorted(names):
            try:
                descriptor = os.open(
                    name,
                    artifact_flags,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise ValueError(
                    f"training bundle artifact is unavailable: {name}"
                ) from exc
            try:
                values[name] = _read_validated_descriptor(
                    descriptor,
                    f"training bundle artifact {name}",
                    lambda name=name: os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    ),
                )
            finally:
                os.close(descriptor)

        if set(os.listdir(directory_fd)) != BUNDLE_FILES:
            raise ValueError("training bundle file set changed while being read")
        directory_after = os.fstat(directory_fd)
        try:
            path_after = bundle.lstat()
        except OSError as exc:
            raise ValueError(
                "training bundle directory path changed while being read"
            ) from exc
        before = _descriptor_snapshot(directory_before)
        if (
            _descriptor_snapshot(directory_after) != before
            or _descriptor_snapshot(path_after) != before
        ):
            raise ValueError("training bundle directory changed while being read")
        return values
    finally:
        os.close(directory_fd)


def verify_private_bundle(out_dir, privacy_index=None):
    """Verify bundle integrity and, when supplied, rederive it from live inputs."""

    values = _read_bundle_files(out_dir)
    try:
        manifest = json.loads(values["manifest.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training bundle manifest is invalid JSON") from exc
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("training bundle manifest schema mismatch")
    supplied_digest = manifest.get("manifest_sha256")
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    expected_digest = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    if supplied_digest != expected_digest:
        raise ValueError("training bundle manifest digest mismatch")
    if set(manifest.get("outputs", {})) != {
        "ledger.json",
        "lineage_fs_targets.tsv",
    }:
        raise ValueError("training bundle output inventory mismatch")
    for name, expected in manifest["outputs"].items():
        if expected != _artifact_record(values[name]):
            raise ValueError(f"training bundle artifact digest mismatch: {name}")
    if manifest.get("code") != _code_records():
        raise ValueError("training bundle code/dependency fingerprint mismatch")
    if manifest.get("e5_revision") != E5_REVISION:
        raise ValueError("training bundle E5 revision mismatch")

    try:
        ledger = json.loads(values["ledger.json"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training bundle ledger is invalid JSON") from exc
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, dict) or set(parameters) != {
        "allow_degenerate_holdout",
        "allow_unresolved_private_targets",
        "decay",
        "fs_root",
        "holdout_frac",
        "owner_excluded_maps",
        "process_expression",
        "split_seed",
        "training_privacy_policy",
    }:
        raise ValueError("training bundle parameter inventory mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"privacy_index"}:
        raise ValueError("training bundle input inventory mismatch")
    privacy_input = inputs["privacy_index"]
    if (
        not isinstance(privacy_input, dict)
        or set(privacy_input) != {"artifact", "path", "provenance"}
        or not _valid_artifact_record(privacy_input.get("artifact"))
        or not isinstance(privacy_input.get("path"), str)
        or not isinstance(privacy_input.get("provenance"), dict)
    ):
        raise ValueError("training bundle privacy input record is malformed")
    provenance = privacy_input.get("provenance", {})
    ledger_privacy = ledger.get("privacy_index", {})
    cross_checks = (
        (ledger.get("bundle_manifest_schema"), manifest.get("schema")),
        (ledger.get("code"), manifest.get("code")),
        (ledger.get("fs_root"), parameters.get("fs_root")),
        (ledger.get("tree_snapshot"), manifest.get("tree_snapshot")),
        (ledger.get("e5_revision"), manifest.get("e5_revision")),
        (ledger.get("counts"), manifest.get("counts")),
        (ledger.get("split_seed"), parameters.get("split_seed")),
        (ledger.get("holdout_frac"), parameters.get("holdout_frac")),
        (
            ledger.get("allow_degenerate_holdout"),
            parameters.get("allow_degenerate_holdout"),
        ),
        (
            ledger.get("training_privacy_policy"),
            parameters.get("training_privacy_policy"),
        ),
        (
            ledger.get("unresolved_private_targets_waived"),
            parameters.get("allow_unresolved_private_targets"),
        ),
        (
            ledger.get("owner_excluded_maps"),
            parameters.get("owner_excluded_maps"),
        ),
        (
            ledger_privacy.get("index_sha256"),
            provenance.get("index_sha256"),
        ),
        (ledger.get("privacy_source"), "content_addressed_index"),
        (ledger_privacy.get("artifact"), privacy_input.get("artifact")),
        (ledger_privacy.get("path"), privacy_input.get("path")),
    )
    if any(left != right for left, right in cross_checks):
        raise ValueError("training bundle manifest/ledger binding mismatch")
    if any(
        ledger_privacy.get(key) != value
        for key, value in provenance.items()
    ):
        raise ValueError("training bundle privacy provenance mismatch")
    _verify_split_derivation(ledger, parameters, manifest)
    catalog = ledger.get("catalog", [])
    if (
        catalog != sorted(set(catalog))
        or ledger.get("catalog_size") != len(catalog)
        or hashlib.sha256("\n".join(catalog).encode()).hexdigest()
        != ledger.get("tree_snapshot")
    ):
        raise ValueError("training bundle catalog snapshot mismatch")
    assert_catalog_complete(ledger.get("rows", []), ledger.get("catalog", []))

    if privacy_index is not None:
        _privacy_path, data, observed_artifact = _read_bound_bytes(
            privacy_index, "privacy index"
        )
        if observed_artifact != privacy_input["artifact"]:
            raise ValueError("training bundle privacy-index binding mismatch")
        privacy_header, index = _parse_privacy_index_bytes(data)
        if _privacy_provenance(privacy_header) != provenance:
            raise ValueError("training bundle privacy-index provenance mismatch")
        _verify_ledger_against_privacy_source(
            ledger,
            parameters.get("fs_root"),
            privacy_header,
            index,
        )

    try:
        target_lines = (
            values["lineage_fs_targets.tsv"].decode("utf-8").splitlines()
        )
    except UnicodeDecodeError as exc:
        raise ValueError("training bundle targets are not UTF-8") from exc
    data_lines = [line for line in target_lines if not line.startswith("#")]
    if len(data_lines) != manifest.get("target_rows"):
        raise ValueError("training bundle target-row count mismatch")
    _validate_training_target_count(
        manifest.get("target_rows"),
        allow_degenerate=parameters["allow_degenerate_holdout"],
    )
    header_values = {}
    for line in target_lines:
        if not line.startswith("# "):
            continue
        columns = line[2:].split("\t")
        if len(columns) == 2:
            header_values[columns[0]] = columns[1]
    expected_headers = {
        "e5_revision": E5_REVISION,
        "privacy_index_artifact_bytes": str(
            privacy_input["artifact"]["bytes"]
        ),
        "privacy_index_artifact_sha256": privacy_input["artifact"]["sha256"],
        "privacy_index_sha256": provenance.get("index_sha256"),
        "process_expression": parameters.get("process_expression"),
        "training_privacy_policy": parameters.get(
            "training_privacy_policy"
        ),
        "tree_snapshot": manifest.get("tree_snapshot"),
    }
    if any(
        header_values.get(key) != value
        for key, value in expected_headers.items()
    ):
        raise ValueError("training bundle target header binding mismatch")

    catalog_set = set(catalog)
    expected_rows = set()
    policy = parameters.get("training_privacy_policy")
    for row in ledger.get("rows", []):
        if (
            row.get("split") != "explore"
            or not row.get("training_usable")
            or not training_privacy_allows(row.get("classification"), policy)
        ):
            continue
        parts = row["dest"].split("/")
        for hop, ancestor_title in enumerate(reversed(parts), start=1):
            expected_rows.add(
                (
                    row["map_path"],
                    row["title"],
                    "/".join(parts[: len(parts) - hop + 1]),
                    ancestor_title,
                    str(hop),
                    f"{DECAY ** (hop - 1):.6f}",
                    row["classification"],
                )
            )
    observed_rows = set()
    for line in data_lines:
        columns = line.split("\t")
        if len(columns) != 7 or columns[2] not in catalog_set:
            raise ValueError("training target references a non-catalog ancestor")
        observed_rows.add(tuple(columns))
    if (
        len(observed_rows) != len(data_lines)
        or observed_rows != expected_rows
        or len(expected_rows) != manifest.get("target_rows")
    ):
        raise ValueError("training bundle target/ledger derivation mismatch")
    return manifest


def install_private_outputs(out_dir, payloads, *, privacy_index):
    """Crash-atomically install a complete local-only bundle without replacement."""

    if set(payloads) != BUNDLE_FILES:
        raise ValueError("private bundle payload set mismatch")
    out = Path(os.path.abspath(os.path.expanduser(str(out_dir))))
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to replace frozen output: {out}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.staging.", dir=str(parent))
    )
    os.chmod(temporary, 0o700)
    installed = False
    try:
        for name in sorted(payloads, key=lambda item: item == "manifest.json"):
            _write_private(temporary / name, payloads[name])
        _fsync_directory(temporary)
        # The complete external-input check belongs on the staging directory:
        # after the no-replace rename succeeds, no verification failure may
        # strand an installed bundle that this invocation already rejected.
        verify_private_bundle(temporary, privacy_index=privacy_index)
        _rename_directory_noreplace(temporary, out)
        installed = True
        _fsync_directory(parent)
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)
    return {name: out / name for name in payloads}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-root", default=DEFAULT_ROOT)
    ap.add_argument("--privacy-index", default=os.path.expanduser(
        "~/mu_data/sm_fs_privacy_index.jsonl"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/mu_data/sm_fs_v2"))
    ap.add_argument("--holdout-frac", type=float, default=0.40)
    ap.add_argument(
        "--allow-degenerate-holdout",
        action="store_true",
        help=(
            "permit a 0 or 1 holdout fraction for diagnostic fixtures only; "
            "the waiver is sealed into the bundle"
        ),
    )
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument(
        "--training-privacy-policy",
        choices=("public-only", "private-only"),
        default="public-only",
        help=(
            "rows allowed into training targets; public-only is the safe default, "
            "while private-only creates a separate local expert target"
        ),
    )
    ap.add_argument(
        "--allow-unresolved-private-targets",
        action="store_true",
        help=(
            "owner waiver for unresolved references below explicit private markers; "
            "recorded in the ledger and normally forbidden for public-only output"
        ),
    )
    ap.add_argument(
        "--exclude-map",
        action="append",
        default=[],
        metavar="RELPATH",
        help=(
            "owner exclusion: drop this indexed map from the corpus entirely (repeatable). "
            "Each path must exist in the privacy index; exclusions are sealed into the "
            "ledger and manifest, and an excluded map's unresolved private-target refs no "
            "longer trip the public-only gate (the referring map itself is gone)"
        ),
    )
    a = ap.parse_args(argv)
    owner_exclusions = sorted(set(a.exclude_map))
    a.holdout_frac = _validate_holdout_fraction(
        a.holdout_frac,
        allow_degenerate=a.allow_degenerate_holdout,
    )

    fs_root = _absolute_path(a.fs_root)
    privacy_path = _absolute_path(a.privacy_index)
    if not privacy_path.exists():
        raise ValueError(
            "training requires a content-addressed privacy index; "
            "build sm_fs_privacy.py first"
        )
    (
        privacy_path,
        privacy_index_artifact,
        privacy_header,
        index,
    ) = _load_bound_privacy_index(privacy_path)
    verify_index_source(fs_root, privacy_header, index)
    print(
        f"privacy index: {len(index)} source-verified records "
        f"(policy {privacy_header['policy_id']}, "
        f"digest {privacy_header['index_sha256'][:16]})"
    )
    missing_exclusions = [rel for rel in owner_exclusions if rel not in index]
    if missing_exclusions:
        raise ValueError(
            f"--exclude-map path(s) not present in the privacy index: "
            f"{missing_exclusions[:3]}"
        )
    unresolved = privacy_header.get("counts", {}).get(
        "unresolved_private_target_refs", 0
    )
    excluded_refs = sum(
        len(index[rel].get("unresolved_private_targets") or [])
        for rel in owner_exclusions
    )
    effective_unresolved = max(0, unresolved - excluded_refs)
    if owner_exclusions:
        print(
            f"owner exclusions: {len(owner_exclusions)} map(s) dropped "
            f"({excluded_refs} unresolved private-target ref(s) retired with them)"
        )
    if (
        a.training_privacy_policy == "public-only"
        and effective_unresolved
        and not a.allow_unresolved_private_targets
    ):
        raise ValueError(
            f"privacy index has {effective_unresolved} unresolved private target "
            "reference(s); resolve them or record an explicit owner waiver"
        )

    rows = []
    observed_privacy = Counter()
    excluded_set = set(owner_exclusions)
    for relative, _path, digest, indexed in _iter_verified_source_maps(
        fs_root, index
    ):
        if relative in excluded_set:
            observed_privacy["owner_excluded"] += 1
            continue
        classification = indexed["classification"]
        observed_privacy[classification] += 1
        if not training_privacy_allows(
            classification, a.training_privacy_policy
        ):
            continue
        row = _expected_ledger_privacy_fields(relative, digest, indexed)
        if not row["dest"]:
            raise ValueError(
                "policy-eligible privacy-index map has no destination: "
                f"{relative}"
            )
        # This is deliberately adjacent to the source/index digest comparison
        # in the iterator: the ledger stores that same digest, never a reread.
        rows.append(row)
    expected_policy_members = {
        relative
        for relative, record in index.items()
        if relative not in excluded_set
        and training_privacy_allows(
            record["classification"], a.training_privacy_policy
        )
    }
    included_members = {row["map_path"] for row in rows}
    if included_members != expected_policy_members:
        raise ValueError(
            "policy-eligible privacy-index map omitted from training bundle"
        )
    # A policy view is a lineage closure, not "all nonsensitive directories".
    # Include every accepted destination and all of its ancestors. This keeps
    # target identities representable while excluding unrelated/unknown branches.
    catalog = catalog_for_rows(rows)
    assert_catalog_complete(rows, catalog)
    tree_snapshot = hashlib.sha256("\n".join(catalog).encode()).hexdigest()
    print(f"rows ({a.training_privacy_policy}): {len(rows)}; "
          f"policy-filtered catalog (exact-path identity): {len(catalog)}; "
          f"tree snapshot {tree_snapshot[:16]}")

    # SUBTREE-BLOCK split: reserve whole depth-K subtrees, so no explore
    # destination is inside any reserved subtree (and vice versa). Ancestors
    # ABOVE block roots are necessarily shared — the documented
    # transductive-at-ancestor limitation, not a leak within blocks.
    splits, n, split_metadata = derive_split(
        rows,
        a.holdout_frac,
        a.split_seed,
        allow_degenerate=a.allow_degenerate_holdout,
    )
    for row, split in zip(rows, splits):
        row["split"] = split
    excluded = n["cross_lineage_excluded"]
    print(f"split (adaptive subtree blocks, base depth "
          f"{split_metadata['base_depth']}, cap {split_metadata['cap']} rows, "
          f"seed {a.split_seed}): "
          f"explore {n['explore']}, "
          f"reserved {n['reserved']} across "
          f"{split_metadata['held_blocks']}/{split_metadata['blocks']} blocks "
          f"(never score; "
          f"whole subtrees); {excluded} lineage-related explore rows excluded")

    ledger = {
        "schema": "sm_fs_corpus.v2",
        "bundle_manifest_schema": BUNDLE_SCHEMA,
        "fs_root": str(fs_root),
        "tree_snapshot": tree_snapshot,
        "e5_revision": E5_REVISION, "split_seed": a.split_seed,
        "holdout_frac": a.holdout_frac, "counts": n, "catalog_size": len(catalog),
        "allow_degenerate_holdout": bool(a.allow_degenerate_holdout),
        "code": _code_records(),
        "privacy_source": "content_addressed_index",
        "privacy_index": {
            **_privacy_provenance(privacy_header),
            "artifact": dict(privacy_index_artifact),
            "observed_counts": dict(sorted(observed_privacy.items())),
            "path": str(privacy_path),
        },
        "training_privacy_policy": a.training_privacy_policy,
        "unresolved_private_targets_waived": bool(
            a.allow_unresolved_private_targets
        ),
        "owner_excluded_maps": owner_exclusions,
        "limitations": ["ancestor components above destination level shared across splits "
                        "(transductive at ancestor level)",
                        "public classification is an owner risk-policy decision, not independent "
                        "public-availability certification",
                        "privacy-index hashes provide content integrity and reproducibility, "
                        "not external authentication"]
        + (
            [
                "degenerate holdout explicitly authorized for diagnostics; "
                "this bundle is not an evaluation reserve"
            ]
            if a.allow_degenerate_holdout
            else []
        ),
        "rows": rows, "catalog": catalog,
    }

    # lineage(fs) training targets — EXPLORE partition only
    target_buffer = io.StringIO()
    target_buffer.write(
        f"# process_expression\t{EXPR}\n"
        f"# tree_snapshot\t{tree_snapshot}\n"
        f"# e5_revision\t{E5_REVISION}\n"
        f"# training_privacy_policy\t{a.training_privacy_policy}\n"
        f"# privacy_index_artifact_bytes\t"
        f"{privacy_index_artifact['bytes']}\n"
        f"# privacy_index_artifact_sha256\t"
        f"{privacy_index_artifact['sha256']}\n"
        f"# privacy_index_sha256\t"
        f"{privacy_header['index_sha256']}\n"
        "# map_path\tmap_title\tancestor_path\tancestor_title\t"
        "hop\ttarget\tclassification\n"
    )
    kept = 0
    catalog_set = set(catalog)
    for r in rows:
        if (
            r["split"] != "explore"
            or not r["training_usable"]
            or not training_privacy_allows(
                r["classification"], a.training_privacy_policy
            )
        ):
            continue
        parts = r["dest"].split("/")
        for hop, ancestor_title in enumerate(reversed(parts), start=1):
            ancestor_path = "/".join(parts[: len(parts) - hop + 1])
            if ancestor_path not in catalog_set:
                raise ValueError(
                    "training target references a non-catalog ancestor: "
                    f"{ancestor_path}"
                )
            target_buffer.write(
                f"{r['map_path']}\t{r['title']}\t{ancestor_path}\t"
                f"{ancestor_title}\t{hop}\t{DECAY ** (hop - 1):.6f}\t"
                f"{r['classification']}\n"
            )
            kept += 1
    _validate_training_target_count(
        kept,
        allow_degenerate=a.allow_degenerate_holdout,
    )
    ledger_bytes = (
        json.dumps(ledger, ensure_ascii=False, indent=0, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    targets_bytes = target_buffer.getvalue().encode("utf-8")
    manifest = build_bundle_manifest(
        ledger=ledger,
        ledger_bytes=ledger_bytes,
        targets_bytes=targets_bytes,
        target_rows=kept,
        fs_root=fs_root,
        privacy_index=privacy_path,
        privacy_index_artifact=privacy_index_artifact,
        privacy_header=privacy_header,
        holdout_frac=a.holdout_frac,
        split_seed=a.split_seed,
        training_privacy_policy=a.training_privacy_policy,
        allow_unresolved_private_targets=a.allow_unresolved_private_targets,
        allow_degenerate_holdout=a.allow_degenerate_holdout,
        owner_excluded_maps=owner_exclusions,
    )
    payloads = {
        "ledger.json": ledger_bytes,
        "lineage_fs_targets.tsv": targets_bytes,
        "manifest.json": canonical_json_bytes(manifest),
    }
    installed = install_private_outputs(
        a.out_dir,
        payloads,
        privacy_index=privacy_path,
    )
    lp = installed["ledger.json"]
    tp = installed["lineage_fs_targets.tsv"]
    mp = installed["manifest.json"]
    print(
        f"ledger -> {lp} (sha {sha_file(lp)[:16]}) — per-row membership, "
        "independently reproducible; mode 0600, no-replace"
    )
    print(
        f"training targets -> {tp} ({kept} rows, expr `{EXPR}`, "
        f"explore + {a.training_privacy_policy} + readable only)"
    )
    print(
        f"binding manifest -> {mp} "
        f"(sha {manifest['manifest_sha256'][:16]}); "
        "bundle installed crash-atomically, mode 0700/0600, no-replace"
    )


if __name__ == "__main__":
    main()
