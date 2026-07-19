#!/usr/bin/env python3
"""Declare explicit local-only inputs for the Pearltrees diffusion compiler.

This utility is intentionally only a declaration planner.  It never searches for
inputs, chooses a newest file, opens source contents, or compiles a graph.  Its
only output is a canonical ``pearltrees-diffusion-source-spec-v1`` in a newly
installed local-only directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import errno
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SOURCE_SPEC_SCHEMA = "pearltrees-diffusion-source-spec-v1"
SPEC_FILENAME = "source_spec.json"
LOCAL_ONLY_MARKER = "LOCAL_ONLY_DO_NOT_PUBLISH"
LOCAL_ONLY_MARKER_PAYLOAD = (
    b"LOCAL ONLY - DO NOT PUBLISH SOURCE PATH DECLARATIONS\n"
)
SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")
ACCOUNT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
WILDCARD_CHARS = frozenset("*?[]{}")


class DeclarationError(ValueError):
    """Fail-closed source declaration or installation error."""


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DeclarationError("source declaration is not canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def _duplicate_checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DeclarationError("source declaration contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> object:
    del token
    raise DeclarationError("source declaration contains a non-finite JSON number")


def _strict_canonical_json(data: bytes) -> dict[str, object]:
    """Decode one exact canonical JSON object, rejecting parser extensions."""

    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonfinite_json,
        )
    except DeclarationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise DeclarationError("source declaration is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DeclarationError("source spec must be an object")
    if _canonical_json(value) != data:
        raise DeclarationError("source declaration bytes are not canonical JSON")
    return value


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_git_worktree_marker(path: Path) -> bool:
    """Recognize a real Git marker, not an unrelated empty ``.git`` path."""

    try:
        if path.is_file():
            with open(path, "rb") as stream:
                return stream.read(64).lstrip().startswith(b"gitdir:")
        return path.is_dir() and (path / "HEAD").is_file()
    except OSError:
        # An unreadable candidate marker is treated conservatively as a marker.
        return True


def _absolute_lexical(path: str | os.PathLike[str]) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise DeclarationError("declared path is invalid") from exc


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject symlinks in every existing component without reading contents."""

    if not path.is_absolute():
        raise DeclarationError(f"{label} path is not absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DeclarationError(f"{label} path could not be validated") from exc
        if stat.S_ISLNK(mode):
            raise DeclarationError(f"{label} path cannot contain a symlink")


def _resolved_declared_input(path_value: str, *, directory: bool) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise DeclarationError("declared source path must be nonempty text")
    if any(character in path_value for character in WILDCARD_CHARS):
        raise DeclarationError("wildcard-looking source paths are forbidden")
    path = _absolute_lexical(path_value)
    _reject_symlink_components(path, "declared source")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise DeclarationError("declared source is unavailable") from exc
    expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise DeclarationError(f"declared source must be a non-symlink {kind}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise DeclarationError("declared source could not be resolved") from exc


def _path_identity(path: Path) -> tuple[int, int]:
    try:
        record = path.stat()
    except OSError as exc:
        raise DeclarationError("declared source identity could not be validated") from exc
    return record.st_dev, record.st_ino


def _source_id(value: str) -> str:
    if not isinstance(value, str) or SOURCE_ID_RE.fullmatch(value) is None:
        raise DeclarationError("source_id must use stable ASCII identifier syntax")
    return value


def _rdf_account(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeclarationError("RDF account must be explicit nonempty text")
    if value != value.strip() or ACCOUNT_RE.fullmatch(value) is None:
        raise DeclarationError("RDF account must use canonical account syntax")
    if value.casefold() == "grous":
        raise DeclarationError("team RDF account typo is forbidden; use groups")
    if value.casefold() == "groups" and value != "groups":
        raise DeclarationError("team RDF account must be the literal groups")
    return value


def _source_entry(kind: str, values: tuple[str, ...]) -> dict[str, str]:
    if kind == "rdf":
        if len(values) != 3:
            raise DeclarationError("RDF declaration fields mismatch")
        source_id, account, path_value = values
        return {
            "account": _rdf_account(account),
            "kind": kind,
            "path": str(_resolved_declared_input(path_value, directory=False)),
            "source_id": _source_id(source_id),
        }
    if len(values) != 2:
        raise DeclarationError("source declaration fields mismatch")
    source_id, path_value = values
    return {
        "kind": kind,
        "path": str(
            _resolved_declared_input(
                path_value,
                directory=(kind == "api_json_dir"),
            )
        ),
        "source_id": _source_id(source_id),
    }


def build_source_spec(
    *,
    snapshot_label: str,
    rdf: tuple[tuple[str, str, str], ...] = (),
    api_json_dir: tuple[tuple[str, str], ...] = (),
    api_sqlite: tuple[tuple[str, str], ...] = (),
    path_jsonl: tuple[tuple[str, str], ...] = (),
    legacy_dag: str | None = None,
) -> dict[str, object]:
    """Build a canonical source specification without reading source contents."""

    if (
        not isinstance(snapshot_label, str)
        or not snapshot_label.strip()
        or snapshot_label != snapshot_label.strip()
    ):
        raise DeclarationError("snapshot_label must be canonical nonempty text")
    declarations: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
        [("rdf", tuple(item)) for item in rdf]
        + [("api_json_dir", tuple(item)) for item in api_json_dir]
        + [("api_sqlite", tuple(item)) for item in api_sqlite]
        + [("path_jsonl", tuple(item)) for item in path_jsonl]
    )
    if not declarations:
        raise DeclarationError("at least one explicit source is required")
    sources = [_source_entry(kind, values) for kind, values in declarations]
    source_ids = [entry["source_id"] for entry in sources]
    if len(set(source_ids)) != len(source_ids):
        raise DeclarationError("source_id values must be globally unique")
    source_identities = [_path_identity(Path(entry["path"])) for entry in sources]
    if len(set(source_identities)) != len(source_identities):
        raise DeclarationError("authoritative source paths must not alias")
    sources.sort(key=lambda entry: entry["source_id"])
    spec: dict[str, object] = {
        "schema": SOURCE_SPEC_SCHEMA,
        "snapshot_label": snapshot_label,
        "sources": sources,
    }
    if legacy_dag is not None:
        legacy_path = _resolved_declared_input(legacy_dag, directory=False)
        if _path_identity(legacy_path) in set(source_identities):
            raise DeclarationError("legacy parity input cannot alias an authoritative source")
        spec["legacy_check"] = {"dag_path": str(legacy_path)}
    return spec


def _validate_canonical_spec(spec: dict[str, object]) -> dict[str, object]:
    if not isinstance(spec, dict):
        raise DeclarationError("source spec must be an object")
    allowed = {"schema", "snapshot_label", "sources", "legacy_check"}
    if set(spec) - allowed or not {"schema", "snapshot_label", "sources"}.issubset(
        spec
    ):
        raise DeclarationError("source spec fields mismatch")
    if spec.get("schema") != SOURCE_SPEC_SCHEMA or not isinstance(
        spec.get("sources"), list
    ):
        raise DeclarationError("source spec schema or sources mismatch")
    grouped: dict[str, list[tuple[str, ...]]] = {
        "rdf": [],
        "api_json_dir": [],
        "api_sqlite": [],
        "path_jsonl": [],
    }
    for entry in spec["sources"]:
        if not isinstance(entry, dict) or entry.get("kind") not in grouped:
            raise DeclarationError("source entry is unsupported")
        kind = entry["kind"]
        expected = (
            {"account", "kind", "path", "source_id"}
            if kind == "rdf"
            else {"kind", "path", "source_id"}
        )
        if set(entry) != expected:
            raise DeclarationError("source entry fields mismatch")
        if kind == "rdf":
            grouped[kind].append(
                (entry["source_id"], entry["account"], entry["path"])
            )
        else:
            grouped[kind].append((entry["source_id"], entry["path"]))
    legacy = spec.get("legacy_check")
    if legacy is not None and (
        not isinstance(legacy, dict) or set(legacy) != {"dag_path"}
    ):
        raise DeclarationError("legacy_check fields mismatch")
    rebuilt = build_source_spec(
        snapshot_label=spec["snapshot_label"],
        rdf=tuple(grouped["rdf"]),
        api_json_dir=tuple(grouped["api_json_dir"]),
        api_sqlite=tuple(grouped["api_sqlite"]),
        path_jsonl=tuple(grouped["path_jsonl"]),
        legacy_dag=legacy["dag_path"] if legacy is not None else None,
    )
    if rebuilt != spec:
        raise DeclarationError("source spec is not in canonical declaration order")
    return rebuilt


def _output_parent_identity(parent: Path) -> tuple[int, int]:
    """Return the identity of one resolved, non-symlink output parent."""

    _reject_symlink_components(parent, "output parent")
    try:
        record = parent.lstat()
        resolved = parent.resolve(strict=True)
    except OSError as exc:
        raise DeclarationError("output parent identity could not be validated") from exc
    if not stat.S_ISDIR(record.st_mode) or resolved != parent:
        raise DeclarationError("output parent must remain a resolved directory")
    return record.st_dev, record.st_ino


def _recheck_output_parent(parent: Path, expected: tuple[int, int]) -> None:
    if _output_parent_identity(parent) != expected:
        raise DeclarationError("output parent identity changed during installation")


def _validate_output_paths(
    output_dir: str | os.PathLike[str],
    local_root: str | os.PathLike[str],
    spec: dict[str, object],
) -> tuple[Path, Path, tuple[int, int]]:
    root_input = _absolute_lexical(local_root)
    output_input = _absolute_lexical(output_dir)
    _reject_symlink_components(root_input, "local root")
    _reject_symlink_components(output_input.parent, "output parent")
    if output_input.is_symlink():
        raise DeclarationError("output directory cannot be a symlink")
    try:
        root_mode = root_input.lstat().st_mode
        parent_mode = output_input.parent.lstat().st_mode
    except OSError as exc:
        raise DeclarationError("output root or parent is unavailable") from exc
    if not stat.S_ISDIR(root_mode) or not stat.S_ISDIR(parent_mode):
        raise DeclarationError("local root and output parent must be directories")
    local_root_resolved = root_input.resolve(strict=True)
    output_parent = output_input.parent.resolve(strict=True)
    output_parent_identity = _output_parent_identity(output_parent)
    output_resolved = output_parent / output_input.name
    if output_resolved.exists() or output_resolved.is_symlink():
        raise DeclarationError("output directory already exists")
    if (
        output_resolved == local_root_resolved
        or not _path_is_within(output_resolved, local_root_resolved)
    ):
        raise DeclarationError("output directory must be below the explicit local root")
    repo = REPO_ROOT.resolve()
    if _path_is_within(output_resolved, repo) or any(
        _is_git_worktree_marker(ancestor / ".git")
        for ancestor in (output_parent, *output_parent.parents)
    ):
        raise DeclarationError("output directory cannot be inside a Git worktree")

    input_paths = [Path(entry["path"]) for entry in spec["sources"]]
    legacy = spec.get("legacy_check")
    if isinstance(legacy, dict):
        input_paths.append(Path(legacy["dag_path"]))
    for input_path in input_paths:
        if (
            input_path == output_resolved
            or _path_is_within(input_path, output_resolved)
            or _path_is_within(output_resolved, input_path)
        ):
            raise DeclarationError("declared input and output paths cannot overlap")
        try:
            input_stat = input_path.stat()
            root_stat = local_root_resolved.stat()
        except OSError as exc:
            raise DeclarationError("input/output alias check failed") from exc
        if (input_stat.st_dev, input_stat.st_ino) == (
            root_stat.st_dev,
            root_stat.st_ino,
        ):
            raise DeclarationError("local root cannot alias an input")
    return output_resolved, local_root_resolved, output_parent_identity


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise DeclarationError("atomic no-replace rename is unavailable")
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
        raise DeclarationError("output appeared during atomic installation")
    raise DeclarationError("atomic no-replace installation failed") from OSError(
        error_number, os.strerror(error_number)
    )


def _write_private(path: Path, data: bytes) -> None:
    try:
        with open(path, "xb") as stream:
            os.chmod(path, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise DeclarationError("local-only declaration could not be written") from exc


def verify_installed_source_spec(
    source_spec: str | os.PathLike[str],
) -> dict[str, object]:
    """Verify and return one complete, private source-declaration bundle.

    Verification deliberately covers both the semantic declaration and its local-only
    installation envelope.  A caller cannot bypass explicit account/path/type checks by
    hand-writing a merely schema-shaped JSON file.
    """

    source_path = _absolute_lexical(source_spec)
    if source_path.name != SPEC_FILENAME:
        raise DeclarationError(f"installed source spec must be named {SPEC_FILENAME}")
    _reject_symlink_components(source_path, "installed source declaration")
    directory = source_path.parent
    marker_path = directory / LOCAL_ONLY_MARKER
    try:
        directory_record = directory.lstat()
        source_record = source_path.lstat()
        marker_record = marker_path.lstat()
        installed_names = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise DeclarationError("installed source declaration cannot be inspected") from exc

    if not stat.S_ISDIR(directory_record.st_mode):
        raise DeclarationError("source declaration bundle must be a directory")
    if stat.S_IMODE(directory_record.st_mode) != 0o700:
        raise DeclarationError("source declaration directory mode must be 0700")
    if installed_names != {SPEC_FILENAME, LOCAL_ONLY_MARKER}:
        raise DeclarationError("source declaration bundle file set mismatch")
    if not stat.S_ISREG(source_record.st_mode) or not stat.S_ISREG(
        marker_record.st_mode
    ):
        raise DeclarationError("source declaration bundle files must be regular files")
    if stat.S_IMODE(source_record.st_mode) != 0o600 or stat.S_IMODE(
        marker_record.st_mode
    ) != 0o600:
        raise DeclarationError("source declaration bundle file modes must be 0600")

    directory_resolved = directory.resolve(strict=True)
    repo = REPO_ROOT.resolve()
    if _path_is_within(directory_resolved, repo) or any(
        _is_git_worktree_marker(ancestor / ".git")
        for ancestor in (directory_resolved, *directory_resolved.parents)
    ):
        raise DeclarationError("source declaration bundle cannot be inside a Git worktree")

    try:
        payload = source_path.read_bytes()
        marker_payload = marker_path.read_bytes()
    except OSError as exc:
        raise DeclarationError("installed source declaration cannot be read") from exc
    if marker_payload != LOCAL_ONLY_MARKER_PAYLOAD:
        raise DeclarationError("source declaration local-only marker mismatch")
    return _validate_canonical_spec(_strict_canonical_json(payload))


def install_source_spec(
    spec: dict[str, object],
    *,
    output_dir: str | os.PathLike[str],
    local_root: str | os.PathLike[str],
) -> dict[str, object]:
    """Atomically install a canonical source spec in a new private directory."""

    spec = _validate_canonical_spec(spec)
    output, _root, output_parent_identity = _validate_output_paths(
        output_dir, local_root, spec
    )
    payload = _canonical_json(spec)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    os.chmod(temporary, 0o700)
    installed = False
    try:
        _write_private(temporary / SPEC_FILENAME, payload)
        _write_private(
            temporary / LOCAL_ONLY_MARKER,
            LOCAL_ONLY_MARKER_PAYLOAD,
        )
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _recheck_output_parent(output.parent, output_parent_identity)
        _rename_directory_noreplace(temporary, output)
        installed = True
        _recheck_output_parent(output.parent, output_parent_identity)
        parent_open_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        parent_fd = os.open(
            output.parent,
            parent_open_flags,
        )
        try:
            parent_record = os.fstat(parent_fd)
            if (parent_record.st_dev, parent_record.st_ino) != output_parent_identity:
                raise DeclarationError(
                    "output parent identity changed before directory synchronization"
                )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        raise DeclarationError("local-only declaration installation failed") from exc
    finally:
        if not installed and temporary.exists():
            shutil.rmtree(temporary)

    installed_spec = verify_installed_source_spec(output / SPEC_FILENAME)
    if installed_spec != spec:
        raise DeclarationError("installed source declaration verification failed")
    counts = Counter(entry["kind"] for entry in installed_spec["sources"])
    return {
        "legacy_check_declared": "legacy_check" in installed_spec,
        "local_only": True,
        "source_count": len(installed_spec["sources"]),
        "source_kind_counts": dict(sorted(counts.items())),
        "spec_written": True,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-label", required=True)
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-only", action="store_true", required=True)
    parser.add_argument(
        "--rdf",
        action="append",
        nargs=3,
        default=[],
        metavar=("SOURCE_ID", "ACCOUNT", "PATH"),
    )
    parser.add_argument(
        "--api-json-dir",
        action="append",
        nargs=2,
        default=[],
        metavar=("SOURCE_ID", "PATH"),
    )
    parser.add_argument(
        "--api-sqlite",
        action="append",
        nargs=2,
        default=[],
        metavar=("SOURCE_ID", "PATH"),
    )
    parser.add_argument(
        "--path-jsonl",
        action="append",
        nargs=2,
        default=[],
        metavar=("SOURCE_ID", "PATH"),
    )
    parser.add_argument("--legacy-dag")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        spec = build_source_spec(
            snapshot_label=args.snapshot_label,
            rdf=tuple(tuple(item) for item in args.rdf),
            api_json_dir=tuple(tuple(item) for item in args.api_json_dir),
            api_sqlite=tuple(tuple(item) for item in args.api_sqlite),
            path_jsonl=tuple(tuple(item) for item in args.path_jsonl),
            legacy_dag=args.legacy_dag,
        )
        summary = install_source_spec(
            spec,
            output_dir=args.output_dir,
            local_root=args.local_root,
        )
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception:
        print(
            json.dumps({"error": "source declaration failed closed"}, sort_keys=True),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
