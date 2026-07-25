#!/usr/bin/env python3
"""Lightweight privacy triage for the SimpleMind filesystem filing corpus.

This is deliberately separate from ``privacy.py``'s conservative public-dataset
scrubber.  The corpus owner permits local use of all three states here; this
index exists so downstream tools can distinguish:

``public``
    No private signal under the owner's stated risk policy, or a topical
    ``Private`` name backed by a normal Pearltrees root link. This is not an
    independent public-availability certification.
``private``
    A contextual filesystem marker, or a map targeted from an exact in-map
    ``private`` container.
``unknown``
    Evidence is ambiguous or the map could not be inspected.

Rules resolve the easy cases.  The production index builder is rule-only until
a versioned, verified passing benchmark lock exists.  Internal benchmark helpers
can exercise a cheap model on synthetic ``marker_vs_topic`` cases, but no model
result can currently enter a corpus index or change a classification.

Map-level propagation is deliberately narrow: only cloud-map references below
an explicit ``private`` topic mark their target maps. It does not recursively
follow every ordinary link inside a newly private target map.

Example:

    python3 sm_fs_privacy.py \
      --root /mnt/c/Users/johnc/Dropbox/root \
      --out ~/mu_data/sm_fs_privacy_index.jsonl

The detailed index is local data.  It records paths and titles and must not be
committed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET
import zipfile

from parse_smmx import label


SCHEMA = "unifyweaver.sm-fs-privacy-index.v1"
POLICY_ID = "sm-fs-owner-triage-v1"
PARSER_PATH = Path(__file__).with_name("parse_smmx.py")
MODEL_REVIEW_DISABLED_REASON = (
    "corpus model review is disabled until a versioned, verified passing "
    "benchmark lock is implemented"
)
CLASSIFICATIONS = frozenset(("public", "private", "unknown"))
INTERPRETATIONS = frozenset(("access_control", "topical", "uncertain"))
MODEL_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "qid": {"type": "string"},
            "interpretation": {
                "type": "string",
                "enum": sorted(INTERPRETATIONS),
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 300,
            },
        },
        "required": ["qid", "interpretation", "reason"],
        "additionalProperties": False,
    },
}
PRIVATE_WORD = re.compile(r"(?i)\bprivate\b")
PRIVATE_OR_PRIVACY_WORD = re.compile(r"(?i)\b(?:private|privacy)\b")
# Access-copy conventions observed in this corpus: ``Private: ...``,
# ``Private_...``, and ``Private - ...``. A lexical subject such as
# ``private-key`` must remain marker-vs-topic evidence, not an automatic seed.
ACCESS_PREFIX = re.compile(r"(?i)^\s*\*?private\*?\s*(?::|_|-\s+)")
PEARLTREES_ID = re.compile(r"(?i)^id\d+$")
PEARLTREES_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
HEADER_FIELDS = frozenset(
    (
        "record_type",
        "schema",
        "policy_id",
        "corpus_root",
        "source_snapshot",
        "classifier_sha256",
        "parser_sha256",
        "model",
        "model_review_authorization",
        "counts",
        "index_sha256",
    )
)
MAP_FIELDS = frozenset(
    (
        "record_type",
        "relative_path",
        "file_sha256",
        "root_title",
        "root_urls",
        "classification",
        "reasons",
        "model_review",
        "unresolved_private_targets",
    )
)
MAP_OPTIONAL_FIELDS = frozenset(("parse_error_type",))
COUNT_FIELDS = frozenset(
    (
        "public",
        "private",
        "unknown",
        "observed",
        "model_access_control",
        "model_topical",
        "model_uncertain",
        "map_parse_errors",
        "unresolved_private_target_refs",
    )
)


class SmFsPrivacyError(ValueError):
    """The filesystem privacy index is malformed or cannot be built safely."""


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


def _read_all_from_fd(fd: int) -> bytes:
    blocks = []
    while True:
        try:
            block = os.read(fd, 1024 * 1024)
        except InterruptedError:
            continue
        if not block:
            return b"".join(blocks)
        blocks.append(block)


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


@contextmanager
def open_corpus_root(corpus_root: Path):
    """Retain a no-follow descriptor for an absolute corpus root.

    Opening every path component relative to its already-open parent prevents
    a concurrent root or ancestor swap from redirecting a private-data scan.
    Callers must keep this context open for enumeration and all member reads.
    """

    corpus_root = Path(os.path.abspath(os.path.normpath(str(corpus_root))))
    try:
        descriptor = os.open(os.path.sep, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise SmFsPrivacyError("cannot open filesystem root safely") from exc
    try:
        for component in corpus_root.parts[1:]:
            try:
                child = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise SmFsPrivacyError(
                    f"cannot open corpus root without following links: "
                    f"{corpus_root}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise SmFsPrivacyError(
                f"corpus root is not a directory: {corpus_root}"
            )
        yield corpus_root, descriptor
    finally:
        os.close(descriptor)


def _open_directory_component(parent_fd: int, name: str, relative: str) -> int:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise SmFsPrivacyError(
                f"refusing symlink corpus directory: {relative}"
            )
        if not stat.S_ISDIR(before.st_mode):
            raise SmFsPrivacyError(
                f"corpus member changed type: {relative}"
            )
        descriptor = os.open(
            name,
            _DIRECTORY_OPEN_FLAGS,
            dir_fd=parent_fd,
        )
    except SmFsPrivacyError:
        raise
    except OSError as exc:
        raise SmFsPrivacyError(
            f"cannot safely open corpus directory: {relative}"
        ) from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise SmFsPrivacyError(
            f"corpus directory changed while opening: {relative}"
        )
    return descriptor


@dataclass(frozen=True)
class CorpusMemberBinding:
    """One member identity plus every directory identity used to reach it."""

    relative: str
    directory_identities: Tuple[Tuple[int, int], ...]
    file_identity: Tuple[int, int]


def _identity(metadata) -> Tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def smmx_member_bindings(root_fd: int) -> List[CorpusMemberBinding]:
    """Enumerate and identity-bind SMMX members below a retained root."""

    bindings: List[CorpusMemberBinding] = []
    root_metadata = os.fstat(root_fd)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise SmFsPrivacyError("retained corpus root changed type")

    def walk(
        directory_fd: int,
        prefix: Tuple[str, ...],
        directory_identities: Tuple[Tuple[int, int], ...],
    ) -> None:
        try:
            names = os.listdir(directory_fd)
        except OSError as exc:
            relative = "/".join(prefix) or "."
            raise SmFsPrivacyError(
                f"cannot enumerate corpus directory: {relative}"
            ) from exc
        for name in sorted(names, key=lambda value: (value.casefold(), value)):
            relative_parts = prefix + (name,)
            relative = "/".join(relative_parts)
            try:
                metadata = os.stat(
                    name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise SmFsPrivacyError(
                    f"cannot inspect corpus member: {relative}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                # A linked directory can hide unindexed maps, and a linked map
                # can escape the retained corpus root. Reject both.
                raise SmFsPrivacyError(
                    f"refusing symlink corpus member: {relative}"
                )
            if name.casefold().endswith(".smmx"):
                if not stat.S_ISREG(metadata.st_mode):
                    raise SmFsPrivacyError(
                        f"refusing non-regular corpus member: {relative}"
                    )
                bindings.append(
                    CorpusMemberBinding(
                        relative=relative,
                        directory_identities=directory_identities,
                        file_identity=_identity(metadata),
                    )
                )
                continue
            if stat.S_ISDIR(metadata.st_mode):
                child = _open_directory_component(
                    directory_fd,
                    name,
                    relative,
                )
                try:
                    walk(
                        child,
                        relative_parts,
                        directory_identities + (_identity(os.fstat(child)),),
                    )
                finally:
                    os.close(child)
                continue

    walk(root_fd, (), (_identity(root_metadata),))
    return bindings


def smmx_member_paths(root_fd: int) -> List[str]:
    """Compatibility view over identity-bound corpus members."""

    return [
        binding.relative for binding in smmx_member_bindings(root_fd)
    ]


def _open_member_parent(
    root_fd: int,
    relative: str,
    expected_directories: Optional[Tuple[Tuple[int, int], ...]] = None,
) -> Tuple[int, str]:
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or relative_path.as_posix() != relative
        or any(part in ("", ".", "..") for part in relative_path.parts)
    ):
        raise SmFsPrivacyError(f"invalid corpus member path: {relative!r}")
    descriptor = os.dup(root_fd)
    try:
        if (
            expected_directories is not None
            and (
                len(expected_directories) != len(relative_path.parts)
                or _identity(os.fstat(descriptor))
                != expected_directories[0]
            )
        ):
            raise SmFsPrivacyError(
                f"corpus directory identity changed: {relative}"
            )
        for depth, component in enumerate(relative_path.parts[:-1], start=1):
            prefix = "/".join(relative_path.parts[:depth])
            child = _open_directory_component(descriptor, component, prefix)
            if (
                expected_directories is not None
                and _identity(os.fstat(child)) != expected_directories[depth]
            ):
                os.close(child)
                raise SmFsPrivacyError(
                    f"corpus directory identity changed: {prefix}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, relative_path.name
    except BaseException:
        os.close(descriptor)
        raise


def read_corpus_member(
    root_fd: int,
    relative: str,
    *,
    binding: Optional[CorpusMemberBinding] = None,
) -> bytes:
    """Read stable exact bytes through a retained root descriptor."""

    if binding is not None and binding.relative != relative:
        raise SmFsPrivacyError("corpus member binding path mismatch")
    parent_fd, name = _open_member_parent(
        root_fd,
        relative,
        None if binding is None else binding.directory_identities,
    )
    descriptor = -1
    try:
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode):
                raise SmFsPrivacyError(
                    f"refusing symlink corpus member: {relative}"
                )
            if not stat.S_ISREG(before.st_mode):
                raise SmFsPrivacyError(
                    f"refusing non-regular corpus member: {relative}"
                )
            if (
                binding is not None
                and _identity(before) != binding.file_identity
            ):
                raise SmFsPrivacyError(
                    f"corpus member identity changed: {relative}"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except SmFsPrivacyError:
            raise
        except OSError as exc:
            raise SmFsPrivacyError(
                f"cannot safely open corpus member: {relative}"
            ) from exc
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino)
            != (opened.st_dev, opened.st_ino)
        ):
            raise SmFsPrivacyError(
                f"corpus member changed while opening: {relative}"
            )
        data = _read_all_from_fd(descriptor)
        after_read = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        confirmed = _read_all_from_fd(descriptor)
        after_confirmation = os.fstat(descriptor)
        try:
            after_name = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise SmFsPrivacyError(
                f"corpus member changed while reading: {relative}"
            ) from exc
        if data != confirmed:
            raise SmFsPrivacyError(
                f"corpus member bytes changed while reading: {relative}"
            )
        identity = (opened.st_dev, opened.st_ino)
        if any(
            (value.st_dev, value.st_ino) != identity
            for value in (after_read, after_confirmation, after_name)
        ):
            raise SmFsPrivacyError(
                f"corpus member changed while reading: {relative}"
            )
        stable = (opened.st_size, opened.st_mtime_ns)
        if any(
            (value.st_size, value.st_mtime_ns) != stable
            for value in (after_read, after_confirmation, after_name)
        ):
            raise SmFsPrivacyError(
                f"corpus member changed while reading: {relative}"
            )
        if len(data) != opened.st_size:
            raise SmFsPrivacyError(
                f"corpus member size changed while reading: {relative}"
            )
        return data
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _read_regular_file_no_follow(path: Path) -> bytes:
    """Read stable exact bytes without ever following the final symlink.

    DrvFs/Dropbox hydration updates ``ctime`` and ``atime`` on the first read
    even when the file identity, size, ``mtime``, and bytes do not change.
    Therefore content stability is established by two identical reads from the
    same descriptor rather than by treating ``ctime`` as a content version.
    """

    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise SmFsPrivacyError(f"refusing symlink file: {path}")
        if not stat.S_ISREG(before.st_mode):
            raise SmFsPrivacyError(f"refusing non-regular file: {path}")
        fd = os.open(path, flags | no_follow)
    except SmFsPrivacyError:
        raise
    except OSError as exc:
        raise SmFsPrivacyError(f"cannot safely open regular file: {path}") from exc

    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise SmFsPrivacyError(f"refusing non-regular file: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise SmFsPrivacyError(f"file changed while opening: {path}")
        data = _read_all_from_fd(fd)
        after_read = os.fstat(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        confirmed = _read_all_from_fd(fd)
        after_confirmation = os.fstat(fd)
        try:
            after_path = path.lstat()
        except OSError as exc:
            raise SmFsPrivacyError(f"file changed while reading: {path}") from exc
        if data != confirmed:
            raise SmFsPrivacyError(f"file bytes changed while reading: {path}")
        identity_fields = ("st_dev", "st_ino")
        if any(
            getattr(opened, field) != getattr(value, field)
            for field in identity_fields
            for value in (after_read, after_confirmation, after_path)
        ):
            raise SmFsPrivacyError(f"file changed while reading: {path}")
        stable_fields = ("st_size", "st_mtime_ns")
        if any(
            getattr(opened, field) != getattr(value, field)
            for field in stable_fields
            for value in (after_read, after_confirmation, after_path)
        ):
            raise SmFsPrivacyError(f"file changed while reading: {path}")
        if len(data) != opened.st_size:
            raise SmFsPrivacyError(f"file size changed while reading: {path}")
        if any(
            not stat.S_ISREG(value.st_mode)
            for value in (after_read, after_confirmation, after_path)
        ):
            raise SmFsPrivacyError(f"file changed type while reading: {path}")
        return data
    finally:
        if fd >= 0:
            os.close(fd)


def sha256_file(path: Path) -> str:
    return sha256_bytes(_read_regular_file_no_follow(path))


def _strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise SmFsPrivacyError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _normal_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\\N", " ").split())


def _exact_private_marker(value: Any) -> bool:
    text = _normal_text(value).casefold()
    text = text.strip(" \t\r\n*[](){}.!?")
    return text == "private"


def _access_prefix(value: Any) -> bool:
    return bool(ACCESS_PREFIX.search(_normal_text(value)))


def _contains_private_word(value: Any) -> bool:
    return bool(PRIVATE_WORD.search(_normal_text(value)))


def _looks_like_privacy_topic_url(url: str) -> bool:
    parsed = urlparse(url)
    searchable = unquote(f"{parsed.path} {parsed.query} {parsed.fragment}")
    return bool(PRIVATE_OR_PRIVACY_WORD.search(searchable.replace("-", " ").replace("_", " ")))


def pearltrees_url_kind(url: str) -> Optional[str]:
    """Return ``public``, ``private_unknown``, or ``None`` for one URL.

    ``/private/id...`` is deliberately ambiguous: in this corpus it is often
    an authentication/export artifact on otherwise ordinary STEM maps.
    """

    if not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
    except ValueError:
        return None
    if parsed.scheme.casefold() not in ("http", "https"):
        return None
    if host not in ("pearltrees.com", "www.pearltrees.com"):
        return None
    # A public root link has exactly /ACCOUNT/SLUG/idNNN. Comparing netloc as
    # well as hostname rejects credentials and ports that merely parse to the
    # trusted hostname.
    if parsed.netloc.casefold() != host:
        return None
    decoded_path = unquote(parsed.path)
    match = re.fullmatch(r"/([^/]+)/([^/]+)/(id\d+)/?", decoded_path, re.I)
    parts = [part for part in decoded_path.split("/") if part]
    if (
        parts
        and any(part.casefold() == "private" for part in parts)
        and PEARLTREES_ID.fullmatch(parts[-1])
    ):
        return "private_unknown"
    if match is None:
        # The unreliable export form is the sole accepted two-component
        # exception, and it remains unknown rather than public.
        private_match = re.fullmatch(r"/private/(id\d+)/?", decoded_path, re.I)
        return "private_unknown" if private_match else None
    account, slug, pearltrees_id = match.groups()
    if (
        account.casefold() != "private"
        and slug.casefold() != "private"
        and PEARLTREES_SEGMENT.fullmatch(account)
        and PEARLTREES_SEGMENT.fullmatch(slug)
        and PEARLTREES_ID.fullmatch(pearltrees_id)
    ):
        return "public"
    return None


def _topic_links(topic, attribute: str) -> List[str]:
    return [
        value
        for value in (link.get(attribute) for link in topic.findall("link"))
        if isinstance(value, str) and value
    ]


def _topic_has_public_topical_link(topic) -> bool:
    for url in _topic_links(topic, "urllink"):
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
        if (
            pearltrees_url_kind(url) != "private_unknown"
            and parsed.scheme in ("http", "https")
            and (host == "wikipedia.org" or host.endswith(".wikipedia.org"))
            and _looks_like_privacy_topic_url(url)
        ):
            return True
    return False


def _inside(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath((str(root), str(path))) == str(root)
    except ValueError:
        return False


def _relative_member_path(corpus_root: Path, path: Path) -> Optional[str]:
    path = Path(os.path.abspath(os.path.normpath(str(path))))
    if not _inside(corpus_root, path):
        return None
    return path.relative_to(corpus_root).as_posix()


def _member_catalog(
    corpus_root: Path, paths: Sequence[Path]
) -> Tuple[Dict[str, Path], Dict[str, Optional[Path]]]:
    exact: Dict[str, Path] = {}
    folded: Dict[str, Optional[Path]] = {}
    for path in paths:
        relative = path.relative_to(corpus_root).as_posix()
        exact[relative] = path
        key = relative.casefold()
        if key in folded and folded[key] != path:
            folded[key] = None
        else:
            folded[key] = path
    return exact, folded


def _catalog_member(
    corpus_root: Path,
    candidate: Path,
    members: Tuple[Mapping[str, Path], Mapping[str, Optional[Path]]],
) -> Optional[Path]:
    relative = _relative_member_path(corpus_root, candidate)
    if relative is None:
        return None
    exact, folded = members
    if relative in exact:
        return exact[relative]
    return folded.get(relative.casefold())


def resolve_cloudmapref(
    corpus_root: Path,
    source: Path,
    ref: str,
    *,
    members: Optional[
        Tuple[Mapping[str, Path], Mapping[str, Optional[Path]]]
    ] = None,
) -> Optional[Path]:
    """Resolve SimpleMind relative and virtual ``/root/...`` map references."""

    corpus_root = Path(os.path.abspath(str(corpus_root)))
    source = Path(os.path.abspath(str(source)))
    normalized = ref.replace("\\", "/").strip()
    if not normalized or normalized == ".":
        return None
    if normalized.casefold().startswith("/root/"):
        candidates = [corpus_root / normalized[len("/root/") :]]
    elif normalized.casefold().startswith("root/"):
        candidates = [corpus_root / normalized[len("root/") :]]
    elif normalized.startswith("/"):
        return None
    else:
        candidates = [
            source.parent / normalized,
            corpus_root / normalized,
        ]
    candidates = [
        Path(os.path.abspath(os.path.normpath(str(candidate))))
        for candidate in candidates
        if _inside(
            corpus_root,
            Path(os.path.abspath(os.path.normpath(str(candidate)))),
        )
    ]
    if not candidates:
        return None
    if members is not None:
        resolved_members = {
            target
            for target in (
                _catalog_member(corpus_root, candidate, members)
                for candidate in candidates
            )
            if target is not None
        }
        return next(iter(resolved_members)) if len(resolved_members) == 1 else None

    regular_candidates = []
    for candidate in candidates:
        try:
            mode = candidate.lstat().st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode):
            regular_candidates.append(candidate)
    distinct = set(regular_candidates)
    if len(distinct) > 1:
        return None
    if len(distinct) == 1:
        return next(iter(distinct))
    # Preserve the resolver's lexical behavior for callers that are checking a
    # virtual /root target before it has been created.
    return candidates[0]


def _private_word_components(
    relative_path: Path,
) -> Tuple[List[str], List[str], List[str]]:
    exact, access, broad = [], [], []
    for component in relative_path.parent.parts:
        if _exact_private_marker(component):
            exact.append(component)
        elif _access_prefix(component):
            access.append(component)
        elif _contains_private_word(component):
            broad.append(component)
    return exact, access, broad


def _load_smmx_bytes(raw: bytes):
    stream = io.BytesIO(raw)
    if zipfile.is_zipfile(stream):
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            canonical = "document/mindmap.xml"
            candidates = []
            for info in archive.infolist():
                normalized = posixpath.normpath(
                    info.filename.replace("\\", "/")
                ).casefold()
                if normalized == canonical.casefold():
                    candidates.append(info)
            if (
                len(candidates) != 1
                or candidates[0].filename != canonical
                or candidates[0].is_dir()
            ):
                raise SmFsPrivacyError(
                    "map archive must contain exactly one unambiguous "
                    "document/mindmap.xml member"
                )
            return ET.fromstring(archive.read(candidates[0]))
    return ET.fromstring(raw)


def _blank_root_placeholder(
    topic,
    children: Mapping[str, Sequence[str]],
    relation_topic_ids: Set[str],
) -> bool:
    topic_id = topic.get("id")
    if (
        label(topic)
        or children.get(topic_id)
        or topic_id in relation_topic_ids
        or (topic.text or "").strip()
    ):
        return False
    if any(
        value
        for key, value in topic.attrib.items()
        if key not in ("id", "parent", "text", "guid") and str(value).strip()
    ):
        return False
    # Links, notes, relations, or any other nested payload mean the root is not
    # demonstrably blank even if its visible text happens to be empty.
    return not list(topic)


def _validated_topic_graph(xml):
    topics = list(xml.findall(".//topic"))
    if not topics:
        raise SmFsPrivacyError("map has no topics")
    topic_by_id = {}
    for topic in topics:
        topic_id = topic.get("id")
        if (
            not isinstance(topic_id, str)
            or not topic_id.strip()
            or topic_id == "-1"
        ):
            raise SmFsPrivacyError("map topic is missing an id")
        if topic_id in topic_by_id:
            raise SmFsPrivacyError(f"duplicate map topic id: {topic_id!r}")
        topic_by_id[topic_id] = topic

    children = defaultdict(list)
    roots = []
    for topic_id, topic in topic_by_id.items():
        parent_id = topic.get("parent")
        if parent_id in (None, "-1"):
            roots.append(topic)
            continue
        if parent_id not in topic_by_id:
            raise SmFsPrivacyError(
                f"map topic {topic_id!r} has missing parent {parent_id!r}"
            )
        children[parent_id].append(topic_id)
    if not roots:
        raise SmFsPrivacyError("map has no root topic")

    for start_id in topic_by_id:
        seen = set()
        topic_id = start_id
        while topic_id is not None:
            if topic_id in seen:
                raise SmFsPrivacyError("map topic parent graph has a cycle")
            seen.add(topic_id)
            parent_id = topic_by_id[topic_id].get("parent")
            topic_id = None if parent_id in (None, "-1") else parent_id

    relation_topic_ids = {
        topic_id
        for relation in xml.findall(".//relation")
        for topic_id in (relation.get("source"), relation.get("target"))
        if topic_id is not None
    }
    if len(roots) == 1:
        root_topic = roots[0]
        extra_roots = False
    else:
        substantive = [
            topic
            for topic in roots
            if not _blank_root_placeholder(
                topic, children, relation_topic_ids
            )
        ]
        if len(substantive) != 1:
            raise SmFsPrivacyError(
                f"cannot distinguish {len(roots)} root topics"
            )
        root_topic = substantive[0]
        if any(
            not _blank_root_placeholder(
                topic, children, relation_topic_ids
            )
            for topic in roots
            if topic is not root_topic
        ):
            raise SmFsPrivacyError("extra root topic is not a blank placeholder")
        extra_roots = True
    return topics, topic_by_id, children, root_topic, extra_roots


def _base_record(
    corpus_root: Path,
    members: Tuple[Mapping[str, Path], Mapping[str, Optional[Path]]],
    root_fd: int,
    binding: CorpusMemberBinding,
) -> Dict[str, Any]:
    relative = binding.relative
    path = corpus_root / PurePosixPath(relative)
    raw = read_corpus_member(root_fd, relative, binding=binding)
    record = {
        "record_type": "map",
        "relative_path": relative,
        "file_sha256": sha256_bytes(raw),
        "root_title": "",
        "root_urls": [],
        "classification": "unknown",
        "reasons": [],
        "model_review": None,
        "unresolved_private_targets": [],
    }
    exact_dirs, access_dirs, broad_dirs = _private_word_components(Path(relative))
    try:
        xml = _load_smmx_bytes(raw)
        topics, topic_by_id, children, root_topic, extra_roots = (
            _validated_topic_graph(xml)
        )
        if extra_roots:
            record["reasons"].append("extra_root_topics_ignored")
        root_title = label(root_topic)
        root_urls = sorted(set(_topic_links(root_topic, "urllink")))
        record["root_title"] = root_title
        record["root_urls"] = root_urls

        public_root = any(pearltrees_url_kind(url) == "public" for url in root_urls)
        private_link = any(
            pearltrees_url_kind(url) == "private_unknown" for url in root_urls
        )
        access_prefix = _access_prefix(path.stem) or _access_prefix(root_title)
        access_seed = bool(exact_dirs or access_dirs) or access_prefix
        broad_private = bool(broad_dirs) or (
            _contains_private_word(path.stem)
            or _contains_private_word(root_title)
        )
        root_topic_evidence = (
            bool(PRIVATE_OR_PRIVACY_WORD.search(root_title))
            and not access_prefix
        ) or any(
            pearltrees_url_kind(url) != "private_unknown"
            and _looks_like_privacy_topic_url(url)
            for url in root_urls
        )

        if public_root and access_seed:
            record["classification"] = "unknown"
            record["reasons"].extend(
                ("public_root_link", "access_marker_conflicts_with_public_root")
            )
        elif public_root:
            record["classification"] = "public"
            record["reasons"].append("public_root_link")
            if broad_private:
                record["reasons"].append("public_root_link_overrides_private_word")
        elif access_seed and root_topic_evidence:
            record["classification"] = "unknown"
            record["reasons"].append("private_marker_with_topical_root")
        elif access_seed:
            record["classification"] = "private"
            record["reasons"].append("private_dir_or_access_prefix")
        elif private_link:
            record["classification"] = "unknown"
            record["reasons"].append("pearltrees_private_link_unknown")
        elif broad_private:
            record["classification"] = "unknown"
            record["reasons"].append("private_word_marker_vs_topic")
        else:
            record["classification"] = "public"
            record["reasons"].append("owner_default_no_private_signal")

        private_targets = set()
        unresolved = set()
        for marker_id, marker in topic_by_id.items():
            if not _exact_private_marker(label(marker)):
                continue
            if _topic_has_public_topical_link(marker):
                continue
            closure, frontier = set(), [marker_id]
            while frontier:
                topic_id = frontier.pop()
                if topic_id in closure:
                    continue
                closure.add(topic_id)
                frontier.extend(children.get(topic_id, ()))
            for topic_id in closure:
                topic = topic_by_id[topic_id]
                for ref in _topic_links(topic, "cloudmapref"):
                    target = resolve_cloudmapref(
                        corpus_root, path, ref, members=members
                    )
                    if target is None:
                        if ref != ".":
                            unresolved.add(ref)
                        continue
                    if target.suffix.casefold() != ".smmx":
                        unresolved.add(ref)
                        continue
                    private_targets.add(target.relative_to(corpus_root).as_posix())
        record["_private_targets"] = sorted(private_targets)
        record["unresolved_private_targets"] = sorted(unresolved)
        if private_targets:
            record["reasons"].append("contains_private_map_links")
        if unresolved:
            record["reasons"].append("unresolved_private_map_links")
    except Exception as exc:
        if exact_dirs or access_dirs or _access_prefix(path.stem):
            record["classification"] = "private"
            record["reasons"].append("private_path_without_readable_map")
        else:
            record["classification"] = "unknown"
            record["reasons"].append("map_parse_error")
        record["parse_error_type"] = type(exc).__name__
        record["_private_targets"] = []
    record["reasons"] = sorted(set(record["reasons"]))
    return record


def _smmx_paths(corpus_root: Path) -> List[Path]:
    """Compatibility wrapper; production scans keep the returned root fd open."""

    with open_corpus_root(corpus_root) as (root, root_fd):
        return [
            root / PurePosixPath(binding.relative)
            for binding in smmx_member_bindings(root_fd)
        ]


def scan_maps(corpus_root: Path, *, workers: int = 8) -> List[Dict[str, Any]]:
    if workers <= 0:
        raise SmFsPrivacyError("workers must be positive")
    with open_corpus_root(corpus_root) as (corpus_root, root_fd):
        bindings = smmx_member_bindings(root_fd)
        paths = [
            corpus_root / PurePosixPath(binding.relative)
            for binding in bindings
        ]
        members = _member_catalog(corpus_root, paths)
        build_record = partial(_base_record, corpus_root, members, root_fd)
        if workers == 1:
            records = [build_record(binding) for binding in bindings]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                records = list(executor.map(build_record, bindings))
    by_path = {record["relative_path"]: record for record in records}
    for source in records:
        for target_path in source.pop("_private_targets", ()):
            target = by_path.get(target_path)
            if target is None:
                source["unresolved_private_targets"] = sorted(
                    set(source["unresolved_private_targets"]) | {target_path}
                )
                source["reasons"] = sorted(
                    set(source["reasons"]) | {"unresolved_private_map_links"}
                )
                continue
            if "public_root_link" in target["reasons"]:
                target["classification"] = "unknown"
                target["reasons"] = sorted(
                    set(target["reasons"])
                    | {"private_cloud_child_conflicts_with_public_root"}
                )
                continue
            target["classification"] = "private"
            target["reasons"] = sorted(
                set(target["reasons"]) | {"private_cloud_child"}
            )
    return records


def review_task(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "qid": sha256_bytes(record["relative_path"].encode("utf-8"))[:20],
        "relative_path": record["relative_path"],
        "root_title": record.get("root_title", ""),
        "root_urls": list(record.get("root_urls", ())),
        "rule_reasons": list(record.get("reasons", ())),
        "allowed_interpretations": sorted(INTERPRETATIONS),
    }


def build_review_prompt(tasks: Sequence[Mapping[str, Any]]) -> str:
    return """You are classifying owner-controlled SimpleMind filing metadata.

For each row, decide whether its use of "private" is:
- access_control: a private copy/branch or access-control marker;
- topical: an ordinary map whose subject happens to include "private", or an
  ordinary map carrying an unreliable pearltrees.com/private export link;
- uncertain: not enough evidence.

A pearltrees.com/private/id... URL is NOT proof of privacy: this export has many
normal STEM maps with that URL. A normal /ACCOUNT/SLUG/id... root link is public
topical evidence. Private:, Private_, and Private - prefixes normally identify
access-controlled copies. Rule reason glossary: private_dir_or_access_prefix
means a filesystem/access-copy marker; private_marker_with_topical_root means
that marker conflicts with a topical title or link; private_word_marker_vs_topic
means the wording alone is ambiguous; pearltrees_private_link_unknown means the
export URL is unreliable.

Apply this precedence:
1. An explicit access-copy name wins over an unreliable /private/id URL.
2. An otherwise ordinary STEM title with only that unreliable URL is topical.
3. A known subject use (privacy law, private-key cryptography, private methods)
   with positive topical evidence is topical.
4. A filesystem/access marker conflicting with a topical title, or a bare
   "private ..." phrase with no decisive context, is uncertain.

Synthetic examples:
- Private - Lab Copy + private_dir_or_access_prefix => access_control.
- Bayesian statistics + only pearltrees_private_link_unknown => topical.
- exact Private folder + Private-key cryptography title => uncertain.

Return ONLY an unfenced JSON array with exactly one object per input row.
Preserve every qid exactly. Each object must have exactly qid, interpretation,
and reason. Keep reason nonempty and at most 300 characters. No markdown.

Rows:
""" + json.dumps(list(tasks), ensure_ascii=False, sort_keys=True)


def _parse_model_response(text: str, expected_qids: Set[str]) -> Dict[str, Dict[str, str]]:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:].lstrip()
    try:
        rows = json.loads(stripped, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, SmFsPrivacyError) as exc:
        raise SmFsPrivacyError(f"model response is not strict JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise SmFsPrivacyError("model response must be a JSON array")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SmFsPrivacyError("model response rows must be objects")
        if set(row) != {"qid", "interpretation", "reason"}:
            raise SmFsPrivacyError("model response row fields changed")
        qid = row["qid"]
        interpretation = row["interpretation"]
        reason = row["reason"]
        if qid not in expected_qids or qid in out:
            raise SmFsPrivacyError("model response qid set changed")
        if interpretation not in INTERPRETATIONS:
            raise SmFsPrivacyError(f"invalid model interpretation: {interpretation!r}")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
            raise SmFsPrivacyError("invalid model reason")
        out[qid] = {"interpretation": interpretation, "reason": reason.strip()}
    if set(out) != expected_qids:
        raise SmFsPrivacyError("model response omitted review rows")
    return out


def apply_model_reviews(
    records: List[Dict[str, Any]],
    *,
    provider: str,
    model: str,
    batch_size: int,
    timeout: int,
    call_llm: Callable[[str, str, str, int], Optional[str]],
) -> Dict[str, int]:
    # Benchmarking uses only build_review_prompt() and _parse_model_response()
    # with synthetic rows. This legacy corpus-facing entry point is sealed off
    # so no caller can accidentally serialize or transmit owner metadata.
    del records, provider, model, batch_size, timeout, call_llm
    raise SmFsPrivacyError(MODEL_REVIEW_DISABLED_REASON)


def _source_snapshot(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    members = [
        {
            "relative_path": record["relative_path"],
            "file_sha256": record["file_sha256"],
        }
        for record in records
    ]
    return {
        "file_count": len(members),
        "members_sha256": sha256_bytes(
            b"".join(canonical_json_bytes(member) for member in members)
        ),
    }


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _require_sorted_unique_strings(value: Any, field: str) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or value != sorted(set(value))
    ):
        raise SmFsPrivacyError(
            f"privacy index {field} must be sorted unique strings"
        )


def _validate_v1_record(record: Any) -> None:
    if not isinstance(record, Mapping) or record.get("record_type") != "map":
        raise SmFsPrivacyError("privacy index map row is malformed")
    fields = set(record)
    if not MAP_FIELDS.issubset(fields) or not fields.issubset(
        MAP_FIELDS | MAP_OPTIONAL_FIELDS
    ):
        raise SmFsPrivacyError("privacy index v1 map fields changed")

    relative = record.get("relative_path")
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise SmFsPrivacyError("privacy index map path is invalid")
    relative_path = PurePosixPath(relative)
    if (
        relative_path.is_absolute()
        or relative_path.as_posix() != relative
        or any(part in ("", ".", "..") for part in relative_path.parts)
        or relative_path.suffix.casefold() != ".smmx"
    ):
        raise SmFsPrivacyError("privacy index map path is invalid")
    if not isinstance(record.get("file_sha256"), str) or not SHA256_HEX.fullmatch(
        record["file_sha256"]
    ):
        raise SmFsPrivacyError("privacy index map digest is invalid")
    if not isinstance(record.get("root_title"), str):
        raise SmFsPrivacyError("privacy index root title is invalid")
    _require_sorted_unique_strings(record.get("root_urls"), "root_urls")
    _require_sorted_unique_strings(record.get("reasons"), "reasons")
    _require_sorted_unique_strings(
        record.get("unresolved_private_targets"),
        "unresolved_private_targets",
    )
    if record.get("classification") not in CLASSIFICATIONS:
        raise SmFsPrivacyError("privacy index map classification is invalid")
    if record.get("model_review") is not None or any(
        reason.startswith("model_") for reason in record["reasons"]
    ):
        raise SmFsPrivacyError("privacy index v1 cannot contain model review data")

    parse_reasons = {
        "map_parse_error",
        "private_path_without_readable_map",
    }
    has_parse_error = bool(parse_reasons.intersection(record["reasons"]))
    parse_error_type = record.get("parse_error_type")
    if (
        has_parse_error
        and (
            "parse_error_type" not in record
            or not isinstance(parse_error_type, str)
            or not parse_error_type
        )
    ) or (not has_parse_error and "parse_error_type" in record):
        raise SmFsPrivacyError("privacy index parse-error metadata is invalid")


def _derived_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    classifications = Counter(record["classification"] for record in records)
    reasons = Counter(
        reason for record in records for reason in record.get("reasons", ())
    )
    return {
        "public": classifications["public"],
        "private": classifications["private"],
        "unknown": classifications["unknown"],
        "observed": len(records),
        "model_access_control": 0,
        "model_topical": 0,
        "model_uncertain": 0,
        "map_parse_errors": reasons["map_parse_error"],
        "unresolved_private_target_refs": sum(
            len(record["unresolved_private_targets"]) for record in records
        ),
    }


def _index_digest(
    header: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> str:
    core = dict(header)
    core.pop("index_sha256", None)
    core["rows_sha256"] = sha256_bytes(
        b"".join(canonical_json_bytes(record) for record in records)
    )
    return sha256_bytes(canonical_json_bytes(core))


def _validate_v1_index(
    header: Any,
    records: Sequence[Any],
    *,
    verify_digest: bool,
) -> None:
    if not isinstance(header, Mapping) or header.get("record_type") != "header":
        raise SmFsPrivacyError("privacy index header is missing")
    if set(header) != HEADER_FIELDS:
        raise SmFsPrivacyError("privacy index v1 header fields changed")
    if header.get("schema") != SCHEMA or header.get("policy_id") != POLICY_ID:
        raise SmFsPrivacyError("privacy index schema/policy changed")
    corpus_root = header.get("corpus_root")
    if (
        not isinstance(corpus_root, str)
        or not corpus_root
        or not os.path.isabs(corpus_root)
        or os.path.abspath(corpus_root) != corpus_root
    ):
        raise SmFsPrivacyError("privacy index corpus root is invalid")
    for field in ("classifier_sha256", "parser_sha256", "index_sha256"):
        value = header.get(field)
        if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
            raise SmFsPrivacyError(f"privacy index {field} is invalid")
    if header.get("model") is not None:
        raise SmFsPrivacyError("privacy index v1 cannot name a model")
    if header.get("model_review_authorization") != {
        "authorized": False,
        "reason": MODEL_REVIEW_DISABLED_REASON,
    }:
        raise SmFsPrivacyError(
            "privacy index v1 model-review authorization changed"
        )

    counts = header.get("counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != COUNT_FIELDS
        or any(not _is_count(value) for value in counts.values())
    ):
        raise SmFsPrivacyError("privacy index v1 counts are invalid")
    snapshot = header.get("source_snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or set(snapshot) != {"file_count", "members_sha256"}
        or not _is_count(snapshot.get("file_count"))
        or not isinstance(snapshot.get("members_sha256"), str)
        or not SHA256_HEX.fullmatch(snapshot["members_sha256"])
    ):
        raise SmFsPrivacyError("privacy index source snapshot is invalid")

    seen = set()
    for record in records:
        _validate_v1_record(record)
        relative = record["relative_path"]
        if relative in seen:
            raise SmFsPrivacyError("privacy index map path is duplicated")
        seen.add(relative)
    if counts != _derived_counts(records):
        raise SmFsPrivacyError("privacy index derived counts changed")
    if snapshot != _source_snapshot(records):
        raise SmFsPrivacyError("privacy index source snapshot changed")
    if verify_digest and header["index_sha256"] != _index_digest(header, records):
        raise SmFsPrivacyError("privacy index digest changed")


def build_index(
    corpus_root: Path,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: int = 20,
    timeout: int = 120,
    workers: int = 8,
    call_llm: Optional[Callable[[str, str, str, int], Optional[str]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if bool(provider) != bool(model):
        raise SmFsPrivacyError("--provider and --model must be supplied together")
    if provider or call_llm is not None:
        raise SmFsPrivacyError(MODEL_REVIEW_DISABLED_REASON)

    classifier_sha256 = sha256_file(Path(__file__))
    parser_sha256 = sha256_file(PARSER_PATH)
    records = scan_maps(corpus_root, workers=workers)
    if sha256_file(Path(__file__)) != classifier_sha256:
        raise SmFsPrivacyError("classifier changed while the index was being built")
    if sha256_file(PARSER_PATH) != parser_sha256:
        raise SmFsPrivacyError("SimpleMind parser changed while the index was being built")
    header = {
        "record_type": "header",
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "corpus_root": str(Path(os.path.abspath(str(corpus_root)))),
        "source_snapshot": _source_snapshot(records),
        "classifier_sha256": classifier_sha256,
        "parser_sha256": parser_sha256,
        "model": None,
        "model_review_authorization": {
            "authorized": False,
            "reason": MODEL_REVIEW_DISABLED_REASON,
        },
        "counts": _derived_counts(records),
    }
    header["index_sha256"] = _index_digest(header, records)
    _validate_v1_index(header, records, verify_digest=True)
    return header, records


def write_index(path: Path, header: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    _validate_v1_index(header, records, verify_digest=True)
    data = canonical_json_bytes(dict(header)) + b"".join(
        canonical_json_bytes(dict(record)) for record in records
    )
    if path.exists():
        raise FileExistsError(
            f"refusing to replace sealed privacy index: {path}; choose a new path"
        )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    installed = False
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        installed = True
        directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        if installed:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def load_index(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    try:
        lines = _read_regular_file_no_follow(path).splitlines()
    except SmFsPrivacyError as exc:
        raise SmFsPrivacyError(f"cannot read privacy index: {path}") from exc
    if not lines:
        raise SmFsPrivacyError("privacy index is empty")
    try:
        values = [
            json.loads(line.decode("utf-8"), object_pairs_hook=_strict_object)
            for line in lines
        ]
    except (UnicodeDecodeError, json.JSONDecodeError, SmFsPrivacyError) as exc:
        raise SmFsPrivacyError(f"privacy index is malformed: {exc}") from exc
    header = values[0]
    records = values[1:]
    _validate_v1_index(header, records, verify_digest=True)
    by_path = {}
    for record in records:
        by_path[record["relative_path"]] = record
    return header, by_path


def verify_index_source(
    corpus_root: Path,
    header: Mapping[str, Any],
    by_path: Mapping[str, Mapping[str, Any]],
    *,
    workers: int = 8,
) -> None:
    """Re-derive the index with the exact classifier and byte-compare every row."""

    corpus_root = Path(os.path.abspath(str(corpus_root)))
    expected_root = header.get("corpus_root")
    if expected_root != str(corpus_root):
        raise SmFsPrivacyError("privacy index corpus root changed")
    classifier_sha256 = sha256_file(Path(__file__))
    parser_sha256 = sha256_file(PARSER_PATH)
    if header.get("classifier_sha256") != classifier_sha256:
        raise SmFsPrivacyError("privacy index classifier changed; rebuild the index")
    if header.get("parser_sha256") != parser_sha256:
        raise SmFsPrivacyError("privacy index SimpleMind parser changed; rebuild the index")
    if workers <= 0:
        raise SmFsPrivacyError("workers must be positive")

    indexed_records = []
    for relative, record in by_path.items():
        if (
            not isinstance(relative, str)
            or not isinstance(record, Mapping)
            or record.get("relative_path") != relative
        ):
            raise SmFsPrivacyError("privacy index map mapping is invalid")
        indexed_records.append(record)
    _validate_v1_index(header, indexed_records, verify_digest=True)

    derived_records = scan_maps(corpus_root, workers=workers)
    if sha256_file(Path(__file__)) != classifier_sha256:
        raise SmFsPrivacyError("classifier changed while the index was verified")
    if sha256_file(PARSER_PATH) != parser_sha256:
        raise SmFsPrivacyError(
            "SimpleMind parser changed while the index was verified"
        )
    derived_by_path = {
        record["relative_path"]: record for record in derived_records
    }
    if set(by_path) != set(derived_by_path):
        raise SmFsPrivacyError("privacy index is stale: map set changed")
    for relative, derived in derived_by_path.items():
        if canonical_json_bytes(by_path[relative]) != canonical_json_bytes(derived):
            if by_path[relative].get("file_sha256") != derived["file_sha256"]:
                raise SmFsPrivacyError(
                    "privacy index is stale: map contents changed"
                )
            raise SmFsPrivacyError(
                "privacy index record differs from exact classifier output"
            )
    _validate_v1_index(header, derived_records, verify_digest=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/c/Users/johnc/Dropbox/root")
    parser.add_argument(
        "--out", default=os.path.expanduser("~/mu_data/sm_fs_privacy_index.jsonl")
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    header, records = build_index(
        Path(args.root),
        workers=args.workers,
    )
    write_index(Path(args.out), header, records)
    counts = header["counts"]
    print(
        "SimpleMind privacy index: "
        f"{counts['public']} public, {counts['private']} private, "
        f"{counts['unknown']} unknown -> {args.out}"
    )
    print(
        "Review perimeter: "
        f"{counts['map_parse_errors']} unreadable maps; "
        f"{counts['unresolved_private_target_refs']} unresolved references "
        "below explicit private markers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
