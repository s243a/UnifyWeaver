#!/usr/bin/env python3
"""SimpleMind FILESYSTEM filing corpus: map ↔ its Dropbox directory = a recorded filing decision.

The .smmx maps live inside a real folder hierarchy, so each map's path is ground-truth filing —
the same single-principal-folder task as Pearltrees (map root ≈ bookmark, parent directory ≈
folder), on a corpus the eval program has NEVER touched. Two disciplines applied at first touch:

  UNSCORED RESERVE — a deterministic 40% of queries is reserved and never scored by this script.
  Because the catalog is placement-derived and the split is not folder-node-disjoint, this is a
  transductive reserve, not a confirmatory holdout. Exploration uses only the 60% split.

  PRIVACY — ``sm_fs_privacy.py`` classifies the frozen filesystem snapshot with contextual rules.
  Corpus model review remains disabled until a verified passing benchmark lock exists. The local
  default excludes records classified private while retaining unknown records;
  ``--privacy-policy public-only`` is the stricter candidate view for owner-approved external use,
  not independent public-availability certification. Per-item outputs stay in ~/mu_data and are
  never committed.

Task: query = map filename stem; true folder = immediate parent directory (title-equivalence
across duplicate directory names); catalog = directories holding >= --min-maps maps. e5 ranking.
Comparators: PT single-folder R@1 0.203 / MRR 0.291; SM parent-level (in-map) 0.180 / 0.320.

  python3 sm_fs_filing.py            # scan, freeze holdout, score the exploration split
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mu_attention
from mu_attention import E5_MODEL, E5_REVISION, build_e5_tables
from sm_fs_privacy import (
    SmFsPrivacyError,
    build_index,
    load_index,
    sha256_file,
    verify_index_source,
    write_index,
)

DEFAULT_ROOT = "/mnt/c/Users/johnc/Dropbox/root"
DEFAULT_PRIVACY_INDEX = os.path.expanduser("~/mu_data/sm_fs_privacy_index.jsonl")


class UnsafePrivatePathError(OSError):
    """A supposedly private artifact path has an unsafe filesystem perimeter."""


def _absolute_without_resolving(path):
    """Make *path* absolute without following any symlink component."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _lstat_or_none(path):
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _prepare_private_parent(parent):
    """Create/validate a private leaf directory without traversing symlinks.

    The leaf must be owned by this process and inaccessible to group/other
    users. Existing ancestors may be public, but may not be symlinks or
    non-sticky group/world-writable directories.
    """
    parent = _absolute_without_resolving(parent)
    chain = list(reversed(parent.parents)) + [parent]
    for component in chain:
        info = _lstat_or_none(component)
        if info is None:
            try:
                os.mkdir(component, 0o700)
            except FileExistsError:
                pass
            info = _lstat_or_none(component)
        if info is None:
            raise UnsafePrivatePathError(
                f"private artifact parent disappeared: {component}"
            )
        if stat.S_ISLNK(info.st_mode):
            raise UnsafePrivatePathError(
                f"refusing symlink parent for private artifact: {component}"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise UnsafePrivatePathError(
                f"private artifact parent is not a directory: {component}"
            )
        mode = stat.S_IMODE(info.st_mode)
        if (
            component != parent
            and mode & 0o022
            and not mode & stat.S_ISVTX
        ):
            raise UnsafePrivatePathError(
                f"refusing writable ancestor for private artifact: {component}"
            )

    leaf_info = os.lstat(parent)
    leaf_mode = stat.S_IMODE(leaf_info.st_mode)
    if leaf_info.st_uid != os.geteuid():
        raise UnsafePrivatePathError(
            f"private artifact parent is not owned by this user: {parent}"
        )
    if leaf_mode & 0o077 or leaf_mode & 0o700 != 0o700:
        raise UnsafePrivatePathError(
            f"private artifact parent must have mode 0700: {parent}"
        )
    return parent


def _validate_private_target(target, *, must_not_exist=False):
    target = _absolute_without_resolving(target)
    _prepare_private_parent(target.parent)
    info = _lstat_or_none(target)
    if info is None:
        return target
    if stat.S_ISLNK(info.st_mode):
        raise UnsafePrivatePathError(
            f"refusing symlink private artifact target: {target}"
        )
    if must_not_exist:
        raise FileExistsError(
            f"refusing to replace frozen ledger: {target}; "
            "choose a new --ledger path"
        )
    if not stat.S_ISREG(info.st_mode):
        raise UnsafePrivatePathError(
            f"private artifact target is not a regular file: {target}"
        )
    if info.st_uid != os.geteuid():
        raise UnsafePrivatePathError(
            f"private artifact target is not owned by this user: {target}"
        )
    if info.st_nlink != 1:
        raise UnsafePrivatePathError(
            f"private artifact target has multiple hard links: {target}"
        )
    return target


def _open_private_parent(target):
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target.parent, flags)
    info = os.fstat(descriptor)
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or mode & 0o077
    ):
        os.close(descriptor)
        raise UnsafePrivatePathError(
            f"private artifact parent changed while opening: {target.parent}"
        )
    return descriptor


def _open_private_regular(target, flags=os.O_RDONLY):
    descriptor = os.open(
        target,
        flags
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
    ):
        os.close(descriptor)
        raise UnsafePrivatePathError(
            f"private artifact target changed while opening: {target}"
        )
    return descriptor


def _atomic_json(path, value):
    target = _validate_private_target(path, must_not_exist=True)
    directory_fd = _open_private_parent(target)
    fd = None
    temporary_name = None
    linked = False
    durable = False
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=str(target.parent)
        )
        temporary_name = Path(temporary).name
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            json.dump(value, handle, ensure_ascii=False, indent=1, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary_name,
            target.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        linked = True
        os.fsync(directory_fd)
        durable = True
    except BaseException:
        if linked and not durable:
            try:
                os.unlink(target.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
        raise
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            if temporary_name is not None:
                os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory_fd)


def privacy_allows(classification, policy):
    if policy == "all-local":
        return True
    if policy == "exclude-private":
        return classification != "private"
    if policy == "public-only":
        return classification == "public"
    raise ValueError(f"unknown privacy policy: {policy}")


def load_or_build_privacy_index(root, index_path):
    path = Path(index_path)
    if not path.exists():
        header, records = build_index(Path(root))
        write_index(path, header, records)
    header, by_path = load_index(path)
    verify_index_source(Path(root), header, by_path)
    return header, by_path


def require_public_privacy_perimeter(header, policy):
    unresolved = header.get("counts", {}).get(
        "unresolved_private_target_refs", 0
    )
    if policy == "public-only" and unresolved:
        raise SmFsPrivacyError(
            f"privacy index has {unresolved} unresolved private target "
            "reference(s); public-only filing is not authorized"
        )


def prepare_private_cache(path):
    cache = _validate_private_target(path)
    if cache.exists():
        descriptor = _open_private_regular(cache)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
    return cache


def harden_private_cache(path):
    cache = _validate_private_target(path)
    if _lstat_or_none(cache) is not None:
        descriptor = _open_private_regular(cache)
        try:
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def filing_provenance(cache_sha256=None):
    provenance = {
        "sm_fs_filing_sha256": sha256_file(Path(__file__)),
        "mu_attention_sha256": sha256_file(Path(mu_attention.__file__)),
        "e5_model_id": E5_MODEL,
        "e5_revision": E5_REVISION,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": mu_attention.torch.__version__,
            "sentence_transformers": _distribution_version(
                "sentence-transformers"
            ),
            "transformers": _distribution_version("transformers"),
        },
    }
    if cache_sha256 is not None:
        provenance["e5_cache_sha256"] = cache_sha256
    return provenance


def _cache_payload_is_usable(payload, names, human):
    if not isinstance(payload, dict):
        return False
    if (
        payload.get("names") != list(names)
        or payload.get("human") != human
        or payload.get("model_name") != E5_MODEL
        or payload.get("model_revision") != E5_REVISION
    ):
        return False
    query = payload.get("query")
    passage = payload.get("passage")
    if not isinstance(query, mu_attention.torch.Tensor):
        return False
    if not isinstance(passage, mu_attention.torch.Tensor):
        return False
    if (
        query.ndim != 2
        or passage.shape != query.shape
        or query.shape[0] != len(names)
        or query.shape[1] == 0
        or query.dtype != mu_attention.torch.float32
        or passage.dtype != mu_attention.torch.float32
        or not query.is_floating_point()
        or not passage.is_floating_point()
    ):
        return False
    return bool(
        mu_attention.torch.isfinite(query).all()
        and mu_attention.torch.isfinite(passage).all()
    )


def _cache_stat_signature(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_nlink,
        stat.S_IMODE(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _validate_cache_descriptor(descriptor):
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise UnsafePrivatePathError(
            "e5 cache descriptor is not a private, singly-linked regular file"
        )
    return info


def _assert_cache_path_still_names(cache, expected):
    actual = _lstat_or_none(cache)
    if actual is None:
        raise UnsafePrivatePathError(
            f"e5 cache disappeared during provenance binding: {cache}"
        )
    if stat.S_ISLNK(actual.st_mode):
        raise UnsafePrivatePathError(
            f"e5 cache became a symlink during provenance binding: {cache}"
        )
    if (
        not stat.S_ISREG(actual.st_mode)
        or actual.st_uid != os.geteuid()
        or actual.st_nlink != 1
        or stat.S_IMODE(actual.st_mode) != 0o600
    ):
        raise UnsafePrivatePathError(
            f"e5 cache became unsafe during provenance binding: {cache}"
        )
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise UnsafePrivatePathError(
            f"e5 cache was replaced during provenance binding: {cache}"
        )
    if _cache_stat_signature(actual) != _cache_stat_signature(expected):
        raise UnsafePrivatePathError(
            f"e5 cache changed during provenance binding: {cache}"
        )


def _read_cache_bytes(descriptor, names):
    before = _validate_cache_descriptor(descriptor)
    raw_tensor_bytes = max(1, len(names)) * 2 * 384 * 4
    if before.st_size > raw_tensor_bytes * 2 + 16 * 1024 * 1024:
        return None

    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = before.st_size
    chunks = []
    while remaining:
        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    extra = os.read(descriptor, 1)
    after = _validate_cache_descriptor(descriptor)
    if (
        remaining
        or extra
        or _cache_stat_signature(before) != _cache_stat_signature(after)
    ):
        raise UnsafePrivatePathError(
            "e5 cache changed while its provenance bytes were read"
        )
    return b"".join(chunks), after


def _load_private_e5_cache_descriptor(descriptor, cache, names):
    exact = _read_cache_bytes(descriptor, names)
    if exact is None:
        _assert_cache_path_still_names(
            cache, _validate_cache_descriptor(descriptor)
        )
        return None
    cache_bytes, cache_info = exact
    try:
        payload = mu_attention.torch.load(
            io.BytesIO(cache_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except Exception:
        # The cache is regenerable; every safe-load failure is a miss.
        payload = None

    # Hash the immutable byte string that torch.load just consumed, then prove
    # the pathname still names that same singly-linked inode. A concurrent
    # rename, symlink, hard-link, or in-place mutation is an error, not a miss.
    cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    current = os.fstat(descriptor)
    if _cache_stat_signature(current) != _cache_stat_signature(cache_info):
        _assert_cache_path_still_names(cache, cache_info)
        raise UnsafePrivatePathError(
            "e5 cache changed during deserialization/provenance binding"
        )
    _assert_cache_path_still_names(cache, cache_info)

    if payload is None:
        return None
    human = [name.replace("_", " ") for name in names]
    if not _cache_payload_is_usable(payload, names, human):
        return None
    idx = {name: index for index, name in enumerate(names)}
    return payload["query"], payload["passage"], idx, cache_sha256


def _load_private_e5_cache(cache, names):
    if _lstat_or_none(cache) is None:
        return None
    descriptor = _open_private_regular(cache)
    try:
        return _load_private_e5_cache_descriptor(descriptor, cache, names)
    finally:
        os.close(descriptor)


def _as_cpu_float_tensor(table):
    if isinstance(table, mu_attention.torch.Tensor):
        tensor = table.detach().to(device="cpu", dtype=mu_attention.torch.float32)
    else:
        tensor = mu_attention.torch.as_tensor(
            table.numpy(), dtype=mu_attention.torch.float32
        )
    return tensor.contiguous()


def build_private_e5_tables(names, cache_path, batch_size=128):
    """Load/build an e5 cache through a no-follow, atomic private-file path."""
    cache = prepare_private_cache(cache_path)
    cached = _load_private_e5_cache(cache, names)
    if cached is not None:
        return cached

    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{cache.name}.build.", dir=str(cache.parent))
    )
    os.chmod(stage_dir, 0o700)
    generated = stage_dir / "generated.pt"
    sanitized = stage_dir / "install.pt"
    try:
        query, passage, _ = build_e5_tables(
            names,
            cache_path=str(generated),
            batch_size=batch_size,
            model_name=E5_MODEL,
            model_revision=E5_REVISION,
        )
        safe_query = _as_cpu_float_tensor(query)
        safe_passage = _as_cpu_float_tensor(passage)
        human = [name.replace("_", " ") for name in names]
        payload = {
            "names": list(names),
            "human": human,
            "model_name": E5_MODEL,
            "model_revision": E5_REVISION,
            "query": safe_query,
            "passage": safe_passage,
        }
        if not _cache_payload_is_usable(payload, names, human):
            raise ValueError("e5 builder returned malformed embedding tables")
        descriptor = os.open(
            sanitized,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                mu_attention.torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())

            source_directory_fd = _open_private_parent(sanitized)
            try:
                destination_directory_fd = _open_private_parent(cache)
                try:
                    os.replace(
                        sanitized.name,
                        cache.name,
                        src_dir_fd=source_directory_fd,
                        dst_dir_fd=destination_directory_fd,
                    )
                    os.fsync(source_directory_fd)
                    os.fsync(destination_directory_fd)
                finally:
                    os.close(destination_directory_fd)
            finally:
                os.close(source_directory_fd)

            installed = _load_private_e5_cache_descriptor(
                descriptor, cache, names
            )
            if installed is None:
                raise RuntimeError(
                    "newly installed e5 cache failed safe validation"
                )
            return installed
        finally:
            os.close(descriptor)
    finally:
        for artifact in (generated, sanitized):
            try:
                artifact.unlink()
            except FileNotFoundError:
                pass
        stage_dir.rmdir()


def discover_filing_rows(root, privacy_by_path, policy):
    """Derive filing rows from a source-verified privacy index.

    ``load_or_build_privacy_index`` has already re-read every source member
    through a retained no-follow root descriptor. Rewalking the path-named
    corpus here would add a second, unbound race surface, so this view is
    derived solely from those verified relative identities.
    """

    rows, counts = [], {"public": 0, "private": 0, "unknown": 0}
    del root  # identities are bound by the verified index, not a second walk
    for relative in sorted(
        privacy_by_path,
        key=lambda value: (value.casefold(), value),
    ):
        privacy = privacy_by_path[relative]
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or path.as_posix() != relative
            or any(part in ("", ".", "..") for part in path.parts)
            or path.suffix.casefold() != ".smmx"
        ):
            raise SmFsPrivacyError(
                f"privacy index contains invalid map identity: {relative!r}"
            )
        classification = privacy["classification"]
        counts[classification] += 1
        if not privacy_allows(classification, policy):
            continue
        parts = list(path.parent.parts)
        stem = path.name[:-5].strip()
        if stem and parts and parts != ["."]:  # root maps have no filing folder
            rows.append(
                {
                    "title": stem,
                    "dir": parts[-1],
                    "path": "/".join(parts),
                    "privacy": classification,
                }
            )
    rows.sort(key=lambda row: (row["path"], row["title"]))
    return rows, counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--min-maps", type=int, default=3, help="catalog floor (= PT min_bm analog)")
    ap.add_argument("--holdout-frac", type=float, default=0.40)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--ledger", default=os.path.expanduser("~/mu_data/sm_fs_ledger_v2.json"))
    ap.add_argument("--e5-cache", default=os.path.expanduser("~/mu_data/sm_fs_e5.pt"))
    ap.add_argument("--privacy-index", default=DEFAULT_PRIVACY_INDEX)
    ap.add_argument(
        "--privacy-policy",
        choices=("exclude-private", "public-only", "all-local"),
        default="exclude-private",
        help="local default excludes only private; public-only is required before external/public use",
    )
    a = ap.parse_args(argv)

    privacy_header, privacy_by_path = load_or_build_privacy_index(
        a.root, a.privacy_index
    )
    require_public_privacy_perimeter(privacy_header, a.privacy_policy)
    rows, privacy_counts = discover_filing_rows(
        a.root, privacy_by_path, a.privacy_policy
    )
    print(
        f"maps with a filing folder ({a.privacy_policy}): {len(rows)} "
        f"[privacy index: {privacy_counts}]"
    )

    # catalog: directories holding >= min_maps maps
    from collections import Counter
    per_dir = Counter(r["path"] for r in rows)
    eligible_paths = {p for p, c in per_dir.items() if c >= a.min_maps}
    rows = [r for r in rows if r["path"] in eligible_paths]
    cat_titles = sorted({r["dir"] for r in rows})
    by_title = {t: i for i, t in enumerate(cat_titles)}
    print(f"catalog: {len(cat_titles)} distinct directory names "
          f"({len(eligible_paths)} dirs >= {a.min_maps} maps); queries: {len(rows)}")

    # FROZEN HOLDOUT at first touch — deterministic by content hash, recorded before any scoring
    def qkey(r):
        return hashlib.sha256(f"{r['path']}/{r['title']}".encode()).hexdigest()
    rows.sort(key=qkey)
    rng = np.random.default_rng(a.split_seed)
    perm = rng.permutation(len(rows))
    n_hold = int(len(rows) * a.holdout_frac)
    hold_idx = set(perm[:n_hold].tolist())
    explore = [r for i, r in enumerate(rows) if i not in hold_idx]
    held_hashes = sorted(qkey(rows[i]) for i in hold_idx)

    # Prepare and bind the exact embedding artifact before sealing the ledger.
    # Ranking/scoring still happens only after the ledger is durable.
    _validate_private_target(a.ledger, must_not_exist=True)
    q_titles = [r["title"] for r in explore]
    names = sorted(set(q_titles) | set(cat_titles))
    qtbl, ptbl, idx, cache_sha256 = build_private_e5_tables(
        names, a.e5_cache, batch_size=128
    )

    ledger = {
        "corpus": "simplemind-fs-filing-v2", "root": a.root, "min_maps": a.min_maps,
        "split_seed": a.split_seed, "holdout_frac": a.holdout_frac,
        "n_total": len(rows), "n_explore": len(explore), "n_reserved": n_hold,
        "reserved_query_hashes_sha256": hashlib.sha256(
            "".join(held_hashes).encode()).hexdigest(),
        "catalog_sha256": hashlib.sha256("\n".join(cat_titles).encode()).hexdigest(),
        "privacy": {
            "policy": a.privacy_policy,
            "index_schema": privacy_header["schema"],
            "index_policy_id": privacy_header["policy_id"],
            "index_sha256": privacy_header["index_sha256"],
            "counts": privacy_counts,
        },
        "provenance": filing_provenance(cache_sha256),
        "status": "unscored transductive reserve; not a node-disjoint confirmatory holdout",
    }
    _atomic_json(a.ledger, ledger)
    print(f"ledger -> {a.ledger} (reserved {n_hold} queries, digest "
          f"{ledger['reserved_query_hashes_sha256'][:16]})")

    # e5 ranking on the EXPLORATION split only
    Q, P = qtbl.numpy(), ptbl.numpy()
    cv = np.stack([P[idx[t]] for t in cat_titles])
    ranks = []
    for r in explore:
        cos = Q[idx[r["title"]]] @ cv.T
        tp = by_title[r["dir"]]
        rk = 1 + int(np.sum(cos > cos[tp]))
        # title-equivalence: best rank over identical directory names
        for j, t in enumerate(cat_titles):
            if t == r["dir"] and j != tp:
                rk = min(rk, 1 + int(np.sum(cos > cos[j])))
        ranks.append(rk)
    rk = np.array(ranks, float)
    print(f"\nSimpleMind-FS filing (exploration split, n={len(rk)}, catalog {len(cat_titles)}):")
    print(f"  MRR {np.mean(1 / rk):.3f}  R@1 {np.mean(rk <= 1):.3f}  R@5 {np.mean(rk <= 5):.3f}  "
          f"R@50 {np.mean(rk <= 50):.3f}  med {int(np.median(rk))}")
    print("  comparators: PT 0.203/0.291 (catalog 335); SM in-map parent 0.180/0.320 (catalog 200)")


if __name__ == "__main__":
    main()
