#!/usr/bin/env python3
"""Fail-closed privacy index for harvested Pearltrees filing data.

The filing tools may send bookmark and folder titles to a hosted judge.  They
therefore need stronger evidence than "not explicitly private": a node is
eligible only when the harvest contains an explicit public claim, no private
claim or private-title marker, and no private/unknown containment ancestor.

Unknown nodes are quarantined, not relabelled private.  The distinction is
preserved in the manifest while both classes remain ineligible for external
tasks.  See ``privacy.py`` for the repository-wide scrub-everywhere rule.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from privacy import is_private_title, propagate, vis_public


POLICY_ID = "pearltrees-public-only-v1"
PUBLIC_CATALOG_POLICY_ID = "pearltrees-public-alphanumeric-title-v1"
SCHEMA = "unifyweaver.pearltrees-privacy-index.v1"


class FilingPrivacyError(ValueError):
    """The harvested snapshot cannot be certified for public-only use."""


def public_catalog_title_eligible(title: Any) -> bool:
    """Outcome-blind folder-title eligibility shared by eval and suggestion."""
    return isinstance(title, str) and any(character.isalnum() for character in title)


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise FilingPrivacyError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _strict_json(data: bytes, source: str):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FilingPrivacyError(f"malformed JSON in {source}: {exc}") from exc


def canonical_tree_id(value: Any) -> str:
    """Return a decimal tree id, rejecting bools and malformed identifiers."""
    if isinstance(value, bool) or value is None:
        raise FilingPrivacyError(f"invalid tree id: {value!r}")
    text = str(value).strip()
    if not text.isdecimal():
        raise FilingPrivacyError(f"invalid tree id: {value!r}")
    return str(int(text))


def visibility_claim(value: Any) -> str:
    """Map a Pearltrees visibility value to public/private/unknown.

    Canonical integer-like values are accepted.  A malformed nonempty value is
    not silently quarantined because it may reflect a changed upstream schema.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return "unknown"
    if isinstance(value, bool):
        raise FilingPrivacyError(f"malformed visibility: {value!r}")
    text = str(value).strip()
    try:
        numeric = int(text)
    except ValueError as exc:
        raise FilingPrivacyError(f"malformed visibility: {value!r}") from exc
    if str(numeric) != text:
        raise FilingPrivacyError(f"noncanonical visibility: {value!r}")
    return "public" if vis_public(value) else "private"


def merge_visibility_claims(claims) -> str:
    claims = tuple(claims)
    if any(claim == "private" for claim in claims):
        return "private"
    if any(claim == "public" for claim in claims):
        return "public"
    return "unknown"


@dataclass(frozen=True)
class PearltreesPrivacyIndex:
    status_by_id: Mapping[str, str]
    reason_by_id: Mapping[str, tuple[str, ...]]
    public_ids: frozenset[str]
    private_ids: frozenset[str]
    quarantined_ids: frozenset[str]
    manifest_sha256: str
    source_snapshot: Mapping[str, Any]
    tree_payloads: Mapping[str, Mapping[str, Any]]
    tree_value_ids: Mapping[str, Any]
    tree_file_sha256: Mapping[str, str]
    public_title_by_id: Mapping[str, str]

    def receipt(self) -> dict[str, Any]:
        """Non-sensitive aggregate record suitable for a hosted-task header."""
        return {
            "schema": SCHEMA,
            "policy_id": POLICY_ID,
            "manifest_sha256": self.manifest_sha256,
            "source_snapshot": dict(self.source_snapshot),
            "counts": {
                "public": len(self.public_ids),
                "private": len(self.private_ids),
                "quarantined": len(self.quarantined_ids),
                "observed": len(self.status_by_id),
            },
        }


def _infer_paths_jsonl(trees_dir: Path) -> Path | None:
    # .../.local/data/pearltrees_api/trees -> .../.local/data/api_tree_paths_v8.jsonl
    candidate = trees_dir.parent.parent / "api_tree_paths_v8.jsonl"
    return candidate if candidate.is_file() else None


def build_pearltrees_privacy_index(
    trees_dir: str | os.PathLike[str],
    *,
    paths_jsonl: str | os.PathLike[str] | None = None,
) -> PearltreesPrivacyIndex:
    """Certify the public subset of a Pearltrees API snapshot.

    Containment is limited to typed API parent/collection edges and validated
    materialized paths.  Shortcut/alias/cross-link pearls are deliberately not
    privacy-propagation edges.
    """
    root = Path(trees_dir)
    if not root.is_dir():
        raise FilingPrivacyError(f"trees directory does not exist: {root}")
    path_file = Path(paths_jsonl) if paths_jsonl is not None else _infer_paths_jsonl(root)

    observations: dict[str, list[str]] = defaultdict(list)
    title_private: dict[str, bool] = defaultdict(bool)
    title_claims: dict[str, list[tuple[int, str]]] = defaultdict(list)
    reasons: dict[str, set[str]] = defaultdict(set)
    children: dict[str, set[str]] = defaultdict(set)
    anchored_nodes: set[str] = set()
    explicit_user_roots: set[str] = set()
    tree_payloads: dict[str, Mapping[str, Any]] = {}
    tree_value_ids: dict[str, Any] = {}
    tree_file_sha256: dict[str, str] = {}
    inventory: list[dict[str, Any]] = []

    def observe(raw_id, title, visibility, source):
        tid = canonical_tree_id(raw_id)
        claim = visibility_claim(visibility)
        observations[tid].append(claim)
        reasons[tid].add(f"{source}:{claim}")
        if isinstance(title, str) and title:
            priority = {"tree": 0, "parentTree": 1, "contentTree": 2}.get(source, 9)
            title_claims[tid].append((priority, title))
        if is_private_title(title):
            title_private[tid] = True
            reasons[tid].add(f"{source}:private-title")
        return tid

    # The directory also carries queue/priority metadata JSON.  Only numeric
    # names are per-tree source records; malformed *numeric* records fail below.
    files = sorted(
        (path for path in root.glob("*.json") if path.stem.isdecimal()),
        key=lambda path: path.name,
    )
    if not files:
        raise FilingPrivacyError(f"no harvested tree JSON files in {root}")
    for path in files:
        raw = path.read_bytes()
        digest = _sha256(raw)
        inventory.append(
            {
                "name": f"trees/{path.name}",
                "size_bytes": len(raw),
                "sha256": digest,
            }
        )
        doc = _strict_json(raw, str(path))
        if not isinstance(doc, dict):
            raise FilingPrivacyError(f"tree file is not an object: {path}")
        api = doc.get("api_response")
        if not isinstance(api, dict):
            raise FilingPrivacyError(f"missing api_response object: {path}")
        tree = api.get("tree")
        if not isinstance(tree, dict):
            raise FilingPrivacyError(f"missing tree object: {path}")
        tid = observe(tree.get("id"), tree.get("title"), tree.get("visibility"), "tree")
        if tid != canonical_tree_id(path.stem):
            raise FilingPrivacyError(
                f"filename/payload tree id mismatch: {path.name} != {tree.get('id')!r}"
            )
        if tid in tree_payloads:
            raise FilingPrivacyError(f"duplicate harvested tree id: {tid}")
        tree_payloads[tid] = tree
        tree_value_ids[tid] = tree.get("id")
        tree_file_sha256[tid] = digest
        if tree.get("isUserRoot") in (1, "1", True):
            explicit_user_roots.add(tid)

        info = api.get("info")
        parent = info.get("parentTree") if isinstance(info, dict) else None
        if isinstance(parent, dict) and parent.get("id") is not None:
            pid = observe(
                parent.get("id"),
                parent.get("title"),
                parent.get("visibility"),
                "parentTree",
            )
            if pid != tid:
                children[pid].add(tid)

        pearls = tree.get("pearls")
        if pearls is None:
            pearls = []
        if not isinstance(pearls, list):
            raise FilingPrivacyError(f"tree.pearls is not a list: {path}")
        for pearl in pearls:
            if not isinstance(pearl, dict) or str(pearl.get("contentType")) != "2":
                continue
            child = pearl.get("contentTree")
            if not isinstance(child, dict) or child.get("id") is None:
                raise FilingPrivacyError(f"collection pearl lacks contentTree id: {path}")
            cid = observe(
                child.get("id"),
                child.get("title") or pearl.get("title"),
                child.get("visibility"),
                "contentTree",
            )
            if is_private_title(pearl.get("title")):
                title_private[cid] = True
                reasons[cid].add("collectionPearl:private-title")
            if cid != tid:
                children[tid].add(cid)

    if path_file is None:
        raise FilingPrivacyError(
            "api_tree_paths_v8.jsonl is required for public-only ancestry certification"
        )
    if path_file is not None:
        raw_paths = path_file.read_bytes()
        inventory.append(
            {
                "name": path_file.name,
                "size_bytes": len(raw_paths),
                "sha256": _sha256(raw_paths),
            }
        )
        for line_no, line in enumerate(raw_paths.splitlines(), 1):
            if not line.strip():
                continue
            row = _strict_json(line, f"{path_file}:{line_no}")
            if not isinstance(row, dict):
                raise FilingPrivacyError(f"path row is not an object: {path_file}:{line_no}")
            raw_ids = row.get("path_ids")
            if not isinstance(raw_ids, list):
                raise FilingPrivacyError(f"path_ids is not a list: {path_file}:{line_no}")
            target = canonical_tree_id(row.get("tree_id"))
            account_rooted = bool(
                raw_ids
                and isinstance(raw_ids[0], str)
                and raw_ids[0].startswith("account:")
            )
            numeric_user_root = bool(
                len(raw_ids) == 1
                and row.get("parent_tree_id") is None
                and canonical_tree_id(raw_ids[0]) == target
            )
            if not account_rooted and not numeric_user_root:
                raise FilingPrivacyError(
                    f"path row lacks an account root: {path_file}:{line_no}"
                )
            if row.get("path_depth") is not None:
                depth = row["path_depth"]
                if isinstance(depth, bool) or not isinstance(depth, int) or depth != len(raw_ids):
                    raise FilingPrivacyError(
                        f"path_depth mismatch at {path_file}:{line_no}"
                    )
            ids = []
            for raw_id in raw_ids:
                text = str(raw_id)
                if text.startswith("account:"):
                    continue
                ids.append(canonical_tree_id(raw_id))
            if not ids:
                raise FilingPrivacyError(f"path row has no numeric ids: {path_file}:{line_no}")
            if ids[-1] != target:
                raise FilingPrivacyError(
                    f"path target mismatch at {path_file}:{line_no}: {ids[-1]} != {target}"
                )
            parent = row.get("parent_tree_id")
            if parent is not None:
                if len(ids) < 2:
                    raise FilingPrivacyError(
                        f"path parent has no preceding id: {path_file}:{line_no}"
                    )
                if canonical_tree_id(parent) != ids[-2]:
                    raise FilingPrivacyError(f"path parent mismatch: {path_file}:{line_no}")
            for parent_id, child_id in zip(ids, ids[1:]):
                if parent_id != child_id:
                    children[parent_id].add(child_id)
            anchored_nodes.update(ids)
            # A private marker anywhere in the materialized display path means
            # the target is below a private ancestor.  When the display lines
            # align with path_ids, seed the exact marked numeric node; otherwise
            # conservatively seed the target.
            if is_private_title(row.get("title")) or is_private_title(row.get("target_text")):
                display = []
                for line in str(row.get("target_text") or "").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("- "):
                        display.append(stripped[2:].strip())
                marked = False
                if len(display) == len(raw_ids):
                    for raw_id, title in zip(raw_ids, display):
                        if (
                            not str(raw_id).startswith("account:")
                            and is_private_title(title)
                        ):
                            marked_id = canonical_tree_id(raw_id)
                            title_private[marked_id] = True
                            reasons[marked_id].add("path:private-marker")
                            marked = True
                if not marked:
                    title_private[target] = True
                    reasons[target].add("path:private-marker")

    direct_private = {
        tid
        for tid, claims in observations.items()
        if merge_visibility_claims(claims) == "private" or title_private[tid]
    }
    direct_unknown = {
        tid
        for tid, claims in observations.items()
        if tid not in direct_private and merge_visibility_claims(claims) == "unknown"
    }
    # Paths can mention nodes absent from API summaries.  Such nodes are
    # uncertified and quarantine their descendants.
    all_nodes = set(observations) | set(children)
    for values in children.values():
        all_nodes.update(values)
    direct_unknown.update(all_nodes - set(observations) - direct_private)
    ancestry_reachable = propagate(anchored_nodes | explicit_user_roots, children)
    direct_unknown.update(
        tid
        for tid, claims in observations.items()
        if merge_visibility_claims(claims) == "public"
        and tid not in ancestry_reachable
        and tid not in direct_private
    )
    for tid in direct_unknown:
        if tid in observations and merge_visibility_claims(observations[tid]) == "public":
            reasons[tid].add("uncertified-ancestry")

    private_closure = propagate(direct_private, children)
    quarantine_closure = propagate(direct_unknown, children) - private_closure
    public_ids = {
        tid
        for tid, claims in observations.items()
        if merge_visibility_claims(claims) == "public"
        and tid not in private_closure
        and tid not in quarantine_closure
    }

    status_by_id = {}
    reason_by_id = {}
    for tid in sorted(all_nodes, key=lambda value: int(value)):
        if tid in private_closure:
            status = "private"
            if tid not in direct_private:
                reasons[tid].add("private-ancestor")
        elif tid in quarantine_closure:
            status = "unknown"
            if tid not in direct_unknown:
                reasons[tid].add("unknown-ancestor")
        elif tid in public_ids:
            status = "public"
        else:
            status = "unknown"
            reasons[tid].add("no-public-evidence")
        status_by_id[tid] = status
        reason_by_id[tid] = tuple(sorted(reasons[tid]))

    decision_rows = [
        {
            "tree_id": tid,
            "status": status_by_id[tid],
            "reasons": list(reason_by_id[tid]),
        }
        for tid in sorted(status_by_id, key=lambda value: int(value))
    ]
    inventory = sorted(inventory, key=lambda row: row["name"])
    source_snapshot = {
        "schema": "unifyweaver.pearltrees-source-snapshot.v1",
        "file_count": len(inventory),
        "total_size_bytes": sum(row["size_bytes"] for row in inventory),
        "members_sha256": _sha256(b"".join(canonical_json_bytes(row) for row in inventory)),
    }
    manifest_core = {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "source_snapshot": source_snapshot,
        "decisions_sha256": _sha256(
            b"".join(canonical_json_bytes(row) for row in decision_rows)
        ),
    }
    manifest_sha256 = _sha256(canonical_json_bytes(manifest_core))
    public_title_by_id = {
        tid: sorted(title_claims[tid], key=lambda item: (item[0], item[1]))[0][1]
        for tid in public_ids
        if title_claims.get(tid)
    }
    return PearltreesPrivacyIndex(
        status_by_id=status_by_id,
        reason_by_id=reason_by_id,
        public_ids=frozenset(public_ids),
        private_ids=frozenset(private_closure),
        quarantined_ids=frozenset(quarantine_closure),
        manifest_sha256=manifest_sha256,
        source_snapshot=source_snapshot,
        tree_payloads=tree_payloads,
        tree_value_ids=tree_value_ids,
        tree_file_sha256=tree_file_sha256,
        public_title_by_id=public_title_by_id,
    )
