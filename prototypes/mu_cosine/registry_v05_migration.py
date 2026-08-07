#!/usr/bin/env python3
"""Registry v0.4 -> v0.5 migration: purely additive, 1:1, no tombstones.

v0.5 adds the `enwiki` substrate atom, the `cowalk` operator, and the
walk/weight enumerated kinds. No existing entry changed meaning, so every
v0.4 identity maps to exactly one v0.5 successor whose SEMANTIC BYTES are
identical — only the digest moves, because REGISTRY_VERSION is in every
preimage (the §5 rule the v0.3->v0.4 migration established). This is the
simplest migration the machinery will ever run, and it exists because the
alternative — growing v0.4 in place — would let two registries share a
label while disagreeing on what is grammatical.

Inventory sources are SEALED BYTES ONLY: the superseded v3 golden bundle's
rows (the canonical v0.4 identity set) and the v0.3->v0.4 manifest's mapped
successors. Nothing is read from live code, so the inventory cannot drift
with the registry it documents.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import process_cards as pc

ROOT = Path(__file__).resolve().parent

FROM_REGISTRY_VERSION = "v0.4"
TO_REGISTRY_VERSION = "v0.5"

#: The sealed v0.4-era sources this migration reads, pinned by digest so a
#: mutated source is detected before any row is trusted.
SEALED_SOURCES = {
    "PROCESS_EXPRESSION_GOLDEN_v3.json":
        "90cc484021150aa9916be2f8c4fdb57b66f3a2e7d18dafff7e40c3c566af8ef7",
    # External review (M3): two of the v0.3->v0.4 manifest's mapped
    # successors are not bundle rows, so a bundle-only inventory left them
    # outside the byte-stability net. Both sealed sources are unioned.
    "REGISTRY_V04_MIGRATION_MANIFEST.json":
        "709c3eb499e1e4ff204e4da3619f6294eb02a2f36515fc5f536a5b1445583cc0",
}

MANIFEST_PATH = ROOT / "REGISTRY_V05_MIGRATION_MANIFEST.json"

#: What v0.5 added — the reverse-containment witness. The additive claim is
#: checked, not asserted: v0.5's name set minus these must equal v0.4's.
ADDED_NAMES = ("cowalk", "enwiki")
ADDED_VALUE_KINDS = ("walk", "weight")


class MigrationError(ValueError):
    """The migration cannot verify its sources or rows; failing closed."""


def _digest(version: str, semantic: str) -> str:
    return hashlib.sha256(f"{version}|{semantic}".encode("utf-8")).hexdigest()


def _sealed(name: str) -> Mapping[str, Any]:
    raw = (ROOT / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SEALED_SOURCES[name]:
        raise MigrationError(f"sealed source {name} does not match its pin")
    return json.loads(raw.decode("utf-8"))


def build_manifest() -> dict[str, Any]:
    bundle = _sealed("PROCESS_EXPRESSION_GOLDEN_v3.json")
    if bundle.get("registry_version") != FROM_REGISTRY_VERSION:
        raise MigrationError("sealed bundle is not a v0.4 artifact")
    v04_manifest = _sealed("REGISTRY_V04_MIGRATION_MANIFEST.json")
    if v04_manifest.get("to_registry_version") != FROM_REGISTRY_VERSION:
        raise MigrationError("sealed v0.4 manifest does not target v0.4")
    identities = [row["canonical_identity_string"] for row in bundle["rows"]]
    identities += [row["new_canonical_bytes"] for row in v04_manifest["rows"]
                   if row["status"] == "mapped"]
    rows = []
    seen: set = set()
    for semantic in identities:
        if semantic in seen:
            continue
        seen.add(semantic)
        # The additive property, checked per row: the v0.4 bytes parse under
        # v0.5 and canonicalize to the SAME bytes.
        node = pc.parse(semantic)
        if pc.canonical_semantic(node) != semantic:
            raise MigrationError(
                f"{semantic!r} is not byte-stable under {TO_REGISTRY_VERSION}; "
                "the additive claim is false and this migration is the wrong "
                "tool — fail closed"
            )
        rows.append({
            "canonical_semantic": semantic,
            "old_semantic_digest": _digest(FROM_REGISTRY_VERSION, semantic),
            "new_semantic_digest": _digest(TO_REGISTRY_VERSION, semantic),
            "status": "mapped",
        })
    rows.sort(key=lambda r: r["canonical_semantic"])
    manifest = {
        "schema": "unifyweaver.registry-migration.v2",
        "from_registry_version": FROM_REGISTRY_VERSION,
        "to_registry_version": TO_REGISTRY_VERSION,
        "additive": True,
        "added_names": list(ADDED_NAMES),
        "added_value_kinds": list(ADDED_VALUE_KINDS),
        "status_counts": {"mapped": len(rows), "tombstoned": 0},
        "rows": rows,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def verify_manifest(manifest: Mapping[str, Any]) -> None:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest["manifest_sha256"]:
        raise MigrationError("manifest_sha256 does not match the manifest body")
    # External review (M2): a self-consistent hash over a WRONG header is
    # still wrong — the verifier binds schema and both version labels, not
    # only the body's internal consistency.
    if manifest.get("schema") != "unifyweaver.registry-migration.v2":
        raise MigrationError("unknown migration manifest schema")
    if manifest.get("from_registry_version") != FROM_REGISTRY_VERSION:
        raise MigrationError("manifest does not migrate FROM v0.4")
    if manifest.get("to_registry_version") != TO_REGISTRY_VERSION:
        raise MigrationError("manifest does not migrate TO v0.5")
    if manifest["status_counts"]["tombstoned"] != 0:
        raise MigrationError("an additive migration cannot tombstone")
    for row in manifest["rows"]:
        if row["old_semantic_digest"] == row["new_semantic_digest"]:
            raise MigrationError(
                f"{row['canonical_semantic']!r}: digest did not move, but "
                "REGISTRY_VERSION is in the preimage (§5)"
            )


def load_and_verify() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    return manifest


def main() -> int:
    manifest = build_manifest()
    if MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH.name} exists; sealed manifests are "
                         "never overwritten")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True)
                             + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.name}: {len(manifest['rows'])} rows, "
          f"sha {manifest['manifest_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
