#!/usr/bin/env python3
"""Stage 4 of registry v0.4: legacy identity inventory + migration manifest.

``DESIGN_registry_v0.4.md`` §5 requires a migration manifest over a frozen
``LegacyIdentityInventory`` (patterns doc §15.1): because ``v0.3`` sits in every
legacy digest preimage, *every* inventory item needs a migration row even where
the spelling is unchanged.

Scope discipline:

- **Legacy bytes come only from sealed sources.** The v0.3 registry no longer
  exists in code, so legacy canonical strings and digests are read from the
  sealed golden bundles and the sealed P1 preregistration — each pinned here by
  file SHA-256 — never recomputed through the current parser. The current
  parser produces only the *new* side of a mapped row.
- **``lineage(fs,…)`` artifacts are out of migration scope** (§15.1: "the
  invalid ``lineage(fs,...)`` artifact is regenerated rather than migrated").
  They were not grammatical under v0.3 — ``fs`` was unregistered — so they have
  no legacy identity to migrate. They parse under v0.4, and regeneration is
  byte-identical because ``sm_fs_freeze.py`` computes targets without the
  registry (*measured*: no ``process_cards`` import; test obligation 7).
- The §15.2 release record and the vNext version axes are NOT this stage:
  they activate with vNext (§15.3), not with the flat v0.4 registry.

Statuses (§15.1): ``mapped`` yields a proposed v0.4 AST plus receipt (factory
verification still required before deployed identity); ``ambiguous`` requires
a ruling; ``tombstoned`` preserves history and forbids promotion.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import process_cards as pc

ROOT = Path(__file__).resolve().parent

INVENTORY_SCHEMA = "unifyweaver.legacy-identity-inventory.v1"
MANIFEST_SCHEMA = "unifyweaver.identity-migration-manifest.v1"
ROW_SCHEMA = "unifyweaver.identity-migration-row.v1"

LEGACY_REGISTRY_VERSION = "v0.3"
#: v0.3 predates registry content-addressing (§15: identifiers are assigned
#: only when checked-in content is frozen), so there is no legacy registry
#: digest. Its behavioral witnesses are the sealed bundles pinned below.
LEGACY_REGISTRY_SHA256 = None
LEGACY_IDENTITY_PREIMAGE = "REGISTRY_VERSION|canonical_identity_string"
CURRENT_IDENTITY_PREIMAGE = "REGISTRY_VERSION|canonical_semantic_identity_string"

INVENTORY_PATH = ROOT / "LEGACY_IDENTITY_INVENTORY_v03.json"
MANIFEST_PATH = ROOT / "REGISTRY_V04_MIGRATION_MANIFEST.json"

#: Every file legacy bytes are read from, pinned so the inventory cannot be
#: rebuilt from silently different sources.
SEALED_SOURCES = {
    "PROCESS_EXPRESSION_GOLDEN_v1.json":
        "b053351a2a419ac58b7ab644afe15c60543846ce8b9d5a3d9bcbc332ca24db29",
    "PROCESS_EXPRESSION_GOLDEN_v2.json":
        "85e6421f5a1347fca5937d1243dc01500a9aa5b7221571b4918248e57ece6344",
    "PROCESS_EXPRESSION_P1_PREREG.json":
        "f5df1929939abe0079c2c0512234360a3a0c02699041591b9eec59f9c64e6998",
}

#: Committed artifacts that DECLARE a legacy expression without a digest.
#: The wiki-lineage header spells exactly the sealed ``lineage-graph``
#: canonical, so it deduplicates onto that identity; it is recorded as a
#: source so the row's reach is visible.
ARTIFACT_DECLARATIONS = {
    "gen_wiki_lineage_v2.py:targets-header": "lineage(graph,decay=0.85)",
}

#: ``lineage(fs,…)`` declarations: out of migration scope, regenerated.
OUT_OF_SCOPE = [
    {
        "artifact": "SM_FS_LINEAGE_RANKING_PREREG.json",
        "expression": "lineage(fs,decay=0.85)",
        "reason": (
            "not grammatical under v0.3 (fs was unregistered), so it has no "
            "legacy identity; parses under v0.4; regenerated, not migrated"
        ),
    },
    {
        "artifact": "sm_fs_freeze.py:EXPR",
        "expression": "lineage(fs,decay=0.85)",
        "reason": (
            "freeze targets are computed without the registry (no "
            "process_cards import), so regeneration under the v0.4 spelling "
            "is byte-identical (test obligation 7)"
        ),
    },
    {
        "artifact": "fine_tune_sm_fs.py:header-assertion",
        "expression": "lineage(fs,decay=0.85)",
        "reason": "consumer of the regenerated freeze artifact; no identity of its own",
    },
]

#: The rulings for legacy spellings that do not parse under v0.4. Keyed by the
#: sealed legacy canonical bytes. Everything else that parses with an
#: unchanged canonical maps mechanically.
RULINGS = {
    "lineage(graph,decay=0.85)": {
        "status": "mapped",
        "successor": 'lineage(simplewiki,mu=graph,estimand="ancestry")',
        "reason": (
            "R1/R2 supply the substrate-and-estimand ruling §15.1 required: "
            "the only committed use walks 100k_cats/category_parent.tsv, "
            "which is the SimpleWiki category graph (measured: "
            "REPORT_repeated_judge_source_regions.md source table; "
            "REPORT_repeated_judge_source_dependence.md), so the substrate is "
            "simplewiki; mu=graph names the deterministic structural "
            "mu-source the legacy spelling conflated with the substrate; the "
            "artifact's hop-decayed ancestor targets are estimand ancestry"
        ),
    },
    "lineage(graph,decay=0.85)@run/2026-07-25": {
        "status": "mapped",
        "successor": 'lineage(simplewiki,mu=graph,estimand="ancestry")',
        "provenance_pins": ["run/2026-07-25"],
        "reason": (
            "same ruling as the unpinned spelling; under R9 the pin is "
            "outside semantic identity, so two legacy identities collapse "
            "onto one v0.4 semantic identity and the pin moves to the "
            "provenance envelope"
        ),
    },
    "lineage(graph,decay=-0.5)": {
        "status": "tombstoned",
        "reason": (
            "fixture-only identity (golden coverage case neg-number): never "
            "deployed, so no fact determines its substrate; the v3 bundle "
            "carries its own negative-decay coverage row "
            "lineage(fs,decay=-0.5); promotion forbidden"
        ),
    },
}


class MigrationError(ValueError):
    """The inventory or manifest is incomplete, drifted, or inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _content_sha(document: Mapping[str, Any], digest_field: str) -> str:
    core = {k: v for k, v in document.items() if k != digest_field}
    return hashlib.sha256(_canonical_json_bytes(core)).hexdigest()


def _read_sealed(name: str) -> bytes:
    raw = (ROOT / name).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != SEALED_SOURCES[name]:
        raise MigrationError(
            f"sealed source {name} does not match its pinned sha256; "
            f"the inventory may not be built from drifted bytes"
        )
    return raw


def registry_content_sha256() -> str:
    """A content witness for the CURRENT registry's signature table.

    v0.4, like v0.3, has no separately frozen registry artifact; this digest
    is a deterministic serialization of the live signature table so migration
    rows can state *which* v0.4 they map onto. It moves whenever any
    signature moves, which is exactly what a witness should do.
    """

    table = {}
    for name, sig in sorted(pc.REGISTRY.items()):
        table[name] = {
            "atom": sig.atom,
            "min_args": sig.min_args,
            "max_args": sig.max_args,
            "arg_types": list(sig.arg_types),
            "variadic_arg_type": sig.variadic_arg_type,
            "kwargs": {
                key: {"kind": spec.kind, "default": spec.default,
                      "required": spec.required}
                for key, spec in sorted(sig.kwargs.items())
            },
            "output": sig.output,
            "modifiers": sorted(sig.modifiers),
        }
    payload = {"registry_version": pc.REGISTRY_VERSION, "signatures": table}
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------


def build_inventory() -> dict[str, Any]:
    items: dict[str, dict[str, Any]] = {}

    def add(digest: str, canonical: str, source: str):
        item = items.setdefault(
            digest, {"identity_key": digest, "canonical": canonical, "sources": []}
        )
        if item["canonical"] != canonical:
            raise MigrationError(
                f"digest {digest} claimed by two different canonical strings"
            )
        if source not in item["sources"]:
            item["sources"].append(source)

    for bundle in ("PROCESS_EXPRESSION_GOLDEN_v1.json",
                   "PROCESS_EXPRESSION_GOLDEN_v2.json"):
        document = json.loads(_read_sealed(bundle).decode("utf-8"))
        if document.get("registry_version") != LEGACY_REGISTRY_VERSION:
            raise MigrationError(f"{bundle} is not a v0.3 bundle")
        for row in document["rows"]:
            add(row["full_process_digest"], row["canonical_identity_string"],
                f"{bundle}:{row['name']}")

    prereg = json.loads(_read_sealed("PROCESS_EXPRESSION_P1_PREREG.json").decode("utf-8"))
    for name, record in sorted(prereg["process_identity"]["processes"].items()):
        add(record["sha256"], record["canonical"],
            f"PROCESS_EXPRESSION_P1_PREREG.json:{name}")

    # Artifact declarations carry no digest; they attach to the sealed
    # identity whose canonical bytes they spell, and may not mint new ones.
    by_canonical = {item["canonical"]: item for item in items.values()}
    for source, expression in sorted(ARTIFACT_DECLARATIONS.items()):
        item = by_canonical.get(expression)
        if item is None:
            raise MigrationError(
                f"artifact declaration {source!r} spells an expression with no "
                f"sealed identity: {expression!r}"
            )
        if source not in item["sources"]:
            item["sources"].append(source)

    document = {
        "schema": INVENTORY_SCHEMA,
        "legacy_registry_version": LEGACY_REGISTRY_VERSION,
        "legacy_registry_sha256": LEGACY_REGISTRY_SHA256,
        "legacy_identity_preimage": LEGACY_IDENTITY_PREIMAGE,
        "sealed_sources": dict(sorted(SEALED_SOURCES.items())),
        "items": [items[key] for key in sorted(items)],
        "out_of_scope": OUT_OF_SCOPE,
        "note": (
            "Legacy canonical bytes and digests are copied from the sealed "
            "sources above, never recomputed through the current parser. "
            "out_of_scope entries have no legacy identity and are "
            "regenerated, not migrated."
        ),
    }
    document["inventory_sha256"] = _content_sha(document, "inventory_sha256")
    return document


def verify_inventory(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("schema") != INVENTORY_SCHEMA:
        raise MigrationError("unsupported inventory schema")
    if _content_sha(document, "inventory_sha256") != document.get("inventory_sha256"):
        raise MigrationError("inventory_sha256 does not bind the inventory content")
    rebuilt = build_inventory()
    if rebuilt["inventory_sha256"] != document["inventory_sha256"]:
        raise MigrationError(
            "inventory drifted from its sealed sources; rebuilding produces "
            "different content"
        )
    return dict(document)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def _new_side(successor_expression: str) -> tuple[str, str]:
    node = pc.parse(successor_expression)
    semantic = pc.canonical_semantic(node)
    digest = hashlib.sha256(
        f"{pc.REGISTRY_VERSION}|{semantic}".encode("utf-8")
    ).hexdigest()
    return semantic, digest


def _migration_row(item: Mapping[str, Any], to_registry_sha: str) -> dict[str, Any]:
    canonical = item["canonical"]
    row = {
        "schema": ROW_SCHEMA,
        "from_registry_version": LEGACY_REGISTRY_VERSION,
        "from_registry_sha256": LEGACY_REGISTRY_SHA256,
        "from_identity_preimage": LEGACY_IDENTITY_PREIMAGE,
        "old_canonical_bytes": canonical,
        "old_full_digest": item["identity_key"],
        "old_factory_fingerprint": None,
        "to_registry_version": pc.REGISTRY_VERSION,
        "to_registry_sha256": to_registry_sha,
        "to_identity_preimage": CURRENT_IDENTITY_PREIMAGE,
        "new_canonical_bytes": None,
        "new_semantic_digest": None,
        "provenance_pins": [],
        "predecessor_kind": "semantic",
        "predecessor_identity_key": item["identity_key"],
        "sources": list(item["sources"]),
    }

    ruling = RULINGS.get(canonical)
    if ruling is not None:
        row["status"] = ruling["status"]
        row["reason"] = ruling["reason"]
        row["provenance_pins"] = list(ruling.get("provenance_pins", []))
        if ruling["status"] == "mapped":
            semantic, digest = _new_side(ruling["successor"])
            row["new_canonical_bytes"] = semantic
            row["new_semantic_digest"] = digest
        return row

    # No ruling needed: the legacy spelling must parse under v0.4 with its
    # canonical bytes unchanged. Anything else is a missing ruling, which is
    # a hard error — never a silent default (§16.15 posture).
    try:
        node = pc.parse(canonical)
    except pc.ParseError as exc:
        raise MigrationError(
            f"legacy identity {canonical!r} does not parse under "
            f"{pc.REGISTRY_VERSION} and has no ruling"
        ) from exc
    if pc.canonical_full(node) != canonical:
        raise MigrationError(
            f"legacy identity {canonical!r} re-canonicalizes differently under "
            f"{pc.REGISTRY_VERSION}; it needs an explicit ruling"
        )
    row["status"] = "mapped"
    row["reason"] = (
        "spelling unchanged under v0.4; the digest still moves because "
        "REGISTRY_VERSION is in the identity preimage (§5)"
    )
    semantic, digest = _new_side(canonical)
    row["new_canonical_bytes"] = semantic
    row["new_semantic_digest"] = digest
    # Pins in the legacy spelling were identity-bearing; under v0.4 they are
    # provenance. (No unchanged-spelling legacy row carries pins today; this
    # keeps the invariant explicit rather than assumed.)
    if node.pins:
        row["provenance_pins"] = list(node.pins)
    return row


def build_manifest(inventory: Mapping[str, Any]) -> dict[str, Any]:
    to_registry_sha = registry_content_sha256()
    rows = [
        _migration_row(item, to_registry_sha) for item in inventory["items"]
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    document = {
        "schema": MANIFEST_SCHEMA,
        "inventory_sha256": inventory["inventory_sha256"],
        "to_registry_version": pc.REGISTRY_VERSION,
        "to_registry_sha256": to_registry_sha,
        "rows": rows,
        "status_counts": dict(sorted(counts.items())),
        "note": (
            "mapped rows are proposed v0.4 semantic identities plus this "
            "receipt; factory verification is still required before any "
            "deployed identity (§15.1). tombstoned rows forbid promotion."
        ),
    }
    document["manifest_sha256"] = _content_sha(document, "manifest_sha256")
    return document


def verify_manifest(
    document: Mapping[str, Any], inventory: Mapping[str, Any]
) -> dict[str, Any]:
    if document.get("schema") != MANIFEST_SCHEMA:
        raise MigrationError("unsupported manifest schema")
    if _content_sha(document, "manifest_sha256") != document.get("manifest_sha256"):
        raise MigrationError("manifest_sha256 does not bind the manifest content")
    if document.get("inventory_sha256") != inventory.get("inventory_sha256"):
        raise MigrationError("manifest is bound to a different inventory")

    rows = document.get("rows")
    if not isinstance(rows, list):
        raise MigrationError("manifest has no rows")
    by_old = {}
    for row in rows:
        if row.get("schema") != ROW_SCHEMA:
            raise MigrationError("row has an unsupported schema")
        if row.get("status") not in ("mapped", "ambiguous", "tombstoned"):
            raise MigrationError(f"illegal status: {row.get('status')!r}")
        if row["old_full_digest"] in by_old:
            raise MigrationError(f"duplicate row for {row['old_full_digest']}")
        by_old[row["old_full_digest"]] = row

    item_keys = {item["identity_key"] for item in inventory["items"]}
    if set(by_old) != item_keys:
        missing = sorted(item_keys - set(by_old))
        extra = sorted(set(by_old) - item_keys)
        raise MigrationError(
            f"manifest coverage mismatch: missing={missing} extra={extra}"
        )

    for row in rows:
        if row["status"] == "mapped":
            semantic, digest = _new_side(row["new_canonical_bytes"])
            if semantic != row["new_canonical_bytes"]:
                raise MigrationError(
                    f"mapped successor is not canonical: {row['new_canonical_bytes']!r}"
                )
            if digest != row["new_semantic_digest"]:
                raise MigrationError(
                    f"new_semantic_digest does not match its own bytes for "
                    f"{row['old_canonical_bytes']!r}"
                )
        else:
            if row.get("new_canonical_bytes") or row.get("new_semantic_digest"):
                raise MigrationError(
                    f"{row['status']} row must not carry new identity bytes"
                )
    return dict(document)


def load_and_verify() -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = verify_inventory(
        json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    )
    manifest = verify_manifest(
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), inventory
    )
    return inventory, manifest


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="write the inventory and manifest files")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing files (a bound manifest must never be rewritten)")
    args = parser.parse_args(argv)

    inventory = build_inventory()
    manifest = build_manifest(inventory)
    verify_manifest(manifest, inventory)
    print(f"inventory: {len(inventory['items'])} items "
          f"({len(inventory['out_of_scope'])} out of scope)  "
          f"sha {inventory['inventory_sha256'][:16]}…")
    print(f"manifest:  {manifest['status_counts']}  "
          f"sha {manifest['manifest_sha256'][:16]}…")

    if args.write:
        for path, document in ((INVENTORY_PATH, inventory), (MANIFEST_PATH, manifest)):
            if path.exists() and not args.force:
                raise SystemExit(f"refusing to overwrite {path.name}; it is bound once written")
            path.write_bytes(
                json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
                .encode("utf-8") + b"\n"
            )
            print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
