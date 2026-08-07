#!/usr/bin/env python3
"""Registry v0.4 stage-4 test obligations (``DESIGN_registry_v0.4.md`` §8, 6-9).

Obligations 1-5 and 10-13 landed with stages 1-3 in
``test_registry_v04_obligations.py``; this file completes the set over the
committed inventory and manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc
import registry_v04_migration as mig
from registry_v04_migration import (
    INVENTORY_PATH,
    MANIFEST_PATH,
    MigrationError,
    build_inventory,
    build_manifest,
    load_and_verify,
    verify_manifest,
)


@pytest.fixture(scope="module")
def committed():
    inventory, manifest = load_and_verify()
    return inventory, manifest


# --------------------------------------------------------------------------
# obligation 6: total coverage, legal statuses, fs out of scope
# --------------------------------------------------------------------------


def test_obligation_6_every_inventory_item_has_exactly_one_row(committed):
    inventory, manifest = committed
    item_keys = {item["identity_key"] for item in inventory["items"]}
    row_keys = [row["old_full_digest"] for row in manifest["rows"]]
    assert sorted(row_keys) == sorted(item_keys)
    assert len(row_keys) == len(set(row_keys))
    assert len(item_keys) == 20  # measured: the deduplicated legacy set


def test_obligation_6_statuses_are_only_the_three_legal_ones(committed):
    _, manifest = committed
    assert {row["status"] for row in manifest["rows"]} <= {
        "mapped", "ambiguous", "tombstoned"
    }
    assert manifest["status_counts"] == {"mapped": 19, "tombstoned": 1}


def test_obligation_6_lineage_fs_rows_are_out_of_scope_not_migrated(committed):
    inventory, manifest = committed
    for row in manifest["rows"]:
        assert "lineage(fs" not in row["old_canonical_bytes"]
    scoped_out = {entry["expression"] for entry in inventory["out_of_scope"]}
    assert scoped_out == {"lineage(fs,decay=0.85)"}
    artifacts = {entry["artifact"] for entry in inventory["out_of_scope"]}
    assert "SM_FS_LINEAGE_RANKING_PREREG.json" in artifacts
    assert "sm_fs_freeze.py:EXPR" in artifacts


def test_obligation_6_every_digest_moves_even_for_unchanged_spellings(committed):
    """§5: REGISTRY_VERSION is in the preimage, so no row may alias digests."""
    _, manifest = committed
    for row in manifest["rows"]:
        if row["status"] == "mapped":
            assert row["new_semantic_digest"] != row["old_full_digest"]


def test_obligation_6_r9_collapse_pinned_and_unpinned_share_the_successor(committed):
    _, manifest = committed
    by_old = {row["old_canonical_bytes"]: row for row in manifest["rows"]}
    plain = by_old["lineage(graph,decay=0.85)"]
    pinned = by_old["lineage(graph,decay=0.85)@run/2026-07-25"]
    assert plain["old_full_digest"] != pinned["old_full_digest"]  # v0.3: two identities
    assert plain["new_semantic_digest"] == pinned["new_semantic_digest"]  # v0.4: one
    assert pinned["provenance_pins"] == ["run/2026-07-25"]
    assert plain["provenance_pins"] == []


def test_obligation_6_tombstone_carries_no_new_identity(committed):
    _, manifest = committed
    (tomb,) = [row for row in manifest["rows"] if row["status"] == "tombstoned"]
    assert tomb["old_canonical_bytes"] == "lineage(graph,decay=-0.5)"
    assert tomb["new_canonical_bytes"] is None
    assert tomb["new_semantic_digest"] is None


def test_manifest_is_reproducible_and_tamper_evident(committed):
    inventory, manifest = committed
    assert build_manifest(build_inventory())["manifest_sha256"] == (
        manifest["manifest_sha256"]
    )
    tampered = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tampered["rows"][0]["status"] = "ambiguous"
    with pytest.raises(MigrationError, match="manifest_sha256"):
        verify_manifest(tampered, inventory)


def test_mapped_successors_are_canonical_v04_identities(committed):
    _, manifest = committed
    for row in manifest["rows"]:
        if row["status"] != "mapped":
            continue
        node = pc.parse(row["new_canonical_bytes"])
        assert pc.canonical_semantic(node) == row["new_canonical_bytes"]
        # The sealed migration's successors are v0.4 identities; the digest
        # preimage uses the migration's pinned TARGET version, not the live
        # registry (which has since moved to v0.5).
        expected = hashlib.sha256(
            f"{mig.TARGET_REGISTRY_VERSION}|{row['new_canonical_bytes']}".encode()
        ).hexdigest()
        assert row["new_semantic_digest"] == expected


def test_inventory_reads_only_pinned_sealed_bytes(committed):
    inventory, _ = committed
    for name, digest in inventory["sealed_sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


# --------------------------------------------------------------------------
# obligation 7: lineage(fs) regeneration is byte-identical
# --------------------------------------------------------------------------


def test_obligation_7_freeze_computes_targets_without_the_registry():
    """The regeneration claim (§5) rests on a measurable fact: sm_fs_freeze
    never imports the registry, so a registry version bump cannot change a
    single byte it writes."""
    source = (ROOT / "sm_fs_freeze.py").read_text(encoding="utf-8")
    assert "process_cards" not in source
    assert "process_identity" not in source


def test_obligation_7_freeze_expression_is_grammatical_under_v04():
    # sm_fs_freeze imports the embedding stack; skip where torch is absent
    # (CI runs the source-scan obligation-7 test above unconditionally).
    pytest.importorskip("torch")
    import sm_fs_freeze

    node = pc.parse(sm_fs_freeze.EXPR)
    # The spelling IS the canonical form, so headers written before and after
    # the bump are byte-identical.
    assert pc.canonical_full(node) == sm_fs_freeze.EXPR == "lineage(fs,decay=0.85)"


# --------------------------------------------------------------------------
# obligation 8: enumerable lineage forms drop from 15 to the substrate set
# --------------------------------------------------------------------------


def test_obligation_8_lineage_argument_forms_are_the_substrate_set():
    declared = pc.REGISTRY["lineage"].arg_types[0]
    substrates = {
        name for name, sig in pc.REGISTRY.items() if sig.output == declared
    }
    # v0.5 added enwiki; the obligation's property (substrate atoms declare
    # no modifiers, so enumerable forms are exactly the atoms) is version-
    # independent and re-measured against the live registry.
    assert substrates == {"pearltrees", "simplemind", "simplewiki", "enwiki", "fs"}
    # Substrate atoms declare no modifiers, so the enumerable forms are
    # exactly the four atoms — down from the measured 15 under v0.3
    # (9 source atoms + 6 modifier variants).
    for name in substrates:
        assert pc.REGISTRY[name].modifiers == frozenset()
    assert len(substrates) == 5 < 15


# --------------------------------------------------------------------------
# obligation 9: every committed process_expression field parses under v0.4
# --------------------------------------------------------------------------


def _iter_expression_fields(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "process_expression" and isinstance(child, str):
                yield child
            else:
                yield from _iter_expression_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_expression_fields(child)


def test_obligation_9_every_committed_process_expression_field_parses():
    found = []
    for path in sorted(ROOT.glob("*.json")):
        if path.name in mig.SEALED_SOURCES and "GOLDEN" in path.name:
            continue  # sealed v0.3 bundles are audit provenance, rows keyed differently
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for expression in _iter_expression_fields(document):
            found.append((path.name, expression))
            pc.parse(expression)  # ParseError = failed obligation
    assert ("SM_FS_LINEAGE_RANKING_PREREG.json", "lineage(fs,decay=0.85)") in found


def test_obligation_9_p1_prereg_expressions_parse_under_v04():
    document = json.loads(
        (ROOT / "PROCESS_EXPRESSION_P1_PREREG.json").read_text(encoding="utf-8")
    )
    for record in document["process_identity"]["processes"].values():
        node = pc.parse(record["expression"])
        assert pc.canonical_semantic(node) == record["canonical"]


def test_obligation_9_wiki_lineage_generator_writes_the_v04_spelling():
    source = (ROOT / "gen_wiki_lineage_v2.py").read_text(encoding="utf-8")
    assert "lineage(graph" not in source
    header = 'lineage(simplewiki,mu=graph,estimand="ancestry")'
    assert header in source
    pc.parse(header)
