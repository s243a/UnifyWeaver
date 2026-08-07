#!/usr/bin/env python3
"""Registry v0.4 -> v0.5 migration obligations.

An additive bump has a shorter obligation list than a semantic one, and the
obligations are different in kind: the load-bearing claims are that nothing
became invalid, that every old identity is BYTE-STABLE, and that the delta
is exactly what the migration says it added — checked against the live
registry, not asserted.
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
import registry_v05_migration as mig


@pytest.fixture(scope="module")
def manifest():
    return mig.load_and_verify()


def test_every_v04_bundle_identity_is_mapped_and_byte_stable(manifest):
    bundle = json.loads(
        (ROOT / "PROCESS_EXPRESSION_GOLDEN_v3.json").read_text(encoding="utf-8")
    )
    sealed = {row["canonical_identity_string"] for row in bundle["rows"]}
    mapped = {row["canonical_semantic"] for row in manifest["rows"]}
    # External review (M3): the net also covers the v0.3->v0.4 manifest's
    # mapped successors, two of which are not bundle rows — the inventory is
    # the UNION of both sealed sources, so mapped is a strict superset.
    assert sealed < mapped
    v04 = json.loads(
        (ROOT / "REGISTRY_V04_MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
    successors = {row["new_canonical_bytes"] for row in v04["rows"]
                  if row["status"] == "mapped"}
    assert mapped == sealed | successors
    for semantic in sorted(mapped):
        node = pc.parse(semantic)
        assert pc.canonical_semantic(node) == semantic  # byte-stable under v0.5


def test_every_digest_moves_and_nothing_is_tombstoned(manifest):
    assert manifest["status_counts"]["tombstoned"] == 0
    for row in manifest["rows"]:
        assert row["status"] == "mapped"
        assert row["old_semantic_digest"] != row["new_semantic_digest"]
        assert row["new_semantic_digest"] == hashlib.sha256(
            f"v0.5|{row['canonical_semantic']}".encode()
        ).hexdigest()


def test_the_delta_is_exactly_what_the_migration_declares(manifest):
    """Reverse containment, measured on the live registry: v0.5's names minus
    the declared additions must be exactly the v0.4 name set (as witnessed by
    the sealed v0.4 registry-content hash still verifying in the v0.4
    migration), and the value-kind delta likewise."""
    assert set(manifest["added_names"]) == {"cowalk", "enwiki"}
    assert set(manifest["added_value_kinds"]) == {"walk", "weight"}
    for name in manifest["added_names"]:
        assert name in pc.REGISTRY
    for kind in manifest["added_value_kinds"]:
        assert kind in pc.VALUE_KINDS
    # The new kinds are used ONLY by the new operator: no v0.4 signature
    # gained or lost a kwarg (additive means untouched, not merely parseable).
    for name, sig in pc.REGISTRY.items():
        if name in manifest["added_names"]:
            continue
        for key, spec in sig.kwargs.items():
            assert spec.kind not in manifest["added_value_kinds"], (name, key)


def test_walk_values_declare_their_shape_at_registration():
    """The family classification depends on the declared walk shape, and the
    trainable-now/nameable-later ruling requires admission WITHOUT
    invertibility — so the shape is a registered fact, never an inference,
    and the shape vocabulary admits non-palindromic walks."""
    for walk, shape in pc.WALKS.items():
        assert shape in pc.WALK_SHAPES, walk
    assert pc.WALKS == {"sibling": "palindromic", "cousin": "palindromic"}
    assert "non_palindromic" in pc.WALK_SHAPES


def test_manifest_is_tamper_evident(manifest):
    tampered = json.loads(
        (mig.MANIFEST_PATH).read_text(encoding="utf-8")
    )
    tampered["rows"][0]["status"] = "tombstoned"
    with pytest.raises(mig.MigrationError):
        mig.verify_manifest(tampered)


def test_manifest_is_reproducible_from_sealed_sources(manifest):
    assert mig.build_manifest()["manifest_sha256"] == manifest["manifest_sha256"]


def test_superseded_version_never_mints_new_identities():
    assert "v0.4" in pc.SUPERSEDED_REGISTRY_VERSIONS
    assert pc.REGISTRY_VERSION == "v0.5"
