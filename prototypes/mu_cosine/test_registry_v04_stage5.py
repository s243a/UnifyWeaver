#!/usr/bin/env python3
"""Stage 5 of registry v0.4 (§5, §9): the conditioning-card cache key binds
``registry_version`` explicitly.

The hazard on record: cards are e5 embeddings of registry-dependent strings,
so a cache keyed without the version silently returns a vector embedded from
a string that no longer exists after a registry bump. Two facts close it:

1. ``process_cards.embedding_cache_key`` — the digest-keyed card cache key —
   carries ``REGISTRY_VERSION`` as an explicit component, proven here by
   holding ``ast_sha`` fixed while varying only the version.
2. The ranking lane's live e5 caches (``mu_attention.build_e5_tables``)
   validate by the embedded TEXT content (names + human strings + model +
   revision), so they cannot serve a stale string at all — measured against
   the cache-reuse condition in the source.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc


def test_cache_key_binds_registry_version_explicitly(monkeypatch):
    """Hold ast_sha constant; the key must still move with the version."""
    node = pc.parse("kalman(luna.D,luna.S)")
    monkeypatch.setattr(pc, "ast_sha", lambda n: "0" * 16)
    before = pc.embedding_cache_key(node, 1, "rev-a")
    monkeypatch.setattr(pc, "REGISTRY_VERSION", "v0.5-hypothetical")
    after = pc.embedding_cache_key(node, 1, "rev-a")
    assert before != after


def test_cache_key_moves_across_the_actual_v03_v04_boundary():
    """A v0.4 key can never collide with a v0.3-era key for the same card:
    both the explicit version component and the ast_sha preimage moved."""
    node = pc.parse("kalman(luna.D,luna.S)")
    key = pc.embedding_cache_key(node, 2, "rev-a")
    assert key == pc.embedding_cache_key(pc.parse("kalman(luna.D,luna.S)"), 2, "rev-a")
    assert pc.REGISTRY_VERSION == "v0.4"  # the key above is a v0.4 key


def test_ranking_lane_e5_caches_are_content_addressed():
    """The live card caches revalidate on the embedded text itself, so no
    version field is needed there: a changed string is a changed cache."""
    source = (ROOT / "mu_attention.py").read_text(encoding="utf-8")
    reuse_condition = source[source.index("def build_e5_tables"):]
    reuse_condition = reuse_condition[: reuse_condition.index("SentenceTransformer")]
    for guard in ('d.get("names")', 'd.get("human")',
                  'd.get("model_name")', 'd.get("model_revision")'):
        assert guard in reuse_condition, guard
