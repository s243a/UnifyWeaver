#!/usr/bin/env python3
"""Offline enforcement tests for the routed filing e5 revision pin."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import filing_assistant
from mu_attention import E5_MODEL, E5_REVISION, build_e5_tables
import routed_queries


def _fake_sentence_transformers():
    class FakeSentenceTransformer:
        calls = []

        def __init__(self, model_name, **kwargs):
            self.calls.append((model_name, dict(kwargs)))

        def encode(self, texts, **_kwargs):
            vectors = np.zeros((len(texts), 4), dtype=np.float32)
            for row, text in enumerate(texts):
                vectors[row, row % 4] = 1.0
                vectors[row, (len(text) + 1) % 4] += 0.25
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors

    return SimpleNamespace(SentenceTransformer=FakeSentenceTransformer), FakeSentenceTransformer


def test_e5_revision_is_the_frozen_immutable_commit():
    assert E5_REVISION == "ffb93f3bd4047442299a41ebb6fa998a38507c52"
    assert len(E5_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in E5_REVISION)


def test_build_e5_tables_forwards_revision_and_reuses_matching_cache(
    tmp_path, monkeypatch
):
    module, fake_model = _fake_sentence_transformers()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    cache = tmp_path / "e5.pt"

    first_q, first_p, first_idx = build_e5_tables(
        ["alpha", "beta"],
        cache_path=cache,
        model_revision=E5_REVISION,
    )
    assert len(fake_model.calls) == 1
    model_name, model_kwargs = fake_model.calls[0]
    assert model_name == E5_MODEL
    assert model_kwargs["revision"] == E5_REVISION
    payload = torch.load(cache, weights_only=False)
    assert payload["model_name"] == E5_MODEL
    assert payload["model_revision"] == E5_REVISION

    class MustNotConstruct:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("matching cache should avoid model construction")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=MustNotConstruct),
    )
    second_q, second_p, second_idx = build_e5_tables(
        ["alpha", "beta"],
        cache_path=cache,
        model_revision=E5_REVISION,
    )
    assert torch.equal(second_q, first_q)
    assert torch.equal(second_p, first_p)
    assert second_idx == first_idx


def test_build_e5_tables_keeps_generic_unpinned_callers_compatible(monkeypatch):
    module, fake_model = _fake_sentence_transformers()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    build_e5_tables(["alpha"], model_name="fixture/model")

    assert fake_model.calls == [("fixture/model", {"device": None})]


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"model_name": E5_MODEL},
        {"model_name": E5_MODEL, "model_revision": "wrong-revision"},
        {"model_name": "other/model", "model_revision": E5_REVISION},
    ],
)
def test_build_e5_tables_invalidates_legacy_or_mismatched_cache(
    tmp_path, monkeypatch, metadata
):
    cache = tmp_path / "e5.pt"
    torch.save(
        {
            "names": ["alpha"],
            "human": ["alpha"],
            "query": torch.full((1, 4), -1.0),
            "passage": torch.full((1, 4), -1.0),
            **metadata,
        },
        cache,
    )
    module, fake_model = _fake_sentence_transformers()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    query, passage, _ = build_e5_tables(
        ["alpha"],
        cache_path=cache,
        model_revision=E5_REVISION,
    )
    assert len(fake_model.calls) == 1
    assert not torch.equal(query, torch.full((1, 4), -1.0))
    assert not torch.equal(passage, torch.full((1, 4), -1.0))
    payload = torch.load(cache, weights_only=False)
    assert payload["model_name"] == E5_MODEL
    assert payload["model_revision"] == E5_REVISION


def test_filing_candidate_and_query_encoders_use_one_pinned_revision(
    tmp_path, monkeypatch
):
    module, fake_model = _fake_sentence_transformers()
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setenv("MU_COSINE_CACHE_DIR", str(tmp_path))
    privacy = SimpleNamespace(manifest_sha256="a" * 64)
    monkeypatch.setattr(
        filing_assistant,
        "load_filing",
        lambda *_args, **_kwargs: (
            [("bookmark one", 1), ("bookmark two", 2)],
            {1: "Folder One", 2: "Folder Two"},
            privacy,
        ),
    )

    filing_assistant.catalog_tables(1, "test")
    filing_assistant.encode_queries(["bookmark one"])

    assert len(fake_model.calls) == 2
    assert all(
        model_name == E5_MODEL and kwargs["revision"] == E5_REVISION
        for model_name, kwargs in fake_model.calls
    )
    assert [
        path.name
        for path in tmp_path.iterdir()
        if E5_REVISION[:12] in path.name
    ] == [
        f"filing_assistant_e5_test_{privacy.manifest_sha256[:12]}_"
        f"{E5_REVISION[:12]}.pt"
    ]


def test_routed_receipt_reports_immutable_ranker_revision(monkeypatch):
    class FakePrivacy:
        manifest_sha256 = "a" * 64
        source_snapshot = {
            "schema": "unifyweaver.pearltrees-source-snapshot.v1",
            "file_count": 1,
            "members_sha256": "b" * 64,
            "total_size_bytes": 1,
        }

        @staticmethod
        def receipt():
            return {
                "schema": "unifyweaver.pearltrees-privacy-index.v1",
                "policy_id": "pearltrees-public-only-v1",
                "counts": {"public": 2, "private": 0, "quarantined": 0},
            }

    monkeypatch.setattr(
        routed_queries,
        "catalog_tables",
        lambda *_args, **_kwargs: (
            [("bookmark", 10)],
            [10, 11],
            ["Folder A", "Folder B"],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            FakePrivacy(),
        ),
    )
    monkeypatch.setattr(
        routed_queries,
        "encode_queries",
        lambda _titles: np.array([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        routed_queries,
        "_implementation_record",
        lambda: {"git_commit": "test", "files_sha256": "c" * 64},
    )

    state = routed_queries.build(
        SimpleNamespace(min_bm=1, max_queries=10, seed=7, top_k=2)
    )
    ranker = state.receipt["ranker"]
    assert ranker["model_id"] == E5_MODEL
    assert ranker["model_revision"] == E5_REVISION
    assert ranker["model_revision_status"] == "immutable-huggingface-commit"
