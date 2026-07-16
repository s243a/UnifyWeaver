#!/usr/bin/env python3
"""Tests for network-free, content-addressed independent embedding caches."""
from dataclasses import replace
import json

import numpy as np
import pytest

from independent_embedding_cache import (
    EmbeddingModelSpec,
    cache_paths,
    canonical_names,
    embedding_texts,
    encode_with_model,
    load_embedding_cache,
    read_pair_nodes,
    validate_embeddings,
    write_embedding_cache,
)


SPEC = EmbeddingModelSpec(
    "example/frozen-embedder",
    "0123456789abcdef",
    "clustering: ",
    3,
    False,
)


class FakeModel:
    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"]
        rows = []
        for index, _text in enumerate(texts):
            value = np.array([index + 1.0, 1.0, 0.5], dtype=np.float32)
            rows.append(value / np.linalg.norm(value))
        return np.asarray(rows)


def test_canonical_names_and_task_texts_are_stable():
    assert canonical_names(("Zeta_node", "Alpha_node")) == ("Alpha_node", "Zeta_node")
    assert embedding_texts(("Alpha_node", "Zeta_node"), SPEC) == (
        "clustering: Alpha node",
        "clustering: Zeta node",
    )
    with pytest.raises(ValueError, match="unique"):
        canonical_names(("x", "x"))
    with pytest.raises(ValueError, match="non-empty"):
        canonical_names(("x", " "))


def test_fake_encoder_receives_frozen_contract_without_model_download():
    names, texts, embeddings = encode_with_model(
        ("Zeta_node", "Alpha_node"), SPEC, FakeModel(), batch_size=2
    )
    assert names == ("Alpha_node", "Zeta_node")
    assert texts[0] == "clustering: Alpha node"
    assert embeddings.shape == (2, 3)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_cache_is_byte_deterministic_and_portable_across_prefixes(tmp_path):
    names, _texts, embeddings = encode_with_model(
        ("Alpha_node", "Zeta_node"), SPEC, FakeModel()
    )
    first_manifest, first_record = write_embedding_cache(
        tmp_path / "first" / "cache",
        names,
        embeddings,
        SPEC,
        package_versions={"numpy": "test"},
    )
    second_manifest, second_record = write_embedding_cache(
        tmp_path / "second" / "renamed",
        names,
        embeddings,
        SPEC,
        package_versions={"numpy": "test"},
    )
    assert first_manifest == second_manifest
    assert first_record == second_record
    for role in ("nodes", "embeddings", "manifest"):
        assert cache_paths(tmp_path / "first" / "cache")[role].read_bytes() == cache_paths(
            tmp_path / "second" / "renamed"
        )[role].read_bytes()
    assert "/tmp" not in repr(first_manifest)


def test_cache_loader_validates_spec_coverage_and_hashes(tmp_path):
    names, _texts, embeddings = encode_with_model(
        ("Alpha_node", "Zeta_node"), SPEC, FakeModel()
    )
    prefix = tmp_path / "cache"
    write_embedding_cache(prefix, names, embeddings, SPEC)
    loaded = load_embedding_cache(
        prefix, expected_spec=SPEC, required_names=("Alpha_node",)
    )
    assert loaded.names == names
    assert set(loaded.by_name()) == set(names)
    assert np.array_equal(loaded.embeddings, embeddings)

    with pytest.raises(ValueError, match="frozen spec"):
        load_embedding_cache(prefix, expected_spec=replace(SPEC, revision="different"))
    with pytest.raises(ValueError, match="misses 1"):
        load_embedding_cache(prefix, required_names=("missing",))

    paths = cache_paths(prefix)
    paths["nodes"].write_bytes(paths["nodes"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="content hash"):
        load_embedding_cache(prefix)


def test_write_rejects_unsorted_names_to_prevent_embedding_misalignment(tmp_path):
    embeddings = np.eye(3, dtype=np.float32)[:2]
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    with pytest.raises(ValueError, match="canonical and sorted"):
        write_embedding_cache(tmp_path / "cache", ("z", "a"), embeddings, SPEC)


def test_embedding_validation_rejects_shape_nonfinite_zero_and_unnormalized():
    names = ("a", "b")
    with pytest.raises(ValueError, match="shape"):
        validate_embeddings(names, np.ones((2, 2)), SPEC)
    bad = np.ones((2, 3), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_embeddings(names, bad, SPEC)
    with pytest.raises(ValueError, match="nonzero"):
        validate_embeddings(names, np.zeros((2, 3)), SPEC)
    with pytest.raises(ValueError, match="L2-normalized"):
        validate_embeddings(names, np.ones((2, 3)), SPEC)


def test_pair_tsv_endpoint_inventory_is_comment_aware_and_canonical(tmp_path):
    path = tmp_path / "pairs.tsv"
    path.write_text(
        "# left\tright\textra\n"
        "Zeta\tAlpha\tone\n"
        "Beta\tZeta\ttwo\n",
        encoding="utf-8",
    )
    assert read_pair_nodes(path) == ("Alpha", "Beta", "Zeta")
    path.write_text("broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_pair_nodes(path)


def test_loader_rejects_manifest_text_contract_tampering(tmp_path):
    names, _texts, embeddings = encode_with_model(("Alpha", "Zeta"), SPEC, FakeModel())
    prefix = tmp_path / "cache"
    write_embedding_cache(prefix, names, embeddings, SPEC)
    paths = cache_paths(prefix)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["input_text_sha256"] = "0" * 64
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="input text hash"):
        load_embedding_cache(prefix)
