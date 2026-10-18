#!/usr/bin/env python3
"""Focused durability and e5-cache tests for ``sm_fs_filing``."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mu_attention
import sm_fs_filing
from sm_fs_filing import (
    E5_MODEL,
    E5_REVISION,
    UnsafePrivatePathError,
    _atomic_json,
    build_private_e5_tables,
    filing_provenance,
    prepare_private_cache,
)


def test_atomic_json_removes_linked_final_when_directory_fsync_fails(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "private" / "ledger.json"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("synthetic pre-durability failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="pre-durability"):
        _atomic_json(ledger, {"frozen": True})

    assert not os.path.lexists(ledger)
    assert list(ledger.parent.iterdir()) == []


def test_atomic_json_is_private_durable_and_never_replaces(tmp_path):
    ledger = tmp_path / "private" / "ledger.json"
    _atomic_json(ledger, {"frozen": True})

    assert json.loads(ledger.read_text(encoding="utf-8")) == {"frozen": True}
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
    assert stat.S_IMODE(ledger.parent.stat().st_mode) == 0o700
    assert not list(ledger.parent.glob(f".{ledger.name}.*"))
    with pytest.raises(FileExistsError, match="refusing to replace"):
        _atomic_json(ledger, {"frozen": False})


@pytest.mark.parametrize("kind", ("ledger", "cache"))
@pytest.mark.parametrize("dangling", (False, True))
def test_private_artifacts_reject_symlink_targets(tmp_path, kind, dangling):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    victim = tmp_path / ("missing" if dangling else "victim")
    if not dangling:
        victim.write_text("do not touch", encoding="utf-8")
    target = private / f"{kind}.dat"
    target.symlink_to(victim)

    with pytest.raises(UnsafePrivatePathError, match="symlink.*target"):
        if kind == "ledger":
            _atomic_json(target, {"unsafe": True})
        else:
            prepare_private_cache(target)
    if dangling:
        assert not victim.exists()
    else:
        assert victim.read_text(encoding="utf-8") == "do not touch"
    assert target.is_symlink()


def test_private_artifacts_reject_symlink_and_nonprivate_parents(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(UnsafePrivatePathError, match="symlink parent"):
        _atomic_json(linked_parent / "ledger.json", {})

    public_parent = tmp_path / "public"
    public_parent.mkdir(mode=0o700)
    public_parent.chmod(0o755)
    with pytest.raises(UnsafePrivatePathError, match="mode 0700"):
        prepare_private_cache(public_parent / "cache.pt")

    writable_ancestor = tmp_path / "writable-ancestor"
    writable_ancestor.mkdir(mode=0o700)
    writable_ancestor.chmod(0o777)
    private_leaf = writable_ancestor / "private"
    private_leaf.mkdir(mode=0o700)
    with pytest.raises(UnsafePrivatePathError, match="writable ancestor"):
        prepare_private_cache(private_leaf / "cache.pt")


def _fake_e5_builder(observed):
    def build(names, **kwargs):
        observed.update(kwargs)
        matrix = torch.eye(len(names), dtype=torch.float32)
        return matrix, matrix.clone(), {
            name: index for index, name in enumerate(names)
        }

    return build


def test_private_cache_rebuild_is_sanitized_atomic_and_hashed(
    tmp_path, monkeypatch
):
    cache = prepare_private_cache(tmp_path / "private" / "e5.pt")
    cache.write_bytes(b"not a torch cache")
    observed = {}
    monkeypatch.setattr(
        sm_fs_filing, "build_e5_tables", _fake_e5_builder(observed)
    )

    query, passage, idx, cache_sha256 = build_private_e5_tables(
        ["Folder", "One"], cache, batch_size=7
    )

    assert set(observed) == {
        "cache_path",
        "batch_size",
        "model_name",
        "model_revision",
    }
    assert observed["batch_size"] == 7
    assert observed["model_name"] == E5_MODEL
    assert observed["model_revision"] == E5_REVISION
    assert Path(observed["cache_path"]).parent != cache.parent
    assert query.shape == passage.shape == (2, 2)
    assert idx == {"Folder": 0, "One": 1}
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    assert cache_sha256 == hashlib.sha256(cache.read_bytes()).hexdigest()
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    assert payload["model_name"] == E5_MODEL
    assert payload["model_revision"] == E5_REVISION
    assert payload["names"] == ["Folder", "One"]
    assert not list(cache.parent.glob(f".{cache.name}.build.*"))


def test_private_cache_reuse_uses_safe_payload_without_rebuilding(
    tmp_path, monkeypatch
):
    names = ["Folder", "One"]
    cache = prepare_private_cache(tmp_path / "private" / "e5.pt")
    payload = {
        "names": names,
        "human": names,
        "model_name": E5_MODEL,
        "model_revision": E5_REVISION,
        "query": torch.eye(2, dtype=torch.float32),
        "passage": torch.eye(2, dtype=torch.float32),
    }
    torch.save(payload, cache)

    monkeypatch.setattr(
        sm_fs_filing,
        "build_e5_tables",
        lambda *_args, **_kwargs: pytest.fail("valid cache was rebuilt"),
    )
    query, passage, idx, digest = build_private_e5_tables(names, cache)

    assert torch.equal(query, payload["query"])
    assert torch.equal(passage, payload["passage"])
    assert idx == {"Folder": 0, "One": 1}
    assert digest == hashlib.sha256(cache.read_bytes()).hexdigest()


def test_private_cache_swap_between_deserialization_and_hash_is_rejected(
    tmp_path, monkeypatch
):
    names = ["Folder", "One"]
    private = tmp_path / "private"
    cache = prepare_private_cache(private / "e5.pt")
    original_payload = {
        "names": names,
        "human": names,
        "model_name": E5_MODEL,
        "model_revision": E5_REVISION,
        "query": torch.eye(2, dtype=torch.float32),
        "passage": torch.eye(2, dtype=torch.float32),
    }
    replacement_payload = {
        **original_payload,
        "query": torch.flip(original_payload["query"], dims=(0,)),
        "passage": torch.flip(original_payload["passage"], dims=(0,)),
    }
    torch.save(original_payload, cache)
    replacement = private / "replacement.pt"
    torch.save(replacement_payload, replacement)
    replacement.chmod(0o600)

    real_load = mu_attention.torch.load
    swapped = False

    def load_then_swap(*args, **kwargs):
        nonlocal swapped
        payload = real_load(*args, **kwargs)
        if not swapped:
            os.replace(replacement, cache)
            swapped = True
        return payload

    monkeypatch.setattr(mu_attention.torch, "load", load_then_swap)
    monkeypatch.setattr(
        sm_fs_filing,
        "build_e5_tables",
        lambda *_args, **_kwargs: pytest.fail(
            "a replacement race must be rejected, not rebuilt"
        ),
    )

    with pytest.raises(UnsafePrivatePathError, match="replaced"):
        build_private_e5_tables(names, cache)
    assert swapped
    assert torch.equal(
        real_load(cache, map_location="cpu", weights_only=True)["query"],
        replacement_payload["query"],
    )


def test_newly_installed_cache_swap_before_hash_is_rejected(
    tmp_path, monkeypatch
):
    names = ["Folder", "One"]
    private = tmp_path / "private"
    cache = prepare_private_cache(private / "e5.pt")
    replacement = private / "replacement.pt"
    replacement_payload = {
        "names": names,
        "human": names,
        "model_name": E5_MODEL,
        "model_revision": E5_REVISION,
        "query": torch.flip(torch.eye(2, dtype=torch.float32), dims=(0,)),
        "passage": torch.flip(torch.eye(2, dtype=torch.float32), dims=(0,)),
    }
    torch.save(replacement_payload, replacement)
    replacement.chmod(0o600)

    observed = {}
    monkeypatch.setattr(
        sm_fs_filing, "build_e5_tables", _fake_e5_builder(observed)
    )
    real_load = mu_attention.torch.load
    swapped = False

    def load_then_swap(*args, **kwargs):
        nonlocal swapped
        payload = real_load(*args, **kwargs)
        if not swapped:
            os.replace(replacement, cache)
            swapped = True
        return payload

    monkeypatch.setattr(mu_attention.torch, "load", load_then_swap)

    with pytest.raises(UnsafePrivatePathError, match="replaced"):
        build_private_e5_tables(names, cache)
    assert swapped
    assert observed["model_name"] == E5_MODEL
    assert observed["model_revision"] == E5_REVISION


def test_mu_attention_cache_deserialization_uses_weights_only(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache.pt"
    names = ["One"]
    torch.save(
        {
            "names": names,
            "human": names,
            "model_name": E5_MODEL,
            "model_revision": E5_REVISION,
            "query": torch.ones(1, 2),
            "passage": torch.ones(1, 2),
        },
        cache,
    )
    real_load = mu_attention.torch.load
    observed = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(mu_attention.torch, "load", recording_load)
    mu_attention.build_e5_tables(
        names,
        cache_path=str(cache),
        model_revision=E5_REVISION,
    )
    assert observed["weights_only"] is True


def test_filing_provenance_binds_model_source_cache_and_runtime():
    provenance = filing_provenance("a" * 64)
    assert provenance["e5_model_id"] == "intfloat/e5-small-v2"
    assert provenance["e5_revision"] == E5_REVISION
    assert provenance["mu_attention_sha256"] == hashlib.sha256(
        (HERE / "mu_attention.py").read_bytes()
    ).hexdigest()
    assert provenance["e5_cache_sha256"] == "a" * 64
    assert set(provenance["runtime_versions"]) == {
        "python",
        "numpy",
        "torch",
        "sentence_transformers",
        "transformers",
    }
    assert all(
        value is None or isinstance(value, str)
        for value in provenance["runtime_versions"].values()
    )


def test_main_seals_cache_hash_and_provenance_before_scoring(
    tmp_path, monkeypatch
):
    rows = [
        {
            "title": "One",
            "dir": "Folder",
            "path": "Folder",
            "privacy": "public",
        }
    ]
    header = {
        "schema": "sm-fs-privacy-test",
        "policy_id": "test-policy",
        "index_sha256": "b" * 64,
        "counts": {"unresolved_private_target_refs": 0},
    }
    monkeypatch.setattr(
        sm_fs_filing,
        "load_or_build_privacy_index",
        lambda *_args: (header, {}),
    )
    monkeypatch.setattr(
        sm_fs_filing,
        "discover_filing_rows",
        lambda *_args: (
            rows,
            {"public": 1, "private": 0, "unknown": 0},
        ),
    )
    cache_hash = "c" * 64

    def fake_private_build(names, _path, batch_size):
        assert batch_size == 128
        matrix = torch.eye(len(names), dtype=torch.float32)
        return (
            matrix,
            matrix,
            {name: index for index, name in enumerate(names)},
            cache_hash,
        )

    monkeypatch.setattr(
        sm_fs_filing, "build_private_e5_tables", fake_private_build
    )
    ledger = tmp_path / "private" / "ledger.json"
    sm_fs_filing.main(
        [
            "--root",
            str(tmp_path / "corpus"),
            "--privacy-index",
            str(tmp_path / "unused.jsonl"),
            "--privacy-policy",
            "public-only",
            "--min-maps",
            "1",
            "--holdout-frac",
            "0",
            "--ledger",
            str(ledger),
            "--e5-cache",
            str(tmp_path / "private" / "e5.pt"),
        ]
    )
    value = json.loads(ledger.read_text(encoding="utf-8"))
    assert value["provenance"]["e5_cache_sha256"] == cache_hash
    assert value["provenance"]["e5_model_id"] == E5_MODEL
    assert value["provenance"]["e5_revision"] == E5_REVISION
    assert "mu_attention_sha256" in value["provenance"]
    assert "runtime_versions" in value["provenance"]
