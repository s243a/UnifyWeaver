#!/usr/bin/env python3
"""Focused tests for the SimpleMind filesystem privacy triage."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sm_fs_filing
import sm_fs_privacy
from sm_fs_filing import (
    discover_filing_rows,
    filing_provenance,
    harden_private_cache,
    prepare_private_cache,
    require_public_privacy_perimeter,
)
from benchmark_sm_fs_privacy_local import score_response, synthetic_cases
from sm_fs_privacy import (
    MODEL_REVIEW_DISABLED_REASON,
    PARSER_PATH,
    SmFsPrivacyError,
    apply_model_reviews,
    build_index,
    canonical_json_bytes,
    load_index,
    pearltrees_url_kind,
    resolve_cloudmapref,
    scan_maps,
    sha256_bytes,
    sha256_file,
    verify_index_source,
    write_index,
)
from mu_attention import E5_REVISION


def _map_xml_bytes(topics):
    document = ET.Element("mindmap")
    for spec in topics:
        attrs = {
            "id": str(spec["id"]),
            "parent": str(spec.get("parent", "-1")),
            "text": spec.get("text", ""),
        }
        topic = ET.SubElement(document, "topic", attrs)
        for url in spec.get("urls", ()):
            ET.SubElement(topic, "link", {"urllink": url})
        for ref in spec.get("refs", ()):
            ET.SubElement(topic, "link", {"cloudmapref": ref})
    return ET.tostring(document)


def _write_map(path: Path, topics):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document/mindmap.xml", _map_xml_bytes(topics))


def _write_map_archive(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


def _record_by_name(records, suffix):
    return next(row for row in records if row["relative_path"].endswith(suffix))


def _resealed_index_bytes(header, records):
    resealed_header = json.loads(json.dumps(header))
    resealed_records = json.loads(json.dumps(records))
    core = dict(resealed_header)
    core.pop("index_sha256", None)
    core["rows_sha256"] = sha256_bytes(
        b"".join(canonical_json_bytes(record) for record in resealed_records)
    )
    resealed_header["index_sha256"] = sha256_bytes(
        canonical_json_bytes(core)
    )
    return canonical_json_bytes(resealed_header) + b"".join(
        canonical_json_bytes(record) for record in resealed_records
    )


def test_pearltrees_private_export_link_is_unknown_not_private():
    assert (
        pearltrees_url_kind("https://www.pearltrees.com/private/id123?access=abc")
        == "private_unknown"
    )
    assert (
        pearltrees_url_kind(
            "https://www.pearltrees.com/s243a/private-hacking/id456"
        )
        == "public"
    )


@pytest.mark.parametrize(
    "url",
    (
        "ftp://www.pearltrees.com/s243a/topic/id456",
        "https://evil@www.pearltrees.com/s243a/topic/id456",
        "https://www.pearltrees.com:443/s243a/topic/id456",
        "https://www.pearltrees.com/s243a/topic/extra/id456",
        "https://www.pearltrees.com/s243a/private/id456",
        "https://www.pearltrees.com/private/topic/id456",
        "https://www.pearltrees.com/s243a/%70rivate/id456",
        "https://www.pearltrees.com/s243a/%2Fprivate/id456",
    ),
)
def test_pearltrees_public_url_grammar_rejects_noncanonical_and_private_paths(url):
    assert pearltrees_url_kind(url) != "public"


def test_smmx_zip_uses_exact_canonical_member_not_preceding_decoy(tmp_path):
    _write_map_archive(
        tmp_path / "Map.smmx",
        (
            (
                "decoy/mindmap.xml",
                _map_xml_bytes([{"id": 0, "text": "Private: decoy"}]),
            ),
            (
                "document/mindmap.xml",
                _map_xml_bytes([{"id": 0, "text": "Canonical public map"}]),
            ),
        ),
    )
    record = scan_maps(tmp_path, workers=1)[0]
    assert record["root_title"] == "Canonical public map"
    assert record["classification"] == "public"


def test_smmx_zip_without_exact_canonical_member_fails_closed(tmp_path):
    _write_map_archive(
        tmp_path / "Map.smmx",
        (
            (
                "decoy/mindmap.xml",
                _map_xml_bytes([{"id": 0, "text": "Decoy"}]),
            ),
        ),
    )
    record = scan_maps(tmp_path, workers=1)[0]
    assert record["classification"] == "unknown"
    assert record["reasons"] == ["map_parse_error"]
    assert record["parse_error_type"] == "SmFsPrivacyError"


@pytest.mark.parametrize(
    "alias",
    (
        "Document/mindmap.xml",
        "document\\mindmap.xml",
        "./document/mindmap.xml",
        "document//mindmap.xml",
        "other/../document/mindmap.xml",
    ),
)
def test_smmx_zip_rejects_ambiguous_canonical_member_alias(tmp_path, alias):
    _write_map_archive(
        tmp_path / "Map.smmx",
        (
            (
                "document/mindmap.xml",
                _map_xml_bytes([{"id": 0, "text": "Canonical"}]),
            ),
            (
                alias,
                _map_xml_bytes([{"id": 0, "text": "Ambiguous"}]),
            ),
        ),
    )
    record = scan_maps(tmp_path, workers=1)[0]
    assert record["classification"] == "unknown"
    assert record["reasons"] == ["map_parse_error"]
    assert record["parse_error_type"] == "SmFsPrivacyError"


def test_smmx_zip_rejects_duplicate_canonical_zipinfo_entries(tmp_path):
    path = tmp_path / "Map.smmx"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_map_archive(
            path,
            (
                (
                    "document/mindmap.xml",
                    _map_xml_bytes([{"id": 0, "text": "First"}]),
                ),
                (
                    "document/mindmap.xml",
                    _map_xml_bytes([{"id": 0, "text": "Second"}]),
                ),
            ),
        )
    record = scan_maps(tmp_path, workers=1)[0]
    assert record["classification"] == "unknown"
    assert record["reasons"] == ["map_parse_error"]
    assert record["parse_error_type"] == "SmFsPrivacyError"


def test_public_topical_root_conflicts_with_private_directory(tmp_path):
    _write_map(
        tmp_path / "Synthetic" / "Private" / "Private Foobarology.smmx",
        [
            {
                "id": 0,
                "text": "Private Foobarology",
                "urls": (
                    "https://www.pearltrees.com/example/foobarology/id456",
                ),
            }
        ],
    )
    record = scan_maps(tmp_path)[0]
    assert record["classification"] == "unknown"
    assert "access_marker_conflicts_with_public_root" in record["reasons"]


def test_extra_blank_root_does_not_hide_canonical_root(tmp_path):
    _write_map(
        tmp_path / "Map.smmx",
        [
            {
                "id": 0,
                "text": "Map",
                "urls": ("https://www.pearltrees.com/s243a/map/id456",),
            },
            {"id": 99, "text": ""},
        ],
    )
    record = scan_maps(tmp_path)[0]
    assert record["classification"] == "public"
    assert "extra_root_topics_ignored" in record["reasons"]


@pytest.mark.parametrize(
    "topics",
    (
        [
            {"id": 0, "text": "Map"},
            {"id": 0, "parent": 0, "text": "duplicate"},
        ],
        [
            {"id": "", "text": "Map"},
        ],
        [
            {"id": 0, "text": "Map"},
            {"id": 1, "parent": 999, "text": "orphan"},
        ],
        [
            {"id": 0, "text": "Map"},
            {"id": 1, "parent": 2, "text": "cycle one"},
            {"id": 2, "parent": 1, "text": "cycle two"},
        ],
        [
            {"id": 0, "text": "Map"},
            {
                "id": 99,
                "text": "",
                "urls": ("https://example.com/not-blank",),
            },
        ],
        [
            {"id": 0, "text": "Map"},
            {"id": 99, "text": ""},
            {"id": 100, "parent": 99, "text": "hidden child"},
        ],
        [
            {"id": 0, "text": "Map"},
            {"id": 99, "text": "second root"},
        ],
    ),
)
def test_invalid_topic_identity_parent_graph_or_extra_root_fails_closed(
    tmp_path, topics
):
    _write_map(tmp_path / "Map.smmx", topics)
    record = scan_maps(tmp_path, workers=1)[0]
    assert record["classification"] == "unknown"
    assert "map_parse_error" in record["reasons"]
    assert record["parse_error_type"] == "SmFsPrivacyError"


def test_private_directory_and_access_prefix_identify_private_duplicate(tmp_path):
    _write_map(
        tmp_path / "Synthetic" / "Private" / "Private_Zorbifold.smmx",
        [
            {
                "id": 0,
                "text": "Private: Zorbifold notes",
                "urls": ("https://www.pearltrees.com/private/id789?access=abc",),
            }
        ],
    )
    record = scan_maps(tmp_path)[0]
    assert record["classification"] == "private"
    assert "private_dir_or_access_prefix" in record["reasons"]


@pytest.mark.parametrize(
    "directory",
    ("Private_Research", "Private: Research", "Private - Research"),
)
def test_access_prefix_directory_components_fail_closed(tmp_path, directory):
    _write_map(
        tmp_path / directory / "Ordinary map.smmx",
        [{"id": 0, "text": "Ordinary map"}],
    )
    record = scan_maps(tmp_path)[0]
    assert record["classification"] == "private"
    assert "private_dir_or_access_prefix" in record["reasons"]


def test_access_prefix_directory_and_public_link_conflict_is_unknown(tmp_path):
    _write_map(
        tmp_path / "Private_Research" / "Ordinary map.smmx",
        [
            {
                "id": 0,
                "text": "Ordinary map",
                "urls": (
                    "https://www.pearltrees.com/s243a/ordinary-map/id42",
                ),
            }
        ],
    )
    record = scan_maps(tmp_path)[0]
    assert record["classification"] == "unknown"
    assert "access_marker_conflicts_with_public_root" in record["reasons"]


def test_private_marker_targets_specific_linked_map_not_its_public_parent(tmp_path):
    public_parent = tmp_path / "Parent.smmx"
    source = tmp_path / "index" / "Index.smmx"
    private_target = tmp_path / "private-maps" / "Course copy.smmx"
    _write_map(public_parent, [{"id": 0, "text": "Public parent"}])
    _write_map(
        private_target,
        [
            {"id": 0, "text": "Course copy"},
            {
                "id": 1,
                "parent": 0,
                "text": "Navigate up",
                "refs": ("../Parent.smmx",),
            },
        ],
    )
    _write_map(
        source,
        [
            {"id": 0, "text": "Index"},
            {"id": 1, "parent": 0, "text": "private"},
            {"id": 2, "parent": 1, "text": "Course"},
            {
                "id": 3,
                "parent": 2,
                "text": "",
                "refs": ("../private-maps/Course copy.smmx",),
            },
        ],
    )
    records = scan_maps(tmp_path)
    assert _record_by_name(records, "Course copy.smmx")["classification"] == "private"
    assert "private_cloud_child" in _record_by_name(
        records, "Course copy.smmx"
    )["reasons"]
    assert _record_by_name(records, "Parent.smmx")["classification"] == "public"


def test_public_root_conflict_with_private_cloud_child_is_unknown(tmp_path):
    target = tmp_path / "Public target.smmx"
    _write_map(
        target,
        [
            {
                "id": 0,
                "text": "Public target",
                "urls": (
                    "https://www.pearltrees.com/s243a/public-target/id456",
                ),
            }
        ],
    )
    _write_map(
        tmp_path / "Index.smmx",
        [
            {"id": 0, "text": "Index"},
            {"id": 1, "parent": 0, "text": "private"},
            {
                "id": 2,
                "parent": 1,
                "text": "",
                "refs": ("Public target.smmx",),
            },
        ],
    )
    record = _record_by_name(scan_maps(tmp_path), "Public target.smmx")
    assert record["classification"] == "unknown"
    assert "private_cloud_child_conflicts_with_public_root" in record["reasons"]


def test_topical_private_key_in_exact_private_directory_is_ambiguous(tmp_path):
    _write_map(
        tmp_path / "Private" / "Private-key cryptography.smmx",
        [{"id": 0, "text": "Private-key cryptography"}],
    )
    record = scan_maps(tmp_path)[0]
    assert record["classification"] == "unknown"
    assert "private_marker_with_topical_root" in record["reasons"]


def test_topical_privacy_link_does_not_mark_cloud_target_private(tmp_path):
    target = tmp_path / "Target.smmx"
    _write_map(target, [{"id": 0, "text": "Target"}])
    _write_map(
        tmp_path / "Index.smmx",
        [
            {"id": 0, "text": "Index"},
            {
                "id": 1,
                "parent": 0,
                "text": "private",
                "urls": ("https://en.wikipedia.org/wiki/Privacy",),
            },
            {"id": 2, "parent": 1, "text": "", "refs": ("Target.smmx",)},
        ],
    )
    records = scan_maps(tmp_path)
    assert _record_by_name(records, "Target.smmx")["classification"] == "public"


def test_unrelated_public_link_does_not_disable_private_marker(tmp_path):
    target = tmp_path / "Target.smmx"
    _write_map(target, [{"id": 0, "text": "Target"}])
    _write_map(
        tmp_path / "Index.smmx",
        [
            {"id": 0, "text": "Index"},
            {
                "id": 1,
                "parent": 0,
                "text": "private",
                "urls": ("https://example.com/unrelated",),
            },
            {"id": 2, "parent": 1, "text": "", "refs": ("Target.smmx",)},
        ],
    )
    record = _record_by_name(scan_maps(tmp_path), "Target.smmx")
    assert record["classification"] == "private"
    assert "private_cloud_child" in record["reasons"]


def test_virtual_root_cloudref_and_outside_ref_resolution(tmp_path):
    source = tmp_path / "branch" / "Source.smmx"
    source.parent.mkdir(parents=True)
    inside = resolve_cloudmapref(tmp_path, source, "/root/other/Target.smmx")
    assert inside == tmp_path / "other" / "Target.smmx"
    assert resolve_cloudmapref(tmp_path, source, "../../../outside.smmx") is None
    assert resolve_cloudmapref(tmp_path, source, ".") is None


def test_legacy_root_relative_cloudref_is_resolved_when_unambiguous(tmp_path):
    source = tmp_path / "branch" / "Source.smmx"
    source.parent.mkdir(parents=True)
    target = tmp_path / "Subjects" / "Target.smmx"
    _write_map(target, [{"id": 0, "text": "Target"}])
    assert (
        resolve_cloudmapref(tmp_path, source, "Subjects/Target.smmx")
        == target
    )


def test_case_different_private_cloud_target_resolves_to_canonical_member(
    tmp_path,
):
    target = tmp_path / "Targets" / "Target.smmx"
    _write_map(target, [{"id": 0, "text": "Target"}])
    _write_map(
        tmp_path / "Index.smmx",
        [
            {"id": 0, "text": "Index"},
            {"id": 1, "parent": 0, "text": "private"},
            {
                "id": 2,
                "parent": 1,
                "text": "",
                "refs": ("targets/target.smmx",),
            },
        ],
    )
    records = scan_maps(tmp_path, workers=1)
    source = _record_by_name(records, "Index.smmx")
    canonical_target = _record_by_name(records, "Targets/Target.smmx")
    assert source["unresolved_private_targets"] == []
    assert canonical_target["classification"] == "private"
    assert "private_cloud_child" in canonical_target["reasons"]


def test_missing_private_cloud_target_is_counted_unresolved(tmp_path):
    _write_map(
        tmp_path / "Index.smmx",
        [
            {"id": 0, "text": "Index"},
            {"id": 1, "parent": 0, "text": "private"},
            {
                "id": 2,
                "parent": 1,
                "text": "",
                "refs": ("Missing/Target.smmx",),
            },
        ],
    )
    header, records = build_index(tmp_path, workers=1)
    assert records[0]["unresolved_private_targets"] == [
        "Missing/Target.smmx"
    ]
    assert "unresolved_private_map_links" in records[0]["reasons"]
    assert header["counts"]["unresolved_private_target_refs"] == 1


def test_apply_model_reviews_cannot_emit_corpus_metadata(tmp_path):
    _write_map(
        tmp_path / "STEM" / "MMSE.smmx",
        [
            {
                "id": 0,
                "text": "Minimum mean square error",
                "urls": ("https://www.pearltrees.com/private/id222?access=x",),
            }
        ],
    )
    records = scan_maps(tmp_path)
    before = json.loads(json.dumps(records))
    called = False

    def must_not_call(*_args):
        nonlocal called
        called = True
        return "[]"

    with pytest.raises(SmFsPrivacyError, match="model review is disabled"):
        apply_model_reviews(
            records,
            provider="ollama-json",
            model="phi3:latest",
            batch_size=20,
            timeout=11,
            call_llm=must_not_call,
        )
    assert called is False
    assert records == before


def test_corpus_model_review_is_disabled_without_verified_lock(tmp_path):
    _write_map(tmp_path / "Map.smmx", [{"id": 0, "text": "Map"}])
    called = False

    def must_not_call(*_args):
        nonlocal called
        called = True
        return "[]"

    with pytest.raises(SmFsPrivacyError, match="verified passing benchmark lock"):
        build_index(
            tmp_path,
            provider="ollama-json",
            model="qwen3:4b",
            call_llm=must_not_call,
        )
    assert called is False


def test_corpus_model_review_rejects_hosted_provider_with_same_closed_gate(
    tmp_path,
):
    _write_map(tmp_path / "Map.smmx", [{"id": 0, "text": "Map"}])
    with pytest.raises(SmFsPrivacyError, match="verified passing benchmark lock"):
        build_index(
            tmp_path,
            provider="openai",
            model="gpt-test",
            call_llm=lambda *_args: "[]",
        )


def test_symlink_corpus_member_is_rejected(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    external = tmp_path / "External.smmx"
    _write_map(external, [{"id": 0, "text": "External"}])
    (corpus / "Alias.smmx").symlink_to(external)
    with pytest.raises(SmFsPrivacyError, match="symlink corpus member"):
        scan_maps(corpus, workers=1)


def test_record_read_uses_no_follow_even_after_discovery(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    member = corpus / "Map.smmx"
    _write_map(member, [{"id": 0, "text": "Map"}])
    external = tmp_path / "External.smmx"
    _write_map(external, [{"id": 0, "text": "External"}])
    real_members = sm_fs_privacy.smmx_member_bindings

    def swap_file(root_fd):
        bindings = real_members(root_fd)
        member.unlink()
        member.symlink_to(external)
        return bindings

    monkeypatch.setattr(
        sm_fs_privacy,
        "smmx_member_bindings",
        swap_file,
    )
    with pytest.raises(SmFsPrivacyError, match="symlink corpus member"):
        scan_maps(corpus, workers=1)


def test_retained_root_descriptor_defeats_root_path_swap(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    _write_map(corpus / "Original.smmx", [{"id": 0, "text": "Original"}])
    replacement = tmp_path / "replacement"
    _write_map(
        replacement / "Original.smmx",
        [{"id": 0, "text": "Replacement"}],
    )
    detached = tmp_path / "detached-corpus"
    real_members = sm_fs_privacy.smmx_member_bindings

    def swap_root(root_fd):
        members = real_members(root_fd)
        corpus.rename(detached)
        corpus.symlink_to(replacement, target_is_directory=True)
        return members

    monkeypatch.setattr(sm_fs_privacy, "smmx_member_bindings", swap_root)
    records = scan_maps(corpus, workers=1)
    assert [record["root_title"] for record in records] == ["Original"]


def test_parent_swap_to_symlink_fails_closed(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    parent = corpus / "Branch"
    _write_map(parent / "Map.smmx", [{"id": 0, "text": "Original"}])
    replacement = tmp_path / "replacement"
    _write_map(
        replacement / "Map.smmx",
        [{"id": 0, "text": "Replacement"}],
    )
    detached = corpus / "Detached"
    real_members = sm_fs_privacy.smmx_member_bindings

    def swap_parent(root_fd):
        members = real_members(root_fd)
        parent.rename(detached)
        parent.symlink_to(replacement, target_is_directory=True)
        return members

    monkeypatch.setattr(sm_fs_privacy, "smmx_member_bindings", swap_parent)
    with pytest.raises(SmFsPrivacyError, match="symlink corpus directory"):
        scan_maps(corpus, workers=1)


def test_parent_swap_to_real_directory_fails_identity_binding(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    parent = corpus / "Branch"
    _write_map(parent / "Map.smmx", [{"id": 0, "text": "Original"}])
    replacement = tmp_path / "replacement"
    _write_map(
        replacement / "Map.smmx",
        [{"id": 0, "text": "Replacement"}],
    )
    detached = corpus / "Detached"
    real_members = sm_fs_privacy.smmx_member_bindings

    def swap_parent(root_fd):
        bindings = real_members(root_fd)
        parent.rename(detached)
        replacement.rename(parent)
        return bindings

    monkeypatch.setattr(sm_fs_privacy, "smmx_member_bindings", swap_parent)
    with pytest.raises(SmFsPrivacyError, match="directory identity changed"):
        scan_maps(corpus, workers=1)


def test_smmx_named_directory_is_rejected(tmp_path):
    (tmp_path / "LooksLikeMap.smmx").mkdir()
    with pytest.raises(SmFsPrivacyError, match="non-regular corpus member"):
        scan_maps(tmp_path, workers=1)


def test_stable_read_allows_drvfs_hydration_ctime_atime_churn(
    tmp_path, monkeypatch
):
    member = tmp_path / "Map.smmx"
    _write_map(member, [{"id": 0, "text": "Map"}])
    expected = member.read_bytes()
    real_fstat = sm_fs_privacy.os.fstat
    calls = 0

    class HydratedStat:
        def __init__(self, base, shift):
            self._base = base
            self.st_ctime_ns = base.st_ctime_ns + shift
            self.st_atime_ns = base.st_atime_ns + shift

        def __getattr__(self, name):
            return getattr(self._base, name)

    def hydrating_fstat(fd):
        nonlocal calls
        calls += 1
        return HydratedStat(real_fstat(fd), calls)

    monkeypatch.setattr(sm_fs_privacy.os, "fstat", hydrating_fstat)
    assert sm_fs_privacy._read_regular_file_no_follow(member) == expected
    assert calls == 3


def test_stable_read_rejects_different_consecutive_descriptor_bytes(
    tmp_path, monkeypatch
):
    member = tmp_path / "Map.smmx"
    _write_map(member, [{"id": 0, "text": "Map"}])
    real_read_all = sm_fs_privacy._read_all_from_fd
    calls = 0

    def mutate_after_first_read(fd):
        nonlocal calls
        data = real_read_all(fd)
        calls += 1
        if calls == 1:
            member.write_bytes(b"X" * len(data))
        return data

    monkeypatch.setattr(
        sm_fs_privacy, "_read_all_from_fd", mutate_after_first_read
    )
    with pytest.raises(SmFsPrivacyError, match="file bytes changed"):
        sm_fs_privacy._read_regular_file_no_follow(member)


def test_unreadable_map_is_unknown_and_does_not_abort_scan(tmp_path):
    broken = tmp_path / "Broken.smmx"
    broken.write_bytes(b"not a zip")
    record = scan_maps(tmp_path)[0]
    assert record["classification"] == "unknown"
    assert record["reasons"] == ["map_parse_error"]
    assert record["parse_error_type"]


def test_index_roundtrip_binds_source_snapshot(tmp_path):
    corpus = tmp_path / "corpus"
    _write_map(corpus / "Folder" / "Map.smmx", [{"id": 0, "text": "Map"}])
    header, records = build_index(corpus)
    assert header["model"] is None
    assert header["model_review_authorization"] == {
        "authorized": False,
        "reason": MODEL_REVIEW_DISABLED_REASON,
    }
    index = tmp_path / "privacy.jsonl"
    write_index(index, header, records)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_index(index, header, records)
    loaded_header, by_path = load_index(index)
    verify_index_source(corpus, loaded_header, by_path)
    assert loaded_header["parser_sha256"] == sha256_file(PARSER_PATH)
    stale_parser = dict(loaded_header)
    stale_parser["parser_sha256"] = "0" * 64
    with pytest.raises(SmFsPrivacyError, match="parser changed"):
        verify_index_source(corpus, stale_parser, by_path)
    (corpus / "Folder" / "Map.smmx").write_bytes(b"changed")
    with pytest.raises(SmFsPrivacyError, match="map contents changed"):
        verify_index_source(corpus, loaded_header, by_path)


def test_load_rejects_resealed_v1_model_and_derived_metadata_tampering(
    tmp_path,
):
    corpus = tmp_path / "corpus"
    _write_map(corpus / "Map.smmx", [{"id": 0, "text": "Map"}])
    original_header, original_records = build_index(corpus, workers=1)

    def set_header_model(header, _records):
        header["model"] = {"provider": "local", "name": "adversarial"}

    def authorize_model_review(header, _records):
        header["model_review_authorization"]["authorized"] = True

    def add_row_model_review(_header, records):
        records[0]["model_review"] = {
            "provider": "local",
            "model": "adversarial",
        }

    def alter_derived_counts(header, _records):
        header["counts"]["public"] += 1

    def alter_derived_snapshot(header, _records):
        header["source_snapshot"]["file_count"] += 1

    cases = (
        ("header-model", set_header_model, "cannot name a model"),
        ("authorization", authorize_model_review, "authorization changed"),
        ("row-model", add_row_model_review, "model review data"),
        ("counts", alter_derived_counts, "derived counts changed"),
        ("snapshot", alter_derived_snapshot, "source snapshot changed"),
    )
    for name, mutate, message in cases:
        header = json.loads(json.dumps(original_header))
        records = json.loads(json.dumps(original_records))
        mutate(header, records)
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(_resealed_index_bytes(header, records))
        with pytest.raises(SmFsPrivacyError, match=message):
            load_index(path)


def test_verify_itself_enforces_v1_no_model_and_derived_counts(tmp_path):
    corpus = tmp_path / "corpus"
    _write_map(corpus / "Map.smmx", [{"id": 0, "text": "Map"}])
    header, records = build_index(corpus, workers=1)
    by_path = {record["relative_path"]: record for record in records}

    by_path["Map.smmx"]["model_review"] = {"model": "adversarial"}
    with pytest.raises(SmFsPrivacyError, match="model review data"):
        verify_index_source(corpus, header, by_path, workers=1)

    by_path["Map.smmx"]["model_review"] = None
    bad_counts = json.loads(json.dumps(header))
    bad_counts["counts"]["public"] += 1
    with pytest.raises(SmFsPrivacyError, match="derived counts changed"):
        verify_index_source(corpus, bad_counts, by_path, workers=1)


def test_resealed_label_tampering_fails_exact_classifier_rederivation(
    tmp_path,
):
    corpus = tmp_path / "corpus"
    _write_map(corpus / "Map.smmx", [{"id": 0, "text": "Map"}])
    header, records = build_index(corpus, workers=1)
    assert records[0]["classification"] == "public"
    records[0]["classification"] = "unknown"
    header["counts"]["public"] = 0
    header["counts"]["unknown"] = 1

    index = tmp_path / "resealed-label.jsonl"
    index.write_bytes(_resealed_index_bytes(header, records))
    loaded_header, by_path = load_index(index)
    with pytest.raises(
        SmFsPrivacyError, match="differs from exact classifier output"
    ):
        verify_index_source(corpus, loaded_header, by_path, workers=1)


def test_filing_policy_filters_private_but_can_retain_unknown(tmp_path):
    root = tmp_path / "corpus"
    _write_map(root / "Public" / "One.smmx", [{"id": 0, "text": "One"}])
    _write_map(
        root / "Private" / "Private_Two.smmx",
        [{"id": 0, "text": "Private: Two"}],
    )
    broken = root / "Unknown" / "Three.smmx"
    broken.parent.mkdir(parents=True)
    broken.write_bytes(b"bad")
    _, records = build_index(root)
    by_path = {row["relative_path"]: row for row in records}

    exclude_private, counts = discover_filing_rows(
        root, by_path, "exclude-private"
    )
    assert {row["privacy"] for row in exclude_private} == {"public", "unknown"}
    assert counts == {"public": 1, "private": 1, "unknown": 1}

    public_only, _ = discover_filing_rows(root, by_path, "public-only")
    assert [row["title"] for row in public_only] == ["One"]

    all_local, _ = discover_filing_rows(root, by_path, "all-local")
    assert len(all_local) == 3


def test_public_filing_refuses_unresolved_private_targets():
    header = {"counts": {"unresolved_private_target_refs": 2}}
    with pytest.raises(SmFsPrivacyError, match="not authorized"):
        require_public_privacy_perimeter(header, "public-only")
    require_public_privacy_perimeter(header, "exclude-private")


def test_embedding_cache_is_private(tmp_path):
    cache = prepare_private_cache(tmp_path / "cache" / "embeddings.pt")
    cache.write_bytes(b"synthetic")
    cache.chmod(0o644)
    harden_private_cache(cache)
    assert (cache.parent.stat().st_mode & 0o777) == 0o700
    assert (cache.stat().st_mode & 0o777) == 0o600


def test_filing_provenance_pins_e5_and_generator_source():
    provenance = filing_provenance()
    assert provenance["e5_revision"] == E5_REVISION
    assert provenance["sm_fs_filing_sha256"] == sha256_file(
        HERE / "sm_fs_filing.py"
    )


def test_filing_runtime_passes_and_records_pinned_e5_revision(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    _write_map(
        corpus / "Folder" / "One.smmx",
        [{"id": 0, "text": "One"}],
    )
    privacy_index = tmp_path / "privacy.jsonl"
    header, records = build_index(corpus, workers=1)
    write_index(privacy_index, header, records)
    observed = {}

    class ArrayResult:
        def __init__(self, array):
            self._array = array

        def numpy(self):
            return self._array

    def fake_build_e5_tables(names, **kwargs):
        observed.update(kwargs)
        matrix = np.eye(len(names), dtype=np.float32)
        return ArrayResult(matrix), ArrayResult(matrix), {
            name: index for index, name in enumerate(names)
        }

    monkeypatch.setattr(
        sm_fs_filing, "build_e5_tables", fake_build_e5_tables
    )
    ledger = tmp_path / "private" / "ledger.json"
    sm_fs_filing.main(
        [
            "--root",
            str(corpus),
            "--privacy-index",
            str(privacy_index),
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
    assert observed["model_revision"] == E5_REVISION
    ledger_value = json.loads(ledger.read_text(encoding="utf-8"))
    expected_provenance = filing_provenance()
    assert {
        key: ledger_value["provenance"][key] for key in expected_provenance
    } == expected_provenance
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        ledger_value["provenance"]["e5_cache_sha256"],
    )


def test_synthetic_benchmark_is_balanced_and_strictly_scored():
    cases = synthetic_cases()
    assert len(cases) == 12
    assert {
        expected: sum(case["expected"] == expected for case in cases)
        for expected in ("access_control", "topical", "uncertain")
    } == {"access_control": 4, "topical": 4, "uncertain": 4}
    response = json.dumps(
        [
            {
                "qid": case["task"]["qid"],
                "interpretation": case["expected"],
                "reason": "synthetic expected label",
            }
            for case in cases
        ]
    )
    score = score_response(response, cases)
    assert score["raw_unfenced_json"] is True
    assert score["schema_ok"] is True
    assert score["correct"] == 12
    assert score["review_eligible_correct"] == 7
