"""Fixture tests for content-bound privacy views, races, and reproduction."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import pytest
import sm_fs_freeze
import sm_fs_privacy
from sm_fs_privacy import build_index, write_index


def write_map(path, title, urls=()):
    document = ET.Element("mindmap")
    root = ET.SubElement(
        document, "topic", {"id": "0", "parent": "-1", "text": title}
    )
    for url in urls:
        ET.SubElement(root, "link", {"urllink": url})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document/mindmap.xml", ET.tostring(document))


def write_unresolved_private_marker_map(path):
    document = ET.Element("mindmap")
    ET.SubElement(
        document, "topic", {"id": "0", "parent": "-1", "text": "Index"}
    )
    ET.SubElement(
        document, "topic", {"id": "1", "parent": "0", "text": "private"}
    )
    holder = ET.SubElement(
        document, "topic", {"id": "2", "parent": "1", "text": ""}
    )
    ET.SubElement(holder, "link", {"cloudmapref": "missing/Target.smmx"})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document/mindmap.xml", ET.tostring(document))


def make_tree(tmp_path):
    tmp_path = tmp_path / "tree"          # corpus root separate from test output dirs
    for p, maps in {
        "Subjects/sci/phy/mech": ["Newton", "Lagrange"],
        "Subjects/sci/phy/quantum": ["Qubits"],
        "Subjects/art/music": ["Jazz", "Blues"],
        "Subjects/art/paint": ["Oil"],
        "Private/journal": ["Secrets"],
        "Subjects/sci/private_notes": ["Hidden"],
        "Other/misc": ["Jazz"],                       # duplicate leaf title in another path
    }.items():
        d = tmp_path / p
        d.mkdir(parents=True, exist_ok=True)
        for m in maps:
            write_map(d / f"{m}.smmx", m)
    write_map(
        tmp_path / "Subjects" / "sci" / "private_notes" / "Hidden.smmx",
        "Hidden",
        ("https://www.pearltrees.com/private/id222?access=synthetic",),
    )
    return tmp_path


def run(
    tmp_path,
    out,
    *,
    privacy_index,
    policy=None,
    holdout_frac="0.4",
    allow_degenerate_holdout=False,
):
    args = [
        "--fs-root", str(tmp_path),
        "--out-dir", str(out),
        "--holdout-frac", holdout_frac,
        "--split-seed", "0",
        "--training-privacy-policy",
        policy or "public-only",
        "--privacy-index", str(privacy_index),
    ]
    if allow_degenerate_holdout:
        args.append("--allow-degenerate-holdout")
    sm_fs_freeze.main(args)
    return json.load(open(out / "ledger.json"))


def build_privacy_index(root, path):
    header, records = build_index(Path(root), workers=1)
    write_index(Path(path), header, records)
    return header


@pytest.mark.parametrize("value", ("-1", "2", "nan", "inf", "-inf", "0", "1"))
def test_holdout_fraction_fails_closed_before_reading_inputs(tmp_path, value):
    with pytest.raises(ValueError, match="holdout"):
        sm_fs_freeze.main(
            [
                "--fs-root",
                str(tmp_path / "missing-corpus"),
                "--privacy-index",
                str(tmp_path / "missing-index"),
                    "--out-dir",
                    str(tmp_path / "out"),
                    f"--holdout-frac={value}",
            ]
        )


def test_nonendpoint_degenerate_waiver_is_rejected():
    with pytest.raises(ValueError, match="valid only.*0 or 1"):
        sm_fs_freeze._validate_holdout_fraction(
            0.4,
            allow_degenerate=True,
        )


def test_realized_single_lineage_split_fails_closed():
    with pytest.raises(ValueError, match="realized split"):
        sm_fs_freeze.derive_split(
            [{"dest": "Synthetic/Only"}],
            0.4,
            0,
        )


def test_zero_usable_training_targets_require_diagnostic_waiver():
    with pytest.raises(ValueError, match="no usable training targets"):
        sm_fs_freeze._validate_training_target_count(
            0,
            allow_degenerate=False,
        )
    sm_fs_freeze._validate_training_target_count(
        0,
        allow_degenerate=True,
    )


def test_source_map_discovery_rejects_symlinks(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tmp_path / "target.smmx"
    write_map(target, "Target")
    (tree / "Alias.smmx").symlink_to(target)

    with pytest.raises(ValueError, match="refusing symlink corpus member"):
        sm_fs_freeze._smmx_paths(tree)


def test_source_verifier_keeps_original_root_across_path_swap(
    tmp_path, monkeypatch
):
    tree = tmp_path / "tree"
    (tree / "Branch").mkdir(parents=True)
    write_map(tree / "Branch" / "Original.smmx", "Original")
    _header, records = build_index(tree, workers=1)
    index = {record["relative_path"]: record for record in records}

    replacement = tmp_path / "replacement"
    (replacement / "Branch").mkdir(parents=True)
    write_map(replacement / "Branch" / "Original.smmx", "Replacement")
    detached = tmp_path / "detached-tree"
    real_members = sm_fs_privacy.smmx_member_bindings

    def swap_root(root_fd):
        members = real_members(root_fd)
        tree.rename(detached)
        tree.symlink_to(replacement, target_is_directory=True)
        return members

    monkeypatch.setattr(sm_fs_privacy, "smmx_member_bindings", swap_root)
    observed = list(sm_fs_freeze._iter_verified_source_maps(tree, index))
    assert [relative for relative, *_rest in observed] == [
        "Branch/Original.smmx"
    ]
    assert observed[0][2] == index["Branch/Original.smmx"]["file_sha256"]


def rewrite_ledger_and_rebind_manifest(out, edit, manifest_edit=None):
    ledger_path = out / "ledger.json"
    manifest_path = out / "manifest.json"
    ledger = json.loads(ledger_path.read_bytes())
    edit(ledger)
    ledger_bytes = (
        json.dumps(ledger, ensure_ascii=False, indent=0, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    ledger_path.write_bytes(ledger_bytes)

    manifest = json.loads(manifest_path.read_bytes())
    manifest["outputs"]["ledger.json"] = {
        "bytes": len(ledger_bytes),
        "sha256": hashlib.sha256(ledger_bytes).hexdigest(),
    }
    if manifest_edit is not None:
        manifest_edit(manifest, ledger)
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        sm_fs_freeze.canonical_json_bytes(core)
    ).hexdigest()
    manifest_path.write_bytes(sm_fs_freeze.canonical_json_bytes(manifest))


def assert_catalog_closes_rows_and_targets(ledger, target_path):
    catalog = set(ledger["catalog"])
    for row in ledger["rows"]:
        assert row["dest"] in catalog
        parts = row["dest"].split("/")
        assert {
            "/".join(parts[:depth]) for depth in range(1, len(parts) + 1)
        }.issubset(catalog)
    for line in target_path.read_text().splitlines():
        if not line.startswith("#"):
            assert line.split("\t")[2] in catalog


def test_content_addressed_privacy_and_split_isolation(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    led = run(tree, tmp_path / "out", privacy_index=privacy_path)
    assert led["privacy_source"] == "content_addressed_index"
    assert {row["classification"] for row in led["rows"]} == {"public"}
    assert not any(path == "Private" or path.startswith("Private/") for path in led["catalog"])
    assert "Subjects/sci/private_notes" not in led["catalog"]
    assert_catalog_closes_rows_and_targets(
        led, tmp_path / "out" / "lineage_fs_targets.tsv"
    )
    # split isolation is bidirectional across the destination lineage.
    res = {r["dest"] for r in led["rows"] if r["split"] == "reserved"}
    exp = {r["dest"] for r in led["rows"] if r["split"] == "explore"}
    for e in exp:
        assert not any(
            e == d or e.startswith(d + "/") or d.startswith(e + "/")
            for d in res
        )
    # all maps of one destination share a split side
    side = {}
    for r in led["rows"]:
        assert side.setdefault(r["dest"], r["split"]) == r["split"]


def test_duplicate_titles_stay_distinct_and_targets_explore_only(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    led = run(tree, out, privacy_index=privacy_path)
    jazz = [r for r in led["rows"] if r["title"] == "Jazz"]
    assert len(jazz) == 2 and len({r["map_path"] for r in jazz}) == 2       # exact-path identity
    hdr = open(out / "lineage_fs_targets.tsv").read().splitlines()
    assert "lineage(fs,decay=0.85)" in hdr[0]                                # process expression
    explore_paths = {r["map_path"] for r in led["rows"] if r["split"] == "explore"}
    for ln in hdr:
        if ln.startswith("#"):
            continue
        assert ln.split("\t")[0] in explore_paths                            # explore-only targets


def test_reproducible_membership(tmp_path):
    t = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(t, privacy_path)
    l1 = run(t, tmp_path / "o1", privacy_index=privacy_path)
    l2 = run(t, tmp_path / "o2", privacy_index=privacy_path)
    m1 = {(r["map_path"], r["split"]) for r in l1["rows"]}
    assert m1 == {(r["map_path"], r["split"]) for r in l2["rows"]}           # identical membership
    assert l1["tree_snapshot"] == l2["tree_snapshot"]
    assert l1["e5_revision"] == l2["e5_revision"]


def test_content_addressed_index_binds_public_only_training_targets(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    privacy_header = build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    ledger = run(
        tree,
        out,
        privacy_index=privacy_path,
        policy="public-only",
        holdout_frac="0",
        allow_degenerate_holdout=True,
    )

    assert ledger["privacy_source"] == "content_addressed_index"
    assert ledger["privacy_index"]["index_sha256"] == privacy_header["index_sha256"]
    assert ledger["training_privacy_policy"] == "public-only"
    by_title = {row["title"]: row for row in ledger["rows"]}
    assert "Secrets" not in by_title
    assert "Hidden" not in by_title
    assert ledger["privacy_index"]["observed_counts"] == {
        classification: privacy_header["counts"][classification]
        for classification in ("private", "public", "unknown")
        if privacy_header["counts"][classification]
    }

    targets = (out / "lineage_fs_targets.tsv").read_text().splitlines()
    data = [line.split("\t") for line in targets if not line.startswith("#")]
    assert data
    assert {row[-1] for row in data} == {"public"}
    assert not any(row[1] in {"Secrets", "Hidden"} for row in data)


def test_private_only_training_is_separate_and_keeps_exact_identity(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    ledger = run(
        tree,
        out,
        privacy_index=privacy_path,
        policy="private-only",
        holdout_frac="0",
        allow_degenerate_holdout=True,
    )

    targets = (out / "lineage_fs_targets.tsv").read_text().splitlines()
    data = [line.split("\t") for line in targets if not line.startswith("#")]
    assert data
    assert {row[-1] for row in data} == {"private"}
    assert {row["classification"] for row in ledger["rows"]} == {"private"}
    assert ledger["catalog"] == sm_fs_freeze.catalog_for_rows(ledger["rows"])
    assert all(
        len(row) == 7
        and row[0].endswith(".smmx")
        and "/" in row[0]
        and row[2]
        for row in data
    )
    assert_catalog_closes_rows_and_targets(
        ledger, out / "lineage_fs_targets.tsv"
    )


def test_private_only_excludes_unreadable_private_map_from_targets(tmp_path):
    tree = make_tree(tmp_path)
    broken = tree / "Private" / "journal" / "Unreadable.smmx"
    broken.write_bytes(b"")
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    ledger = run(
        tree,
        out,
        privacy_index=privacy_path,
        policy="private-only",
        holdout_frac="0",
        allow_degenerate_holdout=True,
    )

    unreadable = next(
        row for row in ledger["rows"] if row["title"] == "Unreadable"
    )
    assert unreadable["training_usable"] is False
    target_text = (out / "lineage_fs_targets.tsv").read_text()
    assert "Unreadable.smmx" not in target_text


def test_public_only_refuses_unaudited_fallback(tmp_path):
    tree = make_tree(tmp_path)
    with pytest.raises(
        ValueError, match="requires a content-addressed privacy index"
    ):
        run(
            tree,
            tmp_path / "out",
            privacy_index=tmp_path / "missing.jsonl",
            policy="public-only",
        )


def test_freezer_rejects_index_after_source_change(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    write_map(tree / "Subjects" / "art" / "music" / "Jazz.smmx", "Changed")
    with pytest.raises(ValueError, match="map contents changed"):
        run(tree, tmp_path / "out", privacy_index=privacy_path)


def test_mixed_case_smmx_member_cannot_vanish_from_bundle(tmp_path):
    tree = make_tree(tmp_path)
    lower = tree / "Subjects" / "art" / "paint" / "Oil.smmx"
    mixed = lower.with_name("Oil.SmMx")
    lower.rename(mixed)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)

    ledger = run(
        tree,
        tmp_path / "out",
        privacy_index=privacy_path,
        holdout_frac="0",
        allow_degenerate_holdout=True,
    )

    row = next(row for row in ledger["rows"] if row["title"] == "Oil")
    assert row["map_path"] == "Subjects/art/paint/Oil.SmMx"
    assert row["sha256"] == hashlib.sha256(mixed.read_bytes()).hexdigest()


def test_policy_eligible_root_map_is_rejected_instead_of_silently_omitted(
    tmp_path,
):
    tree = make_tree(tmp_path)
    write_map(tree / "AtRoot.smmx", "At root")
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="has no destination"):
        run(tree, out, privacy_index=privacy_path)
    assert not out.exists()


def test_source_change_after_initial_check_is_rejected_at_inclusion(
    tmp_path, monkeypatch
):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    target = tree / "Subjects" / "art" / "music" / "Jazz.smmx"
    original_verify = sm_fs_freeze.verify_index_source
    calls = 0

    def verify_then_race(*args, **kwargs):
        nonlocal calls
        original_verify(*args, **kwargs)
        calls += 1
        if calls == 1:
            write_map(target, "Changed after initial verification")

    monkeypatch.setattr(
        sm_fs_freeze, "verify_index_source", verify_then_race
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="map contents changed"):
        run(tree, out, privacy_index=privacy_path)
    assert not out.exists()


def test_policy_eligible_member_removed_after_initial_check_is_rejected(
    tmp_path, monkeypatch
):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    target = tree / "Subjects" / "art" / "paint" / "Oil.smmx"
    original_verify = sm_fs_freeze.verify_index_source
    calls = 0

    def verify_then_remove(*args, **kwargs):
        nonlocal calls
        original_verify(*args, **kwargs)
        calls += 1
        if calls == 1:
            target.unlink()

    monkeypatch.setattr(
        sm_fs_freeze, "verify_index_source", verify_then_remove
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="map set changed"):
        run(tree, out, privacy_index=privacy_path)
    assert not out.exists()


def test_public_only_blocks_unresolved_private_target_without_waiver(tmp_path):
    tree = make_tree(tmp_path)
    write_unresolved_private_marker_map(
        tree / "Subjects" / "sci" / "phy" / "Unresolved.smmx"
    )
    privacy_path = tmp_path / "privacy.jsonl"
    header = build_privacy_index(tree, privacy_path)
    assert header["counts"]["unresolved_private_target_refs"] == 1
    with pytest.raises(ValueError, match="unresolved private target"):
        run(tree, tmp_path / "out", privacy_index=privacy_path)


def test_cross_lineage_rows_are_excluded_bidirectionally():
    rows = [
        {"dest": "A/B", "split": "reserved"},
        {"dest": "A/B/C", "split": "explore"},
        {"dest": "A", "split": "explore"},
        {"dest": "X/Y", "split": "explore"},
    ]
    assert sm_fs_freeze.exclude_cross_lineage(rows) == 2
    assert [row["split"] for row in rows] == [
        "reserved",
        "cross_lineage_excluded",
        "cross_lineage_excluded",
        "explore",
    ]


def test_frozen_outputs_are_private_and_no_replace(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    run(tree, out, privacy_index=privacy_path)
    assert (out.stat().st_mode & 0o777) == 0o700
    assert ((out / "ledger.json").stat().st_mode & 0o777) == 0o600
    assert ((out / "lineage_fs_targets.tsv").stat().st_mode & 0o777) == 0o600
    assert ((out / "manifest.json").stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError, match="refusing to replace"):
        run(tree, out, privacy_index=privacy_path)


def test_bundle_verifier_rejects_hardlinked_artifact(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    run(tree, out, privacy_index=privacy_path)
    os.link(out / "ledger.json", tmp_path / "ledger-hardlink.json")

    with pytest.raises(ValueError, match="exactly one hard link"):
        sm_fs_freeze.verify_private_bundle(out)


def test_bundle_verifier_rejects_symlink_swap_during_descriptor_read(
    tmp_path, monkeypatch
):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    run(tree, out, privacy_index=privacy_path)
    ledger_path = out / "ledger.json"
    replacement = tmp_path / "replacement-ledger.json"
    replacement.write_bytes(ledger_path.read_bytes())
    replacement.chmod(0o600)

    original_read = sm_fs_freeze.os.read
    swapped = False

    def read_then_swap(descriptor, size):
        nonlocal swapped
        data = original_read(descriptor, size)
        if data and not swapped:
            swapped = True
            ledger_path.unlink()
            ledger_path.symlink_to(replacement)
        return data

    monkeypatch.setattr(sm_fs_freeze.os, "read", read_then_swap)
    with pytest.raises(ValueError, match="changed while being read"):
        sm_fs_freeze.verify_private_bundle(out)
    assert swapped


def test_binding_manifest_round_trip_and_provenance(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    privacy_header = build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    ledger = run(
        tree,
        out,
        privacy_index=privacy_path,
        policy="public-only",
        holdout_frac="0",
        allow_degenerate_holdout=True,
    )

    manifest = sm_fs_freeze.verify_private_bundle(
        out, privacy_index=privacy_path
    )
    assert manifest["schema"] == sm_fs_freeze.BUNDLE_SCHEMA
    assert manifest["parameters"] == {
        "allow_degenerate_holdout": True,
        "allow_unresolved_private_targets": False,
        "decay": sm_fs_freeze.DECAY,
        "fs_root": str(tree.resolve()),
        "holdout_frac": 0.0,
        "process_expression": sm_fs_freeze.EXPR,
        "split_seed": 0,
        "training_privacy_policy": "public-only",
    }
    assert ledger["allow_degenerate_holdout"] is True
    assert any(
        "diagnostic" in limitation for limitation in ledger["limitations"]
    )
    assert manifest["inputs"]["privacy_index"]["provenance"][
        "index_sha256"
    ] == privacy_header["index_sha256"]
    privacy_bytes = privacy_path.read_bytes()
    expected_input_artifact = {
        "bytes": len(privacy_bytes),
        "sha256": hashlib.sha256(privacy_bytes).hexdigest(),
    }
    assert (
        manifest["inputs"]["privacy_index"]["artifact"]
        == expected_input_artifact
    )
    assert (
        ledger["privacy_index"]["artifact"] == expected_input_artifact
    )
    assert (
        ledger["privacy_index"]["path"]
        == manifest["inputs"]["privacy_index"]["path"]
        == str(privacy_path.resolve())
    )
    assert manifest["e5_revision"] == sm_fs_freeze.E5_REVISION
    assert set(manifest["code"]) == {
        "mu_attention.py",
        "sm_fs_freeze.py",
        "sm_fs_privacy.py",
    }
    assert all(
        len(record["sha256"]) == 64 for record in manifest["code"].values()
    )
    assert manifest["tree_snapshot"] == ledger["tree_snapshot"]
    for name in ("ledger.json", "lineage_fs_targets.tsv"):
        payload = (out / name).read_bytes()
        assert manifest["outputs"][name] == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def test_all_reserved_diagnostic_bundle_round_trips(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "all-reserved"
    ledger = run(
        tree,
        out,
        privacy_index=privacy_path,
        policy="public-only",
        holdout_frac="1",
        allow_degenerate_holdout=True,
    )

    manifest = sm_fs_freeze.verify_private_bundle(
        out,
        privacy_index=privacy_path,
    )
    assert ledger["counts"]["explore"] == 0
    assert ledger["counts"]["reserved"] == len(ledger["rows"])
    assert manifest["target_rows"] == 0
    assert manifest["parameters"]["allow_degenerate_holdout"] is True
    assert any(
        "diagnostic" in limitation for limitation in ledger["limitations"]
    )


def test_bundle_verifier_rederives_rehashed_split_and_counts(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    run(tree, out, privacy_index=privacy_path)

    def alter_nontraining_split(ledger):
        row = next(
            row for row in ledger["rows"] if row["split"] == "reserved"
        )
        row["split"] = "cross_lineage_excluded"
        ledger["counts"] = {
            split: sum(
                candidate["split"] == split
                for candidate in ledger["rows"]
            )
            for split in (
                "explore",
                "reserved",
                "cross_lineage_excluded",
            )
        }

    rewrite_ledger_and_rebind_manifest(
        out,
        alter_nontraining_split,
        lambda manifest, ledger: manifest.update(counts=ledger["counts"]),
    )

    with pytest.raises(ValueError, match="deterministic split mismatch"):
        sm_fs_freeze.verify_private_bundle(out)


@pytest.mark.parametrize(
    ("field", "value"),
    (("sha256", "0" * 64), ("classification", "unknown")),
)
def test_bundle_verifier_rederives_ledger_privacy_fields_from_live_inputs(
    tmp_path, field, value
):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    run(tree, out, privacy_index=privacy_path)

    rewrite_ledger_and_rebind_manifest(
        out, lambda ledger: ledger["rows"][0].update({field: value})
    )

    with pytest.raises(ValueError, match="privacy/source derivation mismatch"):
        sm_fs_freeze.verify_private_bundle(
            out, privacy_index=privacy_path
        )


def test_bundle_verifier_binds_manifest_input_artifact_to_ledger(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"
    run(tree, out, privacy_index=privacy_path)

    rewrite_ledger_and_rebind_manifest(
        out,
        lambda ledger: ledger["privacy_index"]["artifact"].update(
            sha256="0" * 64
        ),
    )

    with pytest.raises(ValueError, match="manifest/ledger binding mismatch"):
        sm_fs_freeze.verify_private_bundle(out)


def test_bundle_verifier_rejects_artifact_and_privacy_input_tampering(tmp_path):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)

    artifact_out = tmp_path / "artifact-out"
    run(tree, artifact_out, privacy_index=privacy_path)
    with (artifact_out / "lineage_fs_targets.tsv").open("ab") as stream:
        stream.write(b"# tampered\n")
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        sm_fs_freeze.verify_private_bundle(
            artifact_out, privacy_index=privacy_path
        )

    input_out = tmp_path / "input-out"
    run(tree, input_out, privacy_index=privacy_path)
    with privacy_path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="privacy-index binding mismatch"):
        sm_fs_freeze.verify_private_bundle(
            input_out, privacy_index=privacy_path
        )


def test_privacy_index_change_after_bound_read_uses_same_bytes_and_fails_install(
    tmp_path, monkeypatch
):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    original_parse = sm_fs_freeze._parse_privacy_index_bytes

    def change_then_parse(bound_bytes):
        with privacy_path.open("ab") as stream:
            stream.write(b"\n")
        return original_parse(bound_bytes)

    monkeypatch.setattr(
        sm_fs_freeze, "_parse_privacy_index_bytes", change_then_parse
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="privacy-index binding mismatch"):
        run(tree, out, privacy_index=privacy_path)
    assert not out.exists()
    assert not list(tmp_path.glob(".out.staging.*"))


def test_source_change_after_payload_build_fails_before_install(
    tmp_path, monkeypatch
):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    target = tree / "Subjects" / "art" / "music" / "Jazz.smmx"
    original_install = sm_fs_freeze.install_private_outputs

    def change_then_install(out_dir, payloads, *, privacy_index):
        write_map(target, "Changed after payload build")
        return original_install(
            out_dir, payloads, privacy_index=privacy_index
        )

    monkeypatch.setattr(
        sm_fs_freeze, "install_private_outputs", change_then_install
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="map contents changed"):
        run(tree, out, privacy_index=privacy_path)
    assert not out.exists()
    assert not list(tmp_path.glob(".out.staging.*"))


def test_atomic_install_failure_leaves_no_partial_bundle(tmp_path, monkeypatch):
    tree = make_tree(tmp_path)
    privacy_path = tmp_path / "privacy.jsonl"
    build_privacy_index(tree, privacy_path)
    out = tmp_path / "out"

    def fail_rename(source, target):
        raise RuntimeError("synthetic rename failure")

    monkeypatch.setattr(
        sm_fs_freeze, "_rename_directory_noreplace", fail_rename
    )
    with pytest.raises(RuntimeError, match="synthetic rename failure"):
        run(tree, out, privacy_index=privacy_path)
    assert not out.exists()
    assert not list(tmp_path.glob(".out.staging.*"))
