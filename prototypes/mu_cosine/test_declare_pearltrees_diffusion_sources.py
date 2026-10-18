#!/usr/bin/env python3
"""Focused tests for the explicit Pearltrees diffusion source planner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

import declare_pearltrees_diffusion_sources as declare
import prepare_pearltrees_diffusion_snapshot as snapshot


def _declared_inputs(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "private-inputs"
    inputs.mkdir()
    paths = {
        "rdf_groups": inputs / "groups.rdf",
        "rdf_primary": inputs / "primary.rdf",
        "api_json_dir": inputs / "trees",
        "api_sqlite": inputs / "api.db",
        "path_jsonl": inputs / "paths.jsonl",
        "legacy": inputs / "assembled_dag.tsv",
    }
    paths["rdf_groups"].write_bytes(
        b"private team RDF contents are deliberately not parsed"
    )
    paths["rdf_primary"].write_bytes(
        b"private primary RDF contents are deliberately not parsed"
    )
    paths["api_json_dir"].mkdir()
    paths["api_sqlite"].write_bytes(b"not opened by the declaration planner")
    paths["path_jsonl"].write_bytes(b"not read by the declaration planner\n")
    paths["legacy"].write_bytes(b"not read by the declaration planner\n")
    return paths


def _full_spec(tmp_path: Path, *, reverse: bool = False) -> tuple[dict, dict[str, Path]]:
    paths = _declared_inputs(tmp_path)
    rdf = (
        ("rdf-groups", "groups", str(paths["rdf_groups"])),
        ("rdf-primary", "s243a", str(paths["rdf_primary"])),
    )
    if reverse:
        rdf = tuple(reversed(rdf))
    spec = declare.build_source_spec(
        snapshot_label="frozen-local-snapshot",
        rdf=rdf,
        api_json_dir=(("api-files", str(paths["api_json_dir"])),),
        api_sqlite=(("api-db", str(paths["api_sqlite"])),),
        path_jsonl=(("paths", str(paths["path_jsonl"])),),
        legacy_dag=str(paths["legacy"]),
    )
    return spec, paths


def test_exact_compiler_shape_is_deterministic_and_private(tmp_path: Path) -> None:
    spec, paths = _full_spec(tmp_path)
    local_root = tmp_path / "local-only"
    local_root.mkdir()
    first = local_root / "declaration-a"
    second = local_root / "declaration-b"

    summary_a = declare.install_source_spec(
        spec, output_dir=first, local_root=local_root
    )
    reverse_spec = declare.build_source_spec(
        snapshot_label="frozen-local-snapshot",
        rdf=(
            ("rdf-primary", "s243a", str(paths["rdf_primary"])),
            ("rdf-groups", "groups", str(paths["rdf_groups"])),
        ),
        path_jsonl=(("paths", str(paths["path_jsonl"])),),
        api_sqlite=(("api-db", str(paths["api_sqlite"])),),
        api_json_dir=(("api-files", str(paths["api_json_dir"])),),
        legacy_dag=str(paths["legacy"]),
    )
    summary_b = declare.install_source_spec(
        reverse_spec, output_dir=second, local_root=local_root
    )

    assert (first / declare.SPEC_FILENAME).read_bytes() == (
        second / declare.SPEC_FILENAME
    ).read_bytes()
    installed = json.loads((first / declare.SPEC_FILENAME).read_text(encoding="utf-8"))
    assert installed == spec == reverse_spec
    assert declare.verify_installed_source_spec(first / declare.SPEC_FILENAME) == spec
    assert set(installed) == {"legacy_check", "schema", "snapshot_label", "sources"}
    assert [row["source_id"] for row in installed["sources"]] == sorted(
        row["source_id"] for row in installed["sources"]
    )
    assert all(Path(row["path"]).is_absolute() for row in installed["sources"])
    assert [set(row) for row in installed["sources"] if row["kind"] == "rdf"] == [
        {"account", "kind", "path", "source_id"},
        {"account", "kind", "path", "source_id"},
    ]
    assert all(
        "account" not in row for row in installed["sources"] if row["kind"] != "rdf"
    )
    assert installed["legacy_check"] == {"dag_path": str(paths["legacy"].resolve())}
    resolved_sources, resolved_legacy = snapshot._resolve_source_paths(
        first / declare.SPEC_FILENAME, installed
    )
    assert len(resolved_sources) == 5
    assert resolved_legacy == paths["legacy"].resolve()
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert stat.S_IMODE((first / declare.SPEC_FILENAME).stat().st_mode) == 0o600
    assert stat.S_IMODE((first / declare.LOCAL_ONLY_MARKER).stat().st_mode) == 0o600
    assert summary_a == summary_b == {
        "legacy_check_declared": True,
        "local_only": True,
        "source_count": 5,
        "source_kind_counts": {
            "api_json_dir": 1,
            "api_sqlite": 1,
            "path_jsonl": 1,
            "rdf": 2,
        },
        "spec_written": True,
    }


def test_cli_prints_only_aggregate_counts(tmp_path: Path, capsys) -> None:
    private = tmp_path / "private-title-never-print.rdf"
    private.write_bytes(b"private")
    local_root = tmp_path / "local"
    local_root.mkdir()

    status = declare.main(
        [
            "--snapshot-label",
            "secret-label-never-print",
            "--local-root",
            str(local_root),
            "--output-dir",
            str(local_root / "declaration"),
            "--local-only",
            "--rdf",
            "secret-source-id-never-print",
            "groups",
            str(private),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    public = json.loads(captured.out)
    assert public == {
        "legacy_check_declared": False,
        "local_only": True,
        "source_count": 1,
        "source_kind_counts": {"rdf": 1},
        "spec_written": True,
    }
    for private_value in (
        str(private),
        "private-title-never-print",
        "secret-label-never-print",
        "secret-source-id-never-print",
    ):
        assert private_value not in captured.out


def test_api_directory_contents_are_not_scanned(tmp_path: Path) -> None:
    api_dir = tmp_path / "explicit-api-dir"
    api_dir.mkdir()
    (api_dir / "malformed-private.json").write_bytes(b"not JSON")
    (api_dir / "nested-source-link").symlink_to(tmp_path / "missing-private-target")

    spec = declare.build_source_spec(
        snapshot_label="no-scan",
        api_json_dir=(("api-files", str(api_dir)),),
    )

    assert spec["sources"] == [
        {
            "kind": "api_json_dir",
            "path": str(api_dir.resolve()),
            "source_id": "api-files",
        }
    ]


def test_source_ids_are_globally_unique(tmp_path: Path) -> None:
    paths = _declared_inputs(tmp_path)

    with pytest.raises(declare.DeclarationError, match="globally unique"):
        declare.build_source_spec(
            snapshot_label="duplicates",
            rdf=(("duplicate", "groups", str(paths["rdf_groups"])),),
            api_sqlite=(("duplicate", str(paths["api_sqlite"])),),
        )


@pytest.mark.parametrize("hardlink", (False, True), ids=("same-path", "hardlink"))
def test_authoritative_source_path_aliases_are_rejected(
    tmp_path: Path, hardlink: bool
) -> None:
    source = tmp_path / "source.rdf"
    source.write_bytes(b"not read")
    alias = source
    if hardlink:
        alias = tmp_path / "source-hardlink.rdf"
        os.link(source, alias)

    with pytest.raises(declare.DeclarationError, match="source paths must not alias"):
        declare.build_source_spec(
            snapshot_label="no-source-aliases",
            rdf=(("rdf-a", "groups", str(source)), ("rdf-b", "s243a", str(alias))),
        )


def test_legacy_input_cannot_alias_authoritative_source(tmp_path: Path) -> None:
    source = tmp_path / "source.rdf"
    source.write_bytes(b"not read")

    with pytest.raises(declare.DeclarationError, match="legacy parity input cannot alias"):
        declare.build_source_spec(
            snapshot_label="no-legacy-alias",
            rdf=(("rdf", "groups", str(source)),),
            legacy_dag=str(source),
        )


def test_installer_rejects_noncanonical_or_extended_specs(tmp_path: Path) -> None:
    source = tmp_path / "source.rdf"
    source.write_bytes(b"not read")
    second_source = tmp_path / "paths.jsonl"
    second_source.write_bytes(b"not read\n")
    local_root = tmp_path / "local"
    local_root.mkdir()
    spec = declare.build_source_spec(
        snapshot_label="canonical",
        rdf=(("z-source", "groups", str(source)),),
        path_jsonl=(("a-source", str(second_source)),),
    )
    spec["sources"] = list(reversed(spec["sources"]))

    with pytest.raises(declare.DeclarationError, match="canonical declaration order"):
        declare.install_source_spec(
            spec, output_dir=local_root / "noncanonical", local_root=local_root
        )

    extended = declare.build_source_spec(
        snapshot_label="canonical",
        rdf=(("source", "groups", str(source)),),
    )
    extended["publishable"] = False
    with pytest.raises(declare.DeclarationError, match="fields mismatch"):
        declare.install_source_spec(
            extended, output_dir=local_root / "extended", local_root=local_root
        )

    assert list(local_root.iterdir()) == []


@pytest.mark.parametrize("account", ("", " ", "grous", "GROUS", "GROUPS"))
def test_noncanonical_team_accounts_fail_closed(tmp_path: Path, account: str) -> None:
    source = tmp_path / "team.rdf"
    source.write_bytes(b"not read")

    with pytest.raises(declare.DeclarationError, match="account|typo|groups"):
        declare.build_source_spec(
            snapshot_label="account-check",
            rdf=(("team-rdf", account, str(source)),),
        )


def test_filename_never_infers_or_changes_explicit_account(tmp_path: Path) -> None:
    misleading = tmp_path / "grous-and-s243a.rdf"
    misleading.write_bytes(b"not read")

    spec = declare.build_source_spec(
        snapshot_label="explicit-account",
        rdf=(("team-rdf", "groups", str(misleading)),),
    )

    assert spec["sources"][0]["account"] == "groups"


@pytest.mark.parametrize("wildcard", ("*.rdf", "tree?.json", "[12].db", "{a,b}.jsonl"))
def test_wildcard_looking_paths_are_rejected(tmp_path: Path, wildcard: str) -> None:
    wildcard_path = tmp_path / wildcard
    wildcard_path.write_bytes(b"exists but remains forbidden")

    with pytest.raises(declare.DeclarationError, match="wildcard-looking"):
        declare.build_source_spec(
            snapshot_label="no-globs",
            path_jsonl=(("paths", str(wildcard_path)),),
        )


@pytest.mark.parametrize("symlink_parent", (False, True), ids=("leaf", "parent"))
def test_declared_input_symlinks_are_rejected(
    tmp_path: Path, symlink_parent: bool
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_file = real_dir / "source.rdf"
    real_file.write_bytes(b"not read")
    if symlink_parent:
        linked_dir = tmp_path / "linked-dir"
        linked_dir.symlink_to(real_dir, target_is_directory=True)
        declared = linked_dir / "source.rdf"
    else:
        declared = tmp_path / "linked.rdf"
        declared.symlink_to(real_file)

    with pytest.raises(declare.DeclarationError, match="symlink"):
        declare.build_source_spec(
            snapshot_label="no-symlinks",
            rdf=(("rdf", "groups", str(declared)),),
        )


@pytest.mark.parametrize(
    ("kind", "make_directory"),
    (("api_json_dir", False), ("rdf", True), ("api_sqlite", True), ("path_jsonl", True)),
)
def test_declared_input_types_are_checked_only_by_type(
    tmp_path: Path, kind: str, make_directory: bool
) -> None:
    source = tmp_path / "wrong-type"
    source.mkdir() if make_directory else source.write_bytes(b"file")
    kwargs = {kind: (("source", str(source)),)}
    if kind == "rdf":
        kwargs = {kind: (("source", "groups", str(source)),)}

    with pytest.raises(declare.DeclarationError, match="directory|regular file"):
        declare.build_source_spec(snapshot_label="type-check", **kwargs)


def test_output_must_be_new_private_external_directory(tmp_path: Path) -> None:
    source = tmp_path / "source.rdf"
    source.write_bytes(b"not read")
    spec = declare.build_source_spec(
        snapshot_label="output-check",
        rdf=(("rdf", "groups", str(source)),),
    )
    local_root = tmp_path / "local"
    local_root.mkdir()
    existing = local_root / "existing"
    existing.mkdir()
    with pytest.raises(declare.DeclarationError, match="already exists"):
        declare.install_source_spec(spec, output_dir=existing, local_root=local_root)

    target = local_root / "target"
    target.mkdir()
    linked = local_root / "linked-output"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(declare.DeclarationError, match="symlink"):
        declare.install_source_spec(spec, output_dir=linked, local_root=local_root)

    forbidden = declare.REPO_ROOT / "prototypes" / "mu_cosine" / "forbidden-output"
    with pytest.raises(declare.DeclarationError, match="Git worktree"):
        declare.install_source_spec(
            spec,
            output_dir=forbidden,
            local_root=declare.REPO_ROOT,
        )
    assert not forbidden.exists()


def test_atomic_install_never_replaces_a_racing_target(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.rdf"
    source.write_bytes(b"not read")
    spec = declare.build_source_spec(
        snapshot_label="race-check",
        rdf=(("rdf", "groups", str(source)),),
    )
    local_root = tmp_path / "local"
    local_root.mkdir()
    output = local_root / "declaration"
    original = declare._rename_directory_noreplace

    def race(source_dir: Path, target_dir: Path) -> None:
        target_dir.mkdir()
        (target_dir / "winner").write_text("concurrent", encoding="utf-8")
        original(source_dir, target_dir)

    monkeypatch.setattr(declare, "_rename_directory_noreplace", race)

    with pytest.raises(declare.DeclarationError, match="appeared"):
        declare.install_source_spec(spec, output_dir=output, local_root=local_root)

    assert (output / "winner").read_text(encoding="utf-8") == "concurrent"
    assert not (output / declare.SPEC_FILENAME).exists()
    assert not any(path.name.startswith(".declaration.") for path in local_root.iterdir())


def test_output_parent_identity_is_rechecked_before_atomic_promotion(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.rdf"
    source.write_bytes(b"not read")
    spec = declare.build_source_spec(
        snapshot_label="parent-race-check",
        rdf=(("rdf", "groups", str(source)),),
    )
    local_root = tmp_path / "local"
    local_root.mkdir()
    output = local_root / "declaration"
    rename_called = False

    def fail_parent_recheck(parent: Path, expected: tuple[int, int]) -> None:
        del parent, expected
        raise declare.DeclarationError(
            "output parent identity changed during installation"
        )

    def record_rename(source_dir: Path, target_dir: Path) -> None:
        del source_dir, target_dir
        nonlocal rename_called
        rename_called = True

    monkeypatch.setattr(declare, "_recheck_output_parent", fail_parent_recheck)
    monkeypatch.setattr(declare, "_rename_directory_noreplace", record_rename)

    with pytest.raises(declare.DeclarationError, match="parent identity changed"):
        declare.install_source_spec(
            spec,
            output_dir=output,
            local_root=local_root,
        )

    assert rename_called is False
    assert not output.exists()
    assert not any(path.name.startswith(".declaration.") for path in local_root.iterdir())


def test_cli_failure_is_generic_and_does_not_disclose_input(
    tmp_path: Path, capsys
) -> None:
    missing = tmp_path / "very-private-missing-source.rdf"
    local_root = tmp_path / "local"
    local_root.mkdir()

    status = declare.main(
        [
            "--snapshot-label",
            "private-label",
            "--local-root",
            str(local_root),
            "--output-dir",
            str(local_root / "declaration"),
            "--local-only",
            "--rdf",
            "private-source-id",
            "groups",
            str(missing),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": "source declaration failed closed"}
    assert str(missing) not in captured.err
    assert "private-source-id" not in captured.err
    assert "private-label" not in captured.err


def test_empty_declaration_has_no_implicit_default() -> None:
    with pytest.raises(declare.DeclarationError, match="explicit source"):
        declare.build_source_spec(snapshot_label="no-defaults")


def _installed_full_spec(tmp_path: Path) -> tuple[dict, Path]:
    spec, _paths = _full_spec(tmp_path)
    local_root = tmp_path / "local-only"
    local_root.mkdir()
    output = local_root / "declaration"
    declare.install_source_spec(spec, output_dir=output, local_root=local_root)
    return spec, output


def test_verifier_rejects_handwritten_rdf_without_explicit_account(
    tmp_path: Path,
) -> None:
    spec, output = _installed_full_spec(tmp_path)
    malformed = json.loads(json.dumps(spec))
    rdf = next(row for row in malformed["sources"] if row["kind"] == "rdf")
    del rdf["account"]
    (output / declare.SPEC_FILENAME).write_bytes(declare._canonical_json(malformed))

    with pytest.raises(declare.DeclarationError, match="source entry fields mismatch"):
        declare.verify_installed_source_spec(output / declare.SPEC_FILENAME)


def test_verifier_rejects_noncanonical_json_key_order(tmp_path: Path) -> None:
    spec, output = _installed_full_spec(tmp_path)
    reversed_top_level = dict(reversed(tuple(spec.items())))
    payload = (
        json.dumps(
            reversed_top_level,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")
    assert payload != declare._canonical_json(spec)
    (output / declare.SPEC_FILENAME).write_bytes(payload)

    with pytest.raises(declare.DeclarationError, match="not canonical JSON"):
        declare.verify_installed_source_spec(output / declare.SPEC_FILENAME)


@pytest.mark.parametrize(
    ("payload", "error"),
    (
        (b'{"schema":"a","schema":"b"}\n', "duplicate JSON key"),
        (b'{"schema":NaN}\n', "non-finite JSON number"),
        (b'{"schema":"\xff"}\n', "strict UTF-8 JSON"),
    ),
    ids=("duplicate-key", "nonfinite", "invalid-utf8"),
)
def test_verifier_uses_strict_json_parser(
    tmp_path: Path, payload: bytes, error: str
) -> None:
    _spec, output = _installed_full_spec(tmp_path)
    (output / declare.SPEC_FILENAME).write_bytes(payload)

    with pytest.raises(declare.DeclarationError, match=error):
        declare.verify_installed_source_spec(output / declare.SPEC_FILENAME)


@pytest.mark.parametrize(
    "mutation",
    ("directory-mode", "spec-mode", "marker-mode", "extra-file", "marker-bytes"),
)
def test_verifier_rejects_unsafe_bundle_envelope(
    tmp_path: Path, mutation: str
) -> None:
    _spec, output = _installed_full_spec(tmp_path)
    if mutation == "directory-mode":
        os.chmod(output, 0o755)
    elif mutation == "spec-mode":
        os.chmod(output / declare.SPEC_FILENAME, 0o644)
    elif mutation == "marker-mode":
        os.chmod(output / declare.LOCAL_ONLY_MARKER, 0o644)
    elif mutation == "extra-file":
        (output / "unexpected").write_bytes(b"not allowed")
    elif mutation == "marker-bytes":
        (output / declare.LOCAL_ONLY_MARKER).write_bytes(b"not the marker\n")
    else:  # pragma: no cover - parameter exhaustiveness
        raise AssertionError(mutation)

    with pytest.raises(declare.DeclarationError):
        declare.verify_installed_source_spec(output / declare.SPEC_FILENAME)


@pytest.mark.parametrize("symlink_kind", ("spec", "marker", "parent"))
def test_verifier_rejects_symlinked_bundle_paths(
    tmp_path: Path, symlink_kind: str
) -> None:
    _spec, output = _installed_full_spec(tmp_path)
    source_spec = output / declare.SPEC_FILENAME
    if symlink_kind == "spec":
        replacement = tmp_path / "replacement-spec.json"
        replacement.write_bytes(source_spec.read_bytes())
        source_spec.unlink()
        source_spec.symlink_to(replacement)
    elif symlink_kind == "marker":
        marker = output / declare.LOCAL_ONLY_MARKER
        replacement = tmp_path / "replacement-marker"
        replacement.write_bytes(marker.read_bytes())
        marker.unlink()
        marker.symlink_to(replacement)
    elif symlink_kind == "parent":
        linked = tmp_path / "linked-declaration"
        linked.symlink_to(output, target_is_directory=True)
        source_spec = linked / declare.SPEC_FILENAME
    else:  # pragma: no cover - parameter exhaustiveness
        raise AssertionError(symlink_kind)

    with pytest.raises(declare.DeclarationError, match="symlink|regular files"):
        declare.verify_installed_source_spec(source_spec)


def test_verifier_requires_exact_spec_filename(tmp_path: Path) -> None:
    _spec, output = _installed_full_spec(tmp_path)

    with pytest.raises(declare.DeclarationError, match="must be named"):
        declare.verify_installed_source_spec(output / declare.LOCAL_ONLY_MARKER)


def test_verifier_rejects_bundle_inside_real_git_worktree(tmp_path: Path) -> None:
    spec, _paths = _full_spec(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git_marker = checkout / ".git"
    git_marker.mkdir()
    (git_marker / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    output = checkout / "declaration"
    output.mkdir(mode=0o700)
    source_spec = output / declare.SPEC_FILENAME
    source_spec.write_bytes(declare._canonical_json(spec))
    marker = output / declare.LOCAL_ONLY_MARKER
    marker.write_bytes(declare.LOCAL_ONLY_MARKER_PAYLOAD)
    os.chmod(source_spec, 0o600)
    os.chmod(marker, 0o600)

    with pytest.raises(declare.DeclarationError, match="Git worktree"):
        declare.verify_installed_source_spec(source_spec)
