#!/usr/bin/env python3
"""Focused contract tests for the Pearltrees diffusion snapshot compiler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from xml.sax.saxutils import escape

import pytest

import prepare_pearltrees_diffusion_snapshot as snapshot


RESOURCE_CEILING = 1_000_000
PHYSICAL_RELATIONS = {
    "alias": True,
    "collection": True,
    "cross_link": False,
    "path": True,
    "ref": True,
    "shortcut": True,
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _rdf_bytes(
    nodes: list[tuple[int, str, int]],
    *,
    refs: tuple[tuple[int, int], ...] = (),
    aliases: tuple[tuple[int, int], ...] = (),
) -> bytes:
    """Build a small RDF export where relation observations remain public."""
    by_id = {node_id: (title, visibility) for node_id, title, visibility in nodes}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:pt="https://www.pearltrees.com/rdf/0.1/#">',
    ]
    for node_id, title, visibility in nodes:
        lines.extend(
            [
                f'  <pt:Tree rdf:about="https://www.pearltrees.com/synthetic/id{node_id}">',
                f"    <pt:treeId>{node_id}</pt:treeId>",
                f"    <pt:title>{escape(title)}</pt:title>",
                f"    <pt:privacy>{visibility}</pt:privacy>",
                "  </pt:Tree>",
            ]
        )
    for tag, edges in (("RefPearl", refs), ("AliasPearl", aliases)):
        for child, parent in edges:
            title, visibility = by_id[child]
            lines.extend(
                [
                    f"  <pt:{tag}>",
                    "    <pt:parentTree "
                    f'rdf:resource="https://www.pearltrees.com/synthetic/id{parent}" />',
                    "    <pt:seeAlso "
                    f'rdf:resource="https://www.pearltrees.com/synthetic/id{child}" />',
                    f"    <pt:title>{escape(title)}</pt:title>",
                    f"    <pt:privacy>{visibility}</pt:privacy>",
                    f"  </pt:{tag}>",
                ]
            )
    lines.append("</rdf:RDF>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_contract_files(
    case_root: Path,
    sources: list[dict[str, str]],
    *,
    snapshot_label: str = "synthetic-contract-test",
    legacy_path: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    inputs = case_root / "inputs"
    local_root = case_root / "local-snapshots"
    inputs.mkdir(parents=True, exist_ok=True)
    local_root.mkdir()
    policy_path = inputs / "policy.json"
    spec_path = inputs / "sources.json"
    _write_json(
        policy_path,
        {
            "schema": "pearltrees-diffusion-relation-policy-v1",
            "physical_edges": PHYSICAL_RELATIONS,
        },
    )
    spec = {
        "schema": "pearltrees-diffusion-source-spec-v1",
        "snapshot_label": snapshot_label,
        "sources": sources,
    }
    if legacy_path is not None:
        spec["legacy_check"] = {"dag_path": legacy_path}
    _write_json(spec_path, spec)
    return spec_path, policy_path, local_root, local_root / "run"


def _prepare(
    spec_path: Path,
    policy_path: Path,
    local_root: Path,
    run_dir: Path,
) -> dict:
    return snapshot.prepare_snapshot(
        spec_path,
        policy_path,
        run_dir,
        local_root,
        minimum_anchors=1,
        resource_ceiling_bytes=RESOURCE_CEILING,
    )


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_public_edge_source(path: Path) -> None:
    path.write_bytes(
        _rdf_bytes(
            [(1, "Public root", 0), (2, "Public child", 0)],
            refs=((2, 1),),
        )
    )


def _write_normalized_sqlite(
    path: Path,
    *,
    content_type: object,
    parent_visibility: int = 0,
    child_visibility: int = 0,
    raw_fields: dict[str, object] | None = None,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE trees (id TEXT, title TEXT, account TEXT, visibility INTEGER)"
        )
        # Deliberately leave content_type without affinity so tests can verify that
        # the compiler itself rejects noncanonical values instead of SQLite silently
        # coercing them to an integer.
        connection.execute(
            "CREATE TABLE pearls (tree_id TEXT, content_type, content_tree_id TEXT, "
            "content_tree_title TEXT, raw_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO trees VALUES (?,?,?,?)",
            [
                ("1", "Root", "synthetic", parent_visibility),
                ("2", "Child", "synthetic", child_visibility),
            ],
        )
        raw_pearl = {
            "contentTree": {"id": 2, "visibility": child_visibility}
        }
        if raw_fields is not None:
            raw_pearl.update(raw_fields)
        connection.execute(
            "INSERT INTO pearls VALUES (?,?,?,?,?)",
            (
                "1",
                content_type,
                "2",
                "Child",
                json.dumps(raw_pearl),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_privacy_union_propagates_over_ref_but_not_alias(tmp_path: Path) -> None:
    case = tmp_path / "privacy"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "public.rdf").write_bytes(
        _rdf_bytes(
            [
                (1, "Shared root", 0),
                (2, "Contained child", 0),
                (3, "Alias target", 0),
            ],
            refs=((2, 1),),
            aliases=((3, 1),),
        )
    )
    (inputs / "private.rdf").write_bytes(_rdf_bytes([(1, "Shared root", 1)]))
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [
            {"kind": "rdf", "source_id": "public-observation", "path": "public.rdf"},
            {"kind": "rdf", "source_id": "private-observation", "path": "private.rdf"},
        ],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)

    nodes = {row["node_id"]: row for row in _jsonl(run_dir / "nodes.jsonl")}
    exclusions = {
        row["node_id"]: row["reason"] for row in _jsonl(run_dir / "exclusions.jsonl")
    }
    assert nodes["pt:1"]["visibility"] == "private"
    assert exclusions == {"pt:1": "direct_private", "pt:2": "private_descendant"}
    assert nodes["pt:3"]["excluded"] is False
    assert manifest["aggregate"]["privacy_certified"] is True


@pytest.mark.parametrize(
    ("rdf_bytes", "message"),
    [
        (b"<rdf:RDF>", "malformed RDF XML"),
        (
            b'<!DOCTYPE rdf:RDF [<!ENTITY probe "synthetic">]>'
            b'<rdf:RDF xmlns:rdf="urn:rdf"></rdf:RDF>',
            "document type or entity declarations are forbidden",
        ),
    ],
    ids=("malformed", "doctype"),
)
def test_invalid_rdf_fails_closed_without_final_output(
    tmp_path: Path, rdf_bytes: bytes, message: str
) -> None:
    case = tmp_path / message.split()[0]
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "bad.rdf").write_bytes(rdf_bytes)
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "bad-rdf", "path": "bad.rdf"}],
    )

    with pytest.raises(snapshot.SnapshotError, match=message):
        _prepare(spec_path, policy_path, local_root, run_dir)

    assert not run_dir.exists()
    assert list(local_root.iterdir()) == []


def test_relocation_and_source_declaration_order_do_not_change_snapshot(
    tmp_path: Path,
) -> None:
    source_payloads = {
        "source-a": _rdf_bytes(
            [(1, "Root", 0), (2, "Middle", 0)], refs=((2, 1),)
        ),
        "source-b": _rdf_bytes(
            [(2, "Middle", 0), (3, "Leaf", 0)], aliases=((3, 2),)
        ),
    }

    prepared = []
    layouts = (
        ("first-location", (("source-a", "a.rdf"), ("source-b", "b.rdf"))),
        ("relocated", (("source-b", "moved-b.rdf"), ("source-a", "moved-a.rdf"))),
    )
    for dirname, declarations in layouts:
        case = tmp_path / dirname
        inputs = case / "inputs"
        inputs.mkdir(parents=True)
        sources = []
        for source_id, filename in declarations:
            (inputs / filename).write_bytes(source_payloads[source_id])
            sources.append({"kind": "rdf", "source_id": source_id, "path": filename})
        spec_path, policy_path, local_root, run_dir = _write_contract_files(case, sources)
        prepared.append((_prepare(spec_path, policy_path, local_root, run_dir), run_dir))

    (first_manifest, first_run), (second_manifest, second_run) = prepared
    assert first_manifest["snapshot_fingerprint"] == second_manifest["snapshot_fingerprint"]
    assert {
        name: (first_run / name).read_bytes() for name in snapshot.ALL_RUN_FILES
    } == {name: (second_run / name).read_bytes() for name in snapshot.ALL_RUN_FILES}


def test_sqlite_wal_sidecar_is_rejected_without_output(tmp_path: Path) -> None:
    case = tmp_path / "sqlite-wal"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    database = inputs / "api.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE trees (
                id INTEGER, title TEXT, account TEXT, visibility INTEGER
            );
            CREATE TABLE pearls (
                tree_id INTEGER, content_type INTEGER, content_tree_id INTEGER
            );
            INSERT INTO trees VALUES (1, 'Root', 'synthetic', 0);
            """
        )
        connection.commit()
    finally:
        connection.close()
    Path(f"{database}-wal").write_bytes(b"synthetic pending WAL")
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_sqlite", "source_id": "sqlite-api", "path": "api.sqlite"}],
    )

    with pytest.raises(snapshot.SnapshotError, match="SQLite WAL"):
        _prepare(spec_path, policy_path, local_root, run_dir)

    assert not run_dir.exists()
    assert list(local_root.iterdir()) == []


def test_legacy_parity_does_not_change_scientific_fingerprint(tmp_path: Path) -> None:
    prepared = []
    for dirname, include_legacy in (("without-legacy", False), ("with-legacy", True)):
        case = tmp_path / dirname
        inputs = case / "inputs"
        inputs.mkdir(parents=True)
        _write_public_edge_source(inputs / "graph.rdf")
        legacy_path = None
        if include_legacy:
            (inputs / "legacy.tsv").write_text("2\t1\n", encoding="utf-8")
            legacy_path = "legacy.tsv"
        spec_path, policy_path, local_root, run_dir = _write_contract_files(
            case,
            [{"kind": "rdf", "source_id": "graph", "path": "graph.rdf"}],
            legacy_path=legacy_path,
        )
        prepared.append((_prepare(spec_path, policy_path, local_root, run_dir), run_dir))

    (without_manifest, without_run), (with_manifest, with_run) = prepared
    for name in snapshot.ARTIFACT_NAMES:
        if name != "legacy_parity.json":
            assert (without_run / name).read_bytes() == (with_run / name).read_bytes()
    assert (without_run / "legacy_parity.json").read_bytes() != (
        with_run / "legacy_parity.json"
    ).read_bytes()
    assert without_manifest["graph_asset_ready"] is with_manifest["graph_asset_ready"] is True
    assert without_manifest["snapshot_fingerprint"] == with_manifest["snapshot_fingerprint"]


def test_verify_rejects_tampered_artifact(tmp_path: Path) -> None:
    case = tmp_path / "tamper"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_public_edge_source(inputs / "graph.rdf")
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "graph", "path": "graph.rdf"}],
    )
    _prepare(spec_path, policy_path, local_root, run_dir)
    edges_path = run_dir / "physical_edges.tsv"
    edges_path.write_bytes(edges_path.read_bytes() + b"pt:9\tpt:10\t1\n")

    with pytest.raises(snapshot.SnapshotError, match="artifact content record mismatch"):
        snapshot.verify_snapshot(run_dir)


def _rehash_manifest_after_artifact_change(run_dir: Path, changed_name: str) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = (run_dir / changed_name).read_bytes()
    changed_record = snapshot._content_record(changed)
    manifest["artifact_records"][changed_name] = changed_record
    if changed_name != "legacy_parity.json":
        core = manifest["fingerprint_core"]
        core["artifact_records"][changed_name] = changed_record
        core["authoritative_artifact_set_sha256"] = hashlib.sha256(
            snapshot._canonical_json(core["artifact_records"])
        ).hexdigest()
        source_bytes = sum(
            record["size_bytes"]
            for source in core["source_records"]
            for record in source["content_records"]
        )
        observed = source_bytes + sum(
            record["size_bytes"] for record in core["artifact_records"].values()
        )
        core["observed_contract_bytes"] = observed
        manifest["aggregate"]["observed_contract_bytes"] = observed
    manifest["snapshot_fingerprint"] = hashlib.sha256(
        snapshot._canonical_json(manifest["fingerprint_core"])
    ).hexdigest()
    manifest_path.write_bytes(snapshot._canonical_json(manifest))


def test_verify_reconstructs_graph_after_attacker_rehashes_tamper(tmp_path: Path) -> None:
    case = tmp_path / "rehash-tamper"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_public_edge_source(inputs / "graph.rdf")
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "graph", "path": "graph.rdf"}],
    )
    _prepare(spec_path, policy_path, local_root, run_dir)

    adjacency_path = run_dir / "adjacency.jsonl"
    rows = _jsonl(adjacency_path)
    rows[0]["neighbors"] = [rows[0]["node_id"]]
    adjacency_path.write_bytes(snapshot._jsonl_bytes(rows))
    _rehash_manifest_after_artifact_change(run_dir, "adjacency.jsonl")

    with pytest.raises(snapshot.SnapshotError, match="adjacency"):
        snapshot.verify_snapshot(run_dir)


def test_api_info_private_claim_cannot_be_hidden_by_public_tree_claim(tmp_path: Path) -> None:
    case = tmp_path / "api-privacy-union"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    _write_json(
        api_dir / "tree.json",
        {
            "tree_id": 1,
            "api_response": {
                "info": {"id": 1, "title": "Root", "visibility": 1},
                "tree": {
                    "id": 1,
                    "title": "Root",
                    "visibility": 0,
                    "pearls": [
                        {
                            "contentType": 2,
                            "treeId": 1,
                            "contentTree": {
                                "id": 2,
                                "title": "Child",
                                "visibility": 0,
                            },
                        }
                    ],
                },
            },
        },
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )
    manifest = _prepare(spec_path, policy_path, local_root, run_dir)

    nodes = {row["node_id"]: row for row in _jsonl(run_dir / "nodes.jsonl")}
    exclusions = {row["node_id"]: row["reason"] for row in _jsonl(run_dir / "exclusions.jsonl")}
    assert nodes["pt:1"]["visibility"] == "private"
    assert exclusions == {"pt:1": "direct_private", "pt:2": "private_descendant"}
    assert manifest["graph_asset_ready"] is False


def test_missing_api_visibility_claim_does_not_override_known_public_claim(tmp_path: Path) -> None:
    case = tmp_path / "api-known-public"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    _write_json(
        api_dir / "tree.json",
        {
            "tree_id": 1,
            "api_response": {
                "info": {"id": 1, "title": "Root"},
                "tree": {
                    "id": 1,
                    "title": "Root",
                    "visibility": 0,
                    "pearls": [
                        {
                            "contentType": 2,
                            "treeId": 1,
                            "contentTree": {
                                "id": 2,
                                "title": "Child",
                                "visibility": 0,
                            },
                        }
                    ],
                },
            },
        },
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )
    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    nodes = {row["node_id"]: row for row in _jsonl(run_dir / "nodes.jsonl")}

    assert nodes["pt:1"]["visibility"] == "public"
    assert manifest["aggregate"]["privacy_certified"] is True
    assert manifest["graph_asset_ready"] is True


def test_unknown_visibility_nodes_are_retained_but_anchor_ineligible(tmp_path: Path) -> None:
    case = tmp_path / "unknown-path"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    path_file = inputs / "paths.jsonl"
    path_file.write_text(
        json.dumps(
            {
                "account": "synthetic",
                "path_ids": [1, 2],
                "tree_id": 2,
                "title": "Unknown child",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "path_jsonl", "source_id": "paths", "path": "paths.jsonl"}],
    )
    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    eligibility = _jsonl(run_dir / "anchor_eligibility.jsonl")

    assert {row["reason"] for row in eligibility} == {"unknown_visibility"}
    assert manifest["aggregate"]["retained_node_count"] == 2
    assert manifest["aggregate"]["privacy_certified"] is False
    assert manifest["graph_asset_ready"] is False


def test_resource_ceiling_fails_before_installation(tmp_path: Path) -> None:
    case = tmp_path / "resource-ceiling"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_public_edge_source(inputs / "graph.rdf")
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "graph", "path": "graph.rdf"}],
    )

    with pytest.raises(snapshot.SnapshotError, match="byte ceiling"):
        snapshot.prepare_snapshot(
            spec_path,
            policy_path,
            run_dir,
            local_root,
            minimum_anchors=1,
            resource_ceiling_bytes=1,
        )
    assert not run_dir.exists()
    assert list(local_root.iterdir()) == []


def test_normalized_sqlite_schema_compiles_typed_collection_graph(tmp_path: Path) -> None:
    case = tmp_path / "normalized-sqlite"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    database = inputs / "api.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE trees (id TEXT, title TEXT, account TEXT, visibility INTEGER)"
    )
    connection.execute(
        "CREATE TABLE pearls (tree_id TEXT, content_type INTEGER, content_tree_id TEXT, "
        "content_tree_title TEXT, raw_json TEXT, left_index INTEGER, id TEXT)"
    )
    connection.executemany(
        "INSERT INTO trees VALUES (?,?,?,?)",
        [("1", "Root", "synthetic", 0), ("2", "Child", "synthetic", 0)],
    )
    connection.execute(
        "INSERT INTO pearls VALUES (?,?,?,?,?,?,?)",
        (
            "1",
            2,
            "2",
            "Child",
            json.dumps({"contentTree": {"id": 2, "visibility": 0}}),
            1,
            "10",
        ),
    )
    connection.commit()
    connection.close()
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_sqlite", "source_id": "normalized", "path": "api.sqlite"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    assert manifest["graph_asset_ready"] is True
    assert (run_dir / "physical_edges.tsv").read_text(encoding="utf-8").splitlines()[1:] == [
        "pt:1\tpt:2\t1"
    ]


def test_api_response_sqlite_accepts_utf8_blob_without_string_coercion(tmp_path: Path) -> None:
    case = tmp_path / "response-sqlite"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    database = inputs / "responses.sqlite"
    payload = {
        "info": {"id": 1, "title": "Root", "visibility": 0},
        "tree": {
            "id": 1,
            "title": "Root",
            "visibility": 0,
            "pearls": [
                {
                    "contentType": 2,
                    "treeId": 1,
                    "contentTree": {"id": 2, "title": "Child", "visibility": 0},
                }
            ],
        },
    }
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE api_responses (tree_id TEXT, response_json BLOB)")
    connection.execute(
        "INSERT INTO api_responses VALUES (?,?)",
        ("1", sqlite3.Binary(json.dumps(payload).encode("utf-8"))),
    )
    connection.commit()
    connection.close()
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_sqlite", "source_id": "responses", "path": "responses.sqlite"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    assert manifest["graph_asset_ready"] is True
    assert manifest["aggregate"]["physical_edge_count"] == 1


def test_rdf_account_root_parent_is_root_evidence_not_an_edge(tmp_path: Path) -> None:
    case = tmp_path / "rdf-account-root"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "root.rdf").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:pt="https://www.pearltrees.com/rdf/0.1/#">
  <pt:Tree rdf:about="https://www.pearltrees.com/synthetic/id1">
    <pt:treeId>1</pt:treeId><pt:title>Root</pt:title><pt:privacy>0</pt:privacy>
  </pt:Tree>
  <pt:RefPearl>
    <pt:parentTree rdf:resource="https://www.pearltrees.com/synthetic" />
    <pt:seeAlso rdf:resource="https://www.pearltrees.com/synthetic/id1" />
    <pt:title>Root</pt:title><pt:privacy>0</pt:privacy>
  </pt:RefPearl>
</rdf:RDF>
""",
        encoding="utf-8",
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [
            {
                "kind": "rdf",
                "source_id": "rdf-root",
                "path": "root.rdf",
                "account": "synthetic",
            }
        ],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    node = _jsonl(run_dir / "nodes.jsonl")[0]

    assert node["node_id"] == "pt:1"
    assert node["accounts"] == ["synthetic"]
    assert len(node["root_evidence"]) == 1
    assert _jsonl(run_dir / "edge_evidence.jsonl") == []
    assert manifest["aggregate"]["physical_edge_count"] == 0
    assert manifest["aggregate"]["privacy_certified"] is True


def test_rdf_sioc_user_account_and_team_space_root_are_supported(
    tmp_path: Path,
) -> None:
    case = tmp_path / "rdf-team-account-root"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "root.rdf").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:sioc="http://rdfs.org/sioc/ns#"
         xmlns:pt="http://www.pearltrees.com/rdf/0.1/#">
  <sioc:UserAccount rdf:about="https://www.pearltrees.com/t/synthetic#sioc">
    <pt:rootTree rdf:resource="https://www.pearltrees.com/t/synthetic/id1" />
  </sioc:UserAccount>
  <pt:Tree rdf:about="https://www.pearltrees.com/t/synthetic/id1">
    <pt:treeId>1</pt:treeId><pt:title>Root</pt:title><pt:privacy>0</pt:privacy>
  </pt:Tree>
  <pt:RefPearl>
    <pt:parentTree rdf:resource="https://www.pearltrees.com/t/synthetic/" />
    <rdfs:seeAlso rdf:resource="https://www.pearltrees.com/t/synthetic/id1" />
    <pt:title>Root</pt:title><pt:privacy>0</pt:privacy>
  </pt:RefPearl>
</rdf:RDF>
""",
        encoding="utf-8",
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [
            {
                "kind": "rdf",
                "source_id": "rdf-team-root",
                "path": "root.rdf",
                "account": "groups",
            }
        ],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    node = _jsonl(run_dir / "nodes.jsonl")[0]

    assert node["node_id"] == "pt:1"
    assert node["accounts"] == ["groups"]
    assert len(node["root_evidence"]) == 2
    assert _jsonl(run_dir / "edge_evidence.jsonl") == []
    assert manifest["aggregate"]["privacy_certified"] is True


@pytest.mark.parametrize(
    ("case_name", "about_uri", "root_uri", "message"),
    [
        (
            "missing-subject",
            None,
            "https://www.pearltrees.com/synthetic/id1",
            "missing rdf:about",
        ),
        (
            "non-sioc-fragment",
            "https://www.pearltrees.com/synthetic#other",
            "https://www.pearltrees.com/synthetic/id1",
            "canonical account-root URI",
        ),
        (
            "declared-account-mismatch",
            "https://www.pearltrees.com/other",
            "https://www.pearltrees.com/other/id1",
            "subject disagrees with declared account",
        ),
        (
            "root-subject-mismatch",
            "https://www.pearltrees.com/synthetic",
            "https://www.pearltrees.com/other/id1",
            "rootTree disagrees with UserAccount subject",
        ),
    ],
)
def test_rdf_user_account_root_evidence_is_bound_to_subject_and_source(
    tmp_path: Path,
    case_name: str,
    about_uri: str | None,
    root_uri: str,
    message: str,
) -> None:
    case = tmp_path / case_name
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    about_attribute = "" if about_uri is None else f' rdf:about="{about_uri}"'
    (inputs / "bad.rdf").write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:sioc="http://rdfs.org/sioc/ns#"
         xmlns:pt="http://www.pearltrees.com/rdf/0.1/#">
  <sioc:UserAccount{about_attribute}>
    <pt:rootTree rdf:resource="{root_uri}" />
  </sioc:UserAccount>
</rdf:RDF>
""",
        encoding="utf-8",
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [
            {
                "kind": "rdf",
                "source_id": "rdf-user-account",
                "path": "bad.rdf",
                "account": "synthetic",
            }
        ],
    )

    with pytest.raises(snapshot.SnapshotError, match=message):
        _prepare(spec_path, policy_path, local_root, run_dir)
    assert not run_dir.exists()


def test_path_jsonl_account_root_parent_is_committed_as_root_evidence(
    tmp_path: Path,
) -> None:
    case = tmp_path / "path-account-root"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "paths.jsonl").write_text(
        json.dumps(
            {
                "account": "synthetic",
                "parent_tree_id": "account:synthetic",
                "path_ids": ["account:synthetic", 1],
                "title": "Root",
                "tree_id": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "path_jsonl", "source_id": "paths", "path": "paths.jsonl"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    node = _jsonl(run_dir / "nodes.jsonl")[0]

    assert node["accounts"] == ["synthetic"]
    assert len(node["root_evidence"]) == 2
    assert _jsonl(run_dir / "edge_evidence.jsonl") == []
    assert manifest["aggregate"]["retained_node_count"] == 1
    assert manifest["graph_asset_ready"] is False


def test_api_content_type_6_is_noncontainment_cross_link(tmp_path: Path) -> None:
    case = tmp_path / "api-cross-link"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    _write_json(
        api_dir / "tree.json",
        {
            "tree_id": 1,
            "info": {"id": 1, "parentTree": {}, "title": "Private root", "visibility": 1},
            "tree": {
                "id": 1,
                "title": "Private root",
                "visibility": 1,
                "pearls": [
                    {
                        "contentType": 6,
                        "treeId": 1,
                        "contentTree": {"id": 2, "title": "Public peer", "visibility": 0},
                    }
                ],
            },
        },
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    evidence = _jsonl(run_dir / "edge_evidence.jsonl")
    exclusions = {row["node_id"]: row["reason"] for row in _jsonl(run_dir / "exclusions.jsonl")}

    assert exclusions == {"pt:1": "direct_private"}
    assert evidence[0]["relation"] == "cross_link"
    assert evidence[0]["physical_policy_included"] is False
    assert manifest["aggregate"]["physical_edge_count"] == 0
    assert {row["node_id"] for row in _jsonl(run_dir / "nodes.jsonl") if not row["excluded"]} == {"pt:2"}


def test_normalized_sqlite_content_type_6_is_noncontainment_cross_link(
    tmp_path: Path,
) -> None:
    case = tmp_path / "sqlite-cross-link"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_normalized_sqlite(
        inputs / "api.sqlite",
        content_type=6,
        parent_visibility=1,
        child_visibility=0,
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_sqlite", "source_id": "api", "path": "api.sqlite"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    evidence = _jsonl(run_dir / "edge_evidence.jsonl")
    exclusions = {row["node_id"]: row["reason"] for row in _jsonl(run_dir / "exclusions.jsonl")}

    assert exclusions == {"pt:1": "direct_private"}
    assert evidence[0]["relation"] == "cross_link"
    assert evidence[0]["physical_policy_included"] is False
    assert manifest["aggregate"]["physical_edge_count"] == 0


@pytest.mark.parametrize("wrapper", [None, "api_response", "response"], ids=("direct", "api-response", "response"))
@pytest.mark.parametrize("empty_parent", [None, {}], ids=("null-parent", "empty-object-parent"))
def test_empty_api_parent_tree_marks_root_across_wrappers(
    tmp_path: Path, wrapper: str | None, empty_parent: object
) -> None:
    case = tmp_path / f"empty-parent-{wrapper or 'direct'}-{empty_parent is None}"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    response = {
        "info": {
            "id": 1,
            "parentTree": empty_parent,
            "title": "Root",
            "visibility": 0,
        },
        "tree": {"id": 1, "title": "Root", "visibility": 0, "pearls": []},
    }
    payload = {"tree_id": 1, **response} if wrapper is None else {"tree_id": 1, wrapper: response}
    _write_json(api_dir / "tree.json", payload)
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)
    node = _jsonl(run_dir / "nodes.jsonl")[0]

    assert len(node["root_evidence"]) == 1
    assert _jsonl(run_dir / "edge_evidence.jsonl") == []
    assert manifest["aggregate"]["privacy_certified"] is True


@pytest.mark.parametrize(
    "conflict",
    ["content-type", "private-content-tree", "parent-id"],
)
def test_conflicting_api_camel_and_snake_aliases_fail_closed(
    tmp_path: Path, conflict: str
) -> None:
    case = tmp_path / f"alias-conflict-{conflict}"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    pearl = {
        "contentType": 2,
        "treeId": 1,
        "contentTree": {"id": 2, "title": "Child", "visibility": 0},
    }
    if conflict == "content-type":
        pearl["content_type"] = 5
    elif conflict == "private-content-tree":
        # The snake-case claim must not be able to hide a conflicting private claim.
        pearl["content_tree"] = {"id": 2, "title": "Child", "visibility": 1}
    else:
        pearl["tree_id"] = 2
    _write_json(
        api_dir / "tree.json",
        {
            "tree_id": 1,
            "info": {"id": 1, "title": "Root", "visibility": 0},
            "tree": {"id": 1, "title": "Root", "visibility": 0, "pearls": [pearl]},
        },
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )

    with pytest.raises(snapshot.SnapshotError, match="aliases disagree"):
        _prepare(spec_path, policy_path, local_root, run_dir)
    assert not run_dir.exists()


@pytest.mark.parametrize("conflicting_level", ["response", "payload"])
def test_simultaneous_api_pearl_containers_must_agree(
    tmp_path: Path, conflicting_level: str
) -> None:
    case = tmp_path / f"api-pearl-container-{conflicting_level}"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    response = {
        "info": {"id": 1, "parentTree": None, "visibility": 0},
        "tree": {"id": 1, "visibility": 0, "pearls": []},
    }
    payload = {"tree_id": 1, "api_response": response}
    conflicting_pearls = [
        {
            "contentType": 2,
            "treeId": 1,
            "contentTree": {"id": 2, "visibility": 0},
        }
    ]
    if conflicting_level == "response":
        response["pearls"] = conflicting_pearls
    else:
        payload["pearls"] = conflicting_pearls
    _write_json(api_dir / "tree.json", payload)
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )

    with pytest.raises(snapshot.SnapshotError, match="API pearl containers disagree"):
        _prepare(spec_path, policy_path, local_root, run_dir)
    assert not run_dir.exists()


def test_identical_simultaneous_api_pearl_containers_are_accepted(
    tmp_path: Path,
) -> None:
    case = tmp_path / "api-identical-pearl-containers"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    pearls = [
        {
            "contentType": 2,
            "treeId": 1,
            "contentTree": {"id": 2, "title": "Child", "visibility": 0},
        }
    ]
    response = {
        "info": {"id": 1, "parentTree": None, "visibility": 0},
        "tree": {"id": 1, "visibility": 0, "pearls": pearls},
        "pearls": json.loads(json.dumps(pearls)),
    }
    payload = {
        "tree_id": 1,
        "api_response": response,
        "pearls": json.loads(json.dumps(pearls)),
    }
    _write_json(api_dir / "tree.json", payload)
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)

    assert manifest["aggregate"]["physical_edge_count"] == 1
    assert _jsonl(run_dir / "edge_evidence.jsonl")[0]["relation"] == "collection"


@pytest.mark.parametrize("content_type", [2.9, "02"], ids=("fractional", "leading-zero"))
def test_normalized_sqlite_rejects_noncanonical_content_type(
    tmp_path: Path, content_type: object
) -> None:
    case = tmp_path / f"sqlite-content-type-{str(content_type).replace('.', '-') }"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_normalized_sqlite(inputs / "api.sqlite", content_type=content_type)
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_sqlite", "source_id": "api", "path": "api.sqlite"}],
    )

    with pytest.raises(snapshot.SnapshotError, match="canonical integer"):
        _prepare(spec_path, policy_path, local_root, run_dir)
    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("raw_fields", "message"),
    [
        ({"contentType": 5}, "SQLite raw content type disagrees with columns"),
        ({"content_type": 5}, "SQLite raw content type disagrees with columns"),
        ({"treeId": 9}, "SQLite raw parent ID disagrees with columns"),
        ({"tree_id": 9}, "SQLite raw parent ID disagrees with columns"),
    ],
    ids=("camel-type", "snake-type", "camel-parent", "snake-parent"),
)
def test_normalized_sqlite_raw_relation_fields_must_match_columns(
    tmp_path: Path, raw_fields: dict[str, object], message: str
) -> None:
    case = tmp_path / f"sqlite-raw-mismatch-{next(iter(raw_fields))}"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_normalized_sqlite(
        inputs / "api.sqlite",
        content_type=2,
        raw_fields=raw_fields,
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_sqlite", "source_id": "api", "path": "api.sqlite"}],
    )

    with pytest.raises(snapshot.SnapshotError, match=message):
        _prepare(spec_path, policy_path, local_root, run_dir)
    assert not run_dir.exists()


def test_normalized_sqlite_matching_raw_relation_aliases_are_accepted(
    tmp_path: Path,
) -> None:
    case = tmp_path / "sqlite-raw-relation-agreement"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_normalized_sqlite(
        inputs / "api.sqlite",
        content_type=2,
        raw_fields={
            "contentType": 2,
            "content_type": 2,
            "treeId": 1,
            "tree_id": 1,
        },
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_sqlite", "source_id": "api", "path": "api.sqlite"}],
    )

    manifest = _prepare(spec_path, policy_path, local_root, run_dir)

    assert manifest["aggregate"]["physical_edge_count"] == 1
    assert _jsonl(run_dir / "edge_evidence.jsonl")[0]["relation"] == "collection"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            b'xmlns:foreign="urn:foreign"><foreign:Tree '
            b'rdf:about="https://www.pearltrees.com/synthetic/id1" /></rdf:RDF>',
            "entity namespace is not allowlisted",
        ),
        (
            b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            b'xmlns:pt="https://www.pearltrees.com/rdf/0.1/#"><pt:Tree '
            b'rdf:about="https://example.com/synthetic/id1"><pt:treeId>1</pt:treeId>'
            b'</pt:Tree></rdf:RDF>',
            "Pearltrees URI boundary",
        ),
    ],
    ids=("foreign-namespace", "foreign-host"),
)
def test_rdf_namespace_and_uri_boundary_are_fail_closed(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    case = tmp_path / message.split()[0]
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "bad.rdf").write_bytes(payload)
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "rdf", "path": "bad.rdf"}],
    )

    with pytest.raises(snapshot.SnapshotError, match=message):
        _prepare(spec_path, policy_path, local_root, run_dir)
    assert not run_dir.exists()


@pytest.mark.parametrize("policy_kind", ["relation", "privacy"])
def test_verify_rejects_rehashed_embedded_policy_tamper(
    tmp_path: Path, policy_kind: str
) -> None:
    case = tmp_path / f"policy-tamper-{policy_kind}"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_public_edge_source(inputs / "graph.rdf")
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "graph", "path": "graph.rdf"}],
    )
    _prepare(spec_path, policy_path, local_root, run_dir)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if policy_kind == "relation":
        manifest["relation_policy"]["physical_edges"]["ref"] = False
        manifest["fingerprint_core"]["relation_policy"]["physical_edges"]["ref"] = False
        expected = "edge evidence disagrees with relation policy"
    else:
        altered = ["collection", "ref"]
        manifest["privacy_policy"]["propagation_relations"] = altered
        manifest["fingerprint_core"]["privacy_policy"]["propagation_relations"] = altered
        expected = "privacy policy mismatch"
    manifest["snapshot_fingerprint"] = hashlib.sha256(
        snapshot._canonical_json(manifest["fingerprint_core"])
    ).hexdigest()
    manifest_path.write_bytes(snapshot._canonical_json(manifest))

    with pytest.raises(snapshot.SnapshotError, match=expected):
        snapshot.verify_snapshot(run_dir)


def test_atomic_no_replace_install_preserves_concurrent_target(tmp_path: Path) -> None:
    source = tmp_path / "staged"
    target = tmp_path / "run"
    source.mkdir()
    target.mkdir()
    (source / "candidate").write_text("candidate", encoding="utf-8")
    (target / "sentinel").write_text("concurrent", encoding="utf-8")

    with pytest.raises(snapshot.SnapshotError, match="appeared during atomic installation"):
        snapshot._rename_directory_noreplace(source, target)

    assert (target / "sentinel").read_text(encoding="utf-8") == "concurrent"
    assert not (target / "candidate").exists()
    assert (source / "candidate").read_text(encoding="utf-8") == "candidate"


def test_verify_rejects_unbound_repository_commit(tmp_path: Path) -> None:
    case = tmp_path / "commit-binding"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    _write_public_edge_source(inputs / "graph.rdf")
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "graph", "path": "graph.rdf"}],
    )
    _prepare(spec_path, policy_path, local_root, run_dir)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["repository_commit"] = "0" * 40
    manifest_path.write_bytes(snapshot._canonical_json(manifest))

    with pytest.raises(snapshot.SnapshotError, match="repository commit provenance is malformed or unbound"):
        snapshot.verify_snapshot(run_dir)


def test_visibility_evidence_ledger_reconstructs_union_and_rejects_rehashed_tamper(
    tmp_path: Path,
) -> None:
    case = tmp_path / "visibility-ledger"
    inputs = case / "inputs"
    api_dir = inputs / "api"
    api_dir.mkdir(parents=True)
    _write_json(
        api_dir / "tree.json",
        {
            "tree_id": 1,
            "api_response": {
                "info": {"id": 1, "parentTree": {}, "title": "Root", "visibility": 1},
                "tree": {"id": 1, "title": "Root", "visibility": 0, "pearls": []},
            },
        },
    )
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "api_json_dir", "source_id": "api", "path": "api"}],
    )
    _prepare(spec_path, policy_path, local_root, run_dir)

    evidence_path = run_dir / "visibility_evidence.jsonl"
    rows = _jsonl(evidence_path)
    node_rows = [row for row in rows if row["node_id"] == "pt:1"]
    assert {row["visibility"] for row in node_rows} == {"private", "public"}
    assert len({row["record_key"] for row in node_rows}) == 2
    assert _jsonl(run_dir / "nodes.jsonl")[0]["visibility"] == "private"

    for row in rows:
        if row["node_id"] == "pt:1" and row["visibility"] == "private":
            row["visibility"] = "public"
            break
    evidence_path.write_bytes(snapshot._jsonl_bytes(rows))
    _rehash_manifest_after_artifact_change(run_dir, "visibility_evidence.jsonl")

    with pytest.raises(snapshot.SnapshotError, match="node visibility disagrees with source evidence"):
        snapshot.verify_snapshot(run_dir)


def test_rdf_subject_uri_rejects_multiple_id_path_segments(
    tmp_path: Path,
) -> None:
    case = tmp_path / "rdf-multiple-id-segments"
    inputs = case / "inputs"
    inputs.mkdir(parents=True)
    payload = (
        b"<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\" "
        b"xmlns:pt=\"https://www.pearltrees.com/rdf/0.1/#\"><pt:Tree "
        b"rdf:about=\"https://www.pearltrees.com/synthetic/id1/id2\">"
        b"<pt:treeId>1</pt:treeId><pt:privacy>0</pt:privacy>"
        b"</pt:Tree></rdf:RDF>"
    )
    (inputs / "bad.rdf").write_bytes(payload)
    spec_path, policy_path, local_root, run_dir = _write_contract_files(
        case,
        [{"kind": "rdf", "source_id": "rdf", "path": "bad.rdf"}],
    )

    with pytest.raises(
        snapshot.SnapshotError,
        match="exactly one numeric Pearltrees ID",
    ):
        _prepare(spec_path, policy_path, local_root, run_dir)
    assert not run_dir.exists()
