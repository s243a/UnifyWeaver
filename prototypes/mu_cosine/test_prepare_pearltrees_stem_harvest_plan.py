#!/usr/bin/env python3
"""Contract tests for the privacy-aware STEM harvest planner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import stat
from xml.sax.saxutils import escape

import pytest

import prepare_pearltrees_diffusion_snapshot as snapshot
import prepare_pearltrees_stem_harvest_plan as planner


PHYSICAL_RELATIONS = {
    "alias": True,
    "collection": True,
    "cross_link": False,
    "path": True,
    "ref": True,
    "shortcut": True,
}


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _rdf(
    nodes: list[tuple[int, str, int | None]], edges: list[tuple[int, int]]
) -> bytes:
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
            ]
        )
        if visibility is not None:
            lines.append(f"    <pt:privacy>{visibility}</pt:privacy>")
        lines.append("  </pt:Tree>")
    for child, parent in edges:
        title, visibility = by_id[child]
        lines.extend(
            [
                "  <pt:RefPearl>",
                f'    <pt:parentTree rdf:resource="https://www.pearltrees.com/synthetic/id{parent}" />',
                f'    <pt:seeAlso rdf:resource="https://www.pearltrees.com/synthetic/id{child}" />',
                f"    <pt:title>{escape(title)}</pt:title>",
            ]
        )
        if visibility is not None:
            lines.append(f"    <pt:privacy>{visibility}</pt:privacy>")
        lines.append("  </pt:RefPearl>")
    lines.append("</rdf:RDF>")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_snapshot(
    tmp_path: Path,
    *,
    nodes: list[tuple[int, str, int | None]],
    edges: list[tuple[int, int]],
    api_cache: Path,
) -> tuple[Path, dict]:
    inputs = tmp_path / "snapshot-inputs"
    local_root = tmp_path / "snapshots"
    inputs.mkdir()
    local_root.mkdir()
    rdf = inputs / "source.rdf"
    rdf.write_bytes(_rdf(nodes, edges))
    source_spec = inputs / "sources.json"
    policy = inputs / "policy.json"
    _write_json(
        source_spec,
        {
            "schema": "pearltrees-diffusion-source-spec-v1",
            "snapshot_label": "stem-harvest-test",
            "sources": [
                {
                    "account": "synthetic",
                    "kind": "rdf",
                    "path": str(rdf),
                    "source_id": "synthetic-rdf",
                },
                {
                    "kind": "api_sqlite",
                    "path": str(api_cache),
                    "source_id": "synthetic-api-cache",
                },
            ],
        },
    )
    _write_json(
        policy,
        {
            "physical_edges": PHYSICAL_RELATIONS,
            "schema": "pearltrees-diffusion-relation-policy-v1",
        },
    )
    run = local_root / "run"
    manifest = snapshot.prepare_snapshot(
        source_spec,
        policy,
        run,
        local_root,
        minimum_anchors=1,
        resource_ceiling_bytes=1_000_000,
    )
    return run, manifest


def _api_tree(tree_id: int, *, visibility: int = 0, title: str = "Public") -> str:
    return json.dumps(
        {"tree": {"id": tree_id, "pearls": [], "title": title, "visibility": visibility}},
        separators=(",", ":"),
    )


def _make_cache(path: Path, rows: list[tuple[int, str]]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE api_responses (tree_id TEXT PRIMARY KEY, title TEXT, "
            "fetched_at TEXT, children_count INTEGER, response_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO api_responses VALUES (?,?,?,?,?)",
            [(str(tree_id), "", "2026-01-01T00:00:00", 0, payload) for tree_id, payload in rows],
        )
        connection.commit()
    finally:
        connection.close()


def _seed_manifest(
    path: Path,
    fingerprint: str,
    roots: list[int],
    exclude_roots: list[int] | None = None,
    *,
    public_cache_not_before: str = "2026-01-01T00:00:00",
) -> None:
    _write_json(
        path,
        {
            "exclude_roots": [f"pt:{value}" for value in (exclude_roots or [])],
            "public_cache_not_before": public_cache_not_before,
            "roots": [f"pt:{value}" for value in roots],
            "schema": planner.SEED_SCHEMA,
            "snapshot_fingerprint": fingerprint,
        },
    )


def _basic_case(tmp_path: Path, cache_rows: list[tuple[int, str]] | None = None):
    cache = tmp_path / "api.db"
    _make_cache(
        cache,
        cache_rows
        if cache_rows is not None
        else [(1, _api_tree(1)), (2, _api_tree(2))],
    )
    run, manifest = _make_snapshot(
        tmp_path,
        nodes=[
            (1, "STEM root", 0),
            (2, "Fetched branch", 0),
            (3, "Public gap", 0),
            (4, "Unknown frontier", None),
            (5, "Private branch", 1),
            (6, "Private descendant", 0),
            (7, "Behind unresolved", 0),
        ],
        edges=[(2, 1), (3, 2), (4, 2), (5, 2), (6, 5), (7, 3)],
        api_cache=cache,
    )
    seeds = tmp_path / "seeds.json"
    _seed_manifest(seeds, manifest["snapshot_fingerprint"], [1])
    return run, manifest, cache, seeds


def _prepare_basic(tmp_path: Path, cache_rows: list[tuple[int, str]] | None = None):
    run, manifest, cache, seeds = _basic_case(tmp_path, cache_rows)
    local = tmp_path / "plans"
    local.mkdir()
    output = local / "plan"
    result = planner.prepare_plan(
        run,
        cache,
        seeds,
        output,
        local,
        max_hops=8,
        batch_limit=128,
    )
    return output, result, manifest


def test_separates_public_and_unknown_first_frontiers(tmp_path: Path) -> None:
    output, manifest, _ = _prepare_basic(tmp_path)
    public = _json(output / "public_harvest_queue.json")
    revalidation = _json(output / "visibility_revalidation_queue.json")
    assert [row["tree_id"] for row in public["maps"]] == ["3"]
    assert [row["tree_id"] for row in revalidation["maps"]] == ["4"]
    assert public["maps"][0]["reason"] == "public_direct_containment_frontier"
    assert revalidation["maps"][0]["cache_status"] == "missing"
    # Node 7 is behind unresolved node 3; private nodes 5/6 never enter any queue.
    emitted = {row["tree_id"] for row in public["maps"] + revalidation["maps"]}
    assert emitted == {"3", "4"}
    assert manifest["policy"]["unknown_nodes_traversed"] is False


def test_next_round_expands_only_after_public_parent_is_fetched(tmp_path: Path) -> None:
    output, _, _ = _prepare_basic(
        tmp_path,
        [(1, _api_tree(1)), (2, _api_tree(2)), (3, _api_tree(3))],
    )
    public = _json(output / "public_harvest_queue.json")
    assert [row["tree_id"] for row in public["maps"]] == ["7"]


def test_cache_parser_quarantines_masked_auth_signature() -> None:
    assert (
        planner._cache_response_status(
            "pt:2", _api_tree(2, visibility=2, title="*private*")
        )
        == "masked_auth"
    )


def test_cache_parser_private_claim_wins_across_info_and_tree() -> None:
    payload = json.dumps(
        {
            "info": {"id": 2, "title": "Private notes", "visibility": 1},
            "tree": {"id": 2, "title": "Public", "visibility": 0},
        },
        separators=(",", ":"),
    )
    assert planner._cache_response_status("pt:2", payload) == "private_or_restricted"


def test_cache_parser_rejects_disagreeing_ids() -> None:
    payload = json.dumps(
        {
            "info": {"id": 2, "title": "Public", "visibility": 0},
            "tree": {"id": 3, "title": "Public", "visibility": 0},
        },
        separators=(",", ":"),
    )
    assert planner._cache_response_status("pt:2", payload) == "malformed"


def test_stale_public_root_stops_traversal_and_enters_revalidation(tmp_path: Path) -> None:
    run, manifest, cache, seeds = _basic_case(tmp_path)
    _seed_manifest(
        seeds,
        manifest["snapshot_fingerprint"],
        [1],
        public_cache_not_before="2026-01-02T00:00:00",
    )
    local = tmp_path / "plans"
    local.mkdir()
    output = local / "plan"
    planner.prepare_plan(run, cache, seeds, output, local, max_hops=8, batch_limit=128)
    assert _json(output / "public_harvest_queue.json")["count"] == 0
    rows = _json(output / "visibility_revalidation_queue.json")["maps"]
    assert [(row["tree_id"], row["cache_status"]) for row in rows] == [
        ("1", "stale_public")
    ]


def test_scope_exclusion_removes_entire_containment_branch(tmp_path: Path) -> None:
    run, manifest, cache, seeds = _basic_case(tmp_path)
    _seed_manifest(seeds, manifest["snapshot_fingerprint"], [1], [2])
    local = tmp_path / "plans"
    local.mkdir()
    output = local / "plan"
    result = planner.prepare_plan(
        run, cache, seeds, output, local, max_hops=8, batch_limit=128
    )
    assert _json(output / "public_harvest_queue.json")["count"] == 0
    assert _json(output / "visibility_revalidation_queue.json")["count"] == 0
    # The verified snapshot has already removed privacy-excluded physical edges,
    # so the explicit topical closure contains only retained branch members.
    assert result["counts"]["scope_excluded"] == 4


def test_shared_frontier_has_lexicographic_bridge_tier(tmp_path: Path) -> None:
    cache = tmp_path / "api.db"
    _make_cache(cache, [(value, _api_tree(value)) for value in (1, 2, 10, 11)])
    run, manifest = _make_snapshot(
        tmp_path,
        nodes=[
            (1, "Root A", 0),
            (2, "A branch", 0),
            (10, "Root B", 0),
            (11, "B branch", 0),
            (20, "Shared gap", 0),
        ],
        edges=[(2, 1), (11, 10), (20, 2), (20, 11)],
        api_cache=cache,
    )
    seeds = tmp_path / "seeds.json"
    _seed_manifest(seeds, manifest["snapshot_fingerprint"], [10, 1])
    local = tmp_path / "plans"
    local.mkdir()
    output = local / "plan"
    planner.prepare_plan(run, cache, seeds, output, local, max_hops=8, batch_limit=128)
    row = _json(output / "public_harvest_queue.json")["maps"][0]
    assert row["reason"] == "public_shared_containment_frontier"
    assert row["frontier_links"] == 2
    assert row["stem_root_count"] == 2


def test_batch_limit_preserves_deterministic_rank(tmp_path: Path) -> None:
    cache = tmp_path / "api.db"
    _make_cache(cache, [(1, _api_tree(1))])
    run, manifest = _make_snapshot(
        tmp_path,
        nodes=[(1, "Root", 0), (2, "Two", 0), (3, "Three", 0)],
        edges=[(3, 1), (2, 1)],
        api_cache=cache,
    )
    seeds = tmp_path / "seeds.json"
    _seed_manifest(seeds, manifest["snapshot_fingerprint"], [1])
    local = tmp_path / "plans"
    local.mkdir()
    output = local / "plan"
    result = planner.prepare_plan(run, cache, seeds, output, local, max_hops=8, batch_limit=1)
    assert [row["tree_id"] for row in _json(output / "public_harvest_queue.json")["maps"]] == ["2"]
    assert result["counts"]["public_candidates_total"] == 2
    assert result["counts"]["public_queue"] == 1


def test_seed_manifest_is_bound_to_exact_snapshot(tmp_path: Path) -> None:
    run, _, cache, seeds = _basic_case(tmp_path)
    value = _json(seeds)
    value["snapshot_fingerprint"] = "0" * 64
    _write_json(seeds, value)
    local = tmp_path / "plans"
    local.mkdir()
    with pytest.raises(planner.HarvestPlanError, match="schema mismatch"):
        planner.prepare_plan(
            run, cache, seeds, local / "plan", local, max_hops=8, batch_limit=128
        )


def test_installed_plan_is_private_and_tamper_fails(tmp_path: Path) -> None:
    output, manifest, _ = _prepare_basic(tmp_path)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert {stat.S_IMODE(path.stat().st_mode) for path in output.iterdir()} == {0o600}
    assert planner._verify_artifacts(output)["plan_fingerprint"] == manifest["plan_fingerprint"]
    queue = output / "public_harvest_queue.json"
    os.chmod(queue, 0o644)
    with pytest.raises(planner.HarvestPlanError, match="envelope"):
        planner._verify_artifacts(output)


def test_resealed_cross_lane_semantic_tamper_fails(tmp_path: Path) -> None:
    output, _, _ = _prepare_basic(tmp_path)
    queue_path = output / "public_harvest_queue.json"
    queue = _json(queue_path)
    queue["maps"][0]["cache_status"] = "masked_auth"
    queue_bytes = planner._canonical_json(queue)
    queue_path.write_bytes(queue_bytes)
    manifest_path = output / "manifest.json"
    manifest = _json(manifest_path)
    record = planner._artifact_record(queue_bytes)
    manifest["artifacts"]["public_harvest_queue.json"] = record
    manifest["fingerprint_core"]["artifacts"]["public_harvest_queue.json"] = record
    manifest["plan_fingerprint"] = planner._sha256(
        planner._canonical_json(manifest["fingerprint_core"])
    )
    manifest_path.write_bytes(planner._canonical_json(manifest))
    with pytest.raises(planner.HarvestPlanError, match="lane semantics"):
        planner._verify_artifacts(output)


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_cache_sidecar_fails_closed(tmp_path: Path, suffix: str) -> None:
    run, _, cache, seeds = _basic_case(tmp_path)
    Path(f"{cache}{suffix}").write_bytes(b"not-a-real-sidecar")
    local = tmp_path / "plans"
    local.mkdir()
    with pytest.raises(planner.HarvestPlanError, match="sidecars"):
        planner.prepare_plan(
            run, cache, seeds, local / "plan", local, max_hops=8, batch_limit=128
        )


def test_cache_wal_mode_fails_closed(tmp_path: Path) -> None:
    cache = tmp_path / "wal.db"
    _make_cache(cache, [(1, _api_tree(1))])
    connection = sqlite3.connect(cache)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{cache}{suffix}")
        if sidecar.exists():
            sidecar.unlink()
    with pytest.raises(planner.HarvestPlanError, match="WAL-mode"):
        planner._load_api_cache(cache, "2026-01-01T00:00:00")


def test_cache_must_be_exact_api_sqlite_snapshot_source(tmp_path: Path) -> None:
    run, _, _, seeds = _basic_case(tmp_path)
    other = tmp_path / "other.db"
    _make_cache(other, [(1, _api_tree(1)), (2, _api_tree(2)), (3, _api_tree(3))])
    local = tmp_path / "plans"
    local.mkdir()
    with pytest.raises(planner.HarvestPlanError, match="exact source"):
        planner.prepare_plan(
            run, other, seeds, local / "plan", local, max_hops=8, batch_limit=128
        )


def test_excluded_ancestor_of_seed_fails_closed(tmp_path: Path) -> None:
    run, manifest, cache, seeds = _basic_case(tmp_path)
    _seed_manifest(seeds, manifest["snapshot_fingerprint"], [2], [1])
    local = tmp_path / "plans"
    local.mkdir()
    with pytest.raises(planner.HarvestPlanError, match="below an explicit scope exclusion"):
        planner.prepare_plan(
            run, cache, seeds, local / "plan", local, max_hops=8, batch_limit=128
        )


def test_count_schema_is_exact_and_bound_to_fingerprint(tmp_path: Path) -> None:
    output, _, _ = _prepare_basic(tmp_path)
    manifest_path = output / "manifest.json"
    manifest = _json(manifest_path)
    manifest["counts"]["pt:123"] = 1
    manifest["fingerprint_core"]["counts"]["pt:123"] = 1
    manifest["plan_fingerprint"] = planner._sha256(
        planner._canonical_json(manifest["fingerprint_core"])
    )
    manifest_path.write_bytes(planner._canonical_json(manifest))
    with pytest.raises(planner.HarvestPlanError, match="counts are malformed"):
        planner._verify_artifacts(output)


def test_resealed_manifest_cannot_claim_a_different_cache_cutoff(tmp_path: Path) -> None:
    run, _, cache, seeds = _basic_case(tmp_path)
    local = tmp_path / "plans"
    local.mkdir()
    bound_output = local / "plan"
    planner.prepare_plan(
        run, cache, seeds, bound_output, local, max_hops=8, batch_limit=128
    )
    manifest_path = bound_output / "manifest.json"
    manifest = _json(manifest_path)
    manifest["fingerprint_core"]["public_cache_not_before"] = "2026-01-02T00:00:00"
    manifest["plan_fingerprint"] = planner._sha256(
        planner._canonical_json(manifest["fingerprint_core"])
    )
    manifest_path.write_bytes(planner._canonical_json(manifest))
    assert planner._verify_artifacts(bound_output)["plan_fingerprint"] == manifest[
        "plan_fingerprint"
    ]
    with pytest.raises(planner.HarvestPlanError, match="input binding"):
        planner.verify_plan(bound_output, run, cache, seeds)


def test_output_must_be_below_explicit_local_root(tmp_path: Path) -> None:
    run, _, cache, seeds = _basic_case(tmp_path)
    local = tmp_path / "plans"
    local.mkdir()
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(planner.HarvestPlanError, match="below"):
        planner.prepare_plan(
            run,
            cache,
            seeds,
            tmp_path / "elsewhere" / "plan",
            local,
            max_hops=8,
            batch_limit=128,
        )
