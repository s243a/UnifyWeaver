#!/usr/bin/env python3
"""Tests for the SimpleMind within-map campaign sampler."""

import csv
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from sample_product_kalman_simplemind_campaign import (
    candidate_pools,
    load_primary_maps,
    main,
    parse_primary_map,
)


def add_topic(root, topic_id, parent, text, link=None, cloud=None):
    attrs = {"id": str(topic_id), "parent": str(parent), "text": text}
    topic = ET.SubElement(root, "topic", attrs)
    if link:
        ET.SubElement(topic, "link", {"urllink": link})
    if cloud:
        ET.SubElement(topic, "link", {"cloudmapref": cloud})
    return topic


def write_campaign_map(path, prefix):
    root = ET.Element("mindmap")
    add_topic(root, 1, -1, f"Map {prefix}")
    add_topic(root, 2, 1, f"{prefix} node 2", link=f"https://www.pearltrees.com/s243a/{prefix.lower()}-node-2/id2")
    add_topic(root, 50, 2, "")
    add_topic(root, 3, 50, f"{prefix} node 3")
    add_topic(root, 4, 3, f"{prefix} node 4")
    add_topic(root, 5, 4, f"{prefix} node 5")
    add_topic(root, 6, 5, f"{prefix}-node 6", cloud="../Other.smmx")
    add_topic(root, 60, 6, f"{prefix}_node 6")

    add_topic(root, 100, 1, "see also")
    add_topic(root, 101, 100, f"{prefix} associative")
    add_topic(root, 110, 1, "via link")
    add_topic(root, 111, 110, f"{prefix} via-link child")
    add_topic(root, 200, 1, "super Categories")
    add_topic(root, 201, 200, f"{prefix} broader")
    add_topic(root, 300, 1, "Private material")
    add_topic(root, 301, 300, f"{prefix} hidden")
    add_topic(root, 400, 1, "pg1")
    add_topic(root, 401, 400, f"{prefix} navigation child")
    add_topic(root, 500, 1, "wiki", link="https://en.wikipedia.org/wiki/Example")
    ET.SubElement(root, "relation", {"source": "2", "target": "5"})
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_two_node_map(path, root_title, child_title):
    root = ET.Element("mindmap")
    add_topic(root, 1, -1, root_title)
    add_topic(root, 2, 1, child_title)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def test_primary_parser_keeps_content_paths_and_excludes_secondary_structure():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Map A.xml"
        write_campaign_map(path, "A")
        records, source, excluded = parse_primary_map(path)
        assert excluded == {}
        assert source["status"] == "retained"
        assert source["stats"]["private_topics"] == 2
        assert source["stats"]["structural_containers"] == 4
        assert source["stats"]["navigation_topics"] == 1
        assert source["stats"]["cross_map_link_topics"] == 1
        assert source["stats"]["explicit_relations"] == 1
        assert source["stats"]["edge_rejections"]["see_also_ancestry"] == 2
        assert source["stats"]["edge_rejections"]["super_category_ancestry"] == 1
        assert source["stats"]["edge_rejections"]["navigation_ancestry"] == 1

        identities = {
            endpoint["identity"]
            for record in records
            for endpoint in record["endpoints"]
        }
        assert "a associative" not in identities
        assert "a via link child" not in identities
        assert "a broader" not in identities
        assert "a hidden" not in identities
        assert "a navigation child" not in identities
        assert "a node 3" in identities
        leaf = next(
            endpoint
            for record in records
            for endpoint in record["endpoints"]
            if endpoint["identity"] == "a node 6"
        )
        assert leaf["title_aliases"] == ("A_node 6",)
        assert leaf["pearltrees_slugs"] == ()
        linked = next(
            endpoint
            for record in records
            for endpoint in record["endpoints"]
            if endpoint["identity"] == "a node 2"
        )
        assert linked["pearltrees_slugs"] == ("a-node-2",)
        map_root = next(
            endpoint
            for record in records
            for endpoint in record["endpoints"]
            if endpoint["identity"] == "map a"
        )
        assert map_root["enwiki_aliases"] == ("Example",)


def test_cross_map_direction_conflicts_are_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = root / "Alpha.xml"
        second = root / "Beta.xml"
        write_two_node_map(first, "Alpha", "Beta")
        write_two_node_map(second, "Beta", "Alpha")
        dataset = load_primary_maps([first, second])
        _pools, rejected, conflicts = candidate_pools(dataset, hmax=1, seed=0)
        assert rejected["direction_conflict_pairs"] == 1
        assert rejected["direction_conflict_observations"] == 2
        assert conflicts[0]["kind"] == "direction"


def test_cli_balances_maps_and_writes_stable_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        first = root / "Map A.xml"
        second = root / "Map B.xml"
        write_campaign_map(first, "A")
        write_campaign_map(second, "B")
        pairs = root / "pairs.tsv"
        score = root / "score.tsv"
        manifest = root / "manifest.json"
        args = [
            "--maps", str(first), str(second),
            "--pairs", "10",
            "--hmax", "5",
            "--seed", "9",
            "--pairs-tsv", str(pairs),
            "--score-in", str(score),
            "--manifest", str(manifest),
            "--allow-small-sample",
        ]
        assert main(args) == 0
        with open(pairs, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) == 10
        assert {row["corpus"] for row in rows} == {"simplemind"}
        assert {row["graph_view"] for row in rows} == {"within_map_principal_path"}
        for hop in range(1, 6):
            hop_rows = [row for row in rows if int(row["hop"]) == hop]
            assert len(hop_rows) == 2
            assert {row["branch_title"] for row in hop_rows} == {"Map A", "Map B"}
        assert len({tuple(sorted((row["descendant_id"], row["ancestor_id"]))) for row in rows}) == 10
        assert any(
            "a-node-2" in row["descendant_pearltrees_slugs"]
            or "a-node-2" in row["ancestor_pearltrees_slugs"]
            for row in rows
        )
        assert any(
            "Example" in row["descendant_enwiki_aliases"]
            or "Example" in row["ancestor_enwiki_aliases"]
            for row in rows
        )

        score_rows = [
            line.rstrip("\n").split("\t")
            for line in score.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")
        ]
        assert len(score_rows) == 10
        assert all(row[2] == "subtopic" for row in score_rows)
        assert all(row[5:7] == ["mindmap_node", "mindmap_node"] for row in score_rows)

        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["hop_counts"] == {str(hop): 2 for hop in range(1, 6)}
        assert data["dataset_stats"]["maps_retained"] == 2
        assert data["secondary_edge_policy"].startswith("see-also")
        assert data["title_audit"]["semantic_corrections_applied"] is False
        assert data["identity_alias_counts"]["enwiki_aliases"] == 1

        first_bytes = (pairs.read_bytes(), score.read_bytes(), manifest.read_bytes())
        assert main(args) == 0
        assert (pairs.read_bytes(), score.read_bytes(), manifest.read_bytes()) == first_bytes


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} SimpleMind campaign sampler tests passed")
