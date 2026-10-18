#!/usr/bin/env python3
"""Tests for public campaign feature-table preparation."""

from materialize_product_kalman_public_features import (
    PublicFeatureError,
    find_pearltrees_lineage,
    identity_components,
    response_targets,
)


def response(**overrides):
    obj = {
        "element_of": {"mu_fwd": 0.1, "mu_rev": 0.0, "applies": 0.1},
        "subcategory": {"mu_fwd": 0.7, "mu_rev": 0.0, "applies": 0.6},
        "subtopic": {"mu_fwd": 0.8, "mu_rev": 0.0, "applies": 0.7},
        "super_category": {"mu_fwd": 0.0, "mu_rev": 0.8, "applies": 0.1},
        "see_also": {"mu": 0.2, "applies": 0.2},
        "assoc": {"mu": 0.4, "applies": 0.4},
        "none": {"applies": 0.3},
        "unknown": {"mu_fwd": 0.1, "mu_rev": 0.1, "applies": 0.1},
    }
    obj.update(overrides)
    return obj


def pair(pair_id, descendant_id, ancestor_id, descendant_canonical, ancestor_canonical, account="acct"):
    return {
        "pair_id": pair_id,
        "corpus": "pearltrees",
        "account": account,
        "descendant_id": descendant_id,
        "ancestor_id": ancestor_id,
        "descendant_canonical_identity": descendant_canonical,
        "ancestor_canonical_identity": ancestor_canonical,
    }


def test_response_targets_use_frozen_family_order_for_ties():
    d, s, family, ties = response_targets(response())
    assert (d, s) == (0.8, 0.4)
    assert family == "directional"
    assert ties == 1
    obj = response()
    obj["assoc"]["applies"] = 0.7
    d, s, family, ties = response_targets(obj)
    assert family == "directional"
    assert ties == 2


def test_identity_closure_joins_ids_through_canonical_titles():
    rows = [
        pair("p0", "1", "2", "alpha", "shared"),
        pair("p1", "3", "4", "shared", "omega"),
    ]
    identities, groups = identity_components(rows)
    assert identities[("p0", "ancestor")] == identities[("p1", "descendant")]
    assert identities[("p0", "descendant")] != identities[("p1", "ancestor")]
    assert len(groups) == 3


def test_find_pearltrees_lineage_reproduces_frozen_hop_and_source():
    row = {
        "pair_id": "p0", "account": "acct", "source_tree_ids": "tree-b",
        "descendant_id": "c", "ancestor_id": "a", "hop": "2",
    }
    records = [
        {"account": "acct", "tree_id": "tree-a", "path_ids": ("a", "x", "c")},
        {"account": "acct", "tree_id": "tree-b", "path_ids": ("a", "b", "c")},
    ]
    assert find_pearltrees_lineage(row, records)["tree_id"] == "tree-b"
    row["hop"] = "1"
    try:
        find_pearltrees_lineage(row, records)
    except PublicFeatureError as exc:
        assert "no frozen Pearltrees lineage" in str(exc)
    else:
        raise AssertionError("expected mismatched-hop lineage rejection")


def test_response_targets_reject_missing_or_out_of_range_fields():
    obj = response()
    del obj["unknown"]["applies"]
    try:
        response_targets(obj)
    except PublicFeatureError as exc:
        assert "unknown.applies" in str(exc)
    else:
        raise AssertionError("expected missing response field rejection")
    obj = response()
    obj["see_also"]["mu"] = 1.1
    try:
        response_targets(obj)
    except PublicFeatureError as exc:
        assert "out-of-range" in str(exc)
    else:
        raise AssertionError("expected out-of-range response field rejection")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} public feature tests passed")
