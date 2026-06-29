#!/usr/bin/env python3
"""Tests for transitive-closure generation (DESIGN_transitive_relations.md stage 1). Pure python.
Run: `python3 test_transitive_closure.py`."""
from transitive_closure import compose_relation, transitive_pairs, REL_MU


def test_compose_rules():
    assert compose_relation("subcategory", "subcategory") == "subcategory"
    assert compose_relation("subtopic", "subtopic") == "subtopic"
    assert compose_relation("subcategory", "subtopic") == "subtopic"      # mixed downward → looser
    assert compose_relation("element_of", "subcategory") == "element_of"  # C∈A, A⊆B ⇒ C∈B
    assert compose_relation("super_category", "super_category") == "super_category"
    assert compose_relation("bridge", "subcategory") == "subcategory"     # identity passes through
    assert compose_relation("subcategory", "bridge") == "subcategory"
    assert compose_relation("subcategory", "element_of") is None          # downward then membership ⇏
    assert compose_relation("see_also", "subcategory") is None            # lateral never composes


def test_two_hop_product_and_bound():
    # A ⊆ B ⊆ C  (subcategory, 0.90 each)
    pairs = transitive_pairs([("A", "B", "subcategory"), ("B", "C", "subcategory")])
    assert len(pairs) == 1
    p = pairs[0]
    assert (p["src"], p["dst"]) == ("A", "C") and p["hops"] == 2 and p["rel"] == "subcategory"
    assert abs(p["product"] - 0.81) < 1e-9            # 0.9·0.9, the ranking key
    assert abs(p["min_link"] - 0.90) < 1e-9           # min link = the ordinal BOUND


def test_element_of_through_containment():
    # C ∈ A ⊆ B  → C ∈ B
    pairs = transitive_pairs([("C", "A", "element_of"), ("A", "B", "subcategory")])
    assert len(pairs) == 1 and pairs[0]["rel"] == "element_of"
    assert (pairs[0]["src"], pairs[0]["dst"]) == ("C", "B")


def test_three_hop_and_decay_ordering():
    # A⊆B⊆C⊆D : product decays with length; sorted descending so 2-hop (0.81) ranks above 3-hop (0.729)
    e = [("A", "B", "subcategory"), ("B", "C", "subcategory"), ("C", "D", "subcategory")]
    pairs = transitive_pairs(e, max_hops=3)
    prods = [p["product"] for p in pairs]
    assert prods == sorted(prods, reverse=True)        # curriculum order
    ac = next(p for p in pairs if (p["src"], p["dst"]) == ("A", "C"))
    ad = next(p for p in pairs if (p["src"], p["dst"]) == ("A", "D"))
    assert abs(ac["product"] - 0.81) < 1e-9 and abs(ad["product"] - 0.729) < 1e-9
    assert ad["min_link"] == 0.90 and ad["hops"] == 3


def test_dominant_path_keeps_max_product():
    # two paths A→D: A→B→D (0.9·0.9=0.81) and A→C→D via subtopic (0.85·0.85=0.7225); keep the max (0.81)
    e = [("A", "B", "subcategory"), ("B", "D", "subcategory"),
         ("A", "C", "subtopic"), ("C", "D", "subtopic")]
    ad = [p for p in transitive_pairs(e) if (p["src"], p["dst"]) == ("A", "D")]
    assert len(ad) == 1 and abs(ad[0]["product"] - 0.81) < 1e-9      # dominant (max-product) path won


def test_direct_edge_not_emitted_as_transitive():
    # A→C is BOTH reachable via B and a direct edge → excluded (it's direct, not transitive)
    e = [("A", "B", "subcategory"), ("B", "C", "subcategory"), ("A", "C", "subcategory")]
    assert all((p["src"], p["dst"]) != ("A", "C") for p in transitive_pairs(e))


def test_incompatible_chain_pruned():
    # element_of then (reverse) — A⊆B but then B see_also C (lateral, not in REL_MU) → no transitive pair
    e = [("X", "A", "element_of"), ("A", "B", "see_also")]   # see_also not hierarchical → dropped
    assert transitive_pairs(e) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} transitive_closure tests passed")
