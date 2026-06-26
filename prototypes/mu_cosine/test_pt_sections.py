#!/usr/bin/env python3
"""Tests for pt_sections categorisation: exact + fuzzy (typo) + embedding (semantic, via a STUB encoder so
no model/torch is needed; numpy only). Run: `python3 test_pt_sections.py`."""
import numpy as np

from pt_sections import categorize, embed_mode, fuzzy_mode, section_mode, CATEGORY_RELATION


def _stub_encoder():
    """A deterministic stand-in for e5: map keyword presence to one of 5 orthonormal category axes (else a
    neutral vector). Lets us test the embed_mode wiring (per-category max, threshold + margin gate,
    escalation order) without loading the real model."""
    keymap = [(["member", "element", "instance", "subtopic"], 0),
              (["subcategor", "narrower", "child"], 1),
              (["super", "broader", "parent", "general"], 2),
              (["see also", "related", "cross", "associat"], 3),
              (["reference", "source", "reading", "bibliograph", "link", "extern"], 4)]

    def enc(texts):
        out = []
        for t in texts:
            tl = t.lower(); v = np.zeros(5)
            for keys, ax in keymap:
                for k in keys:
                    if k in tl:
                        v[ax] += 1.0
            if v.sum() == 0:
                v = np.ones(5)                            # neutral ⇒ low cosine (0.45) to every single axis
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    return enc


def test_exact_unchanged():
    assert section_mode("Subtopics") == "element_of"
    assert section_mode("Subcategories") == "subcategory"
    cat, method, conf = categorize("Subtopics", "fuzzy")        # exact is tried first even in fuzzy mode
    assert (cat, method, conf) == ("element_of", "exact_phrase", 1.0)


def test_fuzzy_catches_typos():
    for label, want in [("Subtoipcs", "element_of"), ("More Subtoipcs", "element_of"),
                        ("Subcatagories", "subcategory"), ("Super Catagories", "super_category"),
                        ("See Aslo", "see_also")]:
        cat, method, conf = categorize(label, "fuzzy")
        assert cat == want and method == "fuzzy" and conf == 1.0, (label, cat, method, conf)


def test_fuzzy_rejects_topical_and_junk():
    for label in ["Meta", "Papers", "Nonlinear control", "My Groups", "Friends Pages",
                  "Transient response characteristics"]:
        assert categorize(label, "fuzzy")[0] is None, label
        assert fuzzy_mode(label)[1] < 0.78, label                # safely below the threshold


def test_exact_method_does_not_fuzzy_match():
    # method=exact_phrase must NOT rescue a typo (escalation is opt-in)
    assert categorize("Subtoipcs", "exact_phrase")[0] is None


def test_tag_qualifier_pattern():
    # tag/qualifier across a dash, in EITHER order; categorise WHICHEVER segment is the tag.
    # REAL harvest examples — the qualifier (an alphabetical RANGE) comes FIRST, the tag last, and the range's
    # own hyphen ("A-E", "0-9", "N-Z") must NOT be treated as the delimiter (only whitespace-surrounded dashes).
    for label in ["A-E -- Subtopics", "0-9, A-G -- Subtopics", "IT - Subtopics",
                  "N-Z - Subtopics", "PT1 - Subtopics (Information THeory)", "Subtopics - old"]:
        assert categorize(label, "fuzzy")[0] == "element_of", label
    # tag-first order, and the qualifier must neither dilute a fuzzy match nor override the tag:
    assert categorize("See Also -- Foundational", "fuzzy")[0] == "see_also"
    assert categorize("See Aslo -- Foundational", "fuzzy")[:2] == ("see_also", "fuzzy")   # typo'd tag + qualifier
    # qualifier carries a CONFLICTING keyword ('subcategories') — the tag (first matching segment) wins:
    assert categorize("See Also -- Subcategories of X", "fuzzy")[0] == "see_also"
    # PARENTHETICAL qualifiers, handled the same way (real-harvest examples):
    assert categorize("Subtopics (old)", "fuzzy")[0] == "element_of"
    assert categorize("Subtoipcs (old)", "fuzzy")[:2] == ("element_of", "fuzzy")          # paren can't dilute typo
    assert categorize("Further reading (from wikipedia)", "fuzzy")[0] == "reference"       # keyword INSIDE parens
    assert categorize("Reciprocity theorem (disambiguation)", "fuzzy")[0] is None          # topical node, no match


def test_threshold_is_tunable():
    assert categorize("Subtoipcs", "fuzzy", fuzzy_threshold=0.99)[0] is None   # raise the bar ⇒ no match
    assert categorize("Subtoipcs", "fuzzy", fuzzy_threshold=0.70)[0] == "element_of"


def test_embedding_paraphrase_rescue():
    # semantic matches that share NO edit-distance with a keyword (so fuzzy misses) — caught by embedding.
    enc = _stub_encoder()
    assert categorize("Narrower areas", "embedding", encoder=enc)[:2] == ("subcategory", "embedding")
    assert categorize("Parent concepts", "embedding", encoder=enc)[0] == "super_category"
    assert categorize("External links here", "embedding", encoder=enc)[0] == "reference"
    assert categorize("The members", "embedding", encoder=enc)[0] == "element_of"


def test_embedding_rejects_topical():
    # a topical/content name (no relation keyword) must NOT be forced into a relation (the real-harvest risk)
    enc = _stub_encoder()
    for label in ["Quantum mechanics", "Thermodynamics", "Vector calculus"]:
        assert categorize(label, "embedding", encoder=enc)[0] is None, label


def test_embedding_escalation_order():
    # embedding is the LAST rung — exact and fuzzy still win when they fire
    enc = _stub_encoder()
    assert categorize("Subcategories", "embedding", encoder=enc) == ("subcategory", "exact_phrase", 1.0)
    assert categorize("Subtoipcs", "embedding", encoder=enc)[:2] == ("element_of", "fuzzy")
    # and a missing encoder is an explicit error, not a silent miss
    try:
        categorize("Members", "embedding"); assert False
    except ValueError:
        pass


def test_embedding_margin_gate():
    # a label aligned EQUALLY to two categories is ambiguous (likely topical) ⇒ rejected by the margin gate
    enc = _stub_encoder()
    assert embed_mode("members narrower", enc, threshold=0.7, margin=0.05)[0] is None    # tie ⇒ rejected
    assert embed_mode("members narrower", enc, threshold=0.7, margin=0.0)[0] is not None  # gate off ⇒ a pick


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} pt_sections tests passed (torch-free)")
