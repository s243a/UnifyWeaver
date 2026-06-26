#!/usr/bin/env python3
"""Tests for pt_sections section categorisation: exact + the fuzzy (typo-robust) method. Pure-Python
(torch-free). Run: `python3 test_pt_sections.py`."""
from pt_sections import categorize, fuzzy_mode, section_mode, CATEGORY_RELATION


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
    # categorise the leading TAG (before a `-- qualifier`); the qualifier must neither dilute the fuzzy match
    # nor override the tag.
    assert categorize("See Also -- Foundational", "fuzzy")[0] == "see_also"
    assert categorize("See Aslo -- Foundational", "fuzzy")[:2] == ("see_also", "fuzzy")   # typo'd tag + qualifier
    assert categorize("Subtopics -- Advanced", "fuzzy")[0] == "element_of"
    # the qualifier carries a CONFLICTING keyword ('subcategories') — the tag must win
    assert categorize("See Also -- Subcategories of X", "fuzzy")[0] == "see_also"


def test_threshold_is_tunable():
    assert categorize("Subtoipcs", "fuzzy", fuzzy_threshold=0.99)[0] is None   # raise the bar ⇒ no match
    assert categorize("Subtoipcs", "fuzzy", fuzzy_threshold=0.70)[0] == "element_of"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} pt_sections tests passed (torch-free)")
