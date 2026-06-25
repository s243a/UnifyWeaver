#!/usr/bin/env python3
"""Section categorizer — map a Pearltrees SECTION label to a graph RELATION, as a SEPARATE, re-runnable step
over the harvester's RAW capture. The harvester records section text by position id and makes NO relation
judgement; categorization happens here (and in categorize_sections.py / inline in the consumers), so we can
re-categorise cached harvests WITHOUT re-harvesting, and upgrade the matching method over time.

Recording the METHOD makes the categorisation itself PROVENANCE — it slots into the same judge axis as the
rest of the pipeline: `exact_phrase` is deterministic (graph-judge-like, confidence 1.0); `fuzzy` /
`llm_template` are probabilistic (typically <1.0 confidence) and a loose match could later carry a looser μ.

Shared by parse_pearltrees.py, fuse_corpus.py and categorize_sections.py so the rule lives in ONE place.
"""

# category → the graph relation it implies. `reference` (a wiki/encyclopedia reference LIST) is associative.
CATEGORY_RELATION = {
    "subcategory": "subcategory",
    "element_of": "element_of",
    "super_category": "super_category",
    "see_also": "see_also",
    "reference": "see_also",
}


def section_mode(text):
    """EXACT-PHRASE (case-insensitive substring) categoriser: a section label → category, or None when no
    rule fires (the caller then falls back to the structural default — see_also is NOT assumed)."""
    t = (text or "").lower()
    if "subcategor" in t:
        return "subcategory"                          # narrower category
    if "subtopic" in t:
        return "element_of"                           # element / membership
    if ("super" in t and "categor" in t) or "navigate up" in t:
        return "super_category"                       # broader / parent
    if "see also" in t or "via link" in t or "related" in t:
        return "see_also"                             # associative
    if "wiki" in t or "encyclopedia" in t or "reference" in t:
        return "reference"                            # a reference list (→ see_also)
    return None


def categorize(text, method="exact_phrase"):
    """Section label → (category, method, confidence). `exact_phrase` now; `fuzzy` / `llm_template` are
    future methods with the SAME signature (and typically <1.0 confidence)."""
    if method == "exact_phrase":
        cat = section_mode(text)
        return (cat, "exact_phrase", 1.0 if cat else 0.0)
    raise ValueError(f"unknown categorization method: {method!r} (have: exact_phrase)")


def relation_for(text, method="exact_phrase"):
    """Convenience: section label → (relation, method, confidence). `relation` is None when no rule fires."""
    cat, m, conf = categorize(text, method)
    return (CATEGORY_RELATION.get(cat), m, conf)
