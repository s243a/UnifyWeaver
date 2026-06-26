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
    if ((("super" in t or "broad" in t or "parent" in t) and ("categor" in t or "topic" in t))
            or "navigate up" in t):
        return "super_category"                       # broader / parent ("Broad Categories", "Super Topics")
    if "see also" in t or "via link" in t or "related" in t:
        return "see_also"                             # associative
    if "wiki" in t or "encyclopedia" in t or "reference" in t:
        return "reference"                            # a reference list (→ see_also)
    return None


# FUZZY (typo-robust, lexical): canonical section phrase → category, matched by edit-distance so misspelled
# headers (`Subtoipcs`, `More Subtoipcs`, `Subcatagories`) are still categorised instead of falling through to
# the structural default. This GROWS the labelled set + shrinks the inferred noise (REPORT_infer_blend.md).
# Catches TYPOS; an EMBEDDING method (synonyms/paraphrases) and an LLM method (the hard residual) are the next
# escalation layers — cheap → expensive, each catching what the previous misses.
import difflib
import re as _re

FUZZY_KEYS = [
    ("subcategories", "subcategory"), ("subtopics", "element_of"), ("more subtopics", "element_of"),
    ("super categories", "super_category"), ("navigate up", "super_category"),
    ("see also", "see_also"), ("related", "see_also"),
    ("references", "reference"), ("wiki encyclopedia references", "reference"),
]


def _norm(s):
    return " ".join(_re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


# parent-signal guard: a "Broad Categories" / "Super Topics" header is a BROADER (parent) grouping, but a bare
# content token ("categories", "topics") is lexically closer to the SUB* keys ("subcategories", "subtopics")
# than to "super categories" — so unguarded fuzzy flips the DIRECTION (super→sub). When a parent signal pairs
# with a structural noun, force super_category (the user's Pearltrees convention). Requires BOTH so a plain
# "Super Cool Resources" / "My Groups" is NOT swept up.
_PARENT_SIGNAL = _re.compile(r"\b(super|broad(?:er)?|parent|ancestor)\b")
_STRUCT_NOUN = _re.compile(r"(categor|topic|group|theme)")


# tag/qualifier headers: a section header often pairs a category TAG with a free-text qualifier, either across
# a DASH — in EITHER order, "A-E -- Subtopics", "IT - Subtopics" (qualifier first), "Subtopics - old" (tag
# first) — or in PARENTHESES, "Subtopics (old)", "Further reading (from wikipedia)". Split on a
# WHITESPACE-SURROUNDED dash (so alphabetical-range qualifiers "A-E"/"0-9"/"N-Z" are NOT split) OR a paren,
# then categorise WHICHEVER segment matches: the qualifier can't dilute the fuzzy ratio or override the tag,
# and a keyword INSIDE the parens ("from wikipedia") is still found. Real-harvest examples informed this.
_QUALIFIER = _re.compile(r"\s+(?:--+|[-–—])\s+|\s*[()]\s*")   # whitespace-surrounded dash, or a parenthesis


def _segments(text):
    parts = [s.strip() for s in _QUALIFIER.split(text or "") if s.strip()]
    return list(dict.fromkeys(parts + [text]))             # each segment (in order), then the whole label


def fuzzy_mode(text, threshold=0.78):
    """Best edit-distance match of a section label to a canonical keyword → (category, similarity), or
    (None, best) below threshold. Also tries the label's individual TOKENS so a typo'd keyword embedded in a
    longer header still matches."""
    t = _norm(text)
    if not t:
        return (None, 0.0)
    cands = [t] + t.split()
    if (_PARENT_SIGNAL.search(t) and _STRUCT_NOUN.search(t)) or "navigate up" in t:
        r = max(difflib.SequenceMatcher(None, c, "super categories").ratio() for c in cands)
        return ("super_category", max(r, threshold))   # parent-signal guard — never a sub* match
    best_cat, best = None, 0.0
    for key, cat in FUZZY_KEYS:
        r = max(difflib.SequenceMatcher(None, c, key).ratio() for c in cands)
        if r > best:
            best, best_cat = r, cat
    return (best_cat, best) if best >= threshold else (None, best)


def categorize(text, method="exact_phrase", fuzzy_threshold=0.78):
    """Section label → (category, method, confidence). Escalation: `exact_phrase` always tries the literal
    match first (confidence 1.0); `fuzzy` additionally falls back to an edit-distance match (a confident
    fuzzy hit is treated as a LABEL — confidence 1.0 — with provenance `fuzzy`, so it can be audited/
    down-weighted). `llm_template` is the next layer (not yet implemented)."""
    segs = _segments(text)                                 # leading "tag" (before `-- qualifier`), then full
    for s in segs:
        cat = section_mode(s)
        if cat:
            return (cat, "exact_phrase", 1.0)
    if method == "fuzzy":
        for s in segs:
            fcat, r = fuzzy_mode(s, fuzzy_threshold)
            if fcat:
                return (fcat, "fuzzy", 1.0)
    elif method != "exact_phrase":
        raise ValueError(f"unknown categorization method: {method!r} (have: exact_phrase, fuzzy)")
    return (None, method, 0.0)


def relation_for(text, method="exact_phrase", fuzzy_threshold=0.78):
    """Convenience: section label → (relation, method, confidence). `relation` is None when no rule fires."""
    cat, m, conf = categorize(text, method, fuzzy_threshold)
    return (CATEGORY_RELATION.get(cat), m, conf)
