#!/usr/bin/env python3
"""Single source of truth for the mu_cosine PRIVACY rule (scrub-everywhere): private data must NEVER reach
the public dataset, so it is dropped at parse/fuse time, inherited down the subtree, with no include-private
escape hatch. Defined ONCE here and imported by parse_smmx.py, parse_pearltrees.py and fuse_corpus.py so the
rule can't drift between corpora. See DESIGN_provenance_and_representation.md §Privacy.

We err toward dropping (a topical "Private equity" is scrubbed too) and callers LOG every scrub: a false
positive only loses public data; a false negative would LEAK private data."""
import re

# Word-boundary, case-insensitive: the user's `*private*` marker and "… Private …" trigger; `Privacy`
# and `privatize` do NOT.
PRIVATE_RE = re.compile(r"(?i)\bprivate\b")


def is_private_title(t):
    """True if a node's title/label marks it (and, by inheritance, its subtree) private."""
    return bool(t) and bool(PRIVATE_RE.search(t))


def vis_private(v):
    """Pearltrees `visibility`: 0 = public; any other set value is treated as private/restricted."""
    return v is not None and str(v).strip() not in ("", "0")


def vis_public(v):
    """True only for Pearltrees' explicit public value.

    This deliberately is not ``not vis_private(v)``: missing visibility is
    *unknown*, not evidence that a node is safe to send to an external service.
    """
    return v is not None and str(v).strip() == "0"


def propagate(private_seed, children_of):
    """Inherit privacy DOWN a containment graph: given a seed set of private keys and a
    parent_key -> iterable(child_key) map, return the full set including all descendants."""
    private = set(private_seed)
    frontier = list(private)
    while frontier:
        x = frontier.pop()
        for c in children_of.get(x, ()):
            if c not in private:
                private.add(c)
                frontier.append(c)
    return private
