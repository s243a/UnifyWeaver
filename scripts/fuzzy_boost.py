#!/usr/bin/env python3
"""
Fuzzy Boost Module for Bookmark Filing.

Integrates fuzzy logic operations with semantic search for bookmark filing.
Supports term boosting, filtering, and score blending.

Usage:
    from fuzzy_boost import FuzzyBoostEngine, parse_boost_spec

    engine = FuzzyBoostEngine()
    boosted_scores = engine.apply_fuzzy_boost(
        base_scores,
        terms=[('bash', 0.9), ('shell', 0.5)],
        term_scores={'bash': 0.8, 'shell': 0.6},
        mode='dist_or'  # or 'and', 'or', 'union'
    )
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np

# Add fuzzy logic runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))

from fuzzy_logic import (
    f_and, f_or, f_dist_or, f_union, f_not,
    f_and_batch, f_or_batch, f_dist_or_batch, f_union_batch,
    multiply_scores, blend_scores, top_k, apply_filter, apply_boost
)


@dataclass
class BoostSpec:
    """Specification for a fuzzy boost operation."""
    terms: List[Tuple[str, float]]  # [(term, weight), ...]
    mode: str = 'dist_or'  # 'and', 'or', 'dist_or', 'union'
    base_weight: float = 1.0  # For dist_or and union modes


@dataclass
class FilterSpec:
    """Specification for a filter predicate."""
    predicate: str  # 'in_subtree', 'is_type', 'has_depth', etc.
    value: Any  # The value to match


@dataclass
class FuzzyConfig:
    """Configuration for fuzzy boost operations."""
    boost_and: Optional[BoostSpec] = None  # AND boost terms
    boost_or: Optional[BoostSpec] = None   # OR boost terms
    filters: List[FilterSpec] = field(default_factory=list)
    blend_alpha: float = 0.7  # Blend with original scores
    default_term_score: float = 0.5  # Score for unknown terms


def parse_boost_spec(spec_str: str, mode: str = 'dist_or') -> BoostSpec:
    """
    Parse a boost specification string into a BoostSpec.

    Format: "term1:weight1,term2:weight2,..."

    Examples:
        "bash:0.9,shell:0.5" -> [('bash', 0.9), ('shell', 0.5)]
        "python,machine_learning:0.8" -> [('python', 1.0), ('machine_learning', 0.8)]

    Args:
        spec_str: Comma-separated term:weight pairs
        mode: Fuzzy operation mode ('and', 'or', 'dist_or', 'union')

    Returns:
        BoostSpec with parsed terms
    """
    terms = []
    for part in spec_str.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            term, weight_str = part.rsplit(':', 1)
            try:
                weight = float(weight_str)
            except ValueError:
                weight = 1.0
        else:
            term = part
            weight = 1.0
        terms.append((term, weight))

    return BoostSpec(terms=terms, mode=mode)


def parse_filter_spec(spec_str: str) -> FilterSpec:
    """
    Parse a filter specification string.

    Format: "predicate:value"

    Examples:
        "in_subtree:Unix" -> FilterSpec('in_subtree', 'Unix')
        "is_type:tree" -> FilterSpec('is_type', 'tree')
        "has_depth:3" -> FilterSpec('has_depth', 3)
    """
    if ':' not in spec_str:
        return FilterSpec(predicate=spec_str, value=True)

    predicate, value = spec_str.split(':', 1)

    # Try to parse numeric values
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass  # Keep as string

    return FilterSpec(predicate=predicate, value=value)


class FuzzyBoostEngine:
    """
    Engine for applying fuzzy boost operations to bookmark candidates.

    Supports:
    - Term boosting (AND, OR, Distributed OR, Union)
    - Filtering by predicates
    - Score blending
    - Batch processing
    """

    def __init__(self, config: Optional[FuzzyConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or FuzzyConfig()

    def compute_term_scores(
        self,
        candidates: List[dict],
        terms: List[Tuple[str, float]],
        text_field: str = 'title'
    ) -> Dict[str, np.ndarray]:
        """
        Compute term relevance scores for each candidate.

        Uses simple substring matching for now. Could be extended with:
        - Fuzzy string matching
        - Embedding similarity
        - TF-IDF

        Args:
            candidates: List of candidate dicts
            terms: List of (term, weight) tuples
            text_field: Field to search in candidates

        Returns:
            Dict mapping terms to score arrays
        """
        n = len(candidates)
        term_scores = {}

        for term, _ in terms:
            scores = np.zeros(n)
            term_lower = term.lower()

            for i, c in enumerate(candidates):
                # Check title
                title = str(c.get(text_field, '')).lower()
                if term_lower in title:
                    scores[i] = 1.0
                    continue

                # Check path
                path = str(c.get('path', '')).lower()
                if term_lower in path:
                    scores[i] = 0.8  # Slightly lower for path match
                    continue

                # Default: use configured default
                scores[i] = self.config.default_term_score

            term_scores[term] = scores

        return term_scores

    def apply_fuzzy_boost(
        self,
        base_scores: np.ndarray,
        terms: List[Tuple[str, float]],
        term_scores: Dict[str, np.ndarray],
        mode: str = 'dist_or'
    ) -> np.ndarray:
        """
        Apply fuzzy boost to base scores.

        Args:
            base_scores: Original semantic search scores
            terms: List of (term, weight) tuples
            term_scores: Dict mapping terms to score arrays
            mode: 'and', 'or', 'dist_or', 'union'

        Returns:
            Boosted scores
        """
        if mode == 'and':
            boost = f_and_batch(terms, term_scores, self.config.default_term_score)
            return base_scores * boost

        elif mode == 'or':
            boost = f_or_batch(terms, term_scores, self.config.default_term_score)
            return base_scores * boost

        elif mode == 'dist_or':
            return f_dist_or_batch(
                base_scores, terms, term_scores, self.config.default_term_score
            )

        elif mode == 'union':
            return f_union_batch(
                base_scores, terms, term_scores, self.config.default_term_score
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def apply_filter(
        self,
        candidates: List[dict],
        scores: np.ndarray,
        filter_spec: FilterSpec
    ) -> Tuple[List[dict], np.ndarray]:
        """
        Apply a filter predicate to candidates.

        Args:
            candidates: List of candidate dicts
            scores: Score array
            filter_spec: Filter specification

        Returns:
            Tuple of (filtered_candidates, filtered_scores)
        """
        pred = filter_spec.predicate
        val = filter_spec.value

        if pred == 'in_subtree':
            filter_fn = lambda c: str(val) in c.get('path', '')
        elif pred == 'is_type':
            filter_fn = lambda c: c.get('item_type') == val or c.get('type') == val
        elif pred == 'has_depth':
            filter_fn = lambda c: c.get('depth') == val
        elif pred == 'depth_between':
            min_d, max_d = val if isinstance(val, tuple) else (0, val)
            filter_fn = lambda c: min_d <= c.get('depth', 0) <= max_d
        elif pred == 'has_parent':
            filter_fn = lambda c: str(val) in c.get('parent', '')
        elif pred == 'not_in_subtree':
            filter_fn = lambda c: str(val) not in c.get('path', '')
        else:
            # Unknown predicate, keep all
            return candidates, scores

        return apply_filter(candidates, scores, filter_fn)

    def boost_candidates(
        self,
        candidates: List[dict],
        boost_and: Optional[str] = None,
        boost_or: Optional[str] = None,
        filters: Optional[List[str]] = None,
        blend_alpha: float = 0.7
    ) -> List[dict]:
        """
        Apply fuzzy boost and filtering to candidates.

        This is the main entry point for boosting candidates.

        Args:
            candidates: List of candidate dicts with 'score' field
            boost_and: AND boost spec string (e.g., "bash:0.9,shell:0.5")
            boost_or: OR boost spec string
            filters: List of filter spec strings
            blend_alpha: Blend weight (1.0 = full boost, 0.0 = original)

        Returns:
            Candidates with updated scores, re-sorted
        """
        if not candidates:
            return candidates

        # Extract original scores
        original_scores = np.array([c.get('score', 0.0) for c in candidates])
        boosted_scores = original_scores.copy()

        # Apply AND boost
        if boost_and:
            spec = parse_boost_spec(boost_and, mode='and')
            if spec.terms:
                term_scores = self.compute_term_scores(candidates, spec.terms)
                and_boost = f_and_batch(spec.terms, term_scores, self.config.default_term_score)
                boosted_scores = boosted_scores * and_boost

        # Apply OR boost (distributed)
        if boost_or:
            spec = parse_boost_spec(boost_or, mode='dist_or')
            if spec.terms:
                term_scores = self.compute_term_scores(candidates, spec.terms)
                boosted_scores = f_dist_or_batch(
                    boosted_scores, spec.terms, term_scores, self.config.default_term_score
                )

        # Blend with original
        if blend_alpha < 1.0:
            boosted_scores = blend_scores(blend_alpha, boosted_scores, original_scores)

        # Apply filters
        filtered_candidates = candidates
        filtered_scores = boosted_scores

        if filters:
            for filter_str in filters:
                filter_spec = parse_filter_spec(filter_str)
                filtered_candidates, filtered_scores = self.apply_filter(
                    filtered_candidates, filtered_scores, filter_spec
                )

        # Update scores in candidates and re-sort
        result = []
        for c, score in zip(filtered_candidates, filtered_scores):
            c_copy = c.copy()
            c_copy['score'] = float(score)
            c_copy['original_score'] = c.get('score', 0.0)
            result.append(c_copy)

        # Sort by new score
        result.sort(key=lambda x: x['score'], reverse=True)

        # Update ranks
        for i, c in enumerate(result):
            c['rank'] = i + 1

        return result


def boost_filing_candidates(
    candidates: List[dict],
    boost_and: Optional[str] = None,
    boost_or: Optional[str] = None,
    filters: Optional[List[str]] = None,
    blend_alpha: float = 0.7,
    top_k: Optional[int] = None
) -> List[dict]:
    """
    Convenience function to boost filing candidates.

    Args:
        candidates: List of candidate dicts from semantic search
        boost_and: AND boost spec (e.g., "bash:0.9,python:0.5")
        boost_or: OR boost spec
        filters: List of filter specs (e.g., ["in_subtree:Unix", "is_type:tree"])
        blend_alpha: Blend weight with original scores
        top_k: Limit results to top K

    Returns:
        Boosted and filtered candidates
    """
    engine = FuzzyBoostEngine()
    result = engine.boost_candidates(
        candidates,
        boost_and=boost_and,
        boost_or=boost_or,
        filters=filters,
        blend_alpha=blend_alpha
    )

    if top_k:
        result = result[:top_k]

    return result


# =============================================================================
# CLI for testing
# =============================================================================

def main():
    """Test the fuzzy boost module."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test fuzzy boost module")
    parser.add_argument("--candidates", type=str, help="JSON file with candidates")
    parser.add_argument("--boost-and", type=str, help="AND boost spec")
    parser.add_argument("--boost-or", type=str, help="OR boost spec")
    parser.add_argument("--filter", type=str, action="append", help="Filter specs")
    parser.add_argument("--alpha", type=float, default=0.7, help="Blend alpha")
    parser.add_argument("--top-k", type=int, default=10, help="Top K results")

    args = parser.parse_args()

    # Load or create test candidates
    if args.candidates:
        with open(args.candidates) as f:
            candidates = json.load(f)
    else:
        # Test data
        candidates = [
            {"title": "BASH Scripts", "path": "/Unix/BASH", "score": 0.8, "item_type": "tree"},
            {"title": "Shell Scripting Guide", "path": "/Unix/Shell", "score": 0.7, "item_type": "tree"},
            {"title": "Python Programming", "path": "/Programming/Python", "score": 0.6, "item_type": "tree"},
            {"title": "Linux Kernel", "path": "/Unix/Linux", "score": 0.5, "item_type": "tree"},
            {"title": "Machine Learning", "path": "/AI/ML", "score": 0.4, "item_type": "pearl"},
        ]

    print("Original candidates:")
    for c in candidates:
        print(f"  {c['title']}: {c['score']:.4f}")

    # Apply boost
    boosted = boost_filing_candidates(
        candidates,
        boost_and=args.boost_and,
        boost_or=args.boost_or,
        filters=args.filter,
        blend_alpha=args.alpha,
        top_k=args.top_k
    )

    print(f"\nBoosted candidates (alpha={args.alpha}):")
    for c in boosted:
        orig = c.get('original_score', c['score'])
        print(f"  #{c['rank']} {c['title']}: {c['score']:.4f} (was {orig:.4f})")


if __name__ == "__main__":
    main()
