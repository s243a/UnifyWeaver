"""
Fuzzy Logic Runtime for UnifyWeaver

Provides NumPy-based implementations of fuzzy logic operations:
- f_and: Fuzzy AND (product t-norm)
- f_or: Fuzzy OR (probabilistic sum)
- f_dist_or: Distributed OR (base score into each term)
- f_union: Non-distributed OR (base * OR result)
- f_not: Fuzzy NOT (complement)

All operations work on both scalars and NumPy arrays for batch processing.
"""

from typing import List, Tuple, Union, Optional, Callable, Dict, Any
import numpy as np

# Type aliases
Score = Union[float, np.ndarray]
WeightedTerm = Tuple[str, float]  # (term, weight)
TermScores = Dict[str, Score]  # term -> score mapping


def f_and(terms: List[WeightedTerm], term_scores: TermScores,
          default_score: float = 0.5) -> Score:
    """
    Fuzzy AND using product t-norm.

    Formula: w1*t1 * w2*t2 * ...

    Args:
        terms: List of (term, weight) tuples
        term_scores: Dictionary mapping terms to their scores
        default_score: Score to use for unknown terms (default 0.5)

    Returns:
        Product of weighted term scores (scalar or array)

    Example:
        >>> f_and([('bash', 0.9), ('shell', 0.5)], {'bash': 0.8, 'shell': 0.6})
        0.216  # 0.9*0.8 * 0.5*0.6
    """
    result = 1.0
    for term, weight in terms:
        score = term_scores.get(term, default_score)
        result = result * (weight * score)
    return result


def f_or(terms: List[WeightedTerm], term_scores: TermScores,
         default_score: float = 0.5) -> Score:
    """
    Fuzzy OR using probabilistic sum.

    Formula: 1 - (1 - w1*t1) * (1 - w2*t2) * ...

    Args:
        terms: List of (term, weight) tuples
        term_scores: Dictionary mapping terms to their scores
        default_score: Score to use for unknown terms (default 0.5)

    Returns:
        Probabilistic sum of weighted term scores

    Example:
        >>> f_or([('bash', 0.9), ('shell', 0.5)], {'bash': 0.8, 'shell': 0.6})
        0.804  # 1 - (1-0.72)(1-0.3)
    """
    complement = 1.0
    for term, weight in terms:
        score = term_scores.get(term, default_score)
        complement = complement * (1 - weight * score)
    return 1 - complement


def f_dist_or(base_score: Score, terms: List[WeightedTerm],
              term_scores: TermScores, default_score: float = 0.5) -> Score:
    """
    Distributed OR - base score distributed into each term before OR.

    Formula: 1 - (1 - Base*w1*t1) * (1 - Base*w2*t2) * ...

    Note: f_dist_or(1.0, terms, scores) is equivalent to f_or(terms, scores)

    Args:
        base_score: The base score to distribute into each term
        terms: List of (term, weight) tuples
        term_scores: Dictionary mapping terms to their scores
        default_score: Score to use for unknown terms

    Returns:
        Distributed OR result

    Example:
        >>> f_dist_or(0.7, [('bash', 0.9), ('shell', 0.5)], {'bash': 0.8, 'shell': 0.6})
        0.60816  # 1 - (1-0.7*0.72)(1-0.7*0.3)
    """
    complement = 1.0
    for term, weight in terms:
        score = term_scores.get(term, default_score)
        complement = complement * (1 - base_score * weight * score)
    return 1 - complement


def f_union(base_score: Score, terms: List[WeightedTerm],
            term_scores: TermScores, default_score: float = 0.5) -> Score:
    """
    Non-distributed OR (union) - base score multiplies the OR result.

    Formula: Base * (1 - (1-w1*t1)(1-w2*t2)...)

    The difference from f_dist_or is the interaction term:
    - f_union: Sab (union)
    - f_dist_or: S²ab (distributed)

    Args:
        base_score: The base score to multiply with OR result
        terms: List of (term, weight) tuples
        term_scores: Dictionary mapping terms to their scores
        default_score: Score to use for unknown terms

    Returns:
        Base score times OR result

    Example:
        >>> f_union(0.7, [('bash', 0.9), ('shell', 0.5)], {'bash': 0.8, 'shell': 0.6})
        0.5628  # 0.7 * 0.804
    """
    or_result = f_or(terms, term_scores, default_score)
    return base_score * or_result


def f_not(score: Score) -> Score:
    """
    Fuzzy NOT (complement).

    Formula: 1 - score

    Args:
        score: The score to complement

    Returns:
        1 - score

    Example:
        >>> f_not(0.3)
        0.7
    """
    return 1 - score


# =============================================================================
# NumPy Vectorized Versions (for batch processing)
# =============================================================================

def f_and_batch(terms: List[WeightedTerm],
                term_scores_batch: Dict[str, np.ndarray],
                default_score: float = 0.5) -> np.ndarray:
    """
    Batch fuzzy AND for multiple items simultaneously.

    Args:
        terms: List of (term, weight) tuples
        term_scores_batch: Dict mapping terms to score arrays (one per item)
        default_score: Score for unknown terms

    Returns:
        Array of AND results, one per item
    """
    # Determine batch size from first term
    first_term = terms[0][0] if terms else None
    if first_term and first_term in term_scores_batch:
        result = np.ones(len(term_scores_batch[first_term]))
    else:
        return np.array([default_score])

    for term, weight in terms:
        scores = term_scores_batch.get(term, np.full(len(result), default_score))
        result = result * (weight * scores)

    return result


def f_or_batch(terms: List[WeightedTerm],
               term_scores_batch: Dict[str, np.ndarray],
               default_score: float = 0.5) -> np.ndarray:
    """
    Batch fuzzy OR for multiple items simultaneously.

    Args:
        terms: List of (term, weight) tuples
        term_scores_batch: Dict mapping terms to score arrays
        default_score: Score for unknown terms

    Returns:
        Array of OR results, one per item
    """
    first_term = terms[0][0] if terms else None
    if first_term and first_term in term_scores_batch:
        complement = np.ones(len(term_scores_batch[first_term]))
    else:
        return np.array([1 - default_score])

    for term, weight in terms:
        scores = term_scores_batch.get(term, np.full(len(complement), default_score))
        complement = complement * (1 - weight * scores)

    return 1 - complement


def f_dist_or_batch(base_scores: np.ndarray,
                    terms: List[WeightedTerm],
                    term_scores_batch: Dict[str, np.ndarray],
                    default_score: float = 0.5) -> np.ndarray:
    """
    Batch distributed OR for multiple items.

    Args:
        base_scores: Array of base scores (one per item)
        terms: List of (term, weight) tuples
        term_scores_batch: Dict mapping terms to score arrays
        default_score: Score for unknown terms

    Returns:
        Array of distributed OR results
    """
    complement = np.ones_like(base_scores)

    for term, weight in terms:
        scores = term_scores_batch.get(term, np.full_like(base_scores, default_score))
        complement = complement * (1 - base_scores * weight * scores)

    return 1 - complement


def f_union_batch(base_scores: np.ndarray,
                  terms: List[WeightedTerm],
                  term_scores_batch: Dict[str, np.ndarray],
                  default_score: float = 0.5) -> np.ndarray:
    """
    Batch non-distributed OR (union) for multiple items.

    Args:
        base_scores: Array of base scores
        terms: List of (term, weight) tuples
        term_scores_batch: Dict mapping terms to score arrays
        default_score: Score for unknown terms

    Returns:
        Array of union results (base * OR)
    """
    or_result = f_or_batch(terms, term_scores_batch, default_score)
    return base_scores * or_result


# =============================================================================
# Score Combination Utilities
# =============================================================================

def multiply_scores(scores1: np.ndarray, scores2: np.ndarray) -> np.ndarray:
    """Element-wise multiplication of score arrays."""
    return scores1 * scores2


def blend_scores(alpha: float, scores1: np.ndarray,
                 scores2: np.ndarray) -> np.ndarray:
    """
    Blend two score arrays: alpha*scores1 + (1-alpha)*scores2

    Args:
        alpha: Blend factor in [0, 1]
        scores1: First score array
        scores2: Second score array

    Returns:
        Blended scores
    """
    return alpha * scores1 + (1 - alpha) * scores2


def top_k(items: List[Any], scores: np.ndarray, k: int) -> List[Tuple[Any, float]]:
    """
    Get top K items by score.

    Args:
        items: List of items
        scores: Array of scores (same length as items)
        k: Number of top items to return

    Returns:
        List of (item, score) tuples, sorted descending by score
    """
    indices = np.argsort(scores)[::-1][:k]
    return [(items[i], float(scores[i])) for i in indices]


# =============================================================================
# Filter Predicates
# =============================================================================

def apply_filter(items: List[Any], scores: np.ndarray,
                 filter_fn: Callable[[Any], bool]) -> Tuple[List[Any], np.ndarray]:
    """
    Apply a boolean filter to items, keeping only those that pass.

    Args:
        items: List of items
        scores: Array of scores
        filter_fn: Function that returns True for items to keep

    Returns:
        Tuple of (filtered_items, filtered_scores)
    """
    mask = np.array([filter_fn(item) for item in items])
    filtered_items = [item for item, keep in zip(items, mask) if keep]
    filtered_scores = scores[mask]
    return filtered_items, filtered_scores


def apply_boost(items: List[Any], scores: np.ndarray,
                boost_fn: Callable[[Any], float]) -> np.ndarray:
    """
    Apply a fuzzy boost to scores based on item properties.

    Args:
        items: List of items
        scores: Array of scores
        boost_fn: Function that returns boost factor for each item

    Returns:
        Boosted scores (original * boost)
    """
    boosts = np.array([boost_fn(item) for item in items])
    return scores * boosts
