# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
#
# This file is part of UnifyWeaver.
# Licensed under either MIT or Apache-2.0 at your option.

"""
LDA-based Semantic Projection for RAG queries.

Projects query embeddings into answer space using a learned transformation
matrix W derived from Q-A pairs via Linear Discriminant Analysis.

The transformation W is computed as:
    W = A · Q̄ᵀ · (Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ + μI)⁻¹

Where:
    - Q̄: Matrix of cluster centroids (weighted means of questions)
    - A: Matrix of answer embeddings
    - Δw: Weighted residuals (deviations from centroids)
    - λ: Regularization for residual suppression
    - μ: Ridge regularization for numerical stability

See: docs/proposals/SEMANTIC_PROJECTION_LDA.md
     docs/proposals/COMPONENT_REGISTRY.md
"""

import numpy as np
from typing import Optional, Union, List, Tuple
import json
from pathlib import Path


class LDAProjection:
    """LDA-based semantic projection for RAG queries.

    Projects query embeddings into answer space using a learned
    transformation matrix W derived from Q-A pairs via LDA.

    Example usage:
        projection = LDAProjection('models/W_matrix.npy')
        projected = projection.project(query_embedding)
        score = projection.projected_similarity(query_emb, doc_emb)
    """

    def __init__(self, model_file: str, embedding_dim: int = 384):
        """Initialize projection with trained W matrix.

        Args:
            model_file: Path to W matrix (.npy or .json)
            embedding_dim: Expected embedding dimension
        """
        self.embedding_dim = embedding_dim
        self.W: Optional[np.ndarray] = None
        self.model_file = model_file
        self.load_model(model_file)

    def load_model(self, model_file: str) -> None:
        """Load W matrix from file.

        Args:
            model_file: Path to model file (.npy or .json)

        Raises:
            ValueError: If model format is unknown or shape is wrong
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(model_file)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if model_file.endswith('.npy'):
            self.W = np.load(model_file)
        elif model_file.endswith('.json'):
            with open(model_file) as f:
                self.W = np.array(json.load(f))
        else:
            raise ValueError(f"Unknown model format: {model_file}")

        if self.W.shape != (self.embedding_dim, self.embedding_dim):
            raise ValueError(
                f"W shape mismatch: expected ({self.embedding_dim}, {self.embedding_dim}), "
                f"got {self.W.shape}"
            )

    def project(self, query: np.ndarray) -> np.ndarray:
        """Project single query embedding to answer space.

        Args:
            query: Query embedding vector (d,)

        Returns:
            Projected embedding vector (d,)
        """
        if self.W is None:
            raise RuntimeError("Model not loaded")
        return self.W @ query

    def project_batch(self, queries: np.ndarray) -> np.ndarray:
        """Project batch of query embeddings.

        Args:
            queries: Query embeddings matrix (n, d)

        Returns:
            Projected embeddings matrix (n, d)
        """
        if self.W is None:
            raise RuntimeError("Model not loaded")
        return queries @ self.W.T

    def projected_similarity(self, query: np.ndarray, doc: np.ndarray) -> float:
        """Compute similarity between projected query and document.

        Args:
            query: Query embedding (d,)
            doc: Document embedding (d,)

        Returns:
            Cosine similarity score
        """
        projected = self.project(query)
        norm_projected = np.linalg.norm(projected)
        norm_doc = np.linalg.norm(doc)
        if norm_projected == 0 or norm_doc == 0:
            return 0.0
        return float(np.dot(projected, doc) / (norm_projected * norm_doc))

    def projected_similarity_batch(
        self, queries: np.ndarray, docs: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise similarities between projected queries and documents.

        Args:
            queries: Query embeddings (n, d)
            docs: Document embeddings (m, d)

        Returns:
            Similarity matrix (n, m)
        """
        projected = self.project_batch(queries)
        # Normalize
        proj_norms = np.linalg.norm(projected, axis=1, keepdims=True)
        doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
        proj_norms = np.where(proj_norms == 0, 1, proj_norms)
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        projected_normed = projected / proj_norms
        docs_normed = docs / doc_norms
        return projected_normed @ docs_normed.T


def compute_weighted_centroid(
    questions: np.ndarray, max_iter: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively compute weighted centroid (solves bootstrapping problem).

    The bootstrapping problem: to compute the weighted centroid, we need
    weights based on similarity to the centroid - but we need the centroid
    to compute the weights. This is solved iteratively.

    Args:
        questions: Question embeddings (n, d)
        max_iter: Number of iterations (default: 3)

    Returns:
        Tuple of (centroid, weights)
    """
    n = len(questions)
    weights = np.ones(n) / n

    for _ in range(max_iter):
        # Compute weighted centroid
        q_bar = np.sum(weights[:, np.newaxis] * questions, axis=0)
        q_bar = q_bar / np.linalg.norm(q_bar)

        # Recompute weights from similarity to centroid
        sims = questions @ q_bar
        # Softmax
        exp_sims = np.exp(sims - np.max(sims))  # Subtract max for stability
        weights = exp_sims / np.sum(exp_sims)

    return q_bar, weights


def compute_W(
    clusters: List[Tuple[np.ndarray, np.ndarray]],
    lambda_reg: float = 1.0,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Compute transformation matrix W from Q-A clusters.

    Uses LDA-based formulation:
        W = A · Q̄ᵀ · (Q̄ · Q̄ᵀ + λ · Δw · Δwᵀ + μI)⁻¹

    Args:
        clusters: List of (answer_embedding, question_embeddings) pairs
                  where answer_embedding is (d,) and question_embeddings is (n, d)
        lambda_reg: Regularization strength for residual suppression
        ridge: Additional ridge regularization for numerical stability

    Returns:
        W: Transformation matrix (d, d)
    """
    centroids = []
    answers = []
    residuals_weighted = []

    for answer, questions in clusters:
        q_bar, weights = compute_weighted_centroid(questions)
        centroids.append(q_bar)
        answers.append(answer)

        # Compute weighted residuals
        for i, (q, w) in enumerate(zip(questions, weights)):
            delta = q - q_bar
            residuals_weighted.append(np.sqrt(w) * delta)

    Q_bar = np.column_stack(centroids)  # d × m
    A = np.column_stack(answers)  # d × m
    Delta_w = np.column_stack(residuals_weighted)  # d × n_total
    d = Q_bar.shape[0]

    # Compute regularized solution with ridge for numerical stability
    cov = Q_bar @ Q_bar.T + lambda_reg * Delta_w @ Delta_w.T + ridge * np.eye(d)
    W = A @ Q_bar.T @ np.linalg.pinv(cov)

    return W


def save_W(W: np.ndarray, filepath: str) -> None:
    """Save W matrix to file.

    Args:
        W: Transformation matrix
        filepath: Output path (.npy or .json)
    """
    if filepath.endswith('.npy'):
        np.save(filepath, W)
    elif filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(W.tolist(), f)
    else:
        raise ValueError(f"Unknown format: {filepath}")


def load_W(filepath: str) -> np.ndarray:
    """Load W matrix from file.

    Args:
        filepath: Path to model file (.npy or .json)

    Returns:
        W matrix as numpy array
    """
    if filepath.endswith('.npy'):
        return np.load(filepath)
    elif filepath.endswith('.json'):
        with open(filepath) as f:
            return np.array(json.load(f))
    else:
        raise ValueError(f"Unknown format: {filepath}")
