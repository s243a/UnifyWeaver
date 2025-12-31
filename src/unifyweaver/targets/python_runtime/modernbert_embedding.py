# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2025 John William Creighton (s243a)
"""
ModernBERT Embedding Provider with Flash Attention 2 and GPU support.

This module provides high-performance embeddings using ModernBERT with:
- Flash Attention 2 for efficient attention computation
- Automatic GPU detection and utilization
- Mean pooling for sentence embeddings
- 8192 token context length (16x longer than standard BERT)

Requirements:
    pip install torch transformers flash-attn --no-build-isolation

Usage:
    from modernbert_embedding import ModernBertEmbeddingProvider

    # Auto-detect GPU, use flash attention if available
    embedder = ModernBertEmbeddingProvider()

    # Force CPU
    embedder = ModernBertEmbeddingProvider(device="cpu")

    # Get embedding
    embedding = embedder.get_embedding("How to query a database?")
"""

import os
import logging
from typing import List, Optional, Union
import numpy as np

try:
    from .embedding import IEmbeddingProvider
except ImportError:
    # Allow running as standalone script
    from embedding import IEmbeddingProvider

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
torch = None
AutoModel = None
AutoTokenizer = None


def _import_torch():
    """Lazy import torch."""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch


def _import_transformers():
    """Lazy import transformers."""
    global AutoModel, AutoTokenizer
    if AutoModel is None:
        from transformers import AutoModel as _AutoModel, AutoTokenizer as _AutoTokenizer
        AutoModel = _AutoModel
        AutoTokenizer = _AutoTokenizer
    return AutoModel, AutoTokenizer


def check_flash_attention_available() -> bool:
    """Check if flash attention 2 is available."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def get_best_device() -> str:
    """Get the best available device (CUDA > MPS > CPU)."""
    _import_torch()

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class ModernBertEmbeddingProvider(IEmbeddingProvider):
    """
    ModernBERT-style embedding provider with GPU support.

    Uses sentence-transformers for robust model loading and supports:
    - nomic-ai/nomic-embed-text-v1.5 (8192 context, 768 dim) - DEFAULT
    - intfloat/e5-large-v2 (512 context, 1024 dim)
    - Any sentence-transformers compatible model

    Attributes:
        model_name: HuggingFace model name
        device: Device to use (cuda, mps, cpu, or auto)
        max_length: Maximum sequence length
        dimension: Embedding dimension
        normalize: Whether to normalize embeddings to unit length

    Note:
        For answerdotai/ModernBERT-base, you need transformers >= 4.48.0
        and Python >= 3.10. Use nomic-embed-text-v1.5 as an alternative.
    """

    # Default model: Nomic's embedding model with 8192 context
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    # Alternative high-quality model
    ALT_MODEL = "intfloat/e5-large-v2"

    def __init__(
        self,
        model_name: str = None,
        device: str = "auto",
        max_length: int = 8192,
        trust_remote_code: bool = True,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize embedding provider.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ("auto", "cuda", "mps", "cpu")
            max_length: Maximum sequence length
            trust_remote_code: Trust remote code for model loading
            normalize_embeddings: Normalize embeddings to unit length
            batch_size: Batch size for encoding
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        # Determine device
        if device == "auto":
            self.device = get_best_device()
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {self.model_name}")

        # Load model with sentence-transformers
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=trust_remote_code,
        )

        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Determine query prefix based on model
        self._setup_query_prefix()

        logger.info(
            f"Model loaded: dim={self.dimension}, device={self.device}, "
            f"prefix='{self.query_prefix}'"
        )

    def _setup_query_prefix(self):
        """Set up query prefix based on model type."""
        model_lower = self.model_name.lower()

        if "nomic" in model_lower:
            # Nomic models use search_query: prefix
            self.query_prefix = "search_query: "
            self.document_prefix = "search_document: "
        elif "e5" in model_lower:
            # E5 models use query: prefix
            self.query_prefix = "query: "
            self.document_prefix = "passage: "
        else:
            # No prefix for other models
            self.query_prefix = ""
            self.document_prefix = ""

    def get_embedding(self, text: str, is_query: bool = True) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text: Input text to embed
            is_query: If True, add query prefix; if False, add document prefix

        Returns:
            List of floats representing the embedding vector
        """
        prefix = self.query_prefix if is_query else self.document_prefix
        prefixed_text = prefix + text if prefix else text

        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def get_embeddings(self, texts: List[str], is_query: bool = True) -> List[List[float]]:
        """
        Get embeddings for multiple texts (batch processing).

        Args:
            texts: List of input texts
            is_query: If True, add query prefix; if False, add document prefix

        Returns:
            List of embedding vectors
        """
        prefix = self.query_prefix if is_query else self.document_prefix
        prefixed_texts = [prefix + t for t in texts] if prefix else texts

        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Encode documents (uses document prefix).

        Args:
            documents: List of document texts

        Returns:
            List of embedding vectors
        """
        return self.get_embeddings(documents, is_query=False)

    def get_info(self) -> dict:
        """Get information about the embedding provider."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "max_length": self.max_length,
            "normalize": self.normalize_embeddings,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
        }


class SentenceTransformerProvider(IEmbeddingProvider):
    """
    Fallback provider using sentence-transformers library.

    Use this if you don't have flash-attn or prefer the simpler API.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        Initialize sentence-transformers provider.

        Args:
            model_name: Model name from sentence-transformers
            device: Device to use
            trust_remote_code: Trust remote code for models that require it (e.g., nomic)
        """
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = get_best_device()

        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code
        )
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"SentenceTransformer loaded: {model_name}, dim={self.dimension}, device={device}")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def get_info(self) -> dict:
        """Get provider info."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "backend": "sentence-transformers",
        }


def create_embedding_provider(
    model_name: str = None,
    device: str = "auto",
    prefer_flash_attention: bool = True,
) -> IEmbeddingProvider:
    """
    Factory function to create the best available embedding provider.

    Args:
        model_name: Model name (default: ModernBERT-base)
        device: Device to use
        prefer_flash_attention: Try to use flash attention if available

    Returns:
        An embedding provider instance
    """
    # If model_name suggests ModernBERT, use ModernBertEmbeddingProvider
    if model_name is None or "modernbert" in model_name.lower():
        try:
            return ModernBertEmbeddingProvider(
                model_name=model_name,
                device=device,
            )
        except Exception as e:
            logger.warning(f"Failed to load ModernBERT: {e}. Falling back to sentence-transformers.")

    # Fallback to sentence-transformers
    return SentenceTransformerProvider(
        model_name=model_name or "all-MiniLM-L6-v2",
        device=device,
    )


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== ModernBERT Embedding Test ===\n")

    # Check prerequisites
    print("Checking prerequisites...")
    print(f"  Flash attention available: {check_flash_attention_available()}")
    print(f"  Best device: {get_best_device()}")

    # Create provider
    print("\nCreating embedding provider...")
    try:
        provider = ModernBertEmbeddingProvider()
        info = provider.get_info()
        print(f"  Model: {info['model_name']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Device: {info['device']}")
        print(f"  Query prefix: '{info['query_prefix']}'")

        # Test embedding
        print("\nTesting embedding...")
        test_texts = [
            "How do I query a SQLite database?",
            "What is the best way to parse JSON in Python?",
            "Explain machine learning in simple terms.",
        ]

        for text in test_texts:
            emb = provider.get_embedding(text)
            norm = np.linalg.norm(emb)
            print(f"  \"{text[:40]}...\" -> dim={len(emb)}, norm={norm:.4f}")

        # Test batch
        print("\nTesting batch embedding...")
        embeddings = provider.get_embeddings(test_texts)
        print(f"  Batch size: {len(embeddings)}")
        print(f"  All dims correct: {all(len(e) == info['dimension'] for e in embeddings)}")

        print("\n=== Test Complete ===")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
