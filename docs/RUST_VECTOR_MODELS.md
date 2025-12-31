# Rust Vector Search: Embedding Models

## Overview

This document covers embedding models tested and evaluated for UnifyWeaver's Rust vector search implementation using the Candle ML framework.

## Models Tested

### 1. all-MiniLM-L6-v2 (Initial Baseline)

**Status**: ✅ Tested and Working

**Specifications**:
- **Dimensions**: 384
- **Layers**: 6
- **Context Length**: 512 tokens
- **RAM Usage**: ~0.1-0.2 GB
- **Model Size**: ~90 MB
- **Architecture**: BERT
- **Best For**: Fast, lightweight embedding generation with acceptable quality

**Test Results**:
- Successfully tested on CUDA GPU
- Database size: ~134 MB (for Pearltrees RDF dataset)
- Text search: Working
- Semantic search: Working with moderate similarity scores

**HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2`

---

### 2. intfloat/e5-small-v2 (Improved Quality)

**Status**: ✅ Tested and Working (PR #246)

**Specifications**:
- **Dimensions**: 384
- **Layers**: 12
- **Context Length**: 512 tokens
- **RAM Usage**: ~0.5 GB
- **Model Size**: ~128 MB
- **Architecture**: BERT
- **Best For**: Better quality than MiniLM with moderate RAM increase

**Test Results**:
- Successfully tested on CUDA GPU
- Database size: ~134 MB (for Pearltrees RDF dataset)
- Text search: Working
- Semantic search: Working with similarity scores ~0.75
- Notable improvement over all-MiniLM-L6-v2

**HuggingFace**: `intfloat/e5-small-v2`

---

### 3. ModernBERT-base (Long Context, Current)

**Status**: ✅ Tested and Working (PR #248)

**Specifications**:
- **Dimensions**: 768
- **Layers**: 22
- **Context Length**: 8192 tokens (16x longer than BERT)
- **RAM Usage**: ~0.6 GB
- **Model Size**: ~571 MB
- **Architecture**: ModernBERT (with RoPE embeddings)
- **Best For**: Long documents, higher quality embeddings, modern architecture

**Test Results**:
- Successfully tested on CUDA GPU
- Database size: ~258 MB (2x larger due to 768-dim vs 384-dim)
- Text search: Working
- Semantic search: Working with improved scores (0.80-0.83)
- Handles documents up to 8192 tokens
- Better quality than e5-small-v2 with similar RAM footprint

**HuggingFace**: `answerdotai/ModernBERT-base`

**Implementation Notes**:
- Uses attention_mask instead of token_type_ids
- Loads config dynamically from config.json
- Requires candle-transformers 0.9+

---

## Model Comparison Table

| Model | Dims | Layers | Context | RAM (GB) | Size (MB) | DB Size* | Quality | Speed | Best Use Case |
|-------|------|--------|---------|----------|-----------|----------|---------|-------|---------------|
| all-MiniLM-L6-v2 | 384 | 6 | 512 | 0.2 | 90 | 134 MB | Good | Fast | Quick prototyping, resource-constrained |
| e5-small-v2 | 384 | 12 | 512 | 0.5 | 128 | 134 MB | Better | Medium | Balanced quality/performance |
| ModernBERT-base | 768 | 22 | 8192 | 0.6 | 571 | 258 MB | Best | Medium | Long documents, high quality |
| e5-base-v2** | 768 | 12 | 512 | 1.5 | 440 | 258 MB | Better | Slower | Higher quality, moderate context |

\* Database size based on Pearltrees RDF test dataset (~700 nodes)
\*\* Not yet tested, but documented and ready to use

---

## Models Considered But Not Supported

### allenai/longformer-base-4096

**Status**: ❌ Not Supported by Candle

**Why Considered**:
- 4096 token context (8x longer than BERT)
- Good for long documents
- Popular in academic research

**Why Not Used**:
Longformer requires custom sliding window attention implementation that is not currently available in Candle's official model set. As confirmed in [this Perplexity discussion](https://www.perplexity.ai/search/currently-i-m-using-the-all-mi-it2QX3.hRwCxpOfd8fg68w#8), there is no built-in Longformer support in Candle.

**Alternative**: We chose ModernBERT-base instead, which:
- Has even longer context (8192 vs 4096 tokens)
- Is natively supported in candle-transformers 0.9+
- Uses modern RoPE embeddings
- Lower RAM usage (~0.6 GB vs Longformer's estimated ~1.5 GB)

---

## Architecture Details

### BERT Models (all-MiniLM-L6-v2, e5-small-v2, e5-base-v2)

**Forward Pass Requirements**:
- `token_ids`: Input token IDs
- `token_type_ids`: Segment IDs for multi-sentence tasks
- Optional attention mask

**Pooling Strategy**:
- Mean pooling over all tokens
- L2 normalization of final embedding

### ModernBERT

**Forward Pass Requirements**:
- `token_ids`: Input token IDs
- `attention_mask`: Binary mask for valid tokens (no token_type_ids)

**Key Differences**:
- RoPE (Rotary Position Embeddings) instead of absolute position embeddings
- Global attention every N layers for long-range dependencies
- Optimized for longer context windows

**Pooling Strategy**:
- Mean pooling over all tokens
- L2 normalization of final embedding

---

## Model Selection Guide

### Choose all-MiniLM-L6-v2 if:
- You need the fastest embedding generation
- RAM is very constrained (< 0.5 GB available)
- Document length is always < 512 tokens
- Quality is acceptable for your use case

### Choose e5-small-v2 if:
- You want better quality than MiniLM
- You have ~0.5 GB RAM available
- Document length is < 512 tokens
- You need a good balance of speed and quality

### Choose ModernBERT-base if:
- You need to embed long documents (up to 8192 tokens)
- You want the best embedding quality
- You have ~0.6 GB RAM available
- Modern architecture is important

### Choose e5-base-v2 if:
- You want higher-dimensional embeddings (768-dim)
- Document length is < 512 tokens
- You have ~1.5 GB RAM available
- You prioritize quality over context length

---

## Environment Variables

All models support the following configuration:

```bash
# Model Selection
export MODEL_TYPE=modernbert    # or "bert" (default)
export MODEL_DIR=models/modernbert-base-safetensors

# Device Selection
export CANDLE_DEVICE=cuda       # or "cpu", "metal", "auto"

# Optional: Model Identification
export MODEL_NAME=ModernBERT-base
```

---

## Implementation Status

### Completed (PR #246, #248)
- ✅ Multi-model architecture support (BERT + ModernBERT)
- ✅ Environment-based model selection
- ✅ Dynamic config loading for ModernBERT
- ✅ GPU acceleration (CUDA)
- ✅ Separate forward pass logic for each architecture
- ✅ Comprehensive model download documentation

### Tested Models
- ✅ all-MiniLM-L6-v2
- ✅ intfloat/e5-small-v2
- ✅ answerdotai/ModernBERT-base

### Ready to Test
- ⏳ intfloat/e5-base-v2 (documented, not yet tested)

---

## Performance Notes

### GPU vs CPU
All tested models show significant speedup on CUDA GPU:
- Embedding generation: ~5-10x faster on GPU
- Batch processing: Even greater speedup with larger batches

### Database Growth
Database size scales with:
- **Number of documents**: Linear growth
- **Embedding dimensions**: 768-dim produces ~2x larger DB than 384-dim
- **Storage format**: redb provides efficient compression

### Context Length Impact
Longer context models (ModernBERT) don't significantly slow down short documents:
- 8192 token capability doesn't penalize 100-token documents
- Performance scales with actual document length, not max capacity

---

## References

- [Candle ML Framework](https://github.com/huggingface/candle)
- [ModernBERT Paper](https://arxiv.org/abs/2412.13663)
- [E5 Embeddings Paper](https://arxiv.org/abs/2212.03533)
- [Sentence Transformers](https://www.sbert.net/)
- [Longformer Discussion (Perplexity)](https://www.perplexity.ai/search/currently-i-m-using-the-all-mi-it2QX3.hRwCxpOfd8fg68w#8)

---

## See Also

- [MODEL_DOWNLOAD.md](MODEL_DOWNLOAD.md) - Detailed download instructions
- [FUTURE_WORK.md](FUTURE_WORK.md) - Planned improvements and next steps
