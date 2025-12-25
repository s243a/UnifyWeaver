# LDA Philosophy: Why Smoothing Works

**Status:** Philosophical Framework (Not Formal Theory)
**Date:** 2025-12-24

This document collects intuitions and analogies about why frequency-domain smoothing helps in our projection learning. These ideas point in the same direction but **the exact formal relationships are not obvious**. We call this "philosophy" because it isn't directly grounded in derivable theory.

## The Core Observation

FFT smoothing empirically improves projection quality (MRR). We have multiple perspectives on *why*, but no unified formal theory.

---

## 1. Signal Processing Perspectives (Most Rigorous)

### 1.1 Nyquist Sampling

With N clusters, frequency bin k has O(N/k) effective samples.

| Bin | Effective Samples | Reliability |
|-----|-------------------|-------------|
| 0 (DC) | All N | High |
| N/4 | ~4 | Poor |
| N/2 (Nyquist) | ~2 | Very poor |

High-frequency bins are fundamentally under-sampled, regardless of the true signal.

### 1.2 Power/Information Concentration

A finite sample (rect function) becomes a sinc in the frequency domain:
- sinc power falls as 1/f²
- sinc amplitude falls as 1/f

If information ∝ amplitude (like std dev relates to variance):
- Low frequencies carry more information per bin
- High frequencies carry less

**Combined:** High-frequency bins have less information AND fewer samples to estimate them.

---

## 2. The Planck / Ultraviolet Catastrophe Analogy

Classical physics predicted infinite energy at high frequencies (ultraviolet catastrophe). Planck's quantization resolved this with:

```
B(ω) ∝ ω³ / (exp(ℏω/kT) - 1)    [3D]
```

### Dimensional Dependence

- 3D: ω³ in numerator (density of states in 3D k-space)
- 1D: ω¹ in numerator

For our 1D frequency domain:
```
S(ω) ∝ |ω| / (exp(α|ω|) - 1)    [1D analog]
```

The |ω| term accounts for 1D information capacity scaling with 1/f.

### The α Parameter

α plays the role of inverse "data temperature":
- More data → lower α → more high frequencies allowed
- Less data → higher α → more suppression

**This is speculative** - we don't have a formal derivation connecting sample size to α.

---

## 3. Statistical Moments Analogy

Higher moments require more data to estimate reliably:
- Mean (1st moment): n samples
- Variance (2nd moment): n-1 degrees of freedom
- Skewness (3rd moment): fewer effective DoF
- Kurtosis (4th moment): very noisy

High frequencies seem analogous to high moments:
- DC ≈ mean (all data contributes)
- Highest frequency ≈ fine differences (only adjacent pairs)

---

## 4. Risk Aversion / Inverse Temperature (Economics)

From the InfoEcon project: inverse temperature Θ ↔ risk aversion ρ

With uncertain estimates, we should be "risk averse" toward fine distinctions:
- High confidence → trust fine structure
- Low confidence → retreat to coarse structure

This frames α as a "risk aversion" parameter toward high-frequency noise.

---

## 5. The Geometry We Want

### 5.1 The Embedding Premise

Embeddings encode similarity via proximity: **nearby points are semantically similar**.

### 5.2 The Projection Requirement

For cluster i with queries Q_i and target answer A_i:
- Centroid(Q_i) @ W_i ≈ A_i (centroid maps to answer)
- (q - centroid) @ W_i ≈ 0 (deviations suppressed)

### 5.3 The Null Space Structure

Each W_i has:
- **Range:** direction toward the answer (the signal)
- **Kernel/Null Space:** within-cluster variation (should be suppressed)

Because points in a cluster are close together, their deviations from the centroid should map to approximately zero.

### 5.4 How Smoothing Creates This Structure

High-frequency variation in W means:
- Similar clusters → different W matrices
- Similar inputs → different outputs
- **Violates the embedding premise**

Low-frequency structure means:
- Similar clusters → similar W matrices
- Similar inputs → similar outputs
- **Preserves the embedding premise**

Smoothing enforces **Lipschitz continuity** of the mapping:
```
small input change → small output change
```

This is both semantically correct AND numerically stable.

### 5.5 The Frequency-Geometry Correspondence

```
Low frequency signal  →  Centroid projection (preserve)
                         Between-cluster structure
                         Cluster center → Answer

High frequency signal →  Kernel/null space (suppress)
                         Within-cluster variation
                         Maps to ~0
```

---

## 6. Numeric Stability

We need to avoid the transformation amplifying signal outside the kernel space:
- Within-cluster variation should stay small
- If W is noisy (high-frequency), small input differences get amplified
- Low-frequency smoothing keeps the transformation stable

---

## 7. The Dimensionality Question

With d-dimensional embeddings and K basis vectors:
- Effective representation dimensionality: K
- The remaining d-K dimensions are either null space or noise

### Cluster Tightness

- **Tight clusters** (like ModernBERT produces): small within-cluster variance, less smoothing needed
- **Loose clusters** (weak embeddings): large within-cluster variance, more smoothing needed

```
effective_dim ≈ between_cluster_variance / within_cluster_variance
```

---

## 8. Summary: Multiple Perspectives, One Direction

| Perspective | Type | Says |
|-------------|------|------|
| Nyquist | Signal processing (rigorous) | HF bins under-sampled |
| Power concentration | Signal processing (rigorous) | HF bins low information |
| Planck/UV | Physics analogy | Finite data limits resolution |
| Moments | Statistical analogy | Higher moments need more data |
| Risk aversion | Economic analogy | Be conservative with uncertainty |
| Geometry | Intuition | HF violates embedding structure |
| Null space | Intuition | HF should map to kernel |

All of these point the same direction: **suppress high frequencies when data is limited**.

We don't have a unified formal theory connecting them. The signal processing perspectives (Nyquist, power) are most rigorous. The others are suggestive analogies that may share deeper structure.

---

## 9. The ModernBERT Case: When Smoothing Is Unnecessary

Empirical observation: ModernBERT clusters had kernel condition ≈ 1.0 (orthogonal), and smoothing had no effect.

**Explanation:**

With tight clusters (like ModernBERT produces):
- Within-cluster values ≈ centroid
- Deviations are already small
- Already effectively in the null space
- No high-frequency noise to suppress
- Smoothing has nothing to do

With loose clusters (weak embeddings):
- Within-cluster values differ from centroid
- Larger deviations that could get amplified
- High-frequency noise present
- Smoothing needed to push deviations into null space

**The test for whether smoothing helps:**
```
Do within-cluster points stay close to their centroid?
  YES → smoothing unnecessary (already geometric)
  NO  → smoothing helps (enforces geometry)
```

This explains why good embeddings (ModernBERT) don't benefit: they already satisfy the geometric constraint that smoothing would enforce.

---

## 10. Open Questions

1. What is the formal relationship between Nyquist, Planck, and moments?
2. How should α (inverse temperature) be derived from data properties?
3. Does soft exponential decay (Planck) beat hard cutoff empirically?
4. Should smoothing be dimension-aware (different α for different embedding dimensions)?
5. Can we measure cluster tightness and adapt smoothing accordingly?

---

## References

- InfoEcon project: `context/projects/InfoEcon/`
- Planck discussion: `context/Obsidian/A modernized Republic/A sequal/Coupled Kalman Filter/simulator/multi-scale implication/kernals/RKHS/01 kernel space in functional analysis.md`
- Implementation: `src/unifyweaver/targets/python_runtime/planck_smoothing.py`
- FFT smoothing: `src/unifyweaver/targets/python_runtime/fft_smoothing.py`
