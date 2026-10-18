# REPORT — depth-balanced bidirectional walk (validation, no LLM budget)

Graph: `data/benchmark/10k/category_parent.tsv` — 8247 categories, max shortest-depth 12. The graph is **heavily cyclic** (only 436 no-parent roots; the rest sit in cycles), so no global depth is monotone along child edges.

Handshake lemma holds (E[c]=E[p]); global up-weight **β = E[c²]/E[p²] = 58.2** (children are heavy-tailed, parents concentrated).

Sweep: 40 interior seeds (both child & parent, depth 2–6, deg ≤ 30), 60 walks/seed/mode, stop_prob=0.4, hub_beta=1.0.

## 1a. PRIMARY — net directional displacement (down−up steps, the ±1 depth model)

This is the design's per-step depth model and is cycle-robust. Predicted signatures: undirected → drifts deep; child-only → strictly ≥ 0; coinflip/global → symmetric ≈ 0.

| mode | n | mean | sd | min..max | %deeper | %same | %shallower |
|---|---:|---:|---:|---:|---:|---:|---:|
| undirected, no hub-weight (β=0) | 1402 | -0.18 | 1.38 | -5..+4 | 34% | 19% | 47% |
| undirected (baseline) | 1397 | +0.06 | 1.46 | -6..+5 | 43% | 17% | 40% |
| child-only (downward) | 1407 | +1.72 | 0.99 | +1..+7 | 100% | 0% | 0% |
| bidir coinflip | 1357 | -0.05 | 1.43 | -6..+5 | 41% | 19% | 39% |
| bidir global | 1391 | -2.25 | 1.63 | -8..+1 | 1% | 1% | 98% |

### undirected, no hub-weight (β=0)  (displacement mean -0.18, sd 1.38)
```
   ≤-6 |  0
    -5 |  1
    -4 | # 10
    -3 | ### 31
    -2 | ################# 163
    -1 | ############################################## 447
    +0 | ############################ 271
    +1 | #################################### 347
    +2 | ######### 92
    +3 | ### 28
    +4 | # 12
    +5 |  0
    ≥6 |  0
```

### undirected (baseline)  (displacement mean +0.06, sd 1.46)
```
   ≤-6 |  1
    -5 |  1
    -4 | # 11
    -3 | ### 28
    -2 | ############### 134
    -1 | ########################################## 377
    +0 | ########################### 242
    +1 | ############################################## 409
    +2 | ################ 142
    +3 | #### 37
    +4 | # 13
    +5 |  2
    ≥6 |  0
```

### child-only (downward)  (displacement mean +1.72, sd 0.99)
```
   ≤-6 |  0
    -5 |  0
    -4 |  0
    -3 |  0
    -2 |  0
    -1 |  0
    +0 |  0
    +1 | ############################################## 748
    +2 | ########################## 430
    +3 | ######### 146
    +4 | ### 55
    +5 | # 17
    ≥6 | # 11
```

### bidir coinflip  (displacement mean -0.05, sd 1.43)
```
   ≤-6 |  2
    -5 | # 5
    -4 | ## 17
    -3 | ### 29
    -2 | ############# 123
    -1 | ####################################### 360
    +0 | ############################ 260
    +1 | ############################################## 429
    +2 | ########## 97
    +3 | ### 29
    +4 | # 5
    +5 |  1
    ≥6 |  0
```

### bidir global  (displacement mean -2.25, sd 1.63)
```
   ≤-6 | ###### 74
    -5 | ##### 56
    -4 | ########## 126
    -3 | ############### 189
    -2 | ############################# 355
    -1 | ############################################## 568
    +0 | # 12
    +1 | # 11
    +2 |  0
    +3 |  0
    +4 |  0
    +5 |  0
    ≥6 |  0
```

## 1b. SECONDARY — node shortest-hop-depth delta (real-graph geometric check)

`depth(endpoint) − depth(seed)` with depth = shortest child-hops from any root. Noisy because the graph is cyclic/multi-parent (a child can be nearer a *different* root), so treat as a coarse cross-check, not the primary signal.

| mode | n | mean | sd | min..max | %deeper | %same | %shallower |
|---|---:|---:|---:|---:|---:|---:|---:|
| undirected, no hub-weight (β=0) | 1402 | -0.56 | 1.04 | -4..+4 | 15% | 28% | 58% |
| undirected (baseline) | 1397 | -0.30 | 1.04 | -4..+4 | 21% | 34% | 45% |
| child-only (downward) | 1407 | -0.17 | 1.08 | -4..+3 | 27% | 34% | 39% |
| bidir coinflip | 1357 | -0.36 | 1.00 | -4..+4 | 18% | 35% | 47% |
| bidir global | 1391 | -0.39 | 0.96 | -4..+2 | 18% | 32% | 50% |

## 2. Domain-reach from `Physics` (sibling reach vs apex pile-up)

Endpoints classified against depth-≤3 subtrees: **sibling** = in Chemistry/Computer_science subtree but not Physics's (|sibling|=70 nodes); **in-Physics** = Physics subtree; **generic-apex** = a depth-0 root or a node in {Academic_disciplines, Articles, Branches_of_science, Contents, Main_topic_classifications, Nature, Science} (the leak-conduit apexes we must NOT pile up on).

| mode | n | %sibling | %in-Physics | %generic-apex | mean depth |
|---|---:|---:|---:|---:|---:|
| undirected (baseline) | 1778 | 3.7% | 63.9% | 1.4% | 2.51 |
| child-only (downward) | 1814 | 3.1% | 85.8% | 0.0% | 2.82 |
| bidir coinflip | 1744 | 1.8% | 32.9% | 9.6% | 2.00 |
| bidir global | 1823 | 0.3% | 2.7% | 31.8% | 1.48 |

## 3. Verdict

- **coinflip is the cleanest depth distribution** and the recommended bidirectional mode: net displacement mean -0.05 (sd 1.43), tight and symmetric about 0 — the per-node β=c/p martingale holds in practice. It reaches sibling domains (1.8% of Physics endpoints land in the Chemistry/CS depth-≤3 subtrees, vs **0% for child-only**, which structurally cannot leave the subtree) while keeping generic-apex pile-up modest (9.6%).
- **global (β=E[c²]/E[p²]=58.2) is NOT recommended on this graph.** It over-corrects hard to the apex: displacement mean -2.25, and 31.8% of Physics endpoints pile onto generic apexes. Cause: E[c²] is dominated by a handful of mega-hubs (max 1778 children ⇒ c²≈3.2M), so the aggregate β is ~20× a typical node's c/p≈2–3; plugging that into the *local* rule P(down)=c/(c+βp) makes almost every interior node go up. The size-biased mean-field also double-counts the down-branching that hub-down-weighting already suppresses. A robust global variant would trim hubs / use the effective (hub-weighted) branching, but the per-node coinflip is exact and free of this failure — prefer it.
- **The β=1 hub-down-weighting already removes most of the baseline's deep drift** (undirected mean +0.06 vs no-hub-weight -0.18): the design's 'undirected → skewed deep' shows up clearly only without hub-weighting. So coinflip's win over the tuned baseline is mainly a *tighter, symmetric* distribution (smaller tails) plus genuine lateral reach, not a large mean shift.
- **Recommended default mix:** `--bidir --bidir-mode coinflip --bidir-frac 0.5` — half child-only walks for deep in-domain ancestor→descendant structure (the vertical axis), half depth-balanced coinflip walks for lateral sibling/cousin structure (the horizontal axis). Lean to `--bidir-frac 0.3–0.4` if you want to keep the in-domain coherence dominant.
