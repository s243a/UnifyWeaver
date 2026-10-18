# Directional multi-relational μ — `MuAttention` results

Implementation of `DESIGN_directional_attention.md`: a tiny permutation-invariant transformer over a
**set** of tagged tokens (`operator`, `anchor(root)`, `node@gen0`, `{ancestors}@gen_d`, noise-filled
absent slots) on **frozen e5** inputs, learning only the tags (`op_emb` / `anchor_tag` / `gen_emb`), a
2-layer attention block, and a per-operator sigmoid readout. Trained multi-task over operators; the
WAM-Rust core is untouched. 1.2M learned params, ~10 min on CPU.

Reproduce (HF egress for e5):
```bash
python3 train_mu_attention.py --steps 900 --bs 64 --lr 5e-4 \
    --wiki-weight 0.5 --margin-weight 1.0 --wiki-abs 0.5 --llm --save model_3op.pt
```

## The headline — the directional capability the symmetric control structurally cannot have

The #3287 control is `μ(a,b)=cos(f(a),f(b))` — **inherently symmetric** (`μ(a|b)=μ(b|a)`), so on any
*directional* task it is pinned at **50% by construction**: it can never order `child|parent` above
`parent|child`. `MuAttention` makes asymmetry structural (the `anchor`/`node` tags differ, and the root
is the e5 `query:` while the candidate/ancestors are e5 `passage:`):

> **WIKI held-out edge order-accuracy `μ(child|parent) > μ(parent|child)` = 99.1%** (2521 held-out edges;
> mean μ(child|parent)=0.611 vs μ(parent|child)=0.011), vs the symmetric control's structural **50%**.

That is the design's reason to exist, validated on real data.

## Per-operator results

| operator | label source | primary metric | value | control |
|---|---|---|---|---|
| **WIKI** | `category_parent.tsv` edges (free) | held-out edge **order-accuracy** | **99.1%** | 50% (symmetric ceiling) |
| **SYM** | `mu_pairs_scored.tsv` | held-out μ corr (40 pos) / symmetry gap | **+0.335** / 0.159 | **+0.726** |
| **LLM** | `wikipedia_physics_boundary_haiku.tsv` (already bought) | in-sample directional-μ corr (654 band) | **+0.963** | — |

Gate-leak (non-physics μ(·|Physics) ≥ 0.3) — the design's head-to-head metric; control 0/5 probe, 1.1% OOD:

| operator | 5-probe leak | OOD leak (graph dist≥5, 4948 nodes) |
|---|---|---|
| SYM | **0/5** | 1.5% (μ̄ 0.013) |
| WIKI | **0/5** | 1.3% (μ̄ 0.015) |
| LLM | **0/5** | **0.2%** (μ̄ 0.005) |

Every operator's dense map passes `check_feeds_rust.py` (100% name resolution, μ∈[0,1], gated IC
general→specific):
- **SYM** is the membership map — physics 0.88–0.95 (Optics 0.88, Electromagnetism 0.95, Energy 0.92),
  non-physics ≈ 0 (Music 0.005, Football 0.000, Sociology 0.093); top-10 all physics
  (Physics, Subfields_of_physics, Wave_mechanics, Matter, Electromagnetism, …).
- **WIKI** is sparse **by design** — it scores *direct*-edge membership (μ(X|Physics) ≈ "is X a direct
  child of Physics"), so the dense map is near-0 for the 8k indirect nodes; the transitive cone is the
  Rust core's job (`descendant_mu_mass` over the SYM/LLM μ). Its validation is the order-accuracy above.
- **LLM** is the directional boundary-membership signal (in-sample corr +0.963 to the Haiku labels,
  cleanest OOD rejection at 0.2%).

## Honest head-to-head vs the control

- **Wins decisively** on the directional task: WIKI 99.1% vs the control's structural 50%. This is the
  whole point of the design — a single cosine cannot represent order.
- **Matches** the control on the gate-leak 5-probe (0/5 = 0/5).
- **Slightly behind** on OOD leak (SYM 1.5% / WIKI 1.3% vs control 1.1%; LLM 0.2% beats it).
- **Behind** the control on the pure symmetric held-out corr (**SYM +0.335 vs +0.726**). Expected: the
  control is a *single-task symmetric* model tuned for exactly that metric, while SYM here shares one
  backbone with WIKI+LLM (a multi-task trade-off) and the held-out set is only 40 positives. The SYM
  *dense map* is nonetheless strongly discriminative (physics 0.88–0.95 vs non-physics ≈ 0). Closing the
  corr gap (single-task SYM, or a higher `--sym` weight / SYM-only fine-tune) is the obvious next lever.

## SECONDARY — lin-agreement on the NODE-gated IC (#3296), not path-gated

Node-gating makes IC monotone up the DAG, so Lin stays graded: **0.1% saturated (1/1275)** here vs the
**96.7%** path-gated saturation in the control report — confirming the #3296 fix. Pearson(SYM μ,
node-gated Lin) = −0.029 all / −0.033 non-saturated (control non-saturated +0.124). Per the control
report this is a **low-resolution** check (graph-structural vs embedding-semantic similarity diverge);
the primary per-operator metrics above are the verdict, not this.

## Notes / knobs

- Ancestor depth **k=1** (node + direct parents), the design default. Off-manifold unit-norm noise for
  absent/shallow lineage (no learned absent token) doubles as the dropout regulariser (0.2 per-ancestor,
  0.1 whole-lineage); per-node seed at inference ⇒ deterministic dense map. `k` is the next ablation knob.
- One implicit judge (the judge axis is deferred, as the design recommends).
- **No new LLM budget** — the LLM operator reuses the committed boundary fixture (already bought in #3293).
