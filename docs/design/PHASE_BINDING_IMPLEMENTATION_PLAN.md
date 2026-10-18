# Phase Binding — Implementation Plan

Phased rollout for the design specified in
`PHASE_BINDING_SPECIFICATION.md`. Each phase produces a landable PR
with tests; later phases depend on earlier ones for code or for
measurements. Cancel/replan after any phase if measurements contradict
assumptions — the plan is a default, not a contract. In particular the
whole line of work is gated by the P2 A/B: if phase tags do not beat
(or match with fewer parameters) additive tags on the filing ledger,
P3+ do not proceed.

## Sequencing summary

| phase | size | depends on | gates |
|---|---|---|---|
| P0: `phase_binding.py` operator + invariant tests | small | — | pure infrastructure |
| P1: `tag_mode="phase"` in Tokenizer/MuAttention | medium | P0 | warm-start exactness test passes |
| P2: A/B eval, additive vs phase tags | small (compute) | P1 | decision point for everything below |
| P3: fractional depth (`psi_step`) replacing `gen_emb` | small | P1; P2 favorable | interpolation probe shows depth ordering |
| P4: coherence diagnostics for fusion heads | small | P1 | a fusion-head consumer exists |
| P5: ordered-path keys (depth-indexed role phases) | medium | P2, P3 | tree-position experiments resume |

P0–P2 are the core deliverable. P3–P5 are extensions on positive P2
data.

---

## Phase 0 — the operator

### Goal

A standalone, dependency-light module implementing `U_ψ` with the
normative invariants of SPEC §1.2 as tests.

### Deliverables

- `prototypes/mu_cosine/phase_binding.py`:
  - `bind(x, psi) -> Tensor` — batched `irfft(rfft(x)·e^{iψ})`,
    accepting `x: [..., D]`, `psi: [..., D/2+1]` with broadcast;
    structural enforcement of `ψ_0 = ψ_{D/2} = 0` via
    `pad_phases(free: [..., D/2-1]) -> [..., D/2+1]`.
  - `unbind(x, psi) = bind(x, -psi)`.
  - `geometric_step(theta, g, D) -> psi_step` (SPEC §1.3).
  - `coherence(bundle) -> [..., D/2+1]` (SPEC §3).
- `prototypes/mu_cosine/test_phase_binding.py` — asserts I1–I6 at
  `atol=1e-5` on random unit vectors, plus:
  - relative-offset property on fractional depths (I3 with
    `t₁·ψ_step`, `t₂·ψ_step`);
  - `bind(x, 0) == x` exactly-ish (I6, `atol=1e-6`);
  - noise closure smoke test (mean/cov of bound unit noise ≈ unbound).

### Scope notes

No model changes. No CLI. Tests run on CPU in seconds.

---

## Phase 1 — integration behind `tag_mode`

### Goal

Wire phase tags into `MuAttention` such that
`tag_mode="additive"` (default) is byte-identical to today and
`tag_mode="phase"` with zero-init phases is *functionally* identical at
warm start.

### Deliverables

- `mu_attention.py`:
  - `MuAttention.__init__(..., tag_mode="additive")`; new zero-init
    parameters `psi_step [191]`, `psi_role [n_roles,191]`,
    `psi_nodetype [n_nodetype,191]`, `psi_prefix [3,191]` (SPEC §4.1).
    Additive tables retained regardless of mode (checkpoint compat —
    the `judge_name`/`judge_emb` bypass pattern).
  - `forward`: when `tag_mode="phase"`, compose
    `ψ = gen_id·psi_step + psi_role[...] + psi_nodetype[...] +
    psi_prefix[...]` per token from the existing index tensors
    (`gen_id`, `is_anchor`, `nodetype_of`, `prefix_of`, `is_prov`) and
    apply `bind(content, ψ)` **instead of** the corresponding additive
    terms. Operator/provenance content slots unchanged (SPEC §2).
  - Manifest fields plumbed through the existing config dict
    (SPEC §4.2).
- Tests (`test_phase_binding.py` extension or `test_mu_attention_phase.py`):
  - **warm-start exactness**: fresh model, same seed, μ outputs equal
    under both modes at init (`atol=1e-5`);
  - checkpoint round-trip: an additive checkpoint loads into a
    `tag_mode="phase"` model and reproduces its μ at init;
  - gradient smoke test: loss.backward() populates `psi_*` grads and
    only `psi_*` differ after one step in phase mode.

### Scope notes

Training scripts (`train_filing.py`, `train_mu_attention.py`) gain only
a `--tag-mode` flag passthrough. No default behavior changes anywhere.

---

## Phase 2 — A/B evaluation (decision point)

### Goal

Measure, don't assume (PHILOSOPHY principle 3).

### Deliverables

- Two training runs matched in everything but `tag_mode`, under
  `PROTOCOL_filing_ranker_eval.md`, on the standard corpus mix; results
  appended to the training-method ledger (`ARC_filing_program.md` §5
  conventions) as e.g. `REPORT_phase_binding_ab.md`.
- Metrics: the protocol's ranking metrics (MRR/hit@k per operator),
  parameter counts, and wall-clock. Secondary probe: μ sensitivity to
  tag ablation (unbinding vs zeroing).

### Gate

- Phase ≥ additive on the primary ranking metric (within noise, with
  fewer tag parameters, also counts as a pass — 191 vs 384·6 for the
  gen table alone).
- If phase < additive: stop; write the negative result into the ledger
  (the report is a deliverable either way); P3–P5 are cancelled.

---

## Phase 3 — continuous depth

### Goal

Exploit the win P1 only enables: depth as fractional binding.

### Deliverables

- Depth values beyond the `max_gen` cap and non-integer depths (e.g.
  hub-down-weighted expected hop counts from `sample_ancestors`) bind
  via `d·psi_step` with no table growth.
- Interpolation probe script: μ(node|root) as a function of continuous
  synthetic depth on held-out lineages — expect monotone/ordered
  response where the additive table gives arbitrary per-row behavior.
- Optional: retire `gen_emb` rows for phase-mode checkpoints
  (keep loading shim).

---

## Phase 4 — coherence diagnostics

### Goal

Expose superposition coherence (SPEC §3) to the evidence-fusion layer.

### Deliverables

- `coherence()` wired as an optional per-example diagnostic in
  eval paths; logged alongside μ for the fusion heads
  (`DESIGN_amortized_fusion_heads.md` — resultant length as an
  innovation-consistency weight candidate).
- No model changes; measurement only. Adoption into a fusion head is
  its own future decision.

---

## Phase 5 — ordered-path keys

### Goal

Order-sensitive path composition without leaving the torus: per-edge
keys `ψ_edge(r, t) = ψ_role[r] + t·ψ_step` (SPEC §2, commutativity
note), for the typed role paths of
`DESIGN_tree_position_encoding_theory.md` §4.

### Deliverables

- Path-composition variant in the tree-position experiment harness;
  collision test `a/b` vs `b/a` (must separate, unlike bare role sums).
- Comparison against the existing `C(d_t, b_ρt)` composition operators
  under that doc's frozen-manifest rules.

### Scope notes

This phase re-enters open research (tree position encoding); it rides
on that workstream's schedule, not this one's.

---

## Risks

| risk | mitigation |
|---|---|
| Phase tags underperform (attention already compensates for additive distortion) | P2 is an explicit gate; negative result is ledger-worthy and cheap |
| fp32 FFT round-off accumulates through composition | tags are composed in ψ-space (one bind per token), never by repeated binding |
| Hidden coupling: LayerNorm in the encoder re-scales tokens anyway | the claim is about *relative* geometry and invertibility, not norms alone; the ablation probe (P2 secondary) measures it directly |
| Checkpoint drift between modes | both tables always present; mode is a forward-path switch, warm-start equality is a test, not a hope |
