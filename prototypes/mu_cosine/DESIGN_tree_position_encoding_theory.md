# Tree-Position Encodings for Process-Expression ASTs — Implementation Handoff

**Status: future-work design contract, revised 2026-07-25.** This is the position-encoding
companion to [`DESIGN_expression_encoder_future.md`](DESIGN_expression_encoder_future.md).
It inherits that document's gate: do not schedule this work unless the frozen P1 comparison
supports expression conditioning and the deterministic P3 composition baseline has plateaued.
This note changes no P1 or P3 protocol and authorizes no downstream claim.

Current art remains:

- [`process_cards.py`](process_cards.py): registry-driven parsing and canonicalization, plus the
  current shortened `ast_sha` convenience identifier; it does not compose learned tree
  embeddings. The future artifact contract below retains the canonical AST and a full SHA-256
  digest rather than treating a shortened digest as lossless.
- [`DESIGN_process_expression_implementation.md`](DESIGN_process_expression_implementation.md):
  the P0–P4 ladder and deterministic P3 baseline.
- [`ARCHITECTURE_filing_engine.md`](../../docs/design/ARCHITECTURE_filing_engine.md): FUSE-003 and OPENQ-012,
  the eventual conditioning and cross-corpus consumers.

## 0. Background: what this is for

*Added after a cold-reader comprehension test failed (an LLM tutor quizzed on this doc missed the
point of all five of its own questions). This section deliberately restates things the authors
"already know" — the rest of the document opens straight into the contract layer, and a reader
without the project's context has to reverse-engineer the positive picture from prohibitions.*

The process-expression encoder maps a **fixed** canonical AST — the representation of a
distance-type function over a graph, e.g. `max(floor, gamma^hops * lca_frac)` — to semantic
vectors used for downstream training. The tree is an **input, not a learning target**: nothing
in this design updates, rearranges, or generates tree structure. Node positions exist to give
the encoder structural information about where each token sits in the function.

Position information lives at three levels, and most of this document's rules are about keeping
them separate:

| level | what it is | may be used as a key? |
|---|---|---|
| identity | canonical AST + registry + digests | yes — seals, caches, manifests |
| coordinates | materialized typed-path indices (static, per node) | yes — sealed training rows |
| features | position vectors computed from coordinates, each forward pass | **never** |

Coordinates are looked up; features are computed. Depth and lateral (role/ancestor) information
are both **coordinates** and are combined into one position **feature** — first locally per edge
(an operator `C` over the depth and role tables, §3), then along the path (an order-sensitive
encoder `g`, §5). Which operator `C` uses — concatenation, outer product, circular convolution,
bilinear — is an ablation arm: a choice of compression scheme over the same coordinates, not a
fixed part of the design.

Terminology used throughout: **typed role path** (the coordinate), **position encoding** (the
feature), **estimand** (which relation the function estimates — `DESIGN_registry_v0.4.md` R7),
**pin** (an audit/provenance annotation in its own channel, §11), **sealed row** (a training
example whose static bytes are frozen and hashable).

## 1. Structural authority: a typed root-to-node role path

The earlier shorthand `(depth, breadth)` is not a node identity. At depth 3, for example,
`arg:0` below the first argument and `arg:0` below the second argument have the same local
coordinates but different locations and often different meanings.

For every tokenized AST item `u`, freeze its complete root-to-item path

```
p(u) = (rho_1, rho_2, ..., rho_k),    k = depth(u).
```

Each `rho_t` is the typed role of the edge from the item at depth `t-1` to the item at depth
`t`. The exact serialized role schema is authoritative in
[`DESIGN_expression_encoder_future.md`](DESIGN_expression_encoder_future.md):

- `ARG(index, expected_output_type)` for positional expression children, with the registry's
  positional type and zero-based index;
- `KWARG(registry_kw_id, value_type)` for keyword values;
- `LIST_ITEM(index, element_type)` for nested list items;
- ordered `MOD(index)` and `PIN(index)` roles; and
- `LITERAL_BYTE(index)` below exact literal payloads.

The root token has the empty path. `<ROOT>` is a distinguished content token, not an edge role.
This document's lower-case mathematical symbols denote those exact serialized roles; it does not
define a second vocabulary.

The exact path is serialized in the example/split manifest and is recoverable from the resolved
canonical AST. Sibling indices are local to their parent. Keyword roles use the canonical
registry keyword ID, not the source-text order; canonicalization already sorts keywords. No
modulo, clipping, or hash-bucket operation is allowed in this authoritative path.

The pair

```
(depth(u), local_role(u)) = (k, rho_k)
```

is a useful **lossy feature**, not a coordinate system. Distinct paths can share it. Likewise,
any learned position vector is a conditioning feature, never process identity. The canonical AST,
registry version, factory/manifest fingerprint, and their full digests remain authoritative.

## 2. Encoder input contract

The implementation may represent the path as a sequence rather than forcing it into one vector.
For each AST item it must expose:

```
path_role_ids : [max_path_length]  # root-to-item, in order
path_mask     : [max_path_length]  # true entries only; padding is not a role
depth         : integer
parent_index  : integer or ROOT
local_role_id : integer
```

The expression encoder consumes the ordered role sequence or an equivalent parent-pointer tree
whose message-passing steps preserve that order. A commutative sum of edge-role vectors alone
does not meet this contract: it maps `(rho_a, rho_b)` and `(rho_b, rho_a)` to the same value.

For the operator comparisons below, let

- `d_k in R^(D_d)` be the table code for edge depth `k`;
- `b_rho in R^(D_b)` be the table code for typed role `rho`;
- `K_d` and `K_r` be the frozen depth and role vocabulary sizes; and
- `C(d_k, b_rho)` be the local edge feature.

The table parameter count is

```
P_tables = K_d D_d + K_r D_b.
```

Unless stated otherwise, parameter counts below are **in addition** to `P_tables`. Biases are
counted explicitly. The same order-sensitive path aggregator and the same content encoder must be
held fixed within an ablation; otherwise the comparison is not a position-operator comparison.

## 3. Local depth-by-role combination operators

The operations below combine the depth and typed-role code for one edge. None of them, by itself,
represents the complete path.

| family | definition | output dimension | trainable parameters beyond tables |
|---|---|---:|---:|
| Concatenation | `C(d,b) = [d ; b]` | `D_d + D_b` | `0` |
| Outer/Kronecker feature | `C(d,b)_(i,j) = d_i b_j` | `D_d D_b` | `0` |
| Circular convolution | `C(d,b)_t = sum_i d_i b_((t-i) mod D)`; requires `D_d=D_b=D` | `D` | `0` |
| Dense learned bilinear tensor | `C(d,b)_o = sum_(i,j) W_(o,i,j) d_i b_j + c_o` | `D_o` | `D_o D_d D_b + D_o` |
| Scalar bilinear | `C(d,b) = d^T A b + c` | `1` | `D_d D_b + 1` |
| Addition | `C(d,b) = d + b`; requires `D_d=D_b=D` | `D` | `0` |
| Rank-`R` factorized bilinear | `C(d,b) = H ((U d) elementwise_mul (V b)) + c` | `D_o` | `R(D_d + D_b + D_o) + D_o` |

For the factorized form,

```
U in R^(R x D_d),  V in R^(R x D_b),  H in R^(D_o x R),  c in R^(D_o).
```

It is a CP-style rank-`R` factorization of the dense three-way tensor. `R`, `D_o`, and whether
biases are used must be frozen in the run manifest; silently changing them changes the model
class.

For an explicit fixed-projection view of circular convolution, choose the outer-feature vector

```
x = vec(d b^T),    x_(i,j) = d_i b_j,
```

with the `(i,j)` coordinate order frozen. Then

```
d circular_conv b = P x,
P_(t,(i,j)) = 1 if i+j = t (mod D), else 0.
```

Thus circular convolution is one particular fixed linear projection of the outer feature. Any
normalization factor is part of the operator definition and must be pinned; the formula above is
unnormalized.

These families do **not** form a strict information hierarchy. Concatenation allows direct
recovery of its two input codes. An outer feature has scale ambiguity and need not identify an
unknown pair. A lower-dimensional projection can still separate every pair in a finite frozen
vocabulary, while a higher-dimensional learned representation can collide or overfit. “More
dimensions” and “more interactions” are capacity statements, not guaranteed information or
accuracy orderings.

## 4. Complete-path composition

For `p(u) = (rho_1, ..., rho_k)`, form the ordered edge sequence

```
e_t = C(d_t, b_(rho_t)),    t = 1..k.
```

Feed `(e_1, ..., e_k)` and its mask to the shared order-sensitive path encoder, or make the same
information available through parent-indexed tree attention. The root uses a frozen root token.
The output `g(p(u))` may be concatenated with the content token or used as an attention bias, but
the choice must be common across the registered operator ablation.

This separation is load-bearing:

1. the exact typed role path specifies location;
2. `C` controls how depth and the role at one edge interact; and
3. the path encoder controls how successive edges compose.

Do not call `g(p)` lossless or collision-free unless that property is established for the frozen,
finite manifest. It is a learned feature. The exact path and AST digest remain available for
auditing even when two vectors are numerically equal.

## 5. What HRR unbinding does—and does not—provide

Let `z = d circular_conv b`. In the Fourier domain,

```
DFT(z) = DFT(d) elementwise_mul DFT(b).
```

If **one key is known**, and its Fourier coefficients are nonzero and suitably conditioned, the
other code can be recovered by Fourier-domain division. With random unitary or approximately
orthogonal codebooks, circular correlation is often used as an approximate unbinding operation.
Those are codebook and known-key assumptions.

Given only `z`, the unknown pair `(d,b)` is not generally recoverable: many factorizations produce
the same product, and scale/phase ambiguities remain. Therefore this design must not describe HRR
as making the pair “approximately invertible.” Its practical question is empirical separation of
the finite registered depth-role pairs and paths.

## 6. Limited analogies

The outer feature is useful because a bilinear cross-energy is linear in it:

```
d^T A b = inner_product(A, d b^T).
```

For a joint Gaussian whose precision matrix has a depth/role cross-block, a term of this form can
appear in its quadratic energy. That does **not** make `d b^T` a covariance matrix, nor does it
model the Gaussian self-terms or normalization. The analogy is limited to representing a
cross-interaction.

Likewise, an attention logit after learned projections is a scalar bilinear form

```
(W_q d)^T (W_k b).
```

Multiple heads learn multiple such scores and use them to mix value vectors. They do not, in
general, recover or retain the full outer product. Attention is therefore an example of learned
bilinear scoring, not evidence for an information hierarchy among the operators above.

## 7. Typing and identifiability

Depth codes and edge-role codes use separate tables and separate parameter namespaces. This
prevents the trivial role swap that a shared table plus a commutative operator would permit:
“depth 2” cannot be looked up as “arg 2.” It does **not** make the learned factors statistically
identifiable. Permutations, rescalings, rotations, sign changes, and compensating changes in a
downstream projection can leave model outputs unchanged.

Consequently:

- interpret downstream behavior, not individual embedding coordinates;
- never use a position vector as a cache or process key; and
- bind artifacts to the canonical AST digest, registry/canonicalizer versions, split manifest,
  tokenizer version, and position-encoder configuration.

## 8. Measured envelope and overflow behavior

Do not encode the earlier informal claims that the grammar is “depth <= 4” or “breadth <= 8” as
facts. Before implementation, scan:

1. every registered current process;
2. every process in the frozen downstream ledger;
3. the synthetic generator's proposed train/dev/test manifests; and
4. each value/list token emitted by the proposed tokenizer.

Record at least maximum and empirical distributions for AST depth, positional arity, list length,
keyword count, modifier count, pin count, and typed-path length. Freeze the resulting table sizes
and limits before fitting.

The overflow policy must also be frozen. The recommended confirmatory behavior is fail closed:
reject an expression whose path depth or role index exceeds the manifest envelope, report the
count and hashes, and do not clip, wrap, or reuse the final in-range code. A preregistered
`OVERFLOW` role is permissible for an explicitly exploratory deployment, but it is lossy and its
rate must be reported. Expanding a table creates a new model revision and requires a new manifest.

## 9. Required ablation and diagnostics

Preregister one primary downstream metric. Screen the following position variants only on
synthetic train/dev and downstream inner data:

1. no learned position feature;
2. lossy `(depth, local_role)` only;
3. complete typed path with concatenation;
4. complete typed path with addition;
5. complete typed path with circular convolution;
6. complete typed path with rank-`R` factorized bilinear interaction;
7. complete typed path with dense bilinear interaction; and
8. the unprojected outer feature, if its memory and downstream projection fit the frozen resource
   envelope.

The scalar bilinear form is a diagnostic interaction-score arm, not a drop-in vector replacement
unless the downstream interface explicitly accepts one scalar.

Within that inner screen, run both:

- a **matched-output/matched-budget comparison**, choosing `D_d`, `D_b`, `D_o`, and `R` on inner
  data so total trainable parameters are as close as the preregistered tolerance permits; and
- an **unmatched native-capacity diagnostic**, clearly labeled exploratory.

Freeze one selected position arm before any held evaluation. The held comparison contains that
arm, the deterministic P3 position rule, and the no-learned-position baseline—not all eight
variants. This avoids spending the outer set on architecture search.

For every screened arm report table parameters, operator parameters, path-aggregator parameters,
total trainable parameters, output dimensions, peak memory, and measured inference cost. Do not
claim an operator win when only its parameter budget changed.

Before any downstream score, test the representation itself on the frozen manifest:

- exact duplicate-vector count under a pinned numerical tolerance;
- nearest distinct-path cosine and Euclidean distance distributions;
- minimum distance and the identities of the closest distinct paths;
- numerical rank and conditioning of the finite code matrix, labeled diagnostic rather than an
  identifiability proof;
- deterministic inference with dropout disabled, plus reproducible training under pinned
  run/global seeds and deterministic-operation settings; and
- overflow count and rate.

Canonical-AST-derived randomness remains appropriate for the deterministic P3 baseline and
synthetic sample generation. Do not use it to freeze one dropout mask per example while training
the learned encoder.

Required path-order tests include:

- sibling swaps change the exact path and the encoded feature for ordered positional arguments;
- `(rho_a, rho_b)` differs from `(rho_b, rho_a)` when both are valid paths;
- nodes sharing `(depth, local_role)` but having different ancestors remain distinguishable in
  the complete-path arms;
- keyword source order canonicalizes to the same keyword-role path;
- padding and a real role never share a mask state; and
- registered commutative operators receive permutation invariance only if the canonicalizer
  explicitly normalizes that operator. Do not infer commutativity from a function name.

## 10. Exit criteria and handoff artifacts

An implementation PR is complete only when it emits:

- the typed-role vocabulary and digest;
- measured envelope and overflow policy;
- tokenizer, registry, canonicalizer, split, and synthetic-generator digests;
- the exact operator equations, dimensions, ranks, normalizations, seeds, and parameter counts;
- path-order, determinism, overflow, and collision test results; and
- a cross-reference from the encoder/decoder handoff to the selected position arm.

Selection occurs on inner data only. The P3 LOCO/cross-corpus evidence that triggered this
follow-on is already exposed and can support only an adaptive/exploratory comparison. A newly
reserved post-activation process family, corpus, or prospective cohort stays untouched until the
whole encoder configuration, including its position arm, has been frozen. Passing these checks
licenses evaluation of the position feature; it does not establish compositional generalization,
improve a filing metric, or authorize deployment.

## 11. Input channels and pin positions

*Added after the registry v0.4 stage-1 ruling (#4055); encoder-side consequences of the
pin/identity split ruled there (R9 in `DESIGN_registry_v0.4.md`). Nothing here is registry
surface.*

The encoder input has three channels — the function term, the estimand slot, and the pin
channel — and each gets its **own edge-role namespace** in the position encoding, extending the
§7 rule ("depth 2 cannot be looked up as arg 2") to "a pin edge cannot be looked up as an arg
edge." The channels use position very unevenly: the function term is deep and branching (role ⊗
depth does full work), the estimand slot is a single atom (essentially pure role code), and the
pin channel is a flat list (role code + list index, no depth).

**Ablation-stability requirement.** Removing the pin channel must move no other token's
position. This is what makes the pin-visibility comparison (V2 vs V3 cards) clean: with
sequence-indexed positions, deleting the fenced `<PINS>` section would shift every subsequent
index and confound the arms; with channel-scoped structural positions, pin removal is a pure
deletion. This requirement rules out sequence-style positional schemes for the serialized form.

**Pin positions are their target's path.** Envelope pins carry a target role path and node
digest (patterns doc §16.27), so a pin token's position is

```text
position(pin) = path(target node) extended by one pin-role edge
```

The pin sits in its own channel but wears its target's path, making the pin↔node attachment
geometric — read off the shared path prefix — rather than something attention must discover.

**Coordinates are materialized; encodings are computed.** The path *indices* (including the pin
suffix) are static, computed at data-prep time, and are what sealed training rows carry —
consistent with §7's rule never to use a position *vector* as a key. The position *encoding* is
the learned function of those indices, computed each forward pass from tables **shared** with
the target's own position computation. Weight sharing gives lockstep: when training moves the
target's position, the pin's moves with it. Two designs are rejected for breaking that shared
space: a pin-private position table, and freezing the materialized path into a fixed vector
(hash or random projection) — static-and-disconnected reintroduces the discovery burden.
A fixed-phase depth component (RoPE-style `d·ψ_step`, cf. the phase-binding proposal) is
compatible and makes the depth part deterministic before training, shrinking the learned part
to the role tables.

**Robustness comes from channel dropout, not format mixing.** Pin-visible input is an ablation
arm (V2 vs V3), optionally made continuous as stochastic pin-channel dropout within the one
canonical format (pins in channel, path-linked positions). Training with pins sometimes inline
in the term is rejected: it creates a format never seen at serve time and two input
distributions for one semantic identity. Because pin hash tokens are opaque and can only be
memorized, a pin-visible P1 win must be checked against the opaque-process-token baseline it
resembles.

**Importance is read from coordinates, never inferred from arrival order.** A second,
independent rationale for the ablation-stability rule above (two arguments for one rule keep it
from being relitigated): with stream-indexed positions, finding the structurally important
nodes is positional *arithmetic* — locate index 0, reason about what shifted — while with
tree-derived coordinates it is a *lookup*: the root is the empty path at any stream position,
and depth is a deterministic function of the coordinate, so root-ward relevance weighting is
free at readout. Serialization order is canonicalized for the digest, not chosen for the
encoder's benefit; the encoder lineage (MuAttention) is permutation-invariant over the token
set, so arrival order carries nothing and all relevance structure must come from the
coordinates. A sequence-indexed arm added later would silently break this — hence the
requirement is stated, not emergent.

**Two traversals, one join: the stream↔tree mapping.** There are two legitimate orderings of
the same tokens, and the mapping is the join between them — not merely a diagnostic overlay on
one:

1. the **canonical serialization** — pre-order, fixed, digest-bearing. Identity does not care
   about the encoder's convenience, and this traversal never moves;
2. an optional **relevance-ordered encoder stream** — a deliberate permutation
   (most-relevant-first: root functor, estimand, then progressively deeper and lateral detail)
   fed to training. The permutation is legitimate precisely because every token wears its
   tree-derived coordinates, so reordering loses nothing the encoder needs.

Two disciplines govern the second traversal. *The relevance order is a canonical function of
the coordinates, never a curation*: a deterministic sort key over the typed path (channel
priority, then depth, then role priority), versioned like `RENDERER_VERSION` — an
encoder-input-format version, never an identity input. A hand-tuned per-expression ordering
would be an unversioned judgment call baked into training data. And *the payoff must be stated
honestly*: for the permutation-invariant attention lineage a reordered stream produces the same
output, so relevance ordering buys nothing in that forward pass. Where it pays: (a)
**relevance-aware truncation** — a token-budget cut drops the least relevant tail, giving the
envelope-overflow policy a second degradation mode besides fail-closed (truncate and report the
rate, same posture as the preregistered `OVERFLOW` role); (b) **order-sensitive ablation arms**
— causal or recurrent baselines, where order is semantic by construction; (c) **deliberately
order-biased attention** — an arm that biases attention toward earlier tokens (a
primacy/position-decaying bias) makes arrival order carry the relevance signal in the forward
pass itself; the ordering matters exactly when the architecture is made order-sensitive on
purpose, and that is a legitimate design choice to screen, not an assumption to smuggle in.
None of this weakens the coordinates-first requirement above: order may *add* a relevance
signal where an arm is deliberately order-sensitive, but importance must remain readable from
coordinates alone, because the permutation-invariant arms see nothing else.

The mapping's diagnostic uses stand regardless of which traversal feeds the model: readable
attention maps ("the `gamma` leaf attends to the `hop_decay` call node" instead of "token 14
attends to token 7"), translating sequence-arm outputs into tree coordinates so arms compare on
the same diagnostics, and cross-checking the serializer against the tree. Three rules keep the
join honest: (1) *one direction of truth* — both traversals and the mapping are recomputed from
the AST plus their versioned ordering rules, never stored as independent facts, and the tree
wins any conflict; (2) *version-bound* — stream indices are a property of a traversal under a
specific contract (`tok-v2` for the canonical order; the versioned sort key for the encoder
order), and a stored mapping must bind those versions or a re-ordered stream silently mislabels
every analysis; (3) *keys stay tree-side* — §7's "never use a position vector as a key" extends
to stream indices of either traversal, for the same reason: a stream is a projection of the
tree, not an authority.

**Open question — pattern-form input (variables plus a binding channel) as an ablation arm.**
As *identity* design, a binding channel is declined: bindings are identity-determining
(`C=simplemind` and `C=pearltrees` are different processes), so the channel could not be
pin-like (identity-transparent) without making deployed identity a two-artifact (term,
bindings) pair — and the ground form already carries the same semantics in one artifact. As an
*encoder input* choice it needs no identity change at all, same status as pin visibility:
identity stays ground, while the rendering fed to training presents the vNext pattern form —
variable nodes plus a binding channel — letting the model learn corpus-independent structure
(the shape of `max(floor, product(hop_decay(C,γ), lca_frac(C)))` shared across bindings). The
caveat that keeps this an open question rather than a plan: repeated named variables are one
`VarId`, i.e. a *shared* node, which turns the tree into a DAG — and this document's coordinate
system is built on unique root-to-node paths. A twice-occurring variable has two paths. The pin
trick transfers (`position(binding) = path(occurrence)` extended by one bind-role edge, one
binding token per occurrence), but the DAG consequences for identifiability and the §9 path
tests are unexamined. Record, do not build, until the flat-corpus arms have run.
