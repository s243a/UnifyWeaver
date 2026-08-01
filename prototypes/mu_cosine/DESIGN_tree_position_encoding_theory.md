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
position(pin) = path(target node) ⊗ pin-role suffix
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
