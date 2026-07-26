# Process-Expression Corpus Generator — step-2 specification

**Status: specification skeleton, 2026-07-25. Not a frozen generator spec and not an
authorization to train.** This is step 2 of the delegable sequence in
[`DESIGN_expression_encoder_future.md`](DESIGN_expression_encoder_future.md) §11: reversible typed
serialization, finite deterministic generation, structural templates, LOCO splits, and
overlap/privacy validators. It inherits that document's activation gate — frozen P1 supporting
expression conditioning and a plateaued deterministic P3 — and changes no P1 or P3 protocol.

Every quantity below marked *measured* was obtained by running the landed step-1 contract
([`process_expression_contract.py`](process_expression_contract.py)) over the registry and the nine
registered processes. Measurements land as tests in
[`test_process_expression_envelope.py`](test_process_expression_envelope.py), never as prose only.
Numbers that a test does not reproduce are marked *provisional* or *assumed*.

> **vNext boundary:** proposed variables, indexed types, interpretation/representation rules, and
> factory verification live in
> [`DESIGN_process_expression_patterns.md`](DESIGN_process_expression_patterns.md). They do not
> extend this v0.3/`pec-v2` envelope in place. Activating that language requires the separate,
> sealed migration specified there and fresh support measurements.

## 0. Bundle supersession procedure

The encoder handoff §9 states the rule — *any intentional contract change creates a new bundle and
never mutates a sealed result* — but not which pointers must move with it. That gap is how a
current bundle and its documentation drift apart, so the procedure is written down here.

A contract change to the token stream or role-path serialization requires, **in one change**:

1. bump `CONTRACT_VERSION`, with a comment naming what changed and why;
2. seal a **new** bundle file. The generator refuses to overwrite an existing target, because
   rewriting a sealed bundle is the failure this rule exists to prevent;
3. add coverage for the changed behavior to `REQUIRED_COVERAGE_CASES` — the canonical case set
   lives in code, so the committed bundle is reproducible from the module alone rather than from
   command-line flags someone typed once. *If no existing row changed, that is a signal the new
   case is missing, not that the change was harmless;*
4. move `CURRENT_GOLDEN_BUNDLE`, and with it every document that directs a consumer to a bundle:
   `DESIGN_expression_encoder_future.md` §11 step 1, and the track README's two-track table;
5. record the superseded bundle in `SUPERSEDED_GOLDEN_BUNDLES` with its **file digest**. Version
   rejection happens before the checksum is reached, so a corrupted superseded bundle otherwise
   raises exactly the same error as an intact one — retaining it as provenance is only meaningful
   if its bytes are pinned separately.

Superseded bundles stay on disk, are never mutated, and are rejected by the current loader. That
rejection is the designed behavior, not clutter.

## 1. Support and frequency are separate artifacts

The single most important structural decision in this specification:

```text
corpus   (what exists)     -> exhaustive, frozen, content-addressed, LOCO-split
sampler  (what is seen)    -> weighted, versioned, ablatable, floored
```

Exhaustive enumeration fixes the *support*. It guarantees the tokenizer and decoder are never
asked to reconstruct a structure absent from the corpus, and it is what makes a structural
holdout meaningful — a template that was never enumerated cannot be held out.

Exhaustiveness says nothing about *frequency*. Uniform sampling over the support is a choice, not
a neutral default, and it is the wrong one: grammar-legal but production-irrelevant structures
dominate a uniform draw. Frequency is therefore a separate, weighted, recorded artifact.

The separation mirrors the filing track's catalog/policy boundary: the sampler may reweight the
corpus and may never alter it. Corpus manifests, template digests, LOCO assignment, and the golden
fixtures are computed before any weight exists and are unaffected by every weighting arm.

## 2. Corpus — exhaustive enumeration under measured caps

### 2.1 Measured envelope

Over the nine registered processes (*measured*):

| quantity | max observed | registry hard limit |
|---|---:|---|
| AST depth | 3 | unbounded |
| node count | 5 | unbounded |
| positional arity | 3 | `blend` variadic `(2, unbounded)` |
| kwargs per node | 2 | 3 (`routing`) |
| modifiers per node | 1 | 1, validator-enforced |
| pins per node | **0** | unbounded |
| list length | 2 | unbounded |
| string literal bytes | **0 (none present)** | unbounded |
| token count | 120 | — |
| typed role-path length | 5 | — |

Two of these are zero. No registered process carries a pin or a string literal, so real processes
cannot supply that coverage; §3 makes the generator responsible for it.

The fourth envelope source named in
[`DESIGN_tree_position_encoding_theory.md`](DESIGN_tree_position_encoding_theory.md) §8 — every
process in the frozen downstream ledger — does not exist yet. All caps are therefore
**provisional**: a real ledger may force a new manifest revision, which is a normal versioned
change and not a retraction.

### 2.2 Caps

Provisional, with headroom over the measured envelope:

```text
max_depth        = 3      # 2 drops distill-3tier, the only three-operator chain
max_node_count   = 5
max_arity        = 3
max_kwargs_node  = 2
max_list_length  = 2
max_string_bytes = 32
max_tokens       = 512
max_path_length  = 8
```

`max_node_count` is stated explicitly because it does as much pruning as `max_depth`; a depth cap
alone leaves wide shallow trees unbounded.

### 2.3 The type system is the binding constraint

Enumeration does not explode, and the reason is structural rather than a consequence of the caps
(*measured*):

- only atoms produce `source`, so `kalman`/`blend`/`lineage`/`menu` consume flat atoms and cannot
  recurse through their own outputs;
- `e5`'s positional type is `process`, which `_output_matches` treats as a wildcard. This is the
  only recursive channel, and it is what compounds with depth.

At the §2.2 caps this yields **285,478** canonical expressions over **19,131** structural
templates (~14.9 expressions per template; 10,982 templates are singletons). Overflow policy is
fail closed: an expression exceeding any cap is rejected and counted, never clipped or wrapped.

### 2.4 Coverage validation

The generator must reproduce every registered process. *Measured:* **9/9**, including
`distill(e5(routing(e5,sonnet.lineage,menus=[10,20],t=[0.02,0.03])))`. Generator grammar coverage
strictly subsumes production. This is a blocking engineering gate, not a diagnostic.

## 3. Mandatory synthetic coverage

### 3.1 Pins are required, not decorative

*Measured* over the enumerated corpus:

| generator | V1≠V2 | V2≠V3 | all three distinct | mean distinct teacher texts |
|---|---:|---:|---:|---:|
| without pins | 14.8% | **0.0%** | 0.0% | **1.15 of 3** |
| with pins | 15.7% | 70.4% | 11.4% | 1.86 of 3 |

`render()` adds pins at verbosity 3 and nothing else, so V3 differs from V2 *only* by pins. With no
pinned rows, V2 and V3 are byte-identical for 100% of the corpus and the §4.2 alignment head
optimizes one target while appearing to serve three. The generator must therefore emit pins, and
the alignment loss must be **reported per view-pair** rather than as a single mean — otherwise the
collapse is invisible in the training curve.

The fourth view (the P2 natural-language template) does not exist. Until it does, alignment spans
three formal views of which two are frequently identical; this is a stated limitation of the arm,
not a property to be discovered later.

### 3.2 Strings

`routing.manifest` is the only string field and no registered process uses it. The step-1 golden
fixtures already freeze UTF-8 and escaped-string rows, so the contract already obliges the
tokenizer to handle them; shipping a generator that leaves the byte-fallback path untrained would
contradict the fixture authority. A small allowlist suffices — one ASCII, one non-ASCII UTF-8, one
containing an escape.

### 3.3 Privacy

Pins and strings come from a committed synthetic allowlist (`synthetic/pin-*`,
`synthetic-manifest-*`). The generator must never sample repository paths, bookmark titles, private
names, real manifest IDs, environment variables, logs, or user data. A validator rejects any
generated literal outside the allowlist. No private data is required at any point in step 2.

## 4. Sampler — weighting derived from the verification distribution

### 4.1 Every weighted dimension is classified

| class | meaning | requirement |
|---|---|---|
| `measured` | an empirical curve exists | fitted-on population, metric, partition, and grid recorded and hash-bound |
| `witnessed` | observed in production/preregistered experiments | the observation set recorded |
| `assumed` | neither | uniform, and explicitly flagged as assumed |

The generator spec records the class of every weighted dimension. A dimension claiming `measured`
without a receipt fails closed. This makes "weight what works" auditable: the first question about
any proposed weighting is *where is the sweep*, not *does that sound reasonable*.

### 4.2 Structure — witnessed anchor, resolution in the ring

The nine registered processes yield **9 distinct structural templates** (*measured*) — the
witnessed set. They are not a random sample of the grammar; they are the survivors of prior
experimentation, so the witnessed anchor is already a coarse binary effectiveness signal
(survived / unwitnessed).

The refinement that matters: **mass on the witnesses themselves buys little**. Those structures are
already verified and there is nothing left to learn about them. The signal lives in the
*perturbation ring* — the one- and two-edit neighborhoods that preregistered experiments actually
vary: swap the source atom, change a threshold, add a `distill` wrapper, reorder `blend`
arguments. Encoder sharpness across that ring converts directly into experimental signal, because
those are the comparisons the next verification round will draw.

So the weighting targets *resolution along the axes being varied*, not memorization of the points.
Where a preregistered run perturbs `decay`, numeric resolution matters; where it perturbs the
source atom, `ARG`-position resolution matters. This yields an auditable property: after training,
the encoder's high-resolution region can be compared against the axes the experiments varied.

The near-term verification distribution is already written down — the P1 ladder and its
preregistered runs. Weighting derived from the experiment plan has the strongest available
provenance: not *what we guess works*, not merely *what survived*, but *what we have already
committed to testing*. Each new preregistered round revises the weighting spec with its own
receipt.

### 4.3 The floor is non-negotiable

Every enumerated template retains non-zero mass. Down-weighting to zero silently shrinks the
exhaustive corpus to the neighborhood and destroys the far-slice measurement in §6. The mass split
across witnessed / ring / floor is stated as numbers a test reconstructs; if those numbers are an
initial guess, the spec says so.

### 4.4 Weighting is an arm, not an assertion

Uniform versus anchored, same corpus, same splits, same budget. If anchoring does not improve the
near slice, it is dropped.

## 5. Numerics — two channels with different jobs

§3.3 of the encoder handoff separates exact reconstruction from smooth conditioning. They are not
alternatives:

- **Exact channel** — carries the canonical lexical string (`0.85`, byte for byte).
- **Smooth channel** — carries numeric semantics into the latent, which is where fixed Fourier
  features plus FiLM earn their place.

### 5.0 Exactness is layered, and only one layer is mandatory

The three layers have different obligations, and conflating them is how an unnecessary exactness
requirement gets imposed on the model:

```text
tokenizer   exact by construction   (reversibility is a serialization property)
decoder     close by objective      (a learned approximation, graded by tolerance)
identity    independent of both     (canonical bytes retained alongside)
```

**Decoder output is candidate generation and conditioning — never identity.** Relaxed exactness is
safe only while that holds, so it is worth being precise about what currently enforces it and what
does not.

*What holds today:* `process_identity.py` keeps identity outside the bottleneck. Identity is
derived from a validated AST and its retained canonical bytes, never from a latent, so a decoder
rendering `0.84` for `0.85` cannot corrupt an existing identity, cache key, or residual lookup.
Reconstruction always has the reference bytes available.

*What is not yet enforced:* nothing mechanically prevents a caller from passing a decoded
candidate into `full_ast_digest_for_expression` or minting a `deployed_identity` from a decoded
node plus a fingerprint. Those APIs accept any parsed node — they cannot tell where it came from.
The separation is therefore currently a **discipline, not an invariant**, and this document should
not claim otherwise.

Consequently this is recorded as a **step-3 obligation, blocking on the decoder landing**: decoder
output must be carried in an origin-tagged wrapper that the identity and cache APIs refuse, with a
test asserting that a decoded candidate cannot mint an identity, cache key, or residual row. Until
that exists, decoded output is diagnostic-only and no consumer may treat it as a process identity.
The test added in this change proves only that identity is derived from retained bytes rather than
from decoded text; it does not, and cannot yet, guard the boundary.

The tokenizer's round-trip guarantee in §9 is unaffected. It is a property of the serialization
layer, which stays exact and lossless; "close is good" must not leak downward into it.

### 5.0.1 Precedence over the inherited handoff

[`DESIGN_expression_encoder_future.md`](DESIGN_expression_encoder_future.md) is the authoritative
parent and it currently requires exact reconstruction: §4.3 makes exact canonical-byte and
full-digest equality *the primary reconstruction event*, §7 specifies grammar-masked token
cross-entropy, and §8 reports exact AST match as an engineering gate. A tolerance-graded decoder
does not satisfy those clauses, and this document does **not** silently supersede its parent.

The precedence rule for the interim:

- the handoff's **exact-AST reconstruction gate remains primary and unchanged**. It is still
  reported, and still blocking;
- tolerance-based reconstruction (§5.4) is an **additional reported metric** and the objective for
  a registered closeness arm — not a replacement for the gate;
- promoting closeness to the primary reconstruction criterion requires amending the handoff's §4.3,
  §7, and §8 in one change, with the gates restated. That amendment is proposed, not assumed here.

Under `p = 1` the closeness arm reduces to exact token cross-entropy, which is precisely the
handoff's stated loss — so the arm at its default endpoint is already compatible, and only
`p < 1` needs the amendment.

### 5.0.2 Tolerance is per field, and integers are exempt

`menus=10` versus `menus=11`, or `depth=3` versus `depth=4`, are not "close" — they are
behaviorally different processes, and the grammar types them `int`. So:

| field kind | grading |
|---|---|
| `number`, `number_list` | within tolerance |
| `int`, `int_list` | exact after rounding — no tolerance |
| `string` | exact |

Tolerances are spec numbers reproduced by a test, revised when the production distribution moves —
the same rule as the caps. Provisional starting values, absolute unless stated:

```text
tolerance.routing.t      = 0.002     # < half the 0.01 gap between registered tiers
tolerance.margin.t       = 0.002
tolerance.lineage.decay  = 0.01      # decays live in 0.8-0.95
tolerance.blend.w        = 0.01
```

#### Per-field tolerance is not sufficient on its own

Grading list elements independently can destroy a structure that no individual element violates.
The registered thresholds `t=[0.02,0.03]` are `0.01` apart, so an independent absolute tolerance of
`0.005` per element admits `[0.025, 0.025]` — **both elements within tolerance, and the two-tier
routing policy collapsed into one tier**, while the primary metric reports success. The parser does
not prevent this: it validates list kind and non-emptiness but enforces no ordering or separation,
so `routing(e5,haiku,t=[0.03,0.02],...)` and `t=[0.025,0.025]` both parse today.

Two consequences, both binding:

1. **A tolerance must be strictly below half the minimum operational gap** of its field. Hence
   `0.002` for thresholds rather than `0.005`, which was exactly half the gap and therefore
   admitted the collapse.
2. **Ordered fields carry a structural predicate as well.** A decoded `number_list` for a
   threshold field must preserve strict ordering and a minimum separation no smaller than the
   original's. Field-wise tolerance and the structural predicate must both hold; passing only the
   first is a failure.

Where an operational gap is unknown, the honest grading is downstream route-decision equivalence —
does the decoded expression select the same tier — rather than a numeric tolerance chosen for
convenience.

### 5.1 Why the smooth channel is needed, measured before training

*Measured:* `cos(lineage(graph,decay=0.85), lineage(graph,decay=0.5)) = 0.9725` under the pinned
teacher. Two materially different processes receive near-identical alignment targets. The teacher
cannot supply numeric semantics; only the reconstruction path can.

Co-training does create the pressure that forces the latent to carry the distinction — the decoder
must reconstruct the digits through the bottleneck regardless of what the teacher sees. That
mechanism is correct, and it is the reason finding 5.1 argues *for* the numeric path rather than
against the design. But it is not free: exact digit reconstruction competes with structure and
alignment for bottleneck capacity, which is precisely why the numeric pathway is explicit rather
than left to a shared bottleneck.

One consequence must be recorded now rather than discovered later: whatever the decoder learns,
the **alignment view still inherits the teacher's numeric insensitivity**. Any downstream consumer
of the alignment view must either exclude numeric-only distinctions or report them as a known
weakness.

### 5.2 Digit coverage floor

*Measured* directly from the byte tokens the nine registered processes emit: the resolved literals
are `{0.02, 0.03, 0.85, 10, 20}`, so digit bytes `0 1 2 3 5 8` appear and **`4`, `6`, `7`, `9`
never do**.

Under §5.0 this is a **resolution** floor, not a correctness floor. A byte-level decoder can be
close without ever spelling an unseen digit: asked for `0.47` it can render `0.45` from trained
glyphs alone, so missing digits no longer produce a catastrophic failure. They cap how close the
decoder can get — and in the production threshold range that still matters, since a missing `7`
forces `0.07` to `0.06` or `0.08`, a real difference for a margin band and larger than the
§5.0.2 tolerance of 0.002.

So the widened numeric grid stays; only its justification changes. Optimizing for the distribution
the model will actually see is right — thresholds around 0.01–0.1, menu sizes 5–50, decays
0.8–0.95 — and the frequency weighting should reflect exactly that. But the support must still
include every digit byte and every magnitude scale at low frequency. Same support/frequency
separation as §1: shape the distribution for the likely case, never let the tail reach zero.

The coverage invariant lands as a test: every byte value the tokenizer can emit for a numeric
field appears in the training split, in more than one position.

### 5.3 The objective is where closeness lives or dies

Stating tolerance in the spec is not enough. Plain byte-token cross-entropy is an *exactness*
objective — it penalizes `0.84` for `0.85` exactly as hard as `0.31` for `0.85` — so a run that
declares "close is good" and then trains on unmodified token CE quietly optimizes spelling anyway.
Closeness has to be expressed in the loss or it does not exist.

#### Stochastic exact grading

The mechanism is **grading-mode selection**, not a reward term: each graded unit is assigned, by a
seeded draw, to one of two *differentiable* losses.

The unit is the **numeric field**, and the draw is **resampled every step**:

```text
for each numeric field f in a row, independently, at every step:
    L_numeric(f) = exact_token_CE(f)      with probability p
                 = tolerance_loss(f)      with probability 1 - p
p = 0.75    # provisional; ablated over {0, 0.5, 0.75, 1}
```

Both choices are load-bearing and were ambiguous in an earlier draft that wrote the formula
per row while the requirements below assigned a draw per field:

- **Per field, not per row.** A row-level unit makes one wrong digit in a 120-token expression
  decide the grading of every other field in that row, and the two units yield different
  estimators and different correlation structure. Per-field draws keep the signal dense and match
  the per-field tolerance table of §5.0.2.
- **Resampled per step, not fixed.** A permanent assignment is learnable: the decoder can
  discover which examples are never exactly graded and relax on them specifically, which
  defeats the mechanism's central property. Fresh draws keep every field exactly gradable at some
  future step, so "cannot tell which will be strictly graded" stays true throughout training
  rather than only at initialization.

Both branches are ordinary teacher-forced losses over known targets, so gradients flow in the
usual way and `p` only chooses which loss a row contributes. This is deliberate. An earlier
formulation of this section wrote the exact branch as a *verifier reward* computed after discrete
decoding, parsing, and canonicalization — which is not differentiable, supplies no gradient at
all at `p = 1`, and would silently require a policy-gradient estimator with its own sign,
sampling policy, variance control, and credit assignment. None of that is needed: at training
time the target is known, so "grade this row exactly" is just exact cross-entropy.

The masking analogy holds under this reading, and is in fact what motivates it: masking does not
add a reward, it selects which positions are graded.

Two properties make this the right shape rather than a compromise:

- **The decoder cannot tell which rows are strictly graded**, so its optimal policy is exactness
  wherever exactness is affordable. The equilibrium is the profile actually wanted: exact in the
  production region, where values are well trained and cheap to spell, and gracefully close in the
  tail, where exactness costs more capacity than it is worth. `p` is the dial between
  exact-always and close-always.
- **Exact verification remains free, as a metric.** Decode, re-parse, canonicalize, compare bytes:
  the tokenizer's exact reversibility (§5.0) supplies a deterministic oracle at zero cost. It is
  not in the loss, but it is what produces the verified-exact rate below, and what makes the
  collapse diagnostic checkable rather than inferred.

Three requirements on the implementation:

1. **Integer fields grade exactly regardless of the draw** (§5.0.2). They are outside the
   closeness regime entirely, so the draw applies only to `number` and `number_list` fields.
2. **The tolerance branch must not be plain CE.** Falling back to unmodified byte cross-entropy on
   the `(1 - p)` rows silently reimposes exactness and undoes the whole mechanism. The tolerance
   loss is the N2 scalar head or a digit-distance-weighted cross-entropy.
3. **Watch for collapse onto common values.** Exact grading can be satisfied by always emitting the
   most frequent canonical value — `0.85` forever — yielding a high verified-exact rate with dead
   numerics. See the collapse diagnostic in §5.4, which is deliberately decoder-level: encoder
   sensitivity alone cannot detect this.

#### Inference-time reuse, scoped honestly

The verifier can rerank, but only where a reference exists, and the two cases differ:

- **Reconstruction of a known process** — the caller already holds the canonical bytes and digest
  (§5.0), so re-deriving them from a decode is redundant. Reranking buys nothing here; use the
  retained bytes.
- **Novel candidate generation** — no reference bytes exist, so nothing can establish *exactness*.
  Parsing and canonicalizing establish **validity and canonical stability** only: the decode
  parses, type-checks under the pinned registry, and round-trips to itself. That is a useful
  filter and it is not an exactness oracle.

So the recorded claim is narrowed: sample `k` decodings and reject those that fail validity or
canonical round-trip; `k` and the all-candidates-rejected fallback are parameters of whatever
consumer wants this, and are specified there rather than asserted here. The earlier phrasing
("prefer the ones that verify") implied an exactness guarantee that novel generation cannot have.

### 5.4 Numeric-encoding arms

Same frozen corpus and LOCO splits across all three:

| arm | exact channel | smooth channel |
|---|---|---|
| N0 | byte tokens only | none (baseline) |
| N1 | byte tokens | fixed Fourier + FiLM |
| N2 | scalar head rendered through the canonical formatter | fixed Fourier + FiLM |

Evaluation, frozen in advance:

1. **tolerance-based reconstruction** — structure exact, `number` fields within the §5.0.2
   tolerance, `int` fields exact after rounding — weighted by the production numeric distribution.
   This replaces plain exact-match string comparison, which would grade for an objective the
   design does not hold;
2. **verified-exact rate**, reported separately, as the exactness readout;
3. tail robustness under an explicit **digit-holdout** split — train never sees `7`, test does;
   train sees two decimal places, test sees three. This turns "which encoding generalizes" into a
   measurement of the same shape as the template LOCO;
4. latent numeric sensitivity — does the *embedding* separate `decay=0.85` from `decay=0.5`, the
   direct repair target for §5.1;
5. **decoder-level numeric dependence** (below).

#### The collapse diagnostic must be decoder-level

Latent sensitivity and verified-exact rate can both look healthy while the decoder is collapsed.
The encoder may separate `0.85` from `0.5` cleanly while the decoder ignores that coordinate and
emits the majority value; if the majority value is common enough, aggregate exactness stays high
and aggregate sensitivity stays high. Neither aggregate is target-conditioned, so neither can see
it.

The diagnostic must therefore measure whether decoder output *depends on* the target value:

- **macro-averaged reconstruction over distinct target values**, not micro-averaged over rows, so
  a rare value cannot be drowned by a frequent one;
- **field-level confusion** between target and decoded value, which exposes an always-emit-`0.85`
  policy directly;
- **decoded-output variance** conditioned on target — near-zero variance across distinct targets is
  collapse regardless of the aggregate rate.

Reporting latent sensitivity alone would satisfy the letter of "watch for collapse" while missing
the failure it names.

Preregistered prediction, revised for the closeness metric: under tolerance-based grading N2's
"clean exactness story" stops being an advantage and its scalar head becomes the natural fit for
the objective, while N0's weakness moves from tail robustness to plain resolution. N1 remains the
expected middle. A different outcome is a finding, not a failure — which is the point of running
arms rather than asserting a winner.

### 5.5 `decay` as the exemplar measured prior

`decay` is the ideal first `measured` dimension: one-dimensional, executable, and cheap to sweep.
A decay sweep over a real corpus sample under the standing node-disjoint evaluation yields an
empirical curve, and sampling decay proportionally to that curve trains hardest where production
will operate. Nobody hand-sets anything; the weighting *is* a measurement.

Four conditions keep it honest:

1. **The sweep is preregistered** — corpus digest, metric, partition, grid resolution. A curve read
   off an ad-hoc run is the numeric analogue of treating a descriptive curve as a calibrated
   probability, which `RANK-006` already forbids.
2. **Provenance travels with the spec** — the curve is corpus- and metric-dependent and interacts
   with graph density. Record the fitted-on population so a later different-corpus use is visibly
   outside its support, exactly as the filing track treats calibration artifacts.
3. **The floor applies** — the curve says where mass concentrates, not where support ends. Poorly
   performing decay values still appear at low frequency; the exact channel must handle any legal
   value, and today's bad decay may be tomorrow's good one on a denser graph.
4. **No circularity** — the sweep that produces the distribution cannot also serve as evidence that
   the encoder improved numeric sensitivity. The sweep is an input; the encoder needs its own
   held-out basis.

## 6. Structural effectiveness weighting — deferred, and why

Weighting structure by measured effectiveness is the right objective, and it is not available in
v1. The analogy to `decay` breaks in two places:

- **Measurability.** `decay` is one sweep over an executable operator. Structural effectiveness
  means building a factory for each composed pipeline and running a filing evaluation.
  `process_identity.py` marks grammar-valid samples `synthetic_only` and refuses promotion without
  a verified factory fingerprint precisely because most enumerated structures are not executable.
- **Evidence.** Per-structure readings would come from the frozen downstream ledger, which has no
  real rows yet. Effectiveness-weighted structure is a downstream consumer of the ladder this
  encoder is a rung of.

**The circularity trap has teeth.** Part of the encoder's eventual job is screening candidate
structures — finding processes better than the incumbent. Weighting its training distribution by
effectiveness measured with the current e5-based stack would train it to best represent exactly
what the incumbent already prefers. That is `assert_not_circular_grading` one level up: same-model
grading of a same-model-constructed candidate establishes self-consistency, not recovery. Any
structural-effectiveness evidence used for weighting requires an independent evaluator and
inner/outer separation between the runs that fit the weights and the runs that evaluate the
encoder.

**The tractable path is features, not templates.** 19,131 per-template measurements will never
exist, but structural *features* — depth, operator composition, presence of a `routing` stage,
`distill` wrapping, blend arity — can be measured because ledger runs can be designed as
structural ablations. Templates are then weighted by feature profile: a few measured dimensions
combined by a recorded rule. Per-template effectiveness weighting is probably never worth it, and
the corpus floor covers the rest.

Staging: v1 structure is `witnessed` + floor; structural features graduate to `measured` when the
ledger runs land.

## 7. Evaluation — two slices, always reported

The template-digest LOCO split provides the mechanics; the reporting must make the separation
visible:

- **near slice** — held-out templates within the perturbation ring of a witnessed template. This
  estimates performance on the distribution the next verification round will draw from.
- **far slice** — held-out templates far from any witnessed template. This is the canary for
  whether prioritization is collapsing the tail.

Both are reported for every weighting arm, always. Improving the near slice while the far slice
craters is a finding about the weighting, not about the model. Metrics are additionally reported
per template and per depth, because the corpus is dominated by nested `e5(…)` chains — the only
recursive channel — and an aggregate would largely measure how well the model encodes those.

If a future weighting search is called *learned*, its metric, held-out set, and the minimum
far-slice degradation that disqualifies a scheme are written down before the search runs.

## 8. Splits, artifacts, and resources

Structural LOCO by template digest, whole templates to one side. *Measured* at the §2.2 caps:
train 206,851 / dev 23,213 / test 55,414, with **zero** canonical-AST overlap across all three
pairs. The sealed-test transaction of the encoder handoff §6 applies unchanged: the training
worker receives only the train/dev projection.

Resource envelope, worth stating because it constrains storage and versioning: 285,478 rows ×
384 float32 ≈ **438 MB per view**, ~1.3 GB for three views. Teacher caches are content-addressed
and pinned to the exact e5 revision; they are not regenerated silently.

## 9. What lands as tests

Following the standing rule that caps are measured and recorded, never asserted in prose:

1. the measured envelope, per quantity;
2. the registry's one-modifier cap and the resulting unreachability of `MOD(index>0)`;
3. generator coverage of all nine registered processes;
4. corpus and template counts reproduced from the frozen spec;
5. zero canonical-AST and template overlap across LOCO splits;
6. digit-byte coverage of the training split;
7. the V2≡V3 collapse rate with and without pins;
8. privacy-allowlist rejection of any out-of-allowlist literal;
9. round-trip reversibility: canonical AST → tokens → AST for every registry example and generated
   row.

## 10. Deferred

- The P2 natural-language template view, and therefore the fourth alignment view.
- Structural effectiveness weighting (§6) until the ledger supplies feature-level evidence under an
  independent evaluator.
- Any downstream claim whatsoever. This specification produces a corpus and a sampler; it does not
  authorize training, and training does not authorize deployment.
