# Process-Expression Encoder and Reconstruction Decoder — Deferred Implementation Handoff

**Status: future work, 2026-07-24; amended as an implementation handoff.** This remains outside
the active P0–P4 ladder. Do not schedule it unless both activation gates hold:

1. the frozen P1 primary shows that structured expression conditioning earns its keep over flat
   process tokens; and
2. P3's deterministic AST composition has been implemented and has plateaued under its frozen
   multi-process and cross-corpus LOCO evaluation.

If either gate fails, archive this design. A learned encoder is not a way to rescue a failed
conditioning premise. The P1/P3 results used to activate the design are thereby exposed design
evidence; they cannot later be relabeled as the learned encoder's untouched test.

Companion contracts:

- `DESIGN_process_expression_language.md` — grammar, typed registry, canonicalization, and the
  lossless-identity/lossy-card distinction;
- `DESIGN_process_expression_patterns.md` — proposed vNext pattern/ground AST states, optional
  types, interpretation and representation rules, factory verification, and migration boundary.
  It is not active under registry v0.4 or `pec-v3`;
- `DESIGN_process_expression_implementation.md` and `PROTOCOL_process_expression_p1.md` — the
  empirical ladder and activation evidence;
- `DESIGN_tree_position_encoding_theory.md` — candidate compression operators for tree
  positions; the complete role-path contract below is authoritative where depth×breadth alone is
  insufficient;
- `DESIGN_path_operator.md` — fixed Fourier features plus FiLM for continuous process parameters;
- `DESIGN_filing_path_decoder_handoff.md` — the separate active contract for selective filing
  search and new-folder proposals; `DESIGN_lineage_decoder.md` is its historical predecessor.
  Neither is the reconstruction decoder specified here.

## 1. Purpose and non-goals

The proposed component maps a parsed process expression to a reusable conditioning
representation, while an auxiliary decoder verifies that the representation retains the
expression's compositional structure:

```text
canonical typed AST -> typed token/role-path stream -> encoder -> bottleneck z
                                                           |-> view-conditioned e5 alignment
                                                           `-> grammar-constrained AST reconstruction
```

The intended gain is zero-shot conditioning of unseen but structurally familiar compositions.
The component does **not** execute process expressions, infer their factory fingerprints,
generate folder paths, create folders, or replace filing candidate search.

## 2. Identity is outside the learned bottleneck

`z` is a **derived, lossy conditioning representation**. It is never a process identifier, cache
identity, provenance key, or proof of equality. Numerically equal or nearby vectors do not make
two processes identical.

The lossless identity contract remains:

```text
canonical AST bytes
+ exact REGISTRY_VERSION
+ full 64-hex AST digest
+ factory/manifest fingerprint
```

Use the full digest already frozen by `PROTOCOL_process_expression_p1.md`:

```text
sha256((REGISTRY_VERSION + "|" + canonical_identity_string).encode("utf-8"))
```

A deployed process identity also binds that digest to the factory/manifest fingerprint that
realizes the label distribution. Callers must retain the canonical bytes and both fingerprints,
not only a digest. Changing this serialization requires an explicit identity-version migration;
this handoff does not silently introduce a second digest convention.

The current `process_cards.ast_sha()` returns a compact 16-hex convenience key and does not bind
the factory fingerprint. It may remain useful inside current P0 caches, but it is not sufficient
for this handoff's artifacts, split joins, residual lookup, or provenance. New artifacts use the
full digest and fail closed on a compact-only identity.

Grammar-valid synthetic expressions need not have executable factories. Their records therefore
carry a `synthetic_sample_digest` binding the full AST digest to an immutable
`generator_spec_sha256`, and are marked `synthetic_only`. The later corpus manifest records the
generated rows and their digests; it is not an input to its own generation. Synthetic rows cannot
be promoted to deployed process identities without a separately verified factory/manifest
fingerprint.

## 3. Frozen input interface

The encoder accepts a resolved canonical AST data-transfer object, not raw user text and not a
V0–V3 card. Construct it deterministically as:

```text
validated process_cards.Node + pinned registry bytes
    -> resolve every registry default and derived signature field
    -> sort kwargs canonically
    -> resolved canonical AST DTO
```

`Node` itself does not store resolved defaults, `KIND`, or `OUTPUT`; the transform derives them
from the exact pinned registry. The DTO serializes every resolved kwarg, not only `node.kwargs`.
Unknown names, modifiers, kwargs, output types, malformed pins, non-finite numerics, or unsupported
literal shapes are errors.

### 3.1 Reversible typed-AST stream

Version the tokenizer independently of the grammar. Its canonical stream is:

```text
<BOS>
  <NODE> <KIND:atom|apply> <NAME:id> <OUTPUT:type>
    <ARGS>   (<ARG:i> NODE </ARG>)*                         </ARGS>
    <KWARGS> (<KW:key> VALUE </KW>)*                        </KWARGS>
    <MODS>   (<MOD:i> CANONICAL_ASCII_BYTES </MOD>)*        </MODS>
    <PINS>   (<PIN:i> CANONICAL_ASCII_BYTES </PIN>)*        </PINS>
  </NODE>
<EOS>

VALUE :=
    <INT> CANONICAL_ASCII_BYTES </INT>
  | <NUMBER> CANONICAL_ASCII_BYTES </NUMBER>
  | <STRING> UTF8_BYTES </STRING>
  | <LIST> (<ITEM:i> VALUE </ITEM>)* </LIST>
```

`NAME`, `OUTPUT`, kwarg keys, and structural delimiters have finite, versioned vocabulary IDs.
Literal payloads use an explicit 256-byte fallback with start/end delimiters; no hashing,
subword tokenization, or unknown-token replacement is allowed on the reconstruction path.
Numbers use the canonical lexical form produced by the language specification. JSON strings use
exact UTF-8 bytes. Modifiers and pins retain the current grammar's restricted ASCII forms. Thus
stream -> AST -> canonical bytes is reversible.

Derived tokens such as `OUTPUT:type` must agree with the pinned registry during both encoding and
decoding. They do not override it. Atom/operator dual roles are distinguished by `KIND` and
validated against the signature.

### 3.2 Complete role-path positions

Every token carries its complete typed path from the root, not merely `(depth, breadth)`. The root
token has the empty path; `ROOT` is a distinguished token, not an edge step. Non-root path steps
use this authoritative serialized schema:

```text
ARG(index, expected_output_type)
KWARG(registry_kw_id, value_type)
LIST_ITEM(index, element_type)
MOD(index)
PIN(index)
LITERAL_BYTE(index)
```

For example, the first positional child, kwarg `decay`, and first list item are distinct even
when they share a depth and local index. Depth is the path length; breadth is only the index
inside one typed role. Positional encoding must consume the ordered role-step sequence and must
not collapse it to an untyped `(k,j)` pair.

The position-theory note's additive, HRR/convolution, bilinear, and Kronecker options are
compression candidates after this lossless path has been defined. No implementation may claim
that circular convolution is generally invertible after superposition, or use numerical
separability as identity. Whichever candidate is selected must show zero structural aliases on
the generator's finite supported role-path set; otherwise use an explicit role-path lookup or a
less-compressed representation.

### 3.3 Continuous numeric semantics

Exact lexical number tokens remain in the stream for reconstruction. Separately, registry
signatures must mark numeric fields as `continuous` or `discrete` and give every continuous field
a finite semantic domain. A continuous scalar is normalized in that pinned domain, encoded with
fixed sin/cos Fourier frequencies, and passed through a small FiLM module that modulates the
owning node state, following `DESIGN_path_operator.md`.

Frequencies, domains, normalization, and FiLM dimensions are manifest fields. Integers and list
indices remain lexical unless their signature explicitly marks them continuous. Out-of-domain
real values produce an OOD error or a separately registered OOD token; they are never silently
clipped. This two-path design gives smooth parameter geometry without sacrificing exact
reconstruction.

## 4. Encoder, alignment heads, and decoder

### 4.1 Encoder and bottleneck

The first implementation is a small transformer encoder over the typed stream, with role-path
and numeric features added before attention. Its pooled output is `z` with a pinned dimension.
No private corpus text, filing title, judge outcome, or downstream target enters pretraining.

Architectural bounds belong in the manifest: tokenizer/registry versions, maximum tokens/nodes,
layers, heads, hidden width, bottleneck width, dropout, initialization, optimizer, and parameter
count. Exceeding a bound is an explicit new experiment, not an invisible fallback.

### 4.2 View-conditioned e5 alignment

One shared alignment module receives `(z, view_id)` and predicts a vector for each eligible view:

- formal V1;
- formal V2;
- formal V3; and
- the versioned natural-language template view introduced by P2.

For each view `v`, the target is the L2-normalized output of one exact, frozen e5 contract:

```text
t_v = normalize(E5_revision(exact_prefix_v || renderer_v(canonical_AST)))
```

The manifest pins the model repository and immutable revision, pooling rule, dtype, tokenizer
revision, exact prefix including whitespace, renderer/template version, input normalization,
output normalization, and maximum-length policy. Truncation is forbidden unless its policy is
preregistered and recorded per row.

V0 has no semantic card. It is **excluded from alignment pretraining** rather than treated as an
empty-string teacher target. V0 remains downstream conditioning dropout only.

The primary alignment loss is frozen before training (for example cosine distance on normalized
vectors). A CLIP-style expression/template contrastive objective is an optional registered arm,
not a silent replacement for direct alignment.

Downstream integration names exactly one `deployment_view_id`, selected without outer labels, or
a preregistered deterministic mixture with explicit weights. It never averages V1/V2/V3/NL
predictions implicitly. That selected view, renderer/template version, and alignment-head digest
become part of the `NameFunctionCond` adapter receipt.

### 4.3 Reconstruction decoder

The autoregressive decoder receives only:

1. bottleneck `z`; and
2. the already emitted output-token prefix.

It receives no encoder token states, skip connections, source-token copy channel, rendered card,
teacher e5 vector, canonical AST, or factory fingerprint. Teacher forcing may provide the gold
output prefix during training, but evaluation is free-running from `<BOS>`.

At each step a grammar-and-registry state machine masks invalid next tokens. A completed output
must parse, type-check, and canonicalize to an AST. The primary reconstruction event is exact
equality of canonical AST bytes and full AST digest, not surface-string similarity.

This decoder reconstructs a **process expression only**. It has no
`SELECT_EXISTING`, `PROPOSE_NEW`, or `ABSTAIN` actions and must not be described as the filing
path decoder in `DESIGN_filing_path_decoder_handoff.md`.

## 5. NameFunctionCond integration contract

`NameFunctionCond` is currently index-only:

```text
forward(i) = W(name_e5[i]) + residual[i]
```

The learned encoder cannot be wired in by merely replacing the table. Add an explicit external
vector adapter whose public input is a verified, content-bound record:

```text
EncodedProcess {
    e
    full_process_digest
    factory_fingerprint | synthetic_only
    encoder_checkpoint_sha256
    deployment_view_id
    alignment_receipt_sha256
}

forward_external(encoded_process)
    = W(e) + residual[identity_to_row[full deployed identity]]  # registered identity
    = W(e)                                                     # verified unseen identity
```

The alignment output dimension, dtype, normalization, and prefix/view convention must match the
adapter contract. The module performs the full-identity-to-residual lookup internally; callers
cannot supply or pair a raw residual index with an arbitrary vector. The mapping is sealed in the
checkpoint receipt and fails closed on duplicate identities, reordered rows, or a claimed unseen
identity that is already registered. Residual rows map by the full deployed process identity,
never by rendered card or compact `ast_sha`. An unseen expression receives no invented
nearest-neighbour residual.

Before any learned output is used, a parity fixture supplies the existing frozen table vector
through both indexed and external paths and requires bitwise equality where possible, otherwise a
pinned numerical tolerance. This tests adapter wiring independently of encoder approximation.
Existing checkpoints load with the adapter inactive. Activating encoder outputs is a new
downstream arm; `W` is not assumed valid merely because vector dimensions match.

### 5.1 Judge-channel rulings from the R10 bundle (measured, #4069)

The training lane executed R10's judge-channel refinements (prompt-text cards, asymmetric
channel dropout, the slot contract as assertions) and surfaced two questions plus one measured
hazard. Rulings and recordings:

**Ruling — the model-identity line stays.** Haiku and sonnet ran under the *same* frozen
prompt, so pure prompt-text cards would give two judges identical embeddings. That collapse is
not a rendering nicety gone wrong — it is a violation of the recorded doctrine: the
label-generating function is `prompt(Model, PromptText, Harness)` (§1.1 of the registry v0.4
note), and `Model` is an **argument of the function**, not an execution detail. Two judges
sharing `PromptText` but differing in `Model` are different functions and must embed
differently. The prepended identity line is therefore the card-level rendering of the `Model`
argument — ratified, not a workaround. Judge identity carried entirely by residual rows is
rejected for the same reason the channel exists at all: residuals are per-registered-identity
and give an unseen judge nothing, while the card is what lets e5 place it.

**Open question — cross-judge coupling through the shared translation.** *Measured* (#4069):
re-anchoring only the LLM cards moved the untouched graph judge's conditioning by up to ±0.08
through the single shared `W`. The coupling is real and load-bearing, and the function encoder
must make it a **choice**: preserve one shared translation (cross-judge geometry transfers, but
every card edit perturbs every judge) or give judge *families* independent translations/adapters
(isolation, at the cost of transfer). Not ruled here — it needs the function-encoder design
context — but it enters that design as a named decision with a measurement attached, not a
surprise.

**Direction — learned symmetric/asymmetric fusion for filing (owner).** The same experiment
measured prompt-text cards strengthening the directional channel (+0.11–0.13 contribution vs
+0.08 baseline) while weakening the symmetric channel — and the owner's standing observation
sharpens what to do with that: a strong **asymmetric** relation is direct evidence of *where to
file* (filing makes the item a child; the historical misclassification of a symmetric relation
as asymmetric is exactly what made this distinction visible), while symmetric relatedness is
context. The fusion of the two channels should therefore be **learned against the filing
objective** — cross-entropy over filing decisions — and need not be linear: a gating or small
MLP head over the (symmetric, asymmetric) channel scores is in scope. This is a training-lane
experiment; its support obligation is the estimand vocabulary's own split (the symmetric family
`assoc`/`see_also` vs the directional family), so the learned blend is also a first consumer of
`estimand=` as a conditioning input rather than metadata.

**Executed with controls; confirmed in-domain, e5-override out-of-domain** (*measured*,
training lane, on `main`): in-domain (simplewiki) the learned linear blend goes A-dominant
(e5 0.19 / S 0.33 / A 0.61), the gate nearly discards e5 (`we`=0.12) and learns exactly the
hypothesized shape — monotone in A at moderate-to-high S ("trust A when A is strong, fall back
to S otherwise"), with the strongA/weakA stratum gap appearing only in the gate arm. Neither
learned arm beat the fixed μ-max baseline, *by construction*: `ELEM` sits outside the ruled
(S, A) parameterization. Out-of-domain (Pearltrees) the learned blend converges to e5-dominant
(0.88/0.10/0.06) and ties e5-only, while the in-domain-tuned **fixed** blend transported
backwards by 2× — fixed blend weights are not transportable across corpora; a blend learned
against filing CE rediscovers the right regime per domain.

**Direction, second iteration (owner): the relationship is still open, and the next arms must
separate two failure modes.** The owner's reading of the OOD result — *we have not found the
right relationship yet, and it need not be linear* — stands, with one diagnostic discipline
attached: a blend can fail OOD because its **functional form** is wrong, or because its
**inputs carry no signal** there, and no functional form repairs signal-free inputs. So the
next iteration, in order: (1) report per-channel *standalone* OOD performance first — if S and
A alone sit at chance on Pearltrees, the finding is about the μ estimators (corpus-bound walk
structure, the standing honest-NULL memory), not the blend; (2) **extend the parameterization
to (S, A, ELEM)** — granted, and not merely as scope creep: filing *is* predominantly
`element_of` placement, so the three-way blend is exactly the estimand vocabulary's family
split (symmetric family / directional family / membership terminator), completing the
parameterization the R7 enumeration implies — the μ-max baseline is unbeatable from inside
(S, A) because it sees a family the learned arms cannot; (3) a **domain-conditioned gate** arm:
feed the gate cheap substrate statistics (density, branching, depth distribution) so one
learned function can *transfer by conditioning* instead of retraining per corpus — the learned
gate rediscovering each domain's regime is the mechanism-level win already in hand; a gate that
knows *which regime it is in* is the candidate for the relationship that generalizes.

**The blend may be hop-dependent, and the S/A ratio may be a distance proxy (owner).** The
mechanism, observed on Wikipedia: semantic drift over parent hops makes `subcategory`
relationships *read as symmetric* at distance — the directional signal decays toward general
relatedness faster than the symmetric one, so what arrives as "related" may be a drifted
ancestor. Hop distance is unobservable for a new node, but the **ratio of symmetric to
asymmetric signal** is observable and may indicate it — which would give the gate's learned
shape a mechanistic reading (discount A by inferred distance: filing wants a *near* parent, not
a distant ancestor). In-domain this is directly testable because hop distance is ground truth
on the wiki graph: measure A/S as a function of true hop distance before trusting the ratio as
a proxy anywhere. Two confounds carry over from the standing record and must be reported
alongside: in a **non-DAG** substrate multiple parents blur the hop-distance interpretation
(the R1 rider: drift is expected larger under `full_dag` than `principal_tree`), and **typos
masquerade as semantic drift** (title noise splits parents and inflates apparent drift — the
registry note's §7 `decay` estimand fork, observed vs canonicalized titles, is exactly this
confound and remains undecided; its eventual ruling feeds this experiment's interpretation, and
no holdout detects the typo share because the holdout carries the same typos).

## 6. Finite deterministic synthetic corpus

"Synthetic" does not mean unlimited or unconstrained. First freeze a checked-in generator
specification and its `generator_spec_sha256`; after generation, a separate content-addressed
corpus manifest records the outputs. The specification freezes:

- grammar, registry, tokenizer, renderer, and NL-template versions;
- finite allowed operators, atom names, modifiers, placeholder pins, and kwarg domains;
- maximum AST depth, nodes, positional arity, list length, string bytes, and token count;
- finite numeric grids/distributions inside each registered semantic domain;
- target sample counts, enumeration/sampling algorithm, seed derivation, duplicate policy, and
  rejection counters; and
- the exact train/dev/test structural partition.

Pins, strings, and manifests use synthetic placeholders from a committed allowlist. The generator
must not sample repository paths, bookmark titles, private names, real manifest IDs, environment
variables, logs, or user data.

Random choices are counter-based or seeded from `generator_spec_sha256` plus sample index. Rows
are sorted by full AST digest and byte-identical across machines. Duplicate canonical ASTs are
removed before splitting. Only then is the output corpus manifest hashed.

### Structural LOCO partition

Random expression-level splitting is insufficient. Define a structural template by replacing
leaf identities and literal payloads with typed slots while preserving operators, nesting,
argument roles, kwarg keys, modifiers, and list shape. Entire templates and registered
composition families are assigned to one split only. Test contains preregistered held
operator-composition and kwarg/list-shape families; no canonical AST or structural template may
cross train/dev/test.

The split verifier recomputes templates and fails closed on overlap. Downstream process and
cross-corpus LOCO sets remain separate from this synthetic test and are not inspected during
pretraining selection.

### Sealed synthetic-test transaction

The training worker receives only the train/dev projection: no test rows, test teacher vectors,
test metrics, or test-cache paths. The frozen generator specification may name held structural
families, but their realized rows and e5 targets are materialized or released only by a separate
test transaction after the checkpoint, adapter, and evaluation code hashes are sealed.

The transaction records its first read. Once any synthetic-test row, teacher target, or metric has
been revealed, that test is burned for model or architecture selection. A decoder, objective,
tokenizer, position arm, threshold, or checkpoint change after that reveal requires newly frozen
held structural families; the same test may remain only a clearly labeled regression report.

## 7. Objectives and controls

Freeze loss definitions and weights in the run manifest:

```text
L = lambda_recon * grammar-masked token CE
  + lambda_align * mean_v alignment_loss(A(z,v), t_v)
  + optional registered auxiliary losses
```

Weights and checkpoints are selected on synthetic dev only. The synthetic test and downstream
outer sets are single-use reports.

Required controls/ablations:

1. direct frozen formal-card e5;
2. direct frozen NL-template e5;
3. P3 deterministic composition;
4. additive bag of AST nodes;
5. flat process token and unconditional conditioning;
6. shuffled e5 targets within compatible views;
7. reconstruction-only and alignment-only training;
8. untyped depth/breadth positions versus complete role paths; and
9. lexical-only numerics versus lexical plus fixed-Fourier/FiLM numerics.

The shuffled control must break AST-to-teacher correspondence without changing view counts or
length strata. It is not a downstream label permutation.

## 8. Evaluation and activation gates

### Engineering gates

These are exact and blocking:

- canonical AST -> token stream -> AST round-trips every registry example and generated row;
- generator and split manifests reproduce byte-for-byte;
- no full-digest, canonical-AST, or structural-template overlap across splits;
- zero grammar-invalid free-running decodes (grammar masking should make invalid syntax
  unreachable);
- all decoded outputs revalidate under the pinned registry;
- adapter parity passes; and
- artifact load rejects any digest, version, shape, dtype, normalization, prefix, cap, or
  factory-fingerprint mismatch.

### Scientific gates

Before the first training run, a preregistration must freeze numeric reconstruction and alignment
thresholds, training seeds, checkpoint selection, and the downstream decision. Report exact AST
match, field/role accuracy, invalid/OOD rates, e5 cosine-error distribution by view, nearest
teacher retrieval, and results by held structural family.

The P3 multi-process and cross-corpus LOCO evidence used to declare a plateau is **design evidence
and already exposed**. Reusing it for the learned encoder is necessarily exploratory/adaptive.
It may be reported for continuity but cannot serve as a fresh outer comparison.

Only after the encoder and checkpoint are frozen may a newly reserved post-activation process
family, corpus, or prospective cohort be opened. Its primary comparison is the learned encoder
against P3 deterministic composition and the strongest no-new-architecture baseline (formal or
NL-template e5, selected without that new outer set). If no genuinely new evaluation population
exists, the downstream result remains exploratory and cannot claim generalization. The learned
path must satisfy the preregistered superiority target and the existing e5
safety/noninferiority constraint. Multiple ablations are descriptive unless the preregistration
supplies multiplicity handling.

Failure of an engineering gate stops the run. Failure of a synthetic scientific gate blocks
downstream evaluation. Failure of the downstream gate rolls back to deterministic P3 or frozen
e5; it does not trigger tuning on the outer result. Success authorizes an implementation
candidate, not automatic deployment.

## 9. Artifacts and fail-closed provenance

Each run emits a content-addressed bundle containing at least:

- canonical generator specification, `generator_spec_sha256`, and corpus/split manifests;
- full AST and synthetic-sample digests for every row;
- registry, canonicalizer, tokenizer, renderer, and template digests;
- frozen e5 model/revision, tokenizer, prefix, pooling, and normalization contract;
- model/loss/optimizer config, seeds, dependency lock, hardware and peak-memory record;
- checkpoint and adapter hashes;
- train/dev worker-input projection, synthetic-dev selection ledger, sealed-test first-read
  receipt, and synthetic-test report;
- downstream preregistration and, if authorized, the one-shot LOCO report; and
- machine-readable pass/fail records for every gate.

Loaders recompute hashes from bytes and reject unknown/missing fields. A filename, branch name,
compact hash, or self-reported manifest field is not provenance. Any intentional contract change
creates a new bundle and new evaluation; it never mutates a sealed result.

## 10. Privacy and resource envelope

Synthetic pretraining uses only the public grammar/registry and synthetic allowlisted literals.
No private filing data is needed. Real downstream rows retain the privacy and external-processing
rules of their source corpus; neither expressions nor embeddings erase those restrictions.
Private AST literals or derived representations may not be sent to an external model/API without
separate authorization.

The first implementation targets one consumer GPU or CPU execution with explicit caps and
streamed generation; it must not require an in-memory "unlimited" corpus. Record model parameters,
corpus bytes, wall time, peak host RAM, and peak device memory. An out-of-memory event is a failed
run, not permission to shrink data/model or change precision post hoc.

## 11. Delegable implementation sequence

1. **Contract fixtures:** add full-digest/process-identity helpers and golden AST/token/role-path
   vectors without changing current P0 behavior. *Landed:* `process_identity.py`,
   `process_expression_contract.py`, and the frozen golden bundle. The contract module is the
   fixture authority only; the versioned vocabulary IDs and the 256-byte literal fallback belong
   to step 2, which must reproduce those frozen structures exactly.

   **Current bundle: `PROCESS_EXPRESSION_GOLDEN_v4.json` (contract `pec-v4`, registry `v0.5`).**
   Step 2 reproduces this one. `PROCESS_EXPRESSION_GOLDEN_v1.json` and
   `PROCESS_EXPRESSION_GOLDEN_v2.json` are retained as audit-only provenance and are rejected by
   the current loader; reproducing either would fail closed and would miss the v0.4 coverage
   (substrate atoms, positional numeric literals, `mu=` sub-expressions, `estimand=`/`impl=`).
   The authoritative pointer is `process_expression_contract.CURRENT_GOLDEN_BUNDLE`, and this
   line moves with it — see `DESIGN_process_expression_generator.md` §0 for the supersession
   procedure.
2. **Tokenizer and generator:** implement reversible typed serialization, finite deterministic
   generation, structural templates, LOCO splits, and overlap/privacy validators.
3. **Model core:** implement the bounded encoder, complete role-path positions, lexical numeric
   path, fixed-Fourier/FiLM semantic path, and bottleneck-only constrained decoder.
4. **Teacher cache:** materialize V1/V2/V3/NL targets under the exact frozen e5 contract; prove V0
   never enters it.
5. **Adapter:** add the opt-in external-vector path to `NameFunctionCond`, identity-bound residual
   lookup, migration/parity tests, and old-checkpoint compatibility.
6. **Pretrain and freeze:** select on synthetic dev, seal once, then evaluate synthetic test.
7. **Downstream study:** only if all prior gates and the original activation gates pass, freeze
   a new post-activation evaluation population; execute it once. Reuse of the plateau evidence is
   descriptive only.

Minimum tests include grammar-example round trips, atom/operator dual roles, nested args/kwargs,
lists, UTF-8/escaped strings, pins, numeric boundary/OOD cases, role-path non-aliasing, generator
reproduction, structural split rejection, decoder no-skip enforcement, V0 teacher-cache
rejection, shuffled-target control, adapter parity, unseen-residual zero behavior, artifact
tampering, and old-checkpoint loading.
