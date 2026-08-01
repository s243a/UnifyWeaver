# Project Design Philosophy

*Working principles that keep re-deriving themselves across this project's design work. Each one
is stated with the instance that produced it, because a principle without a case is a slogan.
`CONTRIBUTING.md` covers mechanics — licensing, pull requests, code style, testing. This document
covers how design decisions get made and recorded.*

**Status: proposed.** It generalizes from observed practice rather than legislating, and it
touches conventions other lanes rely on, so it is the owner's to ratify, amend, or reject.

## 1. Measure before deciding

The recurring experience is that a measurement changes the decision, not merely its confidence.
Several examples from one design cycle:

| measurement | what it changed |
|---|---|
| 13 of 15 `lineage(...)` forms become unenumerable under a registry split | moved the enumeration gate from "the ruling" to "the registry ships" |
| 24 distinct `{{case}}` values, 22 of them plain atoms | replaced an assumed incompatibility with a two-row migration checklist |
| 307 `.mustache` files, only 13 using `{{match}}` | killed a proposed load-time compatibility shim, whose only benefit was a single code path that would not have been achieved |
| largest real template library has 5 cases | removed the justification for a conversion tool entirely |
| SWI's reader on `'wam-fsharp'` versus `wam-fsharp` | corrected a claim that hyphens were incompatible; they are fine when quoted |

None of these were expensive. All of them were cheaper than the work they prevented.

The corollary matters as much: **a null result from a search is not a finding.** A grep that
misses because its pattern was wrong looks exactly like a grep that misses because the thing is
absent. Report "my search did not hit it," not "it does not exist."

## 2. Prefer an accurate omission to a plausible completion

An incomplete description advertises its incompleteness. A wrong one does not.

The case: a proposal to express a graph judge as `blend(hops(gamma=0.9), lca_frac)`, replacing an
expression that omitted two parameters. But the judge computes `max(floor, gamma^hops · lca_frac)`
— a product under a floor, not a weighted sum. The proposal would have swapped a visible omission
for an invisible inaccuracy, which is worse: *a wrong operator looks complete.*

The same reasoning declined a pattern digest in the vNext frontend. Canonical bytes would have
required fixing a wire format that is genuinely undecided; a structural comparison function does
the job with no commitment. Better to leave a gap that is labelled than to fill it with a guess
that isn't.

## 3. Do not build ahead of the consumer

Infrastructure designed before the thing that uses it acquires an abstraction nobody asked for
and nobody can remove.

Three instances of the same call, all resolved the same way: a structural-matching specification
deferred until a prototype produces real patterns; a type-class mechanism deferred behind the
registry version that actually gates downstream work; a conversion tool deferred because no file
in the repository is large enough to need it.

In each case what gets recorded is the **threshold** rather than a plan — a condition to check,
not a date to wait for. "Revisit when a template library grows large enough that hand-copying its
cases is error-prone" is actionable years later; "build the converter in Q3" is not.

## 4. Put the decision in the artifact, not in the configuration

A guarantee that depends on a flag being set correctly is a discipline. A guarantee that depends
on the shape of the artifact is an invariant.

- A file extension selects the parser, so no existing file can reach a new one — where a
  `structural(true)` option could be inherited or defaulted and silently change how 67 targets
  parse.
- Constraint facts live in a sealed, versioned registry rather than an open Prolog database,
  because anything that can `assert/1` could otherwise void the guarantee invisibly.
- A migration tool that rewrites files produces a reviewable diff; a load-time shim performing
  the same rewrite is invisible at every load.

The test to apply: *if someone sets every option wrongly, does the property still hold?*

## 5. Fail closed, and never default into deployment

Absence means unresolved, not "pick something sensible."

An unknown dialect version is an error rather than a best-effort parse. An unknown implementation
constraint returns no candidates rather than a default. A state type proves its invariant in its
constructor rather than documenting it. Elaboration refuses a term it cannot fully check rather
than passing it along.

The distinction worth keeping sharp: ordinary defaults are fine — a registered parameter default
is a convenience. What is forbidden is **defaulting into something identity-bearing or
deployable**, where a silent choice becomes permanent.

## 6. Order-dependence is a hazard wherever it appears

The same bug recurs in unrelated places, and it is always cheap to prevent and expensive to
discover: clause order in a rule set, case order in a template file, goal order in a residual
store, key order in a caller's mapping, source order in a template resolution chain.

Wherever order is not meaningful, canonicalize — sort, deduplicate, or check for overlap. Where
it *is* meaningful, say so explicitly and version it. What must not happen is inheriting
significance by accident, which is what happens when first-match-wins meets patterns that can
overlap.

## 7. The record is the memory

Conversations do not survive. Documents do.

A design discussion about tree-position encodings — the outer/Kronecker feature, circular
convolution as one fixed projection of it, the bounded quadratic-form analogy — was recoverable
months later only because it had been written into a design note. The conversation itself was
gone, and the agent that had participated in it could not reconstruct it.

So: when a decision is reached, write it down in the same pass. Not the conclusion alone but the
reasoning, the alternatives rejected, and *why* they were rejected — a rejected option with no
recorded reason gets re-proposed. Record open questions as open, so silence is not later mistaken
for consensus.

## 8. Distinguish what was verified from what was inferred

Every claim in a design document is one of: measured, cited, or assumed. Saying which costs a
word and saves an argument.

This project already does it in places — the generator specification marks quantities *measured*
versus *provisional*, and requires that measurements land as tests rather than as prose. The
principle generalizes: quote the code you are describing, cite the file and line, and mark an
inference as an inference.

The reciprocal obligation is to correct plainly when a claim turns out wrong, without
ceremony — state the correction, give the evidence, move on.

## 9. Lead with purpose; the contract comes second

A design document that opens with authorities, prohibitions, and ablation protocol is readable
only by someone who already knows what the artifact is for. The reader who needs the document —
future maintainers, other lanes, models asked to work from it, and the owner verifying it months
later — does not share the author's context.

The instance: a position-encoding design was handed to an LLM tutor for a step-by-step
walkthrough, and the tutor missed the point of **all five** of its own quiz questions — not
randomly, but in two clusters: it never learned what the encoder was *for* (the document nowhere
stated purpose, input, or output), and it collapsed a three-level distinction (identity /
coordinates / features) that the document enforced rule-by-rule but never laid out in one frame.
The consolidated ruling comment one document earlier had failed the same way for the same
reader, and was fixed the same way — a plain-terms block prepended after the fact. The
pattern-stache philosophy note avoided the failure from the start by opening with "The
question."

So: design documents open with **purpose, input, output, and the one table that prevents the
predictable conflation** — before the contract. Precision without comprehensibility fails the
record's main purpose, because the owner is the verification layer. A useful acceptance test: a
fresh reader (or agent) with no conversation history should be able to explain the design back
correctly from the document alone.

## Non-goals

This document does not prescribe architecture, and it does not override any feature-scoped
`*_PHILOSOPHY.md` in this directory. Where a specific design note reaches a different conclusion
for a stated reason, the specific note wins — these are defaults for when nothing more specific
applies.
