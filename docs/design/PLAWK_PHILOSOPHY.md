<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 John William Creighton (s243a)
-->

# plawk — Philosophy

*Project: UnifyWeaver — Hybrid WAM/LLVM Target*
*Author: John Creighton*
*Status: Early Design / Prototype Phase*

> **Companion docs:** [Specification](PLAWK_SPECIFICATION.md) ·
> [Implementation Plan](PLAWK_IMPLEMENTATION_PLAN.md) ·
> [Submodule README](../../examples/plawk/README.md)

---

## 1. Core motivation

Unix shell tools — particularly `awk` and `bash` — have endured for decades
because they offer a simple, composable model: read records from a stream,
apply a pattern-action rule, write results. This model is powerful precisely
because it is minimal.

The goal of this DSL is to preserve that ergonomic simplicity while replacing
the interpreted, string-centric execution model with something rigorous:
compiled Prolog, executed via UnifyWeaver's hybrid WAM/LLVM backend, operating
on binary/typed data structures.

The guiding principle is: **familiar surface, principled interior**.

From the outside, a user writes something that looks and feels like `awk`. Under
the hood, the program is a deterministic Prolog predicate, compiled through
UnifyWeaver to native LLVM code, using binary record structures rather than text
strings.

### 1.1 The performance argument, stated precisely

awk cannot compete with Rust/Haskell/F#/Go for real data-processing work (e.g.
the graph algorithms UnifyWeaver already compiles) for **three** reasons, not one:

1. **Per-field string cost.** Every field is a heap string; `$3 + 1` reparses
   text to a number on every access.
2. **String-keyed arrays.** awk's associative arrays are string-keyed hash
   tables.
3. **Inter-stage serialization.** A pipeline `awk | sort | awk` re-serializes
   every record to text at each `|`.

The WAM addresses (1) and (2) for free: integers stay tagged integers, compound
terms are structure-shared, and keys are terms rather than strings. The LLVM
target's existing machinery (`musttail` TCO, BFS worklist, memo tables) already
lowers the hot patterns to unboxed, iterative native code.

**(3) is the real differentiator** — larger than "binary vs string." Because the
DSL and UnifyWeaver's graph algorithms compile to the *same* engine, a pipeline
like *parse records → build graph → reachability → emit* runs in **one native
binary with no text boundary between stages**. A Unix pipeline cannot do this; a
hand-written Rust program can, but you would write it by hand. The niche is:
awk-level ergonomics, no serialization boundaries, and the ability to drop into
graph reachability mid-stream.

**Honest scope.** This will not beat hand-tuned Rust on raw single-pass
throughput in the near term — boxed WAM cells and residual choicepoints cost
real time. Closing that gap depends on the determinism story below actually
landing in codegen (see the Specification and the reconciliation findings in the
Implementation Plan).

---

## 2. Philosophy of determinism

Standard Prolog is relational and supports backtracking. This is powerful for
reasoning but is an obstacle when compiling to efficient, LLVM-native control
flow. The DSL adopts a **determinism-first stance**:

- Each handler predicate has a declared primary *mode* (`+` inputs, `-` outputs).
- Predicates intended to succeed exactly once are written to do so and leave no
  choicepoints.
- The compiler uses mode information to generate straighter, less-backtracking
  code where possible.
- Backtracking is not removed from Prolog; it is *contained and annotated*.

This mirrors the relationship between Haskell's `IO` monad and pure functions:
backtracking is possible, but it is the exception and must be made explicit.

> **Grounding (see Implementation Plan § Codebase reconciliation).** Mode
> declarations are **already consumed** by UnifyWeaver's codegen: `:- mode`
> flows through `demand_analysis` and `binding_state_analysis` into the WAM
> pipeline, which picks deterministic builtin variants and indexes on input
> positions. A `:- det` *declaration*, by contrast, is **not** consumed today;
> determinism is currently achieved structurally (cut, if-then-else, switch
> indexing, `musttail`). The DSL therefore leans on **modes + structural
> determinism** first, and treats a `det`-directive-driven choicepoint-elision
> pass as a later, measurement-justified addition.

---

## 3. Philosophy of stream abstraction

`awk` hardcodes its stream model: one record per line, fields split by `FS`,
output to stdout. This DSL abstracts that model into three decoupled roles:

- A **Reader** abstracts "how to obtain the next item from an input stream." It
  may read lines, binary records, parsed Prolog terms, or data from a DCG grammar.
- A **Writer** abstracts "how to emit a result." It may write to stdout, a named
  pipe, a binary file, or another stream held in `State`.
- The **Handler** (`{}` body) dispatches between writers or passes control to a
  next stage.

A program written in the DSL is therefore not tied to text I/O. The same
pattern-action logic can process binary network packets, structured log records,
or Prolog terms.

---

## 4. Philosophy of staged development

Complexity is introduced in layers, each stabilized before the next is added:

1. **Core Prolog layer** — `process_all/4`, `reader`, `handler`, `writer`,
   `item_field/3`, mode declarations; validated in plain SWI-Prolog.
2. **UnifyWeaver compilation** — transpile the core to LLVM via the hybrid WAM
   target.
3. **AWK syntactic sugar** — a front-end parser emits Prolog core from awk-like
   `pattern { body }` syntax.
4. **Bash compatibility** — file descriptors, redirection, subprocesses, named
   pipes (much of which already exists as LLVM builtins; see Specification §10).

The DSL syntax is sugar; **the Prolog core is the specification.**

---

## 5. Relationship to the existing AWK target

The name **`plawk`** = "**Prolog awk**": an awk-like surface compiled through
Prolog → WAM → LLVM. It also disambiguates from the existing **AWK target**
(`docs/AWK_TARGET_STATUS.md`, `src/unifyweaver/targets/awk_target.pl`). That
target compiles Prolog *into* awk scripts — the opposite data-flow direction
from `plawk`, which compiles an awk-like surface *down to* Prolog→LLVM. They are
complementary, not competing: one emits awk for portability, the other consumes
an awk-like surface for native performance.

---

## 6. Philosophy of layering: templates below, Prolog above

*Forward-looking; recorded so the direction is not re-derived. Nothing here asks for a
rewrite.*

### 6.1 The rule

**Low-level, mechanical emission belongs in templates. Abstraction and decision-making
belong in Prolog.** This is the same split the other transpiler targets use, and the
reason is the same: Prolog is the portable part. A rule expressed as Prolog clauses can be
re-hosted; a rule expressed as a hand-written string `format/3` inside one of twenty-nine
driver clauses cannot.

The practical test for which layer a thing belongs in: *does it decide, or does it
render?* Choosing a key space from a table's kind decides — Prolog. Turning
`(table, key, delta)` into a `call i64 @wam_assoc_i64_inc(...)` line renders — a template.

### 6.2 Why portability is the point, not tidiness

A future re-host of the compiler — a Go implementation for faster builds is the concrete
candidate — has to carry every decision the compiler makes. Decisions living in Prolog
clauses port as data and logic. Decisions living in the *arrangement* of format strings
across near-duplicate emitters have to be rediscovered by reading them all, which is
precisely how this codebase's recurring defect gets in.

The compile-time motivation is real and measured, not hypothetical: **~5.2 s per program**
for a 20-program corpus, dominated by loading the 21 761-line codegen module before any
work starts, plus `clang`. That cost is why a faster host is attractive, and it is also
why the split matters — a re-host is only cheap if the logic is already separable from the
rendering.

### 6.3 Structural templates: `pattern_stache`

Two template dialects exist, and they are complementary rather than competing:

| | `.mustache` (`template_system.pl`) | `.stache` (`pattern_stache`) |
|---|---|---|
| `{{case v}}` matches | literal text, string-compared | a **Prolog term**, matched by unification |
| variables bound by a match | no such concept | visible to the case body |
| fit | fixed boilerplate with holes | dispatch on the **shape** of a term |

plawk's emitters dispatch on the shape of a term constantly — `var(Name)`, `field(N)`,
`special('NF')`, `assoc(var(A), string(K))`, `concat(Parts)` — which is exactly what
`.stache` is for, and exactly where the duplication lives. So the natural target is *one*
`{{match}}` over the print-field term with one `{{case}}` per field kind, rendered by every
driver, instead of a per-driver clause set that can silently differ.

**Not adopted yet, deliberately.** `pattern_stache` lives on an unmerged branch
(`claude/pattern-stache-dispatcher-prototype-dizes2`,
[`SPEC_pattern_stache.md`](SPEC_pattern_stache.md)) and its spec says it implements *only
what two witnessed consumers needed*. Making plawk a third consumer is a coordination
decision, not a refactor to slip into an unrelated PR — see §6.5.

### 6.4 What the layering would have prevented, concretely

The bare-scalar `END { print NAME }` emitter existed twice, once per END walker. The two
drifted: only one learned string/strnum slots and unset-renders-empty. The same program
text therefore behaved differently depending on whether it happened to also touch an assoc
table — printing `0` where gawk prints empty, and declining where gawk prints the text.

Neither was a missing feature. Both behaviours sat ten lines away in the other copy.
**Deleting the duplicate closed both gaps without implementing either.** That is the
argument in one example: a duplicate does not merely fail to gain behaviour, it *un-does
shipped behaviour* for a subset of programs, silently, and no per-route test suite can see
it because each suite exercises the route it owns.

The measurement that makes this actionable: the END print-field vocabulary is six kinds
(`assoc`, `concat`, `field`, `special`, `string`, `var`) across three walkers — eighteen
cells, about eleven filled. **Every END-shaped gap this campaign has closed was a missing
cell.** A coverage matrix with holes is a defect generator; the holes should be explicit
capability declarations, so a driver that cannot support a kind *declines* rather than
accidentally lacking a clause.

### 6.5 Note for whoever is working on `pattern_stache`

plawk is a plausible **third consumer**, and this is a heads-up rather than a request.
What plawk would need beyond the two witnessed consumers:

- **Whitespace *control*, not byte-identity.** An earlier draft of this bullet demanded
  byte-identical rendering, because plawk's regression tool is a byte-level golden-IR diff
  (a corpus built with `--keep-ll`, compared with `cmp`). That overstated it — the
  constraint belongs to plawk's cheapest verification instrument, not to the dialect:
  clang is indifferent to formatting outside string constants. What the dialect must
  actually provide is *controllable* whitespace around `{{match}}`/`{{case}}`, so an
  author who chooses a byte-faithful migration can have one. Where natural template
  formatting differs cosmetically, plawk has two sanctioned outs, so do not contort a
  template to reproduce a legacy emitter's accidental formatting — fossilizing quirks is
  worse than one verified re-baseline:
  1. **Normalize the diff**: round-trip both sides through `llvm-as | llvm-dis` before
     comparing (verified working on the corpus; formatting-only perturbations vanish).
     The reason naive text normalization is wrong and the round-trip is right: whitespace
     inside `c"..."` string-constant globals is *data* — OFS separators and printf format
     strings live there — and only a parser-aware normalizer preserves it. Renames and
     reorders still show; they are neutral but visible, and fall to the second out.
  2. **Re-baseline deliberately**: verify behaviourally (corpus binaries on fixed inputs,
     the full suite sweep, gawk probes) and re-capture the golden baseline once. Precedent:
     the unset-scalar shared-globals change, recorded in the campaign handoff, which could
     not be byte-identical and was verified on program output instead.
- **Emission of *lists* of lines**, sometimes conditionally (a string slot emits five
  instructions, a numeric slot a different set), with a caller-supplied index threaded
  into every generated SSA name.
- **Selection only, as specified.** The slot-kind and key-space decisions must stay in
  Prolog; the template should receive an already-resolved kind. This matches the spec's
  "it selects; it does not prove" and is the right division anyway per §6.1.

If any of that argues for a dialect change, plawk is not urgent — the duplication is being
removed incrementally in plain Prolog first, which is a prerequisite either way: a template
can only replace emitters that already agree.

**Assessment outcome (recorded by the `pattern_stache` lane in
`prototypes/mu_cosine/RECORD_prospective_consumer_plawk.md`; authoritative details there).**
All three requirements were answered without any dialect change, and two answers carry
obligations back onto plawk's side of a future migration:

- **Requirement 3 is resolved, measured.** Only `{{match}}` gets a free line (preamble
  discard); `{{case}}` and `{{/match}}` are literal boundaries, so
  `{{match k}}\n{{case a}}X\n{{/match}}` renders exactly `"X\n"`. A byte-faithful
  migration is therefore possible, with a residue of one shared line per case — layout
  cost, not gymnastics. No whitespace-exactness machinery was added; the `c"..."` caveat
  above is recorded there as a constraint on any *future* whitespace feature (marker-local
  only, never content-scanning). Their `marker_adjacency` test unit answers this directly.
- **Requirement 2 is expressible.** `{{Key}}` substitutes mid-token with no delimiter
  requirement, so `%end_field_{{I}}_len` works as written. Only a list whose *length* is
  decided inside the template is excluded — which requirement 1 rules out anyway, since
  membership per kind is fixed by the planner.
- **Requirement 1 is structurally enforced, with ONE identified leak path that becomes
  plawk's obligation.** There is no expression sublanguage, so a plan-time set-membership
  test cannot be written in a template, and an unresolved dispatch value is an error, not a
  wildcard. But the dialect permits *priority*: case order is load-bearing under their
  overlap rules, so `{{case interned(K)}}` above `{{case position(K)}}` would encode a
  key-space resolution policy in file order — indistinguishable, by shape, from legitimate
  specific-before-general refinement. **plawk's counter-constraint, for whoever migrates:**
  the planner must hand templates terms whose decision is already a distinguishable tag —
  today's planned vocabulary satisfies this (`lookup_int` / `lookupn` / `strlit` / … are
  functor-distinct) — and no plawk template may contain two cases whose patterns overlap on
  terms plawk emits, *except* pure refinement on a planner-resolved tag value (e.g.
  `lookup_int(T, N, str)` before `lookup_int(T, N, K)`, which branches on a decided `Kind`
  rather than choosing an interpretation). If writing a template ever seems to require
  ordering two cases to pick between interpretations of the same data, the decision has
  leaked out of the planner — fix the planner, not the case order.

### 6.6 Sequencing

1. **Now, in Prolog:** collapse duplicated emitters onto one predicate each, proving
   byte-identity with the golden-IR corpus. Each collapse tends to close real gaps for free
   (§6.4), so this pays its own way and does not depend on any template decision.
2. **Then:** make the coverage matrix explicit — capabilities as data, so a hole is a
   declared decline.
3. **Only then, and only where it helps:** move the *rendering* of agreed emitters into
   `.stache` templates. Rendering that no longer varies between callers is a template;
   rendering that still varies is a decision that has not been factored yet.
