# Record: plawk as a prospective consumer of `pattern_stache`

**Status: record and assessment. No dialect change, no implementation, no integration.**

plawk (`examples/plawk`, an awk → Prolog → WAM/LLVM compiler) is a **prospective** third consumer
of the `pattern_stache` dialect. **Zero code references exist today** — nothing imports the
dialect, no `.stache` file exists under `examples/plawk`, and no dialect change has been
requested. plawk is collapsing its duplicated Prolog emitters first, which is a prerequisite
either way. This file exists so that whenever the dialect next grows, the implementer starts from
a written record instead of rediscovering these constraints — the same role the cowalk delta
plays in [`REPORT_pattern_stache_interpretation.md`](REPORT_pattern_stache_interpretation.md).

Requirements are recorded in **plawk's own corrected ordering, hardest first**. An earlier
version of this record led with byte-identical rendering; that requirement was withdrawn by its
author as an overstatement and is corrected below, along with what this lane got wrong on the
strength of it.

**Provenance.** `docs/design/PLAWK_PHILOSOPHY.md` §6/§6.5 is not reachable from this branch or
from main — the file is here, but its headings stop at "## 5. Relationship to the existing AWK
target" and it contains no occurrence of `pattern_stache`. The §6 text, including the §6.5
correction, is on `claude/plawk-llvm-wam-hybrid-p9ujut` and in PR #4154. The requirements below
are recorded **as relayed**, not as read.

---

## Verdicts

| # | prospective requirement | verdict |
|---|---|---|
| 1 | selection only — resolution stays in Prolog | **the dialect's own boundary, and structurally enforced**, with exactly one leak path: case *priority* |
| 2 | conditional lists of lines with a threaded index | **expressible in today's surface. Not a dialect gap** |
| 3 | whitespace *control* — explicitly not byte-identity | **controllable, measured**; the residue is a layout cost, and **nothing is to be built on plawk's account** |

Measured against `src/unifyweaver/core/pattern_stache.pl` at this commit and pinned by
[`tests/core/test_pattern_stache_whitespace.pl`](../../tests/core/test_pattern_stache_whitespace.pl)
(33 tests). Before that suite, **no test pinned line or byte fidelity at all** — including the
three properties the SPEC's Whitespace section already declares normative — because both
witnessed v1 consumers normalized per rendering with `normalize_space/2`.

---

## Requirement 1 — selection only

*Slot-kind and key-space decisions stay in Prolog; the template receives an already-resolved kind.
plawk has two array key spaces (interned text vs raw integer position) whose resolution depends on
a plan-time set — a decision, not a dispatch, and it must not leak into a template.*

This is the dialect's own boundary, stated in three places:

- **Purpose**: *"It selects; it does not prove … Structural matching improves dispatch, nothing
  else."*
- **Exclusions**, `guards`: not expressible; *"the one guard-shaped question — groundness — is
  owned by the caller's discharge ordering."*
- **Exclusions**, `pattern arithmetic`: not expressible, revisit condition *"never expected."*

The useful part is that the boundary is **structurally enforced, not conventional**. A plan-time
set-membership test cannot be written in a template because there is no expression sublanguage to
write it in — no guards, no arithmetic, no predicate calls, and a case value is a first-order
linear term or a load error. And `nonground_dispatch` is the second half: a dict value that is
still unresolved is an **error**, not a wildcard match, so a template cannot even accidentally
dispatch on something the planner has not decided yet. A key space that resolves at plan time
arrives as a ground atom or it does not arrive.

### The one leak path, stated because "must not leak" deserves a precise answer

The dialect forbids computation but permits **priority**, and priority is a decision. Case order
is semantically load-bearing in two of the three overlap rows:

- *unifiable, neither subsumes* — genuinely order-dependent, first-match-wins. v1 **warns**,
  naming both cases; never silent.
- *later subsumes earlier* — specific-before-general refinement, allowed and **silent** by
  design, because a witnessed consumer needed it (`substrate(pearltrees)` above `substrate(C)`).

So a template author who wrote `{{case interned(K)}}` above `{{case position(K)}}` would be
expressing a resolution *policy* in file order. Nothing in the engine can distinguish that from
legitimate refinement — the trichotomy is about term shape, and both readings have the same
shape. The mitigation is that the decision has to be a *ranking of shapes*, which is far weaker
than a set-membership test and cannot consult the plan-time set at all; and the load-time warning
covers the non-refinement case. Recorded so that "it must not leak" has a measured answer rather
than a reassurance.

---

## Requirement 2 — conditional lists of lines with a threaded index

*A string-slot print emits five instructions, a numeric slot a different set, with a
caller-supplied index threaded into every generated SSA name (`%end_str_ptr_3`,
`%end_field_1_len`).*

**Verdict: expressible today. Not a dialect gap.** Measured end to end — with
`[slot=str_slot(head), 'Idx'=3]`:

```
  %head_ptr_3 = getelementptr i8, i8* %buf, i64 0
  %head_len_3 = call i64 @strlen(i8* %head_ptr_3)
  %end_str_ptr_3 = getelementptr i8, i8* %head_ptr_3, i64 %head_len_3
```

Three v1 features carry it and none is stretched: a case body is arbitrary multi-line literal
text, so "a list of lines" needs no list construct; the case pattern binds the slot name
structurally; and the index is an ordinary dict entry, visible inside the body because unshadowed
outer keys remain visible (SPEC, matching semantics 4). Composite names like `%end_field_1_len`
interpolate the same way — `{{Key}}` substitutes mid-token, with no delimiter requirement.

### The boundary that *is* a gap, so the reading is not misfiled

- **Fixed membership per kind** — five lines for a string slot, two for a numeric slot, chosen by
  kind. **Expressible.** The "list" is literal text.
- **Variable length within one render** — *N* lines for *N* arguments, decided inside the
  template. **Not expressible**, deliberately: `list patterns / store iteration` is an existing
  exclusion whose revisit condition already reads *"a consumer that cannot iterate outside the
  template."* The driver iterates and concatenates; the template dispatches one term.

Requirement 1 makes the first reading nearly certain — a driver that resolves slot kinds is
already iterating. If it is ever the second, that is not a new gap: it is the existing row's
revisit condition firing, and should be raised as such.

### The hazard specific to threading

A case pattern naming a variable spelled like the threaded key **silently shadows it**:

```
{{match slot}}{{case boxed(Idx)}}%inner_{{Idx}}{{/match}}   with [slot=boxed(99), 'Idx'=3]
  →  "%inner_99"        % not "%inner_3"
```

The outer key is untouched after the block — the shadow is lexical and local, exactly as the SPEC
specifies. That is the point: the index spans the whole emission while shadowing is decided one
case at a time, so a template author adding a case makes a global decision from a local view.
Mitigation is a naming convention owned by whoever writes the templates, not a dialect change.
Also note dict keys match by **exact spelling**: a threaded index must be a quoted `'Idx'=3`, since
an unquoted `idx` does not fill `{{Idx}}`.

---

## Requirement 3 — whitespace control, not byte-identity

*plawk's regression tool is a byte-level golden-IR diff, but that constraint belongs to the tool,
not to the dialect. What is needed is that the dialect not impose uncontrollable whitespace around
`{{match}}`/`{{case}}`, so a byte-faithful migration is possible when chosen. Where natural
template formatting differs cosmetically, plawk normalizes through an `llvm-as | llvm-dis`
round-trip or re-baselines with behavioural verification — so **do not add whitespace-exactness
machinery on plawk's account**.*

**Verdict: controllable. A byte-faithful migration is possible, and the cost is layout, not
gymnastics.** Every marker-adjacent newline is the template author's, but only the `{{match}}` tag
gets a free line — the preamble discard. The other three markers are literal boundaries:

| to emit | the tag must | otherwise |
|---|---|---|
| a body starting at its first character | `{{case P}}` shares a line with that character | a leading `\n` enters the body |
| a body ending at its last character | `{{/match}}` follows that character directly | a trailing `\n` enters the body |
| a default body likewise | `{{default}}` shares a line with its first character | a leading `\n` enters the body |

So `{{match k}}\n{{case a}}X\n{{/match}}` renders exactly `"X\n"`, and a block can occupy whole
lines while contributing none of them. Consecutive cases compose at no extra cost: the newline
before the next `{{case}}` is the previous body's own terminator, which is what line-oriented
emission wants. The single residue is that a `{{case}}` tag shares a line with its first
instruction. That is the layout cost, it is what standalone-line semantics would remove, and it
is now that exclusion's revisit condition — **not a reason to act.**

Nothing was built for byte-exactness, and nothing will be. The 33 tests are characterization plus
a home for the three properties the SPEC already declared normative and never pinned; they would
be worth having with no prospective consumer at all.

### The `c"..."` caveat, generalized into a rule for this lane

Their caveat is that bytes inside `c"..."` string-constant globals are semantic data — OFS
separators and printf formats live there — so only a parser-aware normalizer is safe. This
dialect never normalizes text, so the caveat is currently moot. It is recorded because it
constrains any *future* whitespace feature: such a feature must be **marker-local**, deciding
only about bytes adjacent to a tag, and must never scan body content. Standalone-line semantics
as excluded above satisfies that; a trim or reflow option would not, and this caveat is the
reason to refuse one.

---

## What this lane got wrong, and what changed

The correction is worth recording as a finding, not just absorbed.

**Over-attribution from a relayed requirement.** Given "byte-identical rendering", this record
reasoned to a downstream conclusion: a consumer whose only oracle is a byte diff has no local
symptom when a placeholder is unbound, therefore the `missing-key / unbound-placeholder`
exclusion row had found its first named consumer. Both steps were sound; the premise was not
plawk's. A property of someone's *verification instrument* was read as a property of their
*requirements* — and the failure mode is that it looks like evidence. This is what
PROJECT_PHILOSOPHY §8 (verified vs inferred) is for, at one remove: the requirement was relayed
rather than read, and a relayed premise carries no less weight than it was given.

The reusable form, worth the same standing suspicion this dialect gives *"does it decide, or does
it render?"* — **does this constraint belong to the artifact, or to the tool that measures it?**
This lane has its own instance of the error, from the opposite side: the CI leg's `initialization/1`
trap, where swipl's exit status reported "all green" because the *pipeline* could not see a
failure the system had. Both are a measuring instrument's property mistaken for the measured
thing's, once as a false requirement and once as a false pass.

Removed: the claim that this row has a named prospective consumer. It keeps its original
unclaimed revisit condition, and the SPEC says so explicitly so the retraction is visible rather
than merely absent. The hazard itself is real and stays pinned — as a property of the engine,
attributed to nobody.

**A dangling pointer, found while fixing the above.** The SPEC's Whitespace section said mustache
standalone-line semantics were *"an exclusion, below"* — and no such row existed in the exclusions
table. The reference had been dangling since the spec was written. The row now exists, with the
revisit condition this assessment produced.

---

## Where this is recorded

| record | what it holds |
|---|---|
| this file | the three corrected verdicts, the measurements, the leak path, the over-attribution |
| `tests/core/test_pattern_stache_whitespace.pl` | 33 tests; `marker_adjacency` answers requirement 3 |
| `SPEC_pattern_stache.md`, Whitespace | the preamble discard and the marker-adjacency table |
| `SPEC_pattern_stache.md`, exclusions | the standalone-line row that was missing, and one named revisit condition |

No grammar changed, no engine behaviour changed, no exclusion was reopened.
