# Record: plawk as a prospective consumer of `pattern_stache`

**Status: record and assessment. No dialect change, no implementation, no integration.**

plawk (`examples/plawk`, an awk → Prolog → WAM/LLVM compiler) is a **prospective** third consumer
of the `pattern_stache` dialect. **Zero code references exist today** — nothing imports the
dialect, no `.stache` file exists under `examples/plawk`, and no request for a dialect change has
been made. plawk is collapsing its duplicated Prolog emitters first, which is a prerequisite
either way. This file exists so that whenever the dialect next grows, the implementer starts from
a written record instead of rediscovering these constraints — the same role the cowalk delta
plays in [`REPORT_pattern_stache_interpretation.md`](REPORT_pattern_stache_interpretation.md).

**Provenance caveat, stated because it changes how much weight these three requirements carry.**
The assessment they come from is `docs/design/PLAWK_PHILOSOPHY.md` §6 (esp. §6.5), which is **not
reachable from this branch or from main**: the file exists here but its headings stop at "## 5.
Relationship to the existing AWK target" and it contains no occurrence of `pattern_stache`. The
§6 text lives on `claude/plawk-llvm-wam-hybrid-p9ujut`. The three requirements below are
therefore recorded **as relayed**, not as read. Where a verdict depends on precisely what §6
says, that is flagged.

---

## Verdicts

| # | prospective requirement | verdict |
|---|---|---|
| 1 | byte-identical rendering | **already the specified behaviour**, and now measured — with three unpinned hazards, one of which was undocumented |
| 2 | conditional lists of lines with a threaded index | **expressible in today's surface. Not a dialect gap** — but its safe use rests on an existing deliberate exclusion that this consumer is the first to have a reason to reopen |
| 3 | selection only, no computation in templates | **already the spec's own rule**, restated from the consumer side |

Everything below is measured against `src/unifyweaver/core/pattern_stache.pl` at this commit and
pinned by [`tests/core/test_pattern_stache_whitespace.pl`](../../tests/core/test_pattern_stache_whitespace.pl)
(26 tests, new with this record). Nothing here was inferred from the SPEC alone: before this
suite, **no test pinned line or byte fidelity at all**, because both witnessed v1 consumers
normalized whitespace per rendering with `normalize_space/2`. That is exactly the sort of
unmeasured property this lane is not supposed to assert (PROJECT_PHILOSOPHY §1, §8), which is why
the assessment is a suite and not a paragraph.

---

## Requirement 1 — byte-identical rendering

*Their regression tool is a golden-IR diff, so a template layer that reformats whitespace or
reorders lines breaks it.*

### What the engine guarantees, measured

- **Case bodies are literal.** Indentation, interior blank lines, tabs, and `\r\n` all survive
  unchanged. Nothing is trimmed at either end. This is the SPEC's Whitespace section, and those
  rows are now its home rather than prose alone.
- **Nothing is reordered.** Rendering is `Before ++ RenderedBody ++ After` in file order
  (`expand_match_blocks/3`); there is no buffering, no sorting, no reflow.
- **Mustache "standalone line" semantics are absent** — a deliberate v1 exclusion, and the right
  way round for this consumer: the engine will not silently swallow the newline after a lone
  `{{case ...}}` line the way a mustache implementation would.
- **At load**, exactly two things are deleted: blank lines preceding the dialect header, and the
  header line itself. The rest of the file reaches the renderer byte for byte.
- **A non-matching block with no default renders as the empty string** without disturbing the
  text on either side.

So the answer to requirement 1 is yes. The value of having measured it is the three things that
came with it.

### Hazard 1a — the preamble discard (was undocumented)

Text between `{{match k}}` and the **first** `{{case ...}}` is **discarded**: not rendered, not an
error.

```
"PRE{{match k}}JUNK{{case a}}BODY{{/match}}POST"  →  "PREBODYPOST"
```

This is what lets a template put the match tag on its own line without emitting that newline, and
it is the **one place the engine deletes bytes it was handed**. The SPEC's Whitespace section did
not mention it; it does now, with the suite as its home. For a byte-diffing consumer this is
benign once known and confusing once not.

### Hazard 1b — an unbound key is silently verbatim

`SPEC` exclusions: *missing-key / unbound-placeholder changes — left verbatim*, revisit when *a
consumer needs fail-on-unbound rendering*. Measured:

```
render("%end_str_ptr_{{Idx}}", ['Idxx'=3])  →  "%end_str_ptr_{{Idx}}"     % no error
render("%p_{{Idx}}",           [idx=3])     →  "%p_{{Idx}}"               % no error
render("%p_{{Idx}}",           ['Idx'=3])   →  "%p_3"
```

Two things collide here. Placeholder keys match by **exact spelling**, so a threaded index must
be a **quoted** dict key `'Idx'=3` — an unquoted `idx` does not fill `{{Idx}}`. And a key that is
missing, misspelled, or wrongly cased produces **well-formed output containing a template
marker**, with no exception and no warning.

For every consumer so far this was the harmless choice. For a consumer whose only oracle is a
byte diff against a golden file, it is the failure mode with **no local symptom**: the emitter
succeeds, the IR looks like IR, and the mismatch surfaces one layer down as a diff against a file
that may itself need regenerating. This is the first consumer with a reason to want
fail-on-unbound, and the SPEC already says what that would cost — a **new dialect version**, since
it changes output.

**This is the recorded prospective-consumer constraint.** It is not a request, and it is not
being built. It is written down so that whoever next opens the exclusions table sees that this
row has a named potential consumer and a stated reason, rather than an empty revisit condition.

### Hazard 1c — substitution is a sequential global replace

`substitute_placeholders/3` walks the dict in order and replaces each key's marker throughout the
whole text, so a value that itself contains a later key's marker is rescanned, and the result
depends on **dict order**:

```
[a='{{b}}', b=zzz]  →  "A=zzz B=zzz"
[b=zzz, a='{{b}}']  →  "A={{b}} B=zzz"
```

Irrelevant to LLVM IR, which contains no `{{`. Recorded because it is a property of the renderer
that no consumer has yet had to know, and a consumer interpolating text it did not author would.

---

## Requirement 2 — conditional lists of lines with a threaded index

*A string slot emits five instructions, a numeric slot a different set; a caller-supplied index
appears in every generated SSA name (`%end_str_ptr_3`).*

**Verdict: expressible today. Not a dialect gap.** Their own example renders:

```
{{match slot}}
{{case str_slot(Name)}}  %{{Name}}_ptr_{{Idx}} = getelementptr i8, i8* %buf, i64 0
  %{{Name}}_len_{{Idx}} = call i64 @strlen(i8* %{{Name}}_ptr_{{Idx}})
  %end_str_ptr_{{Idx}} = getelementptr i8, i8* %{{Name}}_ptr_{{Idx}}, i64 %{{Name}}_len_{{Idx}}
{{case num_slot(Name)}}  %{{Name}}_val_{{Idx}} = load double, double* %slotbuf
  %end_num_{{Idx}} = fptosi double %{{Name}}_val_{{Idx}} to i64
{{/match}}
```

with `[slot=str_slot(head), 'Idx'=3]` giving, byte for byte:

```
  %head_ptr_3 = getelementptr i8, i8* %buf, i64 0
  %head_len_3 = call i64 @strlen(i8* %head_ptr_3)
  %end_str_ptr_3 = getelementptr i8, i8* %head_ptr_3, i64 %head_len_3
```

Three v1 features carry it and none is stretched: a case body is arbitrary multi-line literal
text, so "a list of lines" needs no list construct; the case pattern binds `Name` structurally;
and the threaded index is an ordinary dict entry, visible inside the body because outer keys not
shadowed remain visible (SPEC, matching semantics 4).

### The boundary that is a gap, stated precisely

The requirement as relayed says "conditional lists". Two readings, and only one is expressible:

- **Fixed membership per kind** — five lines for a string slot, two for a numeric slot, chosen by
  kind. **Expressible**, as above. The "list" is literal text.
- **Variable length within one render** — *N* lines for *N* arguments, decided inside the
  template. **Not expressible**, and deliberately: `list patterns / store iteration` is an
  existing exclusion whose revisit condition already reads *"a consumer that cannot iterate
  outside the template"*. The driver iterates and concatenates; the template dispatches one term.

Which reading §6.5 means is the one thing this record cannot settle from the relayed summary. If
it is the second **and** plawk cannot iterate in Prolog, that is not a new gap — it is the
existing exclusion's revisit condition firing, and it should be raised as such. Given requirement
3, the first reading is far likelier: a driver that already resolves slot kinds is already
iterating.

### The hazard specific to threading

A case pattern that names a variable with the same spelling as the threaded key **silently
shadows it**, per the SPEC's own scoping rule:

```
{{match slot}}{{case boxed(Idx)}}%inner_{{Idx}}{{/match}}   with [slot=boxed(99), 'Idx'=3]
  →  "%inner_99"        % not "%inner_3"
```

The outer key is untouched after the block — the shadow is lexical and local. That is the point:
the index is a property spanning the whole emission, while shadowing is decided one case at a
time, so a template author adding a case is making a global decision from a local view. Same
disease as the "comment at one site explaining a policy that spans four" tell. The mitigation is
a naming convention, not a dialect change, and it belongs to whoever writes the templates. It is
pinned here so it is a known rule rather than a debugging session.

---

## Requirement 3 — selection only

*Slot-kind and key-space decisions stay in Prolog; templates receive resolved kinds.*

Already the spec's rule, in three places, cited rather than paraphrased:

- **Purpose**: *"It selects; it does not prove … Structural matching improves dispatch, nothing
  else."*
- **Exclusions**, `guards`: not expressible; *"the one guard-shaped question — groundness — is
  owned by the caller's discharge ordering."*
- **Exclusions**, `pattern arithmetic`: not expressible, revisit condition *"never expected."*

There is no expression sublanguage to keep out. `{{Key}}`/`{{q:Key}}` interpolate a value the
caller computed; cases select on shape. Requirement 3 is not a concession the dialect makes — it
is the dialect's boundary, arrived at independently from the consumer side, which is the useful
part of the datum.

---

## One thing not in their three requirements

`{{Key}}` renders with `~w`, so **interpolated numbers inherit SWI's writer**. Where that diverges
from another producer's spelling is already measured in this lane — see
[`pattern_stache/pe_number.pl`](pattern_stache/pe_number.pl), built after exactly this assumption
failed against CPython's `repr` in three distinct ways. Integer indices are safe. If any plawk
golden IR carries a float produced by a non-SWI writer, that module is the one to reach for, not
the dialect.

---

## Where this is recorded

| record | what it holds |
|---|---|
| this file | the three verdicts, the measurements, the gap shape |
| `tests/core/test_pattern_stache_whitespace.pl` | 26 tests arming every measured claim above |
| `SPEC_pattern_stache.md`, Whitespace | the preamble discard, previously undocumented |
| `SPEC_pattern_stache.md`, exclusions | a pointer from the two rows this record bears on |

No grammar changed, no engine behaviour changed, no exclusion was reopened.
