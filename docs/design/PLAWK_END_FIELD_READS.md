# plawk: field reads in END (`END { print $1 }`) — design, and what shipped

Status: **implemented** for the straight-line END print (single print, statement
list, concatenation). Every fact below was established by probing or reading the
code, not inferred.

Two things the implementation changed about this design; both are worth reading
before the follow-ons, because each removed work the plan had budgeted for.

**1. The runtime did not need to go in `wam_llvm_target.pl` at all.** The
sequencing constraint recorded at the bottom of this note — "the runtime cannot
land before its wiring", because globals beside `@wam_rt_*` perturb every
program's `.ll` — dissolved once the globals and the two `define`s were emitted as
**program-level IR from the codegen**, which plawk already does for its
`@plawk_foreign_*` / `@plawk_dyncall_*` helpers. They are emitted only when the
gate fires, so a program with no END field read is byte-identical (verified: 15/15
of a mixed golden corpus). The constraint was real; the decomposition it forbade
just turned out to be unnecessary.

**2. Work item 3 ("END field projection") needed no field-emitter change.**
`@wam_transient_atom_from_bytes(i8*, i64)` copies bytes into the shared transient
record buffer and returns the reserved transient atom id — the mechanism
`@wam_getline_file_record` already uses so "the existing `%line` Value immediately
exposes the new `$0`". END re-materialises the retained bytes that way, boxes the
id as a `%Value`, and hands it to `llvm_emit_atom_field_slice/5`, **which was
already parameterised on the record Value** (`+ValueIR` is its first argument).
So no slicing logic is duplicated and no emitter was reparameterised. The plan's
warning against "a second field emitter" was right; the cheaper route was to
notice the existing one already took the record as a parameter.

Note the EOF `%Value` itself carries a *real interned* `end_of_file` atom id, not
the transient id — so rewriting the transient buffer does **not** retarget `%line`
at `end_print`. END must construct its own `%Value`. That was checked, not
assumed, and it is the fact that decides between the two approaches.

**A pre-existing wrong output this work surfaced.**
`{ n++ } END { if (n == 3) print $1 }` printed `end_of_file`, and so did `$0` and
the `else` branch — confirmed against the parent commit, so not introduced here.
`plawk_end_if_branch_ir/8` lowers each branch through the **rule-body** print
emitter, which projects `$N` from `%line`: the identical defect #4100 gated for
END loops, in a driver nobody re-checked. Now a clean decline via
`plawk_end_if_ok/2`, pinned. This is the third time in this campaign that reusing
an emitter imported what it *assumes* rather than only what it does.

## The gap, and why it now has two callers

```awk
{ n++ } END { print $1 }                            DECLINES
{ n++ } END { while (n > 0) { print $1; n-- } }     DECLINES (gated in #4100)
```

awk keeps `$0` and its fields in END, holding the **last record read**. gawk on
three records `a 1 / b 2 / c 3` prints `c`.

The second caller is new. #4100 admitted loops into END, and its loop bodies go
through the rule-body sequence emitter — which lowers `$1` against `%line`
without knowing it has left the record loop. That produced **wrong output**
(`end_of_file`, three times) and had to be gated to a decline. So this work now
converts two declines into one feature, and removes a gate rather than adding one.

## Established fact: the last record is gone by END, not merely unreachable

At `end_print` the transient buffer holds the bytes **`end_of_file`** — the EOF
sentinel the reader interned on its final call. Proven by probe: before the #4100
gate, `END { while (…) print $1 }` printed `end_of_file` per iteration, which is
`$1` of the sentinel text.

So this is not a matter of finding the record; retention must be **explicit**.

Note `%line` at `end_print` is the EOF sentinel `%Value`, which is why
`plawk_assoc_end_if_line_context/5` synthesises a separate transient `%Value` for
record-shaped assoc keys rather than using `%line`. That helper is *not* a
last-record mechanism and cannot be reused as one.

## The mechanism to copy: `@wam_rt_set`

`RT` (the matched record separator) already survives to END, and its runtime is
exactly the shape needed:

- `@wam_rt_buf` / `@wam_rt_cap` — a single **reused, geometrically grown**
  buffer (`realloc` to `max(needlen, cap < 4096 ? 4096 : cap*2)`);
- `@wam_rt_ptr` / `@wam_rt_len` — what readers consume;
- `@wam_rt_set(i8* src, i64 len)` — `memcpy` into it, NUL-terminating;
- `@wam_rt_clear()` — reset to the empty constant.

**Constant memory, one copy per record.** That property is why this is the right
mechanism and interning is the wrong one: `wam_intern_atom` per record would grow
the atom table with every *distinct* record, breaking the streaming invariant the
whole design protects. (The same reasoning is already recorded for
`@wam_strnum_cmp_slices`, which uses a scratch buffer specifically to avoid
interning per-record values.)

Add a **parallel** set of globals rather than reusing RT's: `RT` has its own
meaning and a program can read both. Suggested names `@wam_lastrec_buf/_cap/_ptr/_len`
with `@wam_lastrec_set/2`, structurally copied from the `wam_rt_*` block
(`src/unifyweaver/targets/wam_llvm_target.pl`, ~7567–7640).

## Pay only when used — and the gate already exists

`plawk_end_term_mentions_field/1`, added in #4100 as the *safety* gate for END
loops, is exactly the predicate needed to decide whether to retain: a structural
term walk finding `field(_)` at any depth in the END actions.

So the change **inverts** that gate rather than adding a second one:

- END mentions no field → emit nothing, byte-identical to today (this is what
  keeps the golden diff clean for every existing program);
- END mentions a field → emit the retain call in the record loop, and project END
  fields from the retained buffer.

One predicate, two uses, no second definition to drift — which matters given how
many of this campaign's defects were exactly that.

## Work items

1. **Runtime** (`wam_llvm_target.pl`): `@wam_lastrec_*` globals + `@wam_lastrec_set/2`,
   copied from the `wam_rt_*` block. Declare in the runtime-globals blocks.
2. **Record loop**: after a successful read, when the gate says END needs it,
   `call @wam_lastrec_set(i8* %line_s, i64 <len>)`. Length: the reader gives a
   `%Value`; `@wam_atom_to_string` + `strlen`, or better a length accessor if one
   exists — **check before assuming**.
3. **END field projection**: the field emitters read `%line`. They need a variant
   sourcing the record pointer from `@wam_lastrec_ptr`. Prefer *parameterising the
   record source* over a second field emitter — a second emitter is the
   duplication this line keeps paying for.
4. **Gate inversion** + remove the #4100 field-read decline for END loops.
5. **`NF` in END** falls out of the same retained record and should be included or
   explicitly pinned.

## Hazards, each to be probed not reasoned about

- **`getline`** can grow/relocate the shared transient buffer. The retain copies
  bytes, so it is relocation-safe by construction — but a `getline` in a *rule*
  changes which record is "last"; awk's answer is the last record read into `$0`.
  Probe against gawk.
- **Empty input** (no records): awk gives an empty `$0` in END, not garbage. The
  buffer must start as the empty constant, like `@wam_rt_clear` leaves it.
- **`RS` / paragraph mode**: the retained bytes must be the record *without* its
  separator, matching what `$0` held in the loop.
- **Byte-identity**: no program whose END lacks a field read may change. Verify by
  golden dump before/after over a corpus including several END shapes.

## Sequencing constraint found by attempting it

The runtime half was written (globals + `@wam_lastrec_set/2`, mirroring
`@wam_rt_set`) and then **reverted**, which established a constraint worth
knowing before starting:

**The runtime cannot land before its wiring.** The `wam_rt_*` globals block is
emitted into *every* driver, so adding `@wam_lastrec_*` beside it changes the
`.ll` of **every program** — breaking the byte-identity that golden diffs depend
on, for zero functional gain while the globals are unused. Unexercised runtime IR
is also unverified runtime IR.

So do it as one change: runtime + gated store + END projection + gate inversion,
landing together, with the golden diff run at the end to show that only programs
with an END field read differ. Do **not** split the runtime into a separate
preparatory PR.

For reference, the reverted runtime followed `@wam_rt_set` exactly: 5 globals
(`@.wam_lastrec_empty`, `_ptr`, `_len`, `_buf`, `_cap` — no `suppress`), and a
`define i1 @wam_lastrec_set(i8* %src, i64 %len)` with blocks
clear/need/check/grow/alloc/store/copy/fail, `realloc` growth to
`max(needlen, cap<4096 ? 4096 : cap*2)`, `llvm.memcpy` + NUL terminate, and an
empty `%len` resetting to the empty constant so END sees `""` on empty input.

## Why it was not implemented in the session that designed it

The remaining context was insufficient to hand-write the runtime IR and driver
changes *and* verify them. The failure modes here are a clang failure or wrong
output, both of which this line treats as unacceptable to ship unverified — and
the wrong-output variant is precisely what #4100 had to gate. Better a precise
design than a half-applied change to the record lifetime.

## What shipped, and what is still declined

Landed: `$0` and `$N` in a straight-line END print, an END statement list
(`END { print $1; print $2 }`) and a concatenation (`print $1 " / " $2`).
Honours FS, OFS, ORS and a custom RS. `$N` past `NF` is empty; empty input gives
an empty `$0`; multi-KB records exercise the buffer growth. Every hazard listed
above was probed against gawk rather than reasoned about.

The projection is reachable only through `end_record(FS)`, and
`plawk_end_record_source/4` returns that token **together with** the record-loop
retain IR. One predicate returning both is the point: a driver cannot emit the
projection without the store, and a projection whose bytes were never retained
would print *empty* — silently wrong, the failure mode this line refuses to ship.

Still declined (each pinned in `tests/test_plawk_end_field_reads.pl`, so a later
change flips it deliberately):

- **END `if` branch** — the wrong output described above, now a decline.
  Projecting it means parameterising the record source of the prefixed print
  emitter, which is its own change.
- **END loop body** — #4100's gate, unchanged. Needs the same parameterisation in
  `plawk_scalar_action_sequence_pairs//15`.
- **`printf` argument in END** (`printf "%s\n", $1`) — the END printf emitter has
  no field clause.
- **`NF` in END** — falls out of the same retained record; not wired.
- **END-only programs** (`END { print $1 }` with no rules) — a different driver,
  which does not retain.
- **Binary descriptors** (`binfmt`, `binfmt_union`) — no `%line_s` to copy from;
  `$N` there would read the fixed-width record buffer, a separate feature.

A `$0` **modified** in a rule (`{ $1 = "Z" } END { print $0 }`) is not a divergence
today because those programs decline for unrelated reasons. When they compile, note
that the retain happens at the *top* of the record block, so END would see the
record as read rather than as modified — awk shows the modified one.
