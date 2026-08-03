# plawk: field reads in END (`END { print $1 }`) — design

Status: **designed, not implemented.** Every fact below was established by probing
or reading the code, not inferred. Written so the implementation can start cold.

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

## Why it was not implemented in the session that designed it

The remaining context was insufficient to hand-write the runtime IR and driver
changes *and* verify them. The failure modes here are a clang failure or wrong
output, both of which this line treats as unacceptable to ship unverified — and
the wrong-output variant is precisely what #4100 had to gate. Better a precise
design than a half-applied change to the record lifetime.
