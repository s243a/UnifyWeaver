# plawk campaign — working handoff

Written to survive a context compaction. Everything here was established by
probing or reading code in-session, not inferred.

## Where things are

- Feature line: **`claude/plawk-llvm-wam-hybrid-p9ujut`**. One feature per PR,
  based on the feature line, merged by the user who then says what's next.
- Build: `swipl examples/plawk/bin/plawk build FILE.plawk -o OUT [--keep-ll]`
- Exit codes: **0** built · **2** parse error · **3** parses but outside the
  compilable surface (a clean decline) · **4** clang failure (a miscompile).
- Key files: parser `examples/plawk/parser/plawk_parser.pl`; codegen
  `examples/plawk/codegen/plawk_native_codegen.pl` (~20k lines); runtime IR as a
  single-quoted atom in `src/unifyweaver/targets/wam_llvm_target.pl`.
- Docs kept current: `PLAWK_AWK_FEATURE_AUDIT.md` (a row per feature),
  `PLAWK_SPEC_PLAN_DRIFT_AUDIT.md` (the defect class),
  `PLAWK_END_FIELD_READS.md` (the next feature, designed).

## The recurring defect class — now with three variants

**One property implemented in more than one place, the copies quietly
disagreeing.** Every wrong-output bug found in this campaign was an instance.
Three distinct shapes, all confirmed by real bugs:

0. **A name that describes the implementation, not the property.** The newest and
   nastiest, because the gate reads as correct. `plawk_end_term_reads_record/1` was
   called `plawk_end_term_mentions_field/1` and matched `field(_)` only — and `NF`
   reads the record without being a `field(_)`, so it walked past the gate built to
   catch exactly that, and `END { if (n == 3) print NF }` printed NF of the
   `end_of_file` sentinel for as long as the driver existed. Nobody re-read the gate
   because its name sounded like the property. **Name gates for the property they
   enforce**; when the name and the clauses can drift, the name wins the argument
   and the clauses stay wrong.
0b. **A new producer capturing an older one's surface.** The same shape as 0, one
   level down, and the only variant so far that lives in the **parser**. `TAG` is a
   valid identifier, so registering the scalar-variable pattern (`n > 2`) ahead of
   `tag_eq_pattern//1` made the tag production unreachable: `TAG == 1` parsed as a
   comparison on a variable named TAG, the codegen desugar looked for `tag_pat/1`,
   and the whole tagged-union tag-guard sugar became a clean decline. **Eight tests
   across five suites**, one cause, red for however long.

   Two things make this variant nasty: the failure is a *decline*, so nothing
   crashes or prints wrongly, and DCG clause order carries the semantics invisibly —
   nothing in either production mentions the other. When adding a pattern that can
   match a bare identifier, check what it now shadows, and pin the precedence at the
   **parse** level (assert the term) rather than only downstream, so the next
   ordering change fails with the reason on the label.
1. **Level drift** — the same property asked at surface / spec / plan level, one
   level not taught a new producer. (The original audit doc covers this.)
2. **N emitters or N walkers over one term.** The commonest. Examples: the ORS
   terminator re-derived in **four** whole-record print emitters; the
   string-scalar comparison written **twice** with a text-slot guard in only one
   copy (wrong output); the ternary condition walked by **four** traversals that
   each had to learn `&&`/`||` and `ternary_str/3` independently, one of which had
   no row at all.
3. **Reuse importing assumptions.** Routing a new context through an existing
   emitter inherits what it *assumes*, not only what it does. Three times: END
   string guards inherited a missing text-slot guard (#4078); END loop bodies
   inherited "`%line` is a real record" and printed the EOF sentinel (#4100); and
   the END **`if` branch** inherited the same thing from the rule-body print
   emitter and printed `end_of_file` — undetected until the END field-read work
   probed its neighbourhood, because nobody re-checked that driver when #4100
   found the identical defect next door.

   The lesson that keeps paying: when a bug is found in one context that reuses an
   emitter, **enumerate the emitter's other callers in the same breath**. #4100
   gated loops and stopped; the `if` driver was one grep away.

**A fifth instance, and the cleanest illustration of the class:** awk's
uninitialised value is `""` in string context and `0` in numeric. plawk implemented
that property in *four* slot kinds and got it right in some and wrong in others —
absent array elements were fixed (probe the occupied bit), string scalars were
right (unset atom id 0 renders empty), and **counter scalars were wrong** (an i64
register initialised to 0, rendered numerically), so `{ n++ } END { print n }` on
empty input printed `0`. The property was never stated in one place, so each kind
decided it locally, by its storage.

**Prescription:** when a semantic rule spans representations, the *rule* needs a
named home, not each representation. The fix used the same presence-not-value test
the array path already had — and the giveaway that it was one property, four
implementations, was that fixing one of them (arrays) left the others visibly
inconsistent with it for a whole release.

**Prescriptions that worked:** one shared producer/emitter with callers
parameterised (a *name flavour* parameter can preserve byte-identity — see
#4094); when adding a fast path, check what the general walker's **base case**
does after the last element; for safety gates prefer a **structural term walk**
over a per-action-shape walker, because a new nesting level cannot defeat it.

**And one that worked better than parameterising:** when a new context needs an
emitter to behave differently, consider **rewriting the term** instead of threading
a parameter. The END field reads in `if` branches and loop bodies needed the record
source changed in emitters with 30 `%line` references between them; rewriting
`field(N)` → `end_lastrec_field(N)` before the actions reach those emitters, plus
two clauses on the shared print-expression emitter, left both emitters untouched.
Everything in between — prefix naming, separators, the ORS terminator — never had
to know.

The safety property comes for free if you get the direction right: an unrewritten
`field(N)` reaching a *print* emitter would miscompile, so the rewrite is
structural (same walk as the gate); a rewritten `end_lastrec_field(N)` reaching a
*condition* emitter finds no clause and **declines**. Rewrite indiscriminately and
let the missing clauses be the gate.

## Verification practices (do not skip)

- **gawk 5.2 is the oracle.** Compare output *and* exit status. Probe harness
  pattern: write the program, `rm -f` the binary first (a declining build must not
  run a stale one), build, run, diff against `gawk`.
- **Golden IR diff** for anything claiming "no other program changes": build a
  corpus with `--keep-ll` before and after and diff the `.ll` files. This is how
  #4094's byte-identity claim and #4068's blast radius were established.
- **Probe before concluding, one variable at a time.** Three wrong scoping
  calls this session came from asserting structure without reading it; each was
  settled by a probe that varied one thing. A confound I introduced myself
  (`n--` in every END probe) cost the most.
- **Never pin a driver-wide grep for a common LLVM mnemonic.** `@strcmp` is in
  every EOF check, `@wam_intern_atom` in every input-path intern, `and i1` in
  every read loop. Three tests written that way passed vacuously or failed for
  unrelated reasons. Pin the construct's **own** generated variable names.
- **A fourth of the same shape, in the golden-diff harness:** it tested
  `[ -f PROG.bin.ll ]` to decide whether a program compiled, but a **declining**
  build still writes a `.ll` (the WAM fallback), so the test was always true and a
  declining program contributed fallback IR to the corpus rather than being
  recorded as declined. Key off the **binary** and record the build status. The
  common thread in all four: **an assertion whose failure mode was unreachable.**
  Before trusting a check, ask what would have to be true for it to fail.
- **Flipping a pin:** rewrite it to say what changed, don't delete it. Check
  whether its *rationale* was broader than its truth (the mixed-ternary pin
  blamed "needs a runtime conversion", true only for non-literal arms — so it was
  split, not just inverted). A pin can also be invalidated by a **sibling
  commit** that never touches its file.
- Turn "silently wrong" into either correct behaviour **or a clean decline (3)**,
  and pin the decline so a later PR flips it deliberately.

## Process hazards

- **PR stacking**: merging a stack's base before its dependents strands them —
  GitHub auto-closes PRs whose base branch is deleted. Cost four bookkeeping PRs.
  Merge innermost-first, or retarget before merging a base. Simplest: base each
  PR on the feature line directly.
- Suites are slow (each test does a full clang build, ~8s). Run them in the
  background and monitor; ~25 tests ≈ 4 minutes.
- `swipl` loading the codegen module directly is slow enough to time out; prefer
  probing through `bin/plawk`.
- Commit trailers required:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01HJci66LfkyrStXJPxmMksn`

## Landed this session

posarray drift audit · `printf` in END/BEGIN · nested string-scalar fix · `exit`
in END/BEGIN · BEGIN-only programs · four END statement-list chains · ternary
conditions (parens, string cmp) · reversed comparisons · bare `print` · **ORS on
the whole-record print** (4 emitters) · ternary **string-valued branches** ·
**END string guards** + a counter-vs-string wrong-output fix · **`$0` vs string
literal** · parenthesised whole ternary · `$0` as a ternary condition ·
**`&&`/`||` in ternary conditions** · scalar-var ternary operands · **one emitter
for the string-scalar comparison** (byte-identical) · **mixed ternary branches** ·
**`n--`** · **loops in END** · **`END { print $1 }`** (retained last record,
pay-per-use, + a pre-existing END-`if` wrong output converted to a decline) ·
**field reads in END `if` branches and loop bodies** (term rewrite, not emitter
parameterisation) · **`NF` and `printf` record args in END** (+ the gate renamed to
the property it enforces) · **14 stale assoc expectations** in the prefix-print
suite · **unset scalars render empty** (monotonic assigned-mark).

## A note on the one shared-globals exception

`@plawk_slot_assigned` and `@.plawk_surface_print_unset` live in
`plawk_i64_end_print_globals/3` — the one globals emitter every driver clause calls
— so they appear in **every** program's `.ll`, unlike the pay-per-use
`@plawk_lastrec_*`. That was deliberate: giving the mark table a per-program width
would mean threading the slot count into ~25 driver clauses. The consequence is that
byte-identity golden diffs are not the right check for that change; the corpus was
verified on **program output** instead (only empty-input runs of programs printing
an unset counter differed, all matching gawk). If another change needs shared
globals, expect the same tradeoff and verify the same way.

## `END { print $1 }` — landed

Implemented for the straight-line END print (single print, statement list,
concatenation). See `PLAWK_END_FIELD_READS.md` for the full record, including what
the implementation changed about the design. The load-bearing facts:

- The last record is **gone** by END — the transient buffer holds the bytes
  `end_of_file` (proven by probe), and the EOF `%Value` carries a *real interned*
  atom id, not the transient id, so `%line` cannot be retargeted.
- The record loop copies each record into a reused, geometrically grown buffer
  (`@plawk_lastrec_store`, modelled on `@wam_rt_set`) → **constant memory, one
  memcpy per record**. Interning per record would grow the atom table with every
  distinct record and break the streaming invariant.
- **`@wam_transient_atom_from_bytes/2`** writes bytes into the shared transient
  record buffer and returns the reserved transient atom id — the mechanism
  `getline` already uses to expose a new `$0`. END re-materialises the retained
  bytes that way and boxes the id as a `%Value`.
- **`llvm_emit_atom_field_slice/5` was already parameterised on the record
  Value.** So the projection reuses the *same* slicer every in-loop field read
  uses; no field emitter was changed or duplicated. Worth checking for the
  follow-ons: an emitter may already take as a parameter the thing you were about
  to thread through it.
- `plawk_end_term_mentions_field/1` (#4100's *safety* gate) is **inverted**, not
  restated, to decide whether to retain.
- `plawk_end_record_source/4` returns the capability token **and** the retain IR
  together, so the projection cannot be emitted without its store. A projection
  with no store prints *empty* — silently wrong.
- The runtime is **program-level IR from the codegen**, not the shared runtime
  block. That is what makes it pay-per-use and dissolved the "runtime cannot land
  before its wiring" constraint this doc previously recorded.

## Remaining follow-ons

**END record reads, what is left** — each pinned as a decline in
`tests/test_plawk_end_field_reads.pl`. A field or `NF` in a loop / `if`
**condition** (a fail-safe decline: the rewrite reaches conditions, and no
condition emitter has a clause for `end_lastrec_field(_)` / `end_lastrec_nf`) ·
**`length`** in END (already in the gate, not yet emitted) · **builtins over the
record** in END (`substr($0, …)`, `toupper($1)` — the gate retains for them, the
emitters have no clause) · **END-only** programs (a driver with no retain) ·
`printf` field args in the **assoc / mixed END chain** (a different driver, passes
`no_end_record`) · the **associative** END-`if` branch (refused by
`plawk_assoc_end_if_branch_prints_ok/2`, which allows only string literals there —
an unrelated pre-existing restriction, pinned as such) · `$N` in a `binfmt` END
(reads the record buffer, not a text slice).

`arr[k]--` (needs a row in each of four `inc_assoc` walkers) · `n += -1` (parser
rejects a negative compound-assign delta; odd now `n--` works) · assoc rules
alongside a scalar END loop (state-plan boundary) · float-literal and non-literal
ternary arms (need runtime number→string) · `dec` row in the binfmt action gate ·
the dead `@.plawk_surface_print_line` global (removal costs byte-identity on every
program, so its own PR) · `printf`/`NR` as plain statements in the for-in END
chain · `NR` in a loop-free END list · `printf "%c"` on a string ·
autovivification on read · scalar-var SUBSEP components · empty-string subscripts.
