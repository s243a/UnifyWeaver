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

**Prescriptions that worked:** one shared producer/emitter with callers
parameterised (a *name flavour* parameter can preserve byte-identity — see
#4094); when adding a fast path, check what the general walker's **base case**
does after the last element; for safety gates prefer a **structural term walk**
over a per-action-shape walker, because a new nesting level cannot defeat it.

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
pay-per-use, + a pre-existing END-`if` wrong output converted to a decline).

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

**END field reads, the rest of the surface** — each pinned as a decline in
`tests/test_plawk_end_field_reads.pl`. The `if`-branch and loop-body cases both
need the *record source* parameterised in the emitter they share with rule bodies
(`plawk_prefixed_print_action_ir` and `plawk_scalar_action_sequence_pairs//15`);
do those two together, since it is one parameterisation serving both. Then:
`printf "%s\n", $1` in END (its own argument emitter), `NF` in END (falls out of
the same retained record), END-only programs (a driver with no retain), and `$N`
in a `binfmt` END (reads the record buffer, not a text slice).

`arr[k]--` (needs a row in each of four `inc_assoc` walkers) · `n += -1` (parser
rejects a negative compound-assign delta; odd now `n--` works) · assoc rules
alongside a scalar END loop (state-plan boundary) · float-literal and non-literal
ternary arms (need runtime number→string) · `dec` row in the binfmt action gate ·
the dead `@.plawk_surface_print_line` global (removal costs byte-identity on every
program, so its own PR) · `printf`/`NR` as plain statements in the for-in END
chain · `NR` in a loop-free END list · `printf "%c"` on a string ·
autovivification on read · scalar-var SUBSEP components · empty-string subscripts.
