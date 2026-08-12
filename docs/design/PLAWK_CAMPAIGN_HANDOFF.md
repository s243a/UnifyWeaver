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

**Sizing a follow-on by the wrong relative.** `arr[k]--` was recorded as needing "a
row in each of four `inc_assoc` walkers", because `arr[k]++` is the `inc_assoc`
family. It needed **none**: `arr[k] += N` already parses to `add_assoc/3`, so the
decrement desugars into that family and no walker changed. The estimate had followed
the nearest *spelling* (`++`) instead of the nearest *semantics* (`+= N`) — and
adding `dec_assoc` would have made a third representation of one operation. When a
follow-on is sized as "a row in every walker", first ask which existing family the
surface form belongs to.

**A gate's failure mode decides what a missing row costs — and one kind of gate
fails silently.** The svar-key assoc gap was *four* independent lists of which
scalar-var-keyed action shapes exist (the mixed route's admission gate, the
table-registration walker, the assoc-only body spec, and the strnum read-use gate),
each naming `inc_assoc` and none naming `add_assoc`. Variant 2, so far so familiar.
The instructive part is that **three of the four turn a missing row into a decline
and the fourth turns it into wrong output**, because the strnum read-use gate does
not admit or refuse a program — it decides the key scalar's *representation*. An
unrecognised read deactivates strnum, `k` becomes a plain i64 counter, and the key
silently becomes the decimal of the field *parsed as a number* — `"0"` for `"INFO"`.
So while the other three gates were still refusing `c[k] += N` on its own, a program
mixing the spellings (`{ k = $1; c[k]++; c[k] += 2 }`) passed admission on the `++`,
registered its table on the `++`, **built, and printed an empty line where gawk
prints 3**.

Two things to carry forward. First, when auditing "N lists of one set", **sort the
sites by what happens when a row is missing**; the decline sites are self-reporting
and the representation-selecting ones are not, so they deserve the first look and the
tightest pin (here: assert at the IR level that the key is interned from the field
slice and never from a decimal). Second, a mixed-spelling program is the probe that
finds this class — each spelling alone declined honestly; only the two together
produced a build. **When two spellings of one operation have different coverage,
probe them in the same rule body**, not just side by side.

**Not every "missing row" is a missing row: two key spaces can share one surface
syntax.** `c[5]++` looks like the integer sibling of the `c["x"]++` that landed beside
it, and it is not. plawk has two array conventions and they share the spelling
`arr[N]`: awk semantics (keys are **strings**, so `arr[5]` is interned `"5"`) and
positional tables from `split()` / posarray binds (keyed by the **raw integer**
position, which is how `lookup_int` reads them). *Which space a program means is decided
by the table's kind, not by the subscript*, and the spec-level gate that would admit the
update does not know the kind. Admitting it compiles `{ c[5]++; print c[5] }` into a
store to interned `"5"` and a load from raw `5` — it **builds and prints four empty
lines** where gawk prints 1..4.

So the refusal is not conservatism about a shape; it is a real ambiguity that has to be
resolved before the shape can exist. The tell that it was this and not a missing row:
`c[$1,5]++` already works, so the integer literal builds fine as a *component* — the
same bytes are buildable and only the *lone* form is refused. **When one arity or one
operand position is refused while a strictly more general form works, look for a
collision rather than a gap.**

The collision also bit from the **other side**, and this is the part to remember: the
string-literal key change added a rule-body read `print c["x"]` on the same arity-1 key,
and that read made `{ split($0, a, " "); print a["1"] }` intern `"1"`, miss raw key `1`,
and print an empty line where gawk prints the field — **a pre-existing decline turned
into a wrong output by my own change**, caught only because I probed the *positional*
table after reasoning about the collision on the update side. The row came back out.
Lesson: once you have named a collision as the reason to refuse one direction, probe
**every** direction that now crosses it, including the ones you just added and believed
safe.

**Resolved, and the audit was the valuable part.** Probing the whole matrix (two table
kinds × four subscript forms × read/membership/delete/update) found that *three*
independent sites were each resolving a literal key on their own and disagreeing —
`lookup_int` read raw always, `assoc_delete_lit` interned always, the string-literal
membership probe interned always — and **all three were silent wrong outputs, none a
decline**, because every one of them produces a key that is *type-correct and wrong*
rather than absent. That is the signature of this variant: when the disagreement is over
a *value* both sides consider valid, nothing declines and nothing crashes.

One rule at plan time (`plawk_assoc_literal_key_space/4`) now answers for all of them.
Two details in it are worth keeping:

- **"Absent" must be a key that cannot exist, not a key that probably doesn't.** For
  `a["x"]` on a positional table the tempting shortcut is to intern `"x"` and probe with
  the result; atom ids are small sequential integers, so that id can *coincide* with a
  live position and report a hit on an unrelated element. Raw 0 is used because
  positional keys start at 1 — a correct answer instead of a lucky one.
- **Canonicality needs a round trip.** `number_string/2` reads `"01"` as 1, which would
  alias `a["01"]` onto slot 1; awk's split keys are `"1".."n"`, so `"01"` is a distinct
  absent key. The test requires the text to be byte-identical to the spelling of its own
  value.

The update side is not a resolution problem and does not get a key space: a positional
table's keys are integers, so `a["x"]++` cannot be represented and declines.

**A comment asserting agreement with a shared emitter is not evidence of it — and
two of three arms covered reads as covered.** Variant 2 again, third copy of the same
END scalar print, but the interesting part is entirely about why it survived. The
concat part emitter (`plawk_end_field_print_lines(var(Name), …)`) resolved a scalar's
slot itself and then called the **numeric** render whatever the slot kind was, so
`{ s = $2 } END { print "s=" s }` printed `s=25` — the interned **atom id** as an
i64 — where gawk prints `s=disk`, and an unset string slot printed `s=0` instead of
`s=`. Standalone `END { print s }` was correct throughout, ten lines away.

Two things kept it alive for as long as the emitter existed:

- **The comment on it was true and read as an audit.** It said it used "the SAME
  numeric render the standalone END print uses, so a counter or double in a
  concatenation cannot disagree with one printed on its own about whether an unset
  value is empty." Every word correct — and it names *two of the three arms* of a
  three-arm dispatch. A note about what is shared should say what is **not** shared;
  otherwise it certifies the cases that work and is silent exactly where the risk is.
- **The construct looked tested from every angle except the broken one.** Two suites
  already exercised a scalar in an END concat and both picked a working arm:
  `test_plawk_concat.pl`'s `concat_in_end` uses `s += $1` (a counter), and
  `test_plawk_unset_scalar.pl` pins the unset render for a counter and a double. When
  a construct dispatches on a kind, **walk the kinds as a row**; picking a
  representative is what hides this, and which representative you pick is arbitrary.

The fix was a deletion (defer to `plawk_end_scalar_var_print_lines/4`), so it landed
in all three END walkers at once and the 29 pre-existing golden-corpus programs stayed
byte-identical — none of them had a string scalar in a concat either, which is the
same coverage gap in a third place.

**A contract with only one side implemented — the variant where reading either file
alone shows no defect.** The strongest instance yet, and the one hardest to find by
inspection. Every numeric special on the RIGHT of an `if` / `while` condition
(`if (n < NF)`, `n < NR`, `n < ARGC`, `n < length`) silently compared against a phantom
variable worth 0, so those programs printed nothing where gawk printed records.

The codegen was **already correct and already complete** for that case:
`plawk_while_cond_build/8` carried a `cmp(Lhs, Op, special('NF'))` row for the reversed
operand order, `plawk_while_cond_operand/8` resolved RSTART/RLENGTH/ARGC/NR on *either*
side (it takes a `Side` argument), and `plawk_while_cond_rhs_ok/1` already deferred to
the validator for a `special(_)`. The parser simply never emitted a special on the
right. So the row sat unreachable while the identifier fallback manufactured a phantom.

This is not two implementations of one property disagreeing about a value — it is one
side of a contract never producing what the other side already handles. Neither file
reads as defective: the codegen looks complete, and the parser looks complete because a
fallback covers the case. Two things follow:

- **When you find a codegen row, check that something produces its term.** An emitter
  clause for a shape is evidence of intent, not of reachability. A grep for the term
  constructor on the *producing* side is the check, and it is cheap.
- **A fallback that cannot fail is what converts the gap into wrong output.** Every
  sibling production in the bare-pattern grammar guarded its identifier clause with
  `scalar_cmp_reserved_name/1`; the condition grammar's did not, so an untaught special
  became a variable rather than a decline. **Guard the catch-all**, and the next omission
  reports itself.

It also turned out to be **four** lists of one set, no two agreeing — two in the parser
(`match_special_name//1`, `special_cmp_operand//1`), the emitter rows, and the loop
validator `plawk_match_special/1` (whose name still says "specials set by `match()`",
variant 0 sitting on top of the duplication). The tell that there were four rather than
two: `if (n < length)` worked while `while (n < length)` declined, because only the loop
path consults the validator. **When two spellings of one construct disagree, keep
probing sibling contexts until they all agree** — the count of disagreeing lists is not
knowable from the first two.

**And a verification note this cost real time to learn: this class is invisible to build
status.** All four affected golden programs built *before and after*; only the IR and the
output changed. A sweep that watches exit codes cannot see it. Only gawk output
comparison can.

**When N walkers wrap one shared emitter, the walkers ARE the duplicated list.** The
strongest instance so far, and the one that paid best. Three END print walkers
(scalar / mixed / assoc) each enumerated the print-field vocabulary as clause heads,
and every clause was the same three steps — emit the separator, make ONE per-kind call,
recurse — where that per-kind call was *already exactly* what
`plawk_end_field_print_lines/4` makes for the same kind inside a concatenation. No
logic was duplicated. What was duplicated was the LIST of what may be printed, four
times, with nothing keeping the copies equal — and they had already drifted twice (a
string scalar in a concat printed its atom id; `NF` reached the routes one at a time).

Collapsing them to one delegating clause each took **24 clauses to 13** and turned every
future cell from "one clause per route" into "one clause, everywhere". `length` in END
was the first to land that way: a single row in the shared emitter appeared in the
straight-line print, in a concatenation, in a statement list and in all three routes at
once.

Three things generalise from it:

- **The tell is shape, not size.** A clause that is `Prelude, OneCall(Kind), Recurse`
  where `OneCall` already exists elsewhere for the same `Kind` is not code — it is a
  vocabulary entry. Count those, not lines: a walker with eight such clauses is an
  eight-item list that some other predicate also maintains.
- **A collapse must be provable as a collapse.** 32 golden-corpus programs came out
  byte-identical across this one; only the newly-admitted programs changed. Without that
  check "collapse" and "rewrite" are the same diff.
- **Expect a bonus cell, and pin it.** Delegating the assoc walker meant passing the
  EMPTY scalar plan (which it already did for concat parts), which made the shared
  generic-expression clause reachable, so `END { print 1 + 2, c["x"] }` went from decline
  to correct. An A/B over a 25-program matrix found it; it is now pinned in the tests,
  because an unintended behaviour change nobody wrote down is indistinguishable later
  from a defect.

**The same collapse one level down, and why it is the more useful half.** `NF` had an
`end_lastrec_nf` expression row that was its in-loop `nf` row with `%line` swapped for
the retained record Value; `length` had no such row, and that absence *was* the reason
every END form of `length` declined. Both are now entries in one table of
record-reading i64 leaves parameterised on **which record they read**, with one
retained-record wrapper covering every entry. The pattern is worth naming: when two
contexts differ only in **where a value comes from**, make the source a parameter of
one table rather than writing a second row per operation — otherwise every operation
pays the difference again, and the ones that never do quietly stay unreachable.

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

**Before adding a row per route, ask whether the form has a COMPILE-TIME answer.**
Every gap this campaign closed before the literal-builtin one was closed by teaching
an emitter a new cell, paid for one at a time — six field kinds across three END
walkers. `length("abc")` looked like more of the same: five builtins refused a string
literal in every route while accepting a field in all of them, which reads as a row
of missing cells. It was not. The *answer* to a builtin over a literal is itself a
literal, and literals were already in the vocabulary of every route, so folding at
parse time landed the whole family everywhere at once **with the codegen untouched**
— 41/41 golden-corpus programs byte-identical.

The generalisable question is not "is this foldable" but **what does the answer's
representation cost**. Here it cost nothing because the representation was already
universal. Two things make that check worth running explicitly:

- The check is cheap and mechanical. Build the answer's term by hand and ask the
  driver whether it compiles, in each route. That is what turned this from an
  estimated row of cells into a measurement — and it caught the one position where
  `int(N)` was *not* accepted (an END print field) before any code was written.
- Do it on the SURFACE, not on hand-built ASTs, once the routes involve a plan.
  Hand-building `program([], [rule(always, [inc(var(n)), inc_assoc(c-field(1))])], …)`
  produced a *decline* for a program whose surface equivalent compiles fine, because
  the hand-written term did not take the mixed route at all. Two conclusions were
  drawn from that before it was noticed. Hand-built terms are fine for the scalar
  routes and lie about anything plan-selected.

**Watch for the third statement of a vocabulary — the one that restricts by choosing
a narrow nonterminal instead of by a guard.** The literal-builtin refusal was eight
copies of `Field = field(_)` across two nonterminals, which is the familiar shape, and
they were found and replaced in one pass. The change then *half* worked:
`print length("abc")` compiled and `print length("abc") + 1` did not, because the
arithmetic-operand family stated the same vocabulary a third way — it passed a narrow
nonterminal (`simple_field_expr//1`, `$N` only) as the argument parser. Widening the
guards could not reach it.

Two lessons, and the second is the useful one. First, `grep` for the *guard text* finds
guard-shaped copies only; a restriction can also be spelled as a choice of production,
and that spelling leaves no text in common with the others. Second, the reason it was
easy to miss is that **its name described what it accepted rather than the role it
filled** — `simple_field_expr` names a shape, so nothing connected it to the
vocabulary it was a copy of. That is variant 0 of this campaign's defect class showing
up in a *nonterminal* rather than a gate, and the remedy is the same: it is now
`builtin_string_arg//1`, and it defers to the same `plawk_builtin_string_arg/1` the
other two families use. When hunting N copies of one decision, enumerate by the
decision's *role*, and treat every narrow nonterminal passed as an argument parser as
a candidate copy.

**A widened vocabulary can expose a shadowing producer that was harmless while the
vocabulary was narrow.** Relaxing the builtin argument guard made `length("abc")`
parseable — and revealed that the generic `name(args)` production had been capturing
any reserved builtin name whose arguments happened to be foreign-argument shaped, so
`length("abc")` parsed as a call to a *user Prolog predicate named `length`*. This was
pre-existing and it explained an exit-code split that had looked arbitrary:
`length("abc")` failed as a clean decline (exit 3 — it parsed as a prolog_call the
codegen refused) while `length(v)` failed as a parse error (exit 2 — `v` is not a
foreign argument either, so even the generic clause could not match). It also meant two
paths disagreed about identical text, since a bare print field reached the builtin
production first and an arithmetic operand reached the generic call first.

The tell was an **exit code that did not match the story**: a form outside the
supported surface should decline, but a form outside the *grammar* should be a parse
error, and `length("abc")` was giving the wrong one of the two. When a refusal's exit
code is surprising, the refusal is probably not coming from where you think — read the
parse, not the gate you expected to fire.

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
- **Do not run suites concurrently.** Two `test_plawk_ternary_str_branches` tests
  failed under parallel load and passed when the suite ran alone — a suite builds
  into a fixed path and runs the binary, and concurrent clang invocations make that
  flaky. It cost a wrong diagnosis (two tests reported broken that were fine). Run
  them sequentially, and when a failure looks surprising, re-run that suite by
  itself before believing it.
  **Third occurrence — and its stated cause was wrong, which the next entry explains.**
  `test_plawk_end_length` reported **5** failures while a 192-suite sweep was running and
  **2** on two consecutive re-runs alone (2 being the number the change predicted: the
  pins it was designed to flip). That was recorded as evidence that **load alone** was
  enough, on the grounds that "the two runs did not share build paths — the sweep sets
  its own `TMPDIR` and lives in a separate worktree precisely to avoid that."

  That premise was false. `TMPDIR` does not move where the suites build (next entry), so
  the two runs *were* writing the same `/tmp/uw_plawk_end_length/` paths, and this is the
  same collision as the two occurrences above rather than a new mechanism. The
  operational rule is unchanged and still right — **a suite count produced while any
  other heavy run is in flight is not a result** — but the reason is path collision, not
  load, and that matters because the fix differs: isolate the paths and concurrent runs
  become safe, whereas "load alone" implies they never can be.

  Worth keeping as a worked example of a hazard entry that recorded a correct
  observation with a wrong cause, and stayed wrong for as long as nobody checked the
  premise. The observation was solid; "did not share build paths" was an assumption
  about a flag's behaviour that no one had tested.
- **A broken harness looks exactly like a mass regression — prove the toolchain works
  before believing 13 red suites.** A sweep reported 13 suites failing en masse
  (`absent_key_read` 23 failed / 0 passed, `assoc_body_print` 15/2, `bare_print` 20/5,
  and ten more). Every one passed in isolation at the *same commit*. The cause was
  `TMPDIR` pointing at a directory deleted just before launch: clang writes its
  intermediates there, so every build died with `clang: error: unable to make
  temporary file: No such file or directory`, and the suites faithfully reported the
  missing binaries as failures.

  The shape to recognise: **the failures cluster in the suites that BUILD**, the
  pass/fail split inside each looks arbitrary, and parse-only suites are untouched.
  A real regression in an END emitter does not also take out `argv` and
  `begin_printf`. The sweep now creates its temp roots and compiles a two-line C file
  as a preflight, failing loudly instead of producing 190 suites of plausible red.
  **Preflight the harness, not just the code** — a verification run that cannot fail
  for harness reasons is worth the four lines it costs.
- **`TMPDIR` does not isolate the suites — SWI-Prolog's `tmp_dir` flag is `/tmp`
  regardless.** Two temp roots need redirecting and only one follows the environment:
  clang honours `TMPDIR`, while every suite's build directory comes from
  `current_prolog_flag(tmp_dir, …)`, which is `/tmp` whether or not `TMPDIR` is set
  (verified directly — `TMPDIR=/tmp/x swipl -g "current_prolog_flag(tmp_dir,D)"`
  prints `/tmp`). So the "isolated worktree sweep with its own TMPDIR" was isolated in
  **source** and never in **artifacts**: every suite wrote `/tmp/uw_plawk_<suite>/`,
  the same absolute paths a main-tree run uses.

  This is the actual mechanism behind all three phantom-failure occurrences above, and
  it means the remedy recorded for them ("run suites sequentially") was working for the
  wrong reason — sequential execution *avoids* the collision rather than isolating
  anything, so it was load-bearing in a way nobody knew. `sweep3.sh` moves both roots
  (`TMPDIR` for clang, `set_prolog_flag(tmp_dir, …)` prepended to each suite's goal)
  and **asserts the flag actually moved** before running anything, so a future SWI
  change cannot silently put the artifacts back in `/tmp` while the script still claims
  isolation. Verify what a flag *does*, not what its name suggests it reads from.
- **Match every plunit summary form.** A sweep grepping
  `"All N tests passed|tests failed"` silently misses the single-test form
  (`% test passed`) *and* the singular failure (`% 1 test failed`) — twelve failing
  suites reported as "no summary". Grep `^% .*(passed|failed)` instead. There is a
  *third* form: `% PL-Unit: <name>  passed 0.2 sec` with **no test count and no
  dots**, which means `% No tests to run` — every test filtered out by a
  `condition/1`.
- **A `condition/1`-gated suite that reports zero tests is not evidence about the
  feature — it is evidence about the machine.** Five suites
  (`cache_lmdb`, `multitable_lmdb`, `row_durable_lmdb`, `use_table_lmdb`,
  `use_namespace`) gate on `clang_lmdb_available`, which compiles and links a
  three-line LMDB C probe. Without `liblmdb-dev` they silently run **nothing** and a
  sweep summary still reads green. This cost something real: after four consecutive
  full sweeps in which the LMDB path had never executed, I reported it as
  unverified — and hedged a public claim the user was making about the project on
  that basis. **`apt-get install -y liblmdb-dev` was the whole fix**, after which all
  five suites pass, including `lmdb_histogram_persists` (an LMDB-backed histogram
  that persists and *accumulates across separate runs* of the compiled binary).
  Absence of evidence was a property of the container, not of the code. If a suite
  is gated, install the dependency and run it before drawing any conclusion from its
  silence — and never let a gated suite's silence reach a claim made outside the
  repo.

## Where the durable-store (DB) arc stands

Recorded because this campaign has been elsewhere for weeks and the next DB step is
easy to lose.

Landed and verified: the `BEGIN cache(...) { declare NAME }` surface (phase 1b) with a
**`file`** backend as the default; an opt-in **`lmdb`** backend (phase 5,
`backend "lmdb"`); row-oriented tables; a self-describing store that persists and
validates its schema on open (8.7); `use NAME` attaching to an existing store without
re-`declare` (8.8); and multiple named tables per store — namespaces over LMDB named
sub-DBs, `use ns.table` — (8.9). A multi-table *file* store is a deliberate class-A
compile error; only LMDB routes each table to its own sub-DB.

**Next: the views runtime.** `PLAWK_MULTIPASS_CACHE.md` §3.9 has column projection
landed and the `materialize NAME` surface landed (parsed, its reference validated
against the program's relations); what remains is materialise-and-cache — shared global
column tables, a one-time lazy fill behind a done-flag, and consumer rewiring — so two
passes over one relation stop materialising it twice.

**Why it has waited.** The gawk back-compatibility work displaced it, deliberately and
on the user's judgement: a surface that diverges from awk on ordinary programs is not
trustworthy no matter what the store underneath can do, so closing those divergences
outranks new store features. That has been the right trade — this stretch alone turned
up several *silent wrong outputs* against gawk (unset scalars, END record reads, the
two-key-space collision), each of which would have been inherited by any feature built
on top.

## Baseline

**Measured on `5009181d5` — the feature line with #4171, #4170 and the literal-builtin
fold: all 195 suites sweep clean, 2694 tests passed, 0 failed** (187 report the
`All N tests passed` form; 8 use the single-test form).

Composition, so the next total can be checked rather than trusted:

| tree | suites | tests |
|---|---|---|
| `d6cf0c9f1` (#4162) | 191 | 2586 |
| +#4171 (`end_length` 32, `cond_specials` 26) | 193 | 2644 |
| +#4170 (`end_if_nr` 14) | 194 | 2658 |
| +#4169 (`literal_builtin_fold` 36) | 195 | 2694 |

**Say which tree a baseline describes, and name the commit.** Three different totals
were quoted across four PRs in one day — 193/2644, 192/2622 and 195/2694 — all correct,
all for different trees, and the confusion cost real time. 193/2644 was measured on
#4164's *stacked tip* while the feature line still lacked those two suites, so PRs based
on the line reported totals *below* the recorded baseline and looked like regressions.
A total lower than the baseline is not automatically a regression: check what the
baseline was measured on first, because with stacked branches the answer is usually that
it was a different tree. The table above is the reconciliation; keep it updated with the
commit rather than replacing the number and losing the audit trail.

All five LMDB suites RUN — `liblmdb-dev` is installed in the container, so the
durable-store paths are genuinely exercised rather than skipped; a gated suite
reporting zero tests was always evidence about the machine, never about the code.

*(Earlier in the campaign this read "180 suites, 2323 tests" and went stale for
several PRs. Re-derive it rather than trusting it — and record the commit you
measured, which is what made the discrepancy above diagnosable in one step.)*

This was the first fully green plawk base in the
campaign — worth keeping that way, because twice a change landed on top of a red
suite and the breakage went unnoticed for several PRs (`bare_print` from #4108, and
seven broken `plawk_dyncall_support_ir` ladder rungs). Sweep before claiming a
change is clean, and run the suites **sequentially**.
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
suite · **unset scalars render empty** (monotonic assigned-mark) · **`arr[k]--` / `arr[k] -=
D`** for a field key · **a non-literal `-=` delta** (`n -= $2`, by negating instead of
subtracting) · **scalar-var assoc keys for the whole `add_assoc` family** (four lists
of one set collapsed onto `plawk_assoc_scalar_key_update/4`; one of them was selecting
the key's representation and producing wrong output) · **string-literal assoc keys**
(`c["total"]++`) — the multi-dimensional key builder at arity 1, one relaxed guard (and
one row added, then removed after it turned a decline into a wrong output on positional
tables) · **the positional/interned key-space collision resolved** by one plan-time rule,
fixing three silent wrong outputs (`print c[5]` on an awk table, `delete a[1]` and
`"1" in a` on a split table) and restoring the withdrawn rule-body literal read.

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
**`length`** in END — bare `END { print length }` and `END { print length($0) }`,
which still decline. **Not** `END { print length("abc") }`, which the literal fold
answers at parse time and which is done; the two share a spelling and nothing else,
since one needs the retained record and the other needs no runtime at all.

  This row said "already in the gate, not yet emitted" for most of the campaign and
  **that was wrong** — a correction worth keeping, because it is a defect shape of its
  own. The gate clause is `plawk_end_term_reads_record(special(length))`, and
  `special(length)` occurs in **exactly one place in the repository**: that clause. No
  producer emits it. Both END spellings parse to `length(field(0))`, and the `length`
  *pattern* form uses the bare atom (`special_cmp(length, gt, 3)`), so the clause
  matches nothing and the record is **not** retained for `END { print length }`.

  So the work is not "add an emitter to a gate that already fires" but "teach the gate
  the shape the parser actually produces, then emit" — a different and larger job than
  the row advertised. The general lesson: a gate clause written for a term shape no
  producer emits **reads as done** and sizes the follow-on wrongly, and no test can
  catch it because there is no program that reaches it. `grep` for the *producer* of a
  shape before believing a gate covers it — the campaign's "grep for the behaviour,
  not recollection" rule applies to audit rows as much as to test pins, and this row
  was written by reading the gate rather than probing the program. · **builtins over the
record** in END (`substr($0, …)`, `toupper($1)` — the gate retains for them, the
emitters have no clause; the *literal* forms of all five are done) · **END-only**
programs (a driver with no retain) ·
`printf` field args in the **assoc / mixed END chain** (a different driver, passes
`no_end_record`) · the **associative** END-`if` branch (refused by
`plawk_assoc_end_if_branch_prints_ok/2`, which allows only string literals there —
an unrelated pre-existing restriction, pinned as such) · `$N` in a `binfmt` END
(reads the record buffer, not a text slice).

An **integer-literal** assoc key (`c[5]++`) — blocked by the raw-integer/interned-text
key-space collision above, not by a missing row · **`delete arr[N]` on a positional
· a **non-literal delta at a scalar-var key** (`c[k] += $2` declines, `c[k] -= $2` is a parse error;
the field-key `c[$1] += $2` works, and the scalar `n -= $2` now works, so this is the
last non-literal-delta hole) · assoc rules
alongside a scalar END loop (state-plan boundary) · float-literal and non-literal
ternary arms (need runtime number→string) · `dec` row in the binfmt action gate ·
the dead `@.plawk_surface_print_line` global (removal costs byte-identity on every
program, so its own PR) · `printf`/`NR` as plain statements in the for-in END
chain · `NR` in a loop-free END list · `printf "%c"` on a string ·
autovivification on read · scalar-var SUBSEP components · empty-string subscripts.

**The string-builtin argument vocabulary, what is left** — pinned in
`tests/test_plawk_literal_builtin_fold.pl`, and the three are genuinely different
problems despite reading as one list. A **variable** argument (`v = $2; length(v)`)
is the one that actually needs codegen: the value is not known until runtime, so no
fold can answer it and it wants the runtime projection a field argument already gets
— probably the largest of the three and the most useful. A **nested** call
(`length(toupper("ab"))`) needs only the vocabulary to admit a call, because
`plawk_fold_literal_builtins/2` is written bottom-up and would collapse it in one
pass; the fold is already correct for it, so this is a parser-side change with no
emitter work. `int("3.7")` is deliberately excluded rather than merely unimplemented:
int-over-a-string is numeric coercion with strtod prefix semantics ("3.7abc" is 3,
"abc" is 0, leading whitespace and sign allowed) and belongs with a
string→number fold of its own, tested against those edges, not bolted onto the
string-builtin vocabulary. Also open: **non-ASCII literals**, which decline today
because the fold refuses anything above code point 127 to stay byte-exact with the
runtime — closing it means computing UTF-8 byte offsets in the fold (exact and easy
for `length`, fiddlier for `substr`, whose byte slice can split a character exactly
as the runtime's does).
