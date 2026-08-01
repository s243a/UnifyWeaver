# Report: pattern_stache second consumer — constraints as dispatch keys

*Follow-up to [`REPORT_pattern_stache_prototype.md`](REPORT_pattern_stache_prototype.md) (#4052).
The first consumer — goals → typed AST nodes — needed exactly four pattern shapes, all
first-order, linear, guard-free. That was evidence from one consumer; specifying from one risks
over-fitting. This report applies the same dispatcher, unchanged in its matching machinery, to
the second consumer named in
[`docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md`](../../docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md):
constraint goals selecting their checker by shape.*

**Status: prototype. Core and `templates/` remain untouched.** All new code lives in
[`pattern_stache/`](pattern_stache/).

## What was built

| file | contents |
|---|---|
| `pattern_stache/constraint_check.stache` | the consumer template: each constraint form maps to `checker(How, phase(P))` — checker selection *is* the rendering |
| `pattern_stache/constraint_dispatch.pl` | the driver: mode split (ground vs residual), plan read-back, discharge ledger; executes exactly one checker kind (closed-fact lookup) to show the loop closes |
| `pattern_stache/test_constraint_dispatch.pl` | 12 plunit tests: end-to-end ledger, refinement, compound capture, mode split, extensibility, quoting |

The dispatcher itself (`pattern_stache.pl`) changed in exactly one place: `{{q:Key}}` quoted
interpolation was added to `substitute_placeholders/3` (~10 lines). No matching, scoping,
loading, or overlap machinery changed. That stability across consumers is itself a data point.

Run (from `prototypes/mu_cosine/pattern_stache/`):

```bash
swipl -g run_tests -t halt test_constraint_dispatch.pl   # 12 tests
swipl -g run_tests -t halt test_pattern_stache.pl        # 38 tests, unchanged
swipl -g constraint_dispatch:demo -t halt constraint_dispatch.pl
```

Demo, end to end — the §3 mode table of
[`DESIGN_desugaring_to_prolog_goals.md`](DESIGN_desugaring_to_prolog_goals.md) as a running
ledger:

```text
has_type(principal_tree(pearltrees),substrate(pearltrees))
    => selected(..., pt_lineage_walk(principal_tree(pearltrees)))
has_type(t7,substrate(wikipedia))
    => selected(..., substrate_table(t7,wikipedia))
non_amplifying(min)
    => discharged(non_amplifying(min))
non_amplifying(sum)
    => failed(non_amplifying(sum))
mu_bounded(path,0.5)
    => obligation(..., property_test(mu_bounded(path,0.5)))
in_support(decay,harvest_2026)
    => runtime(..., corpus_data(in_support(decay,harvest_2026)))
has_type(_X,substrate(_C))
    => residual(has_type(_X,substrate(_C)))
frobnicate(x)
    => no_checker(frobnicate(x))
```

Adding a constraint form is adding a `{{case}}`; the driver is untouched (tested:
`new_constraint_form_needs_no_driver_change` extends the template with `disjoint(A, B)` and the
unchanged driver ledgers it correctly). That is the "extensible instead of enumerated" claim,
demonstrated rather than asserted — compare `process_expression_vnext/elaborator.py`, which
hard-codes the single goal type it discharges.

## Do the four shapes still suffice?

**Yes, with one honest refinement of the census.** Every pattern in `constraint_check.stache`
is first-order, linear, guard-free unification against a ground term — the same *rule* the four
shapes instantiated. Consumer 2 stayed inside that rule but used one member of it that consumer
1 never did:

> `has_type(X, substrate(pearltrees))` — a pattern with a **ground leaf where consumer 1 always
> had a variable**.

It costs nothing (unification handles it by construction) but it is load-bearing: it is the
specific half of the refinement idiom — pearltrees gets its dedicated lineage-walk checker,
every other corpus falls through to the generic `substrate_table` case. So the accurate census
across both consumers is not "four shapes" but **one rule with five observed members**:

| # | shape | consumer |
|---|---|---|
| 1 | compound, depth 2, two variables — `has_type(X, substrate(C))` | both |
| 2 | same, sibling functor — `has_type(X, judge(J))` | both |
| 3 | compound, depth 1, one variable — `non_amplifying(Op)` | both |
| 4 | compound, depth 1, two variables, numeric leaf — `mu_bounded(Op, B)` | both |
| 5 | mixed ground-leaf/variable — `has_type(X, substrate(pearltrees))` | constraint dispatch only |

The rule that covers all five: **any first-order, linear, guard-free term, matched by
unification against a ground value**. Nothing outside it appeared.

Also new on the *value* side, not the pattern side: a variable capturing a whole **compound
subterm** (`X = principal_tree(pearltrees)`), where consumer 1's bindings were all atoms and
numbers. Round-trips through render-and-read-back (tested).

## Did constraint dispatch need guards, arithmetic, or list patterns?

**No — and in each case the reason is structural, not luck:**

- **Guards.** The one guard-shaped question — is this goal ground enough to dispatch? — is
  answered *upstream* of dispatch by discharge ordering, exactly as the desugaring doc's rule
  ("dispatch only on goals guaranteed discharged") prescribes. The driver routes nonground goals
  to `residual/1` before the template is consulted (tested), and the dict contract's
  `nonground_dispatch` error from consumer 1 sits behind it as defense in depth (also tested).
  The mode discipline lives in the driver; the pattern language never sees the question.
- **Arithmetic.** `mu_bounded(path, 0.5)` dispatches to a property-test *obligation* precisely
  because a numeric constraint is not checkable by unification. Dispatch has nothing arithmetic
  to do — the number is carried, not computed on. Any arithmetic in patterns would be the
  matcher pretending to prove what it cannot.
- **List patterns.** Iteration over the goal store belongs to the driver (`maplist/3`); the
  template sees one goal at a time. The same driver-iterates/template-dispatches boundary held
  in consumer 1 and was never strained here.

Multi-key dispatch (goal × phase) also never became necessary: the case body *states* the phase;
nested `{{match}}` exists if a real consumer ever needs a second key.

## What constraint dispatch DID need that AST emission didn't

**Re-readable interpolation — `{{q:Key}}` — and it is an interpolation form, not a pattern
shape.** Consumer 1 rendered display text; consumer 2's rendered output is **read back as a
term** (`checker(How, phase(P))`) by the driver. Report 1 flagged plain-`~w` rendering as not
re-readable and deferred a quoted form for lack of demand; consumer 2 produced the demand. The
evidence pair is in `constraint_quoting`:

- with plain `{{S}}`, the value `'my account'` renders unquoted and the plan line no longer
  parses — named error `unreadable_plan(Goal, Line, syntax_error(...))`;
- with `{{q:S}}`, it round-trips exactly; and `~q` is invisible on plain atoms and numbers
  (`min`, `0.5` render identically in both forms).

Two consequences worth recording. First, the philosophy doc's escalation note — templates that
emit *structures* rather than display text move toward being identity-bearing — is no longer
hypothetical pressure: the second consumer's output is structure by design. Nothing here makes
templates identity-bearing, but the spec should acknowledge the direction. Second, the phase
vocabulary (`elaboration | obligation | runtime | no_checker`) is the **closed** half of the
contract — the driver fails closed on an unknown phase (tested) — while constraint forms are the
open half. Extensibility has an axis, and naming it prevents the template from silently growing
a private protocol.

## The refinement idiom is now load-bearing

Consumer 1 exercised specific-before-general only in tests. Consumer 2 needs it in its real
template: `substrate(pearltrees)` above `substrate(C)`. The load-time overlap trichotomy from
#4052 admits this silently (later-subsumes-earlier) while still hard-erroring unreachable cases —
a flat "forbid overlap" rule, the philosophy doc's original first option, **would have rejected
this consumer's natural template**. The trichotomy is validated by use, not just by argument.

## Verdict: green light for a SPECIFICATION

The four shapes — better: the one rule, first-order, linear, guard-free unification against
ground terms, with the five observed members above — **held across both consumers named by the
philosophy doc**. Per the task's own criterion, that is the green light. The specification
should be bounded to exactly:

- patterns: the one rule above, nothing more;
- interpolation: `{{Key}}` (display, `~w`) and `{{q:Key}}` (re-readable, `~q`) — both evidenced;
- dispatch: ground dict values, commit-at-unification, scoped child-dict bindings, load-time
  overlap trichotomy, fail-closed versioned loader — all as answered in #4052;

with extension points **named but not designed**, none having produced a single real pattern
across two consumers: guards, pattern arithmetic, list patterns, regex/glob cases, multi-key
dispatch, partials/delegation.

Unchanged from before: this improves *dispatch*. Closed-fact constraints remain lookups
(demonstrated: the one executed checker), numeric constraints remain property-test obligations,
and nothing here lets unification prove more than it could.
