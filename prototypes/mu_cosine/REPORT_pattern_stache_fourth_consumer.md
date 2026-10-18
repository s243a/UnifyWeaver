# Report: pattern_stache fourth consumer — the where-clause form over pe_emit

*Fifth report in the series
([prototype](REPORT_pattern_stache_prototype.md) →
[second consumer](REPORT_pattern_stache_second_consumer.md) →
[graduation](REPORT_pattern_stache_graduation.md) →
[third consumer](REPORT_pattern_stache_third_consumer.md) → this). First running code for a
piece of recorded vNext surface design: the where-clause binding form of
`DESIGN_desugaring_to_prolog_goals.md` §12, implemented as an input convention in front of the
third consumer's `pe_emit`, per `DESIGN_registry_v0.4.md` §10.1 ("variables stay OUT of the
sealed grammar"). Implemented, not extended: no change to the v0.4 surface, the registry, the
sealed bundles, the production engine, or the three witness consumers.*

## Verdict first: absorbed entirely by the driver architecture — zero pressure on the pattern grammar

The expected answer held, and more strongly than expected: the where-form contributed **zero
new template cases** and never touches the dialect at all. Elaboration — substitution,
occurrence checking, position legality — is driver work discharged strictly *upstream* of
`pe_emit`, exactly like kwarg ordering and default resolution in consumer 3, and exactly like
the mode split in consumer 2: things that vanish before dispatch never become pattern-grammar
demands. The stop-clause did not fire.

## What was built

| file | contents |
|---|---|
| `pattern_stache/pe_where.pl` | the where-elaborator: convention, validation (all fail closed), substitution, discharge into the unchanged `pe_emit` path |
| `pattern_stache/test_pe_where.pl` | 50 tests: the §12 measurement against sealed bytes (11 spellings × semantic + full + structural), the discharge property, and the fail-closed battery |

Run (from `prototypes/mu_cosine/pattern_stache/`):

```bash
swipl -g run_tests -t halt test_pe_where.pl    # 50 tests
```

## The where convention (documented, as required)

```prolog
where(product(hop_decay(C, gamma(0.6)), lca_frac(C)), [C = simplemind])
```

`where(Expr, Bindings)`: `Expr` is a goal term under `pe_emit`'s convention with Prolog
**variables** permitted in value positions; `Bindings` is a list of `Var = Value`. Because the
variables are real Prolog variables, **two occurrences of `C` are one variable, so one binding
reaches both by construction** — §12's enforced functional connection ("a mismatch cannot be
written") comes free from the term representation, not from a check. The caller's term is never
mutated (`copy_term` first, sharing preserved).

## The §12 measurement, re-proven in Prolog against the bundle

Eleven where-spellings were verified three ways each — the where-spelling's output, the direct
repeated-literal spelling's output, and the sealed golden bytes must all agree, on both the
semantic and full surfaces; additionally the *elaborated goal term itself* must be structurally
identical (`==`) to the direct goal:

- **all five golden rows with a repeated atom in bindable positions**, the §12 case proper —
  `blend`, `blend-variadic`, `dir-blend`, `graph-judge` (the §10.1 registered spelling itself),
  `kalman-fused` — each with the repetition factored into one binding reaching two or three
  occurrences;
- **six representatives** covering every value kind a binding can carry: a number (`t=0.03`), a
  whole list (`t([T])`, `T=[1]`), a UTF-8 string (`manifest`), a judge expression + enum atom +
  corpus atom in one three-binding clause (`lineage-haiku`), a binding inside the semantic part
  of a **pinned** expression (full form byte-equal to the pinned golden row), and a variable at
  the root (`where(C, [C = fs])`).

**The discharge property as a test**: a where-term built *from text* with a named variable
(`C`) elaborates to output byte-equal to the `graph-judge` golden row that contains no `where`
substring, no `C`, and no uppercase character at all — the binding provably vanished at
grep level, on top of the byte equality that already implies it. A companion test proves the
caller's own term still holds its unbound variable afterward.

## Fail-closed cases (all tested, all named errors)

| case | error |
|---|---|
| variable left unbound at emit time | `unbound_after_elaboration(Goal)` — ground emission only; residuation is the elaborator's future, not this driver's |
| binding for a variable that does not occur | `dead_binding(V = Val)` — dead bindings hide typos |
| duplicate bindings for one variable | `duplicate_binding_for_one_variable` — refused even when the values agree; a duplicate is a spelling accident |
| value illegal for a position it reaches | `illegal_binding_value(Val, at(Position))` with the position named: `kwarg(margin,t)`, `list_elem(routing,t)`, `arg(lca_frac,1)`, `mod_base` — checked for **every** occurrence the variable reaches, before any substitution |
| binding variable in a pin position | `binding_reaches_pin_channel(pin_name)` |
| pin smuggled in as a binding value | `illegal_binding_value(pin(...), at(...))` |
| malformed binding list / element / bound LHS / non-where term | `bad_binding_list`, `bad_binding`, `not_a_where_term` |

Legality is **structural** (what `pe_emit` can render at that position, per its registry
mirror's kwarg kinds) — not output-type checking. That `hop_decay` and `lca_frac` received the
*same* substrate is enforced here by variable identity; that the substrate is *type-correct*
for both remains vNext's job, and nothing here claims otherwise.

## One interpretation flagged for the owner

§12 rules that bindings "never reach a pin channel" (identity-determining vs.
identity-transparent). This driver reads that as **channel disjointness in both directions**: a
binding variable sitting in a pin position is refused, and a `pin/2` term arriving *as* a
binding value is refused — provenance cannot pass through the binding channel, and bindings
cannot construct provenance. A binding inside the *semantic part* of a pinned expression is of
course fine (verified byte-exact against the pinned golden row). If the owner reads §12 as
constraining only the envelope routing, the two refusals are one predicate each to relax.

## Implementation notes worth recording

- **`findall/3` copies variables.** The first draft collected occurrence positions with
  `findall`, whose template copying silently broke the `==`-identity that "two occurrences are
  one variable" depends on — the exact property this consumer exists to demonstrate. Caught by
  the fail-closed test battery (a list-element check that never fired); fixed by collecting
  with a plain walk. Recorded because it is the one Prolog-representation trap in the §12
  design: anything that copies terms severs the binding connection.
- **`pe_emit` is consulted, not changed.** The elaborator reads consumer 3's non-exported
  registry mirror by module-qualified calls (`pe_emit:pe_kwspec/4` etc.). The alternative —
  duplicating the mirror — would create drift between the two files; the liberty is confined to
  a prototype consuming its own lane's prototype, and is flagged here rather than hidden.
- The vNext Python machinery was not imported, wrapped, or ported; the oracle is the sealed
  bundle bytes throughout. Byte-agreement with vNext's `ground()` is left as the pleasant fact
  §12 already measured on its side.

## Cross-consumer accounting update

Four consumers now stand on the same engine, matching machinery untouched since graduation:
AST emission (bindings into display text), constraint dispatch (bindings read back as terms),
v0.4 canonical emission (dispatch under a sealed byte contract), and the where-form (discharge
*before* dispatch). The pattern census is unchanged by consumer 4 — nothing new reached a
template. The recurring architectural fact, now witnessed three different ways (mode split,
value formatting, binding discharge): **work that can be done before or around dispatch never
becomes a pattern-language feature request.** That is the strongest standing argument that the
v1 grammar is correctly sized.
