# Report: pattern_stache third consumer — goal terms → v0.4 canonical process expressions

*Fourth report in the series
([prototype](REPORT_pattern_stache_prototype.md) →
[second consumer](REPORT_pattern_stache_second_consumer.md) →
[graduation](REPORT_pattern_stache_graduation.md) → this). The first consumer with a sealed
external target: a template-driven emitter from Prolog goal terms to registry-v0.4 canonical
process expressions, verified byte-exact against every row of the sealed
`PROCESS_EXPRESSION_GOLDEN_v3.json`. Built on the PRODUCTION engine
(`src/unifyweaver/core/pattern_stache.pl`); the two prototype witnesses in
`pattern_stache/` are untouched.*

## Verdict first: the witnessed grammar SUFFICED — the stop-clause did not fire

Every template case this consumer needed is a first-order, linear, guard-free pattern — the
same rule the first two consumers established. Byte-exact rendering exercised exactly the four
things the task predicted it would (kwarg ordering, default resolution, nested composition,
quoting of enum values), and **all four landed in the driver, none in the pattern grammar** —
which is where the transpiler-target architecture puts them anyway (the driver plays
`common_generator` + `Config`; the templates play the per-target mustache library). No dialect
v2 material surfaced. Details and honest boundary observations below.

## What was built

| file | contents |
|---|---|
| `pattern_stache/pe_surface.stache` | 16 cases: one per v0.4 operator (`e5`…`lca_frac`, incl. the golden-unexercised `pick`) plus the three structural forms `bare/1`, `modded/2`, `pinned/2` |
| `pattern_stache/pe_value.stache` | 4 cases: `kw(K,V)`, `str(S)`, `list(Items)`, `lit(X)` — the value-syntax layer (`=`, quotes, brackets) |
| `pattern_stache/pe_emit.pl` | the driver: goal convention, v0.4 registry mirror (the Config), introspection, recursion, canonicalization |
| `pattern_stache/test_pe_emit.pl` | 60 tests: all 25 golden rows byte-exact on **both** `canonical_identity_string` and `canonical_full_string`, plus unit tests |

Run (from `prototypes/mu_cosine/pattern_stache/`):

```bash
swipl -g run_tests -t halt test_pe_emit.pl    # 60 tests, incl. 25×2 golden byte checks
```

## The goal convention (documented, as required)

One concrete convention, chosen to look like the elaborated goals of consumer 1:

```prolog
lineage(pearltrees, mu(haiku), estimand(ancestry))
    %  =>  lineage(pearltrees,decay=0.85,estimand="ancestry",mu=haiku)
```

- positional args first, then kwargs as **unary compounds** named by the kwarg
  (unambiguous under v0.4: no kwarg name collides with an operator or atom name);
- `mod(Base, M)` for modified atoms (`mod(luna,'D')` → `luna.D`);
- `pin(Expr, Pin)` for pins (`@run/2026-07-25`, full form only);
- Prolog numbers, lists, and strings for the value kinds; enum values as atoms
  (`estimand(ancestry)`).

## All 25 rows accounted for

All 25 pass byte-exact, semantic and full forms both; none needed individual accounting. The
suite additionally proves the bijection (every golden row has a goal, every goal a row), the
pinned row's exact semantic/full divergence, kwarg-order insensitivity of the goal form,
explicit-default idempotence (`decay(0.85)` given ≡ omitted), JSON escaping (`"a\"b\\c"`,
control characters, UTF-8 passthrough), and per-node pin strip/keep matching
`process_cards._canonical`.

## Where each byte-exact pressure landed

| pressure | landed in | how |
|---|---|---|
| kwarg sorting | driver | `msort/2` on `Key-Value` pairs; atom order = the golden order (`manifest<menus<t`, `decay<estimand<mu`) |
| default resolution | driver (registry mirror) | v0.4 has exactly one default (`lineage` `decay=0.85`); injected when absent |
| nested composition | driver recursion | children emitted first; the template dispatches ONE node on its structural form — the same driver-iterates/template-dispatches boundary as consumers 1–2 |
| enum/string quoting | driver (+ `str/1` template case) | JSON escaping is character rewriting → driver; the quotes themselves are template syntax (`"{{S}}"`) |
| comma placement | driver (`kw_tail/3`) | the positional/kwarg join comma exists only when both sides are non-empty |

## Findings on the dialect (the reason this consumer exists)

**1. `{{q:Key}}` was used ZERO times.** The second consumer's quoted interpolation serves
Prolog-read-back output; this consumer's target language (the v0.4 surface) has *its own*
literal syntax — JSON strings — which is neither `~w` nor `~q`. So value formatting moved to
the driver, exactly as every transpiler target's Config owns `atom_fmt`. This refines the
second report's framing: interpolation forms are **target-language properties**. `{{Key}}`
(display) plus driver-side formatting covered a byte-exact external target; no third
interpolation form is warranted — a `{{json:Key}}` would be building per-target knowledge into
a dialect that 44 target directories would then each want a variant of.

**2. Fixed arity pushed variadic operators into a pre-joined slot.** `blend/product/max` take
open-ended args; their template cases take one slot (`blend({{Args}}{{kw}})`) with the driver
joining. A list pattern would have expressed this in-template — this is the first *observed
pressure* on the "list patterns" exclusion, but not a need: pre-joining is precisely what the
existing targets do, costs three lines, and keeps the template first-order. The spec's revisit
condition ("a consumer that cannot iterate outside the template") was checked and not met.

**3. Uniform surface syntax makes per-operator dispatch earn less here.** All 13 called forms
render as `name(args,kwargs)`, so the 13 operator cases differ only in name and arity. The
value of one-case-per-operator here is the **template-as-registry-library** form the owner's
direction asks for (one case = one clause head per operator, extendable by adding a case), not
shape discrimination. The structural dispatch that *earns* its keep in this consumer is the
form layer (`bare`/`modded`/`pinned`) and the value layer (`kw`/`str`/`list`/`lit`). Recorded
so the eventual cross-consumer accounting stays honest.

**4. One parity assumption is verified only corpus-wide, not in general.** Number spelling
relies on SWI `~w` agreeing with Python `repr` — true for every float in the golden corpus
(verified by the byte tests), but scientific-notation floats (e.g. `1e-05`) could diverge and
no golden row exercises them. Flagged rather than solved: the bytes are the contract, and the
corpus defines the witnessed range.

**5. `pick` is the one registry operator with no golden row.** Its case exists in the template
for registry completeness but is untested by sealed bytes — inference, not measurement, per
PROJECT_PHILOSOPHY §8.

## Pattern census update (cross-consumer)

All new cases stay inside the one rule (first-order, linear, guard-free). New members beyond
the graduated census: fixed-arity operator heads over pre-rendered strings (`routing(S, J)`),
zero-variable atom cases (`margin`), and the two-variable syntax carriers (`kw(K, V)`,
`modded(Base, Mod)`, `pinned(Expr, Pin)`). Nothing outside the rule appeared; nothing in the
rule was missing.

## Spec recording task (item 4)

The refuse-non-linear / tolerate-nesting asymmetry rationale is now recorded in
`SPEC_pattern_stache.md` ("Why non-linearity is refused while nesting is tolerated"): a bare
`f(X, X)` makes the second occurrence an implicit equality test with nothing at the syntax
level marking it — the same unmarked-divergence hazard class as unquoted `Helpers` — while
nesting changes the meaning of nothing around it and is unpromised only for lack of a witness.
Marked flagged-and-recommended, pending the owner's explicit ratification, per the graduation
report.
