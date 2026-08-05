# Tutorial: the pattern_stache stack, end to end

*Every example below is runnable from the repository root, and every shown output is
**verified, not pasted**: `docs/tutorials/check_tutorial.py` extracts each command block from
this file, executes it, and fails CI if the real output differs from what you read here. If
this tutorial drifts from the code, a test breaks.*

## What this stack is, in one breath per layer

**Purpose.** Dispatch on the *shape* of a Prolog term instead of the spelling of a string, and
build emitters and elaborators on top of that dispatch — verified against sealed byte oracles.

**Input.** Prolog terms: template dicts (`Key=Value` lists with ground terms as values), goal
terms (`lineage(pearltrees, mu(haiku))`), goal stores (constraint and binding goals).

**Output.** Rendered text (template layer), registry-v0.4 canonical process-expression strings
(emitter layer), or a typed elaboration state — `ground(Term)` or `pattern(Term, Store)`
(elaborator layer).

The layers, one sentence each, bottom to top:

1. **`pattern_stache`** (`src/unifyweaver/core/pattern_stache.pl`) — a template dialect where
   `{{case}}` values are Prolog terms matched by unification, and the variables a match binds
   become available to that case's body.
2. **`pe_emit`** (`prototypes/mu_cosine/pattern_stache/`) — a template-driven emitter from
   goal terms to v0.4 canonical surface text, byte-exact against the sealed golden bundle.
3. **`pe_where`** — a `where`-clause front end: one binding substitutes every occurrence of a
   variable, then vanishes; the output is byte-identical to writing the literal twice.
4. **`pe_elaborate`** — the elaborator: ground goals discharge (checks are consumed, bindings
   substitute), under-instantiated goals residuate and travel with the term as a pattern
   state.

Before each output block, try to predict what it will say — the point of the examples is that
you can.

## a. Hello, structural dispatch

A `.stache` file must start with the versioned pragma — a file without it, or with a version
the loader doesn't implement, is an error, never a guess. Create a template with two term
patterns and a default:

```console
$ mkdir -p /tmp/pattern_stache_tutorial && printf '%s\n' '{{! dialect(pattern_stache, 1) }}' '{{match goal}}{{case has_type(X, substrate(C))}}term {{X}} walks corpus {{C}}{{case non_amplifying(Op)}}operator {{Op}} composes safely{{default}}no case matched: {{goal}}{{/match}}' > /tmp/pattern_stache_tutorial/hello.stache && cat /tmp/pattern_stache_tutorial/hello.stache
{{! dialect(pattern_stache, 1) }}
{{match goal}}{{case has_type(X, substrate(C))}}term {{X}} walks corpus {{C}}{{case non_amplifying(Op)}}operator {{Op}} composes safely{{default}}no case matched: {{goal}}{{/match}}
```

Now dispatch a term against it. The dict value `has_type(x, substrate(pearltrees))` is a
*term*, not a string; the first case's pattern unifies with it, binding `X=x` and
`C=pearltrees`, and those bindings fill the body's `{{X}}` and `{{C}}`:

```console
$ swipl -g "use_module('src/unifyweaver/core/pattern_stache'), render_stache_file('/tmp/pattern_stache_tutorial/hello.stache', [goal=has_type(x, substrate(pearltrees))], R), write(R), nl" -t halt
term x walks corpus pearltrees
```

**Why this matters:** one case serves *every* corpus. Under string matching you would need one
case per spelling — `has_type_x_substrate_pearltrees`, and so on forever.

A different shape selects a different case:

```console
$ swipl -g "use_module('src/unifyweaver/core/pattern_stache'), render_stache_file('/tmp/pattern_stache_tutorial/hello.stache', [goal=non_amplifying(product)], R), write(R), nl" -t halt
operator product composes safely
```

And an unmatched shape falls to `{{default}}`, which can interpolate the whole term:

```console
$ swipl -g "use_module('src/unifyweaver/core/pattern_stache'), render_stache_file('/tmp/pattern_stache_tutorial/hello.stache', [goal=mystery(42)], R), write(R), nl" -t halt
no case matched: mystery(42)
```

### The silent hazard: bare `Helpers` is a variable

An uppercase-initial case value reads as a Prolog **variable**, which unifies with anything.
A single such case swallows every input — silently:

```console
$ printf '%s\n' '{{! dialect(pattern_stache, 1) }}' '{{match tag}}{{case Helpers}}swallowed {{Helpers}}{{/match}}' > /tmp/pattern_stache_tutorial/hazard.stache && swipl -g "use_module('src/unifyweaver/core/pattern_stache'), render_stache_file('/tmp/pattern_stache_tutorial/hazard.stache', [tag=release], R), write(R), nl" -t halt
swallowed release
```

The author almost certainly meant the *literal* tag `Helpers`. Quoting restores that meaning —
the quoted case matches only the atom `'Helpers'` (the first render below finds no match and
no default, so it produces the empty string, printed quoted so you can see it):

```console
$ printf '{{! dialect(pattern_stache, 1) }}\n{{match tag}}{{case %s}}literal match{{/match}}' "'Helpers'" > /tmp/pattern_stache_tutorial/quoted.stache && swipl -g "use_module('src/unifyweaver/core/pattern_stache'), render_stache_file('/tmp/pattern_stache_tutorial/quoted.stache', [tag=release], R1), format('~q~n', [R1]), render_stache_file('/tmp/pattern_stache_tutorial/quoted.stache', [tag='Helpers'], R2), write(R2), nl" -t halt
""
literal match
```

The hazard is fully silent only in a single-case block. The moment any case follows the bare
variable, the load-time overlap check catches it — a variable pattern subsumes everything, so
the later case is provably unreachable, and unreachable cases are load **errors**:

```console
$ printf '%s\n' '{{! dialect(pattern_stache, 1) }}' '{{match tag}}{{case Helpers}}A{{case helpers}}B{{/match}}' > /tmp/pattern_stache_tutorial/guarded.stache && swipl -g "use_module('src/unifyweaver/core/pattern_stache'), catch(load_stache_file('/tmp/pattern_stache_tutorial/guarded.stache', _), error(pattern_stache(unreachable_case(A, B, _)), _), format('~q~n', [unreachable_case(A, B)]))" -t halt
unreachable_case(case_index(2),subsumed_by(case_index(1)))
```

**Why the refusal:** a dead case means the file's author believes something about their
dispatch that is false. Failing at load is cheaper than debugging silent output later.

## b. Goal term → v0.4 canonical surface, against a sealed oracle

`pe_emit` turns a goal term into registry-v0.4 canonical text: defaults resolved explicitly,
kwargs sorted, enum values quoted. Predict this one from the rules just stated — the input
gives `mu` and `estimand` only, and `lineage` has one registry default (`decay=0.85`):

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_emit'), pe_semantic(lineage(pearltrees, mu(haiku), estimand(ancestry)), S), write(S), nl" -t halt
lineage(pearltrees,decay=0.85,estimand="ancestry",mu=haiku)
```

That exact string is not this tutorial's opinion — it is a row of the sealed golden bundle,
`PROCESS_EXPRESSION_GOLDEN_v3.json`, frozen with a content hash. Read the same bytes straight
from the bundle:

```console
$ python3 -c "import json; d=json.load(open('prototypes/mu_cosine/PROCESS_EXPRESSION_GOLDEN_v3.json')); print([r['canonical_identity_string'] for r in d['rows'] if r['name']=='lineage-haiku'][0])"
lineage(pearltrees,decay=0.85,estimand="ancestry",mu=haiku)
```

**Why sealed oracles:** the emitter is verified by byte equality against all 25 bundle rows
(`test_pe_emit.pl`), so "correct" means "agrees with frozen bytes", never "looks right to
whoever ran it last".

## c. The where-form: one variable, two occurrences, zero survivors

`where(Expr, Bindings)` lets you factor a repeated literal into a variable. Because `C` below
is a real Prolog variable, its two occurrences are *one* variable — one binding reaches both
by construction, and a mismatch cannot even be written:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_where'), where_semantic(where(product(hop_decay(C, gamma(0.6)), lca_frac(C)), [C = simplemind]), S), write(S), nl" -t halt
product(hop_decay(simplemind,gamma=0.6),lca_frac(simplemind))
```

Note what the output does **not** contain: no `where`, no `C`, no binding syntax at all. The
binding is *discharged* — it constructed the term and then vanished; the surface is
byte-identical to writing `simplemind` twice by hand:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_where'), where_semantic(where(product(hop_decay(C, gamma(0.6)), lca_frac(C)), [C = simplemind]), S), ( sub_string(S, _, _, _, \"where\") -> writeln(binding_survived) ; writeln('discharged: no binding syntax survives in the surface') )" -t halt
discharged: no binding syntax survives in the surface
```

**Why bindings vanish:** a binding is identity-*determining* (`C=simplemind` and
`C=pearltrees` are different processes), so it must be consumed before the surface exists —
it is never provenance, and it never reaches the pin channel.

## d. The elaborator: discharge or residuate

The elaborator generalizes the where-form to a full goal store. A store whose goals all
discharge yields `ground(Term)` — and the ground term renders to the same sealed bytes as
example b:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), use_module('prototypes/mu_cosine/pattern_stache/pe_emit'), elaborate(lineage(pearltrees, mu(haiku), estimand(ancestry)), [], State), format('~q~n', [State]), State = ground(G), pe_semantic(G, Bytes), write(Bytes), nl" -t halt
ground(lineage(pearltrees,mu(haiku),estimand(ancestry)))
lineage(pearltrees,decay=0.85,estimand="ancestry",mu=haiku)
```

An **under-instantiated** goal cannot discharge — it *residuates*: it stays in the store, and
the result is a `pattern` state, not a ground one. Here `has_type(X, substrate(C))` has both
`X` and `C` free, so it travels with the term. The output is printed with variables numbered
(`A`, `B`) so it is stable; note the residual store is canonically ordered, and the origin
annotation rides *alongside* the store, never inside it:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), elaborate(product(hop_decay(C, gamma(0.6)), lca_frac(C)), [has_type(X, substrate(C))-'surface:X::substrate[C]'], State, Origins), copy_term(State-Origins, N), numbervars(N, 0, _), N = S2-O2, format('~q~n~q~n', [S2, O2])" -t halt
pattern(product(hop_decay(A,gamma(0.6)),lca_frac(A)),[has_type(B,substrate(A))])
[has_type(B,substrate(A))-'surface:X::substrate[C]']
```

**Why two states:** a term with residual goals is a *pattern*, not a process — it has no
digest and no canonical bytes yet (that is `peid-v1` territory, deliberately fenced). The
typed state makes it impossible to mistake one for the other.

The loop is a **fixpoint** because discharging one goal can unlock another. Watch this store:
in pass one, `has_type(t1, substrate(C))` is nonground (residuates) while `C = simplemind`
discharges by substitution; in pass two, the `has_type` — now ground — checks against the
registry and discharges too. Everything vanishes and the result is ground:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), elaborate(lca_frac(C), [has_type(t1, substrate(C)), C = simplemind], State), format('~q~n', [State])" -t halt
ground(lca_frac(simplemind))
```

### d-addendum. The second oracle: structure, not just strings

Byte equality (example b) checks the *rendered* surface. The bundle also seals each row's
`resolved_ast` — the elaborated node structure as JSON — so the elaborated **term** can be
verified independently of any renderer: elaborate, project the ground term into the
correspondence (`pe_resolved_ast.pl` documents it), and compare structurally against the
sealed JSON, read as data:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), use_module('prototypes/mu_cosine/pattern_stache/pe_resolved_ast'), elaborate(menu(graph, n(10)), [], ground(G)), ( projection_matches_row('menu-required-int', G) -> writeln('structure matches sealed resolved_ast: true') ; writeln(mismatch) )" -t halt
structure matches sealed resolved_ast: true
```

**Why two oracles on the same rows:** with structure checked before rendering, a builder bug
and a renderer bug can no longer mask each other — each would have to fool a different sealed
artifact.

## e. The fail-closed gallery

Every refusal below is a named error thrown at elaboration time. One sentence each on the
hazard the refusal prevents. (Each command catches the error and prints its term with
variables numbered, so the output is stable.)

A **dead binding** — its variable occurs nowhere — is a typo that would otherwise bind
nothing and complain never:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), catch(elaborate(fs, [B = simplemind], _), error(pe_elaborate(E), _), (copy_term(E, EC), numbervars(EC, 0, _), format('~q~n', [EC])))" -t halt
dead_binding(A=simplemind)
```

A **pin smuggled in as a binding value** would put provenance into the identity preimage —
the exact configuration the identity rules exist to make impossible:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), catch(elaborate(lca_frac(C), [C = pin(fs, 'run/1')], _), error(pe_elaborate(E), _), (copy_term(E, EC), numbervars(EC, 0, _), format('~q~n', [EC])))" -t halt
binding_rejected(illegal_binding_value(pin(fs,'run/1'),at(arg(lca_frac,1))),origin(none))
```

An **illegal ground type** (v1.1's eager validation): `substrate(frobnicate)` is ground and
false — no registered name has that output — so letting it residuate would park an
unsatisfiable constraint in a pattern state forever, dormant and unfixable:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), catch(elaborate(fs, [has_type(X, substrate(frobnicate))], _), error(pe_elaborate(E), _), (copy_term(E, EC), numbervars(EC, 0, _), format('~q~n', [EC])))" -t halt
constraint_unsatisfiable(has_type(A,substrate(frobnicate)),origin(none))
```

An **unknown constraint form** can never become known — its functor is not in the elaborator's
ruled scope — so it errors at first sight rather than pretending it might resolve later:

```console
$ swipl -g "use_module('prototypes/mu_cosine/pattern_stache/pe_elaborate'), catch(elaborate(fs, [frobnicate(x)], _), error(pe_elaborate(E), _), (copy_term(E, EC), numbervars(EC, 0, _), format('~q~n', [EC])))" -t halt
unknown_constraint(frobnicate(x),origin(none))
```

## Where to go next

- The dialect's normative contract: `docs/design/SPEC_pattern_stache.md`.
- Why structural matching at all: `docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md`.
- The elaborator's design and its identity fence: `prototypes/mu_cosine/DESIGN_prolog_elaborator.md`.
- The test suites every example here leans on: `tests/core/test_pattern_stache.pl` (engine) and
  `prototypes/mu_cosine/pattern_stache/test_pe_*.pl` (emitter, where-form, elaborator).

*Verification note: `python3 docs/tutorials/check_tutorial.py docs/tutorials/TUTORIAL_pattern_stache.md`
runs every `console` block above from the repository root and diffs real output against the
shown output — it is wired into CI, so this document cannot silently rot.*
