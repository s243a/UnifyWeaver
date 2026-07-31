# Desugaring the process-expression surface to Prolog goals — design note

**Status: design note, not a specification and not an authorization.** It records reasoning
that currently exists only in conversation, so a later reader does not have to reconstruct it.
It changes no contract, gates nothing, and authorizes no implementation. Registry `v0.3`,
`pec-v2`, `tok-v1`, and both sealed golden bundles are untouched.

Companion to [`DESIGN_process_expression_patterns.md`](DESIGN_process_expression_patterns.md)
(the vNext specification, unimplemented) and
[`DESIGN_process_expression_generator.md`](DESIGN_process_expression_generator.md) (step-2
corpus spec). It proposes a *mechanism* for the half of the patterns doc that was specified and
never built — `interpretation/3` and `representation/4` — not a new language.

Nothing here is identity-bearing.

## 1. The claim

Every non-term construct in the surface language is **syntactic sugar for a Prolog goal**:

```text
X::substrate[C]                 ->  has_type(X, substrate(C))
<= NonAmplifying(Op)            ->  non_amplifying(Op)
substrate ownership             ->  owns(S, C)
```

An expression therefore elaborates to **a term plus a goal store**, and the elaborator is a
solver over that store.

The consequences below follow from that one move. They are ordered by how expensive they
become if decided late.

## 2. This is what the existing elaborator already is

`process_expression_vnext` (milestone 2, PR #4028) carries `PatternVar.constraint` for each
unbound variable and discharges the constraints by unification to a fixpoint inside `ground()`.
That is a residual goal store and a solver, specialized to one goal type (`has_type`) with a
hand-rolled solver.

So the general form is not a redesign. It is the generalization of which the landed milestone is
a special case:

```text
PatternAST  =  typed term  +  residual goal store
ground(b)   =  bind, then run the store to a fixpoint
```

This is a more accurate description of the shipped code than the one in its own PR, and it means
the milestone generalizes rather than gets rewritten.

## 3. Compile time versus runtime is a *mode* question

A goal is dischargeable exactly when its arguments are sufficiently instantiated. That is
Mercury's mode discipline rather than its type-class discipline, and it is the more relevant
half here.

| goal | instantiation | discharged |
|---|---|---|
| `non_amplifying(product)` | ground, closed predicate | elaboration, always |
| `has_type(principal_tree(pearltrees), substrate(pearltrees))` | ground | elaboration |
| `has_type(X, substrate(C))` | `X` free | **residuates** — travels with the term |
| `in_support(decay, D)` where support is corpus-dependent | needs data | runtime |

Under-instantiated goals suspend and are carried until they can be discharged. SWI already has
the primitives (`freeze/2`, `when/2`, `dif/2`); the point is that the *policy* — what is checked
when — becomes introspectable rather than hard-coded in an elaborator.

**This project's compile time is elaboration.** Nothing is hashed, sealed, or trained on before
it has been elaborated, so a goal discharged during elaboration gives the same guarantee a
static check would: the illegal expression never reaches a `GroundAST`, and therefore never
reaches identity.

## 4. Generating the AST is a transpiler target

This is the practical payoff, and it is the reason to prefer desugaring over a bespoke checker.

UnifyWeaver is a Prolog transpiler with **67 registered targets**
(`src/unifyweaver/targets/*_target.pl`). They share
[`common_generator.pl`](../../src/unifyweaver/targets/common_generator.pl), which walks a goal
and renders it under a `Config` of format strings:

```prolog
%    access_fmt: Format string for variable access, e.g. "~w.get('arg~w')"
%    atom_fmt:   Format string for atoms, e.g. "'~w'"
%    null_val:   String for unbound/null, e.g. "None"
%    ops:        List of Op-String pairs for operators
```

plus [`template_system.pl`](../../src/unifyweaver/core/template_system.pl) for named-placeholder
substitution and composable template units.

Once the surface desugars to goals, **the typed AST is one more target**: a `Config` and a set of
templates that emit AST nodes instead of Bash, Rust, or WAM text. The same introspection that
produces every other backend produces the structure the encoder embeds. No separate AST builder
has to be written or kept in sync with the language, because the language *is* Prolog goals and
the project already knows how to walk those.

That also makes the semantics rules executable by the system they describe: `interpretation/3`
and `representation/4` would be Prolog predicates compiled through the project's own toolchain.

**Cost to state honestly:** the `mu_cosine` prototypes are Python today and have no `swipl`
dependency; the transpiler does. Making AST generation a transpiler target adds a cross-stack
dependency to a path that currently does not have one. That is a real cost, and it is why this
is a note rather than a plan.

### 4.1 Constraints as template dispatch keys

`template_system.pl` already has match/case dispatch:

```text
{{match key}}{{case v1}}...{{case v2}}...{{default}}...{{/match}}
```

and its matching predicate is a single deliberately-factored clause:

```prolog
case_matches(Value, Pattern) :- Value = Pattern.
```

The file documents the intended extension — glob, PCRE2 regex, or **structured patterns** — and
notes the predicate "is factored out to make this extension straightforward."

**What this is today: exact string match, and nothing more.** The observation that `=` is
unification is about *position*, not capability. `resolve_match/5` stringifies both operands
before the call:

```prolog
member(Key=DictValue, Dict),
atom_string(DictValue, DictValueStr),        % stringified
member(case(CaseValue, CaseBody), Cases),
atom_string(CaseValue, CaseValueStr),        % stringified
case_matches(DictValueStr, CaseValueStr)     % two strings
```

so the clause compares strings. Nothing should be built against a belief that structural
matching already works.

**What the extension would cost.** Three steps, of which only the third is substantial:

1. drop the two `atom_string/2` conversions in `resolve_match/5`;
2. change the Dict contract so values may be terms rather than atoms or strings;
3. **parse `{{case ...}}` bodies as terms.** Case values are currently produced by splitting
   template *text*, so structured patterns require term reading (`read_term_from_atom/3` or
   equivalent) — which reopens operator-table and variable-naming questions the string form
   never had to answer.

**Expressive, not more powerful.** Structural dispatch would let a template select on
`substrate(C)` rather than on the literal string `"substrate"`, with `C` bound for the body —
open patterns instead of a fixed enumeration, which is a real gain. It does not make the
constraint system prove more. Unification supplies *dispatch*; a closed-fact constraint such as
`non_amplifying(Op)` remains a lookup, and a numeric constraint such as `μ(A→C) ≤ min(links)` is
not checkable by unification at all (§8). The proof obligation still discharges once per
operator, by property test.

**Constraints are the safe dispatch keys, and this is why priority matters.** Because
constraints are discharged before ordinary body goals (§3), they are exactly the subset
guaranteed ground at selection time. Ordinary goals may still be residual. The rule:

> Dispatch only on goals guaranteed discharged before selection.

The mode analysis of §3 therefore does a second job — first deciding *when* a goal is checked,
then deciding *which* goals may key a template.

**Template selection is instance resolution.** The correspondence is exact rather than
analogical:

| Mercury / Haskell | here |
|---|---|
| constraint resolved at compile time | constraint discharged at elaboration |
| selects the *instance* | selects the *template* |
| instance supplies the method body | template emits the AST node |
| instance declarations in a module | `{{case}}` blocks in a template file |

Which is what makes a template file usable as a **library**: it is a predicate, `{{case}}` blocks
are its clause heads, the dict is the goal, and rendering is resolution.

That library structure already exists on disk. Mustache files live per target under
`templates/targets/<target>/*.mustache` — `rust_wam`, `scala_wam`, and others — so "one file,
many cases, dispatched by key" is the shape the project already uses for code generation, not a
new idea being proposed for the AST.

Two disciplines follow, both consistent with §6.

**Overlapping patterns become order-dependent.** Exact string match cannot overlap; structured
patterns can. Prolog takes the first matching clause, while Haskell and Mercury forbid
overlapping instances outright. If template selection generates the AST, **template file order
becomes semantically significant** — the same clause-order hazard as the residual store (§6.2)
and as Prolog-text-as-canonical (§8). Either require non-overlapping cases, checked at load, or
make the selection order part of the versioned registry.

**Template *resolution* is a second order-dependence.** `render_named_template/4` takes a
`source_order` strategy list — `generated`, `file`, `cached` — and returns the first that
succeeds, so one template name can resolve to different bytes depending on the strategy passed.
This is not hypothetical: `template_system.pl` already carries the comment *"Inline template is
authoritative; `templates/targets/rust_wam/Cargo.toml.mustache`"*, resolving exactly that
ambiguity by convention. For code generation a comment suffices. If templates generate the AST,
the resolution strategy becomes identity-bearing too, and a convention is not enough.

**Templates are promoted from cache key to identity input.** Today
[`process_cards.py`](process_cards.py) binds `RENDERER_VERSION` (currently `"r2"`) into the *card
cache* key — `(ast_sha, verbosity, RENDERER_VERSION, e5_revision, prefix)` — while identity is
`REGISTRY_VERSION + "|" + canonical`. The renderer affects **display only**. If templates
generate the AST, the template set affects **structure**, moving it out of the cache key and into
the identity preimage. The template set would then need content-hashing and sealing on the same
footing as the registry. That escalation is a decision, not a detail.

## 5. What is embedded, and what is merely checked

Residuation answers a modelling question that would otherwise be settled by taste — whether a
constraint belongs *inside* the structure the encoder embeds:

- **Discharged at elaboration** → a *side condition*. It verified the term; it did not
  constitute it. Not embedded.
- **Still residual at ground time** → *part of the term*. It is unresolved information that
  distinguishes this ground term from another one. Embedded.

The criterion falls out of the mechanism instead of being asserted alongside it. Deciding it
before a trained encoder exists is much cheaper than after.

## 6. Two disciplines that must ship with the sugar

### 6.1 The constraint predicate must be closed and versioned

Mercury's classes are safe partly because instances are closed and coherent. **A Prolog
predicate is an open database**: anything that can `assert/1` can add `non_amplifying(logit)`,
and the guarantee evaporates silently.

So the constraint facts belong in the **sealed, versioned registry**, not in a runtime database,
and must not be assertable at runtime. With closure it is a guarantee; without it, a suggestion —
and the difference is invisible until something has already been trained.

Adding a pooling operator is then a registry bump plus a new instance fact discharged by a
property test, which is the same bar as any other registered entry.

### 6.2 The residual store canonicalizes as a set

If residual goals are part of a term's identity, goal **order** would otherwise change the
digest. That is the same clause-order hazard that argues against making Prolog *text* canonical.

The store must canonicalize sorted and deduplicated — as a set, not a list. Cheap now; a
re-seal later.

## 7. What the sugar must not cost: typed diagnostics

A uniform goal solver naturally reports `goal failed`. The vNext frontend deliberately separates
`ParseError` / `ElaborationError` / `GroundingError`, and emits messages such as

> annotation `judge` conflicts with inferred type `substrate[C]`; `::` asserts or narrows and
> never converts

That property is worth protecting. **Desugar for semantics, but retain which surface construct
each goal came from**, so a failed goal can be reported in the vocabulary the author wrote rather
than as a solver trace. Otherwise the sugar silently buys uniformity with error quality, which
nobody notices until they are debugging.

## 8. Non-goals

This note does **not** propose:

- adopting Mercury as a dependency — it is a design reference. Neither Mercury nor Haskell checks
  numeric properties such as `μ(A→C) ≤ min(links)`; a class is a *tag* whose obligation is
  discharged once per operator by property test, not a proof;
- making Prolog text canonical — the typed AST remains the identity surface, per the patterns
  doc's "surface spelling is not identity";
- any change to registry `v0.3`, `pec-v2`, `tok-v1`, or the sealed bundles;
- unblocking corpus enumeration, which remains gated on the registry `v0.4` **ship** (see
  `DESIGN_process_expression_generator.md` §2).

## 9. Sequencing

The class/constraint machinery is deliberately *after* `v0.4`, because `v0.4` gates enumeration
and the constraint system gates nothing:

1. **`v0.4`** — `estimand=` / `impl=` split, corpora registered as substrates, and the operators
   needed to express what the graph judge computes (`max`, product) rather than mislabel it.
   Safety **by construction**: distinct operators whose argument sets make the illegal
   combination unspellable, with no class mechanism.
2. **Ship `v0.4`** → enumeration unblocks → encoder work proceeds.
3. **vNext / Prolog layer** — desugaring, residuation, closed constraint predicates, and
   `interpretation/3` / `representation/4`, with the AST as a transpiler target.

Step 1 is strictly weaker than step 3: a fixed operator enumeration cannot be extended without a
registry bump. That is an accepted cost, taken so that the encoder is not held behind a
type-system design.

## 10. Why the composition constraint needs any of this

The constraint that motivated the discussion, for the record.

`DESIGN_transitive_relations.md` requires `μ(A→C) ≤ min(links)` — composition along a path must
not amplify. Writing `x_i ∈ [0,1]`, weighted pooling in three spaces behaves differently:

| space | form | bounded by inputs |
|---|---|---|
| linear | `Σ wᵢxᵢ` | between min and max |
| log | `exp(Σ wᵢ log xᵢ) = Π xᵢ^wᵢ` | between min and max when `Σwᵢ = 1`; **`≤ min`** when weights are 1 (the product t-norm) |
| logit | `σ(Σ wᵢ logit xᵢ)`, i.e. odds `Π (xᵢ/(1−xᵢ))^wᵢ` | **not bounded** — agreement amplifies |

So *space* alone is not the axis; **space × normalization** is. Sum-form composes, average-form
fuses. Logit pooling is principled for fusing independent judges of one pair and **illegal** for
composing along a path — a formal test for the fusion/generation distinction that was previously
a convention.

Two riders:

- `logit(0)` and `logit(1)` are infinite, and the graph judge assigns `mu = 1` at identity. The
  `floor` in `max(floor, gamma^hops * lca_frac)` is therefore not only a "do not decay to
  nothing" patch — it is what makes the value representable in logit space at all. A second,
  independent reason `floor` belongs in the grammar.
- Whether logit fusion is *appropriate* for μ is a science question, not a grammar one. Naive-Bayes
  pooling assumes conditional independence, and `DECISIONS_graph_geometry.md` (2026-07-12)
  measured Nomic/MiniLM at 0.894–0.907 correlation while explicitly refusing to treat them as
  "two independent votes." Pooling correlated judges in logit space overstates confidence.

## 11. Open questions

1. Is logit-space fusion admissible for μ at all, given measured judge correlation? (§10)
2. Does AST-as-transpiler-target justify the `swipl` dependency in `mu_cosine`? (§4)
3. Do residual goals enter the embedded structure, and if so does the encoder see them as
   structure or as features? (§5 gives the criterion, not the encoding.)
4. Which surface constructs desugar, and which stay primitive for diagnostic quality? (§7)
