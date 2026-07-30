# Process-Expression Patterns, Types, and Resolution — vNext

**Status: proposed specification; not implemented.** The implemented language is still
[`DESIGN_process_expression_language.md`](DESIGN_process_expression_language.md) with registry
`v0.3`. The implemented structural token-stream and role-path fixture contract is `pec-v2`.
`pec-v2` is not an identity-schema version.

This document specifies the next language boundary without changing any v0.3 identity or sealed
golden file. It introduces:

- a compact functional surface for concrete authoring;
- a Prolog surface for patterns and resolution rules;
- one shared typed AST;
- optional `::` type assertions;
- separate interpretation and representation relations;
- explicit factory verification; and
- distinct semantic identity, resolution evidence, and provenance.

The central rule is:

```text
ground does not imply semantically explicit
semantically explicit does not imply representation-resolved
representation-resolved does not imply factory-verified
```

Only the final state may mint a deployed identity.

## 0. Decisions at a glance

| Concern | Decision |
|---|---|
| Functional notation | Keep positional arguments and Python-like named fields. |
| Declarative notation | Add a Prolog adapter with variables and option terms. |
| Shared meaning | Both adapters elaborate to the same typed AST. Surface spelling is not identity. |
| Optional types | Use `Term::Type`; annotations assert or narrow inferred types and never cast. |
| Suggested variables | `C::corpus`, `S::substrate[C]`, `J::judge`, `E::entity[S]`, `F::function(...)`. |
| Lineage names | `lineage_at` is the family reference; `lineage_op(...)` is the coarse operator. |
| Ambiguous meaning | `interpretation/3` maps a coarse term to an explicit estimand. |
| Ambiguous implementation | `representation/4` maps one explicit estimand plus constraints to equivalent implementations. |
| Approximation | Learned or lossy approximations use a separate relation and become a different process. |
| Pins | `@kind/value` is optional provenance, not a type, selector, trust proof, or semantic default. |
| Deployed identity | Bind a precise semantic AST to a verified stable factory realization. |

`lineage_at` uses `at` as a short suffix for “atom.” The semantic AST records a `Reference` node,
not the suffix. A registry could instead freeze `lineage_atom`; it must choose exactly one
canonical spelling.

## 1. The state machine

The public API uses different types for different states:

```text
PatternAST
  --ground(bindings)-->
GroundAST
  --interpret(...)-->
SemanticGroundAST
  --represent(...)-->
ResolvedGroundAST
  --verify_factory_receipt(...)-->
VerifiedProcess
```

Direct authoring may enter at a later state:

```text
precise surface expression
  --parse + elaborate + prove complete-->
ResolvedGroundAST(origin=direct)
```

The types have these invariants:

```text
PatternAST
    may contain Variable nodes

GroundAST
    contains no Variable nodes
    may still contain coarse families or unresolved semantic choices

SemanticGroundAST
    names one estimand with a complete domain and codomain
    retains any explicit representation constraints separately from the semantic term
    may still have several equivalent implementation representations

ResolvedGroundAST
    contains no variables, coarse nodes, unresolved overloads, or implicit impl choice
    has one complete precise typed AST

VerifiedProcess
    binds that exact ResolvedGroundAST to a verified stable factory realization
```

Identity, deployed caches, residual rows, and production adapters accept only
`VerifiedProcess`. A diagnostic pattern or decoder-candidate cache uses a separate namespace and
cannot be joined to a deployed cache by digest alone.

The deployment predicate is:

```text
deployable(P) :=
    P is VerifiedProcess
    and P.precise_ast is ground
    and P.precise_ast is recursively representation-resolved
    and P.factory_binding verifies against that exact AST and registry
```

Pins are not part of this predicate. They are optional, self-declared provenance that must agree
with a receipt when present.

## 2. One typed AST

The two surface parsers normalize to this conceptual algebra. This is the semantic model; the
canonical wire encoding is frozen separately.

```text
Term :=
    Variable(name)
  | Reference(registry_name)
  | Call(registry_name, positional_args, named_fields)
  | Atom(value)
  | String(value)
  | Int(value)
  | Real(exact_decimal_value)
  | Float64(ieee754_bits)
  | List(items)
  | Modified(term, registered_modifier)
  | Annotated(term, asserted_type)

Type :=
    TypeName(name)
  | IndexedType(name, indices)
  | FunctionType(argument_types, result_type)
  | ListType(element_type)
  | SumType(constructors)

TypedTerm := {
    core: elaborated Term
    inferred_type: Type
}

ProvenanceEnvelope := {
    pins: [ProvenancePin]
    resolution_receipts: [ResolutionReceipt]
    migration_receipt: MigrationReceipt | null
    factory_verification_receipt: FactoryVerificationReceipt | null
    decoded_candidate_receipt: DecodedCandidateReceipt | null
}
```

Provenance is deliberately outside `TypedTerm`. A semantic digest hashes the typed semantic AST,
not pins or the path by which the AST was selected.

The AST distinguishes:

```text
Reference("haiku")       # a registered judge
Call("margin", ...)      # a registered callable
Atom("attention")        # a symbolic enum-like value
String("attention")      # Unicode text
```

These nodes cannot collide even when a human-oriented renderer looks similar.

An annotation is checked during elaboration:

```text
pearltrees
pearltrees::corpus
```

If the registry already declares `pearltrees : corpus`, both normalize to the same typed node:

```text
Reference(name="pearltrees", inferred_type=corpus)
```

### 2.1 Canonical typed-AST encoding

The identity-bearing AST encoding is tagged canonical JSON under schema `pe-typed-ast-v1`. It is
neither functional text nor Prolog text:

```json
{
  "schema": "pe-typed-ast-v1",
  "root": {
    "kind": "call",
    "name": "bounded_depth",
    "type": {
      "kind": "name",
      "name": "depth_limit"
    },
    "args": [
      {
        "kind": "int",
        "type": {"kind": "name", "name": "int"},
        "value": "4"
      }
    ],
    "fields": []
  }
}
```

Indexed types use normalized type/index nodes rather than display strings such as `"S"`.

Node tags are:

```text
var | ref | call | atom | string | int | real | float64 | list | modified
```

Type tags are:

```text
name | indexed | function | list | sum
```

Canonical rules:

1. object fields use the frozen schema and unknown fields fail;
2. call positional arguments retain order;
3. named fields serialize as a name-sorted array of `{name,value}` records;
4. registry defaults are explicit;
5. annotations are absent after successful elaboration;
6. semantic modifiers serialize as `modified` nodes or normalize to a registry-declared explicit
   field; display-only modifiers cannot enter identity;
7. `int` values are canonical decimal strings;
8. `real` values are canonical exact-decimal strings;
9. `float64` values are 16 lowercase hexadecimal digits encoding the 8 IEEE-754 bytes;
10. variables are allowed only in pattern serialization; and
11. provenance pins and receipts are not children of the semantic AST.

Canonical JSON uses UTF-8, sorted keys, compact separators, no duplicate keys, and no non-finite
JSON numbers. Numeric AST values are strings, so host JSON number formatting cannot change
identity.

## 3. Type system

### 3.1 Primitive and domain types

| Type | Meaning |
|---|---|
| `atom` | A symbolic value such as `'attention'`; not a registry reference. |
| `string` | Unicode text. |
| `int` | An unbounded mathematical integer unless a signature narrows it. |
| `real` | A finite exact decimal value used for mathematical parameters. |
| `float64` | An IEEE-754 binary64 value when bit-level runtime representation is semantic. |
| `list[T]` | A homogeneous functional-surface list. |
| `corpus` | A semantic population or collection name. |
| `substrate[C]` | A particular structural/data view of corpus `C`. |
| `entity[S]` | An entity belonging to substrate `S`. |
| `query_context[S]` | A typed query/ground-truth context belonging to `S`. |
| `judge` | A label-producing or scoring authority. |
| `function([A...], R)` | A callable process descriptor from argument types to result type. |
| `scorer[Args,Q]` | A descriptor mapping the typed argument list `Args` to `score_value[Q]`. |
| `relation_scorer[S,Q]` | Alias for `scorer[[entity[S],entity[S]],Q]`. |
| `target_factory[S,T]` | A configured descriptor producing targets of kind `T` over `S`. |
| `score_value[Q]` | An evaluated score on scale/calibration contract `Q`. |
| `target_set_value[S,T]` | An evaluated target set over `S` with target kind `T`. |
| `relation_family` | An abstract family reference such as `lineage_at`. |
| `depth_limit` | `bounded_depth(int)` or `unbounded_depth`. |

`double` may be accepted as a functional-surface alias, but canonical typed ASTs use `float64`.
Most hyperparameters should use `real`; binary64 is normally an implementation detail.

The singular type is `corpus`. Several corpora have type `list[corpus]`. There is no separate
canonical type named `corpora`.

### 3.2 Corpus, substrate, and entity ownership

A corpus identifies a population, not an immutable dataset snapshot:

```text
pearltrees::corpus
```

A substrate identifies the semantic view on which a process operates:

```text
principal_tree(pearltrees)::substrate[pearltrees]
full_dag(pearltrees)::substrate[pearltrees]
filesystem_tree(local_filing)::substrate[local_filing]
```

`principal_tree(pearltrees)` and `full_dag(pearltrees)` are different substrates even if both use
the same underlying corpus. Exact row sets, graph revisions, filters, and split hashes belong in
the verified factory realization.

Entity ownership is indexed:

```text
E::entity[S]
S::substrate[C]
C::corpus
```

This prevents an entity from a SimpleWiki graph being silently passed to a scorer built for a
Pearltrees tree.

vNext retires the v0 catch-all `source` type. A lineage target factory consumes a substrate. A
blend consumes compatible scorers or judges adapted to scorers. Neither is implicitly convertible:

```text
lineage_op(haiku)
# Error: judge is not a substrate.

blend(pearltrees, simplewiki)
# Error: corpora are not scorer descriptors.
```

An explicit adapter may convert a judge into a scorer only by naming the missing semantics:

```text
judge_scorer(
  haiku,
  principal_tree(pearltrees),
  relation='ancestor_membership'
)
```

Its prompt/template and model revision remain factory-bound provenance.

### 3.3 Descriptors are not evaluated values

Process expressions describe computations; they do not contain their evaluated outputs:

```text
structural_lineage_score(...)     # relation_scorer[S,unit_interval]
hop_decay_targets(...)            # target_factory[S,hop_decay]
```

Runtime results have different types:

```text
0.73                              # score_value[unit_interval], after evaluation
{parent_1: 1.0, parent_2: 0.85}   # target_set_value[S,hop_decay]
```

A resolver compares full descriptor signatures, including indexed substrate, domain, codomain,
direction, scale, and estimand ID. Matching only the word `score` or `target_set` is insufficient.

Lineage scoring is relational, not unary. The default domain is:

```text
[entity[S] as destination_or_truth, entity[S] as candidate]
```

A ranking pipeline that needs more query state uses
`[query_context[S], entity[S] as candidate]`. The registry freezes argument roles as well as
types; hiding the destination or truth entity in ambient state is forbidden.

### 3.4 C, S, J, E, and F are variable conventions

Uppercase identifiers are variables in Prolog, so they are not canonical type names:

```text
C::corpus
S::substrate[C]
J::judge
E::entity[S]
F::function([entity[S],entity[S]], score_value[unit_interval])
```

Other useful conventions:

```text
N::int
R::real
A::atom
Text::string
W::list[real]
Cs::list[corpus]
```

In canonical Prolog type syntax, square-bracket type constructors become ordinary terms:

```prolog
C::corpus
S::substrate(C)
J::judge
E::entity(S)
F::function([entity(S),entity(S)], score_value(unit_interval))
W::list(real)
Cs::list(corpus)
```

### 3.5 Inference and annotation

Elaboration must:

1. obtain positional and named-field types from the content-addressed registry;
2. infer literal candidates from their lexical class;
3. unify repeated variables and indexed ownership constraints;
4. check each `::` assertion;
5. apply only registry-declared subtyping and overload rules;
6. reject conflicts and underconstrained terms before resolution; and
7. retain normalized inferred types on every AST node.

`::` asserts or narrows. It never converts:

```text
lineage_op(S)
# S is inferred as substrate[C] for a fresh C.

lineage_op(S::substrate[C])
# Same constraint, stated explicitly.

margin(t=1)
# t is typed real because the signature says real.

margin(t=1::real)
# Same typed AST.

0.85::int
# Error: fractional value cannot satisfy int.

lineage_op(S::judge)
# Error: judge conflicts with substrate[C].
```

Conversions name a policy:

```text
round_to_int(3.7, mode='nearest')
```

Redundant annotations do not alter semantic identity. An annotation that selects a distinct
registered overload does alter the normalized typed AST.

### 3.6 Numeric normalization

The declared field type wins over the source-language runtime type.

For `real`:

- parse a finite decimal exactly;
- normalize equivalent spellings such as `1`, `1.0`, and `1e0` to one canonical value;
- normalize negative decimal zero to zero; and
- reject NaN and infinities.

For `float64`:

- round under the frozen IEEE-754 conversion rule;
- serialize the exact 64 bits in canonical AST bytes; and
- preserve positive and negative zero as distinct bit patterns.

For `int`, fractional and exponent forms that are not exact integers fail.

The legacy `pec-v2` invariant remains independently tested: `margin(t=1)` emits the declared
NUMBER/REAL token kind, never INT, even though the surface lexeme is integral.

## 4. Functional surface

The functional notation extends v0 with variables, optional types, atom literals, and
expression-valued named fields.

```ebnf
Expr          := Primary { "." Modifier } [ "::" Type ] { "@" ProvenancePin }
Primary       := Call
               | Variable
               | Reference
               | AtomLiteral
               | String
               | Number
               | List
               | "(" Expr ")"
Call          := Reference "(" [ Argument { "," Argument } ] ")"
Argument      := Ident "=" Expr | Expr
List          := "[" [ Expr { "," Expr } ] "]"

Variable      := "_" | /[A-Z][A-Za-z0-9_]*/
Reference     := longest match in the versioned registry
AtomLiteral   := a single-quoted symbolic atom
String        := a JSON double-quoted string
Number        := a finite decimal numeric token
Modifier      := the versioned modifier grammar
ProvenancePin := PinKind "/" PinValue

Type          := TypeName
               | TypeName "[" TypeOrIndex { "," TypeOrIndex } "]"
               | "function" "(" "[" [ Type { "," Type } ] "]" "," Type ")"
TypeOrIndex   := Type | Variable | Reference
```

Single and double quotes are different:

```text
'attention'                  # Atom("attention")
"attention"                  # String("attention")
impl='graph_walk'            # semantic enum-like field
note="attention experiment"  # text
```

An unregistered bare lowercase word is not guessed to be an atom. It must be registered as a
reference or written in single quotes.

### 4.1 Functional examples

References and types:

```text
pearltrees
pearltrees::corpus
haiku::judge
lineage_at::relation_family
42::int
0.85::real
'priority'::atom
```

Substrate constructors:

```text
principal_tree(pearltrees)
full_dag(pearltrees)
filesystem_tree(local_filing)
```

Typed patterns:

```text
lineage_op(S::substrate[C])
score(Truth::entity[S], Candidate::entity[S], J::judge)
apply(
  F::function([entity[S],entity[S]],score_value[unit_interval]),
  Truth::entity[S],
  Candidate::entity[S]
)
blend(F1::relation_scorer[S,Q], F2::relation_scorer[S,Q], w=W::list[real])
```

Current-style composition retained where signatures remain valid:

```text
margin(t=0.01)
kalman(luna.D, luna.S)
blend(luna.D, luna.S, w=[0.4,0.6])
e5(routing(e5, haiku, t=[0.02], menus=[10]))
```

Coarse lineage requests:

```text
lineage_op(principal_tree(pearltrees))
lineage_op(
  principal_tree(pearltrees),
  family_spec='lineage_interpretations_v1'
)
lineage_op(principal_tree(pearltrees), estimand='hop_decay')
lineage_op(
  principal_tree(pearltrees),
  estimand='hop_decay',
  impl='graph_walk',
  decay=0.85,
  hop_origin=1,
  direction='ancestor',
  depth=unbounded_depth
)
```

The first is ground but underconstrained. The second is interpretation-ambiguous within a
content-bound family specification. The third names the estimand but not its representation. The
fourth contains enough information to elaborate to a precise process if the registry signature
consumes every field.

## 5. Canonical Prolog surface

The Prolog adapter is for rules and queries, not a second semantic language.

Suggested annotation operator:

```prolog
:- op(650, xfx, ::).
```

Every registered call keeps its positional arguments and adds exactly one final option list:

```text
Functional: f(A, B, x=X, y=Y)
Prolog:     f(A, B, [x(X), y(Y)])
```

This rule is uniform for fixed and variadic arity. It avoids the earlier ad hoc mappings in which
one functor packed all positional arguments into a list while another did not.

Canonical Prolog value wrappers distinguish references from atoms:

```prolog
ref(pearltrees)          % Reference("pearltrees")
ref('luna.D')            % Reference("luna.D")
atom(attention)          % Atom("attention")
str("attention")         % String("attention")
int(4)                   % Int(4)
real(str("0.85"))        % exact decimal Real(0.85)
float64(hex('3feb333333333333'))  % exact binary64 bits
```

The reader runs with:

```prolog
set_prolog_flag(double_quotes, string).
```

The canonical adapter uses `str/1` anyway, so host flags cannot turn a string into a character
list. A complete collision-free table covers every registry spelling, including dots and hyphens.

Equivalent forms:

| Functional | Canonical Prolog |
|---|---|
| `lineage_op(principal_tree(pearltrees))` | `lineage_op(principal_tree(ref(pearltrees),[]),[])` |
| `lineage_op(S::substrate[C], estimand='hop_decay')` | `lineage_op(S::substrate(C), [estimand(atom(hop_decay))])` |
| `blend(luna.D,luna.S,w=[0.4,0.6])` | `blend(ref('luna.D'),ref('luna.S'),[w([real(str("0.4")),real(str("0.6"))])])` |
| `apply(F,E,C)` | `apply(F,E,C,[])` |
| `ancestor_score(S,depth=bounded_depth(4))` | `ancestor_score(S,[depth(bounded_depth(int(4),[]))])` |

Every zero-argument registered call still carries its final option list in canonical Prolog. The
adapter must freeze whether a nullary registry item is a `Reference` or `Call`; it never infers
that choice from empty arguments.

Variables follow Prolog rules:

- every `_` occurrence is a fresh anonymous variable;
- repeated named variables unify;
- `C` and `OtherC` are distinct unless a rule unifies them; and
- pattern digests alpha-normalize named variables but preserve equality structure.

Examples:

```prolog
lineage_op(S::substrate(C), []).

score(Truth::entity(S), Candidate::entity(S), J::judge, []).

apply(
    F::function([entity(S), entity(S)], score_value(unit_interval)),
    Truth::entity(S),
    Candidate::entity(S),
    []
).

blend(
    F1::relation_scorer(S,Q),
    F2::relation_scorer(S,Q),
    [w(W::list(real))]
).
```

The functional and Prolog adapters must produce byte-identical normalized typed ASTs.

Ordinary Prolog floats are forbidden in the canonical adapter because the host reader may round
before elaboration. `real(str(...))` carries a canonical exact decimal token;
`float64(hex(...))` carries exact bits. The adapter has vectors beyond binary64 precision as well
as `1`, `1.0`, `1e0`, positive zero, and negative zero.

## 6. Lineage vocabulary and distinct estimands

The current v0 vocabulary overloads “lineage”:

- `lineage(graph,decay=0.85)` is parsed as an operator;
- `sonnet.lineage` is a modifier; and
- surrounding design documents use lineage for graph structure, learned μ, filing ancestry, and
  target generation.

vNext separates the coarse names:

```text
lineage_at
# Reference, type relation_family.

lineage_op(S::substrate[C], ...)
# Coarse abstract request, type abstract_lineage_process[S].
```

The AST already distinguishes reference from call. The explicit names also help Python, JSON,
registry tables, and readers who do not carry Prolog functor/arity notation.

If the standalone family reference has no use, omit `lineage_at`. Do not invent a zero-argument
call merely for symmetry.

### 6.1 Three lineage-shaped quantities that must not alias

The project currently discusses at least three different estimands:

#### Hop-decay target factory

```text
hop_decay_targets(
  principal_tree(pearltrees),
  decay=0.85,
  hop_origin=1,
  direction='ancestor',
  depth=unbounded_depth
)
```

For ancestor hop `h`, the landed freezer convention is:

```text
target(h) = decay^(h - hop_origin)
```

The substrate view, direction, hop origin, depth limit, duplicate-path policy, and unreachable
policy are semantic.

#### Structural graph score

```text
structural_lineage_score(
  principal_tree(sm_fs),
  floor=0.02,
  gamma=0.85,
  distance='tree_path_hops',
  shared_prefix='lca_depth',
  normalization='destination_depth',
  argument_roles=['destination','candidate'],
  support='structural_nonancestor',
  unreachable='registered_distance_zero_lca',
  numeric='exact_rational_then_float64'
)
```

This is the landed SM-FS **structural-nonancestor branch** expressed with a proposed vNext `sm_fs`
corpus name. Its functional form on that support is:

```text
max(
  0.02,
  0.85^tree_path_hops(destination,candidate)
    * depth(LCA(destination,candidate)) / depth(destination)
)
```

Different roots use the registered unreachable distance and zero LCA fraction. The constructor
computes an exact reduced rational and then one correctly rounded binary64 value. These fields
cannot be reduced to `decay` and `depth`. A versioned estimator specification may replace the
expanded fields only if its content digest binds all of them.

The positive-parent and positive-ancestor branches are not covered by this descriptor: they use
certified six-decimal ASCII targets parsed once to binary64. A full SM-FS target factory is a
piecewise process that names both support partitions and both numeric policies explicitly.

#### Semantic transitive μ

```text
semantic_transitive_mu(
  full_dag(pearltrees),
  judge=haiku,
  relation='ancestor_membership',
  argument_roles=['truth_ancestor','candidate_descendant'],
  direction='ancestor_to_descendant'
)
```

This is a semantic label or learned estimand. It is not made equal to either structural formula
by sharing a geometric decay constant.

These descriptors may have related domains, but they do not have the same estimand ID. They
cannot enter one representation candidate set or be averaged as if they were interchangeable
implementations.

### 6.2 Attention is an approximation or implementation, not an unstated estimand

A learned attention model must say what it approximates:

```text
approximate(
  semantic_transitive_mu(
    full_dag(pearltrees),
    judge=haiku,
    relation='ancestor_membership',
    argument_roles=['truth_ancestor','candidate_descendant'],
    direction='ancestor_to_descendant'
  ),
  estimator=mu_attention(
    input_view='node_root_ancestors',
    max_depth=5,
    score_scale='unit_interval'
  ),
  contract='semantic_mu_approx_v1'
)
```

`temperature=0.7` alone would not specify this process. Input lineage, path selection, depth
handling, output scale, training estimand, and checkpoint/factory binding must all be explicit or
content-bound by the estimator contract.

An approximation has a different precise AST from its reference estimand. It may be accepted
under a tolerance policy, but it is never smuggled into `representation/4` as exact equivalence.

## 7. Interpretation and representation

### 7.1 Interpretation chooses meaning

Interpretation maps a coarse family to one explicit semantic process:

```prolog
interpretation(Abstract, Semantic, RuleId).
```

Before `interpretation/3` runs, the dispatcher handles absent estimands:

- explicit `estimand=...` goes directly to the matching clauses;
- absent `estimand` plus a content-bound `family_spec` expands to the finite, fully defaulted
  explicit requests listed by that specification; and
- absent `estimand` and absent `family_spec` is `underconstrained`, not an implicit enumeration.

The family-spec content digest is a transitive ruleset dependency and appears in the receipt.

For readability, the rules below operate on the elaborated typed terms after the Prolog adapter
has distinguished `ref/1`, `atom/1`, and `str/1`. They are relational pseudocode, not additional
surface-syntax exceptions.

Illustrative rules:

```prolog
interpretation(
    lineage_op(S::substrate(C), Options),
    semantic_request(
        hop_decay_targets(
            S,
            [
                decay(D),
                hop_origin(H),
                direction(Dir),
                depth(Limit)
            ]
        ),
        RepresentationOptions
    ),
    lineage_as_hop_decay_v1
) :-
    option_exact(estimand(hop_decay), Options),
    option_or_default(decay, Options, real_value("0.85"), D),
    option_or_default(hop_origin, Options, int_value("1"), H),
    option_or_default(direction, Options, ancestor, Dir),
    option_or_default(depth, Options, unbounded_depth, Limit),
    extract_options(Options, [impl], RepresentationOptions),
    consume_only(
        Options,
        [estimand, decay, hop_origin, direction, depth],
        RepresentationOptions
    ).

interpretation(
    lineage_op(S::substrate(C), Options),
    semantic_request(
        structural_lineage_score(
            S,
            [
                floor(Floor),
                gamma(Gamma),
                distance(Distance),
                shared_prefix(Prefix),
                normalization(Norm),
                argument_roles(Roles),
                support(Support),
                unreachable(Unreachable),
                numeric(Numeric)
            ]
        ),
        RepresentationOptions
    ),
    lineage_as_structural_score_v1
) :-
    option_exact(estimand(structural_score), Options),
    required_options(
        Options,
        [
            floor(Floor),
            gamma(Gamma),
            distance(Distance),
            shared_prefix(Prefix),
            normalization(Norm),
            argument_roles(Roles),
            support(Support),
            unreachable(Unreachable),
            numeric(Numeric)
        ]
    ),
    extract_options(Options, [impl], RepresentationOptions),
    consume_only(
        Options,
        [estimand, floor, gamma, distance, shared_prefix,
         normalization, argument_roles, support, unreachable, numeric],
        RepresentationOptions
    ).
```

These clauses illustrate the relation; they are not claims about a checked-in Prolog module.
`extract_options/3` transfers, rather than discards, representation fields. `consume_only/3`
verifies that the semantic keys and transferred representation options form an exact partition of
the supplied option occurrences.

A family-bound expression may enumerate several interpretations for exploration:

```text
lineage_op(
  principal_tree(pearltrees),
  family_spec='lineage_interpretations_v1'
)
```

It cannot deploy until one interpretation is explicitly selected and recorded.

### 7.2 Representation preserves meaning

Representation maps one semantic process to an equivalent precise implementation:

```prolog
representation(Semantic, RepresentationConstraints, Precise, RuleId).
```

The second argument is exactly `SemanticGroundAST.representation_constraints`. `impl` is consumed
here, exactly once. Absence of `impl` leaves all otherwise compatible implementations as
candidates; it never creates an implicit deployable default. Unknown `impl` yields
`no_candidates`, and `unique` therefore fails.

For the same `hop_decay_targets` semantic AST, possible exact representations might be:

```text
graph_walk_hop_decay(Semantic)
materialized_hop_decay_table(Semantic)
compiled_hop_decay_lookup(Semantic)
```

They enter one candidate set only if the registry proves identical:

- input substrate and entity ownership;
- output target type;
- estimand ID;
- direction and hop convention;
- exact numeric semantics;
- duplicate and unreachable policy; and
- observable behavior under the declared domain.

The factory realization still distinguishes their code and materialized artifacts.

`impl='graph_walk'` is semantic configuration for representation selection. It may not be hidden
in a pin or silently defaulted for an identity-bearing expression. If a conditioning card elides
named fields, it must not elide `impl` where that would merge behaviorally different processes.

### 7.3 Every option is consumed

Every interpretation and representation rule must consume every supplied named field exactly
once. Unknown, duplicate, misspelled, or inapplicable fields reject the candidate:

```text
lineage_op(
  principal_tree(pearltrees),
  estimand='structural_score',
  decay=0.85
)
# Error: decay is not consumed by structural_score.

lineage_op(
  principal_tree(pearltrees),
  estimand='hop_decay',
  decay=0.85,
  decay=0.5
)
# Error: duplicate field.
```

Rules may use only a deterministic pure allowlist. They may not depend on I/O, dynamic database
state, clocks, random global state, cuts, clause order, or unbounded search.

## 8. Candidate collection and selection

### 8.1 Normative collection

Prolog `setof/3` is useful notation but is not the normative algorithm: it fails on an empty set,
sorts by host term order, and obscures multiple rules deriving the same AST.

The resolver must:

1. collect all derivations under a finite resource cap;
2. return an explicit `no_candidates` result for zero derivations;
3. type-check and default-resolve each result;
4. require complete option consumption;
5. canonicalize each typed candidate;
6. group identical candidate ASTs and retain every sorted deriving rule ID;
7. sort the groups by canonical typed-AST bytes; and
8. hash that ordered structure as the candidate-set digest.

Exhausting the resource cap is an error, not a partial candidate set.

### 8.2 `unique`

```text
resolve(
  lineage_op(
    principal_tree(pearltrees),
    estimand='hop_decay',
    impl='graph_walk'
  ),
  selector=unique
)
```

`unique` succeeds only when the normalized candidate set contains exactly one AST. Zero and
multiple candidates fail.

### 8.3 `priority(policy)`

```text
resolve(
  lineage_op(principal_tree(pearltrees), estimand='hop_decay'),
  selector=priority('hop_decay_impl_policy_v1')
)
```

The policy is content-addressed and defines a total order over **candidate AST digests**, not rule
source order or rule IDs. A surface `first(policy)` alias may normalize to `priority(policy)`.
Bare `first` is invalid.

### 8.4 `random(seed)`

```text
resolve(
  lineage_op(principal_tree(pearltrees), estimand='hop_decay'),
  selector=random(
    seed=20260726,
    scope='per_expression',
    prng='pcg64-v1'
  )
)
```

Identity-bearing random resolution is restricted to `per_expression`. The receipt binds:

- canonical candidate-set digest;
- PRNG algorithm and version;
- seed;
- exact sampling algorithm; and
- selected candidate digest.

Per-row or per-epoch selection does not resolve to one fixed AST. It must instead become an
explicit stochastic process node whose assignment manifest, scope, PRNG, and seed are semantic
and factory-bound.

### 8.5 Aggregation creates a new process

Aggregation is not interpretation and is never implicit. It first requires:

```prolog
compatible_for_aggregation(A, B, CompatibilityReceipt).
```

The receipt proves matching:

- estimand ID;
- indexed domain and codomain;
- direction;
- units and score scale;
- calibration contract;
- support and missing-value semantics; and
- normalization.

Only then may an explicit combiner create a new precise process:

```text
combine_scores(
  [
    weighted_member(
      scorer_a,
      candidate_sha256="<64-hex-candidate-a>",
      weight=0.4
    ),
    weighted_member(
      scorer_b,
      candidate_sha256="<64-hex-candidate-b>",
      weight=0.6
    )
  ],
  aggregate='calibrated_weighted_mean',
  compatibility_sha256="<64-hex-compatibility-receipt>"
)
```

Weights are keyed by full candidate digest, not positional order. Permuting input display order
cannot reassign weights. Incompatible interpretations such as hop targets and semantic μ cannot
be combined by this operator.

The angle-bracket strings are metavariables; identity-bearing records contain exactly 64
lowercase hexadecimal characters.

## 9. Resolution receipts

Interpretation and representation use the same receipt schema with different `stage` values:

```text
ResolutionReceipt {
    schema
    stage: interpretation | representation
    source_registry_version
    source_registry_sha256
    target_registry_version
    target_registry_sha256
    type_system_version
    type_system_sha256
    resolver_version
    resolver_sha256
    prolog_adapter_version
    prolog_adapter_sha256
    prolog_engine_id
    prolog_engine_version
    prolog_engine_artifact_sha256
    prolog_conformance_profile_version
    prolog_conformance_profile_sha256
    pure_allowlist_version
    pure_allowlist_sha256
    resource_policy_version
    resource_policy_sha256
    ruleset_version
    ruleset_sha256
    source_ast_sha256
    candidates: [
        {
            canonical_typed_ast
            typed_ast_sha256
            deriving_rule_ids
        }
    ]
    candidate_set_sha256
    selector
    selected_typed_ast_sha256
}
```

Canonical encoding is the repository's frozen canonical JSON profile:

- UTF-8;
- sorted object keys;
- compact separators;
- no duplicate keys;
- no unknown fields;
- numbers represented by their typed canonical strings; and
- no NaN or infinity.

Receipt digest:

```text
sha256(
  "unifyweaver.process-expression-resolution-receipt.v1" ||
  0x00 ||
  canonical_json_bytes(receipt_without_digest)
)
```

`ruleset_sha256` covers clauses, helper predicates, imported tables, and every transitive
dependency. `candidate_set_sha256` hashes the canonical JSON array of candidate records in
canonical-AST byte order. Verifiers recompute every derived digest; they never trust self-reported
hash fields.

The resource policy freezes inference, depth, candidate, and wall-independent step caps. The
receipt records cap exhaustion as an error and never hashes a partial set. The engine artifact is
an OCI image digest or a canonical executable-plus-standard-library manifest digest; the
conformance profile binds the required reader, term-order-independent, arithmetic, and resource
semantics. Engine, adapter, allowlist, and resource bindings make candidate completeness
reproducible rather than dependent on an ambient Prolog installation. A matching ID/version with
a different artifact digest fails verification.

Resolution history is provenance, not semantic identity:

- direct authoring and rule-based derivation of the same precise AST share a semantic digest;
- `unique` and `priority` selecting the same precise AST share a semantic digest; and
- their distinct receipts remain cross-linked for audit.

If selector behavior itself occurs at runtime, it must be represented as a precise semantic AST
node rather than hidden only in a receipt.

Reference records are discriminated:

```text
DirectReferenceExpression {
    precise_ast
    provenance
}

ResolvedReferenceExpression {
    coarse_ast
    semantic_ast
    precise_ast
    interpretation_receipt
    representation_receipt
    provenance
}
```

A direct precise author does not invent an abstract predecessor or dummy resolution receipt.

## 10. Factory verification and identity

### 10.1 Stable realization versus verification event

A stable factory realization binds the artifacts that can change the produced labels:

```text
FactoryRealization {
    schema
    precise_ast_sha256
    precise_registry_version
    precise_registry_sha256
    artifacts: [
        {
            role
            kind: code | data_manifest | model_checkpoint | prompt_template | configuration
            sha256
        }
    ]
}
```

Roles are registry-declared, for example `implementation`,
`substrate_manifest`, `ancestor_attention_checkpoint`, `judge_checkpoint`, and
`judge_prompt_template`. A role cannot be replaced by another artifact merely because the digest
set is unchanged.

Artifact records sort by `(role, kind, sha256)`. Singleton roles reject duplicates; explicitly
multi-valued roles deduplicate by digest and retain canonical order. Missing required roles,
unknown roles, and a digest moved to the wrong role all fail.

Its stable fingerprint is:

```text
sha256(
  "unifyweaver.process-expression-factory-realization.v1" ||
  0x00 ||
  canonical_json_bytes(realization)
)
```

A verification event is separate:

```text
FactoryVerificationReceipt {
    schema
    realization
    realization_fingerprint
    verification_policy_version
    verification_policy_sha256
    checks
    verifier_identity
    verified_at
    signature_algorithm: ed25519
    signing_key_id
    signed_payload_sha256
    signature_base64
}
```

Time, verifier identity, and signature do not alter the stable realization fingerprint. Rechecking
the same artifacts may produce a new verification receipt without changing process identity.

The signed payload is:

```text
"unifyweaver.process-expression-factory-verification-signed.v1" ||
0x00 ||
canonical_json_bytes(
  receipt without signed_payload_sha256, signature_base64, or receipt_digest
)
```

`signed_payload_sha256` is the SHA-256 of those bytes. `signature_base64` is an Ed25519 signature
over the bytes themselves. The content-addressed verification policy names the trusted key store,
allowed key IDs, and revocations. A verifier resolves `signing_key_id` only through that store and
checks both the payload digest and signature.

Verification-receipt digest:

```text
sha256(
  "unifyweaver.process-expression-factory-verification.v1" ||
  0x00 ||
  canonical_json_bytes(receipt_without_digest)
)
```

The canonical JSON and fail-closed schema rules from §9 apply.

### 10.2 Verification is a binding operation

```text
verify_factory_receipt(
    ResolvedGroundAST,
    FactoryVerificationReceipt,
    Policy
) -> VerifiedProcess | error
```

The verifier requires:

- exact canonical precise AST bytes and digest;
- precise registry version and content digest;
- every required role-keyed code, data, model, prompt, and configuration digest;
- all required artifacts present and content-matching;
- every optional pin consistent with the receipt;
- a passing policy result; and
- no unknown or duplicate receipt fields.

An arbitrary nonempty “factory fingerprint” supplied by a caller is not verification.

### 10.3 Semantic and deployed identity

vNext introduces an explicit identity schema independent of `pec-v3`:

```text
identity_schema_version = "peid-v1"
```

Semantic identity core:

```json
{
  "identity_schema_version": "peid-v1",
  "precise_registry_version": "<version>",
  "precise_registry_sha256": "<64 lowercase hex>",
  "typed_ast_schema_version": "pe-typed-ast-v1",
  "canonical_precise_typed_ast": {
    "schema": "pe-typed-ast-v1",
    "root": "<tagged AST object>"
  }
}
```

Semantic digest:

```text
sha256(
  "unifyweaver.process-expression-semantic-identity.peid-v1" ||
  0x00 ||
  canonical_json_bytes(semantic_identity_core)
)
```

Deployed identity core:

```json
{
  "identity_schema_version": "peid-v1",
  "semantic_process_sha256": "<64 lowercase hex>",
  "factory_realization_sha256": "<64 lowercase hex>"
}
```

Deployed digest:

```text
sha256(
  "unifyweaver.process-expression-deployed-identity.peid-v1" ||
  0x00 ||
  canonical_json_bytes(deployed_identity_core)
)
```

The deployed record retains canonical AST bytes, registry metadata, semantic digest, realization
fingerprint, deployed digest, and verification-receipt digest.

Resolution receipts and pins are excluded from semantic AST bytes. They remain cross-linked
provenance. Two paths to the same precise AST and stable factory therefore have one deployed
identity.

## 11. Pins and checkpoints

### 11.1 Pin rules

vNext pins have registered kinds:

```text
process(...)@implementation_commit/<full-id>
process(...)@graph_manifest/sha256-<64-lowercase-hex>
```

The angle-bracket values above are metavariables, not valid literal examples.

Each pin retains the node it qualifies:

```text
ProvenancePin {
    target_role_path: [RolePathSegment]
    target_typed_ast_sha256
    kind
    value
}
```

`target_role_path` is a tagged segment array:

```text
RolePathSegment :=
    {"kind":"arg",           "index": uint}
  | {"kind":"field",         "name": canonical_field_name}
  | {"kind":"list_item",     "index": uint}
  | {"kind":"modified_base"}
```

Indices are zero-based. Field names are exact registry-canonical Unicode strings and use ordinary
canonical-JSON escaping; paths never use slash-separated display text. The root path is `[]`.
Examples:

```json
[]

[{"kind":"arg","index":0}]

[
  {"kind":"arg","index":0},
  {"kind":"field","name":"estimator"},
  {"kind":"list_item","index":1}
]
```

The wire encoding is the canonical JSON array itself. Segment objects reject unknown fields;
`arg` and `list_item` require only `index`, `field` requires only `name`, and `modified_base`
allows neither. The parser first records the surface attachment; after elaboration it binds the
canonical semantic-node path and digest. Zero-based indices distinguish identical sibling
subtrees even when their node digests match. Moving a pin from one child to another therefore
changes the provenance envelope even though pins remain outside the semantic AST digest.

An interpretation or representation rule may transfer a pin to a new precise node only through
an explicit, receipt-recorded target mapping. Otherwise the pin remains provenance of the source
expression and does not silently qualify the selected implementation.

For identity-bearing envelopes:

- unknown pin kinds fail closed;
- each target/kind pair declares singleton or multi-valued cardinality;
- duplicate singleton target/kind pairs fail;
- conflicting values fail;
- multi-valued target/kind pairs deduplicate and sort by canonical value;
- canonical pin display sorts by target role path, registered kind, then value; and
- a pin disagreeing with a verified receipt fails verification.

Pins are optional. The verified factory realization is authoritative.

Types do not belong in pins:

```text
C::corpus          # correct
C@type/corpus      # wrong
```

Selectors and `impl` do not belong in pins:

```text
lineage_op(S,impl='graph_walk')  # correct
S@impl/graph-walk                # wrong
```

v0 bare pins such as `@e5-small-v2` have no automatic vNext meaning. A migration manifest must map
each one to a registered kind or tombstone the expression. The vNext parser never guesses.

### 11.2 Attention checkpoints are not decoder checkpoints

Checkpoint kinds are role-specific:

| Checkpoint kind | Role | Binding location |
|---|---|---|
| `ancestor_attention_checkpoint` | A learned attention estimator used inside an explicit lineage approximation. | That estimator's factory realization. |
| `process_expression_encoder_checkpoint` | Typed process AST to bottleneck `z`. | Conditioning/adapter receipt, outside process identity. |
| `process_expression_decoder_checkpoint` | Bottleneck `z` to reconstructed process candidate. | Decoded-candidate receipt, outside source-process identity. |
| `filing_decoder_checkpoint` | Filing path/action proposal. | Filing-decision receipt. |
| `judge_model_revision` | Model used to produce labels. | Label-factory realization. |

Therefore an `ancestor_attention_checkpoint` is **not** the process-expression decoder
checkpoint. If the reconstruction decoder internally uses transformer attention, its artifact is
still named `process_expression_decoder_checkpoint`; architecture does not determine scope.

Changing an ancestor-attention estimator checkpoint may change the approximating process factory.
Changing the process-expression encoder or decoder changes a derived conditioning or diagnostic
artifact, not the semantic identity of the process being described.

## 12. Decoder-origin boundary

Decoder output is untrusted candidate generation:

```text
DecodedCandidate {
    raw_output_bytes
    parsed_ground_ast
    receipt
}

DecodedCandidateReceipt {
    schema
    raw_output_sha256
    parsed_ast_sha256
    functional_dialect_version
    functional_dialect_sha256
    parser_version
    parser_sha256
    typed_ast_schema_version
    type_system_version
    type_system_sha256
    registry_version
    registry_sha256
    decoder_checkpoint_sha256
    decoding_config_sha256
    seed
    source_encoded_process_receipt_sha256
}
```

The official decoder parser preserves this wrapper. Parsing and canonicalizing do not clear
origin. Candidate caches remain separate from deployed caches and residual tables.

A caller could copy the text and invoke the ordinary parser. The decisive defense is that even a
freshly parsed `GroundAST` cannot mint identity: deployed identity APIs require `VerifiedProcess`,
and a caller-supplied fingerprint is insufficient.

The only promotion path is:

```text
verify_and_promote(
    DecodedCandidate,
    matching FactoryVerificationReceipt,
    Policy
) -> VerifiedProcess
```

The verifier binds the exact parsed AST and artifacts. This closes text-reparse laundering while
still allowing a decoder to rediscover a genuinely registered and independently verified process.

## 13. End-to-end examples

### 13.1 Typed pattern to ground substrate

Pattern:

```text
hop_decay_targets(
  S::substrate[C],
  decay=D::real,
  hop_origin=H::int,
  direction='ancestor',
  depth=Limit::depth_limit
)
```

Bindings:

```text
C = pearltrees
S = principal_tree(pearltrees)
D = 0.85
H = 1
Limit = bounded_depth(4)
```

Ground semantic term:

```text
hop_decay_targets(
  principal_tree(pearltrees),
  decay=0.85,
  hop_origin=1,
  direction='ancestor',
  depth=bounded_depth(4)
)
```

It is semantically explicit but still needs one exact representation and a verified factory.

### 13.2 Coarse ground expression

```text
lineage_op(
  principal_tree(pearltrees),
  family_spec='lineage_interpretations_v1'
)
```

Possible interpretations:

```text
hop_decay_targets(...)
structural_lineage_score(...)
semantic_transitive_mu(...)
```

These are different estimands. `unique` fails. A versioned interpretation policy may choose one,
but its receipt must show every candidate and the choice. Aggregating them is forbidden.

### 13.3 Explicit estimand, several exact representations

Semantic term:

```text
hop_decay_targets(
  principal_tree(pearltrees),
  decay=0.85,
  hop_origin=1,
  direction='ancestor',
  depth=unbounded_depth
)
```

Equivalent representation candidates:

```text
graph_walk_hop_decay(<semantic-term>)
materialized_hop_decay_table(<semantic-term>)
compiled_hop_decay_lookup(<semantic-term>)
```

Selection:

```text
selector=unique
# Fails if all three are registered.

selector=priority('hop_decay_impl_policy_v1')
# Chooses by a content-bound order over candidate AST digests.

selector=random(seed=7,scope='per_expression',prng='pcg64-v1')
# Chooses reproducibly and records the candidate-set digest.
```

### 13.4 Direct precise authoring

```text
graph_walk_hop_decay(
  hop_decay_targets(
    principal_tree(pearltrees),
    decay=0.85,
    hop_origin=1,
    direction='ancestor',
    depth=unbounded_depth
  )
)
```

If recursively complete, it validates directly as `ResolvedGroundAST(origin=direct)`. It needs no
dummy interpretation receipt. Once bound to the same verified factory, it has the same identity as
the same AST reached through rules.

### 13.5 Function and entity ownership

Pattern:

```text
apply(
  F::function([entity[S],entity[S]],score_value[unit_interval]),
  Destination::entity[S],
  Candidate::entity[S]
)
```

Ground binding:

```text
S = principal_tree(pearltrees)
Destination = entity_ref(S, "destination-17")
Candidate = entity_ref(S, "candidate-42")
F = structural_lineage_score(S, ...)
```

A `simplewiki` entity cannot satisfy either `entity[S]` argument for this binding.

### 13.6 Invalid forms

| Expression or action | Failure |
|---|---|
| `lineage_op(haiku)` | `judge` is not `substrate[C]`. |
| `blend(pearltrees,simplewiki)` | Corpora are not scorer descriptors. |
| `hop_decay_targets(haiku,...)` | Factory requires a substrate. |
| `depth=3.5` | Fractional real does not satisfy `depth_limit`. |
| `depth='unbounded'` | Bare atom is not the `unbounded_depth` constructor. |
| `estimand='structural_score',decay=0.85` | Field is not consumed by that interpretation. |
| duplicate `decay` fields | Duplicate named field. |
| bare `first` | Clause/source order is not a policy. |
| `random()` | Seed, PRNG, and per-expression scope are absent. |
| combine hop targets with semantic μ | Estimand and codomain compatibility fails. |
| use `GroundAST` as deployed identity | It may still be abstract or unverified. |
| pass a nonempty caller fingerprint | It is not a verified factory binding. |
| use decoder text after reparsing | It still lacks `VerifiedProcess`. |

## 14. Current v0 boundary

The following are parser-valid v0.3 forms:

```text
haiku
gpt-5.5-low
luna.D
margin(t=0.01)
kalman(luna.D,luna.S)
blend(luna.D,luna.S,w=[0.4,0.6])
e5(routing(e5,haiku,t=[0.02],menus=[10]))
```

Some parser-valid v0.3 forms expose the type conflation and are not endorsed vNext semantics:

| v0.3 form | Status |
|---|---|
| `lineage(graph,decay=0.85)` | Historically identified target-factory intent, but lacks corpus/substrate view and full estimand fields. Ambiguous migration. |
| `lineage(haiku,decay=0.85)` | Parser accepts it because `haiku` is the broad v0 `source`; semantically invalid. Tombstone unless an explicit reviewed meaning exists. |
| `menu(graph,n=10)` | Parser-valid, but its vNext input and output signature requires separate review. No automatic mapping. |
| `lineage(fs,decay=0.85)` | Not v0.3 parser-valid because `fs` is unregistered. Regenerate under vNext; do not fabricate a legacy identity. |

No v0 identity is parsed or “upgraded” with the vNext parser. The frozen v0.3 parser and legacy
verifier remain authoritative for historical bytes.

## 15. Version axes and sealed migration

vNext versions these axes independently:

| Axis | vNext identifier |
|---|---|
| Functional dialect | `pe-functional-v1` |
| Canonical Prolog adapter | `pe-prolog-v1` |
| Type system | `pe-types-v1` |
| Canonical typed AST | `pe-typed-ast-v1` |
| Abstract registry | version plus content SHA-256 |
| Precise registry | version plus content SHA-256 |
| Interpretation rules | version plus transitive content SHA-256 |
| Representation rules | version plus transitive content SHA-256 |
| Selector policies/PRNG | version plus content SHA-256 |
| Semantic/deployed identity | `peid-v1` |
| Structural stream/role paths | `pec-v3` |
| Golden schema | `unifyweaver.process-expression-golden.v2` |
| Golden bundle identity | bundle ID, independent revision, and full file SHA-256 |
| Release record | `unifyweaver.process-expression-release.v1`, immutable file and SHA-256 |

The registry identifiers are assigned only when their checked-in content is frozen. A friendly
version without its content digest is insufficient.

### 15.1 Identity migration manifest

The migration scope is a frozen `LegacyIdentityInventory`, not the infinite set of grammar-valid
strings. The inventory content-addresses:

- every identity in the v1 and v2 sealed bundles;
- every entry in the checked-in v0.3 `PROCESSES` registry;
- every committed artifact or manifest declaring a v0.3 `process_expression`; and
- any cache inventory explicitly nominated for migration.

The inventory deduplicates exact legacy identity keys and has its own schema, canonical JSON
bytes, and SHA-256. Because v0.3 registry version is in the legacy AST-digest preimage, every
inventory item requires a migration row even if its display spelling appears unchanged:

```text
IdentityMigrationRow {
    schema
    from_registry_version
    from_registry_sha256
    from_identity_preimage
    old_canonical_bytes
    old_full_digest
    old_factory_fingerprint | null
    to_registry_version
    to_registry_sha256
    to_identity_schema_version
    new_canonical_bytes | null
    new_semantic_digest | null
    status: mapped | ambiguous | tombstoned
    reason
    predecessor_kind: semantic | deployed
    predecessor_identity_key
}
```

For a row with a verified legacy factory fingerprint, `predecessor_kind` is `deployed` and the key
binds `old_full_digest|old_factory_fingerprint`. Otherwise it is `semantic` and the key is
`old_full_digest`. The migration manifest is canonical and content-addressed; its exact SHA-256 is
bound in the new golden bundle and release pointer. No optional signature mode is substituted.

- `mapped` yields a proposed precise AST plus migration receipt; factory verification is still
  required before deployed identity.
- `ambiguous` requires a human or versioned policy ruling.
- `tombstoned` preserves history and forbids promotion.

Old and new digests are cross-linked, never aliased. `lineage(graph,...)` cannot auto-map without
a substrate and estimand ruling. The invalid `lineage(fs,...)` artifact is regenerated rather than
migrated.

### 15.2 Release record

One immutable release record binds the migration and golden chain:

```text
ProcessExpressionRelease {
    schema: unifyweaver.process-expression-release.v1
    release_id
    golden_bundle_path
    golden_bundle_id
    golden_bundle_revision
    golden_bundle_sha256
    legacy_inventory_sha256
    migration_manifest_sha256
    abstract_registry_sha256
    precise_registry_sha256
    type_system_sha256
    typed_ast_schema_version
    identity_schema_version
    stream_contract_version
}
```

Release-record digest:

```text
sha256(
  "unifyweaver.process-expression-release.v1" ||
  0x00 ||
  canonical_json_bytes(record_without_digest)
)
```

The reference loader exposes two reviewed constants:

```text
CURRENT_PROCESS_EXPRESSION_RELEASE
CURRENT_PROCESS_EXPRESSION_RELEASE_SHA256
```

The first names an immutable release-record file; the second pins its exact bytes. Verification
recomputes the release digest, then the golden bundle, legacy inventory, migration manifest, and
registry digests, and requires every child to match the release record. A path match without a
digest match fails. Updating “current” creates a new immutable release record and changes both
constants in the same reviewed change.

### 15.3 Golden lifecycle

The sealed legacy files are:

```text
PROCESS_EXPRESSION_GOLDEN_v1.json
sha256 b053351a2a419ac58b7ab644afe15c60543846ce8b9d5a3d9bcbc332ca24db29

PROCESS_EXPRESSION_GOLDEN_v2.json
sha256 85e6421f5a1347fca5937d1243dc01500a9aa5b7221571b4918248e57ece6344
```

Activation of vNext must atomically:

1. freeze registries, dialects, types, AST schema, rulesets, identity schema, `pec-v3`, migration
   inventory, and migration manifest;
2. implement the small reference parser/serializer and fixture generator needed to produce and
   independently verify canonical AST, token, and role-path bytes;
3. create `PROCESS_EXPRESSION_GOLDEN_v3.json` under the new golden schema, with bundle ID
   `process-expression-vnext`, bundle revision `1`, and its independent file SHA-256;
4. bind both registries, both dialects, types, canonical AST, rulesets, identity schema, `pec-v3`,
   migration inventory, and migration manifest;
5. create and verify the immutable release record, then move both current-release constants,
   `CURRENT_GOLDEN_BUNDLE`, and every consumer document together;
6. add v2 to `SUPERSEDED_GOLDEN_BUNDLES` with the exact hash above;
7. retain the existing v1 archive entry and hash;
8. reject any attempt, including override flags, to overwrite a known sealed path; and
9. dispatch legacy verification by the bundle's own registry, identity convention, and stream
   contract.

The reference implementation in step 2 is allowed before sealing because it creates the bytes
that are reviewed and sealed. Production generator, tokenizer, encoder, decoder, training, and
deployment work begins only after step 9. A later registry-only change creates a new bundle
revision and file while retaining `pec-v3` if the structural stream contract is unchanged.

The v3 fixture coverage set is checked in before generation and contains:

- every AST node and type constructor;
- both dialects and all registry punctuation classes;
- variables, repeated variables, `_`, indexed ownership, and redundant/conflicting annotations;
- exact `int`, high-precision `real`, `float64`, positive-zero, and negative-zero cases;
- each lineage estimand, representation constraint, depth constructor, and targeted nested pin;
- direct and rule-derived precise ASTs with receipt vectors;
- role-keyed factory artifacts and decoder-origin records; and
- negative vectors for unknown fields, duplicate fields, type conflicts, unresolved terms,
  invalid pins, cap exhaustion, and tampered receipts.

## 16. Acceptance criteria

A vNext implementation is not complete until tests demonstrate:

1. functional and Prolog surfaces normalize to byte-identical typed ASTs;
2. `Reference`, `Call`, `Atom`, and `String` cannot collide;
3. `_` is fresh per occurrence and repeated named variables unify;
4. alpha-equivalent patterns share a pattern digest while preserving equality structure;
5. `list[real]` maps to Prolog `list(real)` and every indexed type round-trips;
6. all registry names, including dotted and hyphenated names, have collision-free Prolog forms;
7. redundant `::` annotations preserve identity and contradictory annotations fail before hashing;
8. `1`, `1.0`, `1e0`, a beyond-binary64 decimal, positive zero, and negative zero follow the
   declared `int`/`real`/`float64` rules in both dialects;
9. legacy `pec-v2` still emits NUMBER/REAL for `margin(t=1)`;
10. pattern, ground-abstract, semantically incomplete, and unverified terms are rejected by every
    deployed identity, cache, residual, and adapter API;
11. `lineage_op(haiku)`, `hop_decay_targets(haiku,...)`, and
    `blend(pearltrees,simplewiki)` fail type checking;
12. entity/substrate ownership prevents cross-corpus application, and relational lineage scorers
    expose both destination/truth and candidate roles;
13. every supplied option is consumed exactly once; duplicate, unknown, and ignored fields fail;
14. a request without `estimand` or `family_spec` is underconstrained; a content-bound family spec
    expands to its exact finite interpretation set;
15. known `impl` constraints reach `representation/4`, unknown values return `no_candidates`, and
    absence never creates an implicit deployable default;
16. every interpretation candidate names exactly one estimand, and no selector treats distinct
    interpretation candidates as aggregation-compatible;
17. representation candidates have identical full typed signatures and estimand IDs;
18. zero candidates, duplicate derivations, and resource-cap exhaustion have distinct fail-closed
    results;
19. rule evaluation uses only the deterministic pure allowlist;
20. resolver, content-bound Prolog engine artifact/conformance profile, adapter, allowlist, and
    resource-policy bindings reproduce candidate completeness and reject same-version/different-
    artifact engines;
21. candidate ordering is canonical-byte order, independent of Prolog clause order;
22. priority policies order candidate AST digests and bind their own content digests;
23. identity-bearing random selection is reproducible and restricted to per-expression scope;
24. aggregate weights are keyed by digest and compatibility receipts reject scale, direction,
    calibration, support, or missingness mismatches;
25. two derivations of the same precise AST share semantic identity but retain distinct receipts;
26. direct and rule-derived identical precise ASTs share semantic identity;
27. nested pins round-trip with tagged target role path and node digest; identical sibling
    subtrees, escaped field names, and moving a pin to another child remain distinguishable;
28. factory artifact record ordering is invariant, while a role swap, duplicate singleton role,
    missing role, or unknown role fails;
29. factory signatures cover the frozen payload, verify under Ed25519 and a content-bound trust
    policy, and fail for unknown or revoked keys;
30. factory verification rejects a nonempty but unverified fingerprint, missing artifact, digest
    mismatch, pin mismatch, unknown field, duplicate field, tampered receipt, and failing policy;
31. reverifying one stable realization does not change deployed identity;
32. decoder receipts reproduce parsing under the bound dialect, parser, AST schema, type system,
    and registry;
33. decoder candidates, copied text, and reparsed text cannot mint identity without independent
    factory verification;
34. every committed artifact field named `process_expression` parses under its declared registry;
35. migration rows cover every item in the content-bound legacy inventory and use only `mapped`,
    `ambiguous`, or `tombstoned`;
36. historical expressions are verified by the v0.3 parser, never reinterpreted by vNext;
37. the complete v1 and v2 golden files retain the exact SHA-256 values above;
38. bundle ID/revision changes independently of `pec`, and the v3 fixture contains the complete
    declared positive and negative coverage set;
39. release-record verification rejects a tampered record, a current-record digest mismatch, and
    any bundle, inventory, migration, or registry child-digest mismatch;
40. the reference generator, current-release constants, bundle pointer, loader, and documents move
    atomically to the new bundle; and
41. no code path or override flag can mutate a sealed golden bundle or release record.
