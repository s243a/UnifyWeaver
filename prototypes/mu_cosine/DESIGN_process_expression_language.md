# Process-Expression Language — Specification

Declarative functional expressions that name the DATA-GENERATING PROCESS of a training target, and
map deterministically to a conditioning embedding. The model is always told *what function produced
the label it is fitting* — at a chosen level of specificity.

## Grammar (v0, amended per review #3974-r1)

```
Expr     := Atom | Apply | Expr "." Mod              # dotted modifier: variant/channel (sonnet.lineage, luna.D)
Apply    := Name "(" [Args] ")"                      # a Name may be BOTH atom and operator —
Args     := Arg ("," Arg)*                           #   e5 (embedding source) vs e5(...) (ranker over
Arg      := Expr | Kwarg | Pin                       #   an inner process); resolved by the registry
Kwarg    := Ident "=" Val                            # canonical: sorted by kw, defaults ELIDED
Pin      := Expr "@" PinLit                          # provenance pin (V3 only): rev, date, hash
Atom     := Name                                     # any registry-registered leaf source/judge
Val      := Number | List | String | Name
List     := "[" Val ("," Val)* "]"
String   := '"' /[^"]*/ '"'
Number   := /-?[0-9]+(\.[0-9]+)?/ ;  Ident := /[a-z][a-z0-9_]*/ ;  Mod := /[A-Za-z][A-Za-z0-9_-]*/
PinLit   := /[A-Za-z0-9._\/-]+/
Name     := longest match against the registry's reserved name vocabulary
```

**Registry-driven lexing (review r2 item 1):** names like `gpt-5.5-low` contain `.` and `-`, which
would collide with the modifier and minus syntax under naive lexing. Resolution: the lexer
longest-matches `Name` tokens against the versioned registry vocabulary FIRST, so any registered
name — dots, hyphens and all — is one token; `.Mod` parsing applies only to text following a
complete parsed Expr. `Mod` permits uppercase (channel modifiers `luna.D`, `luna.S`). Unregistered
bare words are a parse error (no guessing).

**Typed operator-signature registry (versioned):** every Ident used as an operator has a registry
entry — arity, positional/kw parameter types + defaults, output type (score | pick | target-set),
and whether the same Ident is also a valid atom. The registry version is part of process identity.
The P0 round-trip requirement (parse ∘ render = id) is against THIS grammar + registry, and the
registry must parse every example in this document (the v0 acceptance test).

Examples (the deployed three-tier filing policy, at three verbosities):

```
V1  e5(routing(e5, sonnet))
V2  e5(routing(e5, sonnet.lineage, t=[0.02,0.03], menus=[10,20]))
V3  e5(routing(e5@e5-small-v2, sonnet.lineage@2026-07-23/menu-order-blind,
       t=[0.02,0.03], menus=[10,20], manifest=fcf5e1d6))
```

Other current processes, expressed: `kalman(luna.D, luna.S)` (the "kalman-fused" judge token),
`blend(graph.discrim, llm.element, llm.subcat)` ("dir-blend"), `lineage(graph, decay=0.85)`
(the LINEAGE target factory), `distill(e5(routing(...)))` (the label-factory corpus).

## Process identity vs training cards (review finding 2 — the load-bearing split)

- **Identity is LOSSLESS:** a process is identified by its canonical AST (all args, all pins,
  registry version) plus a factory/manifest fingerprint (hashes of the code + data manifests that
  realize it). This is what P4 provenance headers carry, and what "distinct factories must remain
  distinguishable" means. Two processes are the same iff their canonical ASTs + fingerprints match.
- **Cards are LOSSY, derived views:** V0–V3 renderings are projections of the canonical AST for
  CONDITIONING only. A V1 card deliberately merges distinct processes into an equivalence class —
  that is its job — and is therefore never a process identifier.
- **Cache keys:** embedding caches are keyed on (canonical-AST hash, verbosity level, renderer
  version, embedding model revision, e5 prefix mode) — never on the rendered string alone.

## Canonicalization rules

1. Kwargs sorted; values in canonical numeric form; DEFAULTS ELIDED in renderings (the canonical
   AST keeps explicit resolved values for identity).
2. `judge.modifier` dot-syntax for judge variants (`sonnet.lineage` = sonnet with lineage-context
   menus); `@` pins (model revision, date, prompt hash) rendered at V3 only, always present in
   the canonical AST.
3. One canonical string per (AST, verbosity level, renderer version).

## Verbosity levels

| level | content | role |
|---|---|---|
| V0 | empty card | agnostic/unconditional row (the agnostic-anchor precedent) |
| V1 | fn names + sources, defaults elided | the WORKHORSE — most training rows |
| V2 | + decision-relevant hyperparams (t, N, decay) | distinctions that change the label distribution |
| V3 | + pins (versions, dates, manifest hashes) | provenance/audit; rare in training |

## Expression → embedding

Phase 1 (honest about the code — review finding 3): NameFunctionCond is an integer-indexed frozen
card table + per-index learned residual (`forward(idx) = W(name_e5[idx]) + resid(idx)`,
mu_attention.py:399). So phase 1 = **table expansion**: each registered expression appends a card
row (its rendered string, e5-embedded) and receives a learned residual — the existing judge
onboarding, not a token-free path. For UNSEEN expressions the shared projection gives the dynamic
path `W(e_expr)` with residual = 0 (pure name prior) — a one-line forward generalization, used at
inference/zero-shot only until P3 makes it a trained pathway.

Phase 3 (compositional): tree encoding over the AST — node embedding = card-e5(fn) combined with
position-weighted child embeddings (the `random_operator_embedding` superposition precedent) or a
small learned tree encoder — so UNSEEN compositions (swap a judge, move a threshold) land near
their relatives and can be conditioned on zero-shot.

## Scope

The language names processes; it does not execute them. An expression is valid iff we can map it
to a concrete training-set construction (the target factory that realizes it). Exactness is not
required — the criterion is: two expressions differ iff the label distributions they induce differ
enough to matter (see the philosophy doc's information objective).
