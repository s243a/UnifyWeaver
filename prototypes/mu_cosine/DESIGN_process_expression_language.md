# Process-Expression Language — Specification

Declarative functional expressions that name the DATA-GENERATING PROCESS of a training target, and
map deterministically to a conditioning embedding. The model is always told *what function produced
the label it is fitting* — at a chosen level of specificity.

## Grammar (v0)

```
Expr    := Source | Apply
Apply   := Fn "(" Posargs [ "," Kwargs ] ")"
Posargs := Expr ("," Expr)*
Kwargs  := kw "=" val ("," kw "=" val)*        # canonical: sorted by kw, defaults ELIDED
Source  := e5 | graph | human
         | luna | sonnet | haiku | gpt-5.5-low | gemini | opus | ...   # judge roster
Fn      := routing | pick | kalman | blend | lineage | distill | menu | margin | ...
```

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

## Canonicalization rules

1. Kwargs sorted; values in canonical numeric form; DEFAULTS ELIDED (conciseness is the norm).
2. `judge.modifier` dot-syntax for judge variants (`sonnet.lineage` = sonnet with lineage-context
   menus); `@` suffixes reserved for pins (model revision, date, prompt hash) — V3 only.
3. One canonical string per (expression, verbosity level); the string IS the cache key.

## Verbosity levels

| level | content | role |
|---|---|---|
| V0 | empty card | agnostic/unconditional row (the agnostic-anchor precedent) |
| V1 | fn names + sources, defaults elided | the WORKHORSE — most training rows |
| V2 | + decision-relevant hyperparams (t, N, decay) | distinctions that change the label distribution |
| V3 | + pins (versions, dates, manifest hashes) | provenance/audit; rare in training |

## Expression → embedding

Phase 1 (zero new architecture): canonical string → e5 passage embedding → NameFunctionCond
conditioning residual — exactly the existing judge-card onboarding path (`judge_cards.py`, r=0
name-prior). A new expression needs no new token, no retraining of the pathway.

Phase 3 (compositional): tree encoding over the AST — node embedding = card-e5(fn) combined with
position-weighted child embeddings (the `random_operator_embedding` superposition precedent) or a
small learned tree encoder — so UNSEEN compositions (swap a judge, move a threshold) land near
their relatives and can be conditioned on zero-shot.

## Scope

The language names processes; it does not execute them. An expression is valid iff we can map it
to a concrete training-set construction (the target factory that realizes it). Exactness is not
required — the criterion is: two expressions differ iff the label distributions they induce differ
enough to matter (see the philosophy doc's information objective).
