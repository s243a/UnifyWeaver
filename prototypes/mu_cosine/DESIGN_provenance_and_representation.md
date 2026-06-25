# Provenance axes & node representation — what gets a learned table vs frozen e5

Design decisions from the multi-corpus (SimpleMind + Pearltrees + enwiki) work, on where to spend learned
parameters. Companion to `DESIGN_calibrated_judges.md` §7 (operators / node-types).

## The dividing line: **if the table grows with the data, it is NOT a learned table**

Learned **factored tokens** are only for **small, closed, categorical** axes — a handful of values, each
seen often enough to train. Everything **open-ended** rides **frozen e5** instead (the reason nodes have no
per-node embedding: cold-start safety, no overfitting, no table that grows with the corpus).

| axis | kind | values | mechanism |
|---|---|---|---|
| `op` | closed | SYM / WIKI / ELEM / LLM | learned token + readout row |
| `corpus` (source) | closed | simplewiki / enwiki / pearltrees / mindmap | learned token (maskable provenance) |
| `judge` | closed | haiku / graph / human | learned token (maskable provenance) |
| **`account`** | closed | **`s243a` / `s243a_groups`** | learned token (maskable provenance) — NEW |
| `nodetype` | closed | category / page / mindmap_node / pearltrees_collection | learned token (+ optional transform, below) |
| **node identity** | open | every concept | **frozen e5** (no table) |
| **group** | open | every Pearltrees group | **frozen e5**, optionally a *transform* (below) — NOT a per-group table |

## Account (the two Pearltrees accounts)

`account ∈ {s243a, s243a_groups}` is a 2-value token in the **maskable provenance** slot, alongside
`corpus ⊗ judge`. Maskable ⇒ an account-agnostic default (marginalize it out) plus the ability to condition
when the accounts differ in convention/quality. The field is already in the harvested `trees.account` /
API `asso`, so carrying it through `parse_pearltrees.py` → the graded pairs is free. Gate `--use-account`,
default off; activate once **both** accounts' data is in (single-account data makes `account` collinear with
`corpus` — node-type's collinearity trap again).

## Groups: a transform of e5, never a per-group table

A Pearltrees **group** is open-ended (grows with the data), so it is a **node with frozen e5**, not a
learned table. When group-level conditioning is wanted *beyond* what e5 + the `account` token give, use a
**learned transform of e5** — `group_repr = T(e5_group)` — NOT a per-group row:

- **Parametric & shared.** `T` is the parameters, not a per-entity row, so it never grows with the number
  of groups and **generalizes to unseen groups** (any new group's e5 → `T` → a sensible rep).
- **Piggybacks e5.** Initialise `T` **near identity** ⇒ at warm start a group's rep *is* its e5 (full
  piggyback on e5's pretrained semantics); fine-tuning learns only the group-specific **delta**. Same bet
  as the rest of the model (frozen e5 carries meaning; learned parts are small deltas).
- **Generalises node-type.** Today node-type conditions e5 *additively* (`e5 + nodetype_emb[type]`, a per-
  type shift). A transform conditions it *multiplicatively* (`T_type(e5)`, a per-type projection that can
  rotate/rescale/reweight e5 dims) — strictly more expressive, same cold-start safety.

### Choices to settle when we build it
- **Input e5:** the group's **title** e5 (cheap, uniform) vs the **centroid of its members' e5** (a group
  *is* its contents) vs both concatenated. Lean: member-centroid for collections.
- **Scope:** one `T_group`, or **per-node-type** transforms (`T_category`, `T_page`, `T_group`, …) making
  the whole node-type axis projection-based.
- **Form:** start a **Linear** `d→d` (~150K params at d=384), init near identity; MLP only if it underfits.
- **Compose or replace:** keep the additive `nodetype_emb` *and* add `T`, or let `T` subsume it.

### Discipline (same as node-type)
Init near identity, **gate** (`--group-transform`), and **A/B** it — only once group/multi-account data is
flowing. Adding an expressive transform before the data has group diversity is node-type's collinearity
problem with more parameters to overfit.
