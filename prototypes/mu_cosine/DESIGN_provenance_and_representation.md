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
| **group** | open | every Pearltrees group | **frozen e5 → required transform `T`** added to the member node (below) — NOT a per-group table, never raw e5 |

## Account (the two Pearltrees accounts)

`account ∈ {s243a, s243a_groups}` is a 2-value token in the **maskable provenance** slot, alongside
`corpus ⊗ judge`. Maskable ⇒ an account-agnostic default (marginalize it out) plus the ability to condition
when the accounts differ in convention/quality. The field is already in the harvested `trees.account` /
API `asso`, so carrying it through `parse_pearltrees.py` → the graded pairs is free. Gate `--use-account`,
default off; activate once **both** accounts' data is in (single-account data makes `account` collinear with
`corpus` — node-type's collinearity trap again).

## Groups: a transform of e5, never a per-group table

A Pearltrees **group** is open-ended (grows with the data), so it is a **node with frozen e5**, not a
learned table. Group membership conditions a member node via a **learned transform of e5** that is
**added** to the node — `node_emb += T(e5_group)` — NOT a per-group row, and **never raw e5**:

- **The transform is mandatory, not optional.** A group's e5 is a point in *content space* (same space as
  node e5). Adding it **raw** onto a member node would inject *another concept's vector* into the node, not
  a membership signal. `T` is what **maps e5 → the additive node-role space** — it plays the role the table
  plays for node-type, except its input is a *continuous e5* instead of a *discrete index*:
  - node-type: `index → table-lookup → add` (the row is already a free learned vector in add-space)
  - group:     `e5 → T → add`            (e5 is frozen, so a learned map into add-space is **required**)
- **Parametric & shared.** `T` is the parameters, not a per-entity row, so it never grows with the number
  of groups and **generalizes to unseen groups** (any new group's e5 → `T` → a sensible modifier).
- **Warm-start no-op = zero output, NOT identity.** Zero-init `T`'s final layer so at warm start it **adds
  nothing**, then learns the membership delta — exactly mirroring `nodetype_emb`'s zero-init. (Identity-init
  would *add the raw group e5*, i.e. the wrong thing. Near-identity only applies to the *other* use — when a
  group appears as its **own standalone node** and `T(e5)` *is* its representation; different role.)

### Choices to settle when we build it
- **Input e5:** the group's **title** e5 (cheap, uniform) vs the **centroid of its members' e5** (a group
  *is* its contents) vs both concatenated. Lean: member-centroid for collections.
- **Scope:** one `T_group`, or **per-node-type** transforms (`T_category`, `T_page`, `T_group`, …) making
  the whole node-type axis projection-based.
- **Form:** start a **Linear** `d→d` (~150K params at d=384), init near identity; MLP only if it underfits.
- **Compose or replace:** keep the additive `nodetype_emb` *and* add `T`, or let `T` subsume it.

### Group is a *weak* structural signal (calibration & magnitude)

Group membership is **softer than category membership**: a group is user-curated and may not stay perfectly
on topic (a "Systems Theory" group can hold off-topic pearls). There is a **confidence gradient** among the
structural modifiers:

> **node-type** (a hard fact — a page *is* a page) > **category membership** (curated, reliable) >
> **group membership** (user-curated, *drifty*).

Encode that weakness two ways — both levers we already use elsewhere:

- **Soft / one-sided target.** Train group edges to a *ranged* target, `μ(node|group) ≳ low`, not a sharp
  value (cf. the cross-entropy "it can be a range (e.g. `>`)" idea) — so topical drift isn't punished as
  error the way a curated category miss would be.
- **Small-magnitude modifier.** Extra weight-decay / lower LR on `T` so it learns a **gentle nudge**, not a
  shift that overwrites the node's own e5 (the same "back-prop on it less" lever as the ELEM operator).

The weakness also **reinforces** the transform-over-table choice: a per-group table could *memorise* the
noisy membership, whereas a shared, zero-init, regularised `T` with a soft target is far more robust to drift.

### Learned per-group credibility (v2 extension; v1 stays uniform)

The weakness is **heterogeneous**: groups vary *a lot* in internal source credibility / on-topic-ness — more
than categories do. The model **can** learn which groups are more credible — but **only as a function of
observable group features**, *never* a per-group learned scalar (that is the growing-table violation again).

Implement as a learned **gate** `α(group) ∈ (0,1]` scaling the membership modifier:

```
node_emb += α(group) · T(e5_group)
```

with `α` computed from:
- the group's **e5** (topic), and/or
- the **dispersion of its members' e5** — a *tight* cluster of member embeddings = on-topic/credible →
  larger `α`; a *scattered* group = drifty → smaller `α`. Coherence is computable straight from the members,
  **no table**.

Credible groups nudge harder, drifty ones barely. This turns the *global* "small-magnitude modifier"
regulariser into a **per-group, e5/coherence-derived** one — still cold-start-safe (works on unseen groups),
still no growing table. Start `α` at a neutral constant (≡ v1 behaviour) and learn deviations.

Note the closed-vs-open split holds here too: **account**-level credibility (`s243a` vs `s243a_groups`) is
fine as the small 2-value token; **per-group** credibility is the open case that *must* be a function.

### Alternative: the transform's input embedding (documented, deferred)

`T`'s input is **e5 for now**, but a *different* frozen encoder (e.g. **miniLM**) is a legitimate alternative
input — `T` doesn't care what vector feeds it, so swapping/adding one later is a **localised** change (change
only what goes into `T`). We **defer** it deliberately: adopting it now would fragment the single uniform e5
space every node shares into two encoder spaces to manage — implicit complexity for no present gain. Recorded
as a real option, not ruled out; **stick to e5** until there's a measured reason group needs its own encoder.

## Two buckets, not one: **provenance** (maskable token) vs **structure** (on the node)

The first cut lumped everything "factored" together; sharpen it into two buckets that behave differently:

- **Provenance** — *where a label came from*: `corpus`, `judge`, `account`. Lives on its **own maskable
  token** (mask it ⇒ a provenance-agnostic μ; unmask ⇒ condition on the source). You want to be able to
  *marginalize it out*, so it is deliberately separable from the node.
- **Structure** — *what a node is / what it belongs to*: `node-type`, `group`. **Added to the node's own
  embedding**, per-node, **always on** (it is an intrinsic property of the node, not something you'd average
  away). `mu_attention.py:317` already does this for node-type: `emb += nodetype_emb[type] * mask`.

`account` is provenance (maskable). **`group` is structure** — same bucket as node-type, *not* provenance.
That is the crux of your point.

## Alternatives: where does group conditioning attach?

Group membership is a *node* property, so the design question is **where the group signal enters** and
**where its modifier vector comes from**. Cardinality (many groups, few types) constrains the *source*, not
the *attachment point*.

| # | Attachment | Source of modifier | Verdict |
|---|---|---|---|
| 1 | **Per-group learned axis** (treat group like type/op) | a per-group **table row** | ✗ open-ended → table grows with data; most groups seen too rarely to train a row. This is "adding it to types," and it breaks on **many-groups ≫ few-types**. |
| 2 | **Separate group token** in the sequence (maskable, provenance-style) | `T(e5_group)` | ✗ one per-example token can't say *different nodes belong to different groups* — group is **per-node**, not per-example; + costs a token. |
| 3 | **Added to the node embedding, per-node** (parallel to `nodetype_emb`) | `T(e5_group_of_node)` | ✓ **the lean.** Per-node (each node carries its own group's signal); identical mechanism, warm-start, and gating story as node-type; no extra token. |
| 4 | **Concat + project** the node and group e5 | `W·[e5_node ; T(e5_group)]` | ~ more expressive (interaction terms) but changes the input projection and is hard to init near-identity; hold as an upgrade if (3) underfits. |

**Recommendation: option 3.** `node_emb += T(e5_group_of_node)`, exactly mirroring
`node_emb += nodetype_emb[type]`, with `T` a shared Linear whose **output is zero-initialised** so it starts
as a **no-op** (warm-start-safe, the *same* reason `nodetype_emb` is zero-init — and **not** identity-init,
which would add the raw group e5). Minimal change; respects both constraints — the **per-node** nature of
membership (so it goes on the node, not a global token) and the **open cardinality** (so the modifier is an
e5-*transform* that generalizes, not a per-group row). The transform is *required* here: raw e5 is the wrong
space (see "The transform is mandatory" above).

### The unifying statement

Node-type and group are the **same conditioning** — "what role does this node play" — both **summed into the
node embedding**. They differ only in the *source* of the summed vector, and cardinality picks the source:

> **Closed, small role set → a learned table (lookup).  Open, large role set → a shared transform of e5.**
> Both are added to the node embedding.

Node-type is the degenerate *closed* case; group is the *open* case of one general "node-role" modifier. "Many
more groups than types" is **precisely why** group's modifier must be the e5-transform branch — and why it
still attaches at the very same point type does, which is what you noticed.

### What is "the group's e5"?
The group's **title** e5 (cheap, uniform) vs the **centroid of its member nodes' e5** (a group *is* its
contents). Lean: member-centroid for collections; revisit if titles prove more discriminative.

## Discipline (same as node-type)
Init near identity, **gate** (`--group-transform`), and **A/B** it — only once group/multi-account data is
flowing. Adding an expressive transform before the data has group diversity is node-type's collinearity
problem with more parameters to overfit.
