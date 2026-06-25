# Skill: understanding harvested Pearltrees data

The Pearltrees analogue of `SKILL_understand_smmx.md`: how to read **harvested** Pearltrees data and turn
it into **typed membership relations** (corpus=pearltrees, judge=human). Works two ways: (A) apply the
rules below to the harvested rows yourself, or (B) run `parse_pearltrees.py` and read its typed edge list.

> **Harvesting is private and lives in `.local`.** This skill is *ref-based* — it interprets data the
> harvester already pulled. To (re)harvest, **if it exists, read**
> `.local/tools/browser-automation/SKILL_harvest_pearltrees.md` (the cookie/`getTreeAndPearls` API
> workflow + `.local/tools/browser-automation/scripts/fetch_pearltrees_tree.py`). On a clone without
> `.local`, harvesting is unavailable — work from whatever harvested DB/JSON is present, or skip.

---

## What the harvested data is

A Pearltrees account is a forest of **trees** (collections). The harvest stores them as a SQLite DB
(`.local/data/pearltrees_api/pearltrees_api.db`: `trees`, `pearls`) and per-tree JSON. Each tree contains
**pearls**, distinguished by `contentType`:

| contentType | pearl | what it is |
|---|---|---|
| 2 | **Collection** | a child tree (a sub-collection) — has `content_tree_id`/`content_tree_title` |
| 1 | **PagePearl** | a URL bookmark — has `url` (often a Wikipedia page → a bridge; or an external resource) |
| 5 | **Shortcut** | an alias / cross-reference to another tree elsewhere in the forest |
| 7 | **Section** | a section **header** — retypes the pearls that follow it (next section) |
| 4 | **Root** | the tree's own root pearl |

### The PRINCIPAL relationship + ordering

- **Principal parent-child.** The Pearltrees **tree containment** (a Collection pearl = a child tree) is
  the **one** canonical parent-child the user designated as the *principle* relationship — captured as
  `subtopic` (parent collection → child). The same fact is what an **RDF export** encodes (RDF gives one
  parent per item). The section-header relations below (`subcategory`/`super_category`/`see_also`/…) are
  **secondary annotations** layered on top of it. ("Principle" = the one chosen, not necessarily the best.)
- **Ordering / nested set.** Pearls carry a position: newer data a single `pos`; older data `leftpos` +
  `rightpos` (a nested-set model — equivalent to `pos` when `leftpos == rightpos`, i.e. a leaf). The DB's
  `left_index`/`right_index` are these; the parser reads pearls in `left_index` order so each section
  header scopes the pearls after it.

---

## Section headers retype what follows (like SimpleMind containers)

`Section` pearls (contentType 7) are **headers**: read pearls in `left_index` order, and a header sets the
relation of every pearl after it **until the next header**. The recognised headers:

| section header | relation imposed on the pearls under it |
|---|---|
| `Subcategories` | `subcategory` (narrower category) |
| `Subtopics` / `More Subtopics` | `element_of` (element relations) |
| `Super Categories` | `super_category` (parent — usually a super-category, but could be a page's parent) |
| `Navigate Up` | `super_category` too, but it is **redundant convenience backup** — it just re-points to the **principal parent** (the tree containment) so the user can navigate up from the bottom of the page (the Pearltrees app lacks an up button). Safe to dedup/ignore; it adds no information beyond the principal parent. |
| `See Also` | `see_also` (associative) |
| `Wiki / Encyclopedia type References` | the links here **bridge the whole TREE to enwiki** (collection-level bridge) |
| topical groupers (`Algebra`, `Calculus`, …) / junk (`Meta`, `To sort`, `Friends Pages`) | none — fall back to the contentType default |

## Relation semantics (tree = a collection of pearls)

| pearl | relation | μ reading |
|---|---|---|
| Collection (child tree) | `subtopic` | **high** membership — the child collection is narrower than the parent |
| PagePearl | `element_of` | the page is a **member** of the collection (μ by how central it is, like enwiki page-membership) |
| PagePearl whose `url` is `en.wikipedia.org/...` | **`bridge`** (in addition) | **near-identity** to the enwiki `category`/`page` — same concept, different node-type, the cross-corpus join. **Pearltrees is the bridge-RICH corpus** (e.g. ~48 here vs ~1 in a SimpleMind map) |
| Shortcut (alias) | `assoc` | a deliberate cross-reference — moderate associative |
| Section / Root | — | grouping / root — skip |

**Node identity = the title-derived slug** (e.g. `Network theory` → `network-theory`) — the *same* slug a
SimpleMind node carries in its Pearltrees urllink, so the mindmap and Pearltrees corpora **join on slug**,
and both join to enwiki through `bridge` edges. `node_type ∈ pearltrees_collection | page | category`.

---

## Using the tool

```bash
python3 parse_pearltrees.py --out-prefix pt          # reads the .local harvested DB if present
#   pt_nodes.tsv : node_key  node_type  title  pearltrees_id  enwiki_alias
#   pt_edges.tsv : src_key   dst_key    relation   (subtopic|element_of|bridge|assoc)
```
Pass `--db <path>` for a DB elsewhere. The tool applies exactly the rules above; use the rules by hand only
when reading raw harvested rows directly.

---

## Why this corpus matters

- It's the **bridge spine**: most mindmap nodes link to a Pearltrees collection, and the Pearltrees
  PagePearls carry the enwiki URLs — so Pearltrees is where the `mindmap_node`/`page` ⟷ `category`/`page`
  **bridges** live (the within-operator type diversity that makes the node-type token informative —
  `REPORT_nodetype.md`).
- It joins on **slug** to SimpleMind (`SKILL_understand_smmx.md`) and on **enwiki_alias** to the category /
  page graph — three corpora, one identity fabric. **Many SimpleMind nodes link to a Pearltrees
  collection** (the pearltrees urllink on a topic), so there are **lots of SimpleMind↔Pearltrees bridges**
  on the shared slug — a `mindmap_node` and a `pearltrees_collection` for the *same concept* (different
  node-type, same key). The fusion step (`gen_mindmap_pairs.py`) should keep them as a typed `bridge` pair
  rather than silently collapsing the key, since that type pairing is exactly the within-operator diversity
  the node-type token needs.
