# Skill: understanding a SimpleMind mindmap (`.smmx`)

A skill for reading the user's SimpleMind mindmaps and turning them into **typed membership relations**
(for μ-grading, corpus=mindmap, judge=human). It works **two ways**: (A) read the raw map yourself using
the rules below, or (B) run `parse_smmx.py` as an aid and read its typed edge list. Use the tool when you
have file access; fall back to the rules when you only have the XML pasted in.

---

## What a `.smmx` is

A `.smmx` file is a **zip** containing `document/mindmap.xml`. The XML is a tree of `<topic>` elements:

```xml
<topic id="7" parent="0" guid="…" text="Artificial\Nlife">
  <link urllink="http://www.pearltrees.com/s243a/artificial-life/id60477639"/>
</topic>
```

- `text` — the node label; **`\N` is a line break** (read `"Artificial\Nlife"` as "Artificial life").
- `parent` — the id of the parent topic (`-1` for the root / central theme).
- `<link>` children of a topic:
  - `urllink="…pearltrees.com/s243a/<slug>/id<n>"` — the node's **Pearltrees identity** (the slug is the
    stable node id).
  - `urllink="…en.wikipedia.org/wiki/<Title>"` — a **Wikipedia anchor** for the node (the join key to
    enwiki).
  - `cloudmapref="../Other Map.smmx" element="…"` — a **cross-map link** (relative path) to **another
    mindmap, at its ROOT** (`element` usually absent ⇒ the whole map). **The path direction sets the
    relation** (when no container tags it): a `../` path UP to a **parent folder** ⇒ the target is a
    broader **parent / `super_category`** (the upward parent-tree chain); a path DOWN into a **subfolder**
    ⇒ the target is a narrower **`subcategory`** (a dedicated sub-map expanding this node — e.g. a
    `Chaos theory` node linking down to its own detailed `Chaos theory.smmx`).
- `<relation source="123" target="141"/>` (at the end of the map) — an explicit **associative cross-link**
  between two topics.

---

## The structural nodes — they are NOT topics; they RETYPE their descendants

Some nodes are **containers / scaffolding**, not real concepts. Skip them as nodes, but they change the
**relation** of everything beneath them to the concept the container hangs off. This is the core rule:

| container label | meaning for its descendants | relation |
|---|---|---|
| `See Also`, `Via Link`, `Related` | **see-also**: weakly/associatively related (not membership) | `see_also` |
| `Super Categories`, `Super Category`, `Navigate Up` | **broader / parent** category of the attached node | `super_category` |
| blank (`text=""`) / section headers | usually just visual grouping — pass through, keep the relation from above. **But** a blank node can also be a **list/sequence connector**: in list-like structures (e.g. chapters) an empty node joins items, and it may carry a `cloudmapref` (often `="."` intra-map, with an `element` GUID) that **refers to an adjacent node** — treat that as an associative link (`assoc`) to the referenced node | — / `assoc` |
| a node labelled **`wiki`** / `Wikipedia` / `enwiki` (with a Wikipedia urllink), or any node with a direct `en.wikipedia.org` urllink | a **`bridge`**: the node ↔ the SAME concept in enwiki (a `category` if `Category:…`, else a `page`) — same concept, **different node-type**, possibly different name | `bridge` |

Everything else is a **real node**, and a plain parent→child link (after skipping any containers) is a
**`subtopic`** = membership / narrower-than.

### Worked example (Cybernetics map)
```
Cybernetics
 ├─ Engineering                         → Cybernetics --subtopic--> Engineering   (and cloudmapref to Engineering.smmx)
 ├─ Magnetogenetics
 │   └─ [See Also] → Genetics,          → Magnetogenetics --see_also--> Genetics, Neuroprosthetics,
 │                   Neuroprosthetics,                       Brain–computer interfacing
 │                   Brain–computer…
 ├─ [See Also] → Robotics, Intelligence  → Cybernetics --see_also--> Robotics, Intelligence & mind, …
 └─ [Super Categories] → Applied         → Cybernetics --super_category--> Applied mathematics,
                          mathematics,                       Cognitive science, Systems science
                          Cognitive science…
```
So a child *directly* under `Cybernetics` is a subtopic; a child under a `See Also` that hangs off
`Cybernetics` is a *see-also* of Cybernetics (grandparent), not of the `See Also` node.

---

## Book / course maps — navigation skeleton vs concepts

Some maps index a **book** or **course** and are mostly *navigation*, not concept-membership:

- **Lists via empty containers + sequential names.** An empty node groups list items, whose names carry
  the order: `Modules → [empty] → Week #1, Week #2 …`; `Chapter 2 → 2.1, 2.2 …`. The empty node is plain
  grouping (pass through); the items are the list. (`CAD-111`, `Books (Tensors)`.)
- **Reading order via `<relation>`.** Page/section nodes (`pg18`, `2.3`) are **navigation** (tagged
  `node_type=navigation`); `<relation>` chains between them are `sequence` (page-to-page reading order),
  **not** membership.
- **The concepts hang off `See Also`.** The valuable nodes are the `See Also` targets — each a real concept
  with a Pearltrees slug and often a `wiki` child ⇒ a **`bridge`** (these book/course maps are a richer
  source of bridges than the top-level maps).

**When grading membership, use the `see_also`/`bridge`/concept edges; treat the chapter/week/page skeleton
(`subtopic` between `navigation` nodes, and `sequence`) as book structure, not topic membership.**

---

## How to read a map BY HAND (no tool)

For each real (non-container, non-`wiki`) node:
1. Walk **up** the `parent` chain, skipping container nodes, to the first real node = its **effective
   parent**.
2. The **relation** is set by the nearest container you passed through: `See Also`/`Via Link`/`Related` ⇒
   `see_also`; `Super Categories` ⇒ `super_category`; none (or only blanks) ⇒ `subtopic`.
3. Record the node's **Pearltrees slug** (identity) and any **Wikipedia anchor** (its own wiki urllink, or
   a `wiki`-labelled child).
4. `cloudmapref` on a node (or its blank link-holder child) ⇒ a cross-map edge to that other map's root.
   Set the relation by the **path direction**: `../` (parent folder) ⇒ `super_category` (broader);
   subfolder (down) ⇒ `subcategory` (narrower). **The super-category/parent holder nodes are usually
   UNNAMED** — to name the target you must **open the linked `.smmx` and read its ROOT node's title +
   Pearltrees slug** (the tool does this automatically; `--no-resolve` falls back to the filename).

---

## How to use the tool as an aid

When you have file access, let the parser do steps 1–4:

```bash
python3 parse_smmx.py path/to/Map.smmx --out-prefix map
#   map_nodes.tsv : node_key  title  pearltrees_slug  pearltrees_id  enwiki_alias
#   map_edges.tsv : src_key   dst_key  relation   (subtopic|see_also|super_category|cloudmapref|assoc)
```
Read `map_edges.tsv` instead of eyeballing the XML. (Run with no `--out-prefix` for a summary + the first
edges on stdout.) The tool applies exactly the rules above; use it to avoid mistakes on large maps, and use
the by-hand rules to sanity-check or when only XML is pasted to you.

### Product-Kalman campaign primary view

For the cross-corpus campaign, use `sample_product_kalman_simplemind_campaign.py`, not the parser's full typed-edge
union. Its primary graph is only the plain, content-rooted within-map parent hierarchy, with blank containers
bypassed. It excludes `See Also`/`Via Link`/`Related`, super-category, navigation, wiki-anchor, explicit-relation,
and cross-map edges from hop calculations. Structural labels are matched case-insensitively because real maps use
mixed-case variants. Raw titles, topic IDs, Pearltrees slugs, enwiki aliases, and title aliases remain in the output for
identity closure and audited-title sensitivity; title typos are not silently corrected.

---

## For vision-capable models: render the map to an image

If you are a **vision-capable** model (e.g. Gemini, recent Opus/ChatGPT — *not* a text-only Haiku pass),
a **rendered picture** of the mind map often conveys structure faster than the XML: the spatial layout,
colour palette, branch grouping, and which children hang off a `See Also` vs a `Super Categories` node are
all visible at a glance, and SimpleMind's own visual conventions become legible.

The repo's mind-map renderers can produce that image from the parsed graph (or `.smmx`):
`src/unifyweaver/mindmap/render/` — **`graphviz_renderer.pl`** (→ Graphviz DOT → PNG/SVG, the simplest
static image), **`d3_renderer.pl`** (interactive HTML/SVG), `mm_renderer.pl`, and `smmx_renderer.pl`
(round-trips *back* to `.smmx`). Render with `graphviz_renderer.pl`, then read the resulting image.

Use the image as the **primary** view and the typed edges (`parse_smmx.py`) as the **ground truth** to
resolve anything ambiguous — colours/positions hint at relation type, but the container-retyping rules
above are authoritative. A text-only grader should ignore this section and work from the XML / TSVs.

---

## Turning relations into μ (when grading)

These relations map onto membership strength for `μ(node | root)` grading (human judge):

| relation | μ reading |
|---|---|
| `subtopic` | **high** membership — the child is part of / narrower than the parent topic (in-map hierarchy) |
| `subcategory` | **high** membership — the target map is narrower / a child (downward cloudmapref); μ(target\|node) high |
| `super_category` | the *parent direction* — the target is **broader** (in-map Super Categories, or upward cloudmapref); μ(node\|target) high, reverse low |
| `see_also` | **associative**, not membership — moderate-to-low, symmetric (grade like a boundary pair) |
| `bridge` | **near-identity** across corpora — the mindmap node and its enwiki `category`/`page` are the *same concept* (μ ≈ high). The endpoints differ in **node-type** (mindmap_node ⟷ category/page) and possibly name; this is the cross-corpus join (scarce in maps, more in Pearltrees) |
| `assoc` (explicit relation) | a deliberate cross-link — moderate associative |

The Pearltrees slug is the node identity; the `enwiki_alias`, where present, is the bridge into the enwiki
category/page graph (so a mindmap node can be cross-checked against, or fused with, its Wikipedia data).
