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
| blank (`text=""`) / section headers | just visual grouping — pass through, keep the relation from above | — |
| a node labelled **`wiki`** / `Wikipedia` (with a Wikipedia urllink) | NOT a node — it gives the **enwiki anchor** of its **parent** | (anchor) |

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

---

## Turning relations into μ (when grading)

These relations map onto membership strength for `μ(node | root)` grading (human judge):

| relation | μ reading |
|---|---|
| `subtopic` | **high** membership — the child is part of / narrower than the parent topic (in-map hierarchy) |
| `subcategory` | **high** membership — the target map is narrower / a child (downward cloudmapref); μ(target\|node) high |
| `super_category` | the *parent direction* — the target is **broader** (in-map Super Categories, or upward cloudmapref); μ(node\|target) high, reverse low |
| `see_also` | **associative**, not membership — moderate-to-low, symmetric (grade like a boundary pair) |
| `assoc` (explicit relation) | a deliberate cross-link — moderate associative |

The Pearltrees slug is the node identity; the `enwiki_alias`, where present, is the bridge into the enwiki
category/page graph (so a mindmap node can be cross-checked against, or fused with, its Wikipedia data).
