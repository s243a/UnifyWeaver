# Skill: Mindmap References

Sub-master skill for creating navigable connections between mindmaps and external URLs.

## When to Use

- User asks "how do I link mindmaps together?"
- User wants to connect mindmaps to Pearltrees
- User asks about cloudmapref or urllink attributes
- User needs cross-references between mindmaps
- User wants to enrich mindmaps with external metadata

## Overview

SimpleMind mindmaps support two types of links:

| Link Type | Attribute | Purpose |
|-----------|-----------|---------|
| **External URL** | `urllink` | Link to Pearltrees or other websites |
| **Local Reference** | `cloudmapref` | Link to another local mindmap file |

```xml
<link urllink="https://www.pearltrees.com/s243a/physics/id12345"
      cloudmapref="../Physics/id12345.smmx"/>
```

## Individual Skills

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| `skill_mindmap_linking.md` | Link to Pearltrees URLs | Enriching with external data |
| `skill_mindmap_cross_links.md` | Add cloudmapref for local navigation | Creating internal links |

## Link Types Explained

### External URLs (urllink)

Links to web pages, typically Pearltrees:

```xml
<topic text="Physics">
  <link urllink="https://www.pearltrees.com/s243a/physics/id10390825"/>
</topic>
```

**Use for:**
- Opening original Pearltrees page
- External reference to source material
- Web-based navigation

### Local References (cloudmapref)

Relative paths to other mindmap files:

```xml
<topic text="Physics">
  <link cloudmapref="../Science/Physics_id10390825.smmx"/>
</topic>
```

**Use for:**
- Offline navigation between mindmaps
- Portable collections (work without internet)
- Fast local browsing

## The Linking Pipeline

### Step 1: Build Index

Before adding links, you need an index of mindmap locations:

```bash
python3 scripts/mindmap/build_index.py output/mindmaps/ -o index.json
```

### Step 2: Add Cross-Links

Add cloudmapref attributes to connect mindmaps:

```bash
python3 scripts/mindmap/add_relative_links.py output/mindmaps/*.smmx \
  --index index.json
```

This scans each mindmap for Pearltrees URLs and adds corresponding cloudmapref for local navigation.

See `skill_mindmap_cross_links.md` for details.

### Step 3: Enrich with Pearltrees Data (Optional)

Add URLs and metadata from Pearltrees:

```bash
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap output/my_mindmap.smmx \
  --trees reports/pearltrees_targets.jsonl \
  --projection-model models/federated_model.pkl \
  --output output/enriched.smmx
```

This uses:
- **URL matching** - Match node URLs to Pearltrees
- **Title matching** - Match by normalized title
- **Semantic matching** - Use embeddings when no exact match

See `skill_mindmap_linking.md` for details.

## URL vs cloudmapref Strategies

When generating mindmaps, you can control link placement:

### Both Links on Same Node (Default)

```bash
python3 scripts/generate_mindmap.py ... --index index.json
# urllink on labeled node, cloudmapref on square child node
```

### URL on Main, cloudmapref on Child

```bash
python3 scripts/generate_mindmap.py ... --url-nodes url
```

### cloudmapref on Main, URL on Child

```bash
python3 scripts/generate_mindmap.py ... --url-nodes map
```

## Common Workflows

### Creating Linked Collection from Scratch

```bash
# 1. Generate mindmaps recursively
python3 scripts/generate_mindmap.py \
  --cluster-url "https://www.pearltrees.com/s243a" \
  --recursive --output-dir output/mindmaps/

# 2. Build index
python3 scripts/mindmap/build_index.py output/mindmaps/ -o index.json

# 3. Add cross-links
python3 scripts/mindmap/add_relative_links.py output/mindmaps/**/*.smmx \
  --index index.json
```

### Enriching Exported Mindmap

```bash
# Take a mindmap exported from SimpleMind and enrich it
python3 scripts/mindmap/link_pearltrees.py \
  --mindmap exports/my_ideas.smmx \
  --trees reports/pearltrees_targets.jsonl \
  --url-db data/children_index.db \
  --embeddings datasets/embeddings.npz \
  --projection-model models/model.pkl \
  --output output/my_ideas_enriched.smmx
```

### Update Links After Reorganization

```bash
# After moving files, rebuild index and update links
python3 scripts/mindmap/build_index.py output/organized/ -o index.json
python3 scripts/mindmap/add_relative_links.py output/organized/**/*.smmx \
  --index index.json --update
```

## Link Attributes Reference

| Attribute | Type | Description |
|-----------|------|-------------|
| `urllink` | URL | External web link |
| `cloudmapref` | Path | Relative path to local mindmap |
| `cloudtreeref` | ID | SimpleMind cloud reference (not used) |

## Related

**Parent Skill:**
- `skill_mindmap_tools.md` - Master mindmap skill

**Individual Skills:**
- `skill_mindmap_linking.md` - Link to Pearltrees URLs
- `skill_mindmap_cross_links.md` - Add cloudmapref for local navigation

**Sibling Sub-Masters:**
- `skill_mindmap_indexing.md` - Index system (needed for linking)
- `skill_mindmap_organization.md` - Folder organization

**Code:**
- `scripts/mindmap/add_relative_links.py` - Add cloudmapref links
- `scripts/mindmap/link_pearltrees.py` - Enrich with Pearltrees data
- `scripts/generate_mindmap.py` - Generate mindmaps with links
