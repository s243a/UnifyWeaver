# Skill: Mindmap Tools (Sub-Master)

Managing SimpleMind mindmap files (.smmx) - organization, indexing, linking, and cross-references.

## When to Use

- User asks "how do I work with mindmaps?"
- User wants to organize, link, or manage mindmap files
- User asks about SimpleMind or .smmx files
- User needs to find, rename, or cross-reference mindmaps

## Skill Hierarchy

```
skill_mindmap_bookmark_tools.md (parent)
└── skill_mindmap_tools.md (this file)
    │
    ├── skill_mindmap_organization.md (sub-master)
    │   └── (uses hierarchy_objective from skill_ml_tools.md)
    │
    ├── skill_mindmap_indexing.md (sub-master)
    │   ├── skill_mindmap_index.md - Forward/reverse indexes
    │   └── skill_mindmap_rename.md - Rename with reference updates
    │
    └── skill_mindmap_references.md (sub-master)
        ├── skill_mindmap_linking.md - Enrich with Pearltrees links
        └── skill_mindmap_cross_links.md - Local cloudmapref navigation
```

## Sub-Skills Overview

### Organization & Structure

Semantically organize mindmaps using embeddings and clustering.

**Sub-master:** `skill_mindmap_organization.md`

Note: MST folder grouping and folder suggestion are now under `skill_bookmark_tools.md` since they apply to both mindmaps and bookmarks.

### Indexing & Lookup

Track mindmap locations and maintain consistency when files move or are renamed.

**Sub-master:** `skill_mindmap_indexing.md`

### Linking & References

Create navigable connections between mindmaps and external URLs (Pearltrees).

**Sub-master:** `skill_mindmap_references.md`

## SimpleMind File Format

Mindmaps use the `.smmx` format (SimpleMind XML in a zip):

```
mymap.smmx (zip archive)
└── document.xml
    └── <simplemind-mindmaps>
            └── <mindmap>
                └── <topics>
                    └── <topic text="Root">
                        └── <link urllink="https://..." cloudmapref="../other.smmx"/>
```

**Key attributes:**
- `urllink` - External URL (e.g., Pearltrees page)
- `cloudmapref` - Relative path to another local mindmap

## Typical Workflow

```bash
# 1. Generate mindmaps from Pearltrees data
python3 scripts/generate_mindmap.py --data data.jsonl --recursive --output-dir output/

# 2. Build index (see skill_mindmap_indexing.md)
python3 scripts/mindmap/build_index.py output/ -o index.json

# 3. Add cross-links (see skill_mindmap_references.md)
python3 scripts/mindmap/add_relative_links.py output/*.smmx --index index.json

# 4. Organize into folders (see skill_mindmap_organization.md)
python3 scripts/mindmap/mst_folder_grouping.py --input output/ --output organized/

# 5. Visualize and explore (see skill_density_explorer.md)
python tools/density_explorer/flask_api.py
```

## Related

**Parent Skill:**
- `skill_mindmap_bookmark_tools.md` - Master for mindmaps and bookmarks

**Child Skills:**
- `skill_mindmap_organization.md` - Organization & structure
- `skill_mindmap_indexing.md` - Indexing & lookup
- `skill_mindmap_references.md` - Linking & references

**Sibling Skills:**
- `skill_bookmark_tools.md` - Bookmark filing and folder organization
- `skill_density_explorer.md` - Visualization (shared)

**Documentation:**
- `scripts/mindmap/README.md` - Tool documentation

**Code:**
- `scripts/mindmap/` - All mindmap tools
- `scripts/generate_mindmap.py` - Mindmap generator
