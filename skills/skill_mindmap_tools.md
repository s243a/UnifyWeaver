# Skill: Mindmap Tools

Master skill for managing SimpleMind mindmap files (.smmx) - organization, indexing, linking, and visualization.

## When to Use

- User asks "how do I work with mindmaps?"
- User wants to organize, link, or manage mindmap files
- User asks about SimpleMind or .smmx files
- User needs to find, rename, or cross-reference mindmaps

## Skill Hierarchy

```
skill_mindmap_tools.md (this file)
│
├── Organization & Structure
│   └── skill_mindmap_organization.md (sub-master)
│       ├── skill_mst_folder_grouping.md - MST-based folder organization
│       ├── skill_folder_suggestion.md - Suggest folders for items
│       └── skill_hierarchy_objective.md - Evaluate hierarchy (J = D/(1+H))
│
├── Indexing & Lookup
│   └── skill_mindmap_indexing.md (sub-master)
│       ├── skill_mindmap_index.md - Forward/reverse indexes, storage backends
│       └── skill_mindmap_rename.md - Rename files, update all references
│
├── Linking & References
│   └── skill_mindmap_references.md (sub-master)
│       ├── skill_mindmap_linking.md - Link to Pearltrees URLs
│       └── skill_mindmap_cross_links.md - Add cloudmapref for local navigation
│
└── Visualization
    └── skill_density_explorer.md - Explore embeddings, trees, density
```

## Sub-Skills Overview

### Organization & Structure

Semantically organize mindmaps into folder hierarchies using embeddings and clustering.

**Sub-master:** `skill_mindmap_organization.md`

### Indexing & Lookup

Track mindmap locations and maintain consistency when files move or are renamed.

**Sub-master:** `skill_mindmap_indexing.md`

### Linking & References

Create navigable connections between mindmaps and external URLs.

**Sub-master:** `skill_mindmap_references.md`

### Visualization

Explore embedding spaces and tree structures visually.

**Skill:** `skill_density_explorer.md`

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

**Sub-Master Skills:**
- `skill_mindmap_organization.md` - Organization & structure
- `skill_mindmap_indexing.md` - Indexing & lookup
- `skill_mindmap_references.md` - Linking & references

**Individual Skills:**
- `skill_density_explorer.md` - Visualization

**Documentation:**
- `scripts/mindmap/README.md` - Tool documentation

**Code:**
- `scripts/mindmap/` - All mindmap tools
- `scripts/generate_mindmap.py` - Mindmap generator
