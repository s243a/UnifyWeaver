# Skill: Mindmap Rename

Rename mindmap files and automatically update all cloudmapref references.

## When to Use

- User wants to rename mindmaps from `id12345.smmx` to meaningful names
- User asks to "rename", "batch rename", or "title" mindmaps
- User needs to move/rename without breaking cross-links
- User wants `Title_ID.smmx` format for all files

## Quick Start

### Single File Rename

```bash
# Auto-generate name from root topic
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps/id10380971.smmx \
  --titled

# Explicit new name
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap output/mindmaps/id10380971.smmx \
  --new-name "Quantum_Physics_10380971.smmx"
```

### Batch Rename

```bash
# Rename all mindmaps to Title_ID.smmx format
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps/ \
  --titled
```

## How It Works

When you rename a mindmap:

1. **Extract new name** - From `--new-name` or root topic text (if `--titled`)
2. **Find references** - Scan all mindmaps for `cloudmapref` pointing to old file
3. **Update references** - Recompute relative paths to new filename
4. **Rename file** - Actually rename the file
5. **Update index** - If `--index` provided, update the index

```
Before: Physics/id10380971.smmx
        Other/related.smmx has cloudmapref="../Physics/id10380971.smmx"

After:  Physics/Quantum_Mechanics_10380971.smmx
        Other/related.smmx has cloudmapref="../Physics/Quantum_Mechanics_10380971.smmx"
```

## Commands

### Titled Rename (Recommended)

Generate filename from root topic text:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap "MINDMAP_PATH" \
  --titled
```

**Example:**
- Root topic: "Quantum Mechanics"
- Tree ID: 10380971
- New filename: `Quantum_Mechanics_id10380971.smmx`

### Explicit Rename

Specify exact new filename:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap "MINDMAP_PATH" \
  --new-name "New_Name_id12345.smmx"
```

### Batch Rename

Rename all mindmaps in a directory:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --batch "MINDMAP_DIR" \
  --titled
```

### With Index Update

Update index file after renaming:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap "MINDMAP_PATH" \
  --titled \
  --index index.json
```

### Preview Changes (Dry Run)

See what would happen without making changes:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap "MINDMAP_PATH" \
  --titled \
  --dry-run --verbose
```

## Options

| Option | Description |
|--------|-------------|
| `--mindmap, -m` | Single file to rename |
| `--new-name, -n` | Explicit new filename |
| `--batch, -b` | Directory for batch rename |
| `--titled, -t` | Auto-generate `Title_ID.smmx` from root topic |
| `--id-prefix` | Include "id" prefix (default): `Title_id12345.smmx` |
| `--no-id-prefix` | Omit "id" prefix: `Title_12345.smmx` |
| `--max-title` | Maximum title length (default: 50) |
| `--index, -i` | Index file to update (JSON/TSV/SQLite) |
| `--search-dir, -s` | Directory to search for references |
| `--dry-run, -d` | Preview without changes |
| `--verbose, -v` | Detailed output |

## Filename Formats

| Format | Example | Options |
|--------|---------|---------|
| ID only | `id10380971.smmx` | (original) |
| Title + ID | `Quantum_Mechanics_id10380971.smmx` | `--titled` (default) |
| Title + ID (no prefix) | `Quantum_Mechanics_10380971.smmx` | `--titled --no-id-prefix` |

The tree ID is always preserved to ensure uniqueness.

## Title Sanitization

Root topic text is sanitized for filesystem compatibility:

- Spaces → underscores
- Special characters removed: `/ \ : * ? " < > |`
- Leading/trailing whitespace trimmed
- Truncated to `--max-title` length (default: 50)

**Examples:**
- "Quantum Mechanics & Physics" → `Quantum_Mechanics_&_Physics`
- "The JFK Assassination" → `The_JFK_Assassination`
- "Posts from 2024/01/15" → `Posts_from_2024_01_15`

## Reference Update Process

For each mindmap being renamed:

1. **Build search scope** - Use `--search-dir` or parent directory
2. **Scan for references** - Find all `cloudmapref` attributes containing old filename
3. **Compute new paths** - Calculate relative path from referencing file to new location
4. **Update XML** - Modify cloudmapref attributes in referencing files
5. **Save changes** - Write updated XML back to .smmx files

### Path Computation Example

```
Old: output/Physics/id10380971.smmx
New: output/Physics/Quantum_Mechanics_id10380971.smmx

Reference from: output/Math/algebra.smmx
Old cloudmapref: ../Physics/id10380971.smmx
New cloudmapref: ../Physics/Quantum_Mechanics_id10380971.smmx
```

## Batch Mode Strategy

When batch renaming:

1. **Build reference map first** - Scan all files to find all cross-references
2. **Compute all renames** - Determine all new filenames
3. **Update all references** - Fix cloudmapref attributes
4. **Rename files** - Actually rename (after references are updated)

This prevents broken links during the rename process.

## Common Workflows

### After Initial Import

Convert ID-only filenames to titled format:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps/ \
  --titled \
  --index index.json
```

### Before Sharing Collection

Make filenames human-readable:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/to_share/ \
  --titled \
  --max-title 30 \
  --no-id-prefix
```

### Preview Large Batch

Check what will happen before committing:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --batch output/mindmaps/ \
  --titled \
  --dry-run --verbose 2>&1 | head -100
```

## Troubleshooting

### "Reference not found" warnings

Some cloudmapref attributes may point to missing files:

```bash
# Check for orphan references
python3 scripts/mindmap/verify_links.py output/mindmaps/
```

### Filename collisions

If two mindmaps have the same root topic:

```bash
# The ID suffix ensures uniqueness
# "Physics_id12345.smmx" vs "Physics_id67890.smmx"
```

### Long filenames

Use `--max-title` to limit title length:

```bash
python3 scripts/mindmap/rename_mindmap.py \
  --mindmap file.smmx \
  --titled \
  --max-title 30
```

## Related

**Parent Skills:**
- `skill_mindmap_indexing.md` - Indexing sub-master
- `skill_mindmap_tools.md` - Master mindmap skill

**Sibling Skills:**
- `skill_mindmap_index.md` - Index system (used for lookups)

**Related Skills:**
- `skill_mindmap_cross_links.md` - Cross-references (affected by rename)

**Code:**
- `scripts/mindmap/rename_mindmap.py` - Main rename tool
- `scripts/mindmap/index_store.py` - Index storage
