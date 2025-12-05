# Rebuild Pearltrees Hierarchy Tool

## Purpose

This tool rebuilds the parent-child relationships in a Pearltrees LiteDB database by extracting parent tree IDs from pearl About URLs and populating the `Children` and `ParentTree` fields.

## Why This Is Needed

**Problem:** The current RDF ingestion only extracts text content from XML elements, not RDF resource attributes. In Pearltrees RDF exports, parent-child relationships are encoded as:

```xml
<pt:RefPearl rdf:about="https://www.pearltrees.com/t/hacktivism/id2492215?show=item,18077284">
   <pt:parentTree rdf:resource="https://www.pearltrees.com/t/hacktivism/id2492215" />
</pt:RefPearl>
```

The ingestion extracts `pt:parentTree` as an empty string (text content) instead of the `rdf:resource` attribute value.

**Solution:** This tool reconstructs the hierarchy by parsing pearl About URLs, which contain the parent tree ID in the format: `.../id{PARENT_ID}?show=item,{PEARL_ID}`.

## What It Does

1. **Extracts parent tree IDs** from pearl About URLs using regex pattern `/id(\d+)\?`
2. **Updates tree Children fields** - Builds list of child pearl IDs for each tree
3. **Updates pearl ParentTree fields** - Sets the parent tree ID for each pearl

## Pearltrees Data Model

- **Trees** (`pt:Tree`): Top-level containers (folders/categories)
  - Have `Children` field: list of pearl IDs
  - Do NOT have `ParentTree` (trees don't have parent trees)

- **Pearls** (`pt:Pearl`, `pt:RefPearl`, `pt:AliasPearl`, etc.): Items within trees
  - Have `ParentTree` field: ID of containing tree
  - May reference other trees or external pages

## Usage

```bash
# From UnifyWeaver root directory
dotnet run --project tools/database/RebuildPeartreesHierarchy pt_ingest_test.db
```

Or specify full database path:
```bash
dotnet run --project tools/database/RebuildPeartreesHierarchy /path/to/database.db
```

## Example Output

```
=== Rebuilding Tree Children from About URLs ===

Database: pt_ingest_test.db
Trees: 5002
Pearls: 6865

Extracting parent tree IDs from pearl About URLs...
  Pearl id10311488?show=item,100408713 → Parent tree 10311488
  ...

Processed: 6865 pearls
Pearls with parent: 6725
Unique parent trees: 1996

✓ Updated 1996 trees with Children lists
✓ Updated 6725 pearls with ParentTree field

=== Sample Tree with Children ===
Tree: Social Media
  ID: 10311488
  Children count: 14
    - s243a-wikispaces
    - pearlers
    - ...
```

## Integration with Ingestion

**Recommended workflow:**

1. Run ingestion: `dotnet run --project tmp/pt_ingest_test/pt_ingest_test.csproj`
2. Rebuild hierarchy: `dotnet run --project tools/database/RebuildPeartreesHierarchy pt_ingest_test.db`

**Future improvement:** Fix the RDF ingestion to extract `rdf:resource` attributes so this rebuild step isn't needed.

## Impact on Features

This rebuild enables:
- **Graph navigation**: `GetChildren()`, `GetParent()`, `GetAncestors()`, `GetSiblings()`
- **Tree context visualization**: Shows hierarchical structure in Linux `tree` format
- **Bookmark filing assistant**: Displays candidate locations with their existing content

Without this rebuild, candidate trees appear empty with no children shown.
