# Pearltrees Hierarchy Rebuild Process

## Overview

When ingesting Pearltrees RDF exports into LiteDB, parent-child relationships between trees and pearls are not automatically extracted. This document explains why and how to rebuild the hierarchy.

## The Problem

### RDF Structure
Pearltrees exports use RDF resource references for parent-child relationships:

```xml
<pt:RefPearl rdf:about="https://www.pearltrees.com/user/topic/id12345?show=item,67890">
   <dcterms:title>My Bookmark</dcterms:title>
   <pt:parentTree rdf:resource="https://www.pearltrees.com/user/topic/id12345" />
   <pt:inTreeSinceDate>2024-01-15T10:30:00</pt:inTreeSinceDate>
</pt:RefPearl>
```

The `pt:parentTree` element has an `rdf:resource` attribute containing the parent tree URL, not text content.

### Current Ingestion Limitation
The XPath-based ingestion extracts element **text content** only, not attributes:
- `pt:parentTree` → empty string (no text content)
- `rdf:resource="..."` → not extracted

This results in:
```csharp
pearl.ParentTree = null;  // Should be "12345"
tree.Children = null;     // Should be list of pearl IDs
```

## The Solution

### Workaround: Extract from About URLs
Pearl About URLs contain the parent tree ID in their structure:
```
https://www.pearltrees.com/user/topic/id12345?show=item,67890
                                      ^^^^^
                                   parent tree ID
```

The **RebuildPeartreesHierarchy** tool uses regex to extract these IDs and rebuild the relationships.

### Process

1. **Ingest RDF into LiteDB**
   ```bash
   dotnet run --project tmp/pt_ingest_test/pt_ingest_test.csproj
   ```

2. **Rebuild hierarchy**
   ```bash
   dotnet run --project tools/database/RebuildPeartreesHierarchy pt_ingest_test.db
   ```

3. **Verify results**
   - Check `tree.Children` arrays are populated
   - Check `pearl.ParentTree` fields are set
   - Test graph navigation methods work

## Impact on Features

### Without Hierarchy Rebuild
```
Candidate: "Quantum Mechanics" (similarity: 0.719)
    ├── Quantum Mechanics/        ← CANDIDATE (place new bookmark here)
```
No context! Users can't see what's already in this tree.

### With Hierarchy Rebuild
```
Candidate: "Quantum Mechanics" (similarity: 0.719)
    ├── Quantum Mechanics/        ← CANDIDATE (place new bookmark here)
    │   ├── Wave Function
    │   ├── Schrödinger Equation
    │   ├── Heisenberg Uncertainty
    │   └── Quantum Entanglement
```
Rich context shows existing organization!

### Affected APIs
- `PtSearcher.GetChildren(id)` - Returns empty without rebuild
- `PtSearcher.GetParent(id)` - Returns null without rebuild
- `PtSearcher.GetAncestors(id)` - Returns empty without rebuild
- `PtSearcher.GetSiblings(id)` - Returns empty without rebuild
- `PtSearcher.BuildTreeContext(id, score)` - Shows minimal context
- `PtSearcher.FindBookmarkPlacements(...)` - Shows trees without content

## Future Improvements

### Option 1: Fix RDF Extraction
Modify the Prolog query engine to extract `rdf:resource` attributes:
```prolog
% Current: only gets text content
rdf_property(Element, Property, Value) :-
    xpath(Element, Property, Value).

% Needed: also get rdf:resource attributes
rdf_property(Element, Property, Value) :-
    xpath(Element, Property/@'rdf:resource', Value).
```

### Option 2: Automatic Rebuild
Add rebuild as a post-ingestion step in the main ingestion pipeline:
```csharp
// After ingestion completes
if (embeddingProvider is not null)
{
    Console.WriteLine("Rebuilding hierarchy...");
    PtHierarchyRebuilder.Rebuild(dbPath);
}
```

### Option 3: Alternative Data Source
Use Pearltrees API (if available) instead of RDF export, which might provide structured parent-child data.

## Testing

Verify hierarchy rebuild with:
```csharp
using var searcher = new PtSearcher(dbPath, embeddingProvider);

// Find a tree with children
var results = searcher.SearchSimilar("physics", topK: 1, minScore: 0.5, typeFilter: "pt:Tree");
var testId = results[0].Id;

// Test graph navigation
var children = searcher.GetChildren(testId);
Console.WriteLine($"Children: {children.Count}");  // Should be > 0

// Test tree context
var context = searcher.BuildTreeContext(testId, 1.0);
Console.WriteLine(context);  // Should show children
```

## References

- Tool: `tools/database/RebuildPeartreesHierarchy/`
- Code: `src/unifyweaver/targets/csharp_query_runtime/PtSearcher.cs`
- Example RDF: `context/PT/Example_pearltrees_rdf_export.rdf`
