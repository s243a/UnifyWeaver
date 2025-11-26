<!--
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# LiteDB XML Fixed-Point Crawl (C#)

## `unifyweaver.execution.litedb_xml_fixedpoint`

> [!example-record]
> id: unifyweaver.execution.litedb_xml_fixedpoint
> name: LiteDB XML Fixed-Point Crawl
> platform: csharp

This example shows how to:
- Stream XML fragments (with pearltrees defaults) via `XmlStreamReader`
- Upsert projected records into LiteDB (POCO + Raw `BsonDocument`)
- Iteratively fetch children (parentTree/seeAlso) for N hops (e.g., 5)
- Filter out private entries (`pt:privacy > 1`)

### Files
- `test_data/test_xml_fragments.txt` — scrubbed sample (pearltrees-style, physics-themed)
- `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs` — `XmlStreamReader` with namespace/CDATa support
- (This example uses `BsonDocument` for Raw and a POCO for typed fields.)

### Suggested pipeline (outline)
1. Define a dynamic XML source with `pearltrees(true)` and `record_format(xml)`.
2. Project to a row dictionary with keys: `id`, `title`, `privacy`, `parentTree`, `children[]`, plus `Raw` for the whole map.
3. Upsert into LiteDB (`_id = id`).
4. Query LiteDB for next-hop child tree IDs (from `children` or `parentTree`).
5. Repeat for a bounded number of iterations.

### Minimal C# harness (conceptual)
```csharp
var xmlConfig = new XmlSourceConfig {
    InputPath = "test_data/scrubbed_pearltrees.xml",
    RecordSeparator = RecordSeparatorKind.LineFeed,
    NamespacePrefixes = new Dictionary<string,string> {
        { "http://www.pearltrees.com/rdf/0.1/#", "pt" },
        { "http://purl.org/dc/elements/1.1/", "dcterms" }
    },
    TreatPearltreesCDataAsText = true
};

using var crawler = new PtCrawler("pearltrees.db", xmlConfig);
crawler.IngestOnce(); // single pass over the fragments
// crawler.FixedPoint(seeds, id => BuildConfigForId(id), maxDepth: 5); // future: iterative child expansion
```

### Notes
- `Raw` can capture the full dictionary projection for unmapped fields.
- `Embedding` (not shown here) can be stored as `double[]` for later similarity search; LiteDB doesn’t have native vector search.
- A future enhancement could replace title search with embeddings (e.g., BERT) but is out of scope here.
