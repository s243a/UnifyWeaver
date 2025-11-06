# Future XML Tooling Directions

This note captures ideas for building out UnifyWeaver’s XML capabilities beyond the current `xml_source` plugin. It is a scratchpad—update it as new use cases appear or when we learn more about the tools.

## 1. Engine Landscape (Current & Potential)

| Engine/Tool | Strengths | Weaknesses | Notes / Possible Work |
|-------------|-----------|------------|-----------------------|
| **Python / lxml.etree.iterparse** (`engine(iterparse)`) | True streaming; deterministic memory usage; Python’s full power (XPath, XSLT, schema validation via `lxml` add-ons) | Requires `lxml` wheel; Python dependency | - Extend to expose incremental XPath filters (e.g. allow users to supply Python predicates for complex conditions)<br>- Explore packaging a reusable Python module so other plugins can reuse the streaming logic |
| **xmllint** (`engine(xmllint)`) | Part of `libxml2` suite; fast XPath evaluator; rich CLI options (validation, XInclude, DTD processing) | CLI emits concatenated output; currently relies on embedded Python for splitting/reformatting; streaming support limited to validation | - Investigate pure-shell splitter to drop Python dependency when `namespace_fix(false)`<br>- Add configuration hooks for validation options (`--schema`, `--relaxng`, `--noout` etc.)<br>- Consider piping into other UnifyWeaver-friendly tools (`xmllint | jq`, `| awk`, etc.) |
| **xmlstarlet** (`engine(xmlstarlet)`) | Select + transform; supports large library of operations (edits, transforms, formatting); ubiquitous on Linux | Loads document in memory; CLI syntax is verbose; namespace handling requires manual declarations | - Document best practices for big queries (e.g. how to pre-limit scope)<br>- Provide helper templates for frequent tasks (attribute extraction, XSLT invocation) |
| **Other libxml2 tools** (`xmllint` family) | `xmllint` supports HTML tidy, canonicalisation, etc. | N/A | - Explore additional subcommands (e.g. `--format`, `--c14n`) as dedicated UnifyWeaver transforms |
| **Saxon / Java-based processors** | Full XPath/XQuery/XSLT 3.0; streaming; schema-aware processing | Requires JVM; heavier footprint; licensing for Saxon-EE | - Candidate for advanced transformations; would need firewall/workflow adjustments |
| **Rust/Go XML libs** | High-performance streaming; typed pipelines | Need new runtimes + sandbox permissions | - Worth tracking if we add Rust/Go targets in future |

## 2. Short-Term Ideas

1. **Python-free xmllint splitter**  
   ✅ A Perl-based splitter now ships with `xml_source`. Select it via `xmllint_splitter(perl)`; the default remains the inline Python splitter. Preference rules can hint an order (e.g. `prefer([xmllint_splitter(perl)])`), and the firewall tooling automatically skips any splitter whose executable is denied or missing. Namespace repair still behaves the same, and this path keeps the xmllint engine usable on hosts without Python extras.

2. **XPath helper library**  
   Create a small Prolog helper that builds XPath expressions (including namespace bindings) to avoid hand-rolled strings in multiple places. Could support `descendant::`, attribute filters, etc.

3. **Namespace registry**  
   Move `known_namespace/2` into a shared configuration file so projects can add prefixes once and have them available to all engines (xmllint/xmlstarlet/potential future ones).

4. **Enhanced test coverage**  
   Add sample XML fixtures with:
   - default namespaces (no prefixes),
   - multiple prefixes referencing the same URI,
   - large CDATA sections and mixed content,
   to ensure each engine emits the expected fragments.

5. **Namespace defaults audit**  
   Decide whether `namespace_fix/1` should default to `false` (preserve original prefixes) and document best practices for RDF-oriented pipelines. Update tests and docs accordingly.

## 3. Medium-Term Opportunities

1. **Validation Workflow**  
   - Provide a new plugin or option (`validate(schema(Path))`) that runs `xmllint --schema`, `--relaxng`, or `--dtdvalid`.  
   - Capture validation errors and surface them in UnifyWeaver logs/tests.

2. **XSLT & Transformation Support**  
   - `xmlstarlet tr stylesheet.xsl file.xml` or `xsltproc`.  
   - Could compile into streaming pipelines for transforming XML into CSV/JSON for downstream tools.

3. **Selective Extraction**  
   - Support attribute-only extraction (e.g. stream values of `@rdf:about`).  
   - Possibly integrate with `xmlstarlet sel -T -t -m` for templated output.

4. **Incremental Parsing with Buffers**  
   - Extend iterparse engine to supply context objects (e.g. parent attributes, path) by augmenting the embedded Python code.  
   - Investigate chunking large text nodes for memory control.

## 4. Longer-Term Vision

1. **Unified XML Toolkit Module**  
   - Factor out common namespace/XPath utilities.  
   - Provide a consistent interface so new engines can be swapped in with minimal changes.

2. **Streaming Query Language**  
   - Build a DSL around XPath-lite expressions that choose the right engine automatically, similar to how `jq` does for JSON.  
   - Extend `xml_source` to compile full pipelines (select → transform → output).

3. **Cross-Language Interop**  
   - Explore hooking into `libxml2` bindings for other languages (Rust `xmlparser`, Go `encoding/xml`) for performance-critical tasks.  
   - Could be packaged as optional engines if the tooling is installed.

4. **XML + Semantic Web**  
   - Many datasets (RDF/XML, Atom, RSS) benefit from higher-level handling (e.g. RDF to triples). Consider dedicated plugins that convert XML sources into RDF triples or graph queries.

## 5. Open Questions

- When do we prefer CLI tools over Python, given the extra dependencies?  
- What is the threshold (file size, complexity) where xmlstarlet becomes impractical?  
- Should namespace repair be configurable at the global level instead of per-source?  
- How do we best expose XPath/XSLT knowledge to LLM agents (docs, prompts, helper predicates)?

Document status: **brainstorm**. Refine as we gather more experience with real XML workloads.
