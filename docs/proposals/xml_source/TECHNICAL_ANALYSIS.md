# XML Source Technical Analysis

**Date:** 2025-10-30
**Status:** DRAFT - Under Discussion
**Purpose:** Technical analysis for implementing xml_source plugin

## Executive Summary

This document analyzes implementation options for the `xml_source` plugin, comparing Python-based streaming (lxml) vs shell-based approaches (xmllint/xmlstarlet). The analysis considers memory efficiency, tool availability, UnifyWeaver architecture, and firewall integration.

## Problem Statement

We need to extract XML elements from large RDF/XML files (potentially multi-GB) and stream them as null-delimited records. **Critical requirement:** Memory usage must remain constant - after emitting each element, its memory must be immediately released.

### Example Use Case

Extract Pearltrees RDF data:
```prolog
:- source(xml, pearltrees, [
    xml_file('export.rdf'),
    tags(['pt:Tree', 'pt:Page', 'pt:Note'])
]).
```

Expected behavior: Stream millions of elements without memory growth.

## Current UnifyWeaver Architecture

### Source Plugins
Source plugins generate **Bash scripts** that call tools/languages. They do NOT generate code in other target languages.

**Existing source plugins:**
- `csv_source.pl` - Generates Bash using `awk`
- `json_source.pl` - Generates Bash using `jq`
- `awk_source.pl` - Generates Bash using `awk`
- `http_source.pl` - Generates Bash using `curl`
- `python_source.pl` - Generates Bash that embeds Python via heredoc

### Key Pattern: python_source.pl

The `python_source.pl` plugin (331 lines) establishes the pattern for embedding Python in Bash:

```bash
#!/bin/bash
my_predicate() {
    timeout 30 python3 /dev/fd/3 3<<'PYTHON'
import sys
# Python code here
PYTHON
}
```

**Key features:**
- Python code embedded via heredoc pattern
- Timeout support
- Error handling
- Multiple templates (basic, stdin, sqlite)
- Already tested and working

### Target Languages vs Tools

**UnifyWeaver has 2 target languages:**
1. Bash (default)
2. PowerShell

**Important distinction:**
- **Target language** = Output format of compilation (bash scripts or PowerShell scripts)
- **Tool/Engine** = External program called by generated script (awk, jq, python3, curl, etc.)

**We are NOT adding Python as a target language.** We are using Python as a tool, like we already do with awk, jq, and curl.

## Tool Availability Analysis

### Current System (Ubuntu 22.04 WSL)
```
✅ python3 - Available
❌ lxml - NOT installed (requires: pip3 install lxml)
❌ xmllint - NOT available (requires: apt-get install libxml2-utils)
❌ xmlstarlet - NOT available (requires: apt-get install xmlstarlet)
```

### Typical Linux Server
- **xmllint:** Usually pre-installed (part of libxml2)
- **python3:** Usually pre-installed
- **lxml:** Rarely pre-installed (requires pip)
- **xmlstarlet:** Rarely pre-installed

### macOS
- **xmllint:** Pre-installed
- **python3:** Pre-installed (10.15+)
- **lxml:** Not pre-installed
- **xmlstarlet:** Not available (requires Homebrew)

### Windows (Native)
- **All tools:** Not available by default
- **via WSL:** Use Linux instructions
- **via Cygwin:** Available via package manager
- **via Chocolatey:** xmlstarlet available

## Implementation Options

### Option 1: lxml (Python iterparse) - RECOMMENDED

**Implementation approach:**
```prolog
% Reuse python_source.pl pattern
generate_xml_bash(Pred, File, Tags, Options, BashCode) :-
    generate_lxml_python_code(File, Tags, PythonCode),
    % Use existing python_source template pattern
    render_template(python_streaming_source,
                   [pred=Pred, python_code=PythonCode],
                   BashCode).
```

**Generated Bash:**
```bash
#!/bin/bash
pearltrees_stream() {
    python3 /dev/fd/3 3<<'PYTHON'
import sys
from lxml import etree

file = "export.rdf"
tags = {'pt:Tree', 'pt:Page'}
null = b'\0'

context = etree.iterparse(file, events=('start', 'end'))
event, root = next(context)
nsmap = root.nsmap or {}

def expand(tag):
    if ':' in tag:
        pfx, local = tag.split(':', 1)
        uri = nsmap.get(pfx)
        return f'{{{uri}}}{local}' if uri else tag
    return tag

want = {expand(t) for t in tags}

for event, elem in context:
    if event == 'end' and elem.tag in want:
        sys.stdout.buffer.write(etree.tostring(elem))
        sys.stdout.buffer.write(null)
        # Memory release - THE CRITICAL PART
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        root.clear()
PYTHON
}
```

**Pros:**
- ✅ **True constant memory** (Perplexity confirmed)
- ✅ Reuses existing python_source.pl patterns
- ✅ Robust namespace handling (critical for RDF)
- ✅ Well-tested memory release pattern
- ✅ Firewall already allows python3
- ✅ Works with existing UnifyWeaver architecture
- ✅ Same pattern as csv_source (bash calling tool)

**Cons:**
- ⚠️ Requires lxml installation (`pip3 install lxml`)
- ⚠️ Adds Python library dependency

**Firewall impact:**
```prolog
% Already exists:
allowed_service(bash, executable(python3)) :-
    \+ denied_service(bash, executable(python3)).

% Optional - add module checking:
allowed_python_module(lxml) :-
    \+ denied_python_module(lxml).
```

**Estimated implementation time:** 4-6 hours
- Plugin code: 2-3 hours (reuse python_source.pl)
- Tests: 1-2 hours
- Documentation: 1 hour

### Option 2: xmllint --stream

**Implementation approach:**
```bash
#!/bin/bash
pearltrees_stream() {
    xmllint --stream --pattern '//pt:Tree|//pt:Page' "$xml_file" \
        | grep -E '<(pt:Tree|pt:Page)' \
        | tr '\n' '\0'
}
```

**Note:** This is a simplified example. Real implementation would need more sophisticated parsing.

**Pros:**
- ✅ Usually pre-installed on Linux/macOS
- ✅ No external dependencies
- ✅ Simple implementation

**Cons:**
- ❌ **Perplexity review:** xmllint --stream is "designed for validation/debugging"
- ❌ **Limited functionality:** Pattern matching doesn't serialize full subtrees easily
- ❌ **Partial streaming:** May still buffer internally
- ❌ **Complex namespace handling:** Requires manual XPath setup
- ❌ **Output format issues:** Not designed for null-delimited element streaming

**Firewall impact:**
```prolog
allowed_service(bash, executable(xmllint)) :-
    \+ denied_service(bash, executable(xmllint)).
```

**Estimated implementation time:** 6-8 hours
- More complex due to limited tooling
- Needs custom parsing logic
- Namespace handling complexity

### Option 3: xmlstarlet sel - NOT RECOMMENDED

**Why NOT recommended:**

From Perplexity research:
> "xmlstarlet sel compiles an XSLT and applies it to each input document, which requires loading the XML into a libxml2/libxslt tree rather than using a streaming reader, so memory from matched subtrees is not released immediately after emission."

**Critical issue:** Loads entire file into memory. Fails our primary requirement.

**Verdict:** Do not implement this option.

## Recommendation: Phased Approach

### Phase 1: lxml Implementation (Primary)

**Implement lxml-based engine as the default:**

```prolog
:- source(xml, my_data, [
    xml_file('data.rdf'),
    tags(['pt:Tree']),
    engine(iterparse)  % default
]).
```

**Rationale:**
1. Meets memory requirements (confirmed by Perplexity)
2. Reuses proven python_source.pl patterns
3. Minimal new code needed
4. Firewall already configured for python3

### Phase 2: xmllint Fallback (Optional)

**Add xmllint engine for limited use cases:**

```prolog
:- source(xml, my_data, [
    xml_file('small.xml'),
    tags(['item']),
    engine(xmllint)  % for small files only
]).
```

**Use cases:**
- Environments where lxml unavailable
- Small files where streaming unnecessary
- Simple tag structures without complex namespaces

### Phase 3: Documentation & Testing

**Required documentation:**
- Installation guide (✅ DONE - XML_PARSING_TOOLS_INSTALLATION.md)
- Engine comparison table
- Memory usage benchmarks
- Namespace handling examples

**Required tests:**
- Small RDF file (basic functionality)
- Large RDF file (memory stress test)
- Multiple namespaces (Pearltrees use case)
- Error handling (missing files, bad XML)

## Open Questions for Discussion

### 1. lxml Dependency Management

**Question:** How should we handle the lxml dependency?

**Options:**
A. **Document only** - Installation guide shows how to install, user responsible
B. **Runtime check** - Plugin checks if lxml available, graceful error if not
C. **Auto-install** - Try to pip install lxml if missing (risky)
D. **Fallback** - Try lxml first, fall back to xmllint if unavailable

**My recommendation:** Option B + D
- Check at compile time if lxml available
- Warn user with installation instructions if missing
- Optionally fall back to xmllint for small files

### 2. Engine Auto-Selection

**Question:** Should we auto-detect which engine to use?

**Options:**
A. **Explicit only** - User must specify `engine(iterparse)` or `engine(xmllint)`
B. **Smart default** - Check if lxml available, use it; otherwise try xmllint
C. **File-size based** - Use streaming (lxml) for large files, xmllint for small

**My recommendation:** Option B
- Default to `engine(iterparse)` if lxml available
- Fall back to `engine(xmllint)` if not
- User can override with explicit `engine()` option

### 3. Python Code Generation

**Question:** Should we generate Python code inline (like python_source.pl) or as external files?

**Options:**
A. **Inline (heredoc)** - Python code embedded in Bash script (like python_source.pl)
B. **External file** - Generate .py file alongside .sh file
C. **Hybrid** - Inline for simple cases, external for complex

**My recommendation:** Option A (Inline)
- Consistent with python_source.pl
- Single file deployment
- No file management issues
- Can reference external Python files via python_file() option if needed

### 4. Namespace Handling

**Question:** How should we handle XML namespaces?

**Current approach in Gemini's plan:**
- Extract namespace map from root element
- Expand prefixed tags using nsmap
- Example: `'pt:Tree'` → `'{http://www.pearltrees.com/xmlns/...}Tree'`

**Concerns:**
- What if namespace prefixes differ in the file?
- What if user wants to match by namespace URI directly?

**Options:**
A. **Prefix-based** - Match using prefixes as user provides (pt:Tree)
B. **URI-based** - User provides full URIs, we match exactly
C. **Flexible** - Support both prefix and URI matching

**My recommendation:** Option A with C as extension
- Start with prefix-based (simpler for users)
- Document that prefixes must match file's namespace declarations
- Consider adding URI-based matching in future version

### 5. Memory Verification

**Question:** Should we add memory usage monitoring/verification?

**Options:**
A. **Trust lxml** - Rely on known lxml behavior, no monitoring
B. **Optional debug** - Add `debug(memory)` option for testing
C. **Built-in checks** - Monitor and warn if memory grows unexpectedly

**My recommendation:** Option A for now, B for testing
- lxml iterparse + clear pattern is well-established
- Add memory profiling to test suite
- Document memory characteristics

### 6. Error Handling for Malformed XML

**Question:** What should happen if XML is malformed?

**Options:**
A. **Fail fast** - Stop processing, return error
B. **Skip element** - Log warning, continue with next element
C. **Configurable** - User chooses via `error_handling(fail|skip|silent)`

**My recommendation:** Option C
- Default to `error_handling(fail)` for safety
- Allow `error_handling(skip)` for resilient processing
- Consistent with python_source.pl error handling pattern

## Implementation Checklist

### Prerequisites
- [ ] Review and approve this technical analysis
- [ ] Decide on open questions (1-6 above)
- [ ] Update Gemini's planning docs with decisions

### Phase 1: Core Implementation
- [ ] Create `src/unifyweaver/sources/xml_source.pl`
- [ ] Implement lxml iterparse engine
- [ ] Add firewall rules (python3, lxml module check)
- [ ] Generate Python code using template system
- [ ] Add namespace handling logic

### Phase 2: Testing
- [ ] Create `tests/test_data/sample.rdf` (small test file)
- [ ] Create `tests/core/test_xml_source.pl`
- [ ] Test basic element extraction
- [ ] Test multiple namespaces
- [ ] Test null-delimited output
- [ ] Memory stress test (large file)

### Phase 3: Documentation
- [ ] Update `docs/proposals/xml_source/SPECIFICATION.md`
- [ ] Update `docs/proposals/xml_source/IMPLEMENTATION_STRATEGY.md`
- [ ] Create usage examples
- [ ] Document namespace handling
- [ ] Document memory characteristics

### Phase 4: Integration
- [ ] Add to `examples/data_sources_demo.pl`
- [ ] Update CHANGELOG.md
- [ ] Update README.md (if needed)

### Optional: xmllint Fallback
- [ ] Implement xmllint engine
- [ ] Add engine auto-detection
- [ ] Add engine comparison tests
- [ ] Document engine selection

## Timeline Estimate

**Phase 1 (lxml core):** 4-6 hours
- Plugin implementation: 2-3 hours
- Basic testing: 1-2 hours
- Documentation updates: 1 hour

**Phase 2 (comprehensive testing):** 2-3 hours
- Test suite: 1-2 hours
- Memory profiling: 1 hour

**Phase 3 (xmllint fallback):** 3-4 hours (optional)
- Implementation: 2 hours
- Testing: 1 hour
- Documentation: 1 hour

**Total: 6-9 hours** (core) or **9-13 hours** (with fallback)

## References

- Perplexity research on xmlstarlet/xmllint memory behavior
- Gemini's planning documents in `docs/proposals/xml_source/`
- Existing `src/unifyweaver/sources/python_source.pl` (331 lines)
- Installation guide: `docs/XML_PARSING_TOOLS_INSTALLATION.md`

## Next Steps

1. **Review this analysis** - Discuss open questions
2. **Make decisions** - Resolve questions 1-6
3. **Update planning docs** - Incorporate decisions into Gemini's docs
4. **Share with Gemini** - Provide clear implementation guidance
5. **Begin implementation** - Start with Phase 1 (lxml core)
