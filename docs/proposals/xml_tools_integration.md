# XML Tools Integration - Unifying In-Memory and Streaming Approaches

**Status:** Design Proposal
**Date:** 2025-11-22
**Branch:** feature/pearltrees-extraction

---

## Current State: Two Separate Approaches

### Approach 1: In-Memory (Existing)

**Location:** `playbooks/examples_library/xml_examples.md`

**Usage:**
```prolog
:- source(python, sum_xml_prices, [
    python_inline("
        import xml.etree.ElementTree as ET
        xml_data = '''<products>...</products>'''
        root = ET.fromstring(xml_data)
        # Process...
    ")
]).
```

**Characteristics:**
- ✅ Embedded XML in code
- ✅ Uses UnifyWeaver's source system
- ✅ Dynamic compilation to bash
- ❌ Loads entire XML into memory
- ❌ Limited to small XML (<10MB)

### Approach 2: Streaming (New)

**Location:** `scripts/extract_pearltrees.sh`, `scripts/extract_pearltrees_parallel.sh`

**Usage:**
```bash
# Sequential
./scripts/extract_pearltrees.sh input.rdf output/

# Parallel
./scripts/extract_pearltrees_parallel.sh input.rdf output/ --workers=4
```

**Characteristics:**
- ✅ Handles large files (100MB+)
- ✅ Constant memory (~20KB)
- ✅ Parallel processing support
- ❌ Standalone scripts (not integrated)
- ❌ Different API than source system

---

## Integration Goal

**Create a unified API** where UnifyWeaver's source system automatically chooses the best approach:

```prolog
% Small embedded XML → in-memory
:- source(xml, process_products, [
    xml_inline("<products>...</products>"),
    tag("product")
]).

% Large external file → streaming (auto-detected)
:- source(xml, process_pearltrees, [
    file("pearltrees.rdf"),
    tag("pt:Tree"),
    output("trees.pl")
]).

% Explicit streaming mode
:- source(xml, process_large_catalog, [
    file("catalog.xml"),
    tag("product"),
    mode(streaming),  % Force streaming
    parallel(true),   % Use parallel processing
    workers(4)
]).
```

---

## Integration Strategy

### Option 1: Extend Python Source Plugin

**Modify:** `src/unifyweaver/sources/python_source.pl`

**Add new configuration options:**
```prolog
validate_config(Config) :-
    % Existing options
    (   member(python_inline(_), Config) ; ...
    % NEW: XML streaming options
    ;   member(xml_streaming(Options), Config)
    ->  validate_xml_streaming_options(Options)
    ;   ...
    ).

validate_xml_streaming_options(Options) :-
    member(file(File), Options),
    member(tag(Tag), Options),
    % Optional: mode(streaming/parallel), workers(N)
    !.
```

**Compilation:**
```prolog
compile_source(Pred/Arity, Config, Options, BashCode) :-
    (   member(xml_streaming(XmlOpts), Config)
    ->  compile_xml_streaming(Pred/Arity, XmlOpts, Options, BashCode)
    ;   % Existing compilation
        ...
    ).

compile_xml_streaming(Pred/Arity, XmlOpts, Options, BashCode) :-
    member(file(File), XmlOpts),
    member(tag(Tag), XmlOpts),
    member(output(Output), XmlOpts),

    % Auto-detect mode based on file size
    (   member(mode(Mode), XmlOpts)
    ->  true
    ;   size_file(File, Size),
        (   Size > 10485760  % >10MB
        ->  Mode = streaming
        ;   Mode = in_memory
        )
    ),

    % Generate appropriate bash code
    (   Mode = streaming
    ->  generate_streaming_code(File, Tag, Output, BashCode)
    ;   Mode = parallel
    ->  generate_parallel_code(File, Tag, Output, XmlOpts, BashCode)
    ;   generate_in_memory_code(File, Tag, Output, BashCode)
    ).
```

**Generated Bash Code:**
```prolog
generate_streaming_code(File, Tag, Output, BashCode) :-
    format(atom(BashCode),
        'awk -f scripts/utils/select_xml_elements.awk -v tag="~w" ~w | \\
         python3 scripts/utils/xml_to_prolog_facts.py --element-type=tree > ~w',
        [Tag, File, Output]).

generate_parallel_code(File, Tag, Output, Options, BashCode) :-
    (   member(workers(N), Options)
    ->  true
    ;   N = 4  % Default workers
    ),
    format(atom(BashCode),
        'scripts/extract_pearltrees_parallel.sh ~w ~w --workers=~w',
        [File, Output, N]).
```

**Pros:**
- ✅ Unified API (same `:- source(...)` syntax)
- ✅ Auto-detection (file size → mode)
- ✅ Backward compatible
- ✅ Integrates with existing source system

**Cons:**
- ⚠️ Mixing concerns (Python source now handles XML too)
- ⚠️ Python source becomes more complex

---

### Option 2: New XML Source Plugin

**Create:** `src/unifyweaver/sources/xml_source.pl`

**Structure:**
```prolog
:- module(xml_source, []).

:- initialization(
    register_source_type(xml, xml_source),
    now
).

%% Source info
source_info(info(
    name('XML Data Source'),
    version('1.0.0'),
    description('Process XML files with automatic streaming for large files'),
    supported_arities([1, 2, 3, 4, 5])
)).

%% Validate configuration
validate_config(Config) :-
    % Must have either xml_inline or file
    (   member(xml_inline(_), Config)
    ;   member(file(_), Config)
    ),
    % Must specify tag to extract
    member(tag(_), Config),
    !.

%% Compile to bash
compile_source(Pred/Arity, Config, Options, BashCode) :-
    % Extract configuration
    (   member(file(File), Config)
    ->  InputType = file
    ;   member(xml_inline(XmlData), Config)
    ->  InputType = inline
    ),

    member(tag(Tag), Config),

    (   member(output(Output), Config)
    ->  true
    ;   Output = stdout
    ),

    % Determine processing mode
    determine_mode(InputType, File, Config, Mode),

    % Generate bash code
    generate_bash_code(Mode, InputType, File, Tag, Output, Config, BashCode).

determine_mode(inline, _, _, in_memory) :- !.
determine_mode(file, File, Config, Mode) :-
    (   member(mode(Mode), Config)
    ->  true  % Explicit mode
    ;   size_file(File, Size),
        (   Size > 10485760  % 10MB
        ->  (   member(parallel(true), Config)
            ->  Mode = parallel
            ;   Mode = streaming
            )
        ;   Mode = in_memory
        )
    ).

generate_bash_code(in_memory, inline, _, Tag, Output, Config, BashCode) :-
    member(xml_inline(XmlData), Config),
    % Generate Python code for in-memory processing
    format(atom(BashCode), '
python3 << ''EOF''
import xml.etree.ElementTree as ET

xml_data = """~w"""
root = ET.fromstring(xml_data)

for elem in root.findall(''~w''):
    # Process element
    # Emit facts to ~w
    pass
EOF
', [XmlData, Tag, Output]).

generate_bash_code(streaming, file, File, Tag, Output, _, BashCode) :-
    format(atom(BashCode),
        'awk -f scripts/utils/select_xml_elements.awk -v tag="~w" ~w | \\
         python3 scripts/utils/xml_to_prolog_facts.py --element-type=tree > ~w',
        [Tag, File, Output]).

generate_bash_code(parallel, file, File, Tag, Output, Config, BashCode) :-
    (   member(workers(N), Config)
    ->  true
    ;   N = 4
    ),
    format(atom(BashCode),
        'scripts/extract_pearltrees_parallel.sh ~w ~w --workers=~w',
        [File, Output, N]).
```

**Pros:**
- ✅ Clean separation (XML handling in dedicated module)
- ✅ Simpler to understand
- ✅ Easier to extend
- ✅ Focused API

**Cons:**
- ⚠️ New plugin to maintain
- ⚠️ Duplicates some Python source functionality

---

### Option 3: Bash Pipeline Source Plugin

**Create:** `src/unifyweaver/sources/bash_pipeline_source.pl`

**Concept:** Generic bash pipeline composer

```prolog
:- source(bash_pipeline, extract_trees, [
    stages([
        stage(awk, 'select_xml_elements.awk', [
            tag('pt:Tree')
        ]),
        stage(python, 'xml_to_prolog_facts.py', [
            element_type(tree)
        ])
    ]),
    input_file('pearltrees.rdf'),
    output_file('trees.pl')
]).
```

**Pros:**
- ✅ Generic (works for any pipeline, not just XML)
- ✅ Explicit stages (transparent)
- ✅ Composable

**Cons:**
- ⚠️ More verbose
- ⚠️ Less "magical" auto-detection

---

## Recommendation: Option 2 (New XML Source Plugin)

**Why:**
1. **Clear purpose** - Dedicated to XML processing
2. **Simple API** - Users declare XML sources naturally
3. **Auto-magic** - Automatically chooses best approach
4. **Extensible** - Easy to add features (filtering, transformation types)
5. **Clean code** - Doesn't complicate Python source

**Migration path:**
1. Create `xml_source.pl` plugin
2. Add examples to `xml_examples.md`
3. Update playbooks to show both approaches
4. Existing code continues to work (backward compatible)

---

## Implementation Plan

### Phase 1: Core XML Source Plugin

**File:** `src/unifyweaver/sources/xml_source.pl`

**Features:**
- ✅ `xml_inline(Data)` for embedded XML
- ✅ `file(Path)` for external files
- ✅ `tag(Pattern)` for element selection
- ✅ Auto-detection of mode (in_memory vs streaming)
- ✅ Generate appropriate bash code

**Example:**
```prolog
:- source(xml, get_products, [
    file('catalog.xml'),
    tag('product'),
    output('products.pl')
]).

% Auto-detects: file size → mode
% Small file → in-memory Python
% Large file → streaming awk pipeline
```

### Phase 2: Parallel Support

**Add options:**
```prolog
:- source(xml, get_trees, [
    file('large.rdf'),
    tag('pt:Tree'),
    output('trees.pl'),
    mode(parallel),      % Force parallel
    workers(8)           % Use 8 workers
]).
```

**Generated code:**
```bash
scripts/extract_pearltrees_parallel.sh large.rdf trees.pl --workers=8
```

### Phase 3: Advanced Features

**Filtering:**
```prolog
:- source(xml, get_filtered_pearls, [
    file('pearltrees.rdf'),
    tag('pt:.*Pearl'),
    filter(parent_tree('2492215')),
    output('filtered.pl')
]).
```

**Custom transformation:**
```prolog
:- source(xml, get_custom, [
    file('data.xml'),
    tag('record'),
    transform(python_inline("
        # Custom transformation logic
        def process_element(xml_chunk):
            ...
    ")),
    output('custom.pl')
]).
```

---

## Usage Examples

### Example 1: Small Embedded XML (In-Memory)

```prolog
:- source(xml, sum_prices, [
    xml_inline("
        <products>
            <product><price>100</price></product>
            <product><price>200</price></product>
        </products>
    "),
    tag('product'),
    transform(sum_prices)
]).

% Automatically uses in-memory processing
% Generates Python ElementTree code
```

### Example 2: Large File (Auto-Streaming)

```prolog
:- source(xml, extract_trees, [
    file('pearltrees_export.rdf'),  % 19MB file
    tag('pt:Tree'),
    output('trees.pl')
]).

% Auto-detects: 19MB > 10MB → streaming mode
% Generates: awk | python pipeline
```

### Example 3: Explicit Parallel Processing

```prolog
:- source(xml, extract_parallel, [
    file('huge_catalog.xml'),  % 1GB file
    tag('product'),
    output('products.pl'),
    mode(parallel),
    workers(8)
]).

% Generates: parallel extraction script
% 8 workers process byte-range partitions
```

### Example 4: Filtered Extraction

```prolog
:- source(xml, extract_filtered, [
    file('pearltrees.rdf'),
    tag('pt:.*Pearl'),
    filter([
        parent_tree('2492215')
    ]),
    output('pearls_filtered.pl')
]).

% Generates: awk | filter_by_parent_tree.py | transform
```

---

## Backward Compatibility

**Existing code continues to work:**

```prolog
% Old Python source approach (still works)
:- source(python, process_xml, [
    python_inline("import xml.etree.ElementTree as ET...")
]).
```

**New XML source is additive:**

```prolog
% New XML source approach (cleaner API)
:- source(xml, process_xml, [
    file('data.xml'),
    tag('record')
]).
```

**Both compile to bash,** both work in UnifyWeaver pipelines.

---

## Next Steps

1. ✅ Create `src/unifyweaver/sources/xml_source.pl`
2. ✅ Add tests for different modes
3. ✅ Update `xml_examples.md` with new examples
4. ✅ Document in playbooks
5. ⏳ Get user feedback on API design

---

## Questions for Review

1. **API design:** Does the `:- source(xml, ...)` syntax feel natural?
2. **Auto-detection:** Is 10MB a good threshold for streaming vs in-memory?
3. **Mode names:** `in_memory`, `streaming`, `parallel` - clear enough?
4. **Filtering:** Should filters be first-class or keep as pipeline stage?
5. **Output:** Default to stdout or require explicit output file?

---

**Status:** Ready to implement pending user approval of design.
