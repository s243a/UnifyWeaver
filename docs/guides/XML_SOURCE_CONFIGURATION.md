# XML Source Configuration Guide

**Purpose:** Complete configuration reference for UnifyWeaver's XML processing sources.

---

## Overview: Three Approaches

UnifyWeaver provides three complementary approaches for XML processing:

1. **Python Source** - Use existing `python` source with XML code
2. **XML Source** - Dedicated `xml` source with auto-optimization
3. **Bash Pipeline** - Generic `bash_pipeline` for custom workflows

**All are configurable.** Choose based on your needs.

---

## Approach 1: Python Source with XML

**Use when:** You want full Python/ElementTree control

### Basic Usage

```prolog
:- source(python, process_xml, [
    python_inline("
import xml.etree.ElementTree as ET
# Your Python code here
    ")
]).
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `python_inline(Code)` | atom | required | Embedded Python code |
| `python_file(File)` | atom | - | External Python file |
| `timeout(Seconds)` | number | 30 | Execution timeout |
| `python_interpreter(Path)` | atom | python3 | Python interpreter |
| `input_mode(Mode)` | atom | stdin | Input method |
| `output_format(Format)` | atom | tsv | Output format |

### XML-Specific Patterns

**In-memory (small XML):**
```prolog
:- source(python, process_small, [
    python_inline("
import xml.etree.ElementTree as ET

xml_data = '''<root>...</root>'''
root = ET.fromstring(xml_data)
# Process...
    ")
]).
```

**Streaming (large XML):**
```prolog
:- source(python, process_large, [
    python_inline("
import xml.etree.ElementTree as ET

for event, elem in ET.iterparse('large.xml', events=('end',)):
    if 'record' in elem.tag:
        # Process element
        elem.clear()  # Free memory
    ")
]).
```

---

## Approach 2: XML Source (Dedicated)

**Use when:** You want automatic optimization and simple API

### Basic Usage

```prolog
:- source(xml, extract_elements, [
    file('data.xml'),
    tag('element'),
    output('output.pl')
]).
```

### Complete Configuration

```prolog
:- source(xml, my_extractor, [
    %% INPUT (required, choose one)
    file('path/to/file.xml'),           % External file
    xml_inline('<data>...</data>'),     % Embedded XML

    %% SELECTION (required)
    tag('TagPattern'),                  % Tag to extract (regex)
                                         % Examples: 'product', 'pt:Tree', 'pt:.*Pearl'

    %% MODE SELECTION (optional)
    mode(auto),                         % Auto-detect (default)
    mode(in_memory),                    % Force in-memory
    mode(streaming),                    % Force streaming
    mode(parallel),                     % Force parallel

    %% THRESHOLDS (optional, customize auto-detection)
    streaming_threshold(10485760),      % Bytes (default: 10MB)
                                         % Files larger → streaming mode

    %% PARALLEL OPTIONS (when mode=parallel or auto-detected)
    workers(4),                         % Number of parallel workers (default: 4)
    parallel(true),                     % Enable parallel for large files

    %% PROCESSING
    element_type(tree),                 % Element type for transformation
                                         % Options: tree, pearl, product, custom

    %% OUTPUT
    output('output.pl')                 % Output file (default: stdout)
]).
```

### Mode Selection Details

#### Auto Mode (Default)

```prolog
:- source(xml, auto_detect, [
    file('data.xml'),
    tag('record'),
    mode(auto)  % Can be omitted, this is default
]).
```

**Behavior:**
- File size <10MB → `in_memory` mode
- File size >10MB → `streaming` mode
- If `parallel(true)` and >10MB → `parallel` mode

**Customize threshold:**
```prolog
:- source(xml, custom_threshold, [
    file('data.xml'),
    tag('record'),
    streaming_threshold(52428800)  % 50MB threshold instead of 10MB
]).
```

#### Explicit Modes

**In-Memory:**
```prolog
:- source(xml, force_memory, [
    file('data.xml'),
    tag('record'),
    mode(in_memory)  % Force in-memory even if large
]).
```

**Streaming:**
```prolog
:- source(xml, force_streaming, [
    file('data.xml'),
    tag('record'),
    mode(streaming)  % Force streaming even if small
]).
```

**Parallel:**
```prolog
:- source(xml, force_parallel, [
    file('data.xml'),
    tag('record'),
    mode(parallel),
    workers(8)  % Use 8 workers
]).
```

### Element Types

Currently supported:
- `tree` - Pearltrees tree elements
- `pearl` - Pearltrees pearl elements
- (Extensible - add more in `xml_to_prolog_facts.py`)

---

## Approach 3: Bash Pipeline

**Use when:** You need custom pipeline stages or explicit control

### Basic Usage

```prolog
:- source(bash_pipeline, my_pipeline, [
    stages([
        stage(awk, 'select.awk', [tag='element']),
        stage(python, 'transform.py', [format='facts'])
    ]),
    input_file('input.xml'),
    output_file('output.pl')
]).
```

### Complete Configuration

```prolog
:- source(bash_pipeline, complex_pipeline, [
    %% STAGES (required)
    stages([
        %% AWK stage
        stage(awk, 'script.awk', [
            tag='pattern',           % -v tag="pattern"
            delimiter='\0',          % -v delimiter="\0"
            debug=1                  % -v debug=1
        ]),

        %% Python stage
        stage(python, 'script.py', [
            flag,                    % --flag
            option='value',          % --option=value
            tree_id='123'            % --tree-id=123
        ]),

        %% Bash stage
        stage(bash, 'script.sh', [
            'arg1',
            'arg2'
        ]),

        %% Custom command
        stage(custom, 'grep "pattern" | sort', [])
    ]),

    %% INPUT/OUTPUT
    input_file('input'),      % Input file
    output_file('output')     % Output file
]).
```

### Stage Types

#### AWK Stage

```prolog
stage(awk, 'select_xml_elements.awk', [
    tag='pt:Tree',
    delimiter='\0',
    debug=1
])

% Generates:
% awk -f scripts/utils/select_xml_elements.awk -v tag="pt:Tree" -v delimiter="\0" -v debug=1 input
```

#### Python Stage

```prolog
stage(python, 'xml_to_prolog_facts.py', [
    element_type='tree',
    debug
])

% Generates:
% python3 scripts/utils/xml_to_prolog_facts.py --element-type=tree --debug
```

#### Bash Stage

```prolog
stage(bash, 'process.sh', ['arg1', 'arg2'])

% Generates:
% bash scripts/process.sh arg1 arg2
```

#### Custom Stage

```prolog
stage(custom, 'grep "pattern" | sort -u', [])

% Generates:
% grep "pattern" | sort -u
```

---

## Configuration Scenarios

### Scenario 1: Small Embedded XML

**Requirement:** Process small XML embedded in code

**Best Approach:** Python Source

```prolog
:- source(python, process_embedded, [
    python_inline("
import xml.etree.ElementTree as ET
xml_data = '''<products>...</products>'''
root = ET.fromstring(xml_data)
for product in root.findall('product'):
    print(f'product(...).')
    ")
]).
```

---

### Scenario 2: Medium File (5MB)

**Requirement:** Extract elements from 5MB file

**Best Approach:** XML Source (auto in-memory)

```prolog
:- source(xml, extract_medium, [
    file('catalog.xml'),  % 5MB < 10MB threshold
    tag('product'),
    output('products.pl')
    % Mode automatically: in_memory
]).
```

---

### Scenario 3: Large File (50MB)

**Requirement:** Extract elements from 50MB file

**Best Approach:** XML Source (auto streaming)

```prolog
:- source(xml, extract_large, [
    file('large_catalog.xml'),  % 50MB > 10MB threshold
    tag('product'),
    output('products.pl')
    % Mode automatically: streaming
]).
```

---

### Scenario 4: Very Large File (1GB) with Parallel

**Requirement:** Fast extraction from 1GB file

**Best Approach:** XML Source (parallel mode)

```prolog
:- source(xml, extract_huge, [
    file('huge_catalog.xml'),
    tag('product'),
    mode(parallel),
    workers(8),  % Use 8 cores
    output('products/')  % Output directory
]).
```

---

### Scenario 5: Custom Pipeline with Filtering

**Requirement:** Extract, filter, transform in custom way

**Best Approach:** Bash Pipeline

```prolog
:- source(bash_pipeline, custom_extract, [
    stages([
        stage(awk, 'select_xml_elements.awk', [tag='pt:.*Pearl']),
        stage(python, 'filter_by_parent_tree.py', [tree_id='2492215']),
        stage(python, 'xml_to_prolog_facts.py', [element_type='pearl']),
        stage(custom, 'grep "parent_tree" | sort -u', [])
    ]),
    input_file('pearltrees.rdf'),
    output_file('filtered_pearls.pl')
]).
```

---

## Performance Tuning

### Adjusting Streaming Threshold

**Default:** 10MB

**Increase for more in-memory processing:**
```prolog
:- source(xml, large_memory, [
    file('data.xml'),
    tag('record'),
    streaming_threshold(104857600)  % 100MB threshold
]).
```

**Decrease for earlier streaming:**
```prolog
:- source(xml, early_streaming, [
    file('data.xml'),
    tag('record'),
    streaming_threshold(1048576)  % 1MB threshold
]).
```

### Optimizing Parallel Workers

**Auto-detect CPU count:**
```prolog
% Uses default (4 workers)
:- source(xml, parallel_auto, [
    file('data.xml'),
    tag('record'),
    mode(parallel)
]).
```

**Explicit worker count:**
```prolog
% Match your CPU cores
:- source(xml, parallel_8, [
    file('data.xml'),
    tag('record'),
    mode(parallel),
    workers(8)
]).
```

**Benchmark different worker counts:**
```bash
# Test with 2, 4, 8 workers
for N in 2 4 8; do
    echo "Testing $N workers..."
    time swipl -g "source(xml, test, [file('data.xml'), mode(parallel), workers($N)])"
done
```

---

## Global Configuration

### Setting Defaults

Create `unifyweaver_config.pl`:

```prolog
% Global XML source defaults
:- dynamic xml_config/2.

% Default streaming threshold: 50MB
xml_config(streaming_threshold, 52428800).

% Default parallel workers: 8
xml_config(parallel_workers, 8).

% Default minimum partition size: 5MB
xml_config(min_partition_size, 5242880).
```

Load in your source definitions:
```prolog
:- consult('unifyweaver_config.pl').

:- source(xml, my_extractor, [
    file('data.xml'),
    tag('record')
    % Uses global defaults from config
]).
```

---

## Troubleshooting

### Issue: Auto-detection chooses wrong mode

**Solution:** Use explicit mode

```prolog
% Instead of auto
:- source(xml, my_source, [
    file('data.xml'),
    tag('record'),
    mode(streaming)  % Explicit
]).
```

### Issue: Out of memory with large file

**Solution:** Force streaming or parallel mode

```prolog
:- source(xml, large_file, [
    file('huge.xml'),
    tag('record'),
    mode(streaming)  % Force streaming (constant memory)
]).
```

### Issue: Parallel processing slower than sequential

**Possible causes:**
- File too small (overhead > benefit)
- Too many workers (contention)
- Uneven element distribution

**Solutions:**
```prolog
% Reduce workers
workers(2)

% Use streaming instead
mode(streaming)

% Increase partition size (fewer partitions)
partition_size(20971520)  % 20MB partitions
```

---

## Summary

**Three approaches, all configurable:**

1. **Python Source** - Full control, existing system
2. **XML Source** - Auto-optimization, simple API
3. **Bash Pipeline** - Custom pipelines, explicit stages

**Key configuration points:**
- Mode selection (auto/in_memory/streaming/parallel)
- Thresholds (when to switch modes)
- Workers (parallel processing)
- Element types (transformation)
- Input/Output (files, inline, stdout)

**Choose based on:**
- File size (<10MB → in-memory, >10MB → streaming, >100MB → parallel)
- Complexity (simple → XML source, complex → Python/Pipeline)
- Control (auto → XML source, explicit → Bash pipeline)
