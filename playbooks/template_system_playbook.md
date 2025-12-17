# Playbook: Template System

## Audience
This playbook guides coding agents in using UnifyWeaver's template system to generate code from named templates with placeholder substitution.

## Overview
The template system provides a flexible mechanism for generating code from templates with `{{placeholder}}` syntax. It supports built-in templates, external file loading, caching, and template composition.

Key features:
- **Named placeholder substitution** with `{{name}}` syntax
- **20+ built-in templates** for bash code generation
- **Three source strategies**: generated (hardcoded), file (external), cached (memory)
- **Template composition** for combining multiple templates
- **Transitive closure generation** for graph traversal code

## When to Use This Approach

### Use Template System When:
- Generating bash scripts with consistent structure
- Creating transitive closure/graph traversal implementations
- Need reusable code generation patterns
- Building fact lookup functions from data
- Want template composition for modular code

### Use Other Approaches When:
- Need PowerShell/.NET integration (see `powershell_binding_playbook.md`)
- Generating C# code (see `csharp_codegen_playbook.md`)
- Complex multi-language code generation


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "template_system" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use template system"


## Agent Inputs
Reference the following artifacts:
1. **Bash Executable Records** in `playbooks/examples_library/template_system_examples.md`:
   - `unifyweaver.execution.template_basic_bash` - Basic template rendering
   - `unifyweaver.execution.template_function_bash` - Function generation
   - `unifyweaver.execution.template_transitive_bash` - Transitive closure generation
   - `unifyweaver.execution.template_caching_bash` - Template caching operations
   - `unifyweaver.execution.template_file_bash` - External file loading
   - `unifyweaver.execution.template_test_bash` - Run test suite
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

### Step-by-Step Instructions

#### For Linux/macOS (bash) environment:

**Step 1: Navigate to project root**
```bash
cd /path/to/UnifyWeaver
```

**Step 2: Extract the bash script for basic template example**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.template_basic_bash" \
  playbooks/examples_library/template_system_examples.md \
  > tmp/run_template_basic.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_template_basic.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_template_basic.sh
```

**Expected Output**:
```
=== Template System Basic Demo ===

1. Direct Template Rendering:
   Template: "Hello {{name}}!"
   Values: [name=World]
   Result: Hello World!

2. Multiple Placeholders:
   Result: Hello, Alice! Today is Monday.

3. Named Template (bash_header):
   Result:
#!/bin/bash
# My Generated Script

4. Load Template Only:
   function template:

{{name}}() {
{{body}}
}

Success: Basic template operations complete.
```

### Alternative: Run Function Generation Example

**Step 2b: Extract and run function generation example**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.template_function_bash" \
  playbooks/examples_library/template_system_examples.md \
  > tmp/run_template_function.sh

chmod +x tmp/run_template_function.sh
bash tmp/run_template_function.sh
```

This generates bash functions using the `function` and `stream_check` templates.

### Alternative: Run Transitive Closure Generation

**Step 2c: Generate graph traversal code**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.template_transitive_bash" \
  playbooks/examples_library/template_system_examples.md \
  > tmp/run_template_transitive.sh

chmod +x tmp/run_template_transitive.sh
bash tmp/run_template_transitive.sh
```

This generates complete BFS-based transitive closure implementation for graph traversal.

### Alternative: Run Built-in Test Suite

**Step 2d: Run the template system's test suite**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.template_test_bash" \
  playbooks/examples_library/template_system_examples.md \
  > tmp/run_template_tests.sh

chmod +x tmp/run_template_tests.sh
bash tmp/run_template_tests.sh
```

## Template System Architecture

### Core Components

```
src/unifyweaver/core/
└── template_system.pl    # Core template processing (666 lines)
    - Placeholder substitution engine
    - Named template loading
    - Template composition
    - Caching system
    - Built-in bash templates

templates/
├── simple_function.tmpl.sh              # Example external template
└── dotnet_source_external_compile.tmpl.ps1  # PowerShell template
```

### Core API Predicates

| Predicate | Description |
|-----------|-------------|
| `render_template/3` | Render template string with dictionary |
| `render_named_template/3,4` | Load and render template by name |
| `load_template/2,3` | Load template without rendering |
| `compose_templates/3` | Compose multiple templates |
| `cache_template/2` | Cache template in memory |
| `clear_template_cache/0,1` | Clear cached templates |
| `generate_transitive_closure/4` | Generate graph traversal code |
| `template/2` | Define/access built-in templates |
| `test_template_system/0` | Run built-in test suite |

### Usage Examples

**Direct Template Rendering:**
```prolog
render_template('Hello {{name}}!', [name='World'], Result).
% Result = "Hello World!"
```

**Named Template with Options:**
```prolog
render_named_template(bash_header, [description='My Script'], Result).
% Result = "#!/bin/bash\n# My Script\n"
```

**Template Composition:**
```prolog
compose_templates([bash_header, function], [
    description='Example',
    name='my_func',
    body='    echo "Hello"'
], Result).
```

**Transitive Closure Generation:**
```prolog
generate_transitive_closure(ancestor, parent, [], Code).
% Generates complete BFS-based graph traversal code
```

### Built-in Templates

| Template Name | Purpose |
|---------------|---------|
| `bash_header` | Standard bash script header with description |
| `function` | Bash function definition wrapper |
| `stream_check` | Check if stream/function exists |
| `bfs_init` | BFS initialization with queue setup |
| `bfs_loop` | BFS traversal loop implementation |
| `all_nodes` | Complete node finder function |
| `check_function` | Relationship check function |
| `stream_wrapper` | Stream iteration wrapper |
| `dedup_wrapper` | Deduplication wrapper |
| `xml_awk_field_extraction` | XML field extraction with AWK |
| `facts/array_unary` | Unary fact array declaration |
| `facts/array_binary` | Binary fact associative array |
| `facts/lookup_unary` | Unary fact lookup function |
| `facts/lookup_binary` | Binary fact lookup function |
| `facts/stream_unary` | Unary fact stream function |
| `facts/stream_binary` | Binary fact stream function |
| `facts/reverse_stream_binary` | Reverse binary fact stream |

### Source Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `generated` | Built-in hardcoded templates | Default, always available |
| `file` | External `.tmpl.sh` files | Custom/project-specific templates |
| `cached` | In-memory cached templates | Runtime-defined templates |

**Configuration Example:**
```prolog
% Use file templates first, then generated
render_named_template(my_template, Dict, [source_order([file, generated])], Result).
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `source_order` | `[generated]` | Priority order for template sources |
| `template_dir` | `templates` | Directory for external templates |
| `cache_dir` | `templates/cache` | Directory for cached templates |
| `template_extension` | `.tmpl.sh` | File extension for template files |
| `auto_cache` | `false` | Automatically cache loaded templates |

## Expected Outcome
- Successful execution demonstrates template system functionality
- Generated code is displayed with placeholder substitution complete
- Template metadata and configuration options are shown
- Exit code 0

## Comparison with Other Approaches

| Feature | Template System | Binding System | C# Codegen |
|---------|-----------------|----------------|------------|
| Primary Target | Bash | PowerShell | C# |
| Placeholder Syntax | `{{name}}` | predicate args | DCG rules |
| Built-in Templates | 20+ | 52 bindings | Limited |
| Graph Algorithms | BFS generation | N/A | N/A |
| Effect Tracking | N/A | Built-in | Manual |
| Composition | Yes | Limited | Yes |

## Citations
[1] playbooks/examples_library/template_system_examples.md
[2] src/unifyweaver/core/template_system.pl
[3] templates/simple_function.tmpl.sh
[4] templates/dotnet_source_external_compile.tmpl.ps1
