---
file_type: UnifyWeaver Example Library
---
# Template System Examples for UnifyWeaver

## `unifyweaver.execution.template_basic_bash`

> [!example-record]
> id: unifyweaver.execution.template_basic_bash
> name: Template System Basic Example (Bash)
> platform: bash

This record demonstrates basic template rendering using `render_template/3` and `render_named_template/3` for placeholder substitution.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/template_basic"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/template_basic.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).

:- use_module(library(template_system)).

main :-
    format('=== Template System Basic Demo ===~n~n'),

    % 1. Direct template rendering with render_template/3
    format('1. Direct Template Rendering:~n'),
    render_template('Hello {{name}}!', [name='World'], R1),
    format('   Template: "Hello {{name}}!"~n'),
    format('   Values: [name=World]~n'),
    format('   Result: ~w~n~n', [R1]),

    % 2. Multiple placeholder substitution
    format('2. Multiple Placeholders:~n'),
    render_template('{{greeting}}, {{name}}! Today is {{day}}.',
                   [greeting='Hello', name='Alice', day='Monday'], R2),
    format('   Result: ~w~n~n', [R2]),

    % 3. Load and render named template (bash_header)
    format('3. Named Template (bash_header):~n'),
    render_named_template(bash_header, [description='My Generated Script'], R3),
    format('   Result:~n~w~n', [R3]),

    % 4. Load template without rendering
    format('4. Load Template Only:~n'),
    load_template(function, Template4),
    format('   function template:~n~w~n~n', [Template4]),

    format('Success: Basic template operations complete.~n').

:- main.
:- halt.
PROLOG

echo "Running basic template example..."
swipl -l $TMP_DIR/template_basic.pl
```

## `unifyweaver.execution.template_function_bash`

> [!example-record]
> id: unifyweaver.execution.template_function_bash
> name: Template System Function Generation (Bash)
> platform: bash

This record demonstrates generating bash function code using the `function` template and composing templates.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/template_function"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/template_function.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).

:- use_module(library(template_system)).

main :-
    format('=== Template Function Generation Demo ===~n~n'),

    % Generate a simple bash function
    format('1. Generate Bash Function:~n'),
    render_named_template(function, [
        name='greet_user',
        body='    local user="$1"\n    echo "Hello, $user!"'
    ], FuncCode),
    format('~w~n~n', [FuncCode]),

    % Generate function with stream check
    format('2. Generate Stream Check Function:~n'),
    render_named_template(stream_check, [base='parent'], StreamCheck),
    format('~w~n~n', [StreamCheck]),

    % Compose multiple templates
    format('3. Compose Multiple Templates:~n'),
    compose_templates([bash_header, function], [
        description='Composed Script Example',
        name='my_function',
        body='    echo "Function body here"'
    ], Composed),
    format('~w~n', [Composed]),

    format('Success: Function generation complete.~n').

:- main.
:- halt.
PROLOG

echo "Running function generation example..."
swipl -l $TMP_DIR/template_function.pl
```

## `unifyweaver.execution.template_transitive_bash`

> [!example-record]
> id: unifyweaver.execution.template_transitive_bash
> name: Template System Transitive Closure (Bash)
> platform: bash

This record demonstrates generating transitive closure code using `generate_transitive_closure/4`.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/template_transitive"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/template_transitive.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).

:- use_module(library(template_system)).

main :-
    format('=== Template Transitive Closure Demo ===~n~n'),

    format('Generating transitive closure code for "ancestor" based on "parent"...~n~n'),

    % Generate complete transitive closure implementation
    generate_transitive_closure(ancestor, parent, [], Code),

    format('Generated Code:~n'),
    format('~w~n', [Code]),

    format('~nSuccess: Transitive closure generation complete.~n'),
    format('~nThe generated code includes:~n'),
    format('  - ancestor_all() - finds all descendants via BFS~n'),
    format('  - ancestor_check() - checks specific relationship~n'),
    format('  - ancestor() - main entry point~n').

:- main.
:- halt.
PROLOG

echo "Running transitive closure generation example..."
swipl -l $TMP_DIR/template_transitive.pl
```

## `unifyweaver.execution.template_caching_bash`

> [!example-record]
> id: unifyweaver.execution.template_caching_bash
> name: Template System Caching (Bash)
> platform: bash

This record demonstrates template caching operations with `cache_template/2` and `clear_template_cache/0,1`.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/template_caching"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/template_caching.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).

:- use_module(library(template_system)).

main :-
    format('=== Template Caching Demo ===~n~n'),

    % Cache a custom template
    format('1. Cache Custom Template:~n'),
    cache_template(my_custom_template, '#!/bin/bash\n# {{description}}\necho "Running {{name}}"'),
    format('   Cached "my_custom_template"~n~n'),

    % Render from cache
    format('2. Render from Cache:~n'),
    render_named_template(my_custom_template, [
        description='Custom cached script',
        name='cached_example'
    ], [source_order([cached])], Result),
    format('~w~n~n', [Result]),

    % Check configuration options
    format('3. Configuration Options:~n'),
    template_config_default(DefaultConfig),
    format('   Default config: ~w~n~n', [DefaultConfig]),

    % Clean up cache
    format('4. Clear Cached Template:~n'),
    clear_template_cache(my_custom_template),
    format('   Cleared "my_custom_template" from cache~n~n'),

    format('Success: Caching operations complete.~n').

:- main.
:- halt.
PROLOG

echo "Running template caching example..."
swipl -l $TMP_DIR/template_caching.pl
```

## `unifyweaver.execution.template_file_bash`

> [!example-record]
> id: unifyweaver.execution.template_file_bash
> name: Template System File Loading (Bash)
> platform: bash

This record demonstrates loading templates from external files using the `file` source strategy.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/template_file"
mkdir -p $TMP_DIR

# Write the Prolog script
cat > $TMP_DIR/template_file.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).

:- use_module(library(template_system)).

main :-
    format('=== Template File Loading Demo ===~n~n'),

    % Load template from file (simple_function.tmpl.sh exists in templates/)
    format('1. Load External Template File:~n'),
    (   load_template(simple_function, [source_order([file])], FileTemplate)
    ->  format('   Loaded simple_function.tmpl.sh:~n~w~n~n', [FileTemplate])
    ;   format('   Note: simple_function.tmpl.sh not found, showing generated templates~n~n')
    ),

    % Show available source strategies
    format('2. Source Strategy Priority:~n'),
    format('   - generated: Built-in hardcoded templates~n'),
    format('   - file: External .tmpl.sh files~n'),
    format('   - cached: In-memory cached templates~n~n'),

    % Configure custom source order
    format('3. Custom Source Order:~n'),
    format('   To prioritize file loading: [source_order([file, generated, cached])]~n'),
    format('   To use only generated: [source_order([generated])]~n~n'),

    % List built-in template names
    format('4. Built-in Templates Available:~n'),
    findall(Name, template(Name, _), Names),
    forall(member(N, Names), format('   - ~w~n', [N])),

    format('~nSuccess: File loading demo complete.~n').

:- main.
:- halt.
PROLOG

echo "Running file loading example..."
swipl -l $TMP_DIR/template_file.pl
```

## `unifyweaver.execution.template_test_bash`

> [!example-record]
> id: unifyweaver.execution.template_test_bash
> name: Template System Test Suite (Bash)
> platform: bash

This record runs a custom test suite for the template system that covers the main functionality.

```bash
#!/bin/bash
set -e

# Create temporary directory
TMP_DIR="tmp/template_test"
mkdir -p $TMP_DIR

# Write the Prolog test runner
cat > $TMP_DIR/run_template_tests.pl << 'PROLOG'
:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).

:- use_module(library(template_system)).

main :-
    format('~n========================================~n'),
    format('   UnifyWeaver Template System Tests~n'),
    format('========================================~n~n'),

    % Test 1: Simple substitution
    write('Test 1 - Simple substitution: '),
    render_template('Hello {{name}}!', [name='World'], R1),
    (sub_string(R1, _, _, _, 'Hello World!') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R1]), fail)),

    % Test 2: Multiple substitutions
    write('Test 2 - Multiple substitutions: '),
    render_template('{{greeting}} {{name}}', [greeting='Hello', name='Alice'], R2),
    (sub_string(R2, _, _, _, 'Hello Alice') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R2]), fail)),

    % Test 3: Named template loading
    write('Test 3 - Load named template: '),
    load_template(bash_header, Template3),
    (sub_string(Template3, _, _, _, '#!/bin/bash') -> writeln('PASS') ; (format('FAIL: got ~w~n', [Template3]), fail)),

    % Test 4: Render named template
    write('Test 4 - Render named template: '),
    render_named_template(bash_header, [description='Test Script'], R4),
    (sub_string(R4, _, _, _, '# Test Script') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R4]), fail)),

    % Test 5: Template caching and rendering from cache
    write('Test 5 - Template caching: '),
    cache_template(test_cached, 'Cached {{value}}'),
    render_named_template(test_cached, [value='42'], [source_order([cached])], R5),
    (sub_string(R5, _, _, _, 'Cached 42') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R5]), fail)),
    clear_template_cache(test_cached),

    % Test 6: Function template
    write('Test 6 - Function template: '),
    load_template(function, FuncTemplate),
    (sub_string(FuncTemplate, _, _, _, '{{name}}()') -> writeln('PASS') ; (format('FAIL: got ~w~n', [FuncTemplate]), fail)),

    % Test 7: Template composition
    write('Test 7 - Template composition: '),
    compose_templates([bash_header], [description='Composed'], Composed),
    (sub_string(Composed, _, _, _, '# Composed') -> writeln('PASS') ; (format('FAIL: got ~w~n', [Composed]), fail)),

    format('~n========================================~n'),
    format('   All Template Tests Complete (7/7)~n'),
    format('========================================~n').

:- main.
:- halt.
PROLOG

echo "Running template system tests..."
swipl -l $TMP_DIR/run_template_tests.pl
```

## Additional Notes

### Template System Architecture

The template system consists of the following components:

1. **`template_system.pl`** - Core template processing (666 lines)
   - Placeholder substitution (`{{name}}` syntax)
   - Named template loading
   - Template composition
   - Caching system
   - Built-in bash code generation templates

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
| `template_config/2` | Get configuration option |
| `set_template_config/2` | Set configuration option |

### Built-in Templates

| Template Name | Purpose |
|---------------|---------|
| `bash_header` | Standard bash script header |
| `function` | Bash function definition |
| `stream_check` | Stream/function existence check |
| `bfs_init` | BFS initialization code |
| `bfs_loop` | BFS traversal loop |
| `all_nodes` | Complete node finder |
| `check_function` | Relationship check function |
| `stream_wrapper` | Stream iteration wrapper |
| `dedup_wrapper` | Deduplication wrapper |
| `xml_awk_field_extraction` | XML field extraction |
| `facts/array_unary` | Unary fact array |
| `facts/array_binary` | Binary fact associative array |
| `facts/lookup_unary` | Unary fact lookup |
| `facts/lookup_binary` | Binary fact lookup |
| `facts/stream_unary` | Unary fact stream |
| `facts/stream_binary` | Binary fact stream |

### Source Strategies

| Strategy | Description |
|----------|-------------|
| `generated` | Built-in hardcoded templates (default) |
| `file` | External template files from `templates/` directory |
| `cached` | In-memory cached templates |

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `source_order` | `[generated]` | Priority order for template sources |
| `template_dir` | `templates` | Directory for external templates |
| `cache_dir` | `templates/cache` | Directory for cached templates |
| `template_extension` | `.tmpl.sh` | File extension for template files |
| `auto_cache` | `false` | Automatically cache loaded templates |
