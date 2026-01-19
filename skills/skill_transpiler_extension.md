# Skill: Transpiler Extension

Extend the UnifyWeaver transpiler using sources, bindings, and the component system to integrate foreign tools and libraries.

## When to Use

- User asks "how do I add a new data source?"
- User wants to call Python/Go/Rust stdlib functions from Prolog
- User asks about bindings or foreign function mapping
- User wants to integrate external tools into the transpilation pipeline
- User asks "how do I add support for a new file format?"
- User wants to create reusable components

## Three Extension Mechanisms

| Mechanism | Purpose | Example |
|-----------|---------|---------|
| **Sources** | Read data from external formats | JSON, CSV, SQLite, HTTP |
| **Bindings** | Map predicates to target stdlib | `len/2` → Python's `len()` |
| **Components** | Reusable registered units | Runtime plugins, validators |

## Sources: External Data Integration

Sources let you define predicates that read from external data formats. The transpiler generates target code that processes these sources.

### Declaring a Source

```prolog
:- use_module('src/unifyweaver/sources').

% Define a JSON source
:- source(json, api_response, [
    json_file('data/response.json'),
    jq_filter('.items[] | {id, name, status}'),
    output_format(tsv)
]).

% Now use api_response/3 in your code
process_items :-
    api_response(Id, Name, Status),
    format('~w: ~w (~w)~n', [Id, Name, Status]).
```

### Available Source Types

| Type | Description | Key Options |
|------|-------------|-------------|
| `json` | JSON via jq | `json_file`, `jq_filter`, `output_format` |
| `csv` | CSV/TSV files | `csv_file`, `has_header`, `delimiter` |
| `xml` | XML via xmlstarlet | `xml_file`, `xpath` |
| `yaml` | YAML files | `yaml_file`, `path` |
| `sqlite` | SQLite queries | `db_file`, `query` |
| `http` | HTTP endpoints | `url`, `method`, `headers` |
| `python` | Python functions | `module`, `function` |
| `dotnet` | .NET assemblies | `assembly`, `class` |
| `bash_pipeline` | Shell pipelines | `command` |
| `semantic` | Semantic search | `embeddings`, `query` |

### Creating a Custom Source Plugin

```prolog
% my_source.pl - Custom source plugin
:- module(my_source, []).

:- use_module('../core/dynamic_source_compiler').

% Register on load
:- initialization(
    register_source_type(my_format, my_source),
    now
).

%% source_info(-Info)
source_info(info(
    name('My Format Source'),
    version('1.0.0'),
    description('Process my custom format'),
    supported_arities([1, 2, 3])
)).

%% validate_config(+Config)
validate_config(Config) :-
    member(my_file(File), Config),
    exists_file(File).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
compile_source(Pred/Arity, Config, Options, BashCode) :-
    member(my_file(File), Config),
    format(atom(BashCode), '# Process ~w from ~w~n...', [Pred, File]).
```

## Bindings: Foreign Function Mapping

Bindings map Prolog predicates to target language functions, preserving semantic information for correct code generation.

### Core Binding Predicate

```prolog
%% binding(+Target, +PrologPred, +TargetName, +Inputs, +Outputs, +Options)

% Example: Map string_length/2 to Python's len()
binding(python, string_length/2, 'len', [string], [int], [pure]).

% Example: Map file_exists/1 to Go's os.Stat
binding(go, file_exists/1, 'os.Stat', [path], [],
        [effect(io), returns_error, import('os')]).

% Example: Map with exit status handling for bash
binding(bash, file_exists/1, 'test -f', [path], [],
        [effect(io), exit_status(0=true, _=false)]).
```

### Using Bindings in Your Code

```prolog
:- use_module('src/unifyweaver/bindings/python_bindings').

% Initialize bindings for Python target
:- init_python_bindings.

% Now the transpiler knows how to translate these predicates
process_text(Text, Len) :-
    string_length(Text, Len),   % Translates to: len(text)
    Len > 10.
```

### Creating Custom Bindings

```prolog
% In your Prolog file, declare bindings with directives
:- py_binding(my_hash/2, 'hashlib.sha256', [string], [string],
              [import('hashlib'), pure]).

% Or register programmatically
:- binding_registry:declare_binding(
    python,
    my_custom_func/3,
    'my_module.process',
    [string, int],
    [list],
    [import('my_module'), effect(io)]
).
```

### Binding Options

| Option | Description |
|--------|-------------|
| `pure` | No side effects - can be reordered, memoized |
| `effect(io)` | Performs I/O operations |
| `effect(state)` | Mutates state |
| `effect(throws)` | Can raise exceptions |
| `nondeterministic` | May produce multiple results |
| `deterministic` | Produces exactly one result |
| `import(Module)` | Required import statement |
| `returns_error` | Go-style error return |
| `exit_status(Mapping)` | Shell exit code interpretation |
| `variadic` | Variable argument count |

### Available Binding Modules

| Module | Target | Location |
|--------|--------|----------|
| `python_bindings` | Python | `src/unifyweaver/bindings/python_bindings.pl` |
| `go_bindings` | Go | `src/unifyweaver/bindings/go_bindings.pl` |
| `rust_bindings` | Rust | `src/unifyweaver/bindings/rust_bindings.pl` |
| `bash_bindings` | Bash | `src/unifyweaver/bindings/bash_bindings.pl` |
| `csharp_bindings` | C# | `src/unifyweaver/bindings/csharp_bindings.pl` |
| `powershell_bindings` | PowerShell | `src/unifyweaver/bindings/powershell_bindings.pl` |
| ... | 20+ targets | `src/unifyweaver/bindings/` |

## Component Registry: Reusable Units

The component registry provides a unified framework for managing components across categories (runtime, source, binding).

### Defining a Category

```prolog
:- use_module('src/unifyweaver/core/component_registry').

% Define a new component category
:- define_category(validator, 'Data validation components', [
    requires_compilation(false),
    singleton(false)
]).
```

### Registering a Component Type

```prolog
% Register a component type within the category
:- register_component_type(validator, json_schema, json_validator_module, [
    description('JSON Schema validator'),
    version('1.0.0')
]).
```

### Declaring Component Instances

```prolog
% Declare a specific component instance
:- declare_component(validator, user_input_validator, json_schema, [
    schema_file('schemas/user.json'),
    strict(true)
]).
```

### Using Components

```prolog
% Initialize a component
:- init_component(validator, user_input_validator).

% Invoke a component
validate_user(Input, Result) :-
    invoke_component(validator, user_input_validator, Input, Result).

% For compiled components
generate_validator(Code) :-
    compile_component(validator, user_input_validator, [], Code).
```

### Component Module Interface

When creating a component type, the module must export:

```prolog
:- module(my_component_type, [
    type_info/1,           % type_info(-Info)
    validate_config/1,     % validate_config(+Config)
    init_component/2,      % init_component(+Name, +Config)
    invoke_component/4,    % invoke_component(+Name, +Config, +Input, -Output)
    compile_component/4    % compile_component(+Name, +Config, +Options, -Code)
]).
```

## Adding a New Target Language

To add support for a completely new target language:

### 1. Create the Target Module

```prolog
% src/unifyweaver/targets/mylang_target.pl
:- module(mylang_target, [
    compile_to_mylang/3,
    mylang_template/3
]).

:- use_module('../core/template_system').
:- use_module('../core/target_registry').

% Register target
:- initialization(
    register_target(mylang, mylang_target, [
        extension('.ml'),
        comment_style('(* ... *)'),
        supports([recursion, higher_order])
    ]),
    now
).

%% compile_to_mylang(+Goal, +Options, -Code)
compile_to_mylang(Goal, Options, Code) :-
    % Your compilation logic here
    ...
```

### 2. Create Bindings for the Target

```prolog
% src/unifyweaver/bindings/mylang_bindings.pl
:- module(mylang_bindings, [init_mylang_bindings/0]).

:- use_module('../core/binding_registry').

init_mylang_bindings :-
    % Register stdlib bindings
    declare_binding(mylang, print/1, 'print', [string], [], [effect(io)]),
    declare_binding(mylang, length/2, 'List.length', [list], [int], [pure]),
    ...
```

### 3. Add Templates (Optional)

```prolog
% Templates for code generation patterns
mylang_template(function, [Name, Args, Body], Code) :-
    format(atom(Code), 'let ~w ~w = ~w', [Name, Args, Body]).

mylang_template(if_then_else, [Cond, Then, Else], Code) :-
    format(atom(Code), 'if ~w then ~w else ~w', [Cond, Then, Else]).
```

## Cross-Target Glue

For pipelines spanning multiple targets, use the glue system:

```prolog
:- use_module('src/unifyweaver/glue/shell_glue').

% Generate pipeline: AWK → Python → Rust
generate_pipeline :-
    generate_pipeline(
        [
            step(filter, awk, 'filter.awk', []),
            step(transform, python, 'transform.py', []),
            step(aggregate, rust, 'aggregate', [])
        ],
        [],
        Script
    ),
    write_file('pipeline.sh', Script).
```

## Quick Reference

### Adding a Data Source

```prolog
:- source(TYPE, NAME, [OPTIONS]).
```

### Adding a Binding

```prolog
:- declare_binding(TARGET, PRED/ARITY, TARGET_FUNC, INPUTS, OUTPUTS, OPTIONS).
```

### Registering a Component

```prolog
:- declare_component(CATEGORY, NAME, TYPE, CONFIG).
```

## Related

**Skills:**
- `skill_unifyweaver_compile.md` - Basic compilation
- `skill_json_extraction.md` - JSON source usage

**Documentation:**
- `docs/proposals/BINDING_PREDICATE_PROPOSAL.md` - Binding design
- `docs/proposals/COMPONENT_REGISTRY.md` - Component system
- `docs/guides/cross-target-glue.md` - Multi-target pipelines
- `docs/BINDING_MATRIX.md` - Binding coverage by target

**Education (in `education/` subfolder):**
- `book-07-cross-target-glue/` - Cross-target communication

**Code:**
- `src/unifyweaver/sources/` - Source plugins
- `src/unifyweaver/bindings/` - Target bindings
- `src/unifyweaver/core/binding_registry.pl` - Binding registry
- `src/unifyweaver/core/component_registry.pl` - Component registry
- `src/unifyweaver/core/dynamic_source_compiler.pl` - Source compilation
