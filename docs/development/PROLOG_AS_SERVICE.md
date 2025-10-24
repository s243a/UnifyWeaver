# Prolog as a Service in Bash Scripts

**Status:** Specification for future feature
**Version:** 0.1.0 (Draft)
**Date:** October 23, 2025
**Maturity:** Mixed - Core pattern is production-ready; Discussion section is speculative/research-oriented

---

⚠️ **Document Structure:**
- **Sections marked "Core Pattern" through "Examples Repository"** - Production-ready, tested patterns
- **Section marked "Discussion"** - Speculative research directions and theoretical implications
- **Section marked "Future Work"** - Planned but not yet implemented

---

## Overview

This document specifies a pattern for embedding SWI-Prolog as a service within bash scripts, enabling declarative logic programming to be seamlessly integrated into shell-based workflows. This pattern emerged from the UnifyWeaver v0.0.2 testing infrastructure and has been **validated in production use**.

## Motivation

Traditional bash scripting is:
- **Imperative** - Focuses on "how" rather than "what"
- **Limited** - Lacks pattern matching, backtracking, constraint solving
- **Verbose** - Complex logic requires many lines of procedural code

Prolog offers:
- **Declarative** - Express relationships and constraints directly
- **Powerful** - Pattern matching, unification, backtracking, constraint solving
- **Concise** - Complex queries in a few lines

Combining them provides:
- **Best of both worlds** - Shell orchestration + logic programming
- **Unified workflows** - No need to split into separate scripts
- **Data transformation** - Process bash data with Prolog logic
- **Code generation** - Generate bash from Prolog specifications

## Discussion: Theoretical and Practical Implications

---

⚠️ **SPECULATIVE CONTENT WARNING:**

This section explores theoretical implications and future research directions. While grounded in the production-ready pattern described above, the ideas presented here are:

- **Exploratory** - Not yet implemented or fully validated
- **Research-oriented** - Suitable for academic investigation or experimental prototypes
- **Forward-looking** - May require significant development effort
- **Potentially impractical** - Some ideas may prove unfeasible or unnecessary

**For production use, refer to the "Core Pattern" section and tested "Use Cases" only.**

Readers interested in these research directions are encouraged to:
- Open a GitHub issue for discussion
- Propose implementation approaches
- Share related research or prior art
- Collaborate on experimental prototypes

---

### Higher-Order Pattern Composition

This pattern enables **meta-level programming** where multiple features are composed into a single, cohesive script:

1. **Shell orchestration** - File I/O, process management, environment interaction
2. **Logic programming** - Pattern matching, constraint solving, declarative queries
3. **Code generation** - Dynamic creation of executable code
4. **Data transformation** - Streaming pipelines with declarative logic

Traditional approaches require **separate scripts** for each concern:
```bash
# Traditional: Four separate scripts
./extract_data.sh | ./transform.pl | ./validate.py | ./load.sh
```

With Prolog-as-Service, **one script handles everything**:
```bash
# Unified: Single script with embedded logic
./unified_pipeline.sh
  # Contains: bash orchestration + embedded Prolog services
  # Each service handles different aspects declaratively
```

### Bash Scripts as Compilers

This pattern fundamentally transforms bash scripts from **executors** to **compilers**. The generated bash code can:

1. **Accept specifications** (as bash variables)
2. **Generate Prolog code** dynamically
3. **Compile to executable output** (bash functions, queries, transformations)
4. **Execute the compiled result** immediately

#### Example: Bash Variables Define Prolog Code

```bash
#!/bin/bash
# Configuration via bash variables
DATA_SOURCE="users.csv"
FILTER_COLUMN="age"
FILTER_VALUE="25"
OUTPUT_FORMAT="json"

# Generate Prolog code from bash variables
cat << PROLOG | swipl -q -g "consult(user), compile_and_run, halt" -t halt
:- use_module(library(csv)).
:- use_module(library(http/json)).

compile_and_run :-
    % Bash variables injected as Prolog atoms
    DataSource = '$DATA_SOURCE',
    FilterCol = '$FILTER_COLUMN',
    FilterVal = '$FILTER_VALUE',
    OutputFmt = '$OUTPUT_FORMAT',

    % Compile query from configuration
    compile_filter_query(DataSource, FilterCol, FilterVal, OutputFmt, Query),

    % Execute compiled query
    call(Query).

compile_filter_query(File, Column, Value, Format, Query) :-
    % Generate appropriate query based on format
    (   Format = json
    ->  Query = (load_csv(File), filter_and_output_json(Column, Value))
    ;   Format = tsv
    ->  Query = (load_csv(File), filter_and_output_tsv(Column, Value))
    ;   throw(error(unknown_format(Format)))
    ).
PROLOG
```

This is **compilation at runtime**:
- Bash variables → Prolog specification
- Prolog specification → Compiled query
- Compiled query → Executed immediately

### Multi-Stage Compilation Pipeline

The pattern enables **nested compilation**:

```bash
# Stage 1: Bash compiles configuration to Prolog
CONFIG_FILE="pipeline.conf"

# Stage 2: Prolog compiles data source to bash code
cat << PROLOG | swipl -q -g "consult(user), stage2, halt"
:- use_module('src/unifyweaver/sources').

stage2 :-
    % Read config (from bash environment)
    Config = '$(<$CONFIG_FILE)',

    % Compile to bash code
    compile_source_from_config(Config, BashCode),

    % Output bash code
    write(BashCode).
PROLOG > generated_pipeline.sh

# Stage 3: Execute the generated bash code
chmod +x generated_pipeline.sh
./generated_pipeline.sh
```

This is **three-level compilation**:
1. **Bash** reads config → Prolog
2. **Prolog** compiles config → Bash
3. **Bash** executes generated code

#### Lazy Compilation: Infinite Towers

More profoundly, each compilation stage can **defer to another stage**, creating an **infinite compilation tower** where compilation never needs to finish:

```bash
# Stage N: Generates stage N+1 compiler
compile_stage_N() {
    cat << PROLOG | swipl -q -g "consult(user), generate_next_stage, halt"
    generate_next_stage :-
        % Generate code for the NEXT compiler stage
        % which itself will generate the NEXT stage
        % which will generate...
        current_stage(N),
        NextStage is N + 1,
        generate_compiler_for_stage(NextStage, Code),
        write(Code).
    PROLOG
}
```

**Key insight:** Compilation is **lazy** - each stage only compiles when needed, and can defer to the next stage:

- **Stage 0:** User specification (declarative)
- **Stage 1:** Compiles spec → intermediate representation (still declarative)
- **Stage 2:** Compiles IR → optimized IR (more concrete)
- **Stage 3:** Compiles IR → executable code (when needed)
- **Stage N:** Can always defer to Stage N+1 if more information needed

This enables:
- **Just-in-time** specialization at any stage
- **Partial evaluation** - compile what you know, defer the rest
- **Adaptive compilation** - later stages see runtime data
- **Never-ending optimization** - each execution can trigger deeper compilation

Example of infinite deferral:
```bash
#!/bin/bash
# Stage 0: User's intent
USER_INTENT="process all CSV files efficiently"

# Stage 1: Interpret intent → abstract plan
ABSTRACT_PLAN=$(interpret_intent "$USER_INTENT")

# Stage 2: Generate concrete plan from abstract plan (lazy)
generate_concrete_plan() {
    # Only compile this stage when we know the data format
    if [ -z "$DATA_FORMAT_KNOWN" ]; then
        # Defer to stage 3 - return a compiler that will finish later
        echo "compile_when_data_arrives"
    else
        # We have enough info, generate concrete code
        actual_compile "$ABSTRACT_PLAN" "$DATA_FORMAT"
    fi
}

# Stage 3: Execute (or defer again if needed)
execute() {
    # Might trigger stage 2, which might defer to stage 4...
    eval "$(generate_concrete_plan)"
}
```

**Why this matters:** Traditional compilers must finish before execution. **Lazy compilation** means:
- Compilation interleaves with execution
- Each stage compiles "just enough"
- Optimization can continue indefinitely
- System adapts to runtime conditions

This is a form of **metacircular evaluation** where the compilation process itself is compiled by the next stage.

#### LLM-Assisted Compilation Stages

Each compilation stage could incorporate **LLM (Large Language Model) reasoning**:

```bash
# Stage N: Human intent → LLM → Formal specification
compile_with_llm() {
    USER_REQUEST="$1"

    # LLM interprets natural language → Prolog spec
    PROLOG_SPEC=$(cat << PROMPT | llm_query
    Convert this request to a Prolog data pipeline specification:
    "$USER_REQUEST"

    Output format: Prolog facts using data_source/3 and transformation/3
    PROMPT
    )

    # Stage N+1: Prolog compiles spec → Bash
    echo "$PROLOG_SPEC" | cat << 'PROLOG' | swipl -q -g "consult(user), compile, halt"
    compile :-
        read_spec_from_stdin(Spec),
        compile_to_bash(Spec, BashCode),
        write(BashCode).
    PROLOG
}

# Example usage
compile_with_llm "Extract active users from CSV, filter by age > 25, output as JSON"
```

**Intelligent nested compilation** enables:
- **Natural language** at top level (human intent)
- **LLM reasoning** to bridge natural language → formal spec
- **Prolog logic** for correctness and optimization
- **Bash execution** for actual work
- **Feedback loop** - execution results inform next LLM stage

Each stage can be **as intelligent as needed**:
- **Stage 0:** Natural language (human)
- **Stage 1:** LLM interprets intent → draft specification
- **Stage 2:** Prolog validates/optimizes specification
- **Stage 3:** Generate efficient bash code
- **Stage 4:** Runtime profiling suggests improvements to Stage 1 LLM prompt
- **Stage 5:** LLM generates optimized specification based on profiling data
- **Stage N:** ...continues indefinitely as system learns

This creates a **self-improving compilation tower** where each execution makes the system smarter.

**Note:** This is highly speculative - integrating LLMs into compilation pipelines raises questions about:
- Determinism and reproducibility
- Cost and latency
- Correctness guarantees
- When to use AI vs traditional compilation

Open a GitHub issue to discuss!

### Template Meta-Programming

The pattern supports **template instantiation** where Prolog generates code based on templates:

```bash
#!/bin/bash
# Define template parameters
ENTITY="User"
FIELDS="name:string,age:integer,email:string"
VALIDATIONS="age > 0, email contains @"

# Generate validator code from template
cat << PROLOG | swipl -q -g "consult(user), generate_validator, halt"
generate_validator :-
    Entity = '$ENTITY',
    parse_fields('$FIELDS', FieldList),
    parse_validations('$VALIDATIONS', ValidationList),

    % Generate bash validator function
    format('validate_~w() {~n', [Entity]),
    format('  local json_input="\$1"~n'),
    generate_field_extractors(FieldList),
    generate_validation_checks(ValidationList),
    format('  echo "Valid ~w"~n', [Entity]),
    format('}~n').

% ... template expansion logic ...
PROLOG > validator.sh

# Now we have a generated validator
source validator.sh
validate_User '{"name": "Alice", "age": 25, "email": "alice@example.com"}'
```

### Self-Modifying Scripts

Scripts can **analyze and modify themselves**:

```bash
#!/bin/bash
# This script analyzes its own performance and optimizes itself

run_with_profiling() {
    # Execute with timing
    START=$(date +%s%N)
    "$@"
    END=$(date +%s%N)
    ELAPSED=$((($END - $START) / 1000000))
    echo "$ELAPSED"
}

# Run initial version
TIME1=$(run_with_profiling ./data_pipeline.sh)

# Use Prolog to analyze and optimize
cat << PROLOG | swipl -q -g "consult(user), optimize, halt"
optimize :-
    % Read current script
    read_file_to_string('data_pipeline.sh', Script, []),

    % Analyze bottlenecks (from profiling data)
    Timing = $TIME1,

    % Generate optimized version
    (   Timing > 1000  % > 1 second
    ->  optimize_for_speed(Script, Optimized)
    ;   Optimized = Script
    ),

    % Write optimized version
    open('data_pipeline.optimized.sh', write, Out),
    write(Out, Optimized),
    close(Out).
PROLOG

# Use optimized version if generated
[ -f data_pipeline.optimized.sh ] && mv data_pipeline.optimized.sh data_pipeline.sh
```

### Declarative Feature Composition

Multiple features can be **declaratively composed**:

```bash
#!/bin/bash
# Feature flags via environment
ENABLE_CACHING=${ENABLE_CACHING:-true}
ENABLE_VALIDATION=${ENABLE_VALIDATION:-true}
ENABLE_LOGGING=${ENABLE_LOGGING:-false}

# Generate pipeline with selected features
cat << PROLOG | swipl -q -g "consult(user), compose_pipeline, halt"
compose_pipeline :-
    % Read feature flags from environment
    Caching = '$ENABLE_CACHING',
    Validation = '$ENABLE_VALIDATION',
    Logging = '$ENABLE_LOGGING',

    % Compose pipeline based on flags
    pipeline_features(Caching, Validation, Logging, Features),

    % Generate code with selected features
    maplist(generate_feature_code, Features).

pipeline_features(true, true, true, [cache, validate, log, process]).
pipeline_features(true, true, false, [cache, validate, process]).
pipeline_features(true, false, false, [cache, process]).
pipeline_features(false, false, false, [process]).

generate_feature_code(cache) :-
    format('setup_cache() { mkdir -p /tmp/cache; }~n').
generate_feature_code(validate) :-
    format('validate() { jq empty < "$1" 2>/dev/null; }~n').
generate_feature_code(log) :-
    format('log() { echo "$(date): $*" >> pipeline.log; }~n').
generate_feature_code(process) :-
    format('process() { cat "$1" | jq .data; }~n').
PROLOG > pipeline.sh

source pipeline.sh
# Only functions for enabled features are generated
```

### Homoiconicity in Shell Scripts

This pattern brings **homoiconicity** (code as data) to bash:

```bash
# Bash code can be treated as Prolog data
BASH_FUNCTION='filter() { grep "$1" < "$2"; }'

# Prolog analyzes bash code as data
cat << PROLOG | swipl -q -g "consult(user), analyze, halt"
analyze :-
    BashCode = '$BASH_FUNCTION',

    % Parse bash code (simplified)
    (   sub_string(BashCode, _, _, _, 'grep')
    ->  format('Uses grep - can be optimized with ripgrep~n')
    ;   true
    ),

    % Generate optimized version
    substitute(BashCode, 'grep', 'rg', Optimized),
    format('~w~n', [Optimized]).
PROLOG
```

### Implications for UnifyWeaver

This pattern has profound implications for UnifyWeaver's architecture:

#### 1. Generated Scripts Become Compilers

Current UnifyWeaver:
```
Prolog source → UnifyWeaver compiler → Bash script (executor)
```

With Prolog-as-Service:
```
Prolog source → UnifyWeaver compiler → Bash script (compiler+executor)
                                           ↓
                                    Embedded Prolog service
                                           ↓
                                    Runtime compilation
```

#### 2. Two-Level Compilation Strategy

**Compile-time** (UnifyWeaver generates bash):
- Static optimizations
- Fixed pipeline structure
- Type checking

**Runtime** (Bash invokes Prolog service):
- Dynamic data sources
- Adaptive algorithms
- User configuration

#### 3. Gradual Typing and Refinement

```bash
# Generated script with refinement capability
#!/bin/bash
# Generated by UnifyWeaver

process_data() {
    local input="$1"

    # First pass: Use Prolog to infer data structure
    SCHEMA=$(echo "$input" | cat << 'PROLOG' | swipl -q ...
    infer_schema :-
        read_json_stream(Input),
        analyze_structure(Input, Schema),
        write(Schema).
    PROLOG
    )

    # Second pass: Generate optimized processor for this schema
    cat << PROLOG | swipl -q ...
    generate_optimized_processor('$SCHEMA') :-
        % Generate bash code optimized for this specific schema
        ...
    PROLOG
}
```

#### 4. Self-Optimizing Pipelines

```bash
# Pipeline that learns from execution
run_pipeline() {
    # Collect execution statistics
    STATS=$(run_with_stats ./pipeline)

    # Use Prolog to optimize based on stats
    cat << PROLOG | swipl -q -g "consult(user), optimize('$STATS'), halt"
    optimize(Stats) :-
        parse_stats(Stats, Metrics),
        identify_bottlenecks(Metrics, Bottlenecks),
        generate_optimizations(Bottlenecks, Patches),
        apply_patches(Patches, './pipeline').
    PROLOG
}
```

### Philosophical Implications

#### Blurring Language Boundaries

This pattern dissolves the traditional separation between:
- **Shell scripting** (orchestration)
- **Logic programming** (specification)
- **Compilation** (transformation)

They become **different modes** of the same unified system.

#### Declarative Infrastructure

Instead of imperative scripts, we can write **declarative specifications**:

```bash
#!/bin/bash
# Specification (not implementation)
DESIRED_STATE='
data_source(csv, users, "users.csv").
transformation(filter_active, users, active_users).
validation(email_format, active_users).
output(json, active_users, "output.json").
'

# Prolog compiles specification to implementation
echo "$DESIRED_STATE" | cat << 'PROLOG' | swipl -q -g "consult(user), compile_spec, halt"
compile_spec :-
    % Read specification from stdin
    read_spec(Spec),

    % Generate bash implementation
    compile_to_bash(Spec, BashCode),

    % Execute generated code
    write_and_execute(BashCode).
PROLOG
```

#### Staged Computation

The pattern naturally supports **multi-stage computation**:

- **Stage 0:** Configuration (environment variables, config files)
- **Stage 1:** Bash reads config, generates Prolog code
- **Stage 2:** Prolog compiles config to bash code
- **Stage 3:** Generated bash code executes

Each stage can be **cached, optimized, or skipped** depending on what changed.

### Comparison to Other Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Separate Prolog Scripts** | Clean separation | Inter-process communication overhead, multiple files |
| **Embedded Interpreters** | Full integration | Complex C bindings, portability issues |
| **Code Generation** | Static optimization | No runtime flexibility, regeneration required |
| **Prolog-as-Service** | Simple, flexible, portable | Startup cost per invocation |

**Prolog-as-Service wins on:**
- Simplicity (no C bindings)
- Portability (pure bash + Prolog)
- Flexibility (runtime compilation)
- Composability (easy to combine features)

### Future Research Directions

1. **Optimal staging** - When to compile at which stage?
2. **Partial evaluation** - Pre-compute what can be known statically
3. **JIT compilation** - Cache frequently-used Prolog services
4. **Type-driven generation** - Use Prolog's type system to generate safer bash
5. **Proof-carrying code** - Prolog proves properties, bash enforces them
6. **Genetic programming** - Self-evolving scripts that optimize through evolutionary algorithms
   - Population of script variations
   - Fitness function based on performance metrics
   - Crossover and mutation of bash/Prolog code
   - Automated discovery of optimal implementations
   - **Note:** This is highly speculative - open a GitHub issue to discuss!

## Core Pattern: consult(user) with Heredoc

### Basic Structure

```bash
cat << 'PROLOG' | swipl -q -g "consult(user), GOAL, halt" -t halt
:- use_module(library(lists)).
:- use_module(library(clpfd)).

% Your Prolog code here
solve(X, Y, Z) :-
    X #< Y,
    Y #< Z,
    [X, Y, Z] ins 1..10,
    label([X, Y, Z]).

% Entry point
main :-
    solve(X, Y, Z),
    format('~w ~w ~w~n', [X, Y, Z]),
    fail.
main.
PROLOG
```

### How It Works

1. **Heredoc** - `<< 'PROLOG'` creates a literal string (no variable expansion)
2. **Pipe to stdin** - Prolog code is sent to SWI-Prolog's stdin
3. **consult(user)** - Loads code from stdin as if from a file
4. **Execute goal** - Runs the specified goal (e.g., `main`)
5. **Clean exit** - `halt` terminates Prolog after execution

### Why Single Quotes Matter

```bash
# WRONG - bash expands variables
cat << PROLOG
  solve(X) :- X = $HOME.  # $HOME expanded by bash!
PROLOG

# RIGHT - literal string, Prolog variables preserved
cat << 'PROLOG'
  solve(X) :- X = $HOME.  # Prolog variable $HOME
PROLOG
```

## Use Cases

### 1. Data Source Compilation

**Scenario:** Compile Prolog data source definitions to bash scripts

```bash
# Define data source in Prolog, compile to bash
cat << 'PROLOG' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/json_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- source(json, users, [
    json_file('/tmp/users.json'),
    jq_filter('.users[] | [.id, .name] | @tsv'),
    raw_output(true)
]).

test :-
    compile_dynamic_source(users/2, [], BashCode),
    write_and_execute_bash(BashCode, '', Output),
    format('~w', [Output]).
PROLOG
```

**Output:** Tab-separated user data from JSON

### 2. Constraint Solving

**Scenario:** Solve scheduling constraints with CLP(FD)

```bash
# Find valid meeting times
cat << 'PROLOG' | swipl -q -g "consult(user), schedule, halt" -t halt
:- use_module(library(clpfd)).

schedule :-
    % 3 people, 5 time slots, find non-overlapping meetings
    [A, B, C] ins 1..5,
    all_different([A, B, C]),
    label([A, B, C]),
    format('Person A: slot ~w, Person B: slot ~w, Person C: slot ~w~n', [A, B, C]),
    fail.
schedule.
PROLOG
```

**Output:** All valid scheduling combinations

### 3. Data Transformation Pipeline

**Scenario:** Transform CSV to JSON using Prolog logic

```bash
# Read CSV, apply business logic, output JSON
cat << 'PROLOG' | swipl -q -g "consult(user), transform, halt" -t halt
:- use_module(library(csv)).
:- use_module(library(http/json)).

transform :-
    % Read from stdin, apply discount logic, write JSON
    read_csv_stream(user_input, Rows, []),
    maplist(apply_discount, Rows, Discounted),
    json_write(current_output, Discounted).

apply_discount(row(Product, PriceStr), json([product=Product, price=Final])) :-
    atom_number(PriceStr, Price),
    (Price > 100 -> Final is Price * 0.9 ; Final = Price).
PROLOG
```

**Usage:**
```bash
cat products.csv | <prolog-service> > products.json
```

### 4. Configuration Validation

**Scenario:** Validate complex configuration with constraints

```bash
validate_config() {
    cat << 'PROLOG' | swipl -q -g "consult(user), validate('$1'), halt" -t halt
:- use_module(library(readutil)).

validate(ConfigFile) :-
    read_file_to_terms(ConfigFile, Terms, []),
    maplist(check_term, Terms),
    format('Configuration valid~n').

check_term(setting(Name, Value)) :-
    atom(Name),
    ground(Value),
    !.
check_term(Term) :-
    format('Invalid term: ~w~n', [Term]),
    halt(1).
PROLOG
}

validate_config config.pl || exit 1
```

### 5. Code Generation

**Scenario:** Generate bash functions from Prolog specs

```bash
# Generate bash test functions from Prolog test specs
cat << 'PROLOG' | swipl -q -g "consult(user), generate_tests, halt" -t halt
test_spec(test_auth, 'curl -u user:pass /api/auth').
test_spec(test_data, 'curl /api/data | jq .items').
test_spec(test_health, 'curl /health').

generate_tests :-
    forall(test_spec(Name, Command), (
        format('~w() {~n', [Name]),
        format('    ~w~n', [Command]),
        format('}~n~n')
    )).
PROLOG
```

**Output:** Bash test functions ready to source

## Advanced Patterns

### Pattern 1: Multi-Stage Pipelines

```bash
# Stage 1: Prolog analyzes input
ANALYSIS=$(cat data.txt | cat << 'PROLOG' | swipl -q -g "consult(user), analyze, halt"
analyze :-
    read_file_to_codes(user_input, Codes),
    length(Codes, Len),
    format('~w', [Len]).
PROLOG
)

# Stage 2: Bash uses result
if [ "$ANALYSIS" -gt 1000 ]; then
    echo "Large file detected"
fi
```

### Pattern 2: Interactive Query Service

```bash
# Create reusable query function
query_prolog() {
    local goal="$1"
    cat << PROLOG | swipl -q -g "consult(user), $goal, halt" -t halt
:- use_module(library(lists)).
% ... load your knowledge base ...
PROLOG
}

# Use it multiple times
query_prolog "member(X, [1,2,3]), write(X), nl, fail"
query_prolog "between(1, 5, X), write(X), nl, fail"
```

### Pattern 3: Embedded Knowledge Base

```bash
# Build knowledge base from bash data
cat << 'PROLOG' | swipl -q -g "consult(user), query, halt" -t halt
% Facts from environment
$(env | awk -F= '{print "env_var(\047"$1"\047, \047"$2"\047)."}'

% Query environment
query :-
    env_var('USER', User),
    env_var('HOME', Home),
    format('User ~w has home ~w~n', [User, Home]).
PROLOG
```

### Pattern 4: Error Handling

```bash
# Capture both success/failure and output
run_prolog_service() {
    local output
    local exit_code

    output=$(cat << 'PROLOG' | swipl -q -g "consult(user), main, halt" -t halt 2>&1
main :-
    (   risky_operation
    ->  format('Success~n')
    ;   format('Failed~n', []), halt(1)
    ).

risky_operation :- random(X), X > 0.5.
PROLOG
    )
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Prolog service succeeded: $output"
    else
        echo "Prolog service failed: $output" >&2
        return 1
    fi
}
```

## Future UnifyWeaver Feature: `prolog_service` Command

### Proposed Interface

```bash
# Simple query
prolog_service "member(X, [1,2,3]), write(X), nl, fail"

# With modules
prolog_service -m clpfd "X #< 10, label([X]), write(X), nl, fail"

# From file
prolog_service -f query.pl -g solve

# With input data
cat data.csv | prolog_service -f transform.pl -g "csv_to_json"

# Multiple goals
prolog_service -g "load_data, process, save_results"
```

### Implementation Sketch

```bash
#!/bin/bash
# prolog_service - Run Prolog code as a service

prolog_service() {
    local modules=()
    local goal=""
    local file=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--module)
                modules+=("$2")
                shift 2
                ;;
            -g|--goal)
                goal="$2"
                shift 2
                ;;
            -f|--file)
                file="$2"
                shift 2
                ;;
            *)
                goal="$1"
                shift
                ;;
        esac
    done

    # Build Prolog code
    {
        # Load modules
        for mod in "${modules[@]}"; do
            echo ":- use_module(library($mod))."
        done

        # Load file if specified
        if [[ -n "$file" ]]; then
            cat "$file"
        fi

        # Add goal
        if [[ -n "$goal" ]]; then
            echo "main :- $goal."
        fi
    } | swipl -q -g "consult(user), main, halt" -t halt
}
```

## Best Practices

### 1. Always Use Single-Quoted Heredocs

```bash
# GOOD - Prolog variables preserved
cat << 'PROLOG'
solve(X, Y) :- X = Y.
PROLOG

# BAD - Bash might expand $X, $Y
cat << PROLOG
solve(X, Y) :- X = Y.
PROLOG
```

### 2. Handle Failures Gracefully

```bash
# Pattern: fail-and-succeed to iterate all solutions
main :- solve(X), write(X), nl, fail.
main.  % Succeed after exhausting solutions
```

### 3. Use `-q` Flag for Clean Output

```bash
# Suppress SWI-Prolog banner and prompts
swipl -q -g "consult(user), main, halt" -t halt
```

### 4. Separate Data from Logic

```bash
# Load data separately from logic
cat data.pl logic.pl | swipl -q -g "consult(user), query, halt"
```

### 5. Test Incrementally

```bash
# Test Prolog code in isolation first
cat << 'PROLOG' | swipl
:- use_module(library(lists)).
test :- member(X, [1,2,3]), write(X), nl, fail.
test.
?- test.
^D

# Then integrate into bash script
```

## Performance Considerations

### Startup Cost

- **SWI-Prolog startup:** ~50-100ms
- **Module loading:** +10-50ms per module
- **Acceptable for:** One-time queries, batch processing
- **Not ideal for:** Tight loops with many invocations

### Optimization Strategies

1. **Batch processing** - Process multiple items in one Prolog invocation
2. **Long-running service** - Start Prolog once, query multiple times
3. **Pre-compilation** - Use `.qlf` files for faster loading
4. **Streaming** - Process data incrementally rather than loading all at once

### Example: Batch vs Loop

```bash
# BAD - Start Prolog 100 times
for item in $(seq 1 100); do
    echo "$item" | cat << 'PROLOG' | swipl -q ...
done

# GOOD - Start Prolog once, process all items
seq 1 100 | cat << 'PROLOG' | swipl -q -g "consult(user), process_all, halt"
process_all :-
    read_line_to_codes(user_input, Codes),
    (   Codes == end_of_file
    ->  true
    ;   process_line(Codes),
        process_all
    ).
PROLOG
```

## Security Considerations

### 1. Input Sanitization

```bash
# DANGER - Prolog injection!
USER_INPUT="$1"
cat << PROLOG | swipl ...
query :- Item = $USER_INPUT.
PROLOG

# SAFE - Use stdin for untrusted data
echo "$USER_INPUT" | cat << 'PROLOG' | swipl ...
query :-
    read_line_to_codes(user_input, Codes),
    % ... validate and process safely ...
PROLOG
```

### 2. Resource Limits

```bash
# Limit execution time (10 seconds)
timeout 10 cat << 'PROLOG' | swipl -q -g "consult(user), main, halt"
# ... potentially long-running query ...
PROLOG
```

### 3. File Access Control

```bash
# Restrict file access via sandboxing (if available)
cat << 'PROLOG' | swipl -q -g "consult(user), main, halt"
:- use_module(library(sandbox)).
:- sandbox:safe_primitive(my_safe_operation/1).
PROLOG
```

## Testing and Debugging

### Enable Tracing

```bash
# Add trace/0 to debug
cat << 'PROLOG' | swipl -g "consult(user), trace, main, halt"
main :- solve(X), write(X), nl.
solve(X) :- between(1, 5, X).
PROLOG
```

### Capture Errors

```bash
# Capture stderr separately
ERRORS=$(cat << 'PROLOG' 2>&1 >/dev/null | swipl ...
% ... code that might fail ...
PROLOG
)

if [[ -n "$ERRORS" ]]; then
    echo "Prolog errors: $ERRORS" >&2
fi
```

### Validate Syntax

```bash
# Check syntax before running
validate_prolog() {
    local code="$1"
    echo "$code" | swipl -q -g "consult(user), halt" -t halt 2>&1 | grep -i error
}
```

## Related Documentation

- `docs/development/STDIN_LOADING.md` - Complete guide to loading Prolog from stdin
- `docs/development/testing/v0_0_2_linux_test_plan.md` - Test examples using this pattern
- `education/drafts/advanced-bash-patterns.md` - Bash techniques for Prolog integration

## Future Work

### Short Term
- [ ] Create `prolog_service` wrapper script
- [ ] Add to UnifyWeaver core utilities
- [ ] Document common patterns library
- [ ] Performance benchmarks

### Medium Term
- [ ] Long-running Prolog server mode
- [ ] JSON/structured data exchange format
- [ ] Pre-compiled query cache
- [ ] Integration with UnifyWeaver code generation

### Long Term
- [ ] Prolog microservices architecture
- [ ] HTTP/REST API wrapper
- [ ] Distributed query execution
- [ ] Prolog-bash DSL

## Examples Repository

See `examples/prolog_services/` for complete working examples:
- `csv_transform.sh` - CSV to JSON transformation
- `constraint_solver.sh` - Scheduling with CLP(FD)
- `code_generator.sh` - Generate bash from specs
- `config_validator.sh` - Validate configuration files
- `data_pipeline.sh` - Multi-stage ETL pipeline

## Conclusion

The "Prolog as a Service" pattern enables powerful declarative logic programming within bash scripts without the complexity of separate processes or file management. By leveraging `consult(user)` and heredocs, developers can embed Prolog queries directly in shell scripts, combining the orchestration power of bash with the expressive power of Prolog.

This pattern is production-ready and battle-tested through the UnifyWeaver v0.0.2 test suite, demonstrating real-world applicability for:
- Data transformation
- Code generation
- Constraint solving
- Configuration validation
- Business logic queries

Future UnifyWeaver releases will build on this foundation to provide seamless Prolog-bash integration for data pipeline development.
