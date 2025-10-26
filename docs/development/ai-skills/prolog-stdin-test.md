# Prolog stdin Test Skill

Test SWI-Prolog code using stdin without creating temporary files.

## Context

UnifyWeaver developers often need to quickly test Prolog code snippets, especially when testing data source compilation and bash code generation. This skill helps run Prolog code via stdin using `consult(user)`.

## When to use this skill

- Testing UnifyWeaver data source definitions (CSV, JSON, HTTP, Python)
- Quick experiments with Prolog predicates
- Verifying code generation without creating test files
- Running one-off queries with full module loading

## How it works

SWI-Prolog can load directives, rules, and queries from stdin using `consult(user)`:

```bash
cat << 'EOF' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module(library(lists)).
test :- member(X, [a,b,c]), writeln(X), fail.
test.
EOF
```

## Task Instructions

When the user asks you to test Prolog code via stdin:

1. **Analyze the code** - Check if it has:
   - Module imports (`:- use_module(...)`)
   - Data source definitions (`:- source(...)`)
   - Test predicates
   - Goal to run

2. **Create a heredoc command** using this template:
   ```bash
   cat << 'EOF' | swipl -q -g "consult(user), GOAL, halt" -t halt
   PROLOG_CODE_HERE
   EOF
   ```

3. **Set up test data** if needed:
   - Create JSON/CSV test files in `/tmp/`
   - Ensure required directories exist
   - Initialize databases if testing SQLite sources

4. **Run the test** and show:
   - Generated bash code (if applicable)
   - Execution output
   - Any errors with explanations

5. **Offer to save** the test as a `.pl` file if it's useful for future use

## UnifyWeaver-Specific Patterns

### Testing JSON source:
```bash
# Create test data
cat > /tmp/test.json << 'DATA'
{"users": [{"id": 1, "name": "Alice"}]}
DATA

# Test the source
cat << 'EOF' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/json_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- source(json, users, [
    json_file('/tmp/test.json'),
    jq_filter('.users[] | .name'),
    raw_output(true)
]).

test :-
    compile_dynamic_source(users/2, [], Code),
    write_and_execute_bash(Code, '', Output),
    format('Output: ~w~n', [Output]).
EOF
```

### Testing CSV source:
```bash
# Create test data
cat > /tmp/test.csv << 'DATA'
name,age,city
Alice,25,NYC
Bob,30,SF
DATA

# Test the source
cat << 'EOF' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/csv_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- source(csv, users, [
    csv_file('/tmp/test.csv'),
    has_header(true)
]).

test :-
    compile_dynamic_source(users/2, [], Code),
    write_and_execute_bash(Code, '', Output),
    format('Output: ~w~n', [Output]).
EOF
```

### Testing Python source:
```bash
cat << 'EOF' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/python_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- source(python, process, [
    python_inline('
import sys
for line in sys.stdin:
    print(line.strip().upper())
')
]).

test :-
    compile_dynamic_source(process/2, [], Code),
    write_and_execute_bash(Code, 'hello\nworld\n', Output),
    format('Output: ~w~n', [Output]).
EOF
```

## Common Pitfalls

1. **Forgetting single quotes in EOF** - Use `'EOF'` not `EOF` to prevent variable expansion
2. **Missing initialization goal** - The test predicate must be called in the `-g` flag
3. **Module path issues** - Use relative paths from test_env working directory
4. **Stream handling** - Remember to close streams and handle EOF properly

## References

- `docs/development/STDIN_LOADING.md` - Full stdin loading documentation
- `examples/test_json_source.pl` - Example test file
- [SWI-Prolog consult/1](https://www.swi-prolog.org/pldoc/man?predicate=consult%2F1)
