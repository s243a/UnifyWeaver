# Loading Prolog Code from stdin

This document explains how to load Prolog code from standard input in SWI-Prolog, which is useful for quick experiments and testing without creating temporary files.

## The Correct Approach: `consult(user)`

SWI-Prolog supports loading directives, rules, and queries from stdin using `consult(user)` or `load_files/2` with stream options.

### Method 1: Using `consult(user)`

The `consult(user)` predicate compiles directives and rules from standard input:

```bash
cat test_file.pl | swipl -q -g "consult(user), test, halt" -t halt
```

**Example with heredoc:**

```bash
cat << 'EOF' | swipl -q -g "consult(user), my_test, halt" -t halt
:- use_module(library(lists)).

my_test :-
    member(X, [a, b, c]),
    writeln(X),
    fail.
my_test.
EOF
```

Output:
```
a
b
c
```

### Method 2: Using `load_files/2` with stream

The `load_files/2` predicate supports a `stream(Input)` option:

```prolog
:- initialization(main, main).

main :-
    load_files(stdin_source, [stream(user_input)]),
    test,
    halt.
```

### Method 3: Interactive `[user]` session

At the REPL, you can type `[user].` to enter clauses interactively:

```prolog
?- [user].
|: fact(a).
|: fact(b).
|: fact(c).
|: ^D  % Press CTRL-D to end
true.

?- fact(X).
X = a ;
X = b ;
X = c.
```

## UnifyWeaver-Specific Example

Testing a JSON source without creating a file:

```bash
cat << 'EOF' | swipl -q -g "consult(user), test, halt" -t halt
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/json_source').
:- use_module('src/unifyweaver/core/bash_executor').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- source(json, users, [
    json_file('/tmp/test_data.json'),
    jq_filter('.users[] | [.id, .name, .age, .city] | @tsv'),
    raw_output(true)
]).

test :-
    compile_dynamic_source(users/2, [], Code),
    write_and_execute_bash(Code, '', Output),
    format('Output:~n~w~n', [Output]).
EOF
```

## Why `/dev/stdin` and `<(...)` Don't Work

SWI-Prolog's file loading mechanism checks if files exist using filesystem calls, which don't work with:
- `/dev/stdin` (special file)
- `/dev/fd/*` (process substitution)
- `-` (stdin placeholder used by many Unix tools)

**The documented and portable approach** is to use `consult(user)` or `load_files/2` with stream options.

## Best Practices

### When to use stdin loading:
- ✅ Quick experiments and one-off tests
- ✅ CI/CD pipelines generating test code dynamically
- ✅ Interactive debugging sessions
- ✅ Scripted testing without temporary files

### When to use file-based loading:
- ✅ Production code and repeatable tests
- ✅ Multi-module projects with dependencies
- ✅ Code that needs to be version-controlled
- ✅ Shared development environments

## Command-Line Patterns

### Simple test with inline code:
```bash
echo "test :- write('Hello'), nl." | swipl -q -g "consult(user), test, halt" -t halt
```

### Load and run multiple goals:
```bash
cat mycode.pl | swipl -q -g "consult(user), goal1, goal2, halt" -t halt
```

### Combine with data files:
```bash
cat << 'EOF' | swipl -q -g "consult(user), process('/tmp/data.csv'), halt" -t halt
process(File) :-
    open(File, read, Stream),
    process_stream(Stream),
    close(Stream).
process_stream(Stream) :-
    read_line_to_string(Stream, Line),
    (Line \= end_of_file -> writeln(Line), process_stream(Stream) ; true).
EOF
```

## References

- [SWI-Prolog: consult/1](https://www.swi-prolog.org/pldoc/man?predicate=consult%2F1)
- [SWI-Prolog: Consulting User](https://www.swi-prolog.org/pldoc/man?section=consultuser)
- [SWI-Prolog: load_files/2](https://semanticweb.cs.vu.nl/verrijktkoninkrijk/swish/pldoc/man?section=consulting)
- [SWI-Prolog: Compilation](https://www.swi-prolog.org/pldoc/man?section=compilation)
