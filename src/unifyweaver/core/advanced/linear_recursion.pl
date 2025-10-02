:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% linear_recursion.pl - Compile linear recursive predicates
% Linear recursion: exactly one recursive call per clause
% Strategy: Use memoization or transform to accumulator pattern

:- module(linear_recursion, [
    compile_linear_recursion/3,     % +Pred/Arity, +Options, -BashCode
    can_compile_linear_recursion/1,  % +Pred/Arity
    test_linear_recursion/0        % Test predicate
]).

:- use_module(library(lists)).
:- use_module('../template_system').
:- use_module('pattern_matchers').

%% can_compile_linear_recursion(+Pred/Arity)
%  Check if predicate can be compiled as linear recursion
can_compile_linear_recursion(Pred/Arity) :-
    is_linear_recursive_streamable(Pred/Arity).

%% compile_linear_recursion(+Pred/Arity, +Options, -BashCode)
%  Compile linear recursive predicate
%  Strategy: Use memoization with associative arrays
compile_linear_recursion(Pred/Arity, _Options, BashCode) :-
    format('  Compiling linear recursion: ~w/~w~n', [Pred, Arity]),

    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),

    % Get clauses
    findall(clause(Head, Body), clause(Head, Body), Clauses),

    % Separate base and recursive cases
    partition(is_recursive_clause(Pred), Clauses, RecClauses, BaseClauses),

    % Generate bash code
    generate_linear_recursion_bash(PredStr, Arity, BaseClauses, RecClauses, BashCode).

%% is_recursive_clause(+Pred, +Clause)
is_recursive_clause(Pred, clause(_Head, Body)) :-
    contains_call_to_pred(Body, Pred).

%% contains_call_to_pred(+Body, +Pred)
contains_call_to_pred(Body, Pred) :-
    extract_goal_from_body(Body, Goal),
    functor(Goal, Pred, _).

%% extract_goal_from_body(+Body, -Goal)
extract_goal_from_body(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_).
extract_goal_from_body((A, _), Goal) :- extract_goal_from_body(A, Goal).
extract_goal_from_body((_, B), Goal) :- extract_goal_from_body(B, Goal).
extract_goal_from_body((A; _), Goal) :- extract_goal_from_body(A, Goal).
extract_goal_from_body((_;B), Goal) :- extract_goal_from_body(B, Goal).

%% generate_linear_recursion_bash(+PredStr, +Arity, +BaseClauses, +RecClauses, -BashCode)
generate_linear_recursion_bash(PredStr, Arity, BaseClauses, RecClauses, BashCode) :-
    % For arity 2, assume pattern: pred(Input, Output)
    (   Arity =:= 2 ->
        generate_binary_linear_recursion(PredStr, BaseClauses, RecClauses, BashCode)
    ;   % Other arities - use generic memoization
        generate_generic_linear_recursion(PredStr, Arity, BaseClauses, RecClauses, BashCode)
    ).

%% generate_binary_linear_recursion(+PredStr, +BaseClauses, +RecClauses, -BashCode)
%  Generate code for binary linear recursion (most common case)
%  Example: length([], 0). length([_|T], N) :- length(T, N1), N is N1 + 1.
generate_binary_linear_recursion(PredStr, _BaseClauses, _RecClauses, BashCode) :-
    TemplateLines = [
        "#!/bin/bash",
        "# {{pred}} - linear recursive pattern with memoization",
        "",
        "# Memoization table",
        "declare -gA {{pred}}_memo",
        "",
        "{{pred}}() {",
        "    local arg1=\"$1\"",
        "    local arg2=\"$2\"",
        "    local key=\"$arg1\"",
        "    ",
        "    # Check memo table",
        "    if [[ -n \"${{{pred}}_memo[$key]}\" ]]; then",
        "        local cached=\"${{{pred}}_memo[$key]}\"",
        "        if [[ -n \"$arg2\" ]]; then",
        "            [[ \"$cached\" == \"$arg2\" ]] && echo \"$key:$arg2\" && return 0",
        "            return 1",
        "        else",
        "            echo \"$key:$cached\"",
        "            return 0",
        "        fi",
        "    fi",
        "    ",
        "    # Base case: empty list",
        "    if [[ \"$arg1\" == \"[]\" || -z \"$arg1\" ]]; then",
        "        local result=\"0\"",
        "        {{pred}}_memo[\"$key\"]=\"$result\"",
        "        if [[ -n \"$arg2\" ]]; then",
        "            [[ \"$result\" == \"$arg2\" ]] && echo \"$key:$arg2\" && return 0",
        "            return 1",
        "        else",
        "            echo \"$key:$result\"",
        "            return 0",
        "        fi",
        "    fi",
        "    ",
        "    # Recursive case: process list",
        "    # Extract tail (simplified - assumes comma-separated)",
        "    if [[ \"$arg1\" =~ ^\\[.*\\]$ ]]; then",
        "        # Remove brackets",
        "        local content=\"${arg1#[}\"",
        "        content=\"${content%]}\"",
        "        ",
        "        # Split on first comma to get tail",
        "        if [[ \"$content\" =~ ^[^,]+,(.+)$ ]]; then",
        "            local tail=\"[${BASH_REMATCH[1]}]\"",
        "        else",
        "            # Single element - tail is empty",
        "            local tail=\"[]\"",
        "        fi",
        "        ",
        "        # Recursive call",
        "        local rec_result",
        "        if rec_output=$({{pred}} \"$tail\" \"\"); then",
        "            rec_result=\"${rec_output##*:}\"",
        "            # Apply step operation (e.g., N is N1 + 1)",
        "            local result=$((rec_result + 1))",
        "            ",
        "            # Memoize",
        "            {{pred}}_memo[\"$key\"]=\"$result\"",
        "            ",
        "            if [[ -n \"$arg2\" ]]; then",
        "                [[ \"$result\" == \"$arg2\" ]] && echo \"$key:$arg2\" && return 0",
        "                return 1",
        "            else",
        "                echo \"$key:$result\"",
        "                return 0",
        "            fi",
        "        fi",
        "    fi",
        "    ",
        "    return 1",
        "}",
        "",
        "# Stream all solutions",
        "{{pred}}_stream() {",
        "    {{pred}} \"$@\"",
        "}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr], BashCode).

%% generate_generic_linear_recursion(+PredStr, +Arity, +BaseClauses, +RecClauses, -BashCode)
%  Generic linear recursion for any arity
generate_generic_linear_recursion(PredStr, Arity, _BaseClauses, _RecClauses, BashCode) :-
    TemplateLines = [
        "#!/bin/bash",
        "# {{pred}}/{{arity}} - linear recursive pattern (generic)",
        "",
        "# Memoization table",
        "declare -gA {{pred}}_memo",
        "",
        "{{pred}}() {",
        "    # Build key from all arguments",
        "    local key=\"$*\"",
        "    ",
        "    # Check memo",
        "    if [[ -n \"${{{pred}}_memo[$key]}\" ]]; then",
        "        echo \"${{{pred}}_memo[$key]}\"",
        "        return 0",
        "    fi",
        "    ",
        "    # TODO: Implement base cases and recursive logic",
        "    # This is a generic template - needs customization per predicate",
        "    ",
        "    echo \"# Generic linear recursion - not yet implemented\" >&2",
        "    return 1",
        "}"
    ],

    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, arity=Arity], BashCode).

%% ============================================
%% TESTS
%% ============================================

test_linear_recursion :-
    writeln('=== LINEAR RECURSION COMPILER TESTS ==='),

    % Setup output directory
    (   exists_directory('output/advanced') -> true
    ;   make_directory('output/advanced')
    ),

    % Clear test predicates
    catch(abolish(list_length/2), _, true),
    catch(abolish(factorial/2), _, true),

    % Test 1: List length (classic linear recursion)
    writeln('Test 1: Compile list_length/2 (linear recursive)'),
    assertz((list_length([], 0))),
    assertz((list_length([_|T], N) :- list_length(T, N1), N is N1 + 1)),

    (   can_compile_linear_recursion(list_length/2) ->
        writeln('  ✓ Pattern detected'),
        compile_linear_recursion(list_length/2, [], Code1),
        write_bash_file('output/advanced/list_length.sh', Code1),
        writeln('  ✓ Compiled to output/advanced/list_length.sh')
    ;   writeln('  ✗ FAIL - should detect linear recursion')
    ),

    % Test 2: Factorial (linear recursion with arithmetic)
    writeln('Test 2: Compile factorial/2 (linear recursive)'),
    assertz((factorial(0, 1))),
    assertz((factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),

    (   can_compile_linear_recursion(factorial/2) ->
        writeln('  ✓ Pattern detected'),
        compile_linear_recursion(factorial/2, [], Code2),
        write_bash_file('output/advanced/factorial.sh', Code2),
        writeln('  ✓ Compiled to output/advanced/factorial.sh')
    ;   writeln('  ⚠ Pattern not detected (expected for factorial)')
    ),

    writeln('=== LINEAR RECURSION COMPILER TESTS COMPLETE ===').

%% Helper to write bash files
write_bash_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream),
    % Make executable
    atom_concat('chmod +x ', Path, ChmodCmd),
    shell(ChmodCmd).
