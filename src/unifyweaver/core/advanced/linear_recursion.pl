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
:- use_module('../constraint_analyzer').
:- use_module('pattern_matchers').

%% can_compile_linear_recursion(+Pred/Arity)
%  Check if predicate can be compiled as linear recursion
can_compile_linear_recursion(Pred/Arity) :-
    is_linear_recursive_streamable(Pred/Arity).

%% compile_linear_recursion(+Pred/Arity, +Options, -BashCode)
%  Compile linear recursive predicate
%  Strategy: Use memoization with associative arrays
%  Options: List of Key=Value pairs (e.g., [unique=true, ordered=false])
%  Currently linear recursive predicates return single values (not sets),
%  so deduplication constraints don't apply. Options are reserved for
%  future use (e.g., output language selection).
compile_linear_recursion(Pred/Arity, Options, BashCode) :-
    format('  Compiling linear recursion: ~w/~w~n', [Pred, Arity]),

    % Query constraints (for logging and future use)
    get_constraints(Pred/Arity, Constraints),
    format('  Constraints: ~w~n', [Constraints]),

    % Merge runtime options with constraints
    append(Options, Constraints, AllOptions),
    format('  Final options: ~w~n', [AllOptions]),

    % TODO: Linear recursive patterns return single values, not sets.
    % Deduplication constraints (unique, unordered) may not apply here.
    % Options are kept for future extensibility (e.g., output_lang=python).

    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),

    % Get clauses - use user:clause to access predicates from any module (including test predicates)
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

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

%% ============================================
%% CLAUSE INTROSPECTION (Phase 2)
%% ============================================

%% extract_base_case_info(+BaseClauses, -BaseInput, -BaseOutput)
%  Extract input/output pattern from base case
extract_base_case_info([clause(Head, _Body)|_], BaseInput, BaseOutput) :-
    Head =.. [_Pred, BaseInput, BaseOutput].

%% detect_input_type(+BaseInput, -Type)
%  Determine if input is numeric, list, or other
detect_input_type(BaseInput, Type) :-
    (   BaseInput = [] ->
        Type = list
    ;   integer(BaseInput) ->
        Type = numeric
    ;   Type = unknown
    ).

%% extract_fold_operation(+RecClauses, -FoldExpr)
%  Extract the fold operation from the recursive case
%  Example: F is N * F1 → FoldExpr = N * F1
extract_fold_operation([clause(_Head, Body)|_], FoldExpr) :-
    find_is_expression(Body, _ is FoldExpr).

%% find_is_expression(+Body, -IsGoal)
%  Find the 'is' expression in the body
find_is_expression(Goal, Goal) :-
    Goal = (_ is _), !.
find_is_expression((A, _), IsGoal) :-
    find_is_expression(A, IsGoal), !.
find_is_expression((_, B), IsGoal) :-
    find_is_expression(B, IsGoal).

%% ============================================
%% FOLD-BASED CODE GENERATION (Phase 3 & 4)
%% ============================================

%% generate_linear_recursion_bash(+PredStr, +Arity, +BaseClauses, +RecClauses, -BashCode)
generate_linear_recursion_bash(PredStr, Arity, BaseClauses, RecClauses, BashCode) :-
    % For arity 2, use fold-based approach
    (   Arity =:= 2 ->
        generate_fold_based_recursion(PredStr, BaseClauses, RecClauses, BashCode)
    ;   % Other arities - use generic memoization
        generate_generic_linear_recursion(PredStr, Arity, BaseClauses, RecClauses, BashCode)
    ).

%% generate_fold_based_recursion(+PredStr, +BaseClauses, +RecClauses, -BashCode)
%  Generate fold-based linear recursion code
%  Works for both numeric (factorial) and list (list_length) patterns
generate_fold_based_recursion(PredStr, BaseClauses, RecClauses, BashCode) :-
    % Extract pattern information
    extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    detect_input_type(BaseInput, InputType),
    extract_fold_operation(RecClauses, FoldExpr),

    % Generate based on input type
    (   InputType = numeric ->
        generate_numeric_fold(PredStr, BaseInput, BaseOutput, FoldExpr, BashCode)
    ;   InputType = list ->
        generate_list_fold(PredStr, BaseInput, BaseOutput, FoldExpr, BashCode)
    ;   % Fallback to old template if unknown
        generate_binary_linear_recursion_old(PredStr, BaseClauses, RecClauses, BashCode)
    ).

%% generate_numeric_fold(+PredStr, +BaseInput, +BaseOutput, +FoldExpr, -BashCode)
%  Generate fold-based code for numeric linear recursion (e.g., factorial)
generate_numeric_fold(PredStr, BaseInput, BaseOutput, FoldExpr, BashCode) :-
    % Translate Prolog fold expression to bash
    translate_fold_expr(FoldExpr, BashFoldOp),

    % Generate fold helper library + predicate wrapper
    format(string(BashCode), '#!/bin/bash
# ~w - fold-based linear recursion (numeric)
# Pattern: fold down from N to 1, combining with operation

# Fold helper: fold_left accumulator operation values...
fold_left() {
    local acc="$1"
    local op_func="$2"
    shift 2
    for item in "$@"; do
        acc=$("$op_func" "$item" "$acc")
    done
    echo "$acc"
}

# Range builder: generates N, N-1, ..., 1
build_range_down() {
    local n="$1"
    seq "$n" -1 1
}

# Fold operation for ~w
~w_op() {
    local current="$1"
    local acc="$2"
    echo $((~w))
}

# Main predicate with memoization
declare -gA ~w_memo

~w() {
    local n="$1"
    local expected="$2"

    # Check memo
    if [[ -n "${~w_memo[$n]}" ]]; then
        local cached="${~w_memo[$n]}"
        if [[ -n "$expected" ]]; then
            [[ "$cached" == "$expected" ]] && echo "$n:$expected" && return 0
            return 1
        else
            echo "$n:$cached"
            return 0
        fi
    fi

    # Base case
    if [[ "$n" -eq ~w ]]; then
        local result="~w"
        ~w_memo["$n"]="$result"
        if [[ -n "$expected" ]]; then
            [[ "$result" == "$expected" ]] && echo "$n:$expected" && return 0
            return 1
        else
            echo "$n:$result"
            return 0
        fi
    fi

    # Recursive case using fold
    local range=$(build_range_down "$n")
    local result=$(fold_left ~w "~w_op" $range)

    # Memoize
    ~w_memo["$n"]="$result"

    if [[ -n "$expected" ]]; then
        [[ "$result" == "$expected" ]] && echo "$n:$expected" && return 0
        return 1
    else
        echo "$n:$result"
        return 0
    fi
}

# Stream wrapper
~w_stream() {
    ~w "$@"
}
', [PredStr, PredStr, PredStr, BashFoldOp, PredStr, PredStr, PredStr, BaseInput, BaseOutput, PredStr, BaseOutput, PredStr, PredStr, PredStr, PredStr, PredStr]).

%% translate_fold_expr(+PrologExpr, -BashExpr)
%  Translate Prolog arithmetic expression to bash
%  Example: N * F1 → current * acc
translate_fold_expr(Expr, BashExpr) :-
    translate_expr(Expr, BashExpr).

translate_expr(A * B, BashExpr) :-
    translate_term(A, AT),
    translate_term(B, BT),
    format(string(BashExpr), '~w * ~w', [AT, BT]).
translate_expr(A + B, BashExpr) :-
    translate_term(A, AT),
    translate_term(B, BT),
    format(string(BashExpr), '~w + ~w', [AT, BT]).
translate_expr(A - B, BashExpr) :-
    translate_term(A, AT),
    translate_term(B, BT),
    format(string(BashExpr), '~w - ~w', [AT, BT]).
translate_expr(A / B, BashExpr) :-
    translate_term(A, AT),
    translate_term(B, BT),
    format(string(BashExpr), '~w / ~w', [AT, BT]).
translate_expr(Term, BashExpr) :-
    translate_term(Term, BashExpr).

%% translate_term(+PrologTerm, -BashTerm)
%  Map Prolog variable names to bash variable names
translate_term(N, 'current') :- atom(N), atom_chars(N, [FirstChar|_]), char_type(FirstChar, upper), !.
translate_term(Var, 'acc') :- atom(Var), member(Var, [f1, n1, result, r1]), !.
translate_term(Number, BashTerm) :- integer(Number), !, format(string(BashTerm), '~w', [Number]).
translate_term(Atom, BashTerm) :- format(string(BashTerm), '~w', [Atom]).

%% generate_list_fold(+PredStr, +BaseInput, +BaseOutput, +FoldExpr, -BashCode)
%  Generate fold-based code for list linear recursion (e.g., list_length)
%  TODO: Implement list fold pattern
generate_list_fold(PredStr, _BaseInput, _BaseOutput, _FoldExpr, BashCode) :-
    format(string(BashCode), '#!/bin/bash
# ~w - list fold pattern (NOT YET IMPLEMENTED)
# TODO: Implement list-based fold

~w() {
    echo "List fold not yet implemented" >&2
    return 1
}
', [PredStr, PredStr]).

%% generate_binary_linear_recursion_old(+PredStr, +BaseClauses, +RecClauses, -BashCode)
%  OLD IMPLEMENTATION - kept as fallback
generate_binary_linear_recursion_old(PredStr, _BaseClauses, _RecClauses, BashCode) :-
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
    assertz(user:(list_length([], 0))),
    assertz(user:(list_length([_|T], N) :- list_length(T, N1), N is N1 + 1)),

    (   can_compile_linear_recursion(list_length/2) ->
        writeln('  ✓ Pattern detected'),
        compile_linear_recursion(list_length/2, [], Code1),
        write_bash_file('output/advanced/list_length.sh', Code1),
        writeln('  ✓ Compiled to output/advanced/list_length.sh')
    ;   writeln('  ✗ FAIL - should detect linear recursion')
    ),

    % Test 2: Factorial (linear recursion with arithmetic)
    writeln('Test 2: Compile factorial/2 (linear recursive)'),
    assertz(user:(factorial(0, 1))),
    assertz(user:(factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),

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
