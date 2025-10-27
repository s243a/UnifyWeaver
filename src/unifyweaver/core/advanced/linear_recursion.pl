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
%
%  Constraint effects for linear recursion:
%  - unique(false): Allow duplicate memoization (though unusual for linear recursion)
%  - ordered(true): Use hash-based memoization (faster lookup, preserves order)
%  - ordered(false): Use standard memoization (default)
compile_linear_recursion(Pred/Arity, Options, BashCode) :-
    format('  Compiling linear recursion: ~w/~w~n', [Pred, Arity]),

    % Query constraints
    get_constraints(Pred/Arity, Constraints),
    format('  Constraints: ~w~n', [Constraints]),

    % Merge runtime options with constraints (runtime options override)
    append(Options, Constraints, AllOptions),
    format('  Final options: ~w~n', [AllOptions]),

    % Determine memoization strategy based on constraints
    (   member(unique(false), AllOptions) ->
        format('  Applying unique(false): Memo disabled~n', []),
        MemoEnabled = false
    ;   MemoEnabled = true
    ),

    % Determine memo lookup strategy
    (   member(unordered(false), AllOptions) ->  % ordered = true
        format('  Applying ordered constraint: Using hash-based memo~n', []),
        MemoStrategy = hash
    ;   MemoStrategy = standard
    ),

    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),

    % Get clauses - use user:clause to access predicates from any module (including test predicates)
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Separate base and recursive cases
    partition(is_recursive_clause(Pred), Clauses, RecClauses, BaseClauses),

    % Generate bash code with constraint info
    generate_linear_recursion_bash(PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, BashCode).

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
%  We want the LAST 'is' expression (the output computation), not intermediate ones
extract_fold_operation([clause(_Head, Body)|_], FoldExpr) :-
    find_last_is_expression(Body, _ is FoldExpr).

%% find_last_is_expression(+Body, -IsGoal)
%  Find the LAST 'is' expression in the body (without creating fresh variables)
find_last_is_expression((A, B), IsGoal) :- !,
    (   find_last_is_expression(B, IsGoal) ->
        true
    ;   find_last_is_expression(A, IsGoal)
    ).
find_last_is_expression(Goal, Goal) :-
    Goal = (_ is _).

%% analyze_clause_structure(+RecClauses, -InputVar, -AccVar)
%  Analyze the recursive clause structure to identify variable roles
%  For factorial(N, F) :- ..., factorial(N1, F1), F is N * F1:
%    - InputVar is the first argument of the head (N)
%    - AccVar is the result variable from the recursive call (F1)
analyze_clause_structure([clause(Head, Body)|_], InputVar, AccVar) :-
    % Extract first argument from head (input variable)
    Head =.. [_Pred, InputVar, _OutputVar],

    % Find the recursive call in the body
    find_recursive_call(Body, RecCall),

    % Extract accumulator variable (second argument of recursive call)
    RecCall =.. [_RecPred, _RecInput, AccVar].

%% find_recursive_call(+Body, -RecCall)
%  Find the recursive call in the body
find_recursive_call(Body, RecCall) :-
    extract_goal_from_body(Body, Goal),
    compound(Goal),
    Goal \= (_ is _),
    Goal \= (_ > _),
    Goal \= (_ < _),
    Goal \= (_ =< _),
    Goal \= (_ >= _),
    % Find a goal that's a compound term with 2 args (likely the recursive call)
    functor(Goal, _, 2),
    RecCall = Goal,
    !.
find_recursive_call((A, _B), RecCall) :-
    find_recursive_call(A, RecCall), !.
find_recursive_call((_A, B), RecCall) :-
    find_recursive_call(B, RecCall).

%% ============================================
%% FOLD-BASED CODE GENERATION (Phase 3 & 4)
%% ============================================

%% generate_linear_recursion_bash(+PredStr, +Arity, +BaseClauses, +RecClauses, +MemoEnabled, +MemoStrategy, -BashCode)
generate_linear_recursion_bash(PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, BashCode) :-
    % For arity 2, use fold-based approach
    (   Arity =:= 2 ->
        generate_fold_based_recursion(PredStr, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, BashCode)
    ;   % Other arities - use generic memoization
        generate_generic_linear_recursion(PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, BashCode)
    ).

%% generate_fold_based_recursion(+PredStr, +BaseClauses, +RecClauses, +MemoEnabled, +MemoStrategy, -BashCode)
%  Generate fold-based linear recursion code
%  Works for both numeric (factorial) and list (list_length) patterns
generate_fold_based_recursion(PredStr, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, BashCode) :-
    % Extract pattern information
    extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    detect_input_type(BaseInput, InputType),
    extract_fold_operation(RecClauses, FoldExpr),

    % Generate based on input type
    (   InputType = numeric ->
        generate_numeric_fold(PredStr, BaseInput, BaseOutput, FoldExpr, MemoEnabled, MemoStrategy, BashCode)
    ;   InputType = list ->
        generate_list_fold(PredStr, BaseInput, BaseOutput, FoldExpr, MemoEnabled, MemoStrategy, BashCode)
    ;   % Fallback to old template if unknown
        generate_binary_linear_recursion_old(PredStr, BaseClauses, RecClauses, BashCode)
    ).

%% generate_numeric_fold(+PredStr, +BaseInput, +BaseOutput, +_FoldExpr, +MemoEnabled, +MemoStrategy, -BashCode)
%  Generate fold-based code for numeric linear recursion (e.g., factorial)
%  NOTE: _FoldExpr was already extracted in generate_fold_based_recursion but we re-extract it here
%  to maintain variable identity with the clause analysis
%  MemoEnabled: true/false - whether to use memoization
%  MemoStrategy: standard/hash - memoization lookup strategy
generate_numeric_fold(PredStr, BaseInput, BaseOutput, _FoldExpr, MemoEnabled, MemoStrategy, BashCode) :-
    % Get the recursive clauses to analyze variable roles AND extract fold expr in ONE pass
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(is_recursive_clause(Pred), Clauses, RecClauses, _BaseClauses),

    % Analyze the SAME clause instance to maintain variable identity
    RecClauses = [clause(RHead, RBody)|_],
    RHead =.. [_Pred, InputVar, _OutputVar],
    find_recursive_call(RBody, RecCall),
    RecCall =.. [_RecPred, _RecInput, AccVar],
    find_last_is_expression(RBody, _ is ActualFoldExpr),

    % Translate Prolog fold expression to bash using variable mapping
    translate_fold_expr(ActualFoldExpr, InputVar, AccVar, BashFoldOp),

    % Generate memo declaration (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoDecl), '# Memoization table (~w strategy)~ndeclare -gA ~w_memo~n', [MemoStrategy, PredStr]),
        format(string(MemoCheckCode), '    # Check memo~n    if [[ -n "${~w_memo[$n]}" ]]; then~n        local cached="${~w_memo[$n]}"~n        if [[ -n "$expected" ]]; then~n            [[ "$cached" == "$expected" ]] && echo "$n:$expected" && return 0~n            return 1~n        else~n            echo "$n:$cached"~n            return 0~n        fi~n    fi~n', [PredStr, PredStr]),
        format(string(MemoStoreCode), '    # Memoize~n    ~w_memo["$n"]="$result"~n', [PredStr])
    ;   MemoDecl = '# Memoization disabled (unique=false constraint)\n',
        MemoCheckCode = '',
        MemoStoreCode = ''
    ),

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

# Main predicate
~w

~w() {
    local n="$1"
    local expected="$2"

~w
    # Base case
    if [[ "$n" -eq ~w ]]; then
        local result="~w"
~w
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

~w
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
', [PredStr, PredStr, PredStr, BashFoldOp, MemoDecl, PredStr, MemoCheckCode, BaseInput, BaseOutput, MemoStoreCode, BaseOutput, PredStr, MemoStoreCode, PredStr, PredStr]).


%% translate_fold_expr(+PrologExpr, +InputVar, +AccVar, -BashExpr)
%  Translate Prolog arithmetic expression to bash using clause structure mapping
%  Example: N * F1 with InputVar=N, AccVar=F1 → current * acc
translate_fold_expr(Expr, InputVar, AccVar, BashExpr) :-
    translate_expr(Expr, InputVar, AccVar, BashExpr).

translate_expr(A * B, InputVar, AccVar, BashExpr) :-
    translate_term(A, InputVar, AccVar, AT),
    translate_term(B, InputVar, AccVar, BT),
    format(string(BashExpr), '~w * ~w', [AT, BT]).
translate_expr(A + B, InputVar, AccVar, BashExpr) :-
    translate_term(A, InputVar, AccVar, AT),
    translate_term(B, InputVar, AccVar, BT),
    format(string(BashExpr), '~w + ~w', [AT, BT]).
translate_expr(A - B, InputVar, AccVar, BashExpr) :-
    translate_term(A, InputVar, AccVar, AT),
    translate_term(B, InputVar, AccVar, BT),
    format(string(BashExpr), '~w - ~w', [AT, BT]).
translate_expr(A / B, InputVar, AccVar, BashExpr) :-
    translate_term(A, InputVar, AccVar, AT),
    translate_term(B, InputVar, AccVar, BT),
    format(string(BashExpr), '~w / ~w', [AT, BT]).
translate_expr(Term, InputVar, AccVar, BashExpr) :-
    translate_term(Term, InputVar, AccVar, BashExpr).

%% translate_term(+PrologTerm, +InputVar, +AccVar, -BashTerm)
%  Map Prolog variables to bash variables using structural analysis
%  If the term unifies with InputVar → 'current'
%  If the term unifies with AccVar → 'acc'
%  Otherwise → literal value
translate_term(Term, InputVar, _AccVar, 'current') :-
    Term == InputVar, !.
translate_term(Term, _InputVar, AccVar, 'acc') :-
    Term == AccVar, !.
translate_term(Number, _InputVar, _AccVar, BashTerm) :-
    integer(Number), !,
    format(string(BashTerm), '~w', [Number]).
translate_term(Atom, _InputVar, _AccVar, BashTerm) :-
    format(string(BashTerm), '~w', [Atom]).

%% generate_list_fold(+PredStr, +BaseInput, +BaseOutput, +_FoldExpr, +MemoEnabled, +MemoStrategy, -BashCode)
%  Generate fold-based code for list linear recursion (e.g., list_length)
generate_list_fold(PredStr, BaseInput, BaseOutput, _FoldExpr, MemoEnabled, MemoStrategy, BashCode) :-
    % Get the recursive clauses to analyze variable roles
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(is_recursive_clause(Pred), Clauses, RecClauses, _BaseClauses),

    % Analyze the SAME clause instance to maintain variable identity
    RecClauses = [clause(RHead, RBody)|_],
    RHead =.. [_Pred, _InputVar, _OutputVar],
    find_recursive_call(RBody, RecCall),
    RecCall =.. [_RecPred, _RecInput, AccVar],
    find_last_is_expression(RBody, _ is ActualFoldExpr),

    % For list patterns, the fold expression is typically N1 + 1 (increment)
    % We'll treat this as a constant increment fold
    translate_fold_expr(ActualFoldExpr, _DummyInput, AccVar, BashFoldOp),

    % Generate memo declaration (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoDecl), '# Memoization table (~w strategy)~ndeclare -gA ~w_memo~n', [MemoStrategy, PredStr]),
        format(string(MemoCheckCode), '    # Check memo~n    if [[ -n "${~w_memo[$list]}" ]]; then~n        local cached="${~w_memo[$list]}"~n        if [[ -n "$expected" ]]; then~n            [[ "$cached" == "$expected" ]] && echo "$list:$expected" && return 0~n            return 1~n        else~n            echo "$list:$cached"~n            return 0~n        fi~n    fi~n', [PredStr, PredStr]),
        format(string(MemoStoreCode), '    # Memoize~n    ~w_memo["$list"]="$result"~n', [PredStr])
    ;   MemoDecl = '# Memoization disabled (unique=false constraint)\n',
        MemoCheckCode = '',
        MemoStoreCode = ''
    ),

    % Generate fold helper library + predicate wrapper
    format(string(BashCode), '#!/bin/bash
# ~w - fold-based linear recursion (list)
# Pattern: fold over list elements, combining with operation

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

# Parse list into elements
parse_list() {
    local list="$1"
    # Remove brackets
    list="${list#[}"
    list="${list%]}"
    # Split on commas (simplified - doesn\'t handle nested lists)
    if [[ -n "$list" ]]; then
        echo "$list" | tr \',\' \' \'
    fi
}

# Fold operation for ~w
~w_op() {
    local current="$1"
    local acc="$2"
    # For list_length, we just increment (ignore current element)
    echo $((~w))
}

# Main predicate
~w

~w() {
    local list="$1"
    local expected="$2"

~w
    # Base case
    if [[ "$list" == "~w" || -z "$list" ]]; then
        local result="~w"
~w
        if [[ -n "$expected" ]]; then
            [[ "$result" == "$expected" ]] && echo "$list:$expected" && return 0
            return 1
        else
            echo "$list:$result"
            return 0
        fi
    fi

    # Recursive case using fold
    local elements=$(parse_list "$list")
    local result=$(fold_left ~w "~w_op" $elements)

~w
    if [[ -n "$expected" ]]; then
        [[ "$result" == "$expected" ]] && echo "$list:$expected" && return 0
        return 1
    else
        echo "$list:$result"
        return 0
    fi
}

# Stream wrapper
~w_stream() {
    ~w "$@"
}
', [PredStr, PredStr, PredStr, BashFoldOp, MemoDecl, PredStr, MemoCheckCode, BaseInput, BaseOutput, MemoStoreCode, BaseOutput, PredStr, MemoStoreCode, PredStr, PredStr]).

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

%% generate_generic_linear_recursion(+PredStr, +Arity, +BaseClauses, +RecClauses, +MemoEnabled, +MemoStrategy, -BashCode)
%  Generic linear recursion for any arity
generate_generic_linear_recursion(PredStr, Arity, _BaseClauses, _RecClauses, MemoEnabled, _MemoStrategy, BashCode) :-
    % Generate memo declaration (if enabled)
    (   MemoEnabled = true ->
        MemoDecl = 'declare -gA {{pred}}_memo'
    ;   MemoDecl = '# Memoization disabled (unique=false constraint)'
    ),
    TemplateLines = [
        "#!/bin/bash",
        "# {{pred}}/{{arity}} - linear recursive pattern (generic)",
        "",
        "# Memoization",
        "{{memo_decl}}",
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
    render_template(Template, [pred=PredStr, arity=Arity, memo_decl=MemoDecl], BashCode).

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
