:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% multicall_linear_recursion.pl - Multi-call linear recursion compiler
% Handles predicates with multiple independent recursive calls (e.g., fibonacci)
% Strategy: Memoize each independent call

:- module(multicall_linear_recursion, [
    is_multicall_linear_recursive/1,      % +Pred/Arity - Detect pattern
    can_compile_multicall_linear/1,       % +Pred/Arity - Check compilability
    compile_multicall_linear_recursion/3, % +Pred/Arity, +Options, -BashCode
    test_multicall_linear/0               % Test predicate
]).

:- use_module(library(lists)).
:- use_module('pattern_matchers').

%% ============================================
%% PATTERN DETECTION
%% ============================================

%% is_multicall_linear_recursive(+Pred/Arity)
%  Detect multi-call linear recursion pattern
%  Criteria:
%  1. Multiple recursive calls (2+) per clause
%  2. Calls are independent (no shared variables between them)
%  3. NOT structural decomposition (numeric input like fibonacci)
%  4. Arity = 2 (input/output pattern)
is_multicall_linear_recursive(Pred/Arity) :-
    % Must be binary
    Arity =:= 2,

    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Must have at least one recursive clause
    member(clause(_, SomeBody), Clauses),
    contains_call_to(SomeBody, Pred),

    % Check that head doesn't use structural decomposition
    % (fibonacci(N, F) is OK, tree_sum([V,L,R], Sum) is NOT)
    forall(member(clause(RecHead, RecBody), Clauses), (
        \+ contains_call_to(RecBody, Pred)  % Skip non-recursive clauses
        ;
        (   contains_call_to(RecBody, Pred),
            \+ has_structural_head_in_recursive(RecHead)
        )
    )),

    % Must have 2+ recursive calls
    member(clause(_, RecBody), Clauses),
    contains_call_to(RecBody, Pred),
    count_recursive_calls(RecBody, Pred, Count),
    Count >= 2,

    % Calls should be independent (no shared variables)
    % For now, we trust this - can add explicit check later
    true.

%% has_structural_head_in_recursive(+Head)
%  Check if head uses structural pattern (like [V,L,R])
has_structural_head_in_recursive(Head) :-
    Head =.. [_Pred|Args],
    Args = [FirstArg|_],
    is_structural_input(FirstArg).

%% is_structural_input(+Term)
%  Check if input argument is structural (list decomposition)
%  IMPORTANT: Must check nonvar first to avoid unifying with uninstantiated variables
is_structural_input(Term) :-
    nonvar(Term),
    (   Term = [_,_,_|_], !         % 3+ element list
    ;   Term = [_,_], !              % 2 element list
    ;   Term = [_|T],                % [H|T] with non-var T
        nonvar(T),
        T \= []
    ).

%% can_compile_multicall_linear(+Pred/Arity)
%  Check if we can compile this as multi-call linear recursion
can_compile_multicall_linear(Pred/Arity) :-
    is_multicall_linear_recursive(Pred/Arity).

%% ============================================
%% CODE GENERATION
%% ============================================

%% compile_multicall_linear_recursion(+Pred/Arity, +Options, -BashCode)
%  Compile multi-call linear recursion with memoization
%  Works for predicates like fibonacci that make multiple independent calls
compile_multicall_linear_recursion(Pred/Arity, Options, BashCode) :-
    format('  Compiling multi-call linear recursion: ~w/~w~n', [Pred, Arity]),

    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),

    % Get clauses
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Separate base and recursive cases
    % Need module qualification for partition's meta-call
    partition(multicall_linear_recursion:is_recursive_for_pred(Pred), Clauses, RecClauses, BaseClauses),

    % Determine memoization strategy from options
    (   member(unique(false), Options) ->
        MemoEnabled = false
    ;   MemoEnabled = true
    ),

    % Generate bash code
    generate_multicall_bash(PredStr, BaseClauses, RecClauses, MemoEnabled, BashCode).

%% is_recursive_for_pred(+Pred, +Clause)
is_recursive_for_pred(Pred, clause(_, Body)) :-
    contains_call_to(Body, Pred).

%% generate_multicall_bash(+PredStr, +BaseClauses, +RecClauses, +MemoEnabled, -BashCode)
%  Generate bash code for multi-call linear recursion
generate_multicall_bash(PredStr, BaseClauses, RecClauses, MemoEnabled, BashCode) :-
    % Extract base case info
    BaseClauses = [clause(BaseHead, _)|_],
    BaseHead =.. [_Pred, BaseInput, BaseOutput],

    % Extract recursive case info
    RecClauses = [clause(RecHead, RecBody)|_],
    RecHead =.. [Pred, _InputVar, _OutputVar],

    % Find recursive calls and aggregation
    % Use Pred (atom) not PredStr (string) for functor/3
    findall(Call, (extract_goal_from_body(RecBody, Call), functor(Call, Pred, 2)), RecCalls),
    find_aggregation_expr(RecBody, AggExpr),

    % Generate memo declaration
    (   MemoEnabled = true ->
        format(string(MemoDecl), '# Memoization table~ndeclare -gA ~w_memo~n~n', [PredStr])
    ;   MemoDecl = ''
    ),

    % Generate base case handling for ALL base clauses
    findall(BaseCaseCode,
        (   member(clause(BHead, _), BaseClauses),
            BHead =.. [_P, BInput, BOutput],
            format(string(BaseCaseCode), '    # Base case: ~w~n    if [[ "$n" == "~w" ]]; then~n        echo "~w"~n        return 0~n    fi~n',
                [BInput, BInput, BOutput])
        ),
        BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCase),

    % Generate memo check (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoCheck), '    # Check memo~n    if [[ -n "${~w_memo[$n]}" ]]; then~n        echo "${~w_memo[$n]}"~n        return 0~n    fi~n',
            [PredStr, PredStr])
    ;   MemoCheck = ''
    ),

    % Generate recursive calls (for fibonacci: fib(n-1) and fib(n-2))
    length(RecCalls, NumCalls),
    generate_recursive_calls(PredStr, NumCalls, RecCallsCode),

    % Generate aggregation (for fibonacci: F1 + F2)
    generate_aggregation(AggExpr, NumCalls, AggCode),

    % Generate memo store (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoStore), '    # Store in memo~n    ~w_memo["$n"]="$result"~n', [PredStr])
    ;   MemoStore = ''
    ),

    % Assemble complete function
    format(string(BashCode), '#!/bin/bash
# ~w - multi-call linear recursion with memoization
# Pattern: Multiple independent recursive calls (e.g., fibonacci)

~w~w() {
    local n="$1"

~w
~w
    # Recursive calls
~w
    # Aggregate results
~w

~w    echo "$result"
}
',
        [PredStr, MemoDecl, PredStr, BaseCase, MemoCheck, RecCallsCode, AggCode, MemoStore]).

%% extract_goal_from_body(+Body, -Goal)
extract_goal_from_body(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_).
extract_goal_from_body((A, _), Goal) :- extract_goal_from_body(A, Goal).
extract_goal_from_body((_, B), Goal) :- extract_goal_from_body(B, Goal).

%% find_aggregation_expr(+Body, -Expr)
%  Find the aggregation expression (e.g., F is F1 + F2)
find_aggregation_expr((A, _), Expr) :-
    A = (_ is Expr), !.
find_aggregation_expr((_, B), Expr) :-
    find_aggregation_expr(B, Expr).

%% generate_recursive_calls(+PredStr, +NumCalls, -Code)
%  Generate bash code for recursive calls
generate_recursive_calls(PredStr, 2, Code) :-
    format(string(Code), '    local result1=$( ~w $(( n - 1 )) )~n    local result2=$( ~w $(( n - 2 )) )~n',
        [PredStr, PredStr]).

%% generate_aggregation(+Expr, +NumCalls, -Code)
%  Generate aggregation code
generate_aggregation(_ + _, 2, Code) :-
    format(string(Code), '    local result=$(( result1 + result2 ))~n', []).
generate_aggregation(_ * _, 2, Code) :-
    format(string(Code), '    local result=$(( result1 * result2 ))~n', []).

%% ============================================
%% TESTS
%% ============================================

test_multicall_linear :-
    writeln('=== MULTI-CALL LINEAR RECURSION TESTS ==='),

    % Define fibonacci
    abolish_if_exists(test_fib/2),
    assertz(user:test_fib(0, 0)),
    assertz(user:test_fib(1, 1)),
    assertz(user:(test_fib(N, F) :-
        N > 1,
        N1 is N - 1,
        N2 is N - 2,
        test_fib(N1, F1),
        test_fib(N2, F2),
        F is F1 + F2
    )),

    % Test pattern detection
    writeln('Test 1: Detect fibonacci as multi-call linear'),
    (   is_multicall_linear_recursive(test_fib/2) ->
        writeln('  ✓ PASS - fibonacci detected')
    ;   writeln('  ✗ FAIL - should detect fibonacci')
    ),

    writeln('=== TESTS COMPLETE ===').

abolish_if_exists(Pred/Arity) :-
    functor(Head, Pred, Arity),
    (   current_predicate(Pred/Arity) ->
        abolish(Pred/Arity)
    ;   true
    ).
