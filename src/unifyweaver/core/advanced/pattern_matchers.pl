:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% pattern_matchers.pl - Enhanced recursion pattern detection
% Provides predicates for identifying specific recursion patterns

:- module(pattern_matchers, [
    is_tail_recursive_accumulator/2,      % +Pred/Arity, -AccInfo
    is_linear_recursive_streamable/1,     % +Pred/Arity
    is_mutual_transitive_closure/2,       % +Predicates, -BasePreds
    extract_accumulator_pattern/2,        % +Pred/Arity, -Pattern
    count_recursive_calls/2,              % +Body, -Count
    test_pattern_matchers/0         % Test predicate
]).

:- use_module(library(lists)).

%% is_tail_recursive_accumulator(+Pred/Arity, -AccInfo)
%  Detect tail recursion with accumulator pattern
%  AccInfo = acc_pattern(BaseCase, RecCase, AccPos)
is_tail_recursive_accumulator(Pred/Arity, AccInfo) :-
    functor(Head, Pred, Arity),
    % Use user:clause to access predicates from any module (including test predicates)
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Need at least one base case and one recursive case
    partition(is_recursive_for(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],

    % Check if recursive calls are in tail position
    forall(
        member(clause(_, RecBody), RecClauses),
        is_tail_call(RecBody, Pred)
    ),

    % Try to identify accumulator position
    identify_accumulator_position(Pred/Arity, BaseClauses, RecClauses, AccPos),

    AccInfo = acc_pattern(BaseClauses, RecClauses, AccPos).

%% is_recursive_for(+Pred, +Clause)
is_recursive_for(Pred, clause(_, Body)) :-
    contains_call_to(Body, Pred).

%% contains_call_to(+Body, +Pred)
contains_call_to(Body, Pred) :-
    extract_goal(Body, Goal),
    functor(Goal, Pred, _).

%% extract_goal(+Body, -Goal)
%  Extract individual goals from body
extract_goal(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_),
    \+ Goal = (_;_),
    \+ Goal = (_->_).
extract_goal((A, _), Goal) :- extract_goal(A, Goal).
extract_goal((_, B), Goal) :- extract_goal(B, Goal).
extract_goal((A; _), Goal) :- extract_goal(A, Goal).
extract_goal((_;B), Goal) :- extract_goal(B, Goal).
extract_goal((A -> _), Goal) :- extract_goal(A, Goal).
extract_goal((_ -> B), Goal) :- extract_goal(B, Goal).

%% is_tail_call(+Body, +Pred)
%  Check if recursive call is in tail position
is_tail_call(Body, Pred) :-
    last_goal_in_body(Body, LastGoal),
    functor(LastGoal, Pred, _).

%% last_goal_in_body(+Body, -Goal)
last_goal_in_body(Goal, Goal) :-
    compound(Goal),
    \+ Goal = (_,_).
last_goal_in_body((_, B), Goal) :-
    last_goal_in_body(B, Goal).

%% identify_accumulator_position(+Pred/Arity, +BaseClauses, +RecClauses, -AccPos)
%  Try to identify which argument is the accumulator
%  Heuristic: argument that appears in both base and recursive cases,
%  and is "threaded through" the recursion
identify_accumulator_position(Pred/Arity, BaseClauses, _RecClauses, AccPos) :-
    % For now, simple heuristic: if arity is 3, assume position 2 is accumulator
    % (common pattern: pred(Input, Accumulator, Result))
    (   Arity =:= 3 ->
        AccPos = 2
    ;   Arity =:= 2 ->
        % For arity 2, could be either position
        % Try to detect by looking for base case pattern
        detect_acc_in_binary(Pred, BaseClauses, AccPos)
    ;   % Unknown pattern
        AccPos = unknown
    ).

%% detect_acc_in_binary(+Pred, +BaseClauses, -AccPos)
detect_acc_in_binary(_Pred, BaseClauses, AccPos) :-
    % Look for base case like: pred(Base, Base) or pred([], Result)
    member(clause(Head, _), BaseClauses),
    Head =.. [_|Args],
    (   Args = [A, A] ->
        % Pattern: pred(X, X) - accumulator is position 2
        AccPos = 2
    ;   Args = [[], _] ->
        % Pattern: pred([], Result) - result is position 2
        AccPos = 2
    ;   AccPos = 1
    ).

%% is_linear_recursive_streamable(+Pred/Arity)
%  Check if linear recursion can be compiled to stream processing
%  Linear recursion: exactly one recursive call per clause
is_linear_recursive_streamable(Pred/Arity) :-
    functor(Head, Pred, Arity),
    % Use user:clause to access predicates from any module (including test predicates)
    findall(Body, user:clause(Head, Body), Bodies),

    % All recursive clauses must have exactly one recursive call
    forall(
        (   member(Body, Bodies),
            contains_call_to(Body, Pred)
        ),
        (   count_recursive_calls(Body, Pred, 1)
        )
    ),

    % Additional check: should operate on structured data (lists, trees, etc.)
    % For now, we check arity and assume binary predicates aren't streamable this way
    Arity > 1.

%% count_recursive_calls(+Body, +Pred, -Count)
count_recursive_calls(Body, Pred, Count) :-
    findall(1,
        (   extract_goal(Body, Goal),
            functor(Goal, Pred, _)
        ),
        Ones),
    length(Ones, Count).

%% count_recursive_calls(+Body, -Count)
%  Count all recursive calls in body (any predicate)
count_recursive_calls(Body, Count) :-
    findall(1,
        (   extract_goal(Body, Goal),
            compound(Goal)
        ),
        Ones),
    length(Ones, Count).

%% is_mutual_transitive_closure(+Predicates, -BasePreds)
%  Check if a group of predicates forms a mutual transitive closure
%  This is a generalization of single-predicate transitive closure
is_mutual_transitive_closure(Predicates, BasePreds) :-
    % Each predicate should have:
    % 1. A base case that calls some base predicate
    % 2. A recursive case that chains through the group
    forall(
        member(Pred/Arity, Predicates),
        has_closure_pattern(Pred/Arity, Predicates)
    ),

    % Collect all base predicates referenced
    findall(BasePred,
        (   member(Pred/Arity, Predicates),
            functor(Head, Pred, Arity),
            % Use user:clause to access predicates from any module (including test predicates)
            user:clause(Head, Body),
            \+ contains_call_to_any(Body, Predicates),
            extract_goal(Body, Goal),
            compound(Goal),
            functor(Goal, BasePred, _)
        ),
        BasePredsWithDups),
    sort(BasePredsWithDups, BasePreds).

%% has_closure_pattern(+Pred/Arity, +Group)
has_closure_pattern(Pred/Arity, _Group) :-
    functor(Head, Pred, Arity),
    % Use user:clause to access predicates from any module (including test predicates)
    user:clause(Head, _Body),
    % Simplified check - more sophisticated logic needed
    true.

%% contains_call_to_any(+Body, +Predicates)
contains_call_to_any(Body, Predicates) :-
    member(Pred/_Arity, Predicates),
    contains_call_to(Body, Pred).

%% extract_accumulator_pattern(+Pred/Arity, -Pattern)
%  Extract detailed accumulator pattern information
%  Pattern = pattern(InitValue, StepOp, FinalOp)
extract_accumulator_pattern(Pred/Arity, Pattern) :-
    functor(Head, Pred, Arity),

    % Find base case to get initial value
    % Use user:clause to access predicates from any module (including test predicates)
    user:clause(Head, BaseBody),
    \+ contains_call_to(BaseBody, Pred),
    extract_base_pattern(Head, BaseBody, InitValue),

    % Find recursive case to get step operation
    user:clause(Head, RecBody),
    contains_call_to(RecBody, Pred),
    extract_step_pattern(RecBody, Pred, StepOp),

    Pattern = pattern(InitValue, StepOp, unify).

%% extract_base_pattern(+Head, +Body, -InitValue)
extract_base_pattern(Head, Body, InitValue) :-
    Head =.. [_|Args],
    (   Body = true ->
        % Fact - last arg might be the value
        last(Args, InitValue)
    ;   Body = (_ = InitValue) ->
        % Unification
        true
    ;   % Try to extract from body
        InitValue = unknown
    ).

%% extract_step_pattern(+Body, +Pred, -StepOp)
extract_step_pattern(Body, Pred, StepOp) :-
    % Look for arithmetic operations
    (   extract_goal(Body, Goal),
        Goal =.. [is, _, Expr],
        \+ contains_call_to(Goal, Pred) ->
        StepOp = arithmetic(Expr)
    ;   StepOp = unknown
    ).

%% ============================================
%% TESTS
%% ============================================

test_pattern_matchers :-
    writeln('=== PATTERN MATCHERS TESTS ==='),

    % Clear test predicates
    catch(abolish(test_count/3), _, true),
    catch(abolish(test_length/2), _, true),
    catch(abolish(test_sum/3), _, true),

    % Test 1: Tail recursion with accumulator
    writeln('Test 1: Tail recursive accumulator (count)'),
    assertz(user:(test_count([], Acc, Acc))),
    assertz(user:(test_count([_|T], Acc, N) :- Acc1 is Acc + 1, test_count(T, Acc1, N))),
    (   is_tail_recursive_accumulator(test_count/3, AccInfo1) ->
        format('  ✓ PASS - detected tail recursion with accumulator~n  Info: ~w~n', [AccInfo1])
    ;   writeln('  ✗ FAIL - should detect tail recursive accumulator')
    ),

    % Test 2: Linear recursion (not tail recursive)
    writeln('Test 2: Linear recursion (length)'),
    assertz(user:(test_length([], 0))),
    assertz(user:(test_length([_|T], N) :- test_length(T, N1), N is N1 + 1)),
    (   is_linear_recursive_streamable(test_length/2) ->
        writeln('  ✓ PASS - detected linear recursion')
    ;   writeln('  ✗ FAIL - should detect linear recursion')
    ),

    % Test 3: Count recursive calls
    writeln('Test 3: Count recursive calls'),
    Body1 = (test_count(T, Acc1, N)),
    count_recursive_calls(Body1, test_count, Count1),
    format('  Recursive calls in "test_count(T, Acc1, N)": ~w~n', [Count1]),
    (   Count1 =:= 1 ->
        writeln('  ✓ PASS - counted 1 recursive call')
    ;   writeln('  ✗ FAIL - should count 1 call')
    ),

    % Test 4: Extract accumulator pattern
    writeln('Test 4: Extract accumulator pattern (sum)'),
    assertz(user:(test_sum([], Acc, Acc))),
    assertz(user:(test_sum([H|T], Acc, Sum) :- Acc1 is Acc + H, test_sum(T, Acc1, Sum))),
    (   extract_accumulator_pattern(test_sum/3, Pattern) ->
        format('  ✓ PASS - extracted pattern: ~w~n', [Pattern])
    ;   writeln('  ✗ FAIL - should extract accumulator pattern')
    ),

    writeln('=== PATTERN MATCHERS TESTS COMPLETE ===').
