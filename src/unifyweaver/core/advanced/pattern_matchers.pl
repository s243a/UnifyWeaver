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
    extract_base_pattern/3,               % +Head, +Body, -InitValue (helper for extract_accumulator_pattern)
    extract_step_pattern/3,               % +Body, +PredName, -StepOp (helper for extract_accumulator_pattern)
    count_recursive_calls/2,              % +Body, -Count
    contains_call_to/2,                   % +Body, +Pred (helper for various functions)
    % Linear recursion exclusion predicates
    forbid_linear_recursion/1,            % +Pred/Arity - Mark as not linear
    is_forbidden_linear_recursion/1,      % +Pred/Arity - Check if forbidden
    clear_linear_recursion_forbid/1,      % +Pred/Arity - Clear forbid
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
%  Check if linear recursion can be compiled with memoization
%  Linear recursion: one OR MORE recursive calls per clause, with independent calls
%  Key criteria:
%  1. Body is conjunction (AND chain)
%  2. Variables are unique (no inter-call data flow)
%  3. Recursive call arguments are pre-computed (order independent)
%  4. No side effects (assumed for now)
%  5. NOT structural decomposition (no compound patterns in head)
%  6. NOT forbidden (via forbid_linear_recursion or ordered constraint)
is_linear_recursive_streamable(Pred/Arity) :-
    % Check if forbidden first (fast fail)
    \+ is_forbidden_linear_recursion(Pred/Arity),

    functor(Head, Pred, Arity),
    % Use user:clause to access predicates from any module (including test predicates)
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Must have at least one recursive clause
    member(clause(_, SomeBody), Clauses),
    contains_call_to(SomeBody, Pred),

    % Check that head doesn't use structural decomposition (like [V,L,R])
    forall(member(clause(RecHead, RecBody), Clauses), (
        \+ contains_call_to(RecBody, Pred)  % Skip non-recursive clauses
        ;
        (   contains_call_to(RecBody, Pred),
            \+ has_structural_head_pattern(RecHead)
        )
    )),

    % For each recursive clause, verify linear pattern with independent calls
    forall(
        (   member(clause(_, RecBody), Clauses),
            contains_call_to(RecBody, Pred)
        ),
        (   % Check that recursive call arguments are pre-computed (order independent)
            recursive_args_are_precomputed(RecBody, Pred)
        )
    ),

    % Additional check: should operate on structured data (lists, trees, etc.)
    % For now, we check arity and assume binary predicates aren't streamable this way
    Arity > 1.

%% has_structural_head_pattern(+Head)
%  Check if head uses tree-like structural decomposition
%  e.g., tree_sum([V,L,R], Sum) has structural pattern [V,L,R]
%  BUT list_length([H|T], N) is NOT tree-like - it's linear list decomposition
%
%  Key distinction:
%  - Tree pattern: [V, L, R] - multi-element list (3+ elements representing a tree node)
%  - List pattern: [H|T] or [_|T] - simple cons cell (standard list decomposition)
%
%  Strategy: Only reject multi-element list patterns like [V,L,R] (tree nodes)
%  Allow simple cons patterns [H|T] (linear list decomposition)
has_structural_head_pattern(Head) :-
    Head =.. [_|Args],
    member(Arg, Args),
    is_tree_node_pattern(Arg).

%% is_tree_node_pattern(+Term)
%  Check if a term is a tree node pattern (multi-element list, not cons)
is_tree_node_pattern(Arg) :-
    compound(Arg),
    Arg \= (_ is _),        % Not an 'is' expression
    Arg \= (_ =.. _),       % Not univ
    \+ is_simple_cons(Arg), % Not a simple cons pattern
    !.

%% is_simple_cons(+Term)
%  Check if term is a simple cons pattern: [H|T]
%  Returns true for patterns like [_|T], [H|_], [H|T]
%  Returns false for [V,L,R] or other compound terms
is_simple_cons([_|Tail]) :-
    % Must be a cons with tail as a variable (not multi-element list)
    (   var(Tail) -> true          % [H|T] where T is variable
    ;   Tail = [] -> true          % [H] - single element list (edge case)
    ;   Tail = [_|_] -> fail       % [H,X,...] - multi-element, NOT simple cons
    ;   fail                       % Other structure
    ).

%% recursive_args_are_precomputed(+Body, +Pred)
%  Check that all recursive call arguments are computed/bound before any recursive calls
%  This ensures order independence - all arguments are "pre-computed" scalars/values
%  NOT structural parts from pattern matching
recursive_args_are_precomputed(Body, Pred) :-
    % Extract all goals in order (approximate - good enough for conjunction chains)
    findall(Goal, extract_goal(Body, Goal), Goals),

    % Find all recursive calls
    findall(RecCall, (
        member(RecCall, Goals),
        functor(RecCall, Pred, _)
    ), RecCalls),

    % Find all 'is' expressions (these compute values)
    findall(Var, (
        member(Goal, Goals),
        Goal =.. [is, Var, _]
    ), ComputedVars),

    % For each recursive call, check that INPUT arguments (typically first arg) are computed
    % Output arguments are allowed to be free variables
    forall(member(RecCall, RecCalls), (
        RecCall =.. [_|Args],
        % Get first argument (input position)
        (   Args = [FirstArg|_] ->
            % First arg should be computed (from 'is') or a constant
            % NOT a variable from head pattern matching
            (   var(FirstArg) ->
                % Variable - must be in ComputedVars (from 'is' expression)
                member(FirstArg, ComputedVars)
            ;   true  % Ground term/constant is OK
            )
        ;   true  % No arguments
        )
    )).

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
extract_accumulator_pattern(PredName/Arity, Pattern) :-
    % Find base case to get initial value
    % Use user:clause to access predicates from any module (including test predicates)
    functor(BaseHead, PredName, Arity),
    user:clause(BaseHead, BaseBody),
    \+ contains_call_to(BaseBody, PredName),
    extract_base_pattern(BaseHead, BaseBody, InitValue),

    % Find recursive case to get step operation (use separate head variable!)
    functor(RecHead, PredName, Arity),
    user:clause(RecHead, RecBody),
    contains_call_to(RecBody, PredName),
    extract_step_pattern(RecBody, PredName, StepOp),

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

%% extract_step_pattern(+Body, +PredName, -StepOp)
extract_step_pattern(Body, PredName, StepOp) :-
    % Look for arithmetic operations
    % Extract all goals and find one that is arithmetic (is/2)
    % but doesn't call the predicate itself
    (   extract_goal(Body, Goal),
        Goal =.. [is, _, Expr],
        \+ contains_call_to(Goal, PredName) ->  % Check if this goal calls the predicate
        StepOp = arithmetic(Expr)
    ;   StepOp = unknown
    ).

%% ============================================
%% LINEAR RECURSION EXCLUSION
%% ============================================
%
% This system allows marking predicates that should NOT be compiled
% as linear recursion, even if they match the linear pattern.
%
% Use cases:
% 1. Predicates with ordered constraints (order dependencies)
% 2. Predicates requiring graph recursion (structural traversal)
% 3. Predicates with side effects
%
% Integration with constraint system:
% - Predicates with ordered=true are automatically forbidden
% - Manual forbid/unforbid available for other cases

:- dynamic forbidden_linear_recursion/1.

%% forbid_linear_recursion(+Pred/Arity)
%  Mark a predicate as forbidden for linear recursion compilation
%  This predicate will NOT match is_linear_recursive_streamable/1
%  even if it otherwise meets the criteria.
%
%  Example:
%    forbid_linear_recursion(my_ordered_pred/2)
%
forbid_linear_recursion(Pred/Arity) :-
    retractall(forbidden_linear_recursion(Pred/Arity)),
    assertz(forbidden_linear_recursion(Pred/Arity)).

%% is_forbidden_linear_recursion(+Pred/Arity)
%  Check if a predicate is forbidden for linear recursion compilation
%  Returns true if:
%  1. Explicitly forbidden via forbid_linear_recursion/1
%  2. Has ordered constraint (unordered=false)
%
is_forbidden_linear_recursion(Pred/Arity) :-
    (   % Check explicit forbid
        forbidden_linear_recursion(Pred/Arity) ->
        true
    ;   % Check if predicate has ordered constraint
        % Try to load constraint_analyzer if available
        catch(
            (   current_module(constraint_analyzer) ->
                constraint_analyzer:get_constraints(Pred/Arity, Constraints),
                member(unordered(false), Constraints)
            ;   fail  % Module not loaded
            ),
            _,
            fail  % Error loading - assume not forbidden
        )
    ).

%% clear_linear_recursion_forbid(+Pred/Arity)
%  Remove forbid marking from a predicate
%  Note: This does NOT override constraint-based forbidding
%
clear_linear_recursion_forbid(Pred/Arity) :-
    retractall(forbidden_linear_recursion(Pred/Arity)).

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

    % Test 5: Linear recursion forbid/unforbid
    writeln('Test 5: Forbid linear recursion'),
    catch(abolish(test_fib/2), _, true),
    assertz(user:(test_fib(0, 0))),
    assertz(user:(test_fib(1, 1))),
    assertz(user:(test_fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, test_fib(N1, F1), test_fib(N2, F2), F is F1 + F2)),

    % Should match linear pattern initially
    (   is_linear_recursive_streamable(test_fib/2) ->
        writeln('  ✓ PASS - fibonacci matches linear pattern')
    ;   writeln('  ✗ FAIL - fibonacci should match linear')
    ),

    % Forbid it
    forbid_linear_recursion(test_fib/2),
    (   \+ is_linear_recursive_streamable(test_fib/2) ->
        writeln('  ✓ PASS - forbidden fibonacci does not match linear')
    ;   writeln('  ✗ FAIL - forbidden should not match')
    ),

    % Check is_forbidden works
    (   is_forbidden_linear_recursion(test_fib/2) ->
        writeln('  ✓ PASS - is_forbidden detects forbid')
    ;   writeln('  ✗ FAIL - is_forbidden should return true')
    ),

    % Unforbid it
    clear_linear_recursion_forbid(test_fib/2),
    (   is_linear_recursive_streamable(test_fib/2) ->
        writeln('  ✓ PASS - unforbidden fibonacci matches linear again')
    ;   writeln('  ✗ FAIL - unforbidden should match again')
    ),

    writeln('=== PATTERN MATCHERS TESTS COMPLETE ===').
