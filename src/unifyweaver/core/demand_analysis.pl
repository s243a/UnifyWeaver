:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% demand_analysis.pl — Target-agnostic demand-driven pruning analysis
%
% Detects recursive predicates that benefit from demand-driven pruning:
% predicates where backward reachability from a target fact can eliminate
% branches that can never produce results.
%
% Produces demand_spec(...) terms that targets consume to emit guards.
% Does NOT emit target-specific code — follows the same pattern as
% purity_certificate.pl (analysis produces terms, targets consume them).
%
% The canonical example:
%
%   category_ancestor(Cat, Root, Hops, Visited) :-
%       category_parent(Cat, Mid),             % edge predicate
%       \+ member(Mid, Visited),
%       category_ancestor(Mid, Root, H, [Mid|Visited]),
%       Hops is H + 1.
%
% Demand analysis detects that `category_parent/2` is the step relation,
% and that backward reachability from root_category/1 values can prune
% ~80% of branches — only categories that CAN reach a root are worth
% exploring. A target emits a guard (e.g., `can_reach_root(Mid)`) after
% the edge call, before the recursive call.
%
% See docs/proposals/PROLOG_TARGET_DEMAND_ANALYSIS.md for motivation.

:- module(demand_analysis, [
    %% Detection — does this predicate benefit from demand analysis?
    detect_demand_eligible/4,         % +Pred, +Arity, +Clauses, -DemandSpec

    %% Metadata extraction — for target emitters
    demand_spec_edge_pred/2,          % +DemandSpec, -EdgePred/Arity
    demand_spec_target_arg/2,         % +DemandSpec, -ArgPosition (1-based)
    demand_spec_input_positions/2,    % +DemandSpec, -Positions (0-based)
    demand_spec_guard_points/2,       % +DemandSpec, -GuardPoints
    demand_spec_edge_direction/2,     % +DemandSpec, -direction(FromPos, ToPos)

    %% Convenience
    is_demand_eligible/3              % +Pred, +Arity, +Clauses
]).

:- use_module(library(lists)).

%% ========================================================================
%% Primary detection entry point
%% ========================================================================

%% detect_demand_eligible(+Pred, +Arity, +Clauses, -DemandSpec)
%  Clauses = list of Head-Body pairs (same format as recursive_kernel_detection).
%  Succeeds if Pred/Arity is a recursive transitive predicate with:
%    1. An input mode declaration (user:mode/1) with at least one input arg
%    2. A binary edge predicate called before the recursive self-call
%    3. No aggregate goals in the prefix before the recursive call
%  Returns a demand_spec(...) term that targets can consume.
detect_demand_eligible(Pred, Arity, Clauses, DemandSpec) :-
    % 1. Must have input modes declared
    read_mode_declaration(Pred, Arity, Modes),
    has_input(Modes),
    input_positions(Modes, InputPositions),
    InputPositions \= [],
    % 2. Separate into base and recursive clauses
    partition(is_recursive_clause(Pred, Arity), Clauses, RecClauses, _BaseClauses),
    RecClauses \= [],
    % 3. All recursive clauses must be safe (no aggregates in prefix)
    forall(
        member(_-Body, RecClauses),
        safe_prefix(Pred, Arity, Body)
    ),
    % 4. Extract edge predicate from recursive clause
    RecClauses = [_RecHead-RecBody|_],
    extract_edge_pred(Pred, Arity, RecBody, EdgePred/EdgeArity),
    % 5. Determine which argument is the "target" (the one that reaches
    %    the goal, typically bound to root_category). This is the arg
    %    position that stays constant across recursion (the Ancestor/Root).
    detect_target_arg(Pred, Arity, Clauses, TargetArgPos),
    % 6. Determine edge direction: which arg of the edge predicate is
    %    the "from" (current node) and which is "to" (next hop).
    detect_edge_direction(Pred, Arity, RecBody, EdgePred/EdgeArity, EdgeDirection),
    % 7. Find guard insertion points (after edge call, before recursive call)
    findall(
        guard_point(ClauseIdx, GoalIdx, GuardVar),
        (   nth0(ClauseIdx, RecClauses, _CH-CB),
            find_guard_point(Pred, Arity, CB, EdgePred/EdgeArity, GoalIdx, GuardVar)
        ),
        GuardPoints
    ),
    GuardPoints \= [],
    % Assemble the spec
    DemandSpec = demand_spec(
        Pred/Arity,
        EdgePred/EdgeArity,
        TargetArgPos,
        InputPositions,
        GuardPoints,
        EdgeDirection
    ).

%% is_demand_eligible(+Pred, +Arity, +Clauses)
%  Convenience: just checks eligibility without returning the spec.
is_demand_eligible(Pred, Arity, Clauses) :-
    detect_demand_eligible(Pred, Arity, Clauses, _).

%% ========================================================================
%% Spec accessors — targets use these to extract what they need
%% ========================================================================

demand_spec_edge_pred(demand_spec(_, EP, _, _, _, _), EP).
demand_spec_target_arg(demand_spec(_, _, TA, _, _, _), TA).
demand_spec_input_positions(demand_spec(_, _, _, IP, _, _), IP).
demand_spec_guard_points(demand_spec(_, _, _, _, GP, _), GP).
demand_spec_edge_direction(demand_spec(_, _, _, _, _, ED), ED).

%% ========================================================================
%% Mode declaration reading
%% ========================================================================

%% read_mode_declaration(+Pred, +Arity, -Modes)
%  Reads user:mode/1 declarations. Also accepts demand_mode/1 as an
%  alternative for explicit demand configuration.
%  Modes = list of input/output atoms matching the predicate's arity.
read_mode_declaration(Pred, Arity, Modes) :-
    current_predicate(user:mode/1),
    user:mode(ModeSpec),
    compound(ModeSpec),
    ModeSpec =.. [Pred|ModeArgs],
    length(ModeArgs, Arity),
    maplist(mode_sym, ModeArgs, Modes),
    !.

mode_sym(+, input).
mode_sym(-, output).
mode_sym(?, any).

has_input(Modes) :- member(input, Modes).

input_positions(Modes, Positions) :-
    findall(Pos, nth0(Pos, Modes, input), Positions).

%% ========================================================================
%% Clause classification
%% ========================================================================

%% is_recursive_clause(+Pred, +Arity, +HeadBody)
%  True if the clause body contains a call to Pred/Arity.
is_recursive_clause(Pred, Arity, _Head-Body) :-
    body_to_goals(Body, Goals),
    member(Goal, Goals),
    nonvar(Goal),
    strip_meta(Goal, Plain),
    functor(Plain, Pred, Arity),
    !.

%% safe_prefix(+Pred, +Arity, +Body)
%  The goals before the recursive call must not contain aggregates.
safe_prefix(Pred, Arity, Body) :-
    body_to_goals(Body, Goals),
    append(Prefix, [RecCall|_], Goals),
    nonvar(RecCall),
    strip_meta(RecCall, PlainRec),
    functor(PlainRec, Pred, Arity),
    !,
    \+ (member(G, Prefix), is_aggregate_goal(G)).

%% ========================================================================
%% Edge predicate extraction
%% ========================================================================

%% extract_edge_pred(+Pred, +Arity, +Body, -EdgePred/EdgeArity)
%  Finds a binary fact call in the body that appears BEFORE the recursive
%  call and shares a variable with the head (the "current node" variable).
%  The edge predicate is the step relation in the transitive closure.
extract_edge_pred(Pred, Arity, Body, EdgePred/EdgeArity) :-
    body_to_goals(Body, Goals),
    % Find the recursive call position
    append(Prefix, [RecCall|_], Goals),
    nonvar(RecCall),
    strip_meta(RecCall, PlainRec),
    functor(PlainRec, Pred, Arity),
    !,
    % Find a binary call in the prefix that is NOT negation, NOT a builtin
    member(Goal, Prefix),
    nonvar(Goal),
    strip_meta(Goal, Plain),
    functor(Plain, EdgePred, EdgeArity),
    EdgeArity == 2,
    EdgePred \= member,
    \+ is_builtin_pred(EdgePred/EdgeArity),
    \+ is_negation(Goal).

%% ========================================================================
%% Target argument detection
%% ========================================================================

%% detect_target_arg(+Pred, +Arity, +Clauses, -ArgPos)
%  The "target" argument is the one that stays constant across recursion
%  (passed unchanged from head to recursive call). In category_ancestor/4,
%  this is arg 2 (Ancestor/Root). In closure(+, -), this is arg 2 (Target).
%  ArgPos is 1-based. The target arg can be either input or output —
%  what matters is that it's invariant across the recursion.
detect_target_arg(Pred, Arity, Clauses, ArgPos) :-
    % Look at a recursive clause
    member(Head-Body, Clauses),
    body_to_goals(Body, Goals),
    member(RecCall, Goals),
    nonvar(RecCall),
    strip_meta(RecCall, PlainRec),
    functor(PlainRec, Pred, Arity),
    !,
    % Find which head arg is passed unchanged to the recursive call
    Head =.. [_|HeadArgs],
    PlainRec =.. [_|RecArgs],
    between(1, Arity, ArgPos),
    nth1(ArgPos, HeadArgs, HeadArg),
    nth1(ArgPos, RecArgs, RecArg),
    HeadArg == RecArg,
    % Must not be the "current node" arg (arg 1, which changes at each step)
    ArgPos > 1,
    !.

%% ========================================================================
%% Edge direction detection
%% ========================================================================

%% detect_edge_direction(+Pred, +Arity, +Body, +EdgePred/EA, -Direction)
%  Determines which argument of the edge predicate is the "from" node
%  (shared with the head's first/current arg) and which is the "to" node
%  (shared with the recursive call's first/current arg).
%  Direction = direction(FromArgPos, ToArgPos) where positions are 1-based
%  within the edge predicate.
detect_edge_direction(Pred, Arity, Body, EdgePred/EdgeArity, direction(FromPos, ToPos)) :-
    body_to_goals(Body, Goals),
    % Find the edge call and recursive call
    member(EdgeCall, Goals),
    nonvar(EdgeCall),
    strip_meta(EdgeCall, PlainEdge),
    functor(PlainEdge, EdgePred, EdgeArity),
    member(RecCall, Goals),
    nonvar(RecCall),
    strip_meta(RecCall, PlainRec),
    functor(PlainRec, Pred, Arity),
    !,
    PlainEdge =.. [_|EdgeArgs],
    PlainRec =.. [_|RecArgs],
    % The "to" arg of the edge shares a variable with arg 1 of the recursive call
    nth1(ToPos, EdgeArgs, ToVar),
    nth1(1, RecArgs, RecFirstArg),
    ToVar == RecFirstArg,
    % The "from" is the other arg
    (ToPos =:= 1 -> FromPos = 2 ; FromPos = 1),
    !.
% Fallback: assume from=1, to=2 (child→parent convention)
detect_edge_direction(_Pred, _Arity, _Body, _Edge, direction(1, 2)).

%% ========================================================================
%% Guard insertion point detection
%% ========================================================================

%% find_guard_point(+Pred, +Arity, +Body, +EdgePred/EA, -GoalIdx, -GuardVar)
%  Finds where to insert a demand guard in the clause body. The guard goes
%  AFTER the edge predicate call (which introduces the "next hop" variable)
%  and BEFORE the recursive call. GuardVar is the variable that should be
%  checked against the demand set (the "to" node of the edge).
find_guard_point(Pred, Arity, Body, EdgePred/EdgeArity, GuardIdx, GuardVar) :-
    body_to_goals(Body, Goals),
    % Find edge call index
    nth0(EdgeIdx, Goals, EdgeCall),
    nonvar(EdgeCall),
    strip_meta(EdgeCall, PlainEdge),
    functor(PlainEdge, EdgePred, EdgeArity),
    % Find recursive call index (must come after edge call)
    nth0(RecIdx, Goals, RecCall),
    RecIdx > EdgeIdx,
    nonvar(RecCall),
    strip_meta(RecCall, PlainRec),
    functor(PlainRec, Pred, Arity),
    !,
    % Guard goes right after the edge call
    GuardIdx is EdgeIdx + 1,
    % The variable to guard is the "to" arg of the edge predicate —
    % the variable that will be passed to the recursive call's first arg.
    PlainEdge =.. [_|EdgeArgs],
    PlainRec =.. [_|RecArgs],
    nth1(1, RecArgs, RecFirstArg),
    (   member(Var, EdgeArgs), Var == RecFirstArg
    ->  GuardVar = Var
    ;   % Fallback: guard the second arg of the edge pred
        nth1(2, EdgeArgs, GuardVar)
    ).

%% ========================================================================
%% Utility predicates
%% ========================================================================

%% body_to_goals(+Body, -Goals)
%  Flatten a conjunction into a list of goals.
body_to_goals((A, B), Goals) :-
    !,
    body_to_goals(A, GA),
    body_to_goals(B, GB),
    append(GA, GB, Goals).
body_to_goals(true, []) :- !.
body_to_goals(Goal, [Goal]).

%% strip_meta(+Term, -Plain)
%  Strip module qualifiers and negation wrappers.
strip_meta(_M:T, Plain) :- !, strip_meta(T, Plain).
strip_meta(T, T).

%% is_negation(+Goal)
is_negation(\+ _).
is_negation(not(_)).

%% is_aggregate_goal(+Goal)
is_aggregate_goal(Goal) :-
    nonvar(Goal),
    strip_meta(Goal, Plain),
    functor(Plain, F, _),
    aggregate_functor(F).

aggregate_functor(aggregate_all).
aggregate_functor(aggregate).
aggregate_functor(findall).
aggregate_functor(bagof).
aggregate_functor(setof).

%% is_builtin_pred(+Pred/Arity)
%  Predicates that are definitely not edge relations.
is_builtin_pred(member/2).
is_builtin_pred(length/2).
is_builtin_pred(is/2).
is_builtin_pred((/)/2).  % arithmetic
is_builtin_pred((<)/2).
is_builtin_pred((>)/2).
is_builtin_pred((=<)/2).
is_builtin_pred((>=)/2).
is_builtin_pred((=:=)/2).
is_builtin_pred((=\=)/2).
is_builtin_pred((=)/2).
is_builtin_pred((\=)/2).
is_builtin_pred((!)/0).
is_builtin_pred(true/0).
is_builtin_pred(fail/0).
is_builtin_pred(max_depth/1).
is_builtin_pred(dimension_n/1).
