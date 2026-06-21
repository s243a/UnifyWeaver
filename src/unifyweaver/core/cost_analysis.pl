% cost_analysis.pl
%
% Static cost estimation for Prolog goals, clause bodies, and predicates.
%
% Purpose. Several lowering decisions are currently gated by hard-coded magic
% numbers (the T6 `t6_min_clauses(8)`, the T4/T5 thresholds) or by a *runtime*
% probe (the T7 parallel substrate's `gated_collect`, which samples a few
% branches per call). This module is the shared, compile-time cost machinery
% those gates should consume instead: given a goal / clause body / predicate it
% returns a relative cost estimate and a coarse *tier* (trivial … expensive …
% recursive), so a gate can ask "is the per-branch work of this aggregate large
% enough to parallelise?" or "is this clause body cheap enough that the host
% compiler will fold it?" without a magic number.
%
% Model. Cost is `cost(Weight, Boundedness)`:
%   - Weight       — non-negative relative units (builtins weighted by a small,
%                    overridable table; conjunction adds, disjunction takes the
%                    worst single-solution branch).
%   - Boundedness  — `bounded` or `unbounded`. A goal is unbounded when it
%                    (transitively) calls a recursive predicate or enumerates an
%                    open-ended aggregate (findall/bagof/setof/forall/…): its
%                    cost grows with input, which is exactly the "expensive,
%                    worth parallelising" signal.
%
% Recursion is detected once per program via the call graph (a predicate that is
% reachable from itself is recursive → unbounded); the remaining predicates form
% a DAG whose costs are computed bottom-up with memoisation. The cost table is
% `multifile`/`dynamic` so callers can tune or extend it.

:- module(cost_analysis, [
    build_cost_model/3,        % +Module, +Preds, -Model
    build_cost_model/2,        % +Module, -Model           (all predicates)
    predicate_cost/3,          % +PI, +Model, -Cost
    goal_cost/3,               % +Goal, +Model, -Cost
    clause_body_cost/3,        % +Body, +Model, -Cost
    cost_tier/2,               % +Cost, -Tier
    goal_cost_tier/3,          % +Goal, +Model, -Tier
    predicate_cost_tier/3,     % +PI, +Model, -Tier
    cost_weight/2,             % +Cost, -Weight
    cost_bounded/2,            % +Cost, -Boundedness
    recursive_predicate/2,     % +PI, +Model
    builtin_cost/2,            % ?Functor/Arity, ?Weight    (multifile, overridable)
    cost_tier_threshold/2      % ?Name, ?Value              (multifile, overridable)
]).

:- use_module(library(assoc)).
:- use_module(library(lists)).

:- multifile builtin_cost/2.
:- dynamic   builtin_cost/2.
:- multifile cost_tier_threshold/2.
:- dynamic   cost_tier_threshold/2.

% ============================================================================
% Tunables
% ============================================================================

% Per-predicate-call overhead (added on top of the callee's body cost).
call_overhead(1).
% Default weight for an unrecognised builtin / external predicate.
default_builtin_weight(2).
% Base weight charged for entering a recursive predicate.
recursive_base_weight(5).

% Tier thresholds on Weight (only meaningful for `bounded` costs).
% Overridable by asserting cost_tier_threshold/2.
threshold(Name, Value) :-
    ( cost_tier_threshold(Name, V) -> Value = V ; default_threshold(Name, Value) ).
default_threshold(cheap, 4).        % weight =< cheap  -> trivial/cheap
default_threshold(moderate, 25).    % weight =< moderate -> moderate, else expensive

% ============================================================================
% Default builtin cost table  (relative units; overridable)
% ============================================================================

builtin_cost((true)/0, 0).
builtin_cost((fail)/0, 0).
builtin_cost((false)/0, 0).
builtin_cost((!)/0, 0).
% unification / comparison
builtin_cost((=)/2, 1).
builtin_cost((\=)/2, 1).
builtin_cost((==)/2, 1).
builtin_cost((\==)/2, 1).
builtin_cost((@<)/2, 1).
builtin_cost((@>)/2, 1).
builtin_cost((@=<)/2, 1).
builtin_cost((@>=)/2, 1).
builtin_cost(compare/3, 1).
% arithmetic
builtin_cost(is/2, 2).
builtin_cost((<)/2, 2).
builtin_cost((>)/2, 2).
builtin_cost((=<)/2, 2).
builtin_cost((>=)/2, 2).
builtin_cost((=:=)/2, 2).
builtin_cost((=\=)/2, 2).
builtin_cost(succ/2, 1).
builtin_cost(plus/3, 1).
% type checks
builtin_cost(var/1, 1).
builtin_cost(nonvar/1, 1).
builtin_cost(atom/1, 1).
builtin_cost(atomic/1, 1).
builtin_cost(number/1, 1).
builtin_cost(integer/1, 1).
builtin_cost(float/1, 1).
builtin_cost(compound/1, 1).
builtin_cost(callable/1, 1).
builtin_cost(is_list/1, 2).
builtin_cost(ground/1, 2).
% term construction / inspection
builtin_cost(functor/3, 3).
builtin_cost(arg/3, 2).
builtin_cost((=..)/2, 3).
builtin_cost(copy_term/2, 4).
% atom / string ops (touch character data)
builtin_cost(atom_codes/2, 4).
builtin_cost(atom_chars/2, 4).
builtin_cost(atom_length/2, 2).
builtin_cost(atom_concat/3, 4).
builtin_cost(atom_string/2, 3).
builtin_cost(number_codes/2, 4).
builtin_cost(sub_atom/5, 6).
% list ops
builtin_cost(length/2, 3).
builtin_cost(append/3, 4).
builtin_cost(member/2, 3).
builtin_cost(memberchk/2, 3).
builtin_cost(nth0/3, 3).
builtin_cost(nth1/3, 3).
builtin_cost(reverse/2, 3).
builtin_cost(msort/2, 6).
builtin_cost(sort/2, 6).
% I/O
builtin_cost(write/1, 5).
builtin_cost(writeln/1, 5).
builtin_cost(print/1, 5).
builtin_cost(nl/0, 3).
builtin_cost(format/1, 5).
builtin_cost(format/2, 6).
builtin_cost(format/3, 6).
builtin_cost(read/1, 8).

% ============================================================================
% Cost algebra
% ============================================================================

cost_weight(cost(W, _), W).
cost_bounded(cost(_, B), B).

% Conjunction: add weights; bounded only if both bounded.
cost_seq(cost(W1, B1), cost(W2, B2), cost(W, B)) :-
    W is W1 + W2, bmeet(B1, B2, B).

% Disjunction / alternative: worst single-solution branch.
cost_alt(cost(W1, B1), cost(W2, B2), cost(W, B)) :-
    W is max(W1, W2), bmeet(B1, B2, B).

bmeet(bounded, bounded, bounded) :- !.
bmeet(_, _, unbounded).

% ============================================================================
% Tier classification
% ============================================================================

%% cost_tier(+Cost, -Tier)
%  Tier in {trivial, cheap, moderate, expensive, recursive}. `recursive` is any
%  `unbounded` cost (growth with input); the rest are weight bands on a bounded
%  cost. `expensive` and `recursive` are the "worth parallelising / not flattened
%  by the host compiler" tiers.
cost_tier(cost(_, unbounded), recursive) :- !.
cost_tier(cost(W, bounded), Tier) :-
    threshold(cheap, Cheap),
    threshold(moderate, Moderate),
    (   W =< 1        -> Tier = trivial
    ;   W =< Cheap    -> Tier = cheap
    ;   W =< Moderate -> Tier = moderate
    ;   Tier = expensive
    ).

goal_cost_tier(Goal, Model, Tier) :-
    goal_cost(Goal, Model, Cost), cost_tier(Cost, Tier).

predicate_cost_tier(PI, Model, Tier) :-
    predicate_cost(PI, Model, Cost), cost_tier(Cost, Tier).

% ============================================================================
% Model: recursion set + memoised per-predicate cost
% ============================================================================

% Model = cost_model(Module, RecursiveAssoc, CostAssoc)
%   RecursiveAssoc : PI -> true        (predicates reachable from themselves)
%   CostAssoc      : PI -> cost(W,B)   (memoised predicate costs)

%% build_cost_model(+Module, -Model)
build_cost_model(Module, Model) :-
    findall(Name/Arity,
            ( current_predicate(Module:Name/Arity),
              functor(H, Name, Arity),
              \+ predicate_property(Module:H, built_in),
              clause(Module:H, _) ),
            Preds0),
    sort(Preds0, Preds),
    build_cost_model(Module, Preds, Model).

%% build_cost_model(+Module, +Preds, -Model)
build_cost_model(Module, Preds, cost_model(Module, RecAssoc, CostAssoc)) :-
    build_call_graph(Module, Preds, Graph),
    recursive_set(Preds, Graph, RecAssoc),
    foldl(ensure_pred_cost(Module, RecAssoc), Preds, t, CostAssoc0),
    ( CostAssoc0 == t -> empty_assoc(CostAssoc) ; CostAssoc = CostAssoc0 ).

recursive_predicate(PI, cost_model(_, RecAssoc, _)) :-
    get_assoc(PI, RecAssoc, true).

% --- call graph -------------------------------------------------------------

build_call_graph(Module, Preds, Graph) :-
    foldl(pred_edges(Module, Preds), Preds, t, G0),
    ( G0 == t -> empty_assoc(Graph) ; Graph = G0 ).

pred_edges(Module, Preds, PI, GIn, GOut) :-
    PI = Name/Arity,
    functor(H, Name, Arity),
    findall(Callee,
            ( clause(Module:H, Body),
              body_callee(Body, Preds, Callee) ),
            Callees0),
    sort(Callees0, Callees),
    put_assoc(PI, GIn, Callees, GOut).

% Enumerate the user-predicate callees (restricted to Preds) appearing in a body.
body_callee(Body, Preds, Callee) :-
    body_goals(Body, Goals),
    member(G, Goals),
    nonvar(G),
    goal_user_callees(G, Preds, Callee).

% Flatten control constructs into the list of leaf goals.
body_goals(G, []) :- var(G), !.
body_goals((A, B), Gs) :- !, body_goals(A, GA), body_goals(B, GB), append(GA, GB, Gs).
body_goals((A ; B), Gs) :- !, body_goals(A, GA), body_goals(B, GB), append(GA, GB, Gs).
body_goals((A -> B), Gs) :- !, body_goals(A, GA), body_goals(B, GB), append(GA, GB, Gs).
body_goals((A *-> B), Gs) :- !, body_goals(A, GA), body_goals(B, GB), append(GA, GB, Gs).
body_goals(\+ A, Gs) :- !, body_goals(A, Gs).
body_goals(Goal, Gs) :-
    meta_subgoal(Goal, Sub), !,
    body_goals(Sub, Gs).
body_goals(G, [G]).

% Meta-goals whose first relevant argument is itself a goal to descend into.
meta_subgoal(once(G), G).
meta_subgoal(ignore(G), G).
meta_subgoal(findall(_, G, _), G).
meta_subgoal(findall(_, G, _, _), G).
meta_subgoal(bagof(_, G, _), G).
meta_subgoal(setof(_, G, _), G).
meta_subgoal(aggregate_all(_, G, _), G).
meta_subgoal(forall(C, A), (C, A)).

goal_user_callees(G, Preds, Name/Arity) :-
    functor(G, Name, Arity),
    memberchk(Name/Arity, Preds).

% --- recursion detection (predicate reachable from itself) ------------------

recursive_set(Preds, Graph, RecAssoc) :-
    reachability(Preds, Graph, Reach),
    foldl(mark_recursive(Reach), Preds, t, RecAssoc0),
    ( RecAssoc0 == t -> empty_assoc(RecAssoc) ; RecAssoc = RecAssoc0 ).

mark_recursive(Reach, PI, AIn, AOut) :-
    ( get_assoc(PI, Reach, Set), memberchk(PI, Set)
    ->  put_assoc(PI, AIn, true, AOut)
    ;   AOut = AIn ).

% Transitive closure of the call graph: Reach[PI] = set reachable in >=1 step.
reachability(Preds, Graph, Reach) :-
    foldl(direct_succ(Graph), Preds, t, R0),
    ( R0 == t -> empty_assoc(R1) ; R1 = R0 ),
    close_reach(Preds, R1, Reach).

direct_succ(Graph, PI, AIn, AOut) :-
    ( get_assoc(PI, Graph, Succ) -> true ; Succ = [] ),
    put_assoc(PI, AIn, Succ, AOut).

close_reach(Preds, R0, R) :-
    foldl(expand_reach(R0), Preds, R0, R1),
    ( assoc_equal(Preds, R0, R1) -> R = R1 ; close_reach(Preds, R1, R) ).

expand_reach(Base, PI, AIn, AOut) :-
    get_assoc(PI, AIn, Cur),
    foldl(add_succ_of(Base), Cur, Cur, New0),
    sort(New0, New),
    put_assoc(PI, AIn, New, AOut).

add_succ_of(Base, Q, SIn, SOut) :-
    ( get_assoc(Q, Base, QS) -> append(SIn, QS, SOut) ; SOut = SIn ).

assoc_equal(Preds, A, B) :-
    forall(member(PI, Preds),
           ( get_assoc(PI, A, SA), get_assoc(PI, B, SB),
             sort(SA, S), sort(SB, S) )).

% --- per-predicate cost (memoised over the non-recursive DAG) ---------------

ensure_pred_cost(Module, RecAssoc, PI, MemoIn, MemoOut) :-
    ( MemoIn == t -> empty_assoc(M0) ; M0 = MemoIn ),
    pred_cost_memo(Module, RecAssoc, PI, M0, MemoOut).

pred_cost_memo(_Module, _Rec, PI, Memo, Memo) :-
    get_assoc(PI, Memo, _), !.
pred_cost_memo(_Module, RecAssoc, PI, MemoIn, MemoOut) :-
    get_assoc(PI, RecAssoc, true), !,        % recursive predicate
    recursive_base_weight(W),
    put_assoc(PI, MemoIn, cost(W, unbounded), MemoOut).
pred_cost_memo(Module, RecAssoc, PI, MemoIn, MemoOut) :-
    PI = Name/Arity,
    functor(H, Name, Arity),
    findall(Body, clause(Module:H, Body), Bodies),
    foldl(clause_cost_acc(Module, RecAssoc),
          Bodies, clause_acc(cost(0, bounded), MemoIn), clause_acc(Combined, Memo1)),
    % a predicate with no clauses (shouldn't happen for Preds) costs the default
    ( Bodies == [] -> default_builtin_weight(DW), Cost = cost(DW, bounded) ; Cost = Combined ),
    put_assoc(PI, Memo1, Cost, MemoOut).

% Combine clauses by worst single-solution branch (a call commits to one clause;
% taking the max is the conservative per-call estimate).
clause_cost_acc(Module, RecAssoc, Body,
                clause_acc(AccCost, MemoIn), clause_acc(AccCost1, MemoOut)) :-
    body_cost_memo(Module, RecAssoc, Body, MemoIn, BodyCost, MemoOut),
    cost_alt(AccCost, BodyCost, AccCost1).

% --- body / goal cost with memo threading -----------------------------------

body_cost_memo(_M, _R, G, Memo, cost(0, bounded), Memo) :- var(G), !.
body_cost_memo(_M, _R, true, Memo, cost(0, bounded), Memo) :- !.
body_cost_memo(M, R, (A, B), MemoIn, Cost, MemoOut) :- !,
    body_cost_memo(M, R, A, MemoIn, CA, Memo1),
    body_cost_memo(M, R, B, Memo1, CB, MemoOut),
    cost_seq(CA, CB, Cost).
body_cost_memo(M, R, (C -> T ; E), MemoIn, Cost, MemoOut) :- !,
    body_cost_memo(M, R, C, MemoIn, CC, Memo1),
    body_cost_memo(M, R, T, Memo1, CT, Memo2),
    body_cost_memo(M, R, E, Memo2, CE, MemoOut),
    cost_alt(CT, CE, CTE),
    cost_seq(CC, CTE, Cost).
body_cost_memo(M, R, (A ; B), MemoIn, Cost, MemoOut) :- !,
    body_cost_memo(M, R, A, MemoIn, CA, Memo1),
    body_cost_memo(M, R, B, Memo1, CB, MemoOut),
    cost_alt(CA, CB, Cost).
body_cost_memo(M, R, (C -> T), MemoIn, Cost, MemoOut) :- !,
    body_cost_memo(M, R, C, MemoIn, CC, Memo1),
    body_cost_memo(M, R, T, Memo1, CT, MemoOut),
    cost_seq(CC, CT, Cost).
body_cost_memo(M, R, \+ A, MemoIn, Cost, MemoOut) :- !,
    body_cost_memo(M, R, A, MemoIn, Cost, MemoOut).
body_cost_memo(M, R, once(A), MemoIn, Cost, MemoOut) :- !,
    body_cost_memo(M, R, A, MemoIn, Cost, MemoOut).
body_cost_memo(M, R, ignore(A), MemoIn, Cost, MemoOut) :- !,
    body_cost_memo(M, R, A, MemoIn, Cost, MemoOut).
% Aggregates enumerate the generator to exhaustion → unbounded (cardinality
% unknown statically); the weight carries the per-solution generator cost.
body_cost_memo(M, R, Agg, MemoIn, cost(W, unbounded), MemoOut) :-
    aggregate_goal(Agg, Gen), !,
    body_cost_memo(M, R, Gen, MemoIn, cost(W, _), MemoOut).
body_cost_memo(M, R, forall(C, A), MemoIn, cost(W, unbounded), MemoOut) :- !,
    body_cost_memo(M, R, C, MemoIn, CC, Memo1),
    body_cost_memo(M, R, A, Memo1, CA, MemoOut),
    cost_seq(CC, CA, cost(W, _)).
% A plain goal: user predicate (resolve via memo) or builtin/external.
body_cost_memo(M, R, Goal, MemoIn, Cost, MemoOut) :-
    callable(Goal), functor(Goal, Name, Arity),
    (   user_clause_exists(M, Name/Arity)
    ->  pred_cost_memo(M, R, Name/Arity, MemoIn, MemoOut),
        get_assoc(Name/Arity, MemoOut, Callee),
        call_overhead(OV), Callee = cost(CW, CB),
        W is CW + OV, Cost = cost(W, CB)
    ;   MemoOut = MemoIn,
        builtin_or_default(Name/Arity, W),
        Cost = cost(W, bounded)
    ).

aggregate_goal(findall(_, G, _), G).
aggregate_goal(findall(_, G, _, _), G).
aggregate_goal(bagof(_, G, _), G).
aggregate_goal(setof(_, G, _), G).
aggregate_goal(aggregate_all(_, G, _), G).

user_clause_exists(Module, Name/Arity) :-
    functor(H, Name, Arity),
    \+ predicate_property(Module:H, built_in),
    clause(Module:H, _), !.

builtin_or_default(PI, W) :-
    ( builtin_cost(PI, W0) -> W = W0 ; default_builtin_weight(W) ).

% ============================================================================
% Public lookups against a built model
% ============================================================================

%% predicate_cost(+PI, +Model, -Cost)
predicate_cost(PI, cost_model(_, _, CostAssoc), Cost) :-
    get_assoc(PI, CostAssoc, Cost), !.
predicate_cost(_PI, _Model, cost(W, bounded)) :-
    default_builtin_weight(W).      % unknown predicate: default

%% goal_cost(+Goal, +Model, -Cost)
%  Cost of an arbitrary goal against a built model (no memo mutation needed —
%  predicate costs are already resolved in the model).
goal_cost(Goal, cost_model(Module, RecAssoc, CostAssoc), Cost) :-
    goal_cost_(Goal, Module, RecAssoc, CostAssoc, Cost).

clause_body_cost(Body, Model, Cost) :- goal_cost(Body, Model, Cost).

goal_cost_(G, _, _, _, cost(0, bounded)) :- var(G), !.
goal_cost_(true, _, _, _, cost(0, bounded)) :- !.
goal_cost_((A, B), M, R, C, Cost) :- !,
    goal_cost_(A, M, R, C, CA), goal_cost_(B, M, R, C, CB), cost_seq(CA, CB, Cost).
goal_cost_((Cnd -> T ; E), M, R, C, Cost) :- !,
    goal_cost_(Cnd, M, R, C, CC), goal_cost_(T, M, R, C, CT), goal_cost_(E, M, R, C, CE),
    cost_alt(CT, CE, CTE), cost_seq(CC, CTE, Cost).
goal_cost_((A ; B), M, R, C, Cost) :- !,
    goal_cost_(A, M, R, C, CA), goal_cost_(B, M, R, C, CB), cost_alt(CA, CB, Cost).
goal_cost_((Cnd -> T), M, R, C, Cost) :- !,
    goal_cost_(Cnd, M, R, C, CC), goal_cost_(T, M, R, C, CT), cost_seq(CC, CT, Cost).
goal_cost_(\+ A, M, R, C, Cost) :- !, goal_cost_(A, M, R, C, Cost).
goal_cost_(once(A), M, R, C, Cost) :- !, goal_cost_(A, M, R, C, Cost).
goal_cost_(ignore(A), M, R, C, Cost) :- !, goal_cost_(A, M, R, C, Cost).
goal_cost_(Agg, M, R, C, cost(W, unbounded)) :-
    aggregate_goal(Agg, Gen), !, goal_cost_(Gen, M, R, C, cost(W, _)).
goal_cost_(forall(Cnd, A), M, R, C, cost(W, unbounded)) :- !,
    goal_cost_(Cnd, M, R, C, CC), goal_cost_(A, M, R, C, CA), cost_seq(CC, CA, cost(W, _)).
goal_cost_(Goal, _M, _R, CostAssoc, Cost) :-
    callable(Goal), functor(Goal, Name, Arity),
    (   get_assoc(Name/Arity, CostAssoc, Callee)
    ->  call_overhead(OV), Callee = cost(CW, CB), W is CW + OV, Cost = cost(W, CB)
    ;   builtin_or_default(Name/Arity, W), Cost = cost(W, bounded)
    ).
