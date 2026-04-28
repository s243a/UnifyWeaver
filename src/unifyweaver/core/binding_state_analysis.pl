:- encoding(utf8).
%% binding_state_analysis.pl
%%
%% Forward, no-fixpoint, three-valued binding-state analysis for the
%% WAM target's =../2 PutStructureDyn lowering.
%%
%% See docs/design/WAM_HASKELL_MODE_ANALYSIS_{PHILOSOPHY,SPEC,PLAN}.md
%% for the full design. This module produces, for every goal in a
%% normalised clause body, a `goal_binding(Idx, BeforeEnv, AfterEnv)`
%% record that downstream code generators can consult to decide
%% whether a builtin call can be specialised (e.g. compose-mode
%% =../2 -> PutStructureDyn).
%%
%% Domain (binding_state/1):
%%   unbound  — variable definitely unbound at the program point
%%   bound    — variable definitely bound to a non-variable term
%%   unknown  — analysis cannot prove either (lattice bottom)
%%
%% The analysis is *sound* in the sense that whenever a downstream
%% pass acts on a `bound` or `unbound` answer, that answer must hold
%% on every successful execution path that reaches the program
%% point. When in doubt, the analysis answers `unknown`.

:- module(binding_state_analysis, [
    %% Public API consumed by the WAM target.
    analyse_clause_bindings/3,        % +Head, +Body, -GoalBindings
    binding_state_at/4,               % +GoalIdx, +Var, +GoalBindings, -State
    binding_state_at_var/3,           % +Env, +Var, +ExpectedState
    %% Lower-level helpers (exported for testability and for direct
    %% callers that want to construct envs by hand).
    binding_state/1,                  % ?State
    empty_binding_env/1,              % -Env
    set_binding_state/4,              % +Env0, +Var, +State, -Env1
    get_binding_state/3,              % +Env, +Var, -State
    initial_binding_env/3,            % +Head, +ModeDecl, -Env
    propagate_goal/3,                 % +Goal, +BeforeEnv, -AfterEnv
    meet_env/3                        % +EnvA, +EnvB, -EnvAB
]).

:- use_module(library(lists)).
:- use_module(clause_body_analysis).
:- use_module(demand_analysis).

%% ========================================================================
%% Section 1. Domain
%% ========================================================================

%% binding_state(?State)
%  Three-valued binding state.
binding_state(unbound).
binding_state(bound).
binding_state(unknown).

%% Internal: lattice meet (variable-by-variable).
%%   bound   ⊓ bound   = bound
%%   unbound ⊓ unbound = unbound
%%   anything else      = unknown
state_meet(S, S, S) :- !.
state_meet(_, _, unknown).

%% ========================================================================
%% Section 2. Environment
%%
%% Stored as a flat list of `var(V)-State` pairs where V is the actual
%% Prolog variable. Lookup uses ==/2 (variable identity), so the
%% environment correctly distinguishes fresh-var-with-same-name from
%% the original variable.
%% ========================================================================

%% empty_binding_env(-Env)
empty_binding_env([]).

%% get_binding_state(+Env, +Var, -State)
%  Defaults to `unknown` for variables not in the environment, and
%  for any non-variable term.
get_binding_state(_, Var, bound) :-
    nonvar(Var), !.
get_binding_state(Env, Var, State) :-
    var(Var),
    env_lookup(Env, Var, State0),
    !,
    State = State0.
get_binding_state(_, _, unknown).

%% set_binding_state(+Env0, +Var, +State, -Env1)
%  Replaces or inserts the binding state for Var. Non-variable Var is
%  a no-op (we only track Prolog variables).
set_binding_state(Env0, Var, _, Env0) :-
    nonvar(Var), !.
set_binding_state(Env0, Var, State, Env1) :-
    binding_state(State),
    env_replace(Env0, Var, State, Env1).

env_lookup([Var0-State|_], Var, State) :-
    Var0 == Var, !.
env_lookup([_|Rest], Var, State) :-
    env_lookup(Rest, Var, State).

env_replace([], Var, State, [Var-State]).
env_replace([V0-_|Rest], Var, State, [Var-State|Rest]) :-
    V0 == Var, !.
env_replace([V0-S|Rest0], Var, State, [V0-S|Rest1]) :-
    env_replace(Rest0, Var, State, Rest1).

%% binding_state_at_var(+Env, +Var, +ExpectedState)
%  Convenience predicate used by the WAM lowering: succeeds iff Var
%  is *proven* to be in ExpectedState. `unknown` never matches a
%  concrete expected state.
binding_state_at_var(Env, Var, ExpectedState) :-
    get_binding_state(Env, Var, ExpectedState),
    ExpectedState \== unknown.

%% ========================================================================
%% Section 3. Initial environment from head + mode declaration
%% ========================================================================

%% initial_binding_env(+Head, +ModeDecl, -Env)
%
%  ModeDecl is one of:
%    - `none` (no `:- mode/1` declaration found)
%    - a list of mode atoms (input/output/any) — the same shape
%      that `demand_analysis:read_mode_declaration/3` returns.
%
%  Variables in head sub-terms (e.g. inside `f(X)` when the head arg
%  is `f(X)`) are marked `bound` because head unification will have
%  bound them by the time the body executes.
initial_binding_env(Head, ModeDecl, Env) :-
    Head =.. [_|HeadArgs],
    apply_mode_decl(HeadArgs, ModeDecl, [], Env0),
    propagate_head_structure(HeadArgs, Env0, Env).

apply_mode_decl(_HeadArgs, none, Env, Env) :- !.
apply_mode_decl([], _, Env, Env) :- !.
apply_mode_decl([Arg|RestArgs], [Mode|RestModes], Env0, Env) :-
    !,
    apply_mode_to_arg(Arg, Mode, Env0, Env1),
    apply_mode_decl(RestArgs, RestModes, Env1, Env).
apply_mode_decl([_|RestArgs], [], Env0, Env) :-
    %% Mode list shorter than head — leave remaining args at default.
    apply_mode_decl(RestArgs, [], Env0, Env).

apply_mode_to_arg(Arg, input, Env0, Env1) :-
    !,
    %% mode + (input): caller promises the argument is bound.
    (   var(Arg)
    ->  set_binding_state(Env0, Arg, bound, Env1)
    ;   Env1 = Env0
    ).
apply_mode_to_arg(Arg, output, Env0, Env1) :-
    !,
    %% mode - (output): caller promises the argument is unbound.
    (   var(Arg)
    ->  set_binding_state(Env0, Arg, unbound, Env1)
    ;   %% Inconsistent: the head pattern is a non-variable but mode
        %% says output. Fall through to `unknown` (no entry).
        Env1 = Env0
    ).
apply_mode_to_arg(_Arg, any, Env, Env) :- !.
apply_mode_to_arg(_Arg, _Other, Env, Env).

%% propagate_head_structure(+HeadArgs, +Env0, -Env1)
%  For each non-variable head argument, walk it and mark every Prolog
%  variable contained inside as `bound` — head unification will have
%  bound them at clause entry.
propagate_head_structure([], Env, Env).
propagate_head_structure([Arg|Rest], Env0, Env) :-
    (   var(Arg)
    ->  Env1 = Env0
    ;   mark_subvars_bound(Arg, Env0, Env1)
    ),
    propagate_head_structure(Rest, Env1, Env).

mark_subvars_bound(Term, Env, Env) :-
    (   atomic(Term) ; var(Term) ),
    \+ var(Term),  %% atomic(Term)
    !.
mark_subvars_bound(Var, Env0, Env1) :-
    var(Var), !,
    set_binding_state(Env0, Var, bound, Env1).
mark_subvars_bound(Term, Env0, Env) :-
    Term =.. [_|Args],
    foldl(mark_subvars_bound, Args, Env0, Env).

%% ========================================================================
%% Section 4. Per-goal propagation
%% ========================================================================

%% propagate_goal(+Goal, +BeforeEnv, -AfterEnv)
%  Updates the binding environment based on the success-conditional
%  effect of Goal. The fall-through assumption is "all argument
%  variables become unknown" — most sound, least informative.

%% Module-qualified goals: strip the module wrapper and recurse.
propagate_goal(_M:Goal, Env0, Env) :- !,
    propagate_goal(Goal, Env0, Env).

%% true/fail/cut: no effect on bindings.
propagate_goal(true,  Env, Env) :- !.
propagate_goal(fail,  Env, Env) :- !.
propagate_goal(false, Env, Env) :- !.
propagate_goal(!,     Env, Env) :- !.

%% Negation: doesn't bind anything (as-failure semantics).
propagate_goal(\+(_),  Env, Env) :- !.
propagate_goal(not(_), Env, Env) :- !.

%% Type-test guards that also prove a binding state.
propagate_goal(var(X), Env0, Env) :- !,
    (   var(X)
    ->  set_binding_state(Env0, X, unbound, Env)
    ;   Env = Env0
    ).
propagate_goal(nonvar(X), Env0, Env) :- !,
    (   var(X)
    ->  set_binding_state(Env0, X, bound, Env)
    ;   Env = Env0
    ).
propagate_goal(atom(X),     Env0, Env) :- !, set_var_bound(X, Env0, Env).
propagate_goal(atomic(X),   Env0, Env) :- !, set_var_bound(X, Env0, Env).
propagate_goal(integer(X),  Env0, Env) :- !, set_var_bound(X, Env0, Env).
propagate_goal(float(X),    Env0, Env) :- !, set_var_bound(X, Env0, Env).
propagate_goal(number(X),   Env0, Env) :- !, set_var_bound(X, Env0, Env).
propagate_goal(is_list(X),  Env0, Env) :- !, set_var_bound(X, Env0, Env).
propagate_goal(compound(X), Env0, Env) :- !, set_var_bound(X, Env0, Env).

%% Comparison guards (==, =\=, >, <, etc.): no binding effect.
propagate_goal(Goal, Env, Env) :-
    compound(Goal),
    Goal =.. [Op, _, _],
    is_pure_comparison_op(Op),
    !.

%% Arithmetic: X is Expr binds X.
propagate_goal(X is _, Env0, Env) :- !,
    set_var_bound(X, Env0, Env).

%% Unification: X = Y.
propagate_goal(X = Y, Env0, Env) :- !,
    propagate_unification(X, Y, Env0, Env).

%% Disequality (\= and \==): no binding effect; success-conditional
%% but proves nothing useful for our domain.
propagate_goal(\=(_, _),  Env, Env) :- !.
propagate_goal(\==(_, _), Env, Env) :- !.

%% Term-inspection builtins.
propagate_goal(functor(T, Name, Arity), Env0, Env) :- !,
    propagate_functor(T, Name, Arity, Env0, Env).
propagate_goal(arg(_, _, A), Env0, Env) :- !,
    %% A's runtime state depends on T's contents — we cannot prove
    %% bound or unbound. Mark as unknown.
    set_var_unknown(A, Env0, Env).
propagate_goal(=..(T, L), Env0, Env) :- !,
    propagate_univ(T, L, Env0, Env).
propagate_goal(copy_term(_, C), Env0, Env) :- !,
    set_var_bound(C, Env0, Env).

%% Aggregate builtins: inner goal analysed in isolation; outer env
%% only sees the result variable transition to `bound`.
propagate_goal(findall(_T, _G, R), Env0, Env) :- !,
    set_var_bound(R, Env0, Env).
propagate_goal(bagof(_T, _G, R), Env0, Env) :- !,
    set_var_bound(R, Env0, Env).
propagate_goal(setof(_T, _G, R), Env0, Env) :- !,
    set_var_bound(R, Env0, Env).
propagate_goal(aggregate_all(_T, _G, R), Env0, Env) :- !,
    set_var_bound(R, Env0, Env).

%% Conjunction: sequential composition.
propagate_goal((A, B), Env0, Env) :- !,
    propagate_goal(A, Env0, Env1),
    propagate_goal(B, Env1, Env).

%% If-then-else: walk Then with Cond's bindings, walk Else from Env0,
%% meet the post-states.
propagate_goal((Cond -> Then ; Else), Env0, Env) :- !,
    propagate_goal(Cond, Env0, EnvCond),
    propagate_goal(Then, EnvCond, EnvThen),
    propagate_goal(Else, Env0, EnvElse),
    meet_env(EnvThen, EnvElse, Env).

%% If-then (no else): result of the construct is taken to be the
%% post-state of Cond+Then if matched, or Env0 otherwise. We meet to
%% remain conservative.
propagate_goal((Cond -> Then), Env0, Env) :- !,
    propagate_goal(Cond, Env0, EnvCond),
    propagate_goal(Then, EnvCond, EnvThen),
    meet_env(EnvThen, Env0, Env).

%% Bare disjunction: meet of branches.
propagate_goal((A ; B), Env0, Env) :- !,
    propagate_goal(A, Env0, EnvA),
    propagate_goal(B, Env0, EnvB),
    meet_env(EnvA, EnvB, Env).

%% User predicate calls: classify by mode declaration.
propagate_goal(Goal, Env0, Env) :-
    compound(Goal),
    functor(Goal, Pred, Arity),
    Goal =.. [_|Args],
    !,
    (   demand_analysis:read_mode_declaration(Pred, Arity, Modes)
    ->  apply_call_modes(Args, Modes, Env0, Env)
    ;   set_args_unknown(Args, Env0, Env)
    ).

%% Atom or zero-arity opaque goal: nothing to update.
propagate_goal(_Goal, Env, Env).

%% ----- helpers -----

set_var_bound(Var, Env0, Env) :-
    (   var(Var)
    ->  set_binding_state(Env0, Var, bound, Env)
    ;   Env = Env0
    ).

set_var_unknown(Var, Env0, Env) :-
    (   var(Var)
    ->  set_binding_state(Env0, Var, unknown, Env)
    ;   Env = Env0
    ).

set_args_unknown([], Env, Env).
set_args_unknown([Arg|Rest], Env0, Env) :-
    (   var(Arg)
    ->  set_binding_state(Env0, Arg, unknown, Env1)
    ;   Env1 = Env0
    ),
    set_args_unknown(Rest, Env1, Env).

apply_call_modes([], _, Env, Env) :- !.
apply_call_modes(_, [], Env, Env) :- !.
apply_call_modes([Arg|Rest], [Mode|RestModes], Env0, Env) :-
    apply_call_mode(Arg, Mode, Env0, Env1),
    apply_call_modes(Rest, RestModes, Env1, Env).

apply_call_mode(_Arg, input, Env, Env) :- !.    % caller required to supply bound
apply_call_mode(Arg,  output, Env0, Env) :- !,  % callee promises to bind it
    set_var_bound(Arg, Env0, Env).
apply_call_mode(_Arg, any, Env, Env) :- !.      % `?` mode: leave at pre-call state
                                                % (per WAM_HASKELL_MODE_ANALYSIS_SPEC.md
                                                % §2.3.7). The user has asserted the
                                                % predicate is mode-polymorphic, so a
                                                % proven-bound arg stays proven bound
                                                % across the call.
apply_call_mode(_Arg, _Other, Env, Env).

is_pure_comparison_op(>).
is_pure_comparison_op(<).
is_pure_comparison_op(>=).
is_pure_comparison_op(=<).
is_pure_comparison_op(=:=).
is_pure_comparison_op(=\=).
is_pure_comparison_op(==).
is_pure_comparison_op(@<).
is_pure_comparison_op(@>).
is_pure_comparison_op(@=<).
is_pure_comparison_op(@>=).

%% propagate_unification(+X, +Y, +Env0, -Env1)
propagate_unification(X, Y, Env0, Env) :-
    var(X), var(Y), !,
    get_binding_state(Env0, X, SX),
    get_binding_state(Env0, Y, SY),
    (   ( SX == bound ; SY == bound )
    ->  set_var_bound(X, Env0, Env1),
        set_var_bound(Y, Env1, Env)
    ;   %% Two unbound aliases or any other combination: collapse to
        %% unknown — analysis does not track aliasing.
        set_var_unknown(X, Env0, Env1),
        set_var_unknown(Y, Env1, Env)
    ).
propagate_unification(X, Y, Env0, Env) :-
    var(X), nonvar(Y), !,
    set_var_bound(X, Env0, Env1),
    %% Sub-vars of Y become bound iff Y is fully ground.
    (   ground(Y)
    ->  Env = Env1
    ;   mark_term_vars_unknown(Y, Env1, Env)
    ).
propagate_unification(X, Y, Env0, Env) :-
    nonvar(X), var(Y), !,
    propagate_unification(Y, X, Env0, Env).
propagate_unification(_, _, Env, Env).

mark_term_vars_unknown(Term, Env, Env) :-
    atomic(Term), !.
mark_term_vars_unknown(Var, Env0, Env) :-
    var(Var), !,
    %% leave it alone — caller's unification is success-conditional
    %% and does not bind the sub-var unless it was already bound.
    Env = Env0.
mark_term_vars_unknown(Term, Env0, Env) :-
    Term =.. [_|Args],
    foldl(mark_term_vars_unknown, Args, Env0, Env).

%% propagate_functor(+T, +Name, +Arity, +Env0, -Env)
%  functor(T, Name, Arity) — pre-cond: at least one of T, Name+Arity bound.
propagate_functor(T, Name, Arity, Env0, Env) :-
    get_binding_state(Env0, T, ST),
    get_binding_state(Env0, Name, SN),
    get_binding_state(Env0, Arity, SA),
    (   ST == bound
    ->  set_var_bound(Name, Env0, Env1),
        set_var_bound(Arity, Env1, Env)
    ;   ( SN == bound, SA == bound )
    ->  set_var_bound(T, Env0, Env)
    ;   %% Cannot prove direction. All become unknown.
        set_var_unknown(T, Env0, Env1),
        set_var_unknown(Name, Env1, Env2),
        set_var_unknown(Arity, Env2, Env)
    ).

%% propagate_univ(+T, +L, +Env0, -Env)
%  T =.. L. Pre-cond: at least one of T, L is bound.
%
%  Compose mode (T pre-unbound, L bound):
%    L must be a list whose head is bound to an atom and whose tail
%    is bound. We do a lightweight static check: if L is syntactically
%    a list with head and an atom-or-bound-var first element, treat
%    as compose. Otherwise stay conservative.
propagate_univ(T, L, Env0, Env) :-
    get_binding_state(Env0, T, ST),
    (   ST == bound
    ->  %% Decompose: T -> L list.
        set_var_bound(L, Env0, Env)
    ;   %% Try compose: L is a proper list literal of length ≥ 1
        %% whose first element (the functor name) is provably bound
        %% to an atom or to a variable proven `bound`. The rest of
        %% the list elements are the structure arguments — they
        %% don't need to be bound for the WAM PutStructureDyn
        %% lowering (the emitter handles unbound args via
        %% set_variable).
        univ_compose_list_ok(L, Env0)
    ->  set_var_bound(T, Env0, Env)
    ;   %% Conservative fallback.
        set_var_unknown(T, Env0, Env1),
        set_var_unknown(L, Env1, Env)
    ).

%% univ_compose_list_ok(+L, +Env)
%  L is a proper list literal of length ≥ 1, AND the head of L (the
%  functor name) is statically bound to an atom — i.e. literally an
%  atom, or a variable in Env with state `bound`.
univ_compose_list_ok(L, _Env) :-
    var(L), !, fail.
univ_compose_list_ok([H|T], Env) :-
    proper_list_tail(T),
    head_bound_to_atom(H, Env).

proper_list_tail(T) :- var(T), !, fail.
proper_list_tail([]) :- !.
proper_list_tail([_|T]) :- proper_list_tail(T).

head_bound_to_atom(H, _Env) :- atom(H), !.
head_bound_to_atom(H, Env) :-
    var(H), !,
    get_binding_state(Env, H, bound).

%% ========================================================================
%% Section 5. Meet of two environments
%% ========================================================================

%% meet_env(+EnvA, +EnvB, -EnvAB)
%  Variable-by-variable meet. Variables present in only one env are
%  treated as `unknown` in the other (since absent = default unknown).
meet_env(EnvA, EnvB, EnvMeet) :-
    collect_env_vars(EnvA, EnvB, Vars),
    foldl(meet_one_var(EnvA, EnvB), Vars, [], EnvMeet).

collect_env_vars(EnvA, EnvB, Vars) :-
    pairs_vars(EnvA, VA),
    pairs_vars(EnvB, VB),
    append(VA, VB, All),
    list_unique_by_identity(All, Vars).

pairs_vars([], []).
pairs_vars([V-_|Rest], [V|Vs]) :- pairs_vars(Rest, Vs).

list_unique_by_identity([], []).
list_unique_by_identity([V|Rest], [V|UniqueRest]) :-
    \+ ( member(W, Rest), W == V ), !,
    list_unique_by_identity(Rest, UniqueRest).
list_unique_by_identity([_|Rest], UniqueRest) :-
    list_unique_by_identity(Rest, UniqueRest).

meet_one_var(EnvA, EnvB, V, Env0, Env1) :-
    get_binding_state(EnvA, V, SA),
    get_binding_state(EnvB, V, SB),
    state_meet(SA, SB, SM),
    (   SM == unknown
    ->  Env1 = Env0  % default — no entry
    ;   set_binding_state(Env0, V, SM, Env1)
    ).

%% ========================================================================
%% Section 6. Whole-clause analysis
%% ========================================================================

%% analyse_clause_bindings(+Head, +Body, -GoalBindings)
%
%  Computes per-goal binding-state records for a clause body.
%
%  GoalBindings is a list of `goal_binding(Idx, BeforeEnv, AfterEnv)`
%  records, one per *normalised* body goal (control-flow constructs
%  produce a single record with a meet post-state — they are not
%  decomposed into sub-goal records).
analyse_clause_bindings(Head, Body, GoalBindings) :-
    Head =.. [Pred|_],
    functor(Head, Pred, Arity),
    lookup_modes(Pred, Arity, Modes),
    initial_binding_env(Head, Modes, Env0),
    clause_body_analysis:normalize_goals(Body, Goals),
    walk_body(Goals, 1, Env0, GoalBindings).

%% lookup_modes(+Pred, +Arity, -ModesOrNone)
lookup_modes(Pred, Arity, Modes) :-
    catch(
        demand_analysis:read_mode_declaration(Pred, Arity, Modes),
        _,
        fail
    ),
    !.
lookup_modes(_, _, none).

walk_body([], _, _, []).
walk_body([Goal|Rest], Idx, EnvIn, [goal_binding(Idx, EnvIn, EnvOut)|RestBindings]) :-
    propagate_goal(Goal, EnvIn, EnvOut),
    Idx1 is Idx + 1,
    walk_body(Rest, Idx1, EnvOut, RestBindings).

%% binding_state_at(+GoalIdx, +Var, +GoalBindings, -State)
%  Reads the BeforeEnv of the goal at index GoalIdx; defaults to
%  `unknown` if the goal index is out of range.
binding_state_at(GoalIdx, Var, GoalBindings, State) :-
    member(goal_binding(GoalIdx, BeforeEnv, _), GoalBindings),
    !,
    get_binding_state(BeforeEnv, Var, State).
binding_state_at(_, _, _, unknown).
