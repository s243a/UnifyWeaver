% parallel_gate.pl
%
% First real consumer of the static cost machinery (cost_analysis.pl): the
% compile-time profitability gate for T7 parallel aggregates.
%
% The T7 runtime substrate (rust_runtime/par_aggregate.rs) can fan out the
% independent branches of a forkable aggregate (findall/aggregate_all/bagof/
% setof) across threads, but that only pays off when the *per-branch work* is
% substantial — cheap branches regress 5-200x because each parallel worker must
% clone its own WAM machine (see docs/reports/wam_rust_t7_parallel_perf.md). Its
% built-in gate samples a few branches at runtime; this module supplies the
% *compile-time* decision so the emitter can choose sequential vs parallel
% without a runtime probe (the probe then becomes a cheap confirmation, not the
% decision).
%
% Key insight that keeps this simple: in the cost model, an aggregate generator's
% boundedness/weight already reflects its per-branch work. A pure enumerator
% (`member/2`, `between/3`, fact lookups) is cheap+bounded; a generator that
% calls a recursive predicate is unbounded; one doing heavy per-solution work has
% a high weight. So the decision is just the generator's cost tier — no fragile
% splitting of "enumerator" from "body" needed.

:- module(parallel_gate, [
    forkable_aggregate/3,            % +Goal, -Template, -Generator
    aggregate_parallel_decision/3,   % +Goal, +Model, -Decision   (parallel|sequential)
    goal_parallel_decision/3,        % +Generator, +Model, -Decision
    split_aggregate_generator/5,     % +InnerGoal, +Model, -Enum, -Body, -Frontier
    parallel_aggregate_transform/5,  % +AggGoal, +Model, +Seed, -Helpers, -Plan
    lift_embedded_aggregate/6,       % +Head, +Body, +Model, +Seed, -NewBody, -HelperClause
    parallel_worthy_tier/1           % ?Tier   (multifile, overridable policy)
]).

:- use_module(cost_analysis).
:- use_module(library(lists)).
:- use_module(library(apply)).

:- multifile parallel_worthy_tier/1.
:- dynamic   parallel_worthy_tier/1.

%% forkable_aggregate(+Goal, -Template, -Generator)
%  True when Goal is an aggregate whose solution branches are order-independent
%  and thus candidates for fan-out. (Independence w.r.t. side effects is a
%  separate analysis; this only recognises the *shape*.)
forkable_aggregate(findall(Tmpl, Gen, _),     Tmpl, Gen).
forkable_aggregate(findall(Tmpl, Gen, _, _),  Tmpl, Gen).
forkable_aggregate(aggregate_all(Tmpl, Gen, _), Tmpl, Gen).
forkable_aggregate(bagof(Tmpl, Gen0, _),      Tmpl, Gen) :- strip_caret(Gen0, Gen).
forkable_aggregate(setof(Tmpl, Gen0, _),      Tmpl, Gen) :- strip_caret(Gen0, Gen).

% bagof/setof allow `Var^Goal` existential qualification; the work is in Goal.
strip_caret(_^G0, G) :- !, strip_caret(G0, G).
strip_caret(G, G).

%% aggregate_parallel_decision(+Goal, +Model, -Decision)
%  Decision in {parallel, sequential}. `parallel` only when Goal is a forkable
%  aggregate whose generator is in a parallel-worthy cost tier; everything else
%  (cheap aggregates, non-aggregates) stays sequential — the safe default that
%  never regresses.
aggregate_parallel_decision(Goal, Model, Decision) :-
    ( forkable_aggregate(Goal, _, Gen)
    ->  goal_parallel_decision(Gen, Model, Decision)
    ;   Decision = sequential ).

%% goal_parallel_decision(+Generator, +Model, -Decision)
goal_parallel_decision(Gen, Model, Decision) :-
    goal_cost_tier(Gen, Model, Tier),
    ( worthy(Tier) -> Decision = parallel ; Decision = sequential ).

worthy(Tier) :- parallel_worthy_tier(Tier), !.
worthy(Tier) :- \+ parallel_worthy_tier(_), default_worthy(Tier).

% Default policy: only clear wins fan out. Cheap/trivial/moderate stay sequential
% (conservative — the benchmark showed moderate per-branch work near break-even).
default_worthy(expensive).
default_worthy(recursive).

% ============================================================================
% Route-1 split analysis: enumerator | body
% ============================================================================
%
% For the compile-time generator/body split (the recommended route for the T7
% runtime fork), cut a forkable aggregate's inner goal into:
%
%   - Enum   — a cheap prefix that produces the fan-out bindings, run once;
%   - Body   — the expensive per-branch suffix, run on a cloned machine per
%              Enum solution (the unit `gated_collect` parallelises);
%   - Frontier — the variables Enum binds that Body reads (each branch's input).
%
% This is pure analysis — no codegen, no runtime. It is deliberately
% conservative: it only splits a plain conjunction of pure goals (no cut, no
% control constructs, no side effects), and only where there is a non-empty
% cheap enumerator prefix AND a non-empty body that carries the real work.
% Anything else fails, and the caller keeps the sequential aggregate.

%% split_aggregate_generator(+InnerGoal, +Model, -Enum, -Body, -Frontier)
split_aggregate_generator(InnerGoal, Model, Enum, Body, Frontier) :-
    conj_to_list(InnerGoal, Goals),
    Goals = [_|_],
    splittable_goals(Goals),                 % soundness gate
    split_at_first_expensive(Goals, Model, EnumGs, BodyGs),
    EnumGs = [_|_],                          % a cheap fan-out prefix exists
    BodyGs = [_|_],                          % and a body carrying the work
    list_to_conj(EnumGs, Enum),
    list_to_conj(BodyGs, Body),
    term_variables(Enum, EnumVars),
    term_variables(Body, BodyVars),
    shared_vars(EnumVars, BodyVars, Frontier).

% --- soundness: only pure, splittable conjunctions ---------------------------

splittable_goals(Goals) :- forall(member(G, Goals), splittable_goal(G)).

splittable_goal(G) :- nonvar(G), \+ unsafe_goal(G).

% Goals that break sound parallel splitting (cut, control flow, side effects).
unsafe_goal(!).
unsafe_goal((_ ; _)).
unsafe_goal((_ -> _)).
unsafe_goal((_ *-> _)).
unsafe_goal(\+(_)).
unsafe_goal(not(_)).
unsafe_goal(G) :- functor(G, F, A), side_effecting(F/A).

side_effecting(write/1).      side_effecting(writeln/1).
side_effecting(print/1).      side_effecting(nl/0).
side_effecting(write/2).      side_effecting(nl/1).
side_effecting(format/1).     side_effecting(format/2).
side_effecting(format/3).     side_effecting(read/1).
side_effecting(read_term/2).  side_effecting(read_term/3).
side_effecting(assert/1).     side_effecting(asserta/1).
side_effecting(assertz/1).    side_effecting(retract/1).
side_effecting(retractall/1). side_effecting(abolish/1).
side_effecting(tab/1).        side_effecting(put_char/1).

% --- the cut point: cheap enumerator prefix, then the body -------------------
% Enum = maximal leading run of trivial/cheap goals; Body = the rest, which
% begins at the first goal that actually does work (moderate/expensive/recursive).

split_at_first_expensive([], _, [], []).
split_at_first_expensive([G|Gs], Model, Enum, Body) :-
    ( cheap_goal(G, Model)
    ->  Enum = [G|Enum1],
        split_at_first_expensive(Gs, Model, Enum1, Body)
    ;   Enum = [],
        Body = [G|Gs]
    ).

cheap_goal(G, Model) :-
    goal_cost_tier(G, Model, Tier),
    ( Tier == trivial ; Tier == cheap ), !.

% --- helpers -----------------------------------------------------------------

conj_to_list(G, [G]) :- var(G), !.
conj_to_list((A, B), Gs) :- !, conj_to_list(A, GA), conj_to_list(B, GB), append(GA, GB, Gs).
conj_to_list(G, [G]).

list_to_conj([G], G) :- !.
list_to_conj([G|Gs], (G, Rest)) :- list_to_conj(Gs, Rest).

% Variables present in both lists, compared by identity (==), order of A.
shared_vars(As, Bs, Shared) :-
    include(memberchk_eq(Bs), As, Shared).
memberchk_eq([X|_], V) :- X == V, !.
memberchk_eq([_|T], V) :- memberchk_eq(T, V).

% ============================================================================
% Route-1 source transform: aggregate -> (enum helper, body helper) + plan
% ============================================================================
%
% Rewrite a splittable, parallel-eligible aggregate into two ordinary helper
% predicates plus a parallel plan the runtime executes:
%
%     __enum_Seed(Input)         :- Enum.    % yields each branch's input tuple
%     __body_Seed(Input, Value)  :- Body.    % per branch, yields collected value
%
%     plan = par_aggregate(AggType, EnumName/1, BodyName/2, Result)
%
% The runtime then does: collect Inputs via __enum (sequential, cheap), then
% gated_collect-map __body over the Inputs on cloned machines, and reduce the
% collected Values by AggType into Result. Because __enum and __body are normal
% compiled predicates, this reuses the existing call/aggregate machinery and
% needs no choice-point surgery.
%
% `Input` packs exactly the variables the body (or the collected value) needs
% from the enumerator: vars(Enum) ∩ (vars(Body) ∪ vars(Value)). This is a
% superset of split's Enum∩Body frontier — it also carries value vars the
% enumerator binds — which is what makes the decomposition result-preserving.
%
% Fails (caller keeps the sequential aggregate) unless the aggregate is forkable,
% the gate says parallel, and the inner goal splits soundly.

parallel_aggregate_transform(AggGoal, Model, Seed, Helpers, Plan) :-
    forkable_aggregate(AggGoal, _Template, InnerGoal),
    agg_value_type(AggGoal, AggType, Value),
    aggregate_result(AggGoal, Result),
    goal_parallel_decision(InnerGoal, Model, parallel),
    split_aggregate_generator(InnerGoal, Model, Enum, Body, _Frontier),
    % Input = enum-bound vars that the body or the collected value reads.
    term_variables(Enum, EnumVars),
    term_variables(Body, BodyVars),
    term_variables(Value, ValueVars),
    append(BodyVars, ValueVars, Downstream),
    shared_vars(EnumVars, Downstream, Input),
    InputTuple =.. [ti | Input],
    format(atom(EnumName), '__par_enum_~w', [Seed]),
    format(atom(BodyName), '__par_body_~w', [Seed]),
    EnumHead =.. [EnumName, InputTuple],
    BodyHead =.. [BodyName, InputTuple, Value],
    Helpers = [ (EnumHead :- Enum), (BodyHead :- Body) ],
    Plan = par_aggregate(AggType, EnumName/1, BodyName/2, Result).

% The per-solution value collected, and how it is reduced.
agg_value_type(findall(Tmpl, _, _),          collect, Tmpl) :- !.
agg_value_type(aggregate_all(count, _, _),    count, _Anon) :- !.
agg_value_type(aggregate_all(sum(V), _, _),   sum, V) :- !.
agg_value_type(aggregate_all(max(V), _, _),   max, V) :- !.
agg_value_type(aggregate_all(min(V), _, _),   min, V) :- !.
agg_value_type(aggregate_all(bag(V), _, _),   collect, V) :- !.
agg_value_type(aggregate_all(set(V), _, _),   set, V) :- !.

aggregate_result(findall(_, _, R), R) :- !.
aggregate_result(aggregate_all(_, _, R), R).

% ============================================================================
% Embedded-aggregate lift: aggregate inside a larger clause body -> whole-body
% helper predicate + an in-place call to it.
% ============================================================================
%
% The whole-body injection only fires when a predicate's entire body is the
% aggregate. This lifts an *embedded* aggregate out: given a clause Head :- Body
% where Body is a conjunction of >= 2 goals one of which is a parallel-eligible,
% splittable forkable aggregate, produce
%
%   NewBody     = Body with that aggregate replaced by `__lift_agg_Seed(Ins, R)`
%   HelperClause= (__lift_agg_Seed(Ins, R) :- <the aggregate>)
%
% so the helper is a whole-body aggregate the existing machinery parallelises,
% and the clause calls it (WAM-callable once the helper is registered foreign).
% Ins = variables the aggregate shares with the rest of the clause (Head + other
% goals) minus the result R; R is the helper's last argument. Pure source
% transform — result-preserving (the helper's sequential semantics equal the
% original goal's), no codegen. Fails (clause compiled unchanged) when no
% embedded eligible aggregate is present.

lift_embedded_aggregate(Head, Body, Model, Seed, NewBody, (HelperHead :- AggGoal)) :-
    conj_to_list(Body, Goals),
    Goals = [_, _|_],                                  % embedded: >= 2 goals
    member(AggGoal, Goals), nonvar(AggGoal),
    forkable_aggregate(AggGoal, _Tmpl, Inner),
    aggregate_parallel_decision(AggGoal, Model, parallel),
    split_aggregate_generator(Inner, Model, _, _, _),  % must be splittable
    !,                                                 % commit to first eligible
    aggregate_result(AggGoal, R),
    exclude(==(AggGoal), Goals, OtherGoals),
    term_variables(AggGoal, AggVars),
    term_variables([Head|OtherGoals], RestVars),
    shared_vars(AggVars, RestVars, Shared0),
    exclude(==(R), Shared0, Inputs),
    append(Inputs, [R], HArgs),
    format(atom(Name), '__lift_agg_~w', [Seed]),
    HelperHead =.. [Name|HArgs],
    replace_goal(Goals, AggGoal, HelperHead, NewGoals),
    list_to_conj(NewGoals, NewBody).

% replace the first goal identical (==) to Old with New, preserving order.
replace_goal([G|Gs], Old, New, [Out|Rest]) :-
    ( G == Old -> Out = New, Rest = Gs
    ; Out = G, replace_goal(Gs, Old, New, Rest) ).
