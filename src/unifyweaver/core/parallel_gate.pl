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
    parallel_worthy_tier/1           % ?Tier   (multifile, overridable policy)
]).

:- use_module(cost_analysis).

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
