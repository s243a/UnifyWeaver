% wam_elixir_kernel_dispatch.pl
%
% Pattern recognition for graph-kernel dispatch in the WAM-Elixir
% target. When a predicate matches a recognised pattern AND the
% project Options include `kernel_dispatch(true)`, write_wam_elixir_-
% project/3 emits a dispatch module that delegates to a hand-tuned
% WamRuntime.GraphKernel.* module instead of the WAM-bytecode lower
% chain. See docs/WAM_TARGET_ROADMAP.md for the architectural story
% and benchmarks/wam_elixir_tier2_findall.md for the measured impact
% (PR #1799 reported 218x-1564x for transitive closure).
%
% First pattern shipped: transitive closure.
%
%   tc(X, Z) :- edge(X, Z).
%   tc(X, Z) :- edge(X, Y), tc(Y, Z).
%
% Variants out of scope for this PR (future patterns will extend the
% matcher table):
%   - clause-order swap (recursive case before base case)
%   - right-recursive form `tc(X, Z) :- tc(Y, Z), edge(X, Y)`
%   - other-arity edges, n-ary recursion, etc.

:- module(wam_elixir_kernel_dispatch, [
    match_transitive_closure_pattern/3
]).

%% match_transitive_closure_pattern(+Module, +Pred/+Arity, -EdgePred/+EdgeArity)
%
%  True if Module:Pred/Arity has exactly the canonical 2-clause TC
%  shape over some EdgePred/EdgeArity (where EdgePred is recovered
%  from the clause bodies). Pred and EdgePred must both be 2-ary.
%
%  Alpha-equivalent: variable names in source dont matter — only
%  structural identity.
match_transitive_closure_pattern(Module, Pred/2, EdgePred/2) :-
    atom(Pred),
    functor(QHead, Pred, 2),
    findall(QHead-Body, clause(Module:QHead, Body), Clauses),
    Clauses = [BaseHead-BaseBody, RecHead-RecBody],
    %
    % Base case: pred(A, B) :- edge(A, B).
    %
    BaseHead =.. [Pred, BA, BB],
    var(BA), var(BB), BA \== BB,
    strip_module_call(BaseBody, BaseGoal),
    BaseGoal =.. [EdgePred, EBA, EBB],
    atom(EdgePred), EdgePred \== Pred,
    BA == EBA, BB == EBB,
    %
    % Recursive case: pred(C, D) :- edge(C, E), pred(E, D).
    %
    RecHead =.. [Pred, RC, RD],
    var(RC), var(RD), RC \== RD,
    strip_module_call(RecBody, RecBodyPlain),
    RecBodyPlain = (G1Raw, G2Raw),
    strip_module_call(G1Raw, G1),
    strip_module_call(G2Raw, G2),
    G1 =.. [EdgePred, RG1A, RG1B],
    RG1A == RC,
    var(RG1B), RG1B \== RC, RG1B \== RD,
    G2 =.. [Pred, RG2A, RG2B],
    RG2A == RG1B,
    RG2B == RD.

%% strip_module_call(+Goal, -PlainGoal)
%  Body goals can be module-qualified (`mod:goal(args)`) — sometimes
%  with nested qualifiers (`user:kernel_test:goal`) and sometimes
%  wrapping a conjunction (`user:(g1, g2)`). Strip recursively so
%  the matcher sees bare goal shapes.
strip_module_call(_M:G, Plain) :- !, strip_module_call(G, Plain).
strip_module_call(G, G).
