:- module(astar4_contract_oracle, [
    astar4_oracle_cost/4,
    astar4_oracle_matches/4,
    astar4_oracle_cheaper_detour_edges/1,
    astar4_oracle_cheaper_detour_cost/1,
    astar4_oracle_overestimate_edges/1,
    astar4_oracle_overestimate_heur/1,
    astar4_oracle_overestimate_cost/1,
    astar4_oracle_missing_heur_edges/1,
    astar4_oracle_missing_heur_cost/1,
    astar4_oracle_zero_cycle_edges/1,
    astar4_oracle_zero_cycle_cost/1,
    astar4_oracle_large_chain_edges/2,
    astar4_oracle_large_chain_cost/2,
    astar4_oracle_pred_a_edges/1,
    astar4_oracle_pred_a_cost/1,
    astar4_oracle_pred_b_edges/1,
    astar4_oracle_pred_b_cost/1
]).
:- use_module(library(lists)).
:- use_module(library(dicts)).

%!  astar4_edge_weight(+Edge, -From, -To, -W) is semidet.
astar4_edge_weight(edge(From, To, W0), From, To, W) :-
    number(W0),
    W is float(W0),
    W >= 0.0,
    W =\= -1.0Inf,
    W =\= 1.0Inf,
    \+ (W =\= W).

%!  astar4_oracle_cost(+Edges, +Source, +Target, -Cost) is semidet.
%   Independent finite Dijkstra to a bound Target. Succeeds with Cost=0.0
%   when Source == Target. Fails when Target is unreachable.
astar4_oracle_cost(_Edges, Source, Target, 0.0) :-
    Source == Target, !.
astar4_oracle_cost(Edges, Source, Target, Cost) :-
    put_dict(Source, _{}, 0.0, Dist0),
    astar4_dijkstra(Edges, [(0.0, Source)], Dist0, DistFinal),
    get_dict(Target, DistFinal, Cost).

astar4_oracle_matches(Edges, Source, Target, Expected) :-
    astar4_oracle_cost(Edges, Source, Target, Cost),
    Cost =:= Expected.

astar4_dijkstra(_Edges, [], Dist, Dist) :- !.
astar4_dijkstra(Edges, PQ0, Dist0, Dist) :-
    astar4_pq_pop_min(PQ0, Cost, Node, PQ1),
    (   get_dict(Node, Dist0, Known),
        Cost > Known
    ->  astar4_dijkstra(Edges, PQ1, Dist0, Dist)
    ;   findall(edge(Node, To, W),
                ( member(E, Edges),
                  astar4_edge_weight(E, Node, To, W)
                ),
                Out),
        astar4_relax_all(Out, Cost, Dist0, Dist1, PQ1, PQ2),
        astar4_dijkstra(Edges, PQ2, Dist1, Dist)
    ).

astar4_relax_all([], _Base, Dist, Dist, PQ, PQ).
astar4_relax_all([edge(_From, To, W)|Rest], Base, Dist0, Dist, PQ0, PQ) :-
    NewCost is Base + W,
    (   get_dict(To, Dist0, Old),
        \+ NewCost < Old
    ->  Dist1 = Dist0,
        PQ1 = PQ0
    ;   put_dict(To, Dist0, NewCost, Dist1),
        PQ1 = [(NewCost, To)|PQ0]
    ),
    astar4_relax_all(Rest, Base, Dist1, Dist, PQ1, PQ).

astar4_pq_pop_min([(C, N)|Rest], C, N, Rest) :-
    \+ ( member((C2, _), Rest), C2 < C ), !.
astar4_pq_pop_min([(C, N)|Rest], BestC, BestN, [(C, N)|RestOut]) :-
    astar4_pq_pop_min(Rest, BestC, BestN, RestOut).

% ---- fixtures ----------------------------------------------------------

astar4_oracle_cheaper_detour_edges([
    edge(a, b, 10.0),
    edge(a, c, 1.0),
    edge(c, b, 1.0)
]).
astar4_oracle_cheaper_detour_cost(2.0).

% Overestimating heuristic: direct a->b weight 10, detour a->c->b cost 2.
% Heuristic h(a)=100 would make unsafe first-pop A* take the direct edge
% if it settled incorrectly; Dijkstra optimum is still 2.0.
astar4_oracle_overestimate_edges([
    edge(a, b, 10.0),
    edge(a, c, 1.0),
    edge(c, b, 1.0)
]).
astar4_oracle_overestimate_heur([
    edge(a, b, 100.0),
    edge(c, b, 0.0)
]).
astar4_oracle_overestimate_cost(2.0).

astar4_oracle_missing_heur_edges([
    edge(a, b, 1.0),
    edge(b, c, 1.0)
]).
astar4_oracle_missing_heur_cost(2.0).

astar4_oracle_zero_cycle_edges([
    edge(a, b, 0.0),
    edge(b, a, 0.0),
    edge(b, c, 1.0)
]).
astar4_oracle_zero_cycle_cost(1.0).

astar4_oracle_large_chain_edges(N, Edges) :-
    integer(N), N > 0,
    N1 is N - 1,
    findall(edge(From, To, 1.0),
            ( between(0, N1, I),
              I1 is I + 1,
              atom_concat(n, I, From),
              atom_concat(n, I1, To)
            ),
            Edges).

astar4_oracle_large_chain_cost(N, Cost) :-
    integer(N), N > 0,
    Cost is float(N).

astar4_oracle_pred_a_edges([
    edge(a, b, 1.0),
    edge(b, c, 1.0)
]).
astar4_oracle_pred_a_cost(2.0).

astar4_oracle_pred_b_edges([
    edge(x, y, 3.0),
    edge(y, z, 4.0)
]).
astar4_oracle_pred_b_cost(7.0).
