:- module(wsp3_contract_oracle, [
    wsp3_oracle_min_pairs/3,
    wsp3_oracle_sorted_pairs/3,
    wsp3_oracle_matches_expected/3,
    wsp3_oracle_cheaper_detour_edges/1,
    wsp3_oracle_cheaper_detour_expected/1,
    wsp3_oracle_duplicate_edges/1,
    wsp3_oracle_duplicate_expected/1,
    wsp3_oracle_equal_cost_edges/1,
    wsp3_oracle_equal_cost_expected/1,
    wsp3_oracle_mixed_weights_edges/1,
    wsp3_oracle_mixed_weights_expected/1,
    wsp3_oracle_zero_cost_edges/1,
    wsp3_oracle_zero_cost_expected/1,
    wsp3_oracle_positive_cycle_edges/1,
    wsp3_oracle_positive_cycle_expected/1,
    wsp3_oracle_source_self_loop_edges/1,
    wsp3_oracle_source_self_loop_expected/1,
    wsp3_oracle_sink_edges/1,
    wsp3_oracle_sink_expected/1,
    wsp3_oracle_large_chain_edges/2,
    wsp3_oracle_large_chain_expected/2,
    wsp3_oracle_pred_a_edges/1,
    wsp3_oracle_pred_a_expected/1,
    wsp3_oracle_pred_b_edges/1,
    wsp3_oracle_pred_b_expected/1
]).
:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module(library(dicts)).

%!  wsp3_edge_weight(+Edge, -From, -To, -W) is semidet.
wsp3_edge_weight(edge(From, To, W0), From, To, W) :-
    number(W0),
    W is float(W0),
    W >= 0.0,
    W =\= -1.0Inf,
    W =\= 1.0Inf,
    \+ (W =\= W).

%!  wsp3_oracle_min_pairs(+Edges, +Source, -Pairs) is det.
%   Independent finite Dijkstra oracle. Emits one (Target, FloatCost) for every
%   reachable non-Source target. Source is always excluded.
wsp3_oracle_min_pairs(Edges, Source, Pairs) :-
    put_dict(Source, _{}, 0.0, Dist0),
    wsp3_dijkstra(Edges, [(0.0, Source)], Dist0, DistFinal),
    findall(Target-Cost,
            ( get_dict(Target, DistFinal, Cost),
              Target \== Source
            ),
            Unsorted),
    keysort(Unsorted, Pairs).

wsp3_oracle_sorted_pairs(Edges, Source, Pairs) :-
    wsp3_oracle_min_pairs(Edges, Source, Pairs).

wsp3_oracle_matches_expected(Edges, Source, Expected) :-
    wsp3_oracle_min_pairs(Edges, Source, Actual),
    sort(Expected, ESorted),
    sort(Actual, ASorted),
    ESorted == ASorted.

wsp3_dijkstra(_Edges, [], Dist, Dist) :- !.
wsp3_dijkstra(Edges, PQ0, Dist0, Dist) :-
    wsp3_pq_pop_min(PQ0, Cost, Node, PQ1),
    (   get_dict(Node, Dist0, Known),
        Cost > Known
    ->  wsp3_dijkstra(Edges, PQ1, Dist0, Dist)
    ;   findall(edge(Node, To, W),
                ( member(E, Edges),
                  wsp3_edge_weight(E, Node, To, W)
                ),
                Out),
        wsp3_relax_all(Out, Cost, Dist0, Dist1, PQ1, PQ2),
        wsp3_dijkstra(Edges, PQ2, Dist1, Dist)
    ).

wsp3_relax_all([], _Base, Dist, Dist, PQ, PQ).
wsp3_relax_all([edge(_From, To, W)|Rest], Base, Dist0, Dist, PQ0, PQ) :-
    NewCost is Base + W,
    (   get_dict(To, Dist0, Old),
        \+ NewCost < Old
    ->  Dist1 = Dist0,
        PQ1 = PQ0
    ;   put_dict(To, Dist0, NewCost, Dist1),
        PQ1 = [(NewCost, To)|PQ0]
    ),
    wsp3_relax_all(Rest, Base, Dist1, Dist, PQ1, PQ).

wsp3_pq_pop_min([(C, N)|Rest], C, N, Rest) :-
    \+ ( member((C2, _), Rest), C2 < C ), !.
wsp3_pq_pop_min([(C, N)|Rest], BestC, BestN, [(C, N)|RestOut]) :-
    wsp3_pq_pop_min(Rest, BestC, BestN, RestOut).

% ---- literal fixtures ----------------------------------------------------

wsp3_oracle_cheaper_detour_edges([
    edge(a, b, 10.0),
    edge(a, c, 1.0),
    edge(c, b, 1.0),
    edge(b, d, 1.0)
]).
wsp3_oracle_cheaper_detour_expected([b-2.0, c-1.0, d-3.0]).

wsp3_oracle_duplicate_edges([
    edge(a, b, 5.0),
    edge(a, b, 2.0),
    edge(b, c, 1.0)
]).
wsp3_oracle_duplicate_expected([b-2.0, c-3.0]).

wsp3_oracle_equal_cost_edges([
    edge(a, b, 2.0),
    edge(a, c, 1.0),
    edge(c, b, 1.0),
    edge(b, d, 3.0)
]).
wsp3_oracle_equal_cost_expected([b-2.0, c-1.0, d-5.0]).

wsp3_oracle_mixed_weights_edges([
    edge(a, b, 1),
    edge(b, c, 0.5),
    edge(a, c, 3.0)
]).
wsp3_oracle_mixed_weights_expected([b-1.0, c-1.5]).

wsp3_oracle_zero_cost_edges([
    edge(a, b, 0.0),
    edge(b, a, 0.0),
    edge(b, c, 0.0),
    edge(c, d, 1.0)
]).
wsp3_oracle_zero_cost_expected([b-0.0, c-0.0, d-1.0]).

wsp3_oracle_positive_cycle_edges([
    edge(a, b, 1.0),
    edge(b, c, 1.0),
    edge(c, b, 1.0),
    edge(c, d, 1.0)
]).
wsp3_oracle_positive_cycle_expected([b-1.0, c-2.0, d-3.0]).

wsp3_oracle_source_self_loop_edges([
    edge(a, a, 1.0),
    edge(a, b, 2.0)
]).
wsp3_oracle_source_self_loop_expected([b-2.0]).

wsp3_oracle_sink_edges([
    edge(a, b, 1.0)
]).
wsp3_oracle_sink_expected([]).

wsp3_oracle_large_chain_edges(N, Edges) :-
    integer(N),
    N > 0,
    N1 is N - 1,
    findall(edge(From, To, 1.0),
            ( between(0, N1, I),
              I1 is I + 1,
              atom_concat(n, I, From),
              atom_concat(n, I1, To)
            ),
            Edges).

wsp3_oracle_large_chain_expected(N, Expected) :-
    integer(N),
    N > 0,
    findall(To-Cost,
            ( between(1, N, I),
              atom_concat(n, I, To),
              Cost is float(I)
            ),
            Expected).

wsp3_oracle_pred_a_edges([
    edge(a, b, 1.0),
    edge(b, c, 1.0)
]).
wsp3_oracle_pred_a_expected([b-1.0, c-2.0]).

wsp3_oracle_pred_b_edges([
    edge(x, y, 3.0),
    edge(y, z, 4.0)
]).
wsp3_oracle_pred_b_expected([y-3.0, z-7.0]).
