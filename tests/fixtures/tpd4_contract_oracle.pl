:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% tpd4_contract_oracle.pl — language-neutral finite oracle + fixture matrix
% for the hybrid WAM transitive_parent_distance4 contract (shortest-positive
% parents).
%
% See docs/design/WAM_TRANSITIVE_PARENT_DISTANCE4_CONTRACT.md.
%
% Expected fixture facts are LITERAL and independent — they must not be
% derived by calling the oracle under test. The oracle is provided for
% generator helpers and cross-checks; parity tests assert against the
% literal expectations below.

:- module(tpd4_contract_oracle, [
    tpd4_oracle_triples/3,       % +Edges, +Source, -SortedTriples
    tpd4_oracle_has/5,           % +Edges, +Source, +T, +P, +D
    tpd4_fixture/3,              % ?Name, -Edges, -Sources
    tpd4_fixture_expected/3      % +Name, +Source, -SortedTriples (LITERAL)
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% tpd4_oracle_triples(+Edges, +Source, -SortedTriples) is det.
%  Edges: list of From-To. SortedTriples: list of tpd(T,P,D),
%  sorted by T, then P, then D. Finite BFS with parent sets; dist map
%  tracks nodes discovered via an outgoing edge (not seeded with Source).
tpd4_oracle_triples(Edges, Source, Sorted) :-
    tpd4_bfs(Edges, [Source-0], _{ }, _{ }, Dist, Parents),
    findall(tpd(T, P, D),
            ( get_dict(T, Dist, D),
              get_dict(T, Parents, PSet),
              member(P, PSet)
            ),
            Raw),
    sort(Raw, Sorted).

%% tpd4_oracle_has(+Edges, +Source, +T, +P, +D) is semidet.
tpd4_oracle_has(Edges, Source, T, P, D) :-
    tpd4_oracle_triples(Edges, Source, Sorted),
    memberchk(tpd(T, P, D), Sorted).

%% BFS: queue entries Node-Depth.
%% Dist/Parents are dicts: Dist: Node -> MinDist, Parents: Node -> list of parents.
tpd4_bfs(_, [], Dist, Parents, Dist, Parents) :- !.
tpd4_bfs(Edges, [Node-Depth|Queue], Dist0, Parents0, Dist, Parents) :-
    NextDepth is Depth + 1,
    findall(Next, member(Node-Next, Edges), News0),
    list_to_set(News0, News),
    tpd4_expand(News, Node, NextDepth, Queue, Dist0, Parents0,
                Queue1, Dist1, Parents1),
    tpd4_bfs(Edges, Queue1, Dist1, Parents1, Dist, Parents).

tpd4_expand([], _, _, Queue, Dist, Parents, Queue, Dist, Parents) :- !.
tpd4_expand([N|Ns], Pred, ND, Queue0, Dist0, Parents0, Queue, Dist, Parents) :-
    (   get_dict(N, Dist0, D0)
    ->  (   D0 =:= ND
        ->  get_dict(N, Parents0, Ps0),
            (   memberchk(Pred, Ps0)
            ->  Parents1 = Parents0
            ;   put_dict(N, Parents0, [Pred|Ps0], Parents1)
            ),
            Dist1 = Dist0,
            Queue1 = Queue0
        ;   % longer path — ignore
            Dist1 = Dist0,
            Parents1 = Parents0,
            Queue1 = Queue0
        )
    ;   put_dict(N, Dist0, ND, Dist1),
        put_dict(N, Parents0, [Pred], Parents1),
        append(Queue0, [N-ND], Queue1)
    ),
    tpd4_expand(Ns, Pred, ND, Queue1, Dist1, Parents1, Queue, Dist, Parents).

% ============================================================
% Fixture matrix — edges + query sources
% ============================================================

tpd4_fixture(chain, [a-b, b-c, c-d], [a, c, d, z]).
tpd4_fixture(equal_diamond, [a-b, a-c, b-d, c-d], [a, d]).
tpd4_fixture(unequal_paths, [a-b, b-c, c-t, a-t], [a, t]).
tpd4_fixture(dup_edges, [a-b, a-b, b-c, b-c], [a, b]).
tpd4_fixture(self_loop, [a-a, a-b], [a, b]).
tpd4_fixture(two_cycle, [a-b, b-a], [a, b]).
tpd4_fixture(cycle_exit, [a-b, b-c, c-a, c-d], [a, c, d]).
tpd4_fixture(branch_rejoin, [a-b, a-c, b-e, c-e, e-f], [a]).
tpd4_fixture(sink_disconnected, [a-b, c-d], [a, b, c, z]).

% ============================================================
% Literal expected triples tpd(T,P,D), sorted
% ============================================================

tpd4_fixture_expected(chain, a, [tpd(b, a, 1), tpd(c, b, 2), tpd(d, c, 3)]).
tpd4_fixture_expected(chain, c, [tpd(d, c, 1)]).
tpd4_fixture_expected(chain, d, []).
tpd4_fixture_expected(chain, z, []).

% Equal-shortest diamond: d has parents b and c at distance 2.
tpd4_fixture_expected(equal_diamond, a,
    [tpd(b, a, 1), tpd(c, a, 1), tpd(d, b, 2), tpd(d, c, 2)]).
tpd4_fixture_expected(equal_diamond, d, []).

% Direct a→t beats a→b→c→t (distance 1, parent a — not c).
tpd4_fixture_expected(unequal_paths, a,
    [tpd(b, a, 1), tpd(c, b, 2), tpd(t, a, 1)]).
tpd4_fixture_expected(unequal_paths, t, []).

tpd4_fixture_expected(dup_edges, a, [tpd(b, a, 1), tpd(c, b, 2)]).
tpd4_fixture_expected(dup_edges, b, [tpd(c, b, 1)]).

tpd4_fixture_expected(self_loop, a, [tpd(a, a, 1), tpd(b, a, 1)]).
tpd4_fixture_expected(self_loop, b, []).

tpd4_fixture_expected(two_cycle, a, [tpd(a, b, 2), tpd(b, a, 1)]).
tpd4_fixture_expected(two_cycle, b, [tpd(a, b, 1), tpd(b, a, 2)]).

tpd4_fixture_expected(cycle_exit, a,
    [tpd(a, c, 3), tpd(b, a, 1), tpd(c, b, 2), tpd(d, c, 3)]).
tpd4_fixture_expected(cycle_exit, c,
    [tpd(a, c, 1), tpd(b, a, 2), tpd(c, b, 3), tpd(d, c, 1)]).
tpd4_fixture_expected(cycle_exit, d, []).

tpd4_fixture_expected(branch_rejoin, a,
    [tpd(b, a, 1), tpd(c, a, 1), tpd(e, b, 2), tpd(e, c, 2), tpd(f, e, 3)]).

tpd4_fixture_expected(sink_disconnected, a, [tpd(b, a, 1)]).
tpd4_fixture_expected(sink_disconnected, b, []).
tpd4_fixture_expected(sink_disconnected, c, [tpd(d, c, 1)]).
tpd4_fixture_expected(sink_disconnected, z, []).
