:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% tspd5_contract_oracle.pl — language-neutral finite oracle + fixture matrix
% for the hybrid WAM transitive_step_parent_distance5 contract (shortest-
% positive correlated step/parent).
%
% See docs/design/WAM_TRANSITIVE_STEP_PARENT_DISTANCE5_CONTRACT.md.
%
% Expected fixture facts are LITERAL and independent — they must not be
% derived by calling the oracle under test. The oracle is provided for
% generator helpers and cross-checks; parity tests assert against the
% literal expectations below.

:- module(tspd5_contract_oracle, [
    tspd5_oracle_quads/3,        % +Edges, +Source, -SortedQuads
    tspd5_oracle_has/6,          % +Edges, +Source, +T, +Step, +P, +D
    tspd5_fixture/3,             % ?Name, -Edges, -Sources
    tspd5_fixture_expected/3     % +Name, +Source, -SortedQuads (LITERAL)
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% tspd5_oracle_quads(+Edges, +Source, -SortedQuads) is det.
%  Edges: list of From-To. SortedQuads: list of tspd(T,Step,P,D),
%  sorted. Finite level-synchronous BFS storing correlated (Step,Parent)
%  pairs per Target — never an independent cross-product.
tspd5_oracle_quads(Edges, Source, Sorted) :-
    tspd5_bfs(Edges, Source, [Source-0], _{ }, _{ }, Dist, Pairs),
    findall(tspd(T, Step, P, D),
            ( get_dict(T, Dist, D),
              get_dict(T, Pairs, PairList),
              member(Step-P, PairList)
            ),
            Raw),
    sort(Raw, Sorted).

%% tspd5_oracle_has(+Edges, +Source, +T, +Step, +P, +D) is semidet.
tspd5_oracle_has(Edges, Source, T, Step, P, D) :-
    tspd5_oracle_quads(Edges, Source, Sorted),
    memberchk(tspd(T, Step, P, D), Sorted).

%% Queue entries Node-Depth. Dist: Node->MinDist.
%% Pairs: Node -> sorted unique list of Step-Parent.
tspd5_bfs(_, _, [], Dist, Pairs, Dist, Pairs) :- !.
tspd5_bfs(Edges, Source, [Node-Depth|Queue], Dist0, Pairs0, Dist, Pairs) :-
    NextDepth is Depth + 1,
    findall(Next, member(Node-Next, Edges), News0),
    list_to_set(News0, News),
    tspd5_expand(News, Node, Source, NextDepth, Queue, Dist0, Pairs0,
                 Queue1, Dist1, Pairs1),
    tspd5_bfs(Edges, Source, Queue1, Dist1, Pairs1, Dist, Pairs).

tspd5_expand([], _, _, _, Queue, Dist, Pairs, Queue, Dist, Pairs) :- !.
tspd5_expand([N|Ns], Pred, Source, ND, Queue0, Dist0, Pairs0,
             Queue, Dist, Pairs) :-
    tspd5_cand_pairs(Pred, Source, N, Dist0, Pairs0, Cand0),
    sort(Cand0, Cand),
    (   get_dict(N, Dist0, D0)
    ->  (   D0 =:= ND
        ->  get_dict(N, Pairs0, Old),
            append(Old, Cand, Merged0),
            sort(Merged0, Merged),
            put_dict(N, Pairs0, Merged, Pairs1),
            Dist1 = Dist0,
            Queue1 = Queue0
        ;   Dist1 = Dist0,
            Pairs1 = Pairs0,
            Queue1 = Queue0
        )
    ;   put_dict(N, Dist0, ND, Dist1),
        put_dict(N, Pairs0, Cand, Pairs1),
        append(Queue0, [N-ND], Queue1)
    ),
    tspd5_expand(Ns, Pred, Source, ND, Queue1, Dist1, Pairs1,
                 Queue, Dist, Pairs).

%% Candidate correlated pairs contributed by edge Pred→N at this depth.
tspd5_cand_pairs(Pred, Source, N, _Dist, _Pairs, [N-Source]) :-
    Pred == Source, !.
tspd5_cand_pairs(Pred, _Source, _N, _Dist, Pairs, Cand) :-
    (   get_dict(Pred, Pairs, PairList)
    ->  true
    ;   PairList = []
    ),
    findall(Step-Pred, member(Step-_, PairList), Cand0),
    sort(Cand0, Cand).

% ============================================================
% Fixture matrix — edges + query sources
% ============================================================

tspd5_fixture(chain, [a-b, b-c, c-d], [a, c, d, z]).
tspd5_fixture(equal_diamond, [a-b, a-c, b-d, c-d], [a, d]).
% Adversarial correlated diamond: independent Step×Parent cross-product
% would falsely emit (t,b,q,3) and (t,c,p,3).
tspd5_fixture(correlated_diamond,
    [a-b, a-c, b-p, c-q, p-t, q-t], [a, t]).
tspd5_fixture(multi_parent_one_step,
    [a-s, s-p, s-q, p-t, q-t], [a]).
tspd5_fixture(multi_step_one_parent,
    [a-b, a-c, b-p, c-p, p-t], [a]).
tspd5_fixture(unequal_paths, [a-b, b-c, c-t, a-t], [a, t]).
tspd5_fixture(dup_edges, [a-b, a-b, b-c, b-c], [a, b]).
tspd5_fixture(self_loop, [a-a, a-b], [a, b]).
tspd5_fixture(two_cycle, [a-b, b-a], [a, b]).
tspd5_fixture(cycle_exit, [a-b, b-c, c-a, c-d], [a, c, d]).
tspd5_fixture(sink_disconnected, [a-b, c-d], [a, b, c, z]).
% Mixed-domain rows (numeric atoms) must be ignored by native filters;
% oracle fixtures stay atom-edge only. Parity tests cover mixed rows
% separately via generated programs.

% ============================================================
% Literal expected quads tspd(T,Step,P,D), sorted
% ============================================================

tspd5_fixture_expected(chain, a,
    [tspd(b, b, a, 1), tspd(c, b, b, 2), tspd(d, b, c, 3)]).
tspd5_fixture_expected(chain, c, [tspd(d, d, c, 1)]).
tspd5_fixture_expected(chain, d, []).
tspd5_fixture_expected(chain, z, []).

% Equal-shortest diamond: d has correlated (b,b) and (c,c) at distance 2.
tspd5_fixture_expected(equal_diamond, a,
    [tspd(b, b, a, 1), tspd(c, c, a, 1),
     tspd(d, b, b, 2), tspd(d, c, c, 2)]).
tspd5_fixture_expected(equal_diamond, d, []).

% Adversarial: only correlated pairs — never (t,b,q,3) / (t,c,p,3).
tspd5_fixture_expected(correlated_diamond, a,
    [tspd(b, b, a, 1), tspd(c, c, a, 1),
     tspd(p, b, b, 2), tspd(q, c, c, 2),
     tspd(t, b, p, 3), tspd(t, c, q, 3)]).
tspd5_fixture_expected(correlated_diamond, t, []).

% One first step, multiple equal-shortest parents of t.
tspd5_fixture_expected(multi_parent_one_step, a,
    [tspd(p, s, s, 2), tspd(q, s, s, 2),
     tspd(s, s, a, 1),
     tspd(t, s, p, 3), tspd(t, s, q, 3)]).

% Multiple first steps sharing the same parent of t.
tspd5_fixture_expected(multi_step_one_parent, a,
    [tspd(b, b, a, 1), tspd(c, c, a, 1),
     tspd(p, b, b, 2), tspd(p, c, c, 2),
     tspd(t, b, p, 3), tspd(t, c, p, 3)]).

% Direct a→t beats a→b→c→t.
tspd5_fixture_expected(unequal_paths, a,
    [tspd(b, b, a, 1), tspd(c, b, b, 2), tspd(t, t, a, 1)]).
tspd5_fixture_expected(unequal_paths, t, []).

tspd5_fixture_expected(dup_edges, a,
    [tspd(b, b, a, 1), tspd(c, b, b, 2)]).
tspd5_fixture_expected(dup_edges, b, [tspd(c, c, b, 1)]).

tspd5_fixture_expected(self_loop, a,
    [tspd(a, a, a, 1), tspd(b, b, a, 1)]).
tspd5_fixture_expected(self_loop, b, []).

tspd5_fixture_expected(two_cycle, a,
    [tspd(a, b, b, 2), tspd(b, b, a, 1)]).
tspd5_fixture_expected(two_cycle, b,
    [tspd(a, a, b, 1), tspd(b, a, a, 2)]).

tspd5_fixture_expected(cycle_exit, a,
    [tspd(a, b, c, 3), tspd(b, b, a, 1),
     tspd(c, b, b, 2), tspd(d, b, c, 3)]).
tspd5_fixture_expected(cycle_exit, c,
    [tspd(a, a, c, 1), tspd(b, a, a, 2),
     tspd(c, a, b, 3), tspd(d, d, c, 1)]).
tspd5_fixture_expected(cycle_exit, d, []).

tspd5_fixture_expected(sink_disconnected, a, [tspd(b, b, a, 1)]).
tspd5_fixture_expected(sink_disconnected, b, []).
tspd5_fixture_expected(sink_disconnected, c, [tspd(d, d, c, 1)]).
tspd5_fixture_expected(sink_disconnected, z, []).
