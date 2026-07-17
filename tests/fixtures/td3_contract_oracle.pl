:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% td3_contract_oracle.pl — language-neutral finite oracle + fixture matrix
% for the hybrid WAM transitive_distance3 contract (dist+).
%
% See docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md.
%
% Expected fixture facts are LITERAL and independent — they must not be
% derived by calling the oracle under test. The oracle is provided for
% generator helpers and cross-checks; parity tests assert against the
% literal expectations below.

:- module(td3_contract_oracle, [
    td3_oracle_pairs/3,          % +Edges, +Source, -SortedPairs
    td3_oracle_distance/4,       % +Edges, +Source, +Target, -Dist
    td3_fixture/3,               % ?Name, -Edges, -Sources
    td3_fixture_expected/3       % +Name, +Source, -SortedPairs (LITERAL)
]).

:- use_module(library(lists)).
:- use_module(library(apply)).

%% td3_oracle_pairs(+Edges, +Source, -SortedPairs) is det.
%  Edges: list of From-To. SortedPairs: list of Target-Distance,
%  sorted by Target then Distance. Finite BFS; visited tracks nodes
%  discovered via an outgoing edge (not seeded with Source).
td3_oracle_pairs(Edges, Source, Sorted) :-
    td3_bfs(Edges, [Source-0], [], [], Acc),
    sort(Acc, Sorted).

%% td3_oracle_distance(+Edges, +Source, +Target, -Dist) is semidet.
td3_oracle_distance(Edges, Source, Target, Dist) :-
    td3_oracle_pairs(Edges, Source, Sorted),
    memberchk(Target-Dist, Sorted).

%% BFS: queue entries Node-Depth. Seen tracks edge-discovered nodes.
td3_bfs(_, [], _Seen, Acc, Acc) :- !.
td3_bfs(Edges, [Node-Depth|Queue], Seen, Acc0, Acc) :-
    NextDepth is Depth + 1,
    findall(Next,
            ( member(Node-Next, Edges),
              \+ memberchk(Next, Seen)
            ),
            News0),
    list_to_set(News0, News),
    findall(N-NextDepth, member(N, News), NewPairs),
    append(Seen, News, Seen1),
    append(Acc0, NewPairs, Acc1),
    append(Queue, NewPairs, Queue1),
    td3_bfs(Edges, Queue1, Seen1, Acc1, Acc).

% ============================================================
% Fixture matrix — edges + query sources
% ============================================================

td3_fixture(chain, [a-b, b-c, c-d], [a, c, d, z]).
td3_fixture(equal_diamond, [a-b, a-c, b-d, c-d], [a, d]).
td3_fixture(unequal_paths, [a-b, b-c, c-t, a-t], [a, t]).
td3_fixture(dup_edges, [a-b, a-b, b-c, b-c], [a, b]).
td3_fixture(self_loop, [a-a, a-b], [a, b]).
td3_fixture(two_cycle, [a-b, b-a], [a, b]).
td3_fixture(cycle_exit, [a-b, b-c, c-a, c-d], [a, c, d]).
td3_fixture(sink_disconnected, [a-b, c-d], [a, b, c, z]).

% ============================================================
% Literal expected pairs (Target-Distance), sorted
% ============================================================

td3_fixture_expected(chain, a, [b-1, c-2, d-3]).
td3_fixture_expected(chain, c, [d-1]).
td3_fixture_expected(chain, d, []).
td3_fixture_expected(chain, z, []).

td3_fixture_expected(equal_diamond, a, [b-1, c-1, d-2]).
td3_fixture_expected(equal_diamond, d, []).

% Direct a→t beats a→b→c→t (distance 1, not 3).
td3_fixture_expected(unequal_paths, a, [b-1, c-2, t-1]).
td3_fixture_expected(unequal_paths, t, []).

td3_fixture_expected(dup_edges, a, [b-1, c-2]).
td3_fixture_expected(dup_edges, b, [c-1]).

td3_fixture_expected(self_loop, a, [a-1, b-1]).
td3_fixture_expected(self_loop, b, []).

td3_fixture_expected(two_cycle, a, [a-2, b-1]).
td3_fixture_expected(two_cycle, b, [a-1, b-2]).

td3_fixture_expected(cycle_exit, a, [a-3, b-1, c-2, d-3]).
td3_fixture_expected(cycle_exit, c, [a-1, b-2, c-3, d-1]).
td3_fixture_expected(cycle_exit, d, []).

td3_fixture_expected(sink_disconnected, a, [b-1]).
td3_fixture_expected(sink_disconnected, b, []).
td3_fixture_expected(sink_disconnected, c, [d-1]).
td3_fixture_expected(sink_disconnected, z, []).
