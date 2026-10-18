:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% tc2_contract_oracle.pl — language-neutral finite oracle + fixture matrix
% for the hybrid WAM transitive_closure2 contract (strict R+).
%
% See docs/design/WAM_TRANSITIVE_CLOSURE2_CONTRACT.md.
%
% The oracle is a finite BFS over an explicit edge list. It must NOT be
% implemented by running cyclic Prolog recursion (path proofs / non-termination).

:- module(tc2_contract_oracle, [
    tc2_oracle_reachable/3,     % +Edges, +Source, -SortedTargets
    tc2_oracle_reaches/3,       % +Edges, +Source, +Target
    tc2_fixture/3,              % ?Name, -Edges, -Sources
    tc2_fixture_expected/3      % +Name, +Source, -SortedTargets
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(ordsets)).

%% tc2_oracle_reachable(+Edges, +Source, -SortedTargets) is det.
%  Edges: list of From-To. Strict R+: Target appears iff a ≥1-edge path
%  exists. Source appears iff self-loop or nonempty cycle back.
tc2_oracle_reachable(Edges, Source, Sorted) :-
    tc2_bfs(Edges, [Source], [], [], Acc),
    sort(Acc, Sorted).

%% tc2_oracle_reaches(+Edges, +Source, +Target) is semidet.
tc2_oracle_reaches(Edges, Source, Target) :-
    tc2_oracle_reachable(Edges, Source, Sorted),
    memberchk(Target, Sorted).

%% BFS: queue starts at Source; Seen tracks nodes discovered *via an edge*.
tc2_bfs(_, [], _Seen, Acc, Acc) :- !.
tc2_bfs(Edges, [Node|Queue], Seen, Acc0, Acc) :-
    findall(Next,
            ( member(Node-Next, Edges),
              \+ memberchk(Next, Seen)
            ),
            News0),
    % Preserve first-discovery order for determinism; unique within layer.
    list_to_set(News0, News),
    append(Seen, News, Seen1),
    append(Acc0, News, Acc1),
    append(Queue, News, Queue1),
    tc2_bfs(Edges, Queue1, Seen1, Acc1, Acc).

% ============================================================
% Fixture matrix
% ============================================================

%% tc2_fixture(?Name, -Edges, -QuerySources)
tc2_fixture(chain, [a-b, b-c, c-d], [a, c, d, z]).
tc2_fixture(branch, [a-b, a-c, b-d, c-e], [a, d, e]).
tc2_fixture(diamond, [a-b, a-c, b-d, c-d], [a, d]).
tc2_fixture(dup_edges, [a-b, a-b, b-c, b-c], [a, b]).
tc2_fixture(sink_disconnected, [a-b, c-d], [a, b, c, z]).
tc2_fixture(self_loop, [a-a, a-b], [a, b]).
tc2_fixture(two_cycle, [a-b, b-a], [a, b]).
tc2_fixture(long_cycle_exit, [a-b, b-c, c-a, c-d], [a, c, d]).

%% tc2_fixture_expected(+Name, +Source, -SortedTargets)
%  Independent expected results for every declared fixture query. Keep these
%  literal rather than deriving them through tc2_oracle_reachable/3: the
%  fixture matrix is intended to detect regressions in the oracle itself.
tc2_fixture_expected(chain, a, [b, c, d]).
tc2_fixture_expected(chain, c, [d]).
tc2_fixture_expected(chain, d, []).
tc2_fixture_expected(chain, z, []).

tc2_fixture_expected(branch, a, [b, c, d, e]).
tc2_fixture_expected(branch, d, []).
tc2_fixture_expected(branch, e, []).

tc2_fixture_expected(diamond, a, [b, c, d]).
tc2_fixture_expected(diamond, d, []).

tc2_fixture_expected(dup_edges, a, [b, c]).
tc2_fixture_expected(dup_edges, b, [c]).

tc2_fixture_expected(sink_disconnected, a, [b]).
tc2_fixture_expected(sink_disconnected, b, []).
tc2_fixture_expected(sink_disconnected, c, [d]).
tc2_fixture_expected(sink_disconnected, z, []).

tc2_fixture_expected(self_loop, a, [a, b]).
tc2_fixture_expected(self_loop, b, []).

tc2_fixture_expected(two_cycle, a, [a, b]).
tc2_fixture_expected(two_cycle, b, [a, b]).

tc2_fixture_expected(long_cycle_exit, a, [a, b, c, d]).
tc2_fixture_expected(long_cycle_exit, c, [a, b, c, d]).
tc2_fixture_expected(long_cycle_exit, d, []).
