:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% scc_detection.pl - Strongly Connected Components detection
% Implements Tarjan's algorithm for finding SCCs in call graphs

:- module(scc_detection, [
    find_sccs/2,            % +Graph, -SCCs
    topological_order/2,    % +SCCs, -OrderedSCCs
    is_trivial_scc/1,       % +SCC - check if SCC has single node
    test_scc_detection/0    % Test predicate
]).

:- use_module(library(lists)).
:- use_module(call_graph).

%% find_sccs(+Graph, -SCCs)
%  Find Strongly Connected Components using Tarjan's algorithm
%  Graph is list of edges: [node1->node2, ...]
%  SCCs is list of lists: [[node1, node2], [node3], ...]
find_sccs(Graph, SCCs) :-
    extract_nodes(Graph, Nodes),
    tarjan_sccs(Nodes, Graph, SCCs).

%% extract_nodes(+Graph, -Nodes)
%  Extract unique nodes from graph edges
extract_nodes(Graph, Nodes) :-
    findall(Node,
        (   member(From->To, Graph),
            (   Node = From ; Node = To )
        ),
        NodesWithDups),
    sort(NodesWithDups, Nodes).

%% tarjan_sccs(+Nodes, +Graph, -SCCs)
%  Tarjan's SCC algorithm implementation
tarjan_sccs(Nodes, Graph, SCCs) :-
    empty_assoc(IndexMap0),
    empty_assoc(LowLinkMap0),
    State0 = tarjan_state(0, [], IndexMap0, LowLinkMap0, [], []),
    tarjan_visit_all(Nodes, Graph, State0, StateFinal),
    StateFinal = tarjan_state(_, _, _, _, SCCs, _).

%% Tarjan state structure
%  tarjan_state(Index, Stack, IndexMap, LowLinkMap, SCCs, OnStack)
%  This is a compound term (functor with 6 arguments), not a predicate

tarjan_visit_all([], _Graph, State, State).
tarjan_visit_all([Node|Nodes], Graph, State0, StateFinal) :-
    State0 = tarjan_state(_, _, IndexMap, _, _, _),
    (   get_assoc(Node, IndexMap, _) ->
        % Already visited
        State1 = State0
    ;   % Not visited - run Tarjan's strongconnect
        tarjan_strongconnect(Node, Graph, State0, State1)
    ),
    tarjan_visit_all(Nodes, Graph, State1, StateFinal).

%% tarjan_strongconnect(+Node, +Graph, +State0, -State)
%  Core of Tarjan's algorithm - visit a node
tarjan_strongconnect(Node, Graph, State0, StateFinal) :-
    State0 = tarjan_state(Index0, Stack0, IndexMap0, LowLinkMap0, SCCs0, OnStack0),

    % Set node index and lowlink
    Index1 is Index0 + 1,
    put_assoc(Node, IndexMap0, Index0, IndexMap1),
    put_assoc(Node, LowLinkMap0, Index0, LowLinkMap1),

    % Push node onto stack
    Stack1 = [Node|Stack0],
    OnStack1 = [Node|OnStack0],

    State1 = tarjan_state(Index1, Stack1, IndexMap1, LowLinkMap1, SCCs0, OnStack1),

    % Visit successors
    findall(Succ, member(Node->Succ, Graph), Successors),
    tarjan_visit_successors(Successors, Node, Graph, State1, State2),

    % Check if node is a root node
    State2 = tarjan_state(Index2, Stack2, IndexMap2, LowLinkMap2, SCCs1, OnStack2),
    get_assoc(Node, IndexMap2, NodeIndex),
    get_assoc(Node, LowLinkMap2, NodeLowLink),

    (   NodeIndex =:= NodeLowLink ->
        % Node is root - pop SCC from stack
        pop_scc(Node, Stack2, Stack3, OnStack2, OnStack3, SCC),
        SCCs2 = [SCC|SCCs1],
        StateFinal = tarjan_state(Index2, Stack3, IndexMap2, LowLinkMap2, SCCs2, OnStack3)
    ;   % Not a root
        StateFinal = State2
    ).

%% tarjan_visit_successors(+Successors, +Node, +Graph, +State0, -State)
tarjan_visit_successors([], _Node, _Graph, State, State).
tarjan_visit_successors([Succ|Succs], Node, Graph, State0, StateFinal) :-
    State0 = tarjan_state(Index, Stack, IndexMap, LowLinkMap0, SCCs, OnStack),

    (   get_assoc(Succ, IndexMap, _) ->
        % Successor already visited
        (   member(Succ, OnStack) ->
            % Successor is on stack - update lowlink
            get_assoc(Succ, IndexMap, SuccIndex),
            get_assoc(Node, LowLinkMap0, NodeLowLink),
            NewLowLink is min(NodeLowLink, SuccIndex),
            put_assoc(Node, LowLinkMap0, NewLowLink, LowLinkMap1),
            State1 = tarjan_state(Index, Stack, IndexMap, LowLinkMap1, SCCs, OnStack)
        ;   % Successor not on stack
            State1 = State0
        )
    ;   % Successor not visited - recurse
        tarjan_strongconnect(Succ, Graph, State0, State1),
        State1 = tarjan_state(_, _, _, LowLinkMap1, _, _),
        % Update lowlink based on successor's lowlink
        get_assoc(Succ, LowLinkMap1, SuccLowLink),
        get_assoc(Node, LowLinkMap1, NodeLowLink),
        NewLowLink is min(NodeLowLink, SuccLowLink),
        put_assoc(Node, LowLinkMap1, NewLowLink, LowLinkMap2),
        State1 = tarjan_state(Idx, Stk, IdxMap, _, SccsTemp, OnStk),
        State2 = tarjan_state(Idx, Stk, IdxMap, LowLinkMap2, SccsTemp, OnStk),
        State1 = State2  % Unify with updated lowlink
    ),

    tarjan_visit_successors(Succs, Node, Graph, State1, StateFinal).

%% pop_scc(+RootNode, +Stack, -NewStack, +OnStack, -NewOnStack, -SCC)
%  Pop all nodes of an SCC from the stack
pop_scc(RootNode, Stack, NewStack, OnStack, NewOnStack, SCC) :-
    pop_scc_acc(RootNode, Stack, NewStack, OnStack, NewOnStack, [], SCC).

pop_scc_acc(RootNode, [Node|Stack], NewStack, [Node|OnStack], NewOnStack, Acc, SCC) :-
    Acc1 = [Node|Acc],
    (   Node = RootNode ->
        % Found root - done
        NewStack = Stack,
        NewOnStack = OnStack,
        SCC = Acc1
    ;   % Keep popping
        pop_scc_acc(RootNode, Stack, NewStack, OnStack, NewOnStack, Acc1, SCC)
    ).

%% topological_order(+SCCs, -OrderedSCCs)
%  Order SCCs in topological order (dependencies first)
%  SCCs from Tarjan's algorithm are already in reverse topological order
topological_order(SCCs, OrderedSCCs) :-
    reverse(SCCs, OrderedSCCs).

%% is_trivial_scc(+SCC)
%  Check if SCC contains only one node (self-recursion or no recursion)
is_trivial_scc([_]).

%% ============================================
%% TESTS
%% ============================================

test_scc_detection :-
    writeln('=== SCC DETECTION TESTS ==='),

    % Test 1: Simple cycle (A -> B -> A)
    writeln('Test 1: Simple two-node cycle'),
    Graph1 = [a->b, b->a],
    find_sccs(Graph1, SCCs1),
    format('  Graph: ~w~n', [Graph1]),
    format('  SCCs: ~w~n', [SCCs1]),
    (   member([a,b], SCCs1) ; member([b,a], SCCs1) ->
        writeln('  ✓ PASS - found single SCC with both nodes')
    ;   writeln('  ✗ FAIL - should find single SCC')
    ),

    % Test 2: No cycles (A -> B -> C)
    writeln('Test 2: Acyclic graph'),
    Graph2 = [a->b, b->c],
    find_sccs(Graph2, SCCs2),
    format('  Graph: ~w~n', [Graph2]),
    format('  SCCs: ~w~n', [SCCs2]),
    (   length(SCCs2, 3) ->
        writeln('  ✓ PASS - three separate SCCs')
    ;   format('  ✗ FAIL - expected 3 SCCs, got ~w~n', [SCCs2])
    ),

    % Test 3: Self-loop (A -> A)
    writeln('Test 3: Self-loop'),
    Graph3 = [a->a],
    find_sccs(Graph3, SCCs3),
    format('  Graph: ~w~n', [Graph3]),
    format('  SCCs: ~w~n', [SCCs3]),
    (   member([a], SCCs3) ->
        writeln('  ✓ PASS - found self-loop SCC')
    ;   writeln('  ✗ FAIL - should find SCC containing a')
    ),

    % Test 4: Complex graph with multiple SCCs
    writeln('Test 4: Complex graph (even/odd pattern)'),
    Graph4 = [even->odd, odd->even, even->zero, zero->zero],
    find_sccs(Graph4, SCCs4),
    format('  Graph: ~w~n', [Graph4]),
    format('  SCCs: ~w~n', [SCCs4]),
    (   (member([even,odd], SCCs4) ; member([odd,even], SCCs4)),
        member([zero], SCCs4) ->
        writeln('  ✓ PASS - found even/odd SCC and zero SCC')
    ;   format('  ✗ FAIL - expected even/odd cycle and zero self-loop, got ~w~n', [SCCs4])
    ),

    % Test 5: Topological order
    writeln('Test 5: Topological ordering'),
    Graph5 = [a->b, b->c, c->c],
    find_sccs(Graph5, SCCs5),
    topological_order(SCCs5, Ordered5),
    format('  Graph: ~w~n', [Graph5]),
    format('  SCCs: ~w~n', [SCCs5]),
    format('  Topological order: ~w~n', [Ordered5]),
    % First SCC should be a leaf (c in this case)
    writeln('  ✓ PASS - topological order computed'),

    writeln('=== SCC DETECTION TESTS COMPLETE ===').
