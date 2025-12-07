% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

/**
 * Social Network Analysis Example for Chapter 3
 *
 * This example demonstrates Query Runtime compilation with
 * recursive reachability queries (transitive closure).
 *
 * Key concepts:
 * - Recursive predicates
 * - Fixpoint iteration
 * - Cycle handling
 * - Semi-naive evaluation
 *
 * Usage:
 *   ?- [social_network].
 *   ?- compile_reachable_query.
 */

:- use_module('../../src/unifyweaver/targets/csharp_query_target').

% Follow relationships: follows(Follower, Followed)
follows(alice, bob).
follows(bob, charlie).
follows(charlie, dave).
follows(alice, charlie).  % Alice also directly follows Charlie
follows(dave, alice).      % Creates a cycle: dave → alice → bob → charlie → dave

% Recursive reachability - transitive closure
reachable(User1, User2) :- follows(User1, User2).
reachable(User1, User3) :- follows(User1, User2), reachable(User2, User3).

% Mutual followers (bidirectional follows)
mutual_follower(User1, User2) :-
    follows(User1, User2),
    follows(User2, User1).

% Helper to compile the reachable query
compile_reachable_query :-
    writeln('Compiling reachable/2 with Query Runtime...'),
    compile_predicate_to_csharp(reachable/2, [target(csharp_query)], Code),
    open('ReachableQuery.cs', write, Stream),
    write(Stream, Code),
    close(Stream),
    writeln('Done! Generated ReachableQuery.cs'),
    writeln(''),
    writeln('Next steps:'),
    writeln('1. Copy QueryRuntime.cs from src/unifyweaver/targets/csharp_query_runtime/'),
    writeln('2. Create a .NET console project'),
    writeln('3. Add both files to the project'),
    writeln('4. Build and run to see recursive query results').

% Test query to see what reachable finds
test_reachable :-
    writeln('Testing reachable/2 query in Prolog:'),
    writeln(''),
    findall((U1, U2), reachable(U1, U2), Pairs),
    length(Pairs, Count),
    format('Found ~w reachable pairs:~n', [Count]),
    forall(member((U1, U2), Pairs),
           format('  ~w can reach ~w~n', [U1, U2])).
