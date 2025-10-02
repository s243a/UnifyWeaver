:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% advanced_recursive_compiler.pl - Advanced recursion pattern compiler
% Orchestrates compilation of tail recursion, linear recursion, and mutual recursion
% Uses priority-based strategy: try simpler patterns first

:- module(advanced_recursive_compiler, [
    compile_advanced_recursive/3,   % +Pred/Arity, +Options, -BashCode
    compile_predicate_group/3,      % +Predicates, +Options, -BashCode
    test_advanced_compiler/0
]).

:- use_module(library(lists)).
:- use_module('call_graph').
:- use_module('scc_detection').
:- use_module('pattern_matchers').
:- use_module('tail_recursion').
:- use_module('linear_recursion').
:- use_module('mutual_recursion').

%% compile_advanced_recursive(+Pred/Arity, +Options, -BashCode)
%  Main entry point for advanced recursion compilation
%  Uses priority-based pattern matching:
%  1. Tail recursion (simplest optimization)
%  2. Linear recursion (single recursive call)
%  3. Mutual recursion detection (most complex)
compile_advanced_recursive(Pred/Arity, Options, BashCode) :-
    format('~n=== Advanced Recursive Compilation: ~w/~w ===~n', [Pred, Arity]),

    % Try patterns in priority order
    (   try_tail_recursion(Pred/Arity, Options, BashCode) ->
        format('✓ Compiled as tail recursion~n')
    ;   try_linear_recursion(Pred/Arity, Options, BashCode) ->
        format('✓ Compiled as linear recursion~n')
    ;   try_mutual_recursion_detection(Pred/Arity, Options, BashCode) ->
        format('✓ Compiled as part of mutual recursion group~n')
    ;   % No pattern matched - fail back to caller
        format('✗ No advanced pattern matched~n'),
        fail
    ).

%% try_tail_recursion(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as tail recursion
try_tail_recursion(Pred/Arity, Options, BashCode) :-
    format('  Trying tail recursion pattern...~n'),
    can_compile_tail_recursion(Pred/Arity),
    !,
    compile_tail_recursion(Pred/Arity, Options, BashCode).

%% try_linear_recursion(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as linear recursion
try_linear_recursion(Pred/Arity, Options, BashCode) :-
    format('  Trying linear recursion pattern...~n'),
    can_compile_linear_recursion(Pred/Arity),
    !,
    compile_linear_recursion(Pred/Arity, Options, BashCode).

%% try_mutual_recursion_detection(+Pred/Arity, +Options, -BashCode)
%  Detect if predicate is part of mutual recursion group
%  If so, find the group and compile together
try_mutual_recursion_detection(Pred/Arity, Options, BashCode) :-
    format('  Trying mutual recursion detection...~n'),

    % Find all predicates in this predicate's dependency group
    predicates_in_group(Pred/Arity, Group),

    % Check if group size > 1 (true mutual recursion)
    length(Group, GroupSize),
    GroupSize > 1,

    format('  Found mutual recursion group of size ~w~n', [GroupSize]),

    % Check if group can be compiled as mutual recursion
    can_compile_mutual_recursion(Group),
    !,

    % Compile the entire group
    compile_mutual_recursion(Group, Options, BashCode).

%% compile_predicate_group(+Predicates, +Options, -BashCode)
%  Compile a group of predicates together
%  This is useful when user explicitly specifies a group
compile_predicate_group(Predicates, Options, BashCode) :-
    format('~n=== Compiling Predicate Group ===~n'),
    format('Predicates: ~w~n', [Predicates]),

    % Build call graph for the group
    build_call_graph(Predicates, Graph),
    format('Call graph: ~w~n', [Graph]),

    % Find SCCs
    find_sccs(Graph, SCCs),
    format('SCCs found: ~w~n', [SCCs]),

    % Order SCCs topologically (dependencies first)
    topological_order(SCCs, OrderedSCCs),
    format('Topological order: ~w~n', [OrderedSCCs]),

    % Compile each SCC in order
    compile_sccs(OrderedSCCs, Options, CodeParts),

    % Combine all code parts
    atomic_list_concat(CodeParts, '\n\n# ========================================\n\n', BashCode).

%% compile_sccs(+SCCs, +Options, -CodeParts)
%  Compile each SCC using appropriate strategy
compile_sccs([], _Options, []).
compile_sccs([SCC|Rest], Options, [Code|Codes]) :-
    compile_scc(SCC, Options, Code),
    compile_sccs(Rest, Options, Codes).

%% compile_scc(+SCC, +Options, -Code)
%  Compile a single SCC
compile_scc(SCC, Options, Code) :-
    format('~nCompiling SCC: ~w~n', [SCC]),

    (   is_trivial_scc(SCC) ->
        % Single predicate - try simple patterns first
        SCC = [Pred/Arity],
        format('  Trivial SCC (single predicate): ~w/~w~n', [Pred, Arity]),
        compile_single_predicate(Pred/Arity, Options, Code)
    ;   % Non-trivial SCC - true mutual recursion
        format('  Non-trivial SCC (mutual recursion): ~w~n', [SCC]),
        compile_mutual_recursion(SCC, Options, Code)
    ).

%% compile_single_predicate(+Pred/Arity, +Options, -Code)
%  Compile a single predicate using priority order
compile_single_predicate(Pred/Arity, Options, Code) :-
    (   compile_advanced_recursive(Pred/Arity, Options, Code) ->
        true
    ;   % Fall back to basic compilation
        format('  Using basic recursion compilation~n'),
        Code = "# Basic recursion - not yet implemented"
    ).

%% ============================================
%% TESTS
%% ============================================

test_advanced_compiler :-
    writeln('=== ADVANCED RECURSIVE COMPILER TESTS ==='),
    writeln(''),

    % Setup output directory
    (   exists_directory('output/advanced') -> true
    ;   make_directory('output/advanced')
    ),

    % Test 1: Tail recursion
    writeln('--- Test 1: Tail Recursion (count_items) ---'),
    catch(abolish(count_items/3), _, true),
    assertz((count_items([], Acc, Acc))),
    assertz((count_items([_|T], Acc, N) :- Acc1 is Acc + 1, count_items(T, Acc1, N))),

    (   compile_advanced_recursive(count_items/3, [], Code1) ->
        write_bash_file('output/advanced/test_count.sh', Code1),
        writeln('✓ Test 1 PASSED')
    ;   writeln('✗ Test 1 FAILED')
    ),

    % Test 2: Linear recursion
    writeln(''),
    writeln('--- Test 2: Linear Recursion (list_length) ---'),
    catch(abolish(list_length/2), _, true),
    assertz((list_length([], 0))),
    assertz((list_length([_|T], N) :- list_length(T, N1), N is N1 + 1)),

    (   compile_advanced_recursive(list_length/2, [], Code2) ->
        write_bash_file('output/advanced/test_length.sh', Code2),
        writeln('✓ Test 2 PASSED')
    ;   writeln('✗ Test 2 FAILED')
    ),

    % Test 3: Mutual recursion
    writeln(''),
    writeln('--- Test 3: Mutual Recursion (even/odd) ---'),
    catch(abolish(is_even/1), _, true),
    catch(abolish(is_odd/1), _, true),
    assertz(is_even(0)),
    assertz((is_even(N) :- N > 0, N1 is N - 1, is_odd(N1))),
    assertz(is_odd(1)),
    assertz((is_odd(N) :- N > 1, N1 is N - 1, is_even(N1))),

    (   compile_advanced_recursive(is_even/1, [], Code3) ->
        write_bash_file('output/advanced/test_even_odd.sh', Code3),
        writeln('✓ Test 3 PASSED')
    ;   writeln('✗ Test 3 FAILED (expected - may need explicit group compilation)')
    ),

    % Test 4: Predicate group compilation
    writeln(''),
    writeln('--- Test 4: Predicate Group Compilation ---'),
    Predicates = [is_even/1, is_odd/1],
    (   compile_predicate_group(Predicates, [], Code4) ->
        write_bash_file('output/advanced/test_group.sh', Code4),
        writeln('✓ Test 4 PASSED')
    ;   writeln('✗ Test 4 FAILED')
    ),

    writeln(''),
    writeln('=== ADVANCED RECURSIVE COMPILER TESTS COMPLETE ==='),
    writeln('Check output/advanced/ directory for generated files').

%% Helper to write bash files
write_bash_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream),
    % Make executable
    atom_concat('chmod +x ', Path, ChmodCmd),
    shell(ChmodCmd).
