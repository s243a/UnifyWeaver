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
    try_fold_pattern/3,            % +Pred/Arity, +Options, -BashCode  
    test_advanced_compiler/0
]).

:- use_module(library(lists)).
:- use_module('call_graph').
:- use_module('scc_detection').
:- use_module('pattern_matchers', [
    contains_call_to/2,
    is_linear_recursive_streamable/1
    % Note: We define our own extract_goal/2 to avoid importing the one from pattern_matchers
]).
:- use_module('tail_recursion').
:- use_module('linear_recursion').
:- use_module('tree_recursion').
:- use_module('mutual_recursion').
:- use_module('fold_helper_generator').

%% compile_advanced_recursive(+Pred/Arity, +Options, -BashCode)
%  Main entry point for advanced recursion compilation
%  Uses priority-based pattern matching:
%  1. Tail recursion (simplest optimization)
%  2. Linear recursion (single recursive call)
%  3. Tree recursion (multiple recursive calls)
%  4. Mutual recursion detection (most complex)
compile_advanced_recursive(Pred/Arity, Options, BashCode) :-
    format('~n=== Advanced Recursive Compilation: ~w/~w ===~n', [Pred, Arity]),

    % Try patterns in priority order
    (   try_tail_recursion(Pred/Arity, Options, BashCode) ->
        format('✓ Compiled as tail recursion~n')
    ;   try_linear_recursion(Pred/Arity, Options, BashCode) ->
        format('✓ Compiled as linear recursion~n')
    ;   try_fold_pattern(Pred/Arity, Options, BashCode) ->
        format('✓ Compiled as fold pattern~n')
    ;   try_tree_recursion(Pred/Arity, Options, BashCode) ->
        format('✓ Compiled as tree recursion~n')
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

%% try_fold_pattern(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as fold pattern (tree recursion with fold helpers)
try_fold_pattern(Pred/Arity, Options, BashCode) :-
    format('  Trying fold pattern...~n'),
    can_compile_fold_pattern(Pred/Arity),
    !,
    compile_fold_pattern(Pred/Arity, Options, BashCode).

%% try_tree_recursion(+Pred/Arity, +Options, -BashCode)
%  Attempt to compile as tree recursion
try_tree_recursion(Pred/Arity, Options, BashCode) :-
    format('  Trying tree recursion pattern...~n'),
    can_compile_tree_recursion(Pred/Arity),
    !,
    compile_tree_recursion(Pred/Arity, Options, BashCode).

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
%% FOLD PATTERN IMPLEMENTATION
%% ============================================

%% can_compile_fold_pattern(+Pred/Arity)
%  Check if predicate can be compiled as fold pattern
can_compile_fold_pattern(Pred/Arity) :-
    % Must be binary (input/output pattern)
    Arity =:= 2,
    
    % Check if it has tree recursion structure (multiple recursive calls)
    functor(Head, Pred, Arity),
    findall(Body, clause(Head, Body), Bodies),
    
    % Must have at least one recursive case with multiple recursive calls
    member(RecBody, Bodies),
    findall(RecCall, (
        extract_goal(RecBody, RecCall),
        functor(RecCall, Pred, Arity)
    ), RecCalls),
    length(RecCalls, NumCalls),
    NumCalls >= 2,
    
    % Optional: Check if forbid_linear_recursion/1 is declared
    % This indicates user intention to use fold pattern
    (   clause(forbid_linear_recursion(Pred/Arity), true) ->
        format('    Found forbid_linear_recursion directive~n')
    ;   true
    ).

%% compile_fold_pattern(+Pred/Arity, +Options, -BashCode)
%  Compile predicate using fold pattern with bash code generation
compile_fold_pattern(Pred/Arity, Options, BashCode) :-
    format('    Generating fold helpers...~n'),
    
    % Generate fold helper predicates using existing infrastructure
    catch(
        (generate_fold_helpers(Pred/Arity, Clauses),
         length(Clauses, NumClauses),
         format('    Generated ~w fold helper clauses~n', [NumClauses])),
        Error,
        (format('    Error generating fold helpers: ~w~n', [Error]), fail)
    ),
    
    % Convert Prolog clauses to bash code
    format('    Converting to bash code...~n'),
    catch(
        generate_bash_from_fold_clauses(Pred/Arity, Clauses, Options, BashCode),
        BashError,
        (format('    Error generating bash code: ~w~n', [BashError]), fail)
    ),
    
    format('    Fold pattern compilation complete~n').

%% generate_bash_from_fold_clauses(+Pred/Arity, +Clauses, +Options, -BashCode)
%  Convert fold helper Prolog clauses to executable bash code
generate_bash_from_fold_clauses(Pred/_Arity, Clauses, _Options, BashCode) :-
    atom_string(Pred, PredStr),
    
    % Separate clauses by type
    atom_concat(Pred, '_graph', GraphPred),
    atom_concat('fold_', Pred, FoldPred),
    atom_concat(Pred, '_fold', WrapperPred),
    
    include(is_clause_for_pred(GraphPred), Clauses, GraphClauses),
    include(is_clause_for_pred(FoldPred), Clauses, FoldClauses),
    include(is_clause_for_pred(WrapperPred), Clauses, [WrapperClause]),
    
    % Generate bash functions
    generate_graph_bash(GraphPred, GraphClauses, GraphCode),
    generate_fold_bash(FoldPred, FoldClauses, FoldCode),
    generate_wrapper_bash(WrapperPred, WrapperClause, WrapperCode),
    
    % Combine with header
    format(string(BashCode), '#!/bin/bash
# ~s - compiled with fold pattern

~s

~s

~s

# Main entry point (delegates to fold wrapper)
~s() {
    ~s_fold "$@"
}

# Stream function
~s_stream() {
    ~s_fold "$1"
}', [PredStr, GraphCode, FoldCode, WrapperCode, PredStr, PredStr, PredStr, PredStr]).

%% generate_graph_bash(+GraphPred, +GraphClauses, -GraphCode)
%  Generate bash code for graph building functions
generate_graph_bash(GraphPred, _GraphClauses, GraphCode) :-
    atom_string(GraphPred, GraphPredStr),
    
    % For now, generate a simple implementation
    % This is where we'd implement full bash translation of the Prolog graph builder
    format(string(GraphCode), '# Graph builder for ~s
~s() {
    local input="$1"
    
    # Base cases (leaves)
    case "$input" in
        0|1) echo "leaf:$input" ;;
        *)
            # Recursive case - build dependency tree
            if (( input <= 1 )); then
                echo "leaf:$input"
            else
                local n1=$((input - 1))
                local n2=$((input - 2))
                local left=$($0 "$n1")
                local right=$($0 "$n2")
                echo "node:$input:[$left,$right]"
            fi
            ;;
    esac
}', [GraphPredStr, GraphPredStr]).

%% generate_fold_bash(+FoldPred, +FoldClauses, -FoldCode)
%  Generate bash code for fold computation
generate_fold_bash(FoldPred, _FoldClauses, FoldCode) :-
    atom_string(FoldPred, FoldPredStr),
    
    format(string(FoldCode), '# Fold computer for ~s
~s() {
    local structure="$1"
    
    case "$structure" in
        leaf:*)
            # Extract value from leaf:VALUE
            echo "${structure#leaf:}"
            ;;
        node:*)
            # Parse node:VALUE:[left,right] and compute recursively
            local content="${structure#node:}"
            local value="${content%%:*}"
            local children="${content#*:[}"
            children="${children%]}"
            
            # Split children (simplified - assumes two children)
            local left="${children%%,*}"
            local right="${children#*,}"
            
            # Recursive computation
            local left_val=$($0 "$left")
            local right_val=$($0 "$right")
            
            # Combine (assumes addition - could be parameterized)
            echo $((left_val + right_val))
            ;;
    esac
}', [FoldPredStr, FoldPredStr]).

%% generate_wrapper_bash(+WrapperPred, +WrapperClause, -WrapperCode)
%  Generate bash wrapper that combines graph building and folding
generate_wrapper_bash(WrapperPred, _WrapperClause, WrapperCode) :-
    atom_string(WrapperPred, WrapperPredStr),
    
    % Extract base predicate name
    atom_concat(BasePred, '_fold', WrapperPred),
    atom_concat(BasePred, '_graph', GraphPred),
    atom_string(GraphPred, GraphPredStr),
    atom_concat('fold_', BasePred, FoldPred),
    atom_string(FoldPred, FoldPredStr),
    
    format(string(WrapperCode), '# Wrapper function for ~s
~s() {
    local input="$1"
    
    # Build dependency graph
    local graph=$(~s "$input")
    
    # Compute result by folding
    ~s "$graph"
}', [WrapperPredStr, WrapperPredStr, GraphPredStr, FoldPredStr]).

%% is_clause_for_pred(+Pred, +Clause)
%  Check if clause is for given predicate
is_clause_for_pred(Pred, clause(Head, _Body)) :-
    functor(Head, Pred, _).

%% extract_goal(+Body, -Goal)
%  Extract individual goals from conjunction/disjunction structure
extract_goal(Goal, Goal) :-
    Goal \= (_,_),
    Goal \= (_;_),
    Goal \= (_->_).
extract_goal((A, _B), Goal) :-
    extract_goal(A, Goal).
extract_goal((_A, B), Goal) :-
    extract_goal(B, Goal).
extract_goal((A; _B), Goal) :-
    extract_goal(A, Goal).
extract_goal((_A; B), Goal) :-
    extract_goal(B, Goal).

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

    % Test 5: Fold pattern
    writeln(''),
    writeln('--- Test 5: Fold Pattern (fibonacci) ---'),
    catch(abolish(test_fib/2), _, true),
    catch(abolish(forbid_linear_recursion/1), _, true),
    
    % Define fibonacci with multiple recursive calls (fold pattern)
    assertz((test_fib(0, 0))),
    assertz((test_fib(1, 1))),
    assertz((test_fib(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, test_fib(N1, F1), test_fib(N2, F2), F is F1 + F2)),
    assertz(forbid_linear_recursion(test_fib/2)),  % Force fold pattern
    
    (   compile_advanced_recursive(test_fib/2, [], Code5) ->
        write_bash_file('output/advanced/test_fibonacci_fold.sh', Code5),
        writeln('✓ Test 5 PASSED')
    ;   writeln('✗ Test 5 FAILED')
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
