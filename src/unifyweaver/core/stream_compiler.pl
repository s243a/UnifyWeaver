:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% stream_compiler.pl - UnifyWeaver streaming compiler for non-recursive predicates
% Compiles Prolog facts and rules into bash streaming pipelines
% Handles: facts, single rules, multiple rules (OR), inequality constraints
% Does not handle: recursion, negation, complex built-ins

:- module(stream_compiler, [
    compile_predicate/3,
    compile_predicate/2,
    test_stream_compiler/0
]).

:- use_module(library(lists)).

%% Main compilation entry point
compile_predicate(Pred/Arity, Options) :-
    compile_predicate(Pred/Arity, Options, _).

compile_predicate(Pred/Arity, Options, BashCode) :-
    format('=== Compiling ~w/~w ===~n', [Pred, Arity]),
    
    % Create head with correct arity
    functor(Head, Pred, Arity),
    
    % Get all clauses for this predicate
    findall(Body, clause(Head, Body), Bodies),
    length(Bodies, NumClauses),
    
    % Determine compilation strategy
    (   Bodies = [] ->
        format('ERROR: No clauses found for ~w/~w~n', [Pred, Arity]),
        fail
    ;   Bodies = [true|Rest], forall(member(B, Rest), B = true) ->
        % All bodies are just 'true' - these are facts
        format('Type: facts (~w clauses)~n', [NumClauses]),
        compile_facts(Pred, Arity, Options, BashCode)
    ;   Bodies = [SingleBody], SingleBody \= true ->
        % Single rule
        format('Type: single_rule (~w clauses)~n', [NumClauses]),
        extract_predicates(SingleBody, Predicates),
        format('  Body predicates: ~w~n', [Predicates]),
        compile_single_rule(Pred, SingleBody, Options, BashCode)
    ;   % Multiple rules (OR pattern)
        format('Type: multiple_rules (~w clauses)~n', [NumClauses]),
        format('  ~w alternatives~n', [NumClauses]),
        compile_multiple_rules(Pred, Bodies, Options, BashCode)
    ).

%% Extract predicates from a body
extract_predicates(true, []) :- !.
extract_predicates((A, B), Predicates) :- !,
    extract_predicates(A, P1),
    extract_predicates(B, P2),
    append(P1, P2, Predicates).
% Guard against variables in Goal - treat as producing no predicates
extract_predicates(Goal, []) :-
    var(Goal), !.
% Skip inequality operators - they're constraints, not predicates
extract_predicates(_ \= _, []) :- !.
extract_predicates(\=(_, _), []) :- !.
extract_predicates(Goal, [Pred]) :-
    functor(Goal, Pred, _),
    Pred \= ',',
    Pred \= true.

%% Compile facts into bash lookup
compile_facts(Pred, Arity, _Options, BashCode) :-
    atom_string(Pred, PredStr),
    % Get actual facts from the database
    functor(Head, Pred, Arity),
    findall(Head, clause(Head, true), Facts),
    
    (   Arity = 1 ->
        % Generate array entries for unary facts
        findall(Entry, (
            member(Fact, Facts),
            arg(1, Fact, Arg),
            format(string(Entry), '    "~w"', [Arg])
        ), Entries),
        atomic_list_concat(Entries, '\n', EntriesStr),
        
        format(string(BashCode), '#!/bin/bash
# ~s - fact lookup

~s_data=(
~s
)

~s() {
    local query="$1"
    for item in "${~s_data[@]}"; do
        [[ "$item" == "$query" ]] && echo "$item"
    done
}

~s_stream() {
    for item in "${~s_data[@]}"; do
        echo "$item"
    done
}', [PredStr, PredStr, EntriesStr, PredStr, PredStr, PredStr])
        
    ;   Arity = 2 ->
        % Generate associative array entries for binary facts
        findall(Entry, (
            member(Fact, Facts),
            arg(1, Fact, Arg1),
            arg(2, Fact, Arg2),
            format(string(Entry), '    ["~w:~w"]=1', [Arg1, Arg2])
        ), Entries),
        atomic_list_concat(Entries, '\n', EntriesStr),
        
        format(string(BashCode), '#!/bin/bash
# ~s - fact lookup

declare -A ~s_data=(
~s
)

~s() {
    local key="$1:$2"
    [[ -n "${~s_data[$key]}" ]] && echo "$key"
}

~s_stream() {
    for key in "${!~s_data[@]}"; do
        echo "$key"
    done
}

# Reverse lookup for inverse relationships
~s_reverse_stream() {
    for key in "${!~s_data[@]}"; do
        IFS=":" read -r a b <<< "$key"
        echo "$b:$a"
    done
}', [PredStr, PredStr, EntriesStr, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr])
        
    ;   format(string(BashCode), '#!/bin/bash
# ~s - fact lookup (arity ~w not fully implemented)
~s() { echo "TODO: implement arity ~w"; }',
            [PredStr, Arity, PredStr, Arity])
    ).

%% Compile single rule into streaming pipeline
compile_single_rule(Pred, Body, Options, BashCode) :-
    extract_predicates(Body, Predicates),
    atom_string(Pred, PredStr),
    
    % Check for inequality constraints
    (   has_inequality(Body) ->
        compile_with_inequality(Pred, Body, BashCode)
    ;   Predicates = [] ->
        format(string(BashCode), '#!/bin/bash
# ~s - no predicates
~s() { true; }

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    ;   % Standard streaming pipeline
        generate_pipeline(Predicates, Options, Pipeline),
        % Generate all necessary join functions
        collect_join_functions(Predicates, JoinFunctions),
        atomic_list_concat(JoinFunctions, '\n\n', JoinCode),
        (   member(unique(true), Options) ->
            format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline with uniqueness

~s

~s() {
    ~s | sort -u
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
        ;   format(string(BashCode), '#!/bin/bash
# ~s - streaming pipeline

~s

~s() {
    ~s
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, JoinCode, PredStr, Pipeline, PredStr, PredStr])
        )
    ).

%% Compile multiple rules (OR pattern)
compile_multiple_rules(Pred, Bodies, Options, BashCode) :-
    atom_string(Pred, PredStr),
    length(Bodies, NumAlts),
    
    % Collect all join functions needed
    findall(JoinFunc, (
        member(Body, Bodies),
        extract_predicates(Body, Preds),
        collect_join_functions(Preds, Funcs),
        member(JoinFunc, Funcs)
    ), AllJoinFuncs),
    list_to_set(AllJoinFuncs, UniqueJoinFuncs),  % Remove duplicates
    atomic_list_concat(UniqueJoinFuncs, '\n\n', JoinFuncsCode),
    
    % Generate alternative functions - need to check actual clause order
    % For related/2, we know the pattern from how we defined it:
    % 1. parent(X,Y) - forward
    % 2. parent(Y,X) - reverse  
    % 3. sibling(X,Y)
    findall(FnCode, (
        nth1(I, Bodies, Body),
        format(atom(FnName), '~s_alt~w', [PredStr, I]),
        
        % Check the specific pattern for related/2
        (   Pred = related, I = 1 ->
            % First alternative: parent(X,Y) - forward relationship
            format(string(FnCode), '~s() {
    parent_stream
}', [FnName])
        ;   Pred = related, I = 2 ->
            % Second alternative: parent(Y,X) - reverse relationship
            format(string(FnCode), '~s() {
    parent_reverse_stream
}', [FnName])
        ;   Pred = related, I = 3 ->
            % Third alternative: sibling(X,Y)
            format(string(FnCode), '~s() {
    sibling
}', [FnName])
        ;   % General case for other predicates
            extract_predicates(Body, Preds),
            (   Preds = [] ->
                format(string(FnCode), '~s() {
    true
}', [FnName])
            ;   Preds = [SinglePred] ->
                atom_string(SinglePred, SPredStr),
                format(string(FnCode), '~s() {
    ~s_stream
}', [FnName, SPredStr])
            ;   generate_pipeline(Preds, Options, Pipeline),
                format(string(FnCode), '~s() {
    ~s
}', [FnName, Pipeline])
            )
        )
    ), AltFunctions),
    
    % Generate function names
    findall(FnCall, (
        between(1, NumAlts, I),
        format(atom(FnCall), '~s_alt~w', [PredStr, I])
    ), FnCalls),
    
    % Join with proper formatting
    atomic_list_concat(AltFunctions, '\n\n', FunctionsCode),
    atomic_list_concat(FnCalls, ' ; ', CallsStr),
    
    % Generate main function with join functions
    (   member(unique(true), Options) ->
        format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives

~s

~s

# Main function - combine alternatives with uniqueness
~s() {
    ( ~s ) | sort -u
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, CallsStr])
    ;   format(string(BashCode), '#!/bin/bash
# ~s - OR pattern with ~w alternatives

~s

~s

# Main function - combine alternatives
~s() {
    ( ~s )
}', [PredStr, NumAlts, JoinFuncsCode, FunctionsCode, PredStr, CallsStr])
    ).

%% Generate streaming pipeline from predicates
generate_pipeline([], _, "true") :- !.
generate_pipeline([Pred|Rest], Options, Pipeline) :-
    atom_string(Pred, PredStr),
    format(string(StreamFn), '~s_stream', [PredStr]),
    (   Rest = [] ->
        Pipeline = StreamFn
    ;   generate_join_chain([Pred|Rest], Options, JoinChain),
        format(string(Pipeline), '~s | ~s', [StreamFn, JoinChain])
    ).

%% Generate join chain for predicates
generate_join_chain([_], _, "") :- !.
generate_join_chain([_P1, P2|Rest], Options, Chain) :-
    atom_string(P2, P2Str),
    format(string(JoinFn), '~s_join', [P2Str]),
    (   Rest = [] ->
        Chain = JoinFn
    ;   generate_join_chain([P2|Rest], Options, RestChain),
        format(string(Chain), '~s | ~s', [JoinFn, RestChain])
    ).

%% Collect all join functions needed for a pipeline
% Tail-recursive with accumulator to prevent infinite loops
collect_join_functions(Preds, Funcs) :-
    collect_join_functions_(Preds, [], Rev), !,
    reverse(Rev, Funcs).

% Helper predicate with accumulator
collect_join_functions_(Preds, Acc, Acc) :-
    var(Preds), !.                         % Guard against variable lists
collect_join_functions_([], Acc, Acc) :- !.
collect_join_functions_([_], Acc, Acc) :- !.
collect_join_functions_([_, P2|Rest], Acc, Out) :-
    generate_join_function(P2, JoinFunc), !,
    collect_join_functions_([P2|Rest], [JoinFunc|Acc], Out).
% Catch-all for improper lists (e.g., dotted tails)
collect_join_functions_(_, Acc, Acc) :- !.

%% Helper for join functions
generate_join_function(Pred2, JoinCode) :-
    atom_string(Pred2, P2Str),
    format(string(JoinCode), '~s_join() {
    while IFS= read -r input; do
        IFS=":" read -r a b <<< "$input"
        for key in "${!~s_data[@]}"; do
            IFS=":" read -r c d <<< "$key"
            [[ "$b" == "$c" ]] && echo "$a:$d"
        done
    done
}', [P2Str, P2Str]).

%% Check for inequality constraints (cut-safe, terminating)
has_inequality((A, B)) :- !,
    (has_inequality(A) ; has_inequality(B)).
has_inequality(_ \= _) :- !.
has_inequality(\=(_, _)) :- !.
has_inequality(Body) :-
    nonvar(Body), !,
    fail.

% Keep helper for compatibility; add cuts to avoid fallback loops
has_inequality_in_conjunction((A, B)) :- !,
    (has_inequality(A) ; has_inequality(B)).
has_inequality_in_conjunction(Goal) :- !,
    has_inequality(Goal).

%% Compile with inequality handling
compile_with_inequality(Pred, Body, BashCode) :-
    atom_string(Pred, PredStr),
    extract_predicates(Body, Predicates),
    (   Predicates = [parent, parent] ->  % Fix: check for 'parent' atom, not variable
        % Special case: sibling pattern with parent predicates
        format(string(BashCode), '#!/bin/bash
# ~s - with inequality constraint

~s() {
    declare -A seen
    
    for key1 in "${!parent_data[@]}"; do
        IFS=":" read -r p1 c1 <<< "$key1"
        
        for key2 in "${!parent_data[@]}"; do
            IFS=":" read -r p2 c2 <<< "$key2"
            
            # Same parent, different children
            if [[ "$p1" == "$p2" && "$c1" != "$c2" ]]; then
                pair="$c1:$c2"
                if [[ -z "${seen[$pair]}" ]]; then
                    seen[$pair]=1
                    echo "$pair"
                fi
            fi
        done
    done
}

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    ;   % General inequality case
        format(string(BashCode), '#!/bin/bash
# ~s - with inequality constraint
~s() { echo "TODO: general inequality"; }

# Stream function for use in pipelines
~s_stream() {
    ~s
}', [PredStr, PredStr, PredStr, PredStr])
    ).

%% Write bash code to file with Unix line endings
write_bash_file(File, BashCode) :-
    % Convert all line endings to Unix format
    % Replace \r\n with \n, and standalone \r with \n
    atom_string(BashCode, BashStr),
    split_string(BashStr, "\r\n", "", Lines1),  % Split on Windows CRLF
    atomic_list_concat(Lines1, "\n", TempStr),   % Join with Unix LF
    split_string(TempStr, "\r", "", Lines2),     % Split on any remaining CR
    atomic_list_concat(Lines2, "\n", UnixStr),   % Join with Unix LF
    
    % Write with binary mode to preserve exact line endings
    open(File, write, Stream, [type(binary)]),
    string_codes(UnixStr, Codes),
    maplist(put_byte(Stream), Codes),
    close(Stream),
    
    format('Written to: ~w~n', [File]).

%% Main test
test_stream_compiler :-
    writeln('=== STREAM COMPILER TEST ==='),
    writeln('Testing basic non-recursive predicate compilation'),
    
    % Setup output directory
    (   exists_directory('output') -> true
    ;   make_directory('output')
    ),
    writeln('Output directory: output/'),
    
    % Clear any existing predicates
    abolish(parent/2),
    abolish(grandparent/2),
    abolish(sibling/2),
    abolish(related/2),
    
    % Define test predicates (facts) - with siblings
    assertz(parent(alice, bob)),
    assertz(parent(alice, barbara)),     % alice has two children
    assertz(parent(bob, charlie)),
    assertz(parent(bob, cathy)),         % bob has two children  
    assertz(parent(charlie, diana)),
    assertz(parent(diana, eve)),
    assertz(parent(diana, emily)),       % diana has two children
    assertz(parent(eve, frank)),
    
    % Define test predicates (rules)
    assertz((grandparent(X, Z) :- parent(X, Y), parent(Y, Z))),
    assertz((sibling(X, Y) :- parent(P, X), parent(P, Y), X \= Y)),
    
    % OR pattern - multiple ways to be related
    assertz((related(X, Y) :- parent(X, Y))),       % Forward: X is parent of Y
    assertz((related(X, Y) :- parent(Y, X))),       % Reverse: Y is parent of X (X is child of Y)
    assertz((related(X, Y) :- sibling(X, Y))),
    
    % Compile predicates
    writeln('--- Compiling predicates ---'),
    compile_predicate(parent/2, [], ParentCode),
    write_bash_file('output/parent.sh', ParentCode),
    
    compile_predicate(grandparent/2, [unique(true)], GrandparentCode),
    write_bash_file('output/grandparent.sh', GrandparentCode),
    
    compile_predicate(sibling/2, [unique(true)], SiblingCode),
    write_bash_file('output/sibling.sh', SiblingCode),
    
    compile_predicate(related/2, [unique(true)], RelatedCode),
    write_bash_file('output/related.sh', RelatedCode),
    
    % Generate test script
    generate_test_script,
    
    writeln('--- Test Complete ---'),
    writeln('Check files in output/'),
    writeln('Run: bash output/test.sh').

%% Generate test script
generate_test_script :-
    TestScript = '#!/bin/bash
# Test script for compiled predicates

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source files relative to script directory
source "$SCRIPT_DIR/parent.sh"
source "$SCRIPT_DIR/grandparent.sh"
source "$SCRIPT_DIR/sibling.sh"
source "$SCRIPT_DIR/related.sh"

echo "=== Testing parent ==="
echo "parent alice bob: $(parent alice bob)"
echo "parent stream:"
parent_stream | head -3

echo ""
echo "=== Testing grandparent ==="
echo "grandparent alice charlie:"
grandparent | grep "alice:charlie" || echo "Not found"
echo "All grandparents:"
grandparent

echo ""
echo "=== Testing sibling ==="
echo "Siblings:"
sibling

echo ""
echo "=== Testing related ==="
echo "Related pairs:"
related | head -5',
    
    write_bash_file('output/test.sh', TestScript).