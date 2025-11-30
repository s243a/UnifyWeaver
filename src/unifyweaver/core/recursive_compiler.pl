:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% recursive_compiler.pl - UnifyWeave compiler with recursion support
% Extends stream_compiler to handle recursive predicates
% Uses robust templating system for code generation

:- module(recursive_compiler, [
    compile_recursive/3,
    compile_recursive/2,
    test_recursive_compiler/0
]).

:- use_module('advanced/advanced_recursive_compiler').
:- use_module('advanced/call_graph').
:- use_module(stream_compiler).
:- use_module('../targets/awk_target').
:- use_module('../targets/csharp_target', []).
:- use_module('../targets/csharp_stream_target', []).
:- use_module(template_system).
:- use_module(library(lists)).
:- use_module(constraint_analyzer).
:- use_module(firewall).
:- use_module(preferences).

%% Main entry point - analyze and compile
compile_recursive(Pred/Arity, Options) :-
    compile_recursive(Pred/Arity, Options, _).

compile_recursive(Pred/Arity, RuntimeOptions, GeneratedCode) :-
    % 1. Get constraints and merge with runtime options
    get_constraints(Pred/Arity, DeclaredConstraints),
    preferences:get_final_options(Pred/Arity, RuntimeOptions, FinalOptions),
    merge_options(FinalOptions, DeclaredConstraints, MergedOptions),

    % 2. Get the firewall policy for the predicate.
    firewall:get_firewall_policy(Pred/Arity, Firewall),

    % 3. Determine target backend (default: bash)
    (   member(target(Target), MergedOptions) ->
        true
    ;   Target = bash  % Default target
    ),

    % 4. Validate the request against the firewall.
    (   firewall:enforce_firewall(Pred/Arity, Target, MergedOptions, Firewall)
    ->  % Validation passed, proceed with compilation.
        format('--- Firewall validation passed for ~w. Proceeding with compilation. ---\n', [Pred/Arity]),
        compile_dispatch(Pred/Arity, MergedOptions, Target, GeneratedCode)
    ;   % Validation failed, stop.
        format(user_error, 'Compilation of ~w halted due to firewall policy violation.~n', [Pred/Arity]),
        !, fail
    ).

%% merge_options(RuntimeOpts, Constraints, Merged)
%  Merges runtime options with declared constraints.
merge_options(RuntimeOpts, Constraints, Merged) :-
    (   member(unique(U), RuntimeOpts) -> MergedUnique = [unique(U)] ; MergedUnique = [] ),
    (   member(unordered(O), RuntimeOpts) -> MergedUnordered = [unordered(O)] ; MergedUnordered = [] ),
    append(MergedUnique, MergedUnordered, RuntimeConstraints),
    append(RuntimeConstraints, Constraints, AllConstraints0),
    findall(Opt,
        (   member(Opt, RuntimeOpts),
            functor(Opt, FunctorName, _),
            FunctorName \= unique,
            FunctorName \= unordered
        ),
        OtherRuntimeOpts),
    append(AllConstraints0, OtherRuntimeOpts, AllConstraints),
    list_to_set(AllConstraints, Merged).

%% compile_dispatch(+Pred/Arity, +FinalOptions, +Target, -GeneratedCode)
%  Target-aware compilation logic, now called after validation.
compile_dispatch(Pred/Arity, FinalOptions, Target, GeneratedCode) :-
    format('=== Analyzing ~w/~w ===~n', [Pred, Arity]),
    classify_predicate(Pred/Arity, Classification),
    format('Classification: ~w~n', [Classification]),

    (   Classification = non_recursive ->
        compile_non_recursive(Target, Pred/Arity, FinalOptions, GeneratedCode)
    ;   Target == csharp ->
        compile_recursive_csharp_query(Pred/Arity, FinalOptions, GeneratedCode)
    ;   Classification = transitive_closure(BasePred) ->
        format('Detected transitive closure over ~w~n', [BasePred]),
        compile_transitive_closure(Target, Pred, Arity, BasePred, FinalOptions, GeneratedCode)
    ;   Classification = mutual_recursion -> % Handle mutual recursion
        advanced_recursive_compiler:compile_mutual_recursion(Pred/Arity, FinalOptions, GeneratedCode)
    ;   catch(
            compile_advanced(Target, Pred/Arity, FinalOptions, GeneratedCode),
            error(existence_error(procedure, _), _),
            fail
        ) ->
        format('Compiled using advanced patterns with options: ~w~n', [FinalOptions])
    ;   format('Unknown recursion pattern - using memoization~n', []),
        compile_memoized_recursion(Target, Pred, Arity, FinalOptions, GeneratedCode)
    ).

compile_non_recursive(bash, Pred/Arity, FinalOptions, GeneratedCode) :-
    stream_compiler:compile_predicate(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(awk, Pred/Arity, FinalOptions, GeneratedCode) :-
    awk_target:compile_predicate_to_awk(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(csharp, Pred/Arity, FinalOptions, GeneratedCode) :-
    csharp_stream_target:compile_predicate_to_csharp(Pred/Arity, FinalOptions, GeneratedCode).
compile_non_recursive(Target, Pred/Arity, _Options, _GeneratedCode) :-
    format(user_error, 'Target ~w not supported for non-recursive predicate ~w.~n', [Target, Pred/Arity]),
    fail.

compile_advanced(bash, Pred/Arity, FinalOptions, GeneratedCode) :-
    advanced_recursive_compiler:compile_advanced_recursive(
        Pred/Arity, FinalOptions, GeneratedCode
    ).
compile_advanced(Target, Pred/Arity, _FinalOptions, _GeneratedCode) :-
    format(user_error, 'Advanced recursive compilation for target ~w not implemented (~w).~n',
           [Target, Pred/Arity]),
    fail.

compile_recursive_csharp_query(Pred/Arity, Options, Code) :-
    prepare_csharp_query_options(Options, QueryOptions),
    (   csharp_target:build_query_plan(Pred/Arity, QueryOptions, Plan)
    ->  csharp_target:render_plan_to_csharp(Plan, Code)
    ;   format(user_error, 'C# query target failed to build plan for ~w.~n', [Pred/Arity]),
        fail
    ).

prepare_csharp_query_options(Options, QueryOptions) :-
    exclude(is_target_option, Options, Rest),
    QueryOptions = [target(csharp_query)|Rest].

is_target_option(target(_)).

%% Classify predicate recursion pattern
classify_predicate(Pred/Arity, Classification) :-
    functor(Head, Pred, Arity),
    findall(Body, clause(Head, Body), Bodies),

    % Check for mutual recursion FIRST (before self-recursion check)
    (   call_graph:predicates_in_group(Pred/Arity, Group),
        length(Group, GroupSize),
        GroupSize > 1 ->
        Classification = mutual_recursion
    ;   % Check if self-recursive
        contains_recursive_call(Pred, Bodies) ->
        analyze_recursion_pattern(Pred, Arity, Bodies, Classification)
    ;   Classification = non_recursive
    ).

%% Check if any body contains a recursive call
contains_recursive_call(Pred, Bodies) :-
    member(Body, Bodies),
    contains_goal(Body, Goal),
    functor(Goal, Pred, _).

%% Check if a goal appears in a body
contains_goal(Goal, Goal) :- 
    compound(Goal),
    \+ Goal = (_,_).
contains_goal((A, _), Goal) :- 
    contains_goal(A, Goal).
contains_goal((_, B), Goal) :- 
    contains_goal(B, Goal).
contains_goal((A; _), Goal) :- 
    contains_goal(A, Goal).
contains_goal((_;B), Goal) :- 
    contains_goal(B, Goal).

%% Analyze recursion pattern
analyze_recursion_pattern(Pred, Arity, Bodies, Pattern) :-
    % Separate base cases from recursive cases
    partition(is_recursive_clause(Pred), Bodies, RecClauses, BaseClauses),
    
    % Check for transitive closure pattern
    (   is_transitive_closure(Pred, Arity, BaseClauses, RecClauses, BasePred) ->
        Pattern = transitive_closure(BasePred)
    ;   is_tail_recursive(Pred, RecClauses) ->
        Pattern = tail_recursion
    ;   is_linear_recursive(Pred, RecClauses) ->
        Pattern = linear_recursion
    ;   Pattern = unknown_recursion
    ).

is_recursive_clause(Pred, Body) :-
    contains_goal(Body, Goal),
    functor(Goal, Pred, _).

%% Check for transitive closure pattern
% Two patterns supported:
% 1. Forward: pred(X,Z) :- base(X,Y), pred(Y,Z).  [e.g., ancestor]
% 2. Reverse: pred(X,Z) :- base(Y,X), pred(Y,Z).  [e.g., descendant]
is_transitive_closure(Pred, 2, BaseClauses, RecClauses, BasePred) :-
    % Check base case is a single predicate call
    member(BaseBody, BaseClauses),
    BaseBody \= true,
    functor(BaseBody, BasePred, 2),
    BasePred \= Pred,

    % Check recursive case matches pattern
    member(RecBody, RecClauses),
    RecBody = (BaseCall, RecCall),
    functor(BaseCall, BasePred, 2),
    functor(RecCall, Pred, 2),

    % Try both forward and reverse patterns
    (   % Pattern 1: Forward transitive closure
        % base(X,Y), recursive(Y,Z) - Y flows from base to recursive
        BaseCall =.. [BasePred, _X, Y],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ;   % Pattern 2: Reverse transitive closure
        % base(Y,X), recursive(Y,Z) - Y flows from base to recursive (reversed args)
        BaseCall =.. [BasePred, Y, _X],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ).

is_transitive_closure(_, _, _, _, _) :- fail.

%% Check for tail recursion
is_tail_recursive(Pred, RecClauses) :-
    member(Body, RecClauses),
    last_goal(Body, Goal),
    functor(Goal, Pred, _).

last_goal(Goal, Goal) :- 
    compound(Goal),
    \+ Goal = (_,_).
last_goal((_, B), Goal) :- 
    last_goal(B, Goal).

%% Check for linear recursion
is_linear_recursive(Pred, RecClauses) :-
    member(Body, RecClauses),
    findall(G, contains_goal(Body, G), Goals),
    findall(G, (member(G, Goals), functor(G, Pred, _)), RecGoals),
    length(RecGoals, 1).  % Exactly one recursive call

%% Compile transitive closure pattern
compile_transitive_closure(bash, Pred, _Arity, BasePred, Options, GeneratedCode) :-
    % Use the template_system's new constraint-aware predicate
    template_system:generate_transitive_closure(Pred, BasePred, Options, GeneratedCode),
    !.
compile_transitive_closure(Target, Pred, Arity, BasePred, _Options, _GeneratedCode) :-
    format(user_error,
           'Transitive closure for target ~w not yet supported (~w/~w via ~w).~n',
           [Target, Pred, Arity, BasePred]),
    fail.

%% Compile with memoization for unknown patterns
compile_memoized_recursion(bash, Pred, Arity, Options, GeneratedCode) :-
    compile_plain_recursion(Pred, Arity, Options, GeneratedCode).
compile_memoized_recursion(Target, Pred, Arity, _Options, _GeneratedCode) :-
    format(user_error,
           'Memoized recursion not implemented for target ~w (~w/~w).~n',
           [Target, Pred, Arity]),
    fail.

%% Compile plain recursion as final fallback
compile_plain_recursion(Pred, Arity, Options, GeneratedCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, Arity),  % Currently assumes arity 2 semantics
    
    % Get all clauses
    findall(Body, clause(Head, Body), Bodies),
    
    % Separate base and recursive cases
    partition(is_recursive_clause(Pred), Bodies, RecClauses, BaseClauses),
    
    % Generate base cases
    findall(BaseCode, (
        member(Base, BaseClauses),
        generate_base_case(Pred, Base, BaseCode)
    ), BaseCodes),
    atomic_list_concat(BaseCodes, '\n    ', BaseCodeStr),
    
    % Generate recursive cases  
    findall(RecCode, (
        member(Rec, RecClauses),
        generate_recursive_case(Pred, Rec, RecCode)
    ), RecCodes),
    atomic_list_concat(RecCodes, '\n    ', RecCodeStr),
    
    generate_plain_recursion_template(PredStr, BaseCodeStr, RecCodeStr, Options, GeneratedCode).

%% Generate base case code
generate_base_case(Pred, Body, Code) :-
    atom_string(Pred, PredStr),
    (   Body = true ->
        format(string(Code), '# Base: ~s is a fact
    [[ -n "${~s_data[$key]}" ]] && {
        ~s_memo["$key"]="$key"
        echo "$key"
        return
    }', [PredStr, PredStr, PredStr])
    ;   ( functor(Body, BasePred, 2), atom(BasePred), is_alpha(BasePred) ) ->
        atom_string(BasePred, BaseStr),
        format(string(Code), '# Base: check ~s
    if ~s "$arg1" "$arg2" >/dev/null 2>&1; then
        ~s_memo["$key"]="$key"
        echo "$key"
        return
    fi', [BaseStr, BaseStr, PredStr])
    ;   format(string(Code), '# Complex base case - not implemented', [])
    ).

is_alpha(Atom) :-
    atom_codes(Atom, Codes),
    forall(member(Code, Codes), code_type(Code, alpha)).

%% Generate recursive case code
generate_recursive_case(Pred, Body, Code) :-
    atom_string(Pred, PS),
    % Simplified - just note it's recursive
    format(string(Code), '# Recursive case
    # Body: ~w
    # Would need to decompose and call ~s recursively
    # Implementation depends on specific pattern', [Body, PS]).

%% String replacement helper
string_replace(Input, Find, Replace, Output) :-
    split_string(Input, Find, "", Parts),
    atomic_list_concat(Parts, Replace, Output).

%% Test the recursive compiler
test_recursive_compiler :-
    writeln('=== RECURSIVE COMPILER TEST ==='),
    
    % Setup output directory
    (   exists_directory('output') -> true
    ;   make_directory('output')
    ),
    
    % First ensure base predicates exist
    stream_compiler:test_stream_compiler,  % This sets up parent/2, grandparent/2, etc.
    
    writeln(''),
    writeln('--- Testing Recursive Predicates ---'),
    
    % Clear any existing recursive predicates
    abolish(ancestor/2),
    abolish(descendant/2),
    abolish(reachable/2),
    
    % Define ancestor as transitive closure of parent
    assertz((ancestor(X, Y) :- parent(X, Y))),
    assertz((ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z))),
    
    % Define descendant (reverse of ancestor)
    assertz((descendant(X, Y) :- parent(Y, X))),
    assertz((descendant(X, Z) :- parent(Y, X), descendant(Y, Z))),
    
    % Define reachable (follows related)
    assertz((reachable(X, Y) :- related(X, Y))),
    assertz((reachable(X, Z) :- related(X, Y), reachable(Y, Z))),
    
    % Compile recursive predicates
    compile_recursive(ancestor/2, [], AncestorCode),
    stream_compiler:write_bash_file('output/ancestor.sh', AncestorCode),
    
    compile_recursive(descendant/2, [], DescendantCode),
    stream_compiler:write_bash_file('output/descendant.sh', DescendantCode),
    
    compile_recursive(reachable/2, [], ReachableCode),
    stream_compiler:write_bash_file('output/reachable.sh', ReachableCode),
    
    % Generate extended test script
    generate_recursive_test_script,
    
    writeln('--- Recursive Compilation Complete ---'),
    writeln('Check files in output/'),
    writeln('Run: bash output/test_recursive.sh').

%% Generate test script for recursive predicates
generate_recursive_test_script :-
    TestScript = '#!/bin/bash
# Test script for recursive predicates

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source all files - ORDER MATTERS: dependencies first
source "$SCRIPT_DIR/parent.sh"
source "$SCRIPT_DIR/sibling.sh"      # Must be before related.sh
source "$SCRIPT_DIR/grandparent.sh"  # Also useful to have available
source "$SCRIPT_DIR/ancestor.sh"
source "$SCRIPT_DIR/descendant.sh"
source "$SCRIPT_DIR/related.sh"      # Depends on sibling
source "$SCRIPT_DIR/reachable.sh"

echo "=== Testing ancestor (transitive closure) ==="
echo "All ancestors of charlie:"
ancestor_all charlie

echo ""
echo "Is alice an ancestor of eve?"
ancestor_check alice eve && echo "Yes" || echo "No"

echo ""
echo "Is bob an ancestor of frank?"
ancestor_check bob frank && echo "Yes" || echo "No"

echo ""
echo "=== Testing descendant ==="
echo "All descendants of alice:"
descendant_all alice

echo ""
echo "All descendants of diana (should show eve, emily, frank):"
descendant_all diana

echo ""
echo "All descendants of charlie (should show diana, eve, emily, frank):"
descendant_all charlie

echo ""
echo "=== Testing reachable ==="
echo "All nodes reachable from alice (first 10):"
reachable_all alice | head -10

echo ""
echo "All nodes reachable from eve (first 10):"
reachable_all eve | head -10
',
    stream_compiler:write_bash_file('output/test_recursive.sh', TestScript).

%% ============================================
%% LOCAL TEMPLATE FUNCTIONS
%% ============================================
%% These are specialized for recursive compilation
%% They use template_system:render_template/3 for rendering

%% Generate descendant template
generate_descendant_template(PredStr, BashCode) :-
    Template = '#!/bin/bash
# {{pred}} - finds descendants (children, grandchildren, etc.)

# Iterative BFS implementation
{{pred}}() {
    local start="$1"
    local target="$2"
    
    if [[ -z "$target" ]]; then
        # Mode: {{pred}}(+,-)  Find all descendants
        {{pred}}_all "$start"
    else
        # Mode: {{pred}}(+,+)  Check if descendant
        {{pred}}_check "$start" "$target"
    fi
}

# Find all descendants of start
{{pred}}_all() {
    local start="$1"
    declare -A visited
    declare -A output_seen
    
    # Use work queue for BFS
    local queue_file="/tmp/{{pred}}_queue_$"
    local next_queue="/tmp/{{pred}}_next_$"
    
    trap "rm -f $queue_file $next_queue" EXIT
    
    echo "$start" > "$queue_file"
    visited["$start"]=1
    
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        
        while IFS= read -r current; do
            # Find all children of current (forward direction)
            parent_stream | grep "^$current:" | while IFS=":" read -r from to; do
                if [[ "$from" == "$current" && -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    
                    # Output the descendant relationship
                    local output_key="$start:$to"
                    if [[ -z "${output_seen[$output_key]}" ]]; then
                        output_seen["$output_key"]=1
                        echo "$output_key"
                    fi
                fi
            done
        done < "$queue_file"
        
        mv "$next_queue" "$queue_file"
    done
    
    rm -f "$queue_file" "$next_queue"
}

# Check if target is descendant of start
{{pred}}_check() {
    local start="$1"
    local target="$2"
    local tmpflag="/tmp/{{pred}}_found_$$"
    local timeout_duration="5s"
    
    # Timeout prevents infinite execution, tee prevents SIGPIPE
    timeout "$timeout_duration" {{pred}}_all "$start" | 
    tee >(grep -q "^$start:$target$" && touch "$tmpflag") >/dev/null
    
    if [[ -f "$tmpflag" ]]; then
        echo "$start:$target"
        rm -f "$tmpflag"
        return 0
    else
        rm -f "$tmpflag"
        return 1
    fi
}

# Stream function
{{pred}}_stream() {
    {{pred}}_all "$1"
}',
    template_system:render_template(Template, [pred=PredStr], BashCode).

%% Generate plain recursion template
generate_plain_recursion_template(PredStr, BaseCodeStr, RecCodeStr, Options, BashCode) :-
    constraint_analyzer:get_dedup_strategy(Options, Strategy),
    (   Strategy == no_dedup ->
        MemoSetup = "",
        MemoCheck = ""
    ;   MemoSetup = "declare -gA {{pred}}_memo",
        MemoCheck = 'if [[ -n "${{{pred}}_memo[$key]}" ]]; then echo "${{{pred}}_memo[$key]}"; return; fi'
    ),
    Template = '# Memoization table
{{memo_setup}}

# Main recursive function
{{pred}}_all() {
    local arg1="$1"
    local arg2="$2"
    local key="$arg1:$arg2"
    
    # Check memoization
    {{memo_check}}
    
    # Try base cases first
    {{base_cases}}
    
    # Try recursive cases
    {{rec_cases}}
    
    # Cache miss and no match
    return 1
}

# Wrapper with named pipe support for complex queries
{{pred}}_stream() {
    local pipe="/tmp/{{pred}}_pipe_$$"
    mkfifo "$pipe" 2>/dev/null || true
    
    # Start recursive computation in background
    {{pred}}_all "$@" > "$pipe" &
    
    # Read results from pipe
    cat "$pipe"
    rm -f "$pipe"
}',
    render_template(Template, [
        pred=PredStr,
        base_cases=BaseCodeStr,
        rec_cases=RecCodeStr,
        memo_setup=MemoSetup,
        memo_check=MemoCheck
    ], MainCode),
    render_named_template('dedup_wrapper', [
        pred = PredStr,
        strategy = Strategy,
        main_code = MainCode
    ], BashCode).
