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

:- use_module(library('advanced/advanced_recursive_compiler')).
:- use_module(library(stream_compiler)).
:- use_module(library(template_system)).
:- use_module(library(lists)).
:- use_module(library(firewall)).
:- use_module(library(preferences)).

%% Main entry point - analyze and compile
compile_recursive(Pred/Arity, Options) :-
    compile_recursive(Pred/Arity, Options, _).

compile_recursive(Pred/Arity, RuntimeOptions, BashCode) :-
    % 1. Get the final, merged options from all configuration layers.
    preferences:get_final_options(Pred/Arity, RuntimeOptions, FinalOptions),

    % 2. Get the firewall policy for the predicate.
    (   firewall:rule_firewall(Pred/Arity, Firewall)
    ->  true
    ;   firewall:firewall_default(Firewall)
    ->  true
    ;   Firewall = [] % Default to implicit allow if no rules are found
    ),

    % 3. Validate the request against the firewall.
    % The compilation target is hardcoded to 'bash' for now.
    (   firewall:validate_against_firewall(bash, FinalOptions, Firewall)
    ->  % Validation passed, proceed with compilation.
        format('--- Firewall validation passed for ~w. Proceeding with compilation. ---\n', [Pred/Arity]),
        compile_dispatch(Pred/Arity, FinalOptions, BashCode)
    ;   % Validation failed, stop.
        format(user_error, 'Compilation of ~w halted due to firewall policy violation.~n', [Pred/Arity]),
        !, fail
    ).

%% compile_dispatch(+Pred/Arity, +FinalOptions, -BashCode)
%  The original compilation logic, now called after validation.
compile_dispatch(Pred/Arity, FinalOptions, BashCode) :-
    format('=== Analyzing ~w/~w ===~n', [Pred, Arity]),
    classify_predicate(Pred/Arity, Classification),
    format('Classification: ~w~n', [Classification]),

    (   Classification = non_recursive ->
        % Delegate to stream_compiler
        stream_compiler:compile_predicate(Pred/Arity, FinalOptions, BashCode)
    ;   Classification = transitive_closure(BasePred) ->
        format('Detected transitive closure over ~w~n', [BasePred]),
        compile_transitive_closure(Pred, Arity, BasePred, FinalOptions, BashCode)
    ;   % Try advanced patterns before falling back to memoization
        catch(
            advanced_recursive_compiler:compile_advanced_recursive(
                Pred/Arity, FinalOptions, BashCode
            ),
            error(existence_error(procedure, _), _),
            fail
        ) ->
        format('Compiled using advanced patterns with options: ~w~n', [FinalOptions])
    ;   % Unknown pattern - fall back to memoized recursion
        format('Unknown recursion pattern - using memoization~n', []),
        compile_memoized_recursion(Pred, Arity, FinalOptions, BashCode)
    ).

%% Classify predicate recursion pattern
classify_predicate(Pred/Arity, Classification) :-
    functor(Head, Pred, Arity),
    findall(Body, clause(Head, Body), Bodies),
    
    % Check if recursive
    (   contains_recursive_call(Pred, Bodies) ->
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
% Base: pred(X,Y) :- base_pred(X,Y).
% Recursive: pred(X,Z) :- base_pred(X,Y), pred(Y,Z).
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
    
    % Verify argument flow: base(X,Y), recursive(Y,Z)
    BaseCall =.. [BasePred, _X, Y],
    RecCall =.. [Pred, Y2, _],
    Y == Y2.

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
compile_transitive_closure(Pred, _Arity, BasePred, _Options, BashCode) :-
    % Use the template_system's existing generate_transitive_closure
    template_system:generate_transitive_closure(Pred, BasePred, BashCode),
    !.



%% Compile with memoization for unknown patterns
compile_memoized_recursion(Pred, _Arity, _Options, BashCode) :-
    compile_plain_recursion(Pred, BashCode).

%% Compile plain recursion as final fallback
compile_plain_recursion(Pred, BashCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),  % Assuming arity 2 for now
    
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
    
    generate_plain_recursion_template(PredStr, BaseCodeStr, RecCodeStr, BashCode).

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
    ;   functor(Body, BasePred, 2) ->
        atom_string(BasePred, BaseStr),
        format(string(Code), '# Base: check ~s
    if ~s "$arg1" "$arg2" >/dev/null 2>&1; then
        ~s_memo["$key"]="$key"
        echo "$key"
        return
    fi', [BaseStr, BaseStr, PredStr])
    ;   format(string(Code), '# Complex base case - not implemented', [])
    ).

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
    {{pred}}_all "$start" | grep -q "^$start:$target$" && echo "$start:$target"
}

# Stream function
{{pred}}_stream() {
    {{pred}}_all "$1"
}',
    template_system:render_template(Template, [pred=PredStr], BashCode).

%% Generate plain recursion template
generate_plain_recursion_template(PredStr, BaseCodeStr, RecCodeStr, BashCode) :-
    Template = '#!/bin/bash
# {{pred}} - plain recursive implementation

# Memoization table
declare -gA {{pred}}_memo

# Main recursive function
{{pred}}() {
    local arg1="$1"
    local arg2="$2"
    local key="$arg1:$arg2"
    
    # Check memoization
    if [[ -n "${{{pred}}_memo[$key]}" ]]; then
        echo "${{{pred}}_memo[$key]}"
        return
    fi
    
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
    {{pred}} "$@" > "$pipe" &
    
    # Read results from pipe
    cat "$pipe"
    rm -f "$pipe"
}

# Find all solutions
{{pred}}_all() {
    local start="$1"
    # Generate all possible pairs starting with $start
    # This is a simplified implementation
    {{pred}} "$start" ""
}',
    template_system:render_template(Template, [
        pred=PredStr,
        base_cases=BaseCodeStr,
        rec_cases=RecCodeStr
    ], BashCode).