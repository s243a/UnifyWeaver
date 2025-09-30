:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% template_system.pl - Robust templating system for UnifyWeaver
% Provides named placeholder substitution and composable template units

:- module(template_system, [
    render_template/3,
    compose_templates/3,
    template/2,
    generate_transitive_closure/3,
    test_template_system/0
]).

:- use_module(library(lists)).

%% Named placeholder substitution
% Replaces {{name}} with corresponding value from dictionary
render_template(Template, Dict, Result) :-
    atom_string(Template, TStr),
    render_template_string(TStr, Dict, Result).

% Fixed version using atom_string and sub_atom for reliable replacement
render_template_string(Template, [], Template) :- !.
render_template_string(Template, [Key=Value|Rest], Result) :-
    format(atom(Placeholder), '{{~w}}', [Key]),
    atom_string(Value, ValueStr),
    atom_string(Template, TemplateStr),
    atom_string(Placeholder, PlaceholderStr),
    replace_substring(TemplateStr, PlaceholderStr, ValueStr, Mid),
    render_template_string(Mid, Rest, Result).

% Helper predicate for substring replacement
replace_substring(String, Find, Replace, Result) :-
    string_length(Find, FindLen),
    (   sub_string(String, Before, FindLen, After, Find)
    ->  sub_string(String, 0, Before, _, Prefix),
        Start is Before + FindLen,
        sub_string(String, Start, After, 0, Suffix),
        replace_substring(Suffix, Find, Replace, RestResult),
        string_concat(Prefix, Replace, Part1),
        string_concat(Part1, RestResult, Result)
    ;   Result = String
    ).

%% Compose multiple templates into one
compose_templates([], _, "") :- !.
compose_templates([T|Ts], Dict, Result) :-
    render_template(T, Dict, R1),
    compose_templates(Ts, Dict, Rs),
    format(string(Result), '~s~s', [R1, Rs]).

%% ============================================
%% BASH CODE GENERATION TEMPLATES
%% ============================================

%% Header template
template(bash_header, '#!/bin/bash
# {{description}}
').

%% Function definition template
template(function, '
{{name}}() {
{{body}}
}').

%% Stream check template - checks if function exists
template(stream_check, '
# Check if {{base}}_stream or {{base}} exists
{{base}}_get_stream() {
    if declare -f {{base}}_stream >/dev/null 2>&1; then
        {{base}}_stream
    elif declare -f {{base}} >/dev/null 2>&1; then
        {{base}}
    else
        echo "Error: neither {{base}}_stream nor {{base}} found" >&2
        return 1
    fi
}').

%% BFS initialization template
template(bfs_init, '
    local start="$1"
    declare -A visited
    declare -A output_seen
    local queue_file="/tmp/{{prefix}}_queue_$$"
    local next_queue="/tmp/{{prefix}}_next_$$"
    trap "rm -f $queue_file $next_queue" EXIT
    echo "$start" > "$queue_file"
    visited["$start"]=1').

%% BFS loop template
template(bfs_loop, '
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        
        while IFS= read -r current; do
            {{source}}_get_stream | grep "^$current:" | while IFS=":" read -r from to; do
                if [[ "$from" == "$current" && -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    echo "$start:$to"
                fi
            done
        done < "$queue_file"
        
        mv "$next_queue" "$queue_file"
    done
    
    rm -f "$queue_file" "$next_queue"').

%% All nodes finder template
template(all_nodes, '
{{name}}_all() {
{{bfs_init}}
{{bfs_loop}}
}').

%% Check function template
template(check_function, '
{{name}}_check() {
    local start="$1"
    local target="$2"
    {{name}}_all "$start" | grep -q "^$start:$target$" && echo "$start:$target"
}').

%% Stream wrapper template
template(stream_wrapper, '
{{name}}_stream() {
    local arg="$1"
    if [[ -n "$arg" ]]; then
        {{name}}_all "$arg"
    else
        {{source}}_get_stream | while IFS=":" read -r start _; do
            echo "$start"
        done | sort -u | while read -r start; do
            {{name}}_all "$start"
        done | sort -u
    fi
}').

%% Generate complete transitive closure implementation
generate_transitive_closure(PredName, BaseName, Code) :-
    atom_string(PredName, PredStr),
    atom_string(BaseName, BaseStr),
    
    % Simple template with named placeholders
    Template = '#!/bin/bash
# {{pred}} - transitive closure of {{base}}

# Check for base stream function
{{base}}_get_stream() {
    if declare -f {{base}}_stream >/dev/null 2>&1; then
        {{base}}_stream
    elif declare -f {{base}} >/dev/null 2>&1; then
        {{base}}
    else
        echo "Error: {{base}} not found" >&2
        return 1
    fi
}

# Main function
{{pred}}() {
    local start="$1"
    local target="$2"
    
    if [[ -z "$target" ]]; then
        {{pred}}_all "$start"
    else
        {{pred}}_check "$start" "$target"
    fi
}

# Find all reachable using BFS
{{pred}}_all() {
    local start="$1"
    declare -A visited
    local queue_file="/tmp/{{pred}}_queue_$"
    local next_queue="/tmp/{{pred}}_next_$"
    
    trap "rm -f $queue_file $next_queue" EXIT PIPE
    
    echo "$start" > "$queue_file"
    visited["$start"]=1
    
    while [[ -s "$queue_file" ]]; do
        > "$next_queue"
        
        while IFS= read -r current; do
            # Use process substitution to keep while loop in current shell
            while IFS=":" read -r from to; do
                if [[ "$from" == "$current" && -z "${visited[$to]}" ]]; then
                    visited["$to"]=1
                    echo "$to" >> "$next_queue"
                    echo "$start:$to"
                fi
            done < <({{base}}_get_stream | grep "^$current:")
        done < "$queue_file"
        
        mv "$next_queue" "$queue_file"
    done
    
    rm -f "$queue_file" "$next_queue"
}

# Check specific relationship
{{pred}}_check() {
    local start="$1"
    local target="$2"
    {{pred}}_all "$start" | grep -q "^$start:$target$" && echo "$start:$target"
}

# Stream function
{{pred}}_stream() {
    {{pred}}_all "$1"
}',
    
    % Render with simple dictionary
    render_template(Template, [
        pred = PredStr,
        base = BaseStr
    ], Code).

%% Test the template system
test_template_system :-
    writeln('=== Testing Template System ==='),
    
    % Test 1: Simple substitution
    write('Test 1 - Simple substitution: '),
    render_template('Hello {{name}}!', [name='World'], R1),
    (R1 = 'Hello World!' -> writeln('PASS') ; (format('FAIL: got ~w~n', [R1]), fail)),
    
    % Test 2: Multiple substitutions
    write('Test 2 - Multiple substitutions: '),
    render_template('{{greeting}} {{name}}', [greeting='Hello', name='Alice'], R2),
    (R2 = 'Hello Alice' -> writeln('PASS') ; (format('FAIL: got ~w~n', [R2]), fail)),
    
    % Test 3: Generate transitive closure
    writeln('Test 3 - Generate transitive closure:'),
    generate_transitive_closure(ancestor, parent, Code3),
    (   sub_string(Code3, _, _, _, 'ancestor_all')
    ->  writeln('PASS - contains ancestor_all function')
    ;   writeln('FAIL - missing expected function')
    ),
    
    writeln('=== Template System Tests Complete ===').