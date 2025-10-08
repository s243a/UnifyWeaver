:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% template_system.pl - Robust templating system for UnifyWeaver
% Provides named placeholder substitution and composable template units

:- module(template_system, [
    render_template/3,
    render_named_template/3,
    render_named_template/4,
    compose_templates/3,
    template/2,
    generate_transitive_closure/3,
    load_template/2,
    load_template/3,
    cache_template/2,
    clear_template_cache/0,
    clear_template_cache/1,
    template_config/2,
    set_template_config/2,
    template_config_default/1,
    set_template_config_default/1,
    test_template_system/0
]).

:- use_module(library(lists)).

%% ============================================
%% DYNAMIC PREDICATES (Configuration & Cache)
%% ============================================

:- dynamic cached_template/3.        % cached_template(Name, Template, Timestamp)
:- dynamic template_config/2.        % template_config(Name, Options)
:- dynamic template_config_default/1. % template_config_default(Options)

%% Default configuration
template_config_default([
    source_order([generated]),       % Start conservative: only use hardcoded templates
    template_dir('templates'),
    cache_dir('templates/cache'),
    template_extension('.tmpl.sh'),
    auto_cache(false)
]).

%% ============================================
%% CONFIGURATION MANAGEMENT
%% ============================================

%% set_template_config_default(+Options)
%  Set global default configuration
set_template_config_default(Options) :-
    retractall(template_config_default(_)),
    assertz(template_config_default(Options)).

%% set_template_config(+TemplateName, +Options)
%  Set configuration for specific template
set_template_config(TemplateName, Options) :-
    retractall(template_config(TemplateName, _)),
    assertz(template_config(TemplateName, Options)).

%% get_template_config(+TemplateName, +RuntimeOptions, -MergedConfig)
%  Merge configuration from runtime > per-template > default
get_template_config(TemplateName, RuntimeOptions, MergedConfig) :-
    % Get default config
    template_config_default(DefaultConfig),

    % Get per-template config (if exists)
    (   template_config(TemplateName, TemplateConfig)
    ->  true
    ;   TemplateConfig = []
    ),

    % Merge: runtime overrides template overrides default
    merge_options(RuntimeOptions, TemplateConfig, Temp),
    merge_options(Temp, DefaultConfig, MergedConfig).

%% merge_options(+Priority, +Fallback, -Merged)
%  Merge two option lists, Priority takes precedence
merge_options([], Fallback, Fallback) :- !.
merge_options([H|T], Fallback, Merged) :-
    H =.. [Key, _],
    delete_option(Fallback, Key, Fallback2),
    merge_options(T, Fallback2, Temp),
    Merged = [H|Temp].

%% delete_option(+Options, +Key, -Remaining)
%  Remove all options with given key
delete_option([], _, []) :- !.
delete_option([H|T], Key, Remaining) :-
    H =.. [Key, _],
    !,
    delete_option(T, Key, Remaining).
delete_option([H|T], Key, [H|Remaining]) :-
    delete_option(T, Key, Remaining).

%% get_option(+OptionPattern, +Options, +Default, -Value)
%  Get option value from list, use default if not found
get_option(Key, Options, Default, Value) :-
    OptionPattern =.. [Key, Value],
    (   member(OptionPattern, Options)
    ->  true
    ;   Value = Default
    ).

%% ============================================
%% TEMPLATE CACHING
%% ============================================

%% cache_template(+Name, +Template)
%  Cache a template in memory with timestamp
cache_template(Name, Template) :-
    get_time(Timestamp),
    retractall(cached_template(Name, _, _)),
    assertz(cached_template(Name, Template, Timestamp)).

%% get_cached_template(+Name, -Template)
%  Retrieve cached template if exists
get_cached_template(Name, Template) :-
    cached_template(Name, Template, _).

%% clear_template_cache
%  Clear all cached templates
clear_template_cache :-
    retractall(cached_template(_, _, _)).

%% clear_template_cache(+Name)
%  Clear specific cached template
clear_template_cache(Name) :-
    retractall(cached_template(Name, _, _)).

%% ============================================
%% FILE OPERATIONS
%% ============================================

%% load_template_from_file(+FilePath, -TemplateString)
%  Load template from file
load_template_from_file(FilePath, TemplateString) :-
    exists_file(FilePath),
    read_file_to_string(FilePath, TemplateString, []).

%% find_template_file(+TemplateName, +Config, -FilePath)
%  Find template file path based on configuration
find_template_file(TemplateName, Config, FilePath) :-
    % Check for explicit file_path first
    (   get_option(file_path, Config, none, ExplicitPath),
        ExplicitPath \= none
    ->  FilePath = ExplicitPath
    ;   % Otherwise construct from template_dir + name + extension
        get_option(template_dir, Config, 'templates', Dir),
        get_option(template_extension, Config, '.tmpl.sh', Ext),
        atom_string(TemplateName, NameStr),
        format(string(FilePath), '~w/~w~w', [Dir, NameStr, Ext])
    ).

%% save_template_to_cache_file(+TemplateName, +Template, +Config)
%  Save template to cache directory for inspection/editing
save_template_to_cache_file(TemplateName, Template, Config) :-
    get_option(cache_dir, Config, 'templates/cache', CacheDir),
    get_option(template_extension, Config, '.tmpl.sh', Ext),

    % Create cache directory if it doesn't exist
    (   exists_directory(CacheDir)
    ->  true
    ;   make_directory(CacheDir)
    ),

    % Write template to cache file
    atom_string(TemplateName, NameStr),
    format(string(CachePath), '~w/~w~w', [CacheDir, NameStr, Ext]),
    open(CachePath, write, Stream),
    write(Stream, Template),
    close(Stream).

%% ============================================
%% TEMPLATE LOADING STRATEGIES
%% ============================================

%% try_source(+Name, +Strategy, +Config, -Template)
%  Try loading template from specific strategy
try_source(Name, file, Config, Template) :-
    find_template_file(Name, Config, Path),
    load_template_from_file(Path, Template).

try_source(Name, cached, _Config, Template) :-
    get_cached_template(Name, Template).

try_source(Name, generated, _Config, Template) :-
    template(Name, Template).

%% load_template_with_strategy(+Name, +Config, -Template)
%  Load template using strategy preference order
load_template_with_strategy(Name, Config, Template) :-
    get_option(source_order, Config, [generated], Order),
    try_sources(Name, Order, Config, Template).

%% try_sources(+Name, +Strategies, +Config, -Template)
%  Try strategies in order until one succeeds
try_sources(Name, [Strategy|Rest], Config, Template) :-
    (   try_source(Name, Strategy, Config, Template)
    ->  % Strategy succeeded
        % Check if we should cache this template
        (   Strategy \= cached,  % Don't re-cache already cached template
            get_option(auto_cache, Config, false, AutoCache),
            AutoCache = true
        ->  cache_template(Name, Template),
            save_template_to_cache_file(Name, Template, Config)
        ;   true
        )
    ;   % Strategy failed, try next
        try_sources(Name, Rest, Config, Template)
    ).

%% ============================================
%% PUBLIC API
%% ============================================

%% load_template(+TemplateName, -TemplateString)
%  Load template using default configuration
load_template(TemplateName, TemplateString) :-
    load_template(TemplateName, [], TemplateString).

%% load_template(+TemplateName, +Options, -TemplateString)
%  Load template with runtime options
load_template(TemplateName, Options, TemplateString) :-
    get_template_config(TemplateName, Options, Config),
    load_template_with_strategy(TemplateName, Config, TemplateString).

%% render_named_template(+TemplateName, +Dict, -Result)
%  Load and render template by name
render_named_template(TemplateName, Dict, Result) :-
    render_named_template(TemplateName, Dict, [], Result).

%% render_named_template(+TemplateName, +Dict, +Options, -Result)
%  Load and render template by name with options
render_named_template(TemplateName, Dict, Options, Result) :-
    load_template(TemplateName, Options, Template),
    render_template(Template, Dict, Result).

%% ============================================
%% ORIGINAL TEMPLATE RENDERING (unchanged)
%% ============================================

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
    (sub_string(R1, _, _, _, 'Hello World!') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R1]), fail)),

    % Test 2: Multiple substitutions
    write('Test 2 - Multiple substitutions: '),
    render_template('{{greeting}} {{name}}', [greeting='Hello', name='Alice'], R2),
    (sub_string(R2, _, _, _, 'Hello Alice') -> writeln('PASS') ; (format('FAIL: got ~w~n', [R2]), fail)),

    % Test 3: Generate transitive closure
    writeln('Test 3 - Generate transitive closure:'),
    generate_transitive_closure(ancestor, parent, Code3),
    (   sub_string(Code3, _, _, _, 'ancestor_all')
    ->  writeln('PASS - contains ancestor_all function')
    ;   writeln('FAIL - missing expected function')
    ),

    % Test 4: Template caching
    write('Test 4 - Template caching: '),
    cache_template(test_cached, 'Cached {{value}}'),
    get_cached_template(test_cached, Cached),
    (Cached = 'Cached {{value}}' -> writeln('PASS') ; (format('FAIL: got ~w~n', [Cached]), fail)),

    % Test 5: Load generated template by name
    write('Test 5 - Load by name (generated): '),
    load_template(bash_header, Template5),
    (   sub_string(Template5, _, _, _, '#!/bin/bash')
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [Template5]), fail)
    ),

    % Test 6: Render named template
    write('Test 6 - Render named template: '),
    render_named_template(bash_header, [description='Test Script'], Result6),
    (   sub_string(Result6, _, _, _, '# Test Script')
    ->  writeln('PASS')
    ;   (format('FAIL: got ~w~n', [Result6]), fail)
    ),

    % Test 7: Configuration merging
    write('Test 7 - Configuration merging: '),
    get_template_config(test_template, [source_order([file])], Config7),
    member(source_order([file]), Config7),
    writeln('PASS'),

    % Clean up
    clear_template_cache(test_cached),

    writeln('=== Template System Tests Complete ===').