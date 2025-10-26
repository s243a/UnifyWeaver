:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% http_source.pl - HTTP source plugin for dynamic sources
% Compiles predicates that fetch data from HTTP endpoints with caching

:- module(http_source, [
    compile_source/4,          % +Pred/Arity, +Config, +Options, -BashCode
    validate_config/1,         % +Config
    source_info/1              % -Info
]).

:- use_module(library(lists)).
:- use_module('../core/template_system').
:- use_module('../core/dynamic_source_compiler').

%% Register this plugin on load
:- initialization(
    register_source_type(http, http_source),
    now
).

%% ============================================
%% PLUGIN INTERFACE
%% ============================================

%% source_info(-Info)
%  Provide information about this source plugin
source_info(info(
    name('HTTP Source'),
    version('1.0.0'),
    description('Fetch data from HTTP endpoints with curl/wget and caching support'),
    supported_arities([1, 2])
)).

%% validate_config(+Config)
%  Validate configuration for HTTP source
validate_config(Config) :-
    % Must have url
    (   member(url(URL), Config),
        atom(URL)
    ->  validate_url(URL)
    ;   format('Error: HTTP source requires url(URL)~n', []),
        fail
    ),
    
    % Validate method if specified
    (   member(method(Method), Config)
    ->  (   member(Method, [get, post, put, delete, head])
        ->  true
        ;   format('Error: invalid method ~w, must be get/post/put/delete/head~n', [Method]),
            fail
        )
    ;   true
    ),
    
    % Validate timeout if specified
    (   member(timeout(T), Config)
    ->  (   number(T), T > 0
        ->  true
        ;   format('Error: timeout must be positive number, got ~w~n', [T]),
            fail
        )
    ;   true
    ),
    
    % Validate cache_duration if specified
    (   member(cache_duration(D), Config)
    ->  (   integer(D), D >= 0
        ->  true
        ;   format('Error: cache_duration must be non-negative integer, got ~w~n', [D]),
            fail
        )
    ;   true
    ),
    
    % Validate tool if specified
    (   member(tool(Tool), Config)
    ->  (   member(Tool, [curl, wget])
        ->  true
        ;   format('Error: tool must be curl or wget, got ~w~n', [Tool]),
            fail
        )
    ;   true
    ).

%% validate_url(+URL)
%  Basic URL validation
validate_url(URL) :-
    (   sub_atom(URL, 0, 4, _, 'http')
    ->  true
    ;   format('Warning: URL ~w does not start with http/https~n', [URL])
    ).

%% compile_source(+Pred/Arity, +Config, +Options, -BashCode)
%  Compile HTTP source to bash code
compile_source(Pred/Arity, Config, Options, BashCode) :-
    format('  Compiling HTTP source: ~w/~w~n', [Pred, Arity]),

    % Validate configuration
    validate_config(Config),

    % Merge config and options
    append(Config, Options, AllOptions),

    % Extract required parameters
    member(url(URL), AllOptions),

    % Extract optional parameters with defaults
    (   member(method(Method), AllOptions)
    ->  true
    ;   Method = get  % Default method
    ),
    (   member(timeout(Timeout), AllOptions)
    ->  true
    ;   Timeout = 30  % Default timeout
    ),
    (   member(tool(Tool), AllOptions)
    ->  true
    ;   Tool = curl  % Default tool
    ),
    (   member(cache_file(CacheFile), AllOptions)
    ->  true
    ;   format(atom(CacheFile), '/tmp/~w_cache_$$', [Pred])
    ),
    (   member(cache_duration(CacheDuration), AllOptions)
    ->  true
    ;   CacheDuration = 300  % Default 5 minutes
    ),
    (   member(headers(Headers), AllOptions)
    ->  true
    ;   Headers = []
    ),
    (   member(post_data(PostData), AllOptions)
    ->  true
    ;   PostData = ''
    ),
    (   member(user_agent(UserAgent), AllOptions)
    ->  true
    ;   UserAgent = 'UnifyWeaver/0.0.2'
    ),

    % Generate bash code using template
    atom_string(Pred, PredStr),
    generate_http_bash(PredStr, Arity, URL, Method, Tool, Timeout,
                      CacheFile, CacheDuration, Headers, PostData, UserAgent, BashCode).

%% ============================================
%% BASH CODE GENERATION
%% ============================================

%% generate_http_bash(+PredStr, +Arity, +URL, +Method, +Tool, +Timeout,
%%                    +CacheFile, +CacheDuration, +Headers, +PostData, +UserAgent, -BashCode)
%  Generate bash code for HTTP source
generate_http_bash(PredStr, Arity, URL, Method, Tool, Timeout,
                  CacheFile, CacheDuration, Headers, PostData, UserAgent, BashCode) :-
    
    % Generate HTTP command based on tool
    generate_http_command(Tool, Method, URL, Timeout, Headers, PostData, UserAgent, HttpCommand),
    
    % Generate header options
    generate_header_options(Headers, Tool, HeaderOptions),
    
    % Generate POST data handling
    generate_post_data_options(PostData, Method, Tool, PostOptions),
    
    % Convert method to uppercase for display
    upcase_atom(Method, MethodUpper),
    
    % Render template based on arity and caching
    (   CacheDuration > 0 ->
        TemplateName = http_cached_source
    ;   TemplateName = http_basic_source
    ),
    
    render_named_template(TemplateName,
        [pred=PredStr, url=URL, method=MethodUpper, tool=Tool,
         timeout=Timeout, cache_file=CacheFile, cache_duration=CacheDuration,
         http_command=HttpCommand, header_options=HeaderOptions,
         post_options=PostOptions, user_agent=UserAgent, arity=Arity],
        [source_order([file, generated])],
        BashCode).

%% generate_http_command(+Tool, +Method, +URL, +Timeout, +Headers, +PostData, +UserAgent, -Command)
%  Generate the appropriate HTTP command for curl or wget
generate_http_command(curl, Method, _URL, Timeout, _Headers, _PostData, UserAgent, Command) :-
    format(atom(Command), 'curl -s -X ~w --max-time ~w -A "~w"', [Method, Timeout, UserAgent]).
generate_http_command(wget, Method, _URL, Timeout, _Headers, _PostData, UserAgent, Command) :-
    format(atom(Command), 'wget -qO- --method=~w --timeout=~w --user-agent="~w"', [Method, Timeout, UserAgent]).

%% generate_header_options(+Headers, +Tool, -Options)
%  Generate header options for the chosen tool
generate_header_options([], _, '') :- !.
generate_header_options(Headers, curl, Options) :-
    maplist(curl_header_option, Headers, HeaderOpts),
    atomic_list_concat(HeaderOpts, ' ', Options).
generate_header_options(Headers, wget, Options) :-
    maplist(wget_header_option, Headers, HeaderOpts),
    atomic_list_concat(HeaderOpts, ' ', Options).

curl_header_option(Header, Option) :-
    format(atom(Option), '-H "~w"', [Header]).

wget_header_option(Header, Option) :-
    format(atom(Option), '--header="~w"', [Header]).

%% generate_post_data_options(+PostData, +Method, +Tool, -Options)
%  Generate POST data options
generate_post_data_options('', _, _, '') :- !.
generate_post_data_options(PostData, Method, curl, Options) :-
    (   member(Method, [post, put])
    ->  format(atom(Options), '--data "~w"', [PostData])
    ;   Options = ''
    ).
generate_post_data_options(PostData, Method, wget, Options) :-
    (   member(Method, [post, put])
    ->  format(atom(Options), '--post-data="~w"', [PostData])
    ;   Options = ''
    ).

%% ============================================
%% HARDCODED TEMPLATES (fallback)
%% ============================================

:- multifile template_system:template/2.

% Basic HTTP template - no caching
template_system:template(http_basic_source, '#!/bin/bash
# {{pred}} - HTTP source ({{method}} {{url}})

{{pred}}() {
    local url="{{url}}"
    local additional_args="$*"
    
    # Add any additional arguments to URL as query parameters
    if [[ -n "$additional_args" ]]; then
        if [[ "$url" == *"?"* ]]; then
            url="$url&$additional_args"
        else
            url="$url?$additional_args"
        fi
    fi
    
    # Make HTTP request
    {{http_command}} \c
        {{header_options}} \c
        {{post_options}} \c
        "$url"
    
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "HTTP request failed with exit code $exit_code" >&2
        return $exit_code
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_raw() {
    # Raw output without any processing
    {{pred}} "$@"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

% Cached HTTP template - with caching support
template_system:template(http_cached_source, '#!/bin/bash
# {{pred}} - Cached HTTP source ({{method}} {{url}})

{{pred}}() {
    local url="{{url}}"
    local cache_file="{{cache_file}}"
    local cache_duration="{{cache_duration}}"
    local additional_args="$*"
    
    # Add any additional arguments to URL as query parameters
    if [[ -n "$additional_args" ]]; then
        if [[ "$url" == *"?"* ]]; then
            url="$url&$additional_args"
        else
            url="$url?$additional_args"
        fi
        # Include args in cache file name to avoid conflicts
        cache_file="${cache_file}_$(echo "$additional_args" | md5sum | cut -d\" \" -f1)"
    fi
    
    # Check cache validity
    if [[ -f "$cache_file" ]]; then
        local cache_age=$(( $(date +%s) - $(stat -c%Y "$cache_file" 2>/dev/null || echo 0) ))
        if [[ $cache_age -lt $cache_duration ]]; then
            cat "$cache_file"
            return 0
        fi
    fi
    
    # Make HTTP request and cache result
    local temp_file="${cache_file}.tmp"
    {{http_command}} \c
        {{header_options}} \c
        {{post_options}} \c
        "$url" > "$temp_file"
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        mv "$temp_file" "$cache_file"
        cat "$cache_file"
    else
        rm -f "$temp_file"
        echo "HTTP request failed with exit code $exit_code" >&2
        return $exit_code
    fi
}

{{pred}}_stream() {
    {{pred}}
}

{{pred}}_raw() {
    # Raw output without any processing
    {{pred}} "$@"
}

{{pred}}_cache_clear() {
    # Clear cached data
    rm -f {{cache_file}}*
    echo "Cache cleared for {{pred}}"
}

{{pred}}_cache_info() {
    # Show cache information
    local cache_file="{{cache_file}}"
    if [[ -f "$cache_file" ]]; then
        local cache_age=$(( $(date +%s) - $(stat -c%Y "$cache_file") ))
        local cache_duration="{{cache_duration}}"
        echo "Cache file: $cache_file"
        echo "Cache age: ${cache_age}s (expires in $((cache_duration - cache_age))s)"
        echo "Cache size: $(stat -c%s "$cache_file") bytes"
    else
        echo "No cache file found"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    {{pred}} "$@"
fi
').

%% ============================================
%% PURE POWERSHELL TEMPLATES
%% ============================================

% Pure PowerShell template for HTTP source
template_system:template(http_source_powershell_pure, '# {{pred}} - HTTP source - Pure PowerShell
# URL: {{url}}
# Generated by UnifyWeaver - Pure PowerShell mode (no bash dependency)

function {{pred}} {
    param([string]$Key)

    try {
        # Make HTTP request
        {{#has_headers}}
        $headers = @{
            {{#headers}}
            ''{{header_name}}'' = ''{{header_value}}''{{^last}};{{/last}}
            {{/headers}}
        }
        $response = Invoke-RestMethod -Uri ''{{url}}'' -Headers $headers -ErrorAction Stop
        {{/has_headers}}
        {{^has_headers}}
        $response = Invoke-RestMethod -Uri ''{{url}}'' -ErrorAction Stop
        {{/has_headers}}

        # Handle pagination if configured
        {{#paginated}}
        $allResults = @()
        $page = {{start_page}}
        do {
            $pageUrl = ''{{url}}'' -replace ''{{page_placeholder}}'', $page
            $pageData = Invoke-RestMethod -Uri $pageUrl -ErrorAction Stop
            $allResults += $pageData
            $page++
        } while ($pageData.Count -gt 0 -and $page -le {{max_pages}})
        $response = $allResults
        {{/paginated}}

        # Apply filter logic
        {{#has_key_filter}}
        if ($Key) {
            $results = $response | Where-Object { $_.{{key_field}} -eq $Key }
        } else {
            $results = $response
        }
        {{/has_key_filter}}
        {{^has_key_filter}}
        $results = $response
        {{/has_key_filter}}

        # Format output
        foreach ($item in $results) {
            {{#arity_1}}
            $item.{{field_0}}
            {{/arity_1}}
            {{#arity_2plus}}
            $values = @({{#fields}}$item.{{field}}{{^last}}, {{/last}}{{/fields}})
            $values -join ":"
            {{/arity_2plus}}
        }
    }
    catch {
        Write-Error "HTTP request failed: $_"
        return $null
    }
}

function {{pred}}_stream {
    {{pred}}
}

function {{pred}}_check {
    param([string]$Key)
    $result = {{pred}} $Key
    if ($result) {
        return "$Key exists"
    }
}

# Auto-execute when run directly (not when dot-sourced)
if ($MyInvocation.InvocationName -ne ''.'') {
    {{pred}} @args
}
').
