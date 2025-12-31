/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Example: Log Analysis Pipeline using Cross-Target Glue
 *
 * This example demonstrates a three-stage pipeline:
 * 1. AWK: Parse and filter log entries
 * 2. Python: Analyze and enrich data
 * 3. AWK: Generate summary report
 */

:- use_module('../../src/unifyweaver/glue/shell_glue').
:- use_module('../../src/unifyweaver/core/target_registry').
:- use_module('../../src/unifyweaver/core/target_mapping').

%% ============================================
%% Pipeline Definition
%% ============================================

%% Define the predicates and their targets
:- declare_target(parse_logs/4, awk).
:- declare_target(analyze_errors/4, python).
:- declare_target(summarize/3, awk).

%% ============================================
%% Stage 1: Parse and Filter (AWK)
%% ============================================

% AWK logic to parse log lines and extract errors
parse_logs_logic('
    # Parse Apache-style log format
    # Expected: IP - - [timestamp] "METHOD /path HTTP/1.1" status bytes

    # Extract fields using regex
    if (match($0, /([0-9.]+) .* \\[([^\\]]+)\\] "([A-Z]+) ([^ ]+).*" ([0-9]+) ([0-9]+)/, arr)) {
        ip = arr[1]
        timestamp = arr[2]
        method = arr[3]
        path = arr[4]
        status = arr[5]
        bytes = arr[6]

        # Filter: only errors (4xx and 5xx)
        if (status >= 400) {
').

parse_logs_output_fields([ip, timestamp, path, status]).

generate_parse_logs(Script) :-
    parse_logs_logic(Logic),
    parse_logs_output_fields(Fields),
    FullLogic = '
    # Parse Apache-style log format
    if (match($0, /([0-9.]+) .* \\[([^\\]]+)\\] "[A-Z]+ ([^ ]+).*" ([0-9]+)/, arr)) {
        ip = arr[1]
        timestamp = arr[2]
        path = arr[3]
        status = int(arr[4])

        # Filter: only errors (4xx and 5xx)
        if (status >= 400) {
            print ip "\\t" timestamp "\\t" path "\\t" status
        }
    }',
    format(atom(Script),
'#!/usr/bin/awk -f
# Stage 1: Parse logs and filter errors

{
~w
}
', [FullLogic]).

%% ============================================
%% Stage 2: Analyze and Enrich (Python)
%% ============================================

generate_analyze_errors(Script) :-
    generate_python_script(
        '    # Categorize error type
    status = int(record["status"])
    if status >= 500:
        record["error_type"] = "server_error"
        record["severity"] = "critical"
    elif status >= 400:
        record["error_type"] = "client_error"
        record["severity"] = "warning"

    # Extract endpoint category from path
    path = record["path"]
    if "/api/" in path:
        record["category"] = "api"
    elif "/static/" in path:
        record["category"] = "static"
    else:
        record["category"] = "page"',
        [ip, timestamp, path, status, error_type, severity, category],
        [format(tsv)],
        Script
    ).

%% ============================================
%% Stage 3: Summarize (AWK)
%% ============================================

generate_summarize(Script) :-
    format(atom(Script),
'#!/usr/bin/awk -f
# Stage 3: Summarize errors

BEGIN {
    FS = "\\t"
    OFS = "\\t"
}

{
    ip = $1
    path = $3
    status = $4
    error_type = $5
    severity = $6
    category = $7

    # Count by error type
    error_counts[error_type]++

    # Count by category
    category_counts[category]++

    # Track unique IPs
    ips[ip] = 1

    # Count by status code
    status_counts[status]++

    total++
}

END {
    print "=== Error Summary ==="
    print ""
    print "Total errors: " total
    print "Unique IPs: " length(ips)
    print ""

    print "By Error Type:"
    for (t in error_counts) {
        print "  " t ": " error_counts[t]
    }
    print ""

    print "By Category:"
    for (c in category_counts) {
        print "  " c ": " category_counts[c]
    }
    print ""

    print "By Status Code:"
    for (s in status_counts) {
        print "  " s ": " status_counts[s]
    }
}
', []).

%% ============================================
%% Pipeline Orchestration
%% ============================================

generate_full_pipeline(PipelineScript) :-
    generate_pipeline(
        [
            step(parse, awk, 'parse_logs.awk', []),
            step(analyze, python, 'analyze_errors.py', []),
            step(summarize, awk, 'summarize.awk', [])
        ],
        [],
        PipelineScript
    ).

%% ============================================
%% Main: Generate All Scripts
%% ============================================

generate_all :-
    format('Generating log analysis pipeline scripts...~n~n'),

    % Generate Stage 1
    generate_parse_logs(ParseScript),
    open('parse_logs.awk', write, S1),
    write(S1, ParseScript),
    close(S1),
    format('  Created: parse_logs.awk~n'),

    % Generate Stage 2
    generate_analyze_errors(AnalyzeScript),
    open('analyze_errors.py', write, S2),
    write(S2, AnalyzeScript),
    close(S2),
    format('  Created: analyze_errors.py~n'),

    % Generate Stage 3
    generate_summarize(SummarizeScript),
    open('summarize.awk', write, S3),
    write(S3, SummarizeScript),
    close(S3),
    format('  Created: summarize.awk~n'),

    % Generate Pipeline
    generate_full_pipeline(PipelineScript),
    open('run_pipeline.sh', write, S4),
    write(S4, PipelineScript),
    close(S4),
    format('  Created: run_pipeline.sh~n'),

    format('~nDone! Run with:~n'),
    format('  chmod +x run_pipeline.sh~n'),
    format('  cat access.log | ./run_pipeline.sh~n').

:- initialization(generate_all, main).
