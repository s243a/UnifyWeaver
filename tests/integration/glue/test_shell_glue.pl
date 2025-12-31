/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Integration tests for shell_glue module
 */

:- use_module('../../../src/unifyweaver/glue/shell_glue').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== Shell Glue Integration Tests ===~n~n'),
    run_all_tests,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_awk_script_generation,
    test_python_script_generation,
    test_bash_script_generation,
    test_pipeline_generation,
    test_format_options.

assert_contains(String, Substring, TestName) :-
    (   sub_atom(String, _, _, _, Substring)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED: "~w" not found~n', [TestName, Substring]),
        fail
    ).

assert_true(Goal, TestName) :-
    (   call(Goal)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED~n', [TestName]),
        fail
    ).

%% ============================================
%% Test: AWK Script Generation
%% ============================================

test_awk_script_generation :-
    format('Test: AWK script generation~n'),

    % Basic TSV script
    generate_awk_script(
        '# Filter high salary\n    if (salary > 50000) {',
        [name, dept, salary],
        [format(tsv)],
        AwkScript1
    ),
    assert_contains(AwkScript1, 'FS = "\\t"', 'AWK sets TSV field separator'),
    assert_contains(AwkScript1, 'name = $1', 'AWK assigns field 1'),
    assert_contains(AwkScript1, 'salary = $3', 'AWK assigns field 3'),

    % With header skip
    generate_awk_script(
        '# Process',
        [id, value],
        [format(tsv), header(true)],
        AwkScript2
    ),
    assert_contains(AwkScript2, 'NR == 1 { next }', 'AWK skips header'),

    % JSON output
    generate_awk_script(
        '# Convert',
        [name, age],
        [format(json)],
        AwkScript3
    ),
    assert_contains(AwkScript3, 'print "{', 'AWK outputs JSON'),

    format('~n').

%% ============================================
%% Test: Python Script Generation
%% ============================================

test_python_script_generation :-
    format('Test: Python script generation~n'),

    % Basic TSV script
    generate_python_script(
        '    # Transform logic here\n    record["salary"] = int(record["salary"]) * 1.1',
        [name, dept, salary],
        [format(tsv)],
        PyScript1
    ),
    assert_contains(PyScript1, 'import sys', 'Python imports sys'),
    assert_contains(PyScript1, 'split("\\t")', 'Python splits on tab'),
    assert_contains(PyScript1, 'def process_record', 'Python has process function'),

    % JSON format
    generate_python_script(
        '    # JSON processing',
        [id, data],
        [format(json)],
        PyScript2
    ),
    assert_contains(PyScript2, 'import json', 'Python imports json'),
    assert_contains(PyScript2, 'json.loads', 'Python parses JSON'),
    assert_contains(PyScript2, 'json.dumps', 'Python outputs JSON'),

    % With header
    generate_python_script(
        '    pass',
        [col1, col2],
        [format(tsv), header(true)],
        PyScript3
    ),
    assert_contains(PyScript3, 'skip_header = True', 'Python skips header'),

    format('~n').

%% ============================================
%% Test: Bash Script Generation
%% ============================================

test_bash_script_generation :-
    format('Test: Bash script generation~n'),

    % Basic TSV script
    generate_bash_script(
        '    # Process each record\n    new_value=$((salary * 2))',
        [name, dept, salary],
        [format(tsv)],
        BashScript1
    ),
    assert_contains(BashScript1, 'set -euo pipefail', 'Bash has strict mode'),
    assert_contains(BashScript1, 'IFS=$\'\\t\'', 'Bash uses tab separator'),
    assert_contains(BashScript1, 'read -r name dept salary', 'Bash reads fields'),

    % With header
    generate_bash_script(
        '    echo "Processing"',
        [a, b],
        [format(tsv), header(true)],
        BashScript2
    ),
    assert_contains(BashScript2, 'read -r _header', 'Bash skips header'),

    format('~n').

%% ============================================
%% Test: Pipeline Generation
%% ============================================

test_pipeline_generation :-
    format('Test: Pipeline generation~n'),

    % Simple two-step pipeline
    generate_pipeline(
        [
            step(filter, awk, 'filter.awk', []),
            step(analyze, python, 'analyze.py', [])
        ],
        [],
        Pipeline1
    ),
    assert_contains(Pipeline1, '#!/bin/bash', 'Pipeline is bash script'),
    assert_contains(Pipeline1, 'awk -f "filter.awk"', 'Pipeline includes AWK'),
    assert_contains(Pipeline1, 'python3 "analyze.py"', 'Pipeline includes Python'),
    assert_contains(Pipeline1, '|', 'Pipeline uses pipes'),

    % With input/output files
    generate_pipeline(
        [
            step(process, awk, 'process.awk', [])
        ],
        [input('data.tsv'), output('result.tsv')],
        Pipeline2
    ),
    assert_contains(Pipeline2, 'cat "data.tsv"', 'Pipeline reads input file'),
    assert_contains(Pipeline2, '> "result.tsv"', 'Pipeline writes output file'),

    % Three-step pipeline
    generate_pipeline(
        [
            step(extract, bash, 'extract.sh', []),
            step(transform, python, 'transform.py', []),
            step(load, awk, 'load.awk', [])
        ],
        [],
        Pipeline3
    ),
    assert_contains(Pipeline3, 'bash "extract.sh"', 'Pipeline includes Bash'),
    assert_contains(Pipeline3, 'python3 "transform.py"', 'Pipeline step 2'),
    assert_contains(Pipeline3, 'awk -f "load.awk"', 'Pipeline step 3'),

    format('~n').

%% ============================================
%% Test: Format Options
%% ============================================

test_format_options :-
    format('Test: Format options~n'),

    % Input format
    input_format([input_format(json)], F1),
    assert_true(F1 == json, 'input_format extracts json'),

    input_format([format(csv)], F2),
    assert_true(F2 == csv, 'format option used for input'),

    input_format([], F3),
    assert_true(F3 == tsv, 'default input format is tsv'),

    % Output format
    output_format([output_format(json)], F4),
    assert_true(F4 == json, 'output_format extracts json'),

    output_format([], F5),
    assert_true(F5 == tsv, 'default output format is tsv'),

    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
