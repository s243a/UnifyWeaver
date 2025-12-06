/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Integration tests for native_glue module
 */

:- use_module('../../../src/unifyweaver/glue/native_glue').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== Native Glue Integration Tests ===~n~n'),
    run_all_tests,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_go_code_generation,
    test_rust_code_generation,
    test_go_parallel_generation,
    test_json_mode_generation,
    test_build_scripts,
    test_cross_compilation,
    test_pipeline_orchestration.

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
%% Test: Go Code Generation
%% ============================================

test_go_code_generation :-
    format('Test: Go code generation~n'),

    % Basic TSV main
    generate_go_pipe_main('return fields', [], GoCode1),
    assert_contains(GoCode1, 'package main', 'Go has package main'),
    assert_contains(GoCode1, 'import (', 'Go has imports'),
    assert_contains(GoCode1, 'bufio', 'Go imports bufio'),
    assert_contains(GoCode1, 'strings', 'Go imports strings'),
    assert_contains(GoCode1, 'func process(fields []string) []string', 'Go has process function'),
    assert_contains(GoCode1, 'bufio.NewScanner', 'Go uses scanner'),
    assert_contains(GoCode1, 'strings.Split(line, "\\t")', 'Go splits on tab'),
    assert_contains(GoCode1, 'strings.Join(result, "\\t")', 'Go joins with tab'),

    % Buffer size increase
    assert_contains(GoCode1, '10*1024*1024', 'Go has large buffer'),

    format('~n').

%% ============================================
%% Test: Rust Code Generation
%% ============================================

test_rust_code_generation :-
    format('Test: Rust code generation~n'),

    % Basic TSV main
    generate_rust_pipe_main('Some(fields.iter().map(|s| s.to_string()).collect())', [], RustCode1),
    assert_contains(RustCode1, 'use std::io', 'Rust uses std::io'),
    assert_contains(RustCode1, 'BufRead', 'Rust imports BufRead'),
    assert_contains(RustCode1, 'fn process(fields: &[&str]) -> Option<Vec<String>>', 'Rust has process function'),
    assert_contains(RustCode1, 'stdin.lock().lines()', 'Rust reads lines'),
    assert_contains(RustCode1, 'split(\'\\t\')', 'Rust splits on tab'),
    assert_contains(RustCode1, 'result.join("\\t")', 'Rust joins with tab'),

    % Error handling
    assert_contains(RustCode1, 'match line', 'Rust matches line result'),
    assert_contains(RustCode1, 'Err(e)', 'Rust handles errors'),

    format('~n').

%% ============================================
%% Test: Go Parallel Generation
%% ============================================

test_go_parallel_generation :-
    format('Test: Go parallel code generation~n'),

    % Parallel with 4 workers
    generate_go_pipe_main('return fields', [parallel(4)], GoParallel),
    assert_contains(GoParallel, 'sync', 'Go imports sync'),
    assert_contains(GoParallel, 'sync.WaitGroup', 'Go uses WaitGroup'),
    assert_contains(GoParallel, 'chan string', 'Go uses channels'),
    assert_contains(GoParallel, 'for i := 0; i < 4; i++', 'Go spawns 4 workers'),
    assert_contains(GoParallel, 'go func()', 'Go uses goroutines'),

    format('~n').

%% ============================================
%% Test: JSON Mode Generation
%% ============================================

test_json_mode_generation :-
    format('Test: JSON mode code generation~n'),

    % Go JSON mode
    generate_go_pipe_main('return &record', [format(json), fields([id, name, age])], GoJson),
    assert_contains(GoJson, 'encoding/json', 'Go imports json'),
    assert_contains(GoJson, 'json.Unmarshal', 'Go unmarshals JSON'),
    assert_contains(GoJson, 'json.Marshal', 'Go marshals JSON'),
    assert_contains(GoJson, 'type Record', 'Go has Record type'),
    assert_contains(GoJson, 'json:"id"', 'Go has id JSON tag'),
    assert_contains(GoJson, 'json:"name"', 'Go has name JSON tag'),

    % Rust JSON mode
    generate_rust_pipe_main('Some(record)', [format(json), fields([id, name])], RustJson),
    assert_contains(RustJson, 'serde', 'Rust uses serde'),
    assert_contains(RustJson, 'serde_json', 'Rust uses serde_json'),
    assert_contains(RustJson, 'Deserialize', 'Rust derives Deserialize'),
    assert_contains(RustJson, 'Serialize', 'Rust derives Serialize'),
    assert_contains(RustJson, 'serde_json::from_str', 'Rust parses JSON'),
    assert_contains(RustJson, 'serde_json::to_string', 'Rust outputs JSON'),

    format('~n').

%% ============================================
%% Test: Build Scripts
%% ============================================

test_build_scripts :-
    format('Test: Build script generation~n'),

    % Go build script
    generate_go_build_script('src/filter.go', [], GoBuild),
    assert_contains(GoBuild, '#!/bin/bash', 'Go build is bash script'),
    assert_contains(GoBuild, 'set -euo pipefail', 'Go build has strict mode'),
    assert_contains(GoBuild, 'go build', 'Go build uses go build'),
    assert_contains(GoBuild, 'filter', 'Go build outputs filter'),

    % Go build with optimization
    generate_go_build_script('src/filter.go', [optimize(true)], GoBuildOpt),
    assert_contains(GoBuildOpt, 'ldflags', 'Go optimized build has ldflags'),
    assert_contains(GoBuildOpt, '-s -w', 'Go optimized strips symbols'),

    % Rust build script
    generate_rust_build_script('myproject/src/main.rs', [profile(release)], RustBuild),
    assert_contains(RustBuild, 'cargo build', 'Rust build uses cargo'),
    assert_contains(RustBuild, '--release', 'Rust build is release'),

    format('~n').

%% ============================================
%% Test: Cross-Compilation
%% ============================================

test_cross_compilation :-
    format('Test: Cross-compilation~n'),

    % Get available targets
    cross_compile_targets(Targets),
    length(Targets, NumTargets),
    assert_true(NumTargets >= 5, 'Has at least 5 cross-compile targets'),

    % Go cross-compile script
    generate_cross_compile(go, 'main.go', [linux-amd64, darwin-arm64, windows-amd64], GoXCompile),
    assert_contains(GoXCompile, 'GOOS=linux GOARCH=amd64', 'Go cross-compiles to Linux'),
    assert_contains(GoXCompile, 'GOOS=darwin GOARCH=arm64', 'Go cross-compiles to macOS ARM'),
    assert_contains(GoXCompile, 'GOOS=windows GOARCH=amd64', 'Go cross-compiles to Windows'),
    assert_contains(GoXCompile, '.exe', 'Go Windows has .exe extension'),

    % Rust cross-compile script
    generate_cross_compile(rust, 'src/main.rs', [linux-amd64, darwin-arm64], RustXCompile),
    assert_contains(RustXCompile, 'rustup target add', 'Rust adds targets'),
    assert_contains(RustXCompile, 'cargo build --release --target', 'Rust builds for target'),
    assert_contains(RustXCompile, 'x86_64-unknown-linux-gnu', 'Rust has Linux target'),
    assert_contains(RustXCompile, 'aarch64-apple-darwin', 'Rust has macOS ARM target'),

    format('~n').

%% ============================================
%% Test: Pipeline Orchestration
%% ============================================

test_pipeline_orchestration :-
    format('Test: Pipeline orchestration~n'),

    % Simple pipeline
    generate_native_pipeline(
        [
            step(filter, go, './filter', []),
            step(transform, rust, './transform', []),
            step(aggregate, awk, 'aggregate.awk', [])
        ],
        [],
        Pipeline1
    ),
    assert_contains(Pipeline1, '#!/bin/bash', 'Pipeline is bash script'),
    assert_contains(Pipeline1, 'set -euo pipefail', 'Pipeline has strict mode'),
    assert_contains(Pipeline1, '"./filter"', 'Pipeline includes Go binary'),
    assert_contains(Pipeline1, '"./transform"', 'Pipeline includes Rust binary'),
    assert_contains(Pipeline1, 'awk -f "aggregate.awk"', 'Pipeline includes AWK'),
    assert_contains(Pipeline1, '|', 'Pipeline uses pipes'),

    % Pipeline with input/output files
    generate_native_pipeline(
        [step(process, go, './process', [])],
        [input('data.tsv'), output('result.tsv')],
        Pipeline2
    ),
    assert_contains(Pipeline2, 'cat "data.tsv"', 'Pipeline reads input file'),
    assert_contains(Pipeline2, '> "result.tsv"', 'Pipeline writes output file'),

    % Mixed native and script pipeline
    generate_native_pipeline(
        [
            step(parse, python, 'parse.py', []),
            step(crunch, go, './crunch', []),
            step(format, bash, 'format.sh', [])
        ],
        [],
        Pipeline3
    ),
    assert_contains(Pipeline3, 'python3 "parse.py"', 'Pipeline includes Python'),
    assert_contains(Pipeline3, '"./crunch"', 'Pipeline includes Go'),
    assert_contains(Pipeline3, 'bash "format.sh"', 'Pipeline includes Bash'),

    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
