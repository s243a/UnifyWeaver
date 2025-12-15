#!/usr/bin/env swipl
/**
 * test_backends.pl - Test semantic source with multiple target/backend combinations
 *
 * Tests all supported Target × Backend combinations:
 * - bash + python_onnx (original)
 * - python + python_onnx (new)
 * - powershell + python_onnx (new)
 * - bash + go_service
 * - python + go_service (new)
 * - powershell + go_service (new)
 * - csharp + go_service (new)
 * - bash + rust_candle
 * - python + rust_candle (new)
 * - powershell + rust_candle (new)
 * - csharp + rust_candle (new)
 * - csharp + csharp_native (new)
 */

:- use_module(library(filesex)).
:- use_module(library(plunit)).

% Add parent directory to module search path
:- asserta(file_search_path(unifyweaver, '../src/unifyweaver')).

% Load required modules
:- use_module(unifyweaver('sources/semantic_source')).
:- use_module(unifyweaver('core/dynamic_source_compiler')).

:- begin_tests(semantic_backends).

%% Test: Python + python_onnx
test(python_python_onnx, [true]) :-
    compile_dynamic_source(
        test_papers_py,
        semantic_source(
            python,
            'test_data/papers_vectors.json',
            [
                embedding_backend(python_onnx),
                backend_config(default),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's Python code
    sub_string(CodeStr, _, _, _, "#!/usr/bin/env python3"),
    sub_string(CodeStr, _, _, _, "import"),
    sub_string(CodeStr, _, _, _, "onnxruntime").

%% Test: PowerShell + python_onnx
test(powershell_python_onnx, [true]) :-
    compile_dynamic_source(
        test_papers_ps,
        semantic_source(
            powershell,
            'test_data/papers_vectors.json',
            [
                embedding_backend(python_onnx),
                backend_config(default),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's PowerShell code
    sub_string(CodeStr, _, _, _, "function test_papers_ps"),
    sub_string(CodeStr, _, _, _, "param("),
    sub_string(CodeStr, _, _, _, "$pythonCode").

%% Test: C# + csharp_native
test(csharp_csharp_native, [true]) :-
    compile_dynamic_source(
        test_papers_cs,
        semantic_source(
            csharp,
            'test_data/papers_vectors.json',
            [
                embedding_backend(csharp_native),
                backend_config(default),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's C# code
    sub_string(CodeStr, _, _, _, "using System"),
    sub_string(CodeStr, _, _, _, "namespace SemanticSearch"),
    sub_string(CodeStr, _, _, _, "InferenceSession").

%% Test: Python + go_service
test(python_go_service, [true]) :-
    compile_dynamic_source(
        test_papers_go_py,
        semantic_source(
            python,
            'test_data/papers_vectors.json',
            [
                embedding_backend(go_service),
                backend_config([url('http://localhost:8080')]),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's Python code calling Go service
    sub_string(CodeStr, _, _, _, "#!/usr/bin/env python3"),
    sub_string(CodeStr, _, _, _, "http://localhost:8080"),
    sub_string(CodeStr, _, _, _, "urllib.request").

%% Test: PowerShell + go_service
test(powershell_go_service, [true]) :-
    compile_dynamic_source(
        test_papers_go_ps,
        semantic_source(
            powershell,
            'test_data/papers_vectors.json',
            [
                embedding_backend(go_service),
                backend_config([url('http://localhost:8080')]),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's PowerShell code calling Go service
    sub_string(CodeStr, _, _, _, "function test_papers_go_ps"),
    sub_string(CodeStr, _, _, _, "http://localhost:8080"),
    sub_string(CodeStr, _, _, _, "Invoke-RestMethod").

%% Test: C# + go_service
test(csharp_go_service, [true]) :-
    compile_dynamic_source(
        test_papers_go_cs,
        semantic_source(
            csharp,
            'test_data/papers_vectors.json',
            [
                embedding_backend(go_service),
                backend_config([url('http://localhost:8080')]),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's C# code calling Go service
    sub_string(CodeStr, _, _, _, "using System"),
    sub_string(CodeStr, _, _, _, "HttpClient"),
    sub_string(CodeStr, _, _, _, "http://localhost:8080").

%% Test: Python + rust_candle
test(python_rust_candle, [true]) :-
    compile_dynamic_source(
        test_papers_rust_py,
        semantic_source(
            python,
            'test_data/papers_vectors.json',
            [
                embedding_backend(rust_candle),
                backend_config([binary_path('./rust_semantic_search')]),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's Python code calling Rust binary
    sub_string(CodeStr, _, _, _, "#!/usr/bin/env python3"),
    sub_string(CodeStr, _, _, _, "subprocess"),
    sub_string(CodeStr, _, _, _, "./rust_semantic_search").

%% Test: PowerShell + rust_candle
test(powershell_rust_candle, [true]) :-
    compile_dynamic_source(
        test_papers_rust_ps,
        semantic_source(
            powershell,
            'test_data/papers_vectors.json',
            [
                embedding_backend(rust_candle),
                backend_config([binary_path('./rust_semantic_search')]),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's PowerShell code calling Rust binary
    sub_string(CodeStr, _, _, _, "function test_papers_rust_ps"),
    sub_string(CodeStr, _, _, _, "./rust_semantic_search"),
    sub_string(CodeStr, _, _, _, "& ").

%% Test: C# + rust_candle
test(csharp_rust_candle, [true]) :-
    compile_dynamic_source(
        test_papers_rust_cs,
        semantic_source(
            csharp,
            'test_data/papers_vectors.json',
            [
                embedding_backend(rust_candle),
                backend_config([binary_path('./rust_semantic_search')]),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's C# code calling Rust binary
    sub_string(CodeStr, _, _, _, "using System"),
    sub_string(CodeStr, _, _, _, "ProcessStartInfo"),
    sub_string(CodeStr, _, _, _, "./rust_semantic_search").

%% Test: Bash + python_onnx (original - ensure still works)
test(bash_python_onnx, [true]) :-
    compile_dynamic_source(
        test_papers_bash,
        semantic_source(
            bash,
            'test_data/papers_vectors.json',
            [
                embedding_backend(python_onnx),
                backend_config(default),
                threshold(0.6),
                top_k(5),
                similarity_metric(cosine),
                normalize(true)
            ]
        ),
        Code
    ),
    atom_string(Code, CodeStr),
    % Verify it's bash code
    sub_string(CodeStr, _, _, _, "#!/bin/bash"),
    sub_string(CodeStr, _, _, _, "test_papers_bash()"),
    sub_string(CodeStr, _, _, _, "python3").

:- end_tests(semantic_backends).

%% Run all tests
main :-
    run_tests([semantic_backends]),
    halt(0).

main :-
    format('~nTests FAILED~n', []),
    halt(1).

:- initialization(main, main).
