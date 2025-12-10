#!/usr/bin/env swipl
%% verify_backends.pl - Simple verification of backend combinations

:- initialization(main, main).

:- asserta(file_search_path(unifyweaver, '../src/unifyweaver')).
:- use_module(unifyweaver('sources/semantic_source')).
:- use_module(unifyweaver('core/dynamic_source_compiler')).

main(_Argv) :-
    writeln('=== Verifying Backend Combinations ==='),
    nl,
    test_python_python_onnx,
    test_powershell_python_onnx,
    test_csharp_csharp_native,
    test_python_go_service,
    writeln(''),
    writeln('✓ All tests passed!'),
    halt(0).

main(_) :-
    writeln(''),
    writeln('✗ Tests failed'),
    halt(1).

test_python_python_onnx :-
    writeln('Testing Python + python_onnx...'),
    compile_dynamic_source(
        test_py,
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
    sub_string(CodeStr, _, _, _, "#!/usr/bin/env python3"),
    sub_string(CodeStr, _, _, _, "import"),
    writeln('  ✓ Python + python_onnx working').

test_powershell_python_onnx :-
    writeln('Testing PowerShell + python_onnx...'),
    compile_dynamic_source(
        test_ps,
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
    sub_string(CodeStr, _, _, _, "function test_ps"),
    writeln('  ✓ PowerShell + python_onnx working').

test_csharp_csharp_native :-
    writeln('Testing C# + csharp_native...'),
    compile_dynamic_source(
        test_cs,
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
    sub_string(CodeStr, _, _, _, "using System"),
    sub_string(CodeStr, _, _, _, "namespace SemanticSearch"),
    writeln('  ✓ C# + csharp_native working').

test_python_go_service :-
    writeln('Testing Python + go_service...'),
    compile_dynamic_source(
        test_py_go,
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
    sub_string(CodeStr, _, _, _, "#!/usr/bin/env python3"),
    sub_string(CodeStr, _, _, _, "http://localhost:8080"),
    writeln('  ✓ Python + go_service working').
