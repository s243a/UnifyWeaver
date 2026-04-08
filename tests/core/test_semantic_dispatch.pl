:- encoding(utf8).
% test_semantic_dispatch.pl - Comprehensive test for generic semantic search

:- use_module(unifyweaver(core/semantic_compiler)).
:- use_module(unifyweaver(targets/go_target), [compile_semantic_rule_go/4]).
:- use_module(unifyweaver(targets/python_target), []).
:- use_module(unifyweaver(targets/csharp_target), []).
:- use_module(unifyweaver(targets/rust_target), []).

% 1. Register a multi-target provider with GPU
:- dynamic user:semantic_provider/2.
user:semantic_provider(gpu_search/3, [
    targets([
        target(go, [provider(hugot), model('minilm'), device(gpu)]),
        target(python, [provider(transformers), model('minilm'), device(gpu)]),
        target(csharp, [provider(onnx), model('minilm'), device(gpu)])
    ])
]).

% 2. Register a fallback-only provider
user:semantic_provider(fallback_search/3, [
    fallback([provider(onnx), model('fallback-model')])
]).

test_go_dispatch :-
    format('--- Go Dispatch ---~n'),
    compile_semantic_rule_go('my_search', [query], gpu_search(query, 5, _), Code),
    ( sub_string(Code, _, _, _, "embedder.WithGPU()") -> format('  PASS: GPU initialization found~n') ; format('  FAIL: GPU init missing~n')),
    ( sub_string(Code, _, _, _, "GPU initialization failed") -> format('  PASS: Fallback logic found~n') ; format('  FAIL: Fallback logic missing~n')).

test_python_dispatch :-
    format('--- Python Dispatch ---~n'),
    semantic_compiler:compile_semantic_call(python, gpu_search(query, 5, _), [], Code),
    ( sub_string(Code, _, _, _, "device = \"cuda\"") -> format('  PASS: Device cuda found~n') ; format('  FAIL: Device cuda missing~n')),
    ( sub_string(Code, _, _, _, "torch.cuda.is_available()") -> format('  PASS: Availability check found~n') ; format('  FAIL: Check missing~n')).

test_csharp_dispatch :-
    format('--- C# Dispatch ---~n'),
    semantic_compiler:compile_semantic_call(csharp, gpu_search(query, 5, _), [], Code),
    ( sub_string(Code, _, _, _, "opts.AppendExecutionProvider_DML()") -> format('  PASS: DML GPU provider found~n') ; format('  FAIL: DML missing~n')),
    ( sub_string(Code, _, _, _, "OnnxVectorSearch") -> format('  PASS: ONNX searcher found~n') ; format('  FAIL: ONNX missing~n')).

test_fallback_dispatch :-
    format('--- Fallback Dispatch ---~n'),
    % Should use ONNX fallback for Go
    semantic_compiler:compile_semantic_call(go, fallback_search(q, 10, _), [], Code),
    ( sub_string(Code, _, _, _, "hugot") -> format('  PASS: Dispatched to hugot (default provider for go)~n') ; format('  FAIL: Hugot missing~n')),
    ( sub_string(Code, _, _, _, "fallback-model") -> format('  PASS: Used fallback model name~n') ; format('  FAIL: Wrong model~n')).

test_guard :-
    format('--- Load Guard ---~n'),
    % Calling multiple times should not re-run (this is harder to observe but we verify it still works)
    semantic_compiler:is_semantic_predicate(gpu_search(_,_,_)),
    format('  PASS: Guard didn\'t break resolution~n').

test_all :-
    test_go_dispatch,
    test_python_dispatch,
    test_csharp_dispatch,
    test_fallback_dispatch,
    test_guard.

:- initialization(test_all, main).
