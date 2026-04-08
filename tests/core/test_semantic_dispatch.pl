:- encoding(utf8).
% test_semantic_dispatch.pl - Test for generic semantic search dispatch
%
% Run from repo root:
%   swipl tests/core/test_semantic_dispatch.pl

:- use_module('../../src/unifyweaver/core/semantic_compiler').

% We test dispatch through semantic_compiler rather than loading the full
% target files (which have heavy dependency chains). Instead, define minimal
% multifile dispatch clauses here that mirror the real target implementations.

:- multifile semantic_compiler:semantic_dispatch/5.

% Minimal Go dispatch (mirrors go_target.pl)
semantic_compiler:semantic_dispatch(go, Goal, Provider, VarMap, Code) :-
    Goal =.. [_, Query, TopK | _],
    ( option(provider(hugot), Provider) ; option(provider(onnx), Provider) ),
    !,
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    option(device(Device), Provider, auto),
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),
    (   Device == gpu
    ->  DeviceOpt = 'embedder.WithGPU()'
    ;   Device == cpu
    ->  DeviceOpt = 'embedder.WithCPU()'
    ;   DeviceOpt = '/* auto */'
    ),
    format(string(Code),
        'emb := hugot.New("~w", ~w)\nresults := emb.Search("~w", ~w)',
        [Model, DeviceOpt, QueryExpr, TopKExpr]).

% Minimal Python dispatch (mirrors python_target.pl)
semantic_compiler:semantic_dispatch(python, Goal, Provider, VarMap, Code) :-
    Goal =.. [_, Query, TopK | _],
    option(provider(transformers), Provider),
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    option(device(Device), Provider, auto),
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),
    (   Device == gpu -> DeviceStr = "cuda" ; DeviceStr = "cpu" ),
    format(string(Code),
        'device = "~w"\nif device == "cuda" and not torch.cuda.is_available():\n    device = "cpu"\nmodel = SentenceTransformer("~w", device=device)\nresults = model.search("~w", ~w)',
        [DeviceStr, Model, QueryExpr, TopKExpr]).

% Minimal C# dispatch (mirrors csharp_target.pl)
semantic_compiler:semantic_dispatch(csharp, Goal, Provider, VarMap, Code) :-
    Goal =.. [_, Query, TopK | _],
    option(provider(onnx), Provider),
    option(model(Model), Provider, 'all-MiniLM-L6-v2'),
    option(device(Device), Provider, auto),
    (   member(Query=QueryVar, VarMap) -> QueryExpr = QueryVar ; QueryExpr = Query ),
    (   member(TopK=TopKVar, VarMap) -> TopKExpr = TopKVar ; TopKExpr = TopK ),
    (   Device == gpu
    ->  DeviceInit = 'opts.AppendExecutionProvider_DML();'
    ;   DeviceInit = 'opts.AppendExecutionProvider_CPU();'
    ),
    format(string(Code),
        'var opts = new SessionOptions();\n~w\nvar searcher = new OnnxVectorSearch("~w", opts);\nvar results = searcher.Search("~w", ~w);',
        [DeviceInit, Model, QueryExpr, TopKExpr]).

% ============================================================================
% TEST PROVIDERS
% ============================================================================

:- dynamic user:semantic_provider/2.

% Multi-target provider with GPU
user:semantic_provider(gpu_search/3, [
    targets([
        target(go, [provider(hugot), model('minilm'), device(gpu)]),
        target(python, [provider(transformers), model('minilm'), device(gpu)]),
        target(csharp, [provider(onnx), model('minilm'), device(gpu)])
    ])
]).

% Fallback-only provider (no target-specific config)
user:semantic_provider(fallback_search/3, [
    fallback([provider(hugot), model('fallback-model')])
]).

% ============================================================================
% TESTS
% ============================================================================

test_go_dispatch :-
    format('--- Go Dispatch ---~n'),
    compile_semantic_call(go, gpu_search(query, 5, _), [], Code),
    assert_contains(Code, "WithGPU()", "GPU initialization"),
    assert_contains(Code, "minilm", "model name").

test_python_dispatch :-
    format('--- Python Dispatch ---~n'),
    compile_semantic_call(python, gpu_search(query, 5, _), [], Code),
    assert_contains(Code, "cuda", "CUDA device"),
    assert_contains(Code, "torch.cuda.is_available()", "availability check").

test_csharp_dispatch :-
    format('--- C# Dispatch ---~n'),
    compile_semantic_call(csharp, gpu_search(query, 5, _), [], Code),
    assert_contains(Code, "AppendExecutionProvider_DML()", "DML GPU provider"),
    assert_contains(Code, "OnnxVectorSearch", "ONNX searcher").

test_fallback_dispatch :-
    format('--- Fallback Dispatch ---~n'),
    % Go is not in fallback_search's targets list, so it should use the fallback
    compile_semantic_call(go, fallback_search(q, 10, _), [], Code),
    assert_contains(Code, "fallback-model", "fallback model name").

test_guard :-
    format('--- Load Guard ---~n'),
    % Calling is_semantic_predicate multiple times should be idempotent
    is_semantic_predicate(gpu_search(_,_,_)),
    is_semantic_predicate(gpu_search(_,_,_)),
    format('  PASS: Guard idempotent~n').

test_unknown_target :-
    format('--- Unknown Target ---~n'),
    % A target with no dispatch clause should fail gracefully
    (   compile_semantic_call(elixir, gpu_search(q, 5, _), [], _)
    ->  format('  FAIL: Should not have succeeded for unknown target~n')
    ;   format('  PASS: Correctly failed for unknown target~n')
    ).

% ============================================================================
% FUZZY LOGIC TESTS
% ============================================================================

% Minimal Go fuzzy dispatch (mirrors go_target.pl)
semantic_compiler:fuzzy_dispatch(go, f_and(Terms, _Result), Code) :-
    go_product_terms(Terms, TermCode),
    format(string(Code), '\tresult := 1.0\n~w', [TermCode]).

semantic_compiler:fuzzy_dispatch(go, f_or(Terms, _Result), Code) :-
    go_complement_terms(Terms, TermCode),
    format(string(Code), '\tcomplement := 1.0\n~w\tresult := 1 - complement\n', [TermCode]).

semantic_compiler:fuzzy_dispatch(go, f_not(Score, _Result), Code) :-
    format(string(Code), '\tresult := 1 - ~w\n', [Score]).

% Minimal Python fuzzy dispatch (mirrors python_fuzzy_target.pl)
semantic_compiler:fuzzy_dispatch(python, f_and(Terms, _Result), Code) :-
    py_weighted_terms(Terms, PyTerms),
    format(string(Code), '    result = f_and(~w, _term_scores)\n', [PyTerms]).

semantic_compiler:fuzzy_dispatch(python, f_or(Terms, _Result), Code) :-
    py_weighted_terms(Terms, PyTerms),
    format(string(Code), '    result = f_or(~w, _term_scores)\n', [PyTerms]).

% ---- Inline test helpers ----

go_product_terms([], '').
go_product_terms([w(Term, Weight)|Rest], Code) :-
    go_product_terms(Rest, RestCode),
    format(string(Line), '\tresult *= ~w * termScores["~w"]\n', [Weight, Term]),
    string_concat(Line, RestCode, Code).

go_complement_terms([], '').
go_complement_terms([w(Term, Weight)|Rest], Code) :-
    go_complement_terms(Rest, RestCode),
    format(string(Line), '\tcomplement *= (1 - ~w * termScores["~w"])\n', [Weight, Term]),
    string_concat(Line, RestCode, Code).

py_weighted_terms(Terms, PyTerms) :-
    maplist(py_weighted_term, Terms, Strs),
    atomic_list_concat(Strs, ', ', Joined),
    format(string(PyTerms), '[~w]', [Joined]).
py_weighted_term(w(T, W), S) :- format(string(S), '("~w", ~w)', [T, W]).

% ---- Fuzzy tests ----

test_fuzzy_recognition :-
    format('--- Fuzzy Recognition ---~n'),
    (   is_fuzzy_predicate(f_and([w(a,1)], _))
    ->  format('  PASS: f_and recognized~n')
    ;   format('  FAIL: f_and not recognized~n')
    ),
    (   is_fuzzy_predicate(f_not(0.5, _))
    ->  format('  PASS: f_not recognized~n')
    ;   format('  FAIL: f_not not recognized~n')
    ),
    (   is_fuzzy_predicate(foo(1,2))
    ->  format('  FAIL: foo should not be fuzzy~n')
    ;   format('  PASS: foo correctly rejected~n')
    ).

test_go_fuzzy_and :-
    format('--- Go Fuzzy AND ---~n'),
    compile_fuzzy_call(go, f_and([w(bash, 0.9), w(shell, 0.5)], _), Code),
    assert_contains(Code, "result := 1.0", "product identity"),
    assert_contains(Code, "0.9 * termScores[\"bash\"]", "bash weighted term"),
    assert_contains(Code, "0.5 * termScores[\"shell\"]", "shell weighted term").

test_go_fuzzy_or :-
    format('--- Go Fuzzy OR ---~n'),
    compile_fuzzy_call(go, f_or([w(bash, 0.9), w(shell, 0.5)], _), Code),
    assert_contains(Code, "complement := 1.0", "complement identity"),
    assert_contains(Code, "1 - complement", "probabilistic sum").

test_go_fuzzy_not :-
    format('--- Go Fuzzy NOT ---~n'),
    compile_fuzzy_call(go, f_not(0.3, _), Code),
    assert_contains(Code, "1 - 0.3", "complement").

test_python_fuzzy_and :-
    format('--- Python Fuzzy AND ---~n'),
    compile_fuzzy_call(python, f_and([w(bash, 0.9), w(shell, 0.5)], _), Code),
    assert_contains(Code, "f_and(", "f_and call"),
    assert_contains(Code, "_term_scores", "term scores dict").

test_python_fuzzy_or :-
    format('--- Python Fuzzy OR ---~n'),
    compile_fuzzy_call(python, f_or([w(bash, 0.9)], _), Code),
    assert_contains(Code, "f_or(", "f_or call").

% ---- Helpers ----

assert_contains(String, Substring, Label) :-
    (   sub_string(String, _, _, _, Substring)
    ->  format('  PASS: ~w found~n', [Label])
    ;   format('  FAIL: ~w missing in output~n', [Label]),
        format('  Got: ~w~n', [String])
    ).

test_all :-
    test_go_dispatch,
    test_python_dispatch,
    test_csharp_dispatch,
    test_fallback_dispatch,
    test_guard,
    test_unknown_target,
    test_fuzzy_recognition,
    test_go_fuzzy_and,
    test_go_fuzzy_or,
    test_go_fuzzy_not,
    test_python_fuzzy_and,
    test_python_fuzzy_or.

:- initialization(test_all, main).
