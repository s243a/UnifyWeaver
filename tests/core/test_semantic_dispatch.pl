:- encoding(utf8).
% test_semantic_dispatch.pl - Test for generic semantic search dispatch
%
% Run from repo root:
%   swipl tests/core/test_semantic_dispatch.pl

:- use_module('../../src/unifyweaver/core/semantic_compiler').
:- use_module('../../src/unifyweaver/core/input_source').

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

% Minimal Rust fuzzy dispatch (mirrors rust_target.pl)
semantic_compiler:fuzzy_dispatch(rust, f_and(Terms, _Result), Code) :-
    rust_product_terms(Terms, TermCode),
    format(string(Code), '    let mut result: f64 = 1.0;\n~w', [TermCode]).

semantic_compiler:fuzzy_dispatch(rust, f_or(Terms, _Result), Code) :-
    rust_complement_terms(Terms, TermCode),
    format(string(Code), '    let mut complement: f64 = 1.0;\n~w    let result = 1.0 - complement;\n', [TermCode]).

semantic_compiler:fuzzy_dispatch(rust, f_not(Score, _Result), Code) :-
    format(string(Code), '    let result = 1.0 - ~w;\n', [Score]).

% Minimal C# fuzzy dispatch (mirrors csharp_target.pl)
semantic_compiler:fuzzy_dispatch(csharp, f_and(Terms, _Result), Code) :-
    csharp_product_terms(Terms, TermCode),
    format(string(Code), '    double result = 1.0;\n~w', [TermCode]).

semantic_compiler:fuzzy_dispatch(csharp, f_or(Terms, _Result), Code) :-
    csharp_complement_terms(Terms, TermCode),
    format(string(Code), '    double complement = 1.0;\n~w    double result = 1.0 - complement;\n', [TermCode]).

semantic_compiler:fuzzy_dispatch(csharp, f_not(Score, _Result), Code) :-
    format(string(Code), '    double result = 1.0 - ~w;\n', [Score]).

% Minimal Go batch fuzzy dispatch (mirrors go_target.pl batch ops)
semantic_compiler:fuzzy_dispatch(go, f_and_batch(Terms, _ScoresBatch, _Result), Code) :-
    go_batch_product_terms(Terms, TermCode),
    format(string(Code), '\tn := batchLen(termScoresBatch)\n\tresult := make([]float64, n)\n\tfor i := range result { result[i] = 1.0 }\n~w', [TermCode]).

semantic_compiler:fuzzy_dispatch(go, f_or_batch(Terms, _ScoresBatch, _Result), Code) :-
    go_batch_complement_terms(Terms, TermCode),
    format(string(Code), '\tn := batchLen(termScoresBatch)\n\tcomplement := make([]float64, n)\n\tfor i := range complement { complement[i] = 1.0 }\n~w\tresult := make([]float64, n)\n\tfor i := range result { result[i] = 1 - complement[i] }\n', [TermCode]).

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

rust_product_terms([], '').
rust_product_terms([w(Term, Weight)|Rest], Code) :-
    rust_product_terms(Rest, RestCode),
    format(string(Line), '    result *= ~w * term_scores.get("~w").copied().unwrap_or(0.5);\n', [Weight, Term]),
    string_concat(Line, RestCode, Code).

rust_complement_terms([], '').
rust_complement_terms([w(Term, Weight)|Rest], Code) :-
    rust_complement_terms(Rest, RestCode),
    format(string(Line), '    complement *= 1.0 - ~w * term_scores.get("~w").copied().unwrap_or(0.5);\n', [Weight, Term]),
    string_concat(Line, RestCode, Code).

csharp_product_terms([], '').
csharp_product_terms([w(Term, Weight)|Rest], Code) :-
    csharp_product_terms(Rest, RestCode),
    format(string(Line), '    result *= ~w * (termScores.TryGetValue("~w", out var s_~w) ? s_~w : 0.5);\n', [Weight, Term, Term, Term]),
    string_concat(Line, RestCode, Code).

csharp_complement_terms([], '').
csharp_complement_terms([w(Term, Weight)|Rest], Code) :-
    csharp_complement_terms(Rest, RestCode),
    format(string(Line), '    complement *= 1.0 - ~w * (termScores.TryGetValue("~w", out var s_~w) ? s_~w : 0.5);\n', [Weight, Term, Term, Term]),
    string_concat(Line, RestCode, Code).

go_batch_product_terms([], '').
go_batch_product_terms([w(Term, Weight)|Rest], Code) :-
    go_batch_product_terms(Rest, RestCode),
    format(string(Line), '\tif scores, ok := termScoresBatch["~w"]; ok {\n\t\tfor i := range result { result[i] *= ~w * scores[i] }\n\t}\n', [Term, Weight]),
    string_concat(Line, RestCode, Code).

go_batch_complement_terms([], '').
go_batch_complement_terms([w(Term, Weight)|Rest], Code) :-
    go_batch_complement_terms(Rest, RestCode),
    format(string(Line), '\tif scores, ok := termScoresBatch["~w"]; ok {\n\t\tfor i := range complement { complement[i] *= (1 - ~w * scores[i]) }\n\t}\n', [Term, Weight]),
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

test_rust_fuzzy_and :-
    format('--- Rust Fuzzy AND ---~n'),
    compile_fuzzy_call(rust, f_and([w(bash, 0.9), w(shell, 0.5)], _), Code),
    assert_contains(Code, "let mut result: f64 = 1.0", "product identity"),
    assert_contains(Code, "term_scores.get(\"bash\")", "HashMap lookup"),
    assert_contains(Code, "unwrap_or(0.5)", "default score fallback").

test_rust_fuzzy_or :-
    format('--- Rust Fuzzy OR ---~n'),
    compile_fuzzy_call(rust, f_or([w(bash, 0.9)], _), Code),
    assert_contains(Code, "let mut complement: f64 = 1.0", "complement identity"),
    assert_contains(Code, "1.0 - complement", "probabilistic sum").

test_rust_fuzzy_not :-
    format('--- Rust Fuzzy NOT ---~n'),
    compile_fuzzy_call(rust, f_not(0.3, _), Code),
    assert_contains(Code, "1.0 - 0.3", "complement").

test_csharp_fuzzy_and :-
    format('--- C# Fuzzy AND ---~n'),
    compile_fuzzy_call(csharp, f_and([w(bash, 0.9), w(shell, 0.5)], _), Code),
    assert_contains(Code, "double result = 1.0", "product identity"),
    assert_contains(Code, "TryGetValue(\"bash\"", "Dictionary lookup"),
    assert_contains(Code, "0.5)", "default score fallback").

test_csharp_fuzzy_or :-
    format('--- C# Fuzzy OR ---~n'),
    compile_fuzzy_call(csharp, f_or([w(bash, 0.9)], _), Code),
    assert_contains(Code, "double complement = 1.0", "complement identity"),
    assert_contains(Code, "1.0 - complement", "probabilistic sum").

test_csharp_fuzzy_not :-
    format('--- C# Fuzzy NOT ---~n'),
    compile_fuzzy_call(csharp, f_not(0.3, _), Code),
    assert_contains(Code, "1.0 - 0.3", "complement").

test_go_batch_and :-
    format('--- Go Batch AND ---~n'),
    compile_fuzzy_call(go, f_and_batch([w(bash, 0.9)], scores, _), Code),
    assert_contains(Code, "batchLen(termScoresBatch)", "batch length"),
    assert_contains(Code, "make([]float64, n)", "slice allocation"),
    assert_contains(Code, "termScoresBatch[\"bash\"]", "batch lookup").

test_go_batch_or :-
    format('--- Go Batch OR ---~n'),
    compile_fuzzy_call(go, f_or_batch([w(bash, 0.9)], scores, _), Code),
    assert_contains(Code, "complement[i] = 1.0", "complement init"),
    assert_contains(Code, "1 - complement[i]", "element-wise sum").

% Minimal Rust batch fuzzy dispatch
semantic_compiler:fuzzy_dispatch(rust, f_and_batch(Terms, _SB, _R), Code) :-
    rust_batch_product_terms(Terms, TC),
    format(string(Code), '    let mut result: Vec<f64> = vec![1.0; n];\n~w', [TC]).

semantic_compiler:fuzzy_dispatch(rust, f_or_batch(Terms, _SB, _R), Code) :-
    rust_batch_complement_terms(Terms, TC),
    format(string(Code), '    let mut complement: Vec<f64> = vec![1.0; n];\n~w    let result: Vec<f64> = complement.iter().map(|c| 1.0 - c).collect();\n', [TC]).

% Minimal C# batch fuzzy dispatch
semantic_compiler:fuzzy_dispatch(csharp, f_and_batch(Terms, _SB, _R), Code) :-
    csharp_batch_product_terms(Terms, TC),
    format(string(Code), '    double[] result = Enumerable.Repeat(1.0, n).ToArray();\n~w', [TC]).

semantic_compiler:fuzzy_dispatch(csharp, f_or_batch(Terms, _SB, _R), Code) :-
    csharp_batch_complement_terms(Terms, TC),
    format(string(Code), '    double[] complement = Enumerable.Repeat(1.0, n).ToArray();\n~w    double[] result = complement.Select(c => 1.0 - c).ToArray();\n', [TC]).

rust_batch_product_terms([], '').
rust_batch_product_terms([w(Term, Weight)|Rest], Code) :-
    rust_batch_product_terms(Rest, RestCode),
    format(string(Line), '    if let Some(scores) = term_scores_batch.get("~w") {\n        for (r, s) in result.iter_mut().zip(scores.iter()) { *r *= ~w * s; }\n    }\n', [Term, Weight]),
    string_concat(Line, RestCode, Code).

rust_batch_complement_terms([], '').
rust_batch_complement_terms([w(Term, Weight)|Rest], Code) :-
    rust_batch_complement_terms(Rest, RestCode),
    format(string(Line), '    if let Some(scores) = term_scores_batch.get("~w") {\n        for (c, s) in complement.iter_mut().zip(scores.iter()) { *c *= 1.0 - ~w * s; }\n    }\n', [Term, Weight]),
    string_concat(Line, RestCode, Code).

csharp_batch_product_terms([], '').
csharp_batch_product_terms([w(Term, Weight)|Rest], Code) :-
    csharp_batch_product_terms(Rest, RestCode),
    format(string(Line), '    if (termScoresBatch.TryGetValue("~w", out var scores_~w)) {\n        for (int i = 0; i < result.Length; i++) result[i] *= ~w * scores_~w[i];\n    }\n', [Term, Term, Weight, Term]),
    string_concat(Line, RestCode, Code).

csharp_batch_complement_terms([], '').
csharp_batch_complement_terms([w(Term, Weight)|Rest], Code) :-
    csharp_batch_complement_terms(Rest, RestCode),
    format(string(Line), '    if (termScoresBatch.TryGetValue("~w", out var scores_~w)) {\n        for (int i = 0; i < complement.Length; i++) complement[i] *= 1.0 - ~w * scores_~w[i];\n    }\n', [Term, Term, Weight, Term]),
    string_concat(Line, RestCode, Code).

% ---- New Phase 3 tests ----

test_rust_batch_and :-
    format('--- Rust Batch AND ---~n'),
    compile_fuzzy_call(rust, f_and_batch([w(bash, 0.9)], scores, _), Code),
    assert_contains(Code, "vec![1.0; n]", "batch init"),
    assert_contains(Code, "term_scores_batch.get(\"bash\")", "batch lookup"),
    assert_contains(Code, "iter_mut().zip", "iterator zip").

test_rust_batch_or :-
    format('--- Rust Batch OR ---~n'),
    compile_fuzzy_call(rust, f_or_batch([w(bash, 0.9)], scores, _), Code),
    assert_contains(Code, "1.0 - c", "complement map").

test_csharp_batch_and :-
    format('--- C# Batch AND ---~n'),
    compile_fuzzy_call(csharp, f_and_batch([w(bash, 0.9)], scores, _), Code),
    assert_contains(Code, "Enumerable.Repeat(1.0, n)", "batch init"),
    assert_contains(Code, "TryGetValue(\"bash\"", "batch lookup").

test_csharp_batch_or :-
    format('--- C# Batch OR ---~n'),
    compile_fuzzy_call(csharp, f_or_batch([w(bash, 0.9)], scores, _), Code),
    assert_contains(Code, "Select(c => 1.0 - c)", "complement LINQ").

test_semantic_search_options :-
    format('--- Semantic Search /4 Options ---~n'),
    % Register a provider for test_search/3
    declare_semantic_provider(test_search/3, [
        targets([target(go, [provider(hugot), model('base-model'), device(cpu)])])
    ]),
    % Compile with inline options that override model
    compile_semantic_call(go,
        test_search(query, 5, _, [model('override-model'), threshold(0.7)]),
        [], Code),
    assert_contains(Code, "override-model", "model override from options").

test_vector_source :-
    format('--- Vector DB Source ---~n'),
    resolve_vector_source([input(vector_db("my_index.db", sqlite))], Config),
    Config = vector_db("my_index.db", sqlite),
    format('  PASS: explicit vector_db resolved~n'),
    resolve_vector_source([index("other.db")], Config2),
    Config2 = vector_db("other.db", auto),
    format('  PASS: index shorthand resolved~n'),
    resolve_vector_source([], Config3),
    Config3 = vector_db("data.db", auto),
    format('  PASS: default vector_db resolved~n').

test_vector_init_code :-
    format('--- Vector DB Init Code ---~n'),
    vector_db_init_code(python, vector_db("search.db", _), Code),
    assert_contains(Code, "sqlite3.connect(\"search.db\")", "Python DB init"),
    vector_db_init_code(go, vector_db("search.db", _), GoCode),
    assert_contains(GoCode, "storage.NewStore(\"search.db\")", "Go DB init"),
    vector_db_init_code(rust, vector_db("search.db", _), RustCode),
    assert_contains(RustCode, "Store::open(\"search.db\")", "Rust DB init"),
    vector_db_init_code(csharp, vector_db("search.db", _), CsCode),
    assert_contains(CsCode, "VectorStore(\"search.db\")", "C# DB init").

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
    test_python_fuzzy_or,
    test_rust_fuzzy_and,
    test_rust_fuzzy_or,
    test_rust_fuzzy_not,
    test_csharp_fuzzy_and,
    test_csharp_fuzzy_or,
    test_csharp_fuzzy_not,
    test_go_batch_and,
    test_go_batch_or,
    test_rust_batch_and,
    test_rust_batch_or,
    test_csharp_batch_and,
    test_csharp_batch_or,
    test_semantic_search_options,
    test_vector_source,
    test_vector_init_code.

:- initialization(test_all, main).
