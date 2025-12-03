:- module(test_python_semantic_compilation, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/targets/python_target).

run_tests :-
    run_tests([python_semantic]).

:- begin_tests(python_semantic).

test(semantic_search_compilation) :-
    % Define dummy predicate
    retractall(user:find_similar(_,_)),
    assertz((user:find_similar(Query, Result) :- semantic_search(Query, 5, Result))),
    
    compile_predicate_to_python(user:find_similar/2, [mode(procedural)], PyCode),
    
    % Check for runtime injection
    sub_string(PyCode, _, _, _, "class PtSearcher"),
    sub_string(PyCode, _, _, _, "class PtCrawler"),
    sub_string(PyCode, _, _, _, "_get_runtime().searcher.search"),
    
    retractall(user:find_similar(_,_)).

test(crawler_compilation) :-
    retractall(user:run_crawl(_)),
    assertz((user:run_crawl(Seeds) :- crawler_run(Seeds, 3))),
    
    compile_predicate_to_python(user:run_crawl/1, [mode(procedural)], PyCode),
    
    sub_string(PyCode, _, _, _, "_get_runtime().crawler.crawl"),
    
    retractall(user:run_crawl(_)).

:- end_tests(python_semantic).
