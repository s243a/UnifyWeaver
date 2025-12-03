:- module(test_go_semantic_compilation, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/targets/go_target).

run_tests :-
    run_tests([go_semantic_compilation]).

:- begin_tests(go_semantic_compilation).

test(go_semantic_search) :-
    retractall(user:search_go(_)),
    assertz((user:search_go(Q) :- semantic_search(Q, 5, _))),
    
    compile_predicate_to_go(user:search_go/1, [], Code),
    
    % Check imports
    sub_string(Code, _, _, _, "unifyweaver/targets/go_runtime/search"),
    sub_string(Code, _, _, _, "unifyweaver/targets/go_runtime/embedder"),
    
    % Check logic
    sub_string(Code, _, _, _, "storage.NewStore(\"data.db\")"),
    sub_string(Code, _, _, _, "embedder.NewHugotEmbedder"),
    sub_string(Code, _, _, _, "emb.Embed"),
    sub_string(Code, _, _, _, "search.Search(store, qVec, 5)"),
    
    retractall(user:search_go(_)).

test(go_crawler_run) :-
    retractall(user:run_crawl_const),
    assertz((user:run_crawl_const :- crawler_run(['http://example.com'], 3))),
    
    compile_predicate_to_go(user:run_crawl_const/0, [], Code),
    
    % Check imports
    sub_string(Code, _, _, _, "unifyweaver/targets/go_runtime/crawler"),
    
    % Check logic
    sub_string(Code, _, _, _, "crawler.NewCrawler"),
    sub_string(Code, _, _, _, "craw.Crawl([]string{\"http://example.com\"}, int(3))"),
    
    retractall(user:run_crawl_const).

:- end_tests(go_semantic_compilation).

