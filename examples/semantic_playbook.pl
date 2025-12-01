:- module(semantic_playbook, [main/0]).
:- use_module('../src/unifyweaver/targets/python_target').

% Define indexing logic
index_pt(_) :-
    % Use the actual RDF file path
    File = 'context/PT/pearltrees_export.rdf',
    crawler_run([File], 1).

% Define search logic (generic)
% This predicate expects Query to be passed, but our compiler 
% currently compiles static logic or reads from stdin.
% To make a dynamic search tool, we'd read Query from stdin.
% For this playbook, we'll compile a hardcoded query for demonstration.

% Helper to compile the indexer
compile_indexer :-
    format('Compiling indexer...~n', []),
    compile_predicate_to_python(semantic_playbook:index_pt/1, [mode(procedural)], Code),
    setup_call_cleanup(
        open('run_index.py', write, S),
        write(S, Code),
        close(S)
    ),
    format('Generated run_index.py~n', []).

% Helper to compile a searcher for a specific topic
compile_searcher(Topic) :-
    format('Compiling searcher for "~w"...~n', [Topic]),
    % Dynamic clause generation
    Term = (search_topic(Results) :- semantic_search(Topic, 5, Results)),
    assertz(user:Term),
    compile_predicate_to_python(user:search_topic/1, [mode(procedural)], Code),
    setup_call_cleanup(
        open('run_search.py', write, S),
        write(S, Code),
        close(S)
    ),
    retractall(user:search_topic(_)),
    format('Generated run_search.py~n', []).

main :-
    compile_indexer,
    compile_searcher('physics').
