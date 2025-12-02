:- module(semantic_playbook, [main/0]).
:- use_module('../src/unifyweaver/targets/python_target').

% Define indexing logic
index_pt(_) :-
    % Use the actual RDF file path
    File = 'context/PT/pearltrees_export.rdf',
    crawler_run([File], 1).

% Define search logic (generic)
search_topic(Results) :-
    semantic_search('physics', 5, Results).

% Define RAG logic
summarize_topic(Topic, Summary) :-
    graph_search(Topic, 3, 1, Results),
    Prompt = 'Summarize the following search results (including context from parents/children) regarding the topic.',
    llm_ask(Prompt, Results, Summary).

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

% Helper to compile summarizer
compile_summarizer(Topic) :-
    format('Compiling summarizer for "~w"...~n', [Topic]),
    % Inline the logic
    Prompt = 'Summarize the following search results (including context from parents/children) regarding the topic.',
    Body = (graph_search(Topic, 3, 1, Results), llm_ask(Prompt, Results, S)),
    
    Term = (gen_summary(S) :- Body),
    assertz(user:Term),
    compile_predicate_to_python(user:gen_summary/1, [mode(procedural)], Code),
    setup_call_cleanup(
        open('run_summary.py', write, S),
        write(S, Code),
        close(S)
    ),
    retractall(user:gen_summary(_)),
    format('Generated run_summary.py~n', []).

main :-
    compile_indexer,
    compile_searcher('hacktivism'),
    compile_summarizer('hacktivism').
