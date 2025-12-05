:- module(cpu_rag_playbook, [main/0]).
:- use_module('../src/unifyweaver/targets/python_target').

% 1. Indexing (CPU Mode: No Embeddings)
index_data(_) :-
    File = 'context/PT/pearltrees_export.rdf',
    % Crawl with embedding=false
    crawler_run([File], 1, [embedding(false)]).

% 2. RAG Search (CPU Mode: Text Search + Graph)
answer_question(Question, Answer) :-
    % Use 'text' mode to anchor search via SQL LIKE
    graph_search(Question, 3, 1, [mode(text)], Context),
    
    Prompt = 'Answer the question using the provided context.',
    llm_ask(Prompt, Context, Answer).

% Compilation Helpers
compile_indexer :-
    format('Compiling CPU indexer...~n', []),
    compile_predicate_to_python(cpu_rag_playbook:index_data/1, [mode(procedural)], Code),
    setup_call_cleanup(
        open('run_cpu_index.py', write, S),
        write(S, Code),
        close(S)
    ),
    format('Generated run_cpu_index.py~n', []).

compile_agent(Question) :-
    format('Compiling CPU agent for: "~w"...~n', [Question]),
    Term = (ask_cpu(Ans) :- answer_question(Question, Ans)),
    assertz(user:Term),
    compile_predicate_to_python(user:ask_cpu/1, [mode(procedural)], Code),
    setup_call_cleanup(
        open('run_cpu_agent.py', write, S),
        write(S, Code),
        close(S)
    ),
    retractall(user:ask_cpu(_)),
    format('Generated run_cpu_agent.py~n', []).

main :-
    compile_indexer,
    % Question: "What is Labomedia?" (Should be found via text search)
    compile_agent('Labomedia').
