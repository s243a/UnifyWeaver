:- module(test_python_llm, [run_tests/0]).

:- use_module(library(plunit)).
:- use_module(src/unifyweaver/targets/python_target).

run_tests :-
    run_tests([python_llm]).

:- begin_tests(python_llm).

test(llm_compilation) :-
    retractall(user:ask_gemini(_)),
    assertz((user:ask_gemini(Ans) :- llm_ask("Hello", "Context", Ans))),
    
    compile_predicate_to_python(user:ask_gemini/1, [mode(procedural)], Code),
    
    % Check logic
    sub_string(Code, _, _, _, "_get_runtime().llm.ask"),
    
    % Check runtime injection
    sub_string(Code, _, _, _, "class LLMProvider"),
    
    retractall(user:ask_gemini(_)).

test(chunking_compilation) :-
    retractall(user:do_chunk(_)),
    assertz((user:do_chunk(Chunks) :- chunk_text("Some text", Chunks))),
    
    compile_predicate_to_python(user:do_chunk/1, [mode(procedural)], Code),
    
    sub_string(Code, _, _, _, "_get_runtime().chunker.chunk"),
    sub_string(Code, _, _, _, "class HierarchicalChunker"),
    
    retractall(user:do_chunk(_)).

:- end_tests(python_llm).
