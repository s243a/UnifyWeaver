# Chapter 5: The Semantic Playbook

The **Semantic Playbook** pattern is the recommended way to structure UnifyWeaver applications. It clearly separates **Definition** (Prolog) from **Execution** (Python).

## 5.1 The Playbook Structure

A typical playbook (`semantic_playbook.pl`) has three sections:

1.  **Logic Definition**: The semantic rules (Indexing, Searching, Summarizing).
2.  **Compilation Helpers**: Predicates that generate the Python scripts.
3.  **Main Entry Point**: The orchestrator.

## 5.2 Example Walkthrough

Let's dissect `examples/semantic_playbook.pl`.

### Section 1: Logic

```prolog
% Indexing: Crawl the RDF file
index_pt(_) :-
    crawler_run(['data.rdf'], 1).

% RAG: Search and Summarize
summarize_topic(Topic, Summary) :-
    % 1. Graph Search (Vector + Links)
    graph_search(Topic, 3, 1, Context),
    
    % 2. LLM Synthesis
    Prompt = 'Summarize this context.',
    llm_ask(Prompt, Context, Summary).
```

### Section 2: Compilation Helpers

These predicates tell the compiler *what* to build.

```prolog
compile_summarizer(Topic) :-
    % Create a specialized query for the specific topic
    Term = (gen_summary(S) :- summarize_topic(Topic, S)),
    assertz(Term),
    
    % Compile it to a script
    compile_predicate_to_python(gen_summary/1, [mode(procedural)], Code),
    write_file('run_summary.py', Code).
```

### Section 3: Orchestration

```prolog
main :-
    compile_indexer,
    compile_searcher('hacktivism'),
    compile_summarizer('hacktivism').
```

## 5.3 Running the Playbook

1.  **Generate**: Run the Prolog playbook to create the Python agents.
    ```bash
    swipl -g main -t halt semantic_playbook.pl
    ```
2.  **Execute**: Run the generated Python scripts.
    ```bash
    # Index the data
    python3 run_index.py < inputs.jsonl
    
    # Run the RAG agent
    python3 run_summary.py < inputs.jsonl
    ```

This "Meta-Programming" approach is powerful. You can write a Prolog script that generates hundreds of specialized, optimized Python micro-services, each hard-coded for a specific task or query.

---

## Navigation

**←** [Previous: Chapter 4: Logic and Recursion in Python](04_logic_and_recursion) | [📖 Book 13: Semantic Search](README)
