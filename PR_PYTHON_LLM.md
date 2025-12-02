# feat(python): LLM Integration and Hierarchical Chunking

## Summary
This PR enables UnifyWeaver agents to perform Generative AI tasks by integrating an `LLMProvider` (wrapping the Gemini CLI) and a `HierarchicalChunker` into the Python runtime. It adds new predicates for asking questions and chunking text documents.

## Changes
- **`python_target.pl`**:
    - Added `translate_goal` for `llm_ask(Prompt, Context, Response)` and `chunk_text(Text, Chunks)`.
    - Injects `llm.py` and `chunker.py` into generated scripts.
- **Runtime Library**:
    - `llm.py`: Wraps `gemini` CLI to send prompts.
    - `chunker.py`: Implements macro/micro chunking logic (ported from agentRag).
- **Tests**:
    - `tests/core/test_python_llm.pl`: Verifies compilation of LLM/Chunking predicates.

## Usage
```prolog
summarize_file(File, Summary) :-
    read_file(File, Content),
    llm_ask('Summarize this', Content, Summary).

chunk_file(File, Chunks) :-
    read_file(File, Content),
    chunk_text(Content, Chunks).
```
