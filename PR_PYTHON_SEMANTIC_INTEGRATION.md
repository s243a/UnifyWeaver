# feat(python): Integrate Semantic Runtime into Compiler

## Summary
This PR completes the Python Semantic capability by updating the `python_target` compiler to recognize and translate semantic predicates (`semantic_search/3`, `crawler_run/2`) into calls to the new Runtime Library. It also implements **Runtime Injection**, inlining the runtime code (`crawler.py`, `importer.py`, etc.) directly into the generated script for standalone execution.

## Changes
- **`python_target.pl`**:
    - Added `translate_goal` clauses for:
        - `semantic_search(Query, TopK, Results)`
        - `crawler_run(SeedIds, MaxDepth)`
        - `upsert_object(Id, Type, Data)`
    - Implemented `semantic_runtime_helpers/1` which reads runtime source files and concatenates them into a `SemanticRuntime` singleton wrapper.
    - Updated `helpers/1` to inject this runtime code into generated scripts.
- **Tests**:
    - Added `tests/core/test_python_semantic_compilation.pl` verifying that semantic predicates are correctly compiled into Python code that invokes the runtime components.
- **Documentation**:
    - Updated `docs/PYTHON_TARGET.md` with "Semantic Predicates" section.

## Usage
```prolog
% Crawl physics-related trees
crawl_physics :-
    semantic_search('physics', 5, Seeds),
    crawler_run(Seeds, 2).
```

Compiles to Python:
```python
# ... Inlined Runtime Code ...

def _clause_0(v_0):
    # ...
    _get_runtime().searcher.search("physics", top_k=5)
    # ...
    _get_runtime().crawler.crawl(v_seeds, fetch_xml_func, max_depth=2)
```