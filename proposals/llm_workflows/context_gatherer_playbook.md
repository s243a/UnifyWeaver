---
name: multi_file_context_gatherer
version: 4.0.0
description: Provides cost-aware strategies for gathering context from multiple files.
---

# Playbook: Multi-File Context Gatherer

## 1. Goal

To intelligently search across **multiple files** to find the most relevant context for a final prompt, while optimizing for quality, speed, and API cost.

## 2. Your Role as Economic Strategist

You are a high-level AI agent. Your task is to devise and execute a search plan across a set of files. You must choose the best strategy for each file or the set as a whole, heuristically optimizing against the guidance criteria below. Before you begin, you must state your chosen plan and justify it.

## 3. Tools & Their Interface Specifications

### `local_indexer`
- **Cost:** {{ LOCAL_INDEXER_COST | default: 0.0 }} USD per call (free, local compute)
- **Latency:** {{ LOCAL_INDEXER_LATENCY_MS | default: 500 }}ms per file
- **Interface:**
  ```bash
  local_indexer --query "search query" --files file1.txt file2.txt --top-k 25
  ```
- **Output:** JSON array of `{"file": "path", "chunk": "text", "score": 0.0-1.0}`
- **Error Codes:** 0 (success), 1 (no files found), 2 (index missing)

### `gemini_reranker`
- **Cost:** {{ GEMINI_FLASH_COST | default: 0.002 }} USD per call
- **Latency:** {{ GEMINI_RERANKER_LATENCY_MS | default: 2500 }}ms per call
- **Interface:**
  ```bash
  gemini_reranker --query "search query" --chunks chunk1.txt chunk2.txt --top-k 10
  ```
- **Output:** JSON array of `{"chunk": "text", "confidence": 0.0-1.0, "reasoning": "explanation"}`
- **Error Codes:** 0 (success), 3 (API quota exceeded), 4 (network error)

### `gemini_full_retrieval`
- **Cost:** {{ GEMINI_FLASH_COST | default: 0.002 }} USD per call
- **Latency:** {{ GEMINI_FULL_RETRIEVAL_LATENCY_MS | default: 3000 }}ms per file
- **Interface:**
  ```bash
  gemini_full_retrieval --query "search query" --file large_doc.txt --top-k 5
  ```
- **Output:** JSON array of `{"chunk": "text", "confidence": 0.0-1.0}`
- **Error Codes:** 0 (success), 3 (API quota exceeded), 5 (file too large)

### `grep`
- **Cost:** {{ GREP_COST | default: 0.0 }} USD per call (free)
- **Latency:** {{ GREP_LATENCY_MS | default: 100 }}ms per file
- **Interface:**
  ```bash
  grep -r "keyword" file1.txt file2.txt
  ```
- **Output:** Line-based matches
- **Error Codes:** 0 (matches found), 1 (no matches)

### `git_status`
- **Cost:** {{ GIT_STATUS_COST | default: 0.0 }} USD per call (free)
- **Latency:** {{ GIT_STATUS_LATENCY_MS | default: 50 }}ms
- **Interface:**
  ```bash
  git status --porcelain
  ```
- **Output:** List of modified files
- **Error Codes:** 0 (success), 128 (not a git repo)

## 4. Guidance Criteria for Optimization

### Triage First, Search Second
Do not immediately perform a deep search on all files. Your first step should always be a fast, cheap "triage" to identify a smaller subset of promising files. **Strategy:** Start with a broad `grep` or `local_indexer` query across all files to rank which ones are most likely to contain relevant information.

### Budget Allocation
Allocate your finite budget of time and API cost wisely. **Strategy:** Spend your budget on the most promising files identified during triage. It is better to do a deep search on 2 relevant files than a shallow search on 10 irrelevant ones.

### Single vs. Multiple Files
- **If triage narrows the search to a single file:** It is often acceptable to use a more expensive strategy on that one file (e.g., `gemini_full_retrieval`).
- **If triage identifies multiple promising files:** Use more cost-effective methods. Run `local_indexer` on each, collect the top results, and then use `gemini_reranker` on the combined set.

### Boundary Redefinition
The output of `grep` or a local indexer may have poor contextual boundaries. **Strategy:** After getting initial "hits" from a cheap tool, feed a slightly larger snippet of text around those hits to `gemini_reranker` and ask it to "redefine the boundaries of the most relevant passage."

### Index & Knowledge Management
Your file index may be persistent but can become stale. **Strategy:** Before a search, consider running `git_status` to check for modified files. If a small number of promising files have been modified (≤ {{ SMALL_CHANGE_THRESHOLD | default: 5 }} files), re-index only those files. If many files have changed (> {{ LARGE_CHANGE_THRESHOLD | default: 20 }} files), you may need to perform a full re-index or rely on non-index-based tools like `grep` for your initial triage.

## 5. Strategic Options (The Playbook)

### Strategy 1: Quick Triage
*   **Description:** A fast, broad search to identify promising files.
*   **Best for:** The mandatory first step in any multi-file search.
*   **Plan (`todo`):
    1.  Use `grep` for keyword searches or `local_indexer --query ...` for semantic searches across all target files.
    2.  Analyze the results to create a short list of 2-3 most promising files.

### Strategy 2: Balanced Deep Dive
*   **Description:** A powerful, cost-effective deep search on a small set of promising files.
*   **Best for:** The main analysis phase after triage.
*   **Plan (`todo`):
    1.  For each promising file, use `local_indexer` to get the top 25 candidate chunks.
    2.  Combine all candidate chunks from all promising files into a single list.
    3.  Feed the combined list to the `gemini_reranker`.
    4.  Use the top 5-10 re-ranked results as the final context.

### Strategy 3: Single-File Precision
*   **Description:** A high-accuracy, high-cost search focused on a single, critical file.
*   **Best for:** When triage points to one overwhelmingly important file.
*   **Plan (`todo`):
    1.  Use `gemini_full_retrieval` on the single target file.
    2.  Alternatively, use the "Boundary Redefinition" guidance on hits from `local_indexer`.

## 6. Declarative Strategy Selection Logic

This section provides a formal, machine-readable representation of strategy selection. An LLM can use this for rigorous reasoning about which workflow to execute.

```pseudocode_prolog
% Strategy selection predicates
strategy_selection(context_gatherer, Conditions) :-
    Conditions = [
        % Always start with triage
        initial_step(quick_triage),

        % After triage, select main strategy based on file count
        condition(
            promising_file_count(Count),
            [Count == 1],
            use_strategy(single_file_precision)
        ),
        condition(
            promising_file_count(Count),
            [Count >= 2, Count =< {{ MAX_BALANCED_FILES | default: 5 }}],
            use_strategy(balanced_deep_dive)
        ),
        condition(
            promising_file_count(Count),
            [Count > {{ MAX_BALANCED_FILES | default: 5 }}],
            use_strategy(iterative_triage)  % Re-triage to narrow down further
        ),

        % Budget constraints
        condition(
            remaining_budget(Budget),
            [Budget < {{ MIN_GEMINI_BUDGET | default: 0.01 }}],
            fallback_strategy(local_only)  % Use only free tools
        ),

        % Index freshness check
        condition(
            modified_file_count(ModCount),
            [ModCount =< {{ SMALL_CHANGE_THRESHOLD | default: 5 }}],
            action(reindex_modified_files)
        ),
        condition(
            modified_file_count(ModCount),
            [ModCount > {{ LARGE_CHANGE_THRESHOLD | default: 20 }}],
            action(skip_index, use_grep_triage)
        )
    ].

% Execution plan for each strategy
execution_plan(quick_triage, [
    step(check_index_freshness, [
        tool(git_status),
        output(to_variable('ModifiedFiles'))
    ]),
    step(triage_search, [
        tool(local_indexer),
        args(['--query', from_variable('Query'), '--files', from_variable('AllFiles'), '--top-k', 25]),
        output(to_variable('TriageResults'))
    ]),
    step(analyze_triage, [
        description('Count files with score > 0.5'),
        input(from_variable('TriageResults')),
        output(to_variable('PromisingFileCount'))
    ])
]).

execution_plan(balanced_deep_dive, [
    step(deep_index, [
        tool(local_indexer),
        args(['--query', from_variable('Query'), '--files', from_variable('PromisingFiles'), '--top-k', 25]),
        output(to_variable('DeepResults'))
    ]),
    step(rerank, [
        tool(gemini_reranker),
        args(['--query', from_variable('Query'), '--chunks', from_variable('DeepResults'), '--top-k', 10]),
        output(to_variable('FinalContext'))
    ])
]).

execution_plan(single_file_precision, [
    step(full_retrieval, [
        tool(gemini_full_retrieval),
        args(['--query', from_variable('Query'), '--file', from_variable('TargetFile'), '--top-k', 5]),
        output(to_variable('FinalContext'))
    ])
]).

% Error handling strategies
error_recovery(tool_failure(gemini_reranker), [
    action(log_error),
    fallback(use_local_indexer_results)
]).

error_recovery(tool_failure(gemini_full_retrieval), [
    action(log_error),
    fallback(use_strategy(balanced_deep_dive))
]).

error_recovery(quota_exceeded, [
    action(log_error),
    fallback(local_only_mode)
]).
```

## 7. Configuration Template Variables

These variables can be set during playbook compilation or at runtime:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_INDEXER_COST` | 0.0 | Cost per local indexer call (USD) |
| `LOCAL_INDEXER_LATENCY_MS` | 500 | Average latency (milliseconds) |
| `GEMINI_FLASH_COST` | 0.002 | Cost per Gemini Flash call (USD) |
| `GEMINI_RERANKER_LATENCY_MS` | 2500 | Average reranker latency (ms) |
| `GEMINI_FULL_RETRIEVAL_LATENCY_MS` | 3000 | Average full retrieval latency (ms) |
| `SMALL_CHANGE_THRESHOLD` | 5 | Max files for incremental reindex |
| `LARGE_CHANGE_THRESHOLD` | 20 | Min files to skip indexing |
| `MAX_BALANCED_FILES` | 5 | Max files for balanced deep dive |
| `MIN_GEMINI_BUDGET` | 0.01 | Min budget to use Gemini tools (USD) |
| `API_BUDGET_TOTAL` | 0.10 | Total API budget per query (USD) |
