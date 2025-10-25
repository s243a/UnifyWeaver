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

## 3. Tools & Their Relative Costs

- `local_indexer`: (Cost: Free, Speed: Fast) - Indexes one or more files locally using sentence-transformer embeddings.
- `gemini_reranker`: (Cost: Medium, Speed: Medium) - Re-ranks a list of text snippets for relevance.
- `gemini_full_retrieval`: (Cost: High, Speed: High) - Performs semantic search on a **single** large document.
- `grep`: (Cost: Free, Speed: Very Fast) - Simple keyword search.
- `git_status`: (Cost: Free, Speed: Very Fast) - Checks for modified files.

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
Your file index may be persistent but can become stale. **Strategy:** Before a search, consider running `git_status` to check for modified files. If a small number of promising files have been modified, re-index only those files. If many files have changed, you may need to perform a full re-index or rely on non-index-based tools like `grep` for your initial triage.

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
