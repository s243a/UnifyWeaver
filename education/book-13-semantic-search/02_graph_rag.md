# Chapter 2: Graph Retrieval-Augmented Generation (Graph RAG)

**Retrieval-Augmented Generation (RAG)** is a technique to ground LLM responses in your own data. Standard RAG retrieves documents based on **Vector Similarity**. 

**Graph RAG** goes a step further: it retrieves documents based on **Relationships**.

## 2.1 The Problem with "Flat" Search

Imagine you are searching a knowledge base for "Project Apollo".
*   **Vector Search** finds documents containing "Apollo", "Moon", "NASA".
*   **Missing Context**: It might miss the "Budget Report 1968" because it doesn't explicitly mention "Apollo" often, even though it's the *parent* folder of the Apollo documents.

Relationships (Parent/Child, See Also, Citations) are critical for understanding context.

## 2.2 The UnifyWeaver Approach

UnifyWeaver's Python target implements a **Hybrid Graph RAG**:

1.  **Anchor**: Use Vector Search to find the most relevant "Entry Nodes" (e.g., the "Apollo 11" page).
2.  **Traverse**: Use the Graph (stored in SQLite `links` table) to find neighbors (Parents, Children).
3.  **Synthesize**: Feed the Anchor + Neighbors to the LLM.

### Predicates

*   `semantic_search(Query, TopK, Results)`: Standard Vector Search.
*   `graph_search(Query, TopK, Hops, Results)`: Graph RAG.

## 2.3 Implementation Guide

### Step 1: Ingestion with Links
The `PtCrawler` automatically extracts links from `rdf:resource` attributes (common in RDF/XML).

```prolog
index_data :-
    crawler_run(['data.rdf'], 1). % Crawl depth 1
```

This populates the `links` table in `data.db`:
| Source ID | Target ID |
|-----------|-----------|
| Child     | Parent    |
| Page A    | Page B    |

### Step 2: The Search Query
Use `graph_search/4` to retrieve the context.

```prolog
search_topic(Topic, Results) :-
    % Find top 3 matches, then expand 1 hop (parents/children)
    graph_search(Topic, 3, 1, Results).
```

### Step 3: Summarization
Pass the structured results to the LLM.

```prolog
summarize(Topic, Summary) :-
    graph_search(Topic, 3, 1, Context),
    Prompt = 'Summarize the topic using the provided context.',
    llm_ask(Prompt, Context, Summary).
```

## 2.4 Example: The "Hacktivism" Graph

In our `context/PT` dataset (Pearltrees export):
1.  **Query**: "Hacktivism"
2.  **Anchors**: Finds "Anonymous Culture", "Hacktivism" (Tree).
3.  **Graph Expansion**:
    *   Finds **Children**: "Labomedia", "Tetalab" (which might just be URLs without much text).
    *   Finds **Parents**: The main "Politics" tree.
4.  **Result**: The LLM can now say *"Hacktivism groups include Labomedia and Tetalab,"* even if those specific names weren't in the vector search top hits, because they were linked children of the top hits.

## 2.5 Generated Python Code

The compiler translates `graph_search` into:

```python
_get_runtime().searcher.graph_search(query, top_k=3, hops=1)
```

The runtime (`PtSearcher`) executes the following SQL logic:
1.  `SELECT id FROM embeddings ...` (Vector Search)
2.  `SELECT target_id FROM links WHERE source_id IN (ids)` (Parents)
3.  `SELECT source_id FROM links WHERE target_id IN (ids)` (Children)
4.  Returns a JSON object with `focus`, `parents`, and `children`.

---

## Navigation

**←** [Previous: Chapter 1: The Python Semantic Target](01_introduction) | [📖 Book 13: Semantic Search](README) | [Next: Chapter 3: The Semantic Data Pipeline →](03_semantic_data_pipeline)
