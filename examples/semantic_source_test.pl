%% semantic_source_test.pl
%
% Test file for semantic source plugin
%
% This demonstrates the semantic search source with ONNX embeddings.

:- module(semantic_source_test, [
    find_relevant_papers/3
]).

:- use_module('../src/unifyweaver/sources').
:- use_module('../src/unifyweaver/sources/semantic_source').

%% Test 1: Basic semantic source with minimal config
:- source(semantic, test_papers, [
    vector_store('test_data/papers_vectors.json'),
    embedding_backend(python_onnx, 'all-MiniLM-L6-v2'),
    similarity_threshold(0.6),
    top_k(5)
]).

%% Test 2: Semantic source with explicit model configuration
:- source(semantic, detailed_search, [
    vector_store('test_data/documents.json'),
    embedding_backend(python_onnx, [
        model_path('models/all-MiniLM-L6-v2.onnx'),
        vocab_path('models/vocab.txt'),
        dimensions(384)
    ]),
    similarity_threshold(0.7),
    top_k(10),
    similarity_metric(cosine),
    normalize_vectors(true)
]).

%% Test 3: Higher-level query using semantic source
find_relevant_papers(Query, PaperId, Score) :-
    test_papers(Query, PaperId, Score),
    Score > 0.75.

%% Test 4: Combine with other sources (hybrid search pattern)
% Note: This would require actual CSV source, just showing the pattern
/*
:- source(csv, paper_metadata, [
    csv_file('test_data/papers.csv'),
    has_header(true)
]).

search_with_metadata(Query, Title, Author, Year, Score) :-
    test_papers(Query, PaperId, Score),        % Semantic search
    paper_metadata(PaperId, Title, Author, Year),  % Lookup metadata
    Year >= 2020,                               % Filter by year
    Score > 0.75.
*/
