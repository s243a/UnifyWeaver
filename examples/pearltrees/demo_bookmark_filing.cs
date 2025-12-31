using System;
using LiteDB;
using UnifyWeaver.QueryRuntime;

// Demo: Bookmark Filing Assistant using semantic search and tree formatting
var dbPath = "pt_ingest_test.db";
var modelPath = "models/all-MiniLM-L6-v2/model.onnx";
var vocabPath = "models/all-MiniLM-L6-v2/vocab.txt";

Console.WriteLine("=== Bookmark Filing Assistant Demo ===\n");
Console.WriteLine("Loading database and embedding model...");

var embeddingProvider = new OnnxEmbeddingProvider(modelPath, vocabPath);
using var searcher = new PtSearcher(dbPath, embeddingProvider);

Console.WriteLine("✓ Ready\n");

// Test cases for bookmark filing
var bookmarks = new[]
{
    "Article about quantum entanglement and its applications in quantum computing",
    "Tutorial on classical mechanics and Newton's laws of motion",
    "Research paper on electromagnetic waves and Maxwell's equations"
};

foreach (var bookmark in bookmarks)
{
    Console.WriteLine(new string('=', 100));
    Console.WriteLine();

    var result = searcher.FindBookmarkPlacements(bookmark, topCandidates: 3, minScore: 0.35);
    Console.WriteLine(result);

    Console.WriteLine();
}

Console.WriteLine(new string('=', 100));
Console.WriteLine("\n✓ Demo complete");
