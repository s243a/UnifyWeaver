using System;
using System.IO;
using System.Linq;
using LiteDB;
using UnifyWeaver.QueryRuntime;

// Export physics seeds with threshold 0.35
var modelPath = "models/all-MiniLM-L6-v2/model.onnx";
var vocabPath = "models/all-MiniLM-L6-v2/vocab.txt";
var dbPath = "pt_ingest_test.db";
var outputPath = "physics_seeds_035.txt";

Console.WriteLine("Loading embedding model...");
var embeddingProvider = new OnnxEmbeddingProvider(modelPath, vocabPath);

using var searcher = new PtSearcher(dbPath, embeddingProvider);

Console.WriteLine("Searching for 'physics' with threshold 0.35...");
var treeResults = searcher.SearchSimilar("physics", topK: 1000, minScore: 0.35, typeFilter: "pt:Tree");

Console.WriteLine($"Found {treeResults.Count} tree matches");
Console.WriteLine($"Writing results to {outputPath}...");

using (var writer = new StreamWriter(outputPath))
{
    writer.WriteLine("# Physics-related trees (cosine similarity >= 0.35)");
    writer.WriteLine($"# Query: \"physics\"");
    writer.WriteLine($"# Total matches: {treeResults.Count}");
    writer.WriteLine($"# Similarity metric: Cosine similarity (dot product / (||a|| * ||b||))");
    writer.WriteLine($"# Score range: 0.0 (orthogonal) to 1.0 (identical)");
    writer.WriteLine();
    writer.WriteLine("Rank\tScore\tID\tTitle");
    writer.WriteLine("----\t-----\t--\t-----");

    int rank = 1;
    foreach (var result in treeResults)
    {
        var title = result.Title ?? result.About ?? result.Id;
        writer.WriteLine($"{rank}\t{result.Score:F4}\t{result.Id}\t{title}");
        rank++;
    }
}

Console.WriteLine($"✓ Exported {treeResults.Count} physics trees to {outputPath}");
Console.WriteLine();
Console.WriteLine("Top 10 matches:");
foreach (var result in treeResults.Take(10))
{
    var title = result.Title ?? result.About ?? result.Id;
    if (title.Length > 60) title = title.Substring(0, 60) + "...";
    Console.WriteLine($"  [{result.Score:F4}] {title}");
}
