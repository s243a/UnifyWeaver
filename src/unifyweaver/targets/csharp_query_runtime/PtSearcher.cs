// SPDX-License-Identifier: MIT OR Apache-2.0
using System;
using System.Collections.Generic;
using System.Linq;
using LiteDB;
using UnifyWeaver.QueryRuntime.Pearltrees;

namespace UnifyWeaver.QueryRuntime
{
    /// <summary>
    /// Vector similarity search over Pearltrees embeddings.
    /// </summary>
    public sealed class PtSearcher : IDisposable
    {
        private readonly LiteDatabase _db;
        private readonly ILiteCollection<PtTree> _trees;
        private readonly ILiteCollection<PtPearl> _pearls;
        private readonly ILiteCollection<BsonDocument> _embeddings;
        private readonly IEmbeddingProvider _embeddingProvider;

        public PtSearcher(string dbPath, IEmbeddingProvider embeddingProvider)
        {
            _db = new LiteDatabase(dbPath);
            _trees = _db.GetCollection<PtTree>("trees");
            _pearls = _db.GetCollection<PtPearl>("pearls");
            _embeddings = _db.GetCollection("embeddings");
            _embeddingProvider = embeddingProvider;
        }

        public void Dispose() => _db?.Dispose();

        /// <summary>
        /// Search for documents similar to the query text.
        /// </summary>
        /// <param name="query">Search query text</param>
        /// <param name="topK">Number of results to return (default 10)</param>
        /// <param name="minScore">Minimum similarity score (0-1, default 0.0)</param>
        /// <returns>List of search results with scores</returns>
        public List<SearchResult> SearchSimilar(string query, int topK = 10, double minScore = 0.0)
        {
            if (string.IsNullOrWhiteSpace(query))
            {
                return new List<SearchResult>();
            }

            // Generate embedding for the query
            var queryEmbedding = _embeddingProvider.GetEmbedding(query);

            // Get all embeddings and compute similarities
            var results = new List<SearchResult>();
            foreach (var doc in _embeddings.FindAll())
            {
                var id = doc["_id"].AsString;
                var vectorArray = doc["vector"].AsArray;
                var embedding = vectorArray.Select(v => v.AsDouble).ToArray();

                // Compute cosine similarity
                var similarity = CosineSimilarity(queryEmbedding, embedding);

                if (similarity >= minScore)
                {
                    // Look up the actual document
                    var entity = GetEntity(id);
                    results.Add(new SearchResult
                    {
                        Id = id,
                        Score = similarity,
                        Type = entity?.Type,
                        Title = entity?.Title,
                        About = entity?.About
                    });
                }
            }

            // Sort by score descending and return top-k
            return results
                .OrderByDescending(r => r.Score)
                .Take(topK)
                .ToList();
        }

        /// <summary>
        /// Compute cosine similarity between two vectors.
        /// Returns a value between 0 (orthogonal) and 1 (identical).
        /// </summary>
        private double CosineSimilarity(double[] a, double[] b)
        {
            if (a.Length != b.Length)
            {
                throw new ArgumentException("Vectors must have the same length");
            }

            double dotProduct = 0;
            double normA = 0;
            double normB = 0;

            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            if (normA == 0 || normB == 0)
            {
                return 0;
            }

            return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
        }

        private PtEntity? GetEntity(string id)
        {
            // Try trees first
            var tree = _trees.FindById(id);
            if (tree is not null)
            {
                return new PtEntity
                {
                    Id = tree.Id,
                    Type = tree.Type,
                    Title = tree.Title,
                    About = tree.About,
                    Privacy = tree.Privacy,
                    ParentTree = tree.ParentTree,
                    Children = tree.Children,
                    Raw = tree.Raw
                };
            }

            // Try pearls
            var pearl = _pearls.FindById(id);
            if (pearl is not null)
            {
                return new PtEntity
                {
                    Id = pearl.Id,
                    Type = pearl.Type,
                    Title = pearl.Title,
                    About = pearl.About,
                    Privacy = pearl.Privacy,
                    ParentTree = pearl.ParentTree,
                    Children = pearl.Children,
                    Raw = pearl.Raw
                };
            }

            return null;
        }
    }

    /// <summary>
    /// A search result with similarity score.
    /// </summary>
    public sealed class SearchResult
    {
        public string Id { get; set; } = string.Empty;
        public double Score { get; set; }
        public string? Type { get; set; }
        public string? Title { get; set; }
        public string? About { get; set; }
    }
}
