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
        /// Get document IDs for use as crawler seeds based on semantic search.
        /// Useful for focused crawling: find relevant starting points, then crawl their children.
        /// Example: GetSeedIds("physics quantum mechanics", 100) → Top 100 physics-related IDs
        /// </summary>
        /// <param name="query">Semantic query describing desired topic</param>
        /// <param name="topK">Number of seed IDs to return (default 100)</param>
        /// <param name="minScore">Minimum similarity score (0-1, default 0.5 for quality seeds)</param>
        /// <param name="typeFilter">Optional type filter (e.g., "pt:Tree" for trees only, null for all types)</param>
        /// <returns>List of document IDs ranked by relevance</returns>
        public List<string> GetSeedIds(string query, int topK = 100, double minScore = 0.5, string? typeFilter = null)
        {
            var results = SearchSimilar(query, topK, minScore, typeFilter);
            return results.Select(r => r.Id).ToList();
        }

        /// <summary>
        /// Search for documents similar to the query text.
        /// </summary>
        /// <param name="query">Search query text</param>
        /// <param name="topK">Number of results to return (default 10)</param>
        /// <param name="minScore">Minimum similarity score (0-1, default 0.0)</param>
        /// <param name="typeFilter">Optional type filter (e.g., "pt:Tree" to return only trees, null for all types)</param>
        /// <returns>List of search results with scores</returns>
        public List<SearchResult> SearchSimilar(string query, int topK = 10, double minScore = 0.0, string? typeFilter = null)
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

                    // Apply type filter if specified
                    if (typeFilter != null && entity?.Type != typeFilter)
                    {
                        continue;
                    }

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
        /// Get all children of a tree or pearl.
        /// </summary>
        public List<PtEntity> GetChildren(string id)
        {
            var entity = GetEntity(id);
            if (entity?.Children == null || entity.Children.Count == 0)
            {
                return new List<PtEntity>();
            }

            var children = new List<PtEntity>();
            foreach (var childId in entity.Children)
            {
                var child = GetEntity(childId);
                if (child != null)
                {
                    children.Add(child);
                }
            }
            return children;
        }

        /// <summary>
        /// Get the parent tree of an entity.
        /// </summary>
        public PtEntity? GetParent(string id)
        {
            var entity = GetEntity(id);
            if (entity?.ParentTree == null)
            {
                return null;
            }
            return GetEntity(entity.ParentTree);
        }

        /// <summary>
        /// Get all ancestors (parent, grandparent, ...) up to the root.
        /// Returns list from immediate parent to root.
        /// </summary>
        public List<PtEntity> GetAncestors(string id)
        {
            var ancestors = new List<PtEntity>();
            var current = GetParent(id);
            var visited = new HashSet<string> { id }; // Prevent cycles

            while (current != null && !visited.Contains(current.Id))
            {
                ancestors.Add(current);
                visited.Add(current.Id);
                current = GetParent(current.Id);
            }

            return ancestors;
        }

        /// <summary>
        /// Get siblings (entities with the same parent).
        /// </summary>
        public List<PtEntity> GetSiblings(string id)
        {
            var entity = GetEntity(id);
            if (entity?.ParentTree == null)
            {
                return new List<PtEntity>();
            }

            var parent = GetEntity(entity.ParentTree);
            if (parent?.Children == null)
            {
                return new List<PtEntity>();
            }

            var siblings = new List<PtEntity>();
            foreach (var siblingId in parent.Children)
            {
                if (siblingId != id) // Exclude self
                {
                    var sibling = GetEntity(siblingId);
                    if (sibling != null)
                    {
                        siblings.Add(sibling);
                    }
                }
            }
            return siblings;
        }

        /// <summary>
        /// Find candidate locations for filing a new bookmark and format for LLM presentation.
        /// Returns a formatted string showing each candidate in tree context.
        /// </summary>
        /// <param name="bookmarkDescription">Description of the bookmark to file (title, content, etc.)</param>
        /// <param name="topCandidates">Number of candidate locations to return (default 5)</param>
        /// <param name="minScore">Minimum similarity threshold (default 0.35)</param>
        /// <returns>Formatted string with all candidates and their contexts</returns>
        public string FindBookmarkPlacements(string bookmarkDescription, int topCandidates = 5, double minScore = 0.35)
        {
            // Find semantically similar trees
            var candidates = SearchSimilar(bookmarkDescription, topK: topCandidates, minScore: minScore, typeFilter: "pt:Tree");

            if (candidates.Count == 0)
            {
                return "No suitable placement locations found. Consider lowering the similarity threshold or creating a new category.";
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("=== Bookmark Filing Suggestions ===");
            sb.AppendLine();
            sb.AppendLine($"Bookmark: \"{bookmarkDescription}\"");
            sb.AppendLine($"Found {candidates.Count} candidate location(s):");
            sb.AppendLine();
            sb.AppendLine(new string('=', 80));
            sb.AppendLine();

            for (int i = 0; i < candidates.Count; i++)
            {
                sb.AppendLine($"Option {i + 1}:");
                sb.AppendLine();
                sb.Append(BuildTreeContext(candidates[i].Id, candidates[i].Score));
                sb.AppendLine();
                if (i < candidates.Count - 1)
                {
                    sb.AppendLine(new string('-', 80));
                    sb.AppendLine();
                }
            }

            return sb.ToString();
        }

        /// <summary>
        /// Build a tree-formatted context string for a candidate entity.
        /// Shows ancestors, siblings, the candidate itself (marked), and children.
        /// Format is similar to Linux 'tree' command output.
        /// </summary>
        /// <param name="candidateId">The candidate entity ID</param>
        /// <param name="score">Similarity score for the candidate</param>
        /// <param name="maxChildrenToShow">Maximum children to display (default 10)</param>
        /// <returns>Formatted tree string showing context</returns>
        public string BuildTreeContext(string candidateId, double score, int maxChildrenToShow = 10)
        {
            var candidate = GetEntity(candidateId);
            if (candidate == null)
            {
                return $"Entity {candidateId} not found";
            }

            var sb = new System.Text.StringBuilder();
            var title = candidate.Title ?? candidate.About ?? candidateId;

            // Header with score
            sb.AppendLine($"Candidate: \"{title}\" (similarity: {score:F3})");
            sb.AppendLine();

            // Build ancestor path (root to immediate parent)
            var ancestors = GetAncestors(candidateId);
            ancestors.Reverse(); // Want root first

            // Show ancestor path
            if (ancestors.Count > 0)
            {
                for (int i = 0; i < ancestors.Count; i++)
                {
                    var indent = new string(' ', i * 4);
                    var ancestorTitle = ancestors[i].Title ?? ancestors[i].About ?? ancestors[i].Id;
                    if (ancestorTitle.Length > 60) ancestorTitle = ancestorTitle.Substring(0, 57) + "...";

                    if (i == 0)
                    {
                        sb.AppendLine($"└── {ancestorTitle}/");
                    }
                    else
                    {
                        sb.AppendLine($"{indent}└── {ancestorTitle}/");
                    }
                }
            }

            // Show siblings and candidate at current level
            var siblings = GetSiblings(candidateId);
            var baseIndent = new string(' ', ancestors.Count * 4);

            // Show some siblings before candidate
            foreach (var sibling in siblings.Take(3))
            {
                var siblingTitle = sibling.Title ?? sibling.About ?? sibling.Id;
                if (siblingTitle.Length > 60) siblingTitle = siblingTitle.Substring(0, 57) + "...";
                sb.AppendLine($"{baseIndent}    ├── {siblingTitle}/");
            }

            // Show the candidate (marked with arrow)
            var candTitle = candidate.Title ?? candidate.About ?? candidateId;
            if (candTitle.Length > 50) candTitle = candTitle.Substring(0, 47) + "...";
            sb.AppendLine($"{baseIndent}    ├── {candTitle}/        ← CANDIDATE (place new bookmark here)");

            // Show remaining siblings
            if (siblings.Count > 3)
            {
                foreach (var sibling in siblings.Skip(3).Take(2))
                {
                    var siblingTitle = sibling.Title ?? sibling.About ?? sibling.Id;
                    if (siblingTitle.Length > 60) siblingTitle = siblingTitle.Substring(0, 57) + "...";
                    sb.AppendLine($"{baseIndent}    ├── {siblingTitle}/");
                }
                if (siblings.Count > 5)
                {
                    sb.AppendLine($"{baseIndent}    ├── ... ({siblings.Count - 5} more siblings)");
                }
            }

            // Show children of the candidate
            var children = GetChildren(candidateId);
            if (children.Count > 0)
            {
                var childIndent = baseIndent + "    │   ";
                int shown = 0;
                foreach (var child in children.Take(maxChildrenToShow))
                {
                    var childTitle = child.Title ?? child.About ?? child.Id;
                    if (childTitle.Length > 60) childTitle = childTitle.Substring(0, 57) + "...";
                    var isLast = (shown == Math.Min(children.Count, maxChildrenToShow) - 1);
                    var prefix = isLast && children.Count <= maxChildrenToShow ? "└──" : "├──";
                    sb.AppendLine($"{childIndent}{prefix} {childTitle}");
                    shown++;
                }
                if (children.Count > maxChildrenToShow)
                {
                    sb.AppendLine($"{childIndent}└── ... ({children.Count - maxChildrenToShow} more children)");
                }
            }

            return sb.ToString();
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
