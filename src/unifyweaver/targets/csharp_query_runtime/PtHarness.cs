// SPDX-License-Identifier: MIT OR Apache-2.0
using System;
using System.Collections.Generic;
using System.IO;
using UnifyWeaver.QueryRuntime.Dynamic;

namespace UnifyWeaver.QueryRuntime
{
    public static class PtHarness
    {
        public static void RunIngest(string xmlPath, string dbPath)
        {
            RunIngest(xmlPath, dbPath, embeddingProvider: null, emitEmbeddings: false);
        }

        public static void RunIngest(string xmlPath, string dbPath, bool emitEmbeddings)
        {
            RunIngest(xmlPath, dbPath, embeddingProvider: null, emitEmbeddings: emitEmbeddings);
        }

        public static void RunIngest(string xmlPath, string dbPath, IEmbeddingProvider? embeddingProvider, bool emitEmbeddings)
        {
            var config = new XmlSourceConfig
            {
                InputPath = xmlPath,
                RecordSeparator = RecordSeparatorKind.LineFeed,
                NamespacePrefixes = new Dictionary<string, string>
                {
                    {"http://www.pearltrees.com/rdf/0.1/#", "pt"},
                    {"http://purl.org/dc/elements/1.1/", "dcterms"}
                },
                TreatPearltreesCDataAsText = true
            };

            using var crawler = new PtCrawler(dbPath, config, embeddingProvider);
            crawler.IngestOnce(emitEmbeddings);
        }

        /// <summary>
        /// Perform semantic-driven fixed-point crawl: use semantic search to find relevant starting points,
        /// then crawl through their children to build a focused subset of the knowledge graph.
        /// </summary>
        /// <param name="seedQuery">Semantic query to find starting points (e.g., "physics quantum mechanics")</param>
        /// <param name="sourceDb">Database path containing indexed documents with embeddings</param>
        /// <param name="targetDb">Database path for the crawled subset</param>
        /// <param name="embeddingProvider">Embedding provider for semantic search</param>
        /// <param name="fetchConfig">Function to create XmlSourceConfig for fetching document by ID</param>
        /// <param name="topSeeds">Number of seed documents to start from (default 100)</param>
        /// <param name="minScore">Minimum similarity score for seeds (default 0.5)</param>
        /// <param name="maxDepth">Maximum crawl depth from seeds (default 3)</param>
        /// <param name="typeFilter">Optional type filter for seeds (e.g., "pt:Tree" for trees only, null for all types)</param>
        public static void RunSemanticCrawl(
            string seedQuery,
            string sourceDb,
            string targetDb,
            IEmbeddingProvider embeddingProvider,
            Func<string, XmlSourceConfig> fetchConfig,
            int topSeeds = 100,
            double minScore = 0.5,
            int maxDepth = 3,
            string? typeFilter = null)
        {
            // Use semantic search to find seed IDs
            List<string> seeds;
            using (var searcher = new PtSearcher(sourceDb, embeddingProvider))
            {
                var filterDesc = typeFilter != null ? $" (type: {typeFilter})" : "";
                Console.WriteLine($"Finding seeds via semantic search: \"{seedQuery}\"{filterDesc}");
                seeds = searcher.GetSeedIds(seedQuery, topSeeds, minScore, typeFilter);
                Console.WriteLine($"Found {seeds.Count} seed documents (minScore >= {minScore})");
            }

            if (seeds.Count == 0)
            {
                Console.WriteLine("No seeds found - try lowering minScore or using a different query");
                return;
            }

            // Crawl from the semantic seeds
            Console.WriteLine($"Starting fixed-point crawl from {seeds.Count} seeds (maxDepth={maxDepth})...");
            using var crawler = new PtCrawler(targetDb, fetchConfig(seeds[0]), embeddingProvider);
            crawler.FixedPoint(seeds, fetchConfig, maxDepth);
            Console.WriteLine("Semantic crawl complete");
        }
    }
}
