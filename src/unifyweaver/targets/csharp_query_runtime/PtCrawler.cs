// SPDX-License-Identifier: MIT OR Apache-2.0
using System;
using System.Collections.Generic;
using System.Linq;
using UnifyWeaver.QueryRuntime.Dynamic;
using UnifyWeaver.QueryRuntime.Pearltrees;

namespace UnifyWeaver.QueryRuntime
{
    public sealed class PtCrawler : IDisposable
    {
        private readonly PtImporter _importer;
        private readonly XmlStreamReader _reader;
        private readonly IEmbeddingProvider? _embeddingProvider;

        public PtCrawler(string dbPath, XmlSourceConfig config, IEmbeddingProvider? embeddingProvider = null)
        {
            _importer = new PtImporter(dbPath);
            _reader = new XmlStreamReader(config);
            _embeddingProvider = embeddingProvider;
        }

        public void Dispose()
        {
            _importer.Dispose();
            // Don't dispose embedding provider - we don't own it
        }

        public void IngestOnce(bool emitEmbeddings = false)
        {
            foreach (var row in _reader.Read())
            {
                var map = ToDict(row);
                var entity = PtMapper.Map(map);
                if (entity.Privacy.HasValue && entity.Privacy.Value > 1)
                {
                    continue; // skip private
                }
                _importer.Upsert(entity);
                if (emitEmbeddings && _embeddingProvider is not null)
                {
                    var text = entity.Title ?? entity.About ?? entity.Id;
                    var embedding = _embeddingProvider.GetEmbedding(text);
                    _importer.UpsertEmbedding(entity.Id, embedding);
                }
            }
        }

        public void FixedPoint(IEnumerable<string> seedIds, Func<string, XmlSourceConfig> fetchConfig, int maxDepth = 5)
        {
            var seen = new HashSet<string>(seedIds);
            var frontier = new Queue<string>(seedIds);
            int depth = 0;
            while (frontier.Count > 0 && depth < maxDepth)
            {
                var nextBatch = new List<string>();
                while (frontier.Count > 0)
                {
                    var id = frontier.Dequeue();
                    var cfg = fetchConfig(id);
                    var reader = new XmlStreamReader(cfg);
                    foreach (var row in reader.Read())
                    {
                        var map = ToDict(row);
                        var entity = PtMapper.Map(map);
                        if (entity.Privacy.HasValue && entity.Privacy.Value > 1) continue;
                        _importer.Upsert(entity);
                        if (entity.Children is { } kids)
                        {
                            foreach (var kid in kids)
                            {
                                if (seen.Add(kid))
                                {
                                    nextBatch.Add(kid);
                                }
                            }
                        }
                    }
                }
                foreach (var kid in nextBatch) frontier.Enqueue(kid);
                depth++;
            }
        }

        private IDictionary<string, object?> ToDict(object[] row)
        {
            // expect a single dictionary per row
            if (row.Length == 1 && row[0] is IDictionary<string, object?> dict)
            {
                return dict;
            }
            // fallback: index-based mapping
            var map = new Dictionary<string, object?>();
            for (int i = 0; i < row.Length; i++)
            {
                map[$"col{i}"] = row[i];
            }
            return map;
        }

    }
}
