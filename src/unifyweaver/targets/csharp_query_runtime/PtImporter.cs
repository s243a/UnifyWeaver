// SPDX-License-Identifier: MIT OR Apache-2.0
using System;
using System.Collections.Generic;
using System.Linq;
using LiteDB;
using UnifyWeaver.QueryRuntime.Pearltrees;

namespace UnifyWeaver.QueryRuntime
{
    public sealed class PtImporter
    {
        private readonly LiteDatabase _db;
        private readonly ILiteCollection<PtTree> _trees;
        private readonly ILiteCollection<PtPearl> _pearls;
        private readonly ILiteCollection<BsonDocument> _embeddings; // { _id, vector: array }

        public PtImporter(string dbPath)
        {
            _db = new LiteDatabase(dbPath);
            _trees = _db.GetCollection<PtTree>("trees");
            _pearls = _db.GetCollection<PtPearl>("pearls");
            _embeddings = _db.GetCollection("embeddings");
        }

        public void Dispose() => _db?.Dispose();

        public void Upsert(PtEntity entity)
        {
            switch (entity.Type)
            {
                case "pt:Tree":
                    _trees.Upsert(ToTree(entity));
                    break;
                default:
                    _pearls.Upsert(ToPearl(entity));
                    break;
            }
        }

        public void UpsertEmbedding(string id, IEnumerable<double> vector)
        {
            var doc = new BsonDocument
            {
                ["_id"] = id,
                ["vector"] = new BsonArray(vector.Select(v => new BsonValue(v)))
            };
            _embeddings.Upsert(doc);
        }

        private PtTree ToTree(PtEntity e)
        {
            return new PtTree
            {
                Id = e.Id,
                About = e.About,
                Type = e.Type,
                Title = e.Title,
                Privacy = e.Privacy,
                ParentTree = e.ParentTree,
                Children = e.Children,
                Raw = e.Raw
            };
        }

        private PtPearl ToPearl(PtEntity e)
        {
            return new PtPearl
            {
                Id = e.Id,
                About = e.About,
                Type = e.Type,
                Title = e.Title,
                Privacy = e.Privacy,
                ParentTree = e.ParentTree,
                Children = e.Children,
                Raw = e.Raw
            };
        }
    }
}
