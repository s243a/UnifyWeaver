// SPDX-License-Identifier: MIT OR Apache-2.0

using System.Collections.Generic;
using LiteDB;

namespace UnifyWeaver.QueryRuntime.Pearltrees
{
    public class PtEntity
    {
        [BsonId]
        public string Id { get; set; } = string.Empty;
        public string About { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public string? Title { get; set; }
        public int? Privacy { get; set; }
        public string? ParentTree { get; set; }
        public List<string>? Children { get; set; }
        public BsonDocument Raw { get; set; } = new BsonDocument();
    }

    public class PtTree : PtEntity
    {
    }

    public class PtPearl : PtEntity
    {
    }
}
