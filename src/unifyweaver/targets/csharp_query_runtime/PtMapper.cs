// SPDX-License-Identifier: MIT OR Apache-2.0
using System;
using System.Collections.Generic;
using System.Linq;
using LiteDB;
using UnifyWeaver.QueryRuntime.Pearltrees;

namespace UnifyWeaver.QueryRuntime
{
    public static class PtMapper
    {
        public static PtEntity Map(IDictionary<string, object?> row)
        {
            // Get type - prefer explicit Type field over guessing
            var typeField = GetString(row, "Type");
            var type = !string.IsNullOrWhiteSpace(typeField) ? typeField : GuessType(row) ?? string.Empty;
            var about = GetString(row, "@about") ?? GetString(row, "about") ?? string.Empty;
            var id = ExtractId(row, about);
            var title = GetString(row, "title") ?? GetString(row, "dcterms:title");
            var privacy = GetInt(row, "pt:privacy");
            var parentTree = GetString(row, "pt:parentTree") ?? GetString(row, "parentTree");
            var children = CollectChildren(row);

            var rawDoc = ToBsonDocument(row);

            var entity = new PtEntity
            {
                Id = id,
                About = about,
                Type = type ?? string.Empty,
                Title = title,
                Privacy = privacy,
                ParentTree = parentTree,
                Children = children,
                Raw = rawDoc
            };

            // Upgrade to tree/pearl as needed
            if (string.Equals(type, "pt:Tree", StringComparison.OrdinalIgnoreCase))
            {
                return new PtTree
                {
                    Id = entity.Id,
                    About = entity.About,
                    Type = entity.Type,
                    Title = entity.Title,
                    Privacy = entity.Privacy,
                    ParentTree = entity.ParentTree,
                    Children = entity.Children,
                    Raw = entity.Raw
                };
            }
            return new PtPearl
            {
                Id = entity.Id,
                About = entity.About,
                Type = entity.Type,
                Title = entity.Title,
                Privacy = entity.Privacy,
                ParentTree = entity.ParentTree,
                Children = entity.Children,
                Raw = entity.Raw
            };
        }

        private static string ExtractId(IDictionary<string, object?> row, string about)
        {
            var treeId = GetString(row, "pt:treeId");
            if (!string.IsNullOrEmpty(treeId)) return treeId;
            var refId = GetString(row, "pt:refId");
            if (!string.IsNullOrEmpty(refId)) return refId;
            if (string.IsNullOrEmpty(about)) return Guid.NewGuid().ToString("N");
            var last = about.Split('/').Last();
            return string.IsNullOrEmpty(last) ? about : last;
        }

        private static string? GuessType(IDictionary<string, object?> row)
        {
            if (row.Keys.Any(k => k.StartsWith("pt:treeId", StringComparison.OrdinalIgnoreCase)))
                return "pt:Tree";
            if (row.Keys.Any(k => k.StartsWith("pt:refId", StringComparison.OrdinalIgnoreCase)))
                return "pt:RefPearl";
            return null;
        }

        private static List<string>? CollectChildren(IDictionary<string, object?> row)
        {
            var children = new List<string>();
            if (row.TryGetValue("pt:seeAlso", out var seeAlso) && seeAlso is string sa && !string.IsNullOrWhiteSpace(sa))
                children.Add(sa);
            return children.Count > 0 ? children : null;
        }

        private static string? GetString(IDictionary<string, object?> row, string key)
        {
            if (row.TryGetValue(key, out var val) && val is string s) return s;
            return null;
        }

        private static int? GetInt(IDictionary<string, object?> row, string key)
        {
            if (row.TryGetValue(key, out var val))
            {
                if (val is int i) return i;
                if (val is string s && int.TryParse(s, out var parsed)) return parsed;
            }
            return null;
        }

        private static BsonDocument ToBsonDocument(IDictionary<string, object?> row)
        {
            var doc = new BsonDocument();
            foreach (var kv in row)
            {
                doc[kv.Key] = ToBsonValue(kv.Value);
            }
            return doc;
        }

        private static BsonValue ToBsonValue(object? value)
        {
            return value switch
            {
                null => BsonValue.Null,
                string s => new BsonValue(s),
                int i => new BsonValue(i),
                long l => new BsonValue(l),
                double d => new BsonValue(d),
                IEnumerable<object?> list => new BsonArray(list.Select(ToBsonValue)),
                _ => new BsonValue(value.ToString())
            };
        }
    }
}
