// SPDX-License-Identifier: MIT OR Apache-2.0

using LiteDB;
using System;
using System.Collections.Generic;
using System.Text.Json.Nodes; // For optional JsonNode return type
using System.Text.Json; // For JsonSerializer, if converting BsonDocument to JsonNode

namespace UnifyWeaver.QueryRuntime.Pearltrees
{
    public static class PtEntityExtensions
    {
        /// <summary>
        /// Attempts to get a value from the PtEntity's Raw BsonDocument by key, converting it to the specified type.
        /// If the key is not found or conversion fails, the default value for type T is returned.
        /// </summary>
        /// <typeparam name="T">The target type for the value.</typeparam>
        /// <param name="entity">The PtEntity instance.</param>
        /// <param name="key">The key (field name) to retrieve from the Raw BsonDocument.</param>
        /// <returns>The converted value, or the default value for type T if not found or conversion fails.</returns>
        public static T? GetRawValue<T>(this PtEntity entity, string key)
        {
            if (entity.Raw == null || !entity.Raw.TryGetValue(key, out var bsonValue))
            {
                return default(T);
            }

            try
            {
                // LiteDB's BsonValue has AsType<T>() for convenient conversion
                return bsonValue.AsType<T>();
            }
            catch (Exception ex) // Catch all exceptions from AsType<T> to return default
            {
                // Log the conversion error if desired: Console.WriteLine($"Conversion error for key '{key}': {ex.Message}");
                return default(T);
            }
        }

        /// <summary>
        /// Attempts to get a string value from the PtEntity's Raw BsonDocument by key.
        /// </summary>
        public static string? GetRawString(this PtEntity entity, string key) =>
            entity.GetRawValue<string>(key);

        /// <summary>
        /// Attempts to get an integer value from the PtEntity's Raw BsonDocument by key.
        /// </summary>
        public static int? GetRawInt(this PtEntity entity, string key) =>
            entity.GetRawValue<int?>(key); // Use nullable int to match null return

        /// <summary>
        /// Attempts to get a long integer value from the PtEntity's Raw BsonDocument by key.
        /// </summary>
        public static long? GetRawLong(this PtEntity entity, string key) =>
            entity.GetRawValue<long?>(key);

        /// <summary>
        /// Attempts to get a double-precision floating-point value from the PtEntity's Raw BsonDocument by key.
        /// </summary>
        public static double? GetRawDouble(this PtEntity entity, string key) =>
            entity.GetRawValue<double?>(key);

        /// <summary>
        /// Attempts to get a boolean value from the PtEntity's Raw BsonDocument by key.
        /// </summary>
        public static bool? GetRawBool(this PtEntity entity, string key) =>
            entity.GetRawValue<bool?>(key);
        
        /// <summary>
        /// Attempts to get a BsonDocument value from the PtEntity's Raw BsonDocument by key.
        /// </summary>
        public static BsonDocument? GetRawDocument(this PtEntity entity, string key) =>
            entity.GetRawValue<BsonDocument>(key);

        /// <summary>
        /// Attempts to get a JsonNode representation of a value from the PtEntity's Raw BsonDocument.
        /// This is useful for more advanced, dynamic JSON manipulation.
        /// </summary>
        public static JsonNode? GetRawJsonNode(this PtEntity entity, string key)
        {
            if (entity.Raw == null || !entity.Raw.TryGetValue(key, out var bsonValue))
            {
                return null;
            }
            // Direct conversion from BsonValue to JsonNode is not straightforward and often involves intermediate JSON string.
            // A more robust solution might involve custom LiteDB BsonValue-to-JsonNode mappers.
            try
            {
                // Convert BsonValue to LiteDB object, then serialize with System.Text.Json, then parse to JsonNode.
                // This is a common but potentially inefficient route for complex BsonValues.
                var obj = BsonMapper.Global.Deserialize<object>(bsonValue);
                if (obj == null) return null;
                var jsonString = System.Text.Json.JsonSerializer.Serialize(obj);
                return JsonNode.Parse(jsonString);
            }
            catch (Exception ex)
            {
                // Log conversion error if desired
                return null;
            }
        }
    }
}
