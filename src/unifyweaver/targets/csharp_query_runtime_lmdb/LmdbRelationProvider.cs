// SPDX-License-Identifier: MIT OR Apache-2.0
// Optional LMDB-backed relation provider for C# query runtime artifacts.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.Json;
using LightningDB;

namespace UnifyWeaver.QueryRuntime;

public sealed class LmdbRelationArtifactManifest
{
    public const string CurrentFormat = "unifyweaver.lmdb_relation.v1";

    public string Format { get; set; } = CurrentFormat;

    public int Version { get; set; } = 1;

    public string Backend { get; set; } = "lmdb";

    public string PredicateName { get; set; } = string.Empty;

    public int Arity { get; set; } = 2;

    public string EnvironmentPath { get; set; } = string.Empty;

    public string? DatabaseName { get; set; }

    public bool DupSort { get; set; }

    public string KeyEncoding { get; set; } = "utf8";

    public string ValueEncoding { get; set; } = "utf8";

    public long RowCount { get; set; }

    public string? SourcePath { get; set; }

    public long? SourceLength { get; set; }

    public string? SourceSha256 { get; set; }

    public void Validate(string manifestPath)
    {
        if (!string.Equals(Format, CurrentFormat, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Unsupported LMDB relation artifact format '{Format}': {manifestPath}");
        }

        if (Version != 1)
        {
            throw new InvalidDataException($"Unsupported LMDB relation artifact manifest version {Version}: {manifestPath}");
        }

        if (!string.Equals(Backend, "lmdb", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"LMDB relation artifact has unsupported backend '{Backend}': {manifestPath}");
        }

        if (string.IsNullOrWhiteSpace(PredicateName))
        {
            throw new InvalidDataException($"LMDB relation artifact manifest has no predicate name: {manifestPath}");
        }

        if (Arity != 2)
        {
            throw new InvalidDataException($"LMDB relation artifact smoke provider only supports arity-2 facts: {manifestPath}");
        }

        if (string.IsNullOrWhiteSpace(EnvironmentPath))
        {
            throw new InvalidDataException($"LMDB relation artifact manifest has no environment path: {manifestPath}");
        }

        if (RowCount < 0)
        {
            throw new InvalidDataException($"LMDB relation artifact row count must be non-negative: {manifestPath}");
        }

        ValidateEncoding(KeyEncoding, manifestPath);
        ValidateEncoding(ValueEncoding, manifestPath);
    }

    private static void ValidateEncoding(string encoding, string manifestPath)
    {
        if (!string.Equals(encoding, "utf8", StringComparison.Ordinal) &&
            !string.Equals(encoding, "int32_le", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"LMDB relation artifact has unsupported value encoding '{encoding}': {manifestPath}");
        }
    }
}

public static class LmdbRelationArtifactReader
{
    public static LmdbRelationArtifactManifest LoadManifest(string manifestPath)
    {
        if (manifestPath is null) throw new ArgumentNullException(nameof(manifestPath));
        var manifest = JsonSerializer.Deserialize<LmdbRelationArtifactManifest>(File.ReadAllText(manifestPath, Encoding.UTF8))
            ?? throw new InvalidDataException($"LMDB relation artifact manifest is empty: {manifestPath}");
        manifest.Validate(manifestPath);
        return manifest;
    }

    public static string ResolveEnvironmentPath(LmdbRelationArtifactManifest manifest, string manifestPath)
    {
        var environmentPath = manifest.EnvironmentPath;
        return Path.IsPathRooted(environmentPath)
            ? environmentPath
            : Path.Combine(Path.GetDirectoryName(Path.GetFullPath(manifestPath)) ?? ".", environmentPath);
    }
}

public sealed class LmdbRelationProvider : IIndexedRelationProvider, IRelationCardinalityProvider
{
    private readonly IRelationProvider? _fallback;
    private readonly Dictionary<PredicateId, string> _manifestPaths = new();

    public LmdbRelationProvider(IRelationProvider? fallback = null)
    {
        _fallback = fallback;
    }

    public void RegisterArtifact(PredicateId predicate, string manifestPath)
    {
        if (manifestPath is null) throw new ArgumentNullException(nameof(manifestPath));
        var manifest = LmdbRelationArtifactReader.LoadManifest(manifestPath);
        if (!string.Equals(manifest.PredicateName, predicate.Name, StringComparison.Ordinal) ||
            manifest.Arity != predicate.Arity)
        {
            throw new InvalidDataException($"LMDB relation artifact manifest describes {manifest.PredicateName}/{manifest.Arity}, not {predicate}.");
        }

        _manifestPaths[predicate] = manifestPath;
    }

    public IEnumerable<object[]> GetFacts(PredicateId predicate)
    {
        if (!_manifestPaths.TryGetValue(predicate, out var manifestPath))
        {
            return _fallback?.GetFacts(predicate) ?? Array.Empty<object[]>();
        }

        return ReadRows(manifestPath);
    }

    public bool TryLookupFacts(PredicateId predicate, int columnIndex, IEnumerable<object> keys, out IEnumerable<object[]> facts)
    {
        if (columnIndex == 0 && _manifestPaths.TryGetValue(predicate, out var manifestPath))
        {
            facts = LookupRows(manifestPath, keys);
            return true;
        }

        if (_fallback is IIndexedRelationProvider indexedFallback &&
            indexedFallback.TryLookupFacts(predicate, columnIndex, keys, out facts))
        {
            return true;
        }

        facts = default!;
        return false;
    }

    public bool TryGetRelationCardinality(PredicateId predicate, out long rowCount)
    {
        if (_manifestPaths.TryGetValue(predicate, out var manifestPath))
        {
            rowCount = LmdbRelationArtifactReader.LoadManifest(manifestPath).RowCount;
            return true;
        }

        if (_fallback is IRelationCardinalityProvider cardinalityFallback &&
            cardinalityFallback.TryGetRelationCardinality(predicate, out rowCount))
        {
            return true;
        }

        rowCount = 0;
        return false;
    }

    private static IEnumerable<object[]> ReadRows(string manifestPath)
    {
        var manifest = LmdbRelationArtifactReader.LoadManifest(manifestPath);
        using var env = OpenEnvironment(manifest, manifestPath);
        using var tx = env.BeginTransaction(TransactionBeginFlags.ReadOnly);
        using var db = OpenDatabase(tx, manifest);
        using var cursor = tx.CreateCursor(db);

        for (var current = cursor.First();
             current.resultCode == MDBResultCode.Success;
             current = cursor.Next())
        {
            yield return DecodeRow(manifest, current.key.AsSpan(), current.value.AsSpan());
        }
    }

    private static IEnumerable<object[]> LookupRows(string manifestPath, IEnumerable<object> keys)
    {
        var manifest = LmdbRelationArtifactReader.LoadManifest(manifestPath);
        using var env = OpenEnvironment(manifest, manifestPath);
        using var tx = env.BeginTransaction(TransactionBeginFlags.ReadOnly);
        using var db = OpenDatabase(tx, manifest);
        using var cursor = tx.CreateCursor(db);

        foreach (var key in keys)
        {
            var keyBytes = EncodeValue(key, manifest.KeyEncoding);
            if (manifest.DupSort)
            {
                var result = cursor.Set(keyBytes);
                if (result != MDBResultCode.Success)
                {
                    continue;
                }

                do
                {
                    var current = cursor.GetCurrent();
                    yield return DecodeRow(manifest, current.key.AsSpan(), current.value.AsSpan());
                }
                while (cursor.NextDuplicate().resultCode == MDBResultCode.Success);
            }
            else
            {
                var (resultCode, currentKey, currentValue) = tx.Get(db, keyBytes);
                if (resultCode == MDBResultCode.Success)
                {
                    yield return DecodeRow(manifest, currentKey.AsSpan(), currentValue.AsSpan());
                }
            }
        }
    }

    private static LightningEnvironment OpenEnvironment(LmdbRelationArtifactManifest manifest, string manifestPath)
    {
        var env = new LightningEnvironment(LmdbRelationArtifactReader.ResolveEnvironmentPath(manifest, manifestPath))
        {
            MaxDatabases = string.IsNullOrWhiteSpace(manifest.DatabaseName) ? 0 : 4,
        };
        env.Open(EnvironmentOpenFlags.ReadOnly);
        return env;
    }

    private static LightningDatabase OpenDatabase(LightningTransaction tx, LmdbRelationArtifactManifest manifest)
    {
        return string.IsNullOrWhiteSpace(manifest.DatabaseName)
            ? tx.OpenDatabase()
            : tx.OpenDatabase(manifest.DatabaseName);
    }

    private static object[] DecodeRow(LmdbRelationArtifactManifest manifest, ReadOnlySpan<byte> key, ReadOnlySpan<byte> value)
    {
        return new[]
        {
            DecodeValue(key, manifest.KeyEncoding),
            DecodeValue(value, manifest.ValueEncoding),
        };
    }

    private static object DecodeValue(ReadOnlySpan<byte> bytes, string encoding)
    {
        return encoding switch
        {
            "int32_le" when bytes.Length == 4 => BinaryPrimitives.ReadInt32LittleEndian(bytes),
            "utf8" => Encoding.UTF8.GetString(bytes),
            _ => Convert.ToHexString(bytes),
        };
    }

    private static byte[] EncodeValue(object value, string encoding)
    {
        return encoding switch
        {
            "int32_le" => EncodeInt32(value),
            "utf8" => Encoding.UTF8.GetBytes(Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty),
            _ => throw new InvalidDataException($"Unsupported LMDB relation artifact encoding '{encoding}'."),
        };
    }

    private static byte[] EncodeInt32(object value)
    {
        var buffer = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(buffer, Convert.ToInt32(value, CultureInfo.InvariantCulture));
        return buffer;
    }
}

public sealed class LmdbRelationArtifactProviderFactory : IRelationArtifactProviderFactory
{
    public bool TryOpen(
        PredicateId predicate,
        string manifestPath,
        IRelationProvider? fallback,
        out RelationArtifactProviderOpenResult result)
    {
        if (manifestPath is null) throw new ArgumentNullException(nameof(manifestPath));
        var manifest = LmdbRelationArtifactReader.LoadManifest(manifestPath);
        if (!string.Equals(manifest.Format, LmdbRelationArtifactManifest.CurrentFormat, StringComparison.Ordinal))
        {
            result = default!;
            return false;
        }

        var provider = new LmdbRelationProvider(fallback);
        provider.RegisterArtifact(predicate, manifestPath);
        result = new RelationArtifactProviderOpenResult(provider, "lmdb_artifact");
        return true;
    }
}
