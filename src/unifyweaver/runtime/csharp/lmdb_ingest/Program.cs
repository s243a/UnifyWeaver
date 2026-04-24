// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (s243a)
//
// lmdb_ingest — C# consumer for the UnifyWeaver streaming glue.
//
// Reads TSV records from stdin, filters/projects per env-var schema,
// writes to an LMDB database via LightningDB. Mirrors the contract of
// src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py so the
// two implementations produce byte-identical LMDB output from the
// same input.
//
// Environment configuration (identical to the Python consumer):
//   UW_LMDB_PATH       (required)
//   UW_LMDB_MAP_SIZE   (default 1 GB)
//   UW_LMDB_DBNAME     (default unnamed; dupsort forces "main")
//   UW_LMDB_DUPSORT    ("1" allows multiple values per key)
//   UW_FILTER_COL, UW_FILTER_VAL
//   UW_KEY_COL (default 0), UW_VAL_COL (default 1)
//   UW_KEY_ENCODING, UW_VAL_ENCODING ("int32_le" | "utf8", default "utf8")
//   UW_BATCH_SIZE (default 10000)

using System;
using System.Buffers.Binary;
using System.IO;
using System.Text;
using LightningDB;

static class Program
{
    static string? Env(string name) => Environment.GetEnvironmentVariable(name);

    static int EnvInt(string name, int dflt) =>
        int.TryParse(Env(name), out var v) ? v : dflt;

    static long EnvLong(string name, long dflt) =>
        long.TryParse(Env(name), out var v) ? v : dflt;

    static int? EnvOptInt(string name) =>
        int.TryParse(Env(name), out var v) ? v : null;

    static bool EnvFlag(string name) => Env(name) == "1";

    static int Main(string[] args)
    {
        var lmdbPath = Env("UW_LMDB_PATH");
        if (string.IsNullOrEmpty(lmdbPath))
        {
            Console.Error.WriteLine("ingest_to_lmdb: UW_LMDB_PATH is required");
            return 2;
        }

        var mapSize  = EnvLong("UW_LMDB_MAP_SIZE", 1L << 30);
        var dbName   = Env("UW_LMDB_DBNAME");
        var dupsort  = EnvFlag("UW_LMDB_DUPSORT");
        var filterColVal = EnvOptInt("UW_FILTER_COL");
        var filterVal    = Env("UW_FILTER_VAL");
        var keyCol   = EnvInt("UW_KEY_COL", 0);
        var valCol   = EnvInt("UW_VAL_COL", 1);
        var keyEnc   = Env("UW_KEY_ENCODING") ?? "utf8";
        var valEnc   = Env("UW_VAL_ENCODING") ?? "utf8";
        var batchSize = EnvInt("UW_BATCH_SIZE", 10_000);

        if (filterColVal.HasValue && string.IsNullOrEmpty(filterVal))
        {
            Console.Error.WriteLine(
                "ingest_to_lmdb: UW_FILTER_COL set but UW_FILTER_VAL is not");
            return 2;
        }

        // dupsort requires a named sub-DB; create one implicitly if needed.
        if (dupsort && string.IsNullOrEmpty(dbName)) dbName = "main";

        Directory.CreateDirectory(lmdbPath);
        using var env = new LightningEnvironment(lmdbPath)
        {
            MapSize = mapSize,
            MaxDatabases = string.IsNullOrEmpty(dbName) ? 0 : 4,
        };
        env.Open();

        var dbFlags = DatabaseOpenFlags.Create;
        if (dupsort) dbFlags |= DatabaseOpenFlags.DuplicatesSort;

        int totalIn = 0, totalWritten = 0, skipped = 0;

        var tx = env.BeginTransaction();
        var db = string.IsNullOrEmpty(dbName)
            ? tx.OpenDatabase(configuration: new DatabaseConfiguration { Flags = dbFlags })
            : tx.OpenDatabase(dbName, new DatabaseConfiguration { Flags = dbFlags });

        try
        {
            string? line;
            while ((line = Console.In.ReadLine()) != null)
            {
                totalIn++;
                var cols = line.Split('\t');

                if (filterColVal.HasValue)
                {
                    if (filterColVal.Value >= cols.Length) { skipped++; continue; }
                    if (TsvUnescape(cols[filterColVal.Value]) != filterVal) continue;
                }

                if (keyCol >= cols.Length || valCol >= cols.Length) { skipped++; continue; }

                byte[] keyBytes, valBytes;
                try
                {
                    keyBytes = Encode(cols[keyCol], keyEnc);
                    valBytes = Encode(cols[valCol], valEnc);
                }
                catch (FormatException)
                {
                    skipped++;
                    continue;
                }

                tx.Put(db, keyBytes, valBytes);
                totalWritten++;

                if (totalWritten % batchSize == 0)
                {
                    tx.Commit();
                    tx.Dispose();
                    tx = env.BeginTransaction();
                    db = string.IsNullOrEmpty(dbName)
                        ? tx.OpenDatabase(configuration: new DatabaseConfiguration { Flags = dbFlags })
                        : tx.OpenDatabase(dbName, new DatabaseConfiguration { Flags = dbFlags });
                }
            }

            tx.Commit();
            tx.Dispose();
        }
        catch
        {
            try { tx.Abort(); tx.Dispose(); } catch { }
            throw;
        }

        Console.Error.WriteLine(
            $"ingest_to_lmdb (csharp): in={totalIn} written={totalWritten} skipped={skipped}");
        return 0;
    }

    static byte[] Encode(string value, string encoding)
    {
        switch (encoding)
        {
            case "int32_le":
            {
                var i = int.Parse(value);
                var buf = new byte[4];
                BinaryPrimitives.WriteInt32LittleEndian(buf, i);
                return buf;
            }
            case "utf8":
                return Encoding.UTF8.GetBytes(TsvUnescape(value));
            default:
                throw new ArgumentException($"unknown encoding: {encoding}");
        }
    }

    // Reverse the TSV escapes emitted by mysql_stream (Rust) and
    // parse_inserts.awk. Matches the Python consumer's tsv_unescape
    // exactly so all three implementations agree byte-for-byte.
    static string TsvUnescape(string field)
    {
        if (field.IndexOf('\\') < 0) return field;
        var sb = new StringBuilder(field.Length);
        int i = 0, n = field.Length;
        while (i < n)
        {
            var c = field[i];
            if (c == '\\' && i + 1 < n)
            {
                var nxt = field[i + 1];
                switch (nxt)
                {
                    case '\\': sb.Append('\\'); i += 2; break;
                    case 't':  sb.Append('\t'); i += 2; break;
                    case 'n':  sb.Append('\n'); i += 2; break;
                    case 'r':  sb.Append('\r'); i += 2; break;
                    case 'N':  sb.Append("\\N"); i += 2; break;  // NULL sentinel
                    case 'x':
                        if (i + 3 < n &&
                            int.TryParse(field.Substring(i + 2, 2),
                                System.Globalization.NumberStyles.HexNumber,
                                null, out var b))
                        {
                            sb.Append((char)b);
                            i += 4;
                        }
                        else { sb.Append(c); i++; }
                        break;
                    default: sb.Append(c); i++; break;
                }
            }
            else { sb.Append(c); i++; }
        }
        return sb.ToString();
    }
}
