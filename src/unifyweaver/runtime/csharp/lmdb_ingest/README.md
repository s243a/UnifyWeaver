# lmdb_ingest (C#) — Consumer mirror

C# implementation of the UnifyWeaver streaming-pipeline LMDB consumer.
Parallel to `src/unifyweaver/runtime/python/lmdb_ingest/ingest_to_lmdb.py`
— same environment-variable contract, same TSV input format, same
resulting LMDB output.

## Purpose

Demonstrates **consumer-side target interchangeability**: the same
streaming pipeline produced by any producer target (`mysql_stream`
Rust, `parse_inserts.awk` AWK) can be consumed by either the Python
or the C# implementation. Swapping consumers is a one-line
`declare_target` change in the Prolog pipeline declaration.

## Build

```sh
cd src/unifyweaver/runtime/csharp/lmdb_ingest
dotnet build -c Release
```

## Usage

Reads TSV on stdin; writes LMDB under `UW_LMDB_PATH`. Configured
entirely through environment variables (see `Program.cs` comment
block for the full list).

## Dependencies

- .NET 9.0+ (uses `Console.In`, `BinaryPrimitives`, the standard APIs)
- [LightningDB](https://www.nuget.org/packages/LightningDB) 0.21+
  (managed wrapper around liblmdb)
