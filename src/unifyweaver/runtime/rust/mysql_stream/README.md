# mysql_stream — MySQL INSERT streaming tokenizer

Leaf primitive for the UnifyWeaver cross-target streaming glue. Parses
MySQL dumps (optionally gzipped) and yields every INSERT row as
`Vec<Field>` — no filtering, no column selection.

See `docs/design/cross-target-glue/streaming-pipelines/` for the design.

## Library usage

```rust
use mysql_stream::{iter_mysql_rows, Field};

for row in iter_mysql_rows("simplewiki.sql.gz")? {
    // row: Vec<Field> — all columns of one INSERT row
    if let Some(Field::Str(t)) = row.get(4) {
        if t == b"subcat" {
            // ...
        }
    }
}
```

## Binary usage

```
mysql_stream <path-to-dump.sql[.gz]>
```

Writes every INSERT row as TAB-separated TSV to stdout.

- Integers: decimal
- Strings/bytes: printable ASCII passes through; `\\` `\t` `\n` `\r`
  escape as two-byte sequences; non-ASCII bytes as `\xNN`
- `NULL`: `\N`

## Validation

End-to-end tested against `simplewiki-latest-categorylinks.sql.gz`
(27 MB gzipped, ~230 MB raw). Output row counts match the SQLite
ground truth exactly:

| cl_type | mysql_stream | SQLite ground truth |
|---------|-------------:|--------------------:|
| (total) |    2,206,045 |           2,206,045 |
| subcat  |      297,283 |             297,283 |
| page    |    1,908,512 |           1,908,512 |
| file    |          250 |                 250 |

Throughput: ~28 MB/s of gzipped input on a single core.
