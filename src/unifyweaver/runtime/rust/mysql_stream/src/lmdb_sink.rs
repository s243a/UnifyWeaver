// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2025 John William Creighton (s243a)
//
// mysql_stream_lmdb — direct LMDB sink for MediaWiki categorylinks dumps.
//
// This keeps the parser path in Rust and writes C# query-runtime compatible
// LMDB relation artifacts without a Python or C# row-sink subprocess.

use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use lmdb::{DatabaseFlags, Environment, Transaction, WriteFlags};
use mysql_stream::{iter_mysql_rows, Field};

type DynError = Box<dyn Error>;

const DEFAULT_MAP_SIZE: usize = 1 << 30;
const DEFAULT_BATCH_SIZE: usize = 50_000;
const DB_NAME: &str = "main";

#[derive(Debug, Clone, PartialEq, Eq)]
struct Config {
    dump_path: PathBuf,
    lmdb_path: PathBuf,
    manifest_path: PathBuf,
    max_edges: Option<usize>,
    map_size: usize,
    batch_size: usize,
    fixture_tsv_path: Option<PathBuf>,
    stats_path: Option<PathBuf>,
    refresh: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SinkStats {
    rows_scanned: usize,
    edges_written: usize,
}

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1)) {
        Ok(stats) => {
            eprintln!(
                "mysql_stream_lmdb: scanned={} written={}",
                stats.rows_scanned, stats.edges_written
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("mysql_stream_lmdb: {err}");
            ExitCode::from(1)
        }
    }
}

fn run<I>(args: I) -> Result<SinkStats, DynError>
where
    I: IntoIterator<Item = OsString>,
{
    let config = parse_args(args)?;
    sink_categorylinks_to_lmdb(&config)
}

fn parse_args<I>(args: I) -> Result<Config, DynError>
where
    I: IntoIterator<Item = OsString>,
{
    let mut positional = Vec::new();
    let mut manifest_path = None;
    let mut max_edges = None;
    let mut map_size = DEFAULT_MAP_SIZE;
    let mut batch_size = DEFAULT_BATCH_SIZE;
    let mut fixture_tsv_path = None;
    let mut stats_path = None;
    let mut refresh = false;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.to_string_lossy().as_ref() {
            "--manifest" => {
                manifest_path = Some(PathBuf::from(
                    args.next().ok_or("--manifest requires a path")?,
                ));
            }
            "--max-edges" => {
                max_edges = Some(parse_positive_usize(
                    &args.next().ok_or("--max-edges requires a value")?,
                    "--max-edges",
                )?);
            }
            "--map-size" => {
                map_size = parse_positive_usize(
                    &args.next().ok_or("--map-size requires a value")?,
                    "--map-size",
                )?;
            }
            "--batch-size" => {
                batch_size = parse_positive_usize(
                    &args.next().ok_or("--batch-size requires a value")?,
                    "--batch-size",
                )?;
            }
            "--fixture-tsv" => {
                fixture_tsv_path = Some(PathBuf::from(
                    args.next().ok_or("--fixture-tsv requires a path")?,
                ));
            }
            "--stats" => {
                stats_path = Some(PathBuf::from(args.next().ok_or("--stats requires a path")?));
            }
            "--refresh" => refresh = true,
            "--help" | "-h" => return Err(usage().into()),
            value if value.starts_with("--") => {
                return Err(format!("unknown option: {value}\n{}", usage()).into());
            }
            _ => positional.push(PathBuf::from(arg)),
        }
    }

    if positional.len() != 2 {
        return Err(usage().into());
    }

    let dump_path = positional.remove(0);
    let lmdb_path = positional.remove(0);
    let manifest_path = manifest_path.unwrap_or_else(|| default_manifest_path(&lmdb_path));
    Ok(Config {
        dump_path,
        lmdb_path,
        manifest_path,
        max_edges,
        map_size,
        batch_size,
        fixture_tsv_path,
        stats_path,
        refresh,
    })
}

fn parse_positive_usize(value: &OsString, flag: &str) -> Result<usize, DynError> {
    let parsed = value.to_string_lossy().parse::<usize>()?;
    if parsed == 0 {
        return Err(format!("{flag} must be positive").into());
    }
    Ok(parsed)
}

fn usage() -> String {
    "usage: mysql_stream_lmdb <dump.sql[.gz]> <lmdb-dir> [--manifest <path>] [--max-edges N] [--map-size BYTES] [--batch-size N] [--fixture-tsv <path>] [--stats <path>] [--refresh]".to_string()
}

fn default_manifest_path(lmdb_path: &Path) -> PathBuf {
    let mut manifest = lmdb_path.as_os_str().to_os_string();
    manifest.push(".manifest.json");
    PathBuf::from(manifest)
}

fn sink_categorylinks_to_lmdb(config: &Config) -> Result<SinkStats, DynError> {
    if config.lmdb_path.exists() && config.manifest_path.exists() && !config.refresh {
        let stats = SinkStats {
            rows_scanned: 0,
            edges_written: manifest_row_count(&config.manifest_path).unwrap_or(0),
        };
        write_stats_if_requested(config, &stats)?;
        return Ok(stats);
    }

    if config.lmdb_path.exists() {
        if !config.refresh {
            return Err(format!(
                "LMDB directory exists without reusable manifest; pass --refresh to rebuild: {}",
                config.lmdb_path.display()
            )
            .into());
        }
        fs::remove_dir_all(&config.lmdb_path)?;
    }
    if config.manifest_path.exists() {
        if !config.refresh {
            return Err(format!(
                "manifest exists without reusable LMDB directory; pass --refresh to rebuild: {}",
                config.manifest_path.display()
            )
            .into());
        }
        fs::remove_file(&config.manifest_path)?;
    }

    fs::create_dir_all(&config.lmdb_path)?;
    if let Some(parent) = config.manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let env = Environment::new()
        .set_max_dbs(4)
        .set_map_size(config.map_size)
        .open(&config.lmdb_path)?;
    let db = env.create_db(Some(DB_NAME), DatabaseFlags::DUP_SORT)?;
    let mut txn = env.begin_rw_txn()?;
    let mut fixture_tsv = open_fixture_tsv(config.fixture_tsv_path.as_deref())?;
    let mut rows_scanned = 0usize;
    let mut edges_written = 0usize;

    let dump_path = config
        .dump_path
        .to_str()
        .ok_or("dump path must be valid UTF-8")?;
    for row in iter_mysql_rows(dump_path)? {
        rows_scanned += 1;
        let Some((child, parent)) = categorylinks_subcat_edge(&row) else {
            continue;
        };
        txn.put(
            db,
            &child.as_bytes(),
            &parent.as_bytes(),
            WriteFlags::empty(),
        )?;
        if let Some(writer) = fixture_tsv.as_mut() {
            writeln!(writer, "{child}\t{parent}")?;
        }
        edges_written += 1;

        if edges_written % config.batch_size == 0 {
            txn.commit()?;
            txn = env.begin_rw_txn()?;
        }
        if config
            .max_edges
            .is_some_and(|max_edges| edges_written >= max_edges)
        {
            break;
        }
    }
    txn.commit()?;
    if let Some(mut writer) = fixture_tsv {
        writer.flush()?;
    }

    write_csharp_query_manifest(config, edges_written)?;
    let stats = SinkStats {
        rows_scanned,
        edges_written,
    };
    write_stats_if_requested(config, &stats)?;
    Ok(stats)
}

fn open_fixture_tsv(path: Option<&Path>) -> Result<Option<BufWriter<fs::File>>, DynError> {
    let Some(path) = path else {
        return Ok(None);
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(fs::File::create(path)?);
    writer.write_all(b"child\tparent\n")?;
    Ok(Some(writer))
}

fn categorylinks_subcat_edge(row: &[Field]) -> Option<(String, String)> {
    if !matches!(row.get(4).and_then(Field::as_str), Some("subcat")) {
        return None;
    }
    let child = field_i64(row.get(0)?)?;
    let parent = field_i64(row.get(6)?)?;
    Some((child.to_string(), parent.to_string()))
}

fn field_i64(field: &Field) -> Option<i64> {
    match field {
        Field::Int(value) => Some(*value),
        Field::Str(bytes) => std::str::from_utf8(bytes).ok()?.parse().ok(),
        Field::Null => None,
    }
}

fn write_csharp_query_manifest(config: &Config, row_count: usize) -> Result<(), DynError> {
    let environment_path = manifest_environment_path(&config.lmdb_path, &config.manifest_path);
    let source_length = fs::metadata(&config.dump_path).ok().map(|meta| meta.len());
    let source_path = config.dump_path.display().to_string();
    let source_length_json = source_length
        .map(|length| length.to_string())
        .unwrap_or_else(|| "null".to_string());
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"Format\": \"unifyweaver.lmdb_relation.v1\",\n",
            "  \"Version\": 1,\n",
            "  \"Backend\": \"lmdb\",\n",
            "  \"PredicateName\": \"category_parent\",\n",
            "  \"Arity\": 2,\n",
            "  \"EnvironmentPath\": \"{}\",\n",
            "  \"DatabaseName\": \"{}\",\n",
            "  \"DupSort\": true,\n",
            "  \"KeyEncoding\": \"utf8\",\n",
            "  \"ValueEncoding\": \"utf8\",\n",
            "  \"RowCount\": {},\n",
            "  \"SourcePath\": \"{}\",\n",
            "  \"SourceLength\": {}\n",
            "}}\n"
        ),
        json_escape(&environment_path),
        DB_NAME,
        row_count,
        json_escape(&source_path),
        source_length_json
    );
    fs::write(&config.manifest_path, manifest)?;
    Ok(())
}

fn write_stats_if_requested(config: &Config, stats: &SinkStats) -> Result<(), DynError> {
    let Some(path) = config.stats_path.as_deref() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
        format!(
            "{{\n  \"rows_scanned\": {},\n  \"edges_written\": {}\n}}\n",
            stats.rows_scanned, stats.edges_written
        ),
    )?;
    Ok(())
}

fn manifest_environment_path(lmdb_path: &Path, manifest_path: &Path) -> String {
    let lmdb_parent = lmdb_path.parent().map(Path::to_path_buf);
    let manifest_parent = manifest_path.parent().map(Path::to_path_buf);
    if lmdb_parent == manifest_parent {
        return lmdb_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| lmdb_path.display().to_string());
    }
    lmdb_path.display().to_string()
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", ch as u32);
            }
            ch => out.push(ch),
        }
    }
    out
}

fn manifest_row_count(path: &Path) -> Option<usize> {
    let text = fs::read_to_string(path).ok()?;
    let marker = "\"RowCount\"";
    let start = text.find(marker)? + marker.len();
    let after_colon = text[start..].find(':')? + start + 1;
    let digits: String = text[after_colon..]
        .chars()
        .skip_while(|ch| ch.is_whitespace())
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_categorylinks_subcat_edges_from_page_ids() {
        let row = vec![
            Field::Int(10),
            Field::Int(0),
            Field::Str(b"A".to_vec()),
            Field::Null,
            Field::Str(b"subcat".to_vec()),
            Field::Null,
            Field::Int(20),
        ];
        assert_eq!(
            categorylinks_subcat_edge(&row),
            Some(("10".to_string(), "20".to_string()))
        );
    }

    #[test]
    fn ignores_non_subcat_rows() {
        let row = vec![
            Field::Int(10),
            Field::Int(0),
            Field::Str(b"A".to_vec()),
            Field::Null,
            Field::Str(b"page".to_vec()),
            Field::Null,
            Field::Int(20),
        ];
        assert_eq!(categorylinks_subcat_edge(&row), None);
    }

    #[test]
    fn default_manifest_path_appends_manifest_suffix() {
        assert_eq!(
            default_manifest_path(Path::new("/tmp/category_parent.lmdb")),
            PathBuf::from("/tmp/category_parent.lmdb.manifest.json")
        );
    }

    #[test]
    fn manifest_environment_path_is_relative_for_sibling_manifest() {
        assert_eq!(
            manifest_environment_path(
                Path::new("/tmp/fixture/category_parent.lmdb"),
                Path::new("/tmp/fixture/category_parent.lmdb.manifest.json")
            ),
            "category_parent.lmdb"
        );
    }

    #[test]
    fn json_escape_handles_manifest_paths() {
        assert_eq!(json_escape("a\\b\"c"), "a\\\\b\\\"c");
    }

    #[test]
    fn parse_args_accepts_fixture_tsv_and_stats_paths() {
        let config = parse_args([
            OsString::from("dump.sql.gz"),
            OsString::from("category_parent.lmdb"),
            OsString::from("--fixture-tsv"),
            OsString::from("category_parent.tsv"),
            OsString::from("--stats"),
            OsString::from("stats.json"),
            OsString::from("--max-edges"),
            OsString::from("10"),
        ])
        .expect("valid args");
        assert_eq!(
            config.fixture_tsv_path,
            Some(PathBuf::from("category_parent.tsv"))
        );
        assert_eq!(config.stats_path, Some(PathBuf::from("stats.json")));
        assert_eq!(config.max_edges, Some(10));
    }
}
