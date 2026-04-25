use lmdb::{
    Cursor, Database, DatabaseFlags, Environment, RoTransaction, RwTransaction, Transaction,
    WriteFlags,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

type DynError = Box<dyn Error>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ArtifactDeclaration {
    source: String,
    mode: String,
    kind: String,
    format: String,
    access_contracts: Vec<String>,
    options: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ArtifactManifest {
    predicate: String,
    resolved_mode: String,
    artifact_kind: String,
    format_version: u32,
    physical_format: String,
    exactness: String,
    source_path: String,
    source_hash: String,
    schema_hash: String,
    db_name: String,
    dupsort: bool,
    key_encoding: String,
    value_encoding: String,
    access: Vec<String>,
    target_capabilities: Vec<String>,
    declaration: ArtifactDeclaration,
    files: Vec<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("lmdb_relation_artifact: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), DynError> {
    let args: Vec<String> = env::args().collect();
    match args.as_slice() {
        [_prog, cmd, predicate, tsv_path, artifact_dir] if cmd == "build" => {
            build_command(predicate, Path::new(tsv_path), Path::new(artifact_dir), false)
        }
        [_prog, cmd, predicate, tsv_path, artifact_dir, flag]
            if cmd == "build" && flag == "--dupsort" =>
        {
            build_command(predicate, Path::new(tsv_path), Path::new(artifact_dir), true)
        }
        [_prog, cmd, artifact_dir, predicate, key] if cmd == "get" => {
            get_command(Path::new(artifact_dir), predicate, key)
        }
        [_prog, cmd, artifact_dir, predicate] if cmd == "scan" => {
            scan_command(Path::new(artifact_dir), predicate)
        }
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage:
  lmdb_relation_artifact build <predicate> <tsv-path> <artifact-dir> [--dupsort]
  lmdb_relation_artifact get <artifact-dir> <predicate> <key>
  lmdb_relation_artifact scan <artifact-dir> <predicate>"
    );
}

fn build_command(
    predicate: &str,
    tsv_path: &Path,
    artifact_dir: &Path,
    dupsort: bool,
) -> Result<(), DynError> {
    let rows = read_tsv_rows(tsv_path)?;
    fs::create_dir_all(artifact_dir)?;

    let env = open_env(artifact_dir)?;
    let flags = if dupsort {
        DatabaseFlags::DUP_SORT
    } else {
        DatabaseFlags::empty()
    };
    let db = env.create_db(Some(predicate), flags)?;

    {
        let mut txn = env.begin_rw_txn()?;
        write_rows(&mut txn, db, &rows)?;
        txn.commit()?;
    }

    let manifest = ArtifactManifest {
        predicate: predicate.to_string(),
        resolved_mode: "artifact".to_string(),
        artifact_kind: "lmdb_exact_relation_v1".to_string(),
        format_version: 1,
        physical_format: "lmdb_dupsort_btree".to_string(),
        exactness: "exact".to_string(),
        source_path: tsv_path.display().to_string(),
        source_hash: file_sha256(tsv_path)?,
        schema_hash: "sha256:two_column_utf8_relation_v1".to_string(),
        db_name: predicate.to_string(),
        dupsort,
        key_encoding: "utf8".to_string(),
        value_encoding: "utf8".to_string(),
        access: if dupsort {
            vec![
                "scan".to_string(),
                "arg1_lookup".to_string(),
                "arg1_multi_lookup".to_string(),
            ]
        } else {
            vec!["scan".to_string(), "arg1_lookup".to_string()]
        },
        target_capabilities: vec![
            "mmap".to_string(),
            "mvcc_readers".to_string(),
            "little_endian".to_string(),
        ],
        declaration: ArtifactDeclaration {
            source: "prototype".to_string(),
            mode: "artifact".to_string(),
            kind: if dupsort {
                "adjacency_index".to_string()
            } else {
                "exact_hash_index".to_string()
            },
            format: "lmdb_dupsort_btree".to_string(),
            access_contracts: if dupsort {
                vec![
                    "scan".to_string(),
                    "arg_position_lookup(1)".to_string(),
                    "grouped_values_lookup([2])".to_string(),
                ]
            } else {
                vec![
                    "scan".to_string(),
                    "arg_position_lookup(1)".to_string(),
                    "exact_key_lookup".to_string(),
                ]
            },
            options: if dupsort {
                vec![
                    "key([1])".to_string(),
                    "values([2])".to_string(),
                    "dupsort(true)".to_string(),
                ]
            } else {
                vec!["key([1])".to_string(), "values([2])".to_string()]
            },
        },
        files: vec![
            "data.mdb".to_string(),
            "lock.mdb".to_string(),
            "manifest.json".to_string(),
        ],
    };
    write_manifest(artifact_dir, &manifest)?;

    println!(
        "built predicate={} rows={} dupsort={} artifact_dir={}",
        predicate,
        rows.len(),
        dupsort,
        artifact_dir.display()
    );
    Ok(())
}

fn get_command(artifact_dir: &Path, predicate: &str, key: &str) -> Result<(), DynError> {
    let manifest = load_manifest(artifact_dir)?;
    let env = open_env_readonly(artifact_dir)?;
    let db = env.open_db(Some(predicate))?;
    let txn = env.begin_ro_txn()?;
    let values = get_values(&txn, db, key.as_bytes(), manifest.dupsort)?;
    for value in values {
        println!("{key}\t{}", String::from_utf8_lossy(&value));
    }
    Ok(())
}

fn scan_command(artifact_dir: &Path, predicate: &str) -> Result<(), DynError> {
    let env = open_env_readonly(artifact_dir)?;
    let db = env.open_db(Some(predicate))?;
    let txn = env.begin_ro_txn()?;
    let mut cursor = txn.open_ro_cursor(db)?;
    for (key, value) in cursor.iter_start() {
        println!(
            "{}\t{}",
            String::from_utf8_lossy(key),
            String::from_utf8_lossy(value)
        );
    }
    Ok(())
}

fn open_env(path: &Path) -> Result<Environment, DynError> {
    Ok(Environment::new()
        .set_max_dbs(16)
        .set_map_size(64 * 1024 * 1024)
        .open(path)?)
}

fn open_env_readonly(path: &Path) -> Result<Environment, DynError> {
    Ok(Environment::new()
        .set_max_dbs(16)
        .set_map_size(64 * 1024 * 1024)
        .open(path)?)
}

fn read_tsv_rows(path: &Path) -> Result<Vec<(String, String)>, DynError> {
    let input = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_no, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let (key, value) = line.split_once('\t').ok_or_else(|| {
            let mut msg = String::new();
            let _ = write!(
                &mut msg,
                "invalid TSV row at {}:{}; expected two columns",
                path.display(),
                line_no + 1
            );
            msg
        })?;
        rows.push((key.to_string(), value.to_string()));
    }
    Ok(rows)
}

fn write_rows(
    txn: &mut RwTransaction<'_>,
    db: Database,
    rows: &[(String, String)],
) -> Result<(), DynError> {
    for (key, value) in rows {
        txn.put(db, key, value, WriteFlags::empty())?;
    }
    Ok(())
}

fn get_values(
    txn: &RoTransaction<'_>,
    db: Database,
    key: &[u8],
    dupsort: bool,
) -> Result<Vec<Vec<u8>>, DynError> {
    if !dupsort {
        return match txn.get(db, &key.to_vec()) {
            Ok(value) => Ok(vec![value.to_vec()]),
            Err(lmdb::Error::NotFound) => Ok(Vec::new()),
            Err(err) => Err(Box::new(err)),
        };
    }
    let mut cursor = txn.open_ro_cursor(db)?;
    let mut values = Vec::new();
    let key_buf = key.to_vec();
    for (_k, value) in cursor.iter_dup_of(&key_buf)? {
        values.push(value.to_vec());
    }
    Ok(values)
}

fn write_manifest(artifact_dir: &Path, manifest: &ArtifactManifest) -> Result<(), DynError> {
    let path = artifact_dir.join("manifest.json");
    let content = serde_json::to_string_pretty(manifest)?;
    fs::write(path, content)?;
    Ok(())
}

fn load_manifest(artifact_dir: &Path) -> Result<ArtifactManifest, DynError> {
    let path = artifact_dir.join("manifest.json");
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn file_sha256(path: &Path) -> Result<String, DynError> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn build_and_get_roundtrip() -> Result<(), DynError> {
        let temp_root = unique_temp_dir("lmdb_artifact_roundtrip");
        let tsv_path = temp_root.join("edge.tsv");
        let artifact_dir = temp_root.join("artifact");
        fs::create_dir_all(&temp_root)?;
        fs::write(&tsv_path, "a\t1\nb\t2\n")?;

        build_command("edge/2", &tsv_path, &artifact_dir, false)?;

        let manifest = load_manifest(&artifact_dir)?;
        assert_eq!(manifest.predicate, "edge/2");
        assert!(!manifest.dupsort);
        assert_eq!(manifest.resolved_mode, "artifact");
        assert_eq!(manifest.physical_format, "lmdb_dupsort_btree");
        assert_eq!(manifest.declaration.kind, "exact_hash_index");
        assert!(manifest
            .declaration
            .access_contracts
            .contains(&"exact_key_lookup".to_string()));
        assert_eq!(manifest.access, vec!["scan", "arg1_lookup"]);

        let env = open_env_readonly(&artifact_dir)?;
        let db = env.open_db(Some("edge/2"))?;
        let txn = env.begin_ro_txn()?;
        let values = get_values(&txn, db, b"a", manifest.dupsort)?;
        assert_eq!(values, vec![b"1".to_vec()]);

        let mut cursor = txn.open_ro_cursor(db)?;
        let rows: Vec<(Vec<u8>, Vec<u8>)> = cursor
            .iter_start()
            .map(|(k, v)| (k.to_vec(), v.to_vec()))
            .collect();
        assert_eq!(rows.len(), 2);

        fs::remove_dir_all(temp_root)?;
        Ok(())
    }

    #[test]
    fn build_dupsort_roundtrip() -> Result<(), DynError> {
        let temp_root = unique_temp_dir("lmdb_artifact_dupsort");
        let tsv_path = temp_root.join("edge.tsv");
        let artifact_dir = temp_root.join("artifact");
        fs::create_dir_all(&temp_root)?;
        fs::write(&tsv_path, "a\t1\na\t2\n")?;

        build_command("edge/2", &tsv_path, &artifact_dir, true)?;

        let manifest = load_manifest(&artifact_dir)?;
        assert!(manifest.dupsort);
        assert_eq!(manifest.declaration.kind, "adjacency_index");
        assert!(manifest.access.contains(&"arg1_multi_lookup".to_string()));
        assert!(manifest
            .declaration
            .options
            .contains(&"dupsort(true)".to_string()));

        let env = open_env_readonly(&artifact_dir)?;
        let db = env.open_db(Some("edge/2"))?;
        let txn = env.begin_ro_txn()?;
        let values = get_values(&txn, db, b"a", manifest.dupsort)?;
        assert_eq!(values, vec![b"1".to_vec(), b"2".to_vec()]);

        fs::remove_dir_all(temp_root)?;
        Ok(())
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        env::temp_dir().join(format!("{prefix}_{nanos}"))
    }
}
