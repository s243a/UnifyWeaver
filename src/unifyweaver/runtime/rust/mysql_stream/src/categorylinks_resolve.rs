// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// categorylinks_resolve — multi-dump resolution of MediaWiki categorylinks
// edges into a walkable graph with a consistent ID namespace.
//
// MediaWiki's post-2023 schema separates two ID spaces:
//   cl_from        = page_id        (source page identifier)
//   cl_target_id   = linktarget_id  (indirect reference into the linktarget table)
//
// A single-dump ingest (categorylinks only) mixes these namespaces, so
// multi-hop graph traversal is broken: hop 2 would need to look up the
// parent's page_id, but we only have its linktarget_id.
//
// This module supports three ingest modes:
//
// 1. **Correct (3-dump default)**: categorylinks + linktarget + page.
//    Builds page_id <-> title and linktarget_id -> title maps from the
//    auxiliary dumps (namespace=14 / Category only), then emits edges
//    (cl_from, parent_page_id) where parent_page_id is found by joining
//    cl_target_id -> title -> page_id. Result: fully walkable graph
//    keyed by page_id.
//
// 2. **Compromise (2-dump)**: categorylinks + linktarget. Uses lt_id as
//    the canonical ID space. Categories that appear as cl_target_id
//    somewhere get resolved; cl_from values stay as page_ids. Walkable
//    in the parent direction; leaf categories that never appear as a
//    target are opaque.
//
// 3. **Fallback (1-dump)**: categorylinks only, with --id-method:
//      - raw      : emit (cl_from, cl_target_id) as-is. Multi-hop broken
//                   on new schemas; documented limitation.
//      - position : monotonic intern of every observed raw id into a
//                   single namespace. Stable within one run only.
//      - hash     : deterministic 64-bit hash of (namespace_marker, raw_id)
//                   to merge both namespaces into one. Stable across runs;
//                   collisions possible but vanishingly rare at <10^9 ids.
//
// See docs/design/WAM_LMDB_RESIDENT_INTERNING_PHILOSOPHY.md for the
// existing monotonic-intern design that this composes with.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::{iter_mysql_rows, Field};

/// Namespace constant for Category pages in MediaWiki.
pub const NS_CATEGORY: i64 = 14;

/// How the ingester decides which dumps to consume and how to resolve IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IngestMode {
    /// 3-dump default: categorylinks + linktarget + page. Walkable graph
    /// keyed by page_id. Both auxiliary dumps must be supplied.
    Correct,
    /// 2-dump compromise: categorylinks + linktarget. Walkable graph
    /// keyed by lt_id; cl_from leaves stay as page_ids and are opaque
    /// (cannot be recursed into without a page dump).
    Compromise,
    /// 1-dump fallback: categorylinks only. See [`IdMethod`] for the
    /// strategies for unifying / preserving raw ids.
    Fallback,
}

impl IngestMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "correct" | "3dump" | "3-dump" => Ok(Self::Correct),
            "compromise" | "2dump" | "2-dump" => Ok(Self::Compromise),
            "fallback" | "1dump" | "1-dump" => Ok(Self::Fallback),
            other => Err(format!(
                "unknown --mode value '{other}' (expected: correct, compromise, fallback)"
            )),
        }
    }
}

/// ID generation strategy for the [`IngestMode::Fallback`] single-dump path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdMethod {
    /// Emit raw cl_from / cl_target_id with no transformation. Preserves
    /// the upstream numbers but mixes page_id and linktarget_id spaces
    /// on post-2023 schemas — multi-hop traversal is broken. The only
    /// safe choice for OLD-schema dumps where both sides were page_ids.
    Raw,
    /// Monotonic intern: every observed raw id (regardless of which
    /// column it came from) gets the next sequential synthetic id. Both
    /// namespaces collapse into one. Stable WITHIN a single run; not
    /// stable across runs because input order depends on dump file layout.
    /// This is the safest fallback when correctness of names doesn't
    /// matter but graph structure does.
    Position,
    /// Deterministic hash: `hash64(namespace_marker, raw_id)` produces
    /// a stable 64-bit id across runs. Different namespace markers for
    /// the cl_from and cl_target_id columns ensure they don't accidentally
    /// collide (page_id 5 and lt_id 5 hash to different values). Truncate
    /// to i32 if downstream requires it; collision rate is ~n^2 / 2^32
    /// (negligible below ~10^4 ids, noticeable at 10^6+).
    Hash,
}

impl IdMethod {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "raw" => Ok(Self::Raw),
            "position" | "intern" | "monotonic" => Ok(Self::Position),
            "hash" => Ok(Self::Hash),
            other => Err(format!(
                "unknown --id-method value '{other}' (expected: raw, position, hash)"
            )),
        }
    }
}

/// Build `lt_id -> title` for category-namespace targets only. Reads
/// the entire linktarget dump and keeps only rows where `lt_namespace == 14`.
///
/// Linktarget table layout (post-2023):
///   col 0: lt_id (int)
///   col 1: lt_namespace (int)
///   col 2: lt_title (varbinary)
pub fn build_linktarget_title_map(
    linktarget_dump: &Path,
) -> io::Result<HashMap<i64, Vec<u8>>> {
    let path = linktarget_dump
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "linktarget path not utf-8"))?;
    let mut map = HashMap::new();
    for row in iter_mysql_rows(path)? {
        let Some(ns) = row.get(1).and_then(field_i64) else {
            continue;
        };
        if ns != NS_CATEGORY {
            continue;
        }
        let Some(lt_id) = row.get(0).and_then(field_i64) else {
            continue;
        };
        let Some(title) = row.get(2).and_then(Field::as_bytes) else {
            continue;
        };
        map.insert(lt_id, title.to_vec());
    }
    Ok(map)
}

/// Build `title -> page_id` for category-namespace pages only. Reads the
/// entire page dump and keeps only rows where `page_namespace == 14`.
///
/// Page table layout (early columns, stable since ~2015):
///   col 0: page_id (int)
///   col 1: page_namespace (int)
///   col 2: page_title (varbinary)
pub fn build_page_title_to_id_map(
    page_dump: &Path,
) -> io::Result<HashMap<Vec<u8>, i64>> {
    let path = page_dump
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "page path not utf-8"))?;
    let mut map = HashMap::new();
    for row in iter_mysql_rows(path)? {
        let Some(ns) = row.get(1).and_then(field_i64) else {
            continue;
        };
        if ns != NS_CATEGORY {
            continue;
        }
        let Some(page_id) = row.get(0).and_then(field_i64) else {
            continue;
        };
        let Some(title) = row.get(2).and_then(Field::as_bytes) else {
            continue;
        };
        map.insert(title.to_vec(), page_id);
    }
    Ok(map)
}

/// Resolver state used while streaming categorylinks rows.
///
/// In Correct mode, both maps are populated.
/// In Compromise mode, only `lt_title_map` is populated.
/// In Fallback mode, neither map is populated; [`IdMethod`] decides.
pub struct Resolver {
    pub mode: IngestMode,
    pub id_method: IdMethod,
    pub lt_title_map: HashMap<i64, Vec<u8>>,
    pub page_title_to_id: HashMap<Vec<u8>, i64>,
    /// For [`IdMethod::Position`]: monotonic intern map shared across both
    /// raw-id spaces (page_id and linktarget_id).
    pub intern: HashMap<i64, i64>,
    pub next_id: i64,
    /// Statistics: edges where parent resolution failed (e.g. lt_id not in
    /// linktarget map, or title not in page map).
    pub unresolved_parents: usize,
}

impl Resolver {
    pub fn new(mode: IngestMode, id_method: IdMethod) -> Self {
        Self {
            mode,
            id_method,
            lt_title_map: HashMap::new(),
            page_title_to_id: HashMap::new(),
            intern: HashMap::new(),
            next_id: 0,
            unresolved_parents: 0,
        }
    }

    /// Resolve one categorylinks subcat edge `(cl_from, cl_target_id)`
    /// into the canonical id pair for this ingest mode.
    ///
    /// Returns `None` if the edge cannot be resolved (e.g. the parent's
    /// lt_id is not in the linktarget map). Increments
    /// `unresolved_parents` in that case so the caller can report.
    pub fn resolve_edge(&mut self, cl_from: i64, cl_target_id: i64) -> Option<(i64, i64)> {
        match self.mode {
            IngestMode::Correct => {
                let title = self.lt_title_map.get(&cl_target_id)?;
                let parent_page_id = self.page_title_to_id.get(title.as_slice())?;
                Some((cl_from, *parent_page_id))
            }
            IngestMode::Compromise => {
                // Parent side already canonical (lt_id). Child stays as
                // page_id; the namespaces differ but that's documented.
                if self.lt_title_map.contains_key(&cl_target_id) {
                    Some((cl_from, cl_target_id))
                } else {
                    self.unresolved_parents += 1;
                    None
                }
            }
            IngestMode::Fallback => match self.id_method {
                IdMethod::Raw => Some((cl_from, cl_target_id)),
                IdMethod::Position => {
                    let c = self.intern_or_alloc(cl_from);
                    let p = self.intern_or_alloc(cl_target_id);
                    Some((c, p))
                }
                IdMethod::Hash => {
                    // Salt cl_from and cl_target_id with different markers
                    // so the namespaces never collide accidentally.
                    let c = hash64_with_salt(0xA1, cl_from);
                    let p = hash64_with_salt(0xB2, cl_target_id);
                    Some((c, p))
                }
            },
        }
    }

    fn intern_or_alloc(&mut self, raw: i64) -> i64 {
        if let Some(&existing) = self.intern.get(&raw) {
            return existing;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.intern.insert(raw, id);
        id
    }
}

fn field_i64(field: &Field) -> Option<i64> {
    match field {
        Field::Int(value) => Some(*value),
        Field::Str(bytes) => std::str::from_utf8(bytes).ok()?.parse().ok(),
        // Float columns (e.g. page_random) are never ID columns in the
        // tables we care about; treat as unrepresentable.
        Field::Float(_) => None,
        Field::Null => None,
    }
}

/// 64-bit hash with a 1-byte salt to keep namespaces separate.
/// FNV-1a variant — fast, deterministic, well-distributed for integers.
fn hash64_with_salt(salt: u8, value: i64) -> i64 {
    let mut h: u64 = 0xcbf29ce484222325;
    h ^= salt as u64;
    h = h.wrapping_mul(0x100000001b3);
    for shift in 0..8 {
        let byte = ((value as u64) >> (shift * 8)) as u8;
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_mode_parses_aliases() {
        assert_eq!(IngestMode::parse("correct").unwrap(), IngestMode::Correct);
        assert_eq!(IngestMode::parse("3-dump").unwrap(), IngestMode::Correct);
        assert_eq!(
            IngestMode::parse("compromise").unwrap(),
            IngestMode::Compromise
        );
        assert_eq!(IngestMode::parse("fallback").unwrap(), IngestMode::Fallback);
        assert!(IngestMode::parse("bogus").is_err());
    }

    #[test]
    fn id_method_parses_aliases() {
        assert_eq!(IdMethod::parse("raw").unwrap(), IdMethod::Raw);
        assert_eq!(IdMethod::parse("position").unwrap(), IdMethod::Position);
        assert_eq!(IdMethod::parse("intern").unwrap(), IdMethod::Position);
        assert_eq!(IdMethod::parse("hash").unwrap(), IdMethod::Hash);
        assert!(IdMethod::parse("nope").is_err());
    }

    #[test]
    fn fallback_raw_passes_through() {
        let mut r = Resolver::new(IngestMode::Fallback, IdMethod::Raw);
        assert_eq!(r.resolve_edge(10, 20), Some((10, 20)));
    }

    #[test]
    fn fallback_position_assigns_monotonic_ids() {
        let mut r = Resolver::new(IngestMode::Fallback, IdMethod::Position);
        assert_eq!(r.resolve_edge(100, 200), Some((0, 1)));
        assert_eq!(r.resolve_edge(100, 300), Some((0, 2))); // 100 reuses id 0
        assert_eq!(r.resolve_edge(400, 200), Some((3, 1))); // 200 reuses id 1
    }

    #[test]
    fn fallback_hash_separates_namespaces() {
        let mut r = Resolver::new(IngestMode::Fallback, IdMethod::Hash);
        // Same raw value in cl_from and cl_target_id should hash to
        // different ids because of the salt.
        let (c, p) = r.resolve_edge(42, 42).unwrap();
        assert_ne!(c, p, "namespace salts must produce distinct hashes");
    }

    #[test]
    fn fallback_hash_is_deterministic() {
        let mut r1 = Resolver::new(IngestMode::Fallback, IdMethod::Hash);
        let mut r2 = Resolver::new(IngestMode::Fallback, IdMethod::Hash);
        assert_eq!(r1.resolve_edge(42, 99), r2.resolve_edge(42, 99));
    }

    #[test]
    fn compromise_resolves_when_target_in_linktarget() {
        let mut r = Resolver::new(IngestMode::Compromise, IdMethod::Raw);
        r.lt_title_map.insert(500, b"Physics".to_vec());
        assert_eq!(r.resolve_edge(10, 500), Some((10, 500)));
        assert_eq!(r.unresolved_parents, 0);
        assert_eq!(r.resolve_edge(10, 9999), None);
        assert_eq!(r.unresolved_parents, 1);
    }

    #[test]
    fn correct_joins_target_via_title() {
        let mut r = Resolver::new(IngestMode::Correct, IdMethod::Raw);
        r.lt_title_map.insert(500, b"Physics".to_vec());
        r.page_title_to_id.insert(b"Physics".to_vec(), 7777);
        assert_eq!(r.resolve_edge(10, 500), Some((10, 7777)));
    }

    #[test]
    fn correct_returns_none_when_title_missing() {
        let mut r = Resolver::new(IngestMode::Correct, IdMethod::Raw);
        r.lt_title_map.insert(500, b"Physics".to_vec());
        // title not in page_title_to_id
        assert_eq!(r.resolve_edge(10, 500), None);
    }
}
