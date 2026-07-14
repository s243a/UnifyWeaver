/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * LMDB backend for the plawk multi-pass persistent cache
 * (PLAWK_MULTIPASS_CACHE.md, phase 5). Compiled and linked into a plawk
 * binary ONLY when a program declares a `backend "lmdb"` cache store
 * (the plawk build appends this file and -llmdb in that case); otherwise
 * the pure-LLVM file-backed backend (@wam_cache_load / _commit, emitted as
 * IR) is used and nothing here is linked.
 *
 * Like the file backend, the cache handle is the runtime's in-memory
 * %WamAssocI64Table*: load reads every (key,value) i64 pair from the LMDB
 * store into the table, commit writes the table back. This is eager
 * materialisation -- the LMDB file is the durable format and the working
 * set is the assoc table. (Lazy per-access LMDB -- the larger-than-RAM
 * story -- is a later refinement over the same surface.)
 *
 * The store is a single file (MDB_NOSUBDIR), so `cache("run.lmdb")` names
 * one file (plus an LMDB "run.lmdb-lock" sidecar), matching the file
 * backend's one-path-one-store shape. */

#include <lmdb.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* The runtime's assoc-table helpers (defined as LLVM IR in the same
 * binary). The table pointer is opaque here; we only thread it through. */
extern int64_t wam_assoc_i64_set(void *table, int64_t key, int64_t value);
extern int64_t wam_assoc_i64_iter_next(void *table, int64_t start);
extern int64_t wam_assoc_i64_key_at(void *table, int64_t idx);
extern int64_t wam_assoc_i64_value_at(void *table, int64_t idx);

/* Atom-registry helpers (also LLVM IR in the same binary), for byte-valued
 * (row) tables: resolve a process-local value id to its bytes and back. Used
 * only by the _str variants below -- the durable form stores the row bytes,
 * not the id, so rows survive across runs (mirrors the file backend's
 * @wam_cache_commit_str / _load_str). */
extern const char *wam_atom_to_string(int64_t id);
extern int64_t wam_intern_atom(const char *bytes, int64_t len);

/* 1 GiB map ceiling -- generous for a scratch/aggregation store; the file
 * grows only as needed. A later knob can make this configurable. */
#define WAM_CACHE_LMDB_MAPSIZE ((size_t)1 << 30)

/* Max named sub-DBs openable per env (phase 8.9 multi-table). Each cache op
 * opens at most ONE named sub-DB, so this is a generous ceiling on how many
 * *distinct* named tables a store may hold; a later knob can raise it. An
 * unnamed / single-table store passes subname == NULL and needs no maxdbs. */
#define WAM_CACHE_LMDB_MAXDBS 128

/* Open the store (single NOSUBDIR file), begin a txn, and open the target DB:
 * the unnamed default DB when `subname` is NULL (single-table store), or the
 * named sub-DB `subname` otherwise (multi-table store, phase 8.9). Named DBs
 * require maxdbs set before open; `dbi_flags` carries MDB_CREATE on the write
 * path. Returns 0 with *env/*txn/*dbi set, else non-zero (nothing to clean up
 * -- a missing store or absent sub-DB is a no-op for the caller). */
static int wam_cache_lmdb_open(const char *path, unsigned int env_flags,
                               const char *subname, unsigned int dbi_flags,
                               MDB_env **env_out, MDB_txn **txn_out,
                               MDB_dbi *dbi_out) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    if (mdb_env_create(&env)) return -1;
    mdb_env_set_mapsize(env, WAM_CACHE_LMDB_MAPSIZE);
    if (subname) mdb_env_set_maxdbs(env, WAM_CACHE_LMDB_MAXDBS);
    if (mdb_env_open(env, path, MDB_NOSUBDIR | env_flags, 0644)) {
        mdb_env_close(env);
        return -1;
    }
    if (mdb_txn_begin(env, NULL, (env_flags & MDB_RDONLY), &txn)) {
        mdb_env_close(env);
        return -1;
    }
    if (mdb_dbi_open(txn, subname, dbi_flags, &dbi)) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return -1;
    }
    *env_out = env;
    *txn_out = txn;
    *dbi_out = dbi;
    return 0;
}

/* Load every i64->i64 pair from the `subname` DB (NULL = unnamed) into `table`.
 * A missing store or absent sub-DB is a no-op, so an unseeded cache starts
 * empty and a pre-populated one (a prior run, or a peer) is read in. */
static void wam_cache_load_lmdb_core(void *table, const char *path,
                                     const char *subname) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    MDB_cursor *cur;
    MDB_val k, v;
    if (wam_cache_lmdb_open(path, MDB_RDONLY, subname, 0, &env, &txn, &dbi)) return;
    if (mdb_cursor_open(txn, dbi, &cur)) { mdb_txn_abort(txn); mdb_env_close(env); return; }
    while (mdb_cursor_get(cur, &k, &v, MDB_NEXT) == 0) {
        if (k.mv_size == sizeof(int64_t) && v.mv_size == sizeof(int64_t)) {
            int64_t key, val;
            /* mv_data may be unaligned; copy out. */
            __builtin_memcpy(&key, k.mv_data, sizeof(int64_t));
            __builtin_memcpy(&val, v.mv_data, sizeof(int64_t));
            wam_assoc_i64_set(table, key, val);
        }
    }
    mdb_cursor_close(cur);
    mdb_txn_abort(txn);
    mdb_env_close(env);
}

/* Write every occupied i64 entry of `table` back to the `subname` DB
 * (NULL = unnamed) in one write transaction (insert-or-replace per key). */
static void wam_cache_commit_lmdb_core(void *table, const char *path,
                                       const char *subname) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    int64_t cursor = 0, slot;
    unsigned int dbi_flags = subname ? MDB_CREATE : 0;
    if (wam_cache_lmdb_open(path, 0, subname, dbi_flags, &env, &txn, &dbi)) return;
    while ((slot = wam_assoc_i64_iter_next(table, cursor)) >= 0) {
        int64_t key = wam_assoc_i64_key_at(table, slot);
        int64_t val = wam_assoc_i64_value_at(table, slot);
        MDB_val k = { sizeof(int64_t), &key };
        MDB_val v = { sizeof(int64_t), &val };
        mdb_put(txn, dbi, &k, &v, 0);
        cursor = slot + 1;
    }
    mdb_txn_commit(txn);
    mdb_env_close(env);
}

/* Public i64 entry points: the plain names target the unnamed default DB
 * (single-table store), the _sub names a named sub-DB (multi-table store). */
void wam_cache_load_lmdb(void *table, const char *path) {
    wam_cache_load_lmdb_core(table, path, NULL);
}
void wam_cache_commit_lmdb(void *table, const char *path) {
    wam_cache_commit_lmdb_core(table, path, NULL);
}
void wam_cache_load_lmdb_sub(void *table, const char *path, const char *subname) {
    wam_cache_load_lmdb_core(table, path, subname);
}
void wam_cache_commit_lmdb_sub(void *table, const char *path, const char *subname) {
    wam_cache_commit_lmdb_core(table, path, subname);
}

/* ---- byte-valued (row) LMDB store ---------------------------------------
 * A str-valued (row) table stores an atom id as its i64 value, and that id is
 * process-local -- so the plain i64 path above (which persists the id) does
 * not survive across runs. These variants store the VALUE BYTES instead:
 * commit resolves each value id to its bytes; load re-interns the bytes to a
 * fresh id. Keys stay i64. The schema (when the program declares one) is
 * stored under a distinguished non-8-byte key and validated on open, matching
 * the file backend's self-describing store; data rows are the 8-byte keys, so
 * the schema entry is skipped by size on load. */
static const char WAM_LMDB_SCHEMA_KEY[] = "__wam_schema__";
#define WAM_LMDB_SCHEMA_KEYLEN 14  /* strlen(WAM_LMDB_SCHEMA_KEY), != 8 */

/* Commit row bytes for `table` into the `subname` DB (NULL = unnamed). The
 * declared schema (when present) goes under the distinguished key inside that
 * same DB, so each named sub-DB is independently self-describing. */
static void wam_cache_commit_lmdb_str_core(void *table, const char *path,
                                           const char *subname,
                                           const char *schema) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    int64_t cursor = 0, slot;
    unsigned int dbi_flags = subname ? MDB_CREATE : 0;
    if (wam_cache_lmdb_open(path, 0, subname, dbi_flags, &env, &txn, &dbi)) return;
    if (schema) {
        MDB_val sk = { WAM_LMDB_SCHEMA_KEYLEN, (void *)WAM_LMDB_SCHEMA_KEY };
        MDB_val sv = { strlen(schema), (void *)schema };
        mdb_put(txn, dbi, &sk, &sv, 0);
    }
    while ((slot = wam_assoc_i64_iter_next(table, cursor)) >= 0) {
        int64_t key = wam_assoc_i64_key_at(table, slot);
        int64_t vid = wam_assoc_i64_value_at(table, slot);
        const char *vptr = wam_atom_to_string(vid);
        MDB_val k = { sizeof(int64_t), &key };
        MDB_val v = { strlen(vptr), (void *)vptr };
        mdb_put(txn, dbi, &k, &v, 0);
        cursor = slot + 1;
    }
    mdb_txn_commit(txn);
    mdb_env_close(env);
}

/* Load row bytes from the `subname` DB (NULL = unnamed) into `table`,
 * re-interning each value; the schema (if present in both store and program)
 * is validated first -- a mismatch is a loud non-zero exit, exactly like the
 * file backend -- and the 8-byte data rows skip the schema key by size. */
static void wam_cache_load_lmdb_str_core(void *table, const char *path,
                                         const char *subname,
                                         const char *schema) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    MDB_cursor *cur;
    MDB_val k, v;
    if (wam_cache_lmdb_open(path, MDB_RDONLY, subname, 0, &env, &txn, &dbi)) return;
    if (schema) {
        MDB_val sk = { WAM_LMDB_SCHEMA_KEYLEN, (void *)WAM_LMDB_SCHEMA_KEY };
        MDB_val sv;
        if (mdb_get(txn, dbi, &sk, &sv) == 0) {
            size_t explen = strlen(schema);
            if (sv.mv_size != explen ||
                memcmp(sv.mv_data, schema, explen) != 0) {
                printf("plawk: cache schema mismatch");
                mdb_txn_abort(txn);
                mdb_env_close(env);
                exit(3);
            }
        }
    }
    if (mdb_cursor_open(txn, dbi, &cur)) { mdb_txn_abort(txn); mdb_env_close(env); return; }
    while (mdb_cursor_get(cur, &k, &v, MDB_NEXT) == 0) {
        int64_t key;
        int64_t vid;
        if (k.mv_size != sizeof(int64_t)) continue;  /* skip the schema entry */
        __builtin_memcpy(&key, k.mv_data, sizeof(int64_t));
        vid = wam_intern_atom((const char *)v.mv_data, (int64_t)v.mv_size);
        wam_assoc_i64_set(table, key, vid);
    }
    mdb_cursor_close(cur);
    mdb_txn_abort(txn);
    mdb_env_close(env);
}

/* Public row (str) entry points: plain names target the unnamed default DB
 * (single-table store); _sub names a named sub-DB (multi-table store). */
void wam_cache_commit_lmdb_str(void *table, const char *path,
                               const char *schema) {
    wam_cache_commit_lmdb_str_core(table, path, NULL, schema);
}
void wam_cache_load_lmdb_str(void *table, const char *path,
                             const char *schema) {
    wam_cache_load_lmdb_str_core(table, path, NULL, schema);
}
void wam_cache_commit_lmdb_str_sub(void *table, const char *path,
                                   const char *subname, const char *schema) {
    wam_cache_commit_lmdb_str_core(table, path, subname, schema);
}
void wam_cache_load_lmdb_str_sub(void *table, const char *path,
                                 const char *subname, const char *schema) {
    wam_cache_load_lmdb_str_core(table, path, subname, schema);
}
