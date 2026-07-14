/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Build-time schema probe for the plawk LMDB cache backend
 * (PLAWK_MULTIPASS_CACHE.md phase 8.8/8.9). A str-valued (row) store on the
 * LMDB backend keeps its declared schema under a distinguished key
 * ("__wam_schema__"), written by wam_cache_commit_lmdb_str -- in the unnamed
 * default DB for a single-table store, or inside each named sub-DB for a
 * multi-table store. `use NAME` needs that schema at BUILD time, but LMDB's
 * schema lives inside the B-tree and needs liblmdb to extract, so the plawk
 * build compiles and runs this tiny helper (it already depends on clang +
 * liblmdb for lmdb stores). Given a store path -- and, for a multi-table store,
 * a sub-DB name as argv[2] -- it prints the schema string ("name:type,...") to
 * stdout and exits 0, or exits non-zero when the store / sub-DB is missing or
 * has no schema. NOT linked into the plawk binary; a standalone build-time
 * tool. */

#include <lmdb.h>
#include <stdio.h>
#include <stddef.h>

/* Must match WAM_LMDB_SCHEMA_KEY / _KEYLEN in wam_cache_lmdb.c. */
static const char SCHEMA_KEY[] = "__wam_schema__";
#define SCHEMA_KEYLEN 14

/* Max named sub-DBs (matches WAM_CACHE_LMDB_MAXDBS in wam_cache_lmdb.c); only
 * set when opening a named sub-DB. */
#define SCHEMA_MAXDBS 128

int main(int argc, char **argv) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    MDB_val k, v;
    const char *subname;
    int rc;
    if (argc < 2) return 2;
    subname = (argc >= 3) ? argv[2] : NULL;  /* multi-table: a named sub-DB */
    if (mdb_env_create(&env)) return 1;
    if (subname) mdb_env_set_maxdbs(env, SCHEMA_MAXDBS);
    if (mdb_env_open(env, argv[1], MDB_NOSUBDIR | MDB_RDONLY, 0644)) {
        mdb_env_close(env);
        return 1;
    }
    if (mdb_txn_begin(env, NULL, MDB_RDONLY, &txn)) {
        mdb_env_close(env);
        return 1;
    }
    if (mdb_dbi_open(txn, subname, 0, &dbi)) {  /* NULL = unnamed default DB */
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return 1;
    }
    k.mv_size = SCHEMA_KEYLEN;
    k.mv_data = (void *)SCHEMA_KEY;
    rc = mdb_get(txn, dbi, &k, &v);
    if (rc == 0 && v.mv_size > 0) {
        fwrite(v.mv_data, 1, v.mv_size, stdout);
    }
    mdb_txn_abort(txn);
    mdb_env_close(env);
    return (rc == 0 && v.mv_size > 0) ? 0 : 1;
}
