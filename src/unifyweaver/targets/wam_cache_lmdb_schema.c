/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 John William Creighton (@s243a)
 *
 * Build-time schema probe for the plawk LMDB cache backend
 * (PLAWK_MULTIPASS_CACHE.md phase 8.8). A str-valued (row) store on the LMDB
 * backend keeps its declared schema under a distinguished key
 * ("__wam_schema__"), written by wam_cache_commit_lmdb_str. `use NAME` needs
 * that schema at BUILD time -- but the file backend's schema is a plain file
 * header the plawk build reads directly, whereas LMDB's lives inside the
 * B-tree and needs liblmdb to extract. This tiny helper is compiled and run by
 * the plawk build (which already depends on clang + liblmdb for lmdb stores):
 * given a store path, it prints the schema string ("name:type,...") to stdout
 * and exits 0, or exits non-zero when the store is missing / has no schema.
 * It is NOT linked into the plawk binary; it is a standalone build-time tool. */

#include <lmdb.h>
#include <stdio.h>
#include <stddef.h>

/* Must match WAM_LMDB_SCHEMA_KEY / _KEYLEN in wam_cache_lmdb.c. */
static const char SCHEMA_KEY[] = "__wam_schema__";
#define SCHEMA_KEYLEN 14

int main(int argc, char **argv) {
    MDB_env *env;
    MDB_txn *txn;
    MDB_dbi dbi;
    MDB_val k, v;
    int rc;
    if (argc < 2) return 2;
    if (mdb_env_create(&env)) return 1;
    if (mdb_env_open(env, argv[1], MDB_NOSUBDIR | MDB_RDONLY, 0644)) {
        mdb_env_close(env);
        return 1;
    }
    if (mdb_txn_begin(env, NULL, MDB_RDONLY, &txn)) {
        mdb_env_close(env);
        return 1;
    }
    if (mdb_dbi_open(txn, NULL, 0, &dbi)) {
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
