#include <stddef.h>
#include <jni.h>
#include <lmdb.h>

#include <stdbool.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *key;
    char *value;
} row_buf;

static char *dup_bytes(const char *src, size_t len) {
    char *dst = (char *)malloc(len + 1);
    if (dst == NULL) {
        return NULL;
    }
    memcpy(dst, src, len);
    dst[len] = '\0';
    return dst;
}

static void free_rows(row_buf *rows, size_t count) {
    if (rows == NULL) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        free(rows[i].key);
        free(rows[i].value);
    }
    free(rows);
}

static int append_row(row_buf **rows, size_t *count, size_t *cap, const char *key, size_t key_len, const char *value, size_t value_len) {
    if (*count == *cap) {
        size_t new_cap = (*cap == 0) ? 8 : (*cap * 2);
        row_buf *new_rows = (row_buf *)realloc(*rows, new_cap * sizeof(row_buf));
        if (new_rows == NULL) {
            return ENOMEM;
        }
        *rows = new_rows;
        *cap = new_cap;
    }

    char *key_copy = dup_bytes(key, key_len);
    if (key_copy == NULL) {
        return ENOMEM;
    }
    char *value_copy = dup_bytes(value, value_len);
    if (value_copy == NULL) {
        free(key_copy);
        return ENOMEM;
    }

    (*rows)[*count].key = key_copy;
    (*rows)[*count].value = value_copy;
    *count += 1;
    return MDB_SUCCESS;
}

static int open_store(const char *artifact_dir, const char *db_name, MDB_env **env_out, MDB_txn **txn_out, MDB_dbi *dbi_out) {
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    int rc = mdb_env_create(&env);
    if (rc != MDB_SUCCESS) {
        return rc;
    }
    mdb_env_set_maxdbs(env, 16);
    mdb_env_set_maxreaders(env, 126);
    mdb_env_set_mapsize(env, 64 * 1024 * 1024);

    rc = mdb_env_open(env, artifact_dir, MDB_RDONLY, 0664);
    if (rc != MDB_SUCCESS) {
        mdb_env_close(env);
        return rc;
    }

    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) {
        mdb_env_close(env);
        return rc;
    }

    rc = mdb_dbi_open(txn, db_name, 0, dbi_out);
    if (rc != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return rc;
    }

    *env_out = env;
    *txn_out = txn;
    return MDB_SUCCESS;
}

static void close_store(MDB_env *env, MDB_txn *txn) {
    if (txn != NULL) {
        mdb_txn_abort(txn);
    }
    if (env != NULL) {
        mdb_env_close(env);
    }
}

static jobjectArray rows_to_java(JNIEnv *env, row_buf *rows, size_t count) {
    jclass row_cls = (*env)->FindClass(env, "generated/lmdb/LmdbRow");
    if (row_cls == NULL) {
        return NULL;
    }
    jmethodID ctor = (*env)->GetMethodID(env, row_cls, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V");
    if (ctor == NULL) {
        return NULL;
    }

    jobjectArray result = (*env)->NewObjectArray(env, (jsize)count, row_cls, NULL);
    if (result == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        jstring key = (*env)->NewStringUTF(env, rows[i].key);
        jstring value = (*env)->NewStringUTF(env, rows[i].value);
        if (key == NULL || value == NULL) {
            return NULL;
        }
        jobject row = (*env)->NewObject(env, row_cls, ctor, key, value);
        if (row == NULL) {
            return NULL;
        }
        (*env)->SetObjectArrayElement(env, result, (jsize)i, row);
        (*env)->DeleteLocalRef(env, row);
        (*env)->DeleteLocalRef(env, key);
        (*env)->DeleteLocalRef(env, value);
    }

    return result;
}

JNIEXPORT jobjectArray JNICALL
Java_generated_lmdb_LmdbArtifactJNI_lookupRows(JNIEnv *env, jclass cls, jstring artifact_dir_j, jstring db_name_j, jstring key_j, jboolean dupsort_j) {
    (void)cls;
    const char *artifact_dir = (*env)->GetStringUTFChars(env, artifact_dir_j, NULL);
    const char *db_name = (*env)->GetStringUTFChars(env, db_name_j, NULL);
    const char *key = (*env)->GetStringUTFChars(env, key_j, NULL);
    bool dupsort = dupsort_j == JNI_TRUE;

    MDB_env *mdb_env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int rc = open_store(artifact_dir, db_name, &mdb_env, &txn, &dbi);

    row_buf *rows = NULL;
    size_t count = 0;
    size_t cap = 0;

    if (rc == MDB_SUCCESS) {
        if (dupsort) {
            MDB_cursor *cursor = NULL;
            MDB_val mdb_key;
            MDB_val mdb_val;
            mdb_key.mv_size = strlen(key);
            mdb_key.mv_data = (void *)key;

            rc = mdb_cursor_open(txn, dbi, &cursor);
            if (rc == MDB_SUCCESS) {
                rc = mdb_cursor_get(cursor, &mdb_key, &mdb_val, MDB_SET);
                while (rc == MDB_SUCCESS) {
                    rc = append_row(&rows, &count, &cap,
                                    key, strlen(key),
                                    (const char *)mdb_val.mv_data, mdb_val.mv_size);
                    if (rc != MDB_SUCCESS) {
                        break;
                    }
                    rc = mdb_cursor_get(cursor, &mdb_key, &mdb_val, MDB_NEXT_DUP);
                }
                if (rc == MDB_NOTFOUND) {
                    rc = MDB_SUCCESS;
                }
                mdb_cursor_close(cursor);
            }
        } else {
            MDB_val mdb_key;
            MDB_val mdb_val;
            mdb_key.mv_size = strlen(key);
            mdb_key.mv_data = (void *)key;
            rc = mdb_get(txn, dbi, &mdb_key, &mdb_val);
            if (rc == MDB_SUCCESS) {
                rc = append_row(&rows, &count, &cap,
                                key, strlen(key),
                                (const char *)mdb_val.mv_data, mdb_val.mv_size);
            } else if (rc == MDB_NOTFOUND) {
                rc = MDB_SUCCESS;
            }
        }
    }

    jobjectArray result = NULL;
    if (rc == MDB_SUCCESS) {
        result = rows_to_java(env, rows, count);
    }

    free_rows(rows, count);
    close_store(mdb_env, txn);
    (*env)->ReleaseStringUTFChars(env, artifact_dir_j, artifact_dir);
    (*env)->ReleaseStringUTFChars(env, db_name_j, db_name);
    (*env)->ReleaseStringUTFChars(env, key_j, key);
    return result;
}

JNIEXPORT jobjectArray JNICALL
Java_generated_lmdb_LmdbArtifactJNI_scanRows(JNIEnv *env, jclass cls, jstring artifact_dir_j, jstring db_name_j) {
    (void)cls;
    const char *artifact_dir = (*env)->GetStringUTFChars(env, artifact_dir_j, NULL);
    const char *db_name = (*env)->GetStringUTFChars(env, db_name_j, NULL);

    MDB_env *mdb_env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int rc = open_store(artifact_dir, db_name, &mdb_env, &txn, &dbi);

    row_buf *rows = NULL;
    size_t count = 0;
    size_t cap = 0;

    if (rc == MDB_SUCCESS) {
        MDB_cursor *cursor = NULL;
        rc = mdb_cursor_open(txn, dbi, &cursor);
        if (rc == MDB_SUCCESS) {
            MDB_val key;
            MDB_val value;
            rc = mdb_cursor_get(cursor, &key, &value, MDB_FIRST);
            while (rc == MDB_SUCCESS) {
                rc = append_row(&rows, &count, &cap,
                                (const char *)key.mv_data, key.mv_size,
                                (const char *)value.mv_data, value.mv_size);
                if (rc != MDB_SUCCESS) {
                    break;
                }
                rc = mdb_cursor_get(cursor, &key, &value, MDB_NEXT);
            }
            if (rc == MDB_NOTFOUND) {
                rc = MDB_SUCCESS;
            }
            mdb_cursor_close(cursor);
        }
    }

    jobjectArray result = NULL;
    if (rc == MDB_SUCCESS) {
        result = rows_to_java(env, rows, count);
    }

    free_rows(rows, count);
    close_store(mdb_env, txn);
    (*env)->ReleaseStringUTFChars(env, artifact_dir_j, artifact_dir);
    (*env)->ReleaseStringUTFChars(env, db_name_j, db_name);
    return result;
}
