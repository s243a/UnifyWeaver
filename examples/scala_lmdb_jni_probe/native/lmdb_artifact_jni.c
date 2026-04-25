#include <stddef.h>
#include <jni.h>
#include <lmdb.h>

#include <stdbool.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *dup_c_string(const char *src) {
    size_t len = strlen(src);
    char *dst = (char *)malloc(len + 1);
    if (dst == NULL) {
        return NULL;
    }
    memcpy(dst, src, len + 1);
    return dst;
}

static void append_bytes(char **buffer, size_t *len, size_t *cap, const char *src, size_t src_len) {
    size_t needed = *len + src_len + 1;
    if (needed > *cap) {
        size_t new_cap = (*cap == 0) ? 256 : *cap;
        while (new_cap < needed) {
            new_cap *= 2;
        }
        char *new_buf = (char *)realloc(*buffer, new_cap);
        if (new_buf == NULL) {
            free(*buffer);
            *buffer = NULL;
            *len = 0;
            *cap = 0;
            return;
        }
        *buffer = new_buf;
        *cap = new_cap;
    }
    memcpy(*buffer + *len, src, src_len);
    *len += src_len;
    (*buffer)[*len] = '\0';
}

static jstring error_string(JNIEnv *env, const char *message) {
    return (*env)->NewStringUTF(env, message);
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

JNIEXPORT jstring JNICALL
Java_generated_lmdb_LmdbArtifactJNI_lookupRaw(JNIEnv *env, jclass cls, jstring artifact_dir_j, jstring db_name_j, jstring key_j, jboolean dupsort_j) {
    (void)cls;
    const char *artifact_dir = (*env)->GetStringUTFChars(env, artifact_dir_j, NULL);
    const char *db_name = (*env)->GetStringUTFChars(env, db_name_j, NULL);
    const char *key = (*env)->GetStringUTFChars(env, key_j, NULL);
    bool dupsort = dupsort_j == JNI_TRUE;

    MDB_env *mdb_env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int rc = open_store(artifact_dir, db_name, &mdb_env, &txn, &dbi);
    if (rc != MDB_SUCCESS) {
        (*env)->ReleaseStringUTFChars(env, artifact_dir_j, artifact_dir);
        (*env)->ReleaseStringUTFChars(env, db_name_j, db_name);
        (*env)->ReleaseStringUTFChars(env, key_j, key);
        return error_string(env, mdb_strerror(rc));
    }

    char *buffer = NULL;
    size_t len = 0;
    size_t cap = 0;

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
                append_bytes(&buffer, &len, &cap, key, strlen(key));
                append_bytes(&buffer, &len, &cap, "\t", 1);
                append_bytes(&buffer, &len, &cap, (const char *)mdb_val.mv_data, mdb_val.mv_size);
                append_bytes(&buffer, &len, &cap, "\n", 1);
                if (buffer == NULL) {
                    rc = ENOMEM;
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
            append_bytes(&buffer, &len, &cap, key, strlen(key));
            append_bytes(&buffer, &len, &cap, "\t", 1);
            append_bytes(&buffer, &len, &cap, (const char *)mdb_val.mv_data, mdb_val.mv_size);
            append_bytes(&buffer, &len, &cap, "\n", 1);
            if (buffer == NULL) {
                rc = ENOMEM;
            }
        } else if (rc == MDB_NOTFOUND) {
            rc = MDB_SUCCESS;
        }
    }

    jstring result;
    if (rc != MDB_SUCCESS) {
        result = error_string(env, mdb_strerror(rc));
    } else {
        if (buffer == NULL) {
            buffer = dup_c_string("");
        }
        result = (*env)->NewStringUTF(env, buffer);
    }

    free(buffer);
    close_store(mdb_env, txn);
    (*env)->ReleaseStringUTFChars(env, artifact_dir_j, artifact_dir);
    (*env)->ReleaseStringUTFChars(env, db_name_j, db_name);
    (*env)->ReleaseStringUTFChars(env, key_j, key);
    return result;
}

JNIEXPORT jstring JNICALL
Java_generated_lmdb_LmdbArtifactJNI_scanRaw(JNIEnv *env, jclass cls, jstring artifact_dir_j, jstring db_name_j) {
    (void)cls;
    const char *artifact_dir = (*env)->GetStringUTFChars(env, artifact_dir_j, NULL);
    const char *db_name = (*env)->GetStringUTFChars(env, db_name_j, NULL);

    MDB_env *mdb_env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int rc = open_store(artifact_dir, db_name, &mdb_env, &txn, &dbi);
    if (rc != MDB_SUCCESS) {
        (*env)->ReleaseStringUTFChars(env, artifact_dir_j, artifact_dir);
        (*env)->ReleaseStringUTFChars(env, db_name_j, db_name);
        return error_string(env, mdb_strerror(rc));
    }

    MDB_cursor *cursor = NULL;
    rc = mdb_cursor_open(txn, dbi, &cursor);

    char *buffer = NULL;
    size_t len = 0;
    size_t cap = 0;

    if (rc == MDB_SUCCESS) {
        MDB_val key;
        MDB_val val;
        rc = mdb_cursor_get(cursor, &key, &val, MDB_FIRST);
        while (rc == MDB_SUCCESS) {
            append_bytes(&buffer, &len, &cap, (const char *)key.mv_data, key.mv_size);
            append_bytes(&buffer, &len, &cap, "\t", 1);
            append_bytes(&buffer, &len, &cap, (const char *)val.mv_data, val.mv_size);
            append_bytes(&buffer, &len, &cap, "\n", 1);
            if (buffer == NULL) {
                rc = ENOMEM;
                break;
            }
            rc = mdb_cursor_get(cursor, &key, &val, MDB_NEXT);
        }
        if (rc == MDB_NOTFOUND) {
            rc = MDB_SUCCESS;
        }
        mdb_cursor_close(cursor);
    }

    jstring result;
    if (rc != MDB_SUCCESS) {
        result = error_string(env, mdb_strerror(rc));
    } else {
        if (buffer == NULL) {
            buffer = dup_c_string("");
        }
        result = (*env)->NewStringUTF(env, buffer);
    }

    free(buffer);
    close_store(mdb_env, txn);
    (*env)->ReleaseStringUTFChars(env, artifact_dir_j, artifact_dir);
    (*env)->ReleaseStringUTFChars(env, db_name_j, db_name);
    return result;
}
