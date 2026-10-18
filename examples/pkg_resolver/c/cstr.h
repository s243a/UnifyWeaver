/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Portable C11 replacements for POSIX strdup / getline (not in strict C11). */

static inline char *cstr_dup(const char *s) {
    const char *src = s ? s : "";
    size_t n = strlen(src) + 1;
    char *out = malloc(n);
    if (!out)
        return NULL;
    memcpy(out, src, n);
    return out;
}

static inline char *cstr_dup_or(const char *s, const char *fallback) {
    return cstr_dup(s ? s : fallback);
}

/* Read one line from fp, stripping CR/LF. Returns NULL on EOF with no data. */
static inline char *cstr_read_line(FILE *fp) {
    size_t cap = 128, len = 0;
    char *buf = malloc(cap);
    int c;

    if (!fp || !buf)
        return NULL;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n')
            break;
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF)
                ungetc(next, fp);
            break;
        }
        if (len + 2 > cap) {
            size_t ncap = cap * 2;
            char *n = realloc(buf, ncap);
            if (!n) {
                free(buf);
                return NULL;
            }
            buf = n;
            cap = ncap;
        }
        buf[len++] = (char)c;
    }
    if (c == EOF && len == 0) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}
