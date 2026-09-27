/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#include "term_render.h"
#include "term_heap.h"
#include "cstr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int append_str(char **buf, size_t *len, size_t *cap, const char *s) {
    size_t sl = strlen(s);
    while (*len + sl + 1 > *cap) {
        *cap = *cap ? *cap * 2 : 64;
        char *n = realloc(*buf, *cap);
        if (!n) return 0;
        *buf = n;
    }
    memcpy(*buf + *len, s, sl);
    *len += sl;
    (*buf)[*len] = '\0';
    return 1;
}

static int render_inner(WamState *state, WamValue v, char **buf, size_t *len, size_t *cap);

static int render_list(WamState *state, WamValue head, WamValue tail, char **buf, size_t *len, size_t *cap) {
    if (!append_str(buf, len, cap, "[")) return 0;
    if (!render_inner(state, head, buf, len, cap)) return 0;
    for (;;) {
        WamValue *td = wam_deref_ptr(state, &tail);
        if (td->tag == VAL_ATOM && td->data.atom && strcmp(td->data.atom, "[]") == 0)
            break;
        if (td->tag == VAL_LIST) {
            if (!append_str(buf, len, cap, ",")) return 0;
            if (!render_inner(state, state->H_array[td->data.ref_addr], buf, len, cap)) return 0;
            tail = state->H_array[td->data.ref_addr + 1];
            continue;
        }
        if (td->tag == VAL_STR) {
            char name[128];
            int arity = 0;
            if (term_heap_functor_name(state, td, name, sizeof name, &arity) &&
                arity == 2 && term_heap_is_list_functor(name)) {
                if (!append_str(buf, len, cap, ",")) return 0;
                if (!render_inner(state, state->H_array[td->data.ref_addr + 1], buf, len, cap)) return 0;
                tail = state->H_array[td->data.ref_addr + 2];
                continue;
            }
        }
        break;
    }
    return append_str(buf, len, cap, "]");
}

static int render_inner(WamState *state, WamValue v, char **buf, size_t *len, size_t *cap) {
    WamValue *d = wam_deref_ptr(state, &v);
    char tmp[64];
    switch (d->tag) {
    case VAL_ATOM:
        return append_str(buf, len, cap, d->data.atom ? d->data.atom : "");
    case VAL_INT:
        snprintf(tmp, sizeof tmp, "%d", d->data.integer);
        return append_str(buf, len, cap, tmp);
    case VAL_LIST:
        return render_list(state, state->H_array[d->data.ref_addr], state->H_array[d->data.ref_addr + 1], buf, len, cap);
    case VAL_STR: {
        char name[128];
        int arity = 0;
        if (!term_heap_functor_name(state, d, name, sizeof name, &arity))
            return append_str(buf, len, cap, "?");
        if (arity == 2 && term_heap_is_list_functor(name))
            return render_list(state, state->H_array[d->data.ref_addr + 1], state->H_array[d->data.ref_addr + 2], buf, len, cap);
        if (!append_str(buf, len, cap, name)) return 0;
        if (!append_str(buf, len, cap, "(")) return 0;
        for (int i = 0; i < arity; i++) {
            if (i && !append_str(buf, len, cap, ",")) return 0;
            if (!render_inner(state, state->H_array[d->data.ref_addr + 1 + i], buf, len, cap)) return 0;
        }
        return append_str(buf, len, cap, ")");
    }
    default:
        return append_str(buf, len, cap, "?");
    }
}

char *term_render(WamState *state, WamValue v) {
    size_t len = 0, cap = 0;
    char *buf = NULL;
    if (!render_inner(state, v, &buf, &len, &cap))
        return cstr_dup_or(NULL, "?");
    if (!buf) return cstr_dup("");
    return buf;
}
