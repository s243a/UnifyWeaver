/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#include "term_heap.h"

#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void term_heap_init(TermHeap *th, WamState *state) {
    th->state = state;
    th->error = NULL;
}

static void term_heap_fail(TermHeap *th, const char *message) {
    if (!th->error) th->error = message;
}

bool term_heap_ensure(TermHeap *th, int cells) {
    WamState *s = th->state;
    if (th->error) return false;
    if (cells < 0 || s->H < 0 || cells > INT_MAX - s->H) {
        term_heap_fail(th, "WAM term heap capacity exceeded");
        return false;
    }
    int required = s->H + cells;
    if (required <= s->H_cap && s->H_array) return true;
    int cap = s->H_cap > 0 ? s->H_cap : 64;
    while (required > cap) {
        if (cap > INT_MAX / 2) { cap = required; break; }
        cap *= 2;
    }
    if ((size_t)cap > SIZE_MAX / sizeof(WamValue)) {
        term_heap_fail(th, "WAM term heap capacity exceeded");
        return false;
    }
    WamValue *heap = realloc(s->H_array, sizeof(WamValue) * (size_t)cap);
    if (!heap) {
        term_heap_fail(th, "WAM term heap allocation failed");
        return false;
    }
    s->H_array = heap;
    s->H_cap = cap;
    return true;
}

void *term_heap_calloc(TermHeap *th, size_t count, size_t size) {
    if (th->error) return NULL;
    if (size && count > SIZE_MAX / size) {
        term_heap_fail(th, "WAM term allocation size exceeded");
        return NULL;
    }
    void *p = calloc(count ? count : 1, size);
    if (!p) term_heap_fail(th, "WAM term allocation failed");
    return p;
}

const char *term_heap_intern(TermHeap *th, const char *s) {
    if (th->error) return "";
    const char *result = wam_intern_atom(th->state, s);
    /* The runtime returns the borrowed input on allocation failure. Only
       accept that pointer if it is already owned by the atom table. */
    if (result == s) {
        WamState *state = th->state;
        if (state->atom_table) {
            unsigned h = wam_hash_string(s) & (unsigned)(state->atom_table_size - 1);
            for (AtomEntry *e = state->atom_table[h]; e; e = e->next)
                if (e->str == s) return result;
        }
        term_heap_fail(th, "WAM atom allocation failed");
        return "";
    }
    return result;
}

WamValue term_heap_atom(TermHeap *th, const char *s) {
    const char *atom = term_heap_intern(th, s);
    return th->error ? val_unbound("term_error") : val_atom(atom);
}

WamValue term_heap_int(TermHeap *th, int64_t n) {
    if (n < INT_MIN || n > INT_MAX)
        term_heap_fail(th, "Integer outside WAM C int range");
    return th->error ? val_unbound("term_error") : val_int((int)n);
}

WamValue term_heap_compound(TermHeap *th, const char *name, int arity, const WamValue *args) {
    char functor[128];
    if (arity < 0 || arity == INT_MAX || (arity && !args)) {
        term_heap_fail(th, "Invalid WAM compound arguments");
        return val_unbound("term_error");
    }
    int len = snprintf(functor, sizeof functor, "%s/%d", name, arity);
    if (len < 0 || (size_t)len >= sizeof functor) {
        term_heap_fail(th, "WAM functor name too long");
        return val_unbound("term_error");
    }
    const char *interned = term_heap_intern(th, functor);
    if (!term_heap_ensure(th, 1 + arity)) return val_unbound("term_error");
    WamState *s = th->state;
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom(interned);
    for (int i = 0; i < arity; i++)
        s->H_array[s->H++] = args[i];
    return term;
}

WamValue term_heap_nil(TermHeap *th) {
    return term_heap_atom(th, "[]");
}

WamValue term_heap_list(TermHeap *th, const WamValue *items, size_t count) {
    if (th->error || (count && !items)) {
        term_heap_fail(th, "Invalid WAM list arguments");
        return val_unbound("term_error");
    }
    WamValue acc = term_heap_nil(th);
    for (size_t i = count; i > 0; i--) {
        WamValue pair_args[2] = { items[i - 1], acc };
        acc = term_heap_compound(th, "[|]", 2, pair_args);
        if (th->error) break;
    }
    return acc;
}

int term_heap_functor_name(WamState *state, WamValue *d, char *name, size_t name_sz, int *arity) {
    if (d->tag != VAL_STR)
        return 0;
    WamValue *fn = &state->H_array[d->data.ref_addr];
    if (fn->tag != VAL_ATOM || !fn->data.atom)
        return 0;
    const char *slash = strrchr(fn->data.atom, '/');
    if (!slash || slash == fn->data.atom)
        return 0;
    size_t nlen = (size_t)(slash - fn->data.atom);
    if (nlen + 1 > name_sz)
        return 0;
    memcpy(name, fn->data.atom, nlen);
    name[nlen] = '\0';
    *arity = atoi(slash + 1);
    return 1;
}

int term_heap_is_list_functor(const char *name) {
    return strcmp(name, "[|]") == 0 || strcmp(name, ".") == 0;
}
