/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#include "term_heap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void term_heap_init(TermHeap *th, WamState *state) {
    th->state = state;
}

void term_heap_ensure(TermHeap *th, int cells) {
    WamState *s = th->state;
    if (s->H + cells < s->H_cap)
        return;
    int cap = s->H_cap ? s->H_cap : 64;
    while (s->H + cells >= cap) {
        if (cap > (1 << 28))
            return;
        cap *= 2;
    }
    WamValue *heap = realloc(s->H_array, sizeof(WamValue) * (size_t)cap);
    if (!heap)
        return;
    s->H_array = heap;
    s->H_cap = cap;
}

const char *term_heap_intern(TermHeap *th, const char *s) {
    return wam_intern_atom(th->state, s);
}

WamValue term_heap_atom(TermHeap *th, const char *s) {
    return val_atom(term_heap_intern(th, s));
}

WamValue term_heap_int(TermHeap *th, int n) {
    (void)th;
    return val_int(n);
}

WamValue term_heap_compound(TermHeap *th, const char *name, int arity, const WamValue *args) {
    char functor[128];
    snprintf(functor, sizeof functor, "%s/%d", name, arity);
    term_heap_ensure(th, 1 + arity);
    WamState *s = th->state;
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom(term_heap_intern(th, functor));
    for (int i = 0; i < arity; i++)
        s->H_array[s->H++] = args[i];
    return term;
}

WamValue term_heap_nil(TermHeap *th) {
    return term_heap_atom(th, "[]");
}

WamValue term_heap_list(TermHeap *th, const WamValue *items, size_t count) {
    WamValue acc = term_heap_nil(th);
    for (size_t i = count; i > 0; i--) {
        WamValue pair_args[2] = { items[i - 1], acc };
        acc = term_heap_compound(th, "[|]", 2, pair_args);
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
