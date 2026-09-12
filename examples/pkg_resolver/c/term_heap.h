/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#pragma once

#include "wam_runtime.h"

typedef struct TermHeap {
    WamState *state;
} TermHeap;

void term_heap_init(TermHeap *th, WamState *state);
void term_heap_ensure(TermHeap *th, int cells);

const char *term_heap_intern(TermHeap *th, const char *s);
WamValue term_heap_atom(TermHeap *th, const char *s);
WamValue term_heap_int(TermHeap *th, int n);
WamValue term_heap_compound(TermHeap *th, const char *name, int arity, const WamValue *args);
WamValue term_heap_list(TermHeap *th, const WamValue *items, size_t count);
WamValue term_heap_nil(TermHeap *th);

int term_heap_functor_name(WamState *state, WamValue *d, char *name, size_t name_sz, int *arity);
int term_heap_is_list_functor(const char *name);
