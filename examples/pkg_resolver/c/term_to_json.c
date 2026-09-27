/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#include "term_to_json.h"
#include "term_heap.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    WamValue *items;
    size_t len;
    size_t cap;
} ValueList;

static void vlist_push(ValueList *vl, WamValue v) {
    if (vl->len + 1 > vl->cap) {
        size_t cap = vl->cap ? vl->cap * 2 : 8;
        WamValue *n = realloc(vl->items, cap * sizeof(WamValue));
        if (!n) return;
        vl->items = n;
        vl->cap = cap;
    }
    vl->items[vl->len++] = v;
}

static void vlist_free(ValueList *vl) {
    free(vl->items);
    vl->items = NULL;
    vl->len = vl->cap = 0;
}

static int list_elements(WamState *state, WamValue list, ValueList *out) {
    WamValue cur = list;
    for (;;) {
        WamValue *d = wam_deref_ptr(state, &cur);
        if (d->tag == VAL_ATOM && d->data.atom && strcmp(d->data.atom, "[]") == 0)
            return 1;
        if (d->tag == VAL_LIST) {
            vlist_push(out, state->H_array[d->data.ref_addr]);
            cur = state->H_array[d->data.ref_addr + 1];
            continue;
        }
        if (d->tag == VAL_STR) {
            char name[128];
            int arity = 0;
            if (!term_heap_functor_name(state, d, name, sizeof name, &arity) ||
                arity != 2 || !term_heap_is_list_functor(name))
                return 0;
            vlist_push(out, state->H_array[d->data.ref_addr + 1]);
            cur = state->H_array[d->data.ref_addr + 2];
            continue;
        }
        return 0;
    }
}

Json tj_seg_to_json(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    char fn[128];
    int arity = 0;
    if (!term_heap_functor_name(state, d, fn, sizeof fn, &arity) || strcmp(fn, "s") != 0 || arity < 2)
        return json_null();
    ValueList codes = {0};
    list_elements(state, state->H_array[d->data.ref_addr + 1], &codes);
    char *order = malloc(codes.len + 1);
    if (!order) { vlist_free(&codes); return json_null(); }
    for (size_t i = 0; i < codes.len; i++) {
        WamValue *cv = wam_deref_ptr(state, &codes.items[i]);
        order[i] = (cv->tag == VAL_INT) ? (char)cv->data.integer : '\0';
    }
    order[codes.len] = '\0';
    vlist_free(&codes);
    WamValue *numv = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 2]);
    int64_t num = (numv->tag == VAL_INT) ? numv->data.integer : 0;
    Json arr = json_array();
    json_array_push(&arr, json_string(order));
    free(order);
    json_array_push(&arr, json_int(num));
    return arr;
}

Json tj_ver_to_json(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    char fn[128];
    int arity = 0;
    if (!term_heap_functor_name(state, d, fn, sizeof fn, &arity))
        return json_null();
    if (strcmp(fn, "v") == 0 && arity == 3) {
        Json arr = json_array();
        for (int i = 0; i < 3; i++) {
            WamValue *iv = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1 + i]);
            json_array_push(&arr, json_int(iv->tag == VAL_INT ? iv->data.integer : 0));
        }
        return arr;
    }
    if (strcmp(fn, "deb") == 0 && arity == 3) {
        WamValue *epochv = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1]);
        Json up = json_array();
        Json rev = json_array();
        ValueList upl = {0}, revl = {0};
        list_elements(state, state->H_array[d->data.ref_addr + 2], &upl);
        list_elements(state, state->H_array[d->data.ref_addr + 3], &revl);
        for (size_t i = 0; i < upl.len; i++) {
            Json seg = tj_seg_to_json(state, upl.items[i]);
            json_array_push(&up, seg);
        }
        for (size_t i = 0; i < revl.len; i++) {
            Json seg = tj_seg_to_json(state, revl.items[i]);
            json_array_push(&rev, seg);
        }
        vlist_free(&upl);
        vlist_free(&revl);
        Json deb_arr = json_array();
        json_array_push(&deb_arr, json_int(epochv->tag == VAL_INT ? epochv->data.integer : 0));
        json_array_push(&deb_arr, up);
        json_array_push(&deb_arr, rev);
        Json obj = json_object();
        json_object_set(&obj, "deb", deb_arr);
        return obj;
    }
    return json_null();
}

Json tj_pair_to_json(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    char fn[128];
    int arity = 0;
    if (!term_heap_functor_name(state, d, fn, sizeof fn, &arity) || strcmp(fn, "-") != 0 || arity < 2)
        return json_null();
    WamValue *namev = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1]);
    Json arr = json_array();
    json_array_push(&arr, json_string(namev->tag == VAL_ATOM && namev->data.atom ? namev->data.atom : ""));
    json_array_push(&arr, tj_ver_to_json(state, state->H_array[d->data.ref_addr + 2]));
    return arr;
}

Json tj_sel_to_json(WamState *state, WamValue v) {
    ValueList items = {0};
    list_elements(state, v, &items);
    Json arr = json_array();
    for (size_t i = 0; i < items.len; i++)
        json_array_push(&arr, tj_pair_to_json(state, items.items[i]));
    vlist_free(&items);
    return arr;
}

Json tj_normalize_constraint(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    if (d->tag == VAL_ATOM && d->data.atom && strcmp(d->data.atom, "any") == 0)
        return json_string("any");
    if (d->tag == VAL_STR) {
        char name[128];
        int arity = 0;
        if (!term_heap_functor_name(state, d, name, sizeof name, &arity))
            return json_null();
        if ((strcmp(name, "eq") == 0 || strcmp(name, "gte") == 0 ||
             strcmp(name, "lt") == 0 || strcmp(name, "lte") == 0 ||
             strcmp(name, "gt") == 0) && arity == 1) {
            Json obj = json_object();
            json_object_set(&obj, "op", json_string(name));
            json_object_set(&obj, "v", tj_ver_to_json(state, state->H_array[d->data.ref_addr + 1]));
            return obj;
        }
        if (strcmp(name, "range") == 0 && arity == 2) {
            Json obj = json_object();
            json_object_set(&obj, "op", json_string("range"));
            json_object_set(&obj, "lo", tj_ver_to_json(state, state->H_array[d->data.ref_addr + 1]));
            json_object_set(&obj, "hi", tj_ver_to_json(state, state->H_array[d->data.ref_addr + 2]));
            return obj;
        }
    }
    return json_null();
}

Json tj_normalize_blocked(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    char name[128];
    int arity = 0;
    if (!term_heap_functor_name(state, d, name, sizeof name, &arity) || strcmp(name, "blocked") != 0)
        return json_null();

    if (arity == 1) {
        WamValue *inner = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1]);
        char iname[128];
        int iarity = 0;
        if (term_heap_functor_name(state, inner, iname, sizeof iname, &iarity) &&
            strcmp(iname, "alternatives") == 0 && iarity == 1) {
            ValueList alts = {0};
            list_elements(state, state->H_array[inner->data.ref_addr + 1], &alts);
            Json alts_arr = json_array();
            for (size_t i = 0; i < alts.len; i++) {
                WamValue *alt = wam_deref_ptr(state, &alts.items[i]);
                char aname[128];
                int aarity = 0;
                if (!term_heap_functor_name(state, alt, aname, sizeof aname, &aarity) ||
                    strcmp(aname, "alt") != 0 || aarity < 2)
                    continue;
                WamValue *depv = wam_deref_ptr(state, &state->H_array[alt->data.ref_addr + 1]);
                WamValue *reasonv = wam_deref_ptr(state, &state->H_array[alt->data.ref_addr + 2]);
                Json alt_obj = json_object();
                json_object_set(&alt_obj, "dep", json_string(depv->tag == VAL_ATOM && depv->data.atom ? depv->data.atom : ""));
                if (reasonv->tag == VAL_ATOM && reasonv->data.atom && strcmp(reasonv->data.atom, "unsatisfiable") == 0)
                    json_object_set(&alt_obj, "reason", json_string("unsatisfiable"));
                else
                    json_object_set(&alt_obj, "reason", tj_normalize_blocked(state, *reasonv));
                json_array_push(&alts_arr, alt_obj);
            }
            vlist_free(&alts);
            Json obj = json_object();
            json_object_set(&obj, "alternatives", alts_arr);
            return obj;
        }
    }

    if (arity == 3) {
        WamValue *pkgv = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1]);
        WamValue needs_raw = state->H_array[d->data.ref_addr + 2];
        WamValue *needs_term = wam_deref_ptr(state, &needs_raw);
        char nname[128];
        int narity = 0;
        if (term_heap_functor_name(state, needs_term, nname, sizeof nname, &narity) &&
            strcmp(nname, "needs") == 0 && narity == 1)
            needs_term = wam_deref_ptr(state, &state->H_array[needs_term->data.ref_addr + 1]);
        WamValue *third = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 3]);
        char tname[128];
        int tarity = 0;
        if (!term_heap_functor_name(state, third, tname, sizeof tname, &tarity))
            return json_null();
        const char *pkg_name = (pkgv->tag == VAL_ATOM && pkgv->data.atom) ? pkgv->data.atom : "";
        if (strcmp(tname, "base_has") == 0 && tarity == 1) {
            Json obj = json_object();
            json_object_set(&obj, "name", json_string(pkg_name));
            json_object_set(&obj, "needs", tj_normalize_constraint(state, *needs_term));
            json_object_set(&obj, "base_has", tj_ver_to_json(state, state->H_array[third->data.ref_addr + 1]));
            return obj;
        }
        if (strcmp(tname, "providers") == 0 && tarity == 1) {
            ValueList provs = {0};
            list_elements(state, state->H_array[third->data.ref_addr + 1], &provs);
            Json prov_arr = json_array();
            for (size_t i = 0; i < provs.len; i++)
                json_array_push(&prov_arr, tj_normalize_blocked(state, provs.items[i]));
            vlist_free(&provs);
            Json obj = json_object();
            json_object_set(&obj, "name", json_string(pkg_name));
            json_object_set(&obj, "needs", tj_normalize_constraint(state, *needs_term));
            json_object_set(&obj, "providers", prov_arr);
            return obj;
        }
    }
    return json_null();
}

Json tj_blocked_list_to_json(WamState *state, WamValue v) {
    ValueList items = {0};
    list_elements(state, v, &items);
    Json arr = json_array();
    for (size_t i = 0; i < items.len; i++)
        json_array_push(&arr, tj_normalize_blocked(state, items.items[i]));
    vlist_free(&items);
    return arr;
}

Json tj_normalize_verdict(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    if (d->tag == VAL_ATOM && d->data.atom && strcmp(d->data.atom, "no_candidate") == 0) {
        Json obj = json_object();
        json_object_set(&obj, "verdict", json_string("no_candidate"));
        return obj;
    }
    char name[128];
    int arity = 0;
    if (d->tag == VAL_STR && term_heap_functor_name(state, d, name, sizeof name, &arity) && arity == 1) {
        if (strcmp(name, "safe") == 0) {
            WamValue *cost_arg = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1]);
            const char *cost_str = "";
            char cname[128];
            int carity = 0;
            if (term_heap_functor_name(state, cost_arg, cname, sizeof cname, &carity) &&
                strcmp(cname, "cost") == 0 && carity == 1) {
                WamValue *cv = wam_deref_ptr(state, &state->H_array[cost_arg->data.ref_addr + 1]);
                if (cv->tag == VAL_ATOM && cv->data.atom) cost_str = cv->data.atom;
            } else if (cost_arg->tag == VAL_ATOM && cost_arg->data.atom) {
                cost_str = cost_arg->data.atom;
            }
            Json obj = json_object();
            json_object_set(&obj, "cost", json_string(cost_str));
            json_object_set(&obj, "verdict", json_string("safe"));
            return obj;
        }
        if (strcmp(name, "coordinated") == 0) {
            Json obj = json_object();
            json_object_set(&obj, "set", tj_sel_to_json(state, state->H_array[d->data.ref_addr + 1]));
            json_object_set(&obj, "verdict", json_string("coordinated"));
            return obj;
        }
        if (strcmp(name, "unsafe") == 0) {
            WamValue *rv = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1]);
            Json obj = json_object();
            json_object_set(&obj, "reason", json_string(rv->tag == VAL_ATOM && rv->data.atom ? rv->data.atom : ""));
            json_object_set(&obj, "verdict", json_string("unsafe"));
            return obj;
        }
    }
    return json_null();
}

Json tj_normalize_upgrade(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    if (d->tag == VAL_ATOM && d->data.atom && strcmp(d->data.atom, "no_candidate") == 0) {
        Json obj = json_object();
        json_object_set(&obj, "fail", json_bool(true));
        return obj;
    }
    char name[128];
    int arity = 0;
    if (d->tag == VAL_STR && term_heap_functor_name(state, d, name, sizeof name, &arity)) {
        if (strcmp(name, "ok") == 0 && arity == 1) {
            Json obj = json_object();
            json_object_set(&obj, "ok", tj_sel_to_json(state, state->H_array[d->data.ref_addr + 1]));
            return obj;
        }
        if (strcmp(name, "blocked") == 0) {
            Json inner = tj_normalize_blocked(state, *d);
            Json blocked_env = json_object();
            json_object_set(&blocked_env, "blocked", inner);
            Json obj = json_object();
            json_object_set(&obj, "ok", blocked_env);
            return obj;
        }
    }
    return json_null();
}

Json tj_normalize_audit_term(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    char name[128];
    int arity = 0;
    if (!term_heap_functor_name(state, d, name, sizeof name, &arity) || strcmp(name, "audit") != 0 || arity < 2)
        return json_null();
    WamValue *pkgv = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 1]);
    WamValue *second = wam_deref_ptr(state, &state->H_array[d->data.ref_addr + 2]);
    const char *pkg_name = (pkgv->tag == VAL_ATOM && pkgv->data.atom) ? pkgv->data.atom : "";
    if (second->tag == VAL_ATOM && second->data.atom && strcmp(second->data.atom, "over_frozen") == 0) {
        Json obj = json_object();
        json_object_set(&obj, "kind", json_string("over_frozen"));
        json_object_set(&obj, "name", json_string(pkg_name));
        return obj;
    }
    char sname[128];
    int sarity = 0;
    if (second->tag == VAL_STR && term_heap_functor_name(state, second, sname, sizeof sname, &sarity) && sarity == 1 &&
        (strcmp(sname, "suggest") == 0 || strcmp(sname, "held") == 0)) {
        WamValue *rv = wam_deref_ptr(state, &state->H_array[second->data.ref_addr + 1]);
        Json obj = json_object();
        json_object_set(&obj, "kind", json_string(sname));
        json_object_set(&obj, "name", json_string(pkg_name));
        json_object_set(&obj, "reason", json_string(rv->tag == VAL_ATOM && rv->data.atom ? rv->data.atom : ""));
        return obj;
    }
    return json_null();
}

Json tj_audit_list_to_json(WamState *state, WamValue v) {
    ValueList items = {0};
    list_elements(state, v, &items);
    Json arr = json_array();
    for (size_t i = 0; i < items.len; i++)
        json_array_push(&arr, tj_normalize_audit_term(state, items.items[i]));
    vlist_free(&items);
    return arr;
}

Json tj_term_to_json(WamState *state, WamValue v) {
    WamValue *d = wam_deref_ptr(state, &v);
    switch (d->tag) {
    case VAL_INT:
        return json_int(d->data.integer);
    case VAL_ATOM:
        if (d->data.atom && strcmp(d->data.atom, "[]") == 0)
            return json_array();
        if (d->data.atom && strcmp(d->data.atom, "true") == 0)
            return json_bool(true);
        if (d->data.atom && strcmp(d->data.atom, "false") == 0)
            return json_bool(false);
        return json_string(d->data.atom ? d->data.atom : "");
    case VAL_UNBOUND:
    case VAL_REF:
        return json_null();
    case VAL_LIST: {
        ValueList items = {0};
        list_elements(state, *d, &items);
        Json arr = json_array();
        for (size_t i = 0; i < items.len; i++)
            json_array_push(&arr, tj_term_to_json(state, items.items[i]));
        vlist_free(&items);
        return arr;
    }
    case VAL_STR: {
        char name[128];
        int arity = 0;
        if (!term_heap_functor_name(state, d, name, sizeof name, &arity))
            return json_null();
        if (arity == 2 && term_heap_is_list_functor(name)) {
            ValueList items = {0};
            list_elements(state, *d, &items);
            Json arr = json_array();
            for (size_t i = 0; i < items.len; i++)
                json_array_push(&arr, tj_term_to_json(state, items.items[i]));
            vlist_free(&items);
            return arr;
        }
        if ((strcmp(name, "v") == 0 && arity == 3) || (strcmp(name, "deb") == 0 && arity == 3))
            return tj_ver_to_json(state, *d);
        if (strcmp(name, "s") == 0 && arity == 2)
            return tj_seg_to_json(state, *d);
        if (strcmp(name, "-") == 0 && arity == 2)
            return tj_pair_to_json(state, *d);
        Json arr = json_array();
        json_array_push(&arr, json_string(name));
        for (int i = 0; i < arity; i++)
            json_array_push(&arr, tj_term_to_json(state, state->H_array[d->data.ref_addr + 1 + i]));
        return arr;
    }
    default:
        return json_null();
    }
}
