/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#include "term_build.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static WamValue tb_segs_term(TermHeap *th, const Json *segs) {
    if (!segs || !json_is_array(segs))
        return term_heap_nil(th);
    size_t n = 0;
    const Json *items = json_array_items(segs, &n);
    WamValue *buf = calloc(n > 0 ? n : 1, sizeof(WamValue));
    if (!buf) return term_heap_nil(th);
    for (size_t i = 0; i < n; i++) {
        const Json *seg = &items[i];
        const char *order = "";
        int64_t num = 0;
        if (json_is_array(seg)) {
            size_t sn = 0;
            const Json *sarr = json_array_items(seg, &sn);
            if (sn > 0 && json_is_string(&sarr[0]))
                order = json_as_string(&sarr[0]);
            if (sn > 1 && json_is_int(&sarr[1]))
                num = json_as_int(&sarr[1]);
        }
        size_t olen = strlen(order);
        WamValue *codes = calloc(olen > 0 ? olen : 1, sizeof(WamValue));
        for (size_t j = 0; j < olen; j++)
            codes[j] = term_heap_int(th, (int)(unsigned char)order[j]);
        WamValue code_list = term_heap_list(th, codes, olen);
        free(codes);
        WamValue s_args[2] = { code_list, term_heap_int(th, (int)num) };
        buf[i] = term_heap_compound(th, "s", 2, s_args);
    }
    WamValue out = term_heap_list(th, buf, n);
    free(buf);
    return out;
}

WamValue tb_ver_term(TermHeap *th, const Json *ver) {
    if (json_is_object(ver)) {
        const Json *deb = json_find(ver, "deb");
        if (deb && json_is_array(deb)) {
            size_t n = 0;
            const Json *arr = json_array_items(deb, &n);
            int64_t epoch = (n > 0 && json_is_int(&arr[0])) ? json_as_int(&arr[0]) : 0;
            const Json *up = (n > 1) ? &arr[1] : NULL;
            const Json *rev = (n > 2) ? &arr[2] : NULL;
            WamValue args[3] = {
                term_heap_int(th, (int)epoch),
                tb_segs_term(th, up),
                tb_segs_term(th, rev)
            };
            return term_heap_compound(th, "deb", 3, args);
        }
    }
    if (json_is_array(ver)) {
        size_t n = 0;
        const Json *arr = json_array_items(ver, &n);
        int m = (n > 0 && json_is_int(&arr[0])) ? (int)json_as_int(&arr[0]) : 0;
        int i = (n > 1 && json_is_int(&arr[1])) ? (int)json_as_int(&arr[1]) : 0;
        int p = (n > 2 && json_is_int(&arr[2])) ? (int)json_as_int(&arr[2]) : 0;
        WamValue args[3] = { term_heap_int(th, m), term_heap_int(th, i), term_heap_int(th, p) };
        return term_heap_compound(th, "v", 3, args);
    }
    return val_unbound("invalid_ver");
}

WamValue tb_pair_term(TermHeap *th, const char *name, const Json *ver) {
    WamValue args[2] = { term_heap_atom(th, name), tb_ver_term(th, ver) };
    return term_heap_compound(th, "-", 2, args);
}

WamValue tb_constraint_term(TermHeap *th, const Json *c) {
    if (!c || json_is_null(c))
        return term_heap_atom(th, "any");
    if (json_is_string(c)) {
        if (strcmp(json_as_string(c), "any") == 0)
            return term_heap_atom(th, "any");
        return term_heap_nil(th);
    }
    if (json_is_object(c)) {
        const Json *op = json_find(c, "op");
        if (op && json_is_string(op)) {
            const char *op_s = json_as_string(op);
            if (strcmp(op_s, "eq") == 0 || strcmp(op_s, "gte") == 0 ||
                strcmp(op_s, "lt") == 0 || strcmp(op_s, "lte") == 0 ||
                strcmp(op_s, "gt") == 0) {
                const Json *v = json_find(c, "v");
                WamValue args[1] = { tb_ver_term(th, v ? v : &((Json){0})) };
                return term_heap_compound(th, op_s, 1, args);
            }
            if (strcmp(op_s, "range") == 0) {
                const Json *lo = json_find(c, "lo");
                const Json *hi = json_find(c, "hi");
                WamValue args[2] = { tb_ver_term(th, lo), tb_ver_term(th, hi) };
                return term_heap_compound(th, "range", 2, args);
            }
        }
    }
    return term_heap_nil(th);
}

WamValue tb_hold_term(TermHeap *th, const Json *row) {
    size_t n = 0;
    const Json *arr = json_array_items(row, &n);
    if (n < 2) return term_heap_nil(th);
    WamValue pair = tb_pair_term(th, json_as_string(&arr[0]), &arr[1]);
    if (n >= 3) {
        WamValue args[2] = { pair, term_heap_atom(th, json_as_string(&arr[2])) };
        return term_heap_compound(th, "base", 2, args);
    }
    return pair;
}

WamValue tb_layer_term(TermHeap *th, const Json *row) {
    const char *name = "";
    const Json *name_json = json_find(row, "name");
    if (name_json && json_is_string(name_json))
        name = json_as_string(name_json);
    size_t pn = 0;
    const Json *pkgs_json = json_find(row, "packages");
    const Json *pkgs = pkgs_json ? json_array_items(pkgs_json, &pn) : NULL;
    WamValue *items = calloc(pn > 0 ? pn : 1, sizeof(WamValue));
    for (size_t i = 0; i < pn; i++)
        items[i] = tb_hold_term(th, &pkgs[i]);
    WamValue pkg_list = term_heap_list(th, items, pn);
    free(items);
    WamValue args[2] = { term_heap_atom(th, name), pkg_list };
    return term_heap_compound(th, "layer", 2, args);
}

WamValue tb_alias_term(TermHeap *th, const Json *row) {
    size_t n = 0;
    const Json *arr = json_array_items(row, &n);
    if (n < 2) return term_heap_nil(th);
    WamValue args[2] = {
        term_heap_atom(th, json_as_string(&arr[0])),
        term_heap_atom(th, json_as_string(&arr[1]))
    };
    return term_heap_compound(th, "alias", 2, args);
}

WamValue tb_pkg_term(TermHeap *th, const Json *row) {
    size_t n = 0;
    const Json *arr = json_array_items(row, &n);
    if (n < 2) return term_heap_nil(th);
    WamValue args[2] = { term_heap_atom(th, json_as_string(&arr[0])), tb_ver_term(th, &arr[1]) };
    return term_heap_compound(th, "package", 2, args);
}

WamValue tb_dep_term(TermHeap *th, const Json *row) {
    size_t n = 0;
    const Json *arr = json_array_items(row, &n);
    if (n < 4) return term_heap_nil(th);
    WamValue ver = tb_ver_term(th, &arr[1]);
    WamValue dep_arg;
    if (json_is_object(&arr[2])) {
        const Json *alts_json = json_find(&arr[2], "alternatives");
        size_t an = 0;
        const Json *alts = alts_json ? json_array_items(alts_json, &an) : NULL;
        WamValue *alt_terms = calloc(an > 0 ? an : 1, sizeof(WamValue));
        for (size_t i = 0; i < an; i++) {
            const char *dep_name = "";
            const Json *d = json_find(&alts[i], "dep");
            if (d && json_is_string(d))
                dep_name = json_as_string(d);
            const Json *c = json_find(&alts[i], "constraint");
            WamValue dep_args[2] = {
                term_heap_atom(th, dep_name),
                tb_constraint_term(th, c ? c : &((Json){0}))
            };
            alt_terms[i] = term_heap_compound(th, "dep", 2, dep_args);
        }
        WamValue alt_list = term_heap_list(th, alt_terms, an);
        free(alt_terms);
        WamValue alt_wrap[1] = { alt_list };
        dep_arg = term_heap_compound(th, "alternatives", 1, alt_wrap);
    } else {
        dep_arg = term_heap_atom(th, json_as_string(&arr[2]));
    }
    WamValue args[4] = {
        term_heap_atom(th, json_as_string(&arr[0])),
        ver,
        dep_arg,
        tb_constraint_term(th, &arr[3])
    };
    return term_heap_compound(th, "depends", 4, args);
}

WamValue tb_provide_term(TermHeap *th, const Json *row) {
    size_t n = 0;
    const Json *arr = json_array_items(row, &n);
    if (n < 3) return term_heap_nil(th);
    if (n >= 4 && !json_is_null(&arr[3])) {
        WamValue args[4] = {
            term_heap_atom(th, json_as_string(&arr[0])),
            tb_ver_term(th, &arr[1]),
            term_heap_atom(th, json_as_string(&arr[2])),
            tb_ver_term(th, &arr[3])
        };
        return term_heap_compound(th, "provides", 4, args);
    }
    WamValue args[3] = {
        term_heap_atom(th, json_as_string(&arr[0])),
        tb_ver_term(th, &arr[1]),
        term_heap_atom(th, json_as_string(&arr[2]))
    };
    return term_heap_compound(th, "provides", 3, args);
}

WamValue tb_conf_term(TermHeap *th, const Json *row) {
    size_t n = 0;
    const Json *arr = json_array_items(row, &n);
    if (n < 3) return term_heap_nil(th);
    WamValue args[3] = {
        term_heap_atom(th, json_as_string(&arr[0])),
        tb_ver_term(th, &arr[1]),
        term_heap_atom(th, json_as_string(&arr[2]))
    };
    return term_heap_compound(th, "conflicts", 3, args);
}

WamValue tb_request_term(TermHeap *th, const Json *req) {
    if (json_is_object(req)) {
        const Json *r = json_find(req, "req");
        if (r && json_is_string(r)) {
            const Json *c = json_find(req, "constraint");
            WamValue args[2] = {
                term_heap_atom(th, json_as_string(r)),
                tb_constraint_term(th, c ? c : &((Json){0}))
            };
            return term_heap_compound(th, "req", 2, args);
        }
    }
    if (json_is_string(req))
        return term_heap_atom(th, json_as_string(req));
    return term_heap_nil(th);
}

static WamValue tb_json_array_to_list(TermHeap *th, const Json *arr_json,
                                      WamValue (*map_fn)(TermHeap *, const Json *)) {
    size_t n = 0;
    const Json *items = json_array_items(arr_json, &n);
    WamValue *buf = calloc(n > 0 ? n : 1, sizeof(WamValue));
    for (size_t i = 0; i < n; i++)
        buf[i] = map_fn(th, &items[i]);
    WamValue out = term_heap_list(th, buf, n);
    free(buf);
    return out;
}

WamValue tb_catalog_to_term(TermHeap *th, const Json *catalog) {
    const Json *p = json_find(catalog, "packages");
    const Json *d = json_find(catalog, "depends");
    const Json *c = json_find(catalog, "conflicts");
    const Json *b = json_find(catalog, "base");
    const Json *in = json_find(catalog, "installed");
    const Json *r = json_find(catalog, "requested");

    WamValue pkgs = p && json_is_array(p) ? tb_json_array_to_list(th, p, tb_pkg_term) : term_heap_nil(th);
    WamValue deps = d && json_is_array(d) ? tb_json_array_to_list(th, d, tb_dep_term) : term_heap_nil(th);
    WamValue confs = c && json_is_array(c) ? tb_json_array_to_list(th, c, tb_conf_term) : term_heap_nil(th);
    WamValue base = b && json_is_array(b) ? tb_json_array_to_list(th, b, tb_hold_term) : term_heap_nil(th);

    WamValue inst = term_heap_nil(th);
    if (in && json_is_array(in)) {
        size_t n = 0;
        const Json *items = json_array_items(in, &n);
        WamValue *buf = calloc(n > 0 ? n : 1, sizeof(WamValue));
        for (size_t i = 0; i < n; i++) {
            size_t rn = 0;
            const Json *row = json_array_items(&items[i], &rn);
            if (rn >= 2)
                buf[i] = tb_pair_term(th, json_as_string(&row[0]), &row[1]);
            else
                buf[i] = term_heap_nil(th);
        }
        inst = term_heap_list(th, buf, n);
        free(buf);
    }

    WamValue req = term_heap_nil(th);
    if (r && json_is_array(r)) {
        size_t n = 0;
        const Json *items = json_array_items(r, &n);
        WamValue *buf = calloc(n > 0 ? n : 1, sizeof(WamValue));
        for (size_t i = 0; i < n; i++)
            buf[i] = term_heap_atom(th, json_as_string(&items[i]));
        req = term_heap_list(th, buf, n);
        free(buf);
    }

    WamValue core[6] = { pkgs, deps, confs, base, inst, req };
    const Json *layers_json = json_find(catalog, "layers");
    const Json *excl_json = json_find(catalog, "excluded");
    const Json *alias_json = json_find(catalog, "aliases");
    const Json *prov_json = json_find(catalog, "provides");

    size_t layers_sz = (layers_json && json_is_array(layers_json)) ? json_size(layers_json) : 0;
    size_t excl_sz = (excl_json && json_is_array(excl_json)) ? json_size(excl_json) : 0;
    size_t alias_sz = (alias_json && json_is_array(alias_json)) ? json_size(alias_json) : 0;
    size_t prov_sz = (prov_json && json_is_array(prov_json)) ? json_size(prov_json) : 0;

    if (layers_sz == 0 && excl_sz == 0 && alias_sz == 0 && prov_sz == 0)
        return term_heap_compound(th, "catalog", 6, core);

    WamValue layers = layers_json && json_is_array(layers_json)
        ? tb_json_array_to_list(th, layers_json, tb_layer_term) : term_heap_nil(th);
    WamValue excl = term_heap_nil(th);
    if (excl_json && json_is_array(excl_json)) {
        size_t n = 0;
        const Json *items = json_array_items(excl_json, &n);
        WamValue *buf = calloc(n > 0 ? n : 1, sizeof(WamValue));
        for (size_t i = 0; i < n; i++)
            buf[i] = term_heap_atom(th, json_as_string(&items[i]));
        excl = term_heap_list(th, buf, n);
        free(buf);
    }
    WamValue aliases = alias_json && json_is_array(alias_json)
        ? tb_json_array_to_list(th, alias_json, tb_alias_term) : term_heap_nil(th);

    WamValue nine[9] = { core[0], core[1], core[2], core[3], core[4], core[5], layers, excl, aliases };
    if (prov_sz == 0)
        return term_heap_compound(th, "catalog", 9, nine);

    WamValue provs = prov_json && json_is_array(prov_json)
        ? tb_json_array_to_list(th, prov_json, tb_provide_term) : term_heap_nil(th);
    WamValue ten[10] = {
        nine[0], nine[1], nine[2], nine[3], nine[4], nine[5],
        nine[6], nine[7], nine[8], provs
    };
    return term_heap_compound(th, "catalog", 10, ten);
}
