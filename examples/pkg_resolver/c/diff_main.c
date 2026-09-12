/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* diff_main.c -- JSONL differential runner for WAM-C pkg_resolver. */
#include "wam_runtime.h"
#include "cstr.h"
#include "json.h"
#include "term_build.h"
#include "term_heap.h"
#include "term_to_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "setup_all.inc"

static WamState g_vm;
static TermHeap g_th;

static WamValue list_of_values(WamValue *items, size_t count) {
    return term_heap_list(&g_th, items, count);
}

/* wam_run_predicate converts VAL_UNBOUND args to heap refs, then writes
 * dereferenced results back into state->A[i]. Read those after success. */
static int query_capture(const char *pred_key, WamValue *args, int arity, WamValue *out_cells) {
    if (g_th.error) return WAM_ERR_UNSUPPORTED;
    if (arity < 0 || arity > WAM_MAX_REGS)
        return -1;
    int rc = wam_run_predicate(&g_vm, pred_key, args, arity);
    /* WAM_HALT is logical failure; all other nonzero codes are errors. */
    if (rc != 0 && rc != WAM_HALT) g_vm.error = rc;
    if (rc != 0)
        return rc;
    for (int i = 0; i < arity; i++)
        out_cells[i] = g_vm.A[i];
    return 0;
}

static void json_set_id(Json *out, const Json *row) {
    const Json *id = json_find(row, "id");
    if (id)
        json_object_set(out, "id", json_clone(id));
    else
        json_object_set(out, "id", json_null());
}

static void run_case(const Json *row, Json *out) {
    const Json *cat_ptr = json_find(row, "catalog");
    const Json *q_ptr = json_find(row, "query");
    const Json *args_ptr = json_find(row, "args");
    if (!cat_ptr) {
        json_object_set(out, "crash", json_string("missing catalog"));
        return;
    }
    if (!q_ptr || !json_is_string(q_ptr)) {
        json_object_set(out, "crash", json_string("missing or invalid query"));
        return;
    }
    const char *q = json_as_string(q_ptr);
    WamValue cat_term = tb_catalog_to_term(&g_th, cat_ptr);

    if (strcmp(q, "resolve") == 0 || strcmp(q, "resolve_layered") == 0) {
        size_t rn = 0;
        WamValue *req_vals = NULL;
        if (args_ptr && json_is_array(args_ptr)) {
            const Json *items = json_array_items(args_ptr, &rn);
            req_vals = term_heap_calloc(&g_th, rn > 0 ? rn : 1, sizeof(WamValue));
            if (!req_vals) return;
            for (size_t i = 0; i < rn; i++)
                req_vals[i] = tb_request_term(&g_th, &items[i]);
        } else {
            rn = 0;
            req_vals = term_heap_calloc(&g_th, 1, sizeof(WamValue));
        }
        WamValue reqs_list = list_of_values(req_vals, rn);
        free(req_vals);
        WamValue qargs[3] = { cat_term, reqs_list, val_unbound("Out") };
        const char *pred = strcmp(q, "resolve") == 0 ? "resolve/3" : "resolve_layered/3";
        WamValue cells[3];
        int rc = query_capture(pred, qargs, 3, cells);
        if (rc == 0)
            json_object_set(out, "ok", tj_sel_to_json(&g_vm, cells[2]));
        else
            json_object_set(out, "fail", json_bool(true));
        return;
    }

    if (strcmp(q, "explain_blocked") == 0) {
        if (!args_ptr) { json_object_set(out, "crash", json_string("explain_blocked missing args")); return; }
        WamValue qargs[3] = { cat_term, tb_request_term(&g_th, args_ptr), val_unbound("Out") };
        WamValue cells[3];
        if (query_capture("explain_blocked_list/3", qargs, 3, cells) == 0)
            json_object_set(out, "ok", tj_blocked_list_to_json(&g_vm, cells[2]));
        else
            json_object_set(out, "fail", json_bool(true));
        return;
    }

    if (strcmp(q, "layer_closure") == 0) {
        if (!args_ptr) { json_object_set(out, "crash", json_string("layer_closure missing args")); return; }
        WamValue qargs[3] = { cat_term, tb_request_term(&g_th, args_ptr), val_unbound("Out") };
        WamValue cells[3];
        if (query_capture("layer_closure/3", qargs, 3, cells) == 0)
            json_object_set(out, "ok", tj_term_to_json(&g_vm, cells[2]));
        else
            json_object_set(out, "fail", json_bool(true));
        return;
    }

    if (strcmp(q, "removal_orphans") == 0) {
        if (!args_ptr || !json_is_string(args_ptr)) {
            json_object_set(out, "crash", json_string("removal_orphans missing string args"));
            return;
        }
        WamValue qargs[3] = { cat_term, term_heap_atom(&g_th, json_as_string(args_ptr)), val_unbound("Out") };
        WamValue cells[3];
        if (query_capture("removal_orphans/3", qargs, 3, cells) == 0)
            json_object_set(out, "ok", tj_term_to_json(&g_vm, cells[2]));
        else
            json_object_set(out, "fail", json_bool(true));
        return;
    }

    if (strcmp(q, "safe_upgrade") == 0) {
        if (!args_ptr || !json_is_array(args_ptr) || json_size(args_ptr) < 2) {
            json_object_set(out, "crash", json_string("safe_upgrade expects 2-element args array"));
            return;
        }
        WamValue qargs[4] = {
            cat_term,
            term_heap_atom(&g_th, json_as_string(json_at(args_ptr, 0))),
            tb_ver_term(&g_th, json_at(args_ptr, 1)),
            val_unbound("Out")
        };
        WamValue cells[4];
        if (query_capture("safe_upgrade/4", qargs, 4, cells) == 0)
            json_object_set(out, "ok", tj_normalize_verdict(&g_vm, cells[3]));
        else
            json_object_set(out, "fail", json_bool(true));
        return;
    }

    if (strcmp(q, "upgrade_set") == 0) {
        if (!args_ptr || !json_is_array(args_ptr) || json_size(args_ptr) < 2) {
            json_object_set(out, "crash", json_string("upgrade_set expects 2-element args array"));
            return;
        }
        WamValue qargs[4] = {
            cat_term,
            term_heap_atom(&g_th, json_as_string(json_at(args_ptr, 0))),
            tb_ver_term(&g_th, json_at(args_ptr, 1)),
            val_unbound("Out")
        };
        WamValue cells[4];
        if (query_capture("upgrade_set_result/4", qargs, 4, cells) == 0) {
            Json r = tj_normalize_upgrade(&g_vm, cells[3]);
            const Json *ok = json_find(&r, "ok");
            const Json *fail = json_find(&r, "fail");
            if (ok) json_object_set(out, "ok", json_clone(ok));
            else if (fail) json_object_set(out, "fail", json_clone(fail));
            json_free(&r);
        } else {
            json_object_set(out, "fail", json_bool(true));
        }
        return;
    }

    if (strcmp(q, "freeze_audit") == 0) {
        WamValue qargs[2] = { cat_term, val_unbound("Out") };
        WamValue cells[2];
        if (query_capture("freeze_audit/2", qargs, 2, cells) == 0)
            json_object_set(out, "ok", tj_audit_list_to_json(&g_vm, cells[1]));
        else
            json_object_set(out, "fail", json_bool(true));
        return;
    }

    if (strcmp(q, "dependents") == 0 || strcmp(q, "dependents_installed") == 0) {
        if (!args_ptr || !json_is_string(args_ptr)) {
            json_object_set(out, "crash", json_string("dependents missing string args"));
            return;
        }
        const char *pred = strcmp(q, "dependents") == 0 ? "dependents/3" : "dependents_installed/3";
        WamValue qargs[3] = { cat_term, term_heap_atom(&g_th, json_as_string(args_ptr)), val_unbound("Out") };
        WamValue cells[3];
        if (query_capture(pred, qargs, 3, cells) == 0)
            json_object_set(out, "ok", tj_term_to_json(&g_vm, cells[2]));
        else
            json_object_set(out, "fail", json_bool(true));
        return;
    }

    json_object_set(out, "crash", json_string("unknown query"));
}

int main(void) {
    char *line;
    while ((line = cstr_read_line(stdin)) != NULL) {
        if (line[0] == '\0') {
            free(line);
            continue;
        }

        Json out = json_object();
        JsonParseError err = {0};
        Json row = json_parse(line, &err);
        if (err.message) {
            json_object_set(&out, "crash", json_string(err.message));
        } else {
            /* Each JSONL row owns its entire VM. Tear down choicepoints,
               trail, heap, interned atoms and program tables together, so
               failed or successful queries cannot retain prior row roots. */
            wam_state_init(&g_vm);
            term_heap_init(&g_th, &g_vm);
            if (!g_vm.H_array || !g_vm.TR_array || !g_vm.B_array || !g_vm.E_array)
                g_th.error = "WAM state allocation failed";
            else {
                setup_all_predicates(&g_vm);
                run_case(&row, &out);
            }
            if (g_th.error || g_vm.error) {
                json_free(&out);
                out = json_object();
                json_object_set(&out, "crash", json_string(g_th.error ? g_th.error : "WAM runtime error"));
            }
            json_set_id(&out, &row);
            wam_free_state(&g_vm);
            json_free(&row);
        }
        char *dumped = json_dump(&out);
        puts(dumped);
        fflush(stdout);
        free(dumped);
        json_free(&out);
        free(line);
    }
    return 0;
}
