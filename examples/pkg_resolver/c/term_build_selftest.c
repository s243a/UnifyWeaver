/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* term_build_selftest.c -- catalog JSON -> actual WAM heap terms. */
#include "json.h"
#include "term_build.h"
#include "term_heap.h"
#include "term_render.h"
#include "test_helpers.h"

#include <stdlib.h>

static WamState g_state;
static TermHeap g_th;

static Json parse_ok(const char *text) {
    JsonParseError err = {0};
    Json v = json_parse(text, &err);
    if (err.message) {
        check(0, err.message);
        return json_null();
    }
    return v;
}

int main(void) {
    wam_state_init(&g_state);
    term_heap_init(&g_th, &g_state);

    char *invalid = term_render(&g_state, tb_ver_term(&g_th, &((Json){0})));
    check_str(invalid, "?", "ver_term invalid");
    free(invalid);

    {
        Json v = parse_ok("[0,2,0]");
        char *r = term_render(&g_state, tb_ver_term(&g_th, &v));
        check_str(r, "v(0,2,0)", "ver_term v/3");
        free(r);
        json_free(&v);
    }

    {
        Json v = parse_ok("{\"deb\":[0,[[\"\",1],[\".\",0]],[[\"\",2]]]}");
        WamValue got = tb_ver_term(&g_th, &v);
        char *r = term_render(&g_state, got);
        check_str(r, "deb(0,[s([],1),s([46],0)],[s([],2)])", "ver_term deb/3");
        free(r);
        json_free(&v);
    }

    {
        Json v = parse_ok("\"any\"");
        char *r = term_render(&g_state, tb_constraint_term(&g_th, &v));
        check_str(r, "any", "constraint_term any");
        free(r);
        json_free(&v);
    }

    {
        Json v = parse_ok("{\"op\":\"eq\",\"v\":[0,1,0]}");
        char *r = term_render(&g_state, tb_constraint_term(&g_th, &v));
        check_str(r, "eq(v(0,1,0))", "constraint_term eq");
        free(r);
        json_free(&v);
    }

    {
        Json v = parse_ok("{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}");
        char *r = term_render(&g_state, tb_constraint_term(&g_th, &v));
        check_str(r, "range(v(0,2,0),v(1,0,0))", "constraint_term range");
        free(r);
        json_free(&v);
    }

    {
        Json ver = parse_ok("[0,1,0]");
        char *r = term_render(&g_state, tb_pair_term(&g_th, "p2", &ver));
        check_str(r, "-(p2,v(0,1,0))", "pair_term -/2");
        free(r);
        json_free(&ver);
    }

    {
        const char *CAT6 =
            "{\"packages\":[[\"p0\",[0,2,0]]],\"depends\":[],\"conflicts\":[],"
            "\"base\":[],\"installed\":[],\"requested\":[\"p0\"]}";
        Json cat = parse_ok(CAT6);
        WamValue got = tb_catalog_to_term(&g_th, &cat);
        char *r = term_render(&g_state, got);
        check_str(r, "catalog([package(p0,v(0,2,0))],[],[],[],[],[p0])", "catalog_to_term catalog/6");
        free(r);
        json_free(&cat);
    }

    wam_free_state(&g_state);
    return finish_tests();
}
