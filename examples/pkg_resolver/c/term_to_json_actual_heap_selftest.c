/* SPDX-License-Identifier: MIT OR Apache-2.0 */
/* term_to_json_actual_heap_selftest.c -- build terms on the live WAM heap via
 * term_build / term_heap, serialize with term_to_json, compare compact JSON
 * to fixtures from m2-termtojson-spec.md. */
#include "json.h"
#include "term_build.h"
#include "term_heap.h"
#include "term_to_json.h"
#include "test_helpers.h"

#include <stdlib.h>
#include <string.h>

static WamState g_state;
static TermHeap g_th;

static WamValue tb_int(int n) { return term_heap_int(&g_th, n); }
static WamValue tb_atom(const char *s) { return term_heap_atom(&g_th, s); }

static WamValue tb_ver3(int a, int b, int c) {
    WamValue args[3] = { tb_int(a), tb_int(b), tb_int(c) };
    return term_heap_compound(&g_th, "v", 3, args);
}

static WamValue tb_seg(const char *order, int n) {
    size_t olen = strlen(order);
    WamValue *codes = calloc(olen > 0 ? olen : 1, sizeof(WamValue));
    for (size_t i = 0; i < olen; i++)
        codes[i] = tb_int((int)(unsigned char)order[i]);
    WamValue code_list = term_heap_list(&g_th, codes, olen);
    free(codes);
    WamValue args[2] = { code_list, tb_int(n) };
    return term_heap_compound(&g_th, "s", 2, args);
}

static WamValue tb_list_n(WamValue *items, size_t count) {
    return term_heap_list(&g_th, items, count);
}

static WamValue tb_pair(const char *name, WamValue ver) {
    WamValue args[2] = { tb_atom(name), ver };
    return term_heap_compound(&g_th, "-", 2, args);
}

static WamValue tb_st1(const char *name, WamValue a) {
    WamValue args[1] = { a };
    return term_heap_compound(&g_th, name, 1, args);
}

static WamValue tb_st2(const char *name, WamValue a, WamValue b) {
    WamValue args[2] = { a, b };
    return term_heap_compound(&g_th, name, 2, args);
}

static WamValue tb_st3(const char *name, WamValue a, WamValue b, WamValue c) {
    WamValue args[3] = { a, b, c };
    return term_heap_compound(&g_th, name, 3, args);
}

static void check_json(WamValue term, const char *want, const char *what) {
    Json j = tj_term_to_json(&g_state, term);
    char *got = json_dump(&j);
    check_str(got, want, what);
    free(got);
    json_free(&j);
}

static void check_json_fn(WamValue term, Json (*fn)(WamState *, WamValue), const char *want, const char *what) {
    Json j = fn(&g_state, term);
    char *got = json_dump(&j);
    check_str(got, want, what);
    free(got);
    json_free(&j);
}

int main(void) {
    wam_state_init(&g_state);
    term_heap_init(&g_th, &g_state);

    check_json(tb_int(42), "42", "term_to_json integer");
    check_json(tb_atom("[]"), "[]", "term_to_json atom [] -> empty array");
    check_json(tb_atom("foo"), "\"foo\"", "term_to_json plain atom");
    check_json(tb_atom("any"), "\"any\"", "term_to_json atom any (generic pass-through)");
    {
        WamValue items[3] = { tb_int(1), tb_int(2), tb_int(3) };
        check_json(tb_list_n(items, 3), "[1,2,3]", "term_to_json [|]/2 list chain");
    }
    check_json(tb_ver3(0, 2, 0), "[0,2,0]", "term_to_json routes v/3 to ver_to_json");
    check_json(tb_pair("p2", tb_ver3(0, 1, 0)), "[\"p2\",[0,1,0]]", "term_to_json routes -/2 to pair_to_json");
    {
        WamValue args[2] = { tb_int(1), tb_int(2) };
        check_json(term_heap_compound(&g_th, "foo", 2, args), "[\"foo\",1,2]", "term_to_json untagged compound fallback");
    }

    check_json_fn(tb_ver3(0, 2, 0), tj_ver_to_json, "[0,2,0]", "ver_to_json v/3");
    {
        WamValue segs_up[2] = { tb_seg("", 1), tb_seg(".", 0) };
        WamValue segs_rev[1] = { tb_seg("", 2) };
        WamValue deb_args[3] = { tb_int(0), tb_list_n(segs_up, 2), tb_list_n(segs_rev, 1) };
        check_json_fn(term_heap_compound(&g_th, "deb", 3, deb_args), tj_ver_to_json,
                      "{\"deb\":[0,[[\"\",1],[\".\",0]],[[\"\",2]]]}",
                      "ver_to_json deb/3 (mirrors json_selftest.cpp F2 in reverse)");
    }

    check_json_fn(tb_seg("", 1), tj_seg_to_json, "[\"\",1]", "seg_to_json empty order");
    check_json_fn(tb_seg(".", 0), tj_seg_to_json, "[\".\",0]", "seg_to_json \".\" order (byte 46)");
    check_json_fn(tb_pair("p2", tb_ver3(0, 1, 0)), tj_pair_to_json, "[\"p2\",[0,1,0]]", "pair_to_json");
    {
        WamValue pairs[2] = { tb_pair("p0", tb_ver3(0, 2, 0)), tb_pair("p2", tb_ver3(0, 1, 0)) };
        check_json_fn(tb_list_n(pairs, 2), tj_sel_to_json,
                      "[[\"p0\",[0,2,0]],[\"p2\",[0,1,0]]]", "sel_to_json two pairs");
    }
    check_json_fn(tb_atom("[]"), tj_sel_to_json, "[]", "sel_to_json empty selection");

    check_json_fn(tb_atom("any"), tj_normalize_constraint, "\"any\"", "normalize_constraint any");
    check_json_fn(tb_st1("eq", tb_ver3(0, 1, 0)), tj_normalize_constraint,
                  "{\"op\":\"eq\",\"v\":[0,1,0]}", "normalize_constraint eq");
    check_json_fn(tb_st1("gte", tb_ver3(0, 1, 0)), tj_normalize_constraint,
                  "{\"op\":\"gte\",\"v\":[0,1,0]}", "normalize_constraint gte");
    check_json_fn(tb_st1("lt", tb_ver3(1, 0, 0)), tj_normalize_constraint,
                  "{\"op\":\"lt\",\"v\":[1,0,0]}", "normalize_constraint lt");
    check_json_fn(tb_st1("lte", tb_ver3(1, 0, 0)), tj_normalize_constraint,
                  "{\"op\":\"lte\",\"v\":[1,0,0]}", "normalize_constraint lte");
    check_json_fn(tb_st1("gt", tb_ver3(1, 0, 0)), tj_normalize_constraint,
                  "{\"op\":\"gt\",\"v\":[1,0,0]}", "normalize_constraint gt");
    check_json_fn(tb_st2("range", tb_ver3(0, 2, 0), tb_ver3(1, 0, 0)), tj_normalize_constraint,
                  "{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}", "normalize_constraint range");

    {
        WamValue blocked = tb_st3("blocked", tb_atom("p1"),
                                  tb_st1("needs", tb_atom("any")),
                                  tb_st1("base_has", tb_ver3(0, 0, 5)));
        check_json_fn(blocked, tj_normalize_blocked,
                      "{\"name\":\"p1\",\"needs\":\"any\",\"base_has\":[0,0,5]}",
                      "normalize_blocked base_has (resolver.pl:723/991 shape)");
    }
    {
        WamValue inner = tb_st3("blocked", tb_atom("p2"),
                                  tb_st1("needs", tb_atom("any")),
                                  tb_st1("base_has", tb_ver3(0, 1, 0)));
        WamValue blocked = tb_st3("blocked", tb_atom("virt"),
                                  tb_st1("needs", tb_atom("any")),
                                  tb_st1("providers", tb_list_n(&inner, 1)));
        check_json_fn(blocked, tj_normalize_blocked,
                      "{\"name\":\"virt\",\"needs\":\"any\",\"providers\":[{\"name\":\"p2\",\"needs\":\"any\",\"base_has\":[0,1,0]}]}",
                      "normalize_blocked providers (resolver.pl:727/767-773 shape)");
    }
    {
        WamValue nested = tb_st3("blocked", tb_atom("p6b"),
                                 tb_st1("needs", tb_st1("gte", tb_ver3(0, 1, 0))),
                                 tb_st1("base_has", tb_ver3(0, 0, 1)));
        WamValue alts[2] = {
            tb_st2("alt", tb_atom("p6"), tb_atom("unsatisfiable")),
            tb_st2("alt", tb_atom("p6b"), nested)
        };
        WamValue blocked = tb_st1("blocked", tb_st1("alternatives", tb_list_n(alts, 2)));
        check_json_fn(blocked, tj_normalize_blocked,
                      "{\"alternatives\":[{\"dep\":\"p6\",\"reason\":\"unsatisfiable\"},"
                      "{\"dep\":\"p6b\",\"reason\":{\"name\":\"p6b\",\"needs\":{\"op\":\"gte\",\"v\":[0,1,0]},"
                      "\"base_has\":[0,0,1]}}]}",
                      "normalize_blocked alternatives (resolver.pl:713-716/780-785 shape)");
    }

    check_json_fn(tb_atom("[]"), tj_blocked_list_to_json, "[]", "blocked_list_to_json empty");
    {
        WamValue b1 = tb_st3("blocked", tb_atom("p1"),
                             tb_st1("needs", tb_atom("any")),
                             tb_st1("base_has", tb_ver3(0, 0, 1)));
        WamValue list[1] = { b1 };
        check_json_fn(tb_list_n(list, 1), tj_blocked_list_to_json,
                      "[{\"name\":\"p1\",\"needs\":\"any\",\"base_has\":[0,0,1]}]",
                      "blocked_list_to_json one element");
    }

    check_json_fn(tb_atom("no_candidate"), tj_normalize_verdict,
                  "{\"verdict\":\"no_candidate\"}", "normalize_verdict no_candidate");
    check_json_fn(tb_st1("safe", tb_st1("cost", tb_atom("footprint"))), tj_normalize_verdict,
                  "{\"cost\":\"footprint\",\"verdict\":\"safe\"}", "normalize_verdict safe/cost");
    {
        WamValue set[1] = { tb_pair("p1", tb_ver3(0, 2, 0)) };
        check_json_fn(tb_st1("coordinated", tb_list_n(set, 1)), tj_normalize_verdict,
                      "{\"set\":[[\"p1\",[0,2,0]]],\"verdict\":\"coordinated\"}", "normalize_verdict coordinated");
    }
    check_json_fn(tb_st1("unsafe", tb_atom("modified")), tj_normalize_verdict,
                  "{\"reason\":\"modified\",\"verdict\":\"unsafe\"}", "normalize_verdict unsafe");

    check_json_fn(tb_atom("no_candidate"), tj_normalize_upgrade,
                  "{\"fail\":true}", "normalize_upgrade no_candidate");
    {
        WamValue ok_set[2] = { tb_pair("p1", tb_ver3(0, 2, 0)), tb_pair("p2", tb_ver3(0, 1, 0)) };
        check_json_fn(tb_st1("ok", tb_list_n(ok_set, 2)), tj_normalize_upgrade,
                      "{\"ok\":[[\"p1\",[0,2,0]],[\"p2\",[0,1,0]]]}", "normalize_upgrade ok");
    }
    {
        WamValue b = tb_st3("blocked", tb_atom("p1"),
                            tb_st1("needs", tb_atom("any")),
                            tb_st1("base_has", tb_ver3(0, 0, 1)));
        check_json_fn(b, tj_normalize_upgrade,
                      "{\"ok\":{\"blocked\":{\"name\":\"p1\",\"needs\":\"any\",\"base_has\":[0,0,1]}}}",
                      "normalize_upgrade blocked (resolver.pl:991 -- the only shape this predicate emits)");
    }

    check_json_fn(tb_st2("audit", tb_atom("p10"), tb_atom("over_frozen")), tj_normalize_audit_term,
                  "{\"kind\":\"over_frozen\",\"name\":\"p10\"}", "normalize_audit_term over_frozen");
    check_json_fn(tb_st2("audit", tb_atom("p11"), tb_st1("suggest", tb_atom("abi_anchor"))), tj_normalize_audit_term,
                  "{\"kind\":\"suggest\",\"name\":\"p11\",\"reason\":\"abi_anchor\"}", "normalize_audit_term suggest");
    check_json_fn(tb_st2("audit", tb_atom("p12"), tb_st1("held", tb_atom("layer_shadow"))), tj_normalize_audit_term,
                  "{\"kind\":\"held\",\"name\":\"p12\",\"reason\":\"layer_shadow\"}", "normalize_audit_term held");

    check_json_fn(tb_atom("[]"), tj_audit_list_to_json, "[]", "audit_list_to_json empty");
    {
        WamValue audits[3] = {
            tb_st2("audit", tb_atom("p10"), tb_atom("over_frozen")),
            tb_st2("audit", tb_atom("p11"), tb_st1("suggest", tb_atom("abi_anchor"))),
            tb_st2("audit", tb_atom("p12"), tb_st1("held", tb_atom("layer_shadow")))
        };
        check_json_fn(tb_list_n(audits, 3), tj_audit_list_to_json,
                      "[{\"kind\":\"over_frozen\",\"name\":\"p10\"},"
                      "{\"kind\":\"suggest\",\"name\":\"p11\",\"reason\":\"abi_anchor\"},"
                      "{\"kind\":\"held\",\"name\":\"p12\",\"reason\":\"layer_shadow\"}]",
                      "audit_list_to_json three elements");
    }

    wam_free_state(&g_state);
    return finish_tests();
}
