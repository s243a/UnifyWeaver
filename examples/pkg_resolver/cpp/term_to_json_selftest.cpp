// SPDX-License-Identifier: MIT OR Apache-2.0
// term_to_json_selftest.cpp -- fixed acceptance test for the M2
// term-to-json layer (WAM result Value -> JSON). Exercises the fixtures
// from m2-termtojson-spec.md. Prints PASS/FAIL per check and a summary;
// exits 0 iff every check passed.
#include "json.hpp"
#include "term_to_json.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using wam_cpp::Value;
using wam_cpp::CellPtr;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& what) {
    if (cond) {
        g_pass++;
    } else {
        g_fail++;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

void checkStr(const std::string& got, const std::string& want, const std::string& what) {
    check(got == want, what + " (got \"" + got + "\", want \"" + want + "\")");
}

// ---- local Value-tree builders (fixtures only; NOT part of any shipped
// interface -- a real driver gets these terms from WamState output
// registers, not from hand construction) ----

CellPtr cell(Value v) { return std::make_shared<Value>(std::move(v)); }

Value atomv(std::string s) { return Value::Atom(std::move(s)); }
Value intv(int64_t n) { return Value::Integer(n); }

Value st(const std::string& name, std::vector<Value> args) {
    std::vector<CellPtr> cargs;
    cargs.reserve(args.size());
    for (auto& a : args) cargs.push_back(cell(std::move(a)));
    // Compute the functor string into a local BEFORE the Value::Compound
    // call -- see m2-termbuild-spec.md's writeup of the argument
    // evaluation-order footgun (cargs.size() read after cargs is
    // moved-from would silently yield "name/0").
    std::string functor = name + "/" + std::to_string(cargs.size());
    return Value::Compound(functor, std::move(cargs));
}

Value listv(std::vector<Value> items) {
    Value acc = Value::Atom("[]");
    for (auto it = items.rbegin(); it != items.rend(); ++it) {
        std::vector<CellPtr> args;
        args.push_back(cell(std::move(*it)));
        args.push_back(cell(std::move(acc)));
        acc = Value::Compound("[|]/2", std::move(args));
    }
    return acc;
}

Value ver3(int64_t a, int64_t b, int64_t c) { return st("v", {intv(a), intv(b), intv(c)}); }

Value seg(std::string order, int64_t n) {
    std::vector<Value> codes;
    for (unsigned char ch : order) codes.push_back(intv(static_cast<int64_t>(ch)));
    return st("s", {listv(std::move(codes)), intv(n)});
}

Value pair(const std::string& name, Value ver) { return st("-", {atomv(name), std::move(ver)}); }

}  // namespace

int main() {
    using json::dump;
    using namespace term_json;

    // ---- term_to_json: generic leaves ----
    checkStr(dump(term_to_json(intv(42))), "42", "term_to_json integer");
    checkStr(dump(term_to_json(atomv("[]"))), "[]", "term_to_json atom [] -> empty array");
    checkStr(dump(term_to_json(atomv("foo"))), "\"foo\"", "term_to_json plain atom");
    checkStr(dump(term_to_json(atomv("any"))), "\"any\"", "term_to_json atom any (generic pass-through)");
    checkStr(dump(term_to_json(listv({intv(1), intv(2), intv(3)}))), "[1,2,3]",
             "term_to_json [|]/2 list chain");
    checkStr(dump(term_to_json(ver3(0, 2, 0))), "[0,2,0]", "term_to_json routes v/3 to ver_to_json");
    checkStr(dump(term_to_json(pair("p2", ver3(0, 1, 0)))), "[\"p2\",[0,1,0]]",
             "term_to_json routes -/2 to pair_to_json");
    checkStr(dump(term_to_json(st("foo", {intv(1), intv(2)}))), "[\"foo\",1,2]",
             "term_to_json untagged compound fallback");

    // ---- ver_to_json ----
    checkStr(dump(ver_to_json(ver3(0, 2, 0))), "[0,2,0]", "ver_to_json v/3");
    {
        Value deb = st("deb", {intv(0), listv({seg("", 1), seg(".", 0)}), listv({seg("", 2)})});
        checkStr(dump(ver_to_json(deb)), "{\"deb\":[0,[[\"\",1],[\".\",0]],[[\"\",2]]]}",
                 "ver_to_json deb/3 (mirrors json_selftest.cpp F2 in reverse)");
    }

    // ---- seg_to_json ----
    checkStr(dump(seg_to_json(seg("", 1))), "[\"\",1]", "seg_to_json empty order");
    checkStr(dump(seg_to_json(seg(".", 0))), "[\".\",0]", "seg_to_json \".\" order (byte 46)");

    // ---- pair_to_json ----
    checkStr(dump(pair_to_json(pair("p2", ver3(0, 1, 0)))), "[\"p2\",[0,1,0]]", "pair_to_json");

    // ---- sel_to_json ----
    checkStr(dump(sel_to_json(listv({pair("p0", ver3(0, 2, 0)), pair("p2", ver3(0, 1, 0))}))),
             "[[\"p0\",[0,2,0]],[\"p2\",[0,1,0]]]", "sel_to_json two pairs");
    checkStr(dump(sel_to_json(atomv("[]"))), "[]", "sel_to_json empty selection");

    // ---- normalize_constraint ----
    checkStr(dump(normalize_constraint(atomv("any"))), "\"any\"", "normalize_constraint any");
    checkStr(dump(normalize_constraint(st("eq", {ver3(0, 1, 0)}))), "{\"op\":\"eq\",\"v\":[0,1,0]}",
             "normalize_constraint eq");
    checkStr(dump(normalize_constraint(st("gte", {ver3(0, 1, 0)}))), "{\"op\":\"gte\",\"v\":[0,1,0]}",
             "normalize_constraint gte");
    checkStr(dump(normalize_constraint(st("lt", {ver3(1, 0, 0)}))), "{\"op\":\"lt\",\"v\":[1,0,0]}",
             "normalize_constraint lt");
    checkStr(dump(normalize_constraint(st("lte", {ver3(1, 0, 0)}))), "{\"op\":\"lte\",\"v\":[1,0,0]}",
             "normalize_constraint lte");
    checkStr(dump(normalize_constraint(st("gt", {ver3(1, 0, 0)}))), "{\"op\":\"gt\",\"v\":[1,0,0]}",
             "normalize_constraint gt");
    checkStr(dump(normalize_constraint(st("range", {ver3(0, 2, 0), ver3(1, 0, 0)}))),
             "{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}", "normalize_constraint range");

    // ---- normalize_blocked ----
    {
        Value b = st("blocked", {atomv("p1"), st("needs", {st("gte", {ver3(0, 1, 0)})}),
                                  st("base_has", {ver3(0, 0, 5)})});
        checkStr(dump(normalize_blocked(b)),
                 "{\"name\":\"p1\",\"needs\":{\"op\":\"gte\",\"v\":[0,1,0]},\"base_has\":[0,0,5]}",
                 "normalize_blocked base_has (resolver.pl:723/991 shape)");
    }
    {
        Value inner = st("blocked", {atomv("p2"), st("needs", {atomv("any")}), st("base_has", {ver3(0, 1, 0)})});
        Value b = st("blocked", {atomv("virt"), st("needs", {atomv("any")}), st("providers", {listv({inner})})});
        checkStr(dump(normalize_blocked(b)),
                 "{\"name\":\"virt\",\"needs\":\"any\",\"providers\":[{\"name\":\"p2\",\"needs\":\"any\",\"base_has\":[0,1,0]}]}",
                 "normalize_blocked providers (resolver.pl:727/767-773 shape)");
    }
    {
        Value nested = st("blocked", {atomv("p6b"), st("needs", {st("gte", {ver3(0, 1, 0)})}),
                                       st("base_has", {ver3(0, 0, 1)})});
        Value alts = listv({st("alt", {atomv("p6"), atomv("unsatisfiable")}),
                             st("alt", {atomv("p6b"), nested})});
        Value b = st("blocked", {st("alternatives", {alts})});
        checkStr(dump(normalize_blocked(b)),
                 "{\"alternatives\":[{\"dep\":\"p6\",\"reason\":\"unsatisfiable\"},"
                 "{\"dep\":\"p6b\",\"reason\":{\"name\":\"p6b\",\"needs\":{\"op\":\"gte\",\"v\":[0,1,0]},"
                 "\"base_has\":[0,0,1]}}]}",
                 "normalize_blocked alternatives (resolver.pl:713-716/780-785 shape)");
    }

    // ---- blocked_list_to_json ----
    checkStr(dump(blocked_list_to_json(atomv("[]"))), "[]", "blocked_list_to_json empty");
    {
        Value b1 = st("blocked", {atomv("p1"), st("needs", {atomv("any")}), st("base_has", {ver3(0, 0, 1)})});
        checkStr(dump(blocked_list_to_json(listv({b1}))),
                 "[{\"name\":\"p1\",\"needs\":\"any\",\"base_has\":[0,0,1]}]",
                 "blocked_list_to_json one element");
    }

    // ---- normalize_verdict (resolver.pl:952-967) ----
    checkStr(dump(normalize_verdict(atomv("no_candidate"))), "{\"verdict\":\"no_candidate\"}",
             "normalize_verdict no_candidate");
    checkStr(dump(normalize_verdict(st("safe", {st("cost", {atomv("footprint")})}))),
             "{\"cost\":\"footprint\",\"verdict\":\"safe\"}", "normalize_verdict safe/cost");
    checkStr(dump(normalize_verdict(st("coordinated", {listv({pair("p1", ver3(0, 2, 0))})}))),
             "{\"set\":[[\"p1\",[0,2,0]]],\"verdict\":\"coordinated\"}", "normalize_verdict coordinated");
    checkStr(dump(normalize_verdict(st("unsafe", {atomv("modified")}))),
             "{\"reason\":\"modified\",\"verdict\":\"unsafe\"}", "normalize_verdict unsafe");

    // ---- normalize_upgrade (resolver.pl:974-993) ----
    checkStr(dump(normalize_upgrade(atomv("no_candidate"))), "{\"fail\":true}", "normalize_upgrade no_candidate");
    checkStr(dump(normalize_upgrade(st("ok", {listv({pair("p1", ver3(0, 2, 0)), pair("p2", ver3(0, 1, 0))})}))),
             "{\"ok\":[[\"p1\",[0,2,0]],[\"p2\",[0,1,0]]]}", "normalize_upgrade ok");
    {
        Value b = st("blocked", {atomv("p1"), st("needs", {atomv("any")}), st("base_has", {ver3(0, 0, 1)})});
        checkStr(dump(normalize_upgrade(b)),
                 "{\"ok\":{\"blocked\":{\"name\":\"p1\",\"needs\":\"any\",\"base_has\":[0,0,1]}}}",
                 "normalize_upgrade blocked (resolver.pl:991 -- the only shape this predicate emits)");
    }

    // ---- normalize_audit_term (resolver.pl:1054-1069) ----
    checkStr(dump(normalize_audit_term(st("audit", {atomv("p10"), atomv("over_frozen")}))),
             "{\"kind\":\"over_frozen\",\"name\":\"p10\"}", "normalize_audit_term over_frozen");
    checkStr(dump(normalize_audit_term(st("audit", {atomv("p11"), st("suggest", {atomv("abi_anchor")})}))),
             "{\"kind\":\"suggest\",\"name\":\"p11\",\"reason\":\"abi_anchor\"}",
             "normalize_audit_term suggest");
    checkStr(dump(normalize_audit_term(st("audit", {atomv("p12"), st("held", {atomv("layer_shadow")})}))),
             "{\"kind\":\"held\",\"name\":\"p12\",\"reason\":\"layer_shadow\"}", "normalize_audit_term held");

    // ---- audit_list_to_json ----
    checkStr(dump(audit_list_to_json(atomv("[]"))), "[]", "audit_list_to_json empty");
    {
        Value a1 = st("audit", {atomv("p10"), atomv("over_frozen")});
        Value a2 = st("audit", {atomv("p11"), st("suggest", {atomv("abi_anchor")})});
        Value a3 = st("audit", {atomv("p12"), st("held", {atomv("layer_shadow")})});
        checkStr(dump(audit_list_to_json(listv({a1, a2, a3}))),
                 "[{\"kind\":\"over_frozen\",\"name\":\"p10\"},"
                 "{\"kind\":\"suggest\",\"name\":\"p11\",\"reason\":\"abi_anchor\"},"
                 "{\"kind\":\"held\",\"name\":\"p12\",\"reason\":\"layer_shadow\"}]",
                 "audit_list_to_json three elements");
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
