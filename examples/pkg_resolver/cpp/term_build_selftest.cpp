// SPDX-License-Identifier: MIT OR Apache-2.0
// term_build_selftest.cpp -- fixed acceptance test for the M2 term-builder
// layer (catalog JSON -> WAM Value terms). Exercises the fixtures from
// m2-termbuild-spec.md. Prints PASS/FAIL per check and a summary; exits 0
// iff every check passed.
#include "json.hpp"
#include "term_build.hpp"

#include <cstdio>
#include <memory>
#include <string>

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

// Debug-only canonical renderer, private to this test file (NOT part of
// term_build.hpp's public API). Atom -> name, Integer -> decimal, a
// "[|]/2" chain -> [a,b,c] bracket notation, every other compound ->
// name(arg,arg,...) where name is the functor with its "/arity" suffix
// stripped. This intentionally does NOT re-derive arity from args.size()
// (that would hide the exact bug this spec warns about) -- it just prints
// what's actually stored in Value::s / Value::args.
std::string render(const Value& v) {
    switch (v.tag) {
        case Value::Tag::Atom:
            return v.s;
        case Value::Tag::Integer:
            return std::to_string(v.i);
        case Value::Tag::Compound: {
            auto slash = v.s.rfind('/');
            std::string name = slash == std::string::npos ? v.s : v.s.substr(0, slash);
            if (name == "[|]" && v.args.size() == 2) {
                std::string out = "[";
                const Value* cur = &v;
                bool first = true;
                while (cur->tag == Value::Tag::Compound && cur->s == "[|]/2") {
                    if (!first) out += ",";
                    first = false;
                    out += render(*cur->args[0]);
                    cur = cur->args[1].get();
                }
                out += "]";
                return out;
            }
            std::string out = name + "(";
            for (size_t i = 0; i < v.args.size(); ++i) {
                if (i) out += ",";
                out += render(*v.args[i]);
            }
            out += ")";
            return out;
        }
        default:
            return "?";
    }
}

}  // namespace

int main() {
    using json::parse;
    using term_build::alias_term;
    using term_build::catalog_to_term;
    using term_build::conf_term;
    using term_build::constraint_term;
    using term_build::dep_term;
    using term_build::hold_term;
    using term_build::layer_term;
    using term_build::pair_term;
    using term_build::pkg_term;
    using term_build::provide_term;
    using term_build::request_term;
    using term_build::ver_term;

    // ---- ver_term: v/3 ----
    checkStr(render(ver_term(parse("[0,2,0]"))), "v(0,2,0)", "ver_term v/3");

    // ---- ver_term: deb/3 (fixture = json_selftest.cpp's F2) ----
    {
        Value got = ver_term(parse("{\"deb\":[0,[[\"\",1],[\".\",0]],[[\"\",2]]]}"));
        checkStr(render(got), "deb(0,[s([],1),s([46],0)],[s([],2)])", "ver_term deb/3");
        check(got.tag == Value::Tag::Compound && got.s == "deb/3", "ver_term deb/3 raw functor");
    }

    // ---- constraint_term: every op ----
    checkStr(render(constraint_term(parse("\"any\""))), "any", "constraint_term any(string)");
    checkStr(render(constraint_term(parse("{\"op\":\"eq\",\"v\":[0,1,0]}"))), "eq(v(0,1,0))", "constraint_term eq");
    checkStr(render(constraint_term(parse("{\"op\":\"gte\",\"v\":[0,1,0]}"))), "gte(v(0,1,0))", "constraint_term gte");
    checkStr(render(constraint_term(parse("{\"op\":\"lt\",\"v\":[1,0,0]}"))), "lt(v(1,0,0))", "constraint_term lt");
    checkStr(render(constraint_term(parse("{\"op\":\"lte\",\"v\":[1,0,0]}"))), "lte(v(1,0,0))", "constraint_term lte");
    checkStr(render(constraint_term(parse("{\"op\":\"gt\",\"v\":[1,0,0]}"))), "gt(v(1,0,0))", "constraint_term gt");
    checkStr(render(constraint_term(parse("{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}"))),
             "range(v(0,2,0),v(1,0,0))", "constraint_term range");

    // ---- pair_term: the Name-Ver "-/2" pair (base entries, installed entries) ----
    {
        Value got = pair_term("p2", parse("[0,1,0]"));
        checkStr(render(got), "-(p2,v(0,1,0))", "pair_term -/2");
        check(got.tag == Value::Tag::Compound && got.s == "-/2", "pair_term raw functor is \"-/2\"");
    }

    // ---- hold_term: 2-elem (bare Name-Ver) vs 3-elem (base/2 with Reason) ----
    checkStr(render(hold_term(parse("[\"p2\",[0,1,0]]"))), "-(p2,v(0,1,0))", "hold_term 2-elem row");
    checkStr(render(hold_term(parse("[\"p10\",[0,2,0],\"abi_anchor\"]"))),
             "base(-(p10,v(0,2,0)),abi_anchor)", "hold_term 3-elem row");

    // ---- layer_term ----
    checkStr(render(layer_term(parse("{\"name\":\"devx\",\"packages\":[[\"p7\",[0,1,0]]]}"))),
             "layer(devx,[-(p7,v(0,1,0))])", "layer_term");

    // ---- alias_term ----
    checkStr(render(alias_term(parse("[\"py\",\"python3\"]"))), "alias(py,python3)", "alias_term");

    // ---- pkg_term ----
    checkStr(render(pkg_term(parse("[\"p0\",[0,2,0]]"))), "package(p0,v(0,2,0))", "pkg_term");

    // ---- dep_term: plain dep name + range constraint (F1 depends[0]) ----
    checkStr(render(dep_term(parse(
                 "[\"p1\",[0,1,0],\"p0\",{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}]"))),
             "depends(p1,v(0,1,0),p0,range(v(0,2,0),v(1,0,0)))", "dep_term plain");

    // ---- dep_term: alternatives ----
    checkStr(render(dep_term(parse(
                 "[\"p9\",[0,1,0],{\"alternatives\":[{\"dep\":\"p6\",\"constraint\":\"any\"},"
                 "{\"dep\":\"p6b\",\"constraint\":{\"op\":\"gte\",\"v\":[0,1,0]}}]},\"any\"]"))),
             "depends(p9,v(0,1,0),alternatives([dep(p6,any),dep(p6b,gte(v(0,1,0)))]),any)",
             "dep_term alternatives");

    // ---- provide_term: unversioned (3-elem) vs versioned (4-elem) ----
    checkStr(render(provide_term(parse("[\"p2\",[0,1,0],\"virt-a\"]"))),
             "provides(p2,v(0,1,0),virt-a)", "provide_term unversioned");
    checkStr(render(provide_term(parse("[\"p2\",[0,1,0],\"virt-a\",[1,0,0]]"))),
             "provides(p2,v(0,1,0),virt-a,v(1,0,0))", "provide_term versioned");

    // ---- conf_term ----
    checkStr(render(conf_term(parse("[\"p0\",[0,2,0],\"p8\"]"))),
             "conflicts(p0,v(0,2,0),p8)", "conf_term");

    // ---- request_term: bare atom vs req(Name,Constraint) object ----
    checkStr(render(request_term(parse("\"p0\""))), "p0", "request_term bare");
    checkStr(render(request_term(parse("{\"req\":\"p0\",\"constraint\":{\"op\":\"gte\",\"v\":[0,1,0]}}"))),
             "req(p0,gte(v(0,1,0)))", "request_term req-object");
    checkStr(render(request_term(parse("{\"req\":\"p0\"}"))), "req(p0,any)",
             "request_term req-object, no constraint key -> any");

    // ---- catalog_to_term: catalog/6 (layers/excluded/aliases/provides all absent-or-empty) ----
    {
        const char* CAT6 =
            "{\"packages\":[[\"p0\",[0,2,0]]],\"depends\":[],\"conflicts\":[],"
            "\"base\":[],\"installed\":[],\"requested\":[\"p0\"]}";
        Value got = catalog_to_term(parse(CAT6));
        checkStr(render(got), "catalog([package(p0,v(0,2,0))],[],[],[],[],[p0])", "catalog_to_term catalog/6");
        check(got.args.size() == 6 && got.s == "catalog/6", "catalog_to_term catalog/6 raw arity");
    }

    // ---- catalog_to_term: catalog/9 (F1 fixture -- layers non-empty, excluded/aliases empty, no provides) ----
    {
        const char* F1CAT =
            "{\"packages\":[[\"p0\",[0,2,0]],[\"p1\",[0,1,0]],[\"p2\",[0,1,0]],"
            "[\"p3\",[0,2,0]],[\"p4\",[0,2,0]],[\"p5\",[0,1,0]],[\"p6\",[0,1,0]],[\"p7\",[0,1,0]],[\"p8\",[0,1,0]],"
            "[\"p9\",[0,1,0]]],\"depends\":[[\"p1\",[0,1,0],\"p0\",{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}],"
            "[\"p2\",[0,1,0],\"p0\",\"any\"],[\"p3\",[0,2,0],\"p2\",{\"op\":\"gte\",\"v\":[0,1,0]}],"
            "[\"p4\",[0,2,0],\"p2\",\"any\"],[\"p7\",[0,1,0],\"p1\",{\"op\":\"gte\",\"v\":[0,1,0]}],"
            "[\"p8\",[0,1,0],\"p5\",{\"op\":\"eq\",\"v\":[0,1,0]}],[\"p9\",[0,1,0],\"p6\",\"any\"]],"
            "\"conflicts\":[[\"p0\",[0,2,0],\"p8\"]],\"base\":[],\"installed\":[[\"p2\",[0,1,0]]],\"requested\":[],"
            "\"layers\":[{\"name\":\"devx\",\"packages\":[[\"p7\",[0,1,0]]]}],\"excluded\":[],\"aliases\":[]}";
        Value got = catalog_to_term(parse(F1CAT));
        checkStr(render(got),
                 "catalog([package(p0,v(0,2,0)),package(p1,v(0,1,0)),package(p2,v(0,1,0)),package(p3,v(0,2,0)),"
                 "package(p4,v(0,2,0)),package(p5,v(0,1,0)),package(p6,v(0,1,0)),package(p7,v(0,1,0)),"
                 "package(p8,v(0,1,0)),package(p9,v(0,1,0))],"
                 "[depends(p1,v(0,1,0),p0,range(v(0,2,0),v(1,0,0))),depends(p2,v(0,1,0),p0,any),"
                 "depends(p3,v(0,2,0),p2,gte(v(0,1,0))),depends(p4,v(0,2,0),p2,any),"
                 "depends(p7,v(0,1,0),p1,gte(v(0,1,0))),depends(p8,v(0,1,0),p5,eq(v(0,1,0))),"
                 "depends(p9,v(0,1,0),p6,any)],"
                 "[conflicts(p0,v(0,2,0),p8)],[],[-(p2,v(0,1,0))],[],"
                 "[layer(devx,[-(p7,v(0,1,0))])],[],[])",
                 "catalog_to_term catalog/9 (F1)");
        check(got.args.size() == 9 && got.s == "catalog/9", "catalog_to_term catalog/9 raw arity");
    }

    // ---- catalog_to_term: catalog/10 (provides non-empty; layers/excluded/aliases empty but present) ----
    {
        const char* CAT10 =
            "{\"packages\":[[\"p0\",[0,2,0]]],\"depends\":[],\"conflicts\":[],"
            "\"base\":[[\"p0\",[0,2,0],\"blanket\"]],\"installed\":[],\"requested\":[],"
            "\"layers\":[],\"excluded\":[],\"aliases\":[],"
            "\"provides\":[[\"p2\",[0,1,0],\"virt-a\"],[\"p3\",[0,1,0],\"virt-b\",[2,0,0]]]}";
        Value got = catalog_to_term(parse(CAT10));
        checkStr(render(got),
                 "catalog([package(p0,v(0,2,0))],[],[],[base(-(p0,v(0,2,0)),blanket)],[],[],[],[],[],"
                 "[provides(p2,v(0,1,0),virt-a),provides(p3,v(0,1,0),virt-b,v(2,0,0))])",
                 "catalog_to_term catalog/10");
        check(got.args.size() == 10 && got.s == "catalog/10", "catalog_to_term catalog/10 raw arity");
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
