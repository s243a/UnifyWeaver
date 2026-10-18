// SPDX-License-Identifier: MIT OR Apache-2.0
// json_selftest.cpp -- fixed acceptance test for the M2 JSON layer.
// Exercises the fixtures from m2-json-layer-spec.md §3. Prints PASS/FAIL
// per check and a summary; exits 0 iff every check passed.
#include "json.hpp"

#include <cstdio>
#include <string>

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

void checkInt(int64_t got, int64_t want, const std::string& what) {
    check(got == want, what + " (got " + std::to_string(got) + ", want " + std::to_string(want) + ")");
}

}  // namespace

int main() {
    using json::Json;
    using json::parse;
    using json::dump;

    // ---- F1: full catalog case object ----
    const std::string F1 =
        "{\"id\":\"g368\",\"catalog\":{\"packages\":[[\"p0\",[0,2,0]],[\"p1\",[0,1,0]],[\"p2\",[0,1,0]],"
        "[\"p3\",[0,2,0]],[\"p4\",[0,2,0]],[\"p5\",[0,1,0]],[\"p6\",[0,1,0]],[\"p7\",[0,1,0]],[\"p8\",[0,1,0]],"
        "[\"p9\",[0,1,0]]],\"depends\":[[\"p1\",[0,1,0],\"p0\",{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}],"
        "[\"p2\",[0,1,0],\"p0\",\"any\"],[\"p3\",[0,2,0],\"p2\",{\"op\":\"gte\",\"v\":[0,1,0]}],"
        "[\"p4\",[0,2,0],\"p2\",\"any\"],[\"p7\",[0,1,0],\"p1\",{\"op\":\"gte\",\"v\":[0,1,0]}],"
        "[\"p8\",[0,1,0],\"p5\",{\"op\":\"eq\",\"v\":[0,1,0]}],[\"p9\",[0,1,0],\"p6\",\"any\"]],"
        "\"conflicts\":[[\"p0\",[0,2,0],\"p8\"]],\"base\":[],\"installed\":[[\"p2\",[0,1,0]]],\"requested\":[],"
        "\"layers\":[{\"name\":\"devx\",\"packages\":[[\"p7\",[0,1,0]]]}],\"excluded\":[],\"aliases\":[]},"
        "\"query\":\"dependents\",\"args\":\"p0\"}";
    {
        Json root = parse(F1);
        checkStr(dump(root), F1, "F1 round trip");
        check(root.isObject(), "F1 root isObject");
        checkStr(root.find("id")->asString(), "g368", "F1 id");
        checkStr(root.find("query")->asString(), "dependents", "F1 query");
        check(root.find("args")->isString(), "F1 args isString");
        checkStr(root.find("args")->asString(), "p0", "F1 args value");
        const Json* cat = root.find("catalog");
        check(cat != nullptr, "F1 catalog present");
        checkInt(static_cast<int64_t>(cat->find("packages")->size()), 10, "F1 packages.size");
        const Json& pkg0 = cat->find("packages")->asArray()[0];
        checkStr(pkg0.asArray()[0].asString(), "p0", "F1 packages[0][0]");
        const Json& pkg0ver = pkg0.asArray()[1];
        checkInt(pkg0ver.asArray()[0].asInt(), 0, "F1 packages[0][1][0]");
        checkInt(pkg0ver.asArray()[1].asInt(), 2, "F1 packages[0][1][1]");
        checkInt(pkg0ver.asArray()[2].asInt(), 0, "F1 packages[0][1][2]");
        const Json& dep0 = cat->find("depends")->asArray()[0];
        const Json& dep0c = dep0.asArray()[3];
        checkStr(dep0c.find("op")->asString(), "range", "F1 depends[0][3].op");
        checkInt(dep0c.find("lo")->asArray()[0].asInt(), 0, "F1 depends[0][3].lo[0]");
        checkInt(dep0c.find("lo")->asArray()[1].asInt(), 2, "F1 depends[0][3].lo[1]");
        checkInt(dep0c.find("hi")->asArray()[0].asInt(), 1, "F1 depends[0][3].hi[0]");
        checkInt(static_cast<int64_t>(cat->find("conflicts")->asArray()[0].size()), 3, "F1 conflicts[0].size");
        checkInt(static_cast<int64_t>(cat->find("base")->size()), 0, "F1 base.size");
        checkStr(cat->find("layers")->asArray()[0].find("name")->asString(), "devx", "F1 layers[0].name");
        checkInt(static_cast<int64_t>(cat->find("excluded")->size()), 0, "F1 excluded.size");
        check(cat->has("aliases"), "F1 catalog.has(aliases)");
        checkInt(static_cast<int64_t>(cat->find("aliases")->size()), 0, "F1 aliases.size");
    }

    // ---- F2: nested deb version object ----
    const std::string F2 = "{\"deb\":[0,[[\"\",1],[\".\",0]],[[\"\",2]]]}";
    {
        Json root = parse(F2);
        checkStr(dump(root), F2, "F2 round trip");
        const Json& deb = *root.find("deb");
        checkInt(deb.asArray()[0].asInt(), 0, "F2 deb[0] epoch");
        checkInt(static_cast<int64_t>(deb.asArray()[1].size()), 2, "F2 deb[1].size");
        checkStr(deb.asArray()[1].asArray()[0].asArray()[0].asString(), "", "F2 deb[1][0][0] empty string");
        checkInt(deb.asArray()[1].asArray()[0].asArray()[1].asInt(), 1, "F2 deb[1][0][1]");
        checkStr(deb.asArray()[1].asArray()[1].asArray()[0].asString(), ".", "F2 deb[1][1][0]");
        checkInt(deb.asArray()[1].asArray()[1].asArray()[1].asInt(), 0, "F2 deb[1][1][1]");
        checkInt(static_cast<int64_t>(deb.asArray()[2].size()), 1, "F2 deb[2].size");
        checkStr(deb.asArray()[2].asArray()[0].asArray()[0].asString(), "", "F2 deb[2][0][0] empty string");
        checkInt(deb.asArray()[2].asArray()[0].asArray()[1].asInt(), 2, "F2 deb[2][0][1]");
    }

    // ---- F3: range constraint object ----
    const std::string F3 = "{\"op\":\"range\",\"lo\":[0,2,0],\"hi\":[1,0,0]}";
    {
        Json root = parse(F3);
        checkStr(dump(root), F3, "F3 round trip");
        checkStr(root.find("op")->asString(), "range", "F3 op");
        checkInt(root.find("lo")->asArray()[0].asInt(), 0, "F3 lo[0]");
        checkInt(root.find("lo")->asArray()[1].asInt(), 2, "F3 lo[1]");
        checkInt(root.find("hi")->asArray()[0].asInt(), 1, "F3 hi[0]");
        check(root.find("nope") == nullptr, "F3 find missing key -> nullptr");
        check(!root.has("nope"), "F3 has missing key -> false");
    }

    // ---- F4: small packages array ----
    const std::string F4 = "[[\"p0\",[0,2,0]],[\"p1\",[0,1,0]]]";
    {
        Json root = parse(F4);
        checkStr(dump(root), F4, "F4 round trip");
        check(root.isArray(), "F4 root isArray");
        checkInt(static_cast<int64_t>(root.size()), 2, "F4 size");
        checkStr(root[0].asArray()[0].asString(), "p0", "F4 [0][0]");
        checkInt(root[0].asArray()[1].asArray()[1].asInt(), 2, "F4 [0][1][1]");
        checkStr(root[1].asArray()[0].asString(), "p1", "F4 [1][0]");
    }

    // ---- F5: string escaping + bool + negative int (synthetic) ----
    const std::string F5 =
        "{\"ok\":true,\"note\":\"line1\\nline2\\ttab \\\"quoted\\\" back\\\\slash\",\"n\":-7}";
    {
        Json root = parse(F5);
        checkStr(dump(root), F5, "F5 round trip");
        check(root.find("ok")->asBool() == true, "F5 ok");
        checkStr(root.find("note")->asString(),
                 "line1\nline2\ttab \"quoted\" back\\slash", "F5 note (decoded escapes)");
        checkInt(root.find("n")->asInt(), -7, "F5 n");
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
