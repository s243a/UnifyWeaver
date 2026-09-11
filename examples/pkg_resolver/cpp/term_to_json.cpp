#include "term_to_json.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace term_json {

using wam_cpp::Value;

namespace {

std::string functor_name(const Value& v) {
    if (v.tag != Value::Tag::Compound) return "";
    auto slash = v.s.rfind('/');
    if (slash == std::string::npos) return v.s;
    return v.s.substr(0, slash);
}

std::vector<const Value*> list_elements(const Value& list) {
    std::vector<const Value*> res;
    const Value* cur = &list;
    while (cur->tag == Value::Tag::Compound && cur->s == "[|]/2" && cur->args.size() == 2) {
        res.push_back(cur->args[0].get());
        cur = cur->args[1].get();
    }
    return res;
}

} // namespace

json::Json seg_to_json(const Value& seg) {
    if (seg.tag != Value::Tag::Compound || seg.s != "s/2" || seg.args.size() < 2) {
        throw std::runtime_error("term_json: invalid seg term");
    }
    std::string order;
    for (const Value* code_val : list_elements(*seg.args[0])) {
        if (code_val->tag == Value::Tag::Integer) {
            order.push_back(static_cast<char>(code_val->i));
        }
    }
    int64_t num = (seg.args[1]->tag == Value::Tag::Integer) ? seg.args[1]->i : 0;
    json::Json arr = json::Json::makeArray();
    arr.push_back(json::Json::makeString(std::move(order)));
    arr.push_back(json::Json::makeInt(num));
    return arr;
}

json::Json ver_to_json(const Value& ver) {
    if (ver.tag == Value::Tag::Compound) {
        if (ver.s == "v/3" && ver.args.size() == 3) {
            json::Json arr = json::Json::makeArray();
            arr.push_back(json::Json::makeInt(ver.args[0]->i));
            arr.push_back(json::Json::makeInt(ver.args[1]->i));
            arr.push_back(json::Json::makeInt(ver.args[2]->i));
            return arr;
        }
        if (ver.s == "deb/3" && ver.args.size() == 3) {
            int64_t epoch = ver.args[0]->i;
            json::Json up = json::Json::makeArray();
            for (const Value* item : list_elements(*ver.args[1])) {
                up.push_back(seg_to_json(*item));
            }
            json::Json rev = json::Json::makeArray();
            for (const Value* item : list_elements(*ver.args[2])) {
                rev.push_back(seg_to_json(*item));
            }
            json::Json deb_arr = json::Json::makeArray();
            deb_arr.push_back(json::Json::makeInt(epoch));
            deb_arr.push_back(std::move(up));
            deb_arr.push_back(std::move(rev));

            json::Json obj = json::Json::makeObject();
            obj.set("deb", std::move(deb_arr));
            return obj;
        }
    }
    throw std::runtime_error("term_json: invalid ver term");
}

json::Json pair_to_json(const Value& pair) {
    if (pair.tag != Value::Tag::Compound || pair.s != "-/2" || pair.args.size() < 2) {
        throw std::runtime_error("term_json: invalid pair term");
    }
    json::Json arr = json::Json::makeArray();
    arr.push_back(json::Json::makeString(pair.args[0]->s));
    arr.push_back(ver_to_json(*pair.args[1]));
    return arr;
}

json::Json sel_to_json(const Value& list) {
    json::Json arr = json::Json::makeArray();
    for (const Value* item : list_elements(list)) {
        arr.push_back(pair_to_json(*item));
    }
    return arr;
}

json::Json normalize_constraint(const Value& c) {
    if (c.tag == Value::Tag::Atom && c.s == "any") {
        return json::Json::makeString("any");
    }
    if (c.tag == Value::Tag::Compound) {
        std::string name = functor_name(c);
        if ((name == "eq" || name == "gte" || name == "lt" || name == "lte" || name == "gt") && c.args.size() == 1) {
            json::Json obj = json::Json::makeObject();
            obj.set("op", json::Json::makeString(name));
            obj.set("v", ver_to_json(*c.args[0]));
            return obj;
        }
        if (name == "range" && c.args.size() == 2) {
            json::Json obj = json::Json::makeObject();
            obj.set("op", json::Json::makeString("range"));
            obj.set("lo", ver_to_json(*c.args[0]));
            obj.set("hi", ver_to_json(*c.args[1]));
            return obj;
        }
    }
    throw std::runtime_error("term_json: unknown constraint term");
}

json::Json normalize_blocked(const Value& blocked) {
    if (blocked.tag == Value::Tag::Compound) {
        std::string name = functor_name(blocked);
        if (name == "blocked") {
            // Shape 3: blocked(alternatives([...]))
            if (blocked.args.size() == 1) {
                const Value& inner = *blocked.args[0];
                if (inner.tag == Value::Tag::Compound && functor_name(inner) == "alternatives" && inner.args.size() == 1) {
                    json::Json alts_arr = json::Json::makeArray();
                    for (const Value* alt : list_elements(*inner.args[0])) {
                        if (alt->tag == Value::Tag::Compound && functor_name(*alt) == "alt" && alt->args.size() == 2) {
                            json::Json alt_obj = json::Json::makeObject();
                            alt_obj.set("dep", json::Json::makeString(alt->args[0]->s));
                            if (alt->args[1]->tag == Value::Tag::Atom && alt->args[1]->s == "unsatisfiable") {
                                alt_obj.set("reason", json::Json::makeString("unsatisfiable"));
                            } else {
                                alt_obj.set("reason", normalize_blocked(*alt->args[1]));
                            }
                            alts_arr.push_back(std::move(alt_obj));
                        }
                    }
                    json::Json obj = json::Json::makeObject();
                    obj.set("alternatives", std::move(alts_arr));
                    return obj;
                }
            }
            // Shape 1 or 2: blocked(N, needs(C), base_has(V)) or blocked(N, needs(C), providers(Ps))
            if (blocked.args.size() == 3) {
                std::string pkg_name = blocked.args[0]->s;
                const Value* needs_term = blocked.args[1].get();
                if (needs_term->tag == Value::Tag::Compound && functor_name(*needs_term) == "needs" && needs_term->args.size() == 1) {
                    needs_term = needs_term->args[0].get();
                }
                const Value* third = blocked.args[2].get();
                std::string third_name = functor_name(*third);

                if (third_name == "base_has" && third->args.size() == 1) {
                    json::Json obj = json::Json::makeObject();
                    obj.set("name", json::Json::makeString(pkg_name));
                    obj.set("needs", normalize_constraint(*needs_term));
                    obj.set("base_has", ver_to_json(*third->args[0]));
                    return obj;
                }
                if (third_name == "providers" && third->args.size() == 1) {
                    json::Json prov_arr = json::Json::makeArray();
                    for (const Value* p : list_elements(*third->args[0])) {
                        prov_arr.push_back(normalize_blocked(*p));
                    }
                    json::Json obj = json::Json::makeObject();
                    obj.set("name", json::Json::makeString(pkg_name));
                    obj.set("needs", normalize_constraint(*needs_term));
                    obj.set("providers", std::move(prov_arr));
                    return obj;
                }
            }
        }
    }
    throw std::runtime_error("term_json: unknown blocked term");
}

json::Json blocked_list_to_json(const Value& list) {
    json::Json arr = json::Json::makeArray();
    for (const Value* item : list_elements(list)) {
        arr.push_back(normalize_blocked(*item));
    }
    return arr;
}

json::Json normalize_verdict(const Value& verdict) {
    if (verdict.tag == Value::Tag::Atom && verdict.s == "no_candidate") {
        json::Json obj = json::Json::makeObject();
        obj.set("verdict", json::Json::makeString("no_candidate"));
        return obj;
    }
    if (verdict.tag == Value::Tag::Compound && verdict.args.size() == 1) {
        std::string name = functor_name(verdict);
        if (name == "safe") {
            const Value* cost_arg = verdict.args[0].get();
            std::string cost_str;
            if (cost_arg->tag == Value::Tag::Compound && functor_name(*cost_arg) == "cost" && cost_arg->args.size() == 1) {
                cost_str = cost_arg->args[0]->s;
            } else {
                cost_str = cost_arg->s;
            }
            json::Json obj = json::Json::makeObject();
            obj.set("cost", json::Json::makeString(cost_str));
            obj.set("verdict", json::Json::makeString("safe"));
            return obj;
        }
        if (name == "coordinated") {
            json::Json obj = json::Json::makeObject();
            obj.set("set", sel_to_json(*verdict.args[0]));
            obj.set("verdict", json::Json::makeString("coordinated"));
            return obj;
        }
        if (name == "unsafe") {
            json::Json obj = json::Json::makeObject();
            obj.set("reason", json::Json::makeString(verdict.args[0]->s));
            obj.set("verdict", json::Json::makeString("unsafe"));
            return obj;
        }
    }
    throw std::runtime_error("term_json: unknown verdict term");
}

json::Json normalize_upgrade(const Value& result) {
    if (result.tag == Value::Tag::Atom && result.s == "no_candidate") {
        json::Json obj = json::Json::makeObject();
        obj.set("fail", json::Json::makeBool(true));
        return obj;
    }
    if (result.tag == Value::Tag::Compound) {
        std::string name = functor_name(result);
        if (name == "ok" && result.args.size() == 1) {
            json::Json obj = json::Json::makeObject();
            obj.set("ok", sel_to_json(*result.args[0]));
            return obj;
        }
        if (name == "blocked") {
            json::Json inner = normalize_blocked(result);
            json::Json blocked_envelope = json::Json::makeObject();
            blocked_envelope.set("blocked", std::move(inner));
            json::Json obj = json::Json::makeObject();
            obj.set("ok", std::move(blocked_envelope));
            return obj;
        }
    }
    throw std::runtime_error("term_json: unknown upgrade term");
}

json::Json normalize_audit_term(const Value& audit) {
    if (audit.tag == Value::Tag::Compound && functor_name(audit) == "audit" && audit.args.size() == 2) {
        std::string name = audit.args[0]->s;
        const Value* second = audit.args[1].get();
        if (second->tag == Value::Tag::Atom && second->s == "over_frozen") {
            json::Json obj = json::Json::makeObject();
            obj.set("kind", json::Json::makeString("over_frozen"));
            obj.set("name", json::Json::makeString(name));
            return obj;
        }
        if (second->tag == Value::Tag::Compound && second->args.size() == 1) {
            std::string kind = functor_name(*second);
            if (kind == "suggest" || kind == "held") {
                json::Json obj = json::Json::makeObject();
                obj.set("kind", json::Json::makeString(kind));
                obj.set("name", json::Json::makeString(name));
                obj.set("reason", json::Json::makeString(second->args[0]->s));
                return obj;
            }
        }
    }
    throw std::runtime_error("term_json: unknown audit term");
}

json::Json audit_list_to_json(const Value& list) {
    json::Json arr = json::Json::makeArray();
    for (const Value* item : list_elements(list)) {
        arr.push_back(normalize_audit_term(*item));
    }
    return arr;
}

json::Json term_to_json(const Value& term) {
    switch (term.tag) {
        case Value::Tag::Integer:
            return json::Json::makeInt(term.i);
        case Value::Tag::Atom:
            if (term.s == "[]") {
                return json::Json::makeArray();
            }
            if (term.s == "true") {
                return json::Json::makeBool(true);
            }
            if (term.s == "false") {
                return json::Json::makeBool(false);
            }
            return json::Json::makeString(term.s);
        case Value::Tag::Uninit:
        case Value::Tag::Unbound:
            return json::Json::makeNull();
        case Value::Tag::Compound: {
            if (term.s == "[|]/2" && term.args.size() == 2) {
                json::Json arr = json::Json::makeArray();
                for (const Value* item : list_elements(term)) {
                    arr.push_back(term_to_json(*item));
                }
                return arr;
            }
            if (term.s == "v/3" || term.s == "deb/3") {
                return ver_to_json(term);
            }
            if (term.s == "s/2") {
                return seg_to_json(term);
            }
            if (term.s == "-/2") {
                return pair_to_json(term);
            }
            std::string name = functor_name(term);
            json::Json arr = json::Json::makeArray();
            arr.push_back(json::Json::makeString(name));
            for (const auto& a : term.args) {
                arr.push_back(term_to_json(*a));
            }
            return arr;
        }
        default:
            return json::Json::makeNull();
    }
}

}  // namespace term_json
