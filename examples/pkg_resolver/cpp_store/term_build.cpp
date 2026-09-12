#include "term_build.hpp"
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <utility>

namespace term_build {

using wam_cpp::Value;
using wam_cpp::CellPtr;

namespace {

Value atom(const std::string& name) {
    return Value::Atom(name);
}

Value integer(int64_t n) {
    return Value::Integer(n);
}

Value st(const std::string& name, std::vector<Value> args) {
    std::string functor = name + "/" + std::to_string(args.size());
    std::vector<CellPtr> cargs;
    cargs.reserve(args.size());
    for (auto& a : args) {
        cargs.push_back(std::make_shared<Value>(std::move(a)));
    }
    return Value::Compound(std::move(functor), std::move(cargs));
}

Value list_of(std::vector<Value> items) {
    Value cur = Value::Atom("[]");
    for (auto it = items.rbegin(); it != items.rend(); ++it) {
        std::vector<CellPtr> cargs;
        cargs.reserve(2);
        cargs.push_back(std::make_shared<Value>(std::move(*it)));
        cargs.push_back(std::make_shared<Value>(std::move(cur)));
        cur = Value::Compound("[|]/2", std::move(cargs));
    }
    return cur;
}

} // namespace

Value segs_term(const json::Json* segs) {
    if (!segs || !segs->isArray()) {
        return Value::Atom("[]");
    }
    std::vector<Value> items;
    for (const auto& seg : segs->asArray()) {
        std::string order = "";
        int64_t num = 0;
        if (seg.isArray()) {
            const auto& s = seg.asArray();
            if (s.size() > 0 && s[0].isString()) {
                order = s[0].asString();
            }
            if (s.size() > 1 && s[1].isInt()) {
                num = s[1].asInt();
            }
        }
        std::vector<Value> codes;
        for (unsigned char c : order) {
            codes.push_back(integer(static_cast<int64_t>(c)));
        }
        items.push_back(st("s", { list_of(std::move(codes)), integer(num) }));
    }
    return list_of(std::move(items));
}

Value ver_term(const json::Json& ver) {
    if (ver.isObject()) {
        const json::Json* d = ver.find("deb");
        if (d && d->isArray()) {
            const auto& arr = d->asArray();
            int64_t epoch = 0;
            const json::Json* up = nullptr;
            const json::Json* rev = nullptr;
            if (arr.size() > 0 && arr[0].isInt()) {
                epoch = arr[0].asInt();
            }
            if (arr.size() > 1) {
                up = &arr[1];
            }
            if (arr.size() > 2) {
                rev = &arr[2];
            }
            return st("deb", { integer(epoch), segs_term(up), segs_term(rev) });
        }
    }
    if (ver.isArray()) {
        const auto& a = ver.asArray();
        int64_t m = a.size() > 0 && a[0].isInt() ? a[0].asInt() : 0;
        int64_t i = a.size() > 1 && a[1].isInt() ? a[1].asInt() : 0;
        int64_t p = a.size() > 2 && a[2].isInt() ? a[2].asInt() : 0;
        return st("v", { integer(m), integer(i), integer(p) });
    }
    throw std::runtime_error("term_build: invalid version");
}

Value pair_term(const std::string& name, const json::Json& ver) {
    return st("-", { atom(name), ver_term(ver) });
}

Value constraint_term(const json::Json& c) {
    if (c.isNull()) {
        return atom("any");
    }
    if (c.isString()) {
        if (c.asString() == "any") {
            return atom("any");
        }
        throw std::runtime_error("term_build: unknown string constraint: " + c.asString());
    }
    if (c.isObject()) {
        const json::Json* op_ptr = c.find("op");
        if (op_ptr && op_ptr->isString()) {
            const std::string& op = op_ptr->asString();
            if (op == "eq" || op == "gte" || op == "lt" || op == "lte" || op == "gt") {
                const json::Json* v_ptr = c.find("v");
                if (!v_ptr) {
                    throw std::runtime_error("term_build: missing 'v' in constraint");
                }
                return st(op, { ver_term(*v_ptr) });
            } else if (op == "range") {
                const json::Json* lo_ptr = c.find("lo");
                const json::Json* hi_ptr = c.find("hi");
                if (!lo_ptr || !hi_ptr) {
                    throw std::runtime_error("term_build: missing 'lo'/'hi' in range constraint");
                }
                return st("range", { ver_term(*lo_ptr), ver_term(*hi_ptr) });
            }
        }
    }
    throw std::runtime_error("term_build: unknown constraint");
}

Value hold_term(const json::Json& row) {
    if (!row.isArray() || row.size() < 2) {
        throw std::runtime_error("term_build: hold_term expects array of at least 2 elements");
    }
    const auto& arr = row.asArray();
    std::string name = arr[0].asString();
    Value pair = pair_term(name, arr[1]);
    if (arr.size() >= 3) {
        std::string reason = arr[2].asString();
        return st("base", { std::move(pair), atom(reason) });
    }
    return pair;
}

Value layer_term(const json::Json& row) {
    std::string name = "";
    const json::Json* name_json = row.find("name");
    if (name_json && name_json->isString()) {
        name = name_json->asString();
    }
    std::vector<Value> pkgs;
    const json::Json* pkgs_json = row.find("packages");
    if (pkgs_json && pkgs_json->isArray()) {
        for (const auto& p : pkgs_json->asArray()) {
            pkgs.push_back(hold_term(p));
        }
    }
    return st("layer", { atom(name), list_of(std::move(pkgs)) });
}

Value alias_term(const json::Json& row) {
    if (!row.isArray() || row.size() < 2) {
        throw std::runtime_error("term_build: alias_term expects 2-element array");
    }
    return st("alias", { atom(row[0].asString()), atom(row[1].asString()) });
}

Value pkg_term(const json::Json& row) {
    if (!row.isArray() || row.size() < 2) {
        throw std::runtime_error("term_build: pkg_term expects 2-element array");
    }
    return st("package", { atom(row[0].asString()), ver_term(row[1]) });
}

Value dep_term(const json::Json& row) {
    if (!row.isArray() || row.size() < 4) {
        throw std::runtime_error("term_build: dep_term expects array of at least 4 elements");
    }
    std::string name = row[0].asString();
    Value ver = ver_term(row[1]);
    const json::Json& third = row[2];
    Value dep_arg;
    bool handled_alts = false;
    if (third.isObject()) {
        const json::Json* alts_json = third.find("alternatives");
        if (alts_json && alts_json->isArray()) {
            std::vector<Value> alt_terms;
            for (const auto& a : alts_json->asArray()) {
                std::string dep_name = "";
                const json::Json* d = a.find("dep");
                if (d && d->isString()) {
                    dep_name = d->asString();
                }
                const json::Json* c = a.find("constraint");
                alt_terms.push_back(st("dep", { atom(dep_name), constraint_term(c ? *c : json::Json::makeNull()) }));
            }
            dep_arg = st("alternatives", { list_of(std::move(alt_terms)) });
            handled_alts = true;
        }
    }
    if (!handled_alts) {
        dep_arg = atom(third.asString());
    }
    Value constraint = constraint_term(row[3]);
    return st("depends", { atom(name), std::move(ver), std::move(dep_arg), std::move(constraint) });
}

Value provide_term(const json::Json& row) {
    if (!row.isArray() || row.size() < 3) {
        throw std::runtime_error("term_build: provide_term expects at least 3 elements");
    }
    std::string name = row[0].asString();
    Value ver = ver_term(row[1]);
    std::string virt = row[2].asString();
    if (row.size() >= 4 && !row[3].isNull()) {
        Value virt_ver = ver_term(row[3]);
        return st("provides", { atom(name), std::move(ver), atom(virt), std::move(virt_ver) });
    }
    return st("provides", { atom(name), std::move(ver), atom(virt) });
}

Value conf_term(const json::Json& row) {
    if (!row.isArray() || row.size() < 3) {
        throw std::runtime_error("term_build: conf_term expects 3-element array");
    }
    return st("conflicts", { atom(row[0].asString()), ver_term(row[1]), atom(row[2].asString()) });
}

Value request_term(const json::Json& req) {
    if (req.isObject()) {
        const json::Json* r = req.find("req");
        if (r && r->isString()) {
            const json::Json* c = req.find("constraint");
            return st("req", { atom(r->asString()), constraint_term(c ? *c : json::Json::makeNull()) });
        }
    }
    if (req.isString()) {
        return atom(req.asString());
    }
    throw std::runtime_error("term_build: invalid request_term");
}

Value catalog_to_term(const json::Json& catalog) {
    std::vector<Value> pkgs;
    if (const json::Json* p = catalog.find("packages")) {
        if (p->isArray()) {
            for (const auto& item : p->asArray()) {
                pkgs.push_back(pkg_term(item));
            }
        }
    }

    std::vector<Value> deps;
    if (const json::Json* d = catalog.find("depends")) {
        if (d->isArray()) {
            for (const auto& item : d->asArray()) {
                deps.push_back(dep_term(item));
            }
        }
    }

    std::vector<Value> confs;
    if (const json::Json* c = catalog.find("conflicts")) {
        if (c->isArray()) {
            for (const auto& item : c->asArray()) {
                confs.push_back(conf_term(item));
            }
        }
    }

    std::vector<Value> base;
    if (const json::Json* b = catalog.find("base")) {
        if (b->isArray()) {
            for (const auto& item : b->asArray()) {
                base.push_back(hold_term(item));
            }
        }
    }

    std::vector<Value> inst;
    if (const json::Json* in = catalog.find("installed")) {
        if (in->isArray()) {
            for (const auto& item : in->asArray()) {
                if (item.isArray() && item.size() >= 2) {
                    inst.push_back(pair_term(item[0].asString(), item[1]));
                }
            }
        }
    }

    std::vector<Value> req;
    if (const json::Json* r = catalog.find("requested")) {
        if (r->isArray()) {
            for (const auto& item : r->asArray()) {
                req.push_back(atom(item.asString()));
            }
        }
    }

    std::vector<Value> core;
    core.push_back(list_of(std::move(pkgs)));
    core.push_back(list_of(std::move(deps)));
    core.push_back(list_of(std::move(confs)));
    core.push_back(list_of(std::move(base)));
    core.push_back(list_of(std::move(inst)));
    core.push_back(list_of(std::move(req)));

    const json::Json* layers_json = catalog.find("layers");
    const json::Json* excl_json = catalog.find("excluded");
    const json::Json* alias_json = catalog.find("aliases");
    const json::Json* prov_json = catalog.find("provides");

    size_t layers_sz = (layers_json && layers_json->isArray()) ? layers_json->size() : 0;
    size_t excl_sz = (excl_json && excl_json->isArray()) ? excl_json->size() : 0;
    size_t alias_sz = (alias_json && alias_json->isArray()) ? alias_json->size() : 0;
    size_t prov_sz = (prov_json && prov_json->isArray()) ? prov_json->size() : 0;

    if (layers_sz == 0 && excl_sz == 0 && alias_sz == 0 && prov_sz == 0) {
        return st("catalog", std::move(core));
    }

    std::vector<Value> layers;
    if (layers_json && layers_json->isArray()) {
        for (const auto& item : layers_json->asArray()) {
            layers.push_back(layer_term(item));
        }
    }

    std::vector<Value> excl;
    if (excl_json && excl_json->isArray()) {
        for (const auto& item : excl_json->asArray()) {
            excl.push_back(atom(item.asString()));
        }
    }

    std::vector<Value> aliases;
    if (alias_json && alias_json->isArray()) {
        for (const auto& item : alias_json->asArray()) {
            aliases.push_back(alias_term(item));
        }
    }

    std::vector<Value> nine = std::move(core);
    nine.push_back(list_of(std::move(layers)));
    nine.push_back(list_of(std::move(excl)));
    nine.push_back(list_of(std::move(aliases)));

    if (prov_sz == 0) {
        return st("catalog", std::move(nine));
    }

    std::vector<Value> provs;
    if (prov_json && prov_json->isArray()) {
        for (const auto& item : prov_json->asArray()) {
            provs.push_back(provide_term(item));
        }
    }

    std::vector<Value> ten = std::move(nine);
    ten.push_back(list_of(std::move(provs)));
    return st("catalog", std::move(ten));
}

}  // namespace term_build
