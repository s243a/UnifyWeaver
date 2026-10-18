#pragma once
#include "json.hpp"
#include "wam_runtime.h"

namespace term_build {

using wam_cpp::Value;

Value ver_term(const json::Json& ver);
Value segs_term(const json::Json* segs);  // nullptr (absent) => []
Value pair_term(const std::string& name, const json::Json& ver);
Value constraint_term(const json::Json& c);
Value hold_term(const json::Json& row);
Value layer_term(const json::Json& row);
Value alias_term(const json::Json& row);
Value pkg_term(const json::Json& row);
Value dep_term(const json::Json& row);
Value provide_term(const json::Json& row);
Value conf_term(const json::Json& row);
Value request_term(const json::Json& req);
Value catalog_to_term(const json::Json& catalog);

}  // namespace term_build
