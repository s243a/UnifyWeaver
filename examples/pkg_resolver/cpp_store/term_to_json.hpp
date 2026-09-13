#pragma once
#include "json.hpp"
#include "wam_runtime.h"

namespace term_json {

using wam_cpp::Value;

json::Json term_to_json(const Value& term);

json::Json ver_to_json(const Value& ver);
json::Json seg_to_json(const Value& seg);
json::Json pair_to_json(const Value& pair);
json::Json sel_to_json(const Value& list);

json::Json normalize_constraint(const Value& c);
json::Json normalize_blocked(const Value& blocked);
json::Json blocked_list_to_json(const Value& list);
json::Json normalize_verdict(const Value& verdict);
json::Json normalize_upgrade(const Value& result);
json::Json normalize_audit_term(const Value& audit);
json::Json audit_list_to_json(const Value& list);

}  // namespace term_json
