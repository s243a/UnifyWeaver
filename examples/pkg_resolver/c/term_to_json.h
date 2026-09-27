/* SPDX-License-Identifier: MIT OR Apache-2.0 */
#pragma once

#include "json.h"
#include "wam_runtime.h"

Json tj_seg_to_json(WamState *state, WamValue v);
Json tj_ver_to_json(WamState *state, WamValue v);
Json tj_pair_to_json(WamState *state, WamValue v);
Json tj_sel_to_json(WamState *state, WamValue v);
Json tj_normalize_constraint(WamState *state, WamValue v);
Json tj_normalize_blocked(WamState *state, WamValue v);
Json tj_blocked_list_to_json(WamState *state, WamValue v);
Json tj_normalize_verdict(WamState *state, WamValue v);
Json tj_normalize_upgrade(WamState *state, WamValue v);
Json tj_normalize_audit_term(WamState *state, WamValue v);
Json tj_audit_list_to_json(WamState *state, WamValue v);
Json tj_term_to_json(WamState *state, WamValue v);
