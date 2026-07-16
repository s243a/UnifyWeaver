% test_wam_c_fact_table_emit.pl
%
% T9 fact-table inline for the C target — codegen-level checks (no gcc needed).
% C already had a deterministic static-row-table + first-arg bucket scan; T9
% makes the in-window fact lowering backtrackable (it drives wam_fact_table_scan,
% which leaves a WAM_FACT_TABLE_RETRY choice point) and adds the t9 window /
% opt-out / oversized semantics. Verifies the planner gating and the emitted C.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_c_target').

:- dynamic user:ek/2.
user:ek(a, 1). user:ek(a, 2). user:ek(b, 3).
user:ek(a, 4). user:ek(c, 5). user:ek(b, 6).

plan(Opts, Plans) :-
    plan_wam_c_lowered_helpers([user:ek/2], [lowered_helpers(true)|Opts], [], Plans).

:- begin_tests(wam_c_fact_table_emit).

% default in-range (6 facts, window [4,256]) -> backtrackable fact_table plan.
test(default_in_range_fact_table) :-
    plan([t9_min_rows(4)], Plans),
    assertion(member(wam_c_lowered_helper_plan('ek/2', _, lowered, fact_table(_)), Plans)).

% explicit opt-out -> deterministic fact_only (no choice point).
test(opt_out_fact_only) :-
    plan([t9_min_rows(4), fact_table_inline(false)], Plans),
    assertion(member(wam_c_lowered_helper_plan('ek/2', _, lowered, fact_only(_)), Plans)),
    assertion(\+ member(wam_c_lowered_helper_plan('ek/2', _, lowered, fact_table(_)), Plans)).

% below t9_min_rows -> keep the cheap deterministic scanner.
test(below_min_fact_only) :-
    plan([t9_min_rows(100)], Plans),
    assertion(member(wam_c_lowered_helper_plan('ek/2', _, lowered, fact_only(_)), Plans)).

% above t9_max_rows -> not inlined as a fact table (deterministic fallback).
test(above_cap_fact_only) :-
    plan([t9_min_rows(2), t9_max_rows(5)], Plans),
    assertion(member(wam_c_lowered_helper_plan('ek/2', _, lowered, fact_only(_)), Plans)),
    assertion(\+ member(wam_c_lowered_helper_plan('ek/2', _, lowered, fact_table(_)), Plans)).

% classifier yields fact_info with the rows when in-window.
test(classify_fact_info) :-
    wam_c_fact_table_classify(user:ek/2, [t9_min_rows(4)], fact_info(2, Rows)),
    assertion(length(Rows, 6)).

% emitted C drives the backtrackable scan, with a static row table + bucket index.
test(emitted_c_uses_scan) :-
    wam_c_fact_table_helper_for_predicate(user:ek/2,
        [[a,1],[a,2],[b,3],[a,4],[c,5],[b,6]], Key, Code, Setup),
    assertion(Key == 'ek/2'),
    assertion(sub_string(Code, _, _, _, "wam_fact_table_scan")),
    assertion(sub_string(Code, _, _, _, "_rows[][2]")),
    assertion(sub_string(Code, _, _, _, "switch (bucket)")),
    assertion(sub_string(Setup, _, _, _, "wam_register_foreign_predicate")).

% value-literal mapping (atom/int) in the emitted row table.
test(value_literals) :-
    wam_c_fact_table_helper_for_predicate(user:ek/2,
        [[foo, 7], [bar, -3]], _Key, Code, _Setup),
    assertion(sub_string(Code, _, _, _, ".tag = VAL_ATOM, .data.atom = \"foo\"")),
    assertion(sub_string(Code, _, _, _, ".tag = VAL_INT, .data.integer = 7")),
    assertion(sub_string(Code, _, _, _, ".tag = VAL_INT, .data.integer = -3")).

:- end_tests(wam_c_fact_table_emit).
