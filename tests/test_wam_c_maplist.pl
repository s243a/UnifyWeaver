:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Focused C maplist/2: generated/compiled C vs SWI for finite proper lists
% and conservative unary bodyless facts.
% Open/improper/cyclic/non-list and ineligible goals (body-bearing, multi-clause,
% compound closures, unknown predicates) are diagnosed with WAM_ERR_UNSUPPORTED,
% not treated as silent logical failure.
%
%   swipl -g run_tests -t halt tests/test_wam_c_maplist.pl

:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module(library(process)).

:- dynamic test_failed/0.
:- dynamic tests_already_ran/0.

% Base fact predicates used as goals
:- dynamic user:is_v3/1.
:- dynamic user:tag_val/1.
:- dynamic user:same_pair/1.
:- dynamic user:has_body/1.
:- dynamic user:multi_clause/1.

% Test harness predicates
:- dynamic user:wam_maplist_q/2.
:- dynamic user:wam_maplist_empty_unbound/1.
:- dynamic user:wam_maplist_empty_unknown/1.
:- dynamic user:wam_maplist_is_v3/1.
:- dynamic user:wam_maplist_tag/1.
:- dynamic user:wam_maplist_instantiate/1.
:- dynamic user:wam_maplist_same_pair/1.
:- dynamic user:wam_maplist_same_pair_bind/2.
:- dynamic user:wam_maplist_backtrack/1.
:- dynamic user:wam_maplist_continuation/3.
:- dynamic user:wam_maplist_chained/3.
:- dynamic user:wam_maplist_unknown/1.
:- dynamic user:wam_maplist_body/1.
:- dynamic user:wam_maplist_multi/1.
:- dynamic user:wam_maplist_compound/1.
:- dynamic user:wam_maplist_shared_conflict/1.
:- dynamic user:wam_control_overwrite_a0/2.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

wam_c_temp_root('/tmp').

wam_c_temp_path(Prefix, Stamp, Path) :-
    wam_c_temp_root(Root),
    format(atom(Path), '~w/~w_~w', [Root, Prefix, Stamp]).

gcc_available :-
    catch(process_create(path(gcc), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          _, fail),
    process_wait(Pid, exit(0)).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).

cleanup_maplist_preds :-
    retractall(user:is_v3(_)),
    retractall(user:tag_val(_)),
    retractall(user:same_pair(_)),
    retractall(user:has_body(_)),
    retractall(user:multi_clause(_)),
    retractall(user:wam_maplist_q(_, _)),
    retractall(user:wam_maplist_empty_unbound(_)),
    retractall(user:wam_maplist_empty_unknown(_)),
    retractall(user:wam_maplist_is_v3(_)),
    retractall(user:wam_maplist_tag(_)),
    retractall(user:wam_maplist_instantiate(_)),
    retractall(user:wam_maplist_same_pair(_)),
    retractall(user:wam_maplist_same_pair_bind(_, _)),
    retractall(user:wam_maplist_backtrack(_)),
    retractall(user:wam_maplist_continuation(_, _, _)),
    retractall(user:wam_maplist_chained(_, _, _)),
    retractall(user:wam_maplist_unknown(_)),
    retractall(user:wam_maplist_body(_)),
    retractall(user:wam_maplist_multi(_)),
    retractall(user:wam_maplist_compound(_)),
    retractall(user:wam_maplist_shared_conflict(_)),
    retractall(user:wam_control_overwrite_a0(_, _)).

setup_maplist_preds :-
    cleanup_maplist_preds,
    % Supported unary bodyless facts
    assertz((user:is_v3(v(_, _, _)))),
    assertz((user:tag_val(t(42)))),
    assertz((user:same_pair(pair(X, X)))),
    % Ineligible goals for conservative subset
    assertz((user:has_body(X) :- integer(X))),
    assertz((user:multi_clause(a))),
    assertz((user:multi_clause(b))),
    % Caller test predicates
    assertz((user:wam_maplist_q(G, L) :- maplist(G, L))),
    assertz((user:wam_maplist_empty_unbound(R) :- maplist(_, []), R = ok)),
    assertz((user:wam_maplist_empty_unknown(R) :- maplist(no_such_predicate_xyz, []), R = ok)),
    assertz((user:wam_maplist_is_v3(L) :- maplist(is_v3, L))),
    assertz((user:wam_maplist_tag(L) :- maplist(tag_val, L))),
    assertz((user:wam_maplist_instantiate(Out) :- L = [X, X], maplist(tag_val, L), Out = L)),
    assertz((user:wam_maplist_same_pair(L) :- maplist(same_pair, L))),
    assertz((user:wam_maplist_same_pair_bind(L, Out) :- maplist(same_pair, L), Out = L)),
    assertz((user:wam_maplist_backtrack(Out) :- ( maplist(tag_val, [X, bad(1)]) ; X = restored ), Out = X)),
    assertz((user:wam_maplist_shared_conflict(L) :- L = [pair(X, a), pair(X, b)], maplist(same_pair, L))),
    assertz((user:wam_maplist_continuation(L, S1, S2) :- S1 = start, maplist(is_v3, L), S2 = done)),
    assertz((user:wam_maplist_chained(L1, L2, Res) :- maplist(is_v3, L1), maplist(same_pair, L2), Res = both_ok)),
    assertz((user:wam_maplist_unknown(L) :- maplist(no_such_pred_xyz, L))),
    assertz((user:wam_maplist_body(L) :- maplist(has_body, L))),
    assertz((user:wam_maplist_multi(L) :- maplist(multi_clause, L))),
    assertz((user:wam_maplist_compound(L) :- maplist(tag(1), L))),
    % Control: builtin call overwrites A0, but retained query argument handle remains observable
    assertz((user:wam_control_overwrite_a0(In, Out) :- Out = In, atom(hello))).

% SWI Oracle for supported ground cases
ground_swi(control_builtin_overwrite, ok, retained_ok) :-
    user:wam_control_overwrite_a0(retained_ok, Out),
    Out == retained_ok.
ground_swi(empty_unbound, ok, ok) :-
    maplist(_, []).
ground_swi(empty_unknown, ok, ok) :-
    maplist(no_such_predicate_xyz, []).
ground_swi(is_v3_valid, ok, [v(1, 0, 0), v(2, 0, 0)]) :-
    maplist(is_v3, [v(1, 0, 0), v(2, 0, 0)]).
ground_swi(is_v3_single, ok, [v(a, b, c)]) :-
    maplist(is_v3, [v(a, b, c)]).
ground_swi(is_v3_empty, ok, []) :-
    maplist(is_v3, []).
ground_swi(is_v3_fail_first, fail, _) :-
    \+ maplist(is_v3, [deb(0, [], []), v(1, 0, 0)]).
ground_swi(is_v3_fail_second, fail, _) :-
    \+ maplist(is_v3, [v(1, 0, 0), deb(0, [], [])]).
ground_swi(instantiate_shared, ok, [t(42), t(42)]) :-
    L = [X, X], maplist(tag_val, L), L == [t(42), t(42)].
ground_swi(same_pair_valid, ok, [pair(a, a), pair(1, 1)]) :-
    maplist(same_pair, [pair(a, a), pair(1, 1)]).
ground_swi(same_pair_mismatch, fail, _) :-
    \+ maplist(same_pair, [pair(a, b)]).
ground_swi(same_pair_bind, ok, [pair(10, 10)]) :-
    L = [pair(10, _Y)], maplist(same_pair, L), L == [pair(10, 10)].
ground_swi(backtrack_restore, ok, restored) :-
    ( maplist(tag_val, [X, bad(1)]) ; X = restored ), X == restored.
ground_swi(shared_conflict, fail, _) :-
    \+ (L = [pair(X, a), pair(X, b)], maplist(same_pair, L)).
ground_swi(continuation, ok, done) :-
    _S1 = start, maplist(is_v3, [v(1, 2, 3)]), _S2 = done.
ground_swi(chained, ok, both_ok) :-
    maplist(is_v3, [v(1, 2, 3)]), maplist(same_pair, [pair(x, x)]).

% Oracle tokens for unsupported cases (must produce explicit WAM_ERR_UNSUPPORTED diagnostics)
token_swi(unknown_goal, runtime_error, unsupported_ok).
token_swi(body_goal, runtime_error, unsupported_ok).
token_swi(multi_clause_goal, runtime_error, unsupported_ok).
token_swi(compound_goal, runtime_error, unsupported_ok).
token_swi(var_goal, runtime_error, unsupported_ok).
token_swi(open_list, runtime_error, open_ok).
token_swi(improper_list, runtime_error, improper_ok).
token_swi(cyclic_list, runtime_error, cyclic_ok).
token_swi(non_list, runtime_error, non_list_ok).

parse_c_cases([], []).
parse_c_cases([Line|Rest], Cases) :-
    (   sub_string(Line, 0, _, _, "CASE ")
    ->  sub_string(Line, 5, _, 0, Id0),
        atom_string(Id, Id0),
        parse_c_status(Rest, Id, Case, Rest2),
        Cases = [Case|Cases2],
        parse_c_cases(Rest2, Cases2)
    ;   parse_c_cases(Rest, Cases)
    ).

parse_c_status([], Id, c_case(Id, missing_status, _), []).
parse_c_status([Line|Rest], Id, Case, RestOut) :-
    (   sub_string(Line, 0, _, _, "STATUS ")
    ->  sub_string(Line, 7, _, 0, St0),
        atom_string(Status, St0),
        parse_c_term(Rest, Id, Status, Case, RestOut)
    ;   sub_string(Line, 0, _, _, "CASE ")
    ->  Case = c_case(Id, missing_status, _),
        RestOut = [Line|Rest]
    ;   parse_c_status(Rest, Id, Case, RestOut)
    ).

parse_c_term([], Id, Status, c_case(Id, Status, _), []).
parse_c_term([Line|Rest], Id, Status, Case, RestOut) :-
    (   sub_string(Line, 0, _, _, "TERM ")
    ->  sub_string(Line, 5, _, 0, TermStr),
        (   catch(term_string(Term, TermStr), _, fail)
        ->  true
        ;   Term = parse_error(TermStr)
        ),
        Case = c_case(Id, Status, Term),
        RestOut = Rest
    ;   Status == fail
    ->  Case = c_case(Id, fail, _),
        RestOut = [Line|Rest]
    ;   sub_string(Line, 0, _, _, "CASE ")
    ->  Case = c_case(Id, Status, _),
        RestOut = [Line|Rest]
    ;   parse_c_term(Rest, Id, Status, Case, RestOut)
    ).

same_success(Expected, Actual) :-
    ground(Expected),
    ground(Actual),
    Actual == Expected.

compare_ground(Id, CCases) :-
    ground_swi(Id, SWIStatus, Expected),
    (   member(c_case(Id, CStatus, CTerm), CCases)
    ->  true
    ;   format(atom(R), 'missing C output for ~w', [Id]),
        fail_test(Id, R),
        fail
    ),
    (   SWIStatus == fail
    ->  (   CStatus == fail
        ->  pass(Id)
        ;   format(atom(R), 'expected fail vs C ~w ~q', [CStatus, CTerm]),
            fail_test(Id, R),
            fail
        )
    ;   CStatus == ok,
        same_success(Expected, CTerm)
    ->  pass(Id)
    ;   format(atom(R), 'SWI ~w ~q vs C ~w ~q', [SWIStatus, Expected, CStatus, CTerm]),
        fail_test(Id, R),
        fail
    ).

compare_token(Id, CCases) :-
    (   token_swi(Id, ExpectedStatus, ExpectedToken)
    ->  true
    ;   fail_test(Id, 'missing SWI token oracle'),
        fail
    ),
    (   member(c_case(Id, CStatus, CTerm), CCases)
    ->  true
    ;   format(atom(R), 'missing C output for ~w', [Id]),
        fail_test(Id, R),
        fail
    ),
    (   CStatus == ExpectedStatus,
        CTerm == ExpectedToken
    ->  pass(Id)
    ;   format(atom(R), 'expected ~w ~q vs C ~w ~q',
               [ExpectedStatus, ExpectedToken, CStatus, CTerm]),
        fail_test(Id, R),
        fail
    ).

compile_one(PI, Code) :-
    compile_predicate_to_wam(PI, [], Wam),
    compile_wam_predicate_to_c(PI, Wam, [], Code).

test_generation_maplist_builtin :-
    Test = 'maplist/2: generated runtime contains maplist handling',
    (   compile_wam_helpers_to_c([], HelpersCode),
        atom_string(HelpersCode, HelpersS),
        compile_step_wam_to_c([], StepCode),
        atom_string(StepCode, StepS),
        (   sub_string(HelpersS, _, _, _, 'maplist/2')
        ;   sub_string(StepS, _, _, _, 'maplist/2')
        )
    ->  pass(Test)
    ;   fail_test(Test, 'maplist/2 handler missing from generated runtime')
    ).

test_wam_emits_maplist_builtin :-
    Test = 'maplist/2: WAM text emits builtin_call maplist/2',
    setup_maplist_preds,
    (   compile_predicate_to_wam(user:wam_maplist_is_v3/1, [], Wam),
        sub_string(Wam, _, _, _, 'builtin_call maplist/2')
    ->  pass(Test)
    ;   fail_test(Test, 'wam_maplist_is_v3/1 WAM missing builtin_call maplist/2')
    ).

test_wrong_answer_rejected :-
    Test = 'maplist/2: SWI comparator rejects wrong answer and wrong status',
    (   same_success(ok, ok),
        \+ same_success(ok, fail),
        \+ same_success([a, b], [a]),
        \+ same_success([a, b], [b, a])
    ->  pass(Test)
    ;   fail_test(Test, 'comparator control failed')
    ).

test_metadata_controls :-
    Test = 'maplist/2: metadata eligibility controls',
    setup_maplist_preds,
    (   % 1. valid source single fact canonical WAM eligible
        compile_predicate_to_wam(user:is_v3/1, [], WamV3),
        wam_c_fact_eligible(user:is_v3/1, WamV3, 1),
        compile_predicate_to_wam(user:tag_val/1, [], WamTag),
        wam_c_fact_eligible(user:tag_val/1, WamTag, 1),
        % 2. modified/caller-supplied WAM rejected
        atom_concat(WamV3, '\n  noop', TamperedWam),
        \+ wam_c_fact_eligible(user:is_v3/1, TamperedWam, 1),
        % 3. foreign/body/multiclause/control-transfer stream rejected
        \+ wam_c_fact_eligible(user:read/1, 'read/1:\n  proceed\n', 1),
        compile_predicate_to_wam(user:has_body/1, [], WamBody),
        \+ wam_c_fact_eligible(user:has_body/1, WamBody, 1),
        compile_predicate_to_wam(user:multi_clause/1, [], WamMulti),
        \+ wam_c_fact_eligible(user:multi_clause/1, WamMulti, 1),
        \+ wam_c_fact_eligible(user:is_v3/1, 'is_v3/1:\n  call foo/1\n  proceed\n', 1),
        \+ wam_c_fact_eligible(user:is_v3/1, 'is_v3/1:\n  execute foo/1\n', 1),
        % 4. mismatched label rejected
        \+ wam_c_fact_eligible(user:is_v3/1, 'wrong_label/1:\n  proceed\n', 1)
    ->  pass(Test)
    ;   fail_test(Test, 'metadata eligibility control check failed')
    ).

test_compiled_c_maplist_matches_swi :-
    Test = 'maplist/2: compiled C solutions match SWI and reject invalid/ineligible cases',
    (   gcc_available
    ->  (   run_compiled_c_maplist
        ->  pass(Test)
        ;   fail_test(Test, 'compiled C maplist executable failed or mismatched SWI/expected diagnostics')
        )
    ;   fail_test(Test, 'gcc unavailable; compiled behavior not verified')
    ).

run_compiled_c_maplist :-
    setup_maplist_preds,
    compile_one(user:is_v3/1, V3Code),
    compile_one(user:tag_val/1, TagCode),
    compile_one(user:same_pair/1, SamePairCode),
    compile_one(user:has_body/1, HasBodyCode),
    compile_one(user:multi_clause/1, MultiCode),
    compile_one(user:wam_maplist_q/2, QCode),
    compile_one(user:wam_maplist_empty_unbound/1, EmptyUnboundCode),
    compile_one(user:wam_maplist_empty_unknown/1, EmptyUnknownCode),
    compile_one(user:wam_maplist_is_v3/1, IsV3Code),
    compile_one(user:wam_maplist_instantiate/1, InstantiateCode),
    compile_one(user:wam_maplist_same_pair/1, SamePairCallerCode),
    compile_one(user:wam_maplist_same_pair_bind/2, SamePairBindCode),
    compile_one(user:wam_maplist_backtrack/1, BacktrackCode),
    compile_one(user:wam_maplist_continuation/3, ContinuationCode),
    compile_one(user:wam_maplist_chained/3, ChainedCode),
    compile_one(user:wam_maplist_unknown/1, UnknownCallerCode),
    compile_one(user:wam_maplist_body/1, BodyCallerCode),
    compile_one(user:wam_maplist_multi/1, MultiCallerCode),
    compile_one(user:wam_maplist_compound/1, CompoundCallerCode),
    compile_one(user:wam_maplist_shared_conflict/1, SharedConflictCode),
    compile_one(user:wam_control_overwrite_a0/2, ControlCode),
    compile_wam_runtime_to_c([], RuntimeCode),
    get_time(Now),
    Stamp is round(Now * 1000000),
    wam_c_temp_path('unifyweaver_wam_c_maplist', Stamp, TmpBase),
    format(atom(RuntimePath), '~w_runtime.c', [TmpBase]),
    format(atom(PredPath), '~w_pred.c', [TmpBase]),
    format(atom(DriverPath), '~w_driver.c', [TmpBase]),
    format(atom(ExePath), '~w_bin', [TmpBase]),
    write_text_file(RuntimePath, RuntimeCode),
    atomic_list_concat([
        V3Code, TagCode, SamePairCode, HasBodyCode, MultiCode,
        QCode, EmptyUnboundCode, EmptyUnknownCode, IsV3Code,
        InstantiateCode, SamePairCallerCode, SamePairBindCode,
        BacktrackCode, ContinuationCode, ChainedCode,
        UnknownCallerCode, BodyCallerCode, MultiCallerCode, CompoundCallerCode,
        SharedConflictCode, ControlCode
    ], '\n\n', PredCode),
    format(atom(PredTranslationUnit), '#include "wam_runtime.h"~n~n~w', [PredCode]),
    write_text_file(PredPath, PredTranslationUnit),
    maplist_driver_c_source(DriverCode),
    write_text_file(DriverPath, DriverCode),
    IncludeDir = 'src/unifyweaver/targets/wam_c_runtime',
    process_create(path(gcc),
                   ['-std=c11', '-Wall', '-Wextra', '-I', IncludeDir,
                    RuntimePath, PredPath, DriverPath, '-lm', '-o', ExePath],
                   [process(Pid)]),
    process_wait(Pid, GccStatus),
    (   GccStatus == exit(0)
    ->  true
    ;   fail_test('maplist/2 C compile', 'gcc compilation failed'),
        fail
    ),
    process_create(path(timeout), ['10', ExePath],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(RunPid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(RunPid, RunStatus),
    (   ErrStr == ""
    ->  true
    ;   format(user_error, '~w', [ErrStr])
    ),
    (   RunStatus == exit(0)
    ->  true
    ;   format(atom(R), 'C test driver exited with ~w', [RunStatus]),
        fail_test('maplist/2 C runner exit', R)
    ),
    split_string(OutStr, "\n", "", Lines),
    parse_c_cases(Lines, CCases),
    GroundCases = [
        empty_unbound, empty_unknown, is_v3_valid, is_v3_single,
        is_v3_empty, is_v3_fail_first, is_v3_fail_second,
        instantiate_shared, same_pair_valid, same_pair_mismatch,
        same_pair_bind, backtrack_restore, continuation, chained,
        shared_conflict, control_builtin_overwrite
    ],
    TokenCases = [
        unknown_goal, body_goal, multi_clause_goal, compound_goal,
        var_goal, open_list, improper_list, cyclic_list, non_list
    ],
    findall(Id, (member(Id, GroundCases), \+ compare_ground(Id, CCases)), GroundBads),
    findall(Id, (member(Id, TokenCases), \+ compare_token(Id, CCases)), TokenBads),
    GroundBads == [],
    TokenBads == [].

maplist_driver_c_source(DriverCode) :-
    DriverCode =
'#include "wam_runtime.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup_is_v3_1(WamState *state);
void setup_tag_val_1(WamState *state);
void setup_same_pair_1(WamState *state);
void setup_has_body_1(WamState *state);
void setup_multi_clause_1(WamState *state);
void setup_wam_maplist_q_2(WamState *state);
void setup_wam_maplist_empty_unbound_1(WamState *state);
void setup_wam_maplist_empty_unknown_1(WamState *state);
void setup_wam_maplist_is_v3_1(WamState *state);
void setup_wam_maplist_instantiate_1(WamState *state);
void setup_wam_maplist_same_pair_1(WamState *state);
void setup_wam_maplist_same_pair_bind_2(WamState *state);
void setup_wam_maplist_backtrack_1(WamState *state);
void setup_wam_maplist_continuation_3(WamState *state);
void setup_wam_maplist_chained_3(WamState *state);
void setup_wam_maplist_unknown_1(WamState *state);
void setup_wam_maplist_body_1(WamState *state);
void setup_wam_maplist_multi_1(WamState *state);
void setup_wam_maplist_compound_1(WamState *state);
void setup_wam_maplist_shared_conflict_1(WamState *state);
void setup_wam_control_overwrite_a0_2(WamState *state);

static void ensure_h(WamState *s, int n) {
    if (s->H + n < s->H_cap)
        return;
    int cap = s->H_cap ? s->H_cap : 64;
    while (s->H + n >= cap) {
        if (cap > 1 << 28) {
            fprintf(stderr, "ensure_h: heap capacity limit exceeded\\n");
            exit(2);
        }
        cap *= 2;
    }
    WamValue *heap = realloc(s->H_array, sizeof(WamValue) * (size_t)cap);
    if (!heap) {
        fprintf(stderr, "ensure_h: heap allocation failed\\n");
        exit(2);
    }
    s->H_array = heap;
    s->H_cap = cap;
}

static WamValue cons(WamState *s, WamValue h, WamValue t) {
    ensure_h(s, 2);
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = s->H;
    s->H_array[s->H++] = h;
    s->H_array[s->H++] = t;
    return list;
}

static WamValue nil_atom(void) {
    return val_atom("[]");
}

static int is_nil_cell(WamValue *d) {
    return d->tag == VAL_ATOM && d->data.atom && strcmp(d->data.atom, "[]") == 0;
}

static WamValue mkv3(WamState *s, WamValue a, WamValue b, WamValue c) {
    ensure_h(s, 4);
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom("v/3");
    s->H_array[s->H++] = a;
    s->H_array[s->H++] = b;
    s->H_array[s->H++] = c;
    return term;
}

static WamValue mkdeb(WamState *s, WamValue a, WamValue b, WamValue c) {
    ensure_h(s, 4);
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom("deb/3");
    s->H_array[s->H++] = a;
    s->H_array[s->H++] = b;
    s->H_array[s->H++] = c;
    return term;
}

static WamValue mkpair(WamState *s, WamValue a, WamValue b) {
    ensure_h(s, 3);
    WamValue term;
    term.tag = VAL_STR;
    term.data.ref_addr = s->H;
    s->H_array[s->H++] = val_atom("pair/2");
    s->H_array[s->H++] = a;
    s->H_array[s->H++] = b;
    return term;
}

static int is_plain_atom(const char *str) {
    if (str == NULL || str[0] == 0)
        return 0;
    if (!(str[0] >= 97 && str[0] <= 122))
        return 0;
    for (const char *p = str + 1; *p; p++) {
        if (!(isalnum((unsigned char)*p) || *p == 95))
            return 0;
    }
    return 1;
}

static void print_atom(const char *str) {
    if (str == NULL) {
        fputc(39, stdout);
        fputc(39, stdout);
        return;
    }
    if (strcmp(str, "[]") == 0) {
        fputs("[]", stdout);
        return;
    }
    if (is_plain_atom(str)) {
        fputs(str, stdout);
        return;
    }
    fputc(39, stdout);
    for (const char *p = str; *p; p++) {
        if (*p == 92 || *p == 39)
            fputc(92, stdout);
        fputc(*p, stdout);
    }
    fputc(39, stdout);
}

static int split_functor(const char *qualified, char *name, size_t name_sz, int *arity) {
    const char *slash = strrchr(qualified, 47);
    if (slash == NULL || slash == qualified)
        return 0;
    size_t nlen = (size_t)(slash - qualified);
    if (nlen + 1 > name_sz)
        return 0;
    memcpy(name, qualified, nlen);
    name[nlen] = 0;
    *arity = atoi(slash + 1);
    return 1;
}

static void print_term(WamState *state, WamValue v, int depth);

static int is_list_functor(const char *name) {
    return strcmp(name, ".") == 0 || strcmp(name, "[|]") == 0;
}

static void print_list_from_cells(WamState *state, WamValue head, WamValue tail, int depth) {
    fputs("[", stdout);
    print_term(state, head, depth + 1);
    for (;;) {
        WamValue *td = wam_deref_ptr(state, &tail);
        if (is_nil_cell(td))
            break;
        int c = wam_cons_head_addr(state, td);
        if (c >= 0) {
            fputs(",", stdout);
            print_term(state, state->H_array[c], depth + 1);
            tail = state->H_array[c + 1];
            continue;
        }
        fputs("|", stdout);
        print_term(state, tail, depth + 1);
        break;
    }
    fputs("]", stdout);
}

static void print_term(WamState *state, WamValue v, int depth) {
    if (depth > 128) {
        fputs("...", stdout);
        return;
    }
    WamValue *d = wam_deref_ptr(state, &v);
    switch (d->tag) {
    case VAL_ATOM:
        print_atom(d->data.atom);
        return;
    case VAL_INT:
        printf("%d", d->data.integer);
        return;
    case VAL_FLOAT:
        printf("%g", d->data.floating);
        return;
    case VAL_UNBOUND:
        fputs("_", stdout);
        return;
    case VAL_LIST:
        print_list_from_cells(state,
                              state->H_array[d->data.ref_addr],
                              state->H_array[d->data.ref_addr + 1],
                              depth);
        return;
    case VAL_STR: {
        WamValue *fn = &state->H_array[d->data.ref_addr];
        char name[128];
        int arity = 0;
        if (fn->tag != VAL_ATOM || fn->data.atom == NULL ||
            !split_functor(fn->data.atom, name, sizeof name, &arity)) {
            fputs("\'<struct>\'", stdout);
            return;
        }
        if (arity == 2 && is_list_functor(name)) {
            print_list_from_cells(state,
                                  state->H_array[d->data.ref_addr + 1],
                                  state->H_array[d->data.ref_addr + 2],
                                  depth);
            return;
        }
        print_atom(name);
        if (arity <= 0)
            return;
        fputs("(", stdout);
        for (int i = 0; i < arity; i++) {
            if (i > 0)
                fputs(",", stdout);
            print_term(state, state->H_array[d->data.ref_addr + 1 + i], depth + 1);
        }
        fputs(")", stdout);
        return;
    }
    case VAL_REF:
        fputs("_", stdout);
        return;
    default:
        fputs("\'<unknown>\'", stdout);
        return;
    }
}

static void emit_case(const char *id, const char *status, WamState *state, WamValue *term) {
    printf("CASE %s\\n", id);
    printf("STATUS %s\\n", status);
    if (term) {
        fputs("TERM ", stdout);
        print_term(state, *term, 0);
        printf("\\n");
    }
    fflush(stdout);
}

static void emit_token(const char *id, const char *status, const char *token) {
    printf("CASE %s\\n", id);
    printf("STATUS %s\\n", status);
    printf("TERM %s\\n", token);
    fflush(stdout);
}

static int is_unsupported_maplist(WamState *s, int rc) {
    return (rc == WAM_ERR_UNSUPPORTED || s->error == WAM_ERR_UNSUPPORTED) &&
           s->error_op != NULL && strcmp(s->error_op, "maplist/2") == 0;
}

static void setup_all(WamState *s) {
    setup_is_v3_1(s);
    setup_tag_val_1(s);
    setup_same_pair_1(s);
    setup_has_body_1(s);
    setup_multi_clause_1(s);
    setup_wam_maplist_q_2(s);
    setup_wam_maplist_empty_unbound_1(s);
    setup_wam_maplist_empty_unknown_1(s);
    setup_wam_maplist_is_v3_1(s);
    setup_wam_maplist_instantiate_1(s);
    setup_wam_maplist_same_pair_1(s);
    setup_wam_maplist_same_pair_bind_2(s);
    setup_wam_maplist_backtrack_1(s);
    setup_wam_maplist_continuation_3(s);
    setup_wam_maplist_chained_3(s);
    setup_wam_maplist_unknown_1(s);
    setup_wam_maplist_body_1(s);
    setup_wam_maplist_multi_1(s);
    setup_wam_maplist_compound_1(s);
    setup_wam_maplist_shared_conflict_1(s);
    setup_wam_control_overwrite_a0_2(s);
}

static void run_c_metadata_controls(void) {
    WamState s;
    wam_state_init(&s);

    // 1. ordinary re-registration clears eligibility (C check)
    wam_register_predicate(&s, "test_fact/1", 42);
    wam_set_predicate_fact_eligible(&s, "test_fact/1", true);
    const PredEntry *e1 = wam_lookup_fact_eligible_entry(&s, "test_fact");
    if (!e1 || !(e1->flags & WAM_PRED_FACT_ELIGIBLE)) {
        fprintf(stderr, "C metadata control: test_fact/1 was not marked eligible\\n");
        exit(1);
    }
    wam_register_predicate(&s, "test_fact/1", 99);
    const PredEntry *e2 = wam_lookup_fact_eligible_entry(&s, "test_fact");
    if (e2 != NULL) {
        fprintf(stderr, "C metadata control: re-registration did not clear fact eligibility\\n");
        exit(1);
    }

    // 2. slash-containing literal goal must not alias another predicate (C lookup check)
    wam_register_predicate(&s, "aliased/1", 100);
    wam_set_predicate_fact_eligible(&s, "aliased/1", true);
    if (!wam_lookup_fact_eligible_entry(&s, "aliased")) {
        fprintf(stderr, "C metadata control: lookup aliased failed\\n");
        exit(1);
    }
    if (wam_lookup_fact_eligible_entry(&s, "aliased/1") != NULL) {
        fprintf(stderr, "C metadata control: slash-containing goal must not alias\\n");
        exit(1);
    }
    if (wam_lookup_fact_eligible_entry(&s, "foo/bar") != NULL) {
        fprintf(stderr, "C metadata control: slash-containing literal goal must return NULL\\n");
        exit(1);
    }

    // 3. overlong name rejected
    char overlong[300];
    memset(overlong, 97, sizeof(overlong) - 1);
    overlong[sizeof(overlong) - 1] = 0;
    if (wam_lookup_fact_eligible_entry(&s, overlong) != NULL) {
        fprintf(stderr, "C metadata control: overlong goal name was not rejected\\n");
        exit(1);
    }

    wam_free_state(&s);
}

int main(void) {
    run_c_metadata_controls();
    // 1. empty_unbound
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        ensure_h(&s, 1);
        int r_addr = s.H++;
        s.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        WamValue args[1] = { r_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_empty_unbound/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("empty_unbound", "ok", &s, &r_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("empty_unbound", "fail", &s, NULL);
        } else {
            emit_case("empty_unbound", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 2. empty_unknown
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        ensure_h(&s, 1);
        int r_addr = s.H++;
        s.H_array[r_addr] = val_unbound("R");
        WamValue r_ref = { .tag = VAL_REF, .data = { .ref_addr = r_addr } };
        WamValue args[1] = { r_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_empty_unknown/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("empty_unknown", "ok", &s, &r_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("empty_unknown", "fail", &s, NULL);
        } else {
            emit_case("empty_unknown", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 3. is_v3_valid
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = mkv3(&s, val_int(1), val_int(0), val_int(0));
        WamValue v2 = mkv3(&s, val_int(2), val_int(0), val_int(0));
        WamValue list = cons(&s, v1, cons(&s, v2, nil_atom()));
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("is_v3_valid", "ok", &s, &list);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("is_v3_valid", "fail", &s, NULL);
        } else {
            emit_case("is_v3_valid", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 4. is_v3_single
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = mkv3(&s, val_atom("a"), val_atom("b"), val_atom("c"));
        WamValue list = cons(&s, v1, nil_atom());
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("is_v3_single", "ok", &s, &list);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("is_v3_single", "fail", &s, NULL);
        } else {
            emit_case("is_v3_single", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 5. is_v3_empty
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue list = nil_atom();
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("is_v3_empty", "ok", &s, &list);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("is_v3_empty", "fail", &s, NULL);
        } else {
            emit_case("is_v3_empty", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 6. is_v3_fail_first
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue deb = mkdeb(&s, val_int(0), nil_atom(), nil_atom());
        WamValue v1 = mkv3(&s, val_int(1), val_int(0), val_int(0));
        WamValue list = cons(&s, deb, cons(&s, v1, nil_atom()));
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("is_v3_fail_first", "ok", &s, &list);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("is_v3_fail_first", "fail", &s, NULL);
        } else {
            emit_case("is_v3_fail_first", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 7. is_v3_fail_second
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = mkv3(&s, val_int(1), val_int(0), val_int(0));
        WamValue deb = mkdeb(&s, val_int(0), nil_atom(), nil_atom());
        WamValue list = cons(&s, v1, cons(&s, deb, nil_atom()));
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("is_v3_fail_second", "ok", &s, &list);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("is_v3_fail_second", "fail", &s, NULL);
        } else {
            emit_case("is_v3_fail_second", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 8. instantiate_shared
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        ensure_h(&s, 1);
        int out_addr = s.H++;
        s.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_instantiate/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("instantiate_shared", "ok", &s, &out_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("instantiate_shared", "fail", &s, NULL);
        } else {
            emit_case("instantiate_shared", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 9. same_pair_valid
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue p1 = mkpair(&s, val_atom("a"), val_atom("a"));
        WamValue p2 = mkpair(&s, val_int(1), val_int(1));
        WamValue list = cons(&s, p1, cons(&s, p2, nil_atom()));
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_same_pair/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("same_pair_valid", "ok", &s, &list);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("same_pair_valid", "fail", &s, NULL);
        } else {
            emit_case("same_pair_valid", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 10. same_pair_mismatch
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue p1 = mkpair(&s, val_atom("a"), val_atom("b"));
        WamValue list = cons(&s, p1, nil_atom());
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_same_pair/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("same_pair_mismatch", "ok", &s, &list);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("same_pair_mismatch", "fail", &s, NULL);
        } else {
            emit_case("same_pair_mismatch", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 11. same_pair_bind
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        ensure_h(&s, 3);
        int y_addr = s.H++;
        s.H_array[y_addr] = val_unbound("Y");
        WamValue y_ref = { .tag = VAL_REF, .data = { .ref_addr = y_addr } };
        WamValue p1 = mkpair(&s, val_int(10), y_ref);
        WamValue list = cons(&s, p1, nil_atom());
        int out_addr = s.H++;
        s.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[2] = { list, out_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_same_pair_bind/2", args, 2);
        if (rc == 0 && s.error == 0) {
            emit_case("same_pair_bind", "ok", &s, &out_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("same_pair_bind", "fail", &s, NULL);
        } else {
            emit_case("same_pair_bind", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 12. backtrack_restore
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        ensure_h(&s, 1);
        int out_addr = s.H++;
        s.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[1] = { out_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_backtrack/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("backtrack_restore", "ok", &s, &out_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("backtrack_restore", "fail", &s, NULL);
        } else {
            emit_case("backtrack_restore", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 13. continuation
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = mkv3(&s, val_int(1), val_int(2), val_int(3));
        WamValue list = cons(&s, v1, nil_atom());
        ensure_h(&s, 2);
        int s1_addr = s.H++;
        s.H_array[s1_addr] = val_unbound("S1");
        WamValue s1_ref = { .tag = VAL_REF, .data = { .ref_addr = s1_addr } };
        int s2_addr = s.H++;
        s.H_array[s2_addr] = val_unbound("S2");
        WamValue s2_ref = { .tag = VAL_REF, .data = { .ref_addr = s2_addr } };
        WamValue args[3] = { list, s1_ref, s2_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_continuation/3", args, 3);
        if (rc == 0 && s.error == 0) {
            emit_case("continuation", "ok", &s, &s2_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("continuation", "fail", &s, NULL);
        } else {
            emit_case("continuation", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 14. chained
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = mkv3(&s, val_int(1), val_int(2), val_int(3));
        WamValue list1 = cons(&s, v1, nil_atom());
        WamValue p1 = mkpair(&s, val_atom("x"), val_atom("x"));
        WamValue list2 = cons(&s, p1, nil_atom());
        ensure_h(&s, 1);
        int res_addr = s.H++;
        s.H_array[res_addr] = val_unbound("Res");
        WamValue res_ref = { .tag = VAL_REF, .data = { .ref_addr = res_addr } };
        WamValue args[3] = { list1, list2, res_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_chained/3", args, 3);
        if (rc == 0 && s.error == 0) {
            emit_case("chained", "ok", &s, &res_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("chained", "fail", &s, NULL);
        } else {
            emit_case("chained", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 15. unknown_goal
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = val_int(1);
        WamValue list = cons(&s, v1, nil_atom());
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_unknown/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("unknown_goal", "runtime_error", "unsupported_ok");
        } else {
            emit_token("unknown_goal", "fail", "unexpected_status");
        }
        wam_free_state(&s);
    }

    // 16. body_goal
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = val_int(1);
        WamValue list = cons(&s, v1, nil_atom());
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_body/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("body_goal", "runtime_error", "unsupported_ok");
        } else {
            emit_token("body_goal", "fail", "unexpected_status");
        }
        wam_free_state(&s);
    }

    // 17. multi_clause_goal
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = val_atom("a");
        WamValue list = cons(&s, v1, nil_atom());
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_multi/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("multi_clause_goal", "runtime_error", "unsupported_ok");
        } else {
            emit_token("multi_clause_goal", "fail", "unexpected_status");
        }
        wam_free_state(&s);
    }

    // 18. compound_goal
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = val_int(1);
        WamValue list = cons(&s, v1, nil_atom());
        WamValue args[1] = { list };
        int rc = wam_run_predicate(&s, "wam_maplist_compound/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("compound_goal", "runtime_error", "unsupported_ok");
        } else {
            emit_token("compound_goal", "fail", "unexpected_status");
        }
        wam_free_state(&s);
    }

    // 19. var_goal
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue v1 = val_int(1);
        WamValue list = cons(&s, v1, nil_atom());
        WamValue args[2] = { val_unbound("G"), list };
        int rc = wam_run_predicate(&s, "wam_maplist_q/2", args, 2);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("var_goal", "runtime_error", "unsupported_ok");
        } else {
            emit_token("var_goal", "fail", "unexpected_status");
        }
        wam_free_state(&s);
    }

    // 20. open_list
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        ensure_h(&s, 3);
        int var_addr = s.H;
        s.H_array[s.H++] = val_unbound("_");
        WamValue tail_var;
        tail_var.tag = VAL_REF;
        tail_var.data.ref_addr = var_addr;
        WamValue open_l = cons(&s, mkv3(&s, val_int(1), val_int(0), val_int(0)), tail_var);
        WamValue args[1] = { open_l };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("open_list", "runtime_error", "open_ok");
        } else {
            emit_token("open_list", "fail", "open_bad");
        }
        wam_free_state(&s);
    }

    // 21. improper_list
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue imp_l = cons(&s, mkv3(&s, val_int(1), val_int(0), val_int(0)), val_atom("bad_tail"));
        WamValue args[1] = { imp_l };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("improper_list", "runtime_error", "improper_ok");
        } else {
            emit_token("improper_list", "fail", "improper_bad");
        }
        wam_free_state(&s);
    }

    // 22. cyclic_list
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue head_val = mkv3(&s, val_int(1), val_int(0), val_int(0));
        ensure_h(&s, 2);
        int c_addr = s.H;
        s.H += 2;
        s.H_array[c_addr] = head_val;
        WamValue self_ref;
        self_ref.tag = VAL_LIST;
        self_ref.data.ref_addr = c_addr;
        s.H_array[c_addr + 1] = self_ref;
        WamValue cyclic_l = self_ref;
        WamValue args[1] = { cyclic_l };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("cyclic_list", "runtime_error", "cyclic_ok");
        } else {
            emit_token("cyclic_list", "fail", "cyclic_bad");
        }
        wam_free_state(&s);
    }

    // 23. non_list
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue non_l = val_atom("not_a_list");
        WamValue args[1] = { non_l };
        int rc = wam_run_predicate(&s, "wam_maplist_is_v3/1", args, 1);
        if (is_unsupported_maplist(&s, rc)) {
            emit_token("non_list", "runtime_error", "non_list_ok");
        } else {
            emit_token("non_list", "fail", "non_list_bad");
        }
        wam_free_state(&s);
    }

    // 24. shared_conflict
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        ensure_h(&s, 1);
        int l_addr = s.H++;
        s.H_array[l_addr] = val_unbound("L");
        WamValue l_ref = { .tag = VAL_REF, .data = { .ref_addr = l_addr } };
        WamValue args[1] = { l_ref };
        int rc = wam_run_predicate(&s, "wam_maplist_shared_conflict/1", args, 1);
        if (rc == 0 && s.error == 0) {
            emit_case("shared_conflict", "ok", &s, &l_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("shared_conflict", "fail", &s, NULL);
        } else {
            emit_case("shared_conflict", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    // 25. control_builtin_overwrite
    {
        WamState s;
        wam_state_init(&s);
        setup_all(&s);
        WamValue in_val = val_atom("retained_ok");
        ensure_h(&s, 1);
        int out_addr = s.H++;
        s.H_array[out_addr] = val_unbound("Out");
        WamValue out_ref = { .tag = VAL_REF, .data = { .ref_addr = out_addr } };
        WamValue args[2] = { in_val, out_ref };
        int rc = wam_run_predicate(&s, "wam_control_overwrite_a0/2", args, 2);
        WamValue a0_deref = *wam_deref_ptr(&s, &s.A[0]);
        bool a0_was_overwritten = !val_equal(a0_deref, in_val);
        if (rc == 0 && s.error == 0 && a0_was_overwritten) {
            emit_case("control_builtin_overwrite", "ok", &s, &out_ref);
        } else if (rc == WAM_HALT && s.error == 0) {
            emit_case("control_builtin_overwrite", "fail", &s, NULL);
        } else {
            emit_case("control_builtin_overwrite", "runtime_error", &s, NULL);
        }
        wam_free_state(&s);
    }

    return 0;
}
'.

run_tests :-
    (   tests_already_ran
    ->  true
    ;   assert(tests_already_ran),
        run_tests_once
    ).

run_tests_once :-
    format('~n=== WAM-C maplist/2 Tests ===~n~n'),
    setup_maplist_preds,
    test_generation_maplist_builtin,
    test_wam_emits_maplist_builtin,
    test_wrong_answer_rejected,
    test_metadata_controls,
    test_compiled_c_maplist_matches_swi,
    cleanup_maplist_preds,
    format('~n=== WAM-C maplist/2 Tests Complete ===~n'),
    (   test_failed -> halt(1) ; true ).

:- initialization(run_tests, main).
