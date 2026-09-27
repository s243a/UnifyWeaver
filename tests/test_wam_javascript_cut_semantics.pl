:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_javascript_cut_semantics.pl
%
% Cut and choice-point barrier conformance for the JS WAM backend.
%
% Every probe pNN is a failure-driven loop that prints ALL solutions of
% pNN_t/1.  The SAME clauses run under SWI-Prolog (the oracle) and under
% Node, in four emit modes, and the two stdout streams must match exactly:
%
%   interpreter      -- everything on the WAM interpreter
%   mixed            -- every eligible predicate lowered to a JS function
%   functions        -- same, forced
%   mixed(Helpers)   -- only the pNN_h/_a/_b/_c/_g/_r helpers lowered, so
%                       the interpreted caller / lowered callee boundary
%                       (the shape the D44 cut bug lived in) is exercised
%                       in BOTH directions
%
% The barrier model these probes pin down is written up in
% docs/WAM_JAVASCRIPT_STATUS.md, section "Cut and choice-point barriers".
%
%   swipl -q -g run_tests -t halt tests/test_wam_javascript_cut_semantics.pl

:- module(test_wam_javascript_cut_semantics,
          [test_wam_javascript_cut_semantics/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_javascript_target',
              [write_wam_javascript_project/3]).
:- use_module('../src/unifyweaver/targets/wam_javascript_lowered_emitter',
              [wam_javascript_explain_lower/3]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

% ---------------------------------------------------------------------
% The probe program.  Kept as ONE source of truth: the same clause terms
% are asserted into user: (so SWI runs them) and handed to the JS WAM
% emitter (so Node runs them).  There is no second copy to drift.
% ---------------------------------------------------------------------

%% cut_probe(-Name, -Context)
%  Name is pNN; Context names the barrier context the probe pins down.
cut_probe(p01, 'neck cut in a callee reached by Execute (D44 shape)').
cut_probe(p02, 'mid-body cut').
cut_probe(p03, 'last-goal cut').
cut_probe(p04, 'cut in a callee tail-called from a nondet caller').
cut_probe(p05, 'cut in an if-then-else CONDITION (condition CPs only)').
cut_probe(p06, 'cut in an ITE condition that then fails').
cut_probe(p07, 'cut in the THEN branch cuts the enclosing clause').
cut_probe(p08, 'cut in the ELSE branch cuts the enclosing clause').
cut_probe(p09, 'cut inside call/1 is local').
cut_probe(p10, 'cut inside call((G, !)) is local to the call').
cut_probe(p11, 'cut inside \\+ is local to the negation').
cut_probe(p12, 'cut inside a findall inner goal').
cut_probe(p13, 'cut inside findall; caller still nondeterministic after').
cut_probe(p14, 'cut inside a bagof inner goal').
cut_probe(p15, 'cut inside a setof inner goal').
cut_probe(p16, 'cut inside an aggregate_all inner goal').
cut_probe(p17, 'once/1 after a nondeterministic goal').
cut_probe(p18, 'once/1 does not cut the enclosing predicate clauses').
cut_probe(p19, 'deep recursion: cut binds only that activation').
cut_probe(p20, 'cut after between/3 (a CP-creating builtin)').
cut_probe(p21, 'cut after member/2').
cut_probe(p22, 'caller keeps member/2 alternatives across a cutting callee').
cut_probe(p23, '-> inside \\+').
cut_probe(p24, 'nested ITE with a cut in the inner condition').
cut_probe(p25, 'disjunction with a cut in the left branch').
cut_probe(p26, 'disjunction with a cut in the right branch').
cut_probe(p27, 'cut followed by more nondeterminism in the same clause').
cut_probe(p28, 'cutting helper invoked from inside findall').
cut_probe(p29, 'findall with a cut, nested inside \\+').
cut_probe(p30, 'cut in an ITE condition inside a nondeterministic caller').
cut_probe(p31, 'forall/2 (soft-cut rewrite)').
cut_probe(p32, 'cut inside the ACTION of forall/2').
cut_probe(p33, 'cutting callee two frames deep').
cut_probe(p34, 'cut in the first clause guards the later clauses').
cut_probe(p35, 'cut with a choice point created before the guard').

cut_probe_clause(d(1)).
cut_probe_clause(d(2)).
cut_probe_clause(d(3)).
cut_probe_clause(e(a)).
cut_probe_clause(e(b)).

cut_probe_clause((p01_h(X) :- !, X = one)).
cut_probe_clause((p01_h(_) :- fail)).
cut_probe_clause((p01_t(r(X, Y)) :- d(X), p01_h(Y))).

cut_probe_clause((p02_h(X, Y) :- d(X), X > 1, !, Y = big)).
cut_probe_clause(p02_h(_, small)).
cut_probe_clause((p02_t(r(X, Y)) :- p02_h(X, Y))).

cut_probe_clause((p03_h(X) :- d(X), !)).
cut_probe_clause(p03_h(99)).
cut_probe_clause((p03_t(X) :- p03_h(X))).

cut_probe_clause((p04_h(X) :- d(X), !)).
cut_probe_clause(p04_h(0)).
cut_probe_clause((p04_c(Y) :- e(_), p04_h(Y))).
cut_probe_clause((p04_t(Y) :- p04_c(Y))).

cut_probe_clause((p05_h(X, R) :- ( d(X), ! -> R = then ; R = else ))).
cut_probe_clause((p05_t(r(X, R)) :- p05_h(X, R))).

cut_probe_clause((p06_h(R) :- ( d(_), !, fail -> R = then ; R = else ))).
cut_probe_clause(p06_h(second)).
cut_probe_clause((p06_t(R) :- p06_h(R))).

cut_probe_clause((p07_h(X, R) :- d(X), ( X > 1 -> !, R = big ; R = small ))).
cut_probe_clause(p07_h(_, none)).
cut_probe_clause((p07_t(r(X, R)) :- p07_h(X, R))).

cut_probe_clause((p08_h(X, R) :- d(X), ( X > 2 -> R = big ; !, R = small ))).
cut_probe_clause(p08_h(_, none)).
cut_probe_clause((p08_t(r(X, R)) :- p08_h(X, R))).

cut_probe_clause((p09_h(X) :- d(X), call(!))).
cut_probe_clause((p09_t(X) :- p09_h(X))).

cut_probe_clause((p10_h(X) :- call((d(X), !)))).
cut_probe_clause(p10_h(9)).
cut_probe_clause((p10_t(X) :- p10_h(X))).

cut_probe_clause((p11_h(X) :- d(X), \+ ( e(_), !, fail ))).
cut_probe_clause((p11_t(X) :- p11_h(X))).

cut_probe_clause((p12_h(L) :- findall(X, (d(X), !), L))).
cut_probe_clause((p12_t(L) :- p12_h(L))).

cut_probe_clause((p13_h(r(Y, L)) :- e(Y), findall(X, (d(X), !), L))).
cut_probe_clause((p13_t(R) :- p13_h(R))).

cut_probe_clause((p14_h(L) :- bagof(X, (d(X), !), L))).
cut_probe_clause((p14_t(L) :- p14_h(L))).

cut_probe_clause((p15_h(L) :- setof(X, (d(X), !), L))).
cut_probe_clause((p15_t(L) :- p15_h(L))).

cut_probe_clause((p16_h(N) :- aggregate_all(count, (d(_), !), N))).
cut_probe_clause((p16_t(N) :- p16_h(N))).

cut_probe_clause((p17_h(X) :- d(X), once(e(_)))).
cut_probe_clause((p17_t(X) :- p17_h(X))).

cut_probe_clause((p18_h(X) :- once(d(X)))).
cut_probe_clause(p18_h(9)).
cut_probe_clause((p18_t(X) :- p18_h(X))).

cut_probe_clause(p19_r([], [])).
cut_probe_clause((p19_r([H|T], [H2|T2]) :-
    ( H > 1 -> H2 = big ; H2 = H ), !, p19_r(T, T2))).
cut_probe_clause((p19_t(L) :- p19_r([1, 2, 3], L))).

cut_probe_clause((p20_h(X) :- between(1, 3, X), X > 1, !)).
cut_probe_clause(p20_h(0)).
cut_probe_clause((p20_t(X) :- p20_h(X))).

cut_probe_clause((p21_h(X) :- member(X, [a, b, c]), !)).
cut_probe_clause(p21_h(z)).
cut_probe_clause((p21_t(X) :- p21_h(X))).

cut_probe_clause(p22_g(a, one)).
cut_probe_clause(p22_g(b, two)).
cut_probe_clause((p22_h(K, V) :- p22_g(K, V), !)).
cut_probe_clause((p22_t(r(K, V)) :- member(K, [a, b]), p22_h(K, V))).

cut_probe_clause((p23_h(X) :- d(X), \+ ( X > 1 -> fail ; true ))).
cut_probe_clause((p23_t(X) :- p23_h(X))).

cut_probe_clause((p24_h(X, R) :- d(X), ( ( X > 1, ! ) -> R = hi ; R = lo ))).
cut_probe_clause((p24_t(r(X, R)) :- p24_h(X, R))).

cut_probe_clause((p25_h(X) :- ( d(X), ! ; X = z ))).
cut_probe_clause(p25_h(9)).
cut_probe_clause((p25_t(X) :- p25_h(X))).

cut_probe_clause((p26_h(X) :- ( fail ; d(X), ! ))).
cut_probe_clause(p26_h(9)).
cut_probe_clause((p26_t(X) :- p26_h(X))).

cut_probe_clause((p27_h(r(X, Y)) :- d(X), !, e(Y))).
cut_probe_clause((p27_t(R) :- p27_h(R))).

cut_probe_clause((p28_h(X) :- d(X), !)).
cut_probe_clause((p28_c(L) :- findall(Y, (e(_), p28_h(Y)), L))).
cut_probe_clause((p28_t(L) :- p28_c(L))).

cut_probe_clause((p29_h :- \+ ( findall(X, (d(X), !), L), L == [] ))).
cut_probe_clause((p29_t(ok) :- p29_h)).

cut_probe_clause((p30_h(X, R) :- ( d(X), !, X > 1 -> R = yes ; R = no ))).
cut_probe_clause((p30_c(r(Y, R)) :- e(Y), p30_h(_X, R))).
cut_probe_clause((p30_t(R) :- p30_c(R))).

cut_probe_clause((p31_h(ok) :- forall(d(X), X > 0))).
cut_probe_clause((p31_t(R) :- p31_h(R))).

cut_probe_clause((p32_h(ok) :- forall(d(X), (e(_), !, X > 0)))).
cut_probe_clause((p32_t(R) :- p32_h(R))).

cut_probe_clause((p33_a(X) :- d(X), !)).
cut_probe_clause((p33_b(X) :- p33_a(X))).
cut_probe_clause((p33_c(r(Y, X)) :- e(Y), p33_b(X))).
cut_probe_clause((p33_t(R) :- p33_c(R))).

cut_probe_clause((p34_h(X, one) :- X == 1, !)).
cut_probe_clause((p34_h(X, two) :- X == 2, !)).
cut_probe_clause(p34_h(_, many)).
cut_probe_clause((p34_t(r(X, R)) :- d(X), p34_h(X, R))).

cut_probe_clause((p35_h(X, Y) :- member(Y, [p, q]), X > 1, !)).
cut_probe_clause(p35_h(_, none)).
cut_probe_clause((p35_t(r(X, Y)) :- d(X), p35_h(X, Y))).

% ---------------------------------------------------------------------
% Installation
% ---------------------------------------------------------------------

cut_probe_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

install_cut_probes :-
    findall(PI, (cut_probe_clause(C), cut_probe_head(C, PI)), PIs0),
    sort(PIs0, PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(cut_probe(N, _),
           ( functor(D, N, 0), catch(retractall(user:D), _, true) )),
    forall(cut_probe_clause(C), assertz(user:C)),
    % Failure-driven driver: print every solution of pNN_t/1, then stop.
    % The driver itself holds no cut, so what the probe measures is the
    % cut inside pNN_t's callees.
    forall(cut_probe(N, _),
           ( atom_concat(N, '_t', TName),
             TG =.. [TName, R],
             assertz(user:(N :- ( TG, write(R), nl, fail ; true ))) )).

cut_probe_preds(Preds) :-
    findall(user:PI, (cut_probe_clause(C), cut_probe_head(C, PI)), P0),
    findall(user:(N/0), cut_probe(N, _), P1),
    append(P0, P1, P2),
    sort(P2, Preds).

% pNN_h / _a / _b / _c / _g / _r are the "helper" tier: lowering only
% those puts a lowered callee under an interpreted caller AND an
% interpreted callee under a lowered caller.
cut_probe_helper_preds(Hot) :-
    findall(F/A,
            (   cut_probe_clause(C), cut_probe_head(C, F/A),
                atom_concat(_, Suffix, F),
                atom_length(Suffix, 2),
                memberchk(Suffix, ['_h', '_a', '_b', '_c', '_g', '_r'])
            ), Hot0),
    sort(Hot0, Hot).

% ---------------------------------------------------------------------
% Running
% ---------------------------------------------------------------------

swi_probe_output(N, Out) :-
    with_output_to(string(Raw),
                   ( catch(user:N, _, true) -> true ; true )),
    normalize_output(Raw, Out).

node_probe_output(Dir, N, Out) :-
    directory_file_path(Dir, 'js', JsDir),
    format(atom(Key), '~w/0', [N]),
    process_create(path(node), ['generated_program.js', Key],
                   [cwd(JsDir), stdout(pipe(O)), stderr(pipe(E)),
                    process(Pid)]),
    read_string(O, _, S1),
    read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    atomic_list_concat([S1, S2], Raw),
    normalize_output(Raw, Out).

% The generated main prints an A-register dump plus a trailing
% true/false; strip those and all whitespace so only the probe's own
% write/1 lines are compared.
normalize_output(Raw, Out) :-
    split_string(Raw, "\n", "\r", Lines0),
    exclude(cut_probe_noise_line, Lines0, Lines),
    atomic_list_concat(Lines, "\n", Joined),
    split_string(Joined, " \t", "", Parts),
    atomic_list_concat(Parts, '', Out).

cut_probe_noise_line("").
cut_probe_noise_line("true").
cut_probe_noise_line("false").
cut_probe_noise_line(L) :-
    string_concat("A", Rest, L),
    sub_string(Rest, B, _, _, " = "),
    sub_string(Rest, 0, B, _, Num),
    number_string(_, Num).

compile_cut_probes(Mode, Dir) :-
    cut_probe_preds(Preds),
    (   Mode == helpers
    ->  cut_probe_helper_preds(Hot), Opts = [emit_mode(mixed(Hot))]
    ;   Opts = [emit_mode(Mode)]
    ),
    format(atom(Dir), 'output/js_wam_cut_semantics_~w', [Mode]),
    make_directory_path(Dir),
    write_wam_javascript_project(Preds, Opts, Dir).

%% run_cut_probe_mode(+Mode)
%  Compile in Mode, then require an EXACT match with SWI for every probe.
run_cut_probe_mode(Mode) :-
    compile_cut_probes(Mode, Dir),
    findall(fail(N, Ctx, Swi, Node),
            (   cut_probe(N, Ctx),
                swi_probe_output(N, Swi),
                node_probe_output(Dir, N, Node),
                Swi \== Node
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, J), Fails),
               format(user_error,
                      'CUT PROBE DIVERGENCE [~w] ~w (~w)~n  swi : ~q~n  node: ~q~n',
                      [Mode, N, Ctx, S, J])),
        fail
    ).

test_wam_javascript_cut_semantics :-
    run_tests(js_wam_cut_semantics).

:- begin_tests(js_wam_cut_semantics).

test(interpreter_matches_swi, [setup(install_cut_probes)]) :-
    run_cut_probe_mode(interpreter).

test(helper_lowering_matches_swi, [setup(install_cut_probes)]) :-
    run_cut_probe_mode(helpers).

test(mixed_matches_swi, [setup(install_cut_probes)]) :-
    run_cut_probe_mode(mixed).

test(functions_matches_swi, [setup(install_cut_probes)]) :-
    run_cut_probe_mode(functions).

% A commit-less try_me_else block is a plain disjunction (A ; B), not an
% if-then-else.  Lowering it as ite(A, [], B) deletes B as a retry
% alternative; the emitter must decline.
test(plain_disjunction_not_lowered_as_ite, [setup(install_cut_probes)]) :-
    compile_predicate_to_wam_text(user:p25_h/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_explain_lower(user:p25_h/1, Wam, Decision),
    assertion(Decision = fallback(_)).

% A predicate whose clauses are neither first-argument exclusive nor
% cut-committed can yield more than one solution; a first-solution
% lowering would hide the rest.
test(multi_solution_predicate_not_lowered, [setup(install_cut_probes)]) :-
    compile_predicate_to_wam_text(user:p18_h/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_explain_lower(user:p18_h/1, Wam, Decision),
    assertion(Decision = fallback(_)).

% A CP-creating builtin outside a commit wrapper keeps the predicate on
% the interpreter (the rule that used to name only member/2).
test(nondet_builtin_taints_caller, [setup(install_cut_probes)]) :-
    compile_predicate_to_wam_text(user:p20_h/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_explain_lower(user:p20_h/1, Wam, Decision),
    assertion(Decision = fallback(_)).

% Cut-committed clause chains and first-argument-exclusive fact tables
% must STILL lower: the refusals above are targeted, not blanket.
test(committed_clause_chain_still_lowers, [setup(install_cut_probes)]) :-
    compile_predicate_to_wam_text(user:p34_h/2,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_explain_lower(user:p34_h/2, Wam, Decision),
    assertion(Decision = lower(_)).

test(first_arg_exclusive_facts_still_lower, [setup(install_cut_probes)]) :-
    compile_predicate_to_wam_text(user:d/1,
        [ite_use_y_level(true), inline_bagof_setof(true)], Wam),
    wam_javascript_explain_lower(user:d/1, Wam, Decision),
    assertion(Decision = lower(_)).

:- end_tests(js_wam_cut_semantics).
