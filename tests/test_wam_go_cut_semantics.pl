:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_wam_go_cut_semantics.pl
%
% Port of tests/test_wam_javascript_cut_semantics.pl to the Go WAM
% backend. Every probe pNN is a failure-driven loop that prints ALL
% solutions of pNN_t/1. The SAME clauses run under SWI-Prolog (the
% oracle) and under the compiled Go binary; stdout must match after
% whitespace / true-false / A-register dump stripping.
%
% Go has no emit_mode (interpreter/mixed/functions). This suite drives
% prefer_wam(true) only. JS lowering-refusal tests are N/A here.
%
% Probes whose shapes wam_go cannot compile are recorded in
% cut_probe_refused/2 and must FAIL TO COMPILE LOUDLY — never emit
% wrong solutions. Currently every probe compiles.
%
%   swipl -q -g run_tests -t halt tests/test_wam_go_cut_semantics.pl

:- module(test_wam_go_cut_semantics,
          [test_wam_go_cut_semantics/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_go_target',
              [write_wam_go_project/3]).

% ---------------------------------------------------------------------
% The probe program. Same clause terms as the JS suite.
% ---------------------------------------------------------------------

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

% Shapes wam_go cannot compile: Name-Reason. Empty means all 35 compile.
:- dynamic cut_probe_refused/2.

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
    forall(cut_probe(N, _),
           ( atom_concat(N, '_t', TName),
             TG =.. [TName, R],
             assertz(user:(N :- ( TG, write(R), nl, fail ; true ))) )).

cut_probe_preds(Preds) :-
    findall(user:PI, (cut_probe_clause(C), cut_probe_head(C, PI)), P0),
    findall(user:(N/0), cut_probe(N, _), P1),
    append(P0, P1, P2),
    sort(P2, Preds).

go_cap(Atom, Cap) :-
    atom_codes(Atom, [F|R]),
    code_type(U, to_upper(F)),
    atom_codes(Cap, [U|R]).

swi_probe_output(N, Out) :-
    with_output_to(string(Raw),
                   ( catch(user:N, _, true) -> true ; true )),
    normalize_output(Raw, Out).

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

go_available :-
    catch(
        ( process_create(path(go), ['version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

cut_probe_go_main(Src) :-
    findall(Case,
            (   cut_probe(N, _),
                \+ cut_probe_refused(N, _),
                go_cap(N, Cap),
                format(string(Case),
'	case "~w":
		run(wam.~wCode, wam.~wLabels, wam.~wStartPC)
', [N, Cap, Cap, Cap])
            ), Cases),
    atomic_list_concat(Cases, CasesTxt),
    format(string(Src),
'package main

import (
	"fmt"
	"os"
	wam "cutsem"
)

func run(code []wam.Instruction, labels map[string]int, pc int) {
	vm := wam.NewWamState(code, labels)
	vm.PC = pc
	_ = vm.Run()
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: runprobe pNN")
		os.Exit(2)
	}
	switch os.Args[1] {
~w	default:
		fmt.Fprintln(os.Stderr, "unknown probe", os.Args[1])
		os.Exit(2)
	}
}
', [CasesTxt]).

go_probe_output(Bin, N, Out) :-
    process_create(Bin, [N],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1),
    read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    atomic_list_concat([S1, S2], Raw),
    normalize_output(Raw, Out).

compile_cut_probes(Dir, Bin) :-
    cut_probe_preds(Preds),
    Dir = 'output/test_wam_go_cut_semantics_gen',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_go_project(Preds,
                         [prefer_wam(true),
                          module_name(cutsem),
                          package_name(wam)],
                         Dir),
    directory_file_path(Dir, 'lib.go', LibPath),
    read_file_to_string(LibPath, Lib, []),
    (   sub_string(Lib, _, _, _, ': compilation failed')
    ->  format(user_error,
               'CUT PROBE: a predicate failed to compile (lib.go contains "compilation failed")~n',
               []),
        fail
    ;   true
    ),
    directory_file_path(Dir, 'cmd', CmdDir),
    directory_file_path(CmdDir, 'run', RunDir),
    make_directory_path(RunDir),
    directory_file_path(RunDir, 'main.go', MainPath),
    cut_probe_go_main(MainSrc),
    setup_call_cleanup(
        open(MainPath, write, MS),
        write(MS, MainSrc),
        close(MS)),
    directory_file_path(Dir, 'go.mod', GoModPath),
    read_file_to_string(GoModPath, GoModOld, []),
    atomic_list_concat([GoModOld, "\nreplace cutsem => ../../\n"], GoModNew),
    setup_call_cleanup(
        open(GoModPath, write, GS),
        write(GS, GoModNew),
        close(GS)),
    directory_file_path(RunDir, 'runprobe', Bin),
    format(atom(BuildCmd), 'cd ~w && go build -o runprobe .', [RunDir]),
    process_create(path(sh), ['-c', BuildCmd],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutStr),
    read_string(Err, _, ErrStr),
    close(Out), close(Err),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, '~n[go build failed]~n~w~w~n', [OutStr, ErrStr]),
        throw(go_build_failed(Status))
    ).

test_wam_go_cut_semantics :-
    run_tests(wam_go_cut_semantics).

:- begin_tests(wam_go_cut_semantics, [condition(go_available)]).

test(prefer_wam_matches_swi, [setup(install_cut_probes)]) :-
    compile_cut_probes(Dir, Bin),
    findall(N, cut_probe(N, _), All),
    findall(N, cut_probe_refused(N, _), Refused),
    length(All, NAll),
    length(Refused, NRef),
    NRun is NAll - NRef,
    format(user_error,
           'cut-semantics: ~w probes (~w run vs SWI, ~w refused-loudly)~n',
           [NAll, NRun, NRef]),
    forall(cut_probe_refused(N, Why),
           format(user_error, '  REFUSED ~w: ~w~n', [N, Why])),
    findall(fail(N, Ctx, Swi, Go),
            (   cut_probe(N, Ctx),
                \+ cut_probe_refused(N, _),
                swi_probe_output(N, Swi),
                go_probe_output(Bin, N, Go),
                Swi \== Go
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, G), Fails),
               format(user_error,
                      'CUT PROBE DIVERGENCE ~w (~w)~n  swi: ~q~n  go : ~q~n',
                      [N, Ctx, S, G])),
        fail
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_go_cut_semantics).
