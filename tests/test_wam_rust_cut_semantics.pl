:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_cut_semantics.pl
%
% Cut and choice-point barrier conformance for the Rust WAM backend
% (docs/WAM_BACKEND_CONVENTIONS.md §9, "Cut is a barrier, never a stack
% wipe"). This is the Rust port of tests/test_wam_javascript_cut_semantics.pl
% — the SAME 35 probes, so a divergence found on one backend is directly
% comparable on the other.
%
% Every probe pNN is a failure-driven loop that prints ALL solutions of
% pNN_t/1. The same clauses run under SWI-Prolog (the oracle) and under the
% compiled Rust binary, and the two stdout streams must match exactly.
%
% Configurations exercised (Rust's tiers differ from JS's, so the mode names
% do too):
%
%   interpreter   -- default write_wam_rust_project: T9 fact tables and
%                    native lowering are live, ITE/cut-bearing predicates run
%                    on the WAM interpreter. This is the mixed tier: a
%                    natively-lowered / fact-table callee under an
%                    interpreted caller.
%   pure          -- fact_table_inline(false) + no_kernels(true): everything
%                    on the WAM interpreter.
%   functions     -- emit_mode(functions): the lowered emitter is offered
%                    every predicate it will accept.
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_cut_semantics.pl

:- module(test_wam_rust_cut_semantics,
          [test_wam_rust_cut_semantics/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

% ---------------------------------------------------------------------
% The probe program. ONE source of truth: the same clause terms are
% asserted into user: (so SWI runs them) and handed to the Rust WAM
% emitter (so the compiled binary runs them). There is no second copy
% to drift.
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

% ---------------------------------------------------------------------
% Running
% ---------------------------------------------------------------------

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

cut_probe_mode_options(interpreter, [emit_mode(interpreter)]).
cut_probe_mode_options(pure, [emit_mode(interpreter),
                              fact_table_inline(false),
                              no_kernels(true)]).
cut_probe_mode_options(functions, [emit_mode(functions)]).

compile_cut_probes(Mode, Dir) :-
    cut_probe_preds(Preds),
    cut_probe_mode_options(Mode, ModeOpts),
    format(atom(Dir), 'output/rust_wam_cut_semantics_~w', [Mode]),
    make_directory_path(Dir),
    append([module_name('cutprobe'), wam_fallback(true)], ModeOpts, Opts),
    write_wam_rust_project(Preds, Opts, Dir),
    write_probe_driver(Dir),
    format(atom(BuildCmd), 'cd ~w && cargo build --release --bin cut_probe 2>&1', [Dir]),
    process_create(path(sh), ['-c', BuildCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo build output]~n~w~n", [OutStr]),
        throw(rust_cut_probe_build_failed(Mode, Status))
    ).

% The driver runs one zero-arity predicate by label and lets the program's
% own write/1 do the printing. Nothing about cut semantics lives here.
%
% `--lowered pNN` instead calls the LOWERED Rust function for pNN_t/1
% directly, which is the only way to reach Rust's lowered tier: a lowered
% predicate keeps its WAM entry in the shared table, so an interpreted
% caller still goes through the interpreter. That mode is first-solution by
% the lowered tier's own contract, so it is oracled against `once/1`.
write_probe_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/cut_probe'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    probe_driver_source(Head, Tail),
    % Only emit an arm for a probe the emitter actually lowered -- the
    % symbol does not exist otherwise, and in interpreter modes none do.
    lowered_probe_names(Dir, Names),
    findall(Arm,
            ( member(N, Names),
              format(atom(Arm),
                     '        "~w" => run_lowered(&mut vm, cutprobe::lowered_~w_t_1),~n',
                     [N, N]) ),
            Arms),
    atomic_list_concat(Arms, ArmsText),
    setup_call_cleanup(open(Path, write, S),
                       format(S, '~w~w~w', [Head, ArmsText, Tail]),
                       close(S)).

%% lowered_probe_names(+Dir, -Names)
%  The pNN whose pNN_t/1 the lowered emitter accepted, read back from the
%  generated crate.
lowered_probe_names(Dir, Names) :-
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    (   exists_file(LibRs)
    ->  read_file_to_string(LibRs, S, []),
        findall(N,
                (   cut_probe(N, _),
                    format(atom(Sym), 'pub fn lowered_~w_t_1', [N]),
                    sub_string(S, _, _, _, Sym)
                ), Names)
    ;   Names = []
    ).

probe_driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// cut_probe driver: run one 0-arity predicate by its WAM label and let the
// compiled program's own write/1 produce the output compared against SWI.
#![allow(unused_imports)]
use cutprobe::state::WamState;
use cutprobe::value::Value;
use cutprobe::{setup_foreign_predicates, shared_wam_program};

const OUT: &str = \"_cut_probe_out\";

#[allow(dead_code)]
fn run_lowered(vm: &mut WamState, f: fn(&mut WamState) -> bool) {
    vm.reset_query();
    vm.cp = 0;
    vm.set_reg(\"A1\", Value::Unbound(OUT.to_string()));
    if f(vm) {
        // Read the answer from A1, falling back to the binding table: a
        // lowered `get_constant` may overwrite the register slot instead of
        // binding the variable that was in it, and a lowered list walk may
        // overwrite A1 with a recursive tail while still binding the
        // variable. Trying both covers each shape.
        let bound = vm.deref_heap(&Value::Unbound(OUT.to_string()));
        let out = if bound.is_unbound() {
            vm.get_reg(\"A1\").map(|v| vm.deref_heap(&v)).unwrap_or(bound)
        } else {
            bound
        };
        println!(\"{}\", out);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);

    if args.get(1).map(|s| s.as_str()) == Some(\"--lowered\") {
        let name = match args.get(2) {
            Some(n) => n.clone(),
            None => { eprintln!(\"usage: cut_probe --lowered <pNN>\"); std::process::exit(2); }
        };
        match name.as_str() {
",
"            other => { eprintln!(\"NO_LOWERED {}\", other); std::process::exit(3); }
        }
        return;
    }

    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: cut_probe <pred/arity>\"); std::process::exit(2); }
    };
    let target = match vm.labels.get(&key) {
        Some(pc) => *pc,
        None => { eprintln!(\"NO_LABEL {}\", key); std::process::exit(3); }
    };
    vm.reset_query();
    vm.cp = 0;
    vm.pc = target;
    let ok = vm.run();
    eprintln!(\"run={}\", ok);
}
").

swi_probe_output(N, Out) :-
    with_output_to(string(Raw),
                   ( catch(user:N, _, true) -> true ; true )),
    normalize_output(Raw, Out).

rust_probe_output(Dir, N, Out) :-
    atomic_list_concat([Dir, '/target/release/cut_probe'], Bin),
    format(atom(Key), '~w/0', [N]),
    process_create(Bin, [Key],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1),
    read_string(E, _, _S2),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    normalize_output(S1, Out).

normalize_output(Raw, Out) :-
    split_string(Raw, "\n", "\r", Lines0),
    exclude(cut_probe_noise_line, Lines0, Lines),
    atomic_list_concat(Lines, "\n", Joined),
    split_string(Joined, " \t", "", Parts),
    atomic_list_concat(Parts, '', Out).

cut_probe_noise_line("").

%% run_cut_probe_mode(+Mode)
%  Compile in Mode, then require an EXACT match with SWI for every probe.
run_cut_probe_mode(Mode) :-
    compile_cut_probes(Mode, Dir),
    findall(fail(N, Ctx, Swi, Rust),
            (   cut_probe(N, Ctx),
                swi_probe_output(N, Swi),
                rust_probe_output(Dir, N, Rust),
                Swi \== Rust
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, R), Fails),
               format(user_error,
                      'CUT PROBE DIVERGENCE [~w] ~w (~w)~n  swi : ~q~n  rust: ~q~n',
                      [Mode, N, Ctx, S, R])),
        length(Fails, NF),
        format(user_error, '[~w] ~w/35 probes diverged~n', [Mode, NF]),
        fail
    ).

% --- lowered tier -----------------------------------------------------
% A lowered Rust function returns through the host stack, so it is
% first-solution by construction (§9, "Lowered tiers are first-solution").
% The oracle is therefore SWI's `once/1`, and what the probe pins down is
% that the lowered function's single answer is the RIGHT one -- i.e. that
% the cut committed to the same clause SWI committed to -- and that the
% emitter declined to lower anything whose first answer would be wrong.

swi_once_output(N, Out) :-
    atom_concat(N, '_t', TName),
    TG =.. [TName, R],
    with_output_to(string(Raw),
                   ( catch(( user:TG -> write(R), nl ; true ), _, true) )),
    normalize_output(Raw, Out).

rust_lowered_output(Dir, N, Out) :-
    atomic_list_concat([Dir, '/target/release/cut_probe'], Bin),
    process_create(Bin, ['--lowered', N],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1),
    read_string(E, _, _S2),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    normalize_output(S1, Out).

run_lowered_probes(Dir) :-
    lowered_probe_names(Dir, Names),
    length(Names, NL),
    format(user_error, '[lowered] ~w/35 probe entry predicates were lowered~n', [NL]),
    findall(fail(N, Ctx, Swi, Rust),
            (   member(N, Names),
                cut_probe(N, Ctx),
                swi_once_output(N, Swi),
                rust_lowered_output(Dir, N, Rust),
                Swi \== Rust
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, R), Fails),
               format(user_error,
                      'LOWERED CUT PROBE DIVERGENCE ~w (~w)~n  swi once: ~q~n  rust    : ~q~n',
                      [N, Ctx, S, R])),
        length(Fails, NF),
        format(user_error, '[lowered] ~w/35 probes diverged~n', [NF]),
        fail
    ).

test_wam_rust_cut_semantics :-
    run_tests(rust_wam_cut_semantics).

:- begin_tests(rust_wam_cut_semantics, [condition(cargo_available)]).

test(interpreter_matches_swi, [setup(install_cut_probes)]) :-
    once(run_cut_probe_mode(interpreter)).

test(pure_interpreter_matches_swi, [setup(install_cut_probes)]) :-
    once(run_cut_probe_mode(pure)).

test(functions_matches_swi, [setup(install_cut_probes)]) :-
    once(run_cut_probe_mode(functions)).

% Lowered tier: call the generated Rust function for each pNN_t/1 directly.
% Oracled against once/1 -- the lowered tier is first-solution by contract.
test(lowered_first_solution_matches_swi, [setup(install_cut_probes)]) :-
    once(( compile_cut_probes(functions, Dir), run_lowered_probes(Dir) )).

:- end_tests(rust_wam_cut_semantics).
