:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_builtin_probes.pl
%
% Minimal SWI-oracled probes for the wam_rust runtime defects the uw-resolve
% whole-program exercise (examples/pkg_resolver/rust/) forced out. Each probe
% is the smallest program that reproduces one defect; every one of them was
% RED before the corresponding fix and is green after.
%
%   b01 empty list rejected by list builtins        (CONVENTIONS §1)
%   b02 zero-solution findall halted the machine    (silent wrong answer)
%   b03 ==/2 did not alias [] with the empty list   (CONVENTIONS §1)
%   b04 Execute of a runtime builtin silently failed (CONVENTIONS §7, gap A3)
%   b05 Call of a runtime builtin silently failed    (CONVENTIONS §7)
%   b06 call/1 was not implemented at all
%   b07 bagof/3 and setof/3 were never inlined nor implemented
%
% Every probe prints its answers with write/1; the compiled Rust binary's
% stdout must match SWI's byte for byte (whitespace-insensitive).
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_builtin_probes.pl

:- module(test_wam_rust_builtin_probes,
          [test_wam_rust_builtin_probes/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

builtin_probe(b01, 'empty list reaches list builtins as the atom []').
builtin_probe(b02, 'findall with zero solutions must not halt the machine').
builtin_probe(b03, '==/2 aliases the atom [] with the empty list').
builtin_probe(b04, 'Execute of a runtime-implemented builtin (last goal)').
builtin_probe(b05, 'Call of a runtime-implemented builtin (non-last goal)').
builtin_probe(b06, 'call/1 with a conjunction and a local cut').
builtin_probe(b07, 'bagof/3 and setof/3 inline into the aggregate frame').

builtin_probe_clause(bd(1)).
builtin_probe_clause(bd(3)).
builtin_probe_clause(bd(2)).

% b01: sort/2, keysort/2, sum_list/2, msort/2 and last/2 over the EMPTY list.
% `put_constant []` delivers the atom [], which these builtins used to reject
% outright, so `sort([], X)` FAILED and took the whole clause with it.
builtin_probe_clause((b01_h(r(A, B, C, D)) :-
    sort([], A), msort([], B), sum_list([], C), keysort([], D))).
builtin_probe_clause((b01 :- ( b01_h(R), write(R), nl, fail ; true ))).

% b02: a findall whose goal has no solutions. The aggregate frame recorded
% its continuation in EndAggregate, which never ran, so the finalisation
% jumped to PC 0 -- HALT, reported as success, dropping `write/1` entirely.
builtin_probe_clause((b02_h(r(L, M)) :-
    findall(X, (bd(X), X > 9), L),
    findall(Y, bd(Y), M))).
builtin_probe_clause((b02 :- ( b02_h(R), write(R), nl, fail ; true ))).

% b03: the empty list returned by a builtin is Value::List([]); the [] a
% clause writes is the atom. Raw structural equality made them different.
builtin_probe_clause((b03_h(yes) :- findall(X, (bd(X), X > 9), L), L == [])).
builtin_probe_clause(b03_h(no)).
builtin_probe_clause((b03 :- ( b03_h(R), write(R), nl, fail ; true ))).

% b04: a clause whose LAST goal is a builtin outside is_builtin_pred/2, so it
% is emitted as `execute <name>` with no label. Every one of these bound its
% outputs and then reported FAILURE.
builtin_probe_clause((b04_a(X, Y) :- sub_atom(X, 0, 3, _, Y))).
builtin_probe_clause((b04_b(L, S) :- sum_list(L, S))).
builtin_probe_clause((b04_c(X, Y) :- upcase_atom(X, Y))).
builtin_probe_clause((b04_d(L, R) :- last(L, R))).
builtin_probe_clause((b04_e(L, N, E) :- nth0(N, L, E))).
builtin_probe_clause((b04_h(r(A, B, C, D, E)) :-
    b04_a(abcdef, A),
    b04_b([1, 2, 3], B),
    b04_c(abc, C),
    b04_d([1, 2, 3], D),
    b04_e([a, b, c], 1, E))).
builtin_probe_clause((b04 :- ( b04_h(R), write(R), nl, fail ; true ))).

% b05: the same class in NON-last position (`call <name>`), which used to
% route only atomic/1 plus the three is_iso_meta_builtin names.
builtin_probe_clause((b05_h(r(A, B)) :-
    atom_length(hello, A),
    A > 0,
    upcase_atom(xy, B),
    B \== zz)).
builtin_probe_clause((b05 :- ( b05_h(R), write(R), nl, fail ; true ))).

% b06: call/1 did not exist -- `execute call/1` found no label and failed, so
% every clause ending in a meta-call was dead. The cut inside the call is
% local to it (CONVENTIONS §9, barrier-raising contexts).
builtin_probe_clause((b06_h(X) :- call((bd(X), !)))).
builtin_probe_clause(b06_h(9)).
builtin_probe_clause((b06_g(X) :- bd(X), call(!))).
builtin_probe_clause((b06 :-
    ( b06_h(R), write(h(R)), nl, fail ; true ),
    ( b06_g(S), write(g(S)), nl, fail ; true ))).

% b07: bagof/setof were never inlined into begin_aggregate/end_aggregate, so
% they arrived as `execute bagof/3` -- no label, no builtin, always false.
builtin_probe_clause((b07_h(r(A, B)) :- bagof(X, bd(X), A), setof(Y, bd(Y), B))).
builtin_probe_clause((b07_e(none) :- \+ bagof(X, (bd(X), X > 9), X))).
builtin_probe_clause((b07 :-
    ( b07_h(R), write(R), nl, fail ; true ),
    ( b07_e(E), write(E), nl, fail ; true ))).

% ---------------------------------------------------------------------

builtin_probe_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

install_builtin_probes :-
    findall(PI, (builtin_probe_clause(C), builtin_probe_head(C, PI)), PIs0),
    sort(PIs0, PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(builtin_probe_clause(C), assertz(user:C)).

builtin_probe_preds(Preds) :-
    findall(user:PI, (builtin_probe_clause(C), builtin_probe_head(C, PI)), P0),
    sort(P0, Preds).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

compile_builtin_probes(Dir) :-
    builtin_probe_preds(Preds),
    Dir = 'output/rust_wam_builtin_probes',
    make_directory_path(Dir),
    write_wam_rust_project(Preds,
        [module_name('bprobe'), wam_fallback(true), emit_mode(interpreter)],
        Dir),
    write_probe_driver(Dir),
    format(atom(Cmd), 'cd ~w && cargo build --release --bin bprobe_run 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo build output]~n~w~n", [OutStr]),
        throw(rust_builtin_probe_build_failed(Status))
    ).

write_probe_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/bprobe_run'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    driver_source(Src),
    setup_call_cleanup(open(Path, write, S), write(S, Src), close(S)).

driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
use bprobe::state::WamState;
use bprobe::{setup_foreign_predicates, shared_wam_program};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: bprobe_run <pred/arity>\"); std::process::exit(2); }
    };
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);
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

swi_output(N, Out) :-
    with_output_to(string(Raw),
                   ( catch(user:N, _, true) -> true ; true )),
    normalize(Raw, Out).

rust_output(Dir, N, Out) :-
    atomic_list_concat([Dir, '/target/release/bprobe_run'], Bin),
    format(atom(Key), '~w/0', [N]),
    process_create(Bin, [Key],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1),
    read_string(E, _, _),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    normalize(S1, Out).

normalize(Raw, Out) :-
    split_string(Raw, "\n", "\r", Lines0),
    exclude(==(""), Lines0, Lines),
    atomic_list_concat(Lines, "\n", Joined),
    split_string(Joined, " \t", "", Parts),
    atomic_list_concat(Parts, '', Out).

test_wam_rust_builtin_probes :-
    run_tests(wam_rust_builtin_probes).

:- begin_tests(wam_rust_builtin_probes, [condition(cargo_available)]).

test(probes_match_swi, [setup(install_builtin_probes)]) :-
    once(run_builtin_probes).

:- end_tests(wam_rust_builtin_probes).

run_builtin_probes :-
    compile_builtin_probes(Dir),
    findall(fail(N, Ctx, Swi, Rust),
            (   builtin_probe(N, Ctx),
                swi_output(N, Swi),
                rust_output(Dir, N, Rust),
                Swi \== Rust
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, R), Fails),
               format(user_error,
                      'BUILTIN PROBE DIVERGENCE ~w (~w)~n  swi : ~q~n  rust: ~q~n',
                      [N, Ctx, S, R])),
        fail
    ).
