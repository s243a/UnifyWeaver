:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_maplist_predsort.pl
%
% P3 uw-resolve emits BuiltinCall maplist/N (is_v3/1) and predsort/3
% (cmp_ver/3). Pin the Rust runtime ports, empty-list maplist (atom []),
% user-comparator register hygiene (A3 must survive the less-func), and
% UW_WAM_WARN_UNKNOWN on an unresolved Call.
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_maplist_predsort.pl

:- module(test_wam_rust_maplist_predsort,
          [test_wam_rust_maplist_predsort/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

mp_probe(mp01, 'maplist/2 over v/3 succeeds; over deb/3 fails').
mp_probe(mp02, 'maplist/3 user goal binds the output list').
mp_probe(mp03, 'maplist/2 over the empty list (atom [])').
mp_probe(mp04, 'predsort/3 with compare/3').
mp_probe(mp05, 'predsort/3 with a user comparator that writes A1-A3').
mp_probe(mp06, 'predsort/3 then reverse/2 in an Allocate clause (env isolation)').

mp_probe_clause(is_v3(v(_, _, _))).
mp_probe_clause((inc1(X, Y) :- Y is X + 1)).
mp_probe_clause((cmp_n(<, A, B) :- A < B, !)).
mp_probe_clause((cmp_n(>, A, B) :- B < A, !)).
mp_probe_clause(cmp_n(=, _, _)).

mp_probe_clause((mp01_h(r(ok, fail)) :-
    maplist(is_v3, [v(1, 0, 0), v(2, 0, 0)]),
    \+ maplist(is_v3, [deb(0, [], [])]))).
mp_probe_clause((mp01 :- ( mp01_h(R), write(R), nl, fail ; true ))).

mp_probe_clause((mp02_h(r(L)) :- maplist(inc1, [1, 2, 3], L))).
mp_probe_clause((mp02 :- ( mp02_h(R), write(R), nl, fail ; true ))).

mp_probe_clause((mp03_h(r(yes)) :- maplist(is_v3, []))).
mp_probe_clause((mp03 :- ( mp03_h(R), write(R), nl, fail ; true ))).

mp_probe_clause((mp04_h(r(S)) :-
    predsort(compare, [v(2, 0, 0), v(1, 0, 0)], S))).
mp_probe_clause((mp04 :- ( mp04_h(R), write(R), nl, fail ; true ))).

mp_probe_clause((mp05_h(r(S)) :- predsort(cmp_n, [3, 1, 2], S))).
mp_probe_clause((mp05 :- ( mp05_h(R), write(R), nl, fail ; true ))).

% The sort_versions_desc shape: predsort binds an intermediate, then
% reverse/2, then the clause Deallocate. A nested run() that pops the
% caller Env makes reverse succeed and Deallocate fail.
mp_probe_clause((mp06_h(r(S)) :-
    predsort(cmp_n, [3, 1, 2], A),
    reverse(A, S))).
mp_probe_clause((mp06 :- ( mp06_h(R), write(R), nl, fail ; true ))).

mp_probe_clause((mp_unknown :- totally_missing_pred(x))).

mp_probe_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

install_mp_probes :-
    findall(PI, (mp_probe_clause(C), mp_probe_head(C, PI)), PIs0),
    sort(PIs0, PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(mp_probe_clause(C), assertz(user:C)).

mp_probe_preds(Preds) :-
    findall(user:PI, (mp_probe_clause(C), mp_probe_head(C, PI)), P0),
    sort(P0, Preds).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

compile_mp_probes(Dir) :-
    mp_probe_preds(Preds),
    Dir = 'output/rust_wam_maplist_predsort',
    make_directory_path(Dir),
    write_wam_rust_project(Preds,
        [module_name('mpprobe'), wam_fallback(true), emit_mode(interpreter)],
        Dir),
    write_probe_driver(Dir),
    format(atom(Cmd), 'cd ~w && cargo build --release --bin mpprobe_run 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo build output]~n~w~n", [OutStr]),
        throw(rust_maplist_predsort_build_failed(Status))
    ).

write_probe_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/mpprobe_run'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    driver_source(Src),
    setup_call_cleanup(open(Path, write, S), write(S, Src), close(S)).

driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
use mpprobe::state::WamState;
use mpprobe::{setup_foreign_predicates, shared_wam_program};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: mpprobe_run <pred/arity>\"); std::process::exit(2); }
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
    atomic_list_concat([Dir, '/target/release/mpprobe_run'], Bin),
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

test_wam_rust_maplist_predsort :-
    run_tests(wam_rust_maplist_predsort).

:- begin_tests(wam_rust_maplist_predsort, [condition(cargo_available)]).

test(maplist_predsort_match_swi, [setup(install_mp_probes)]) :-
    once(run_mp_probes).

test(warn_unknown_builtin, [setup(install_mp_probes)]) :-
    once(run_warn_unknown).

:- end_tests(wam_rust_maplist_predsort).

run_mp_probes :-
    once(compile_mp_probes(Dir)),
    findall(fail(N, Ctx, Swi, Rust),
            (   mp_probe(N, Ctx),
                swi_output(N, Swi),
                rust_output(Dir, N, Rust),
                Swi \== Rust
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, R), Fails),
               format(user_error,
                      'MAPLIST/PREDSORT PROBE DIVERGENCE ~w (~w)~n  swi : ~q~n  rust: ~q~n',
                      [N, Ctx, S, R])),
        fail
    ).

run_warn_unknown :-
    once(compile_mp_probes(Dir)),
    atomic_list_concat([Dir, '/target/release/mpprobe_run'], Bin),
    format(atom(Cmd), 'UW_WAM_WARN_UNKNOWN=1 "~w" mp_unknown/0', [Bin]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _),
    read_string(E, _, Err),
    close(O), close(E),
    process_wait(Pid, exit(_)),
    (   sub_string(Err, _, _, _, '[wam_rust]'),
        sub_string(Err, _, _, _, 'unresolved goal')
    ->  true
    ;   format(user_error, 'WARN-UNKNOWN expected [wam_rust] unresolved goal, got:~n~w~n', [Err]),
        fail
    ).
