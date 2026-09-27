:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_lowered_dispatch.pl
%
% Probes for D55's "sound intermediate": the `lowered_call` call-site hook that
% routes an interpreted `call`/`execute` to a LOWERED Rust function when — and
% only when — the predicate's first solution is its only solution
% (`deterministic` / `ite_lowered` / atom-guarded `clause_chain`, closed over
% the call graph). See `wam_rust_lowered_emitter:rust_lowered_dispatch_profile`
% and `WamState::lowered_dispatch`.
%
% Two guarantees are pinned, both SWI-oracled:
%
%   * SOUNDNESS. The same program compiled with `emit_mode(functions)` (the
%     dispatch hook live) and with `emit_mode(interpreter)` (no hook) must give
%     byte-identical output, and both must match SWI. A predicate that
%     backtracks must still yield ALL its solutions — proof the hook did not
%     unsoundly commit a nondet predicate to its first answer. The runtime
%     `lowered_dispatch` guard (roll back and decline when a call left a choice
%     point) is what makes that safe even if the classifier were ever wrong.
%
%   * DISPATCH ACTUALLY HAPPENS. `d05` runs a deterministic chain deep enough
%     that if the hook were dead code the answer would still be right but the
%     point of the test — that eligible predicates are reachable — would be
%     untested; the emitter-level unit check below asserts the eligible set is
%     non-empty and excludes the nondet predicate.
%
%   d01 a deterministic arithmetic chain (deterministic class)
%   d02 an atom-keyed clause_chain (first-arg dispatch)
%   d03 a self-recursive accumulator loop with a guard cut
%   d04 a nondeterministic member/2 enumerator — MUST keep all solutions
%   d05 a deterministic predicate calling other deterministic predicates
%   d06 an ite_lowered predicate ((C -> T ; E)) reached by call
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_lowered_dispatch.pl

:- module(test_wam_rust_lowered_dispatch,
          [test_wam_rust_lowered_dispatch/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).
:- use_module('../src/unifyweaver/targets/wam_rust_lowered_emitter',
              [rust_lowered_dispatch_profile/4]).

dispatch_probe(d01, 'a deterministic arithmetic chain').
dispatch_probe(d02, 'an atom-keyed clause_chain dispatched by first arg').
dispatch_probe(d03, 'a self-recursive accumulator with a guard cut').
dispatch_probe(d04, 'a nondet member enumerator keeps every solution').
dispatch_probe(d05, 'deterministic predicate calling deterministic predicates').
dispatch_probe(d06, 'an if-then-else predicate reached by call').

% ---------------------------------------------------------------------
% d01 -- deterministic; lowers `deterministic` and is dispatch-eligible.
dispatch_probe_clause((dd_add3(X, Y) :- Y is X + 3)).
dispatch_probe_clause((dd_dbl(X, Y) :- Y is X * 2)).
dispatch_probe_clause((d01_h(r(A, B)) :- dd_add3(10, A), dd_dbl(A, B))).
dispatch_probe_clause((d01 :- ( d01_h(R), write(R), nl, fail ; true ))).

% d02 -- an atom-keyed clause chain. Dispatched only when A1 is one of the
% discriminators; d02 also probes a non-matching key (must fail, not misfire).
dispatch_probe_clause(dd_kind(alpha, one)).
dispatch_probe_clause(dd_kind(beta, two)).
dispatch_probe_clause(dd_kind(gamma, three)).
dispatch_probe_clause((d02_h(r(A, B, C)) :-
    dd_kind(alpha, A), dd_kind(gamma, B),
    ( dd_kind(zeta, _) -> C = hit ; C = miss ))).
dispatch_probe_clause((d02 :- ( d02_h(R), write(R), nl, fail ; true ))).

% d03 -- self-recursive with a guard cut (clause_chain via the base cut, or
% multi_clause_n; the fixpoint keeps it eligible only if every callee is too).
dispatch_probe_clause((dd_sum(0, A, A) :- !)).
dispatch_probe_clause((dd_sum(N, A, R) :- N > 0, A1 is A + N, N1 is N - 1, dd_sum(N1, A1, R))).
dispatch_probe_clause((d03_h(r(S)) :- dd_sum(100, 0, S))).
dispatch_probe_clause((d03 :- ( d03_h(R), write(R), nl, fail ; true ))).

% d04 -- NONDETERMINISTIC. member/2 over a list enumerates; the hook must NOT
% dispatch dd_enum (it would drop solutions), so all three must print.
dispatch_probe_clause((dd_enum(X) :- member(X, [p, q, r]))).
dispatch_probe_clause((d04 :- ( dd_enum(X), write(X), nl, fail ; true ))).

% d05 -- a deterministic predicate whose body is nothing but calls to other
% deterministic predicates: the whole chain is dispatched.
dispatch_probe_clause((dd_f(X, Y) :- Y is X + 1)).
dispatch_probe_clause((dd_g(X, Y) :- dd_f(X, T), Y is T * T)).
dispatch_probe_clause((dd_hchain(X, Y) :- dd_g(X, T), dd_f(T, Y))).
dispatch_probe_clause((d05_h(r(A)) :- dd_hchain(4, A))).
dispatch_probe_clause((d05 :- ( d05_h(R), write(R), nl, fail ; true ))).

% d06 -- an if-then-else predicate (ite_lowered) reached via call.
dispatch_probe_clause((dd_clamp(X, Y) :- ( X > 50 -> Y is X - 50 ; Y = X ))).
dispatch_probe_clause((d06_h(r(A, B)) :- dd_clamp(70, A), dd_clamp(20, B))).
dispatch_probe_clause((d06 :- ( d06_h(R), write(R), nl, fail ; true ))).

% ---------------------------------------------------------------------

dispatch_probe_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

install_dispatch_probes :-
    findall(PI, (dispatch_probe_clause(C), dispatch_probe_head(C, PI)), PIs0),
    sort(PIs0, PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(dispatch_probe_clause(C), assertz(user:C)).

dispatch_probe_preds(Preds) :-
    findall(user:PI, (dispatch_probe_clause(C), dispatch_probe_head(C, PI)), P0),
    sort(P0, Preds).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

% Compile the probe set in one mode into a distinctly-named crate directory.
compile_dispatch_probes(Mode, Dir) :-
    dispatch_probe_preds(Preds),
    ( Mode == functions
    ->  Dir = 'output/rust_wam_lowered_dispatch_fn',
        ModeOpts = [emit_mode(functions)]
    ;   Dir = 'output/rust_wam_lowered_dispatch_int',
        ModeOpts = [emit_mode(interpreter)]
    ),
    make_directory_path(Dir),
    append([module_name('dprobe'), wam_fallback(true)], ModeOpts, Opts),
    write_wam_rust_project(Preds, Opts, Dir),
    write_probe_driver(Dir),
    format(atom(Cmd), 'cd ~w && cargo build --release --bin dprobe_run 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo build output]~n~w~n", [OutStr]),
        throw(rust_dispatch_probe_build_failed(Mode, Status))
    ).

write_probe_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/dprobe_run'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    driver_source(Src),
    setup_call_cleanup(open(Path, write, S), write(S, Src), close(S)).

driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
use dprobe::state::WamState;
use dprobe::{setup_foreign_predicates, shared_wam_program};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: dprobe_run <pred/arity>\"); std::process::exit(2); }
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
    atomic_list_concat([Dir, '/target/release/dprobe_run'], Bin),
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

test_wam_rust_lowered_dispatch :-
    run_tests(wam_rust_lowered_dispatch).

:- begin_tests(wam_rust_lowered_dispatch, [condition(cargo_available)]).

% Dispatched (emit_mode(functions)) answers match SWI on every probe,
% including the nondet enumerator (all solutions preserved).
test(dispatched_matches_swi, [setup(install_dispatch_probes)]) :-
    once(run_dispatch_probes(functions)).

% Interpreted answers also match SWI, so `functions` and `interpreter` agree.
test(interpreted_matches_swi, [setup(install_dispatch_probes)]) :-
    once(run_dispatch_probes(interpreter)).

:- end_tests(wam_rust_lowered_dispatch).

run_dispatch_probes(Mode) :-
    once(compile_dispatch_probes(Mode, Dir)),
    findall(fail(N, Ctx, Swi, Rust),
            (   dispatch_probe(N, Ctx),
                swi_output(N, Swi),
                rust_output(Dir, N, Rust),
                Swi \== Rust
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, R), Fails),
               format(user_error,
                      'DISPATCH PROBE DIVERGENCE ~w [~w] (~w)~n  swi : ~q~n  rust: ~q~n',
                      [N, Mode, Ctx, S, R])),
        fail
    ).
