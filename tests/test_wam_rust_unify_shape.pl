:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_unify_shape.pl
%
% SWI-oracled behaviour probes for the SHAPE-FIRST restructuring of
% `WamState::unify` (templates/targets/rust_wam/state.rs.mustache): the runtime
% now dispatches on the raw (undereferenced) cell shape, following only the
% variable-binding chain by reference, and materialises at most one level of a
% term per side before element-wise work. Deep traversal happens only when both
% sides are genuinely compound. The recorded quadratic (`deref_heap` walked
% BOTH arguments eagerly before the match) is gone; these probes pin that the
% observable ANSWERS are unchanged — the change is behaviour-preserving.
%
%   u01 an output variable unified against the suffix of a long list
%   u02 var-var unification, both directions, then a later binding
%   u03 compound-compound unification, nested, with shared sub-terms
%   u04 cons-spelling aliasing: [|]/2 built structure vs put_list list
%   u05 empty-list aliasing: the atom [] vs Value::List([])
%   u06 occurs-free deep chains: two N-cell lists unified element-wise
%   u07 partial list (unbound tail) unified, then the tail filled
%   u08 a long compound (arity via =..) unified argument-wise
%   u09 unify failure deep inside two otherwise-equal compounds
%
% The suffix/deep-chain probes (u01, u06) are the ones that were quadratic;
% they are kept small here (correctness, not timing — the timing curves live in
% the round's scratch harness and the status doc). Every probe prints its
% answers with write/1; the compiled Rust binary's stdout must match SWI's byte
% for byte (whitespace-insensitive).
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_unify_shape.pl

:- module(test_wam_rust_unify_shape,
          [test_wam_rust_unify_shape/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

unify_probe(u01, 'output variable unified against a long list suffix').
unify_probe(u02, 'var-var unification both directions, then a binding').
unify_probe(u03, 'nested compound-compound unification with shared sub-terms').
unify_probe(u04, 'cons-spelling aliasing between a built structure and a list').
unify_probe(u05, 'empty-list aliasing: the atom [] vs the empty list').
unify_probe(u06, 'two N-cell lists unified element-wise').
unify_probe(u07, 'a partial list unified, then its tail filled').
unify_probe(u08, 'a wide compound unified argument-wise').
unify_probe(u09, 'a unify failure deep inside two equal-shaped compounds').

% ---------------------------------------------------------------------
% u01 -- peel off a prefix, then unify a fresh output var against the
% remaining suffix. This is the exact shape the D59 write-up isolated as
% O(len)-per-step: `Rest = [X|More]` handing a suffix back through an output
% argument. The answer is just the suffix; the point is it is CORRECT (and now
% linear).
unify_probe_clause((u01_drop(0, L, L) :- !)).
unify_probe_clause((u01_drop(N, [_|T], R) :- N > 0, N1 is N - 1, u01_drop(N1, T, R))).
unify_probe_clause((u01_h(r(Suf)) :-
    numlist(1, 12, L),
    u01_drop(8, L, Suf))).
unify_probe_clause((u01 :- ( u01_h(R), write(R), nl, fail ; true ))).

% u02 -- X = Y (var-var), then bind through one end and read the other. Both
% assignment directions exercised.
unify_probe_clause((u02_h(r(A, B)) :-
    X = Y,          % var-var, n1 == n2 is false, binds one to the other
    Y = hello,
    A = X,
    P = Q,
    P = world,
    B = Q)).
unify_probe_clause((u02 :- ( u02_h(R), write(R), nl, fail ; true ))).

% u03 -- nested compounds; the inner term is shared between two arguments of
% the outer, so a spine-sharing runtime must still compare it correctly.
unify_probe_clause((u03_h(r(A, B, C)) :-
    Inner = point(3, 4),
    T1 = seg(Inner, Inner),
    T2 = seg(point(X, Y), point(3, 4)),
    T1 = T2,
    A = X, B = Y, C = T1)).
unify_probe_clause((u03 :- ( u03_h(R), write(R), nl, fail ; true ))).

% u04 -- a list built with `[H|T]` (which the compiler lowers through a
% cons structure) unified against one written as a flat list literal.
unify_probe_clause((u04_h(r(H, T, Ok)) :-
    L1 = [a, b, c, d],
    L2 = [H|T],
    ( L1 = L2 -> Ok = yes ; Ok = no ))).
unify_probe_clause((u04 :- ( u04_h(R), write(R), nl, fail ; true ))).

% u05 -- empty-list aliasing. E comes back from a builtin as Value::List([]);
% [] written in the clause is the atom. They must unify.
unify_probe_clause((u05_h(r(Ok1, Ok2)) :-
    findall(Z, (member(Z, [1,2,3]), Z > 9), E),
    ( E = [] -> Ok1 = yes ; Ok1 = no ),
    ( [] = E -> Ok2 = yes ; Ok2 = no ))).
unify_probe_clause((u05 :- ( u05_h(R), write(R), nl, fail ; true ))).

% u06 -- two independently-built N-cell lists unified element-wise. The spine
% walk is a loop; a wrong bound would drop or duplicate elements.
unify_probe_clause((u06_h(r(Ok, Sum)) :-
    numlist(1, 30, A),
    numlist(1, 30, B),
    ( A = B -> Ok = yes ; Ok = no ),
    sum_list(A, Sum))).
unify_probe_clause((u06 :- ( u06_h(R), write(R), nl, fail ; true ))).

% u07 -- a partial list [1,2,3|T] with T unbound is unified against
% [1,2,3,4,5]; T must bind to [4,5].
unify_probe_clause((u07_h(r(T)) :-
    P = [1, 2, 3|T],
    P = [1, 2, 3, 4, 5])).
unify_probe_clause((u07 :- ( u07_h(R), write(R), nl, fail ; true ))).

% u08 -- a wide compound assembled with =.. then unified argument-wise.
unify_probe_clause((u08_h(r(Ok, Args)) :-
    T1 =.. [f, 1, 2, 3, 4, 5, 6],
    T2 = f(A, B, C, D, E, F),
    ( T1 = T2 -> Ok = yes ; Ok = no ),
    Args = [A, B, C, D, E, F])).
unify_probe_clause((u08 :- ( u08_h(R), write(R), nl, fail ; true ))).

% u09 -- two compounds that agree everywhere except one deep leaf; unify must
% FAIL, and must not have left a partial binding visible.
unify_probe_clause((u09_h(r(Ok, X)) :-
    T1 = a(b(c(1, 2), d(X)), e),
    T2 = a(b(c(1, 9), d(7)), e),
    ( T1 = T2 -> Ok = unexpected_success ; Ok = failed_as_expected ),
    ( var(X) -> X = still_unbound ; true ))).
unify_probe_clause((u09 :- ( u09_h(R), write(R), nl, fail ; true ))).

% ---------------------------------------------------------------------

unify_probe_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

install_unify_probes :-
    findall(PI, (unify_probe_clause(C), unify_probe_head(C, PI)), PIs0),
    sort(PIs0, PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(unify_probe_clause(C), assertz(user:C)).

unify_probe_preds(Preds) :-
    findall(user:PI, (unify_probe_clause(C), unify_probe_head(C, PI)), P0),
    sort(P0, Preds).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

compile_unify_probes(Dir) :-
    unify_probe_preds(Preds),
    Dir = 'output/rust_wam_unify_shape',
    make_directory_path(Dir),
    write_wam_rust_project(Preds,
        [module_name('uprobe'), wam_fallback(true), emit_mode(interpreter)],
        Dir),
    write_probe_driver(Dir),
    format(atom(Cmd), 'cd ~w && cargo build --release --bin uprobe_run 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo build output]~n~w~n", [OutStr]),
        throw(rust_unify_probe_build_failed(Status))
    ).

write_probe_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/uprobe_run'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    driver_source(Src),
    setup_call_cleanup(open(Path, write, S), write(S, Src), close(S)).

driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
use uprobe::state::WamState;
use uprobe::{setup_foreign_predicates, shared_wam_program};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: uprobe_run <pred/arity>\"); std::process::exit(2); }
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
    atomic_list_concat([Dir, '/target/release/uprobe_run'], Bin),
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

test_wam_rust_unify_shape :-
    run_tests(wam_rust_unify_shape).

:- begin_tests(wam_rust_unify_shape, [condition(cargo_available)]).

test(unify_probes_match_swi, [setup(install_unify_probes)]) :-
    once(run_unify_probes).

:- end_tests(wam_rust_unify_shape).

run_unify_probes :-
    once(compile_unify_probes(Dir)),
    findall(fail(N, Ctx, Swi, Rust),
            (   unify_probe(N, Ctx),
                swi_output(N, Swi),
                rust_output(Dir, N, Rust),
                Swi \== Rust
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, R), Fails),
               format(user_error,
                      'UNIFY SHAPE PROBE DIVERGENCE ~w (~w)~n  swi : ~q~n  rust: ~q~n',
                      [N, Ctx, S, R])),
        fail
    ).
