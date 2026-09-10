:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_term_sharing.pl
%
% SWI-oracled probes for the STRUCTURAL SHARING representation of terms in the
% wam_rust runtime (templates/targets/rust_wam/value.rs.mustache): a compound's
% argument vector and a list's element vector are one refcounted spine, and a
% list TAIL is that same spine viewed one element further along
% (`Args { spine, off }`). Cloning a term is therefore O(1) and peeling a list
% allocates nothing.
%
% Sharing is only sound because terms in this runtime are IMMUTABLE — a
% variable binding lives in `WamState::bindings` keyed by the variable's name,
% never inside the term cell, and the trail restores bindings rather than
% cells. These probes pin exactly that: every way a program can observe whether
% two terms secretly share storage.
%
%   s01 peeling a list under backtracking must not disturb the original
%   s02 a tail bound to another variable, and a new list built on that tail
%   s03 bindings made on a FAILED branch are undone inside a shared term
%   s04 a variable inside a shared compound, observed before and after binding
%   s05 list builtins applied to a shared tail (length/msort/sort)
%   s06 nested lists: an element that is itself a shared list
%   s07 a register holding a shared term across choice points
%   s08 append/nth0/last/reverse over a shared tail
%   s09 two lists consed onto ONE shared tail
%
% Every probe prints its answers with write/1; the compiled Rust binary's
% stdout must match SWI's byte for byte (whitespace-insensitive).
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_term_sharing.pl

:- module(test_wam_rust_term_sharing,
          [test_wam_rust_term_sharing/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

sharing_probe(s01, 'peeling a list under backtracking leaves the original intact').
sharing_probe(s02, 'a tail bound to a variable, and a new list built on it').
sharing_probe(s03, 'bindings on a failed branch are undone inside a shared term').
sharing_probe(s04, 'a variable inside a shared compound before and after binding').
sharing_probe(s05, 'list builtins over a shared tail').
sharing_probe(s06, 'an element that is itself a shared list').
sharing_probe(s07, 'a register holding a shared term across choice points').
sharing_probe(s08, 'append/nth0/last/reverse over a shared tail').
sharing_probe(s09, 'two lists consed onto one shared tail').

% ---------------------------------------------------------------------
% s01 -- peeling. Each recursive step hands the callee the SAME spine one
% element along; if that view were ever mutated (or if backtracking restored
% the wrong one), the caller's L would change between solutions.
sharing_probe_clause(s01_pick([H|_], H)).
sharing_probe_clause((s01_pick([_|T], X) :- s01_pick(T, X))).
sharing_probe_clause((s01_h(r(L, X)) :- L = [a, b, c, d], s01_pick(L, X))).
sharing_probe_clause((s01 :- ( s01_h(R), write(R), nl, fail ; true ))).

% s02 -- the tail is handed out as a term in its own right and then used to
% build a longer list. The parent must not see the new head.
sharing_probe_clause((s02_h(r(A, B, C)) :-
    A = [1, 2, 3],
    A = [_|B],
    C = [0|B])).
sharing_probe_clause((s02 :- ( s02_h(R), write(R), nl, fail ; true ))).

% s03 -- TRAIL RESTORE. The first clause of s03_bind/1 binds the variable that
% lives inside the shared compound and then fails; the second must see it
% unbound again. With shared spines a stale binding would be visible through
% every alias of the spine at once.
sharing_probe_clause((s03_bind(X) :- X = 1, fail)).
sharing_probe_clause(s03_bind(_)).
sharing_probe_clause((s03_h(r(S, K)) :-
    T = f(A, kept),
    s03_bind(A),
    ( var(A) -> S = unbound ; S = A ),
    T = f(_, K))).
sharing_probe_clause((s03 :- ( s03_h(R), write(R), nl, fail ; true ))).

% s04 -- the deref fast path returns the SAME spine when nothing under it
% moved. Binding a variable inside the spine afterwards must still be visible
% through the term (the result is never cached).
%
% The argument is read back by head unification rather than by `arg/3`:
% `arg/3` over a body-constructed compound hands back the construction-time
% placeholder instead of the argument (`T = k(5,2), arg(1,T,V)` gives an
% unbound V), a PRE-EXISTING defect that reproduces identically on the
% deep-copy runtime and is recorded in docs/WAM_RUST_STATUS.md.
sharing_probe_clause((s04_h(r(S1, V)) :-
    T = k(X, 2),
    ( var(X) -> S1 = unbound ; S1 = bound ),
    X = 7,
    T = k(V, _))).
sharing_probe_clause((s04 :- ( s04_h(R), write(R), nl, fail ; true ))).

% s05 -- builtins take their list argument through deref_list_arg, which now
% hands back the shared spine instead of copying it.
sharing_probe_clause((s05_h(r(W, T, N, S)) :-
    W = [3, 1, 2],
    W = [_|T],
    length(T, N),
    msort(T, S))).
sharing_probe_clause((s05 :- ( s05_h(R), write(R), nl, fail ; true ))).

% s06 -- an element of a list is itself a list, so the outer spine holds a
% Value that owns an inner spine. Peeling the outer one must not reach in.
sharing_probe_clause((s06_h(r(L, Sub, All)) :-
    L = [[1, 2], [3, 4]],
    L = [Sub|_],
    findall(X, member(X, Sub), All))).
sharing_probe_clause((s06 :- ( s06_h(R), write(R), nl, fail ; true ))).

% s07 -- a choice point saves the registers; with sharing the saved register
% is a pointer into the same spine, so restoring must give back the same list
% and not a half-consumed view of it.
sharing_probe_clause(s07_copy([], [])).
sharing_probe_clause((s07_copy([H|T], [H|R]) :- s07_copy(T, R))).
sharing_probe_clause((s07_h(r(L, C, X)) :-
    L = [a, b, c],
    s07_copy(L, C),
    member(X, C))).
sharing_probe_clause((s07 :- ( s07_h(R), write(R), nl, fail ; true ))).

% s08 -- the list builtins that rebuild a spine (append/reverse) and those
% that only read one (nth0/last), all applied to a shared tail.
sharing_probe_clause((s08_h(r(T, A, N, Z, V)) :-
    L = [1, 2, 3, 4],
    L = [_|T],
    append(T, [9], A),
    nth0(0, T, N),
    last(T, Z),
    reverse(T, V))).
sharing_probe_clause((s08 :- ( s08_h(R), write(R), nl, fail ; true ))).

% s09 -- one tail, two different heads. Prepending copies precisely because
% the slot before the shared window may belong to another view.
sharing_probe_clause((s09_h(r(A, B, C)) :-
    A = [x, y],
    B = [p|A],
    C = [q|A])).
sharing_probe_clause((s09 :- ( s09_h(R), write(R), nl, fail ; true ))).

% ---------------------------------------------------------------------

sharing_probe_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

install_sharing_probes :-
    findall(PI, (sharing_probe_clause(C), sharing_probe_head(C, PI)), PIs0),
    sort(PIs0, PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(sharing_probe_clause(C), assertz(user:C)).

sharing_probe_preds(Preds) :-
    findall(user:PI, (sharing_probe_clause(C), sharing_probe_head(C, PI)), P0),
    sort(P0, Preds).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

compile_sharing_probes(Dir) :-
    sharing_probe_preds(Preds),
    Dir = 'output/rust_wam_term_sharing',
    make_directory_path(Dir),
    write_wam_rust_project(Preds,
        [module_name('sprobe'), wam_fallback(true), emit_mode(interpreter)],
        Dir),
    write_probe_driver(Dir),
    format(atom(Cmd), 'cd ~w && cargo build --release --bin sprobe_run 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo build output]~n~w~n", [OutStr]),
        throw(rust_sharing_probe_build_failed(Status))
    ).

write_probe_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/sprobe_run'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    driver_source(Src),
    setup_call_cleanup(open(Path, write, S), write(S, Src), close(S)).

driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
use sprobe::state::WamState;
use sprobe::{setup_foreign_predicates, shared_wam_program};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: sprobe_run <pred/arity>\"); std::process::exit(2); }
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
    atomic_list_concat([Dir, '/target/release/sprobe_run'], Bin),
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

test_wam_rust_term_sharing :-
    run_tests(wam_rust_term_sharing).

:- begin_tests(wam_rust_term_sharing, [condition(cargo_available)]).

test(sharing_probes_match_swi, [setup(install_sharing_probes)]) :-
    once(run_sharing_probes).

:- end_tests(wam_rust_term_sharing).

run_sharing_probes :-
    once(compile_sharing_probes(Dir)),
    findall(fail(N, Ctx, Swi, Rust),
            (   sharing_probe(N, Ctx),
                swi_output(N, Swi),
                rust_output(Dir, N, Rust),
                Swi \== Rust
            ), Fails),
    (   Fails == []
    ->  true
    ;   forall(member(fail(N, Ctx, S, R), Fails),
               format(user_error,
                      'SHARING PROBE DIVERGENCE ~w (~w)~n  swi : ~q~n  rust: ~q~n',
                      [N, Ctx, S, R])),
        fail
    ).
