% test_wam_rust_lowered_t4.pl
%
% End-to-end execution test for the Rust T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from Scala.
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers (emit_mode(functions)) to ALL clauses inline: each
% clause is an immediately-invoked closure, tried in order with a
% trail/register/heap/stack restore (vm.lo_restore_clause) between attempts.
% The first clause that succeeds wins (first-solution / deterministic-prefix
% semantics); the function never returns to the interpreter for clauses 2+,
% unlike the multi_clause_1 (clause-1 only) shape.
%
% Pins (the cases preload a BOUND first arg; the payoff is the non-first
% clauses running natively):
%   * grade/2 — fact chain with a REPEATED first arg (alice in clauses 1 & 3),
%               so not a distinct-first-arg chain; grade(alice,c) needs clause
%               3, grade(bob,b) needs clause 2;
%   * rel/2   — RULE chain with a VARIABLE first arg (=/2 body); rel(q,two)
%               needs clause 2 (and clause 1 clobbers A2 via put_constant,
%               which the restore must undo).
%
% Skipped automatically when `cargo` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/wam_rust_lowered_emitter').

:- dynamic user:grade/2.
:- dynamic user:rel/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_rust_lowered_t4, [condition(cargo_available)]).

% Both predicates must lower as T4 (multi_clause_n), not multi_clause_1.
test(gate_picks_multi_clause_n) :-
    forall(member(PI, [grade/2, rel/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_rust_lowerable(PI, W, Reason),
             assertion(Reason == multi_clause_n) )).

test(t4_exec_parity) :-
    Dir = 'output/test_wam_rust_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_rust_project(
        [user:grade/2, user:rel/2],
        [module_name('t4rust'), wam_fallback(true), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/src/lib.rs'], LibPath),
    ( exists_file(LibPath) -> read_file_to_string(LibPath, LibSrc, []) ; LibSrc = "" ),
    assertion(sub_string(LibSrc, _, _, _, "T4 all-clauses inline")),
    atomic_list_concat([Dir, '/tests'], TestsDir),
    make_directory_path(TestsDir),
    atomic_list_concat([Dir, '/tests/t4_exec.rs'], TestPath),
    rust_t4_source(RustSrc),
    write_file_t4(TestPath, RustSrc),
    format(atom(TestCmd), 'cd ~w && cargo test --test t4_exec 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo test output]~n~w~n", [OutStr]),
        throw(rust_t4_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_rust_lowered_t4).

write_file_t4(Path, Text) :-
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Text), close(S)).

% Calls each lowered function with the query arguments preloaded (first arg
% bound) and asserts the boolean outcome. Exercises the non-first clauses
% (grade clauses 2 & 3, rel clause 2) — the T4 payoff — plus no-match cases.
rust_t4_source(
"use t4rust::state::WamState;
use t4rust::value::Value;
use t4rust::{lowered_grade_2, lowered_rel_2};
use std::collections::HashMap;

fn call(setup: &dyn Fn(&mut WamState), f: fn(&mut WamState) -> bool) -> bool {
    let mut vm = WamState::new(vec![], HashMap::new());
    setup(&mut vm);
    f(&mut vm)
}
fn a(s: &str) -> Value { Value::Atom(s.to_string()) }

#[test]
fn t4_parity() {
    let cases: Vec<(&str, bool, bool)> = vec![
        (\"grade(alice,a)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"alice\")); vm.set_reg(\"A2\", a(\"a\")); }, lowered_grade_2), true),
        (\"grade(bob,b)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"bob\"));   vm.set_reg(\"A2\", a(\"b\")); }, lowered_grade_2), true),
        (\"grade(alice,c)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"alice\")); vm.set_reg(\"A2\", a(\"c\")); }, lowered_grade_2), true),
        (\"grade(alice,b)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"alice\")); vm.set_reg(\"A2\", a(\"b\")); }, lowered_grade_2), false),
        (\"grade(carol,a)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"carol\")); vm.set_reg(\"A2\", a(\"a\")); }, lowered_grade_2), false),
        (\"grade(bob,c)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"bob\"));   vm.set_reg(\"A2\", a(\"c\")); }, lowered_grade_2), false),
        (\"rel(p,one)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"p\")); vm.set_reg(\"A2\", a(\"one\")); }, lowered_rel_2), true),
        (\"rel(q,two)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"q\")); vm.set_reg(\"A2\", a(\"two\")); }, lowered_rel_2), true),
        (\"rel(p,two)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"p\")); vm.set_reg(\"A2\", a(\"two\")); }, lowered_rel_2), false),
        (\"rel(q,one)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"q\")); vm.set_reg(\"A2\", a(\"one\")); }, lowered_rel_2), false),
    ];
    let mut failures = Vec::new();
    for (name, got, want) in cases {
        if got != want { failures.push(format!(\"{}: got {}, want {}\", name, got, want)); }
    }
    assert!(failures.is_empty(), \"failures: {:?}\", failures);
}
").
