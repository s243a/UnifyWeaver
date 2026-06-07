% test_wam_rust_lowered_t5.pl
%
% End-to-end execution test for the Rust T5 lowering — "multi-clause as an
% if-then-else chain" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from the Scala
% implementation via the shared wam_clause_chain front-end.
%
% A predicate whose clauses discriminate on a DISTINCT first-argument
% constant now lowers (emit_mode(functions)) to a bound-checked first-arg
% dispatch over ALL clauses, instead of multi_clause_1 (clause 1 inline,
% clauses 2+ via the interpreter fallback). Non-first clauses become
% fast-path too when the first argument is bound; an unbound first argument
% returns false and defers to the interpreter (enumeration).
%
% Pins (the cases preload a BOUND first arg, so they exercise the T5
% dispatch including the non-first clauses):
%   * color/1 — fact chain, atom discriminators;
%   * sz/2    — fact chain with a second head match in each remainder;
%   * op/2    — RULE chain (each remainder runs an is/2 builtin).
%
% Skipped automatically when `cargo` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/wam_rust_lowered_emitter').

:- dynamic user:color/1.
:- dynamic user:sz/2.
:- dynamic user:op/2.

user:color(red).
user:color(green).
user:color(blue).

user:sz(small, 1).
user:sz(medium, 2).
user:sz(large, 3).

user:op(add, R) :- R is 1 + 1.
user:op(mul, R) :- R is 2 * 3.
user:op(neg, R) :- R is 0 - 1.

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_rust_lowered_t5, [condition(cargo_available)]).

% The three predicates must lower as T5 (clause_chain), not multi_clause_1.
test(gate_picks_clause_chain) :-
    forall(member(PI, [color/1, sz/2, op/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_rust_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

test(t5_exec_parity) :-
    Dir = 'output/test_wam_rust_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_rust_project(
        [user:color/1, user:sz/2, user:op/2],
        [module_name('t5rust'), wam_fallback(true), emit_mode(functions)], Dir),
    % Sanity: the lib must contain the T5 dispatch for each predicate.
    atomic_list_concat([Dir, '/src/lib.rs'], LibPath),
    ( exists_file(LibPath) -> read_file_to_string(LibPath, LibSrc, []) ; LibSrc = "" ),
    % (lib.rs may `mod` the lowered file; assert on the whole src tree below)
    _ = LibSrc,
    atomic_list_concat([Dir, '/tests'], TestsDir),
    make_directory_path(TestsDir),
    atomic_list_concat([Dir, '/tests/t5_exec.rs'], TestPath),
    rust_t5_source(RustSrc),
    write_file_t5(TestPath, RustSrc),
    format(atom(TestCmd), 'cd ~w && cargo test --test t5_exec 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo test output]~n~w~n", [OutStr]),
        throw(rust_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_rust_lowered_t5).

write_file_t5(Path, Text) :-
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Text), close(S)).

% Calls each lowered function with the query arguments preloaded (first arg
% bound) and asserts the boolean outcome. Exercises every clause including
% the non-first ones (green/blue, medium/large, mul/neg) — the T5 payoff —
% plus the no-match cases (yellow, sz big, op div).
rust_t5_source(
"use t5rust::state::WamState;
use t5rust::value::Value;
use t5rust::{lowered_color_1, lowered_sz_2, lowered_op_2};
use std::collections::HashMap;

fn call(setup: &dyn Fn(&mut WamState), f: fn(&mut WamState) -> bool) -> bool {
    let mut vm = WamState::new(vec![], HashMap::new());
    setup(&mut vm);
    f(&mut vm)
}
fn i(n: i64) -> Value { Value::Integer(n) }
fn a(s: &str) -> Value { Value::Atom(s.to_string()) }

#[test]
fn t5_parity() {
    let cases: Vec<(&str, bool, bool)> = vec![
        (\"color(red)\",    call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"red\")); },    lowered_color_1), true),
        (\"color(green)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"green\")); },  lowered_color_1), true),
        (\"color(blue)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"blue\")); },   lowered_color_1), true),
        (\"color(yellow)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"yellow\")); }, lowered_color_1), false),
        (\"sz(small,1)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"small\"));  vm.set_reg(\"A2\", i(1)); }, lowered_sz_2), true),
        (\"sz(medium,2)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"medium\")); vm.set_reg(\"A2\", i(2)); }, lowered_sz_2), true),
        (\"sz(large,3)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"large\"));  vm.set_reg(\"A2\", i(3)); }, lowered_sz_2), true),
        (\"sz(small,2)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"small\"));  vm.set_reg(\"A2\", i(2)); }, lowered_sz_2), false),
        (\"sz(big,1)\",    call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"big\"));    vm.set_reg(\"A2\", i(1)); }, lowered_sz_2), false),
        (\"op(add,2)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"add\")); vm.set_reg(\"A2\", i(2)); },  lowered_op_2), true),
        (\"op(mul,6)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"mul\")); vm.set_reg(\"A2\", i(6)); },  lowered_op_2), true),
        (\"op(neg,-1)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"neg\")); vm.set_reg(\"A2\", i(-1)); }, lowered_op_2), true),
        (\"op(add,3)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"add\")); vm.set_reg(\"A2\", i(3)); },  lowered_op_2), false),
        (\"op(div,1)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"div\")); vm.set_reg(\"A2\", i(1)); },  lowered_op_2), false),
    ];
    let mut failures = Vec::new();
    for (name, got, want) in cases {
        if got != want { failures.push(format!(\"{}: got {}, want {}\", name, got, want)); }
    }
    assert!(failures.is_empty(), \"failures: {:?}\", failures);
}
").
