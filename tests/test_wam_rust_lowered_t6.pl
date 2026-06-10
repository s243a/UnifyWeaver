% test_wam_rust_lowered_t6.pl
%
% End-to-end execution test for the Rust T6 lowering — "first-argument
% indexing / native match" (lowering type T6 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md).
%
% T6 shares the wam_clause_chain front-end with T5 but, when the
% discriminators are all atoms AND the clause count meets the gate
% (`t6_min_clauses`, default 8), emits a native two-stage `match` — a string
% switch mapping the first arg's atom to a Copy selector index, then an
% integer jump table dispatching to the clause remainder — instead of the T5
% if-cascade. For many clauses this is ~2.6-12.7x over the linear cascade
% (measured, generated code; see docs/reports/wam_rust_dispatch_alloc_perf.md);
% for few clauses the gate declines (the compiler flattens the cascade to the
% same code), so T6 only fires above the threshold.
%
% Pins:
%   * shade/1 (12 atom facts) — must lower as T6, and dispatch correctly for a
%     first clause, a middle clause, the last clause, a no-match atom, and an
%     unbound / non-atom first argument (both defer/fail -> false).
%   * grade/2 (12 atom RULE clauses, each remainder runs an is/2 builtin) —
%     T6 with a non-trivial remainder.
%   * few/1 (3 atom facts) — below the gate, must STAY T5 (the cascade).
%
% Skipped automatically when `cargo` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/wam_rust_lowered_emitter').

:- dynamic user:shade/1.
:- dynamic user:grade/2.
:- dynamic user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10). user:shade(s11). user:shade(s12).

user:grade(g01, R) :- R is 1 + 0.
user:grade(g02, R) :- R is 1 + 1.
user:grade(g03, R) :- R is 1 + 2.
user:grade(g04, R) :- R is 1 + 3.
user:grade(g05, R) :- R is 1 + 4.
user:grade(g06, R) :- R is 1 + 5.
user:grade(g07, R) :- R is 1 + 6.
user:grade(g08, R) :- R is 1 + 7.
user:grade(g09, R) :- R is 1 + 8.
user:grade(g10, R) :- R is 1 + 9.
user:grade(g11, R) :- R is 1 + 10.
user:grade(g12, R) :- R is 1 + 11.

user:few(a). user:few(b). user:few(c).

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_rust_lowered_t6, [condition(cargo_available)]).

% The many-clause atom predicates lower as T6 (clause_chain front-end + the
% gated native-match back-end); the few-clause one stays T5.
test(gate_picks_clause_chain) :-
    forall(member(PI, [shade/1, grade/2, few/1]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_rust_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

% The emitted source must use the native match for the many-clause atom
% predicates and the cascade for the few-clause one.
test(emits_t6_for_many_t5_for_few) :-
    Dir = 'output/test_wam_rust_t6_shape',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_rust_project(
        [user:shade/1, user:grade/2, user:few/1],
        [module_name('t6shape'), wam_fallback(true), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/src/lib.rs'], LibPath),
    read_file_to_string(LibPath, Src, []),
    assertion(sub_string(Src, _, _, _, "lowered_shade_1") ),
    assertion(sub_string(Src, _, _, _, "(T6 first-argument indexing / native match)") ),
    assertion(sub_string(Src, _, _, _, "lowered_grade_2") ),
    % few/1 (3 clauses, below gate) must stay the T5 cascade.
    assertion(sub_string(Src, _, _, _, "lowered_few_1 — lowered from few/1 (T5 first-argument dispatch)") ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

test(t6_exec_parity) :-
    Dir = 'output/test_wam_rust_t6_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_rust_project(
        [user:shade/1, user:grade/2],
        [module_name('t6rust'), wam_fallback(true), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/tests'], TestsDir),
    make_directory_path(TestsDir),
    atomic_list_concat([Dir, '/tests/t6_exec.rs'], TestPath),
    rust_t6_source(RustSrc),
    write_file_t6(TestPath, RustSrc),
    format(atom(TestCmd), 'cd ~w && cargo test --test t6_exec 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo test output]~n~w~n", [OutStr]),
        throw(rust_t6_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_rust_lowered_t6).

write_file_t6(Path, Text) :-
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Text), close(S)).

% Calls each lowered function with the first arg preloaded and asserts the
% boolean outcome: first / middle / last clause hit, a no-match atom, a
% non-atom value, and an unbound register (the last two defer/fail -> false).
rust_t6_source(
"use t6rust::state::WamState;
use t6rust::value::Value;
use t6rust::{lowered_shade_1, lowered_grade_2};
use std::collections::HashMap;

fn call(setup: &dyn Fn(&mut WamState), f: fn(&mut WamState) -> bool) -> bool {
    let mut vm = WamState::new(vec![], HashMap::new());
    setup(&mut vm);
    f(&mut vm)
}
fn i(n: i64) -> Value { Value::Integer(n) }
fn a(s: &str) -> Value { Value::Atom(s.to_string()) }

#[test]
fn t6_parity() {
    let cases: Vec<(&str, bool, bool)> = vec![
        (\"shade(s01)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"s01\")); }, lowered_shade_1), true),
        (\"shade(s07)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"s07\")); }, lowered_shade_1), true),
        (\"shade(s12)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"s12\")); }, lowered_shade_1), true),
        (\"shade(s99)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"s99\")); }, lowered_shade_1), false),
        (\"shade(42)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(42)); },     lowered_shade_1), false),
        (\"shade(_)\",    call(&|_vm: &mut WamState| {},                               lowered_shade_1), false),
        (\"grade(g01,1)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"g01\")); vm.set_reg(\"A2\", i(1)); },  lowered_grade_2), true),
        (\"grade(g06,6)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"g06\")); vm.set_reg(\"A2\", i(6)); },  lowered_grade_2), true),
        (\"grade(g12,12)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"g12\")); vm.set_reg(\"A2\", i(12)); }, lowered_grade_2), true),
        (\"grade(g06,9)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"g06\")); vm.set_reg(\"A2\", i(9)); },  lowered_grade_2), false),
        (\"grade(gxx,1)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"gxx\")); vm.set_reg(\"A2\", i(1)); },  lowered_grade_2), false),
    ];
    let mut failures = Vec::new();
    for (name, got, want) in cases {
        if got != want { failures.push(format!(\"{}: got {}, want {}\", name, got, want)); }
    }
    assert!(failures.is_empty(), \"failures: {:?}\", failures);
}
").
