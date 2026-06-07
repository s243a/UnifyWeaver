% test_wam_rust_lowered_ite_exec.pl
%
% End-to-end execution test for Rust if-then-else / negation / once
% lowering (emit_mode(functions)).
%
% Generates a WAM Rust project for a set of ITE predicates with the lowered
% emitter enabled, compiles it with the real `cargo` toolchain, and runs a
% Rust integration test that calls each lowered function and asserts the
% boolean result is correct. Counterpart to the Go test
% test_wam_go_lowered_ite_exec.pl. Pins:
%
%   * sequential ITEs   — rseqite(10,pos,small) must be false;
%   * nested ITEs       — rnestite/2;
%   * negation (\+)     — rneg/1 (commits with !/0, not cut_ite);
%   * binding condition — rundoite/2, whose condition binds a fresh var then
%                         fails; Rust's get_reg derefs through the binding
%                         table, so unwind_trail_to alone restores it.
%
% Before this fix the lowered emitter no-op'd try_me_else/cut_ite/jump/
% trust_me, emitting the condition + then + else as one flat conjunction
% (unifying the output with both branch values), so the function always
% failed. Skipped automatically when `cargo` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_rust_target').

:- dynamic user:rite/2.
:- dynamic user:rneg/1.
:- dynamic user:rseqite/3.
:- dynamic user:rnestite/2.
:- dynamic user:rundoite/2.

user:rite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:rneg(X)          :- \+ X > 0.
user:rseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:rnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).
user:rundoite(X, R)   :- ( (Y = a, Y = b) -> R = then ; R = els ), X = Y.

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

:- begin_tests(wam_rust_lowered_ite_exec, [condition(cargo_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_rust_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM Rust project with the lowered emitter enabled.
    write_wam_rust_project(
        [user:rite/2, user:rneg/1, user:rseqite/3, user:rnestite/2, user:rundoite/2],
        [module_name('iterust'), wam_fallback(true), emit_mode(functions)], Dir),
    % 2. Integration test source. `cargo test` also builds the bench bin
    %    (src/main.rs), so this run additionally guards that the whole
    %    generated project compiles, not just the lowered library.
    atomic_list_concat([Dir, '/tests'], TestsDir),
    make_directory_path(TestsDir),
    atomic_list_concat([Dir, '/tests/ite_exec.rs'], TestPath),
    rust_test_source(RustSrc),
    write_file(TestPath, RustSrc),
    % 3. Compile (lib + bench bin + test) and run the integration test.
    format(atom(TestCmd), 'cd ~w && cargo test --test ite_exec 2>&1', [Dir]),
    process_create(path(sh), ['-c', TestCmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0)
    ->  true
    ;   format(user_error, "~n[cargo test output]~n~w~n", [OutStr]),
        throw(rust_test_failed(Status))
    ).

:- end_tests(wam_rust_lowered_ite_exec).

write_file(Path, Text) :-
    setup_call_cleanup(open(Path, write, S), write(S, Text), close(S)).

% Calls each lowered function with the query arguments preloaded into the
% A-registers and asserts the boolean outcome. rseqite(10,pos,small)=false
% and rundoite(c,els)=true are the discriminating cases.
rust_test_source(
"use wam_lib::state::WamState;
use wam_lib::value::Value;
use wam_lib::{lowered_rite_2, lowered_rneg_1, lowered_rseqite_3, lowered_rnestite_2, lowered_rundoite_2};
use std::collections::HashMap;

fn call(setup: &dyn Fn(&mut WamState), f: fn(&mut WamState) -> bool) -> bool {
    let mut vm = WamState::new(vec![], HashMap::new());
    setup(&mut vm);
    f(&mut vm)
}
fn i(n: i64) -> Value { Value::Integer(n) }
fn a(s: &str) -> Value { Value::Atom(s.to_string()) }

#[test]
fn ite_parity() {
    let cases: Vec<(&str, bool, bool)> = vec![
        (\"rite(5,pos)\",    call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(5));  vm.set_reg(\"A2\", a(\"pos\")); },    lowered_rite_2), true),
        (\"rite(5,nonpos)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(5));  vm.set_reg(\"A2\", a(\"nonpos\")); }, lowered_rite_2), false),
        (\"rite(-1,nonpos)\",call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(-1)); vm.set_reg(\"A2\", a(\"nonpos\")); }, lowered_rite_2), true),
        (\"rite(-1,pos)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(-1)); vm.set_reg(\"A2\", a(\"pos\")); },    lowered_rite_2), false),
        (\"rneg(5)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(5)); },  lowered_rneg_1), false),
        (\"rneg(-1)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(-1)); }, lowered_rneg_1), true),
        (\"rneg(0)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(0)); },  lowered_rneg_1), true),
        (\"rseqite(10,pos,big)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(10)); vm.set_reg(\"A2\", a(\"pos\")); vm.set_reg(\"A3\", a(\"big\")); },   lowered_rseqite_3), true),
        (\"rseqite(10,pos,small)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(10)); vm.set_reg(\"A2\", a(\"pos\")); vm.set_reg(\"A3\", a(\"small\")); }, lowered_rseqite_3), false),
        (\"rseqite(3,pos,small)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(3));  vm.set_reg(\"A2\", a(\"pos\")); vm.set_reg(\"A3\", a(\"small\")); }, lowered_rseqite_3), true),
        (\"rseqite(-1,nonpos,small)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(-1)); vm.set_reg(\"A2\", a(\"nonpos\")); vm.set_reg(\"A3\", a(\"small\")); }, lowered_rseqite_3), true),
        (\"rnestite(20,big)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(20)); vm.set_reg(\"A2\", a(\"big\")); },   lowered_rnestite_2), true),
        (\"rnestite(5,small)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(5));  vm.set_reg(\"A2\", a(\"small\")); }, lowered_rnestite_2), true),
        (\"rnestite(-1,neg)\",   call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(-1)); vm.set_reg(\"A2\", a(\"neg\")); },   lowered_rnestite_2), true),
        (\"rnestite(20,small)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", i(20)); vm.set_reg(\"A2\", a(\"small\")); }, lowered_rnestite_2), false),
        (\"rundoite(c,els)\",  call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"c\")); vm.set_reg(\"A2\", a(\"els\")); },  lowered_rundoite_2), true),
        (\"rundoite(c,then)\", call(&|vm: &mut WamState| { vm.set_reg(\"A1\", a(\"c\")); vm.set_reg(\"A2\", a(\"then\")); }, lowered_rundoite_2), false),
    ];
    let mut failures = Vec::new();
    for (name, got, want) in cases {
        if got != want { failures.push(format!(\"{}: got {}, want {}\", name, got, want)); }
    }
    assert!(failures.is_empty(), \"failures: {:?}\", failures);
}
").
