:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

% Executable regression for transactional, correlated foreign-result tuples
% in the generated Rust WAM runtime.

:- use_module(library(filesex)).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_rust_target').

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _,
        fail).

:- begin_tests(wam_rust_foreign_tuple_aliases,
               [condition(cargo_available)]).

test(correlated_alias_candidates_are_transactional) :-
    Dir = 'output/test_wam_rust_foreign_tuple_aliases',
    setup_call_cleanup(
        prepare_rust_alias_project(Dir),
        run_cargo_alias_test(Dir),
        cleanup_test_dir(Dir)).

:- end_tests(wam_rust_foreign_tuple_aliases).

prepare_rust_alias_project(Dir) :-
    cleanup_test_dir(Dir),
    write_wam_rust_project([], [module_name(alias_runtime)], Dir),
    directory_file_path(Dir, tests, TestsDir),
    make_directory_path(TestsDir),
    directory_file_path(TestsDir, 'foreign_tuple_aliases.rs', TestPath),
    rust_alias_test_source(Source),
    setup_call_cleanup(
        open(TestPath, write, Stream, [encoding(utf8)]),
        format(Stream, '~s', [Source]),
        close(Stream)).

run_cargo_alias_test(Dir) :-
    absolute_file_name(Dir, AbsDir),
    process_create(path(cargo),
                   ['test', '--test', 'foreign_tuple_aliases', '--quiet'],
                   [cwd(AbsDir), stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, OutText),
    read_string(Err, _, ErrText),
    close(Out),
    close(Err),
    process_wait(Pid, Status),
    (   Status == exit(0)
    ->  true
    ;   throw(error(rust_alias_test_failed(Status, OutText, ErrText), _))
    ).

cleanup_test_dir(Dir) :-
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

rust_alias_test_source(
"use std::collections::HashMap;

use alias_runtime::state::WamState;
use alias_runtime::value::Value;

fn atom(name: &str) -> Value {
    Value::Atom(name.to_string())
}

fn tuple(left: &str, right: &str, distance: i64) -> Value {
    Value::Str(
        \"__tuple__\".to_string(),
        vec![atom(left), atom(right), Value::Integer(distance)],
    )
}

fn machine() -> (WamState, Value, Value) {
    let mut vm = WamState::new(Vec::new(), HashMap::new());
    vm.register_foreign_result_layout(\"alias/3\", \"tuple:3\");
    vm.register_foreign_result_mode(\"alias/3\", \"stream\");
    (
        vm,
        Value::Unbound(\"Shared\".to_string()),
        Value::Unbound(\"Distance\".to_string()),
    )
}

fn assert_atom(value: Value, expected: &str) {
    match value {
        Value::Atom(actual) => assert_eq!(actual, expected),
        other => panic!(\"expected atom {expected}, got {other:?}\"),
    }
}

fn assert_integer(value: Value, expected: i64) {
    match value {
        Value::Integer(actual) => assert_eq!(actual, expected),
        other => panic!(\"expected integer {expected}, got {other:?}\"),
    }
}

#[test]
fn skips_an_incompatible_first_tuple_without_leaking_it() {
    let (mut vm, shared, distance) = machine();
    let results = vec![tuple(\"left\", \"right\", 1), tuple(\"same\", \"same\", 2)];

    assert!(vm.finish_foreign_results(
        \"alias/3\",
        vec![shared.clone(), shared.clone(), distance.clone()],
        results,
    ));
    assert_atom(vm.deref_var(&shared), \"same\");
    assert_integer(vm.deref_var(&distance), 2);
    assert!(vm.choice_points.is_empty());
}

#[test]
fn retry_skips_an_incompatible_tuple_and_finds_a_later_match() {
    let (mut vm, shared, distance) = machine();
    let results = vec![
        tuple(\"first\", \"first\", 1),
        tuple(\"bad\", \"worse\", 2),
        tuple(\"second\", \"second\", 3),
    ];

    assert!(vm.finish_foreign_results(
        \"alias/3\",
        vec![shared.clone(), shared.clone(), distance.clone()],
        results,
    ));
    assert_atom(vm.deref_var(&shared), \"first\");
    assert_eq!(vm.choice_points.len(), 1);

    assert!(vm.backtrack());
    assert_atom(vm.deref_var(&shared), \"second\");
    assert_integer(vm.deref_var(&distance), 3);
    assert!(vm.choice_points.is_empty());
}

#[test]
fn exhausted_retry_restores_bindings_trail_and_choice_point() {
    let (mut vm, shared, distance) = machine();
    let results = vec![
        tuple(\"first\", \"first\", 1),
        tuple(\"bad\", \"worse\", 2),
    ];

    assert!(vm.finish_foreign_results(
        \"alias/3\",
        vec![shared.clone(), shared.clone(), distance.clone()],
        results,
    ));
    assert_eq!(vm.choice_points.len(), 1);

    assert!(!vm.backtrack());
    assert!(matches!(vm.deref_var(&shared), Value::Unbound(_)));
    assert!(matches!(vm.deref_var(&distance), Value::Unbound(_)));
    assert!(vm.bindings.is_empty());
    assert!(vm.trail.is_empty());
    assert!(vm.choice_points.is_empty());
}

#[test]
fn tspd5_emits_correlated_shortest_positive_diamond() {
    // Adversarial diamond: a→b,a→c,b→p,c→q,p→t,q→t.
    // Correlated pairs only — never Step×Parent cross-products
    // such as (t,b,q,3) / (t,c,p,3).
    let mut vm = WamState::new(Vec::new(), HashMap::new());
    vm.register_indexed_atom_fact2_pairs(\"edge\", &[
        (\"a\", \"b\"),
        (\"a\", \"c\"),
        (\"b\", \"p\"),
        (\"c\", \"q\"),
        (\"p\", \"t\"),
        (\"q\", \"t\"),
    ]);
    let mut out = Vec::new();
    vm.collect_native_transitive_step_parent_distance_results(\"a\", \"edge\", &mut out);
    out.sort();

    assert_eq!(out, vec![
        (\"b\".to_string(), \"b\".to_string(), \"a\".to_string(), 1),
        (\"c\".to_string(), \"c\".to_string(), \"a\".to_string(), 1),
        (\"p\".to_string(), \"b\".to_string(), \"b\".to_string(), 2),
        (\"q\".to_string(), \"c\".to_string(), \"c\".to_string(), 2),
        (\"t\".to_string(), \"b\".to_string(), \"p\".to_string(), 3),
        (\"t\".to_string(), \"c\".to_string(), \"q\".to_string(), 3),
    ]);
    assert!(!out.iter().any(|q| q == &(\"t\".to_string(), \"b\".to_string(), \"q\".to_string(), 3)));
    assert!(!out.iter().any(|q| q == &(\"t\".to_string(), \"c\".to_string(), \"p\".to_string(), 3)));
}
").
