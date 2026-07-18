:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_rust_builtin_parity.pl - Execution tests for the Rust WAM
% builtin parity sweep (F# parity campaign): term ordering, list
% utilities, atom/string text ops, integer relations, and the
% nondeterministic builtins (between/3, select/3, nth0/nth1 with
% unbound index, atom_concat/3 split mode).
%
% Two layers:
%   1. Compiled-predicate paths (Prolog clause -> WAM -> Rust ->
%      cargo test) for the nondeterministic builtins, verifying
%      builtin_call emission + choice-point/backtrack integration.
%   2. Direct execute_builtin unit coverage for the deterministic
%      families (register protocol, no WAM program needed).
%
% Usage: swipl -g run_tests -t halt tests/test_wam_rust_builtin_parity.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module(library(process)).
:- use_module(library(filesex)).

:- dynamic test_failed/0.

pass(Test) :- format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (   test_failed -> true ; assert(test_failed) ).

%% Predicates exercising builtins through the full compile pipeline.
:- dynamic t_between/1.
t_between(X) :- between(1, 3, X).

:- dynamic t_msort/1.
t_msort(S) :- msort([banana, apple, cherry, apple], S).

:- dynamic t_sort/1.
t_sort(S) :- sort([b, a, b], S).

:- dynamic t_sort4/1.
t_sort4(S) :-
    sort(2, '@=<', [row(a, 2), row(b, 1), row(c, 2)], S).

:- dynamic t_concat_split/2.
t_concat_split(A, B) :- atom_concat(A, B, abc).

:- dynamic t_select/2.
t_select(X, R) :- select(X, [a, b, c], R).

:- dynamic t_atomic/1.
t_atomic(X) :- atomic(X).

:- dynamic t_atomic_number/1.
t_atomic_number(X) :- atomic(X), number(X).

:- dynamic t_string_codes/1.
t_string_codes(Codes) :- string_codes(hello, Codes).

:- dynamic t_string_chars/1.
t_string_chars(Chars) :- string_chars(hi, Chars).

:- dynamic t_string_code/1.
t_string_code(Code) :- string_code(2, hello, Code).

:- dynamic t_output_aliases/0.
t_output_aliases :- print(alias), writeln(done).

:- dynamic t_tab/0.
t_tab :- tab(2).

:- dynamic t_output_family/0.
t_output_family :-
    put_char(x),
    put_code(121),
    write_canonical('two words'),
    write_canonical(node('has space', 7)).

:- dynamic t_format_capture/3.
t_format_capture(Atom, String, Codes) :-
    format(atom(Atom), 'X=~w ~a ~d ~s~n~~',
           [node(1), hello, 42, [111, 107]]),
    format(string(String), '~p~t~a', [foo(bar), done]),
    format(codes(Codes), ab, []).

:- dynamic t_format_output/0.
t_format_output :-
    format('F1~~'),
    format('|~w/~a/~d/~s~n', [node(1), hello, 42, [111, 107]]).

:- dynamic t_filesystem_query/1.
t_filesystem_query(Files) :-
    exists_file('Cargo.toml'),
    exists_directory('src/bin'),
    directory_files('src/bin', Files).

:- dynamic t_file_metadata/2.
t_file_metadata(Size, Time) :-
    size_file('Cargo.toml', Size),
    time_file('Cargo.toml', Time).

:- dynamic t_system_queries/3.
t_system_queries(Time, Pid, Path) :-
    get_time(Time),
    getpid(Pid),
    getenv('PATH', Path).

:- dynamic t_process_commands/2.
t_process_commands(Status, Output) :-
    shell('exit 0'),
    shell('exit 7', Status),
    system_to_atom('printf compiled-output', Output),
    atom_to_system('cat > process_command_compiled_fixture.txt',
                   'compiled-input').

:- dynamic t_sleep/0.
t_sleep :- sleep(0.001).

:- dynamic t_environment_mutation/1.
t_environment_mutation(Value) :-
    setenv('UNIFYWEAVER_RUST_WAM_COMPILED_ENV_8C19', first),
    setenv('UNIFYWEAVER_RUST_WAM_COMPILED_ENV_8C19', second),
    getenv('UNIFYWEAVER_RUST_WAM_COMPILED_ENV_8C19', Value),
    unsetenv('UNIFYWEAVER_RUST_WAM_COMPILED_ENV_8C19').

:- dynamic t_split_string/1.
t_split_string(Parts) :-
    split_string(' alpha, beta ,, gamma ', ',', ' ', Parts).

:- dynamic t_atom_split/1.
t_atom_split(Parts) :- atom_split('a,,b', ',', Parts).

:- dynamic t_atom_checks/0.
t_atom_checks :-
    atom_starts_with('prefix-middle-suffix', prefix),
    atom_ends_with('prefix-middle-suffix', suffix),
    atom_contains('prefix-middle-suffix', middle).

:- dynamic t_text_inspection/2.
t_text_inspection(Sub, After) :-
    sub_atom(hello, 1, 3, After, Sub),
    code_type(65, alpha),
    code_type(65, upper),
    code_type(95, csym).

:- dynamic t_path_utilities/5.
t_path_utilities(Base, Dir, Stem, Ext, Full) :-
    file_base_name('/tmp/archive.tar.gz', Base),
    file_directory_name('/tmp/archive.tar.gz', Dir),
    file_name_extension(Stem, Ext, '/tmp/archive.tar.gz'),
    is_absolute_file_name('/tmp/archive.tar.gz'),
    path_join('/tmp/base', 'child.txt', Full).

:- dynamic t_filesystem_identity/2.
t_filesystem_identity(Real, Target) :-
    realpath('identity_fixture/link.txt', Real),
    read_link('identity_fixture/link.txt', Target),
    same_file('identity_fixture/source.txt', 'identity_fixture/hard.txt').

:- dynamic t_filesystem_mutation/0.
t_filesystem_mutation :-
    make_directory('mutation_compiled_fixture'),
    copy_file('Cargo.toml', 'mutation_compiled_fixture/copied.toml'),
    rename_file('mutation_compiled_fixture/copied.toml',
                'mutation_compiled_fixture/renamed.toml'),
    delete_file('mutation_compiled_fixture/renamed.toml'),
    delete_directory('mutation_compiled_fixture').

:- dynamic t_file_content_io/1.
t_file_content_io(Content) :-
    write_atom_to_file('file_content_compiled_fixture.txt', alpha),
    append_atom_to_file('file_content_compiled_fixture.txt', '-beta'),
    read_file_to_atom('file_content_compiled_fixture.txt', Content),
    delete_file('file_content_compiled_fixture.txt').

:- dynamic t_file_links_truncate/1.
t_file_links_truncate(Content) :-
    make_directory('file_link_compiled_fixture'),
    write_atom_to_file('file_link_compiled_fixture/source.txt', abcdef),
    link('file_link_compiled_fixture/source.txt',
         'file_link_compiled_fixture/hard.txt'),
    symlink('source.txt', 'file_link_compiled_fixture/symbolic.txt'),
    truncate('file_link_compiled_fixture/hard.txt', 3),
    read_file_to_atom('file_link_compiled_fixture/symbolic.txt', Content),
    delete_file('file_link_compiled_fixture/symbolic.txt'),
    delete_file('file_link_compiled_fixture/hard.txt'),
    delete_file('file_link_compiled_fixture/source.txt'),
    delete_directory('file_link_compiled_fixture').

:- dynamic t_pairs_project/2.
t_pairs_project(Keys, Values) :-
    Pairs = [a-1, b-2],
    pairs_keys(Pairs, Keys),
    pairs_values(Pairs, Values).

:- dynamic t_pairs_split/2.
t_pairs_split(Keys, Values) :- pairs_keys_values([a-1, b-2], Keys, Values).

:- dynamic t_pairs_zip/1.
t_pairs_zip(Pairs) :- pairs_keys_values(Pairs, [a, b], [1, 2]).

%% catch/throw + succ predicates (ISO meta-builtin Call fallback path).
:- dynamic t_thrower/0.
t_thrower :- throw(oops(42)).

:- dynamic t_deep/0.
:- dynamic t_mid/0.
t_deep :- t_mid.
t_mid :- t_thrower.

:- dynamic t_catch_match/1.
t_catch_match(R) :- catch(t_thrower, oops(X), R = caught(X)).

:- dynamic t_catch_deep/1.
t_catch_deep(R) :- catch(t_deep, oops(X), R = X).

:- dynamic t_catch_nomatch/0.
t_catch_nomatch :- catch(t_thrower, other(_), true).

:- dynamic t_catch_nothrow/1.
t_catch_nothrow(X) :- catch(member(X, [a]), _Any, fail).

:- dynamic t_catch_failgoal/0.
t_catch_failgoal :- catch(fail, _Any, true).

:- dynamic t_catch_nested/1.
t_catch_nested(R) :- catch(catch(t_thrower, nomatch(_), fail), oops(X), R = X).

:- dynamic t_must_be/3.
t_must_be(TypeCaught, InstCaught, DomainCaught) :-
    must_be(atom, hello),
    must_be(integer, 42),
    must_be(number, 2.5),
    must_be(compound, node(x)),
    must_be(list, [a, b]),
    must_be(ground, node(a, [1, 2])),
    catch(must_be(atom, 5),
          error(type_error(atom, 5), _), TypeCaught = caught),
    catch(must_be(integer, _),
          error(instantiation_error, _), InstCaught = caught),
    catch(must_be(unknown_type, 5),
          error(domain_error(type, unknown_type), _), DomainCaught = caught).

:- dynamic t_succ_fwd/1.
t_succ_fwd(Y) :- succ(2, Y).

:- dynamic t_succ_rev/1.
t_succ_rev(X) :- succ(X, 3).

%% maplist family (P5): user-defined goals called per element through
%% the call_goal_value meta-call machinery.
:- dynamic double/2.
double(X, Y) :- Y is X * 2.

:- dynamic pos/1.
pos(X) :- X > 0.

:- dynamic acc_add/3.
acc_add(X, A0, A) :- A is A0 + X.

:- dynamic t_maplist/1.
t_maplist(L) :- maplist(double, [1, 2, 3], L).

:- dynamic t_maplist_check/0.
t_maplist_check :- maplist(pos, [1, 2, 3]).

:- dynamic t_maplist_fail/0.
t_maplist_fail :- maplist(pos, [1, -2, 3]).

:- dynamic t_include/1.
t_include(I) :- include(pos, [1, -2, 3], I).

:- dynamic t_exclude/1.
t_exclude(E) :- exclude(pos, [1, -2, 3], E).

:- dynamic t_partition/2.
t_partition(I, E) :- partition(pos, [1, -2, 3], I, E).

:- dynamic t_foldl/1.
t_foldl(S) :- foldl(acc_add, [1, 2, 3], 0, S).

:- dynamic t_alc_join/1.
t_alc_join(A) :- atomic_list_concat([a, b, c], '-', A).

:- dynamic t_alc_split/1.
t_alc_split(L) :- atomic_list_concat(L, '-', 'x-y-z').

cargo_available :-
    catch(
        (process_create(path(cargo), ['--version'],
                        [stdout(null), stderr(null), process(Pid)]),
         process_wait(Pid, exit(0))),
        _, fail).

write_output_probe(TmpDir) :-
    directory_file_path(TmpDir, 'src/bin', BinDir),
    make_directory_path(BinDir),
    directory_file_path(BinDir, 'output_probe.rs', ProbePath),
    ProbeContent = '
use builtin_parity_test::state::WamState;
use builtin_parity_test::{t_format_output_0, t_output_family_0};
use std::collections::HashMap;

fn main() {
    let mut vm = WamState::new(vec![], HashMap::new());
    if !t_output_family_0(&mut vm) {
        std::process::exit(1);
    }
    if !t_format_output_0(&mut vm) {
        std::process::exit(1);
    }
}
',
    setup_call_cleanup(
        open(ProbePath, write, Out, [encoding(utf8)]),
        write(Out, ProbeContent),
        close(Out)).

run_output_probe(TmpDir, ExitCode, Output) :-
    process_create(path(cargo), ['run', '--quiet', '--bin', 'output_probe'],
                   [cwd(TmpDir), stdout(pipe(Stream)), stderr(null), process(Pid)]),
    read_string(Stream, _, Output),
    close(Stream),
    process_wait(Pid, exit(ExitCode)).

test_builtin_parity_execution :-
    Test = 'WAM-Rust builtin parity: end-to-end execution (cargo test)',
    TmpDir = 'output/test_wam_rust_builtin_parity',
    (   \+ cargo_available
    ->  format('[SKIP] ~w (cargo not found)~n', [Test]),
        pass(Test)
    ;   (exists_directory(TmpDir) -> delete_directory_and_contents(TmpDir) ; true),
        write_wam_rust_project(
            [user:t_between/1, user:t_msort/1, user:t_sort/1, user:t_sort4/1,
             user:t_concat_split/2, user:t_select/2,
             user:t_atomic/1, user:t_atomic_number/1,
             user:t_string_codes/1, user:t_string_chars/1,
             user:t_string_code/1,
             user:t_output_aliases/0,
             user:t_tab/0,
             user:t_output_family/0,
             user:t_format_capture/3,
             user:t_format_output/0,
             user:t_filesystem_query/1,
             user:t_file_metadata/2,
             user:t_system_queries/3,
             user:t_process_commands/2,
             user:t_sleep/0,
             user:t_environment_mutation/1,
             user:t_split_string/1, user:t_atom_split/1, user:t_atom_checks/0,
             user:t_text_inspection/2,
             user:t_path_utilities/5,
             user:t_filesystem_identity/2,
             user:t_filesystem_mutation/0,
             user:t_file_content_io/1,
             user:t_file_links_truncate/1,
             user:t_pairs_project/2, user:t_pairs_split/2, user:t_pairs_zip/1,
             user:t_thrower/0, user:t_deep/0, user:t_mid/0,
             user:t_catch_match/1, user:t_catch_deep/1,
             user:t_catch_nomatch/0, user:t_catch_nothrow/1,
             user:t_catch_failgoal/0, user:t_catch_nested/1,
             user:t_must_be/3,
             user:t_succ_fwd/1, user:t_succ_rev/1,
             user:double/2, user:pos/1, user:acc_add/3,
             user:t_maplist/1, user:t_maplist_check/0, user:t_maplist_fail/0,
             user:t_include/1, user:t_exclude/1, user:t_partition/2,
             user:t_foldl/1, user:t_alc_join/1, user:t_alc_split/1],
            [module_name('builtin_parity_test'), wam_fallback(true)],
            TmpDir),
        write_output_probe(TmpDir),
        directory_file_path(TmpDir, 'tests', TestsDir),
        make_directory_path(TestsDir),
        directory_file_path(TestsDir, 'integration_test.rs', TestPath),
        TestContent = '
use builtin_parity_test::state::WamState;
use builtin_parity_test::value::Value;
use builtin_parity_test::{t_between_1, t_msort_1, t_sort_1, t_sort4_1, t_concat_split_2, t_select_2,
    t_atomic_1, t_atomic_number_1,
    t_string_codes_1, t_string_chars_1,
    t_string_code_1,
    t_output_aliases_0,
    t_tab_0,
    t_output_family_0,
    t_format_capture_3,
    t_filesystem_query_1,
    t_file_metadata_2,
    t_system_queries_3,
    t_process_commands_2,
    t_sleep_0,
    t_environment_mutation_1,
    t_split_string_1, t_atom_split_1, t_atom_checks_0,
    t_text_inspection_2,
    t_path_utilities_5,
    t_filesystem_identity_2,
    t_filesystem_mutation_0,
    t_file_content_io_1,
    t_file_links_truncate_1,
    t_pairs_project_2, t_pairs_split_2, t_pairs_zip_1,
    t_catch_match_1, t_catch_deep_1, t_catch_nomatch_0, t_catch_nothrow_1,
    t_catch_failgoal_0, t_catch_nested_1, t_must_be_3,
    t_succ_fwd_1, t_succ_rev_1,
    t_maplist_1, t_maplist_check_0, t_maplist_fail_0, t_include_1, t_exclude_1,
    t_partition_2, t_foldl_1, t_alc_join_1, t_alc_split_1};
use std::collections::HashMap;

fn vmnew() -> WamState {
    WamState::new(vec![], HashMap::new())
}

fn a(s: &str) -> Value { Value::Atom(s.to_string()) }
fn i(n: i64) -> Value { Value::Integer(n) }
fn ub(n: &str) -> Value { Value::Unbound(n.to_string()) }

fn read_var(vm: &WamState, name: &str) -> Value {
    match vm.bindings.get(name) {
        Some(v) => vm.deref_heap(&vm.deref_var(v)),
        None => Value::Unbound(name.to_string()),
    }
}

fn thrown_formal(vm: &WamState) -> Value {
    match vm.thrown_ball.as_ref() {
        Some(Value::Str(name, args)) if name == "error" && args.len() == 2 => {
            assert!(matches!(&args[1], Value::Unbound(_)),
                "error context must be a fresh variable");
            args[0].clone()
        }
        other => panic!("expected error/2 ball, got {:?}", other),
    }
}

// ---- layer 1: compiled-predicate paths -------------------------------

#[test]
fn test_between_enumeration_compiled() {
    let mut vm = vmnew();
    assert!(t_between_1(&mut vm, ub("X")), "between(1,3,X) first solution");
    let mut got = Vec::new();
    loop {
        match read_var(&vm, "X") {
            Value::Integer(n) => got.push(n),
            other => panic!("expected integer, got {:?}", other),
        }
        if !vm.backtrack() { break; }
    }
    assert_eq!(got, vec![1, 2, 3]);
}

#[test]
fn test_msort_sort_compiled() {
    let mut vm = vmnew();
    assert!(t_msort_1(&mut vm, ub("S")));
    assert_eq!(read_var(&vm, "S"),
        Value::List(vec![a("apple"), a("apple"), a("banana"), a("cherry")]),
        "msort keeps duplicates in standard order");
    let mut vm2 = vmnew();
    assert!(t_sort_1(&mut vm2, ub("S")));
    assert_eq!(read_var(&vm2, "S"), Value::List(vec![a("a"), a("b")]),
        "sort dedupes");

    let mut vm3 = vmnew();
    assert!(t_sort4_1(&mut vm3, ub("S")));
    assert_eq!(read_var(&vm3, "S"), Value::List(vec![
        Value::Str("row".to_string(), vec![a("b"), i(1)]),
        Value::Str("row".to_string(), vec![a("a"), i(2)]),
        Value::Str("row".to_string(), vec![a("c"), i(2)]),
    ]), "sort/4 keeps equal keys stable for @=<");
}

#[test]
fn test_atom_concat_split_compiled() {
    let mut vm = vmnew();
    assert!(t_concat_split_2(&mut vm, ub("A"), ub("B")));
    let mut got = Vec::new();
    loop {
        let pa = match read_var(&vm, "A") { Value::Atom(s) => s, o => panic!("A: {:?}", o) };
        let pb = match read_var(&vm, "B") { Value::Atom(s) => s, o => panic!("B: {:?}", o) };
        got.push((pa, pb));
        if !vm.backtrack() { break; }
    }
    got.sort();
    assert_eq!(got, vec![
        ("".to_string(), "abc".to_string()),
        ("a".to_string(), "bc".to_string()),
        ("ab".to_string(), "c".to_string()),
        ("abc".to_string(), "".to_string()),
    ]);
}

#[test]
fn test_select_enumeration_compiled() {
    let mut vm = vmnew();
    assert!(t_select_2(&mut vm, ub("X"), ub("R")));
    let mut got = Vec::new();
    loop {
        let x = match read_var(&vm, "X") { Value::Atom(s) => s, o => panic!("X: {:?}", o) };
        let r = match read_var(&vm, "R") {
            Value::List(items) => items.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(","),
            o => panic!("R: {:?}", o),
        };
        got.push((x, r));
        if !vm.backtrack() { break; }
    }
    got.sort();
    assert_eq!(got, vec![
        ("a".to_string(), "b,c".to_string()),
        ("b".to_string(), "a,c".to_string()),
        ("c".to_string(), "a,b".to_string()),
    ]);
}

#[test]
fn test_atomic_compiled() {
    let mut atom_vm = vmnew();
    assert!(t_atomic_1(&mut atom_vm, a("x")));
    let mut number_vm = vmnew();
    assert!(t_atomic_1(&mut number_vm, i(42)));
    let mut compound_vm = vmnew();
    assert!(!t_atomic_1(&mut compound_vm,
        Value::Str("f".to_string(), vec![a("x")])));
    let mut var_vm = vmnew();
    assert!(!t_atomic_1(&mut var_vm, ub("X")));

    let mut non_tail_ok_vm = vmnew();
    assert!(t_atomic_number_1(&mut non_tail_ok_vm, i(42)));
    let mut non_tail_fail_vm = vmnew();
    assert!(!t_atomic_number_1(&mut non_tail_fail_vm, a("x")));
}

#[test]
fn test_string_codes_chars_compiled() {
    let mut codes_vm = vmnew();
    assert!(t_string_codes_1(&mut codes_vm, ub("Codes")));
    assert_eq!(read_var(&codes_vm, "Codes"),
        Value::List(vec![i(104), i(101), i(108), i(108), i(111)]));

    let mut chars_vm = vmnew();
    assert!(t_string_chars_1(&mut chars_vm, ub("Chars")));
    assert_eq!(read_var(&chars_vm, "Chars"), Value::List(vec![a("h"), a("i")]));

    let mut code_vm = vmnew();
    assert!(t_string_code_1(&mut code_vm, ub("Code")));
    assert_eq!(read_var(&code_vm, "Code"), i(101));
}

#[test]
fn test_output_aliases_compiled() {
    let mut vm = vmnew();
    assert!(t_output_aliases_0(&mut vm));

    let mut tab_vm = vmnew();
    assert!(t_tab_0(&mut tab_vm));

    let mut output_vm = vmnew();
    assert!(t_output_family_0(&mut output_vm));
}

#[test]
fn test_format_capture_compiled() {
    let mut vm = vmnew();
    assert!(t_format_capture_3(&mut vm, ub("Atom"), ub("String"), ub("Codes")));
    assert_eq!(read_var(&vm, "Atom"),
        a(&format!("X=node(1) hello 42 ok{}~", char::from(10))));
    assert_eq!(read_var(&vm, "String"),
        a(&format!("foo(bar){}done", char::from(9))));
    assert_eq!(read_var(&vm, "Codes"), Value::List(vec![i(97), i(98)]));
}

#[test]
fn test_filesystem_query_compiled() {
    let mut vm = vmnew();
    assert!(t_filesystem_query_1(&mut vm, ub("Files")));
    assert_eq!(read_var(&vm, "Files"), Value::List(vec![
        a("."), a(".."), a("output_probe.rs")
    ]));
}

#[test]
fn test_file_metadata_compiled() {
    let mut vm = vmnew();
    assert!(t_file_metadata_2(&mut vm, ub("Size"), ub("Time")));
    match read_var(&vm, "Size") {
        Value::Integer(size) => assert!(size > 0),
        other => panic!("expected integer file size, got {:?}", other),
    }
    match read_var(&vm, "Time") {
        Value::Float(time) => assert!(time.is_finite() && time > 0.0),
        other => panic!("expected float modification time, got {:?}", other),
    }
}

#[test]
fn test_system_queries_compiled() {
    let mut vm = vmnew();
    assert!(t_system_queries_3(&mut vm, ub("Time"), ub("Pid"), ub("Path")));
    match read_var(&vm, "Time") {
        Value::Float(time) => assert!(time.is_finite() && time > 0.0),
        other => panic!("expected float wall time, got {:?}", other),
    }
    assert_eq!(read_var(&vm, "Pid"), i(i64::from(std::process::id())));
    assert_eq!(read_var(&vm, "Path"),
        a(&std::env::var("PATH").expect("PATH must be available to cargo test")));
}

#[cfg(unix)]
#[test]
fn test_process_commands_compiled() {
    let fixture = std::path::Path::new("process_command_compiled_fixture.txt");
    let _ = std::fs::remove_file(fixture);

    let mut vm = vmnew();
    assert!(t_process_commands_2(&mut vm, ub("Status"), ub("Output")));
    assert_eq!(read_var(&vm, "Status"), i(7));
    assert_eq!(read_var(&vm, "Output"), a("compiled-output"));
    assert_eq!(std::fs::read_to_string(fixture).unwrap(), "compiled-input");

    std::fs::remove_file(fixture).unwrap();
}

#[test]
fn test_sleep_compiled() {
    let mut vm = vmnew();
    assert!(t_sleep_0(&mut vm));
}

#[test]
fn test_environment_mutation_compiled() {
    let key = "UNIFYWEAVER_RUST_WAM_COMPILED_ENV_8C19";
    std::env::remove_var(key);

    let mut vm = vmnew();
    assert!(t_environment_mutation_1(&mut vm, ub("Value")));
    assert_eq!(read_var(&vm, "Value"), a("second"));
    assert!(std::env::var_os(key).is_none());
}

#[test]
fn test_text_utility_family_compiled() {
    let mut split_vm = vmnew();
    assert!(t_split_string_1(&mut split_vm, ub("Parts")));
    assert_eq!(read_var(&split_vm, "Parts"), Value::List(vec![
        a("alpha"), a("beta"), a(""), a("gamma")
    ]));

    let mut atom_split_vm = vmnew();
    assert!(t_atom_split_1(&mut atom_split_vm, ub("Parts")));
    assert_eq!(read_var(&atom_split_vm, "Parts"), Value::List(vec![
        a("a"), a(""), a("b")
    ]));

    let mut checks_vm = vmnew();
    assert!(t_atom_checks_0(&mut checks_vm));

    let mut inspection_vm = vmnew();
    assert!(t_text_inspection_2(&mut inspection_vm, ub("Sub"), ub("After")));
    assert_eq!(read_var(&inspection_vm, "Sub"), a("ell"));
    assert_eq!(read_var(&inspection_vm, "After"), i(1));
}

#[test]
fn test_path_utility_family_compiled() {
    let mut vm = vmnew();
    assert!(t_path_utilities_5(
        &mut vm, ub("Base"), ub("Dir"), ub("Stem"), ub("Ext"), ub("Full")));
    assert_eq!(read_var(&vm, "Base"), a("archive.tar.gz"));
    assert_eq!(read_var(&vm, "Dir"), a("/tmp"));
    assert_eq!(read_var(&vm, "Stem"), a("/tmp/archive.tar"));
    assert_eq!(read_var(&vm, "Ext"), a("gz"));
    assert_eq!(read_var(&vm, "Full"), a("/tmp/base/child.txt"));
}

#[cfg(unix)]
#[test]
fn test_filesystem_identity_compiled_and_direct() {
    let fixture = std::path::Path::new("identity_fixture");
    let _ = std::fs::remove_dir_all(fixture);
    std::fs::create_dir(fixture).unwrap();
    std::fs::write(fixture.join("source.txt"), b"identity").unwrap();
    std::fs::hard_link(fixture.join("source.txt"), fixture.join("hard.txt")).unwrap();
    std::os::unix::fs::symlink("source.txt", fixture.join("link.txt")).unwrap();

    let expected_real = std::fs::canonicalize(fixture.join("link.txt"))
        .unwrap().into_os_string().into_string().unwrap();
    let mut vm = vmnew();
    assert!(t_filesystem_identity_2(&mut vm, ub("Real"), ub("Target")));
    assert_eq!(read_var(&vm, "Real"), a(&expected_real));
    assert_eq!(read_var(&vm, "Target"), a("source.txt"));

    let source = fixture.join("source.txt").to_str().unwrap().to_string();
    let hard = fixture.join("hard.txt").to_str().unwrap().to_string();
    let link = fixture.join("link.txt").to_str().unwrap().to_string();
    assert!(call2("same_file/2", a(&source), a(&hard)).0);
    assert!(call2("same_file/2", a(&source), a(&link)).0,
        "metadata follows symlinks");
    std::fs::write(fixture.join("other.txt"), b"other").unwrap();
    let other = fixture.join("other.txt").to_str().unwrap().to_string();
    assert!(!call2("same_file/2", a(&source), a(&other)).0);

    let (real_ok, real_vm) = call2("realpath/2", a(&link), ub("Real"));
    assert!(real_ok);
    assert_eq!(read_var(&real_vm, "Real"), a(&expected_real));
    let (link_ok, link_vm) = call2("read_link/2", a(&link), ub("Target"));
    assert!(link_ok);
    assert_eq!(read_var(&link_vm, "Target"), a("source.txt"));

    std::fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn test_filesystem_identity_validation() {
    assert!(!call2("realpath/2", a("missing-identity-path"), ub("Real")).0);
    assert!(!call2("read_link/2", a("missing-identity-link"), ub("Target")).0);
    assert!(!call2("same_file/2", a("missing-a"), a("missing-b")).0);
    assert!(!call2("realpath/2", i(1), ub("Real")).0);
    assert!(!call2("read_link/2", ub("Path"), ub("Target")).0);
    assert!(!call2("same_file/2", a("path"), i(2)).0);
}

#[test]
fn test_filesystem_mutation_compiled() {
    let fixture = std::path::Path::new("mutation_compiled_fixture");
    let _ = std::fs::remove_dir_all(fixture);

    let mut vm = vmnew();
    assert!(t_filesystem_mutation_0(&mut vm));
    assert!(!fixture.exists());
}

#[test]
fn test_filesystem_mutation_direct() {
    let fixture = std::path::Path::new("mutation_direct_fixture");
    let _ = std::fs::remove_dir_all(fixture);
    let fixture_text = fixture.to_str().unwrap();

    assert!(call1("make_directory/1", a(fixture_text)).0);
    assert!(fixture.is_dir());
    assert!(!call1("make_directory/1", a(fixture_text)).0);

    let copied = fixture.join("copied.toml");
    let copied_text = copied.to_str().unwrap();
    assert!(call2("copy_file/2", a("Cargo.toml"), a(copied_text)).0);
    assert_eq!(std::fs::read(&copied).unwrap(), std::fs::read("Cargo.toml").unwrap());

    let renamed = fixture.join("renamed.toml");
    let renamed_text = renamed.to_str().unwrap();
    assert!(call2("rename_file/2", a(copied_text), a(renamed_text)).0);
    assert!(!copied.exists());
    assert!(renamed.is_file());

    assert!(!call1("delete_directory/1", a(fixture_text)).0,
        "delete_directory/1 must reject a nonempty directory");
    assert!(call1("delete_file/1", a(renamed_text)).0);
    assert!(!call1("delete_file/1", a(renamed_text)).0);
    assert!(call1("delete_directory/1", a(fixture_text)).0);
    assert!(!call1("delete_directory/1", a(fixture_text)).0);

    assert!(!call1("make_directory/1", ub("Path")).0);
    assert!(!call1("delete_file/1", i(7)).0);
    assert!(!call2("copy_file/2", a("missing-mutation-source"), a("unused")).0);
    assert!(!call2("copy_file/2", a("Cargo.toml"), ub("Destination")).0);
    assert!(!call2("rename_file/2", ub("Source"), a("unused")).0);
}

#[test]
fn test_file_content_io_compiled() {
    let fixture = std::path::Path::new("file_content_compiled_fixture.txt");
    let _ = std::fs::remove_file(fixture);

    let mut vm = vmnew();
    assert!(t_file_content_io_1(&mut vm, ub("Content")));
    assert_eq!(read_var(&vm, "Content"), a("alpha-beta"));
    assert!(!fixture.exists());
}

#[test]
fn test_file_content_io_direct() {
    let path = "file_content_direct_fixture.txt";
    let _ = std::fs::remove_file(path);

    assert!(call2("write_atom_to_file/2", a(path), a("longer content")).0);
    assert!(call2("write_atom_to_file/2", a(path), a("short")).0);
    assert_eq!(std::fs::read(path).unwrap(), b"short");
    assert!(call2("append_atom_to_file/2", a(path), a("-tail")).0);
    assert_eq!(std::fs::read(path).unwrap(), b"short-tail");

    let (read_ok, read_vm) = call2("read_file_to_atom/2", a(path), ub("Content"));
    assert!(read_ok);
    assert_eq!(read_var(&read_vm, "Content"), a("short-tail"));
    assert!(call2("read_file_to_atom/2", a(path), a("short-tail")).0);
    assert!(!call2("read_file_to_atom/2", a(path), a("wrong")).0);

    std::fs::remove_file(path).unwrap();
    assert!(call2("append_atom_to_file/2", a(path), a("created")).0,
        "append_atom_to_file/2 creates a missing file");
    assert_eq!(std::fs::read(path).unwrap(), b"created");

    std::fs::write(path, [0xff]).unwrap();
    assert!(!call2("read_file_to_atom/2", a(path), ub("Content")).0,
        "Rust atoms require UTF-8 file content");
    assert!(!call2("read_file_to_atom/2", a("missing-file-content"), ub("Content")).0);
    assert!(!call2("read_file_to_atom/2", i(7), ub("Content")).0);
    assert!(!call2("write_atom_to_file/2", a(path), i(7)).0);
    assert!(!call2("append_atom_to_file/2", ub("Path"), a("content")).0);

    std::fs::remove_file(path).unwrap();
}

#[cfg(unix)]
#[test]
fn test_file_links_truncate_compiled() {
    let fixture = std::path::Path::new("file_link_compiled_fixture");
    let _ = std::fs::remove_dir_all(fixture);

    let mut vm = vmnew();
    assert!(t_file_links_truncate_1(&mut vm, ub("Content")));
    assert_eq!(read_var(&vm, "Content"), a("abc"));
    assert!(!fixture.exists());
}

#[cfg(unix)]
#[test]
fn test_file_links_truncate_direct() {
    let fixture = std::path::Path::new("file_link_direct_fixture");
    let _ = std::fs::remove_dir_all(fixture);
    std::fs::create_dir(fixture).unwrap();

    let source = fixture.join("source.txt");
    let hard = fixture.join("hard.txt");
    let symbolic = fixture.join("symbolic.txt");
    let dangling = fixture.join("dangling.txt");
    std::fs::write(&source, b"abcdef").unwrap();
    let source_text = source.to_str().unwrap();
    let hard_text = hard.to_str().unwrap();
    let symbolic_text = symbolic.to_str().unwrap();
    let dangling_text = dangling.to_str().unwrap();

    assert!(call2("link/2", a(source_text), a(hard_text)).0);
    assert!(call2("same_file/2", a(source_text), a(hard_text)).0);
    assert!(!call2("link/2", a(source_text), a(hard_text)).0);

    assert!(call2("symlink/2", a("source.txt"), a(symbolic_text)).0);
    assert_eq!(std::fs::read(&symbolic).unwrap(), b"abcdef");
    assert!(!call2("symlink/2", a("source.txt"), a(symbolic_text)).0);
    assert!(call2("symlink/2", a("missing.txt"), a(dangling_text)).0);
    assert!(std::fs::symlink_metadata(&dangling).unwrap().file_type().is_symlink());

    assert!(call2("truncate/2", a(hard_text), i(3)).0);
    assert_eq!(std::fs::read(&source).unwrap(), b"abc");
    assert!(call2("truncate/2", a(source_text), i(6)).0);
    assert_eq!(std::fs::read(&source).unwrap(), vec![97, 98, 99, 0, 0, 0]);

    assert!(!call2("link/2", a("missing-link-source"), a("unused-link")).0);
    assert!(!call2("link/2", i(1), a("unused-link")).0);
    assert!(!call2("symlink/2", a("target"), i(2)).0);
    assert!(!call2("truncate/2", a("missing-truncate-path"), i(1)).0);
    assert!(!call2("truncate/2", a(source_text), i(-1)).0);
    assert!(!call2("truncate/2", a(source_text), a("3")).0);

    std::fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn test_pairs_family_compiled() {
    let mut project_vm = vmnew();
    assert!(t_pairs_project_2(&mut project_vm, ub("Keys"), ub("Values")));
    assert_eq!(read_var(&project_vm, "Keys"), Value::List(vec![a("a"), a("b")]));
    assert_eq!(read_var(&project_vm, "Values"), Value::List(vec![i(1), i(2)]));

    let mut split_vm = vmnew();
    assert!(t_pairs_split_2(&mut split_vm, ub("Keys"), ub("Values")));
    assert_eq!(read_var(&split_vm, "Keys"), Value::List(vec![a("a"), a("b")]));
    assert_eq!(read_var(&split_vm, "Values"), Value::List(vec![i(1), i(2)]));

    let mut zip_vm = vmnew();
    assert!(t_pairs_zip_1(&mut zip_vm, ub("Pairs")));
    assert_eq!(read_var(&zip_vm, "Pairs"), Value::List(vec![
        Value::Str("-".to_string(), vec![a("a"), i(1)]),
        Value::Str("-".to_string(), vec![a("b"), i(2)]),
    ]));
}

#[test]
fn test_catch_matching_catcher_runs_recovery() {
    let mut vm = vmnew();
    assert!(t_catch_match_1(&mut vm, ub("R")),
        "catch(throw(oops(42)), oops(X), R = caught(X)) must succeed");
    assert_eq!(read_var(&vm, "R"),
        Value::Str("caught".to_string(), vec![i(42)]));
    assert!(vm.thrown_ball.is_none(), "ball consumed");
}

#[test]
fn test_catch_propagates_through_calls() {
    let mut vm = vmnew();
    assert!(t_catch_deep_1(&mut vm, ub("R")),
        "throw two predicate calls deep must reach the catcher");
    assert_eq!(read_var(&vm, "R"), i(42));
}

#[test]
fn test_catch_nonmatching_catcher_rethrows() {
    let mut vm = vmnew();
    assert!(!t_catch_nomatch_0(&mut vm),
        "non-unifying catcher must rethrow (uncaught at top = failure)");
    assert!(vm.thrown_ball.is_some(), "ball still in flight after rethrow");
}

#[test]
fn test_catch_transparent_when_no_throw() {
    let mut vm = vmnew();
    assert!(t_catch_nothrow_1(&mut vm, ub("X")));
    assert_eq!(read_var(&vm, "X"), a("a"));
    assert!(vm.thrown_ball.is_none());
}

#[test]
fn test_catch_plain_goal_failure_fails() {
    let mut vm = vmnew();
    assert!(!t_catch_failgoal_0(&mut vm),
        "catch(fail, _, true) fails — recovery only runs on throw");
    assert!(vm.thrown_ball.is_none());
}

#[test]
fn test_catch_nested_inner_rethrows_outer_catches() {
    let mut vm = vmnew();
    assert!(t_catch_nested_1(&mut vm, ub("R")));
    assert_eq!(read_var(&vm, "R"), i(42));
    assert!(vm.thrown_ball.is_none());
}

#[test]
fn test_must_be_compiled_success_and_errors() {
    let mut vm = vmnew();
    assert!(t_must_be_3(
        &mut vm, ub("TypeCaught"), ub("InstCaught"), ub("DomainCaught")));
    assert_eq!(read_var(&vm, "TypeCaught"), a("caught"));
    assert_eq!(read_var(&vm, "InstCaught"), a("caught"));
    assert_eq!(read_var(&vm, "DomainCaught"), a("caught"));
    assert!(vm.thrown_ball.is_none(), "all errors were caught");
}

#[test]
fn test_succ_both_modes_compiled() {
    let mut vm = vmnew();
    assert!(t_succ_fwd_1(&mut vm, ub("Y")));
    assert_eq!(read_var(&vm, "Y"), i(3));
    let mut vm2 = vmnew();
    assert!(t_succ_rev_1(&mut vm2, ub("X")));
    assert_eq!(read_var(&vm2, "X"), i(2));
}

#[test]
fn test_succ_direct_edge_cases() {
    let (ok, vm) = call2("succ/2", i(0), ub("Y"));
    assert!(ok);
    assert_eq!(read_var(&vm, "Y"), i(1));
    assert!(!call2("succ/2", ub("X"), i(0)).0, "succ(X, 0) has no natural X");
    assert!(!call2("succ/2", i(-1), ub("Y")).0, "negative X fails");
    assert!(call2("succ/2", i(2), i(3)).0);
    assert!(!call2("succ/2", i(2), i(4)).0);
}

#[test]
fn test_maplist_transform_compiled() {
    let mut vm = vmnew();
    assert!(t_maplist_1(&mut vm, ub("L")),
        "maplist(double, [1,2,3], L) must succeed");
    assert_eq!(read_var(&vm, "L"), Value::List(vec![i(2), i(4), i(6)]));
}

#[test]
fn test_maplist_check_and_fail_compiled() {
    let mut vm = vmnew();
    assert!(t_maplist_check_0(&mut vm), "all elements positive");
    let mut vm2 = vmnew();
    assert!(!t_maplist_fail_0(&mut vm2), "one negative element fails maplist");
}

#[test]
fn test_include_exclude_partition_compiled() {
    let mut vm = vmnew();
    assert!(t_include_1(&mut vm, ub("I")));
    assert_eq!(read_var(&vm, "I"), Value::List(vec![i(1), i(3)]));
    let mut vm2 = vmnew();
    assert!(t_exclude_1(&mut vm2, ub("E")));
    assert_eq!(read_var(&vm2, "E"), Value::List(vec![i(-2)]));
    let mut vm3 = vmnew();
    assert!(t_partition_2(&mut vm3, ub("I"), ub("E")));
    assert_eq!(read_var(&vm3, "I"), Value::List(vec![i(1), i(3)]));
    assert_eq!(read_var(&vm3, "E"), Value::List(vec![i(-2)]));
}

#[test]
fn test_foldl_compiled() {
    let mut vm = vmnew();
    assert!(t_foldl_1(&mut vm, ub("S")),
        "foldl(acc_add, [1,2,3], 0, S) must succeed");
    assert_eq!(read_var(&vm, "S"), i(6));
}

#[test]
fn test_atomic_list_concat_compiled() {
    let mut vm = vmnew();
    assert!(t_alc_join_1(&mut vm, ub("A")));
    assert_eq!(read_var(&vm, "A"), a("a-b-c"));
    let mut vm2 = vmnew();
    assert!(t_alc_split_1(&mut vm2, ub("L")));
    assert_eq!(read_var(&vm2, "L"), Value::List(vec![a("x"), a("y"), a("z")]));
}

#[test]
fn test_atomic_list_concat_direct() {
    let (ok, vm) = call2("atomic_list_concat/2",
        Value::List(vec![a("foo"), i(7), a("bar")]), ub("A"));
    assert!(ok);
    assert_eq!(read_var(&vm, "A"), a("foo7bar"));
    assert!(call3("atomic_list_concat/3",
        Value::List(vec![a("x"), a("y")]), a(","), a("x,y")).0);
    assert!(!call3("atomic_list_concat/3",
        Value::List(vec![a("x"), a("y")]), a(","), a("x;y")).0);
    // Split with empty separator must fail (would not terminate).
    assert!(!call3("atomic_list_concat/3", ub("L"), a(""), a("xyz")).0);
}

#[test]
fn test_char_type_direct() {
    assert!(call2("char_type/2", a("x"), a("alpha")).0);
    assert!(!call2("char_type/2", a("1"), a("alpha")).0);
    assert!(call2("char_type/2", a("1"), a("alnum")).0);
    assert!(call2("char_type/2", a("_"), a("csym")).0);
    assert!(call2("char_type/2", a(" "), a("space")).0);
    assert!(call2("char_type/2", a("!"), a("punct")).0);
    assert!(call2("char_type/2", a("A"), a("upper")).0);
    assert!(!call2("char_type/2", a("a"), a("upper")).0);
    let (ok, vm) = call2("char_type/2", a("7"),
        Value::Str("digit/1".to_string(), vec![ub("W")]));
    assert!(ok);
    assert_eq!(read_var(&vm, "W"), i(7));
    let (ok2, vm2) = call2("char_type/2", a("B"),
        Value::Str("to_lower/1".to_string(), vec![ub("L")]));
    assert!(ok2);
    assert_eq!(read_var(&vm2, "L"), a("b"));
    let (ok3, vm3) = call2("char_type/2", a("Q"),
        Value::Str("upper/1".to_string(), vec![ub("L")]));
    assert!(ok3);
    assert_eq!(read_var(&vm3, "L"), a("q"));
    assert!(!call2("char_type/2", a("q"),
        Value::Str("upper/1".to_string(), vec![ub("L")])).0);
}

#[test]
fn test_code_type_direct() {
    assert!(call2("code_type/2", i(65), a("alpha")).0);
    assert!(call2("code_type/2", i(65), a("upper")).0);
    assert!(call2("code_type/2", i(97), a("lower")).0);
    assert!(call2("code_type/2", i(48), a("digit")).0);
    assert!(call2("code_type/2", i(95), a("csym")).0);
    assert!(call2("code_type/2", i(95), a("csymf")).0);
    assert!(call2("code_type/2", i(10), a("space")).0);
    assert!(call2("code_type/2", i(9), a("white")).0);
    assert!(call2("code_type/2", i(33), a("punct")).0);
    assert!(call2("code_type/2", i(33), a("graph")).0);
    assert!(call2("code_type/2", i(127), a("ascii")).0);
    assert!(!call2("code_type/2", i(128), a("ascii")).0);
    assert!(call2("code_type/2", i(10), a("newline")).0);
    assert!(call2("code_type/2", i(13), a("end_of_line")).0);
    assert!(!call2("code_type/2", i(-1), a("ascii")).0);
    assert!(!call2("code_type/2", i(256), a("ascii")).0);
    assert!(!call2("code_type/2", a("A"), a("alpha")).0);
    assert!(!call2("code_type/2", ub("Code"), a("alpha")).0);
    assert!(!call2("code_type/2", i(65), ub("Type")).0);
    assert!(!call2("code_type/2", i(65), a("unknown")).0);
}

#[test]
fn test_sub_atom_direct() {
    let (ok, vm) = call5("sub_atom/5",
        a("hello"), i(1), i(3), ub("After"), ub("Sub"));
    assert!(ok);
    assert_eq!(read_var(&vm, "After"), i(1));
    assert_eq!(read_var(&vm, "Sub"), a("ell"));

    assert!(call5("sub_atom/5", a("hello"), i(0), i(5), i(0), a("hello")).0);
    let (empty_ok, empty_vm) = call5("sub_atom/5",
        a("hello"), i(2), i(0), ub("After"), ub("Sub"));
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "After"), i(3));
    assert_eq!(read_var(&empty_vm, "Sub"), a(""));

    assert!(!call5("sub_atom/5", a("hello"), i(1), i(3), i(2), a("ell")).0);
    let (rollback_ok, rollback_vm) = call5("sub_atom/5",
        a("hello"), i(1), i(3), ub("After"), a("wrong"));
    assert!(!rollback_ok);
    assert_eq!(read_var(&rollback_vm, "After"), ub("After"),
        "failed Sub unification must roll back the After binding");

    assert!(!call5("sub_atom/5", i(42), i(0), i(1), ub("After"), ub("Sub")).0);
    assert!(!call5("sub_atom/5", ub("Atom"), i(0), i(1), ub("After"), ub("Sub")).0);
    assert!(!call5("sub_atom/5", a("hello"), ub("Before"), i(1), ub("After"), ub("Sub")).0);
    assert!(!call5("sub_atom/5", a("hello"), i(0), ub("Length"), ub("After"), ub("Sub")).0);
    assert!(!call5("sub_atom/5", a("hello"), i(-1), i(1), ub("After"), ub("Sub")).0);
    assert!(!call5("sub_atom/5", a("hello"), i(1), i(-1), ub("After"), ub("Sub")).0);
    assert!(!call5("sub_atom/5", a("hello"), i(4), i(2), ub("After"), ub("Sub")).0);
    assert!(!call5("sub_atom/5", a("hello"), i(i64::MAX), i(i64::MAX),
        ub("After"), ub("Sub")).0);
}

#[test]
fn test_must_be_direct() {
    assert!(call2("must_be/2", a("atom"), a("hello")).0);
    assert!(call2("must_be/2", a("integer"), i(7)).0);
    assert!(call2("must_be/2", a("float"), Value::Float(2.5)).0);
    assert!(call2("must_be/2", a("number"), i(7)).0);
    assert!(call2("must_be/2", a("compound"),
        Value::Str("node".to_string(), vec![i(1)])).0);
    assert!(call2("must_be/2", a("compound"), Value::List(vec![i(1)])).0);
    assert!(call2("must_be/2", a("atomic"), Value::List(vec![])).0);
    assert!(call2("must_be/2", a("callable"), Value::List(vec![i(1)])).0);
    assert!(call2("must_be/2", a("boolean"), Value::Bool(true)).0);
    assert!(call2("must_be/2", a("list"), Value::List(vec![a("x")])).0);
    assert!(call2("must_be/2", a("ground"),
        Value::Str("node".to_string(), vec![a("x"), i(1)])).0);
    assert!(call2("must_be/2", a("var"), ub("X")).0);
    assert!(call2("must_be/2", a("nonvar"), i(1)).0);

    let (type_ok, type_vm) = call2("must_be/2", a("atom"), i(5));
    assert!(!type_ok);
    assert_eq!(thrown_formal(&type_vm), Value::Str(
        "type_error".to_string(), vec![a("atom"), i(5)]));

    let (inst_ok, inst_vm) = call2("must_be/2", a("integer"), ub("X"));
    assert!(!inst_ok);
    assert_eq!(thrown_formal(&inst_vm), a("instantiation_error"));

    let nested = Value::Str("node".to_string(), vec![ub("X"), i(2)]);
    let (ground_ok, ground_vm) = call2("must_be/2", a("ground"), nested);
    assert!(!ground_ok);
    assert_eq!(thrown_formal(&ground_vm), a("instantiation_error"));

    let (domain_ok, domain_vm) = call2("must_be/2", a("unknown_type"), i(5));
    assert!(!domain_ok);
    assert_eq!(thrown_formal(&domain_vm), Value::Str(
        "domain_error".to_string(), vec![a("type"), a("unknown_type")]));

    let (type_inst_ok, type_inst_vm) = call2("must_be/2", ub("Type"), i(5));
    assert!(!type_inst_ok);
    assert_eq!(thrown_formal(&type_inst_vm), a("instantiation_error"));

    let (type_atom_ok, type_atom_vm) = call2("must_be/2", i(7), i(5));
    assert!(!type_atom_ok);
    assert_eq!(thrown_formal(&type_atom_vm), Value::Str(
        "type_error".to_string(), vec![a("atom"), i(7)]));

    let (nonvar_ok, nonvar_vm) = call2("must_be/2", a("nonvar"), ub("X"));
    assert!(!nonvar_ok);
    assert_eq!(thrown_formal(&nonvar_vm), Value::Str(
        "type_error".to_string(), vec![a("nonvar"), ub("X")]));
}

// ---- layer 2: direct execute_builtin coverage ------------------------

fn call1(op: &str, a1: Value) -> (bool, WamState) {
    let mut vm = vmnew();
    vm.set_reg("A1", a1);
    let ok = vm.execute_builtin(op, 1);
    (ok, vm)
}

fn call2(op: &str, a1: Value, a2: Value) -> (bool, WamState) {
    let mut vm = vmnew();
    vm.set_reg("A1", a1);
    vm.set_reg("A2", a2);
    let ok = vm.execute_builtin(op, 2);
    (ok, vm)
}

fn call3(op: &str, a1: Value, a2: Value, a3: Value) -> (bool, WamState) {
    let mut vm = vmnew();
    vm.set_reg("A1", a1);
    vm.set_reg("A2", a2);
    vm.set_reg("A3", a3);
    let ok = vm.execute_builtin(op, 3);
    (ok, vm)
}

fn call4(op: &str, a1: Value, a2: Value, a3: Value, a4: Value) -> (bool, WamState) {
    let mut vm = vmnew();
    vm.set_reg("A1", a1);
    vm.set_reg("A2", a2);
    vm.set_reg("A3", a3);
    vm.set_reg("A4", a4);
    let ok = vm.execute_builtin(op, 4);
    (ok, vm)
}

fn call5(op: &str, a1: Value, a2: Value, a3: Value, a4: Value, a5: Value) -> (bool, WamState) {
    let mut vm = vmnew();
    vm.set_reg("A1", a1);
    vm.set_reg("A2", a2);
    vm.set_reg("A3", a3);
    vm.set_reg("A4", a4);
    vm.set_reg("A5", a5);
    let ok = vm.execute_builtin(op, 5);
    (ok, vm)
}

#[test]
fn test_not_unify_and_not_equal() {
    assert!(call2("\\\\=/2", a("x"), a("y")).0);
    assert!(!call2("\\\\=/2", a("x"), a("x")).0);
    assert!(!call2("\\\\=/2", ub("V"), a("x")).0, "var unifies with anything");
    assert!(call2("\\\\==/2", a("x"), a("y")).0);
    assert!(!call2("\\\\==/2", i(7), i(7)).0);
    assert!(call2("\\\\==/2", ub("V"), a("x")).0, "unbound var differs from atom");
}

#[test]
fn test_standard_order() {
    assert!(call2("@</2", a("a"), a("b")).0);
    assert!(!call2("@</2", a("b"), a("a")).0);
    assert!(call2("@</2", i(99), a("a")).0, "numbers precede atoms");
    assert!(call2("@</2", a("zzz"), Value::Str("f/1".to_string(), vec![i(1)])).0,
        "atoms precede compounds");
    assert!(call2("@=</2", a("a"), a("a")).0);
    assert!(call2("@>/2", Value::List(vec![i(1), i(2)]), Value::List(vec![i(1), i(1)])).0);
    assert!(call2("@>=/2", a("b"), a("b")).0);
    let (ok, vm) = call3("compare/3", ub("O"), i(1), i(2));
    assert!(ok);
    assert_eq!(read_var(&vm, "O"), a("<"));
    let (ok2, vm2) = call3("compare/3", ub("O"), a("x"), a("x"));
    assert!(ok2);
    assert_eq!(read_var(&vm2, "O"), a("="));
}

#[test]
fn test_keysort() {
    let pair = |k: Value, v: Value| Value::Str("-/2".to_string(), vec![k, v]);
    // read_var goes through deref_heap, which normalizes the functor
    // spelling "-/2" to the display name "-" — compare against that.
    let dpair = |k: Value, v: Value| Value::Str("-".to_string(), vec![k, v]);
    let (ok, vm) = call2("keysort/2",
        Value::List(vec![pair(a("b"), i(1)), pair(a("a"), i(2)), pair(a("b"), i(0))]),
        ub("S"));
    assert!(ok);
    assert_eq!(read_var(&vm, "S"), Value::List(vec![
        dpair(a("a"), i(2)), dpair(a("b"), i(1)), dpair(a("b"), i(0)),
    ]), "stable by key, payload order preserved");
}

#[test]
fn test_sort4_modes_and_keys() {
    let row = |name: &str, key: i64| {
        Value::Str("row/2".to_string(), vec![a(name), i(key)])
    };
    let drow = |name: &str, key: i64| {
        Value::Str("row".to_string(), vec![a(name), i(key)])
    };
    let rows = Value::List(vec![row("a", 2), row("b", 1), row("c", 2)]);

    let (ascending_ok, ascending_vm) = call4(
        "sort/4", i(2), a("@=<"), rows.clone(), ub("Sorted"));
    assert!(ascending_ok);
    assert_eq!(read_var(&ascending_vm, "Sorted"), Value::List(vec![
        drow("b", 1), drow("a", 2), drow("c", 2),
    ]));

    let (ascending_unique_ok, ascending_unique_vm) = call4(
        "sort/4", i(2), a("@<"), rows.clone(), ub("Sorted"));
    assert!(ascending_unique_ok);
    assert_eq!(read_var(&ascending_unique_vm, "Sorted"), Value::List(vec![
        drow("b", 1), drow("a", 2),
    ]));

    let (descending_ok, descending_vm) = call4(
        "sort/4", i(2), a("@>="), rows.clone(), ub("Sorted"));
    assert!(descending_ok);
    assert_eq!(read_var(&descending_vm, "Sorted"), Value::List(vec![
        drow("a", 2), drow("c", 2), drow("b", 1),
    ]));

    let (descending_unique_ok, descending_unique_vm) = call4(
        "sort/4", i(2), a("@>"), rows, ub("Sorted"));
    assert!(descending_unique_ok);
    assert_eq!(read_var(&descending_unique_vm, "Sorted"), Value::List(vec![
        drow("a", 2), drow("b", 1),
    ]));

    let (whole_ok, whole_vm) = call4(
        "sort/4", i(0), a("@<"),
        Value::List(vec![a("b"), a("a"), a("b")]), ub("Sorted"));
    assert!(whole_ok);
    assert_eq!(read_var(&whole_vm, "Sorted"), Value::List(vec![a("a"), a("b")]));

    let (list_key_ok, list_key_vm) = call4(
        "sort/4", i(1), a("@=<"),
        Value::List(vec![
            Value::List(vec![i(2), a("two")]),
            Value::List(vec![i(1), a("one")]),
        ]),
        ub("Sorted"),
    );
    assert!(list_key_ok);
    assert_eq!(read_var(&list_key_vm, "Sorted"), Value::List(vec![
        Value::List(vec![i(1), a("one")]),
        Value::List(vec![i(2), a("two")]),
    ]));
}

#[test]
fn test_sort4_validation_and_rollback() {
    let list = Value::List(vec![a("b"), a("a")]);
    assert!(!call4("sort/4", i(-1), a("@<"), list.clone(), ub("Sorted")).0);
    assert!(!call4("sort/4", a("key"), a("@<"), list.clone(), ub("Sorted")).0);
    assert!(!call4("sort/4", i(0), a("ascending"), list.clone(), ub("Sorted")).0);
    assert!(!call4("sort/4", i(0), a("@<"), a("not-a-list"), ub("Sorted")).0);

    let (empty_ok, empty_vm) = call4(
        "sort/4", i(0), a("@=<"), Value::List(vec![]), ub("Sorted"));
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "Sorted"), Value::List(vec![]));

    let (rollback_ok, rollback_vm) = call4(
        "sort/4",
        i(0),
        a("@=<"),
        list,
        Value::List(vec![ub("First"), a("wrong")]),
    );
    assert!(!rollback_ok);
    assert_eq!(read_var(&rollback_vm, "First"), ub("First"));
}

#[test]
fn test_pairs_family_direct() {
    let pair = |k: Value, v: Value| Value::Str("-/2".to_string(), vec![k, v]);
    let pairs = Value::List(vec![pair(a("a"), i(1)), pair(a("b"), i(2))]);

    let (keys_ok, keys_vm) = call2("pairs_keys/2", pairs.clone(), ub("Keys"));
    assert!(keys_ok);
    assert_eq!(read_var(&keys_vm, "Keys"), Value::List(vec![a("a"), a("b")]));

    let (values_ok, values_vm) = call2("pairs_values/2", pairs.clone(), ub("Values"));
    assert!(values_ok);
    assert_eq!(read_var(&values_vm, "Values"), Value::List(vec![i(1), i(2)]));

    let (split_ok, split_vm) = call3(
        "pairs_keys_values/3", pairs, ub("Keys"), ub("Values"));
    assert!(split_ok);
    assert_eq!(read_var(&split_vm, "Keys"), Value::List(vec![a("a"), a("b")]));
    assert_eq!(read_var(&split_vm, "Values"), Value::List(vec![i(1), i(2)]));

    let (zip_ok, zip_vm) = call3(
        "pairs_keys_values/3",
        ub("Pairs"),
        Value::List(vec![a("a"), a("b")]),
        Value::List(vec![i(1), i(2)]),
    );
    assert!(zip_ok);
    assert_eq!(read_var(&zip_vm, "Pairs"), Value::List(vec![
        Value::Str("-".to_string(), vec![a("a"), i(1)]),
        Value::Str("-".to_string(), vec![a("b"), i(2)]),
    ]));

    let (empty_ok, empty_vm) = call3(
        "pairs_keys_values/3", ub("Pairs"), Value::List(vec![]), Value::List(vec![]));
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "Pairs"), Value::List(vec![]));

    assert!(!call3(
        "pairs_keys_values/3",
        ub("Pairs"),
        Value::List(vec![a("a")]),
        Value::List(vec![i(1), i(2)]),
    ).0);
    assert!(!call2(
        "pairs_keys/2",
        Value::List(vec![Value::Str("other/2".to_string(), vec![a("a"), i(1)])]),
        ub("Keys"),
    ).0);

    let (rollback_ok, rollback_vm) = call3(
        "pairs_keys_values/3",
        Value::List(vec![pair(ub("X"), i(1))]),
        Value::List(vec![a("bound")]),
        Value::List(vec![i(2)]),
    );
    assert!(!rollback_ok);
    assert_eq!(read_var(&rollback_vm, "X"), ub("X"));
}

#[test]
fn test_list_utilities() {
    assert!(call2("memberchk/2", a("b"), Value::List(vec![a("a"), a("b")])).0);
    assert!(!call2("memberchk/2", a("z"), Value::List(vec![a("a"), a("b")])).0);
    let (ok, vm) = call2("last/2", Value::List(vec![i(1), i(2), i(3)]), ub("L"));
    assert!(ok);
    assert_eq!(read_var(&vm, "L"), i(3));
    let (ok2, vm2) = call3("nth0/3", i(1), Value::List(vec![a("x"), a("y")]), ub("E"));
    assert!(ok2);
    assert_eq!(read_var(&vm2, "E"), a("y"));
    let (ok3, vm3) = call3("nth1/3", i(1), Value::List(vec![a("x"), a("y")]), ub("E"));
    assert!(ok3);
    assert_eq!(read_var(&vm3, "E"), a("x"));
    assert!(!call3("nth0/3", i(5), Value::List(vec![a("x")]), ub("E")).0);
    let (ok4, vm4) = call3("numlist/3", i(2), i(5), ub("L"));
    assert!(ok4);
    assert_eq!(read_var(&vm4, "L"), Value::List(vec![i(2), i(3), i(4), i(5)]));
    assert!(!call3("numlist/3", i(5), i(2), ub("L")).0);
    let (ok5, vm5) = call3("delete/3",
        Value::List(vec![a("a"), a("b"), a("a"), a("c")]), a("a"), ub("R"));
    assert!(ok5);
    assert_eq!(read_var(&vm5, "R"), Value::List(vec![a("b"), a("c")]));
    let (ok6, vm6) = call3(
        "subtract/3",
        Value::List(vec![a("a"), a("b"), a("a"), a("c"), a("d")]),
        Value::List(vec![a("a"), a("d")]),
        ub("R"),
    );
    assert!(ok6);
    assert_eq!(read_var(&vm6, "R"), Value::List(vec![a("b"), a("c")]));
    let pair = Value::Str("pair/2".to_string(), vec![a("x"), i(1)]);
    let (ok7, vm7) = call3(
        "subtract/3",
        Value::List(vec![pair.clone(), a("keep")]),
        Value::List(vec![pair]),
        ub("R"),
    );
    assert!(ok7);
    assert_eq!(read_var(&vm7, "R"), Value::List(vec![a("keep")]));
    assert!(!call3("subtract/3", a("not_a_list"), Value::List(vec![]), ub("R")).0);
    assert!(!call3("subtract/3", Value::List(vec![]), a("not_a_list"), ub("R")).0);
}

#[test]
fn test_nth_unbound_index_enumerates() {
    let mut vm = vmnew();
    vm.set_reg("A1", ub("N"));
    vm.set_reg("A2", Value::List(vec![a("x"), a("y")]));
    vm.set_reg("A3", ub("E"));
    assert!(vm.execute_builtin("nth1/3", 3));
    let mut got = Vec::new();
    loop {
        let n = match read_var(&vm, "N") { Value::Integer(n) => n, o => panic!("N: {:?}", o) };
        let e = match read_var(&vm, "E") { Value::Atom(s) => s, o => panic!("E: {:?}", o) };
        got.push((n, e));
        if !vm.backtrack() { break; }
    }
    assert_eq!(got, vec![(1, "x".to_string()), (2, "y".to_string())]);
}

#[test]
fn test_between_check_and_plus() {
    assert!(call3("between/3", i(1), i(5), i(3)).0);
    assert!(!call3("between/3", i(1), i(5), i(9)).0);
    let (ok, vm) = call3("plus/3", i(2), i(3), ub("Z"));
    assert!(ok);
    assert_eq!(read_var(&vm, "Z"), i(5));
    let (ok2, vm2) = call3("plus/3", i(2), ub("Y"), i(7));
    assert!(ok2);
    assert_eq!(read_var(&vm2, "Y"), i(5));
    let (ok3, vm3) = call3("plus/3", ub("X"), i(3), i(7));
    assert!(ok3);
    assert_eq!(read_var(&vm3, "X"), i(4));
    assert!(call3("plus/3", i(2), i(3), i(5)).0);
    assert!(!call3("plus/3", i(2), i(3), i(6)).0);
}

#[test]
fn test_numeric_list_folds() {
    let nums = Value::List(vec![i(3), i(1), i(2)]);
    let (ok, vm) = call2("sum_list/2", nums.clone(), ub("S"));
    assert!(ok);
    assert_eq!(read_var(&vm, "S"), i(6));
    let (ok2, vm2) = call2("max_list/2", nums.clone(), ub("M"));
    assert!(ok2);
    assert_eq!(read_var(&vm2, "M"), i(3));
    let (ok3, vm3) = call2("min_list/2", nums, ub("M"));
    assert!(ok3);
    assert_eq!(read_var(&vm3, "M"), i(1));
    assert!(!call2("max_list/2", Value::List(vec![]), ub("M")).0);
}

#[test]
fn test_atom_text_ops() {
    let (ok, vm) = call2("atom_length/2", a("hello"), ub("L"));
    assert!(ok);
    assert_eq!(read_var(&vm, "L"), i(5));
    let (ok2, vm2) = call3("atom_concat/3", a("foo"), a("bar"), ub("C"));
    assert!(ok2);
    assert_eq!(read_var(&vm2, "C"), a("foobar"));
    assert!(call3("atom_concat/3", a("foo"), a("bar"), a("foobar")).0);
    assert!(!call3("atom_concat/3", a("foo"), a("bar"), a("foobaz")).0);
    let (ok3, vm3) = call2("char_code/2", a("A"), ub("C"));
    assert!(ok3);
    assert_eq!(read_var(&vm3, "C"), i(65));
    let (ok4, vm4) = call2("char_code/2", ub("Ch"), i(98));
    assert!(ok4);
    assert_eq!(read_var(&vm4, "Ch"), a("b"));
    let (string_code_ok, string_code_vm) = call3("string_code/3", i(2), a("abc"), ub("Code"));
    assert!(string_code_ok);
    assert_eq!(read_var(&string_code_vm, "Code"), i(98));
    assert!(call3("string_code/3", i(1), a("é"), i(233)).0);
    assert!(!call3("string_code/3", i(0), a("abc"), ub("Code")).0);
    assert!(!call3("string_code/3", i(4), a("abc"), ub("Code")).0);
    assert!(!call3("string_code/3", ub("Index"), a("abc"), ub("Code")).0);
    assert!(!call3("string_code/3", i(1), ub("String"), ub("Code")).0);
    assert!(!call3("string_code/3", i(1), a("abc"), i(98)).0);
    let (ok5, vm5) = call2("atom_chars/2", a("hi"), ub("Cs"));
    assert!(ok5);
    assert_eq!(read_var(&vm5, "Cs"), Value::List(vec![a("h"), a("i")]));
    let (ok6, vm6) = call2("atom_chars/2", ub("A"), Value::List(vec![a("h"), a("i")]));
    assert!(ok6);
    assert_eq!(read_var(&vm6, "A"), a("hi"));
    let (string_codes_ok, string_codes_vm) = call2("string_codes/2", a("hi"), ub("Codes"));
    assert!(string_codes_ok);
    assert_eq!(read_var(&string_codes_vm, "Codes"), Value::List(vec![i(104), i(105)]));
    let (string_codes_rev_ok, string_codes_rev_vm) = call2(
        "string_codes/2", ub("String"), Value::List(vec![i(111), i(107)]));
    assert!(string_codes_rev_ok);
    assert_eq!(read_var(&string_codes_rev_vm, "String"), a("ok"));
    let (string_chars_ok, string_chars_vm) = call2("string_chars/2", a("hi"), ub("Chars"));
    assert!(string_chars_ok);
    assert_eq!(read_var(&string_chars_vm, "Chars"), Value::List(vec![a("h"), a("i")]));
    let (string_chars_rev_ok, string_chars_rev_vm) = call2(
        "string_chars/2", ub("String"), Value::List(vec![a("o"), a("k")]));
    assert!(string_chars_rev_ok);
    assert_eq!(read_var(&string_chars_rev_vm, "String"), a("ok"));
    let (ok7, vm7) = call2("upcase_atom/2", a("aBc"), ub("U"));
    assert!(ok7);
    assert_eq!(read_var(&vm7, "U"), a("ABC"));
    let (ok8, vm8) = call2("downcase_atom/2", a("aBc"), ub("D"));
    assert!(ok8);
    assert_eq!(read_var(&vm8, "D"), a("abc"));
    let (ok9, vm9) = call2("atom_number/2", a("42"), ub("N"));
    assert!(ok9);
    assert_eq!(read_var(&vm9, "N"), i(42));
    let (ok10, vm10) = call2("atom_number/2", ub("A"), i(7));
    assert!(ok10);
    assert_eq!(read_var(&vm10, "A"), a("7"));
    assert!(!call2("atom_number/2", a("notanum"), ub("N")).0);
    let (ok_chars, vm_chars) = call2("number_chars/2", i(-42), ub("Cs"));
    assert!(ok_chars);
    assert_eq!(
        read_var(&vm_chars, "Cs"),
        Value::List(vec![a("-"), a("4"), a("2")]),
    );
    let (ok_float, vm_float) = call2(
        "number_chars/2",
        ub("N"),
        Value::List(vec![a("3"), a("."), a("5")]),
    );
    assert!(ok_float);
    assert_eq!(read_var(&vm_float, "N"), Value::Float(3.5));
    assert!(!call2("number_chars/2", ub("N"), Value::List(vec![])).0);
    assert!(!call2("number_chars/2", ub("N"), Value::List(vec![a("12")])).0);
    assert!(!call2("number_chars/2", ub("N"), Value::List(vec![a("x")])).0);
    let (ok11, vm11) = call2("atom_string/2", a("x"), ub("S"));
    assert!(ok11);
    assert_eq!(read_var(&vm11, "S"), a("x"));
    let (ok12, vm12) = call2("string_to_atom/2", a("y"), ub("A"));
    assert!(ok12);
    assert_eq!(read_var(&vm12, "A"), a("y"));
}

#[test]
fn test_text_decomposition_direct() {
    let (split_ok, split_vm) = call4(
        "split_string/4", a(" alpha; beta,,gamma "), a(";,"), a(" "), ub("Parts"));
    assert!(split_ok);
    assert_eq!(read_var(&split_vm, "Parts"), Value::List(vec![
        a("alpha"), a("beta"), a(""), a("gamma")
    ]));

    let (no_sep_ok, no_sep_vm) = call4(
        "split_string/4", a("  hello  "), a(""), a(" "), ub("Parts"));
    assert!(no_sep_ok);
    assert_eq!(read_var(&no_sep_vm, "Parts"), Value::List(vec![a("hello")]));

    let (unicode_ok, unicode_vm) = call4(
        "split_string/4", a(" α·β·γ "), a("·"), a(" "), ub("Parts"));
    assert!(unicode_ok);
    assert_eq!(read_var(&unicode_vm, "Parts"), Value::List(vec![
        a("α"), a("β"), a("γ")
    ]));
    assert!(!call4("split_string/4", ub("Text"), a(","), a(""), ub("Parts")).0);

    let (rollback_ok, rollback_vm) = call4(
        "split_string/4",
        a("a,b"),
        a(","),
        a(""),
        Value::List(vec![ub("First"), a("wrong")]),
    );
    assert!(!rollback_ok);
    assert_eq!(read_var(&rollback_vm, "First"), ub("First"));

    let (atom_split_ok, atom_split_vm) = call3(
        "atom_split/3", a("a,,b"), a(","), ub("Parts"));
    assert!(atom_split_ok);
    assert_eq!(read_var(&atom_split_vm, "Parts"), Value::List(vec![
        a("a"), a(""), a("b")
    ]));

    let (empty_ok, empty_vm) = call3("atom_split/3", a(""), a(","), ub("Parts"));
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "Parts"), Value::List(vec![a("")]));

    let (unicode_split_ok, unicode_split_vm) = call3(
        "atom_split/3", a("left·right"), a("·"), ub("Parts"));
    assert!(unicode_split_ok);
    assert_eq!(read_var(&unicode_split_vm, "Parts"), Value::List(vec![
        a("left"), a("right")
    ]));
    assert!(!call3("atom_split/3", a("a--b"), a("--"), ub("Parts")).0);
    assert!(!call3("atom_split/3", a("abc"), a(""), ub("Parts")).0);
    assert!(!call3("atom_split/3", i(123), a(","), ub("Parts")).0);
}

#[test]
fn test_atom_match_checks_direct() {
    assert!(call2("atom_starts_with/2", a("prefix-value"), a("prefix")).0);
    assert!(!call2("atom_starts_with/2", a("prefix-value"), a("value")).0);
    assert!(call2("atom_ends_with/2", a("value-suffix"), a("suffix")).0);
    assert!(!call2("atom_ends_with/2", a("value-suffix"), a("value")).0);
    assert!(call2("atom_contains/2", a("prefix-middle-suffix"), a("middle")).0);
    assert!(!call2("atom_contains/2", a("prefix-middle-suffix"), a("absent")).0);
    assert!(call2("atom_starts_with/2", a("anything"), a("")).0);
    assert!(call2("atom_ends_with/2", a("anything"), a("")).0);
    assert!(call2("atom_contains/2", a("anything"), a("")).0);
    assert!(!call2("atom_contains/2", i(123), a("2")).0);
    assert!(!call2("atom_contains/2", a("123"), i(2)).0);
}

#[test]
fn test_path_components_direct() {
    let (base_ok, base_vm) = call2("file_base_name/2", a("/tmp/file.txt"), ub("Base"));
    assert!(base_ok);
    assert_eq!(read_var(&base_vm, "Base"), a("file.txt"));
    let (plain_ok, plain_vm) = call2("file_base_name/2", a("plain"), ub("Base"));
    assert!(plain_ok);
    assert_eq!(read_var(&plain_vm, "Base"), a("plain"));
    let (trailing_ok, trailing_vm) = call2("file_base_name/2", a("/tmp/"), ub("Base"));
    assert!(trailing_ok);
    assert_eq!(read_var(&trailing_vm, "Base"), a(""));
    assert!(!call2("file_base_name/2", ub("Path"), ub("Base")).0);

    let (dir_ok, dir_vm) = call2("file_directory_name/2", a("/tmp/file.txt"), ub("Dir"));
    assert!(dir_ok);
    assert_eq!(read_var(&dir_vm, "Dir"), a("/tmp"));
    let (root_ok, root_vm) = call2("file_directory_name/2", a("/file.txt"), ub("Dir"));
    assert!(root_ok);
    assert_eq!(read_var(&root_vm, "Dir"), a("/"));
    let (relative_ok, relative_vm) = call2("file_directory_name/2", a("file.txt"), ub("Dir"));
    assert!(relative_ok);
    assert_eq!(read_var(&relative_vm, "Dir"), a("."));
    let (empty_ok, empty_vm) = call2("file_directory_name/2", a(""), ub("Dir"));
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "Dir"), a("."));
}

#[test]
fn test_file_name_extension_direct() {
    let (split_ok, split_vm) = call3(
        "file_name_extension/3", ub("Base"), ub("Ext"), a("/tmp/archive.tar.gz"));
    assert!(split_ok);
    assert_eq!(read_var(&split_vm, "Base"), a("/tmp/archive.tar"));
    assert_eq!(read_var(&split_vm, "Ext"), a("gz"));

    let (hidden_ok, hidden_vm) = call3(
        "file_name_extension/3", ub("Base"), ub("Ext"), a("/tmp/.hidden"));
    assert!(hidden_ok);
    assert_eq!(read_var(&hidden_vm, "Base"), a("/tmp/.hidden"));
    assert_eq!(read_var(&hidden_vm, "Ext"), a(""));

    let (hidden_ext_ok, hidden_ext_vm) = call3(
        "file_name_extension/3", ub("Base"), ub("Ext"), a("/tmp/.hidden.txt"));
    assert!(hidden_ext_ok);
    assert_eq!(read_var(&hidden_ext_vm, "Base"), a("/tmp/.hidden"));
    assert_eq!(read_var(&hidden_ext_vm, "Ext"), a("txt"));

    let (join_ok, join_vm) = call3(
        "file_name_extension/3", a("/tmp/report"), a("csv"), ub("File"));
    assert!(join_ok);
    assert_eq!(read_var(&join_vm, "File"), a("/tmp/report.csv"));
    let (empty_ext_ok, empty_ext_vm) = call3(
        "file_name_extension/3", a("/tmp/report"), a(""), ub("File"));
    assert!(empty_ext_ok);
    assert_eq!(read_var(&empty_ext_vm, "File"), a("/tmp/report"));

    let (rollback_ok, rollback_vm) = call3(
        "file_name_extension/3", ub("Base"), a("wrong"), a("name.txt"));
    assert!(!rollback_ok);
    assert_eq!(read_var(&rollback_vm, "Base"), ub("Base"));
    assert!(!call3("file_name_extension/3", ub("Base"), a("txt"), ub("File")).0);
}

#[test]
fn test_path_queries_and_join_direct() {
    assert!(call1("is_absolute_file_name/1", a("/tmp/file")).0);
    assert!(!call1("is_absolute_file_name/1", a("tmp/file")).0);
    assert!(!call1("is_absolute_file_name/1", a("")).0);
    assert!(!call1("is_absolute_file_name/1", i(1)).0);

    let (join_ok, join_vm) = call3("path_join/3", a("/tmp/base"), a("child"), ub("Full"));
    assert!(join_ok);
    assert_eq!(read_var(&join_vm, "Full"), a("/tmp/base/child"));
    let (slash_ok, slash_vm) = call3("path_join/3", a("/tmp/base/"), a("child"), ub("Full"));
    assert!(slash_ok);
    assert_eq!(read_var(&slash_vm, "Full"), a("/tmp/base/child"));
    let (absolute_ok, absolute_vm) = call3(
        "path_join/3", a("/ignored"), a("/absolute/file"), ub("Full"));
    assert!(absolute_ok);
    assert_eq!(read_var(&absolute_vm, "Full"), a("/absolute/file"));
    let (empty_base_ok, empty_base_vm) = call3("path_join/3", a(""), a("child"), ub("Full"));
    assert!(empty_base_ok);
    assert_eq!(read_var(&empty_base_vm, "Full"), a("child"));
    let (empty_rel_ok, empty_rel_vm) = call3("path_join/3", a("base"), a(""), ub("Full"));
    assert!(empty_rel_ok);
    assert_eq!(read_var(&empty_rel_vm, "Full"), a("base"));
    assert!(!call3("path_join/3", a("base"), ub("Rel"), ub("Full")).0);
    assert!(!call3("path_join/3", a("base"), a("child"), a("wrong")).0);
}

#[test]
fn test_term_variables_direct() {
    let term = Value::Str(
        "outer/3".to_string(),
        vec![
            ub("X"),
            Value::Str("inner/3".to_string(), vec![ub("Y"), ub("X"), ub("Z")]),
            Value::List(vec![ub("Z"), ub("Y")]),
        ],
    );
    let (ok, vm) = call2("term_variables/2", term, ub("Vars"));
    assert!(ok);
    assert_eq!(
        read_var(&vm, "Vars"),
        Value::List(vec![ub("X"), ub("Y"), ub("Z")]),
    );
    let (ground_ok, ground_vm) = call2(
        "term_variables/2",
        Value::Str("ground/2".to_string(), vec![a("x"), i(1)]),
        ub("Vars"),
    );
    assert!(ground_ok);
    assert_eq!(read_var(&ground_vm, "Vars"), Value::List(vec![]));
    assert!(!call2("term_variables/2", ub("X"), Value::List(vec![])).0);
}

#[test]
fn test_numbervars_direct() {
    let term = Value::Str(
        "outer/3".to_string(),
        vec![
            ub("X"),
            Value::Str("inner/2".to_string(), vec![ub("Y"), ub("X")]),
            ub("Z"),
        ],
    );
    let (ok, vm) = call3("numbervars/3", term, i(3), ub("End"));
    assert!(ok);
    assert_eq!(read_var(&vm, "X"), Value::Str("$VAR".to_string(), vec![i(3)]));
    assert_eq!(read_var(&vm, "Y"), Value::Str("$VAR".to_string(), vec![i(4)]));
    assert_eq!(read_var(&vm, "Z"), Value::Str("$VAR".to_string(), vec![i(5)]));
    assert_eq!(read_var(&vm, "End"), i(6));

    let (ground_ok, ground_vm) = call3(
        "numbervars/3",
        Value::Str("ground/2".to_string(), vec![a("x"), i(1)]),
        i(7),
        ub("End"),
    );
    assert!(ground_ok);
    assert_eq!(read_var(&ground_vm, "End"), i(7));
    assert!(!call3("numbervars/3", ub("X"), ub("Start"), ub("End")).0);

    let (mismatch_ok, mismatch_vm) = call3(
        "numbervars/3",
        Value::Str("pair/2".to_string(), vec![ub("X"), ub("Y")]),
        i(0),
        i(9),
    );
    assert!(!mismatch_ok);
    assert_eq!(read_var(&mismatch_vm, "X"), ub("X"));
    assert_eq!(read_var(&mismatch_vm, "Y"), ub("Y"));

    let (overflow_ok, overflow_vm) =
        call3("numbervars/3", ub("X"), i(i64::MAX), ub("End"));
    assert!(!overflow_ok);
    assert_eq!(read_var(&overflow_vm, "X"), ub("X"));
}

#[test]
fn test_unifiable_direct() {
    let display_eq = |left: Value, right: Value| {
        Value::Str("=".to_string(), vec![left, right])
    };
    let raw_eq = |left: Value, right: Value| {
        Value::Str("=/2".to_string(), vec![left, right])
    };

    let (simple_ok, simple_vm) = call3(
        "unifiable/3",
        ub("X"),
        Value::Str("foo/1".to_string(), vec![a("a")]),
        ub("Bindings"),
    );
    assert!(simple_ok);
    assert_eq!(read_var(&simple_vm, "X"), ub("X"));
    assert_eq!(
        read_var(&simple_vm, "Bindings"),
        Value::List(vec![display_eq(
            ub("X"),
            Value::Str("foo".to_string(), vec![a("a")]),
        )]),
    );

    let (two_ok, two_vm) = call3(
        "unifiable/3",
        Value::Str("p/2".to_string(), vec![ub("X"), ub("Y")]),
        Value::Str("p/2".to_string(), vec![i(1), i(2)]),
        ub("Bindings"),
    );
    assert!(two_ok);
    assert_eq!(
        read_var(&two_vm, "Bindings"),
        Value::List(vec![display_eq(ub("X"), i(1)), display_eq(ub("Y"), i(2))]),
    );
    assert_eq!(read_var(&two_vm, "X"), ub("X"));
    assert_eq!(read_var(&two_vm, "Y"), ub("Y"));

    let (alias_ok, alias_vm) = call3("unifiable/3", ub("X"), ub("Y"), ub("Bindings"));
    assert!(alias_ok);
    assert_eq!(
        read_var(&alias_vm, "Bindings"),
        Value::List(vec![display_eq(ub("X"), ub("Y"))]),
    );
    assert_eq!(read_var(&alias_vm, "X"), ub("X"));
    assert_eq!(read_var(&alias_vm, "Y"), ub("Y"));

    let (ground_ok, ground_vm) = call3("unifiable/3", a("same"), a("same"), ub("B"));
    assert!(ground_ok);
    assert_eq!(read_var(&ground_vm, "B"), Value::List(vec![]));

    let (fail_ok, fail_vm) = call3(
        "unifiable/3",
        Value::Str("p/2".to_string(), vec![ub("X"), a("a")]),
        Value::Str("p/2".to_string(), vec![i(1), a("b")]),
        ub("Bindings"),
    );
    assert!(!fail_ok);
    assert_eq!(read_var(&fail_vm, "X"), ub("X"));

    let constrained = Value::List(vec![raw_eq(ub("X"), a("foo"))]);
    let (constrained_ok, constrained_vm) =
        call3("unifiable/3", ub("X"), ub("Y"), constrained);
    assert!(constrained_ok);
    assert_eq!(read_var(&constrained_vm, "X"), ub("X"));
    assert_eq!(read_var(&constrained_vm, "Y"), a("foo"));

    let mismatched = Value::List(vec![
        raw_eq(ub("Capture"), i(1)),
        raw_eq(ub("Y"), i(9)),
    ]);
    let (mismatch_ok, mismatch_vm) = call3(
        "unifiable/3",
        Value::Str("p/2".to_string(), vec![ub("X"), ub("Y")]),
        Value::Str("p/2".to_string(), vec![i(1), i(2)]),
        mismatched,
    );
    assert!(!mismatch_ok);
    assert_eq!(read_var(&mismatch_vm, "X"), ub("X"));
    assert_eq!(read_var(&mismatch_vm, "Y"), ub("Y"));
    assert_eq!(read_var(&mismatch_vm, "Capture"), ub("Capture"));
}

#[test]
fn test_variant_equivalence_direct() {
    assert!(call2("=@=/2", ub("X"), ub("Y")).0);
    assert!(call2("=@=/2", a("same"), a("same")).0);
    assert!(!call2("=@=/2", a("left"), a("right")).0);

    let shared_left = Value::Str("f/2".to_string(), vec![ub("X"), ub("X")]);
    let shared_right = Value::Str("f/2".to_string(), vec![ub("A"), ub("A")]);
    let distinct_right = Value::Str("f/2".to_string(), vec![ub("A"), ub("B")]);
    let (shared_ok, shared_vm) = call2("=@=/2", shared_left.clone(), shared_right);
    assert!(shared_ok);
    assert_eq!(read_var(&shared_vm, "X"), ub("X"));
    assert_eq!(read_var(&shared_vm, "A"), ub("A"));
    assert!(!call2("=@=/2", shared_left.clone(), distinct_right.clone()).0);
    assert!(!call2("=@=/2", distinct_right, shared_left).0);

    let nested_left = Value::Str(
        "outer/2".to_string(),
        vec![ub("X"), Value::List(vec![ub("Y"), ub("X")])],
    );
    let nested_right = Value::Str(
        "outer/2".to_string(),
        vec![ub("A"), Value::List(vec![ub("B"), ub("A")])],
    );
    assert!(call2("=@=/2", nested_left.clone(), nested_right.clone()).0);
    assert!(!call2(
        "=@=/2",
        nested_left.clone(),
        Value::Str("other/2".to_string(), vec![ub("A"), ub("B")]),
    ).0);
    assert!(call2("\\\\=@=/2", nested_left, a("different")).0);
    assert!(!call2("\\\\=@=/2", nested_right.clone(), nested_right).0);

    assert!(call2("=@=/2", Value::List(vec![]), a("[]")).0);
    assert!(call2(
        "=@=/2",
        Value::List(vec![ub("X")]),
        Value::Str("[|]/2".to_string(), vec![ub("Y"), a("[]")]),
    ).0);
}

#[test]
fn test_intersection_direct() {
    let (ok, vm) = call3(
        "intersection/3",
        Value::List(vec![a("a"), a("b"), a("c"), a("b")]),
        Value::List(vec![a("b"), a("d")]),
        ub("Common"),
    );
    assert!(ok);
    assert_eq!(read_var(&vm, "Common"), Value::List(vec![a("b"), a("b")]));

    let (empty_ok, empty_vm) = call3(
        "intersection/3",
        Value::List(vec![]),
        Value::List(vec![a("a")]),
        ub("Common"),
    );
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "Common"), Value::List(vec![]));
    assert!(!call3("intersection/3", a("bad"), Value::List(vec![]), ub("R")).0);
    assert!(!call3("intersection/3", Value::List(vec![]), a("bad"), ub("R")).0);

    let (vars_ok, vars_vm) = call3(
        "intersection/3",
        Value::List(vec![ub("X"), ub("Y")]),
        Value::List(vec![a("a")]),
        ub("Common"),
    );
    assert!(vars_ok);
    assert_eq!(read_var(&vars_vm, "X"), a("a"));
    assert_eq!(read_var(&vars_vm, "Y"), a("a"));
    assert_eq!(read_var(&vars_vm, "Common"), Value::List(vec![a("a"), a("a")]));

    let (retry_ok, retry_vm) = call3(
        "intersection/3",
        Value::List(vec![Value::Str("f/1".to_string(), vec![ub("X")])]),
        Value::List(vec![
            Value::Str("g/1".to_string(), vec![a("wrong")]),
            Value::Str("f/1".to_string(), vec![a("right")]),
        ]),
        ub("Common"),
    );
    assert!(retry_ok);
    assert_eq!(read_var(&retry_vm, "X"), a("right"));

    let (miss_ok, miss_vm) = call3(
        "intersection/3",
        Value::List(vec![Value::Str("f/2".to_string(), vec![ub("X"), a("a")])]),
        Value::List(vec![Value::Str("f/2".to_string(), vec![a("bound"), a("b")])]),
        ub("Common"),
    );
    assert!(miss_ok);
    assert_eq!(read_var(&miss_vm, "X"), ub("X"));
    assert_eq!(read_var(&miss_vm, "Common"), Value::List(vec![]));

    let (mismatch_ok, mismatch_vm) = call3(
        "intersection/3",
        Value::List(vec![ub("X")]),
        Value::List(vec![a("a")]),
        Value::List(vec![]),
    );
    assert!(!mismatch_ok);
    assert_eq!(read_var(&mismatch_vm, "X"), ub("X"));
}

#[test]
fn test_union_direct() {
    let (ok, vm) = call3(
        "union/3",
        Value::List(vec![a("a"), a("b"), a("c")]),
        Value::List(vec![a("b"), a("d"), a("e")]),
        ub("Union"),
    );
    assert!(ok);
    assert_eq!(
        read_var(&vm, "Union"),
        Value::List(vec![a("a"), a("b"), a("c"), a("d"), a("e")]),
    );

    let (dupes_ok, dupes_vm) = call3(
        "union/3",
        Value::List(vec![a("a"), a("a"), a("b")]),
        Value::List(vec![a("a"), a("c")]),
        ub("Union"),
    );
    assert!(dupes_ok);
    assert_eq!(
        read_var(&dupes_vm, "Union"),
        Value::List(vec![a("a"), a("a"), a("b"), a("c")]),
    );

    let (empty_ok, empty_vm) = call3(
        "union/3",
        Value::List(vec![]),
        Value::List(vec![a("x"), a("y")]),
        ub("Union"),
    );
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "Union"), Value::List(vec![a("x"), a("y")]));
    assert!(!call3("union/3", a("bad"), Value::List(vec![]), ub("R")).0);
    assert!(!call3("union/3", Value::List(vec![]), a("bad"), ub("R")).0);

    let (vars_ok, vars_vm) = call3(
        "union/3",
        Value::List(vec![ub("X")]),
        Value::List(vec![a("a"), a("b")]),
        ub("Union"),
    );
    assert!(vars_ok);
    assert_eq!(read_var(&vars_vm, "X"), a("a"));
    assert_eq!(read_var(&vars_vm, "Union"), Value::List(vec![a("a"), a("b")]));

    let (miss_ok, miss_vm) = call3(
        "union/3",
        Value::List(vec![Value::Str("f/2".to_string(), vec![ub("X"), a("a")])]),
        Value::List(vec![
            Value::Str("f/2".to_string(), vec![a("bound"), a("b")]),
            a("keep"),
        ]),
        ub("Union"),
    );
    assert!(miss_ok);
    assert_eq!(read_var(&miss_vm, "X"), ub("X"));
    assert_eq!(
        read_var(&miss_vm, "Union"),
        Value::List(vec![
            Value::Str("f".to_string(), vec![ub("X"), a("a")]),
            Value::Str("f".to_string(), vec![a("bound"), a("b")]),
            a("keep"),
        ]),
    );

    let (mismatch_ok, mismatch_vm) = call3(
        "union/3",
        Value::List(vec![ub("X")]),
        Value::List(vec![a("a")]),
        Value::List(vec![]),
    );
    assert!(!mismatch_ok);
    assert_eq!(read_var(&mismatch_vm, "X"), ub("X"));
}

#[test]
fn test_list_to_set_direct() {
    let (ok, vm) = call2(
        "list_to_set/2",
        Value::List(vec![a("a"), a("b"), a("c"), a("a"), a("b"), a("d")]),
        ub("Set"),
    );
    assert!(ok);
    assert_eq!(
        read_var(&vm, "Set"),
        Value::List(vec![a("a"), a("b"), a("c"), a("d")]),
    );

    let (empty_ok, empty_vm) =
        call2("list_to_set/2", Value::List(vec![]), ub("Set"));
    assert!(empty_ok);
    assert_eq!(read_var(&empty_vm, "Set"), Value::List(vec![]));
    assert!(!call2("list_to_set/2", a("bad"), ub("Set")).0);

    let pair = Value::Str("pair/2".to_string(), vec![a("x"), i(1)]);
    let (compound_ok, compound_vm) = call2(
        "list_to_set/2",
        Value::List(vec![pair.clone(), a("keep"), pair]),
        ub("Set"),
    );
    assert!(compound_ok);
    assert_eq!(
        read_var(&compound_vm, "Set"),
        Value::List(vec![
            Value::Str("pair".to_string(), vec![a("x"), i(1)]),
            a("keep"),
        ]),
    );

    let (vars_ok, vars_vm) = call2(
        "list_to_set/2",
        Value::List(vec![ub("X"), a("a"), a("b")]),
        ub("Set"),
    );
    assert!(vars_ok);
    assert_eq!(read_var(&vars_vm, "X"), a("a"));
    assert_eq!(read_var(&vars_vm, "Set"), Value::List(vec![a("a"), a("b")]));

    let (miss_ok, miss_vm) = call2(
        "list_to_set/2",
        Value::List(vec![
            Value::Str("f/2".to_string(), vec![ub("X"), a("a")]),
            Value::Str("f/2".to_string(), vec![a("bound"), a("b")]),
        ]),
        ub("Set"),
    );
    assert!(miss_ok);
    assert_eq!(read_var(&miss_vm, "X"), ub("X"));
    assert_eq!(
        read_var(&miss_vm, "Set"),
        Value::List(vec![
            Value::Str("f".to_string(), vec![ub("X"), a("a")]),
            Value::Str("f".to_string(), vec![a("bound"), a("b")]),
        ]),
    );

    let (mismatch_ok, mismatch_vm) = call2(
        "list_to_set/2",
        Value::List(vec![ub("X"), a("a")]),
        Value::List(vec![]),
    );
    assert!(!mismatch_ok);
    assert_eq!(read_var(&mismatch_vm, "X"), ub("X"));
}

#[test]
fn test_ground() {
    assert!(call1("ground/1", a("x")).0);
    assert!(call1("ground/1", Value::List(vec![i(1), a("b")])).0);
    assert!(!call1("ground/1", ub("V")).0);
    assert!(!call1("ground/1", Value::List(vec![i(1), ub("V")])).0);
    assert!(!call1("ground/1", Value::Str("f/1".to_string(), vec![ub("V")])).0);
}

#[test]
fn test_atomic_direct() {
    assert!(call1("atomic/1", a("x")).0);
    assert!(call1("atomic/1", i(1)).0);
    assert!(call1("atomic/1", Value::Float(1.5)).0);
    assert!(call1("atomic/1", Value::List(vec![])).0);
    assert!(!call1("atomic/1", Value::List(vec![a("x")])).0);
    assert!(!call1("atomic/1", Value::Str("f".to_string(), vec![a("x")])).0);
    assert!(!call1("atomic/1", ub("X")).0);
}

#[test]
fn test_tab_direct() {
    assert!(call1("tab/1", i(3)).0);
    assert!(call1("tab/1", i(0)).0);
    assert!(!call1("tab/1", i(-1)).0);
    assert!(!call1("tab/1", ub("N")).0);
    assert!(!call1("tab/1", a("three")).0);
}

#[test]
fn test_single_term_output_direct() {
    assert!(call1("put_char/1", a("x")).0);
    assert!(call1("put_char/1", a("é")).0);
    assert!(!call1("put_char/1", a("")).0);
    assert!(!call1("put_char/1", a("xy")).0);
    assert!(!call1("put_char/1", i(120)).0);
    assert!(!call1("put_char/1", ub("C")).0);

    assert!(call1("put_code/1", i(121)).0);
    assert!(!call1("put_code/1", i(-1)).0);
    assert!(!call1("put_code/1", i(0x110000)).0);
    assert!(!call1("put_code/1", i(0xd800)).0);
    assert!(!call1("put_code/1", a("x")).0);
    assert!(!call1("put_code/1", ub("Code")).0);

    assert!(call1("write_canonical/1",
        Value::Str("node/2".to_string(), vec![a("has space"), i(7)])).0);
    let mut vm = vmnew();
    assert!(!vm.execute_builtin("write_canonical/1", 1));
}

#[test]
fn test_format_direct() {
    let args = Value::List(vec![
        Value::Str("node".to_string(), vec![i(1)]),
        a("hello"),
        i(42),
        Value::List(vec![i(111), i(107)]),
    ]);
    let destination = Value::Str("atom".to_string(), vec![ub("Out")]);
    let (ok, vm) = call3("format/3", destination,
        a("~w|~a|~d|~s|~t|~~"), args);
    assert!(ok);
    assert_eq!(read_var(&vm, "Out"),
        a(&format!("node(1)|hello|42|ok|{}|~", char::from(9))));

    let string_dest = Value::Str("string/1".to_string(), vec![ub("Out")]);
    let (string_ok, string_vm) = call3("format/3", string_dest,
        a("~p"), Value::List(vec![Value::List(vec![a("a"), i(2)])]));
    assert!(string_ok);
    assert_eq!(read_var(&string_vm, "Out"), a("[a, 2]"));

    let codes_dest = Value::Str("codes".to_string(), vec![ub("Codes")]);
    let (codes_ok, codes_vm) = call3("format/3", codes_dest,
        a("ab"), Value::List(vec![]));
    assert!(codes_ok);
    assert_eq!(read_var(&codes_vm, "Codes"), Value::List(vec![i(97), i(98)]));

    assert!(call3("format/3",
        Value::Str("atom".to_string(), vec![a("ok")]),
        a("ok"), Value::List(vec![])).0);
    assert!(!call3("format/3",
        Value::Str("atom".to_string(), vec![a("wrong")]),
        a("ok"), Value::List(vec![])).0);
    assert!(!call3("format/3", a("unknown"), a("ok"), Value::List(vec![])).0);
    assert!(!call2("format/2", a("~w"), Value::List(vec![])).0);
    assert!(!call2("format/2", a("ok"), i(7)).0);
    assert!(!call2("format/2", Value::Float(2.5), Value::List(vec![])).0);
}

#[test]
fn test_filesystem_query_direct() {
    assert!(call1("exists_file/1", a("Cargo.toml")).0);
    assert!(!call1("exists_file/1", a("src/bin")).0);
    assert!(!call1("exists_file/1", a("definitely_missing_rust_wam_path")).0);
    assert!(!call1("exists_file/1", ub("Path")).0);
    assert!(!call1("exists_file/1", i(7)).0);

    assert!(call1("exists_directory/1", a("src/bin")).0);
    assert!(!call1("exists_directory/1", a("Cargo.toml")).0);
    assert!(!call1("exists_directory/1", a("definitely_missing_rust_wam_path")).0);
    assert!(!call1("exists_directory/1", ub("Path")).0);
    assert!(!call1("exists_directory/1", i(7)).0);

    let expected = Value::List(vec![a("."), a(".."), a("output_probe.rs")]);
    let (ok, vm) = call2("directory_files/2", a("src/bin"), ub("Files"));
    assert!(ok);
    assert_eq!(read_var(&vm, "Files"), expected);
    assert!(!call2("directory_files/2", a("Cargo.toml"), ub("Files")).0);
    assert!(!call2("directory_files/2", a("definitely_missing_rust_wam_path"), ub("Files")).0);
    assert!(!call2("directory_files/2", ub("Path"), ub("Files")).0);
    assert!(!call2("directory_files/2", i(7), ub("Files")).0);

    let (matched, _) = call2("directory_files/2", a("src/bin"), expected.clone());
    assert!(matched);
    let (mismatched, mismatch_vm) = call2("directory_files/2", a("src/bin"),
        Value::List(vec![ub("Head"), a("wrong"), ub("Tail")]));
    assert!(!mismatched);
    assert_eq!(read_var(&mismatch_vm, "Head"), ub("Head"));
}

#[test]
fn test_file_metadata_direct() {
    let (size_ok, size_vm) = call2("size_file/2", a("Cargo.toml"), ub("Size"));
    assert!(size_ok);
    let size = match read_var(&size_vm, "Size") {
        Value::Integer(size) => size,
        other => panic!("expected integer file size, got {:?}", other),
    };
    assert!(size > 0);
    assert!(call2("size_file/2", a("Cargo.toml"), i(size)).0);
    assert!(!call2("size_file/2", a("Cargo.toml"), i(size + 1)).0);
    assert!(!call2("size_file/2", a("definitely_missing_rust_wam_path"), ub("Size")).0);
    assert!(!call2("size_file/2", ub("Path"), ub("Size")).0);
    assert!(!call2("size_file/2", i(7), ub("Size")).0);

    let (time_ok, time_vm) = call2("time_file/2", a("Cargo.toml"), ub("Time"));
    assert!(time_ok);
    let time = match read_var(&time_vm, "Time") {
        Value::Float(time) => time,
        other => panic!("expected float modification time, got {:?}", other),
    };
    assert!(time.is_finite() && time > 0.0);
    assert!(call2("time_file/2", a("Cargo.toml"), Value::Float(time)).0);
    assert!(!call2("time_file/2", a("Cargo.toml"), Value::Float(time + 1.0)).0);
    assert!(!call2("time_file/2", a("Cargo.toml"), i(time as i64)).0);
    assert!(!call2("time_file/2", a("definitely_missing_rust_wam_path"), ub("Time")).0);
    assert!(!call2("time_file/2", ub("Path"), ub("Time")).0);
    assert!(!call2("time_file/2", i(7), ub("Time")).0);
}

#[test]
fn test_system_queries_direct() {
    let (time_ok, time_vm) = call1("get_time/1", ub("Time"));
    assert!(time_ok);
    match read_var(&time_vm, "Time") {
        Value::Float(time) => assert!(time.is_finite() && time > 0.0),
        other => panic!("expected float wall time, got {:?}", other),
    }
    assert!(!call1("get_time/1", Value::Float(0.0)).0);
    assert!(!call1("get_time/1", i(0)).0);
    assert!(!vmnew().execute_builtin("get_time/1", 1));

    let pid = i64::from(std::process::id());
    let (pid_ok, pid_vm) = call1("getpid/1", ub("Pid"));
    assert!(pid_ok);
    assert_eq!(read_var(&pid_vm, "Pid"), i(pid));
    assert!(call1("getpid/1", i(pid)).0);
    assert!(!call1("getpid/1", i(pid + 1)).0);
    assert!(!call1("getpid/1", a("pid")).0);
    assert!(!vmnew().execute_builtin("getpid/1", 1));

    let expected_path = std::env::var("PATH").expect("PATH must be available to cargo test");
    let (env_ok, env_vm) = call2("getenv/2", a("PATH"), ub("Value"));
    assert!(env_ok);
    assert_eq!(read_var(&env_vm, "Value"), a(&expected_path));
    assert!(call2("getenv/2", a("PATH"), a(&expected_path)).0);
    assert!(!call2("getenv/2", a("PATH"), a("definitely_not_the_path")).0);
    assert!(!call2("getenv/2", a("UNIFYWEAVER_ENV_MUST_NOT_EXIST_7F3A"), ub("Value")).0);
    assert!(!call2("getenv/2", ub("Name"), ub("Value")).0);
    assert!(!call2("getenv/2", i(7), ub("Value")).0);
    let mut missing_output = vmnew();
    missing_output.set_reg("A1", a("PATH"));
    assert!(!missing_output.execute_builtin("getenv/2", 2));
}

#[cfg(unix)]
#[test]
fn test_process_commands_direct() {
    assert!(call1("shell/1", a("exit 0")).0);
    assert!(!call1("shell/1", a("exit 3")).0);
    assert!(!call1("shell/1", i(7)).0);
    assert!(!vmnew().execute_builtin("shell/1", 1));

    let (status_ok, status_vm) = call2("shell/2", a("exit 23"), ub("Status"));
    assert!(status_ok);
    assert_eq!(read_var(&status_vm, "Status"), i(23));
    assert!(call2("shell/2", a("exit 0"), i(0)).0);
    assert!(!call2("shell/2", a("exit 9"), i(8)).0);
    assert!(!call2("shell/2", ub("Command"), ub("Status")).0);

    let (capture_ok, capture_vm) = call2(
        "system_to_atom/2", a("printf direct-output"), ub("Output"));
    assert!(capture_ok);
    assert_eq!(read_var(&capture_vm, "Output"), a("direct-output"));
    assert!(call2("system_to_atom/2", a("exit 5"), a("")).0,
        "stdout capture succeeds independently of the exit status");
    assert!(!call2("system_to_atom/2", a("printf wrong"), a("right")).0);
    assert!(!call2("system_to_atom/2", i(7), ub("Output")).0);

    let fixture = std::path::Path::new("process_command_direct_fixture.txt");
    let _ = std::fs::remove_file(fixture);
    assert!(call2(
        "atom_to_system/2",
        a("cat > process_command_direct_fixture.txt"),
        a("direct-input")).0);
    assert_eq!(std::fs::read_to_string(fixture).unwrap(), "direct-input");
    assert!(!call2("atom_to_system/2", a("exit 4"), a("input")).0);
    assert!(!call2("atom_to_system/2", i(7), a("input")).0);
    assert!(!call2("atom_to_system/2", a("cat"), i(7)).0);
    std::fs::remove_file(fixture).unwrap();
}

#[test]
fn test_sleep_direct() {
    assert!(call1("sleep/1", i(0)).0);
    assert!(call1("sleep/1", Value::Float(0.0)).0);

    let started = std::time::Instant::now();
    assert!(call1("sleep/1", Value::Float(0.005)).0);
    assert!(started.elapsed() >= std::time::Duration::from_millis(4));

    assert!(!call1("sleep/1", i(-1)).0);
    assert!(!call1("sleep/1", Value::Float(-0.001)).0);
    assert!(!call1("sleep/1", Value::Float(f64::NAN)).0);
    assert!(!call1("sleep/1", Value::Float(f64::INFINITY)).0);
    assert!(!call1("sleep/1", a("soon")).0);
    assert!(!call1("sleep/1", ub("Seconds")).0);
}

#[test]
fn test_environment_mutation_direct() {
    let key = "UNIFYWEAVER_RUST_WAM_DIRECT_ENV_5A27";
    std::env::remove_var(key);

    assert!(call2("setenv/2", a(key), a("first")).0);
    assert_eq!(std::env::var(key).unwrap(), "first");
    assert!(call2("setenv/2", a(key), a("second")).0);
    assert_eq!(std::env::var(key).unwrap(), "second");
    assert!(call1("unsetenv/1", a(key)).0);
    assert!(std::env::var_os(key).is_none());
    assert!(call1("unsetenv/1", a(key)).0,
        "unsetenv/1 is idempotent for a missing name");

    let nul_name = String::from_utf8(vec![66, 65, 68, 0, 78, 65, 77, 69]).unwrap();
    let nul_value = String::from_utf8(vec![98, 97, 100, 0, 118, 97, 108]).unwrap();
    assert!(!call2("setenv/2", a(""), a("value")).0);
    assert!(!call2("setenv/2", a("BAD=NAME"), a("value")).0);
    assert!(!call2("setenv/2", a(&nul_name), a("value")).0);
    assert!(!call2("setenv/2", a(key), a(&nul_value)).0);
    assert!(!call2("setenv/2", ub("Name"), a("value")).0);
    assert!(!call2("setenv/2", a(key), i(7)).0);
    assert!(!call1("unsetenv/1", a("")).0);
    assert!(!call1("unsetenv/1", a("BAD=NAME")).0);
    assert!(!call1("unsetenv/1", a(&nul_name)).0);
    assert!(!call1("unsetenv/1", i(7)).0);

    std::env::remove_var(key);
}
',
        setup_call_cleanup(
            open(TestPath, write, Out, [encoding(utf8)]),
            write(Out, TestContent),
            close(Out)),
        format(atom(Cmd), 'cd "~w" && cargo test -- --nocapture 2>&1', [TmpDir]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Stream)), process(Pid)]),
        read_string(Stream, _, Output),
        close(Stream),
        process_wait(Pid, exit(ExitCode)),
        (   ExitCode == 0,
            sub_string(Output, _, _, _, "test result: ok")
        ->  run_output_probe(TmpDir, ProbeExitCode, ProbeOutput),
            ExpectedOutput = "xy'two words'node('has space', 7)F1~|node(1)/hello/42/ok\n",
            (   ProbeExitCode == 0,
                ProbeOutput == ExpectedOutput
            ->  pass(Test)
            ;   format('--- output probe ---~nexpected: ~q~nactual:   ~q~n--- end ---~n',
                       [ExpectedOutput, ProbeOutput]),
                fail_test(Test, 'output probe failed')
            )
        ;   format('--- cargo test output ---~n~w~n--- end ---~n', [Output]),
            fail_test(Test, 'cargo test failed')
        )
    ).

run_tests :-
    format('=== WAM-Rust builtin parity tests ===~n'),
    test_builtin_parity_execution,
    (   test_failed
    ->  format('~n=== SOME TESTS FAILED ===~n'), halt(1)
    ;   format('~n=== ALL TESTS PASSED ===~n')
    ).
