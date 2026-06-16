% test_wam_rust_switch_on_constant_keydrop_exec.pl
%
% Regression for the switch_on_constant key-dropping bug: the WAM first-arg index
% (switch_on_constant / *_pc) is emitted in CLAUSE order (k0,k1,..,k9,k10,..),
% which is NOT lexicographically sorted (k10 < k2). The runtime looked up Atom
% keys with binary_search_by_key (which assumes a lexically sorted table), so it
% mis-navigated and dropped keys whose clause order differs from lexical order
% (e.g. k5 -> no match). Fixed to a linear scan by value equality.
%
% Forces the T4/WAM shared-table path with fact_table_inline(false) (T9 fact
% tables are the default in-range and would otherwise mask this). cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic ek/2.
% 25 unique atom keys in clause/numeric order (k0..k24): clause order != lexical
% order, which is exactly what tripped the binary search.
mk :- forall(between(0, 24, N), ( atom_concat(k, N, K), assertz(ek(K, N)) )).
:- initialization(mk).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_switch_on_constant_keydrop_exec).

test(every_key_resolves,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_switch_keydrop'))]) :-
    Dir = 'output/test_switch_keydrop',
    safe_rmdir(Dir),
    % fact_table_inline(false) -> the WAM shared table + switch_on_constant path
    once(write_wam_rust_project(
        [user:ek/2], [module_name(sk), fact_table_inline(false)], Dir)),
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    assertion(sub_string(Src, _, _, _, "SwitchOnConstant")),  % the indexed path
    assertion(\+ sub_string(Src, _, _, _, "Strategy: fact_table")),  % not T9
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/sk_test.rs', TestPath),
    TestSrc = '
use sk::value::Value;
use sk::state::WamState;
use sk::{ek_2, shared_wam_program};

fn lookup(n: i64) -> Option<i64> {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    if ek_2(&mut vm, Value::Atom(format!("k{}", n)), Value::Unbound("X".into())) {
        if let Some(Value::Integer(v)) = vm.bindings.get("X").cloned() { return Some(v); }
    }
    None
}

#[test]
fn every_key_resolves() {
    // Before the fix, keys like k5/k10/k11 (clause order != lexical order) were
    // dropped by the binary search. Every key must now resolve to its own value.
    for n in 0..25 {
        assert_eq!(lookup(n), Some(n), "key k{} must resolve to {}", n, n);
    }
    // an unindexed key must still fail cleanly
    assert_eq!(lookup(999), None, "unknown key fails");
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test sk_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[switch keydrop exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_switch_on_constant_keydrop_exec).
