% test_wam_rust_repeated_key_fact_exec.pl
%
% Regression for the first-argument-indexing bug: a fact predicate with a
% REPEATED first-argument key (rk(10,3). rk(10,4).) failed to enumerate when the
% second arg was unbound. Root cause: switch_on_constant maps a multi-clause key
% to the "default" sentinel (= fall through to the try_me_else chain), but the
% SwitchOnConstant runtime arm treated "default" as a real label, found none, and
% returned false -- dropping every clause of that key. Fixed to fall through.
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic rk/2.
rk(10, 3). rk(10, 4).   % repeated first-arg key 10 (multi-clause -> "default")
rk(20, 5).              % distinct key 20 (single clause -> direct jump)

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_repeated_key_fact_exec).

test(repeated_first_arg_key_enumerates,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_rk_exec'))]) :-
    Dir = 'output/test_rk_exec',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:rk/2], [module_name(rk)], Dir)),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/rk_test.rs', TestPath),
    TestSrc = '
use rk::value::Value;
use rk::state::WamState;
use rk::rk_2;

fn enum_x(n: i64) -> Vec<i64> {
    let mut vm = WamState::new(vec![], std::collections::HashMap::new());
    let mut out = Vec::new();
    if rk_2(&mut vm, Value::Integer(n), Value::Unbound("X".to_string())) {
        loop {
            if let Some(Value::Integer(x)) = vm.bindings.get("X").cloned() { out.push(x); }
            if !vm.backtrack() || !vm.run() { break; }
        }
    }
    out
}

#[test]
fn repeated_key_enumerates() {
    assert_eq!(enum_x(10), vec![3, 4], "repeated key 10 must yield both clauses");
    assert_eq!(enum_x(20), vec![5], "distinct key 20");
    assert_eq!(enum_x(99), Vec::<i64>::new(), "unindexed key fails");
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test rk_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[repeated-key exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_repeated_key_fact_exec).
