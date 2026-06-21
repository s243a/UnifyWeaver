% test_wam_rust_bridge_foreign_dispatch.pl
%
% INCREMENT 1 of WAM_RUST_BRIDGE_CLUSTERING.md: the fan-out bridge detector is
% reachable from Prolog as a FOREIGN PREDICATE through execute_foreign_predicate.
% Mirrors test_wam_rust_boundary_foreign_dispatch.pl: it validates the full
% dispatch path a compiled query takes —
%
%   register category_bridge_score/2 as a foreign predicate
%     -> native_kind "category_bridge_score"
%     -> result_mode "deterministic", layout "tuple:1"
%     -> config: edge_pred / mu_pred / threshold
%   load edge facts (ffi_facts) + a mu map (register_ffi_mu)
%   set A1 = node atom, A2 = unbound, call
%     execute_foreign_predicate("category_bridge_score", 2)
%   read the class atom back from A2
%
% and asserts a node with two disjoint in-domain child branches classifies as
% 'bridge'. (The real-data classification — Subfields_of_physics -> bridge,
% Matter -> leak_conduit on the 10k fixture — is the cargo-gated boundary_cache
% test wikipedia_bridge_foreign_predicate_atoms, which renders without swipl.)
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic bfp_fact/2.
bfp_fact(a, b).   % keeps the project non-trivial; the bridge graph is
bfp_fact(b, c).   % registered inside the cargo test below.

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_bridge_foreign_dispatch).

test(bridge_detector_reachable_via_foreign_dispatch,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bridge_dispatch'))]) :-
    Dir = 'output/test_bridge_dispatch',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:bfp_fact/2], [module_name(bfp)], Dir)),
    % the generated runtime must carry the bridge dispatch arm + the WamState glue.
    atom_concat(Dir, '/src/state.rs', StateRs),
    read_file_to_string(StateRs, Src, []),
    sub_string(Src, _, _, _, "\"category_bridge_score\" =>"),
    sub_string(Src, _, _, _, "fn build_bridge_scores_named"),
    sub_string(Src, _, _, _, "fn register_ffi_mu"),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bridge_test.rs', TestPath),
    TestSrc = '
use bfp::state::WamState;
use bfp::value::Value;
use std::collections::HashMap;

// Drive category_bridge_score/2 THROUGH the foreign-dispatch path: A1 = node
// atom (in), A2 = fresh output var, read the class atom back from A2. A fresh
// output-variable name per call mirrors the WAM allocating a new var per goal.
fn dispatch_class(vm: &mut WamState, node: &str, tag: usize) -> Option<String> {
    vm.set_reg("A1", Value::Atom(node.to_string()));
    vm.set_reg("A2", Value::Unbound(format!("BClass{}", tag)));
    let ok = vm.execute_foreign_predicate("category_bridge_score", 2);
    if !ok { return None; }
    match vm.deref_var(&vm.get_reg_raw("A2").unwrap()) {
        Value::Atom(s) => Some(s),
        _ => None,
    }
}

fn setup() -> WamState {
    let mut vm = WamState::new(vec![], HashMap::new());
    // org fans out into two DISJOINT in-domain child branches (a -> a1, b -> b1):
    // n_eff = 2, in-domain fraction = 1.0 => Bridge. org is the only fan-out>=2 node.
    vm.register_ffi_fact_pairs("category_parent", &[
        ("a", "org"), ("b", "org"), ("a1", "a"), ("b1", "b"),
    ]);
    vm.register_ffi_mu("category_mu", &[
        ("org", 1.0), ("a", 1.0), ("b", 1.0), ("a1", 1.0), ("b1", 1.0),
    ]);
    // register the foreign kernel exactly as the compiler would lower it.
    let key = "category_bridge_score/2";
    vm.register_foreign_predicate(key);
    vm.register_foreign_native_kind(key, "category_bridge_score");
    vm.register_foreign_result_mode(key, "deterministic");
    vm.register_foreign_result_layout(key, "tuple:1");
    vm.register_foreign_string_config(key, "edge_pred", "category_parent");
    vm.register_foreign_string_config(key, "mu_pred", "category_mu");
    vm.register_foreign_f64_config(key, "threshold", 0.3);
    vm
}

#[test]
fn bridge_node_classifies_as_bridge_via_dispatch() {
    let mut vm = setup();
    let got = dispatch_class(&mut vm, "org", 0).expect("dispatch should bind A2");
    assert_eq!(got, "bridge", "org has two disjoint in-domain branches => bridge");
    // A leaf (no fan-out) is not a candidate => dispatch fails (no solution).
    assert!(dispatch_class(&mut vm, "a1", 1).is_none(), "a leaf is not a bridge candidate");
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bridge_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[bridge foreign dispatch FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_bridge_foreign_dispatch).
