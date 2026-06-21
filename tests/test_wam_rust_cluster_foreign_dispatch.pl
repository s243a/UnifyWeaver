% test_wam_rust_cluster_foreign_dispatch.pl
%
% INCREMENT 2 of WAM_RUST_BRIDGE_CLUSTERING.md: the bridge-informed clustering
% primitive is reachable from Prolog as FOREIGN PREDICATES through
% execute_foreign_predicate. Mirrors test_wam_rust_bridge_foreign_dispatch.pl.
% Validates the full dispatch path:
%
%   register category_cluster/2 + cluster_members/2 as foreign predicates
%     -> native_kind "category_cluster" / "cluster_members"
%     -> config: edge_pred / mu_pred / threshold / tau_pure
%   load edge facts (ffi_facts) + a mu map (register_ffi_mu)
%   set A1 = node atom, A2 = unbound, call
%     execute_foreign_predicate("category_cluster", 2) -> cluster id (integer)
%   then cluster_members(ClusterId, Member) -> a member atom
%
% Asserts that two domains joined only through a leak conduit land in different
% clusters once the conduit is cut. (tau_pure is pinned for a deterministic cut
% on this small graph; the auto-calibration is exercised by the gated real-data
% test.)
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic cfp_fact/2.
cfp_fact(a, b).
cfp_fact(b, c).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_cluster_foreign_dispatch).

test(clustering_reachable_via_foreign_dispatch,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_cluster_dispatch'))]) :-
    Dir = 'output/test_cluster_dispatch',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:cfp_fact/2], [module_name(cfp)], Dir)),
    atom_concat(Dir, '/src/state.rs', StateRs),
    read_file_to_string(StateRs, Src, []),
    sub_string(Src, _, _, _, "\"category_cluster\" =>"),
    sub_string(Src, _, _, _, "\"cluster_members\" =>"),
    sub_string(Src, _, _, _, "fn build_clusters_named"),
    % cluster_by_bridges lives in boundary_cache.rs (the lib core), reused by state.rs.
    atom_concat(Dir, '/src/boundary_cache.rs', BcRs),
    read_file_to_string(BcRs, BcSrc, []),
    sub_string(BcSrc, _, _, _, "fn cluster_by_bridges"),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/cluster_test.rs', TestPath),
    TestSrc = '
use cfp::state::WamState;
use cfp::value::Value;
use std::collections::HashMap;

fn dispatch_cluster(vm: &mut WamState, node: &str, tag: usize) -> Option<i64> {
    vm.set_reg("A1", Value::Atom(node.to_string()));
    vm.set_reg("A2", Value::Unbound(format!("Cl{}", tag)));
    if !vm.execute_foreign_predicate("category_cluster", 2) { return None; }
    match vm.deref_var(&vm.get_reg_raw("A2").unwrap()) {
        Value::Integer(n) => Some(n),
        _ => None,
    }
}

fn dispatch_member(vm: &mut WamState, cluster: i64, tag: usize) -> Option<String> {
    vm.set_reg("A1", Value::Integer(cluster));
    vm.set_reg("A2", Value::Unbound(format!("Mem{}", tag)));
    if !vm.execute_foreign_predicate("cluster_members", 2) { return None; }
    match vm.deref_var(&vm.get_reg_raw("A2").unwrap()) {
        Value::Atom(s) => Some(s),
        _ => None,
    }
}

fn setup() -> WamState {
    let mut vm = WamState::new(vec![], HashMap::new());
    // D1 = {d1, x1}, D2 = {d2, y1}; both domain roots hang under the apex L, which also fans into a
    // big OUT-of-domain subtree => L is a LeakConduit (low in-domain fraction, distinct in-domain
    // children). Cutting L separates D1 from D2.
    let mut edges: Vec<(&str, &str)> = vec![("d1","L"),("x1","d1"),("d2","L"),("y1","d2")];
    let oods = ["o1","o2","o3","o4","o5","o6","o7","o8","o9","o10","o11","o12","o13","o14","o15"];
    for o in oods.iter() { edges.push((o, "L")); }
    vm.register_ffi_fact_pairs("category_parent", &edges);
    let mut mu: Vec<(&str, f64)> = vec![("L",1.0),("d1",1.0),("x1",1.0),("d2",1.0),("y1",1.0)];
    for o in oods.iter() { mu.push((o, 0.0)); }
    vm.register_ffi_mu("category_mu", &mu);
    for key in ["category_cluster/2", "cluster_members/2"] {
        vm.register_foreign_predicate(key);
        vm.register_foreign_string_config(key, "edge_pred", "category_parent");
        vm.register_foreign_string_config(key, "mu_pred", "category_mu");
        vm.register_foreign_f64_config(key, "threshold", 0.3);
        vm.register_foreign_f64_config(key, "tau_pure", 0.3); // pin the cut on this small graph
    }
    vm.register_foreign_native_kind("category_cluster/2", "category_cluster");
    vm.register_foreign_result_mode("category_cluster/2", "deterministic");
    vm.register_foreign_result_layout("category_cluster/2", "tuple:1");
    vm.register_foreign_native_kind("cluster_members/2", "cluster_members");
    vm.register_foreign_result_mode("cluster_members/2", "stream");
    vm.register_foreign_result_layout("cluster_members/2", "tuple:1");
    vm
}

#[test]
fn leak_cut_separates_domains_via_dispatch() {
    let mut vm = setup();
    let c_d1 = dispatch_cluster(&mut vm, "d1", 0).expect("d1 has a cluster");
    let c_x1 = dispatch_cluster(&mut vm, "x1", 1).expect("x1 has a cluster");
    let c_d2 = dispatch_cluster(&mut vm, "d2", 2).expect("d2 has a cluster");
    let c_y1 = dispatch_cluster(&mut vm, "y1", 3).expect("y1 has a cluster");
    assert_eq!(c_d1, c_x1, "D1 = {{d1,x1}} share a cluster");
    assert_eq!(c_d2, c_y1, "D2 = {{d2,y1}} share a cluster");
    assert_ne!(c_d1, c_d2, "cutting the leak conduit L separates D1 from D2");
    // cluster_members of D1 binds to a D1 member atom.
    let m = dispatch_member(&mut vm, c_d1, 4).expect("D1 cluster has members");
    assert!(m == "d1" || m == "x1", "a D1 member, got {:?}", m);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test cluster_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[cluster foreign dispatch FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_cluster_foreign_dispatch).
