% test_wam_rust_boundary_lowering.pl
%
% P2c WIRING / lowering: the COMPILER chooses the boundary-distribution
% optimization for a detected category_ancestor kernel, gated by the project
% option boundary_optimization(true) (default off).
%
% Where test_wam_rust_boundary_foreign_dispatch.pl validates the runtime
% dispatch path, this validates the *codegen* decision:
%   - default (no option): the kernel lowers to the 4-ary streaming
%     category_ancestor native kind (unchanged behaviour).
%   - boundary_optimization(true): the kernel is UPGRADED to the 3-ary
%     category_ancestor_boundary native kind — deterministic result mode,
%     tuple(1) layout, with weight_n / result_extractor config — and a public
%     3-ary wrapper Pred(Cat, Root, Result) is emitted.
%
% The cargo-gated case builds the upgraded crate and calls the emitted wrapper,
% asserting it reproduces the production hop-stream aggregate.
%
% See WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md §5 (P2c-wiring/lowering).

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).
:- use_module(library(lists)).

:- dynamic bcat_ancestor/4.
:- dynamic bcat_parent/2.
:- dynamic max_depth/1.

max_depth(10).

bcat_ancestor(Cat, Parent, 1, Visited) :-
    bcat_parent(Cat, Parent),
    \+ member(Parent, Visited).
bcat_ancestor(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD), length(Visited, D), D < MaxD, !,
    bcat_parent(Cat, Mid),
    \+ member(Mid, Visited),
    bcat_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

gen(Options, Dir) :-
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:bcat_ancestor/4], [module_name(bl)|Options], Dir)).

read_src(Dir, File, Src) :-
    atom_concat(Dir, File, Path),
    read_file_to_string(Path, Src, []).

:- begin_tests(wam_rust_boundary_lowering).

% Default: no boundary_optimization -> stays the 4-ary streaming kernel.
test(default_keeps_streaming_category_ancestor,
     [cleanup(safe_rmdir('output/test_bl_default'))]) :-
    Dir = 'output/test_bl_default',
    gen([], Dir),
    read_src(Dir, '/src/lib.rs', Lib),
    sub_string(Lib, _, _, _, "register_foreign_native_kind(\"bcat_ancestor/4\", \"category_ancestor\")"),
    sub_string(Lib, _, _, _, "register_foreign_result_mode(\"bcat_ancestor/4\", \"stream\")"),
    % the boundary kind must NOT appear in the registration
    \+ sub_string(Lib, _, _, _, "\"category_ancestor_boundary\"").

% boundary_optimization(true) -> upgraded to the 3-ary deterministic kernel.
test(option_upgrades_to_boundary_kernel,
     [cleanup(safe_rmdir('output/test_bl_boundary'))]) :-
    Dir = 'output/test_bl_boundary',
    gen([boundary_optimization(true)], Dir),
    read_src(Dir, '/src/lib.rs', Lib),
    % registration: 3-ary key, boundary native kind, deterministic, tuple(1)
    sub_string(Lib, _, _, _, "register_foreign_native_kind(\"bcat_ancestor/3\", \"category_ancestor_boundary\")"),
    sub_string(Lib, _, _, _, "register_foreign_result_mode(\"bcat_ancestor/3\", \"deterministic\")"),
    sub_string(Lib, _, _, _, "register_foreign_result_layout(\"bcat_ancestor/3\", \"tuple(1)\")"),
    % config carried over + defaults injected
    sub_string(Lib, _, _, _, "register_foreign_usize_config(\"bcat_ancestor/3\", \"max_depth\", 10)"),
    sub_string(Lib, _, _, _, "register_foreign_string_config(\"bcat_ancestor/3\", \"edge_pred\", \"bcat_parent\")"),
    sub_string(Lib, _, _, _, "register_foreign_f64_config(\"bcat_ancestor/3\", \"weight_n\", 2.0)"),
    sub_string(Lib, _, _, _, "register_foreign_string_config(\"bcat_ancestor/3\", \"result_extractor\", \"scalar\")"),
    % the public 3-ary wrapper is emitted
    sub_string(Lib, _, _, _, "pub fn bcat_ancestor(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool").

% result-extractor / weight_n tunables flow into the registration.
test(option_tunables_flow_through,
     [cleanup(safe_rmdir('output/test_bl_tunables'))]) :-
    Dir = 'output/test_bl_tunables',
    gen([boundary_optimization(true),
         boundary_weight_n(3.0),
         boundary_result_extractor(effective_distance)], Dir),
    read_src(Dir, '/src/lib.rs', Lib),
    sub_string(Lib, _, _, _, "register_foreign_f64_config(\"bcat_ancestor/3\", \"weight_n\", 3.0)"),
    sub_string(Lib, _, _, _, "register_foreign_string_config(\"bcat_ancestor/3\", \"result_extractor\", \"effective_distance\")").

% shortest_distance extractor (increment 2): the config flows through, and the
% dispatch routes it through the min-plus distance cache, not the histogram.
test(option_shortest_distance_extractor,
     [cleanup(safe_rmdir('output/test_bl_dist'))]) :-
    Dir = 'output/test_bl_dist',
    gen([boundary_optimization(true),
         boundary_result_extractor(shortest_distance)], Dir),
    read_src(Dir, '/src/lib.rs', Lib),
    sub_string(Lib, _, _, _, "register_foreign_string_config(\"bcat_ancestor/3\", \"result_extractor\", \"shortest_distance\")"),
    read_src(Dir, '/src/state.rs', St),
    sub_string(St, _, _, _, "extractor == \"shortest_distance\""),
    sub_string(St, _, _, _, "category_ancestor_boundary_distance(cat_id, root_id").

% cargo-gated: the upgraded crate builds and the emitted wrapper reproduces the
% production hop-stream aggregate.
test(boundary_wrapper_matches_production,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bl_exec'))]) :-
    Dir = 'output/test_bl_exec',
    gen([boundary_optimization(true)], Dir),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bl_test.rs', TestPath),
    TestSrc = '
use bl::state::WamState;
use bl::value::Value;
use std::collections::HashMap;

fn ws_production(vm: &WamState, seed: u32, root: u32, budget: usize, n: f64) -> f64 {
    let acc = vm.resolve_edge_accessor("bcat_parent");
    let mut hops: Vec<i64> = Vec::new();
    let mut vis = vec![seed];
    vm.collect_native_category_ancestor_hops(seed, root, &mut vis, budget, &acc, 0, &mut hops);
    hops.iter().map(|&h| (h as f64).powf(-n)).sum()
}

#[test]
fn wrapper_matches_production() {
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.register_ffi_fact_pairs("bcat_parent", &[
        ("5","4"),("5","2"),("4","3"),("4","1"),("3","1"),("3","2"),
        ("1","0"),("2","0"),("6","5"),("7","4"),("8","6"),("8","7"),
    ]);
    let root = vm.intern_atom("0");
    let budget = 10usize;
    let n = 2.0f64;
    let bnodes: Vec<u32> = ["1","2"].iter().map(|s| vm.intern_atom(s)).collect();
    vm.build_boundary_suffix(&bnodes, root, budget, "bcat_parent");
    let mut checked = 0;
    for sname in ["3","4","5","6","7","8"] {
        let seed = vm.intern_atom(sname);
        let wp = ws_production(&vm, seed, root, budget, n);
        // call the emitted 3-ary wrapper (self-registers the kernel).
        let out = Value::Unbound(format!("Out{}", checked));
        let ok = bl::bcat_ancestor(&mut vm, Value::Atom(sname.to_string()),
                                   Value::Atom("0".to_string()), out);
        assert!(ok, "wrapper should succeed for seed {}", sname);
        let got = match vm.deref_var(&vm.get_reg_raw("A3").unwrap()) {
            Value::Float(f) => f, other => panic!("expected Float, got {:?}", other) };
        assert!((wp - got).abs() < 1e-12, "seed {}: production {} != wrapper {}", sname, wp, got);
        checked += 1;
    }
    assert_eq!(checked, 6);
}

// Same wrapper, but the side-table is populated via the root-near SELECTION path
// (set_min_dist + build_boundary_suffix_root_near) rather than an explicit band.
#[test]
fn wrapper_with_root_near_precompute() {
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.register_ffi_fact_pairs("bcat_parent", &[
        ("5","4"),("5","2"),("4","3"),("4","1"),("3","1"),("3","2"),
        ("1","0"),("2","0"),("6","5"),("7","4"),("8","6"),("8","7"),
    ]);
    let root = vm.intern_atom("0");
    let budget = 10usize;
    let n = 2.0f64;
    // distance-to-root for the graph.
    let mut md: HashMap<i32, i32> = HashMap::new();
    for (name, d) in [("0",0),("1",1),("2",1),("3",2),("4",2),("5",2),("6",3),("7",3),("8",4)] {
        md.insert(vm.intern_atom(name) as i32, d);
    }
    vm.set_min_dist(&md);
    // pick + precompute the root-near band (<= 2 hops): {1,2,3,4,5}.
    vm.build_boundary_suffix_root_near(root, 2, budget, "bcat_parent");
    assert!(!vm.boundary_suffix.is_empty(), "root-near precompute should populate the side-table");
    let mut checked = 0;
    for sname in ["3","4","5","6","7","8"] {
        let seed = vm.intern_atom(sname);
        let wp = ws_production(&vm, seed, root, budget, n);
        let out = Value::Unbound(format!("RnOut{}", checked));
        let ok = bl::bcat_ancestor(&mut vm, Value::Atom(sname.to_string()),
                                   Value::Atom("0".to_string()), out);
        assert!(ok, "wrapper should succeed for seed {}", sname);
        let got = match vm.deref_var(&vm.get_reg_raw("A3").unwrap()) {
            Value::Float(f) => f, other => panic!("expected Float, got {:?}", other) };
        assert!((wp - got).abs() < 1e-12, "seed {}: production {} != root-near {}", sname, wp, got);
        checked += 1;
    }
    assert_eq!(checked, 6);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bl_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[boundary lowering exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

% cargo-gated: the shortest_distance wrapper returns the cycle-correct shortest
% hop-distance to root (the min-plus closure), via the distance cache.
test(wrapper_shortest_distance_matches_closure,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bl_distexec'))]) :-
    Dir = 'output/test_bl_distexec',
    gen([boundary_optimization(true),
         boundary_result_extractor(shortest_distance)], Dir),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bld_test.rs', TestPath),
    TestSrc = '
use bl::state::WamState;
use bl::value::Value;
use std::collections::HashMap;

#[test]
fn wrapper_returns_shortest_distance() {
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.register_ffi_fact_pairs("bcat_parent", &[
        ("5","4"),("5","2"),("4","3"),("4","1"),("3","1"),("3","2"),
        ("1","0"),("2","0"),("6","5"),("7","4"),("8","6"),("8","7"),
    ]);
    let root = vm.intern_atom("0");
    // precompute the distance cache over a root-near band.
    let bnodes: Vec<u32> = ["1","2"].iter().map(|s| vm.intern_atom(s)).collect();
    assert!(vm.build_boundary_distances(&bnodes, root, "bcat_parent"));
    // shortest hop-distances to root for the diamond DAG.
    let want: HashMap<&str, i64> =
        [("3",2),("4",2),("5",2),("6",3),("7",3),("8",4)].into_iter().collect();
    let mut checked = 0;
    for sname in ["3","4","5","6","7","8"] {
        let out = Value::Unbound(format!("D{}", checked));
        let ok = bl::bcat_ancestor(&mut vm, Value::Atom(sname.to_string()),
                                   Value::Atom("0".to_string()), out);
        assert!(ok, "wrapper should succeed for seed {}", sname);
        let got = match vm.deref_var(&vm.get_reg_raw("A3").unwrap()) {
            Value::Integer(i) => i, other => panic!("expected Integer, got {:?}", other) };
        assert_eq!(got, want[sname], "seed {}: shortest distance", sname);
        checked += 1;
    }
    assert_eq!(checked, 6);
    // and it is still correct with NO precompute (empty cache -> plain BFS).
    let mut vm2 = WamState::new(vec![], HashMap::new());
    vm2.register_ffi_fact_pairs("bcat_parent", &[("5","2"),("2","0")]);
    let out = Value::Unbound("E".to_string());
    assert!(bl::bcat_ancestor(&mut vm2, Value::Atom("5".to_string()),
                              Value::Atom("0".to_string()), out));
    match vm2.deref_var(&vm2.get_reg_raw("A3").unwrap()) {
        Value::Integer(i) => assert_eq!(i, 2, "5->2->0 = 2 with empty cache"),
        other => panic!("expected Integer, got {:?}", other) };
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bld_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[shortest_distance exec FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_boundary_lowering).
