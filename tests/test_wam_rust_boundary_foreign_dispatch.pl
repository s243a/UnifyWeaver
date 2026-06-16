% test_wam_rust_boundary_foreign_dispatch.pl
%
% P2c WIRING: the boundary-distribution optimization is reachable as a FOREIGN
% KERNEL through execute_foreign_predicate. Where test_wam_rust_boundary_kernel_
% exec.pl validates the kernel METHOD directly, this validates the full dispatch
% path a compiled query takes:
%
%   register category_ancestor_boundary/3 as a foreign predicate
%     -> native_kind "category_ancestor_boundary"
%     -> result_mode "deterministic", layout "tuple:1"
%     -> config: max_depth / edge_pred / weight_n / result_extractor
%   set A1/A2/A3, call execute_foreign_predicate("category_ancestor_boundary", 3)
%   read the ONE deterministic result back from A3
%
% and asserts the spliced scalar equals the PRODUCTION kernel's weighted-power
% aggregate, for both the "scalar" and "effective_distance" extractors. This is
% the wiring that lets the compiler substitute the boundary kernel for the
% enumerating kernel. See WAM_RUST_BOUNDARY_DISTRIBUTION_SPECIFICATION.md §5/§6.
%
% cargo-gated.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic bd_fact/2.
bd_fact(a, b).   % keeps the project non-trivial; the graph for the kernels is
bd_fact(b, c).   % registered inside the cargo test via register_ffi_fact_pairs.

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_boundary_foreign_dispatch).

test(boundary_kernel_reachable_via_foreign_dispatch,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bd_dispatch'))]) :-
    Dir = 'output/test_bd_dispatch',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:bd_fact/2], [module_name(bd)], Dir)),
    % the runtime must carry both the foreign dispatch arm and the kernel.
    atom_concat(Dir, '/src/state.rs', LibRs),
    read_file_to_string(LibRs, Src, []),
    sub_string(Src, _, _, _, "\"category_ancestor_boundary\" =>"),
    sub_string(Src, _, _, _, "collect_native_category_ancestor_boundary_hist"),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bd_test.rs', TestPath),
    TestSrc = '
use bd::state::WamState;
use bd::value::Value;
use std::collections::HashMap;

// weighted_power(N=2) aggregate from the PRODUCTION kernel (hop stream).
fn ws_production(vm: &WamState, seed: u32, root: u32, budget: usize, n: f64) -> f64 {
    let acc = vm.resolve_edge_accessor("category_parent");
    let mut hops: Vec<i64> = Vec::new();
    let mut vis = vec![seed];
    vm.collect_native_category_ancestor_hops(seed, root, &mut vis, budget, &acc, 0, &mut hops);
    hops.iter().map(|&h| (h as f64).powf(-n)).sum()
}

// Drive the boundary kernel THROUGH the foreign-dispatch path and read the
// single deterministic result back out of A3. A fresh output-variable name per
// call mirrors the WAM allocating a new var per goal (a reused name would stay
// bound from the previous call and the deterministic unify would then reject a
// different result).
fn dispatch_scalar(vm: &mut WamState, seed_name: &str, root_name: &str, tag: usize) -> Option<Value> {
    vm.set_reg("A1", Value::Atom(seed_name.to_string()));
    vm.set_reg("A2", Value::Atom(root_name.to_string()));
    vm.set_reg("A3", Value::Unbound(format!("BdOut{}", tag)));
    let ok = vm.execute_foreign_predicate("category_ancestor_boundary", 3);
    if !ok { return None; }
    Some(vm.deref_var(&vm.get_reg_raw("A3").unwrap()))
}

fn setup(extractor: &str) -> WamState {
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.register_ffi_fact_pairs("category_parent", &[
        ("5","4"),("5","2"),("4","3"),("4","1"),("3","1"),("3","2"),
        ("1","0"),("2","0"),("6","5"),("7","4"),("8","6"),("8","7"),
    ]);
    let root = vm.intern_atom("0");
    let budget = 10usize;
    // boundary band near the root
    let bnodes: Vec<u32> = ["1","2"].iter().map(|s| vm.intern_atom(s)).collect();
    vm.build_boundary_suffix(&bnodes, root, budget, "category_parent");
    // register the foreign kernel exactly as the compiler would lower it.
    let key = "category_ancestor_boundary/3";
    vm.register_foreign_predicate(key);
    vm.register_foreign_native_kind(key, "category_ancestor_boundary");
    vm.register_foreign_result_mode(key, "deterministic");
    vm.register_foreign_result_layout(key, "tuple:1");
    vm.register_foreign_usize_config(key, "max_depth", budget);
    vm.register_foreign_string_config(key, "edge_pred", "category_parent");
    vm.register_foreign_f64_config(key, "weight_n", 2.0);
    vm.register_foreign_string_config(key, "result_extractor", extractor);
    vm
}

#[test]
fn scalar_extractor_matches_production() {
    let mut vm = setup("scalar");
    let root = vm.intern_atom("0");
    let budget = 10usize;
    let n = 2.0f64;
    let mut checked = 0;
    for sname in ["3","4","5","6","7","8"] {
        let seed = vm.intern_atom(sname);
        let wp = ws_production(&vm, seed, root, budget, n);
        let got = dispatch_scalar(&mut vm, sname, "0", checked).expect("dispatch should bind A3");
        let wb = match got { Value::Float(f) => f, other => panic!("expected Float, got {:?}", other) };
        assert!((wp - wb).abs() < 1e-12, "seed {}: production {} != dispatch {}", sname, wp, wb);
        assert!(wp > 0.0, "seed {} should reach root", sname);
        checked += 1;
    }
    assert_eq!(checked, 6);
}

#[test]
fn effective_distance_extractor_matches_closed_form() {
    let mut vm = setup("effective_distance");
    let root = vm.intern_atom("0");
    let budget = 10usize;
    let n = 2.0f64;
    let mut checked = 0;
    for sname in ["3","4","5","6","7","8"] {
        let seed = vm.intern_atom(sname);
        let ws = ws_production(&vm, seed, root, budget, n);
        let expect = ws.powf(-1.0 / n);   // d_eff = WeightSum^(-1/N)
        let got = dispatch_scalar(&mut vm, sname, "0", checked).expect("dispatch should bind A3");
        let deff = match got { Value::Float(f) => f, other => panic!("expected Float, got {:?}", other) };
        assert!((expect - deff).abs() < 1e-12, "seed {}: d_eff {} != {}", sname, deff, expect);
        checked += 1;
    }
    assert_eq!(checked, 6);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bd_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[boundary foreign dispatch FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_boundary_foreign_dispatch).
