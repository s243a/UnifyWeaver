% test_wam_rust_boundary_repr_lmdb.pl
%
% Wire choose_representation into the persisted boundary cache (the storage win,
% end to end): WamState::boundary_suffix_reprs picks a fitted form (exact /
% tail-pruned / binomial / mixture) per cached histogram; LmdbFactSource::
% save_boundary_reprs persists them to the boundary_basis_repr sub-db; a later run
% load_boundary_reprs decodes + EXPANDS them back to histograms within the epsilon_K
% certificate. Validated in a generated lmdb_zero crate.
%
% cargo-gated (needs the lmdb-zero crate to build).

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic brl_fact/2.
brl_fact(a, b).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_boundary_repr_lmdb).

test(boundary_repr_persist_roundtrip,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_brl'))]) :-
    Dir = 'output/test_brl',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:brl_fact/2],
        [module_name(brl), lmdb_mode(cursor), lmdb_crate(lmdb_zero)], Dir)),
    atom_concat(Dir, '/src/lmdb_fact_source.rs', LmdbRs),
    read_file_to_string(LmdbRs, LSrc, []),
    sub_string(LSrc, _, _, _, "pub fn save_boundary_reprs"),
    sub_string(LSrc, _, _, _, "pub fn load_boundary_reprs"),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/brl_test.rs', TestPath),
    TestSrc = '
use brl::state::WamState;
use brl::lmdb_fact_source::LmdbFactSource;
use brl::boundary_cache::{HistRepr, binomial_pmf, cdf_max_error};
use lmdb_zero as lmdb;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn boundary_repr_roundtrip() {
    let dir = std::env::temp_dir().join(format!("brl_it_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.to_str().unwrap();
    {
        let env = unsafe {
            let mut b = lmdb::EnvBuilder::new().unwrap();
            b.set_maxdbs(16).unwrap();
            b.set_mapsize(64 * 1024 * 1024).unwrap();
            b.open(path, lmdb::open::Flags::empty(), 0o600).unwrap()
        };
        let env = Arc::new(env);
        for name in ["s2i", "i2s"] {
            let _ = lmdb::Database::open(Arc::clone(&env), Some(name),
                &lmdb::DatabaseOptions::new(lmdb::db::CREATE)).unwrap();
        }
        for name in ["category_parent", "category_child"] {
            let _ = lmdb::Database::open(Arc::clone(&env), Some(name),
                &lmdb::DatabaseOptions::new(lmdb::db::CREATE | lmdb::db::DUPSORT)).unwrap();
        }
    }
    let src = LmdbFactSource::open(path).unwrap();
    // a long, compressible (binomial-shaped) histogram + a short one.
    let long: Vec<u64> = binomial_pmf(59, 0.4).iter()
        .map(|&pr| (pr * 1_000_000.0).round() as u64).collect();
    let short = vec![900u64, 1, 1];
    let mut table: HashMap<u32, Vec<u64>> = HashMap::new();
    table.insert(1, long.clone());
    table.insert(2, short.clone());
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.set_boundary_suffix(&table);
    // choose a representation per node (work trigger 10, epsilon_K 0.01).
    let reprs = vm.boundary_suffix_reprs(10, 0.01);
    assert!(!matches!(reprs.get(&1).unwrap(), HistRepr::Exact(_)),
        "long binomial-shaped histogram should compress to a parametric form");
    assert!(matches!(reprs.get(&2).unwrap(), HistRepr::Exact(_)),
        "short histogram stays exact");
    // persist + reload (expanded back to counts).
    src.save_boundary_reprs(&reprs).unwrap();
    let loaded = src.load_boundary_reprs().unwrap();
    assert!(cdf_max_error(&long, loaded.get(&1).unwrap()) <= 0.01 + 1e-6,
        "expanded fit within the epsilon_K certificate");
    assert_eq!(loaded.get(&2).unwrap(), &short, "exact node round-trips identically");
    let _ = std::fs::remove_dir_all(&dir);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test brl_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[boundary repr lmdb FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_boundary_repr_lmdb).
