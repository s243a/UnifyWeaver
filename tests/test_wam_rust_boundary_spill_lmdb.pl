% test_wam_rust_boundary_spill_lmdb.pl
%
% Non-default mid-sweep spill, real LMDB backend (spec §8b): when the live frontier
% exceeds its budget, build_boundary_suffix_sweep_with_spill spills live histograms
% to the boundary_spill LMDB sub-db (LmdbFactSource as a SpillSink) and reloads them
% on demand, COMPLETING the sweep with identical band results instead of
% stop-at-depth. Validated in a generated lmdb_zero crate.
%
% cargo-gated (needs the lmdb-zero crate to build).

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic bsp_fact/2.
bsp_fact(a, b).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_boundary_spill_lmdb).

test(lmdb_spill_completes_and_matches,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bsp'))]) :-
    Dir = 'output/test_bsp',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:bsp_fact/2],
        [module_name(bsp), lmdb_mode(cursor), lmdb_crate(lmdb_zero)], Dir)),
    atom_concat(Dir, '/src/lmdb_fact_source.rs', LmdbRs),
    read_file_to_string(LmdbRs, LSrc, []),
    sub_string(LSrc, _, _, _, "impl crate::state::SpillSink for LmdbFactSource"),
    atom_concat(Dir, '/src/state.rs', StateRs),
    read_file_to_string(StateRs, SSrc, []),
    sub_string(SSrc, _, _, _, "pub fn build_boundary_suffix_sweep_with_spill"),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bsp_test.rs', TestPath),
    TestSrc = '
use bsp::state::WamState;
use bsp::lmdb_fact_source::LmdbFactSource;
use lmdb_zero as lmdb;
use std::collections::HashMap;
use std::sync::Arc;

fn snapshot(vm: &WamState) -> Vec<(u32, Vec<u64>)> {
    let mut t: Vec<(u32, Vec<u64>)> = vm.boundary_suffix.iter().map(|(&k, v)| {
        let mut h = v.clone();
        while h.len() > 1 && h.last() == Some(&0) { h.pop(); }
        (k, h)
    }).collect();
    t.sort_by_key(|(k, _)| *k);
    t
}

#[test]
fn lmdb_spill_completes_and_matches() {
    // temp env with the four Phase-1 sub-dbs so LmdbFactSource::open succeeds.
    let dir = std::env::temp_dir().join(format!("bsp_it_{}", std::process::id()));
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
    let mut sink = LmdbFactSource::open(path).unwrap();

    // VM with EAGER edges (the sweep reads the cone from ffi_facts); LMDB is only
    // the spill target. Diamond DAG toward root 0.
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.register_ffi_fact_pairs("category_parent", &[
        ("5","4"),("5","2"),("4","3"),("4","1"),("3","1"),("3","2"),("1","0"),("2","0"),
    ]);
    let root = vm.intern_atom("0");
    let band: Vec<u32> = ["1","2","3","4"].iter().map(|s| vm.intern_atom(s)).collect();
    // reference: unbudgeted sweep
    vm.build_boundary_suffix_sweep(&band, root, 8, "category_parent", 0, 0).unwrap();
    let refr = snapshot(&vm);
    // tiny live budget WITH the LMDB spill sink -> completes via spill, identical.
    let s = vm.build_boundary_suffix_sweep_with_spill(&band, root, 8, "category_parent", 0, 1, &mut sink).unwrap();
    assert!(!s.stopped_early, "spill must not stop early");
    assert!(s.spilled > 0, "spill should have written live entries to LMDB");
    assert_eq!(snapshot(&vm), refr, "LMDB-spilled sweep must produce identical band results");
    let _ = std::fs::remove_dir_all(&dir);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bsp_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[boundary spill lmdb FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_boundary_spill_lmdb).
