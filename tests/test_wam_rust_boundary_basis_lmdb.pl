% test_wam_rust_boundary_basis_lmdb.pl
%
% P2c-wiring/precompute persistence: the boundary precompute can be SPILLED to /
% PERSISTED in the `boundary_basis` LMDB sub-db and reloaded across runs, so the
% precompute is not repeated per run (spec §8b/§9). Validates, in a generated
% lmdb_zero-mode crate:
%   - LmdbFactSource::save_boundary_basis(table) writes node->histogram entries,
%   - load_boundary_basis() reads them back identically,
%   - a fresh LmdbFactSource::open over the same env still sees them (cross-run),
%   - boundary_cache::{encode_hist,decode_hist} are the packing format.
%
% cargo-gated (also needs the lmdb-zero crate to build).

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic bbl_fact/2.
bbl_fact(a, b).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- begin_tests(wam_rust_boundary_basis_lmdb).

test(boundary_basis_persist_roundtrip,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_bbl'))]) :-
    Dir = 'output/test_bbl',
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:bbl_fact/2],
        [module_name(bbl), lmdb_mode(cursor), lmdb_crate(lmdb_zero)], Dir)),
    % the runtime must carry the persistence methods + the packing format.
    atom_concat(Dir, '/src/lmdb_fact_source.rs', LmdbRs),
    read_file_to_string(LmdbRs, LSrc, []),
    sub_string(LSrc, _, _, _, "pub fn save_boundary_basis"),
    sub_string(LSrc, _, _, _, "pub fn load_boundary_basis"),
    atom_concat(Dir, '/src/boundary_cache.rs', BcRs),
    read_file_to_string(BcRs, BSrc, []),
    sub_string(BSrc, _, _, _, "pub fn encode_hist"),
    sub_string(BSrc, _, _, _, "pub fn decode_hist"),
    atom_concat(Dir, '/tests', TestsDir),
    make_directory_path(TestsDir),
    atom_concat(Dir, '/tests/bbl_test.rs', TestPath),
    TestSrc = '
use bbl::lmdb_fact_source::LmdbFactSource;
use lmdb_zero as lmdb;
use std::collections::HashMap;
use std::sync::Arc;

// Create the Phase-1 sub-dbs (empty) so LmdbFactSource::open succeeds, then
// round-trip a boundary_basis table through save/load and a fresh re-open.
#[test]
fn boundary_basis_roundtrip() {
    let dir = std::env::temp_dir().join(format!("bbl_it_{}", std::process::id()));
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
    assert!(src.load_boundary_basis().unwrap().is_empty(), "no sub-db yet -> empty");
    let mut table: HashMap<u32, Vec<u64>> = HashMap::new();
    table.insert(1, vec![0, 1]);
    table.insert(2, vec![0, 0, 3, 1]);
    table.insert(7, vec![]);
    src.save_boundary_basis(&table).unwrap();
    assert_eq!(src.load_boundary_basis().unwrap(), table, "round-trip");
    // fresh open reads the persisted table (cross-run persistence).
    let src2 = LmdbFactSource::open(path).unwrap();
    assert_eq!(src2.load_boundary_basis().unwrap(), table, "persisted across re-open");
    let _ = std::fs::remove_dir_all(&dir);
}',
    setup_call_cleanup(open(TestPath, write, S),
                       format(S, "~w", [TestSrc]), close(S)),
    format(atom(Cmd), 'cd "~w" && cargo test --test bbl_test -- --nocapture 2>&1', [Dir]),
    setup_call_cleanup(
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
        ( read_string(Out, _, OutS), read_string(Err, _, ErrS) ),
        ( close(Out), close(Err) )),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutS, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[boundary_basis lmdb FAILED] status=~w~n~w~n~w~n", [Status, OutS, ErrS]),
        fail ).

:- end_tests(wam_rust_boundary_basis_lmdb).
