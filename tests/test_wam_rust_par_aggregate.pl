% test_wam_rust_par_aggregate.pl
%
% Compiles and runs the unit tests embedded in the T7 (parallel / Tier-2)
% runtime substrate src/unifyweaver/targets/rust_runtime/par_aggregate.rs
% (the gated, chunked, machine-forking parallel collector that the WAM Rust
% target's BeginAggregate path will call). The .rs file carries its own
% #[cfg(test)] battery — correctness (parallel result-set == sequential, in
% generator order), the adaptive gate (sequential for cheap branches, parallel
% for expensive), and a real speedup sanity check. This test just drives
% `rustc --test` on it so the substrate is guarded by the Prolog suite.
%
% Skipped automatically when rustc is unavailable.

:- use_module(library(process)).

rustc_available :-
    catch(( process_create(path(rustc), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_rust_par_aggregate, [condition(rustc_available)]).

test(par_aggregate_unit_tests) :-
    module_path(Src),
    Dir = 'output/test_wam_rust_par_aggregate',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    atomic_list_concat([Dir, '/par_test'], Bin),
    % rustc --test -O <src> -o <bin>
    process_create(path(rustc), ['--test', '-O', Src, '-o', Bin],
                   [stdout(pipe(CO)), stderr(pipe(CE)), process(CPid)]),
    read_string(CO, _, _), read_string(CE, _, CErr), close(CO), close(CE),
    process_wait(CPid, CStatus),
    ( CStatus == exit(0)
    ->  true
    ;   format(user_error, "~n[par_aggregate compile failed]~n~w~n", [CErr]),
        throw(par_aggregate_compile_failed(CStatus)) ),
    % run the test binary
    process_create(Bin, [],
                   [stdout(pipe(RO)), stderr(std), process(RPid)]),
    read_string(RO, _, ROut), close(RO),
    process_wait(RPid, RStatus),
    ( RStatus == exit(0), sub_string(ROut, _, _, _, "test result: ok")
    ->  true
    ;   format(user_error, "~n[par_aggregate tests failed]~n~w~n", [ROut]),
        throw(par_aggregate_tests_failed(RStatus)) ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_rust_par_aggregate).

module_path(Abs) :-
    Rel = 'src/unifyweaver/targets/rust_runtime/par_aggregate.rs',
    ( exists_file(Rel)
    ->  absolute_file_name(Rel, Abs)
    ;   % run from tests/ dir
        absolute_file_name('../src/unifyweaver/targets/rust_runtime/par_aggregate.rs', Abs)
    ).
