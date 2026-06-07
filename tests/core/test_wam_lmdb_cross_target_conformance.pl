% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_lmdb_cross_target_conformance.pl
%
% Cross-target conformance for the LMDB-backed category_ancestor workload.
% The classic-program conformance harness (test_wam_cross_target_conformance.pl)
% covers boolean classic predicates but has NO LMDB-mode coverage -- which is
% why the Rust cursor s2i id-space bug (PR #2772: tuple_count 14678 vs the
% correct 970 on int-native fixtures) shipped with no failing test. This fills
% that gap: it builds a tiny INT-NATIVE Phase-1 LMDB fixture (graph keyed by raw
% page ids, with a deliberately NON-IDENTITY s2i so a path that wrongly resolves
% seeds through s2i gets a wrong answer), then runs each available target's LMDB
% category_ancestor benchmark and asserts every mode agrees with the oracle.
%
% Oracle (tiny fixture, max_depth >= 3): exactly 3 of the 4 seeds reach root 2.
%
% Targets / legs:
%   - F# eager/lazy/cached  (always, when `dotnet` is present)
%   - Rust cursor int-native (when `cargo` is present AND env UW_CONFORMANCE_RUST
%     is set -- a fresh cargo build is multi-minute, so it is opt-in)
% Missing toolchains / python-lmdb => the relevant legs SKIP (never fail), matching
% the classic harness's philosophy.
%
% Prerequisites: python3 + the `lmdb` package; .NET 8 SDK for the F# legs;
% cargo + UW_CONFORMANCE_RUST=1 for the Rust leg.

:- encoding(utf8).
:- use_module(library(process)).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).

% The tiny fixture's known answer (see build_tiny_int_native_lmdb.py).
oracle_solution_count(3).

repo_root(Root) :-
    % tests/core/ -> repo root is two dirs up from this file.
    source_file(repo_root(_), ThisFile),
    file_directory_name(ThisFile, CoreDir),
    file_directory_name(CoreDir, TestsDir),
    file_directory_name(TestsDir, Root).

have_exe(Exe) :-
    catch(( process_create(path(Exe), ['--version'],
                           [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, _) ), _, fail).

% --- run a command, capture combined stdout+stderr + exit code ---
% Inherits the parent environment (no env/1 replacement) so PATH/HOME/locale
% survive; per-run vars are set with setenv/2 before spawning. (A replaced env
% strips PATH -> cargo fails, and strips the locale -> swipl hits "Encoding
% cannot represent character" on UTF-8 templates.)
run_cmd(ExePath, Args, Cwd, ExitCode, OutText) :-
    setup_call_cleanup(
        process_create(ExePath, Args,
                       [cwd(Cwd),
                        stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
        ( read_string(Out, _, OutS),
          read_string(Err, _, ErrS),
          process_wait(PID, exit(ExitCode)),
          string_concat(OutS, ErrS, OutText) ),
        ( catch(close(Out), _, true), catch(close(Err), _, true) )).

% extract the integer following `Key=` (first match) from Text.
extract_int(Key, Text, N) :-
    string_concat(Key, "=", Pat),
    sub_string(Text, B, _, _, Pat), !,
    string_concat(Pat, "", _),
    string_length(Pat, PL),
    Start is B + PL,
    sub_string(Text, Start, _, 0, Rest),
    read_leading_int(Rest, N).

read_leading_int(Str, N) :-
    string_chars(Str, Chars),
    take_digits(Chars, Ds),
    Ds \= [],
    number_codes_from_chars(Ds, N).
take_digits([C|Cs], [C|Ds]) :- char_type(C, digit), !, take_digits(Cs, Ds).
take_digits(_, []).
number_codes_from_chars(Chars, N) :- atom_chars(A, Chars), atom_number(A, N).

% ============================================================
% Fixture
% ============================================================
build_fixture(FixDir, ok) :-
    repo_root(Root),
    directory_file_path(Root, 'tests/helpers/build_tiny_int_native_lmdb.py', Builder),
    catch(delete_directory_and_contents(FixDir), _, true),
    ( catch(( process_create(path(python3), [Builder, FixDir],
                             [stdout(null), stderr(pipe(E)), process(PID)]),
              read_string(E, _, ErrS), close(E),
              process_wait(PID, exit(Code)) ), Err, (Code = -1, ErrS = Err))
    -> ( Code == 0 -> true
       ; format('  [SKIP] fixture build failed (python3/lmdb?): ~w~n', [ErrS]), fail )
    ;  format('  [SKIP] python3 not available for fixture build~n', []), fail ).

% ============================================================
% F# legs (eager / lazy / cached)
% ============================================================
fsharp_leg(FixDir, Mode, Result) :-
    repo_root(Root),
    format(atom(ProjDir), '/tmp/uw_lmdbconf_fs_~w', [Mode]),
    catch(delete_directory_and_contents(ProjDir), _, true),
    make_directory_path(ProjDir),
    directory_file_path(Root, 'examples/benchmark/generate_wam_fsharp_optimized_benchmark.pl', Gen),
    run_cmd(path(swipl), ['-q', '-s', Gen, '--', FixDir, ProjDir, Mode], Root, GExit, GOut),
    ( GExit \== 0
    -> Result = error(generate, GOut)
    ;  run_cmd(path(dotnet), ['build', '-c', 'Release', '--nologo', '-v', 'quiet'],
               ProjDir, BExit, BOut),
       ( BExit \== 0
       -> Result = error(build, BOut)
       ;  directory_file_path(ProjDir, 'bin/Release/net8.0/wam-fsharp-optimized-bench.dll', Dll),
          run_cmd(path(dotnet), [Dll, FixDir, '1'], ProjDir, RExit, ROut),
          ( RExit == 0, extract_int("solutions", ROut, N)
          -> Result = count(N)
          ;  Result = error(run, ROut) ) ) ).

% ============================================================
% Rust leg (cursor int-native) -- opt-in (slow build)
% ============================================================
rust_leg(FixDir, Result) :-
    repo_root(Root),
    ProjDir = '/tmp/uw_lmdbconf_rust',
    directory_file_path(Root, 'examples/benchmark/generate_wam_rust_matrix_benchmark.pl', Gen),
    run_cmd(path(swipl),
            ['-q', '-s', Gen, '--', 'dummy', ProjDir,
             'accumulated', 'functions', 'kernels_on', 'cursor', 'auto', 'cached', '0'],
            Root, GExit, GOut),
    ( GExit \== 0 -> Result = error(generate, GOut)
    ; run_cmd(path(cargo), ['build', '--release'], ProjDir, BExit, BOut),
      ( BExit \== 0 -> Result = error(build, BOut)
      ; directory_file_path(ProjDir, 'target/release/bench', Bin),
        % WAM_INT_NATIVE/ROOT/THREADS set via setenv in main before this leg.
        run_cmd(Bin, [FixDir], ProjDir, RExit, ROut),
        ( RExit == 0, extract_int("tuple_count", ROut, N)
        -> Result = count(N)
        ;  Result = error(run, ROut) ) ) ).

% ============================================================
% Driver
% ============================================================
check_leg(Name, count(N), Oracle, Pass) :-
    ( N =:= Oracle
    -> format('  [PASS] ~w: ~w solutions = oracle~n', [Name, N]), Pass = pass
    ;  format('  [FAIL] ~w: ~w solutions != oracle ~w~n', [Name, N, Oracle]), Pass = fail ).
check_leg(Name, error(Stage, _), _, fail) :-
    format('  [FAIL] ~w: ~w error~n', [Name, Stage]).

main :-
    % UTF-8 locale so swipl can render the UTF-8 codegen templates; dotnet roll-
    % forward (generated fsproj targets net8.0); Rust int-native cursor flags.
    % Set in this process so spawned children inherit them (run_cmd does not
    % replace the env).
    catch(setenv('LANG', 'C.UTF-8'), _, true),
    catch(setenv('DOTNET_ROLL_FORWARD', 'Major'), _, true),
    catch(setenv('DOTNET_CLI_TELEMETRY_OPTOUT', '1'), _, true),
    catch(setenv('DOTNET_NOLOGO', '1'), _, true),
    catch(setenv('WAM_INT_NATIVE', '1'), _, true),
    catch(setenv('WAM_ROOT_ID', '2'), _, true),
    catch(setenv('WAM_THREADS', '1'), _, true),
    catch(setenv('CARGO_TERM_COLOR', 'never'), _, true),
    oracle_solution_count(Oracle),
    format('=== WAM LMDB cross-target conformance (oracle: ~w reach root) ===~n', [Oracle]),
    FixDir = '/tmp/uw_lmdbconf_fixture',
    ( build_fixture(FixDir, ok)
    -> format('Fixture built at ~w~n', [FixDir]),
       findall(P, run_all_legs(FixDir, Oracle, P), Passes),
       ( Passes == []
       -> format('All legs skipped (no toolchains) -- nothing to assert.~n', []), halt(0)
       ; ( memberchk(fail, Passes)
         -> format('~nSome legs FAILED~n', []), halt(1)
         ;  format('~nAll ~w leg(s) passed~n', [Passes]), halt(0) ) )
    ;  format('Fixture unavailable -- SKIP whole test.~n', []), halt(0) ).

run_all_legs(FixDir, Oracle, Pass) :-
    ( have_exe(dotnet)
    -> member(Mode, [lmdb_eager, lmdb_lazy, lmdb_cached]),
       fsharp_leg(FixDir, Mode, R),
       format(atom(Name), 'F#/~w', [Mode]),
       check_leg(Name, R, Oracle, Pass)
    ;  format('  [SKIP] F# legs: dotnet not on PATH~n', []), fail ).
run_all_legs(FixDir, Oracle, Pass) :-
    ( getenv('UW_CONFORMANCE_RUST', _), have_exe(cargo)
    -> rust_leg(FixDir, R),
       check_leg('Rust/cursor-int-native', R, Oracle, Pass)
    ;  format('  [SKIP] Rust leg: set UW_CONFORMANCE_RUST=1 + cargo on PATH~n', []), fail ).

:- initialization(main).
