% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_lmdb_smoke.pl - F# WAM LMDB fact-source end-to-end smoke test
%
% Creates a small synthetic Phase 1 LMDB fixture (via Python), generates
% an F# project with lmdb_path(LmdbDir), builds with dotnet, and
% verifies that the generated code can read facts from the LMDB and
% dispatch WAM predicates against them.
%
% Prerequisites:
%   - python3 with the 'lmdb' package (pip3 install lmdb)
%   - .NET 8 SDK on PATH
%   - LANG=C.UTF-8

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).

run_dotnet(Args, Dir, ExitCode, OutText) :-
    setup_call_cleanup(
        process_create(path(dotnet), Args,
                       [cwd(Dir),
                        stdout(pipe(Out)),
                        stderr(pipe(Err)),
                        process(PID)]),
        ( read_string(Out, _, OutStr),
          read_string(Err, _, ErrStr),
          process_wait(PID, exit(ExitCode)),
          string_concat(OutStr, ErrStr, OutText)
        ),
        ( catch(close(Out), _, true), catch(close(Err), _, true) )).

main :-
    LmdbDir = '/tmp/uw_fsharp_lmdb_fixture',
    ProjectDir = '/tmp/uw_fsharp_lmdb_smoke',
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Step 1: Generate synthetic Phase 1 LMDB (10 parents, 3 children each)
    format('Generating LMDB fixture...~n'),
    setup_call_cleanup(
        process_create(path(python3),
                       ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
                        LmdbDir,
                        '--parents', '5',
                        '--children-per-parent', '2',
                        '--refresh'],
                       [stdout(pipe(PyOut)),
                        stderr(pipe(PyErr)),
                        process(PyPID)]),
        ( read_string(PyOut, _, _PyOutS),
          read_string(PyErr, _, PyErrS),
          process_wait(PyPID, exit(PyExit))
        ),
        ( catch(close(PyOut), _, true), catch(close(PyErr), _, true) )),
    (   PyExit == 0
    ->  format('LMDB fixture generated.~n')
    ;   format('LMDB generation failed (exit ~w): ~w~n', [PyExit, PyErrS]),
        halt(1)
    ),

    %% Step 2: Generate F# project with lmdb_path.
    %% The test predicate: load category_parent from LMDB, check that
    %% parent of child 6 is parent 1 (per the synthetic fixture pattern:
    %% parents 1..5, children 6..15, each parent gets 2 children).
    %% Child 6 -> parent 1, child 7 -> parent 1, child 8 -> parent 2, etc.
    make_directory_path(ProjectDir),
    write_wam_fsharp_project(
        [],
        [no_kernels(true),
         module_name('uw_fs_lmdb_smoke'),
         lmdb_path(LmdbDir)],
        ProjectDir),
    format('Project generated at ~w~n', [ProjectDir]),

    %% Step 3: Write a custom Program.fs that:
    %%   - Opens the LMDB
    %%   - Loads category_parent into a Map<int, int list>
    %%   - Verifies expected edges exist
    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    atom_string(LmdbDir, LmdbDirStr),
    atom_string(LmdbDir, LmdbDirStr),
    string_concat(LmdbDirStr, "\"
    eprintfn \"Opening LMDB at: %s\" lmdbPath
    let env = openEnv lmdbPath
    let cp = loadCategoryParent env

    let totalEdges = cp |> Map.fold (fun acc _ vs -> acc + List.length vs) 0
    assertTrue \"total edges = 10\" (totalEdges = 10)

    assertTrue \"child 6 -> parent 1\" (Map.tryFind 6 cp = Some [1])
    assertTrue \"child 7 -> parent 1\" (Map.tryFind 7 cp = Some [1])
    assertTrue \"child 8 -> parent 2\" (Map.tryFind 8 cp = Some [2])
    assertTrue \"child 14 -> parent 5\" (Map.tryFind 14 cp = Some [5])

    assertTrue \"child 99 -> None\" (Map.tryFind 99 cp = None)
    assertTrue \"parent 1 not a child\" (Map.tryFind 1 cp = None)

    let cc = loadCategoryChild env
    let ccTotalEdges = cc |> Map.fold (fun acc _ vs -> acc + List.length vs) 0
    assertTrue \"reverse total edges = 10\" (ccTotalEdges = 10)
    assertTrue \"parent 1 -> [6; 7]\" (Map.tryFind 1 cc = Some [6; 7])

    env.Dispose()

    printfn \"RESULT %d/%d\" passes (passes + fails)
    if fails > 0 then 1 else 0
", DriverTail),
    string_concat("module Program

open System
open LmdbFactSource

let mutable passes = 0
let mutable fails = 0

let assertTrue (name: string) (cond: bool) =
    if cond then
        passes <- passes + 1
        printfn \"[PASS] %s\" name
    else
        fails <- fails + 1
        printfn \"[FAIL] %s\" name

[<EntryPoint>]
let main _argv =
    let lmdbPath = \"", DriverTail, DriverCode),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverCode),
    close(OW),

    %% Step 4: Build
    format('Building...~n'),
    run_dotnet(['build', '--nologo', '-v', 'minimal'], ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('--- build output ---~n~w~n----~n', [BuildOut]),
        format('BUILD FAILED~n'),
        halt(1)
    ),

    %% Step 5: Run
    format('Running...~n'),
    run_dotnet(['run', '--no-build', '-c', 'Debug'], ProjectDir, RunExit, RunOut),
    format('--- run output (exit=~w) ---~n~w~n----~n', [RunExit, RunOut]),
    (   RunExit == 0
    ->  halt(0)
    ;   halt(1)
    ).
