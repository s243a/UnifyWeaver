% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_fsharp_csr_smoke.pl - E2E smoke test for F# CSR reader
%
% Generates a synthetic Phase 1 LMDB fixture, builds a CSR artifact
% from it, then generates an F# project that opens the CSR and
% verifies lookups match the LMDB source.
%
% Prerequisites: python3 + lmdb, .NET 8 SDK

:- encoding(utf8).
:- use_module('../../src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).
:- use_module(library(filesex), [delete_directory_and_contents/1,
                                  make_directory_path/1,
                                  directory_file_path/3]).
:- use_module(library(process)).

run_cmd(Prog, Args, Dir, ExitCode, OutText) :-
    setup_call_cleanup(
        process_create(path(Prog), Args,
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
    LmdbDir = '/tmp/uw_fsharp_csr_smoke_lmdb',
    CsrDir  = '/tmp/uw_fsharp_csr_smoke_csr',
    ProjectDir = '/tmp/uw_fsharp_csr_smoke_proj',
    catch(delete_directory_and_contents(LmdbDir), _, true),
    catch(delete_directory_and_contents(CsrDir), _, true),
    catch(delete_directory_and_contents(ProjectDir), _, true),

    %% Step 1: Generate LMDB fixture (50 parents x 4 children = 200 edges)
    format('Step 1: Generating LMDB fixture (50 parents x 4 children)...~n'),
    run_cmd(python3,
            ['examples/benchmark/generate_synthetic_phase1_lmdb.py',
             LmdbDir, '--parents', '50', '--children-per-parent', '4', '--refresh'],
            '.', PyExit, _),
    (PyExit == 0 -> true ; format('LMDB gen failed~n'), halt(1)),

    %% Step 2: Build CSR artifact from LMDB
    format('Step 2: Building CSR artifact...~n'),
    run_cmd(python3,
            ['examples/benchmark/build_reverse_csr_artifact.py',
             LmdbDir, CsrDir, '--refresh'],
            '.', CsrExit, CsrOut),
    (CsrExit == 0 -> true ; format('CSR build failed: ~w~n', [CsrOut]), halt(1)),

    %% Step 3: Generate F# project with CSR
    format('Step 3: Generating F# project with CSR...~n'),
    make_directory_path(ProjectDir),
    write_wam_fsharp_project([], [no_kernels(true), module_name('uw_fs_csr_smoke'), csr_path(CsrDir)], ProjectDir),

    %% Step 4: Write smoke test driver
    atom_string(CsrDir, CsrDirStr),

    DriverCode = "module Program

open System
open CsrReader

[<EntryPoint>]
let main _argv =
    let csrPath = \"",
    DriverTail = "\"
    use csr = openCsr csrPath
    printfn \"CSR opened: parents=%d edges=%d\" csr.ParentCount csr.EdgeCount

    // The synthetic fixture has parent IDs 1..50, each with children.
    // Parent 1 has children [51..54], parent 2 has [55..58], etc.
    // (generate_synthetic_phase1_lmdb.py: child = parents + parent*children_per + c)
    let mutable totalChildren = 0
    let mutable mismatches = 0
    let allParents = csr.Parents()
    for parent in allParents do
        let children = (csr :> WamTypes.ILookupSource).Lookup(parent)
        totalChildren <- totalChildren + children.Length
        if children.Length = 0 then
            printfn \"WARN: parent %d has 0 children\" parent
            mismatches <- mismatches + 1

    // Verify a non-existent parent returns empty
    let ghost = (csr :> WamTypes.ILookupSource).Lookup(999999)
    if ghost.Length <> 0 then
        printfn \"FAIL: non-existent parent returned %d children\" ghost.Length
        mismatches <- mismatches + 1

    // Verify specific parent: parent 1 should have 4 children
    let p1Children = (csr :> WamTypes.ILookupSource).Lookup(1)
    if p1Children.Length <> 4 then
        printfn \"FAIL: parent 1 has %d children, expected 4\" p1Children.Length
        mismatches <- mismatches + 1

    printfn \"\"
    printfn \"CSR Smoke Test Results\"
    printfn \"=====================\"
    printfn \"Parents:        %d\" allParents.Length
    printfn \"Total children: %d\" totalChildren
    printfn \"Mismatches:     %d\" mismatches
    if mismatches = 0 && totalChildren = csr.EdgeCount then
        printfn \"RESULT OK\"
        0
    else
        printfn \"RESULT FAIL\"
        1
",

    string_concat(DriverCode, CsrDirStr, P1),
    string_concat(P1, DriverTail, DriverFull),

    directory_file_path(ProjectDir, 'Program.fs', ProgPath),
    open(ProgPath, write, OW, [encoding(utf8)]),
    write(OW, DriverFull),
    close(OW),

    %% Step 5: Build
    format('Step 5: Building (Release)...~n'),
    run_cmd(dotnet, ['build', '--nologo', '-v', 'minimal', '-c', 'Release'], ProjectDir, BuildExit, BuildOut),
    (   BuildExit == 0
    ->  format('Build OK.~n')
    ;   format('BUILD FAILED:~n~w~n', [BuildOut]), halt(1)
    ),

    %% Step 6: Run
    format('Step 6: Running smoke test...~n~n'),
    run_cmd(dotnet, ['run', '--no-build', '-c', 'Release'], ProjectDir, RunExit, RunOut),
    format('~w~n', [RunOut]),
    halt(RunExit).
