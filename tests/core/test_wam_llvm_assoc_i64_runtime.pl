:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3]).

:- dynamic user:assoc_i64_marker/0.

user:assoc_i64_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_assoc_i64_runtime, [condition(clang_available)]).

test(assoc_i64_counts_colliding_keys_and_missing_key) :-
    run_assoc_i64_smoke.

run_assoc_i64_smoke :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_assoc_i64_runtime', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'assoc_i64_runtime.ll', LLPath),
    write_wam_llvm_project(
        [ user:assoc_i64_marker/0 ],
        [ module_name('assoc_i64_runtime') ],
        LLPath),
    assoc_i64_driver_ir(DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'assoc_i64_runtime_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> assertion(OutStr == "")
    ;  format(user_error, "~n[assoc i64 runtime output]~n~w~n", [OutStr]),
       throw(assoc_i64_runtime_failed(Status))
    ),
    !.

assoc_i64_driver_ir('
define i32 @main() {
entry:
  %table = call %WamAssocI64Table* @wam_assoc_i64_new(i64 4)
  %table_null = icmp eq %WamAssocI64Table* %table, null
  br i1 %table_null, label %alloc_fail, label %exercise

exercise:
  ; 1 and 5 collide when capacity is 4, so this also covers probing.
  %a1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 1, i64 1)
  %a2 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 1, i64 1)
  %b1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 5, i64 1)
  %got_a = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 1)
  %got_b = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 5)
  %got_missing = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 9)
  %a_ok = icmp eq i64 %got_a, 2
  %b_ok = icmp eq i64 %got_b, 1
  %missing_ok = icmp eq i64 %got_missing, 0
  %ab_ok = and i1 %a_ok, %b_ok
  %all_ok = and i1 %ab_ok, %missing_ok
  call void @wam_assoc_i64_free(%WamAssocI64Table* %table)
  br i1 %all_ok, label %ok, label %bad_counts

ok:
  ret i32 0

alloc_fail:
  ret i32 90

bad_counts:
  ret i32 91
}
').

:- end_tests(wam_llvm_assoc_i64_runtime).

:- initialization(run_tests, main).
