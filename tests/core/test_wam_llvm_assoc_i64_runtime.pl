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
    assoc_i64_collision_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_assoc_i64_runtime', DriverIR).

test(assoc_i64_resizes_across_many_numeric_keys) :-
    assoc_i64_resize_stress_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_assoc_i64_resize_stress', DriverIR).

test(assoc_i64_delete_backward_shifts_probe_chain) :-
    assoc_i64_delete_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_assoc_i64_delete', DriverIR).

test(str_split_into_populates_positions) :-
    str_split_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_str_split_into', DriverIR).

run_assoc_i64_smoke(Name, DriverIR) :-
    tmp_root(Root),
    directory_file_path(Root, Name, Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'assoc_i64_runtime.ll', LLPath),
    write_wam_llvm_project(
        [ user:assoc_i64_marker/0 ],
        [ module_name('assoc_i64_runtime') ],
        LLPath),
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

assoc_i64_collision_driver_ir('
define i32 @main() {
entry:
  %table = call %WamAssocI64Table* @wam_assoc_i64_new(i64 4)
  %table_null = icmp eq %WamAssocI64Table* %table, null
  br i1 %table_null, label %alloc_fail, label %exercise

exercise:
  ; 1, 5, and 9 collide when capacity is 4. The third unique key also crosses
  ; the table growth threshold, so subsequent lookups prove resize rehashing.
  %a1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 1, i64 1)
  %a2 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 1, i64 1)
  %b1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 5, i64 1)
  %c1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 9, i64 1)
  %got_a = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 1)
  %got_b = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 5)
  %got_c = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 9)
  %got_missing = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 13)
  %a_ok = icmp eq i64 %got_a, 2
  %b_ok = icmp eq i64 %got_b, 1
  %c_ok = icmp eq i64 %got_c, 1
  %missing_ok = icmp eq i64 %got_missing, 0
  %ab_ok = and i1 %a_ok, %b_ok
  %abc_ok = and i1 %ab_ok, %c_ok
  %all_ok = and i1 %abc_ok, %missing_ok
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

assoc_i64_resize_stress_driver_ir('
define i32 @main() {
entry:
  %table = call %WamAssocI64Table* @wam_assoc_i64_new(i64 4)
  %table_null = icmp eq %WamAssocI64Table* %table, null
  br i1 %table_null, label %alloc_fail, label %insert_loop

insert_loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %insert_step ]
  %done = icmp uge i64 %i, 3000
  br i1 %done, label %check_counts, label %insert_body

insert_body:
  %inserted = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 %i, i64 1)
  %insert_ok = icmp eq i64 %inserted, 1
  br i1 %insert_ok, label %insert_step, label %bad_counts

insert_step:
  %i_next = add i64 %i, 1
  br label %insert_loop

check_counts:
  %zero_again = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 0, i64 1)
  %got_zero = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 0)
  %got_last = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 2999)
  %got_missing = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 4000)
  %zero_ok = icmp eq i64 %got_zero, 2
  %last_ok = icmp eq i64 %got_last, 1
  %missing_ok = icmp eq i64 %got_missing, 0
  %zl_ok = and i1 %zero_ok, %last_ok
  %all_ok = and i1 %zl_ok, %missing_ok
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

% Deleting a key in the middle of a probe chain must pull the following
% colliding key back so it stays reachable (backward-shift), and preserve its
% value. Keys 1/17/33 all hash to slot 1 at cap 16 -> slots 1,2,3. Delete 17
% (slot 2); 33 (home 1) shifts back to slot 2. Deleting a missing key is a
% no-op. After re-adding the deleted key it counts fresh from 0.
assoc_i64_delete_driver_ir('
define i32 @main() {
entry:
  %table = call %WamAssocI64Table* @wam_assoc_i64_new(i64 16)
  %table_null = icmp eq %WamAssocI64Table* %table, null
  br i1 %table_null, label %alloc_fail, label %exercise

exercise:
  %a1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 1, i64 1)
  %b1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 17, i64 1)
  %c1 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 33, i64 1)
  %c2 = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 33, i64 1)
  ; delete the middle of the 1/17/33 chain
  call void @wam_assoc_i64_delete(%WamAssocI64Table* %table, i64 17)
  ; deleting an absent key is a no-op
  call void @wam_assoc_i64_delete(%WamAssocI64Table* %table, i64 99)
  %got_a = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 1)
  %got_del = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 17)
  %got_c = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 33)
  ; re-adding the deleted key counts fresh from 0
  %b_re = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %table, i64 17, i64 1)
  %a_ok = icmp eq i64 %got_a, 1
  %del_ok = icmp eq i64 %got_del, 0
  %c_ok = icmp eq i64 %got_c, 2
  %re_ok = icmp eq i64 %b_re, 1
  %ad_ok = and i1 %a_ok, %del_ok
  %adc_ok = and i1 %ad_ok, %c_ok
  %all_ok = and i1 %adc_ok, %re_ok
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

% Split "a,b,c" on ',' into a table keyed 1/2/3; the values are the interned ids
% of "a"/"b"/"c" (interned up front for comparison; interning is canonical) and
% the returned count is 3.
str_split_driver_ir('
@.uw_split_input = private constant [6 x i8] c"a,b,c\00"

define i32 @main() {
entry:
  %table = call %WamAssocI64Table* @wam_assoc_i64_new(i64 8)
  %tn = icmp eq %WamAssocI64Table* %table, null
  br i1 %tn, label %alloc_fail, label %go

go:
  %sp = getelementptr [6 x i8], [6 x i8]* @.uw_split_input, i64 0, i64 0
  %ida = call i64 @wam_intern_atom(i8* %sp, i64 1)
  %bptr = getelementptr [6 x i8], [6 x i8]* @.uw_split_input, i64 0, i64 2
  %idb = call i64 @wam_intern_atom(i8* %bptr, i64 1)
  %cptr = getelementptr [6 x i8], [6 x i8]* @.uw_split_input, i64 0, i64 4
  %idc = call i64 @wam_intern_atom(i8* %cptr, i64 1)
  %n = call i64 @wam_str_split_into(%WamAssocI64Table* %table, i8* %sp, i64 5, i8 44)
  %got1 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 1)
  %got2 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 2)
  %got3 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 3)
  %n_ok = icmp eq i64 %n, 3
  %g1_ok = icmp eq i64 %got1, %ida
  %g2_ok = icmp eq i64 %got2, %idb
  %g3_ok = icmp eq i64 %got3, %idc
  %a1 = and i1 %n_ok, %g1_ok
  %a2 = and i1 %a1, %g2_ok
  %a3 = and i1 %a2, %g3_ok
  call void @wam_assoc_i64_free(%WamAssocI64Table* %table)
  br i1 %a3, label %ok, label %bad

ok:
  ret i32 0

alloc_fail:
  ret i32 90

bad:
  ret i32 91
}
').

:- end_tests(wam_llvm_assoc_i64_runtime).

:- initialization(run_tests, main).
