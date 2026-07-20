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

test(assoc_i64_exists_tracks_occupancy_not_value) :-
    assoc_i64_exists_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_assoc_i64_exists', DriverIR).

test(str_split_into_populates_positions) :-
    str_split_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_str_split_into', DriverIR).

test(fields_buffer_split_edit_join) :-
    fields_buffer_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_fields_buffer', DriverIR).

test(regex_fs_field_slice_and_count) :-
    regex_fs_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_regex_fs', DriverIR).

test(regex_match_and_gsub) :-
    regex_match_gsub_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_regex_match_gsub', DriverIR).

test(strnum_looks_numeric_and_cmp) :-
    strnum_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_strnum', DriverIR).

test(strnum_cmp_int) :-
    strnum_cmp_int_driver_ir(DriverIR),
    run_assoc_i64_smoke('uw_wam_strnum_cmp_int', DriverIR).

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

% Existence is an occupancy query, not `get(Key) != 0`: key 1 is deliberately
% stored with value zero and must still exist. Keys 1 and 17 collide at cap 16;
% deleting 1 must leave 17 reachable after the backward-shift repair.
assoc_i64_exists_driver_ir('
define i32 @main() {
entry:
  %table = call %WamAssocI64Table* @wam_assoc_i64_new(i64 16)
  %table_null = icmp eq %WamAssocI64Table* %table, null
  br i1 %table_null, label %alloc_fail, label %exercise

exercise:
  %zero_set = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %table, i64 1, i64 0)
  %neighbor_set = call i64 @wam_assoc_i64_set(%WamAssocI64Table* %table, i64 17, i64 7)
  %zero_exists = call i1 @wam_assoc_i64_exists(%WamAssocI64Table* %table, i64 1)
  %zero_value = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 1)
  %zero_value_ok = icmp eq i64 %zero_value, 0
  %absent_exists = call i1 @wam_assoc_i64_exists(%WamAssocI64Table* %table, i64 33)
  %absent_ok = xor i1 %absent_exists, true
  call void @wam_assoc_i64_delete(%WamAssocI64Table* %table, i64 1)
  %deleted_exists = call i1 @wam_assoc_i64_exists(%WamAssocI64Table* %table, i64 1)
  %deleted_ok = xor i1 %deleted_exists, true
  %neighbor_exists = call i1 @wam_assoc_i64_exists(%WamAssocI64Table* %table, i64 17)
  %neighbor_value = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %table, i64 17)
  %neighbor_value_ok = icmp eq i64 %neighbor_value, 7
  %a = and i1 %zero_exists, %zero_value_ok
  %b = and i1 %a, %absent_ok
  %c = and i1 %b, %deleted_ok
  %d = and i1 %c, %neighbor_exists
  %all_ok = and i1 %d, %neighbor_value_ok
  call void @wam_assoc_i64_free(%WamAssocI64Table* %table)
  br i1 %all_ok, label %ok, label %bad

ok:
  ret i32 0

alloc_fail:
  ret i32 90

bad:
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

% Exercise the editable field buffer end to end: split "a,b,c" on ',', edit a
% field, set one past the end (padding with empties), copy a field, and join with
% OFS -- comparing the joined text to expected literals via strcmp. Slices point
% into the record text (no interning), and join rebuilds $0 once.
fields_buffer_driver_ir('
@.uw_fb_abc = private constant [6 x i8] c"a,b,c\00"
@.uw_fb_X = private constant [2 x i8] c"X\00"
@.uw_fb_Z = private constant [2 x i8] c"Z\00"
@.uw_fb_aXc = private constant [6 x i8] c"a,X,c\00"
@.uw_fb_aXc__Z = private constant [9 x i8] c"a,X,c,,Z\00"
@.uw_fb_cXc__Z = private constant [9 x i8] c"c,X,c,,Z\00"

define i32 @main() {
entry:
  %abc = getelementptr [6 x i8], [6 x i8]* @.uw_fb_abc, i64 0, i64 0
  %vX = getelementptr [2 x i8], [2 x i8]* @.uw_fb_X, i64 0, i64 0
  %vZ = getelementptr [2 x i8], [2 x i8]* @.uw_fb_Z, i64 0, i64 0
  %e_aXc = getelementptr [6 x i8], [6 x i8]* @.uw_fb_aXc, i64 0, i64 0
  %e_aXc__Z = getelementptr [9 x i8], [9 x i8]* @.uw_fb_aXc__Z, i64 0, i64 0
  %e_cXc__Z = getelementptr [9 x i8], [9 x i8]* @.uw_fb_cXc__Z, i64 0, i64 0

  %fb = call %WamFieldBuf* @wam_fields_new(i8* %abc, i8 44)

  ; field 2 = X, join with comma -> a,X,c
  call void @wam_fields_set(%WamFieldBuf* %fb, i64 2, i8* %vX, i64 1)
  %j1 = call i8* @wam_fields_join(%WamFieldBuf* %fb, i8 44)
  %c1 = call i32 @strcmp(i8* %j1, i8* %e_aXc)
  %c1_ok = icmp eq i32 %c1, 0

  ; field 5 = Z (pads fields 4), join -> a,X,c,,Z
  call void @wam_fields_set(%WamFieldBuf* %fb, i64 5, i8* %vZ, i64 1)
  %j2 = call i8* @wam_fields_join(%WamFieldBuf* %fb, i8 44)
  %c2 = call i32 @strcmp(i8* %j2, i8* %e_aXc__Z)
  %c2_ok = icmp eq i32 %c2, 0

  ; field 1 = field 3 (copy slice "c"), join -> c,X,c,,Z
  %s3 = call %WamSlice @wam_fields_get(%WamFieldBuf* %fb, i64 3)
  %s3p = extractvalue %WamSlice %s3, 0
  %s3l = extractvalue %WamSlice %s3, 1
  call void @wam_fields_set(%WamFieldBuf* %fb, i64 1, i8* %s3p, i64 %s3l)
  %j3 = call i8* @wam_fields_join(%WamFieldBuf* %fb, i8 44)
  %c3 = call i32 @strcmp(i8* %j3, i8* %e_cXc__Z)
  %c3_ok = icmp eq i32 %c3, 0

  call void @wam_fields_free(%WamFieldBuf* %fb)

  %a1 = and i1 %c1_ok, %c2_ok
  %a2 = and i1 %a1, %c3_ok
  br i1 %a2, label %ok, label %bad

ok:
  ret i32 0

bad:
  ret i32 91
}
').

% Exercise the regex/multi-char FS field splitter through the public field
% functions with the sentinel separator (i8 0). Two patterns: a multi-char
% literal "::" over "a::b::c", and a regex ", *" over "a, b,c" (comma then
% optional spaces). Checks NF and per-field bytes; the cache is reset between
% patterns since one FS regex is compiled per program.
regex_fs_driver_ir('
@.uw_rfs_colons = private constant [3 x i8] c"::\00"
@.uw_rfs_rec1 = private constant [8 x i8] c"a::b::c\00"
@.uw_rfs_comma = private constant [4 x i8] c", *\00"
@.uw_rfs_rec2 = private constant [7 x i8] c"a, b,c\00"

define i32 @main() {
entry:
  ; --- multi-char literal FS "::" over "a::b::c" ---
  %p1 = getelementptr [3 x i8], [3 x i8]* @.uw_rfs_colons, i64 0, i64 0
  store i8* %p1, i8** @wam_fs_regex_pattern_ptr
  store i8* null, i8** @wam_fs_regex_cache
  %r1p = getelementptr [8 x i8], [8 x i8]* @.uw_rfs_rec1, i64 0, i64 0
  %id1 = call i64 @wam_intern_atom(i8* %r1p, i64 7)
  %v1a = insertvalue %Value undef, i32 0, 0
  %v1 = insertvalue %Value %v1a, i64 %id1, 1

  %n1 = call i64 @wam_atom_field_count_value(%Value %v1, i8 0)
  %n1_ok = icmp eq i64 %n1, 3

  %f1 = call %WamSlice @wam_atom_field_slice_value(%Value %v1, i64 1, i8 0)
  %f1p = extractvalue %WamSlice %f1, 0
  %f1l = extractvalue %WamSlice %f1, 1
  %f1l_ok = icmp eq i64 %f1l, 1
  %f1c = load i8, i8* %f1p
  %f1c_ok = icmp eq i8 %f1c, 97

  %f2 = call %WamSlice @wam_atom_field_slice_value(%Value %v1, i64 2, i8 0)
  %f2p = extractvalue %WamSlice %f2, 0
  %f2l = extractvalue %WamSlice %f2, 1
  %f2l_ok = icmp eq i64 %f2l, 1
  %f2c = load i8, i8* %f2p
  %f2c_ok = icmp eq i8 %f2c, 98

  %f3 = call %WamSlice @wam_atom_field_slice_value(%Value %v1, i64 3, i8 0)
  %f3p = extractvalue %WamSlice %f3, 0
  %f3c = load i8, i8* %f3p
  %f3c_ok = icmp eq i8 %f3c, 99

  %f4 = call %WamSlice @wam_atom_field_slice_value(%Value %v1, i64 4, i8 0)
  %f4p = extractvalue %WamSlice %f4, 0
  %f4_missing = icmp eq i8* %f4p, null

  ; --- regex FS ", *" over "a, b,c" ---
  %p2 = getelementptr [4 x i8], [4 x i8]* @.uw_rfs_comma, i64 0, i64 0
  store i8* %p2, i8** @wam_fs_regex_pattern_ptr
  store i8* null, i8** @wam_fs_regex_cache
  %r2p = getelementptr [7 x i8], [7 x i8]* @.uw_rfs_rec2, i64 0, i64 0
  %id2 = call i64 @wam_intern_atom(i8* %r2p, i64 6)
  %v2a = insertvalue %Value undef, i32 0, 0
  %v2 = insertvalue %Value %v2a, i64 %id2, 1

  %n2 = call i64 @wam_atom_field_count_value(%Value %v2, i8 0)
  %n2_ok = icmp eq i64 %n2, 3

  %g2 = call %WamSlice @wam_atom_field_slice_value(%Value %v2, i64 2, i8 0)
  %g2p = extractvalue %WamSlice %g2, 0
  %g2l = extractvalue %WamSlice %g2, 1
  %g2l_ok = icmp eq i64 %g2l, 1
  %g2c = load i8, i8* %g2p
  %g2c_ok = icmp eq i8 %g2c, 98

  %a1 = and i1 %n1_ok, %f1l_ok
  %a2 = and i1 %a1, %f1c_ok
  %a3 = and i1 %a2, %f2l_ok
  %a4 = and i1 %a3, %f2c_ok
  %a5 = and i1 %a4, %f3c_ok
  %a6 = and i1 %a5, %f4_missing
  %a7 = and i1 %a6, %n2_ok
  %a8 = and i1 %a7, %g2l_ok
  %a9 = and i1 %a8, %g2c_ok
  br i1 %a9, label %ok, label %bad

ok:
  ret i32 0

bad:
  ret i32 91
}
').

% Exercise @wam_regex_match (RSTART/RLENGTH out-params + returned position) and
% @wam_regex_gsub (global gsub, non-global sub, and & expansion), comparing gsub
% results against interned expected strings.
regex_match_gsub_driver_ir('
@.uw_mg_pat = private constant [7 x i8] c"[0-9]+\00"
@.uw_mg_str = private constant [10 x i8] c"abc123def\00"
@.uw_mg_gstr = private constant [10 x i8] c"a1b22c333\00"
@.uw_mg_hash = private constant [2 x i8] c"#\00"
@.uw_mg_ghash_exp = private constant [7 x i8] c"a#b#c#\00"
@.uw_mg_shash_exp = private constant [10 x i8] c"a#b22c333\00"
@.uw_mg_amp = private constant [4 x i8] c"[&]\00"
@.uw_mg_astr = private constant [6 x i8] c"a1b22\00"
@.uw_mg_amp_exp = private constant [10 x i8] c"a[1]b[22]\00"
@mg_c1 = internal global i8* null
@mg_c2 = internal global i8* null
@mg_c3 = internal global i8* null
@mg_c4 = internal global i8* null

define i32 @main() {
entry:
  %pat = getelementptr [7 x i8], [7 x i8]* @.uw_mg_pat, i64 0, i64 0
  %rstart = alloca i64, align 8
  %rlength = alloca i64, align 8
  %cnt = alloca i64, align 8

  ; match "[0-9]+" in "abc123def" -> pos 4, rlength 3
  %mstr = getelementptr [10 x i8], [10 x i8]* @.uw_mg_str, i64 0, i64 0
  %pos = call i64 @wam_regex_match(i8* %mstr, i64 9, i8* %pat, i8** @mg_c1, i64* %rstart, i64* %rlength)
  %rs = load i64, i64* %rstart
  %rl = load i64, i64* %rlength
  %pos_ok = icmp eq i64 %pos, 4
  %rs_ok = icmp eq i64 %rs, 4
  %rl_ok = icmp eq i64 %rl, 3

  ; gsub "[0-9]+" -> "#" (global) over "a1b22c333" -> "a#b#c#", count 3
  %gstr = getelementptr [10 x i8], [10 x i8]* @.uw_mg_gstr, i64 0, i64 0
  %hash = getelementptr [2 x i8], [2 x i8]* @.uw_mg_hash, i64 0, i64 0
  %gid = call i64 @wam_regex_gsub(i8* %gstr, i64 9, i8* %pat, i8** @mg_c2, i8* %hash, i64 1, i1 true, i64* %cnt)
  %gcnt = load i64, i64* %cnt
  %gexp_p = getelementptr [7 x i8], [7 x i8]* @.uw_mg_ghash_exp, i64 0, i64 0
  %gexp = call i64 @wam_intern_atom(i8* %gexp_p, i64 6)
  %gid_ok = icmp eq i64 %gid, %gexp
  %gcnt_ok = icmp eq i64 %gcnt, 3

  ; sub "[0-9]+" -> "#" (non-global) -> "a#b22c333", count 1
  %sid = call i64 @wam_regex_gsub(i8* %gstr, i64 9, i8* %pat, i8** @mg_c3, i8* %hash, i64 1, i1 false, i64* %cnt)
  %scnt = load i64, i64* %cnt
  %sexp_p = getelementptr [10 x i8], [10 x i8]* @.uw_mg_shash_exp, i64 0, i64 0
  %sexp = call i64 @wam_intern_atom(i8* %sexp_p, i64 9)
  %sid_ok = icmp eq i64 %sid, %sexp
  %scnt_ok = icmp eq i64 %scnt, 1

  ; gsub "[0-9]+" -> "[&]" (global) over "a1b22" -> "a[1]b[22]", count 2
  %astr = getelementptr [6 x i8], [6 x i8]* @.uw_mg_astr, i64 0, i64 0
  %amp = getelementptr [4 x i8], [4 x i8]* @.uw_mg_amp, i64 0, i64 0
  %aid = call i64 @wam_regex_gsub(i8* %astr, i64 5, i8* %pat, i8** @mg_c4, i8* %amp, i64 3, i1 true, i64* %cnt)
  %acnt = load i64, i64* %cnt
  %aexp_p = getelementptr [10 x i8], [10 x i8]* @.uw_mg_amp_exp, i64 0, i64 0
  %aexp = call i64 @wam_intern_atom(i8* %aexp_p, i64 9)
  %aid_ok = icmp eq i64 %aid, %aexp
  %acnt_ok = icmp eq i64 %acnt, 2

  %a1 = and i1 %pos_ok, %rs_ok
  %a2 = and i1 %a1, %rl_ok
  %a3 = and i1 %a2, %gid_ok
  %a4 = and i1 %a3, %gcnt_ok
  %a5 = and i1 %a4, %sid_ok
  %a6 = and i1 %a5, %scnt_ok
  %a7 = and i1 %a6, %aid_ok
  %a8 = and i1 %a7, %acnt_ok
  br i1 %a8, label %ok, label %bad

ok:
  ret i32 0

bad:
  ret i32 91
}
').

% Exercise the strnum primitives (PLAWK_STRNUM_DUALITY.md step 1). First the
% recogniser @wam_looks_numeric over numeric / non-numeric / blank-padded /
% empty inputs, then @wam_strnum_cmp over the POSIX kind table (0=number,
% 1=strnum, 2=string literal) -- including the classic "10 9" (numeric) vs
% "10 9x" (lexical) divergence and the "2" vs "10" case where numeric and
% lexical disagree in sign.
strnum_driver_ir('
@.uw_sn_10 = private constant [3 x i8] c"10\00"
@.uw_sn_314 = private constant [5 x i8] c"3.14\00"
@.uw_sn_pad = private constant [7 x i8] c"  42  \00"
@.uw_sn_neg = private constant [3 x i8] c"-5\00"
@.uw_sn_exp = private constant [4 x i8] c"1e3\00"
@.uw_sn_10x = private constant [4 x i8] c"10x\00"
@.uw_sn_abc = private constant [4 x i8] c"abc\00"
@.uw_sn_empty = private constant [1 x i8] c"\00"
@.uw_sn_blanks = private constant [3 x i8] c"  \00"
@.uw_sn_9 = private constant [2 x i8] c"9\00"
@.uw_sn_9x = private constant [3 x i8] c"9x\00"
@.uw_sn_2 = private constant [2 x i8] c"2\00"

define i32 @main() {
entry:
  %p10 = getelementptr [3 x i8], [3 x i8]* @.uw_sn_10, i64 0, i64 0
  %p314 = getelementptr [5 x i8], [5 x i8]* @.uw_sn_314, i64 0, i64 0
  %ppad = getelementptr [7 x i8], [7 x i8]* @.uw_sn_pad, i64 0, i64 0
  %pneg = getelementptr [3 x i8], [3 x i8]* @.uw_sn_neg, i64 0, i64 0
  %pexp = getelementptr [4 x i8], [4 x i8]* @.uw_sn_exp, i64 0, i64 0
  %p10x = getelementptr [4 x i8], [4 x i8]* @.uw_sn_10x, i64 0, i64 0
  %pabc = getelementptr [4 x i8], [4 x i8]* @.uw_sn_abc, i64 0, i64 0
  %pempty = getelementptr [1 x i8], [1 x i8]* @.uw_sn_empty, i64 0, i64 0
  %pblanks = getelementptr [3 x i8], [3 x i8]* @.uw_sn_blanks, i64 0, i64 0
  %p9 = getelementptr [2 x i8], [2 x i8]* @.uw_sn_9, i64 0, i64 0
  %p9x = getelementptr [3 x i8], [3 x i8]* @.uw_sn_9x, i64 0, i64 0
  %p2 = getelementptr [2 x i8], [2 x i8]* @.uw_sn_2, i64 0, i64 0

  ; --- @wam_looks_numeric ---
  %ln_10 = call i1 @wam_looks_numeric(i8* %p10)
  %ln_10_ok = icmp eq i1 %ln_10, 1
  %ln_314 = call i1 @wam_looks_numeric(i8* %p314)
  %ln_314_ok = icmp eq i1 %ln_314, 1
  %ln_pad = call i1 @wam_looks_numeric(i8* %ppad)
  %ln_pad_ok = icmp eq i1 %ln_pad, 1
  %ln_neg = call i1 @wam_looks_numeric(i8* %pneg)
  %ln_neg_ok = icmp eq i1 %ln_neg, 1
  %ln_exp = call i1 @wam_looks_numeric(i8* %pexp)
  %ln_exp_ok = icmp eq i1 %ln_exp, 1
  %ln_10x = call i1 @wam_looks_numeric(i8* %p10x)
  %ln_10x_ok = icmp eq i1 %ln_10x, 0
  %ln_abc = call i1 @wam_looks_numeric(i8* %pabc)
  %ln_abc_ok = icmp eq i1 %ln_abc, 0
  %ln_empty = call i1 @wam_looks_numeric(i8* %pempty)
  %ln_empty_ok = icmp eq i1 %ln_empty, 0
  %ln_blanks = call i1 @wam_looks_numeric(i8* %pblanks)
  %ln_blanks_ok = icmp eq i1 %ln_blanks, 0

  ; --- @wam_strnum_cmp (kinds 0=number 1=strnum 2=string) ---
  ; "10" strnum vs "9" strnum -> both numeric -> 10 > 9 -> +1
  %c1 = call i32 @wam_strnum_cmp(i8* %p10, i8 1, i8* %p9, i8 1)
  %c1_ok = icmp eq i32 %c1, 1
  ; "10" strnum vs "9x" strnum -> 9x not numeric -> lexical -> "10" < "9x" -> -1
  %c2 = call i32 @wam_strnum_cmp(i8* %p10, i8 1, i8* %p9x, i8 1)
  %c2_ok = icmp eq i32 %c2, -1
  ; "10" strnum vs "10" string literal -> literal never numeric -> lexical eq -> 0
  %c3 = call i32 @wam_strnum_cmp(i8* %p10, i8 1, i8* %p10, i8 2)
  %c3_ok = icmp eq i32 %c3, 0
  ; "10" number vs "9" number -> numeric -> +1
  %c4 = call i32 @wam_strnum_cmp(i8* %p10, i8 0, i8* %p9, i8 0)
  %c4_ok = icmp eq i32 %c4, 1
  ; "abc" strnum vs "abc" strnum -> both non-numeric -> lexical eq -> 0
  %c5 = call i32 @wam_strnum_cmp(i8* %pabc, i8 1, i8* %pabc, i8 1)
  %c5_ok = icmp eq i32 %c5, 0
  ; "2" strnum vs "10" strnum -> numeric -> 2 < 10 -> -1 (lexical would be +1)
  %c6 = call i32 @wam_strnum_cmp(i8* %p2, i8 1, i8* %p10, i8 1)
  %c6_ok = icmp eq i32 %c6, -1
  ; "10" number vs "9x" strnum -> 9x not numeric -> lexical -> "10" < "9x" -> -1
  %c7 = call i32 @wam_strnum_cmp(i8* %p10, i8 0, i8* %p9x, i8 1)
  %c7_ok = icmp eq i32 %c7, -1

  %a1 = and i1 %ln_10_ok, %ln_314_ok
  %a2 = and i1 %a1, %ln_pad_ok
  %a3 = and i1 %a2, %ln_neg_ok
  %a4 = and i1 %a3, %ln_exp_ok
  %a5 = and i1 %a4, %ln_10x_ok
  %a6 = and i1 %a5, %ln_abc_ok
  %a7 = and i1 %a6, %ln_empty_ok
  %a8 = and i1 %a7, %ln_blanks_ok
  %a9 = and i1 %a8, %c1_ok
  %a10 = and i1 %a9, %c2_ok
  %a11 = and i1 %a10, %c3_ok
  %a12 = and i1 %a11, %c4_ok
  %a13 = and i1 %a12, %c5_ok
  %a14 = and i1 %a13, %c6_ok
  %a15 = and i1 %a14, %c7_ok
  br i1 %a15, label %ok, label %bad

ok:
  ret i32 0

bad:
  ret i32 91
}
').

% Exercise @wam_strnum_cmp_int (step 3b): a strnum string operand vs an integer.
% Numeric when the string looks numeric (5 vs 3 -> +1, 2 vs 3 -> -1, 5 vs 5 ->
% 0, "05" vs 5 -> 0), lexical otherwise ("abc" vs 3 -> "abc">"3" -> +1, "5x" vs
% 5 -> "5x">"5" -> +1).
strnum_cmp_int_driver_ir('
@.uw_ci_5 = private constant [2 x i8] c"5\00"
@.uw_ci_2 = private constant [2 x i8] c"2\00"
@.uw_ci_abc = private constant [4 x i8] c"abc\00"
@.uw_ci_05 = private constant [3 x i8] c"05\00"
@.uw_ci_5x = private constant [3 x i8] c"5x\00"

define i32 @main() {
entry:
  %p5 = getelementptr [2 x i8], [2 x i8]* @.uw_ci_5, i64 0, i64 0
  %p2 = getelementptr [2 x i8], [2 x i8]* @.uw_ci_2, i64 0, i64 0
  %pabc = getelementptr [4 x i8], [4 x i8]* @.uw_ci_abc, i64 0, i64 0
  %p05 = getelementptr [3 x i8], [3 x i8]* @.uw_ci_05, i64 0, i64 0
  %p5x = getelementptr [3 x i8], [3 x i8]* @.uw_ci_5x, i64 0, i64 0

  %c1 = call i32 @wam_strnum_cmp_int(i8* %p5, i8 1, i64 3)
  %c1_ok = icmp eq i32 %c1, 1
  %c2 = call i32 @wam_strnum_cmp_int(i8* %p2, i8 1, i64 3)
  %c2_ok = icmp eq i32 %c2, -1
  %c3 = call i32 @wam_strnum_cmp_int(i8* %pabc, i8 1, i64 3)
  %c3_ok = icmp eq i32 %c3, 1
  %c4 = call i32 @wam_strnum_cmp_int(i8* %p5, i8 1, i64 5)
  %c4_ok = icmp eq i32 %c4, 0
  %c5 = call i32 @wam_strnum_cmp_int(i8* %p05, i8 1, i64 5)
  %c5_ok = icmp eq i32 %c5, 0
  %c6 = call i32 @wam_strnum_cmp_int(i8* %p5x, i8 1, i64 5)
  %c6_ok = icmp eq i32 %c6, 1

  %a1 = and i1 %c1_ok, %c2_ok
  %a2 = and i1 %a1, %c3_ok
  %a3 = and i1 %a2, %c4_ok
  %a4 = and i1 %a3, %c5_ok
  %a5 = and i1 %a4, %c6_ok
  br i1 %a5, label %ok, label %bad

ok:
  ret i32 0

bad:
  ret i32 91
}
').

:- end_tests(wam_llvm_assoc_i64_runtime).

:- initialization(run_tests, main).
