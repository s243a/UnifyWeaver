:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/core/plawk_core').

:- dynamic user:plawk_native_initial_state/1.
:- dynamic user:plawk_native_line_handler/3.
:- dynamic user:plawk_native_error_line/1.
:- dynamic user:plawk_native_readout/4.

user:plawk_native_initial_state(state([], [], 0, none)).

user:plawk_native_line_handler(Line, State0, StateN) :-
    increment_counter(State0, State1),
    (   plawk_native_error_line(Line)
    ->  append_output(Line, State1, StateN)
    ;   StateN = State1
    ).

user:plawk_native_error_line(Line) :-
    sub_atom(Line, 0, 5, _, 'ERROR').

user:plawk_native_readout(State, Count, FirstError, SecondError) :-
    state_counter(State, Count),
    state_outputs(State, [FirstError, SecondError]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_native_outer_loop_driver, [condition(clang_available)]).

test(native_loop_threads_plawk_state_through_wam_handler_calls) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_native_outer_loop_driver', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'plawk_native_outer_loop.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_native_initial_state/1,
          user:plawk_native_line_handler/3,
          user:plawk_native_error_line/1,
          user:plawk_native_readout/4,
          plawk_core:normalize_outputs/2
        ],
        [module_name('plawk_native_outer_loop')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_native_outer_loop_driver_ir(InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_native_outer_loop_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk native outer loop driver output]~n~w~n",
              [OutStr]),
       throw(plawk_native_outer_loop_driver_failed(Status))
    ),
    !.

:- end_tests(plawk_native_outer_loop_driver).

plawk_native_outer_loop_driver_ir(InstrCount, LabelCount, DriverIR) :-
    format(atom(DriverIR),
'@.plawk_line_info = private constant [13 x i8] c"INFO boot ok\\00"
@.plawk_line_error1 = private constant [16 x i8] c"ERROR disk full\\00"
@.plawk_line_warn = private constant [13 x i8] c"WARN cpu hot\\00"
@.plawk_line_error2 = private constant [15 x i8] c"ERROR net down\\00"
@.plawk_expect_first = private constant [16 x i8] c"ERROR disk full\\00"
@.plawk_expect_second = private constant [15 x i8] c"ERROR net down\\00"

define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %init_pc = load i32, i32* @plawk_native_initial_state_start_pc
  %handler_pc = load i32, i32* @plawk_native_line_handler_start_pc
  %readout_pc = load i32, i32* @plawk_native_readout_start_pc
  %info_ptr = getelementptr [13 x i8], [13 x i8]* @.plawk_line_info, i32 0, i32 0
  %error1_ptr = getelementptr [16 x i8], [16 x i8]* @.plawk_line_error1, i32 0, i32 0
  %warn_ptr = getelementptr [13 x i8], [13 x i8]* @.plawk_line_warn, i32 0, i32 0
  %error2_ptr = getelementptr [15 x i8], [15 x i8]* @.plawk_line_error2, i32 0, i32 0
  %info_id = call i64 @wam_intern_atom(i8* %info_ptr, i64 12)
  %error1_id = call i64 @wam_intern_atom(i8* %error1_ptr, i64 15)
  %warn_id = call i64 @wam_intern_atom(i8* %warn_ptr, i64 12)
  %error2_id = call i64 @wam_intern_atom(i8* %error2_ptr, i64 14)
  %info0 = insertvalue %Value undef, i32 0, 0
  %info_v = insertvalue %Value %info0, i64 %info_id, 1
  %error10 = insertvalue %Value undef, i32 0, 0
  %error1_v = insertvalue %Value %error10, i64 %error1_id, 1
  %warn0 = insertvalue %Value undef, i32 0, 0
  %warn_v = insertvalue %Value %warn0, i64 %warn_id, 1
  %error20 = insertvalue %Value undef, i32 0, 0
  %error2_v = insertvalue %Value %error20, i64 %error2_id, 1
  %state0 = call %Value @plawk_call_initial_state(%WamState* %vm, i32 %init_pc)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next_i, %call_handler ]
  %state = phi %Value [ %state0, %entry ], [ %state_next, %call_handler ]
  switch i32 %i, label %done [
    i32 0, label %line_info
    i32 1, label %line_error1
    i32 2, label %line_warn
    i32 3, label %line_error2
  ]

line_info:
  br label %call_handler

line_error1:
  br label %call_handler

line_warn:
  br label %call_handler

line_error2:
  br label %call_handler

call_handler:
  %line = phi %Value [ %info_v, %line_info ], [ %error1_v, %line_error1 ], [ %warn_v, %line_warn ], [ %error2_v, %line_error2 ]
  %state_next = call %Value @plawk_call_line_handler(%WamState* %vm, i32 %handler_pc, %Value %line, %Value %state)
  %next_i = add i32 %i, 1
  br label %loop

done:
  %final_tag = extractvalue %Value %state, 0
  %final_payload = extractvalue %Value %state, 1
  %final_is_int = icmp eq i32 %final_tag, 1
  %final_is_bad_payload = icmp slt i64 %final_payload, 0
  %final_is_bad = and i1 %final_is_int, %final_is_bad_payload
  br i1 %final_is_bad, label %fail_final_state, label %prepare_readout

prepare_readout:
  %unb = call %Value @value_unbound(i8* null)
  %count_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %first_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %second_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %count_ref = call %Value @value_ref(i32 %count_addr)
  %first_ref = call %Value @value_ref(i32 %first_addr)
  %second_ref = call %Value @value_ref(i32 %second_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %readout_pc)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %state)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %count_ref)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %first_ref)
  call void @wam_set_reg(%WamState* %vm, i32 3, %Value %second_ref)
  %readout_ok = call i1 @run_loop(%WamState* %vm)
  br i1 %readout_ok, label %check_count, label %fail_readout

check_count:
  %count_v = call %Value @wam_deref_value(%WamState* %vm, %Value %count_ref)
  %count_tag = extractvalue %Value %count_v, 0
  %count_is_int = icmp eq i32 %count_tag, 1
  br i1 %count_is_int, label %check_count_value, label %fail_count_tag

check_count_value:
  %count_payload = extractvalue %Value %count_v, 1
  %count_ok = icmp eq i64 %count_payload, 4
  br i1 %count_ok, label %check_output_tags, label %fail_count_value

check_output_tags:
  %first_v = call %Value @wam_deref_value(%WamState* %vm, %Value %first_ref)
  %second_v = call %Value @wam_deref_value(%WamState* %vm, %Value %second_ref)
  %first_tag = extractvalue %Value %first_v, 0
  %second_tag = extractvalue %Value %second_v, 0
  %first_is_atom = icmp eq i32 %first_tag, 0
  %second_is_atom = icmp eq i32 %second_tag, 0
  %tags_ok = and i1 %first_is_atom, %second_is_atom
  br i1 %tags_ok, label %check_output_strings, label %fail_output_tags

check_output_strings:
  %first_id = extractvalue %Value %first_v, 1
  %second_id = extractvalue %Value %second_v, 1
  %first_s = call i8* @wam_atom_to_string(i64 %first_id)
  %second_s = call i8* @wam_atom_to_string(i64 %second_id)
  %expect_first = getelementptr [16 x i8], [16 x i8]* @.plawk_expect_first, i32 0, i32 0
  %expect_second = getelementptr [15 x i8], [15 x i8]* @.plawk_expect_second, i32 0, i32 0
  %cmp_first = call i32 @strcmp(i8* %first_s, i8* %expect_first)
  %cmp_second = call i32 @strcmp(i8* %second_s, i8* %expect_second)
  %first_ok = icmp eq i32 %cmp_first, 0
  %second_ok = icmp eq i32 %cmp_second, 0
  %strings_ok = and i1 %first_ok, %second_ok
  br i1 %strings_ok, label %success, label %fail_output_strings

success:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0

fail_final_state:
  call void @wam_state_free(%WamState* %vm)
  ret i32 40

fail_readout:
  call void @wam_state_free(%WamState* %vm)
  ret i32 10

fail_count_tag:
  call void @wam_state_free(%WamState* %vm)
  ret i32 20

fail_count_value:
  call void @wam_state_free(%WamState* %vm)
  ret i32 21

fail_output_tags:
  call void @wam_state_free(%WamState* %vm)
  ret i32 30

fail_output_strings:
  call void @wam_state_free(%WamState* %vm)
  ret i32 31
}

define %Value @plawk_call_initial_state(%WamState* %vm, i32 %start_pc) {
entry:
  %unb = call %Value @value_unbound(i8* null)
  %state_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %state_ref = call %Value @value_ref(i32 %state_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %start_pc)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %state_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %done, label %failed

done:
  %state = call %Value @wam_deref_value(%WamState* %vm, %Value %state_ref)
  ret %Value %state

failed:
  %bad = call %Value @value_integer(i64 -1)
  ret %Value %bad
}

define %Value @plawk_call_line_handler(%WamState* %vm, i32 %start_pc, %Value %line, %Value %state0) {
entry:
  %unb = call %Value @value_unbound(i8* null)
  %state_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %state_ref = call %Value @value_ref(i32 %state_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %start_pc)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %line)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %state0)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %state_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %done, label %failed

done:
  %state = call %Value @wam_deref_value(%WamState* %vm, %Value %state_ref)
  ret %Value %state

failed:
  %bad = call %Value @value_integer(i64 -1)
  ret %Value %bad
}
',
        [InstrCount, InstrCount, InstrCount,
         LabelCount, LabelCount, LabelCount]).
