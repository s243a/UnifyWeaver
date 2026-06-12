:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/core/plawk_core').

:- dynamic user:plawk_native_stream_initial_state/1.
:- dynamic user:plawk_native_stream_line_handler/3.
:- dynamic user:plawk_native_stream_error_line/1.
:- dynamic user:plawk_native_stream_readout/4.

user:plawk_native_stream_initial_state(state([], [], 0, none)).

user:plawk_native_stream_line_handler(Line, State0, StateN) :-
    increment_counter(State0, State1),
    (   plawk_native_stream_error_line(Line)
    ->  append_output(Line, State1, StateN)
    ;   StateN = State1
    ).

user:plawk_native_stream_error_line(Line) :-
    sub_atom(Line, 0, 5, _, 'ERROR').

user:plawk_native_stream_readout(State, Count, FirstError, SecondError) :-
    state_counter(State, Count),
    state_outputs(State, [FirstError, SecondError]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_native_stream_loop_driver, [condition(clang_available)]).

test(native_loop_reads_file_and_threads_plawk_state) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_native_stream_loop_driver', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, 'INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n', []),
        close(In)),
    directory_file_path(Dir, 'plawk_native_stream_loop.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_native_stream_initial_state/1,
          user:plawk_native_stream_line_handler/3,
          user:plawk_native_stream_error_line/1,
          user:plawk_native_stream_readout/4,
          plawk_core:normalize_outputs/2
        ],
        [module_name('plawk_native_stream_loop')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_native_stream_loop_driver_ir(InputPath, InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_native_stream_loop_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk native stream loop driver output]~n~w~n",
              [OutStr]),
       throw(plawk_native_stream_loop_driver_failed(Status))
    ),
    !.

:- end_tests(plawk_native_stream_loop_driver).

plawk_native_stream_loop_driver_ir(InputPath, InstrCount, LabelCount, DriverIR) :-
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    format(atom(DriverIR),
'@.plawk_stream_path = private constant [~w x i8] c"~w\\00"
@.plawk_stream_eof = private constant [12 x i8] c"end_of_file\\00"
@.plawk_expect_first = private constant [16 x i8] c"ERROR disk full\\00"
@.plawk_expect_second = private constant [15 x i8] c"ERROR net down\\00"

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_stream_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %init_pc = load i32, i32* @plawk_native_stream_initial_state_start_pc
  %handler_pc = load i32, i32* @plawk_native_stream_line_handler_start_pc
  %readout_pc = load i32, i32* @plawk_native_stream_readout_start_pc
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %init_state, label %fail_open

init_state:
  %state0 = call %Value @plawk_call_initial_state(%WamState* %vm, i32 %init_pc)
  br label %loop

loop:
  %state = phi %Value [ %state0, %init_state ], [ %state_next, %call_handler ]
  %line = call %Value @wam_stream_read_line_value(%Value %handle)
  %line_tag = extractvalue %Value %line, 0
  %line_payload = extractvalue %Value %line, 1
  %line_is_int = icmp eq i32 %line_tag, 1
  %line_bad_payload = icmp slt i64 %line_payload, 0
  %line_bad = and i1 %line_is_int, %line_bad_payload
  br i1 %line_bad, label %fail_read, label %check_line_atom

check_line_atom:
  %line_is_atom = icmp eq i32 %line_tag, 0
  br i1 %line_is_atom, label %check_eof, label %fail_line_tag

check_eof:
  %line_s = call i8* @wam_atom_to_string(i64 %line_payload)
  %eof_s = getelementptr [12 x i8], [12 x i8]* @.plawk_stream_eof, i32 0, i32 0
  %eof_cmp = call i32 @strcmp(i8* %line_s, i8* %eof_s)
  %is_eof = icmp eq i32 %eof_cmp, 0
  br i1 %is_eof, label %close_stream, label %call_handler

call_handler:
  %state_next = call %Value @plawk_call_line_handler(%WamState* %vm, i32 %handler_pc, %Value %line, %Value %state)
  %next_tag = extractvalue %Value %state_next, 0
  %next_payload = extractvalue %Value %state_next, 1
  %next_is_int = icmp eq i32 %next_tag, 1
  %next_is_bad_payload = icmp slt i64 %next_payload, 0
  %next_is_bad = and i1 %next_is_int, %next_is_bad_payload
  br i1 %next_is_bad, label %fail_handler, label %loop

close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %prepare_readout, label %fail_close

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

fail_open:
  call void @wam_state_free(%WamState* %vm)
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  call void @wam_state_free(%WamState* %vm)
  ret i32 11

fail_line_tag:
  %close_ignore_line_tag = call i1 @wam_stream_close_value(%Value %handle)
  call void @wam_state_free(%WamState* %vm)
  ret i32 12

fail_handler:
  %close_ignore_handler = call i1 @wam_stream_close_value(%Value %handle)
  call void @wam_state_free(%WamState* %vm)
  ret i32 13

fail_close:
  call void @wam_state_free(%WamState* %vm)
  ret i32 14

fail_readout:
  call void @wam_state_free(%WamState* %vm)
  ret i32 20

fail_count_tag:
  call void @wam_state_free(%WamState* %vm)
  ret i32 30

fail_count_value:
  call void @wam_state_free(%WamState* %vm)
  ret i32 31

fail_output_tags:
  call void @wam_state_free(%WamState* %vm)
  ret i32 40

fail_output_strings:
  call void @wam_state_free(%WamState* %vm)
  ret i32 41
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
        [BytesLen, PathBytes,
         BytesLen, BytesLen, PathLen,
         InstrCount, InstrCount, InstrCount,
         LabelCount, LabelCount, LabelCount]).

llvm_c_bytes([], '').
llvm_c_bytes([Code | Rest], Bytes) :-
    llvm_c_byte(Code, Byte),
    llvm_c_bytes(Rest, Tail),
    atom_concat(Byte, Tail, Bytes).

llvm_c_byte(Code, Byte) :-
    format(atom(Byte), '\\~|~`0t~16r~2+', [Code]).
