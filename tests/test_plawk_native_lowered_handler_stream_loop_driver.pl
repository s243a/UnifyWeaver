:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:plawk_native_lowered_marker/0.

user:plawk_native_lowered_marker.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_native_lowered_handler_stream_loop_driver, [condition(clang_available)]).

test(native_loop_lowers_prefix_handler_without_wam_call) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_native_lowered_handler_stream_loop_driver', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, 'INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n', []),
        close(In)),
    directory_file_path(Dir, 'plawk_native_lowered_handler_stream_loop.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_native_lowered_marker/0 ],
        [module_name('plawk_native_lowered_handler_stream_loop')], LLPath),
    plawk_native_lowered_handler_stream_loop_driver_ir(InputPath, DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@run_loop')),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_native_lowered_handler_stream_loop_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk native lowered handler stream loop output]~n~w~n",
              [OutStr]),
       throw(plawk_native_lowered_handler_stream_loop_driver_failed(Status))
    ),
    !.

:- end_tests(plawk_native_lowered_handler_stream_loop_driver).

plawk_native_lowered_handler_stream_loop_driver_ir(InputPath, DriverIR) :-
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    llvm_emit_atom_prefix_guard(plawk_lowered_prefix, '%line', 'ERROR',
        '%is_match', PrefixGuardGlobalIR-PrefixGuardCallIR),
    format(atom(DriverIR),
'@.plawk_lowered_path = private constant [~w x i8] c"~w\\00"
@.plawk_lowered_eof = private constant [12 x i8] c"end_of_file\\00"
@.plawk_expect_first = private constant [16 x i8] c"ERROR disk full\\00"
@.plawk_expect_second = private constant [15 x i8] c"ERROR net down\\00"
~w

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.plawk_lowered_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %outputs = alloca [2 x %Value]
  %handle = call %Value @wam_stream_open_value(%Value %path)
  %handle_tag = extractvalue %Value %handle, 0
  %handle_is_int = icmp eq i32 %handle_tag, 1
  br i1 %handle_is_int, label %check_handle_value, label %fail_open

check_handle_value:
  %handle_payload = extractvalue %Value %handle, 1
  %handle_ok = icmp sgt i64 %handle_payload, 0
  br i1 %handle_ok, label %loop, label %fail_open

loop:
  %record_count = phi i64 [ 0, %check_handle_value ], [ %record_count_next, %continue_loop ]
  %output_count = phi i64 [ 0, %check_handle_value ], [ %output_count_next, %continue_loop ]
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
  %eof_s = getelementptr [12 x i8], [12 x i8]* @.plawk_lowered_eof, i32 0, i32 0
  %eof_cmp = call i32 @strcmp(i8* %line_s, i8* %eof_s)
  %is_eof = icmp eq i32 %eof_cmp, 0
  br i1 %is_eof, label %close_stream, label %lowered_match

lowered_match:
~w
  %record_count_inc = add i64 %record_count, 1
  br i1 %is_match, label %append_output, label %no_output

append_output:
  switch i64 %output_count, label %fail_output_capacity [
    i64 0, label %append_first
    i64 1, label %append_second
  ]

append_first:
  %out0 = getelementptr [2 x %Value], [2 x %Value]* %outputs, i32 0, i32 0
  store %Value %line, %Value* %out0
  br label %continue_loop

append_second:
  %out1 = getelementptr [2 x %Value], [2 x %Value]* %outputs, i32 0, i32 1
  store %Value %line, %Value* %out1
  br label %continue_loop

no_output:
  br label %continue_loop

continue_loop:
  %record_count_next = phi i64 [ %record_count_inc, %append_first ], [ %record_count_inc, %append_second ], [ %record_count_inc, %no_output ]
  %output_count_next = phi i64 [ 1, %append_first ], [ 2, %append_second ], [ %output_count, %no_output ]
  br label %loop

close_stream:
  %close_ok = call i1 @wam_stream_close_value(%Value %handle)
  br i1 %close_ok, label %check_counts, label %fail_close

check_counts:
  %record_count_ok = icmp eq i64 %record_count, 4
  br i1 %record_count_ok, label %check_output_count, label %fail_record_count

check_output_count:
  %output_count_ok = icmp eq i64 %output_count, 2
  br i1 %output_count_ok, label %check_output_tags, label %fail_output_count

check_output_tags:
  %first_ptr = getelementptr [2 x %Value], [2 x %Value]* %outputs, i32 0, i32 0
  %second_ptr = getelementptr [2 x %Value], [2 x %Value]* %outputs, i32 0, i32 1
  %first_v = load %Value, %Value* %first_ptr
  %second_v = load %Value, %Value* %second_ptr
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
  ret i32 0

fail_open:
  ret i32 10

fail_read:
  %close_ignore_read = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 11

fail_line_tag:
  %close_ignore_line_tag = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 12

fail_output_capacity:
  %close_ignore_capacity = call i1 @wam_stream_close_value(%Value %handle)
  ret i32 15

fail_close:
  ret i32 16

fail_record_count:
  ret i32 20

fail_output_count:
  ret i32 21

fail_output_tags:
  ret i32 30

fail_output_strings:
  ret i32 31
}
',
        [BytesLen, PathBytes,
         PrefixGuardGlobalIR,
         BytesLen, BytesLen, PathLen, PrefixGuardCallIR]).

llvm_c_bytes([], '').
llvm_c_bytes([Code | Rest], Bytes) :-
    llvm_c_byte(Code, Byte),
    llvm_c_bytes(Rest, Tail),
    atom_concat(Byte, Tail, Bytes).

llvm_c_byte(Code, Byte) :-
    format(atom(Byte), '\\~|~`0t~16r~2+', [Code]).
