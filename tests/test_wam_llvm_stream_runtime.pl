:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:llvm_stream_reader_probe/4.

user:llvm_stream_reader_probe(Path, First, Second, EOF) :-
    stream_open(Path, Handle),
    read_line(Handle, First),
    read_line(Handle, Second),
    read_line(Handle, EOF),
    stream_close(Handle).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_stream_runtime, [condition(clang_available)]).

test(stream_open_read_line_close_executes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_llvm_stream_runtime', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, 'alpha\r\n\r\n', []),
        close(In)),
    directory_file_path(Dir, 'stream_probe.ll', LLPath),
    write_wam_llvm_project(
        [user:llvm_stream_reader_probe/4],
        [module_name('stream_probe')], LLPath),
    read_file_to_string(LLPath, Src, []),
    assertion(once(sub_string(Src, _, _, _, 'i64 166'))),
    assertion(once(sub_string(Src, _, _, _, 'i64 167'))),
    assertion(once(sub_string(Src, _, _, _, 'i64 168'))),
    stream_driver_ir(InputPath, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'stream_probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[llvm stream runtime output]~n~w~n", [OutStr]),
       throw(llvm_stream_runtime_failed(Status))
    ),
    !.

:- end_tests(wam_llvm_stream_runtime).

stream_driver_ir(InputPath, DriverIR) :-
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    format(atom(DriverIR),
'@.stream_runtime_path = private constant [~w x i8] c"~w\\00"
@.stream_expect_alpha = private constant [6 x i8] c"alpha\\00"
@.stream_expect_empty = private constant [1 x i8] c"\\00"
@.stream_expect_eof = private constant [12 x i8] c"end_of_file\\00"

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.stream_runtime_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([21 x %Instruction], [21 x %Instruction]* @module_code, i32 0, i32 0),
      i32 21,
      i32* getelementptr ([1 x i32], [1 x i32]* @module_labels, i32 0, i32 0),
      i32 1)
  %unb = call %Value @value_unbound(i8* null)
  %addr1 = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %addr2 = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %addr3 = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %ref1 = call %Value @value_ref(i32 %addr1)
  %ref2 = call %Value @value_ref(i32 %addr2)
  %ref3 = call %Value @value_ref(i32 %addr3)
  call void @wam_set_pc(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %path)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %ref1)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %ref2)
  call void @wam_set_reg(%WamState* %vm, i32 3, %Value %ref3)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %check_tags, label %fail_run

check_tags:
  ; Argument registers are caller-saved and A2 is reused for each
  ; read_line/2 output. Inspect the caller-owned heap refs directly.
  %v1 = call %Value @wam_deref_value(%WamState* %vm, %Value %ref1)
  %v2 = call %Value @wam_deref_value(%WamState* %vm, %Value %ref2)
  %v3 = call %Value @wam_deref_value(%WamState* %vm, %Value %ref3)
  %tag1 = extractvalue %Value %v1, 0
  %tag2 = extractvalue %Value %v2, 0
  %tag3 = extractvalue %Value %v3, 0
  %tag1_ok = icmp eq i32 %tag1, 0
  %tag2_ok = icmp eq i32 %tag2, 0
  %tag3_ok = icmp eq i32 %tag3, 0
  %tags_12 = and i1 %tag1_ok, %tag2_ok
  %tags_ok = and i1 %tags_12, %tag3_ok
  br i1 %tags_ok, label %check_strings, label %fail_tags

check_strings:
  %id1 = extractvalue %Value %v1, 1
  %id2 = extractvalue %Value %v2, 1
  %id3 = extractvalue %Value %v3, 1
  %s1 = call i8* @wam_atom_to_string(i64 %id1)
  %s2 = call i8* @wam_atom_to_string(i64 %id2)
  %s3 = call i8* @wam_atom_to_string(i64 %id3)
  %exp1 = getelementptr [6 x i8], [6 x i8]* @.stream_expect_alpha, i32 0, i32 0
  %exp2 = getelementptr [1 x i8], [1 x i8]* @.stream_expect_empty, i32 0, i32 0
  %exp3 = getelementptr [12 x i8], [12 x i8]* @.stream_expect_eof, i32 0, i32 0
  %cmp1 = call i32 @strcmp(i8* %s1, i8* %exp1)
  %cmp2 = call i32 @strcmp(i8* %s2, i8* %exp2)
  %cmp3 = call i32 @strcmp(i8* %s3, i8* %exp3)
  %c1_ok = icmp eq i32 %cmp1, 0
  %c2_ok = icmp eq i32 %cmp2, 0
  %c3_ok = icmp eq i32 %cmp3, 0
  br i1 %c1_ok, label %check_second, label %fail_first

check_second:
  br i1 %c2_ok, label %check_eof, label %fail_second

check_eof:
  br i1 %c3_ok, label %success, label %fail_eof

success:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0

fail_run:
  ret i32 10

fail_tags:
  call void @wam_state_free(%WamState* %vm)
  ret i32 20

fail_first:
  call void @wam_state_free(%WamState* %vm)
  ret i32 31

fail_second:
  call void @wam_state_free(%WamState* %vm)
  ret i32 32

fail_eof:
  call void @wam_state_free(%WamState* %vm)
  ret i32 33
}
',
        [BytesLen, PathBytes,
         BytesLen, BytesLen, PathLen]).

llvm_c_bytes([], '').
llvm_c_bytes([Code | Rest], Bytes) :-
    llvm_c_byte(Code, Byte),
    llvm_c_bytes(Rest, Tail),
    atom_concat(Byte, Tail, Bytes).

llvm_c_byte(Code, Byte) :-
    format(atom(Byte), '\\~|~`0t~16r~2+', [Code]).
