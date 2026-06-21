:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:llvm_stream_state_two_lines/3.
:- dynamic user:llvm_stream_state_next/4.

user:llvm_stream_state_two_lines(Path, First, Second) :-
    llvm_stream_state_next(Path, First, state([], [], 0, none), State1),
    llvm_stream_state_next(Path, Second, State1, State2),
    State2 = state(Handle, _OutputStreams, _Counter, _UserFields),
    stream_close(Handle).

user:llvm_stream_state_next(Path, Line,
                            state([], OutputStreams, Counter, UserFields),
                            state(Handle, OutputStreams, Counter, UserFields)) :-
    stream_open(Path, Handle),
    read_line(Handle, Line).

user:llvm_stream_state_next(_Path, Line,
                            state(Handle, OutputStreams, Counter, UserFields),
                            state(Handle, OutputStreams, Counter, UserFields)) :-
    read_line(Handle, Line).


clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_stream_state_runtime, [condition(clang_available)]).

test(native_handle_survives_state_compound) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_llvm_stream_state_runtime', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, 'alpha\nbeta\n', []),
        close(In)),
    directory_file_path(Dir, 'stream_state_runtime.ll', LLPath),
    write_wam_llvm_project(
        [ user:llvm_stream_state_two_lines/3,
          user:llvm_stream_state_next/4
        ],
        [module_name('stream_state_runtime')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    stream_state_driver_ir(InputPath, InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'stream_state_runtime_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[wam llvm stream-state runtime output]~n~w~n", [OutStr]),
       throw(wam_llvm_stream_state_runtime_failed(Status))
    ),
    !.

:- end_tests(wam_llvm_stream_state_runtime).

stream_state_driver_ir(InputPath, InstrCount, LabelCount, DriverIR) :-
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    format(atom(DriverIR),
'@.stream_state_path = private constant [~w x i8] c"~w\\00"
@.stream_state_expect_alpha = private constant [6 x i8] c"alpha\\00"
@.stream_state_expect_beta = private constant [5 x i8] c"beta\\00"

define i32 @main() {
entry:
  %path_ptr = getelementptr [~w x i8], [~w x i8]* @.stream_state_path, i32 0, i32 0
  %path_id = call i64 @wam_intern_atom(i8* %path_ptr, i64 ~w)
  %path0 = insertvalue %Value undef, i32 0, 0
  %path = insertvalue %Value %path0, i64 %path_id, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %unb = call %Value @value_unbound(i8* null)
  %first_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %second_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %first_ref = call %Value @value_ref(i32 %first_addr)
  %second_ref = call %Value @value_ref(i32 %second_addr)
  call void @wam_set_pc(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %path)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %first_ref)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %second_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %check_tags, label %fail_run

check_tags:
  %first_v = call %Value @wam_deref_value(%WamState* %vm, %Value %first_ref)
  %second_v = call %Value @wam_deref_value(%WamState* %vm, %Value %second_ref)
  %first_tag = extractvalue %Value %first_v, 0
  %second_tag = extractvalue %Value %second_v, 0
  %first_is_atom = icmp eq i32 %first_tag, 0
  %second_is_atom = icmp eq i32 %second_tag, 0
  %tags_ok = and i1 %first_is_atom, %second_is_atom
  br i1 %tags_ok, label %check_strings, label %fail_tags

check_strings:
  %first_id = extractvalue %Value %first_v, 1
  %second_id = extractvalue %Value %second_v, 1
  %first_s = call i8* @wam_atom_to_string(i64 %first_id)
  %second_s = call i8* @wam_atom_to_string(i64 %second_id)
  %exp_first = getelementptr [6 x i8], [6 x i8]* @.stream_state_expect_alpha, i32 0, i32 0
  %exp_second = getelementptr [5 x i8], [5 x i8]* @.stream_state_expect_beta, i32 0, i32 0
  %cmp_first = call i32 @strcmp(i8* %first_s, i8* %exp_first)
  %cmp_second = call i32 @strcmp(i8* %second_s, i8* %exp_second)
  %first_ok = icmp eq i32 %cmp_first, 0
  %second_ok = icmp eq i32 %cmp_second, 0
  %strings_ok = and i1 %first_ok, %second_ok
  br i1 %strings_ok, label %success, label %fail_strings

success:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0

fail_run:
  ret i32 10

fail_tags:
  call void @wam_state_free(%WamState* %vm)
  ret i32 20

fail_strings:
  call void @wam_state_free(%WamState* %vm)
  ret i32 30
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
