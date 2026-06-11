:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/core/plawk_core').

:- dynamic user:plawk_stream_count_errors/4.
:- dynamic user:plawk_stream_loop/3.
:- dynamic user:plawk_stream_error_handler/4.

user:plawk_stream_count_errors(Path, Count, FirstError, SecondError) :-
    stream_open(Path, Handle),
    plawk_stream_loop(Handle, state([], [], 0, none), StateN),
    stream_close(Handle),
    state_counter(StateN, Count),
    state_outputs(StateN, [FirstError, SecondError]).

user:plawk_stream_loop(Handle, State0, StateN) :-
    read_line(Handle, Line),
    (   Line == end_of_file
    ->  StateN = State0
    ;   atom_split(Line, ' ', Fields),
        Item = record(text, Line, Fields),
        plawk_stream_error_handler(Item, State0, State1, yes),
        plawk_stream_loop(Handle, State1, StateN)
    ).

user:plawk_stream_error_handler(Item, State0, StateN, yes) :-
    increment_counter(State0, State1),
    (   item_field(1, Item, 'ERROR')
    ->  item_field(0, Item, Line),
        append_output(Line, State1, StateN)
    ;   StateN = State1
    ).


clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_compiled_stream_core, [condition(clang_available)]).

test(compiled_stream_counts_and_collects_error_lines) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_compiled_stream_core', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.txt', InputPath),
    setup_call_cleanup(
        open(InputPath, write, In, [type(binary)]),
        format(In, 'INFO boot ok\nERROR disk full\nWARN cpu hot\nERROR net down\n', []),
        close(In)),
    directory_file_path(Dir, 'plawk_stream_core.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_stream_count_errors/4,
          user:plawk_stream_loop/3,
          user:plawk_stream_error_handler/4,
          plawk_core:normalize_outputs/2
        ],
        [module_name('plawk_stream_core')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_stream_driver_ir(InputPath, InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_stream_core_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk compiled stream core output]~n~w~n", [OutStr]),
       throw(plawk_compiled_stream_core_failed(Status))
    ),
    !.

:- end_tests(plawk_compiled_stream_core).

plawk_stream_driver_ir(InputPath, InstrCount, LabelCount, DriverIR) :-
    atom_codes(InputPath, PathCodes),
    length(PathCodes, PathLen),
    BytesLen is PathLen + 1,
    llvm_c_bytes(PathCodes, PathBytes),
    format(atom(DriverIR),
'@.plawk_stream_path = private constant [~w x i8] c"~w\\00"
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
  %unb = call %Value @value_unbound(i8* null)
  %count_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %first_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %second_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %count_ref = call %Value @value_ref(i32 %count_addr)
  %first_ref = call %Value @value_ref(i32 %first_addr)
  %second_ref = call %Value @value_ref(i32 %second_addr)
  call void @wam_set_pc(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %path)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %count_ref)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %first_ref)
  call void @wam_set_reg(%WamState* %vm, i32 3, %Value %second_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %check_count, label %fail_run

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

fail_run:
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
