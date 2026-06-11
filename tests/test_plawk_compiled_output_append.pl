:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/core/plawk_core').

:- dynamic user:plawk_append_output_probe/2.

user:plawk_append_output_probe(First, Second) :-
    Item = record(text, 'ERROR disk full', ['ERROR', disk, full]),
    item_field(0, Item, Line),
    append_output(Line, state([], [], 0, none), State1),
    append_output('ERROR net down', State1, State2),
    state_outputs(State2, Outputs),
    Outputs = [First, Second].

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_compiled_output_append, [condition(clang_available)]).

test(compiled_append_output_accumulates_lines) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_compiled_output_append', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'plawk_output_append.ll', LLPath),
    write_wam_llvm_project(
        [ user:plawk_append_output_probe/2,
          plawk_core:normalize_outputs/2
        ],
        [module_name('plawk_output_append')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_output_append_driver_ir(InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'plawk_output_append_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[plawk compiled output append output]~n~w~n", [OutStr]),
       throw(plawk_compiled_output_append_failed(Status))
    ),
    !.

:- end_tests(plawk_compiled_output_append).

plawk_output_append_driver_ir(InstrCount, LabelCount, DriverIR) :-
    format(atom(DriverIR),
'@.plawk_expect_first = private constant [16 x i8] c"ERROR disk full\\00"
@.plawk_expect_second = private constant [15 x i8] c"ERROR net down\\00"

define i32 @main() {
entry:
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
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %first_ref)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %second_ref)
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
  %expect_first = getelementptr [16 x i8], [16 x i8]* @.plawk_expect_first, i32 0, i32 0
  %expect_second = getelementptr [15 x i8], [15 x i8]* @.plawk_expect_second, i32 0, i32 0
  %cmp_first = call i32 @strcmp(i8* %first_s, i8* %expect_first)
  %cmp_second = call i32 @strcmp(i8* %second_s, i8* %expect_second)
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
        [InstrCount, InstrCount, InstrCount,
         LabelCount, LabelCount, LabelCount]).
