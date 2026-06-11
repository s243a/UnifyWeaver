:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:uw_ite_else_probe/1.
:- dynamic user:uw_allocating_false/1.

user:uw_ite_else_probe(Result) :-
    (   uw_allocating_false(token)
    ->  Result = then
    ;   Result = else
    ).

user:uw_allocating_false(Value) :-
    Pair = pair(Value, keep),
    Pair = pair(no, keep).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_choicepoint_stack_restore, [condition(clang_available)]).

test(failed_allocating_call_enters_ite_else_branch) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_llvm_choicepoint_stack_restore', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'choicepoint_stack_restore.ll', LLPath),
    write_wam_llvm_project(
        [ user:uw_ite_else_probe/1,
          user:uw_allocating_false/1
        ],
        [module_name('choicepoint_stack_restore')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    choicepoint_stack_restore_driver_ir(InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'choicepoint_stack_restore_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[wam llvm choicepoint stack restore output]~n~w~n",
              [OutStr]),
       throw(wam_llvm_choicepoint_stack_restore_failed(Status))
    ),
    !.

:- end_tests(wam_llvm_choicepoint_stack_restore).

choicepoint_stack_restore_driver_ir(InstrCount, LabelCount, DriverIR) :-
    format(atom(DriverIR),
'@.expect_else = private constant [5 x i8] c"else\\00"

define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %unb = call %Value @value_unbound(i8* null)
  %result_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %result_ref = call %Value @value_ref(i32 %result_addr)
  call void @wam_set_pc(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %result_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %check_tag, label %fail_run

check_tag:
  %result_v = call %Value @wam_deref_value(%WamState* %vm, %Value %result_ref)
  %result_tag = extractvalue %Value %result_v, 0
  %result_is_atom = icmp eq i32 %result_tag, 0
  br i1 %result_is_atom, label %check_string, label %fail_tag

check_string:
  %result_id = extractvalue %Value %result_v, 1
  %result_s = call i8* @wam_atom_to_string(i64 %result_id)
  %expect = getelementptr [5 x i8], [5 x i8]* @.expect_else, i32 0, i32 0
  %cmp = call i32 @strcmp(i8* %result_s, i8* %expect)
  %strings_ok = icmp eq i32 %cmp, 0
  br i1 %strings_ok, label %success, label %fail_string

success:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0

fail_run:
  ret i32 10

fail_tag:
  call void @wam_state_free(%WamState* %vm)
  ret i32 20

fail_string:
  call void @wam_state_free(%WamState* %vm)
  ret i32 30
}
',
        [InstrCount, InstrCount, InstrCount,
         LabelCount, LabelCount, LabelCount]).
