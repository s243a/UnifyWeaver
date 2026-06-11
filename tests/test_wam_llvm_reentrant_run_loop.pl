:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:uw_native_loop_step/3.

user:uw_native_loop_step(_Item, Count0, CountN) :-
    CountN is Count0 + 1.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_reentrant_run_loop, [condition(clang_available)]).

test(native_driver_reuses_vm_for_repeated_wam_handler_calls) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_llvm_reentrant_run_loop', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'reentrant_run_loop.ll', LLPath),
    write_wam_llvm_project(
        [ user:uw_native_loop_step/3 ],
        [module_name('reentrant_run_loop')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    reentrant_run_loop_driver_ir(InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'reentrant_run_loop_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[wam llvm reentrant run-loop output]~n~w~n",
              [OutStr]),
       throw(wam_llvm_reentrant_run_loop_failed(Status))
    ),
    !.

:- end_tests(wam_llvm_reentrant_run_loop).

reentrant_run_loop_driver_ir(InstrCount, LabelCount, DriverIR) :-
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %step_pc = load i32, i32* @uw_native_loop_step_start_pc
  %item = call %Value @value_integer(i64 0)
  %zero = call %Value @value_integer(i64 0)
  %one = call %Value @wam_call_count_step(%WamState* %vm, i32 %step_pc, %Value %item, %Value %zero)
  %two = call %Value @wam_call_count_step(%WamState* %vm, i32 %step_pc, %Value %item, %Value %one)
  %three = call %Value @wam_call_count_step(%WamState* %vm, i32 %step_pc, %Value %item, %Value %two)
  %tag = extractvalue %Value %three, 0
  %is_int = icmp eq i32 %tag, 1
  br i1 %is_int, label %check_value, label %fail_tag

check_value:
  %payload = extractvalue %Value %three, 1
  %ok = icmp eq i64 %payload, 3
  br i1 %ok, label %success, label %fail_value

success:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0

fail_tag:
  call void @wam_state_free(%WamState* %vm)
  ret i32 20

fail_value:
  call void @wam_state_free(%WamState* %vm)
  ret i32 30
}

define %Value @wam_call_count_step(%WamState* %vm, i32 %start_pc, %Value %item, %Value %count0) {
entry:
  %unb = call %Value @value_unbound(i8* null)
  %out_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %out_ref = call %Value @value_ref(i32 %out_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %start_pc)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %item)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %count0)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %out_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %done, label %failed

done:
  %out = call %Value @wam_deref_value(%WamState* %vm, %Value %out_ref)
  ret %Value %out

failed:
  %bad = call %Value @value_integer(i64 -1)
  ret %Value %bad
}
',
        [InstrCount, InstrCount, InstrCount,
         LabelCount, LabelCount, LabelCount]).
