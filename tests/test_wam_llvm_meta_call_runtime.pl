:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:llvm_meta_closure_count/1.
:- dynamic user:llvm_meta_count_handler/5.

user:llvm_meta_closure_count(Count) :-
    call(llvm_meta_count_handler(Count),
         item,
         state([], [], 0, none),
         _StateN,
         no).

user:llvm_meta_count_handler(Count, _Item, State0, StateN, no) :-
    State0 = state(_InputStreams, OutputStreams, _Counter, UserFields),
    Count = 1,
    StateN = state([], OutputStreams, 1, UserFields).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_meta_call_runtime, [condition(clang_available)]).

test(compound_closure_binds_captured_argument) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_llvm_meta_call_runtime', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'meta_call_runtime.ll', LLPath),
    write_wam_llvm_project(
        [ user:llvm_meta_closure_count/1,
          user:llvm_meta_count_handler/5
        ],
        [module_name('meta_call_runtime')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    meta_call_driver_ir(InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'meta_call_runtime_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[wam llvm meta-call runtime output]~n~w~n", [OutStr]),
       throw(wam_llvm_meta_call_runtime_failed(Status))
    ),
    !.

:- end_tests(wam_llvm_meta_call_runtime).

meta_call_driver_ir(InstrCount, LabelCount, DriverIR) :-
    format(atom(DriverIR),
'
define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %unb = call %Value @value_unbound(i8* null)
  %count_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %count_ref = call %Value @value_ref(i32 %count_addr)
  call void @wam_set_pc(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %count_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %check_count, label %fail_run

check_count:
  %count_v = call %Value @wam_deref_value(%WamState* %vm, %Value %count_ref)
  %count_tag = extractvalue %Value %count_v, 0
  %count_is_int = icmp eq i32 %count_tag, 1
  br i1 %count_is_int, label %check_payload, label %fail_tag

check_payload:
  %count_payload = extractvalue %Value %count_v, 1
  %count_ok = icmp eq i64 %count_payload, 1
  br i1 %count_ok, label %success, label %fail_count

success:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0

fail_run:
  ret i32 10

fail_tag:
  call void @wam_state_free(%WamState* %vm)
  ret i32 20

fail_count:
  call void @wam_state_free(%WamState* %vm)
  ret i32 30
}
',
        [InstrCount, InstrCount, InstrCount, LabelCount, LabelCount, LabelCount]).
