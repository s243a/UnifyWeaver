:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:uw_quoted_atoms/2.

user:uw_quoted_atoms('ERROR disk full', 'it''s bad').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_llvm_quoted_atom_literals, [condition(clang_available)]).

test(quoted_atom_tokens_intern_without_outer_quotes) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_wam_llvm_quoted_atom_literals', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'quoted_atom_literals.ll', LLPath),
    write_wam_llvm_project(
        [user:uw_quoted_atoms/2],
        [module_name('quoted_atom_literals')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    quoted_atom_driver_ir(InstrCount, LabelCount, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'quoted_atom_literals_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1 && ~w',
        [LLPath, BinPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    ( Status == exit(0)
    -> true
    ;  format(user_error, "~n[wam llvm quoted atom literal output]~n~w~n",
              [OutStr]),
       throw(wam_llvm_quoted_atom_literals_failed(Status))
    ),
    !.

:- end_tests(wam_llvm_quoted_atom_literals).

quoted_atom_driver_ir(InstrCount, LabelCount, DriverIR) :-
    format(atom(DriverIR),
'@.expect_spaced = private constant [16 x i8] c"ERROR disk full\\00"
@.expect_escaped = private constant [9 x i8] c"it\\27s bad\\00"

define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %start_pc = load i32, i32* @uw_quoted_atoms_start_pc
  %unb = call %Value @value_unbound(i8* null)
  %a_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %b_addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %a_ref = call %Value @value_ref(i32 %a_addr)
  %b_ref = call %Value @value_ref(i32 %b_addr)
  call void @wam_prepare_call(%WamState* %vm, i32 %start_pc)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a_ref)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %b_ref)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %check_tags, label %fail_run

check_tags:
  %a_v = call %Value @wam_deref_value(%WamState* %vm, %Value %a_ref)
  %b_v = call %Value @wam_deref_value(%WamState* %vm, %Value %b_ref)
  %a_tag = extractvalue %Value %a_v, 0
  %b_tag = extractvalue %Value %b_v, 0
  %a_is_atom = icmp eq i32 %a_tag, 0
  %b_is_atom = icmp eq i32 %b_tag, 0
  %tags_ok = and i1 %a_is_atom, %b_is_atom
  br i1 %tags_ok, label %check_strings, label %fail_tags

check_strings:
  %a_id = extractvalue %Value %a_v, 1
  %b_id = extractvalue %Value %b_v, 1
  %a_s = call i8* @wam_atom_to_string(i64 %a_id)
  %b_s = call i8* @wam_atom_to_string(i64 %b_id)
  %expect_a = getelementptr [16 x i8], [16 x i8]* @.expect_spaced, i32 0, i32 0
  %expect_b = getelementptr [9 x i8], [9 x i8]* @.expect_escaped, i32 0, i32 0
  %cmp_a = call i32 @strcmp(i8* %a_s, i8* %expect_a)
  %cmp_b = call i32 @strcmp(i8* %b_s, i8* %expect_b)
  %a_ok = icmp eq i32 %cmp_a, 0
  %b_ok = icmp eq i32 %cmp_b, 0
  %strings_ok = and i1 %a_ok, %b_ok
  br i1 %strings_ok, label %success, label %fail_strings

success:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0

fail_run:
  call void @wam_state_free(%WamState* %vm)
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
