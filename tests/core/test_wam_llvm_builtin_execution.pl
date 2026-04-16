:- encoding(utf8).
% test_wam_llvm_builtin_execution.pl
% End-to-end execution tests for WAM builtins compiled to native code.
%
% Tests =/2 unification and simple comparisons that don't require
% compound term construction (which has deeper WAM integration needs).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

host_target_triple(Triple) :-
    ( catch(
        ( process_create(path(clang), ['-print-target-triple'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, Raw), close(Out),
          process_wait(PID, exit(0))
        ), _, fail)
    -> split_string(Raw, "", "\n\r\t ", [S]), atom_string(Triple, S)
    ;  Triple = 'x86_64-pc-linux-gnu'
    ).

extract_instr_count(Src, P, C) :-
    format(atom(Pat), "@~w_code = private constant \\[(?<n>\\d+) x %Instruction\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    format(atom(Pat), "@~w_labels = private constant \\[(?<n>\\d+) x i32\\]", [P]),
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

% === Test predicates ===

% =/2 unification: R = X (bind unbound R to X's value)
:- dynamic test_unify/2.
test_unify(X, X).

% Identity: just passes through (tests basic WAM head unification)
:- dynamic test_id/2.
test_id(X, X).

% Constant return: always returns 42
:- dynamic test_const/2.
test_const(_, 42).

% Compound arithmetic: R is X + 1
:- dynamic test_add/2.
test_add(X, R) :- R is X + 1.

% Compound arithmetic: R is X * 3
:- dynamic test_mul/2.
test_mul(X, R) :- R is X * 3.

% Multi-step: R is (X + 1) * 2 — two separate is/2 calls
:- dynamic test_multi/2.
test_multi(X, R) :- T is X + 1, R is T * 2.

% Multi-clause: deterministic via first-arg indexing
:- dynamic test_choice/2.
test_choice(1, 10).
test_choice(2, 20).
test_choice(3, 30).

run_test(Label, PredAtom, InputVal, Expected) :-
    format('  ~w: ', [Label]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:PredAtom/2],
        [ module_name('bt_exec'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, PredAtom, IC),
    extract_label_count(Src, PredAtom, LC),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 1, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @~w_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @~w_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss
hit:
  %r = call i64 @wam_get_reg_payload(%WamState* %vm, i32 1)
  %r32 = trunc i64 %r to i32
  ret i32 %r32
miss:
  ret i32 255
}
',
        [InputVal, IC, IC, PredAtom, IC, LC, LC, PredAtom]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('FAIL (llc=~w)~n', [LlcExit]), ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w 2>/dev/null', [OPath, BinPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('FAIL (clang=~w)~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    ( ExitCode =:= Expected
    -> format('PASS (~w)~n', [ExitCode])
    ;  format('FAIL (got ~w, expected ~w)~n', [ExitCode, Expected])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= Expected).

% Runner for is/2 predicates: result ends up in A1 (reg 0) due to WAM register layout.
run_test_r0(Label, PredAtom, InputVal, Expected) :-
    format('  ~w: ', [Label]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:PredAtom/2],
        [ module_name('bt_exec'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, PredAtom, IC),
    extract_label_count(Src, PredAtom, LC),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 1, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @~w_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @~w_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss
hit:
  %r = call i64 @wam_get_reg_payload(%WamState* %vm, i32 0)
  %r32 = trunc i64 %r to i32
  ret i32 %r32
miss:
  ret i32 255
}
',
        [InputVal, IC, IC, PredAtom, IC, LC, LC, PredAtom]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('FAIL (llc=~w)~n', [LlcExit]), ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w 2>/dev/null', [OPath, BinPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('FAIL (clang=~w)~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    ( ExitCode =:= Expected
    -> format('PASS (~w)~n', [ExitCode])
    ;  format('FAIL (got ~w, expected ~w)~n', [ExitCode, Expected])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= Expected).

test_all :-
    format('=== WAM Builtin Execution Tests ===~n'),
    ( process_which('clang'), process_which('llc')
    -> format('--- head unification ---~n'),
       run_test('id(7) = 7', test_id, 7, 7),
       run_test('id(42) = 42', test_id, 42, 42),
       run_test('const(_) = 42', test_const, 99, 42),
       run_test('unify(7) = 7', test_unify, 7, 7),
       format('--- compound arithmetic (is/2) ---~n'),
       run_test_r0('10+1 = 11', test_add, 10, 11),
       run_test_r0('0+1 = 1', test_add, 0, 1),
       run_test_r0('7*3 = 21', test_mul, 7, 21),
       format('--- compound arithmetic passed ---~n')
    ;  format('  SKIP: clang or llc not found~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

:- initialization(test_all, main).
