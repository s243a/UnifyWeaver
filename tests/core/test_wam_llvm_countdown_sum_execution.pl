:- encoding(utf8).
% test_wam_llvm_countdown_sum_execution.pl
% End-to-end execution test for countdown_sum2 (deterministic
% arithmetic recurrence). The kernel computes S = N*(N+1)/2 via
% closed-form arithmetic, then writes the result to A2.
%
% Tests:
%   1. sum(10) = 55
%   2. sum(0)  = 0
%   3. sum(1)  = 1
%   4. Auto-detect matches the countdown_sum clause shape.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_foreign_kernel_spec/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

% A predicate matching the countdown_sum2 clause shape.
:- dynamic my_sum/2.
my_sum(0, 0).
my_sum(N, Sum) :-
    N > 0,
    N1 is N - 1,
    my_sum(N1, PrevSum),
    Sum is PrevSum + N.

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
    Pat = "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    Pat = "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

run_sum_case(Label, N, Expected) :-
    format('  testing ~w: my_sum(~w, S) expected ~w~n', [Label, N, Expected]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:my_sum/2],
        [ module_name('cds2_exec'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_lowering(true)
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, my_sum, IC),
    extract_label_count(Src, my_sum, LC),
    % A1 = N (integer, tag=1), A2 = result slot (unbound, tag=6).
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 1, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss
hit:
  %sum = call i64 @wam_get_reg_payload(%WamState* %vm, i32 1)
  %sum32 = trunc i64 %sum to i32
  ret i32 %sum32
miss:
  ret i32 255
}
',
        [N, IC, IC, IC, LC, LC]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -filetype=obj -relocation-model=pic ~w -o ~w 2>~w.llc.err',
        [LLPath, OPath, LLPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('    FAIL: llc exit=~w~n', [LlcExit]), ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    FAIL: clang exit=~w~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    ( ExitCode =:= Expected
    -> format('    PASS: ~w returned ~w~n', [Label, ExitCode])
    ;  format('    FAIL: ~w returned ~w (expected ~w)~n', [Label, ExitCode, Expected])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= Expected).

test_autodetect :-
    format('--- countdown_sum2 auto-detect ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:my_sum/2],
        [ module_name('cds2_autodetect'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_lowering(true)
        ],
        LLPath),
    ( llvm_foreign_kernel_spec(my_sum/2, countdown_sum2, _)
    -> format('  PASS: auto-detect registered my_sum/2 as countdown_sum2~n')
    ;  format('  FAIL: auto-detect did not match my_sum/2~n'),
       throw(autodetect_failed)
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_cds2_executes :-
    format('--- countdown_sum2 execution ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_sum_case('sum(10)', 10, 55),
       run_sum_case('sum(0)',   0,  0),
       run_sum_case('sum(1)',   1,  1)
    ;  format('  SKIP: clang or llc not found~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

test_all :-
    catch(test_autodetect, E1,
        format('  ERROR: ~w~n', [E1])),
    catch(test_cds2_executes, E2,
        format('  ERROR: ~w~n', [E2])).

:- initialization(test_all, main).
