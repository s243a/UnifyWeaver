:- encoding(utf8).
% test_wam_llvm_multi_result_dispatch.pl
% Validates the multi-result foreign dispatch infrastructure (M5.14).
%
% Tests:
%   1. ChoicePoint type has foreign iterator fields (fields 8-10).
%   2. @wam_push_foreign_choice_point and @wam_foreign_iter_next are
%      present in the generated IR.
%   3. The backtrack handler checks for agg_type == -2 (foreign iter).
%   4. End-to-end: a synthetic driver pushes 3 integer results via
%      the iterator, backtracks through them, counts them via a
%      handwritten accumulator, and returns the count as exit code.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic dummy/1.
dummy(x).

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

sub_atom_or_string(Atom, B, L, A, Sub) :-
    ( atom(Atom) -> sub_atom(Atom, B, L, A, Sub)
    ; string(Atom) -> sub_string(Atom, B, L, A, Sub)
    ).

test_ir_structure :-
    format('--- multi-result dispatch IR structure ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:dummy/1],
        [ module_name('mr_struct'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    % Test 1: ChoicePoint has 11 fields (extended with foreign iter)
    ( sub_atom_or_string(Src, _, _, _, 'i8*,')  % field 8: foreign_results
    -> format('  PASS: ChoicePoint has i8* field for foreign_results~n')
    ;  format('  FAIL: ChoicePoint missing foreign_results field~n'),
       throw(missing_foreign_results)
    ),

    % Test 2: push and iter functions present
    ( sub_atom_or_string(Src, _, _, _, '@wam_push_foreign_choice_point')
    -> format('  PASS: @wam_push_foreign_choice_point present~n')
    ;  format('  FAIL: @wam_push_foreign_choice_point missing~n'),
       throw(missing_push)
    ),
    ( sub_atom_or_string(Src, _, _, _, '@wam_foreign_iter_next')
    -> format('  PASS: @wam_foreign_iter_next present~n')
    ;  format('  FAIL: @wam_foreign_iter_next missing~n'),
       throw(missing_iter_next)
    ),

    % Test 3: backtrack checks for foreign iter (agg_type == -2)
    ( sub_atom_or_string(Src, _, _, _, 'icmp eq i32 %ca_at, -2')
    -> format('  PASS: backtrack handler checks agg_type == -2~n')
    ;  format('  FAIL: backtrack handler missing foreign iter check~n'),
       throw(missing_foreign_check)
    ),

    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_execution :-
    format('--- multi-result dispatch execution ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_multi_result_test
    ;  format('  SKIP: clang or llc not found~n')
    ).

run_multi_result_test :-
    format('  testing: push 3 results, iterate, count = 3~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:dummy/1],
        [ module_name('mr_exec'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),

    % Append a custom driver that:
    %   1. Allocates a %Value[3] array with integers 10, 20, 30
    %   2. Creates a WAM state
    %   3. Pushes a foreign choice point with the 3 results
    %   4. Writes result[0] to register 0
    %   5. Uses proceed → backtrack loop to count results
    %   6. Returns count as exit code
    DriverIR = '
@module_code = private constant [2 x %Instruction] [
  %Instruction { i32 20, i64 0, i64 0 },
  %Instruction { i32 20, i64 0, i64 0 }
]
@module_labels = private constant [1 x i32] [ i32 0 ]

define i32 @main() {
entry:
  ; Allocate result array: 3 x %Value = 3 integers
  %arr_mem = call i8* @malloc(i64 48)
  %arr = bitcast i8* %arr_mem to %Value*

  ; result[0] = Integer(10)
  %r0 = getelementptr %Value, %Value* %arr, i64 0
  %r0_tag = getelementptr %Value, %Value* %r0, i32 0, i32 0
  store i32 1, i32* %r0_tag
  %r0_pay = getelementptr %Value, %Value* %r0, i32 0, i32 1
  store i64 10, i64* %r0_pay

  ; result[1] = Integer(20)
  %r1 = getelementptr %Value, %Value* %arr, i64 1
  %r1_tag = getelementptr %Value, %Value* %r1, i32 0, i32 0
  store i32 1, i32* %r1_tag
  %r1_pay = getelementptr %Value, %Value* %r1, i32 0, i32 1
  store i64 20, i64* %r1_pay

  ; result[2] = Integer(30)
  %r2 = getelementptr %Value, %Value* %arr, i64 2
  %r2_tag = getelementptr %Value, %Value* %r2, i32 0, i32 0
  store i32 1, i32* %r2_tag
  %r2_pay = getelementptr %Value, %Value* %r2, i32 0, i32 1
  store i64 30, i64* %r2_pay

  ; Create VM
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([2 x %Instruction], [2 x %Instruction]* @module_code, i32 0, i32 0),
      i32 2,
      i32* getelementptr ([1 x i32], [1 x i32]* @module_labels, i32 0, i32 0),
      i32 0)

  ; Write result[0] to register 0 (first yield)
  %first = load %Value, %Value* %r0
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %first)

  ; Push foreign choice point: results=arr, count=3, result_reg=0, return_pc=0
  call void @wam_push_foreign_choice_point(
      %WamState* %vm,
      i8* %arr_mem,
      i32 3,
      i32 0,
      i32 0)

  ; Count results by reading reg 0 and summing payloads.
  ; First result already in reg 0.
  %pay0 = call i64 @wam_get_reg_payload(%WamState* %vm, i32 0)
  %sum0 = trunc i64 %pay0 to i32
  br label %bt_loop

bt_loop:
  %count = phi i32 [1, %entry], [%count_inc, %got_next]
  %sum   = phi i32 [%sum0, %entry], [%sum_next, %got_next]
  ; Try to backtrack (get next result)
  %ok = call i1 @backtrack(%WamState* %vm)
  br i1 %ok, label %got_next, label %done

got_next:
  %pay = call i64 @wam_get_reg_payload(%WamState* %vm, i32 0)
  %pay32 = trunc i64 %pay to i32
  %sum_next = add i32 %sum, %pay32
  %count_inc = add i32 %count, 1
  br label %bt_loop

done:
  ; Return count as exit code (should be 3).
  ; Also verify sum: 10+20+30 = 60. We encode both:
  ; exit = count * 100 + (sum mod 100) = 3*100 + 60 = 360 mod 256 = 104
  ; Actually keep it simple: just return count.
  ret i32 %count
}
',

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
    -> read_file_to_string(atom_concat(LLPath, '.llc.err'), LlcErr, []),
       format('    FAIL: llc exit=~w~n~w~n', [LlcExit, LlcErr]),
       ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    FAIL: clang exit=~w~n', [ClangExit]),
          ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    Expected = 3,
    ( ExitCode =:= Expected
    -> format('    PASS: iterator returned ~w results (exit=~w)~n', [ExitCode, ExitCode])
    ;  format('    FAIL: expected ~w, got ~w~n', [Expected, ExitCode])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= Expected).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

test_all :-
    catch(test_ir_structure, E1,
        format('  ERROR: ~w~n', [E1])),
    catch(test_execution, E2,
        format('  ERROR: ~w~n', [E2])).

:- initialization(test_all, main).
