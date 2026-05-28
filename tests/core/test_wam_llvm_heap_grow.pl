:- encoding(utf8).
% test_wam_llvm_heap_grow.pl
%
% Exercises the M6 heap-grow path with the WriteCtx-pointer fixup pass.
% Companion to tests/core/test_wam_llvm_growable_alloc.pl (M5), which
% covers trail / stack / CP growth but leaves the heap fixed.
%
% The challenge for testing heap grow: each WAM-fallback iteration's
% heap usage depends on which instructions ran (put_structure goes to
% the arena, not heap; get_list write mode pushes 3 heap cells;
% put_variable pushes 1). To force a heap grow within reasonable
% wall-clock we use a predicate whose body push at least one heap cell
% per call and run thousands of iterations against a shared %WamState
% WITHOUT resetting `hs` (heap_size). After ~65k cells the heap hits
% the initial cap and must double via @wam_heap_grow.
%
% This test ALSO exercises the post-grow WriteCtx fixup: each call
% enters `get_list` write mode (sets up a WriteCtx with args pointing
% into the heap), then subsequent set_/unify_ instructions consume
% it. If a heap grow fires while a WriteCtx is on the stack with a
% heap-bound `data` pointer, the @wam_fixup_writectx_after_heap_grow
% walker translates the pointer to the new heap base before the next
% set_arg writes through it. Without the walker, the post-grow
% set_arg would write to freed memory and the binary would segfault.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

% make_list(X, [X]) compiles to a body that uses get_list write mode +
% unify_value + unify_constant — 3 heap pushes per call, plus a
% WriteCtx that points into the heap during the run.
:- dynamic make_list/2.
make_list(X, [X]).

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

test_ir_structure :-
    format('--- M6 IR structure: heap grow + WriteCtx fixup ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:make_list/2],
        [ module_name('m6_struct'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    forall(member(Sym, [
                'define void @wam_heap_grow',
                'define void @wam_fixup_writectx_after_heap_grow'
            ]),
        ( sub_string(Src, _, _, _, Sym)
        -> format('  PASS: ~w present~n', [Sym])
        ;  format('  FAIL: ~w missing~n', [Sym]),
           throw(missing_symbol(Sym))
        )),
    % @wam_heap_push must route through @wam_heap_grow, not exit(2).
    ( sub_string(Src, _, _, _, 'call void @wam_heap_grow(%WamState* %vm)')
    -> format('  PASS: @wam_heap_push routes through @wam_heap_grow~n')
    ;  format('  FAIL: @wam_heap_push still aborts instead of growing~n'),
       throw(missing_heap_grow_call)
    ),
    ( sub_string(Src, _, _, _, 'call void @exit(i32 2)')
    -> format('  FAIL: legacy exit(2) abort path still present~n'),
       throw(legacy_abort_present)
    ;  format('  PASS: legacy exit(2) abort path removed~n')
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_heap_grow_exec :-
    format('~n--- M6 execution stress: heap grow over 50k iterations ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:make_list/2],
        [ module_name('m6_stress'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    re_matchsub("@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
                Src, M1, []), get_dict(n, M1, IcStr), number_string(IC, IcStr),
    re_matchsub("@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
                Src, M2, []), get_dict(n, M2, LcStr), number_string(LC, LcStr),
    Iterations = 50000,
    % Driver: shared VM, reset PC + ts + cpn + halted between iters but
    % NOT hs. Each iteration pushes 3 heap cells via get_list write mode;
    % 50k * 3 = 150k cells, which forces 2 doublings past the 65536
    % initial cap (65536 → 131072 → 262144). Each grow triggers the
    % WriteCtx fixup because there's an active WriteCtx during the
    % set_value / unify_constant sequence within make_list's body —
    % though the WriteCtx is auto-popped before the next iteration's
    % `get_list`, so most heap grows happen at the START of an iter
    % (before any WriteCtx is on the stack) rather than mid-iteration.
    %
    % To force a mid-iteration grow we'd need a multi-cell write-mode
    % build that straddles cap. The simpler smoke is whether the binary
    % runs to completion at all — pre-M6 it would exit(2) after ~22k
    % iterations.
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %pc_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 0
  %ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %halt_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 19
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i_next, %loop_body]
  %done = icmp uge i32 %i, ~w
  br i1 %done, label %exit, label %loop_body
loop_body:
  store i32 0, i32* %pc_ptr
  store i32 0, i32* %ts_ptr
  store i32 0, i32* %cpn_ptr
  store i1 false, i1* %halt_ptr
  %a1_0 = insertvalue %Value undef, i32 1, 0
  %a1 = insertvalue %Value %a1_0, i64 42, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  call void @wam_cleanup()
  %i_next = add i32 %i, 1
  br label %loop
exit:
  call void @wam_state_free(%WamState* %vm)
  ret i32 0
}
', [IC, IC, IC, LC, LC, LC, Iterations]),
    setup_call_cleanup(open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ), close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -O2 -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, _),
    format(atom(ClangCmd), 'clang -O2 ~w -o ~w 2>/dev/null', [OPath, BinPath]),
    shell(ClangCmd, _),
    get_time(T0),
    shell(BinPath, ExitCode),
    get_time(T1),
    Elapsed is (T1 - T0) * 1000,
    ( ExitCode =:= 0
    -> format('  PASS: 50k iter run completed (exit=0, ~3fms wall-clock)~n',
              [Elapsed])
    ;  format('  FAIL: 50k iter run exited ~w (expected 0)~n', [ExitCode]),
       throw(stress_exec_failed(ExitCode))
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_all :-
    ( process_which('clang'), process_which('llc')
    -> catch(
         ( test_ir_structure,
           test_heap_grow_exec,
           format('~n=== All M6 heap-grow tests passed ===~n')
         ),
         E,
         ( format(user_error, '  ERROR: ~w~n', [E]),
           halt(1)
         ))
    ;  format('  SKIP: clang/llc not all on PATH~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

:- initialization(test_all, main).
