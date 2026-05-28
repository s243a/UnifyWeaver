:- encoding(utf8).
% test_wam_llvm_growable_alloc.pl
%
% Exercises the M5 doubling-realloc grow paths for the trail, stack,
% and choice-point array. The wam_state_new initial caps (trail 65536,
% CPs 1024, stack 1024) are too high for a smoke test to exceed in a
% reasonable wall-clock budget, so this test:
%
%   1. IR-structure check — confirm `@wam_trail_grow`, `@wam_stack_grow`,
%      and `@wam_cp_grow` are emitted into the module, and that the
%      push sites (`@wam_trail_binding`, `@wam_push_unify_ctx`,
%      `@wam_push_write_ctx`, `@wam_push_foreign_choice_point`, and the
%      try_me_else / allocate @step cases) call the ensure_capacity
%      helpers.
%
%   2. Execution stress — call a simple WAM-fallback predicate ~200k
%      times against ONE shared %WamState without resetting the trail.
%      Each invocation trails ~4 register bindings, so 200k iterations
%      hit ~800k trail entries — well past the 65536 initial cap. The
%      test verifies the resulting binary runs to completion without
%      OOM-aborting (which is what the pre-M5 fixed-size code did).

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic add1/2.
add1(X, R) :- R is X + 1.

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
    format('--- M5 IR structure: grow helpers + ensure_capacity calls ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:add1/2],
        [ module_name('m5_struct'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    forall(member(Sym, [
                'define void @wam_trail_grow',
                'define void @wam_stack_grow',
                'define void @wam_cp_grow',
                'define void @wam_stack_ensure_capacity',
                'define void @wam_cp_ensure_capacity'
            ]),
        ( sub_string(Src, _, _, _, Sym)
        -> format('  PASS: ~w present~n', [Sym])
        ;  format('  FAIL: ~w missing~n', [Sym]),
           throw(missing_symbol(Sym))
        )),
    % Trail / stack / CP push sites must route through the grow helpers.
    forall(member(Pattern-Site, [
                'call void @wam_trail_grow'             - 'wam_trail_binding',
                'call void @wam_stack_ensure_capacity'  - 'allocate or push_*',
                'call void @wam_cp_ensure_capacity'     - 'try_me_else / begin_aggregate / push_foreign_cp'
            ]),
        ( sub_string(Src, _, _, _, Pattern)
        -> format('  PASS: ~w wired at ~w~n', [Pattern, Site])
        ;  format('  FAIL: ~w not wired (~w)~n', [Pattern, Site]),
           throw(unwired(Pattern))
        )),
    % @realloc declaration must be in the module (or stubbed for WASM).
    ( sub_string(Src, _, _, _, '@realloc')
    -> format('  PASS: @realloc declared / stubbed~n')
    ;  format('  FAIL: @realloc not in module~n'),
       throw(missing_realloc)
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_trail_grow_exec :-
    format('~n--- M5 execution stress: trail grow over 200k iterations ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:add1/2],
        [ module_name('m5_stress'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    re_matchsub("@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
                Src, M1, []), get_dict(n, M1, IcStr), number_string(IC, IcStr),
    re_matchsub("@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
                Src, M2, []), get_dict(n, M2, LcStr), number_string(LC, LcStr),
    % Driver: allocate ONE %WamState, loop 200k times calling add1's
    % bytecode WITHOUT resetting trail / heap (only PC + halt flag).
    % Each iteration trails ~4 entries via get_variable/put_value/is.
    % At iteration ~16k the initial trail (cap=65536) hits cap and
    % must grow — repeatedly — to reach the 200k-iter target.
    %
    % We also reset arena per-iter via wam_cleanup (the existing M3
    % path) since put_structure builds a fresh +/2 compound each call.
    Iterations = 200000,
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  %pc_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 0
  %hs_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 6
  %cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %halt_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 19
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i_next, %loop_body]
  %done = icmp uge i32 %i, ~w
  br i1 %done, label %exit, label %loop_body
loop_body:
  ; Reset only what would otherwise OOM each iter: PC, heap, CP count,
  ; halted. Crucially we do NOT reset the trail size — bindings
  ; accumulate, forcing the trail to grow past its initial cap.
  store i32 0, i32* %pc_ptr
  store i32 0, i32* %hs_ptr
  store i32 0, i32* %cpn_ptr
  store i1 false, i1* %halt_ptr
  %a1_0 = insertvalue %Value undef, i32 1, 0
  %a1 = insertvalue %Value %a1_0, i64 17, 1
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
    -> format('  PASS: 200k iter run completed (exit=0, ~3fms wall-clock)~n',
              [Elapsed])
    ;  format('  FAIL: 200k iter run exited ~w (expected 0)~n', [ExitCode]),
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
           test_trail_grow_exec,
           format('~n=== All M5 growable-allocator tests passed ===~n')
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
