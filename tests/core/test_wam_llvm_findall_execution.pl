:- encoding(utf8).
% test_wam_llvm_findall_execution.pl
%
% M9: end-to-end execution test for findall / aggregate_all(bag/set).
%
% Pre-M9 the wam_apply_aggregation runtime helper returned a sentinel
% Atom for `collect` (the agg_type findall lowers to), so
% `findall(X, Goal, L)` always returned `L = Atom_0` instead of an
% actual list. Three coupled fixes in this milestone:
%
%   1. wam_apply_aggregation collect_case
%      (templates/targets/llvm_wam/state.ll.mustache): builds a
%      cons-cell chain from the accumulator entries, terminated by an
%      empty-list Atom (id read from @wam_empty_list_atom_id).
%
%   2. wam_llvm_target.pl write_wam_llvm_project: registers the [|]/2
%      functor string global eagerly + interns "[]" + emits
%      @wam_empty_list_atom_id at module scope. Programs that use
%      findall but never explicitly construct a cons cell now still
%      get a valid module.
%
%   3. wam_switch_on_constant deref + wam_finalize_aggregate bind-via-Ref:
%      first-arg-indexed multi-clause predicates called from inside an
%      aggregate body had their A1 passed as a Ref, the pre-M9
%      `wam_get_reg` (no deref) saw the Ref payload as a literal and
%      never matched; the result reg was also written via wam_set_reg
%      which overwrote the Ref instead of binding through it, so the
%      result never reached the caller's output variable.
%
% This test exercises all three by compiling
%
%     my_fact(11).  my_fact(22).  my_fact(33).
%     test_collect(L) :- findall(X, my_fact(X), L).
%
% then running a hand-rolled LLVM driver that calls test_collect with
% an unbound A1 backed by a heap cell, derefs the result, and walks
% the cons-cell chain counting entries.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3, clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

:- dynamic my_fact/1.
my_fact(11).
my_fact(22).
my_fact(33).

:- dynamic test_collect/1.
test_collect(L) :- findall(X, my_fact(X), L).

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
    format('--- M9 IR structure: collect_case + module globals ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:my_fact/1, user:test_collect/1],
        [ module_name('m9_struct'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    % Module globals must exist for collect_case + cons-cell build.
    forall(member(Pat-Tag, [
                '@wam_empty_list_atom_id = private constant i64'  - 'empty_list atom id',
                '@.fn__5B_7C_5D = private constant'               - '[|]/2 functor string'
            ]),
        ( sub_string(Src, _, _, _, Pat)
        -> format('  PASS: ~w global emitted~n', [Tag])
        ;  format('  FAIL: ~w global missing (~w)~n', [Tag, Pat]),
           throw(missing_global(Tag))
        )),
    % The collect_case label must be present in the @wam_apply_aggregation
    % function (its switch must have an `i32 4` arm now).
    ( sub_string(Src, _, _, _, 'collect_case')
    -> format('  PASS: collect_case label present in apply_aggregation~n')
    ;  format('  FAIL: collect_case missing~n'),
       throw(missing_collect_case)
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_findall_runs :-
    format('~n--- M9 execution: findall(X, my_fact(X), L) ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, S0), close(S0),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:my_fact/1, user:test_collect/1],
        [ module_name('m9_exec'),
          target_triple(Triple),
          target_datalayout('')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    re_matchsub("@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
                Src, M1, []), get_dict(n, M1, IcStr), number_string(IC, IcStr),
    re_matchsub("@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
                Src, M2, []), get_dict(n, M2, LcStr), number_string(LC, LcStr),
    % test_collect's entry sets PC to a value we can extract from the
    % emitted entry function. The relevant line shape is
    %   call void @wam_set_pc(%WamState* %vm, i32 <PC>)
    % within the @test_collect definition.
    extract_test_collect_pc(Src, PC),
    % Driver IR: allocate state, push an unbound heap cell, set A1 to
    % a Ref pointing at it, run test_collect, deref A1, count cons cells.
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
    i32 ~w,
    i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
    i32 ~w)
  %unb = call %Value @value_unbound(i8* null)
  %addr = call i32 @wam_heap_push(%WamState* %vm, %Value %unb)
  %a1 = call %Value @value_ref(i32 %addr)
  call void @wam_set_pc(%WamState* %vm, i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %read, label %fail

read:
  %a1_now = call %Value @wam_get_reg_deref(%WamState* %vm, i32 0)
  br label %walk

walk:
  %cur = phi %Value [ %a1_now, %read ], [ %next_deref, %tail_cell ]
  %n = phi i32 [ 0, %read ], [ %n_inc, %tail_cell ]
  %tag = extractvalue %Value %cur, 0
  %is_compound = icmp eq i32 %tag, 3
  br i1 %is_compound, label %tail_cell, label %done

tail_cell:
  ; Cons cell: args[0] = head, args[1] = tail (Ref or empty-list Atom).
  %payload = extractvalue %Value %cur, 1
  %cp = inttoptr i64 %payload to %Compound*
  %args_slot = getelementptr %Compound, %Compound* %cp, i32 0, i32 2
  %args = load %Value*, %Value** %args_slot
  %tail_slot = getelementptr %Value, %Value* %args, i32 1
  %next = load %Value, %Value* %tail_slot
  %next_deref = call %Value @wam_deref_value(%WamState* %vm, %Value %next)
  %n_inc = add i32 %n, 1
  br label %walk

done:
  call void @wam_state_free(%WamState* %vm)
  ret i32 %n

fail:
  call void @wam_state_free(%WamState* %vm)
  ret i32 255
}
', [IC, IC, IC, LC, LC, LC, PC]),
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
    shell(BinPath, ExitCode),
    Expected = 3,
    ( ExitCode =:= Expected
    -> format('  PASS: findall returned a ~w-element list~n', [Expected])
    ; ExitCode =:= 255
    -> format('  FAIL: run_loop returned false (test_collect failed)~n'),
       throw(test_collect_failed)
    ;  format('  FAIL: expected ~w elements, got ~w~n', [Expected, ExitCode]),
       throw(wrong_count(ExitCode))
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs.

extract_test_collect_pc(Src, PC) :-
    % Find the test_collect entry function and extract its set_pc value.
    re_matchsub(
        "define i1 @test_collect\\([^)]*\\)[^{]*\\{[^}]*?call void @wam_set_pc\\(%WamState\\* %vm, i32 (?<pc>\\d+)\\)",
        Src, M, []),
    get_dict(pc, M, PcStr),
    number_string(PC, PcStr).

test_all :-
    ( process_which('clang'), process_which('llc')
    -> catch(
         ( test_ir_structure,
           test_findall_runs,
           format('~n=== All M9 findall-execution tests passed ===~n')
         ),
         E,
         ( format(user_error, '  ERROR: ~w~n', [E]),
           halt(1) ))
    ;  format('  SKIP: clang or llc not on PATH~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

:- initialization(test_all, main).
