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

extract_instr_count(Src, _P, C) :-
    % After the cross-pred label fix, all wam-fallback predicates share
    % @module_code / @module_labels. The pred-name argument is kept for
    % backward source compatibility but is ignored in the match.
    re_matchsub("@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
                Src, M, []),
    get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, _P, C) :-
    re_matchsub("@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
                Src, M, []),
    get_dict(n, M, NS), number_string(C, NS).

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

% Compound arithmetic: R is X ** 3 (integer power)
:- dynamic test_pow/2.
test_pow(X, R) :- R is X ** 3.

% Compound arithmetic: R is 2 ** X (variable exponent)
:- dynamic test_pow2/2.
test_pow2(X, R) :- R is 2 ** X.

% Multi-step: R is (X + 1) * 2 — two separate is/2 calls
:- dynamic test_multi/2.
test_multi(X, R) :- T is X + 1, R is T * 2.

% Multi-clause: deterministic via first-arg indexing
:- dynamic test_choice/2.
test_choice(1, 10).
test_choice(2, 20).
test_choice(3, 30).

% M10: list construction + recursive multi-clause traversal.
% Exercises put_list / get_list (the Compound representation fix) and
% the disjoint X/Y register layout: my_mem's first clause has no
% `allocate` and writes X1, X2 -- under the pre-M10 ABI those slot
% indices aliased the caller's Y1, Y2 and silently corrupted the
% outer `R` variable.
:- dynamic my_mem/2.
my_mem(X, [X|_]).
my_mem(X, [_|T]) :- my_mem(X, T).

:- dynamic test_mem_first/2.
test_mem_first(_, R) :- L = [11, 22, 33], my_mem(11, L), R = 11.

:- dynamic test_mem_second/2.
test_mem_second(_, R) :- L = [11, 22, 33], my_mem(22, L), R = 22.

:- dynamic test_mem_third/2.
test_mem_third(_, R) :- L = [11, 22, 33], my_mem(33, L), R = 33.

% M10: msort/2 builtin -- sort a list of integers, return the head.
% The trailing `R is X` forces the result into A1 so the run_test_r0
% driver (which reads reg 0) sees the value.
:- dynamic test_msort_head/2.
test_msort_head(_, R) :-
    msort([33, 11, 22], Sorted),
    Sorted = [X|_],
    R is X.

:- dynamic test_msort_second/2.
test_msort_second(_, R) :-
    msort([33, 11, 22], Sorted),
    Sorted = [_, X|_],
    R is X.

:- dynamic test_msort_third/2.
test_msort_third(_, R) :-
    msort([33, 11, 22], Sorted),
    Sorted = [_, _, X],
    R is X.

% Idempotent on a single element.
:- dynamic test_msort_one/2.
test_msort_one(_, R) :-
    msort([42], Sorted),
    Sorted = [X],
    R is X.

% msort preserves duplicates (unlike sort/2): expect [1, 1, 2, 3, 3].
:- dynamic test_msort_dups/2.
test_msort_dups(_, R) :-
    msort([3, 1, 3, 2, 1], Sorted),
    Sorted = [_, X|_],
    R is X.

% M10: setof/3 via aggregate_all -> sort + dedup. The agg_type_id
% routes set/setof to id 6, which inserts a sort+dedup pass before
% building the cons-cell chain. Drives off a small dynamic fact
% base so the inner goal yields a deterministic multi-set.
:- dynamic color/1.
color(red).
color(blue).
color(red).    % duplicate
color(green).
color(blue).   % duplicate

:- dynamic test_setof_count/2.
test_setof_count(_, R) :-
    setof(C, color(C), Cs),
    length(Cs, N),
    R is N.

% M11: findall over a Compound template. Pre-M11, end_aggregate
% only dereferenced atomic values; Compound entries shared args
% pointers that pointed into the per-iteration heap region, so
% every accumulated entry collapsed to the LAST iteration's values
% after backtrack rewound the heap. wam_freeze_value now deep-
% copies the Compound's args onto the arena so each entry is
% self-contained.
:- dynamic pair/2.
pair(1, 10).
pair(2, 20).
pair(3, 30).

:- dynamic test_findall_pair_first_key/2.
test_findall_pair_first_key(_, R) :-
    findall(K-V, pair(K, V), Pairs),
    Pairs = [K1-_|_],
    R is K1.

:- dynamic test_findall_pair_first_val/2.
test_findall_pair_first_val(_, R) :-
    findall(K-V, pair(K, V), Pairs),
    Pairs = [_-V1|_],
    R is V1.

:- dynamic test_findall_pair_last_key/2.
test_findall_pair_last_key(_, R) :-
    findall(K-V, pair(K, V), Pairs),
    Pairs = [_, _, K3-_],
    R is K3.

:- dynamic test_findall_pair_count/2.
test_findall_pair_count(_, R) :-
    findall(K-V, pair(K, V), Pairs),
    length(Pairs, N),
    R is N.

% M10: \+/1 negation-as-failure via inline (G -> fail ; true) rewrite
% in the WAM compiler. No runtime metacall: the bytecode goes through
% the existing if-then-else (try_me_else / cut_ite / trust_me) chain.
%
% Coverage is partial: the inline rewrite only behaves correctly when
% the inner goal FAILS (typical "negation of an absent fact" use).
% \+ of a SUCCEEDING goal currently mis-succeeds because the LLVM
% target's cut_ite naively pops one CP, which is the inner retry CP
% rather than the ITE guard CP -- proper get_level/cut Y_n is M11.
:- dynamic in_basket/1.
in_basket(apple).
in_basket(bread).
in_basket(milk).

% \+ of an absent item -> succeeds.
:- dynamic test_not_absent/2.
test_not_absent(_, R) :-
    \+ in_basket(soap),
    R is 7.

% \+ of a present item -> fails the whole goal -> predicate fails,
% run_test_r0 maps that to exit 255 (the `miss:` branch).
:- dynamic test_not_present/2.
test_not_present(_, R) :-
    \+ in_basket(apple),
    R is 7.

% Chain: \+ followed by another check. Exercises that the inline
% expansion leaves the env / Y-reg state consistent for the next
% goal.
:- dynamic test_not_then/2.
test_not_then(_, R) :-
    \+ in_basket(soap),
    in_basket(bread),
    R is 13.

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
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
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
        [InputVal, IC, IC, IC, LC, LC, LC]),
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
% Accepts either a single predicate atom (compiled solo) or a Pred/Helpers
% pair where Helpers is a list of additional Pred/Arity to include in the
% module (used by M10 list tests that need both the entry pred and the
% user-defined member/2 it calls).
run_test_r0(Label, Pred, InputVal, Expected) :-
    ( Pred = PredAtom + Helpers -> true
    ; PredAtom = Pred, Helpers = []
    ),
    format('  ~w: ', [Label]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    % Entry must be FIRST so it gets start_pc = 0 (run_test_r0's driver
    % calls @run_loop directly without wam_set_pc -- new VMs start at PC 0).
    findall(user:P/A, member(P/A, Helpers), HelperPreds),
    AllPreds = [user:PredAtom/2 | HelperPreds],
    write_wam_llvm_project(
        AllPreds,
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
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss
hit:
  ; M10: deref reg 0 before reading the payload. get_variable now
  ; promotes a direct-Unbound input Ai to a Ref-into-heap so callees
  ; can bind through it; once is/2 binds, reg 0 is a Ref whose
  ; payload is the heap address, NOT the result value. Deref-then-
  ; payload gets the actual integer the test expects.
  %r_raw = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %r_d = call %Value @wam_deref_value(%WamState* %vm, %Value %r_raw)
  %r_pay = extractvalue %Value %r_d, 1
  %r32 = trunc i64 %r_pay to i32
  ret i32 %r32
miss:
  ret i32 255
}
',
        [InputVal, IC, IC, IC, LC, LC, LC]),
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
       run_test_r0('2**3 = 8', test_pow, 2, 8),
       run_test_r0('3**3 = 27', test_pow, 3, 27),
       run_test_r0('5**3 = 125', test_pow, 5, 125),
       run_test_r0('2**5 = 32', test_pow2, 5, 32),
       run_test_r0('2**7 = 128', test_pow2, 7, 128),
       format('--- M10 list traversal (put_list + member-style) ---~n'),
       run_test_r0('mem_first [11,22,33] -> 11', test_mem_first + [my_mem/2], 0, 11),
       run_test_r0('mem_second [11,22,33] -> 22', test_mem_second + [my_mem/2], 0, 22),
       run_test_r0('mem_third [11,22,33] -> 33', test_mem_third + [my_mem/2], 0, 33),
       format('--- M10 msort/2 builtin ---~n'),
       run_test_r0('msort_head [33,11,22] -> 11', test_msort_head, 0, 11),
       run_test_r0('msort_second [33,11,22] -> 22', test_msort_second, 0, 22),
       run_test_r0('msort_third [33,11,22] -> 33', test_msort_third, 0, 33),
       run_test_r0('msort_one [42] -> 42', test_msort_one, 0, 42),
       run_test_r0('msort_dups [3,1,3,2,1] -> 1', test_msort_dups, 0, 1),
       format('--- M10 setof/3 (sort + dedup) ---~n'),
       run_test_r0('setof color/1 count -> 3',
                   test_setof_count + [color/1], 0, 3),
       format('--- M11 findall over compound template ---~n'),
       run_test_r0('findall pair(K,V), first key -> 1',
                   test_findall_pair_first_key + [pair/2], 0, 1),
       run_test_r0('findall pair(K,V), first val -> 10',
                   test_findall_pair_first_val + [pair/2], 0, 10),
       run_test_r0('findall pair(K,V), third key -> 3',
                   test_findall_pair_last_key + [pair/2], 0, 3),
       run_test_r0('findall pair(K,V), count -> 3',
                   test_findall_pair_count + [pair/2], 0, 3),
       format('--- M10 \\+ negation-as-failure (inline rewrite) ---~n'),
       run_test_r0('\\+ in_basket(soap) -> succeeds, R=7',
                   test_not_absent + [in_basket/1], 0, 7),
       run_test_r0('\\+ in_basket(apple) -> fails (exit 255)',
                   test_not_present + [in_basket/1], 0, 255),
       run_test_r0('\\+ then in_basket(bread), R=13',
                   test_not_then + [in_basket/1], 0, 13),
       format('--- multi-clause (first-arg indexing) ---~n'),
       run_test('choice(1) = 10', test_choice, 1, 10),
       run_test('choice(2) = 20', test_choice, 2, 20),
       run_test('choice(3) = 30', test_choice, 3, 30)
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
