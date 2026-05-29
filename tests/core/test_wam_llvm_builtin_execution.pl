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

% M11: float `**` with negative exponent. Pre-M11, eval_arith was
% integer-only and returned 0 for negative exp; now eval_arith_value
% detects the **/2 pattern at the top level and computes via floating
% point. The test predicates just bind R via is/2; a custom driver
% (run_pow_test below) reads reg 0''s tag and float bits so we can
% verify the result type AND value without needing float arithmetic
% to compose through eval_arith yet (which is M12 work).
:- dynamic test_pow_neg_eighth/2.
test_pow_neg_eighth(_, R) :- R is 2 ** -3.   % expect Float(0.125)

:- dynamic test_pow_neg_fifth/2.
test_pow_neg_fifth(_, R) :- R is 5 ** -1.    % expect Float(0.2)

:- dynamic test_pow_neg_two/2.
test_pow_neg_two(_, R) :- R is 10 ** -2.     % expect Float(0.01)

% M13: full float propagation through +, -, *, / with mixed operands.
% These all return Float (any float operand floats the result, or the
% division is non-exact). Verified via the same run_pow_test driver
% which reads the result tag (must be Float = 2) and scales the
% double to an integer the shell exit code can carry.
:- dynamic test_div_inexact/2.
test_div_inexact(_, R) :- R is 1 / 4.        % Float(0.25)

:- dynamic test_div_exact/2.
test_div_exact(_, R) :- R is 6 / 2.          % Integer(3) -- exact

:- dynamic test_float_add/2.
test_float_add(_, R) :- X is 1 / 4, R is X + 1.   % 0.25 + 1 = 1.25

:- dynamic test_float_mul/2.
test_float_mul(_, R) :- X is 1 / 2, R is X * 3.   % 0.5 * 3 = 1.5

:- dynamic test_float_sub/2.
test_float_sub(_, R) :- X is 1 / 4, R is 2 - X.   % 2 - 0.25 = 1.75

:- dynamic test_float_neg/2.
test_float_neg(_, R) :- X is 1 / 4, R is -X.      % -0.25

:- dynamic test_float_chain/2.
test_float_chain(_, R) :- R is (1 / 2) + (1 / 4). % 0.5 + 0.25 = 0.75

:- dynamic test_float_pow_chain/2.
test_float_pow_chain(_, R) :- R is (2 ** -2) + 0.5. % 0.25 + 0.5 = 0.75

% M14: float-aware comparison ops. Pre-M14, builtin_gt/lt/etc read
% the register payload as raw i64 -- meaningless when one operand
% is a Float because float bits aren''t the numeric value. Also,
% comparisons over a Compound expression like `X > Y * 2` mis-fed
% the unmaterialised Compound pointer to icmp.
:- dynamic test_cmp_float_gt/2.
test_cmp_float_gt(_, R) :-
    X is 1 / 4,        % Float 0.25
    ( X > 0 -> R is 1 ; R is 0 ).   % expect 1

:- dynamic test_cmp_float_gt_neg/2.
test_cmp_float_gt_neg(_, R) :-
    X is -1 / 4,       % Float -0.25
    ( X > 0 -> R is 1 ; R is 0 ).   % expect 0

:- dynamic test_cmp_float_eq/2.
test_cmp_float_eq(_, R) :-
    X is 1 / 2,        % Float 0.5
    ( X =:= 0.5 -> R is 1 ; R is 0 ). % expect 1

:- dynamic test_cmp_compound_expr/2.
test_cmp_compound_expr(_, R) :-
    Y = 3,
    % Direct comparison of Y * 2 against 5 -- arithmetic eval on both sides.
    ( Y * 2 > 5 -> R is 1 ; R is 0 ).   % 6 > 5 -> 1

% M14: aggregate_all(sum(F), ...) over Float-producing inner goal.
% The sum_case now scans for any Float entry and switches to a
% double accumulator instead of integer add-of-bits.
:- dynamic node/1.
node(2).
node(4).
node(8).

:- dynamic test_sum_int/2.
test_sum_int(_, R) :-
    aggregate_all(sum(N), node(N), S),   % 2 + 4 + 8 = 14, stays Integer
    R is S.                              % route through is/2 -> A1

:- dynamic test_sum_float/2.
test_sum_float(_, R) :-
    aggregate_all(sum(W), (node(N), W is 1 / N), Sum),
    % Sum = 1/2 + 1/4 + 1/8 = 0.875
    R is Sum.

% M12: format/2 -- prints to stdout. Tested by capturing the shell
% stdout and comparing against expected string. Each predicate
% does ONE format call and then succeeds (no R binding -- the
% format-test driver reads from stdout, not the exit code).
:- dynamic test_fmt_literal/2.
test_fmt_literal(_, R) :-
    format('hello world~n', []),
    R is 1.

:- dynamic test_fmt_int/2.
test_fmt_int(_, R) :-
    format('n=~w~n', [42]),
    R is 1.

:- dynamic test_fmt_two_ints/2.
test_fmt_two_ints(_, R) :-
    format('~w + ~w~n', [3, 4]),
    R is 1.

:- dynamic test_fmt_atom/2.
test_fmt_atom(_, R) :-
    format('color=~w~n', [red]),
    R is 1.

:- dynamic test_fmt_tilde_escape/2.
test_fmt_tilde_escape(_, R) :-
    format('about ~~w~n', []),
    R is 1.

% M15: precision directives ~Nf (fixed-point) and ~Ne (scientific).
% Parses the digit run between ~ and f/e at runtime, then routes the
% next arg through printf "%.*f" / "%.*e" with the parsed precision.
:- dynamic test_fmt_6f/2.
test_fmt_6f(_, R) :-
    X is 1 / 4,                       % Float 0.25
    format('~6f~n', [X]),
    R is 1.

:- dynamic test_fmt_3f/2.
test_fmt_3f(_, R) :-
    X is 1 / 8,                       % Float 0.125
    format('d=~3f~n', [X]),
    R is 1.

:- dynamic test_fmt_int_via_f/2.
test_fmt_int_via_f(_, R) :-
    % Integer arg routed through ~Nf: should promote via sitofp.
    format('n=~2f~n', [7]),
    R is 1.

:- dynamic test_fmt_2e/2.
test_fmt_2e(_, R) :-
    X is 1 / 4000,                    % Float 0.00025
    format('e=~2e~n', [X]),
    R is 1.

% Runner for format/2 tests: captures stdout, compares against an
% expected string. The test predicate must end with `R is 1` so the
% module compiles cleanly and the call succeeds.
run_fmt_test(Label, PredAtom, ExpectedStdout) :-
    format('  ~w: ', [Label]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:PredAtom/2],
        [ module_name('fmt_t'),
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
  %a1 = insertvalue %Value %a1_0, i64 0, 1
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
  ret i32 0
}
', [IC, IC, IC, LC, LC, LC]),
    setup_call_cleanup(open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ), close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    atom_concat(LLPath, '.stdout', StdoutPath),
    format(atom(LlcCmd),
        'llc -O0 -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, _),
    format(atom(ClangCmd), 'clang -O0 ~w -o ~w 2>/dev/null', [OPath, BinPath]),
    shell(ClangCmd, _),
    format(atom(RunCmd), '~w > ~w 2>&1', [BinPath, StdoutPath]),
    shell(RunCmd, _),
    read_file_to_string(StdoutPath, ActualStdout, []),
    ( ActualStdout == ExpectedStdout
    -> format('PASS~n')
    ;  format('FAIL~n    got: ~q~n    expected: ~q~n',
              [ActualStdout, ExpectedStdout])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    catch(delete_file(StdoutPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ActualStdout == ExpectedStdout).

% Custom driver: returns 1000 * float_value cast to i32 so the shell
% exit code carries enough precision to verify the result. Tag is
% checked separately via the IR -- if it''s not Float (tag=2) we
% return 254 as a sentinel so a wrong-type result is distinguishable
% from a "right tag, wrong value" mismatch.
run_pow_test(Label, Preds, EntryPred, ScaleI, ExpectedI) :-
    format('  ~w: ', [Label]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    AllPreds = [user:EntryPred/2 | Preds],
    write_wam_llvm_project(AllPreds,
        [module_name('pow_t'), target_triple(Triple), target_datalayout('')],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, EntryPred, IC),
    extract_label_count(Src, EntryPred, LC),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 1, 0
  %a1 = insertvalue %Value %a1_0, i64 0, 1
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
  %r_raw = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %r_d = call %Value @wam_deref_value(%WamState* %vm, %Value %r_raw)
  %r_tag = extractvalue %Value %r_d, 0
  %is_float = icmp eq i32 %r_tag, 2
  br i1 %is_float, label %scale_float, label %wrong_tag
scale_float:
  %r_bits = extractvalue %Value %r_d, 1
  %r_d_val = bitcast i64 %r_bits to double
  %scale = sitofp i32 ~w to double
  %scaled = fmul double %r_d_val, %scale
  %scaled_i = fptosi double %scaled to i32
  ret i32 %scaled_i
wrong_tag:
  ret i32 254
miss:
  ret i32 255
}
', [IC, IC, IC, LC, LC, LC, ScaleI]),
    setup_call_cleanup(open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ), close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -O0 -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, _),
    format(atom(ClangCmd), 'clang -O0 ~w -o ~w 2>/dev/null', [OPath, BinPath]),
    shell(ClangCmd, _),
    shell(BinPath, ExitCode),
    ( ExitCode =:= ExpectedI
    -> format('PASS (~w)~n', [ExitCode])
    ; ExitCode =:= 254
    -> format('FAIL (result was not a Float -- got non-2 tag)~n')
    ;  format('FAIL (got ~w, expected ~w)~n', [ExitCode, ExpectedI])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= ExpectedI).

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
       format('--- M11 float ** with negative exponent ---~n'),
       run_pow_test('2**-3 -> Float 0.125, *1000 -> 125',
                    [], test_pow_neg_eighth, 1000, 125),
       run_pow_test('5**-1 -> Float 0.2, *100 -> 20',
                    [], test_pow_neg_fifth, 100, 20),
       run_pow_test('10**-2 -> Float 0.01, *10000 -> 100',
                    [], test_pow_neg_two, 10000, 100),
       format('--- M13 float arithmetic propagation ---~n'),
       run_pow_test('1/4 -> Float 0.25, *100 -> 25',
                    [], test_div_inexact, 100, 25),
       run_test_r0('6/2 -> Integer 3 (exact)', test_div_exact, 0, 3),
       run_pow_test('(1/4)+1 -> Float 1.25, *100 -> 125',
                    [], test_float_add, 100, 125),
       run_pow_test('(1/2)*3 -> Float 1.5, *100 -> 150',
                    [], test_float_mul, 100, 150),
       run_pow_test('2-(1/4) -> Float 1.75, *100 -> 175',
                    [], test_float_sub, 100, 175),
       run_pow_test('-(1/4) -> Float -0.25, *-100 -> 25',
                    [], test_float_neg, -100, 25),
       run_pow_test('(1/2)+(1/4) -> Float 0.75, *100 -> 75',
                    [], test_float_chain, 100, 75),
       run_pow_test('(2**-2)+0.5 -> Float 0.75, *100 -> 75',
                    [], test_float_pow_chain, 100, 75),
       format('--- M14 float-aware comparisons + float sum ---~n'),
       run_test_r0('1/4 > 0 -> 1', test_cmp_float_gt, 0, 1),
       run_test_r0('-1/4 > 0 -> 0', test_cmp_float_gt_neg, 0, 0),
       run_test_r0('1/2 =:= 0.5 -> 1', test_cmp_float_eq, 0, 1),
       run_test_r0('Y*2 > 5 when Y=3 -> 1', test_cmp_compound_expr, 0, 1),
       run_test_r0('sum(node) int -> 14', test_sum_int + [node/1], 0, 14),
       run_pow_test('sum(1/N) -> 0.875, *100 -> 87',
                    [node/1], test_sum_float, 100, 87),
       format('--- M12 format/2 stdout printing ---~n'),
       run_fmt_test('literal "hello world"', test_fmt_literal,
                    "hello world\n"),
       run_fmt_test('"n=~w~n" with 42', test_fmt_int,
                    "n=42\n"),
       run_fmt_test('"~w + ~w~n" with 3, 4', test_fmt_two_ints,
                    "3 + 4\n"),
       run_fmt_test('"color=~w~n" with red', test_fmt_atom,
                    "color=red\n"),
       run_fmt_test('"about ~~w~n" tilde escape', test_fmt_tilde_escape,
                    "about ~w\n"),
       format('--- M15 precision directives (~~Nf / ~~Ne) ---~n'),
       run_fmt_test('"~6f~n" with 0.25', test_fmt_6f,
                    "0.250000\n"),
       run_fmt_test('"d=~3f~n" with 0.125', test_fmt_3f,
                    "d=0.125\n"),
       run_fmt_test('"n=~2f~n" with integer 7', test_fmt_int_via_f,
                    "n=7.00\n"),
       run_fmt_test('"e=~2e~n" with 0.00025', test_fmt_2e,
                    "e=2.50e-04\n"),
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
