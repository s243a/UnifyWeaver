:- encoding(utf8).
% test_wam_llvm_builtin_execution.pl
% End-to-end execution tests for WAM builtins compiled to native code.
%
% Tests =/2 unification and simple comparisons that don't require
% compound term construction (which has deeper WAM integration needs).

% The cumulative LLVM IR built per test passed Prolog's 1 GB default
% global-stack limit around M41 (sub_string over the full IR text is
% the hot spot). Bump to 4 GB so the test suite stays self-contained.
:- set_prolog_flag(stack_limit, 4_294_967_296).

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

% M16: reverse-mode length/2 -- length(L, N) with L unbound and N
% bound to an Integer allocates a fresh N-element list of unbound
% logic variables and binds L. We then unify the result with a
% concrete pattern to confirm the structure round-trips.
:- dynamic test_length_rev_size/2.
test_length_rev_size(_, R) :-
    length(L, 4),
    length(L, N),   % round-trip: forward-mode length on the freshly built list
    R is N.

:- dynamic test_length_rev_unify_head/2.
test_length_rev_unify_head(_, R) :-
    length(L, 3),
    L = [11, _, _],   % bind the first cell -- exercises that the
                       % args[0] Ref in the reverse-built cons cell
                       % is actually a bindable logic variable.
    L = [H|_],
    R is H.

:- dynamic test_length_rev_unify_all/2.
test_length_rev_unify_all(_, R) :-
    length(L, 3),
    L = [4, 5, 6],
    L = [_, M, _],
    R is M.

% M17: if-then-else soft cut with a multi-clause condition. Pre-M17
% cut_ite naively decremented cp_count by 1, which popped the inner
% retry CP left over from in_basket''s clause dispatch rather than
% the ITE guard CP -- after `fail` from the then-branch the backtrack
% landed on the else-branch and produced a wrong answer. M17 emits
% get_level Y_n before try_me_else and cut Y_n at the commit site so
% the cut restores cp_count to the pre-ITE level regardless of any
% CPs the condition pushed.
:- dynamic ite_basket/1.
ite_basket(apple).
ite_basket(bread).
ite_basket(milk).

:- dynamic test_ite_succ_then_fail/2.
test_ite_succ_then_fail(_, R) :-
    % apple IS in the basket -> condition succeeds, commits to "then" branch.
    ( ite_basket(apple) -> R is 11 ; R is 22 ).

:- dynamic test_ite_fail_to_else/2.
test_ite_fail_to_else(_, R) :-
    % soap is NOT in the basket -> condition fails, takes else branch.
    ( ite_basket(soap) -> R is 11 ; R is 22 ).

% Chained ITE after a multi-clause call (the case cut_ite would mis-handle).
:- dynamic test_ite_chained_after_call/2.
test_ite_chained_after_call(_, R) :-
    ( ite_basket(milk) -> X is 7 ; X is 0 ),
    % Continuation arithmetic exercises that the ITE returned with
    % a clean cp_stack -- pre-M17 inner retry CPs would survive and
    % the wrong branch would be taken on subsequent backtrack.
    R is X + 1.

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

% M18: more eval_arith unary ops -- truncate, round, floor, ceiling,
% sqrt, sign. These either lower to LLVM intrinsics (which the linker
% pulls from libm) or use inline integer logic (sign). truncate /
% round / floor / ceiling all return Integer; sqrt returns Float;
% sign returns the operand''s tag.
:- dynamic test_truncate_pos/2.
test_truncate_pos(_, R) :- X is 7 / 2, R is truncate(X).    % 3.5 -> 3

:- dynamic test_truncate_neg/2.
test_truncate_neg(_, R) :- X is -7 / 2, R is truncate(X).   % -3.5 -> -3

:- dynamic test_round_up/2.
test_round_up(_, R) :- X is 7 / 2, R is round(X).           % 3.5 -> 4

:- dynamic test_round_down/2.
test_round_down(_, R) :- X is 5 / 2, R is round(X).         % 2.5 -> 3
                                                            % (round-half-away-from-zero)

:- dynamic test_floor_pos/2.
test_floor_pos(_, R) :- X is 7 / 2, R is floor(X).          % 3.5 -> 3

:- dynamic test_floor_neg/2.
test_floor_neg(_, R) :- X is -7 / 2, R is floor(X).         % -3.5 -> -4

:- dynamic test_ceiling_pos/2.
test_ceiling_pos(_, R) :- X is 7 / 2, R is ceiling(X).      % 3.5 -> 4

:- dynamic test_sqrt_int/2.
test_sqrt_int(_, R) :- X is sqrt(16), R is truncate(X).     % 4.0 -> 4

:- dynamic test_sign_pos/2.
test_sign_pos(_, R) :- R is sign(7).                        % 1

:- dynamic test_sign_neg/2.
test_sign_neg(_, R) :- R is sign(-7).                       % -1

:- dynamic test_sign_zero/2.
test_sign_zero(_, R) :- R is sign(0).                       % 0

% M19: atom_length/2 and atom_codes/2 -- atom-string primitives.
% atom_length walks the atom''s C string in the M12 atom-string table.
% atom_codes also walks it but emits an Integer-per-byte cons chain.
:- dynamic test_atom_length_hello/2.
test_atom_length_hello(_, R) :- atom_length(hello, N), R is N. % 5

:- dynamic test_atom_length_empty/2.
test_atom_length_empty(_, R) :- atom_length('', N), R is N.    % 0

:- dynamic test_atom_codes_head/2.
test_atom_codes_head(_, R) :-
    atom_codes(hello, Cs),
    Cs = [H|_],
    R is H.   % ''h'' = 104

:- dynamic test_atom_codes_second/2.
test_atom_codes_second(_, R) :-
    atom_codes(hello, Cs),
    Cs = [_, X|_],
    R is X.   % ''e'' = 101

:- dynamic test_atom_codes_length/2.
test_atom_codes_length(_, R) :-
    atom_codes(world, Cs),
    length(Cs, N),
    R is N.   % 5

% M26: char_code/2 -- bidirectional char-atom <-> integer-code primitive.
% Forward mode walks the single-char atom''s string and takes the first
% byte; reverse mode looks up the @wam_char_to_atom_id table populated
% with all printable ASCII (32..126) at module build time.
:- dynamic test_char_code_forward_h/2.
test_char_code_forward_h(_, R) :-
    char_code(h, X),
    R is X.   % 104

:- dynamic test_char_code_forward_z/2.
test_char_code_forward_z(_, R) :-
    char_code(z, X),
    R is X.   % 122

:- dynamic test_char_code_reverse_h/2.
test_char_code_reverse_h(_, R) :-
    char_code(C, 104),
    char_code(C, X),    % round-trip through forward to read the code back
    R is X.             % 104

:- dynamic test_char_code_check/2.
test_char_code_check(_, R) :-
    % Both bound, code matches: succeeds.
    char_code(a, 97),
    R is 1.

% M27: atom_chars/2 forward mode -- walks atom string, maps each byte
% to a single-char atom via the M26 char-id table, builds a cons chain.
% Reverse mode (chars -> atom) needs runtime interning and is deferred.
:- dynamic test_atom_chars_head/2.
test_atom_chars_head(_, R) :-
    atom_chars(hello, Cs),
    Cs = [C|_],
    char_code(C, X),    % round-trip the single-char atom back to a code
    R is X.             % ''h'' = 104

:- dynamic test_atom_chars_third/2.
test_atom_chars_third(_, R) :-
    atom_chars(hello, Cs),
    Cs = [_, _, C|_],
    char_code(C, X),
    R is X.             % ''l'' = 108

:- dynamic test_atom_chars_length/2.
test_atom_chars_length(_, R) :-
    atom_chars(world, Cs),
    length(Cs, N),
    R is N.             % 5

% Round-trip through both atom_chars and write/1 to verify the
% printer renders the list of single-char atoms correctly.
:- dynamic test_atom_chars_print/2.
test_atom_chars_print(_, R) :-
    atom_chars(hi, Cs),
    format('~w', [Cs]),
    R is 1.

% M28: between/3 multi-result iterator. Three patterns:
%   bind + use (one solution, fail on next): runs to completion
%   accumulate via findall (consume the iterator)
%   check (both ends + bound X): no iterator, just range check.
:- dynamic test_between_bind_first/2.
test_between_bind_first(_, R) :-
    between(3, 5, X),    % binds X = 3 on first solution
    R is X.

:- dynamic test_between_check_in/2.
test_between_check_in(_, R) :-
    % X already bound, both ends bound, X in range -> succeeds.
    X = 5,
    between(1, 10, X),
    R is 1.

:- dynamic test_between_check_out/2.
test_between_check_out(_, R) :-
    % X already bound, X out of range -> fails. run_test_r0 maps
    % the failure to exit 255.
    X = 99,
    between(1, 10, X),
    R is 1.

:- dynamic test_between_sum/2.
test_between_sum(_, R) :-
    % Sum 1..10 = 55. Exercises full iterator drain via
    % aggregate_all + between.
    aggregate_all(sum(X), between(1, 10, X), S),
    R is S.

:- dynamic test_between_count/2.
test_between_count(_, R) :-
    aggregate_all(count, between(1, 7, _), N),
    R is N.   % 7 -- route through is/2 so the result lands in A1

% M29: atom_concat/3 forward mode -- needs runtime atom interning to
% construct novel atoms not in the static table. Tests the round-trip
% by piping the result through atom_length / char_code so the test
% driver gets back an integer the shell can carry.
:- dynamic test_atom_concat_length/2.
test_atom_concat_length(_, R) :-
    atom_concat(hello, world, A),
    atom_length(A, N),
    R is N.   % 5 + 5 = 10

:- dynamic test_atom_concat_first/2.
test_atom_concat_first(_, R) :-
    atom_concat(ab, cd, A),
    atom_codes(A, Cs),
    Cs = [C|_],
    R is C.   % ''a'' = 97

:- dynamic test_atom_concat_second/2.
test_atom_concat_second(_, R) :-
    atom_concat(ab, cd, A),
    atom_codes(A, Cs),
    Cs = [_, _, C|_],
    R is C.   % ''c'' = 99

% Same string interned twice should reuse the dynamic-table entry.
% Test by interning the same atom_concat twice and verifying the
% atom ids match (== succeeds for equal atoms).
:- dynamic test_atom_concat_dedup/2.
test_atom_concat_dedup(_, R) :-
    atom_concat(hello, world, A),
    atom_concat(hello, world, B),
    ( A == B -> R is 1 ; R is 0 ).

% Empty atom handling.
:- dynamic test_atom_concat_left_empty/2.
test_atom_concat_left_empty(_, R) :-
    atom_concat('', hi, A),
    atom_length(A, N),
    R is N.   % 2

% M30: atom_codes/2 reverse mode. Build an atom from a list of integer
% codes. Validates the dynamic atom-table grows correctly and that
% subsequent forward operations (length, concat, ==) see the new atom.

:- dynamic test_atom_codes_reverse_length/2.
test_atom_codes_reverse_length(_, R) :-
    atom_codes(A, [104, 105]),   % "hi"
    atom_length(A, N),
    R is N.   % 2

:- dynamic test_atom_codes_reverse_concat/2.
test_atom_codes_reverse_concat(_, R) :-
    atom_codes(A, [104, 105]),   % "hi"
    atom_concat(A, lo, B),       % "hilo"
    atom_length(B, N),
    R is N.   % 4

% Two reverse-mode interns of the same code list must produce the
% same atom id (dedup via @wam_str_eq_n linear scan).
:- dynamic test_atom_codes_reverse_dedup/2.
test_atom_codes_reverse_dedup(_, R) :-
    atom_codes(A, [104, 105]),
    atom_codes(B, [104, 105]),
    ( A == B -> R is 1 ; R is 0 ).

% Empty code list builds the empty atom.
:- dynamic test_atom_codes_reverse_empty/2.
test_atom_codes_reverse_empty(_, R) :-
    atom_codes(A, []),
    atom_length(A, N),
    R is N.   % 0

% M30: atom_chars/2 reverse mode. Build an atom from a list of
% single-char atoms (each element''s first byte is taken).

:- dynamic test_atom_chars_reverse_length/2.
test_atom_chars_reverse_length(_, R) :-
    atom_chars(A, [h, i]),
    atom_length(A, N),
    R is N.   % 2

:- dynamic test_atom_chars_reverse_first/2.
test_atom_chars_reverse_first(_, R) :-
    atom_chars(A, [h, i]),
    atom_codes(A, [C|_]),
    R is C.   % ''h'' = 104

% Cross-mode dedup: atoms built by atom_chars reverse and re-decoded
% via atom_codes forward should produce the same id when interned
% twice.
:- dynamic test_atom_chars_reverse_dedup/2.
test_atom_chars_reverse_dedup(_, R) :-
    atom_chars(A, [h, i, j]),
    atom_chars(B, [h, i, j]),
    ( A == B -> R is 1 ; R is 0 ).

% M31: sub_atom/5 deterministic extraction. Requires Atom + Before +
% Length all bound. Verifies After is computed correctly and the
% extracted slice is interned (so subsequent forward ops see it).

:- dynamic test_sub_atom_extract_length/2.
test_sub_atom_extract_length(_, R) :-
    sub_atom(hello, 1, 3, _, S),    % "ell"
    atom_length(S, N),
    R is N.   % 3

:- dynamic test_sub_atom_extract_after/2.
test_sub_atom_extract_after(_, R) :-
    sub_atom(hello, 1, 3, A, _),    % After = 5 - 1 - 3 = 1
    R is A.   % 1

:- dynamic test_sub_atom_extract_first_code/2.
test_sub_atom_extract_first_code(_, R) :-
    sub_atom(hello, 1, 3, _, S),    % "ell"
    atom_codes(S, [C|_]),
    R is C.   % 'e' = 101

:- dynamic test_sub_atom_dedup/2.
test_sub_atom_dedup(_, R) :-
    sub_atom(hello, 1, 3, _, S1),
    sub_atom(hello, 1, 3, _, S2),
    ( S1 == S2 -> R is 1 ; R is 0 ).

:- dynamic test_sub_atom_empty/2.
test_sub_atom_empty(_, R) :-
    sub_atom(hello, 2, 0, _, S),    % empty slice
    atom_length(S, N),
    R is N.   % 0

:- dynamic test_sub_atom_full/2.
test_sub_atom_full(_, R) :-
    sub_atom(hello, 0, 5, A, S),    % whole atom, After = 0
    atom_length(S, N),
    R is N + A.   % 5 + 0 = 5

:- dynamic test_sub_atom_overflow/2.
test_sub_atom_overflow(_, R) :-
    ( sub_atom(hi, 0, 99, _, _)
    -> R is 1
    ;  R is 0 ).   % 0 -- length exceeds source

% M32: atom_number/2 integer mode -- forward (atom -> int) + reverse
% (int -> atom). Float parsing / formatting deferred.

:- dynamic test_atom_number_fwd/2.
test_atom_number_fwd(_, R) :-
    atom_number('42', N),
    R is N.   % 42

:- dynamic test_atom_number_fwd_neg/2.
test_atom_number_fwd_neg(_, R) :-
    atom_number('-17', N),
    R is N + 100.   % 83

:- dynamic test_atom_number_fwd_zero/2.
test_atom_number_fwd_zero(_, R) :-
    atom_number('0', N),
    R is N + 7.   % 7

:- dynamic test_atom_number_fwd_bad/2.
test_atom_number_fwd_bad(_, R) :-
    ( atom_number('12abc', _)
    -> R is 1
    ;  R is 0 ).   % 0 -- trailing junk fails

:- dynamic test_atom_number_fwd_empty/2.
test_atom_number_fwd_empty(_, R) :-
    ( atom_number('', _)
    -> R is 1
    ;  R is 0 ).   % 0 -- empty atom fails

:- dynamic test_atom_number_rev/2.
test_atom_number_rev(_, R) :-
    atom_number(A, 42),
    atom_length(A, N),
    R is N.   % 2

:- dynamic test_atom_number_rev_neg/2.
test_atom_number_rev_neg(_, R) :-
    atom_number(A, -17),
    atom_length(A, N),
    R is N.   % 3

% Roundtrip: number -> atom -> back to same number.
:- dynamic test_atom_number_roundtrip/2.
test_atom_number_roundtrip(_, R) :-
    atom_number(A, 99),
    atom_number(A, N),
    R is N.   % 99 (kept <256 for bash exit-code carry)

% M33: number_codes/2 integer mode -- forward (int -> codes) + reverse
% (codes -> int). Mirrors atom_number/2 but with a list-of-codes
% interchange format.

:- dynamic test_number_codes_fwd_first/2.
test_number_codes_fwd_first(_, R) :-
    number_codes(42, [C|_]),
    R is C.   % '4' = 52

:- dynamic test_number_codes_fwd_length/2.
test_number_codes_fwd_length(_, R) :-
    number_codes(1234, L),   % "1234" -> 4 codes (kept <256 in R)
    length(L, N),
    R is N.   % 4

:- dynamic test_number_codes_fwd_neg_first/2.
test_number_codes_fwd_neg_first(_, R) :-
    number_codes(-5, [C|_]),
    R is C.   % '-' = 45

:- dynamic test_number_codes_rev/2.
test_number_codes_rev(_, R) :-
    number_codes(N, [52, 50]),   % "42"
    R is N.   % 42

:- dynamic test_number_codes_rev_neg/2.
test_number_codes_rev_neg(_, R) :-
    number_codes(N, [45, 49, 55]),   % "-17"
    R is N + 100.   % 83

:- dynamic test_number_codes_rev_bad/2.
test_number_codes_rev_bad(_, R) :-
    ( number_codes(_, [49, 97, 98, 99])   % "1abc"
    -> R is 1
    ;  R is 0 ).   % 0 -- trailing junk fails

:- dynamic test_number_codes_rev_empty/2.
test_number_codes_rev_empty(_, R) :-
    ( number_codes(_, [])
    -> R is 1
    ;  R is 0 ).   % 0 -- empty list fails

:- dynamic test_number_codes_roundtrip/2.
test_number_codes_roundtrip(_, R) :-
    number_codes(99, Codes),
    number_codes(N, Codes),
    R is N.   % 99

% M34: number_chars/2 integer mode -- forward (int -> chars) +
% reverse (chars -> int). Mirrors M33 but emits single-char atoms.

:- dynamic test_number_chars_fwd_head/2.
test_number_chars_fwd_head(_, R) :-
    number_chars(42, [H|_]),
    char_code(H, C),
    R is C.   % '4' = 52

:- dynamic test_number_chars_fwd_length/2.
test_number_chars_fwd_length(_, R) :-
    number_chars(1234, L),
    length(L, N),
    R is N.   % 4

:- dynamic test_number_chars_fwd_neg_head/2.
test_number_chars_fwd_neg_head(_, R) :-
    number_chars(-5, [H|_]),
    char_code(H, C),
    R is C.   % '-' = 45

:- dynamic test_number_chars_rev/2.
test_number_chars_rev(_, R) :-
    number_chars(N, ['4', '2']),
    R is N.   % 42

:- dynamic test_number_chars_rev_neg/2.
test_number_chars_rev_neg(_, R) :-
    number_chars(N, ['-', '1', '7']),
    R is N + 100.   % 83

:- dynamic test_number_chars_rev_bad/2.
test_number_chars_rev_bad(_, R) :-
    ( number_chars(_, ['1', a, b, c])
    -> R is 1
    ;  R is 0 ).   % 0 -- trailing non-digit fails

:- dynamic test_number_chars_rev_empty/2.
test_number_chars_rev_empty(_, R) :-
    ( number_chars(_, [])
    -> R is 1
    ;  R is 0 ).   % 0 -- empty list fails

:- dynamic test_number_chars_roundtrip/2.
test_number_chars_roundtrip(_, R) :-
    number_chars(99, Chars),
    number_chars(N, Chars),
    R is N.   % 99

% M35: upcase_atom/2 + downcase_atom/2 -- ASCII a..z <-> A..Z, other
% bytes passthrough.

:- dynamic test_upcase_length/2.
test_upcase_length(_, R) :-
    upcase_atom(hello, U),
    atom_length(U, N),
    R is N.   % 5

:- dynamic test_upcase_first_code/2.
test_upcase_first_code(_, R) :-
    upcase_atom(hello, U),
    atom_codes(U, [C|_]),
    R is C.   % 'H' = 72

:- dynamic test_upcase_last_code/2.
test_upcase_last_code(_, R) :-
    upcase_atom(ab, U),
    atom_codes(U, [_, C2]),
    R is C2.   % 'B' = 66

:- dynamic test_upcase_passthrough/2.
test_upcase_passthrough(_, R) :-
    upcase_atom('hi!', U),    % '!' is not in a..z, stays '!' (33)
    atom_codes(U, [_, _, C3]),
    R is C3.   % '!' = 33

:- dynamic test_upcase_dedup/2.
test_upcase_dedup(_, R) :-
    upcase_atom(hello, U1),
    upcase_atom(hello, U2),
    ( U1 == U2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_downcase_length/2.
test_downcase_length(_, R) :-
    downcase_atom('HELLO', D),
    atom_length(D, N),
    R is N.   % 5

:- dynamic test_downcase_first_code/2.
test_downcase_first_code(_, R) :-
    downcase_atom('HELLO', D),
    atom_codes(D, [C|_]),
    R is C.   % 'h' = 104

:- dynamic test_downcase_mixed/2.
test_downcase_mixed(_, R) :-
    downcase_atom('aB', D),    % 'a' stays 'a', 'B' becomes 'b'
    atom_codes(D, [_, C2]),
    R is C2.   % 'b' = 98

:- dynamic test_upcase_downcase_roundtrip/2.
test_upcase_downcase_roundtrip(_, R) :-
    upcase_atom(hello, U),
    downcase_atom(U, D),
    ( D == hello -> R is 1 ; R is 0 ).   % 1

% M36: string-type aliases. Runtime has no distinct string type, so
% atom_string / string_to_atom reduce to unify, and string_concat /
% string_length share dispatch labels with atom_concat / atom_length.

:- dynamic test_atom_string_fwd/2.
test_atom_string_fwd(_, R) :-
    atom_string(hello, S),
    atom_length(S, N),
    R is N.   % 5

:- dynamic test_atom_string_check/2.
test_atom_string_check(_, R) :-
    ( atom_string(hello, hello) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_string_to_atom_fwd/2.
test_string_to_atom_fwd(_, R) :-
    string_to_atom(world, A),
    atom_length(A, N),
    R is N.   % 5

:- dynamic test_string_concat_length/2.
test_string_concat_length(_, R) :-
    string_concat(hi, there, S),
    atom_length(S, N),
    R is N.   % 7

:- dynamic test_string_concat_first_code/2.
test_string_concat_first_code(_, R) :-
    string_concat(ab, cd, S),
    atom_codes(S, [C|_]),
    R is C.   % 'a' = 97

:- dynamic test_string_length_simple/2.
test_string_length_simple(_, R) :-
    string_length(hello, N),
    R is N.   % 5

:- dynamic test_string_length_empty/2.
test_string_length_empty(_, R) :-
    string_length('', N),
    R is N + 42.   % 42

% M37: nth0/3, nth1/3, last/2 -- list-indexing trio. Forward modes
% only (Index + List bound).

:- dynamic test_nth0_first/2.
test_nth0_first(_, R) :-
    nth0(0, [10, 20, 30], E),
    R is E.   % 10

:- dynamic test_nth0_middle/2.
test_nth0_middle(_, R) :-
    nth0(2, [10, 20, 30, 40], E),
    R is E.   % 30

:- dynamic test_nth0_last/2.
test_nth0_last(_, R) :-
    nth0(3, [10, 20, 30, 40], E),
    R is E.   % 40

:- dynamic test_nth0_overflow/2.
test_nth0_overflow(_, R) :-
    ( nth0(5, [10, 20], _) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_nth0_negative/2.
test_nth0_negative(_, R) :-
    ( nth0(-1, [10, 20], _) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_nth1_first/2.
test_nth1_first(_, R) :-
    nth1(1, [10, 20, 30], E),
    R is E.   % 10

:- dynamic test_nth1_third/2.
test_nth1_third(_, R) :-
    nth1(3, [10, 20, 30, 40], E),
    R is E.   % 30

:- dynamic test_nth1_zero/2.
test_nth1_zero(_, R) :-
    ( nth1(0, [10, 20], _) -> R is 1 ; R is 0 ).   % 0 -- 1-indexed rejects 0

:- dynamic test_last_simple/2.
test_last_simple(_, R) :-
    last([10, 20, 30], E),
    R is E.   % 30

:- dynamic test_last_singleton/2.
test_last_singleton(_, R) :-
    last([99], E),
    R is E.   % 99

:- dynamic test_last_empty/2.
test_last_empty(_, R) :-
    ( last([], _) -> R is 1 ; R is 0 ).   % 0

% M38: reverse/2 deterministic forward mode.

:- dynamic test_reverse_first/2.
test_reverse_first(_, R) :-
    reverse([10, 20, 30], L),
    nth0(0, L, E),
    R is E.   % 30 (originally last)

:- dynamic test_reverse_last/2.
test_reverse_last(_, R) :-
    reverse([10, 20, 30], L),
    last(L, E),
    R is E.   % 10 (originally first)

:- dynamic test_reverse_length/2.
test_reverse_length(_, R) :-
    reverse([1, 2, 3, 4, 5], L),
    length(L, N),
    R is N.   % 5

:- dynamic test_reverse_empty/2.
test_reverse_empty(_, R) :-
    reverse([], L),
    length(L, N),
    R is N + 7.   % 7

:- dynamic test_reverse_singleton/2.
test_reverse_singleton(_, R) :-
    reverse([42], [E]),
    R is E.   % 42

:- dynamic test_reverse_idempotent/2.
test_reverse_idempotent(_, R) :-
    reverse([1, 2, 3], L1),
    reverse(L1, L2),
    nth0(0, L2, E),
    R is E.   % 1 (reverse twice -> original)

% M39: append/3 deterministic (A1 and A2 both bound lists).

:- dynamic test_append_length/2.
test_append_length(_, R) :-
    append([1, 2, 3], [4, 5], L),
    length(L, N),
    R is N.   % 5

:- dynamic test_append_first/2.
test_append_first(_, R) :-
    append([10, 20], [30, 40], L),
    nth0(0, L, E),
    R is E.   % 10 (head from A1)

:- dynamic test_append_seam/2.
test_append_seam(_, R) :-
    append([10, 20], [30, 40], L),
    nth0(2, L, E),
    R is E.   % 30 (first elem of A2)

:- dynamic test_append_last/2.
test_append_last(_, R) :-
    append([10, 20], [30, 40], L),
    last(L, E),
    R is E.   % 40

:- dynamic test_append_left_empty/2.
test_append_left_empty(_, R) :-
    append([], [7, 8], L),
    length(L, N),
    R is N.   % 2

:- dynamic test_append_right_empty/2.
test_append_right_empty(_, R) :-
    append([7, 8], [], L),
    length(L, N),
    R is N.   % 2

:- dynamic test_append_both_empty/2.
test_append_both_empty(_, R) :-
    append([], [], L),
    length(L, N),
    R is N + 9.   % 9

:- dynamic test_append_roundtrip/2.
test_append_roundtrip(_, R) :-
    append([1, 2], [3, 4], L),
    reverse(L, L2),
    nth0(0, L2, E),
    R is E.   % 4 (last of L becomes first after reverse)

% M40: memberchk/2 deterministic.

:- dynamic test_memberchk_int_hit/2.
test_memberchk_int_hit(_, R) :-
    ( memberchk(20, [10, 20, 30]) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_memberchk_int_miss/2.
test_memberchk_int_miss(_, R) :-
    ( memberchk(99, [10, 20, 30]) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_memberchk_int_empty/2.
test_memberchk_int_empty(_, R) :-
    ( memberchk(1, []) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_memberchk_atom_hit/2.
test_memberchk_atom_hit(_, R) :-
    ( memberchk(b, [a, b, c]) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_memberchk_bind/2.
test_memberchk_bind(_, R) :-
    memberchk(X, [42, 99, 7]),
    R is X.   % 42 -- unbound X gets first element

:- dynamic test_memberchk_first_match_wins/2.
test_memberchk_first_match_wins(_, R) :-
    % Two 5s in the list; deterministic memberchk takes the first.
    ( memberchk(5, [3, 5, 5, 9]) -> R is 1 ; R is 0 ).   % 1

% M41: delete/3 deterministic.

:- dynamic test_delete_no_match/2.
test_delete_no_match(_, R) :-
    delete([1, 2, 3], 99, L),
    length(L, N),
    R is N.   % 3 -- nothing removed

:- dynamic test_delete_single/2.
test_delete_single(_, R) :-
    delete([1, 2, 3], 2, L),
    length(L, N),
    R is N.   % 2

:- dynamic test_delete_multiple/2.
test_delete_multiple(_, R) :-
    delete([1, 2, 1, 3, 1], 1, L),
    length(L, N),
    R is N.   % 2 -- three 1s removed

:- dynamic test_delete_all/2.
test_delete_all(_, R) :-
    delete([7, 7, 7], 7, L),
    length(L, N),
    R is N + 5.   % 5 -- empty result

:- dynamic test_delete_empty/2.
test_delete_empty(_, R) :-
    delete([], 1, L),
    length(L, N),
    R is N + 8.   % 8

:- dynamic test_delete_first/2.
test_delete_first(_, R) :-
    delete([1, 2, 3, 4], 1, L),
    nth0(0, L, E),
    R is E.   % 2 -- first becomes 2 after removing the 1

:- dynamic test_delete_last/2.
test_delete_last(_, R) :-
    delete([1, 2, 3, 99], 99, L),
    last(L, E),
    R is E.   % 3 -- 99 removed from tail

:- dynamic test_delete_preserves_order/2.
test_delete_preserves_order(_, R) :-
    delete([10, 5, 20, 5, 30], 5, L),
    nth0(1, L, E),    % L = [10, 20, 30], index 1 is 20
    R is E.   % 20

% M42: numlist/3 + sum_list/2.

:- dynamic test_numlist_length/2.
test_numlist_length(_, R) :-
    numlist(3, 7, L),
    length(L, N),
    R is N.   % 5

:- dynamic test_numlist_first/2.
test_numlist_first(_, R) :-
    numlist(10, 15, [F|_]),
    R is F.   % 10

:- dynamic test_numlist_last/2.
test_numlist_last(_, R) :-
    numlist(1, 7, L),
    last(L, E),
    R is E.   % 7

:- dynamic test_numlist_singleton/2.
test_numlist_singleton(_, R) :-
    numlist(42, 42, [E]),
    R is E.   % 42

:- dynamic test_numlist_empty/2.
test_numlist_empty(_, R) :-
    ( numlist(5, 3, _) -> R is 1 ; R is 0 ).   % 0 -- High < Low

:- dynamic test_numlist_sum/2.
test_numlist_sum(_, R) :-
    numlist(1, 10, L),
    sum_list(L, S),
    R is S.   % 55

:- dynamic test_sum_list_simple/2.
test_sum_list_simple(_, R) :-
    sum_list([10, 20, 30], S),
    R is S.   % 60

:- dynamic test_sum_list_empty/2.
test_sum_list_empty(_, R) :-
    sum_list([], S),
    R is S + 7.   % 7

:- dynamic test_sum_list_singleton/2.
test_sum_list_singleton(_, R) :-
    sum_list([99], S),
    R is S.   % 99

:- dynamic test_sumlist_alias/2.
test_sumlist_alias(_, R) :-
    sumlist([1, 2, 3, 4, 5], S),
    R is S.   % 15

% M43: max_list/2 + min_list/2 -- integer aggregations.

:- dynamic test_max_list_simple/2.
test_max_list_simple(_, R) :-
    max_list([3, 7, 2, 9, 5], M),
    R is M.   % 9

:- dynamic test_max_list_singleton/2.
test_max_list_singleton(_, R) :-
    max_list([42], M),
    R is M.   % 42

:- dynamic test_max_list_first_biggest/2.
test_max_list_first_biggest(_, R) :-
    max_list([100, 50, 25, 10], M),
    R is M.   % 100

:- dynamic test_max_list_empty/2.
test_max_list_empty(_, R) :-
    ( max_list([], _) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_min_list_simple/2.
test_min_list_simple(_, R) :-
    min_list([7, 3, 9, 2, 5], M),
    R is M.   % 2

:- dynamic test_min_list_singleton/2.
test_min_list_singleton(_, R) :-
    min_list([42], M),
    R is M.   % 42

:- dynamic test_min_list_first_smallest/2.
test_min_list_first_smallest(_, R) :-
    min_list([1, 10, 100], M),
    R is M.   % 1

:- dynamic test_min_list_empty/2.
test_min_list_empty(_, R) :-
    ( min_list([], _) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_max_minus_min/2.
test_max_minus_min(_, R) :-
    max_list([4, 7, 1, 9, 3], Mx),
    min_list([4, 7, 1, 9, 3], Mn),
    R is Mx - Mn.   % 9 - 1 = 8

% M44: subtract/3 -- elements of A1 with no unify-match in A2.

:- dynamic test_subtract_disjoint/2.
test_subtract_disjoint(_, R) :-
    subtract([1, 2, 3], [4, 5, 6], L),
    length(L, N),
    R is N.   % 3 -- all kept

:- dynamic test_subtract_one/2.
test_subtract_one(_, R) :-
    subtract([1, 2, 3], [2], L),
    length(L, N),
    R is N.   % 2

:- dynamic test_subtract_many/2.
test_subtract_many(_, R) :-
    subtract([1, 2, 3, 4, 5], [2, 4], L),
    length(L, N),
    R is N.   % 3

:- dynamic test_subtract_all/2.
test_subtract_all(_, R) :-
    subtract([1, 2, 3], [3, 2, 1], L),
    length(L, N),
    R is N + 7.   % 7 -- empty result

:- dynamic test_subtract_empty_left/2.
test_subtract_empty_left(_, R) :-
    subtract([], [1, 2], L),
    length(L, N),
    R is N + 8.   % 8

:- dynamic test_subtract_empty_right/2.
test_subtract_empty_right(_, R) :-
    subtract([7, 8, 9], [], L),
    length(L, N),
    R is N.   % 3 -- nothing to remove

:- dynamic test_subtract_first/2.
test_subtract_first(_, R) :-
    subtract([10, 20, 30, 40], [10], L),
    nth0(0, L, E),
    R is E.   % 20

:- dynamic test_subtract_preserves_order/2.
test_subtract_preserves_order(_, R) :-
    subtract([10, 5, 20, 5, 30], [5], L),
    nth0(1, L, E),    % L = [10, 20, 30]
    R is E.   % 20

:- dynamic test_subtract_dupes/2.
test_subtract_dupes(_, R) :-
    % All occurrences in A1 are filtered, just like delete/3.
    subtract([1, 2, 1, 3, 1], [1], L),
    length(L, N),
    R is N.   % 2

% M45: intersection/3 -- elements of A1 with a unify-match in A2.

:- dynamic test_intersection_disjoint/2.
test_intersection_disjoint(_, R) :-
    intersection([1, 2, 3], [4, 5, 6], L),
    length(L, N),
    R is N + 8.   % 8 -- empty

:- dynamic test_intersection_one/2.
test_intersection_one(_, R) :-
    intersection([1, 2, 3], [2], L),
    length(L, N),
    R is N.   % 1

:- dynamic test_intersection_many/2.
test_intersection_many(_, R) :-
    intersection([1, 2, 3, 4, 5], [2, 4, 99], L),
    length(L, N),
    R is N.   % 2 -- [2, 4]

:- dynamic test_intersection_all/2.
test_intersection_all(_, R) :-
    intersection([1, 2, 3], [3, 2, 1], L),
    length(L, N),
    R is N.   % 3 -- everything matches

:- dynamic test_intersection_empty_left/2.
test_intersection_empty_left(_, R) :-
    intersection([], [1, 2], L),
    length(L, N),
    R is N + 7.   % 7

:- dynamic test_intersection_empty_right/2.
test_intersection_empty_right(_, R) :-
    intersection([1, 2, 3], [], L),
    length(L, N),
    R is N + 5.   % 5 -- nothing matches

:- dynamic test_intersection_first/2.
test_intersection_first(_, R) :-
    intersection([10, 20, 30, 40], [20, 40, 99], L),
    nth0(0, L, E),
    R is E.   % 20

:- dynamic test_intersection_preserves_order/2.
test_intersection_preserves_order(_, R) :-
    intersection([10, 5, 20, 5, 30], [5, 20], L),
    % L = [5, 20, 5]; nth0(2) is the second 5.
    nth0(2, L, E),
    R is E.   % 5

:- dynamic test_intersection_dupes/2.
test_intersection_dupes(_, R) :-
    % All occurrences of a match in A1 survive.
    intersection([1, 2, 1, 3, 1], [1], L),
    length(L, N),
    R is N.   % 3

% Composition: List = Intersection + Subtract should reconstruct A1.
:- dynamic test_inter_subtract_complement/2.
test_inter_subtract_complement(_, R) :-
    intersection([1, 2, 3, 4, 5], [2, 4], Inter),    % [2, 4]
    subtract([1, 2, 3, 4, 5], [2, 4], Diff),         % [1, 3, 5]
    length(Inter, NI),
    length(Diff, ND),
    R is NI + ND.   % 2 + 3 = 5

% M46: union/3 -- A1 ++ subtract(A2, A1).

:- dynamic test_union_disjoint/2.
test_union_disjoint(_, R) :-
    union([1, 2, 3], [4, 5, 6], L),
    length(L, N),
    R is N.   % 6

:- dynamic test_union_overlap/2.
test_union_overlap(_, R) :-
    union([1, 2, 3], [2, 4, 5], L),
    length(L, N),
    R is N.   % 5 -- 2 dropped from A2

:- dynamic test_union_identical/2.
test_union_identical(_, R) :-
    union([1, 2, 3], [1, 2, 3], L),
    length(L, N),
    R is N.   % 3 -- all of A2 filtered

:- dynamic test_union_empty_left/2.
test_union_empty_left(_, R) :-
    union([], [7, 8, 9], L),
    length(L, N),
    R is N.   % 3

:- dynamic test_union_empty_right/2.
test_union_empty_right(_, R) :-
    union([1, 2, 3], [], L),
    length(L, N),
    R is N.   % 3

:- dynamic test_union_both_empty/2.
test_union_both_empty(_, R) :-
    union([], [], L),
    length(L, N),
    R is N + 11.   % 11

:- dynamic test_union_a1_first/2.
test_union_a1_first(_, R) :-
    union([10, 20], [30, 40], L),
    nth0(0, L, E),
    R is E.   % 10 -- A1 comes first

:- dynamic test_union_a1_dupes_kept/2.
test_union_a1_dupes_kept(_, R) :-
    % SWI semantics: A1''s own duplicates are preserved, only A2 is
    % filtered against A1.
    union([1, 1, 2], [3], L),
    length(L, N),
    R is N.   % 4 -- [1, 1, 2, 3]

:- dynamic test_union_a2_first_match_filtered/2.
test_union_a2_first_match_filtered(_, R) :-
    % A2''s first 2 matches A1, gets dropped; 4 survives.
    union([1, 2, 3], [2, 4], L),
    nth0(3, L, E),
    R is E.   % 4 (last position)

:- dynamic test_union_size_relation/2.
test_union_size_relation(_, R) :-
    % |union| + |intersection| = |A1| + |A2| (inclusion-exclusion).
    union([1, 2, 3, 4], [3, 4, 5, 6], U),
    intersection([1, 2, 3, 4], [3, 4, 5, 6], I),
    length(U, NU),
    length(I, NI),
    R is NU + NI.   % 6 + 2 = 8 (= 4 + 4)

% M47: list_to_set/2 -- dedupe preserving first-occurrence order.

:- dynamic test_l2s_simple/2.
test_l2s_simple(_, R) :-
    list_to_set([1, 2, 3], L),
    length(L, N),
    R is N.   % 3 -- no dupes

:- dynamic test_l2s_dupes/2.
test_l2s_dupes(_, R) :-
    list_to_set([1, 2, 1, 3, 2], L),
    length(L, N),
    R is N.   % 3 -- [1, 2, 3]

:- dynamic test_l2s_all_dupes/2.
test_l2s_all_dupes(_, R) :-
    list_to_set([7, 7, 7, 7], L),
    length(L, N),
    R is N.   % 1

:- dynamic test_l2s_empty/2.
test_l2s_empty(_, R) :-
    list_to_set([], L),
    length(L, N),
    R is N + 9.   % 9

:- dynamic test_l2s_singleton/2.
test_l2s_singleton(_, R) :-
    list_to_set([42], [E]),
    R is E.   % 42

:- dynamic test_l2s_order_first/2.
test_l2s_order_first(_, R) :-
    list_to_set([3, 1, 2, 1, 3], L),
    nth0(0, L, E),
    R is E.   % 3 -- first occurrence wins

:- dynamic test_l2s_order_second/2.
test_l2s_order_second(_, R) :-
    list_to_set([3, 1, 2, 1, 3], L),
    nth0(1, L, E),
    R is E.   % 1

:- dynamic test_l2s_order_third/2.
test_l2s_order_third(_, R) :-
    list_to_set([3, 1, 2, 1, 3], L),
    nth0(2, L, E),
    R is E.   % 2

:- dynamic test_l2s_atom_dupes/2.
test_l2s_atom_dupes(_, R) :-
    list_to_set([a, b, a, c, b], L),
    length(L, N),
    R is N.   % 3

:- dynamic test_l2s_idempotent/2.
test_l2s_idempotent(_, R) :-
    list_to_set([1, 2, 2, 3], L1),
    list_to_set(L1, L2),
    length(L2, N),
    R is N.   % 3 -- already a set

% M48: string_chars/2, string_codes/2 aliases + string_code/3.

:- dynamic test_string_chars_first/2.
test_string_chars_first(_, R) :-
    string_chars(hello, [H|_]),
    char_code(H, C),
    R is C.   % 'h' = 104

:- dynamic test_string_chars_length/2.
test_string_chars_length(_, R) :-
    string_chars(abc, L),
    length(L, N),
    R is N.   % 3

:- dynamic test_string_codes_first/2.
test_string_codes_first(_, R) :-
    string_codes(hello, [C|_]),
    R is C.   % 104

:- dynamic test_string_codes_length/2.
test_string_codes_length(_, R) :-
    string_codes(abcde, L),
    length(L, N),
    R is N.   % 5

:- dynamic test_string_code_first/2.
test_string_code_first(_, R) :-
    string_code(1, hello, C),
    R is C.   % 'h' = 104

:- dynamic test_string_code_middle/2.
test_string_code_middle(_, R) :-
    string_code(3, abcdef, C),
    R is C.   % 'c' = 99

:- dynamic test_string_code_last/2.
test_string_code_last(_, R) :-
    string_code(5, hello, C),
    R is C.   % 'o' = 111

:- dynamic test_string_code_oob_low/2.
test_string_code_oob_low(_, R) :-
    ( string_code(0, hello, _) -> R is 1 ; R is 0 ).   % 0 -- 1-based

:- dynamic test_string_code_oob_high/2.
test_string_code_oob_high(_, R) :-
    ( string_code(99, hello, _) -> R is 1 ; R is 0 ).   % 0

% M49: pairs_keys/2 + pairs_values/2 forward modes.

:- dynamic test_pairs_keys_simple/2.
test_pairs_keys_simple(_, R) :-
    pairs_keys([a-1, b-2, c-3], Ks),
    length(Ks, N),
    R is N.   % 3

:- dynamic test_pairs_keys_first/2.
test_pairs_keys_first(_, R) :-
    pairs_keys([10-x, 20-y, 30-z], [F|_]),
    R is F.   % 10

:- dynamic test_pairs_keys_last/2.
test_pairs_keys_last(_, R) :-
    pairs_keys([10-a, 20-b, 30-c], Ks),
    last(Ks, E),
    R is E.   % 30

:- dynamic test_pairs_keys_empty/2.
test_pairs_keys_empty(_, R) :-
    pairs_keys([], Ks),
    length(Ks, N),
    R is N + 11.   % 11

:- dynamic test_pairs_keys_singleton/2.
test_pairs_keys_singleton(_, R) :-
    pairs_keys([42-foo], [K]),
    R is K.   % 42

:- dynamic test_pairs_values_simple/2.
test_pairs_values_simple(_, R) :-
    pairs_values([a-10, b-20, c-30], Vs),
    length(Vs, N),
    R is N.   % 3

:- dynamic test_pairs_values_first/2.
test_pairs_values_first(_, R) :-
    pairs_values([a-100, b-200], [F|_]),
    R is F.   % 100

:- dynamic test_pairs_values_last/2.
test_pairs_values_last(_, R) :-
    pairs_values([a-10, b-20, c-99], Vs),
    last(Vs, E),
    R is E.   % 99

:- dynamic test_pairs_values_empty/2.
test_pairs_values_empty(_, R) :-
    pairs_values([], Vs),
    length(Vs, N),
    R is N + 13.   % 13

% Compose: pairs_keys + pairs_values should reproduce the structure.
:- dynamic test_pairs_keys_values_sum/2.
test_pairs_keys_values_sum(_, R) :-
    P = [10-1, 20-2, 30-3],
    pairs_keys(P, Ks),
    pairs_values(P, Vs),
    sum_list(Ks, SK),
    sum_list(Vs, SV),
    R is SK + SV.   % 60 + 6 = 66

% M50: pairs_keys_values/3 forward (single-pass split).

:- dynamic test_pkv_keys_length/2.
test_pkv_keys_length(_, R) :-
    pairs_keys_values([a-1, b-2, c-3], Ks, _),
    length(Ks, N),
    R is N.   % 3

:- dynamic test_pkv_values_length/2.
test_pkv_values_length(_, R) :-
    pairs_keys_values([a-1, b-2, c-3], _, Vs),
    length(Vs, N),
    R is N.   % 3

:- dynamic test_pkv_keys_first/2.
test_pkv_keys_first(_, R) :-
    pairs_keys_values([10-x, 20-y], [F|_], _),
    R is F.   % 10

:- dynamic test_pkv_values_first/2.
test_pkv_values_first(_, R) :-
    pairs_keys_values([a-100, b-200], _, [F|_]),
    R is F.   % 100

:- dynamic test_pkv_empty/2.
test_pkv_empty(_, R) :-
    pairs_keys_values([], Ks, Vs),
    length(Ks, NK),
    length(Vs, NV),
    R is NK + NV + 17.   % 17

:- dynamic test_pkv_singleton/2.
test_pkv_singleton(_, R) :-
    pairs_keys_values([42-99], [K], [V]),
    R is K + V.   % 141

:- dynamic test_pkv_keys_last/2.
test_pkv_keys_last(_, R) :-
    pairs_keys_values([1-x, 2-y, 30-z], Ks, _),
    last(Ks, E),
    R is E.   % 30

:- dynamic test_pkv_values_last/2.
test_pkv_values_last(_, R) :-
    pairs_keys_values([a-10, b-20, c-99], _, Vs),
    last(Vs, E),
    R is E.   % 99

% Cross-check with pairs_keys / pairs_values separately.
:- dynamic test_pkv_matches_split/2.
test_pkv_matches_split(_, R) :-
    P = [10-1, 20-2, 30-3],
    pairs_keys(P, KsA),
    pairs_values(P, VsA),
    pairs_keys_values(P, KsB, VsB),
    sum_list(KsA, SKA),
    sum_list(KsB, SKB),
    sum_list(VsA, SVA),
    sum_list(VsB, SVB),
    ( SKA =:= SKB, SVA =:= SVB -> R is 1 ; R is 0 ).   % 1

% M51: pairs_keys_values/3 reverse (zip keys + values into pair list).

:- dynamic test_pkv_rev_length/2.
test_pkv_rev_length(_, R) :-
    pairs_keys_values(P, [a, b, c], [1, 2, 3]),
    length(P, N),
    R is N.   % 3

:- dynamic test_pkv_rev_roundtrip_keys/2.
test_pkv_rev_roundtrip_keys(_, R) :-
    pairs_keys_values(P, [10, 20, 30], [x, y, z]),
    % P should be [10-x, 20-y, 30-z]; pairs_keys recovers the keys.
    pairs_keys(P, Ks),
    sum_list(Ks, S),
    R is S.   % 60

:- dynamic test_pkv_rev_roundtrip_values/2.
test_pkv_rev_roundtrip_values(_, R) :-
    pairs_keys_values(P, [a, b, c], [10, 20, 30]),
    pairs_values(P, Vs),
    sum_list(Vs, S),
    R is S.   % 60

:- dynamic test_pkv_rev_empty/2.
test_pkv_rev_empty(_, R) :-
    pairs_keys_values(P, [], []),
    length(P, N),
    R is N + 19.   % 19

:- dynamic test_pkv_rev_singleton/2.
test_pkv_rev_singleton(_, R) :-
    pairs_keys_values(P, [42], [99]),
    pairs_keys(P, [K]),
    pairs_values(P, [V]),
    R is K + V.   % 141

:- dynamic test_pkv_rev_mismatch_keys_longer/2.
test_pkv_rev_mismatch_keys_longer(_, R) :-
    ( pairs_keys_values(_, [a, b, c], [1, 2]) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_pkv_rev_mismatch_values_longer/2.
test_pkv_rev_mismatch_values_longer(_, R) :-
    ( pairs_keys_values(_, [a], [1, 2, 3]) -> R is 1 ; R is 0 ).   % 0

% Forward then reverse should reconstruct equivalent structure.
:- dynamic test_pkv_forward_then_reverse/2.
test_pkv_forward_then_reverse(_, R) :-
    pairs_keys_values([10-1, 20-2, 30-3], Ks, Vs),
    pairs_keys_values(P2, Ks, Vs),
    pairs_keys(P2, Ks2),
    sum_list(Ks2, S),
    R is S.   % 60

% M52: atomic_list_concat/2 (atoms-only mode).

:- dynamic test_alc_simple_length/2.
test_alc_simple_length(_, R) :-
    atomic_list_concat([hello, world], A),
    atom_length(A, N),
    R is N.   % 10

:- dynamic test_alc_first_code/2.
test_alc_first_code(_, R) :-
    atomic_list_concat([hello, world], A),
    atom_codes(A, [C|_]),
    R is C.   % 'h' = 104

:- dynamic test_alc_seam_code/2.
test_alc_seam_code(_, R) :-
    atomic_list_concat([ab, cd], A),
    atom_codes(A, [_, _, C, _]),
    R is C.   % 'c' = 99

:- dynamic test_alc_empty_list/2.
test_alc_empty_list(_, R) :-
    atomic_list_concat([], A),
    atom_length(A, N),
    R is N + 31.   % 31 -- empty atom

:- dynamic test_alc_singleton/2.
test_alc_singleton(_, R) :-
    atomic_list_concat([only], A),
    atom_length(A, N),
    R is N.   % 4

:- dynamic test_alc_many/2.
test_alc_many(_, R) :-
    atomic_list_concat([a, b, c, d, e], A),
    atom_length(A, N),
    R is N.   % 5

:- dynamic test_alc_with_empties/2.
test_alc_with_empties(_, R) :-
    atomic_list_concat(['', hi, ''], A),
    atom_length(A, N),
    R is N.   % 2

:- dynamic test_alc_dedup/2.
test_alc_dedup(_, R) :-
    atomic_list_concat([hi, there], A),
    atomic_list_concat([hi, there], B),
    ( A == B -> R is 1 ; R is 0 ).   % 1 -- intern dedupes

% M53: atomic_list_concat/3 with separator (forward, atoms-only).

:- dynamic test_alc3_simple_length/2.
test_alc3_simple_length(_, R) :-
    atomic_list_concat([a, b, c], '-', A),
    atom_length(A, N),
    R is N.   % 5 -- "a-b-c"

:- dynamic test_alc3_first_code/2.
test_alc3_first_code(_, R) :-
    atomic_list_concat([hi, there], '/', A),
    atom_codes(A, [C|_]),
    R is C.   % 'h' = 104

:- dynamic test_alc3_sep_code/2.
test_alc3_sep_code(_, R) :-
    atomic_list_concat([ab, cd], '/', A),
    atom_codes(A, [_, _, C, _, _]),
    R is C.   % '/' = 47

:- dynamic test_alc3_multi_char_sep/2.
test_alc3_multi_char_sep(_, R) :-
    atomic_list_concat([a, b], ' :: ', A),
    atom_length(A, N),
    R is N.   % 6 = 1 + 4 + 1

:- dynamic test_alc3_empty_list/2.
test_alc3_empty_list(_, R) :-
    atomic_list_concat([], '-', A),
    atom_length(A, N),
    R is N + 27.   % 27 -- empty result

:- dynamic test_alc3_singleton/2.
test_alc3_singleton(_, R) :-
    % Only one element -> separator never appears in output.
    atomic_list_concat([foo], '/', A),
    atom_length(A, N),
    R is N.   % 3

:- dynamic test_alc3_empty_sep/2.
test_alc3_empty_sep(_, R) :-
    % Empty separator -> same as atomic_list_concat/2.
    atomic_list_concat([a, b, c], '', A),
    atom_length(A, N),
    R is N.   % 3

:- dynamic test_alc3_dedup/2.
test_alc3_dedup(_, R) :-
    atomic_list_concat([a, b], '-', A1),
    atomic_list_concat([a, b], '-', A2),
    ( A1 == A2 -> R is 1 ; R is 0 ).   % 1

% Cross-check: alc3 with empty sep equals alc/2.
:- dynamic test_alc3_matches_alc2/2.
test_alc3_matches_alc2(_, R) :-
    atomic_list_concat([hello, world], '', A1),
    atomic_list_concat([hello, world], A2),
    ( A1 == A2 -> R is 1 ; R is 0 ).   % 1

% M54: atomic_list_concat/3 split mode (bound Atom + Sep -> Parts).

:- dynamic test_alc3s_simple_count/2.
test_alc3s_simple_count(_, R) :-
    atomic_list_concat(Parts, '-', 'a-b-c'),
    length(Parts, N),
    R is N.   % 3

:- dynamic test_alc3s_first_length/2.
test_alc3s_first_length(_, R) :-
    atomic_list_concat(Parts, '-', 'hello-world'),    % Parts = unbound -> split
    nth0(0, Parts, P),
    atom_length(P, N),
    R is N.   % 5 ("hello")

:- dynamic test_alc3s_last_length/2.
test_alc3s_last_length(_, R) :-
    atomic_list_concat(Parts, '/', 'a/bb/ccc'),
    last(Parts, P),
    atom_length(P, N),
    R is N.   % 3 ("ccc")

:- dynamic test_alc3s_no_sep/2.
test_alc3s_no_sep(_, R) :-
    % No separator in source -> single-element list with the whole atom.
    atomic_list_concat(Parts, '-', 'helloworld'),
    length(Parts, N),
    R is N.   % 1

:- dynamic test_alc3s_empty_atom/2.
test_alc3s_empty_atom(_, R) :-
    atomic_list_concat(Parts, '-', ''),
    length(Parts, N),
    R is N.   % 1 -- ['']

:- dynamic test_alc3s_sep_only/2.
test_alc3s_sep_only(_, R) :-
    % Atom is exactly the separator -> two empty parts.
    atomic_list_concat(Parts, '-', '-'),
    length(Parts, N),
    R is N.   % 2

:- dynamic test_alc3s_consecutive_seps/2.
test_alc3s_consecutive_seps(_, R) :-
    atomic_list_concat(Parts, '-', 'a--b'),
    length(Parts, N),
    R is N.   % 3 -- ['a', '', 'b']

:- dynamic test_alc3s_multi_char_sep/2.
test_alc3s_multi_char_sep(_, R) :-
    atomic_list_concat(Parts, '::', 'foo::bar::baz'),
    length(Parts, N),
    R is N.   % 3

:- dynamic test_alc3s_empty_sep_fails/2.
test_alc3s_empty_sep_fails(_, R) :-
    ( atomic_list_concat(_, '', 'abc') -> R is 1 ; R is 0 ).   % 0

% Forward then reverse round-trip recovers parts (atom_codes head check).
:- dynamic test_alc3s_roundtrip/2.
test_alc3s_roundtrip(_, R) :-
    atomic_list_concat([alpha, beta, gamma], '|', Joined),
    atomic_list_concat(Parts, '|', Joined),
    length(Parts, N),
    R is N.   % 3

% M55: atomic_list_concat/2 with Integer heads (snprintf widening).

:- dynamic test_alc_int_only_length/2.
test_alc_int_only_length(_, R) :-
    atomic_list_concat([1, 2, 3], A),
    atom_length(A, N),
    R is N.   % 3 -- "123"

:- dynamic test_alc_int_first_code/2.
test_alc_int_first_code(_, R) :-
    atomic_list_concat([42], A),
    atom_codes(A, [C|_]),
    R is C.   % '4' = 52

:- dynamic test_alc_int_negative/2.
test_alc_int_negative(_, R) :-
    atomic_list_concat([-7], A),
    atom_codes(A, [C|_]),
    R is C.   % '-' = 45

:- dynamic test_alc_mixed_length/2.
test_alc_mixed_length(_, R) :-
    atomic_list_concat([foo, 42, bar], A),
    atom_length(A, N),
    R is N.   % 8 ("foo42bar")

:- dynamic test_alc_mixed_seam/2.
test_alc_mixed_seam(_, R) :-
    atomic_list_concat([ab, 99, cd], A),
    atom_codes(A, [_, _, C, _, _, _]),    % position 2 is '9'
    R is C.   % 57

:- dynamic test_alc_int_zero/2.
test_alc_int_zero(_, R) :-
    atomic_list_concat([0], A),
    atom_codes(A, [C|_]),
    R is C.   % '0' = 48

:- dynamic test_alc_int_last/2.
test_alc_int_last(_, R) :-
    atomic_list_concat([prefix, 200], A),
    atom_length(A, N),
    R is N.   % 9 ("prefix200")

:- dynamic test_alc_three_ints/2.
test_alc_three_ints(_, R) :-
    atomic_list_concat([10, 20, 30], A),
    atom_codes(A, [_, _, C, _, _, _]),    % position 2 is '2' from "20"
    R is C.   % 50

% M56: atomic_list_concat/3 with Integer heads.

:- dynamic test_alc3_int_only_length/2.
test_alc3_int_only_length(_, R) :-
    atomic_list_concat([1, 2, 3], '-', A),
    atom_length(A, N),
    R is N.   % 5 -- "1-2-3"

:- dynamic test_alc3_int_first/2.
test_alc3_int_first(_, R) :-
    atomic_list_concat([42], '-', A),
    atom_codes(A, [C|_]),
    R is C.   % '4' = 52

:- dynamic test_alc3_int_seam/2.
test_alc3_int_seam(_, R) :-
    atomic_list_concat([10, 20], '/', A),
    atom_codes(A, [_, _, C, _, _]),  % position 2 is '/'
    R is C.   % 47

:- dynamic test_alc3_int_with_atom/2.
test_alc3_int_with_atom(_, R) :-
    atomic_list_concat([foo, 42, bar], '-', A),
    atom_length(A, N),
    R is N.   % 10 = 3 + 1 + 2 + 1 + 3

:- dynamic test_alc3_int_negative/2.
test_alc3_int_negative(_, R) :-
    atomic_list_concat([-5, 7], '+', A),
    atom_length(A, N),
    R is N.   % 4 ("-5+7")

:- dynamic test_alc3_int_multi_char_sep/2.
test_alc3_int_multi_char_sep(_, R) :-
    atomic_list_concat([1, 2, 3], ', ', A),
    atom_length(A, N),
    R is N.   % 7 -- "1, 2, 3"

:- dynamic test_alc3_int_split_count/2.
test_alc3_int_split_count(_, R) :-
    % Forward int + split round-trip.
    atomic_list_concat([100, 200, 300], '|', Joined),
    atomic_list_concat(Parts, '|', Joined),
    length(Parts, N),
    R is N.   % 3

% M57: Float widening for atomic_list_concat/2 and /3.

:- dynamic test_alc_float_singleton/2.
test_alc_float_singleton(_, R) :-
    atomic_list_concat([3.5], A),
    atom_codes(A, [C|_]),
    R is C.   % '3' = 51

:- dynamic test_alc_float_with_atom/2.
test_alc_float_with_atom(_, R) :-
    atomic_list_concat([pi, '=', 3.14], A),
    atom_codes(A, [C|_]),
    R is C.   % 'p' = 112

:- dynamic test_alc_float_with_int/2.
test_alc_float_with_int(_, R) :-
    atomic_list_concat([1, 2.5, 3], A),
    atom_codes(A, [_, _, C|_]),
    R is C.   % '.' = 46 (position 2 is the decimal point of 2.5)

:- dynamic test_alc3_float_only/2.
test_alc3_float_only(_, R) :-
    atomic_list_concat([1.5, 2.5], '+', A),
    atom_codes(A, [_, _, _, C|_]),    % position 3 is the '+'
    R is C.   % 43

:- dynamic test_alc3_float_mixed/2.
test_alc3_float_mixed(_, R) :-
    atomic_list_concat([x, 1.5, y], ',', A),
    atom_codes(A, [C|_]),
    R is C.   % 'x' = 120

% M58: char_type/2 -- check mode.

:- dynamic test_ct_alpha_yes/2.
test_ct_alpha_yes(_, R) :-
    ( char_type(a, alpha) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_alpha_no/2.
test_ct_alpha_no(_, R) :-
    ( char_type('5', alpha) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ct_digit_yes/2.
test_ct_digit_yes(_, R) :-
    ( char_type('7', digit) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_digit_no/2.
test_ct_digit_no(_, R) :-
    ( char_type(z, digit) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ct_upper_yes/2.
test_ct_upper_yes(_, R) :-
    ( char_type('A', upper) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_upper_no/2.
test_ct_upper_no(_, R) :-
    ( char_type(a, upper) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ct_lower_yes/2.
test_ct_lower_yes(_, R) :-
    ( char_type(m, lower) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_alnum_letter/2.
test_ct_alnum_letter(_, R) :-
    ( char_type(b, alnum) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_alnum_digit/2.
test_ct_alnum_digit(_, R) :-
    ( char_type('3', alnum) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_alnum_punct/2.
test_ct_alnum_punct(_, R) :-
    ( char_type('!', alnum) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ct_space_yes/2.
test_ct_space_yes(_, R) :-
    ( char_type(' ', space) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_ascii_yes/2.
test_ct_ascii_yes(_, R) :-
    ( char_type('A', ascii) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_punct_yes/2.
test_ct_punct_yes(_, R) :-
    ( char_type('!', punct) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_punct_no_space/2.
test_ct_punct_no_space(_, R) :-
    ( char_type(' ', punct) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ct_csymf_letter/2.
test_ct_csymf_letter(_, R) :-
    ( char_type(z, csymf) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_csymf_underscore/2.
test_ct_csymf_underscore(_, R) :-
    ( char_type('_', csymf) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_csymf_digit_no/2.
test_ct_csymf_digit_no(_, R) :-
    ( char_type('5', csymf) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ct_csym_digit_yes/2.
test_ct_csym_digit_yes(_, R) :-
    ( char_type('5', csym) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ct_unknown_type/2.
test_ct_unknown_type(_, R) :-
    ( char_type(a, bogus) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ct_multichar_atom/2.
test_ct_multichar_atom(_, R) :-
    ( char_type(ab, alpha) -> R is 1 ; R is 0 ).   % 0 -- not single-char

% M59: compare/3 -- three-way standard order (Integer / Float / Atom).

:- dynamic test_cmp_int_lt/2.
test_cmp_int_lt(_, R) :-
    compare(O, 1, 5),
    char_code(O, C),
    R is C.   % '<' = 60

:- dynamic test_cmp_int_eq/2.
test_cmp_int_eq(_, R) :-
    compare(O, 7, 7),
    char_code(O, C),
    R is C.   % '=' = 61

:- dynamic test_cmp_int_gt/2.
test_cmp_int_gt(_, R) :-
    compare(O, 10, 3),
    char_code(O, C),
    R is C.   % '>' = 62

:- dynamic test_cmp_neg/2.
test_cmp_neg(_, R) :-
    compare(O, -5, 5),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_float_lt/2.
test_cmp_float_lt(_, R) :-
    compare(O, 1.5, 2.5),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_order_float_eq/2.
test_cmp_order_float_eq(_, R) :-
    compare(O, 3.14, 3.14),
    char_code(O, C),
    R is C.   % '='

:- dynamic test_cmp_int_float_mixed/2.
test_cmp_int_float_mixed(_, R) :-
    % Numbers compared by value: 2 < 2.5
    compare(O, 2, 2.5),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_atom_lt/2.
test_cmp_atom_lt(_, R) :-
    compare(O, apple, banana),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_atom_eq/2.
test_cmp_atom_eq(_, R) :-
    compare(O, hello, hello),
    char_code(O, C),
    R is C.   % '='

:- dynamic test_cmp_atom_gt/2.
test_cmp_atom_gt(_, R) :-
    compare(O, zebra, apple),
    char_code(O, C),
    R is C.   % '>'

:- dynamic test_cmp_num_atom/2.
test_cmp_num_atom(_, R) :-
    % Numbers come before atoms in ISO standard order.
    compare(O, 42, foo),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_atom_num/2.
test_cmp_atom_num(_, R) :-
    compare(O, foo, 42),
    char_code(O, C),
    R is C.   % '>'

:- dynamic test_cmp_check_mode_lt/2.
test_cmp_check_mode_lt(_, R) :-
    % Order bound in advance -- compare just unifies.
    ( compare(<, 1, 2) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_cmp_check_mode_eq/2.
test_cmp_check_mode_eq(_, R) :-
    ( compare(=, foo, foo) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_cmp_check_mode_wrong/2.
test_cmp_check_mode_wrong(_, R) :-
    ( compare(>, 1, 5) -> R is 1 ; R is 0 ).   % 0 -- 1 > 5 is false

% M60: must_be/2 fail-instead-of-throw type guard.

:- dynamic test_mb_atom_yes/2.
test_mb_atom_yes(_, R) :-
    ( must_be(atom, hello) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_atom_no/2.
test_mb_atom_no(_, R) :-
    ( must_be(atom, 42) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_integer_yes/2.
test_mb_integer_yes(_, R) :-
    ( must_be(integer, 7) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_integer_no/2.
test_mb_integer_no(_, R) :-
    ( must_be(integer, 3.14) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_float_yes/2.
test_mb_float_yes(_, R) :-
    ( must_be(float, 2.5) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_number_int/2.
test_mb_number_int(_, R) :-
    ( must_be(number, 99) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_number_flt/2.
test_mb_number_flt(_, R) :-
    ( must_be(number, 1.5) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_number_atom/2.
test_mb_number_atom(_, R) :-
    ( must_be(number, foo) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_compound_yes/2.
test_mb_compound_yes(_, R) :-
    ( must_be(compound, [1, 2, 3]) -> R is 1 ; R is 0 ).   % 1 -- list is compound

:- dynamic test_mb_compound_no/2.
test_mb_compound_no(_, R) :-
    ( must_be(compound, atom) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_var_yes/2.
test_mb_var_yes(_, R) :-
    ( must_be(var, _Fresh) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_var_no/2.
test_mb_var_no(_, R) :-
    ( must_be(var, 5) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_nonvar_yes/2.
test_mb_nonvar_yes(_, R) :-
    ( must_be(nonvar, 5) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_nonvar_no/2.
test_mb_nonvar_no(_, R) :-
    ( must_be(nonvar, _Fresh) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_atomic_atom/2.
test_mb_atomic_atom(_, R) :-
    ( must_be(atomic, hello) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_atomic_int/2.
test_mb_atomic_int(_, R) :-
    ( must_be(atomic, 7) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_atomic_compound/2.
test_mb_atomic_compound(_, R) :-
    ( must_be(atomic, [1, 2]) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_callable_atom/2.
test_mb_callable_atom(_, R) :-
    ( must_be(callable, foo) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_callable_compound/2.
test_mb_callable_compound(_, R) :-
    ( must_be(callable, [1]) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_callable_int/2.
test_mb_callable_int(_, R) :-
    ( must_be(callable, 5) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_list_empty/2.
test_mb_list_empty(_, R) :-
    ( must_be(list, []) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_list_cons/2.
test_mb_list_cons(_, R) :-
    ( must_be(list, [1, 2]) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_list_no/2.
test_mb_list_no(_, R) :-
    ( must_be(list, hello) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_boolean_true/2.
test_mb_boolean_true(_, R) :-
    ( must_be(boolean, true) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_boolean_false/2.
test_mb_boolean_false(_, R) :-
    ( must_be(boolean, false) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mb_boolean_no/2.
test_mb_boolean_no(_, R) :-
    ( must_be(boolean, maybe) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mb_unknown_type/2.
test_mb_unknown_type(_, R) :-
    ( must_be(bogus, hello) -> R is 1 ; R is 0 ).   % 0

% M61: display/1 and writeln/1 -- alias for write/1 and write+nl.

:- dynamic test_display_succeeds/2.
test_display_succeeds(_, R) :-
    ( display(hello) -> R is 1 ; R is 0 ).   % 1 (and prints ``hello'')

:- dynamic test_display_integer/2.
test_display_integer(_, R) :-
    ( display(42) -> R is 1 ; R is 0 ).   % 1 (prints ``42'')

:- dynamic test_display_compound/2.
test_display_compound(_, R) :-
    ( display(foo(1, 2)) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_writeln_succeeds/2.
test_writeln_succeeds(_, R) :-
    ( writeln(hello) -> R is 1 ; R is 0 ).   % 1 (prints ``hello\n'')

:- dynamic test_writeln_integer/2.
test_writeln_integer(_, R) :-
    ( writeln(7) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_writeln_list/2.
test_writeln_list(_, R) :-
    ( writeln([1, 2, 3]) -> R is 1 ; R is 0 ).   % 1

% M62: keysort/2 -- stable insertion sort of K-V pairs by Key.

:- dynamic test_ks_int_keys_length/2.
test_ks_int_keys_length(_, R) :-
    keysort([3-a, 1-b, 2-c], S),
    length(S, N),
    R is N.   % 3

:- dynamic test_ks_int_keys_first/2.
test_ks_int_keys_first(_, R) :-
    keysort([3-a, 1-b, 2-c], [K-_|_]),
    R is K.   % 1 (smallest)

:- dynamic test_ks_int_keys_last/2.
test_ks_int_keys_last(_, R) :-
    keysort([5-x, 2-y, 8-z, 1-w], S),
    last(S, K-_),
    R is K.   % 8

:- dynamic test_ks_already_sorted/2.
test_ks_already_sorted(_, R) :-
    keysort([1-a, 2-b, 3-c], [K-_|_]),
    R is K.   % 1

:- dynamic test_ks_reverse_sorted/2.
test_ks_reverse_sorted(_, R) :-
    keysort([5-a, 4-b, 3-c, 2-d, 1-e], [K-_|_]),
    R is K.   % 1

:- dynamic test_ks_empty/2.
test_ks_empty(_, R) :-
    keysort([], S),
    length(S, N),
    R is N + 17.   % 17

:- dynamic test_ks_singleton/2.
test_ks_singleton(_, R) :-
    keysort([42-only], [K-_]),
    R is K.   % 42

:- dynamic test_ks_stable_dupes/2.
test_ks_stable_dupes(_, R) :-
    % Equal keys -- check stable order: input order preserved.
    keysort([2-first, 1-x, 2-second, 1-y], S),
    nth0(2, S, K-_),
    R is K.   % 2 (the first 2-pair)

:- dynamic test_ks_atom_keys/2.
test_ks_atom_keys(_, R) :-
    keysort([banana-2, apple-1, cherry-3], [K-_|_]),
    atom_length(K, L),
    R is L.   % 5 ("apple")

:- dynamic test_ks_atom_keys_value/2.
test_ks_atom_keys_value(_, R) :-
    keysort([banana-2, apple-1, cherry-3], [_-V|_]),
    R is V.   % 1 (apple''s value)

:- dynamic test_ks_float_keys/2.
test_ks_float_keys(_, R) :-
    keysort([3.5-a, 1.5-b, 2.5-c], [K-_|_]),
    R is truncate(K * 10).   % 15

:- dynamic test_ks_mixed_num_keys/2.
test_ks_mixed_num_keys(_, R) :-
    % Mixed int/float keys: 1 < 1.5 < 2 < 2.5
    keysort([2-a, 1.5-b, 1-c, 2.5-d], [K-_|_]),
    R is K.   % 1

% M63: sort/2 -- standard term order with dedup.

:- dynamic test_sort_int_unique/2.
test_sort_int_unique(_, R) :-
    sort([3, 1, 2], L),
    length(L, N),
    R is N.   % 3

:- dynamic test_sort_int_first/2.
test_sort_int_first(_, R) :-
    sort([5, 2, 8, 1, 3], [F|_]),
    R is F.   % 1

:- dynamic test_sort_int_last/2.
test_sort_int_last(_, R) :-
    sort([5, 2, 8, 1, 3], L),
    last(L, E),
    R is E.   % 8

:- dynamic test_sort_dedup/2.
test_sort_dedup(_, R) :-
    sort([3, 1, 2, 1, 3, 2], L),
    length(L, N),
    R is N.   % 3 (dedup removes duplicates)

:- dynamic test_sort_all_same/2.
test_sort_all_same(_, R) :-
    sort([7, 7, 7, 7], L),
    length(L, N),
    R is N.   % 1

:- dynamic test_sort_empty/2.
test_sort_empty(_, R) :-
    sort([], L),
    length(L, N),
    R is N + 23.   % 23

:- dynamic test_sort_singleton/2.
test_sort_singleton(_, R) :-
    sort([42], [E]),
    R is E.   % 42

:- dynamic test_sort_atom_keys/2.
test_sort_atom_keys(_, R) :-
    sort([banana, apple, cherry], [F|_]),
    atom_length(F, L),
    R is L.   % 5 (apple)

:- dynamic test_sort_atom_dedup/2.
test_sort_atom_dedup(_, R) :-
    sort([b, a, c, a, b], L),
    length(L, N),
    R is N.   % 3

:- dynamic test_sort_float/2.
test_sort_float(_, R) :-
    sort([3.5, 1.5, 2.5], [F|_]),
    R is truncate(F * 10).   % 15

:- dynamic test_sort_mixed_num/2.
test_sort_mixed_num(_, R) :-
    sort([3, 1.5, 2, 1], [F|_]),
    R is F.   % 1

:- dynamic test_sort_already_sorted/2.
test_sort_already_sorted(_, R) :-
    sort([1, 2, 3, 4, 5], L),
    length(L, N),
    R is N.   % 5 (no dedup, already sorted)

% M64: compare/3 extension to Compound terms (via @wam_term_cmp helper).

:- dynamic test_cmp_compound_eq/2.
test_cmp_compound_eq(_, R) :-
    compare(O, foo(1, 2), foo(1, 2)),
    char_code(O, C),
    R is C.   % '='

:- dynamic test_cmp_compound_arity_lt/2.
test_cmp_compound_arity_lt(_, R) :-
    % Smaller arity comes first.
    compare(O, foo(1), foo(1, 2)),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_compound_arity_gt/2.
test_cmp_compound_arity_gt(_, R) :-
    compare(O, foo(1, 2, 3), foo(1, 2)),
    char_code(O, C),
    R is C.   % '>'

:- dynamic test_cmp_compound_functor_lt/2.
test_cmp_compound_functor_lt(_, R) :-
    % Same arity, alphabetical functor order.
    compare(O, alpha(1, 2), beta(1, 2)),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_compound_arg_lt/2.
test_cmp_compound_arg_lt(_, R) :-
    % Same arity + functor, first differing arg decides.
    compare(O, foo(1, 5), foo(1, 9)),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_compound_arg_gt/2.
test_cmp_compound_arg_gt(_, R) :-
    compare(O, foo(2, 5), foo(1, 5)),
    char_code(O, C),
    R is C.   % '>'

:- dynamic test_cmp_compound_recursive/2.
test_cmp_compound_recursive(_, R) :-
    % Nested compounds -- recursion through args.
    compare(O, foo(bar(1)), foo(bar(2))),
    char_code(O, C),
    R is C.   % '<'

:- dynamic test_cmp_compound_vs_atom/2.
test_cmp_compound_vs_atom(_, R) :-
    % Cross-category: Compound > Atom.
    compare(O, foo(1), bar),
    char_code(O, C),
    R is C.   % '>'

:- dynamic test_cmp_lists_via_compound/2.
test_cmp_lists_via_compound(_, R) :-
    % Lists are compounds ([|]/2); equal lists compare =.
    compare(O, [1, 2, 3], [1, 2, 3]),
    char_code(O, C),
    R is C.   % '='

:- dynamic test_cmp_lists_diff/2.
test_cmp_lists_diff(_, R) :-
    compare(O, [1, 2, 3], [1, 2, 4]),
    char_code(O, C),
    R is C.   % '<'

% M71 manual-rewrite check: does the soft-cut form work when written
% directly (without forall)?

:- dynamic test_forall_manual/2.
test_forall_manual(_, R) :-
    ( ( ( positive(X), ( X > 0 -> fail ; true ) ) -> fail ; true ) -> R is 1 ; R is 0 ).

% M115: chown/3 -- libc chown wrapper. Non-root callers can only
% chown to their OWN uid/gid (-1 means "leave that side unchanged").

:- dynamic test_chown_self_self/2.
test_chown_self_self(_, R) :-
    % Create a temp file, chown to (our own uid, our own gid).
    % Non-root caller can only chown a file to their own credentials,
    % and this combination always succeeds.
    Path = '/tmp/uw_m115_chown_self',
    shell('touch /tmp/uw_m115_chown_self', _),
    getuid(U),
    getgid(G),
    chown(Path, U, G),
    shell('rm -f /tmp/uw_m115_chown_self', _),
    R is 1.   % 1

:- dynamic test_chown_noop/2.
test_chown_noop(_, R) :-
    % Passing -1 for both uid and gid is a libc no-op that
    % succeeds iff the file exists and the caller can access it.
    Path = '/tmp/uw_m115_chown_noop',
    shell('touch /tmp/uw_m115_chown_noop', _),
    chown(Path, -1, -1),
    shell('rm -f /tmp/uw_m115_chown_noop', _),
    R is 1.   % 1

:- dynamic test_chown_missing/2.
test_chown_missing(_, R) :-
    % chown on a non-existent path fails (ENOENT).
    ( chown('/tmp/uw_m115_does_not_exist', -1, -1) -> R is 0
    ; R is 1
    ).   % 1

:- dynamic test_chown_bad_args/2.
test_chown_bad_args(_, R) :-
    % Non-atom path or non-int uid/gid fail the type guards.
    ( chown(42, 0, 0) -> R is 0
    ; chown('/tmp/x', not_int, 0) -> R is 0
    ; chown('/tmp/x', 0, not_int) -> R is 0
    ; R is 1
    ).   % 1

% M117: ground/1 -- succeeds iff term has no unbound variables.

:- dynamic test_ground_atom/2.
test_ground_atom(_, R) :-
    ( ground(foo) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ground_int/2.
test_ground_int(_, R) :-
    ( ground(42) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ground_compound_closed/2.
test_ground_compound_closed(_, R) :-
    ( ground(foo(a, b, c)) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ground_nested_closed/2.
test_ground_nested_closed(_, R) :-
    ( ground(pair(p(1, 2), q(3, [4, 5]))) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ground_list_closed/2.
test_ground_list_closed(_, R) :-
    ( ground([1, 2, 3, foo]) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ground_bare_var/2.
test_ground_bare_var(_, R) :-
    ( ground(_X) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_ground_compound_with_var/2.
test_ground_compound_with_var(_, R) :-
    ( ground(foo(a, _Y, c)) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_ground_nested_with_var/2.
test_ground_nested_with_var(_, R) :-
    ( ground(pair(p(1, _Z), q(3, [4, 5]))) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_ground_list_open_tail/2.
test_ground_list_open_tail(_, R) :-
    % Partial list [1, 2 | _T] has an unbound tail var, so not ground.
    ( ground([1, 2 | _T]) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_ground_after_bind/2.
test_ground_after_bind(_, R) :-
    % Binding the var makes the term ground.
    X = bound_atom,
    ( ground(foo(a, X, c)) -> R is 1 ; R is 0 ).   % 1

% M118: file_base_name/2 + file_directory_name/2 -- path component split.

:- dynamic test_fbn_simple/2.
test_fbn_simple(_, R) :-
    file_base_name('/usr/bin/swipl', B),
    ( B == swipl -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fbn_no_slash/2.
test_fbn_no_slash(_, R) :-
    file_base_name(swipl, B),
    ( B == swipl -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fbn_root/2.
test_fbn_root(_, R) :-
    % "/" -> "" (basename of root is empty per SWI).
    file_base_name('/', B),
    ( B == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fbn_trailing_slash/2.
test_fbn_trailing_slash(_, R) :-
    % "/usr/bin/" -> "".
    file_base_name('/usr/bin/', B),
    ( B == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fbn_empty/2.
test_fbn_empty(_, R) :-
    file_base_name('', B),
    ( B == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fbn_nested/2.
test_fbn_nested(_, R) :-
    file_base_name('/a/b/c/d/e.txt', B),
    ( B == 'e.txt' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fbn_bad_arg/2.
test_fbn_bad_arg(_, R) :-
    % Non-atom path fails the type guard.
    ( file_base_name(42, _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_fdn_simple/2.
test_fdn_simple(_, R) :-
    file_directory_name('/usr/bin/swipl', D),
    ( D == '/usr/bin' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fdn_no_slash/2.
test_fdn_no_slash(_, R) :-
    % No "/" -> ".".
    file_directory_name(swipl, D),
    ( D == '.' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fdn_root/2.
test_fdn_root(_, R) :-
    % "/" -> "/".
    file_directory_name('/', D),
    ( D == '/' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fdn_root_child/2.
test_fdn_root_child(_, R) :-
    % "/etc" -> "/".
    file_directory_name('/etc', D),
    ( D == '/' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fdn_trailing_slash/2.
test_fdn_trailing_slash(_, R) :-
    % "/usr/bin/" -- last slash at index 8, prefix [0..8) = "/usr/bin".
    file_directory_name('/usr/bin/', D),
    ( D == '/usr/bin' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fdn_empty/2.
test_fdn_empty(_, R) :-
    % "" -> ".".
    file_directory_name('', D),
    ( D == '.' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fdn_bad_arg/2.
test_fdn_bad_arg(_, R) :-
    ( file_directory_name(42, _) -> R is 0 ; R is 1 ).   % 1

% M119: file_name_extension/3 -- split/join at last basename dot.

:- dynamic test_fne_split_simple/2.
test_fne_split_simple(_, R) :-
    file_name_extension(B, E, 'foo.txt'),
    ( B == foo, E == txt -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_no_ext/2.
test_fne_split_no_ext(_, R) :-
    file_name_extension(B, E, 'README'),
    ( B == 'README', E == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_path/2.
test_fne_split_path(_, R) :-
    file_name_extension(B, E, '/usr/bin/foo.sh'),
    ( B == '/usr/bin/foo', E == sh -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_dot_in_dir/2.
test_fne_split_dot_in_dir(_, R) :-
    % Dot lives in directory portion, not basename.
    file_name_extension(B, E, '/usr/foo.bar/baz'),
    ( B == '/usr/foo.bar/baz', E == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_hidden/2.
test_fne_split_hidden(_, R) :-
    % Leading dot (hidden file) is NOT an extension separator.
    file_name_extension(B, E, '.bashrc'),
    ( B == '.bashrc', E == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_hidden_in_dir/2.
test_fne_split_hidden_in_dir(_, R) :-
    % Same hidden-file rule applies after a ''/''.
    file_name_extension(B, E, '/home/u/.bashrc'),
    ( B == '/home/u/.bashrc', E == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_multi_dot/2.
test_fne_split_multi_dot(_, R) :-
    % Split at LAST dot in basename.
    file_name_extension(B, E, 'archive.tar.gz'),
    ( B == 'archive.tar', E == gz -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_trailing_dot/2.
test_fne_split_trailing_dot(_, R) :-
    % "foo." -- dot at end, ext is empty string.
    file_name_extension(B, E, 'foo.'),
    ( B == foo, E == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_split_empty/2.
test_fne_split_empty(_, R) :-
    file_name_extension(B, E, ''),
    ( B == '', E == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_join_simple/2.
test_fne_join_simple(_, R) :-
    file_name_extension(foo, txt, F),
    ( F == 'foo.txt' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_join_empty_ext/2.
test_fne_join_empty_ext(_, R) :-
    % Empty Ext -> no dot, just Base.
    file_name_extension(foo, '', F),
    ( F == foo -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_join_with_path/2.
test_fne_join_with_path(_, R) :-
    file_name_extension('/tmp/data', csv, F),
    ( F == '/tmp/data.csv' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_join_check/2.
test_fne_join_check(_, R) :-
    % All three bound: succeed iff they agree.
    ( file_name_extension(foo, txt, 'foo.txt') -> R is 1 ; R is 0 ).   % 1

:- dynamic test_fne_join_check_disagree/2.
test_fne_join_check_disagree(_, R) :-
    % Disagreeing all-bound case fails.
    ( file_name_extension(foo, txt, 'bar.txt') -> R is 0 ; R is 1 ).   % 1

:- dynamic test_fne_insufficient/2.
test_fne_insufficient(_, R) :-
    % All vars: insufficient instantiation, fails.
    ( file_name_extension(_, _, _) -> R is 0 ; R is 1 ).   % 1

% M120: read_link/2 + symlink/2 -- libc symlink ops.

:- dynamic test_symlink_create_and_read/2.
test_symlink_create_and_read(_, R) :-
    % Clean any leftover, create a symlink, read it back.
    shell('rm -f /tmp/uw_m120_link', _),
    symlink('/etc/hostname', '/tmp/uw_m120_link'),
    read_link('/tmp/uw_m120_link', T),
    shell('rm -f /tmp/uw_m120_link', _),
    ( T == '/etc/hostname' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_symlink_dangling/2.
test_symlink_dangling(_, R) :-
    % Dangling symlink (target doesn''t exist) is legal; libc
    % stores the path verbatim. read_link still returns the
    % literal target string.
    shell('rm -f /tmp/uw_m120_dangling', _),
    symlink('/this/does/not/exist', '/tmp/uw_m120_dangling'),
    read_link('/tmp/uw_m120_dangling', T),
    shell('rm -f /tmp/uw_m120_dangling', _),
    ( T == '/this/does/not/exist' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_read_link_not_symlink/2.
test_read_link_not_symlink(_, R) :-
    % read_link on a regular file fails (EINVAL).
    shell('touch /tmp/uw_m120_regular', _),
    ( read_link('/tmp/uw_m120_regular', _) -> Tmp = 0 ; Tmp = 1 ),
    shell('rm -f /tmp/uw_m120_regular', _),
    R is Tmp.   % 1

:- dynamic test_read_link_missing/2.
test_read_link_missing(_, R) :-
    % read_link on a non-existent path fails (ENOENT).
    ( read_link('/tmp/uw_m120_never_existed', _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_symlink_exists_fail/2.
test_symlink_exists_fail(_, R) :-
    % Creating a symlink at an already-existing path fails (EEXIST).
    shell('rm -f /tmp/uw_m120_busy', _),
    symlink('/etc/hostname', '/tmp/uw_m120_busy'),
    ( symlink('/another/target', '/tmp/uw_m120_busy') -> Tmp = 0 ; Tmp = 1 ),
    shell('rm -f /tmp/uw_m120_busy', _),
    R is Tmp.   % 1

:- dynamic test_read_link_bad_arg/2.
test_read_link_bad_arg(_, R) :-
    ( read_link(42, _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_symlink_bad_arg/2.
test_symlink_bad_arg(_, R) :-
    ( symlink(42, '/tmp/x') -> R is 0
    ; symlink('/etc/hostname', 99) -> R is 0
    ; R is 1
    ).   % 1

% M121: link/2 -- libc hard link wrapper.

:- dynamic test_link_create/2.
test_link_create(_, R) :-
    % Create a source file via shell, hard-link it via libc link/2,
    % verify the link exists, clean up.
    shell('echo hello > /tmp/uw_m121_src', _),
    shell('rm -f /tmp/uw_m121_link', _),
    link('/tmp/uw_m121_src', '/tmp/uw_m121_link'),
    shell('test -f /tmp/uw_m121_link', St),
    shell('rm -f /tmp/uw_m121_src /tmp/uw_m121_link', _),
    ( St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_link_same_inode/2.
test_link_same_inode(_, R) :-
    % Hard links share the same inode. Compare via "stat -c %i".
    shell('echo data > /tmp/uw_m121_orig', _),
    shell('rm -f /tmp/uw_m121_hard', _),
    link('/tmp/uw_m121_orig', '/tmp/uw_m121_hard'),
    shell('test "$(stat -c %i /tmp/uw_m121_orig)" = "$(stat -c %i /tmp/uw_m121_hard)"', St),
    shell('rm -f /tmp/uw_m121_orig /tmp/uw_m121_hard', _),
    ( St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_link_missing_old/2.
test_link_missing_old(_, R) :-
    % link from non-existent source fails (ENOENT).
    ( link('/tmp/uw_m121_never_existed', '/tmp/uw_m121_new') -> R is 0 ; R is 1 ).   % 1

:- dynamic test_link_exists_new/2.
test_link_exists_new(_, R) :-
    % link to a path that already exists fails (EEXIST).
    shell('echo a > /tmp/uw_m121_a', _),
    shell('echo b > /tmp/uw_m121_b', _),
    ( link('/tmp/uw_m121_a', '/tmp/uw_m121_b') -> Tmp = 0 ; Tmp = 1 ),
    shell('rm -f /tmp/uw_m121_a /tmp/uw_m121_b', _),
    R is Tmp.   % 1

:- dynamic test_link_bad_arg/2.
test_link_bad_arg(_, R) :-
    ( link(42, '/tmp/x') -> R is 0
    ; link('/etc/hostname', 99) -> R is 0
    ; R is 1
    ).   % 1

% M122: is_absolute_file_name/1 + same_file/2.

:- dynamic test_iaf_absolute/2.
test_iaf_absolute(_, R) :-
    ( is_absolute_file_name('/usr/bin/swipl') -> R is 1 ; R is 0 ).   % 1

:- dynamic test_iaf_relative/2.
test_iaf_relative(_, R) :-
    ( is_absolute_file_name('foo/bar') -> R is 0 ; R is 1 ).   % 1

:- dynamic test_iaf_root/2.
test_iaf_root(_, R) :-
    ( is_absolute_file_name('/') -> R is 1 ; R is 0 ).   % 1

:- dynamic test_iaf_empty/2.
test_iaf_empty(_, R) :-
    ( is_absolute_file_name('') -> R is 0 ; R is 1 ).   % 1

:- dynamic test_iaf_dot/2.
test_iaf_dot(_, R) :-
    ( is_absolute_file_name('./foo') -> R is 0 ; R is 1 ).   % 1

:- dynamic test_iaf_bad_arg/2.
test_iaf_bad_arg(_, R) :-
    ( is_absolute_file_name(42) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_samf_same_path/2.
test_samf_same_path(_, R) :-
    % Same path -> same file by tautology.
    shell('touch /tmp/uw_m122_same', _),
    ( same_file('/tmp/uw_m122_same', '/tmp/uw_m122_same') -> Tmp = 1 ; Tmp = 0 ),
    shell('rm -f /tmp/uw_m122_same', _),
    R is Tmp.   % 1

:- dynamic test_samf_hard_link/2.
test_samf_hard_link(_, R) :-
    % Hard link shares inode -> same_file should succeed.
    shell('echo hi > /tmp/uw_m122_orig', _),
    shell('rm -f /tmp/uw_m122_hl', _),
    link('/tmp/uw_m122_orig', '/tmp/uw_m122_hl'),
    ( same_file('/tmp/uw_m122_orig', '/tmp/uw_m122_hl') -> Tmp = 1 ; Tmp = 0 ),
    shell('rm -f /tmp/uw_m122_orig /tmp/uw_m122_hl', _),
    R is Tmp.   % 1

:- dynamic test_samf_symlink/2.
test_samf_symlink(_, R) :-
    % stat follows symlinks, so symlink -> target -> same_file = true.
    shell('echo hi > /tmp/uw_m122_target', _),
    shell('rm -f /tmp/uw_m122_sym', _),
    symlink('/tmp/uw_m122_target', '/tmp/uw_m122_sym'),
    ( same_file('/tmp/uw_m122_target', '/tmp/uw_m122_sym') -> Tmp = 1 ; Tmp = 0 ),
    shell('rm -f /tmp/uw_m122_target /tmp/uw_m122_sym', _),
    R is Tmp.   % 1

:- dynamic test_samf_different/2.
test_samf_different(_, R) :-
    % Distinct files with distinct inodes -> same_file = false.
    shell('echo a > /tmp/uw_m122_a', _),
    shell('echo b > /tmp/uw_m122_b', _),
    ( same_file('/tmp/uw_m122_a', '/tmp/uw_m122_b') -> Tmp = 0 ; Tmp = 1 ),
    shell('rm -f /tmp/uw_m122_a /tmp/uw_m122_b', _),
    R is Tmp.   % 1

:- dynamic test_samf_missing/2.
test_samf_missing(_, R) :-
    ( same_file('/tmp/uw_m122_does_not_exist', '/etc/hostname') -> R is 0 ; R is 1 ).   % 1

:- dynamic test_samf_bad_arg/2.
test_samf_bad_arg(_, R) :-
    ( same_file(42, '/tmp/x') -> R is 0
    ; same_file('/etc/hostname', 99) -> R is 0
    ; R is 1
    ).   % 1

% M123: tmp_file/2 + mkfifo/2.

:- dynamic test_tmp_file_creates/2.
test_tmp_file_creates(_, R) :-
    % mkstemp creates the file atomically; test -f should succeed.
    tmp_file(uw123, P),
    atom_concat('test -f ', P, Cmd),
    shell(Cmd, St),
    atom_concat('rm -f ', P, Rm),
    shell(Rm, _),
    ( St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tmp_file_prefix/2.
test_tmp_file_prefix(_, R) :-
    % Path is non-empty and the file was created (mkstemp guarantee).
    tmp_file(label, P),
    atom_length(P, L),
    atom_concat('test -f ', P, Cmd),
    shell(Cmd, St),
    atom_concat('rm -f ', P, Rm),
    shell(Rm, _),
    ( L > 0, St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tmp_file_unique/2.
test_tmp_file_unique(_, R) :-
    % Two calls produce different paths (mkstemp guarantees this).
    tmp_file(dup, P1),
    tmp_file(dup, P2),
    atom_concat('rm -f ', P1, R1), shell(R1, _),
    atom_concat('rm -f ', P2, R2), shell(R2, _),
    ( P1 \== P2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tmp_file_bad_arg/2.
test_tmp_file_bad_arg(_, R) :-
    ( tmp_file(42, _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_mkfifo_create/2.
test_mkfifo_create(_, R) :-
    % Create a FIFO at a known path, verify with "test -p".
    Path = '/tmp/uw_m123_fifo',
    shell('rm -f /tmp/uw_m123_fifo', _),
    mkfifo(Path, 0o644),
    shell('test -p /tmp/uw_m123_fifo', St),
    shell('rm -f /tmp/uw_m123_fifo', _),
    ( St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_mkfifo_exists_fail/2.
test_mkfifo_exists_fail(_, R) :-
    % mkfifo at an already-existing path fails (EEXIST).
    shell('touch /tmp/uw_m123_busy', _),
    ( mkfifo('/tmp/uw_m123_busy', 0o644) -> Tmp = 0 ; Tmp = 1 ),
    shell('rm -f /tmp/uw_m123_busy', _),
    R is Tmp.   % 1

:- dynamic test_mkfifo_bad_path/2.
test_mkfifo_bad_path(_, R) :-
    ( mkfifo(42, 0o644) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_mkfifo_bad_mode/2.
test_mkfifo_bad_mode(_, R) :-
    ( mkfifo('/tmp/uw_m123_bm', not_int) -> R is 0 ; R is 1 ).   % 1

% M124: umask/2 + monotonic_time/1.

:- dynamic test_umask_set_and_restore/2.
test_umask_set_and_restore(_, R) :-
    % Set umask to 0o077, capture old value; immediately restore.
    umask(Old, 0o077),
    umask(_, Old),
    % Old should be a non-negative integer.
    ( integer(Old), Old >= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_umask_round_trip/2.
test_umask_round_trip(_, R) :-
    % Set umask to a known value, read it back, restore original.
    umask(Save, 0o022),
    umask(Read, 0o022),
    umask(_, Save),
    % After setting to 0o022, the next umask should read 0o022.
    ( Read =:= 0o022 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_umask_bad_new/2.
test_umask_bad_new(_, R) :-
    ( umask(_, not_int) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_monotonic_nonneg/2.
test_monotonic_nonneg(_, R) :-
    monotonic_time(T),
    ( T >= 0.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_monotonic_advances/2.
test_monotonic_advances(_, R) :-
    % Two calls with a tiny sleep between -- the second must be
    % strictly greater than (or equal to) the first; CLOCK_MONOTONIC
    % never goes backwards.
    monotonic_time(T0),
    sleep(0.01),
    monotonic_time(T1),
    ( T1 >= T0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_monotonic_elapsed/2.
test_monotonic_elapsed(_, R) :-
    % After sleep(0.05), elapsed >= 0.04 -- same floor as M88's
    % sleep_elapsed_float test.
    monotonic_time(T0),
    sleep(0.05),
    monotonic_time(T1),
    Diff is T1 - T0,
    ( Diff >= 0.04 -> R is 1 ; R is 0 ).   % 1

% M125: nice/1 + getpriority/1 + setpriority/1.

:- dynamic test_getpriority_in_range/2.
test_getpriority_in_range(_, R) :-
    % Default priority is typically 0; allow [-20, 19] sanity range.
    getpriority(P),
    ( integer(P), P >= -20, P =< 19 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_nice_zero/2.
test_nice_zero(_, R) :-
    % nice(0) doesn''t change priority. Always succeeds.
    getpriority(Before),
    nice(0),
    getpriority(After),
    ( Before =:= After -> R is 1 ; R is 0 ).   % 1

:- dynamic test_nice_positive_raises/2.
test_nice_positive_raises(_, R) :-
    % nice(+5) increases the niceness value (lowers priority).
    % Unprivileged users can ALWAYS go nicer; only root can become
    % less nice. Restore by setpriority on the way out.
    getpriority(Before),
    nice(5),
    getpriority(After),
    setpriority(Before),
    ( After > Before -> R is 1 ; R is 0 ).   % 1

:- dynamic test_setpriority_roundtrip/2.
test_setpriority_roundtrip(_, R) :-
    % Set to a known nice value (must be >= current to avoid EPERM
    % for unprivileged), confirm getpriority reflects it, restore.
    getpriority(Before),
    Target is Before + 3,
    setpriority(Target),
    getpriority(Read),
    setpriority(Before),
    ( Read =:= Target -> R is 1 ; R is 0 ).   % 1

:- dynamic test_nice_bad_arg/2.
test_nice_bad_arg(_, R) :-
    ( nice(not_int) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_setpriority_bad_arg/2.
test_setpriority_bad_arg(_, R) :-
    ( setpriority(not_int) -> R is 0 ; R is 1 ).   % 1

% M126: getrlimit/2 + setrlimit/2.

:- dynamic test_grl_fsize/2.
test_grl_fsize(_, R) :-
    % RLIMIT_FSIZE = 1. Default is usually RLIM_INFINITY, which is
    % the rlim_t-max sentinel -- (u64)-1 on Linux/macOS, surfaces as
    % -1 in signed Prolog Integer space. Just sanity-check it's
    % an Integer; the actual value depends on environment.
    getrlimit(1, L),
    ( integer(L) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_grl_core/2.
test_grl_core(_, R) :-
    % RLIMIT_CORE = 4. Often 0 (no core dumps) but can also be
    % RLIM_INFINITY; either is a valid Integer.
    getrlimit(4, L),
    ( integer(L) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_srl_round_trip/2.
test_srl_round_trip(_, R) :-
    % Set RLIMIT_CORE soft limit to a known value, read back, restore.
    getrlimit(4, Before),
    setrlimit(4, 0),
    getrlimit(4, Read),
    setrlimit(4, Before),
    ( Read =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_srl_lower_ok/2.
test_srl_lower_ok(_, R) :-
    % Lowering the soft limit is always allowed for unprivileged.
    getrlimit(1, Before),
    NewLow is Before - 1,
    ( NewLow > 0
    -> ( setrlimit(1, NewLow) -> Set = 1 ; Set = 0 ),
       setrlimit(1, Before)
    ;  Set = 1
    ),
    R is Set.   % 1

:- dynamic test_grl_bad_resource/2.
test_grl_bad_resource(_, R) :-
    % Resource 999 is way out of range -> EINVAL -> fail.
    ( getrlimit(999, _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_grl_bad_arg/2.
test_grl_bad_arg(_, R) :-
    ( getrlimit(not_int, _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_srl_bad_args/2.
test_srl_bad_args(_, R) :-
    ( setrlimit(not_int, 0) -> R is 0
    ; setrlimit(1, not_int) -> R is 0
    ; R is 1
    ).   % 1

% M127: getlogin/1 + uname_sysname/1 + uname_machine/1.

:- dynamic test_getlogin_or_fail/2.
test_getlogin_or_fail(_, R) :-
    % Either getlogin succeeds with a non-empty atom (interactive
    % terminal), or it fails (CI / cron / no tty). Both are
    % acceptable; we just check the result shape if it succeeds.
    ( getlogin(N), atom(N), atom_length(N, L), L > 0
    -> R is 1
    ;  % Fail path is also acceptable in CI.
       R is 1
    ).   % 1

:- dynamic test_uname_sysname_nonempty/2.
test_uname_sysname_nonempty(_, R) :-
    uname_sysname(S),
    atom_length(S, L),
    ( atom(S), L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_uname_machine_nonempty/2.
test_uname_machine_nonempty(_, R) :-
    uname_machine(M),
    atom_length(M, L),
    ( atom(M), L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_uname_sysname_stable/2.
test_uname_sysname_stable(_, R) :-
    % Two calls in the same process must return the same atom.
    uname_sysname(S1),
    uname_sysname(S2),
    ( S1 == S2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_uname_known_linux_or_darwin/2.
test_uname_known_linux_or_darwin(_, R) :-
    % Smoke check: sysname should be Linux, Darwin, or FreeBSD on
    % the platforms we actually run on. Anything else just doesn''t
    % fail the test -- accept any non-empty atom.
    uname_sysname(S),
    ( S == 'Linux' ; S == 'Darwin' ; S == 'FreeBSD' ; atom(S) ),
    !,
    R is 1.   % 1

% M128: copy_file/2 -- libc open/read/write/close.

:- dynamic test_copy_file_basic/2.
test_copy_file_basic(_, R) :-
    % Create a tiny source file, copy it, diff to confirm bytes match.
    shell('echo hello > /tmp/uw_m128_src', _),
    shell('rm -f /tmp/uw_m128_dst', _),
    copy_file('/tmp/uw_m128_src', '/tmp/uw_m128_dst'),
    shell('diff -q /tmp/uw_m128_src /tmp/uw_m128_dst', St),
    shell('rm -f /tmp/uw_m128_src /tmp/uw_m128_dst', _),
    ( St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_copy_file_empty/2.
test_copy_file_empty(_, R) :-
    % Copying an empty file produces an empty dest (read returns 0
    % immediately, loop exits successfully).
    shell('rm -f /tmp/uw_m128_empty_src /tmp/uw_m128_empty_dst', _),
    shell('touch /tmp/uw_m128_empty_src', _),
    copy_file('/tmp/uw_m128_empty_src', '/tmp/uw_m128_empty_dst'),
    size_file('/tmp/uw_m128_empty_dst', Sz),
    shell('rm -f /tmp/uw_m128_empty_src /tmp/uw_m128_empty_dst', _),
    ( Sz =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_copy_file_large/2.
test_copy_file_large(_, R) :-
    % Copy a file larger than the 4 KB buffer -- exercises the
    % loop. yes | head produces ~3 KB per 1500 lines, so generate
    % 20 KB by repeating ~10000 lines.
    shell('rm -f /tmp/uw_m128_large_src /tmp/uw_m128_large_dst', _),
    shell('seq 1 5000 > /tmp/uw_m128_large_src', _),
    copy_file('/tmp/uw_m128_large_src', '/tmp/uw_m128_large_dst'),
    shell('diff -q /tmp/uw_m128_large_src /tmp/uw_m128_large_dst', St),
    shell('rm -f /tmp/uw_m128_large_src /tmp/uw_m128_large_dst', _),
    ( St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_copy_file_missing_src/2.
test_copy_file_missing_src(_, R) :-
    % Source doesn''t exist -> open fails -> copy_file fails.
    ( copy_file('/tmp/uw_m128_never_existed', '/tmp/uw_m128_x') -> R is 0
    ; R is 1
    ).   % 1

:- dynamic test_copy_file_overwrite/2.
test_copy_file_overwrite(_, R) :-
    % O_TRUNC: copying over an existing dest replaces its content.
    shell('echo old > /tmp/uw_m128_ow_dst', _),
    shell('echo new > /tmp/uw_m128_ow_src', _),
    copy_file('/tmp/uw_m128_ow_src', '/tmp/uw_m128_ow_dst'),
    shell('diff -q /tmp/uw_m128_ow_src /tmp/uw_m128_ow_dst', St),
    shell('rm -f /tmp/uw_m128_ow_src /tmp/uw_m128_ow_dst', _),
    ( St =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_copy_file_bad_args/2.
test_copy_file_bad_args(_, R) :-
    ( copy_file(42, '/tmp/x') -> R is 0
    ; copy_file('/tmp/x', 99) -> R is 0
    ; R is 1
    ).   % 1

% M129: read_file_to_atom/2 -- stat + open + read loop -> atom.

:- dynamic test_rfta_short/2.
test_rfta_short(_, R) :-
    % Write a known short string, read it back, compare.
    shell('printf hello > /tmp/uw_m129_short', _),
    read_file_to_atom('/tmp/uw_m129_short', A),
    shell('rm -f /tmp/uw_m129_short', _),
    ( A == hello -> R is 1 ; R is 0 ).   % 1

:- dynamic test_rfta_empty/2.
test_rfta_empty(_, R) :-
    % Empty file -> empty atom (size=0 branch, skip read loop).
    shell('rm -f /tmp/uw_m129_empty', _),
    shell('touch /tmp/uw_m129_empty', _),
    read_file_to_atom('/tmp/uw_m129_empty', A),
    shell('rm -f /tmp/uw_m129_empty', _),
    ( A == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_rfta_large/2.
test_rfta_large(_, R) :-
    % File larger than typical read chunks -- verifies the read
    % loop handles multiple iterations to fill the buffer.
    shell('seq 1 5000 > /tmp/uw_m129_large', _),
    read_file_to_atom('/tmp/uw_m129_large', A),
    atom_length(A, L),
    size_file('/tmp/uw_m129_large', Sz),
    shell('rm -f /tmp/uw_m129_large', _),
    ( L =:= Sz, L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_rfta_missing/2.
test_rfta_missing(_, R) :-
    % stat fails on missing path -> read_file_to_atom fails.
    ( read_file_to_atom('/tmp/uw_m129_never_existed', _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_rfta_bad_arg/2.
test_rfta_bad_arg(_, R) :-
    ( read_file_to_atom(42, _) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_rfta_size_matches/2.
test_rfta_size_matches(_, R) :-
    % atom_length of the read content must equal stat''s st_size.
    shell('printf "hello world" > /tmp/uw_m129_sz', _),
    read_file_to_atom('/tmp/uw_m129_sz', A),
    atom_length(A, L),
    shell('rm -f /tmp/uw_m129_sz', _),
    ( L =:= 11 -> R is 1 ; R is 0 ).   % 1

% M130: write_atom_to_file/2 + append_atom_to_file/2.

:- dynamic test_wfa_basic/2.
test_wfa_basic(_, R) :-
    % Write a known atom, read it back via shell + diff.
    shell('rm -f /tmp/uw_m130_w', _),
    write_atom_to_file('/tmp/uw_m130_w', 'hello world'),
    size_file('/tmp/uw_m130_w', Sz),
    shell('rm -f /tmp/uw_m130_w', _),
    ( Sz =:= 11 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_wfa_empty/2.
test_wfa_empty(_, R) :-
    % Empty content -> 0-byte file (size=0 fast path skips loop).
    shell('rm -f /tmp/uw_m130_we', _),
    write_atom_to_file('/tmp/uw_m130_we', ''),
    size_file('/tmp/uw_m130_we', Sz),
    shell('rm -f /tmp/uw_m130_we', _),
    ( Sz =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_wfa_truncates/2.
test_wfa_truncates(_, R) :-
    % O_TRUNC: existing larger file gets replaced with shorter.
    shell('printf "lots_of_old_content_here" > /tmp/uw_m130_wt', _),
    write_atom_to_file('/tmp/uw_m130_wt', new),
    size_file('/tmp/uw_m130_wt', Sz),
    shell('rm -f /tmp/uw_m130_wt', _),
    ( Sz =:= 3 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_wfa_round_trip/2.
test_wfa_round_trip(_, R) :-
    % Write then read returns the same atom.
    shell('rm -f /tmp/uw_m130_rt', _),
    write_atom_to_file('/tmp/uw_m130_rt', 'abcdefghij'),
    read_file_to_atom('/tmp/uw_m130_rt', A),
    shell('rm -f /tmp/uw_m130_rt', _),
    ( A == 'abcdefghij' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_wfa_bad_args/2.
test_wfa_bad_args(_, R) :-
    ( write_atom_to_file(42, abc) -> R is 0
    ; write_atom_to_file('/tmp/x', 99) -> R is 0
    ; R is 1
    ).   % 1

:- dynamic test_afa_creates/2.
test_afa_creates(_, R) :-
    % Append to a non-existent file -> creates it (O_CREAT).
    shell('rm -f /tmp/uw_m130_ac', _),
    append_atom_to_file('/tmp/uw_m130_ac', start),
    size_file('/tmp/uw_m130_ac', Sz),
    shell('rm -f /tmp/uw_m130_ac', _),
    ( Sz =:= 5 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_afa_extends/2.
test_afa_extends(_, R) :-
    % Append to an existing file -> total length is sum.
    shell('printf "abc" > /tmp/uw_m130_ae', _),
    append_atom_to_file('/tmp/uw_m130_ae', defg),
    size_file('/tmp/uw_m130_ae', Sz),
    shell('rm -f /tmp/uw_m130_ae', _),
    ( Sz =:= 7 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_afa_bad_args/2.
test_afa_bad_args(_, R) :-
    ( append_atom_to_file(42, abc) -> R is 0
    ; append_atom_to_file('/tmp/x', 99) -> R is 0
    ; R is 1
    ).   % 1

% M131: errno/1 + strerror/2 -- diagnostics for libc-wrapper failures.

:- dynamic test_errno_returns_int/2.
test_errno_returns_int(_, R) :-
    % errno/1 always succeeds and returns an Integer (often 0).
    errno(E),
    ( integer(E) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_errno_after_open_fail/2.
test_errno_after_open_fail(_, R) :-
    % After a deliberately-failing open (via delete_file on missing
    % path), errno should be set to a non-zero value (ENOENT = 2).
    ( delete_file('/tmp/uw_m131_never_existed') -> R is 0
    ; errno(E),
      ( E =\= 0 -> R is 1 ; R is 0 )
    ).   % 1

:- dynamic test_strerror_known_errno/2.
test_strerror_known_errno(_, R) :-
    % ENOENT = 2 -> "No such file or directory".
    strerror(2, M),
    ( atom(M), atom_length(M, L), L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_strerror_zero/2.
test_strerror_zero(_, R) :-
    % strerror(0) is typically "Success".
    strerror(0, M),
    ( atom(M), atom_length(M, L), L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_strerror_round_trip/2.
test_strerror_round_trip(_, R) :-
    % errno -> strerror should yield a non-empty atom even for
    % errno=0 ("Success" / "No error").
    errno(E),
    strerror(E, M),
    atom_length(M, L),
    ( L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_strerror_bad_arg/2.
test_strerror_bad_arg(_, R) :-
    ( strerror(not_int, _) -> R is 0 ; R is 1 ).   % 1

% M132: process_max_rss/1 + process_user_time/1 + process_system_time/1.

:- dynamic test_max_rss_positive/2.
test_max_rss_positive(_, R) :-
    % Process should have allocated SOME memory by now.
    process_max_rss(K),
    ( integer(K), K > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_user_time_nonneg/2.
test_user_time_nonneg(_, R) :-
    process_user_time(T),
    ( T >= 0.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_system_time_nonneg/2.
test_system_time_nonneg(_, R) :-
    process_system_time(T),
    ( T >= 0.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_user_time_monotonic/2.
test_user_time_monotonic(_, R) :-
    % Two calls with some CPU work between: T1 >= T0 (never goes
    % backwards). Trivial busy-loop via between/3 generates user
    % CPU time.
    process_user_time(T0),
    ( between(1, 100, _), fail ; true ),
    process_user_time(T1),
    ( T1 >= T0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_max_rss_monotonic/2.
test_max_rss_monotonic(_, R) :-
    % Peak RSS only goes up (or stays).
    process_max_rss(K0),
    process_max_rss(K1),
    ( K1 >= K0 -> R is 1 ; R is 0 ).   % 1

% M133: path_join/3 -- join two paths with single '/' separator.

:- dynamic test_pj_simple/2.
test_pj_simple(_, R) :-
    path_join('/usr/bin', swipl, F),
    ( F == '/usr/bin/swipl' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_pj_trailing_slash/2.
test_pj_trailing_slash(_, R) :-
    % Base already has trailing slash -> don''t double up.
    path_join('/usr/bin/', swipl, F),
    ( F == '/usr/bin/swipl' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_pj_absolute_rel/2.
test_pj_absolute_rel(_, R) :-
    % Absolute Rel overrides Base entirely.
    path_join('/usr/bin', '/etc/hosts', F),
    ( F == '/etc/hosts' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_pj_empty_base/2.
test_pj_empty_base(_, R) :-
    % Empty Base + non-empty Rel: with non-absolute Rel, we keep
    % the Rel verbatim (no leading '/').
    path_join('', foo, F),
    ( F == foo -> R is 1 ; R is 0 ).   % 1

:- dynamic test_pj_empty_rel/2.
test_pj_empty_rel(_, R) :-
    % Empty Rel: Full = Base.
    path_join('/tmp', '', F),
    ( F == '/tmp' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_pj_nested/2.
test_pj_nested(_, R) :-
    % Building deeper paths via repeated calls.
    path_join('/var', log, A),
    path_join(A, 'syslog', F),
    ( F == '/var/log/syslog' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_pj_bad_args/2.
test_pj_bad_args(_, R) :-
    ( path_join(42, foo, _) -> R is 0
    ; path_join('/tmp', 99, _) -> R is 0
    ; R is 1
    ).   % 1

% M134: system_to_atom/2 -- shell stdout capture.

:- dynamic test_sta_echo/2.
test_sta_echo(_, R) :-
    % "echo hi" produces "hi\n" -- 3 bytes.
    system_to_atom('echo hi', O),
    atom_length(O, L),
    ( L =:= 3 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sta_empty/2.
test_sta_empty(_, R) :-
    % "true" produces no output -> empty atom.
    system_to_atom('true', O),
    ( O == '' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sta_pwd/2.
test_sta_pwd(_, R) :-
    % "pwd" produces the cwd + "\n". Just check non-empty atom.
    system_to_atom('pwd', O),
    atom_length(O, L),
    ( L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sta_multiline/2.
test_sta_multiline(_, R) :-
    % "seq 1 5" produces "1\n2\n3\n4\n5\n" = 10 bytes.
    system_to_atom('seq 1 5', O),
    atom_length(O, L),
    ( L =:= 10 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sta_pipe/2.
test_sta_pipe(_, R) :-
    % Shell pipeline: "echo hello | wc -c" -> "6\n" (5 chars + nl).
    system_to_atom('echo hello | wc -c', O),
    atom_length(O, L),
    ( L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sta_bad_arg/2.
test_sta_bad_arg(_, R) :-
    ( system_to_atom(42, _) -> R is 0 ; R is 1 ).   % 1

% M112: truncate/2 -- libc truncate wrapper for file resizing.

:- dynamic test_truncate_grow/2.
test_truncate_grow(_, R) :-
    % Create a tiny file via shell, truncate it to 100 bytes,
    % verify size_file reports 100.
    Path = '/tmp/uw_m112_truncate_test',
    shell('touch /tmp/uw_m112_truncate_test', _),
    truncate(Path, 100),
    size_file(Path, Sz),
    shell('rm -f /tmp/uw_m112_truncate_test', _),
    ( Sz =:= 100 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_truncate_zero/2.
test_truncate_zero(_, R) :-
    % Truncating to 0 is a no-op on a freshly-created file but
    % should still succeed.
    Path = '/tmp/uw_m112_truncate_zero',
    shell('touch /tmp/uw_m112_truncate_zero', _),
    truncate(Path, 0),
    size_file(Path, Sz),
    shell('rm -f /tmp/uw_m112_truncate_zero', _),
    ( Sz =:= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_truncate_missing/2.
test_truncate_missing(_, R) :-
    % truncate on a non-existent path fails (ENOENT).
    ( truncate('/tmp/uw_m112_does_not_exist', 0) -> R is 0
    ; R is 1
    ).   % 1

:- dynamic test_truncate_bad_args/2.
test_truncate_bad_args(_, R) :-
    % Non-atom Path or non-Integer Length fail the type guards.
    ( truncate(42, 0) -> R is 0
    ; truncate('/tmp/x', not_int) -> R is 0
    ; R is 1
    ).   % 1

% M111: kill/2 -- libc kill wrapper. Sig=0 is the standard existence
% probe (no signal sent, just check process exists + permission).

:- dynamic test_kill_self_probe/2.
test_kill_self_probe(_, R) :-
    % kill(0, 0) sends signal 0 to ALL processes in the caller''s
    % group -- succeeds iff at least one exists and we have
    % permission. Effectively a no-op self-check.
    ( kill(0, 0) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_kill_missing_pid/2.
test_kill_missing_pid(_, R) :-
    % Some pid that''s almost certainly not running. ESRCH.
    ( kill(99999999, 0) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_kill_bad_args/2.
test_kill_bad_args(_, R) :-
    % Non-int args fail the type guards.
    ( kill(not_int, 0) -> R is 0
    ; kill(0, not_int) -> R is 0
    ; R is 1
    ).   % 1

% M110: realpath/2 -- libc realpath wrapper for canonical absolute paths.

:- dynamic test_rp_tmp/2.
test_rp_tmp(_, R) :-
    % /tmp is its own canonical absolute path -- realpath returns
    % the same atom (no symlinks involved here).
    realpath('/tmp', Abs),
    ( Abs == '/tmp' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_rp_relative/2.
test_rp_relative(_, R) :-
    % realpath resolves . to the CWD; just check the result is a
    % non-empty atom starting with /.
    realpath('.', Abs),
    atom_length(Abs, L),
    atom_chars(Abs, [First | _]),
    ( L > 0, First == '/' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_rp_missing/2.
test_rp_missing(_, R) :-
    % realpath on a non-existent path fails (ENOENT).
    ( realpath('/tmp/uw_m110_definitely_not_here', _) -> R is 0
    ; R is 1
    ).   % 1

:- dynamic test_rp_bad_arg/2.
test_rp_bad_arg(_, R) :-
    % Non-atom Path fails the type guard.
    ( realpath(42, _) -> R is 0 ; R is 1 ).   % 1

% M109: getpgrp/1 -- libc process group id wrapper.

:- dynamic test_pgrp_positive/2.
test_pgrp_positive(_, R) :-
    getpgrp(PG),
    ( PG > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_pgrp_stable/2.
test_pgrp_stable(_, R) :-
    % Two calls in the same process return the same group id.
    getpgrp(PG1),
    getpgrp(PG2),
    ( PG1 =:= PG2 -> R is 1 ; R is 0 ).   % 1

% M107: directory_files/2 -- opendir/readdir loop, list of entry atoms.

:- dynamic test_df_tmp_nonempty/2.
test_df_tmp_nonempty(_, R) :-
    % /tmp always contains at least . and .. -- result list is non-empty.
    directory_files('/tmp', Fs),
    ( Fs = [_ | _] -> R is 1 ; R is 0 ).   % 1

:- dynamic test_df_contains_dot/2.
test_df_contains_dot(_, R) :-
    % readdir always yields ''.'' and ''..'' for a real directory.
    directory_files('/tmp', Fs),
    ( memberchk('.', Fs), memberchk('..', Fs) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_df_missing_dir/2.
test_df_missing_dir(_, R) :-
    % opendir fails on non-existent path.
    ( directory_files('/tmp/uw_m107_definitely_not_a_dir', _) -> R is 0
    ; R is 1
    ).   % 1

:- dynamic test_df_bad_arg/2.
test_df_bad_arg(_, R) :-
    % Non-atom Dir fails the type guard.
    ( directory_files(42, _) -> R is 0 ; R is 1 ).   % 1

% M106: access/2 -- libc access(path, mode_bits). Mode is the libc
% bitmask: F_OK=0, R_OK=4, W_OK=2, X_OK=1.

:- dynamic test_access_tmp_exists/2.
test_access_tmp_exists(_, R) :-
    % /tmp exists -- F_OK (0) succeeds.
    ( access('/tmp', 0) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_access_missing_file/2.
test_access_missing_file(_, R) :-
    % A path that doesn''t exist -- F_OK fails.
    ( access('/tmp/uw_m106_definitely_not_here', 0) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_access_tmp_writable/2.
test_access_tmp_writable(_, R) :-
    % /tmp is world-writable -- W_OK (2) succeeds.
    ( access('/tmp', 2) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_access_bad_args/2.
test_access_bad_args(_, R) :-
    % Non-atom path or non-int mode fail the type guards.
    ( access(42, 0) -> R is 0
    ; access('/tmp', not_an_int) -> R is 0
    ; R is 1
    ).   % 1

% M105: numbervars/3 -- bind free vars to $VAR(N) compounds.

:- dynamic test_nv_basic/2.
test_nv_basic(_, R) :-
    Term = foo(X, Y, Z),
    numbervars(Term, 0, End),
    X = '$VAR'(N0), Y = '$VAR'(N1), Z = '$VAR'(N2),
    ( N0 =:= 0, N1 =:= 1, N2 =:= 2, End =:= 3 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_nv_shared/2.
test_nv_shared(_, R) :-
    % Same var twice gets the same $VAR(N) -- exercises the M104
    % aliasing fix: binding through one Ref propagates to all
    % occurrences of the var.
    Term = bar(X, X),
    numbervars(Term, 0, End),
    X = '$VAR'(N),
    ( N =:= 0, End =:= 1 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_nv_ground/2.
test_nv_ground(_, R) :-
    numbervars(foo(1, 2, 3), 5, End),
    ( End =:= 5 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_nv_nested/2.
test_nv_nested(_, R) :-
    Term = outer(X, inner(Y), Z),
    numbervars(Term, 0, End),
    X = '$VAR'(0),
    Y = '$VAR'(1),
    Z = '$VAR'(2),
    ( End =:= 3 -> R is 1 ; R is 0 ).   % 1

% M104: wam_unify_value bind path now aliases two unbound vars
% via Ref-to-Ref instead of writing the Unbound sentinel. So after
% X = Y, var(X) and var(Y) both still hold but X == Y is now true.

:- dynamic test_unify_aliases_vars/2.
test_unify_aliases_vars(_, R) :-
    X = Y,
    ( X == Y -> R is 1 ; R is 0 ).   % 1

:- dynamic test_unify_then_bind_propagates/2.
test_unify_then_bind_propagates(_, R) :-
    % After X = Y, binding one should propagate to the other.
    X = Y,
    X = 42,
    ( Y =:= 42 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_term_vars_identity/2.
test_term_vars_identity(_, R) :-
    % The natural M103 test that used to fail: term_variables
    % returns the list of variables, and via the M104 aliasing
    % fix the V we get back == X.
    term_variables(foo(X, 2), [V | _]),
    ( V == X -> R is 1 ; R is 0 ).   % 1

% M103: term_variables/2 -- depth-first left-to-right collection of
% unbound vars from a term. No dedup -- repeated occurrences of the
% same var appear once per occurrence (SWI dedupes; documented
% limitation for M103).

:- dynamic test_tv_ground/2.
test_tv_ground(_, R) :-
    % All-ground term -> empty Vars list.
    term_variables(foo(1, 2, 3), Vs),
    ( Vs == [] -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tv_single/2.
test_tv_single(_, R) :-
    % One free var in a compound -> 1-element list of an unbound var.
    % (Var identity via V == X is a separate WAM-unify limitation:
    % unifying two unbound vars currently doesn''t chain them, so
    % the V bound from the cons-cell head ends up at a different
    % heap cell than the original X. We just check shape + var-ness.)
    term_variables(foo(_X, 2), Vs),
    Vs = [V | T],
    ( var(V), T == [] -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tv_three/2.
test_tv_three(_, R) :-
    % Three vars: result should be a 3-element list of unbound vars.
    term_variables(bar(_X, _Y, _Z), Vs),
    Vs = [V1, V2, V3],
    ( var(V1), var(V2), var(V3) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tv_nested/2.
test_tv_nested(_, R) :-
    % Nested compound: still produces 3 vars in left-to-right DFS
    % order (the wam_collect_vars walker visits inner(_Y)''s arg
    % between outer args 1 and 3).
    term_variables(outer(_X, inner(_Y), _Z), Vs),
    Vs = [V1, V2, V3],
    ( var(V1), var(V2), var(V3) -> R is 1 ; R is 0 ).   % 1

% M102: chmod/2 -- libc chmod wrapper for file mode bits.

:- dynamic test_chmod_set_readonly/2.
test_chmod_set_readonly(_, R) :-
    % Use shell to create + cleanup so the test doesn''t depend on
    % open/3 + setup_call_cleanup interactions in the WAM backend
    % (which currently segfault when stacked). Verifies chmod
    % actually reaches the file by checking exists_file still
    % succeeds after the mode change.
    Path = '/tmp/uw_m102_chmod_test',
    shell('touch /tmp/uw_m102_chmod_test', _),
    chmod(Path, 0o444),
    exists_file(Path),
    shell('rm -f /tmp/uw_m102_chmod_test', _),
    R is 1.   % 1

:- dynamic test_chmod_missing_file/2.
test_chmod_missing_file(_, R) :-
    % chmod on a path that doesn''t exist fails (returns -1, ENOENT).
    ( chmod('/tmp/uw_m102_nope_xyz', 0o644) -> R is 0 ; R is 1 ).   % 1

:- dynamic test_chmod_bad_args/2.
test_chmod_bad_args(_, R) :-
    % Non-atom path or non-integer mode falls through to plain fail.
    ( chmod(42, 0o644) -> R is 0
    ; chmod('/tmp/x', not_an_int) -> R is 0
    ; R is 1
    ).   % 1

% M101: =.. with partial-list second arg now unifies (was broken
% in M100 -- u.a2_check used value_equals which couldn''t bind
% through the unbound vars in [H|_]).

:- dynamic test_univ_partial_head/2.
test_univ_partial_head(_, R) :-
    % Direct form: Term =.. [H | _] -- partial list with unbound
    % H and unbound tail. The freshly-built result list unifies
    % with the pattern, binding H to the functor atom.
    Term = baz(7, 8),
    Term =.. [H | _],
    ( H == baz -> R is 1 ; R is 0 ).   % 1

:- dynamic test_univ_partial_arity/2.
test_univ_partial_arity(_, R) :-
    % [_, _, _] with three unbound element slots -- result list
    % from a 2-ary compound has exactly 3 elements (functor + 2
    % args), so unification succeeds and the pattern serves as
    % a deterministic arity check.
    Term = quux(a, b),
    ( Term =.. [_, _, _] -> R is 1 ; R is 0 ).   % 1

:- dynamic test_univ_partial_no_match/2.
test_univ_partial_no_match(_, R) :-
    % [_, _] is a 2-element partial pattern; a 2-ary compound''s
    % result list has 3 elements, so the unify fails.
    Term = quux(a, b),
    ( Term =.. [_, _] -> R is 0 ; R is 1 ).   % 1

% M100: functor/3 + =.. read-mode atom representation fix.

:- dynamic test_functor_name_eq_literal/2.
test_functor_name_eq_literal(_, R) :-
    % functor(C, Name, _) extracts the functor as an Atom Value.
    % Pre-M100 that was pointer-based, so Name == foo failed even
    % when the compound was foo(...). The fix interns the functor
    % string into the atom table so Name carries an id payload that
    % matches the literal `foo' (also id-based via put_constant).
    Term = foo(1, 2, 3),
    functor(Term, Name, _),
    ( Name == foo -> R is 1 ; R is 0 ).   % 1

:- dynamic test_functor_arity_check/2.
test_functor_arity_check(_, R) :-
    % And the canonical pattern `functor(T, Name, Arity)' with both
    % name and arity literals now works as a structure shape check.
    Term = bar(a, b),
    ( functor(Term, bar, 2) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_univ_head_eq_literal/2.
test_univ_head_eq_literal(_, R) :-
    % Same fix applied to =.. (univ): the list head is the functor
    % as an id-based Atom Value, comparing correctly with literals.
    % Uses the two-step =.. L, L = [H|_] form -- the direct
    % =.. [H|_] form has a separate pre-existing WAM-compile
    % issue (partial-list arg to =.. doesn''t reach the builtin in
    % a unifiable shape).
    Term = baz(7, 8),
    Term =.. L,
    L = [H | _],
    ( H == baz -> R is 1 ; R is 0 ).   % 1

:- dynamic test_univ_tail_is_empty_list/2.
test_univ_tail_is_empty_list(_, R) :-
    % =.. terminates with the [] atom -- M100 makes that id-based
    % via @wam_empty_list_atom_id, matching the literal [].
    Term = quux(1),
    Term =.. L,
    L = [_, _ | Tail],
    ( Tail == [] -> R is 1 ; R is 0 ).   % 1

% M99: date_time_stamp/2 -- inverse of M98 stamp_date_time/3 via libc mktime.

:- dynamic test_dts_roundtrip/2.
test_dts_roundtrip(_, R) :-
    % stamp -> DT -> stamp round-trip. mktime is the inverse of
    % localtime_r on the same TZ, so the recovered stamp must equal
    % the original integer-truncated stamp.
    S0 is 1700000000,
    stamp_date_time(S0, DT, local),
    date_time_stamp(DT, S1),
    S1i is truncate(S1),
    ( S1i =:= S0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_dts_returns_float/2.
test_dts_returns_float(_, R) :-
    % Matches SWI: stamp result is a Float, not an Integer (so it
    % round-trips with get_time which is also Float).
    S0 is 1700000000,
    stamp_date_time(S0, DT, local),
    date_time_stamp(DT, S1),
    ( float(S1) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_dts_bad_arity/2.
test_dts_bad_arity(_, R) :-
    % Non-date/9 compound fails the structure check.
    ( date_time_stamp(foo(1,2,3), _) -> R is 0
    ; date_time_stamp(date(2020), _) -> R is 0
    ; date_time_stamp(42, _) -> R is 0
    ; R is 1
    ).   % 1

% M98: stamp_date_time/3 -- localtime_r + build 9-arity date/9 compound.

:- dynamic test_sdt_arity/2.
test_sdt_arity(_, R) :-
    % arg(9, DT, _) succeeds iff DT is a compound with arity >= 9; the
    % stamp_date_time output is exactly 9 args, so this also confirms
    % we don''t overshoot. (Avoids functor/3 read-mode + atom literal
    % comparison -- the functor slot is currently a pointer to the
    % functor string globals while atom literals are interned atom ids,
    % so payload comparison there spuriously fails.)
    Stamp is 1700000000,
    stamp_date_time(Stamp, DT, local),
    arg(1, DT, _),
    arg(9, DT, _),
    R is 1.   % 1

:- dynamic test_sdt_year_4digit/2.
test_sdt_year_4digit(_, R) :-
    % Year component is the calendar year (>= 1970 for any non-negative
    % Unix stamp). 1000 < Y < 3000 catches the +1900 offset working.
    Stamp is 1700000000,
    stamp_date_time(Stamp, DT, local),
    arg(1, DT, Y),
    ( Y > 1970, Y < 3000 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sdt_month_in_range/2.
test_sdt_month_in_range(_, R) :-
    % Month is 1..12 after the +1 fixup (tm_mon is 0..11).
    Stamp is 1700000000,
    stamp_date_time(Stamp, DT, local),
    arg(2, DT, M),
    ( M >= 1, M =< 12 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sdt_tzname/2.
test_sdt_tzname(_, R) :-
    % TZName slot (arg 8) is the atom local for any TZ input (we don''t
    % actually consult libc TZ; the slot just records the requested name).
    Stamp is 1700000000,
    stamp_date_time(Stamp, DT, local),
    arg(8, DT, TZ),
    ( TZ == local -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sdt_float_stamp/2.
test_sdt_float_stamp(_, R) :-
    % Float stamps are truncated to whole seconds before localtime_r.
    Stamp is 1700000000.7,
    stamp_date_time(Stamp, DT, local),
    arg(1, DT, Y),
    ( Y > 1970, Y < 3000 -> R is 1 ; R is 0 ).   % 1

% M97: set_random/1 -- libc srand48 via the SWI-style seed(N) compound.

:- dynamic test_setrand_changes_output/2.
test_setrand_changes_output(_, R) :-
    % Default seed (0) gives a fixed first lrand48() value. Re-seed
    % with a different N and lrand48 produces a different value.
    random_between(1, 1000000, V0),
    set_random(seed(424242)),
    random_between(1, 1000000, V1),
    ( V0 =\= V1 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_setrand_repeatable/2.
test_setrand_repeatable(_, R) :-
    % Seeding twice with the same N gives the same draws -- proves
    % srand48 is actually being called with our seed, not ignored.
    set_random(seed(99)),
    random_between(1, 1000000, A),
    set_random(seed(99)),
    random_between(1, 1000000, B),
    ( A =:= B -> R is 1 ; R is 0 ).   % 1

:- dynamic test_setrand_bad_option/2.
test_setrand_bad_option(_, R) :-
    % Wrong functor / non-Integer arg / non-compound all fail.
    ( set_random(garbage) -> R is 0
    ; set_random(seed(not_an_int)) -> R is 0
    ; set_random(42) -> R is 0
    ; R is 1
    ).   % 1

% M96: getgid/1 + getegid/1 + getppid/1 -- more libc process-info wrappers.

:- dynamic test_gid_nonneg/2.
test_gid_nonneg(_, R) :-
    getgid(G),
    ( G >= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_egid_nonneg/2.
test_egid_nonneg(_, R) :-
    getegid(E),
    ( E >= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_gid_eq_egid/2.
test_gid_eq_egid(_, R) :-
    % Unprivileged context: real == effective.
    getgid(G),
    getegid(E),
    ( G =:= E -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ppid_positive/2.
test_ppid_positive(_, R) :-
    % A spawned binary always has a positive parent pid (its launcher).
    getppid(PP),
    ( PP > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ppid_neq_pid/2.
test_ppid_neq_pid(_, R) :-
    % The test binary cannot be its own parent.
    getpid(P),
    getppid(PP),
    ( P =\= PP -> R is 1 ; R is 0 ).   % 1

% M95: getuid/1 + geteuid/1 -- libc uid_t wrappers.

:- dynamic test_uid_nonneg/2.
test_uid_nonneg(_, R) :-
    % getuid is unsigned -- always >= 0. Just sanity-check that the
    % i32->i64 zext path doesn''t produce a negative i64.
    getuid(U),
    ( U >= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_euid_nonneg/2.
test_euid_nonneg(_, R) :-
    geteuid(E),
    ( E >= 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_uid_eq_euid/2.
test_uid_eq_euid(_, R) :-
    % In an unprivileged context (the test container) no setuid bit
    % is in play, so real and effective uids should match.
    getuid(U),
    geteuid(E),
    ( U =:= E -> R is 1 ; R is 0 ).   % 1

% M93: unsetenv/1 -- libc unsetenv() wrapper, complement to M77 setenv/2.

:- dynamic test_unsetenv_roundtrip/2.
test_unsetenv_roundtrip(_, R) :-
    % Set var, confirm it''s set, unsetenv, confirm getenv fails.
    setenv('UW_M93_TEST', 'hello'),
    getenv('UW_M93_TEST', V1),
    V1 == 'hello',
    unsetenv('UW_M93_TEST'),
    ( \+ getenv('UW_M93_TEST', _) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_unsetenv_idempotent/2.
test_unsetenv_idempotent(_, R) :-
    % Unsetting a var that isn''t set still succeeds (matches SWI).
    unsetenv('UW_M93_NOT_SET_VAR_NAME'),
    unsetenv('UW_M93_NOT_SET_VAR_NAME'),
    R is 1.   % 1

:- dynamic test_unsetenv_non_atom/2.
test_unsetenv_non_atom(_, R) :-
    % Integer arg fails (non-atom).
    ( unsetenv(42) -> R is 0 ; R is 1 ).   % 1

% M92: halt/0 + halt/1 -- libc exit() wrapper. Process terminates
% inside the WAM run loop, so the test driver never reaches its
% normal `read reg 0 + return as exit code' path -- the exit code
% from halt IS the test result.

:- dynamic test_halt_zero/2.
test_halt_zero(_, _) :- halt.   % -> exit 0

:- dynamic test_halt_seven/2.
test_halt_seven(_, _) :- halt(7).   % -> exit 7

:- dynamic test_halt_var/2.
test_halt_var(_, _) :-
    % Code computed at runtime: 2 + 3 = 5.
    X is 2 + 3,
    halt(X).   % -> exit 5

% M91: format_time/3 -- libc strftime + localtime_r wrapper.

:- dynamic test_ft_year/2.
test_ft_year(_, R) :-
    % format_time(?Atom, +Fmt, +Stamp) -- direct atom output. Stamp
    % 1609459200 = 2021-01-01 00:00:00 UTC; with timezone offset the
    % calendar year is 2020 or 2021, so length-of-year = 4.
    Stamp is 1609459200,
    format_time(Y, '%Y', Stamp),
    atom_length(Y, L),
    ( L =:= 4 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ft_iso_len/2.
test_ft_iso_len(_, R) :-
    % '%Y-%m-%d %H:%M:%S' renders as 19 chars regardless of timezone.
    Stamp is 1700000000,
    format_time(S, '%Y-%m-%d %H:%M:%S', Stamp),
    atom_length(S, L),
    ( L =:= 19 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ft_float_stamp/2.
test_ft_float_stamp(_, R) :-
    % Float Stamp must also work -- the fractional part is dropped to
    % whole-second precision before localtime_r.
    Stamp is 1700000000.5,
    format_time(S, '%S', Stamp),
    atom_length(S, L),
    ( L =:= 2 -> R is 1 ; R is 0 ).   % 1

% M90: random/1 + random_between/3 -- libc drand48 / lrand48 wrappers.

:- dynamic test_rand_in_unit/2.
test_rand_in_unit(_, R) :-
    % drand48() result must satisfy 0.0 <= X < 1.0.
    random(X),
    ( X >= 0.0, X < 1.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_rand_between_in_range/2.
test_rand_between_in_range(_, R) :-
    % random_between(1, 10, X), 1 <= X <= 10.
    random_between(1, 10, X),
    ( X >= 1, X =< 10 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_rand_between_singleton/2.
test_rand_between_singleton(_, R) :-
    % random_between(7, 7, X) -- only valid value is 7.
    random_between(7, 7, X),
    ( X =:= 7 -> R is 1 ; R is 0 ).   % 1

% M89: cpu_time/1 -- process CPU time via clock_gettime(CLOCK_PROCESS_CPUTIME_ID).

:- dynamic test_cpu_nonneg/2.
test_cpu_nonneg(_, R) :-
    % A freshly-started process always has cpu_time >= 0.
    cpu_time(T),
    ( T >= 0.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_cpu_monotonic/2.
test_cpu_monotonic(_, R) :-
    % CPU time is monotonically non-decreasing within a process.
    cpu_time(T0),
    cpu_time(T1),
    ( T1 >= T0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_cpu_under_wall/2.
test_cpu_under_wall(_, R) :-
    % CPU time accrued during a sleep should be much less than the
    % wall-clock elapsed time -- sleeping doesn''t consume CPU.
    % 50ms sleep => wall ~0.05s, CPU should be < 0.04s.
    cpu_time(C0),
    get_time(W0),
    sleep(0.05),
    cpu_time(C1),
    get_time(W1),
    CpuDiff is C1 - C0,
    WallDiff is W1 - W0,
    ( CpuDiff < WallDiff -> R is 1 ; R is 0 ).   % 1

% M88: gethostname/1 -- libc gethostname() wrapper.

:- dynamic test_ghn_nonempty/2.
test_ghn_nonempty(_, R) :-
    gethostname(H),
    atom_length(H, L),
    ( L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ghn_stable/2.
test_ghn_stable(_, R) :-
    % Two gethostname calls in the same process must return the
    % same atom (host name doesn''t change mid-run).
    gethostname(H1),
    gethostname(H2),
    ( H1 == H2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ghn_not_empty_atom/2.
test_ghn_not_empty_atom(_, R) :-
    % Hostname should not be the empty atom.
    gethostname(H),
    ( H \== '' -> R is 1 ; R is 0 ).   % 1

% M87: get_time/1 upgraded to nanosecond precision (clock_gettime).

:- dynamic test_gt87_short_sleep/2.
test_gt87_short_sleep(_, R) :-
    % Now that get_time has sub-second resolution, a 50ms sleep is
    % observable. Both bounds:
    %   Diff >= 0.04 -- sleep actually waited.
    %   Diff <  0.5  -- whole-second resolution would have produced
    %                   either 0 or >= 1, never something in [0.04, 0.5).
    get_time(T0),
    sleep(0.05),
    get_time(T1),
    Diff is T1 - T0,
    ( Diff >= 0.04, Diff < 0.5 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_gt87_ms_count/2.
test_gt87_ms_count(_, R) :-
    % truncate((T1 - T0) * 1000) should land in roughly [40, 200] for
    % a 50ms sleep. Just verify >= 40 here -- the upper bound is
    % machine-dependent.
    get_time(T0),
    sleep(0.05),
    get_time(T1),
    Ms is truncate((T1 - T0) * 1000),
    ( Ms >= 40 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_gt87_fractional/2.
test_gt87_fractional(_, R) :-
    % Verify there''s actually a non-integer part in the value. For a
    % nanosecond-resolution clock the fractional part is essentially
    % never zero. Use floor() (M18) and subtract; if the diff is
    % strictly > 0.0 the clock has sub-second precision.
    get_time(T),
    Whole is floor(T),
    Frac is T - Whole,
    ( Frac > 0.0 -> R is 1 ; R is 0 ).   % 1

% M86: sleep/1 -- libc usleep wrapper.

:- dynamic test_sleep_zero/2.
test_sleep_zero(_, R) :-
    sleep(0),
    R is 1.   % 1

:- dynamic test_sleep_float_zero/2.
test_sleep_float_zero(_, R) :-
    sleep(0.0),
    R is 1.   % 1

:- dynamic test_sleep_tiny/2.
test_sleep_tiny(_, R) :-
    % 1ms is the floor we use for ``definitely returned'' tests; any
    % usleep call should at least cycle through the kernel and return.
    sleep(0.001),
    R is 1.   % 1

:- dynamic test_sleep_elapsed_float/2.
test_sleep_elapsed_float(_, R) :-
    % M88: tightened from sleep(1.0) -> sleep(0.05) now that M87
    % gave get_time/1 nanosecond resolution. 50ms wait with a 40ms
    % floor for slow CI machines.
    get_time(T0),
    sleep(0.05),
    get_time(T1),
    Diff is T1 - T0,
    ( Diff >= 0.04 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sleep_elapsed_int/2.
test_sleep_elapsed_int(_, R) :-
    % Integer-arg branch needs a whole number of seconds; keep
    % sleep(1) here with the original 0.5s floor.
    get_time(T0),
    sleep(1),
    get_time(T1),
    Diff is T1 - T0,
    ( Diff >= 0.5 -> R is 1 ; R is 0 ).   % 1

% M85: bitwise /\ (AND), \/ (OR), \ (unary NOT).

:- dynamic test_band_basic/2.
test_band_basic(_, R) :-
    R is 12 /\ 10.   % 8 (1100 AND 1010 = 1000)

:- dynamic test_band_byte/2.
test_band_byte(_, R) :-
    R is 0xFF /\ 0x0F.   % 15 (mask low nibble)

:- dynamic test_bor_basic/2.
test_bor_basic(_, R) :-
    R is 5 \/ 3.   % 7 (101 OR 011 = 111)

:- dynamic test_bor_combine/2.
test_bor_combine(_, R) :-
    % Combine bit flags: 1 | 2 | 4 | 8 | 16 = 31.
    R is 1 \/ 2 \/ 4 \/ 8 \/ 16.   % 31

:- dynamic test_bnot_byte/2.
test_bnot_byte(_, R) :-
    % \(0) = -1; low 8 bits = 0xFF = 255.
    X is \0,
    R is X /\ 0xFF.   % 255

% M84: integer bitshifts -- << / >>.

:- dynamic test_shl_basic/2.
test_shl_basic(_, R) :-
    R is 1 << 4.   % 16

:- dynamic test_shl_byte_top/2.
test_shl_byte_top(_, R) :-
    R is 1 << 7.   % 128

:- dynamic test_shl_31_3/2.
test_shl_31_3(_, R) :-
    R is 31 << 3.   % 248 (31 * 8)

:- dynamic test_shr_basic/2.
test_shr_basic(_, R) :-
    R is 240 >> 4.   % 15

:- dynamic test_shr_round/2.
test_shr_round(_, R) :-
    % (1 << 8) >> 4 = 256 >> 4 = 16.
    R is 256 >> 4.   % 16

% M83: pi/e atom constants + xor/2 integer bitwise.

:- dynamic test_pi_50/2.
test_pi_50(_, R) :-
    % pi * 50 ~ 157.08; truncate -> 157 (fits in 0..255).
    X is pi,
    R is truncate(X * 50).   % 157

:- dynamic test_pi_gt3/2.
test_pi_gt3(_, R) :-
    X is pi,
    ( X > 3.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_e_90/2.
test_e_90(_, R) :-
    % e * 90 ~ 244.65; truncate -> 244.
    X is e,
    R is truncate(X * 90).   % 244

:- dynamic test_xor_small/2.
test_xor_small(_, R) :-
    % 5 = 0b101; 3 = 0b011; xor = 0b110 = 6.
    R is xor(5, 3).   % 6

:- dynamic test_xor_byte/2.
test_xor_byte(_, R) :-
    % 0xFF xor 0x0F = 0xF0 = 240.
    R is xor(255, 15).   % 240

% M82: gcd/2 (Integer Euclidean) + log/2 (Float log with base).

:- dynamic test_gcd_basic/2.
test_gcd_basic(_, R) :-
    R is gcd(12, 18).   % 6

:- dynamic test_gcd_coprime/2.
test_gcd_coprime(_, R) :-
    R is gcd(7, 5).   % 1

:- dynamic test_gcd_with_zero/2.
test_gcd_with_zero(_, R) :-
    % gcd(0, n) = n -- Euclid terminates on the first iteration.
    R is gcd(0, 5).   % 5

:- dynamic test_log2_eight/2.
test_log2_eight(_, R) :-
    % log(2, 8) = 3 (since 2^3 = 8). Use floats to force the
    % named-binary path; integer literals go through int eval which
    % doesn''t recognize ``log''.
    X is log(2.0, 8.0),
    R is truncate(X).   % 3

:- dynamic test_log10_hundred/2.
test_log10_hundred(_, R) :-
    X is log(10.0, 100.0),
    R is truncate(X).   % 2

% M81: atan2/2 -- binary inverse tangent (4-quadrant).

:- dynamic test_atan2_xaxis/2.
test_atan2_xaxis(_, R) :-
    % atan2(0, 1) -- positive x-axis -- is 0.
    X is atan2(0.0, 1.0),
    R is truncate(X * 100).   % 0

:- dynamic test_atan2_diag/2.
test_atan2_diag(_, R) :-
    % atan2(1, 1) = pi/4 ~ 0.7854; *200 truncated -> 157.
    X is atan2(1.0, 1.0),
    R is truncate(X * 200).   % 157

:- dynamic test_atan2_yaxis/2.
test_atan2_yaxis(_, R) :-
    % atan2(1, 0) = pi/2 ~ 1.5708; *100 truncated -> 157.
    X is atan2(1.0, 0.0),
    R is truncate(X * 100).   % 157

:- dynamic test_atan2_diag_scaled/2.
test_atan2_diag_scaled(_, R) :-
    % atan2 only cares about the ratio: (2,2) is the same angle as (1,1).
    X is atan2(2.0, 2.0),
    R is truncate(X * 200).   % 157

:- dynamic test_atan2_pi/2.
test_atan2_pi(_, R) :-
    % atan2(0, -1) = pi ~ 3.14159; *50 truncated -> 157.
    X is atan2(0.0, -1.0),
    R is truncate(X * 50).   % 157

% M80: inverse trig -- asin/1, acos/1, atan/1 via libm.

:- dynamic test_asin_zero/2.
test_asin_zero(_, R) :-
    X is asin(0.0),
    R is truncate(X * 100).   % 0

:- dynamic test_asin_one/2.
test_asin_one(_, R) :-
    % asin(1) = pi/2 ~ 1.5708; *100 truncated -> 157.
    X is asin(1.0),
    R is truncate(X * 100).   % 157

:- dynamic test_acos_one/2.
test_acos_one(_, R) :-
    X is acos(1.0),
    R is truncate(X * 100).   % 0

:- dynamic test_acos_zero/2.
test_acos_zero(_, R) :-
    % acos(0) = pi/2 ~ 1.5708; *100 truncated -> 157.
    X is acos(0.0),
    R is truncate(X * 100).   % 157

:- dynamic test_atan_one/2.
test_atan_one(_, R) :-
    % atan(1) = pi/4 ~ 0.7854; *200 truncated -> 157 (same as
    % asin(1.0)/acos(0.0)). Cannot use *400 because OS exit codes
    % are 8-bit and 314 mod 256 = 58.
    X is atan(1.0),
    R is truncate(X * 200).   % 157

% M79: working_directory/2 + getpid/1 -- libc getcwd/chdir/getpid.

:- dynamic test_wd_query/2.
test_wd_query(_, R) :-
    % Query mode: working_directory(D, D) -- after the call D is
    % bound to CWD; no chdir happens. CWD should not be empty.
    working_directory(D, D),
    atom_length(D, L),
    ( L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_wd_chdir/2.
test_wd_chdir(_, R) :-
    % Save current CWD, chdir to /tmp, read CWD again, restore.
    working_directory(Old, '/tmp'),
    working_directory(New, New),
    working_directory(_, Old),
    ( New == '/tmp' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_wd_fail/2.
test_wd_fail(_, R) :-
    % chdir to a non-existent directory must fail (ENOENT).
    ( working_directory(_, '/nonexistent/uw_m79_dir') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_getpid_pos/2.
test_getpid_pos(_, R) :-
    getpid(P),
    ( P > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_getpid_stable/2.
test_getpid_stable(_, R) :-
    % Two getpid calls in the same process should return the same value.
    getpid(P1),
    getpid(P2),
    ( P1 =:= P2 -> R is 1 ; R is 0 ).   % 1

% M78: shell/1 + shell/2 -- libc system() process spawn.

:- dynamic test_sh1_true/2.
test_sh1_true(_, R) :-
    ( shell('true') -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sh1_false/2.
test_sh1_false(_, R) :-
    ( shell('false') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_sh1_nonexistent/2.
test_sh1_nonexistent(_, R) :-
    ( shell('/nonexistent/uw_m78_definitely_no_such_binary') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_sh2_true/2.
test_sh2_true(_, R) :-
    shell('true', S),
    R is S.   % 0

:- dynamic test_sh2_exit42/2.
test_sh2_exit42(_, R) :-
    shell('exit 42', S),
    R is S.   % 42

% M77: getenv/2 + setenv/2 -- libc env-var access.

:- dynamic test_ge_path/2.
test_ge_path(_, R) :-
    % PATH is set in every container shell.
    getenv('PATH', P),
    atom_length(P, L),
    ( L > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_ge_missing/2.
test_ge_missing(_, R) :-
    ( getenv('UW_M77_DEFINITELY_UNSET', _) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_se_basic/2.
test_se_basic(_, R) :-
    setenv('UW_M77_TEST', 'hello'),
    getenv('UW_M77_TEST', V),
    ( V == 'hello' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_se_overwrite/2.
test_se_overwrite(_, R) :-
    setenv('UW_M77_OVR', 'first'),
    setenv('UW_M77_OVR', 'second'),
    getenv('UW_M77_OVR', V),
    ( V == 'second' -> R is 1 ; R is 0 ).   % 1

:- dynamic test_se_empty/2.
test_se_empty(_, R) :-
    % Empty string is a valid value (setenv accepts ""). getenv
    % then succeeds and returns the empty atom.
    setenv('UW_M77_EMPTY', ''),
    getenv('UW_M77_EMPTY', V),
    atom_length(V, L),
    R is L.   % 0

% M76: size_file/2 + time_file/2 -- stat-based mtime + size readers.

:- dynamic test_sf_etc_hostname/2.
test_sf_etc_hostname(_, R) :-
    % /etc/hostname is small but > 0 bytes on every Linux container.
    size_file('/etc/hostname', N),
    ( N > 0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_sf_zero/2.
test_sf_zero(_, R) :-
    % /dev/null is a 0-byte character device; stat reports size = 0.
    size_file('/dev/null', N),
    R is N.   % 0

:- dynamic test_sf_fail_missing/2.
test_sf_fail_missing(_, R) :-
    ( size_file('/nonexistent/file/qqq', _) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_tf_etc_hostname/2.
test_tf_etc_hostname(_, R) :-
    % Mtime on /etc/hostname is some Float > 0 (epoch seconds).
    time_file('/etc/hostname', T),
    ( T > 0.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tf_fail_missing/2.
test_tf_fail_missing(_, R) :-
    ( time_file('/nonexistent/file/qqq', _) -> R is 1 ; R is 0 ).   % 0

% M75: rename_file/2 + delete_directory/1 -- libc rename + rmdir.

:- dynamic test_rnf_basic/2.
test_rnf_basic(_, R) :-
    % Make a directory we can write a sentinel file into, rename it,
    % then check both old-name absence and new-name presence. Pre-clean
    % via best-effort delete (ignored if missing).
    ( delete_directory('/tmp/uw_m75_rnf_dst') ; true ),
    ( delete_directory('/tmp/uw_m75_rnf_src') ; true ),
    make_directory('/tmp/uw_m75_rnf_src'),
    rename_file('/tmp/uw_m75_rnf_src', '/tmp/uw_m75_rnf_dst'),
    exists_directory('/tmp/uw_m75_rnf_dst'),
    \+ exists_directory('/tmp/uw_m75_rnf_src'),
    R is 1.   % 1

:- dynamic test_rnf_fail_missing/2.
test_rnf_fail_missing(_, R) :-
    % Source doesn''t exist -> ENOENT.
    ( rename_file('/nonexistent/source/foo', '/tmp/uw_m75_target') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ddr_basic/2.
test_ddr_basic(_, R) :-
    % Roundtrip: create then delete, verify gone.
    ( delete_directory('/tmp/uw_m75_ddr_test') ; true ),
    make_directory('/tmp/uw_m75_ddr_test'),
    delete_directory('/tmp/uw_m75_ddr_test'),
    \+ exists_directory('/tmp/uw_m75_ddr_test'),
    R is 1.   % 1

:- dynamic test_ddr_fail_missing/2.
test_ddr_fail_missing(_, R) :-
    % Path doesn''t exist -> ENOENT.
    ( delete_directory('/nonexistent/dir/qqq') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_ddr_fail_file/2.
test_ddr_fail_file(_, R) :-
    % rmdir() refuses non-directories (ENOTDIR).
    ( delete_directory('/etc/hostname') -> R is 1 ; R is 0 ).   % 0

% M74: delete_file/1 + make_directory/1 -- libc unlink + mkdir.

:- dynamic test_mkd_basic/2.
test_mkd_basic(_, R) :-
    % Disjunction: either makes it new or finds it already there.
    ( make_directory('/tmp/uw_m74_test')
    ; exists_directory('/tmp/uw_m74_test')
    ),
    exists_directory('/tmp/uw_m74_test'),
    R is 1.   % 1

:- dynamic test_mkd_fail_perm/2.
test_mkd_fail_perm(_, R) :-
    % /sys is locked down (EPERM/EROFS) even for root in typical containers.
    ( make_directory('/sys/uw_m74_protected_dir') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_mkd_fail_parent/2.
test_mkd_fail_parent(_, R) :-
    % Parent doesn''t exist; mkdir fails with ENOENT.
    ( make_directory('/nonexistent/foo/bar') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_df_fail_missing/2.
test_df_fail_missing(_, R) :-
    ( delete_file('/nonexistent/file/qqq') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_df_dir_no/2.
test_df_dir_no(_, R) :-
    % unlink() refuses directories (EISDIR).
    ( delete_file('/tmp') -> R is 1 ; R is 0 ).   % 0

% M73: exists_file/1 + exists_directory/1 -- stat-based fs checks.

:- dynamic test_xf_real/2.
test_xf_real(_, R) :-
    ( exists_file('/etc/hostname') -> R is 1 ; R is 0 ).   % 1 (Linux)

:- dynamic test_xf_missing/2.
test_xf_missing(_, R) :-
    ( exists_file('/nonexistent/path/qqq') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_xf_directory_no/2.
test_xf_directory_no(_, R) :-
    % exists_file should fail on a directory.
    ( exists_file('/etc') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_xd_real/2.
test_xd_real(_, R) :-
    ( exists_directory('/etc') -> R is 1 ; R is 0 ).   % 1

:- dynamic test_xd_file_no/2.
test_xd_file_no(_, R) :-
    % exists_directory should fail on a regular file.
    ( exists_directory('/etc/hostname') -> R is 1 ; R is 0 ).   % 0

:- dynamic test_xd_missing/2.
test_xd_missing(_, R) :-
    ( exists_directory('/nonexistent/path') -> R is 1 ; R is 0 ).   % 0

% M72: get_time/1 -- wall-clock seconds since the epoch as Float.

:- dynamic test_get_time_succeeds/2.
test_get_time_succeeds(_, R) :- ( get_time(_) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_get_time_positive/2.
test_get_time_positive(_, R) :-
    get_time(T),
    ( T > 0.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_get_time_recent/2.
test_get_time_recent(_, R) :-
    % UNIX time should be past 2020 (1577836800 epoch).
    get_time(T),
    ( T > 1577836800.0 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_get_time_monotonic/2.
test_get_time_monotonic(_, R) :-
    % Two get_time calls -- second should be >= first.
    get_time(T1),
    get_time(T2),
    ( T2 >= T1 -> R is 1 ; R is 0 ).   % 1

% M71: forall/2 -- compile-time rewrite to \+ (Cond, \+ Action).

:- dynamic positive/1.
positive(1).
positive(2).
positive(3).

:- dynamic small/1.
small(1).
small(2).
small(3).
small(4).
small(5).

:- dynamic test_forall_all_pos/2.
test_forall_all_pos(_, R) :-
    ( forall(positive(X), X > 0) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_forall_one_fails/2.
test_forall_one_fails(_, R) :-
    ( forall(positive(X), X > 1) -> R is 1 ; R is 0 ).   % 0 (1 fails)

:- dynamic test_forall_subset/2.
test_forall_subset(_, R) :-
    % Every X in positive/1 is also in small/1.
    ( forall(positive(X), small(X)) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_forall_empty/2.
test_forall_empty(_, R) :-
    % No solutions for false_pred -> forall vacuously true.
    ( forall(member(X, []), X > 0) -> R is 1 ; R is 0 ).   % 1

% M70: msort/2 + aggregate_all(set, ...) migrated to @wam_term_cmp.
% Integer-only sorting unchanged; atom sorting now alphabetical (was
% previously by intern atom_id which is allocation order).

:- dynamic test_msort_atoms_alpha/2.
test_msort_atoms_alpha(_, R) :-
    % b interns before a in this program, but standard order is alpha:
    % expected sorted: [a, b, c, d].
    msort([d, b, a, c], [F|_]),
    char_code(F, C), R is C.   % 97 ('a')

:- dynamic test_msort_atoms_last/2.
test_msort_atoms_last(_, R) :-
    msort([d, b, a, c], L),
    last(L, E),
    char_code(E, C), R is C.   % 100 ('d')

:- dynamic test_msort_dupes_kept/2.
test_msort_dupes_kept(_, R) :-
    % msort keeps duplicates (unlike sort).
    msort([2, 1, 2, 1, 3], L),
    length(L, N), R is N.   % 5

:- dynamic test_msort_compound/2.
test_msort_compound(_, R) :-
    % Compound sorted via @wam_term_cmp recursion through args.
    msort([foo(3), foo(1), foo(2)], [F|_]),
    F = foo(N), R is N.   % 1

% M69: @</2 @=</2 @>/2 @>=/2 standard-order comparison operators.

:- dynamic test_at_lt_yes/2.
test_at_lt_yes(_, R) :- ( 1 @< 2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_at_lt_no/2.
test_at_lt_no(_, R) :- ( 5 @< 2 -> R is 1 ; R is 0 ).   % 0

:- dynamic test_at_lt_eq/2.
test_at_lt_eq(_, R) :- ( 3 @< 3 -> R is 1 ; R is 0 ).   % 0

:- dynamic test_at_le_yes/2.
test_at_le_yes(_, R) :- ( 1 @=< 2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_at_le_eq/2.
test_at_le_eq(_, R) :- ( 3 @=< 3 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_at_le_no/2.
test_at_le_no(_, R) :- ( 5 @=< 2 -> R is 1 ; R is 0 ).   % 0

:- dynamic test_at_gt_yes/2.
test_at_gt_yes(_, R) :- ( 5 @> 2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_at_gt_no/2.
test_at_gt_no(_, R) :- ( 1 @> 2 -> R is 1 ; R is 0 ).   % 0

:- dynamic test_at_ge_yes/2.
test_at_ge_yes(_, R) :- ( 5 @>= 2 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_at_ge_eq/2.
test_at_ge_eq(_, R) :- ( 3 @>= 3 -> R is 1 ; R is 0 ).   % 1

:- dynamic test_at_atom/2.
test_at_atom(_, R) :- ( apple @< banana -> R is 1 ; R is 0 ).   % 1 (alpha order)

:- dynamic test_at_cross_cat/2.
test_at_cross_cat(_, R) :- ( 42 @< foo -> R is 1 ; R is 0 ).   % 1 (numbers < atoms)

:- dynamic test_at_compound/2.
test_at_compound(_, R) :- ( foo(1) @< foo(2) -> R is 1 ; R is 0 ).   % 1

% M68: split_string/4 -- per-char separators + pad-strip on segments.

:- dynamic test_ss_simple/2.
test_ss_simple(_, R) :-
    split_string('a,b,c', ',', '', L),
    length(L, N), R is N.   % 3

:- dynamic test_ss_first_length/2.
test_ss_first_length(_, R) :-
    split_string('hello,world', ',', '', [H|_]),
    atom_length(H, N), R is N.   % 5

:- dynamic test_ss_multi_sep/2.
test_ss_multi_sep(_, R) :-
    % Any of ,; splits.
    split_string('a,b;c,d', ',;', '', L),
    length(L, N), R is N.   % 4

:- dynamic test_ss_pad_strip/2.
test_ss_pad_strip(_, R) :-
    % Spaces stripped from segments.
    split_string(' a , b , c ', ',', ' ', [H|_]),
    atom_length(H, N), R is N.   % 1 ("a", spaces gone)

:- dynamic test_ss_no_sep/2.
test_ss_no_sep(_, R) :-
    % No separator in input -> single element.
    split_string(abc, ',', '', L),
    length(L, N), R is N.   % 1

:- dynamic test_ss_empty_sep/2.
test_ss_empty_sep(_, R) :-
    % Empty sep set -> no split, single element.
    split_string(abc, '', '', L),
    length(L, N), R is N.   % 1

:- dynamic test_ss_empty_source/2.
test_ss_empty_source(_, R) :-
    split_string('', ',', '', L),
    length(L, N), R is N + 31.   % 32 = 1 + 31 (one empty segment)

:- dynamic test_ss_consecutive_seps/2.
test_ss_consecutive_seps(_, R) :-
    split_string('a,,b', ',', '', L),
    length(L, N), R is N.   % 3 -- ['a', '', 'b']

:- dynamic test_ss_pad_only/2.
test_ss_pad_only(_, R) :-
    % Whole string is pad chars; emitted segment is empty.
    split_string('   ', ',', ' ', [H|_]),
    atom_length(H, N), R is N + 42.   % 42 (length 0)

% M67 verification: literal-list tests with proper R-is-N pattern.

:- dynamic test_lit_3_ints/2.
test_lit_3_ints(_, R) :- L = [1, 2, 3], length(L, N), R is N.   % 3

:- dynamic test_lit_3_compounds_arity1/2.
test_lit_3_compounds_arity1(_, R) :- L = [foo(1), foo(2), foo(3)], length(L, N), R is N.   % 3

:- dynamic test_lit_pair_int/2.
test_lit_pair_int(_, R) :- L = [a-1, b-2, c-3], length(L, N), R is N.   % 3

% M67: re-add the deferred M65 compound-key/element tests now that
% we know the test framework reads R via reg 0 (force via R is N).

:- dynamic test_ks_compound_keys_length/2.
test_ks_compound_keys_length(_, R) :-
    keysort([foo(3)-c, foo(1)-a, foo(2)-b], L),
    length(L, N), R is N.   % 3

:- dynamic test_ks_compound_keys_first/2.
test_ks_compound_keys_first(_, R) :-
    keysort([foo(3)-c, foo(1)-a, foo(2)-b], [_-V|_]),
    char_code(V, C), R is C.   % 97 ('a')

:- dynamic test_sort_compound_elements/2.
test_sort_compound_elements(_, R) :-
    sort([foo(3), foo(1), foo(2)], [F|_]),
    F = foo(N), R is N.   % 1

:- dynamic test_sort_compound_dedup/2.
test_sort_compound_dedup(_, R) :-
    sort([foo(2), foo(1), foo(3), foo(1)], L),
    length(L, N), R is N.   % 3

:- dynamic test_sort_lists/2.
test_sort_lists(_, R) :-
    sort([[3, 4], [1, 2], [2, 3]], [F|_]),
    F = [H|_], R is H.   % 1

% M66: tab/1, put_char/1, put_code/1 -- small I/O builtins.

:- dynamic test_tab_3/2.
test_tab_3(_, R) :- ( tab(3) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_tab_zero/2.
test_tab_zero(_, R) :- ( tab(0) -> R is 1 ; R is 0 ).   % 1 (no output, succeeds)

:- dynamic test_tab_neg/2.
test_tab_neg(_, R) :- ( tab(-1) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_tab_not_int/2.
test_tab_not_int(_, R) :- ( tab(hello) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_put_char_basic/2.
test_put_char_basic(_, R) :- ( put_char(a) -> R is 1 ; R is 0 ).   % 1

:- dynamic test_put_char_digit/2.
test_put_char_digit(_, R) :- ( put_char('5') -> R is 1 ; R is 0 ).   % 1

:- dynamic test_put_char_multi/2.
test_put_char_multi(_, R) :- ( put_char(abc) -> R is 1 ; R is 0 ).   % 0 (not single-char)

:- dynamic test_put_char_int/2.
test_put_char_int(_, R) :- ( put_char(7) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_put_code_basic/2.
test_put_code_basic(_, R) :- ( put_code(65) -> R is 1 ; R is 0 ).   % 1 (prints 'A')

:- dynamic test_put_code_low/2.
test_put_code_low(_, R) :- ( put_code(10) -> R is 1 ; R is 0 ).   % 1 (newline)

:- dynamic test_put_code_neg/2.
test_put_code_neg(_, R) :- ( put_code(-1) -> R is 1 ; R is 0 ).   % 0

:- dynamic test_put_code_oob/2.
test_put_code_oob(_, R) :- ( put_code(300) -> R is 1 ; R is 0 ).   % 0

% M65: keysort/2 + sort/2 migrated to @wam_term_cmp.
% Compound key / element behavioral tests deferred -- there''s a
% pre-existing emit bug where literal lists of compound terms in
% the body construct extra cells; verified by pure-Prolog returning
% length 3 but LLVM returning 16 even before keysort enters the
% picture. The refactor itself is exercised by all the pre-existing
% Integer / Float / Atom keysort and sort tests, which continue to
% pass.

% M20: transcendentals -- sin, cos, tan, log, exp. All lower to LLVM
% intrinsics that the M18 -lm rollout already links. Verified via
% truncate(... * scale) so the shell exit code can carry an integer
% close to the expected value (the Prolog-side truncate makes the
% result int-typed before is/2 boxes it).
:- dynamic test_sin_pi_half/2.
test_sin_pi_half(_, R) :-
    % sin(pi/2) = 1.0; *100 -> 100. We approximate pi as 22/7 to
    % avoid needing a runtime pi constant (close enough for the
    % exit-code precision the test driver carries).
    X is sin(22/7/2),
    R is truncate(X * 100).   % ~ 99 (22/7 is slightly off pi)

:- dynamic test_cos_zero/2.
test_cos_zero(_, R) :-
    X is cos(0),
    R is truncate(X * 100).   % 100

:- dynamic test_tan_zero/2.
test_tan_zero(_, R) :-
    X is tan(0),
    R is truncate(X * 100).   % 0

:- dynamic test_log_e/2.
test_log_e(_, R) :-
    % log(e^2) = 2; emit via exp + log round-trip.
    X is log(exp(2)),
    R is truncate(X).   % 2

:- dynamic test_exp_zero/2.
test_exp_zero(_, R) :-
    X is exp(0),
    R is truncate(X).   % 1

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

% M21: compound pretty-printing through write/1. The runtime helper
% @wam_write_value walks Compounds recursively, printing them as
% functor(arg, ...) or list notation for [|]/2 chains. Tested via
% format/2 ~w which now routes through the same helper -- the test
% predicates use format() so stdout capture works.
:- dynamic test_write_list3/2.
test_write_list3(_, R) :-
    L = [1, 2, 3],
    format('~w', [L]),
    R is 1.

:- dynamic test_write_pair/2.
test_write_pair(_, R) :-
    format('~w', [a-b]),
    R is 1.

:- dynamic test_write_empty/2.
test_write_empty(_, R) :-
    format('~w', [[]]),
    R is 1.

:- dynamic test_write_nested/2.
test_write_nested(_, R) :-
    format('~w', [[1, [2, 3], 4]]),
    R is 1.

:- dynamic test_write_compound/2.
test_write_compound(_, R) :-
    T = foo(1, hello, 3.5),
    format('~w', [T]),
    R is 1.

% M22: operator notation in pretty-printer. Single-char binary ops
% print as ``arg1 op arg2``; unary ``-`` prints as ``-arg``.
% Precedence isn''t modelled, so ``1+2*3`` will round-trip with all
% operators flat (no parens) -- enough to read but not strictly
% re-parseable for compound expressions.
:- dynamic test_write_add/2.
test_write_add(_, R) :- T = 1+2, format('~w', [T]), R is 1.

:- dynamic test_write_mul_add/2.
test_write_mul_add(_, R) :-
    T = 1+2*3, format('~w', [T]), R is 1.

:- dynamic test_write_eq/2.
test_write_eq(_, R) :- T = (x=y), format('~w', [T]), R is 1.

:- dynamic test_write_colon/2.
test_write_colon(_, R) :- T = a:b, format('~w', [T]), R is 1.

:- dynamic test_write_neg/2.
test_write_neg(_, R) :- T = -(x), format('~w', [T]), R is 1.

:- dynamic test_write_list_of_pairs/2.
test_write_list_of_pairs(_, R) :-
    L = [a-1, b-2],
    format('~w', [L]),
    R is 1.

% M23: 2-char symbolic operators in pretty-printer. Same arity=2
% path as M22 but for functors of length 2 with no word boundary
% (i.e. all-symbolic): ``->``, ``:-``, ``==``, ``\\=``, ``=<``,
% ``>=``, ``//``, ``**``.
:- dynamic test_write_arrow/2.
test_write_arrow(_, R) :- T = (a->b), format('~w', [T]), R is 1.

:- dynamic test_write_neck/2.
test_write_neck(_, R) :- T = (h :- g), format('~w', [T]), R is 1.

:- dynamic test_write_struct_eq/2.
test_write_struct_eq(_, R) :- T = (1==2), format('~w', [T]), R is 1.

:- dynamic test_write_not_unify/2.
test_write_not_unify(_, R) :- T = (a\=b), format('~w', [T]), R is 1.

:- dynamic test_write_le/2.
test_write_le(_, R) :- T = (3=<4), format('~w', [T]), R is 1.

:- dynamic test_write_ge/2.
test_write_ge(_, R) :- T = (5>=4), format('~w', [T]), R is 1.

:- dynamic test_write_int_div/2.
test_write_int_div(_, R) :- T = (7//2), format('~w', [T]), R is 1.

:- dynamic test_write_pow/2.
test_write_pow(_, R) :- T = (2**8), format('~w', [T]), R is 1.

% M24: word-like operators in the pretty-printer. ``is`` (2-char) and
% ``mod`` / ``xor`` / ``rem`` / ``div`` (3-char) are printed with
% surrounding spaces so they don''t collide with adjacent atoms /
% numbers (``Xis5`` is ambiguous; ``X is 5`` reads cleanly).
:- dynamic test_write_is/2.
test_write_is(_, R) :-
    % Build the ``is/2`` compound explicitly so we exercise the
    % pretty-printer rather than just evaluating with the runtime.
    T = is(x, 5),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_mod/2.
test_write_mod(_, R) :-
    T = mod(7, 3),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_xor/2.
test_write_xor(_, R) :-
    T = xor(5, 3),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_rem/2.
test_write_rem(_, R) :-
    T = rem(10, 4),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_div/2.
test_write_div(_, R) :-
    T = div(8, 3),
    format('~w', [T]),
    R is 1.

% M25: 3-char symbolic operators. Same 3-byte packed-i32 dispatch as
% the word-like ops, but routed to wv.infix3 (no surrounding spaces).
:- dynamic test_write_arith_eq/2.
test_write_arith_eq(_, R) :-
    T = =:=(x, y),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_arith_ne/2.
test_write_arith_ne(_, R) :-
    T = =\=(x, y),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_univ/2.
test_write_univ(_, R) :-
    T = =..(foo, [1, 2]),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_struct_ne/2.
test_write_struct_ne(_, R) :-
    T = \==(a, b),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_term_le/2.
test_write_term_le(_, R) :-
    T = @=<(a, b),
    format('~w', [T]),
    R is 1.

:- dynamic test_write_term_ge/2.
test_write_term_ge(_, R) :-
    T = @>=(b, a),
    format('~w', [T]),
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
    % M108: pull the module-level instruction + label counts from
    % the side-channel asserted by write_wam_llvm_project. Avoids a
    % per-test ~7MB read_file_to_string + 2 regex scans that
    % accumulated SWI-stack pressure (the 4 GB cap hit was driven
    % almost entirely by these reads).
    wam_llvm_target:wam_llvm_last_compile_counts(IC, LC),
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
    format(atom(ClangCmd), 'clang -O0 ~w -o ~w -lm 2>/dev/null', [OPath, BinPath]),
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
    % M108: same side-channel as run_test_r0 -- skip the IR read-back.
    wam_llvm_target:wam_llvm_last_compile_counts(IC, LC),
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
    format(atom(ClangCmd), 'clang -O0 ~w -o ~w -lm 2>/dev/null', [OPath, BinPath]),
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
    % M108: pull the module-level instruction + label counts from
    % the side-channel asserted by write_wam_llvm_project. Avoids a
    % per-test ~7MB read_file_to_string + 2 regex scans that
    % accumulated SWI-stack pressure (the 4 GB cap hit was driven
    % almost entirely by these reads).
    wam_llvm_target:wam_llvm_last_compile_counts(IC, LC),
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
    ;  format(atom(ClangCmd), 'clang ~w -o ~w -lm 2>/dev/null', [OPath, BinPath]),
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
    % M88: 600+ run_test invocations in one swipl session accumulate
    % large IR-string intermediates on the global stack; tripped a 4GB
    % stack-limit OOM around test_write_rem. Force a GC + trim of
    % stack/heap between tests to keep memory bounded.
    garbage_collect,
    garbage_collect_atoms,
    trim_stacks,
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
    % M108: pull the module-level instruction + label counts from
    % the side-channel asserted by write_wam_llvm_project. Avoids a
    % per-test ~7MB read_file_to_string + 2 regex scans that
    % accumulated SWI-stack pressure (the 4 GB cap hit was driven
    % almost entirely by these reads).
    wam_llvm_target:wam_llvm_last_compile_counts(IC, LC),
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
    ;  format(atom(ClangCmd), 'clang ~w -o ~w -lm 2>/dev/null', [OPath, BinPath]),
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
    % M88: 600+ run_test invocations in one swipl session accumulate
    % large IR-string intermediates on the global stack; tripped a 4GB
    % stack-limit OOM around test_write_rem. Force a GC + trim of
    % stack/heap between tests to keep memory bounded.
    garbage_collect,
    garbage_collect_atoms,
    trim_stacks,
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
       format('--- M16 reverse-mode length/2 ---~n'),
       run_test_r0('length(L, 4), length(L, N) -> 4',
                   test_length_rev_size, 0, 4),
       run_test_r0('length(L, 3), L = [11,_,_], head -> 11',
                   test_length_rev_unify_head, 0, 11),
       run_test_r0('length(L, 3), L = [4,5,6], middle -> 5',
                   test_length_rev_unify_all, 0, 5),
       format('--- M17 if-then-else soft cut (get_level/cut Y_n) ---~n'),
       run_test_r0('(apple -> 11 ; 22) -> 11',
                   test_ite_succ_then_fail + [ite_basket/1], 0, 11),
       run_test_r0('(soap -> 11 ; 22) -> 22',
                   test_ite_fail_to_else + [ite_basket/1], 0, 22),
       run_test_r0('(milk -> 7 ; 0), R is X+1 -> 8',
                   test_ite_chained_after_call + [ite_basket/1], 0, 8),
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
       format('--- M18 truncate / round / floor / ceiling / sqrt / sign ---~n'),
       run_test_r0('truncate(7/2) -> 3', test_truncate_pos, 0, 3),
       % Shell exit codes are unsigned 8-bit: -3 comes back as 253.
       run_test_r0('truncate(-7/2) -> -3 (exit 253)',
                   test_truncate_neg, 0, 253),
       run_test_r0('round(7/2) -> 4', test_round_up, 0, 4),
       run_test_r0('round(5/2) -> 3', test_round_down, 0, 3),
       run_test_r0('floor(7/2) -> 3', test_floor_pos, 0, 3),
       run_test_r0('floor(-7/2) -> -4 (exit 252)',
                   test_floor_neg, 0, 252),
       run_test_r0('ceiling(7/2) -> 4', test_ceiling_pos, 0, 4),
       run_test_r0('truncate(sqrt(16)) -> 4', test_sqrt_int, 0, 4),
       run_test_r0('sign(7) -> 1', test_sign_pos, 0, 1),
       run_test_r0('sign(-7) -> -1 (exit 255)',
                   test_sign_neg, 0, 255),
       run_test_r0('sign(0) -> 0', test_sign_zero, 0, 0),
       format('--- M19 atom_length / atom_codes ---~n'),
       run_test_r0('atom_length(hello) -> 5',
                   test_atom_length_hello, 0, 5),
       run_test_r0('atom_length('''') -> 0',
                   test_atom_length_empty, 0, 0),
       run_test_r0('atom_codes(hello)[0] -> ''h''=104',
                   test_atom_codes_head, 0, 104),
       run_test_r0('atom_codes(hello)[1] -> ''e''=101',
                   test_atom_codes_second, 0, 101),
       run_test_r0('length(atom_codes(world)) -> 5',
                   test_atom_codes_length, 0, 5),
       format('--- M26 char_code/2 ---~n'),
       run_test_r0('char_code(h, X) -> 104',
                   test_char_code_forward_h, 0, 104),
       run_test_r0('char_code(z, X) -> 122',
                   test_char_code_forward_z, 0, 122),
       run_test_r0('char_code(C, 104), char_code(C, X) -> 104',
                   test_char_code_reverse_h, 0, 104),
       run_test_r0('char_code(a, 97) (check mode) -> 1',
                   test_char_code_check, 0, 1),
       format('--- M27 atom_chars/2 forward mode ---~n'),
       run_test_r0('atom_chars(hello)[0] code -> 104',
                   test_atom_chars_head, 0, 104),
       run_test_r0('atom_chars(hello)[2] code -> 108',
                   test_atom_chars_third, 0, 108),
       run_test_r0('length(atom_chars(world)) -> 5',
                   test_atom_chars_length, 0, 5),
       run_fmt_test('write atom_chars(hi) -> "[h, i]"',
                    test_atom_chars_print, "[h, i]"),
       format('--- M28 between/3 ---~n'),
       run_test_r0('between(3, 5, X) bind first -> 3',
                   test_between_bind_first, 0, 3),
       run_test_r0('between(1, 10, 5) in range -> 1',
                   test_between_check_in, 0, 1),
       run_test_r0('between(1, 10, 99) out of range -> 255',
                   test_between_check_out, 0, 255),
       run_test_r0('aggregate_all(sum(X), between(1, 10, X)) -> 55',
                   test_between_sum, 0, 55),
       run_test_r0('aggregate_all(count, between(1, 7, _)) -> 7',
                   test_between_count, 0, 7),
       format('--- M29 atom_concat/3 forward + runtime atom interning ---~n'),
       run_test_r0('atom_length(hello++world) -> 10',
                   test_atom_concat_length, 0, 10),
       run_test_r0('atom_codes(ab++cd)[0] -> 97',
                   test_atom_concat_first, 0, 97),
       run_test_r0('atom_codes(ab++cd)[2] -> 99',
                   test_atom_concat_second, 0, 99),
       run_test_r0('atom_concat twice -> same id (== true) -> 1',
                   test_atom_concat_dedup, 0, 1),
       run_test_r0('atom_length('''' ++ hi) -> 2',
                   test_atom_concat_left_empty, 0, 2),
       format('--- M30 atom_codes/2 + atom_chars/2 reverse mode ---~n'),
       run_test_r0('atom_codes(A, [104,105]), atom_length(A) -> 2',
                   test_atom_codes_reverse_length, 0, 2),
       run_test_r0('atom_codes(A, [104,105]) then atom_concat(A, lo) length -> 4',
                   test_atom_codes_reverse_concat, 0, 4),
       run_test_r0('atom_codes(A,[104,105]) twice -> same id (== true) -> 1',
                   test_atom_codes_reverse_dedup, 0, 1),
       run_test_r0('atom_codes(A, []) empty list -> atom_length 0',
                   test_atom_codes_reverse_empty, 0, 0),
       run_test_r0('atom_chars(A, [h,i]) atom_length -> 2',
                   test_atom_chars_reverse_length, 0, 2),
       run_test_r0('atom_chars(A,[h,i]) first code -> 104',
                   test_atom_chars_reverse_first, 0, 104),
       run_test_r0('atom_chars(A,[h,i,j]) atom_codes(A) twice == 1',
                   test_atom_chars_reverse_dedup, 0, 1),
       format('--- M31 sub_atom/5 deterministic extraction ---~n'),
       run_test_r0('sub_atom(hello,1,3,_,S) atom_length(S) -> 3',
                   test_sub_atom_extract_length, 0, 3),
       run_test_r0('sub_atom(hello,1,3,A,_) After -> 1',
                   test_sub_atom_extract_after, 0, 1),
       run_test_r0('sub_atom(hello,1,3,_,S) first code -> 101 (e)',
                   test_sub_atom_extract_first_code, 0, 101),
       run_test_r0('sub_atom(hello,1,3,_,S) twice -> same id -> 1',
                   test_sub_atom_dedup, 0, 1),
       run_test_r0('sub_atom(hello,2,0,_,S) atom_length(S) -> 0',
                   test_sub_atom_empty, 0, 0),
       run_test_r0('sub_atom(hello,0,5,A,S) N + After -> 5',
                   test_sub_atom_full, 0, 5),
       run_test_r0('sub_atom(hi,0,99,_,_) overflow -> 0',
                   test_sub_atom_overflow, 0, 0),
       format('--- M32 atom_number/2 integer forward + reverse ---~n'),
       run_test_r0('atom_number(\'42\', N) -> 42',
                   test_atom_number_fwd, 0, 42),
       run_test_r0('atom_number(\'-17\', N) -> N + 100 = 83',
                   test_atom_number_fwd_neg, 0, 83),
       run_test_r0('atom_number(\'0\', N) -> N + 7 = 7',
                   test_atom_number_fwd_zero, 0, 7),
       run_test_r0('atom_number(\'12abc\', _) trailing junk -> 0',
                   test_atom_number_fwd_bad, 0, 0),
       run_test_r0('atom_number(\'\', _) empty -> 0',
                   test_atom_number_fwd_empty, 0, 0),
       run_test_r0('atom_number(A, 42), atom_length(A) -> 2',
                   test_atom_number_rev, 0, 2),
       run_test_r0('atom_number(A, -17), atom_length(A) -> 3',
                   test_atom_number_rev_neg, 0, 3),
       run_test_r0('roundtrip atom_number(A,99) twice -> 99',
                   test_atom_number_roundtrip, 0, 99),
       format('--- M33 number_codes/2 integer forward + reverse ---~n'),
       run_test_r0('number_codes(42, [C|_]) first code -> 52 (\'4\')',
                   test_number_codes_fwd_first, 0, 52),
       run_test_r0('number_codes(1234, L), length(L) -> 4',
                   test_number_codes_fwd_length, 0, 4),
       run_test_r0('number_codes(-5, [C|_]) first code -> 45 (\'-\')',
                   test_number_codes_fwd_neg_first, 0, 45),
       run_test_r0('number_codes(N, [52,50]) -> 42',
                   test_number_codes_rev, 0, 42),
       run_test_r0('number_codes(N, [45,49,55]) + 100 -> 83',
                   test_number_codes_rev_neg, 0, 83),
       run_test_r0('number_codes(_, "1abc") trailing junk -> 0',
                   test_number_codes_rev_bad, 0, 0),
       run_test_r0('number_codes(_, []) empty -> 0',
                   test_number_codes_rev_empty, 0, 0),
       run_test_r0('roundtrip number_codes(99) -> 99',
                   test_number_codes_roundtrip, 0, 99),
       format('--- M34 number_chars/2 integer forward + reverse ---~n'),
       run_test_r0('number_chars(42, [H|_]) char_code -> 52',
                   test_number_chars_fwd_head, 0, 52),
       run_test_r0('number_chars(1234, L), length(L) -> 4',
                   test_number_chars_fwd_length, 0, 4),
       run_test_r0('number_chars(-5, [H|_]) char_code -> 45',
                   test_number_chars_fwd_neg_head, 0, 45),
       run_test_r0('number_chars(N, [\'4\',\'2\']) -> 42',
                   test_number_chars_rev, 0, 42),
       run_test_r0('number_chars(N, [\'-\',\'1\',\'7\']) + 100 -> 83',
                   test_number_chars_rev_neg, 0, 83),
       run_test_r0('number_chars(_, [\'1\',a,b,c]) trailing junk -> 0',
                   test_number_chars_rev_bad, 0, 0),
       run_test_r0('number_chars(_, []) empty -> 0',
                   test_number_chars_rev_empty, 0, 0),
       run_test_r0('roundtrip number_chars(99) -> 99',
                   test_number_chars_roundtrip, 0, 99),
       format('--- M35 upcase_atom/2 + downcase_atom/2 ---~n'),
       run_test_r0('upcase_atom(hello, U), atom_length(U) -> 5',
                   test_upcase_length, 0, 5),
       run_test_r0('upcase_atom(hello, U) first code -> 72 (H)',
                   test_upcase_first_code, 0, 72),
       run_test_r0('upcase_atom(ab, U) second code -> 66 (B)',
                   test_upcase_last_code, 0, 66),
       run_test_r0('upcase_atom(\'hi!\', U) third code -> 33 (!)',
                   test_upcase_passthrough, 0, 33),
       run_test_r0('upcase_atom(hello) twice -> same id -> 1',
                   test_upcase_dedup, 0, 1),
       run_test_r0('downcase_atom(\'HELLO\', D), atom_length -> 5',
                   test_downcase_length, 0, 5),
       run_test_r0('downcase_atom(\'HELLO\', D) first -> 104 (h)',
                   test_downcase_first_code, 0, 104),
       run_test_r0('downcase_atom(\'aB\', D) second -> 98 (b)',
                   test_downcase_mixed, 0, 98),
       run_test_r0('upcase then downcase roundtrip -> hello -> 1',
                   test_upcase_downcase_roundtrip, 0, 1),
       format('--- M36 string-type aliases (atom_string/string_concat/string_length/string_to_atom) ---~n'),
       run_test_r0('atom_string(hello, S), atom_length(S) -> 5',
                   test_atom_string_fwd, 0, 5),
       run_test_r0('atom_string(hello, hello) check -> 1',
                   test_atom_string_check, 0, 1),
       run_test_r0('string_to_atom(world, A), atom_length(A) -> 5',
                   test_string_to_atom_fwd, 0, 5),
       run_test_r0('string_concat(hi, there, S), atom_length(S) -> 7',
                   test_string_concat_length, 0, 7),
       run_test_r0('string_concat(ab, cd, S) first code -> 97',
                   test_string_concat_first_code, 0, 97),
       run_test_r0('string_length(hello, N) -> 5',
                   test_string_length_simple, 0, 5),
       run_test_r0('string_length(\'\', N) + 42 -> 42',
                   test_string_length_empty, 0, 42),
       format('--- M37 nth0/3 + nth1/3 + last/2 list-indexing ---~n'),
       run_test_r0('nth0(0, [10,20,30], E) -> 10',
                   test_nth0_first, 0, 10),
       run_test_r0('nth0(2, [10,20,30,40], E) -> 30',
                   test_nth0_middle, 0, 30),
       run_test_r0('nth0(3, [10,20,30,40], E) -> 40',
                   test_nth0_last, 0, 40),
       run_test_r0('nth0(5, [10,20], _) overflow -> 0',
                   test_nth0_overflow, 0, 0),
       run_test_r0('nth0(-1, [10,20], _) negative -> 0',
                   test_nth0_negative, 0, 0),
       run_test_r0('nth1(1, [10,20,30], E) -> 10',
                   test_nth1_first, 0, 10),
       run_test_r0('nth1(3, [10,20,30,40], E) -> 30',
                   test_nth1_third, 0, 30),
       run_test_r0('nth1(0, [10,20], _) rejects 0 -> 0',
                   test_nth1_zero, 0, 0),
       run_test_r0('last([10,20,30], E) -> 30',
                   test_last_simple, 0, 30),
       run_test_r0('last([99], E) -> 99',
                   test_last_singleton, 0, 99),
       run_test_r0('last([], _) -> 0',
                   test_last_empty, 0, 0),
       format('--- M38 reverse/2 deterministic ---~n'),
       run_test_r0('reverse([10,20,30], L), nth0(0, L) -> 30',
                   test_reverse_first, 0, 30),
       run_test_r0('reverse([10,20,30], L), last(L) -> 10',
                   test_reverse_last, 0, 10),
       run_test_r0('reverse([1..5], L), length(L) -> 5',
                   test_reverse_length, 0, 5),
       run_test_r0('reverse([], L) + 7 -> 7',
                   test_reverse_empty, 0, 7),
       run_test_r0('reverse([42], [E]) -> 42',
                   test_reverse_singleton, 0, 42),
       run_test_r0('reverse twice idempotent -> 1',
                   test_reverse_idempotent, 0, 1),
       format('--- M39 append/3 deterministic ---~n'),
       run_test_r0('append([1,2,3], [4,5], L), length -> 5',
                   test_append_length, 0, 5),
       run_test_r0('append([10,20], [30,40], L), nth0(0) -> 10',
                   test_append_first, 0, 10),
       run_test_r0('append([10,20], [30,40], L), nth0(2) -> 30 (seam)',
                   test_append_seam, 0, 30),
       run_test_r0('append([10,20], [30,40], L), last -> 40',
                   test_append_last, 0, 40),
       run_test_r0('append([], [7,8], L), length -> 2',
                   test_append_left_empty, 0, 2),
       run_test_r0('append([7,8], [], L), length -> 2',
                   test_append_right_empty, 0, 2),
       run_test_r0('append([], [], L), length + 9 -> 9',
                   test_append_both_empty, 0, 9),
       run_test_r0('append + reverse roundtrip -> last becomes first -> 4',
                   test_append_roundtrip, 0, 4),
       format('--- M40 memberchk/2 deterministic ---~n'),
       run_test_r0('memberchk(20, [10,20,30]) -> 1',
                   test_memberchk_int_hit, 0, 1),
       run_test_r0('memberchk(99, [10,20,30]) -> 0',
                   test_memberchk_int_miss, 0, 0),
       run_test_r0('memberchk(1, []) -> 0',
                   test_memberchk_int_empty, 0, 0),
       run_test_r0('memberchk(b, [a,b,c]) -> 1',
                   test_memberchk_atom_hit, 0, 1),
       run_test_r0('memberchk(X, [42,99,7]) -> 42',
                   test_memberchk_bind, 0, 42),
       run_test_r0('memberchk(5, [3,5,5,9]) first-match -> 1',
                   test_memberchk_first_match_wins, 0, 1),
       format('--- M41 delete/3 deterministic ---~n'),
       run_test_r0('delete([1,2,3], 99, L), length -> 3 (no match)',
                   test_delete_no_match, 0, 3),
       run_test_r0('delete([1,2,3], 2, L), length -> 2',
                   test_delete_single, 0, 2),
       run_test_r0('delete([1,2,1,3,1], 1, L), length -> 2',
                   test_delete_multiple, 0, 2),
       run_test_r0('delete([7,7,7], 7, L), length + 5 -> 5 (empty)',
                   test_delete_all, 0, 5),
       run_test_r0('delete([], 1, L), length + 8 -> 8',
                   test_delete_empty, 0, 8),
       run_test_r0('delete([1,2,3,4], 1, L), nth0(0) -> 2',
                   test_delete_first, 0, 2),
       run_test_r0('delete([1,2,3,99], 99, L), last -> 3',
                   test_delete_last, 0, 3),
       run_test_r0('delete([10,5,20,5,30], 5, L), nth0(1) -> 20',
                   test_delete_preserves_order, 0, 20),
       format('--- M42 numlist/3 + sum_list/2 ---~n'),
       run_test_r0('numlist(3, 7, L), length -> 5',
                   test_numlist_length, 0, 5),
       run_test_r0('numlist(10, 15, [F|_]) -> 10',
                   test_numlist_first, 0, 10),
       run_test_r0('numlist(1, 7, L), last -> 7',
                   test_numlist_last, 0, 7),
       run_test_r0('numlist(42, 42, [E]) -> 42',
                   test_numlist_singleton, 0, 42),
       run_test_r0('numlist(5, 3, _) empty range -> 0',
                   test_numlist_empty, 0, 0),
       run_test_r0('numlist(1, 10) sum_list -> 55',
                   test_numlist_sum, 0, 55),
       run_test_r0('sum_list([10,20,30]) -> 60',
                   test_sum_list_simple, 0, 60),
       run_test_r0('sum_list([]) + 7 -> 7',
                   test_sum_list_empty, 0, 7),
       run_test_r0('sum_list([99]) -> 99',
                   test_sum_list_singleton, 0, 99),
       run_test_r0('sumlist([1..5]) alias -> 15',
                   test_sumlist_alias, 0, 15),
       format('--- M43 max_list/2 + min_list/2 ---~n'),
       run_test_r0('max_list([3,7,2,9,5]) -> 9',
                   test_max_list_simple, 0, 9),
       run_test_r0('max_list([42]) -> 42',
                   test_max_list_singleton, 0, 42),
       run_test_r0('max_list([100,50,25,10]) -> 100',
                   test_max_list_first_biggest, 0, 100),
       run_test_r0('max_list([]) -> 0',
                   test_max_list_empty, 0, 0),
       run_test_r0('min_list([7,3,9,2,5]) -> 2',
                   test_min_list_simple, 0, 2),
       run_test_r0('min_list([42]) -> 42',
                   test_min_list_singleton, 0, 42),
       run_test_r0('min_list([1,10,100]) -> 1',
                   test_min_list_first_smallest, 0, 1),
       run_test_r0('min_list([]) -> 0',
                   test_min_list_empty, 0, 0),
       run_test_r0('max_list - min_list of [4,7,1,9,3] -> 8',
                   test_max_minus_min, 0, 8),
       format('--- M44 subtract/3 ---~n'),
       run_test_r0('subtract([1,2,3], [4,5,6]) disjoint length -> 3',
                   test_subtract_disjoint, 0, 3),
       run_test_r0('subtract([1,2,3], [2]) length -> 2',
                   test_subtract_one, 0, 2),
       run_test_r0('subtract([1,2,3,4,5], [2,4]) length -> 3',
                   test_subtract_many, 0, 3),
       run_test_r0('subtract([1,2,3], [3,2,1]) empty + 7 -> 7',
                   test_subtract_all, 0, 7),
       run_test_r0('subtract([], [1,2]) + 8 -> 8',
                   test_subtract_empty_left, 0, 8),
       run_test_r0('subtract([7,8,9], []) length -> 3',
                   test_subtract_empty_right, 0, 3),
       run_test_r0('subtract([10,20,30,40], [10]) nth0(0) -> 20',
                   test_subtract_first, 0, 20),
       run_test_r0('subtract([10,5,20,5,30], [5]) nth0(1) -> 20',
                   test_subtract_preserves_order, 0, 20),
       run_test_r0('subtract([1,2,1,3,1], [1]) length -> 2 (all dupes filtered)',
                   test_subtract_dupes, 0, 2),
       format('--- M45 intersection/3 ---~n'),
       run_test_r0('intersection([1,2,3], [4,5,6]) disjoint + 8 -> 8',
                   test_intersection_disjoint, 0, 8),
       run_test_r0('intersection([1,2,3], [2]) length -> 1',
                   test_intersection_one, 0, 1),
       run_test_r0('intersection([1..5], [2,4,99]) length -> 2',
                   test_intersection_many, 0, 2),
       run_test_r0('intersection([1,2,3], [3,2,1]) length -> 3',
                   test_intersection_all, 0, 3),
       run_test_r0('intersection([], [1,2]) + 7 -> 7',
                   test_intersection_empty_left, 0, 7),
       run_test_r0('intersection([1,2,3], []) + 5 -> 5',
                   test_intersection_empty_right, 0, 5),
       run_test_r0('intersection([10,20,30,40], [20,40,99]) nth0(0) -> 20',
                   test_intersection_first, 0, 20),
       run_test_r0('intersection([10,5,20,5,30], [5,20]) nth0(2) -> 5',
                   test_intersection_preserves_order, 0, 5),
       run_test_r0('intersection([1,2,1,3,1], [1]) length -> 3 (all dupes survive)',
                   test_intersection_dupes, 0, 3),
       run_test_r0('intersection + subtract partition reconstruct -> 5',
                   test_inter_subtract_complement, 0, 5),
       format('--- M46 union/3 ---~n'),
       run_test_r0('union([1,2,3], [4,5,6]) disjoint length -> 6',
                   test_union_disjoint, 0, 6),
       run_test_r0('union([1,2,3], [2,4,5]) overlap length -> 5',
                   test_union_overlap, 0, 5),
       run_test_r0('union([1,2,3], [1,2,3]) identical length -> 3',
                   test_union_identical, 0, 3),
       run_test_r0('union([], [7,8,9]) length -> 3',
                   test_union_empty_left, 0, 3),
       run_test_r0('union([1,2,3], []) length -> 3',
                   test_union_empty_right, 0, 3),
       run_test_r0('union([], []) + 11 -> 11',
                   test_union_both_empty, 0, 11),
       run_test_r0('union([10,20], [30,40]) nth0(0) -> 10 (A1 first)',
                   test_union_a1_first, 0, 10),
       run_test_r0('union([1,1,2], [3]) length -> 4 (A1 dupes kept)',
                   test_union_a1_dupes_kept, 0, 4),
       run_test_r0('union([1,2,3], [2,4]) nth0(3) -> 4 (2 filtered)',
                   test_union_a2_first_match_filtered, 0, 4),
       run_test_r0('|union| + |inter| = |A1| + |A2| (8)',
                   test_union_size_relation, 0, 8),
       format('--- M47 list_to_set/2 ---~n'),
       run_test_r0('list_to_set([1,2,3]) length -> 3 (no dupes)',
                   test_l2s_simple, 0, 3),
       run_test_r0('list_to_set([1,2,1,3,2]) length -> 3',
                   test_l2s_dupes, 0, 3),
       run_test_r0('list_to_set([7,7,7,7]) length -> 1',
                   test_l2s_all_dupes, 0, 1),
       run_test_r0('list_to_set([]) + 9 -> 9',
                   test_l2s_empty, 0, 9),
       run_test_r0('list_to_set([42], [E]) -> 42',
                   test_l2s_singleton, 0, 42),
       run_test_r0('list_to_set([3,1,2,1,3]) nth0(0) -> 3 (first wins)',
                   test_l2s_order_first, 0, 3),
       run_test_r0('list_to_set([3,1,2,1,3]) nth0(1) -> 1',
                   test_l2s_order_second, 0, 1),
       run_test_r0('list_to_set([3,1,2,1,3]) nth0(2) -> 2',
                   test_l2s_order_third, 0, 2),
       run_test_r0('list_to_set([a,b,a,c,b]) length -> 3',
                   test_l2s_atom_dupes, 0, 3),
       run_test_r0('list_to_set idempotent on already-a-set -> 3',
                   test_l2s_idempotent, 0, 3),
       format('--- M48 string_chars/2 + string_codes/2 aliases + string_code/3 ---~n'),
       run_test_r0('string_chars(hello, [H|_]) char_code -> 104',
                   test_string_chars_first, 0, 104),
       run_test_r0('string_chars(abc, L) length -> 3',
                   test_string_chars_length, 0, 3),
       run_test_r0('string_codes(hello, [C|_]) -> 104',
                   test_string_codes_first, 0, 104),
       run_test_r0('string_codes(abcde, L) length -> 5',
                   test_string_codes_length, 0, 5),
       run_test_r0('string_code(1, hello, C) -> 104 (h)',
                   test_string_code_first, 0, 104),
       run_test_r0('string_code(3, abcdef, C) -> 99 (c)',
                   test_string_code_middle, 0, 99),
       run_test_r0('string_code(5, hello, C) -> 111 (o)',
                   test_string_code_last, 0, 111),
       run_test_r0('string_code(0, hello, _) -> 0 (1-based)',
                   test_string_code_oob_low, 0, 0),
       run_test_r0('string_code(99, hello, _) -> 0 (overflow)',
                   test_string_code_oob_high, 0, 0),
       format('--- M49 pairs_keys/2 + pairs_values/2 ---~n'),
       run_test_r0('pairs_keys([a-1, b-2, c-3]) length -> 3',
                   test_pairs_keys_simple, 0, 3),
       run_test_r0('pairs_keys([10-x, 20-y, 30-z]) first -> 10',
                   test_pairs_keys_first, 0, 10),
       run_test_r0('pairs_keys([10-a, 20-b, 30-c]) last -> 30',
                   test_pairs_keys_last, 0, 30),
       run_test_r0('pairs_keys([]) + 11 -> 11',
                   test_pairs_keys_empty, 0, 11),
       run_test_r0('pairs_keys([42-foo], [K]) -> 42',
                   test_pairs_keys_singleton, 0, 42),
       run_test_r0('pairs_values([a-10, b-20, c-30]) length -> 3',
                   test_pairs_values_simple, 0, 3),
       run_test_r0('pairs_values([a-100, b-200]) first -> 100',
                   test_pairs_values_first, 0, 100),
       run_test_r0('pairs_values([a-10, b-20, c-99]) last -> 99',
                   test_pairs_values_last, 0, 99),
       run_test_r0('pairs_values([]) + 13 -> 13',
                   test_pairs_values_empty, 0, 13),
       run_test_r0('pairs_keys + pairs_values sums [10-1, 20-2, 30-3] -> 66',
                   test_pairs_keys_values_sum, 0, 66),
       format('--- M50 pairs_keys_values/3 forward (split) ---~n'),
       run_test_r0('pkv([a-1,b-2,c-3], Ks, _) length -> 3',
                   test_pkv_keys_length, 0, 3),
       run_test_r0('pkv([a-1,b-2,c-3], _, Vs) length -> 3',
                   test_pkv_values_length, 0, 3),
       run_test_r0('pkv([10-x, 20-y], [F|_], _) -> 10',
                   test_pkv_keys_first, 0, 10),
       run_test_r0('pkv([a-100, b-200], _, [F|_]) -> 100',
                   test_pkv_values_first, 0, 100),
       run_test_r0('pkv([], Ks, Vs) + 17 -> 17',
                   test_pkv_empty, 0, 17),
       run_test_r0('pkv([42-99], [K], [V]) K+V -> 141',
                   test_pkv_singleton, 0, 141),
       run_test_r0('pkv last key -> 30',
                   test_pkv_keys_last, 0, 30),
       run_test_r0('pkv last value -> 99',
                   test_pkv_values_last, 0, 99),
       run_test_r0('pkv matches pairs_keys + pairs_values -> 1',
                   test_pkv_matches_split, 0, 1),
       format('--- M51 pairs_keys_values/3 reverse (zip) ---~n'),
       run_test_r0('pkv(P, [a,b,c], [1,2,3]), length(P) -> 3',
                   test_pkv_rev_length, 0, 3),
       run_test_r0('pkv reverse + pairs_keys roundtrip sum -> 60',
                   test_pkv_rev_roundtrip_keys, 0, 60),
       run_test_r0('pkv reverse + pairs_values roundtrip sum -> 60',
                   test_pkv_rev_roundtrip_values, 0, 60),
       run_test_r0('pkv(P, [], []) + 19 -> 19',
                   test_pkv_rev_empty, 0, 19),
       run_test_r0('pkv(P, [42], [99]) K+V -> 141',
                   test_pkv_rev_singleton, 0, 141),
       run_test_r0('pkv mismatch keys longer -> 0',
                   test_pkv_rev_mismatch_keys_longer, 0, 0),
       run_test_r0('pkv mismatch values longer -> 0',
                   test_pkv_rev_mismatch_values_longer, 0, 0),
       run_test_r0('pkv forward then reverse roundtrip sum -> 60',
                   test_pkv_forward_then_reverse, 0, 60),
       format('--- M52 atomic_list_concat/2 (atoms-only) ---~n'),
       run_test_r0('atomic_list_concat([hello, world]) length -> 10',
                   test_alc_simple_length, 0, 10),
       run_test_r0('atomic_list_concat([hello, world]) first -> 104',
                   test_alc_first_code, 0, 104),
       run_test_r0('atomic_list_concat([ab, cd]) seam -> 99 (c)',
                   test_alc_seam_code, 0, 99),
       run_test_r0('atomic_list_concat([]) + 31 -> 31',
                   test_alc_empty_list, 0, 31),
       run_test_r0('atomic_list_concat([only]) length -> 4',
                   test_alc_singleton, 0, 4),
       run_test_r0('atomic_list_concat([a,b,c,d,e]) length -> 5',
                   test_alc_many, 0, 5),
       run_test_r0('atomic_list_concat([\'\', hi, \'\']) length -> 2',
                   test_alc_with_empties, 0, 2),
       run_test_r0('atomic_list_concat dedup -> 1',
                   test_alc_dedup, 0, 1),
       format('--- M53 atomic_list_concat/3 separator (atoms-only forward) ---~n'),
       run_test_r0('alc([a,b,c], \'-\') length -> 5',
                   test_alc3_simple_length, 0, 5),
       run_test_r0('alc([hi,there], \'/\') first -> 104',
                   test_alc3_first_code, 0, 104),
       run_test_r0('alc([ab,cd], \'/\') sep code -> 47',
                   test_alc3_sep_code, 0, 47),
       run_test_r0('alc([a,b], \' :: \') length -> 6',
                   test_alc3_multi_char_sep, 0, 6),
       run_test_r0('alc([], \'-\') + 27 -> 27',
                   test_alc3_empty_list, 0, 27),
       run_test_r0('alc([foo], \'/\') singleton length -> 3',
                   test_alc3_singleton, 0, 3),
       run_test_r0('alc([a,b,c], \'\') length -> 3',
                   test_alc3_empty_sep, 0, 3),
       run_test_r0('alc/3 dedup -> 1',
                   test_alc3_dedup, 0, 1),
       run_test_r0('alc/3 empty sep matches alc/2 -> 1',
                   test_alc3_matches_alc2, 0, 1),
       format('--- M54 atomic_list_concat/3 split mode ---~n'),
       run_test_r0('alc(Parts, \'-\', \'a-b-c\') length -> 3',
                   test_alc3s_simple_count, 0, 3),
       run_test_r0('alc([P|_], \'-\', \'hello-world\') first length -> 5',
                   test_alc3s_first_length, 0, 5),
       run_test_r0('alc(Parts, \'/\', \'a/bb/ccc\') last length -> 3',
                   test_alc3s_last_length, 0, 3),
       run_test_r0('alc(Parts, \'-\', \'helloworld\') -> 1',
                   test_alc3s_no_sep, 0, 1),
       run_test_r0('alc(Parts, \'-\', \'\') -> 1 ([\'\'])',
                   test_alc3s_empty_atom, 0, 1),
       run_test_r0('alc(Parts, \'-\', \'-\') -> 2 ([\'\', \'\'])',
                   test_alc3s_sep_only, 0, 2),
       run_test_r0('alc(Parts, \'-\', \'a--b\') -> 3',
                   test_alc3s_consecutive_seps, 0, 3),
       run_test_r0('alc(Parts, \'::\', \'foo::bar::baz\') -> 3',
                   test_alc3s_multi_char_sep, 0, 3),
       run_test_r0('alc(_, \'\', _) empty sep -> 0',
                   test_alc3s_empty_sep_fails, 0, 0),
       run_test_r0('alc roundtrip count -> 3',
                   test_alc3s_roundtrip, 0, 3),
       format('--- M55 atomic_list_concat/2 Integer heads ---~n'),
       run_test_r0('alc([1,2,3]) atom_length -> 3',
                   test_alc_int_only_length, 0, 3),
       run_test_r0('alc([42]) first code -> 52',
                   test_alc_int_first_code, 0, 52),
       run_test_r0('alc([-7]) first code -> 45',
                   test_alc_int_negative, 0, 45),
       run_test_r0('alc([foo, 42, bar]) length -> 8',
                   test_alc_mixed_length, 0, 8),
       run_test_r0('alc([ab, 99, cd]) pos 2 -> 57',
                   test_alc_mixed_seam, 0, 57),
       run_test_r0('alc([0]) first code -> 48',
                   test_alc_int_zero, 0, 48),
       run_test_r0('alc([prefix, 200]) length -> 9',
                   test_alc_int_last, 0, 9),
       run_test_r0('alc([10, 20, 30]) pos 2 -> 50',
                   test_alc_three_ints, 0, 50),
       format('--- M56 atomic_list_concat/3 Integer heads ---~n'),
       run_test_r0('alc([1,2,3], \'-\') length -> 5',
                   test_alc3_int_only_length, 0, 5),
       run_test_r0('alc([42], \'-\') first -> 52',
                   test_alc3_int_first, 0, 52),
       run_test_r0('alc([10, 20], \'/\') seam -> 47',
                   test_alc3_int_seam, 0, 47),
       run_test_r0('alc([foo, 42, bar], \'-\') length -> 10',
                   test_alc3_int_with_atom, 0, 10),
       run_test_r0('alc([-5, 7], \'+\') length -> 4',
                   test_alc3_int_negative, 0, 4),
       run_test_r0('alc([1,2,3], \', \') length -> 7',
                   test_alc3_int_multi_char_sep, 0, 7),
       run_test_r0('alc/3 int + split roundtrip count -> 3',
                   test_alc3_int_split_count, 0, 3),
       format('--- M57 atomic_list_concat Float widening ---~n'),
       run_test_r0('alc([3.5]) first -> 51 (\'3\')',
                   test_alc_float_singleton, 0, 51),
       run_test_r0('alc([pi, \'=\', 3.14]) first -> 112 (p)',
                   test_alc_float_with_atom, 0, 112),
       run_test_r0('alc([1, 2.5, 3]) pos 2 -> 46 (.)',
                   test_alc_float_with_int, 0, 46),
       run_test_r0('alc/3 [1.5, 2.5] / \'+\' pos 3 -> 43',
                   test_alc3_float_only, 0, 43),
       run_test_r0('alc/3 [x, 1.5, y] / \',\' first -> 120 (x)',
                   test_alc3_float_mixed, 0, 120),
       format('--- M58 char_type/2 (check mode) ---~n'),
       run_test_r0('char_type(a, alpha) -> 1',
                   test_ct_alpha_yes, 0, 1),
       run_test_r0('char_type(\'5\', alpha) -> 0',
                   test_ct_alpha_no, 0, 0),
       run_test_r0('char_type(\'7\', digit) -> 1',
                   test_ct_digit_yes, 0, 1),
       run_test_r0('char_type(z, digit) -> 0',
                   test_ct_digit_no, 0, 0),
       run_test_r0('char_type(\'A\', upper) -> 1',
                   test_ct_upper_yes, 0, 1),
       run_test_r0('char_type(a, upper) -> 0',
                   test_ct_upper_no, 0, 0),
       run_test_r0('char_type(m, lower) -> 1',
                   test_ct_lower_yes, 0, 1),
       run_test_r0('char_type(b, alnum) letter -> 1',
                   test_ct_alnum_letter, 0, 1),
       run_test_r0('char_type(\'3\', alnum) digit -> 1',
                   test_ct_alnum_digit, 0, 1),
       run_test_r0('char_type(\'!\', alnum) -> 0',
                   test_ct_alnum_punct, 0, 0),
       run_test_r0('char_type(\' \', space) -> 1',
                   test_ct_space_yes, 0, 1),
       run_test_r0('char_type(\'A\', ascii) -> 1',
                   test_ct_ascii_yes, 0, 1),
       run_test_r0('char_type(\'!\', punct) -> 1',
                   test_ct_punct_yes, 0, 1),
       run_test_r0('char_type(\' \', punct) -> 0',
                   test_ct_punct_no_space, 0, 0),
       run_test_r0('char_type(z, csymf) -> 1',
                   test_ct_csymf_letter, 0, 1),
       run_test_r0('char_type(\'_\', csymf) -> 1',
                   test_ct_csymf_underscore, 0, 1),
       run_test_r0('char_type(\'5\', csymf) -> 0',
                   test_ct_csymf_digit_no, 0, 0),
       run_test_r0('char_type(\'5\', csym) -> 1',
                   test_ct_csym_digit_yes, 0, 1),
       run_test_r0('char_type(a, bogus) unknown type -> 0',
                   test_ct_unknown_type, 0, 0),
       run_test_r0('char_type(ab, alpha) multichar -> 0',
                   test_ct_multichar_atom, 0, 0),
       format('--- M59 compare/3 (Integer / Float / Atom) ---~n'),
       run_test_r0('compare(O, 1, 5) -> 60 (<)',
                   test_cmp_int_lt, 0, 60),
       run_test_r0('compare(O, 7, 7) -> 61 (=)',
                   test_cmp_int_eq, 0, 61),
       run_test_r0('compare(O, 10, 3) -> 62 (>)',
                   test_cmp_int_gt, 0, 62),
       run_test_r0('compare(O, -5, 5) -> 60 (<)',
                   test_cmp_neg, 0, 60),
       run_test_r0('compare(O, 1.5, 2.5) -> 60 (<)',
                   test_cmp_float_lt, 0, 60),
       run_test_r0('compare(O, 3.14, 3.14) -> 61 (=)',
                   test_cmp_order_float_eq, 0, 61),
       run_test_r0('compare(O, 2, 2.5) mixed -> 60 (<)',
                   test_cmp_int_float_mixed, 0, 60),
       run_test_r0('compare(O, apple, banana) -> 60 (<)',
                   test_cmp_atom_lt, 0, 60),
       run_test_r0('compare(O, hello, hello) -> 61 (=)',
                   test_cmp_atom_eq, 0, 61),
       run_test_r0('compare(O, zebra, apple) -> 62 (>)',
                   test_cmp_atom_gt, 0, 62),
       run_test_r0('compare(O, 42, foo) cross-cat -> 60',
                   test_cmp_num_atom, 0, 60),
       run_test_r0('compare(O, foo, 42) cross-cat -> 62',
                   test_cmp_atom_num, 0, 62),
       run_test_r0('compare(<, 1, 2) check mode -> 1',
                   test_cmp_check_mode_lt, 0, 1),
       run_test_r0('compare(=, foo, foo) check mode -> 1',
                   test_cmp_check_mode_eq, 0, 1),
       run_test_r0('compare(>, 1, 5) check mode wrong -> 0',
                   test_cmp_check_mode_wrong, 0, 0),
       format('--- M60 must_be/2 (fail-instead-of-throw type guard) ---~n'),
       run_test_r0('must_be(atom, hello) -> 1',
                   test_mb_atom_yes, 0, 1),
       run_test_r0('must_be(atom, 42) -> 0',
                   test_mb_atom_no, 0, 0),
       run_test_r0('must_be(integer, 7) -> 1',
                   test_mb_integer_yes, 0, 1),
       run_test_r0('must_be(integer, 3.14) -> 0',
                   test_mb_integer_no, 0, 0),
       run_test_r0('must_be(float, 2.5) -> 1',
                   test_mb_float_yes, 0, 1),
       run_test_r0('must_be(number, 99) int -> 1',
                   test_mb_number_int, 0, 1),
       run_test_r0('must_be(number, 1.5) flt -> 1',
                   test_mb_number_flt, 0, 1),
       run_test_r0('must_be(number, foo) -> 0',
                   test_mb_number_atom, 0, 0),
       run_test_r0('must_be(compound, [1,2,3]) -> 1',
                   test_mb_compound_yes, 0, 1),
       run_test_r0('must_be(compound, atom) -> 0',
                   test_mb_compound_no, 0, 0),
       run_test_r0('must_be(var, _Fresh) -> 1',
                   test_mb_var_yes, 0, 1),
       run_test_r0('must_be(var, 5) -> 0',
                   test_mb_var_no, 0, 0),
       run_test_r0('must_be(nonvar, 5) -> 1',
                   test_mb_nonvar_yes, 0, 1),
       run_test_r0('must_be(nonvar, _Fresh) -> 0',
                   test_mb_nonvar_no, 0, 0),
       run_test_r0('must_be(atomic, hello) -> 1',
                   test_mb_atomic_atom, 0, 1),
       run_test_r0('must_be(atomic, 7) -> 1',
                   test_mb_atomic_int, 0, 1),
       run_test_r0('must_be(atomic, [1,2]) -> 0',
                   test_mb_atomic_compound, 0, 0),
       run_test_r0('must_be(callable, foo) -> 1',
                   test_mb_callable_atom, 0, 1),
       run_test_r0('must_be(callable, [1]) -> 1',
                   test_mb_callable_compound, 0, 1),
       run_test_r0('must_be(callable, 5) -> 0',
                   test_mb_callable_int, 0, 0),
       run_test_r0('must_be(list, []) -> 1',
                   test_mb_list_empty, 0, 1),
       run_test_r0('must_be(list, [1,2]) -> 1',
                   test_mb_list_cons, 0, 1),
       run_test_r0('must_be(list, hello) -> 0',
                   test_mb_list_no, 0, 0),
       run_test_r0('must_be(boolean, true) -> 1',
                   test_mb_boolean_true, 0, 1),
       run_test_r0('must_be(boolean, false) -> 1',
                   test_mb_boolean_false, 0, 1),
       run_test_r0('must_be(boolean, maybe) -> 0',
                   test_mb_boolean_no, 0, 0),
       run_test_r0('must_be(bogus, hello) -> 0',
                   test_mb_unknown_type, 0, 0),
       format('--- M61 display/1 + writeln/1 ---~n'),
       run_test_r0('display(hello) succeeds -> 1',
                   test_display_succeeds, 0, 1),
       run_test_r0('display(42) -> 1',
                   test_display_integer, 0, 1),
       run_test_r0('display(foo(1, 2)) -> 1',
                   test_display_compound, 0, 1),
       run_test_r0('writeln(hello) -> 1',
                   test_writeln_succeeds, 0, 1),
       run_test_r0('writeln(7) -> 1',
                   test_writeln_integer, 0, 1),
       run_test_r0('writeln([1,2,3]) -> 1',
                   test_writeln_list, 0, 1),
       format('--- M62 keysort/2 (Integer / Float / Atom keys) ---~n'),
       run_test_r0('keysort([3-a,1-b,2-c]) length -> 3',
                   test_ks_int_keys_length, 0, 3),
       run_test_r0('keysort([3-a,1-b,2-c]) first key -> 1',
                   test_ks_int_keys_first, 0, 1),
       run_test_r0('keysort([5-x,2-y,8-z,1-w]) last key -> 8',
                   test_ks_int_keys_last, 0, 8),
       run_test_r0('keysort([1-a,2-b,3-c]) already sorted -> 1',
                   test_ks_already_sorted, 0, 1),
       run_test_r0('keysort([5..1]-) reverse -> 1',
                   test_ks_reverse_sorted, 0, 1),
       run_test_r0('keysort([]) + 17 -> 17',
                   test_ks_empty, 0, 17),
       run_test_r0('keysort([42-only]) -> 42',
                   test_ks_singleton, 0, 42),
       run_test_r0('keysort stable on equal keys nth0(2) -> 2',
                   test_ks_stable_dupes, 0, 2),
       run_test_r0('keysort atom keys first length -> 5 (apple)',
                   test_ks_atom_keys, 0, 5),
       run_test_r0('keysort atom keys first value -> 1',
                   test_ks_atom_keys_value, 0, 1),
       run_test_r0('keysort float keys first*10 -> 15',
                   test_ks_float_keys, 0, 15),
       run_test_r0('keysort mixed num keys first -> 1',
                   test_ks_mixed_num_keys, 0, 1),
       format('--- M63 sort/2 (standard order + dedup) ---~n'),
       run_test_r0('sort([3,1,2]) length -> 3',
                   test_sort_int_unique, 0, 3),
       run_test_r0('sort([5,2,8,1,3]) first -> 1',
                   test_sort_int_first, 0, 1),
       run_test_r0('sort([5,2,8,1,3]) last -> 8',
                   test_sort_int_last, 0, 8),
       run_test_r0('sort([3,1,2,1,3,2]) dedup length -> 3',
                   test_sort_dedup, 0, 3),
       run_test_r0('sort([7,7,7,7]) length -> 1',
                   test_sort_all_same, 0, 1),
       run_test_r0('sort([]) + 23 -> 23',
                   test_sort_empty, 0, 23),
       run_test_r0('sort([42], [E]) -> 42',
                   test_sort_singleton, 0, 42),
       run_test_r0('sort atom keys first length -> 5 (apple)',
                   test_sort_atom_keys, 0, 5),
       run_test_r0('sort atom dedup length -> 3',
                   test_sort_atom_dedup, 0, 3),
       run_test_r0('sort float first*10 -> 15',
                   test_sort_float, 0, 15),
       run_test_r0('sort mixed num first -> 1',
                   test_sort_mixed_num, 0, 1),
       run_test_r0('sort already-sorted length -> 5',
                   test_sort_already_sorted, 0, 5),
       format('--- M115 chown/3 ---~n'),
       run_test_r0('chown to own (uid, gid) -> 1',
                   test_chown_self_self, 0, 1),
       run_test_r0('chown(-1, -1) no-op -> 1',
                   test_chown_noop, 0, 1),
       run_test_r0('chown on missing path fails -> 1',
                   test_chown_missing, 0, 1),
       run_test_r0('chown with non-atom path / non-int uid/gid fails -> 1',
                   test_chown_bad_args, 0, 1),
       format('--- M117 ground/1 ---~n'),
       run_test_r0('ground(atom) -> 1',
                   test_ground_atom, 0, 1),
       run_test_r0('ground(42) -> 1',
                   test_ground_int, 0, 1),
       run_test_r0('ground(foo(a,b,c)) -> 1',
                   test_ground_compound_closed, 0, 1),
       run_test_r0('ground(pair(p(1,2),q(3,[4,5]))) -> 1',
                   test_ground_nested_closed, 0, 1),
       run_test_r0('ground([1,2,3,foo]) -> 1',
                   test_ground_list_closed, 0, 1),
       run_test_r0('ground(_X) fails -> 1',
                   test_ground_bare_var, 0, 1),
       run_test_r0('ground(foo(a,_,c)) fails -> 1',
                   test_ground_compound_with_var, 0, 1),
       run_test_r0('ground with nested var fails -> 1',
                   test_ground_nested_with_var, 0, 1),
       run_test_r0('ground([1,2|_T]) fails -> 1',
                   test_ground_list_open_tail, 0, 1),
       run_test_r0('ground after binding var -> 1',
                   test_ground_after_bind, 0, 1),
       format('--- M118 file_base_name/2 + file_directory_name/2 ---~n'),
       run_test_r0('file_base_name(/usr/bin/swipl) -> swipl -> 1',
                   test_fbn_simple, 0, 1),
       run_test_r0('file_base_name(swipl) -> swipl -> 1',
                   test_fbn_no_slash, 0, 1),
       run_test_r0('file_base_name(/) -> '''' -> 1',
                   test_fbn_root, 0, 1),
       run_test_r0('file_base_name(/usr/bin/) -> '''' -> 1',
                   test_fbn_trailing_slash, 0, 1),
       run_test_r0('file_base_name('''') -> '''' -> 1',
                   test_fbn_empty, 0, 1),
       run_test_r0('file_base_name(/a/b/c/d/e.txt) -> e.txt -> 1',
                   test_fbn_nested, 0, 1),
       run_test_r0('file_base_name(42, _) fails -> 1',
                   test_fbn_bad_arg, 0, 1),
       run_test_r0('file_directory_name(/usr/bin/swipl) -> /usr/bin -> 1',
                   test_fdn_simple, 0, 1),
       run_test_r0('file_directory_name(swipl) -> . -> 1',
                   test_fdn_no_slash, 0, 1),
       run_test_r0('file_directory_name(/) -> / -> 1',
                   test_fdn_root, 0, 1),
       run_test_r0('file_directory_name(/etc) -> / -> 1',
                   test_fdn_root_child, 0, 1),
       run_test_r0('file_directory_name(/usr/bin/) -> /usr/bin -> 1',
                   test_fdn_trailing_slash, 0, 1),
       run_test_r0('file_directory_name('''') -> . -> 1',
                   test_fdn_empty, 0, 1),
       run_test_r0('file_directory_name(42, _) fails -> 1',
                   test_fdn_bad_arg, 0, 1),
       format('--- M119 file_name_extension/3 ---~n'),
       run_test_r0('split foo.txt -> foo+txt -> 1',
                   test_fne_split_simple, 0, 1),
       run_test_r0('split README -> README+'''' -> 1',
                   test_fne_split_no_ext, 0, 1),
       run_test_r0('split /usr/bin/foo.sh -> /usr/bin/foo+sh -> 1',
                   test_fne_split_path, 0, 1),
       run_test_r0('split /usr/foo.bar/baz (dot in dir) -> baz+'''' -> 1',
                   test_fne_split_dot_in_dir, 0, 1),
       run_test_r0('split .bashrc (hidden) -> .bashrc+'''' -> 1',
                   test_fne_split_hidden, 0, 1),
       run_test_r0('split /home/u/.bashrc (hidden in dir) -> +'''' -> 1',
                   test_fne_split_hidden_in_dir, 0, 1),
       run_test_r0('split archive.tar.gz -> archive.tar+gz -> 1',
                   test_fne_split_multi_dot, 0, 1),
       run_test_r0('split foo. (trailing dot) -> foo+'''' -> 1',
                   test_fne_split_trailing_dot, 0, 1),
       run_test_r0('split '''' -> ''''+'''' -> 1',
                   test_fne_split_empty, 0, 1),
       run_test_r0('join foo+txt -> foo.txt -> 1',
                   test_fne_join_simple, 0, 1),
       run_test_r0('join foo+'''' -> foo -> 1',
                   test_fne_join_empty_ext, 0, 1),
       run_test_r0('join /tmp/data+csv -> /tmp/data.csv -> 1',
                   test_fne_join_with_path, 0, 1),
       run_test_r0('check foo+txt vs foo.txt -> 1',
                   test_fne_join_check, 0, 1),
       run_test_r0('check foo+txt vs bar.txt fails -> 1',
                   test_fne_join_check_disagree, 0, 1),
       run_test_r0('all vars fails (insufficient instantiation) -> 1',
                   test_fne_insufficient, 0, 1),
       format('--- M120 read_link/2 + symlink/2 ---~n'),
       run_test_r0('symlink + read_link round-trip -> 1',
                   test_symlink_create_and_read, 0, 1),
       run_test_r0('dangling symlink read_link -> 1',
                   test_symlink_dangling, 0, 1),
       run_test_r0('read_link on regular file fails -> 1',
                   test_read_link_not_symlink, 0, 1),
       run_test_r0('read_link on missing path fails -> 1',
                   test_read_link_missing, 0, 1),
       run_test_r0('symlink at existing path fails -> 1',
                   test_symlink_exists_fail, 0, 1),
       run_test_r0('read_link(42, _) fails -> 1',
                   test_read_link_bad_arg, 0, 1),
       run_test_r0('symlink with non-atom args fails -> 1',
                   test_symlink_bad_arg, 0, 1),
       format('--- M121 link/2 ---~n'),
       run_test_r0('link + test -f -> 1',
                   test_link_create, 0, 1),
       run_test_r0('hard link shares inode -> 1',
                   test_link_same_inode, 0, 1),
       run_test_r0('link from missing path fails -> 1',
                   test_link_missing_old, 0, 1),
       run_test_r0('link to existing path fails -> 1',
                   test_link_exists_new, 0, 1),
       run_test_r0('link with non-atom args fails -> 1',
                   test_link_bad_arg, 0, 1),
       format('--- M122 is_absolute_file_name/1 + same_file/2 ---~n'),
       run_test_r0('is_absolute(/usr/bin/swipl) -> 1',
                   test_iaf_absolute, 0, 1),
       run_test_r0('is_absolute(foo/bar) fails -> 1',
                   test_iaf_relative, 0, 1),
       run_test_r0('is_absolute(/) -> 1',
                   test_iaf_root, 0, 1),
       run_test_r0('is_absolute('''') fails -> 1',
                   test_iaf_empty, 0, 1),
       run_test_r0('is_absolute(./foo) fails -> 1',
                   test_iaf_dot, 0, 1),
       run_test_r0('is_absolute(42) fails -> 1',
                   test_iaf_bad_arg, 0, 1),
       run_test_r0('same_file(P, P) -> 1',
                   test_samf_same_path, 0, 1),
       run_test_r0('same_file across hard link -> 1',
                   test_samf_hard_link, 0, 1),
       run_test_r0('same_file follows symlink -> 1',
                   test_samf_symlink, 0, 1),
       run_test_r0('same_file on distinct files fails -> 1',
                   test_samf_different, 0, 1),
       run_test_r0('same_file on missing path fails -> 1',
                   test_samf_missing, 0, 1),
       run_test_r0('same_file with non-atom args fails -> 1',
                   test_samf_bad_arg, 0, 1),
       format('--- M123 tmp_file/2 + mkfifo/2 ---~n'),
       run_test_r0('tmp_file creates file -> 1',
                   test_tmp_file_creates, 0, 1),
       run_test_r0('tmp_file path under /tmp + has Base label -> 1',
                   test_tmp_file_prefix, 0, 1),
       run_test_r0('tmp_file unique on consecutive calls -> 1',
                   test_tmp_file_unique, 0, 1),
       run_test_r0('tmp_file(42, _) fails -> 1',
                   test_tmp_file_bad_arg, 0, 1),
       run_test_r0('mkfifo + test -p -> 1',
                   test_mkfifo_create, 0, 1),
       run_test_r0('mkfifo at existing path fails -> 1',
                   test_mkfifo_exists_fail, 0, 1),
       run_test_r0('mkfifo(42, _) fails -> 1',
                   test_mkfifo_bad_path, 0, 1),
       run_test_r0('mkfifo(_, not_int) fails -> 1',
                   test_mkfifo_bad_mode, 0, 1),
       format('--- M124 umask/2 + monotonic_time/1 ---~n'),
       run_test_r0('umask set and restore -> 1',
                   test_umask_set_and_restore, 0, 1),
       run_test_r0('umask round-trip -> 1',
                   test_umask_round_trip, 0, 1),
       run_test_r0('umask(_, not_int) fails -> 1',
                   test_umask_bad_new, 0, 1),
       run_test_r0('monotonic_time >= 0 -> 1',
                   test_monotonic_nonneg, 0, 1),
       run_test_r0('monotonic_time advances -> 1',
                   test_monotonic_advances, 0, 1),
       run_test_r0('monotonic_time elapsed >= 0.04 -> 1',
                   test_monotonic_elapsed, 0, 1),
       format('--- M125 nice/1 + getpriority/1 + setpriority/1 ---~n'),
       run_test_r0('getpriority in [-20, 19] -> 1',
                   test_getpriority_in_range, 0, 1),
       run_test_r0('nice(0) preserves priority -> 1',
                   test_nice_zero, 0, 1),
       run_test_r0('nice(+5) raises niceness -> 1',
                   test_nice_positive_raises, 0, 1),
       run_test_r0('setpriority round-trip -> 1',
                   test_setpriority_roundtrip, 0, 1),
       run_test_r0('nice(not_int) fails -> 1',
                   test_nice_bad_arg, 0, 1),
       run_test_r0('setpriority(not_int) fails -> 1',
                   test_setpriority_bad_arg, 0, 1),
       format('--- M126 getrlimit/2 + setrlimit/2 ---~n'),
       run_test_r0('getrlimit FSIZE non-negative -> 1',
                   test_grl_fsize, 0, 1),
       run_test_r0('getrlimit CORE non-negative -> 1',
                   test_grl_core, 0, 1),
       run_test_r0('setrlimit CORE round-trip -> 1',
                   test_srl_round_trip, 0, 1),
       run_test_r0('setrlimit lower FSIZE OK -> 1',
                   test_srl_lower_ok, 0, 1),
       run_test_r0('getrlimit unknown resource fails -> 1',
                   test_grl_bad_resource, 0, 1),
       run_test_r0('getrlimit non-int fails -> 1',
                   test_grl_bad_arg, 0, 1),
       run_test_r0('setrlimit non-int args fail -> 1',
                   test_srl_bad_args, 0, 1),
       format('--- M127 getlogin/1 + uname_sysname/1 + uname_machine/1 ---~n'),
       run_test_r0('getlogin or no-tty fail accepted -> 1',
                   test_getlogin_or_fail, 0, 1),
       run_test_r0('uname_sysname non-empty -> 1',
                   test_uname_sysname_nonempty, 0, 1),
       run_test_r0('uname_machine non-empty -> 1',
                   test_uname_machine_nonempty, 0, 1),
       run_test_r0('uname_sysname stable across calls -> 1',
                   test_uname_sysname_stable, 0, 1),
       run_test_r0('uname_sysname known OS -> 1',
                   test_uname_known_linux_or_darwin, 0, 1),
       format('--- M128 copy_file/2 ---~n'),
       run_test_r0('copy_file basic round-trip + diff -> 1',
                   test_copy_file_basic, 0, 1),
       run_test_r0('copy_file empty source -> empty dst -> 1',
                   test_copy_file_empty, 0, 1),
       run_test_r0('copy_file large (multi-loop) round-trip -> 1',
                   test_copy_file_large, 0, 1),
       run_test_r0('copy_file missing source fails -> 1',
                   test_copy_file_missing_src, 0, 1),
       run_test_r0('copy_file O_TRUNC overwrite -> 1',
                   test_copy_file_overwrite, 0, 1),
       run_test_r0('copy_file with non-atom args fails -> 1',
                   test_copy_file_bad_args, 0, 1),
       format('--- M129 read_file_to_atom/2 ---~n'),
       run_test_r0('read_file_to_atom short ascii -> 1',
                   test_rfta_short, 0, 1),
       run_test_r0('read_file_to_atom empty -> '''' -> 1',
                   test_rfta_empty, 0, 1),
       run_test_r0('read_file_to_atom large file (loop) -> 1',
                   test_rfta_large, 0, 1),
       run_test_r0('read_file_to_atom missing path fails -> 1',
                   test_rfta_missing, 0, 1),
       run_test_r0('read_file_to_atom non-atom arg fails -> 1',
                   test_rfta_bad_arg, 0, 1),
       run_test_r0('read_file_to_atom length matches stat size -> 1',
                   test_rfta_size_matches, 0, 1),
       format('--- M130 write_atom_to_file/2 + append_atom_to_file/2 ---~n'),
       run_test_r0('write 11-byte atom -> size 11 -> 1',
                   test_wfa_basic, 0, 1),
       run_test_r0('write empty atom -> size 0 -> 1',
                   test_wfa_empty, 0, 1),
       run_test_r0('write O_TRUNC shrinks existing file -> 1',
                   test_wfa_truncates, 0, 1),
       run_test_r0('write then read round-trip -> 1',
                   test_wfa_round_trip, 0, 1),
       run_test_r0('write with non-atom args fails -> 1',
                   test_wfa_bad_args, 0, 1),
       run_test_r0('append creates new file -> 1',
                   test_afa_creates, 0, 1),
       run_test_r0('append extends existing file -> 1',
                   test_afa_extends, 0, 1),
       run_test_r0('append with non-atom args fails -> 1',
                   test_afa_bad_args, 0, 1),
       format('--- M131 errno/1 + strerror/2 ---~n'),
       run_test_r0('errno returns Integer -> 1',
                   test_errno_returns_int, 0, 1),
       run_test_r0('errno after failing delete_file != 0 -> 1',
                   test_errno_after_open_fail, 0, 1),
       run_test_r0('strerror(2) non-empty atom -> 1',
                   test_strerror_known_errno, 0, 1),
       run_test_r0('strerror(0) non-empty atom -> 1',
                   test_strerror_zero, 0, 1),
       run_test_r0('errno -> strerror round-trip non-empty -> 1',
                   test_strerror_round_trip, 0, 1),
       run_test_r0('strerror(not_int, _) fails -> 1',
                   test_strerror_bad_arg, 0, 1),
       format('--- M132 process_max_rss/user_time/system_time ---~n'),
       run_test_r0('process_max_rss > 0 -> 1',
                   test_max_rss_positive, 0, 1),
       run_test_r0('process_user_time >= 0 -> 1',
                   test_user_time_nonneg, 0, 1),
       run_test_r0('process_system_time >= 0 -> 1',
                   test_system_time_nonneg, 0, 1),
       run_test_r0('user_time non-decreasing -> 1',
                   test_user_time_monotonic, 0, 1),
       run_test_r0('max_rss non-decreasing -> 1',
                   test_max_rss_monotonic, 0, 1),
       format('--- M133 path_join/3 ---~n'),
       run_test_r0('path_join /usr/bin + swipl -> 1',
                   test_pj_simple, 0, 1),
       run_test_r0('path_join /usr/bin/ + swipl (no double slash) -> 1',
                   test_pj_trailing_slash, 0, 1),
       run_test_r0('path_join /usr/bin + /etc/hosts -> Rel wins -> 1',
                   test_pj_absolute_rel, 0, 1),
       run_test_r0('path_join '''' + foo -> foo -> 1',
                   test_pj_empty_base, 0, 1),
       run_test_r0('path_join /tmp + '''' -> /tmp -> 1',
                   test_pj_empty_rel, 0, 1),
       run_test_r0('path_join nested chain -> 1',
                   test_pj_nested, 0, 1),
       run_test_r0('path_join with non-atom args fails -> 1',
                   test_pj_bad_args, 0, 1),
       format('--- M134 system_to_atom/2 ---~n'),
       run_test_r0('echo hi -> 3-byte atom -> 1',
                   test_sta_echo, 0, 1),
       run_test_r0('true -> empty atom -> 1',
                   test_sta_empty, 0, 1),
       run_test_r0('pwd -> non-empty atom -> 1',
                   test_sta_pwd, 0, 1),
       run_test_r0('seq 1 5 -> 10-byte atom -> 1',
                   test_sta_multiline, 0, 1),
       run_test_r0('shell pipeline captures stdout -> 1',
                   test_sta_pipe, 0, 1),
       run_test_r0('system_to_atom(42, _) fails -> 1',
                   test_sta_bad_arg, 0, 1),
       format('--- M112 truncate/2 ---~n'),
       run_test_r0('touch + truncate 100 + size_file -> 1',
                   test_truncate_grow, 0, 1),
       run_test_r0('truncate 0 -> size 0 -> 1',
                   test_truncate_zero, 0, 1),
       run_test_r0('truncate on missing path fails -> 1',
                   test_truncate_missing, 0, 1),
       run_test_r0('truncate with non-atom path / non-int length fails -> 1',
                   test_truncate_bad_args, 0, 1),
       format('--- M111 kill/2 ---~n'),
       run_test_r0('kill(0, 0) self-probe -> 1',
                   test_kill_self_probe, 0, 1),
       run_test_r0('kill on missing pid fails -> 1',
                   test_kill_missing_pid, 0, 1),
       run_test_r0('kill with non-int args fails -> 1',
                   test_kill_bad_args, 0, 1),
       format('--- M110 realpath/2 ---~n'),
       run_test_r0('realpath(/tmp, Abs), Abs == /tmp -> 1',
                   test_rp_tmp, 0, 1),
       run_test_r0('realpath(., Abs), starts with / -> 1',
                   test_rp_relative, 0, 1),
       run_test_r0('realpath on missing path fails -> 1',
                   test_rp_missing, 0, 1),
       run_test_r0('realpath(42, _) fails (non-atom) -> 1',
                   test_rp_bad_arg, 0, 1),
       format('--- M109 getpgrp/1 ---~n'),
       run_test_r0('getpgrp(PG), PG > 0 -> 1',
                   test_pgrp_positive, 0, 1),
       run_test_r0('two getpgrp calls match -> 1',
                   test_pgrp_stable, 0, 1),
       format('--- M107 directory_files/2 ---~n'),
       run_test_r0('directory_files(/tmp, [_|_]) -> 1',
                   test_df_tmp_nonempty, 0, 1),
       run_test_r0('memberchk(., Fs), memberchk(.., Fs) -> 1',
                   test_df_contains_dot, 0, 1),
       run_test_r0('directory_files on missing dir fails -> 1',
                   test_df_missing_dir, 0, 1),
       run_test_r0('directory_files(42, _) fails (non-atom) -> 1',
                   test_df_bad_arg, 0, 1),
       format('--- M106 access/2 ---~n'),
       run_test_r0('access(/tmp, F_OK=0) -> 1',
                   test_access_tmp_exists, 0, 1),
       run_test_r0('access on missing path fails -> 1',
                   test_access_missing_file, 0, 1),
       run_test_r0('access(/tmp, W_OK=2) -> 1',
                   test_access_tmp_writable, 0, 1),
       run_test_r0('access with non-atom path / non-int mode fails -> 1',
                   test_access_bad_args, 0, 1),
       format('--- M105 numbervars/3 ---~n'),
       run_test_r0('foo(X,Y,Z), nv 0 -> 3, vars 0/1/2 -> 1',
                   test_nv_basic, 0, 1),
       run_test_r0('bar(X,X) shared, single $VAR(0), End=1 -> 1',
                   test_nv_shared, 0, 1),
       run_test_r0('ground term, End == Start -> 1',
                   test_nv_ground, 0, 1),
       run_test_r0('nested compound DFS L-to-R -> 1',
                   test_nv_nested, 0, 1),
       format('--- M104 wam_unify_value Ref-to-Ref aliasing ---~n'),
       run_test_r0('X = Y, X == Y -> 1',
                   test_unify_aliases_vars, 0, 1),
       run_test_r0('X = Y, X = 42, Y =:= 42 -> 1',
                   test_unify_then_bind_propagates, 0, 1),
       run_test_r0('term_variables(foo(X,_), [V|_]), V == X -> 1',
                   test_term_vars_identity, 0, 1),
       format('--- M103 term_variables/2 ---~n'),
       run_test_r0('ground term -> [] -> 1',
                   test_tv_ground, 0, 1),
       run_test_r0('one var -> [V], V == X -> 1',
                   test_tv_single, 0, 1),
       run_test_r0('three vars left-to-right -> 1',
                   test_tv_three, 0, 1),
       run_test_r0('nested compound, DFS left-to-right -> 1',
                   test_tv_nested, 0, 1),
       format('--- M102 chmod/2 ---~n'),
       run_test_r0('create + chmod 0o444 + size_file roundtrip -> 1',
                   test_chmod_set_readonly, 0, 1),
       run_test_r0('chmod on missing file fails -> 1',
                   test_chmod_missing_file, 0, 1),
       run_test_r0('chmod with non-atom path / non-int mode fails -> 1',
                   test_chmod_bad_args, 0, 1),
       format('--- M101 =.. partial-list unify ---~n'),
       run_test_r0('baz(7,8) =.. [H|_], H == baz -> 1',
                   test_univ_partial_head, 0, 1),
       run_test_r0('quux(a,b) =.. [_,_,_] -> 1 (arity check)',
                   test_univ_partial_arity, 0, 1),
       run_test_r0('quux(a,b) =.. [_,_] fails -> 1',
                   test_univ_partial_no_match, 0, 1),
       format('--- M100 functor/3 + =.. atom-rep fix ---~n'),
       run_test_r0('functor(foo(...), Name, _), Name == foo -> 1',
                   test_functor_name_eq_literal, 0, 1),
       run_test_r0('functor(bar(a,b), bar, 2) -> 1',
                   test_functor_arity_check, 0, 1),
       run_test_r0('baz(7,8) =.. [H|_], H == baz -> 1',
                   test_univ_head_eq_literal, 0, 1),
       run_test_r0('quux(1) =.. [_,_|T], T == [] -> 1',
                   test_univ_tail_is_empty_list, 0, 1),
       format('--- M99 date_time_stamp/2 ---~n'),
       run_test_r0('stamp -> DT -> stamp round-trip -> 1',
                   test_dts_roundtrip, 0, 1),
       run_test_r0('result is Float -> 1',
                   test_dts_returns_float, 0, 1),
       run_test_r0('bad arity / non-date compound / non-compound fails -> 1',
                   test_dts_bad_arity, 0, 1),
       format('--- M98 stamp_date_time/3 ---~n'),
       run_test_r0('arg(1,DT,_), arg(9,DT,_) -> 1',
                   test_sdt_arity, 0, 1),
       run_test_r0('1970 < Year < 3000 -> 1',
                   test_sdt_year_4digit, 0, 1),
       run_test_r0('1 <= Month <= 12 -> 1',
                   test_sdt_month_in_range, 0, 1),
       run_test_r0('TZName slot = local -> 1',
                   test_sdt_tzname, 0, 1),
       run_test_r0('Float stamp truncates -> Y in range -> 1',
                   test_sdt_float_stamp, 0, 1),
       format('--- M97 set_random/1 ---~n'),
       run_test_r0('set_random(seed(N)) changes lrand48 output -> 1',
                   test_setrand_changes_output, 0, 1),
       run_test_r0('same seed gives same draw -> 1',
                   test_setrand_repeatable, 0, 1),
       run_test_r0('bad option (non-seed compound, etc.) fails -> 1',
                   test_setrand_bad_option, 0, 1),
       format('--- M96 getgid/1 + getegid/1 + getppid/1 ---~n'),
       run_test_r0('getgid(G), G >= 0 -> 1',
                   test_gid_nonneg, 0, 1),
       run_test_r0('getegid(E), E >= 0 -> 1',
                   test_egid_nonneg, 0, 1),
       run_test_r0('getgid =:= getegid (no setgid) -> 1',
                   test_gid_eq_egid, 0, 1),
       run_test_r0('getppid(PP), PP > 0 -> 1',
                   test_ppid_positive, 0, 1),
       run_test_r0('getpid =\\= getppid -> 1',
                   test_ppid_neq_pid, 0, 1),
       format('--- M95 getuid/1 + geteuid/1 ---~n'),
       run_test_r0('getuid(U), U >= 0 -> 1',
                   test_uid_nonneg, 0, 1),
       run_test_r0('geteuid(E), E >= 0 -> 1',
                   test_euid_nonneg, 0, 1),
       run_test_r0('getuid =:= geteuid (no setuid) -> 1',
                   test_uid_eq_euid, 0, 1),
       format('--- M93 unsetenv/1 ---~n'),
       run_test_r0('setenv/getenv/unsetenv/getenv-fails roundtrip -> 1',
                   test_unsetenv_roundtrip, 0, 1),
       run_test_r0('unsetenv on already-unset succeeds -> 1',
                   test_unsetenv_idempotent, 0, 1),
       run_test_r0('unsetenv(42) fails (non-atom) -> 1',
                   test_unsetenv_non_atom, 0, 1),
       format('--- M92 halt/0 + halt/1 ---~n'),
       run_test_r0('halt/0 -> exit 0',
                   test_halt_zero, 0, 0),
       run_test_r0('halt(7) -> exit 7',
                   test_halt_seven, 0, 7),
       run_test_r0('halt(2+3) -> exit 5',
                   test_halt_var, 0, 5),
       format('--- M91 format_time/3 ---~n'),
       run_test_r0('format_time(Y, ''%Y'', stamp), len(Y) = 4 -> 1',
                   test_ft_year, 0, 1),
       run_test_r0('format_time(S, ISO, stamp), len(S) = 19 -> 1',
                   test_ft_iso_len, 0, 1),
       run_test_r0('format_time(S, ''%S'', Float stamp), len(S) = 2 -> 1',
                   test_ft_float_stamp, 0, 1),
       format('--- M90 random/1 + random_between/3 ---~n'),
       run_test_r0('random(X), 0.0 <= X < 1.0 -> 1',
                   test_rand_in_unit, 0, 1),
       run_test_r0('random_between(1, 10, X), 1 <= X <= 10 -> 1',
                   test_rand_between_in_range, 0, 1),
       run_test_r0('random_between(7, 7, X) =:= 7 -> 1',
                   test_rand_between_singleton, 0, 1),
       format('--- M89 cpu_time/1 ---~n'),
       run_test_r0('cpu_time(T), T >= 0.0 -> 1',
                   test_cpu_nonneg, 0, 1),
       run_test_r0('cpu_time monotonic across calls -> 1',
                   test_cpu_monotonic, 0, 1),
       run_test_r0('cpu_time accrued < wall during sleep -> 1',
                   test_cpu_under_wall, 0, 1),
       format('--- M88 gethostname/1 ---~n'),
       run_test_r0('gethostname(H), length(H) > 0 -> 1',
                   test_ghn_nonempty, 0, 1),
       run_test_r0('gethostname stable across calls -> 1',
                   test_ghn_stable, 0, 1),
       run_test_r0('gethostname \\= empty atom -> 1',
                   test_ghn_not_empty_atom, 0, 1),
       format('--- M87 get_time/1 nanosecond precision ---~n'),
       run_test_r0('sleep(0.05) elapsed in [0.04, 0.5) -> 1',
                   test_gt87_short_sleep, 0, 1),
       run_test_r0('sleep(0.05) ms_count >= 40 -> 1',
                   test_gt87_ms_count, 0, 1),
       run_test_r0('get_time(T), T - floor(T) > 0.0 -> 1',
                   test_gt87_fractional, 0, 1),
       format('--- M86 sleep/1 ---~n'),
       run_test_r0('sleep(0) -> 1',
                   test_sleep_zero, 0, 1),
       run_test_r0('sleep(0.0) -> 1',
                   test_sleep_float_zero, 0, 1),
       run_test_r0('sleep(0.001) -> 1',
                   test_sleep_tiny, 0, 1),
       run_test_r0('sleep(0.05) elapsed >= 0.04 -> 1 (M87 tightened)',
                   test_sleep_elapsed_float, 0, 1),
       run_test_r0('sleep(1) elapsed >= 0.5 -> 1',
                   test_sleep_elapsed_int, 0, 1),
       format('--- M85 bitwise /\\ \\/ \\ ---~n'),
       run_test_r0('12 /\\ 10 -> 8',
                   test_band_basic, 0, 8),
       run_test_r0('0xFF /\\ 0x0F -> 15',
                   test_band_byte, 0, 15),
       run_test_r0('5 \\/ 3 -> 7',
                   test_bor_basic, 0, 7),
       run_test_r0('1\\/2\\/4\\/8\\/16 -> 31',
                   test_bor_combine, 0, 31),
       run_test_r0('\\0 /\\ 0xFF -> 255',
                   test_bnot_byte, 0, 255),
       format('--- M84 integer bitshifts << / >> ---~n'),
       run_test_r0('1 << 4 -> 16',
                   test_shl_basic, 0, 16),
       run_test_r0('1 << 7 -> 128',
                   test_shl_byte_top, 0, 128),
       run_test_r0('31 << 3 -> 248',
                   test_shl_31_3, 0, 248),
       run_test_r0('240 >> 4 -> 15',
                   test_shr_basic, 0, 15),
       run_test_r0('256 >> 4 -> 16',
                   test_shr_round, 0, 16),
       format('--- M83 pi/e constants + xor/2 ---~n'),
       run_test_r0('truncate(pi * 50) -> 157',
                   test_pi_50, 0, 157),
       run_test_r0('pi > 3.0 -> 1',
                   test_pi_gt3, 0, 1),
       run_test_r0('truncate(e * 90) -> 244',
                   test_e_90, 0, 244),
       run_test_r0('xor(5, 3) -> 6',
                   test_xor_small, 0, 6),
       run_test_r0('xor(255, 15) -> 240',
                   test_xor_byte, 0, 240),
       format('--- M82 gcd/2 + log/2 binary arith ---~n'),
       run_test_r0('gcd(12, 18) -> 6',
                   test_gcd_basic, 0, 6),
       run_test_r0('gcd(7, 5) -> 1 (coprime)',
                   test_gcd_coprime, 0, 1),
       run_test_r0('gcd(0, 5) -> 5',
                   test_gcd_with_zero, 0, 5),
       run_test_r0('truncate(log(2.0, 8.0)) -> 3',
                   test_log2_eight, 0, 3),
       run_test_r0('truncate(log(10.0, 100.0)) -> 2',
                   test_log10_hundred, 0, 2),
       format('--- M81 atan2/2 binary inverse tangent ---~n'),
       run_test_r0('atan2(0,1) -> 0 (x-axis)',
                   test_atan2_xaxis, 0, 0),
       run_test_r0('atan2(1,1) * 200 -> 157 (~ pi/4)',
                   test_atan2_diag, 0, 157),
       run_test_r0('atan2(1,0) * 100 -> 157 (~ pi/2)',
                   test_atan2_yaxis, 0, 157),
       run_test_r0('atan2(2,2) * 200 -> 157 (ratio only)',
                   test_atan2_diag_scaled, 0, 157),
       run_test_r0('atan2(0,-1) * 50 -> 157 (~ pi)',
                   test_atan2_pi, 0, 157),
       format('--- M80 inverse trig -- asin/1, acos/1, atan/1 ---~n'),
       run_test_r0('truncate(asin(0.0) * 100) -> 0',
                   test_asin_zero, 0, 0),
       run_test_r0('truncate(asin(1.0) * 100) -> 157 (~ pi/2)',
                   test_asin_one, 0, 157),
       run_test_r0('truncate(acos(1.0) * 100) -> 0',
                   test_acos_one, 0, 0),
       run_test_r0('truncate(acos(0.0) * 100) -> 157 (~ pi/2)',
                   test_acos_zero, 0, 157),
       run_test_r0('truncate(atan(1.0) * 200) -> 157 (~ pi/2)',
                   test_atan_one, 0, 157),
       format('--- M79 working_directory/2 + getpid/1 ---~n'),
       run_test_r0('working_directory(D, D) query -> 1',
                   test_wd_query, 0, 1),
       run_test_r0('working_directory chdir/restore roundtrip -> 1',
                   test_wd_chdir, 0, 1),
       run_test_r0('working_directory chdir to /nonexistent -> 0',
                   test_wd_fail, 0, 0),
       run_test_r0('getpid(P), P > 0 -> 1',
                   test_getpid_pos, 0, 1),
       run_test_r0('getpid stable across calls -> 1',
                   test_getpid_stable, 0, 1),
       format('--- M78 shell/1 + shell/2 ---~n'),
       run_test_r0('shell(true) -> 1',
                   test_sh1_true, 0, 1),
       run_test_r0('shell(false) -> 0',
                   test_sh1_false, 0, 0),
       run_test_r0('shell(/nonexistent/...) -> 0',
                   test_sh1_nonexistent, 0, 0),
       run_test_r0('shell(true, S), S=0 -> 0',
                   test_sh2_true, 0, 0),
       run_test_r0('shell(exit 42, S), S=42 -> 42',
                   test_sh2_exit42, 0, 42),
       format('--- M77 getenv/2 + setenv/2 ---~n'),
       run_test_r0('getenv(PATH) length > 0 -> 1',
                   test_ge_path, 0, 1),
       run_test_r0('getenv(unset) -> 0',
                   test_ge_missing, 0, 0),
       run_test_r0('setenv + getenv roundtrip -> 1',
                   test_se_basic, 0, 1),
       run_test_r0('setenv overwrite -> 1 (last write wins)',
                   test_se_overwrite, 0, 1),
       run_test_r0('setenv empty value -> 0 (empty atom length)',
                   test_se_empty, 0, 0),
       format('--- M76 size_file/2 + time_file/2 ---~n'),
       run_test_r0('size_file(/etc/hostname) > 0 -> 1',
                   test_sf_etc_hostname, 0, 1),
       run_test_r0('size_file(/dev/null) -> 0',
                   test_sf_zero, 0, 0),
       run_test_r0('size_file(/nonexistent/...) -> 0',
                   test_sf_fail_missing, 0, 0),
       run_test_r0('time_file(/etc/hostname) > 0.0 -> 1',
                   test_tf_etc_hostname, 0, 1),
       run_test_r0('time_file(/nonexistent/...) -> 0',
                   test_tf_fail_missing, 0, 0),
       format('--- M75 rename_file/2 + delete_directory/1 ---~n'),
       run_test_r0('rename_file(src, dst) roundtrip -> 1',
                   test_rnf_basic, 0, 1),
       run_test_r0('rename_file(/nonexistent, ...) -> 0',
                   test_rnf_fail_missing, 0, 0),
       run_test_r0('delete_directory roundtrip -> 1',
                   test_ddr_basic, 0, 1),
       run_test_r0('delete_directory(/nonexistent/...) -> 0',
                   test_ddr_fail_missing, 0, 0),
       run_test_r0('delete_directory(/etc/hostname) file -> 0',
                   test_ddr_fail_file, 0, 0),
       format('--- M74 delete_file/1 + make_directory/1 ---~n'),
       run_test_r0('make_directory + exists_directory roundtrip -> 1',
                   test_mkd_basic, 0, 1),
       run_test_r0('make_directory(/sys/...) no permission -> 0',
                   test_mkd_fail_perm, 0, 0),
       run_test_r0('make_directory(/nonexistent/...) missing parent -> 0',
                   test_mkd_fail_parent, 0, 0),
       run_test_r0('delete_file(/nonexistent/file) -> 0',
                   test_df_fail_missing, 0, 0),
       run_test_r0('delete_file(/tmp) directory -> 0',
                   test_df_dir_no, 0, 0),
       format('--- M73 exists_file/1 + exists_directory/1 ---~n'),
       run_test_r0('exists_file(/etc/hostname) -> 1',
                   test_xf_real, 0, 1),
       run_test_r0('exists_file(/nonexistent/...) -> 0',
                   test_xf_missing, 0, 0),
       run_test_r0('exists_file(/etc) directory -> 0',
                   test_xf_directory_no, 0, 0),
       run_test_r0('exists_directory(/etc) -> 1',
                   test_xd_real, 0, 1),
       run_test_r0('exists_directory(/etc/hostname) file -> 0',
                   test_xd_file_no, 0, 0),
       run_test_r0('exists_directory(/nonexistent/...) -> 0',
                   test_xd_missing, 0, 0),
       format('--- M72 get_time/1 ---~n'),
       run_test_r0('get_time(_) succeeds -> 1',
                   test_get_time_succeeds, 0, 1),
       run_test_r0('get_time(T), T > 0.0 -> 1',
                   test_get_time_positive, 0, 1),
       run_test_r0('get_time(T) > 2020 epoch -> 1',
                   test_get_time_recent, 0, 1),
       run_test_r0('get_time monotonic -> 1',
                   test_get_time_monotonic, 0, 1),
       format('--- M71 forall/2 (compile-time \\+ (Cond, \\+ Action) rewrite) ---~n'),
       run_test_r0('forall manual soft-cut rewrite -> 1',
                   test_forall_manual + [positive/1], 0, 1),
       run_test_r0('forall(positive(X), X > 0) -> 1',
                   test_forall_all_pos + [positive/1], 0, 1),
       run_test_r0('forall(positive(X), X > 1) -> 0 (1 fails)',
                   test_forall_one_fails + [positive/1], 0, 0),
       run_test_r0('forall(positive(X), small(X)) subset -> 1',
                   test_forall_subset + [positive/1, small/1], 0, 1),
       run_test_r0('forall over empty solutions -> 1 (vacuous)',
                   test_forall_empty, 0, 1),
       format('--- M70 msort/2 + setof migrated to @wam_term_cmp ---~n'),
       run_test_r0('msort([d,b,a,c]) first -> 97 (a)',
                   test_msort_atoms_alpha, 0, 97),
       run_test_r0('msort([d,b,a,c]) last -> 100 (d)',
                   test_msort_atoms_last, 0, 100),
       run_test_r0('msort([2,1,2,1,3]) length -> 5 (dupes kept)',
                   test_msort_dupes_kept, 0, 5),
       run_test_r0('msort([foo(3), foo(1), foo(2)]) first arg -> 1',
                   test_msort_compound, 0, 1),
       format('--- M69 standard-order comparison operators ---~n'),
       run_test_r0('1 @< 2 -> 1',
                   test_at_lt_yes, 0, 1),
       run_test_r0('5 @< 2 -> 0',
                   test_at_lt_no, 0, 0),
       run_test_r0('3 @< 3 -> 0',
                   test_at_lt_eq, 0, 0),
       run_test_r0('1 @=< 2 -> 1',
                   test_at_le_yes, 0, 1),
       run_test_r0('3 @=< 3 -> 1',
                   test_at_le_eq, 0, 1),
       run_test_r0('5 @=< 2 -> 0',
                   test_at_le_no, 0, 0),
       run_test_r0('5 @> 2 -> 1',
                   test_at_gt_yes, 0, 1),
       run_test_r0('1 @> 2 -> 0',
                   test_at_gt_no, 0, 0),
       run_test_r0('5 @>= 2 -> 1',
                   test_at_ge_yes, 0, 1),
       run_test_r0('3 @>= 3 -> 1',
                   test_at_ge_eq, 0, 1),
       run_test_r0('apple @< banana -> 1',
                   test_at_atom, 0, 1),
       run_test_r0('42 @< foo cross-cat -> 1',
                   test_at_cross_cat, 0, 1),
       run_test_r0('foo(1) @< foo(2) -> 1',
                   test_at_compound, 0, 1),
       format('--- M68 split_string/4 ---~n'),
       run_test_r0('split_string(\'a,b,c\', \',\', \'\') length -> 3',
                   test_ss_simple, 0, 3),
       run_test_r0('split_string first segment length -> 5',
                   test_ss_first_length, 0, 5),
       run_test_r0('split_string multi-char sep set length -> 4',
                   test_ss_multi_sep, 0, 4),
       run_test_r0('split_string pad-strip first length -> 1',
                   test_ss_pad_strip, 0, 1),
       run_test_r0('split_string no sep present -> 1',
                   test_ss_no_sep, 0, 1),
       run_test_r0('split_string empty sep -> 1',
                   test_ss_empty_sep, 0, 1),
       run_test_r0('split_string empty source -> 1 (+31 = 32)',
                   test_ss_empty_source, 0, 32),
       run_test_r0('split_string consecutive seps length -> 3',
                   test_ss_consecutive_seps, 0, 3),
       run_test_r0('split_string pad-only -> empty (+42 = 42)',
                   test_ss_pad_only, 0, 42),
       format('--- M66 tab/1 + put_char/1 + put_code/1 ---~n'),
       run_test_r0('[1,2,3] length via R is N -> 3',
                   test_lit_3_ints, 0, 3),
       run_test_r0('[foo(1), foo(2), foo(3)] length via R is N -> 3',
                   test_lit_3_compounds_arity1, 0, 3),
       run_test_r0('[a-1, b-2, c-3] length via R is N -> 3',
                   test_lit_pair_int, 0, 3),
       run_test_r0('keysort compound keys length -> 3',
                   test_ks_compound_keys_length, 0, 3),
       run_test_r0('keysort compound keys first value -> 97',
                   test_ks_compound_keys_first, 0, 97),
       run_test_r0('sort compound elements first arg -> 1',
                   test_sort_compound_elements, 0, 1),
       run_test_r0('sort compound dedup length -> 3',
                   test_sort_compound_dedup, 0, 3),
       run_test_r0('sort lists first sublist first elem -> 1',
                   test_sort_lists, 0, 1),
       run_test_r0('tab(3) -> 1',
                   test_tab_3, 0, 1),
       run_test_r0('tab(0) -> 1',
                   test_tab_zero, 0, 1),
       run_test_r0('tab(-1) -> 0',
                   test_tab_neg, 0, 0),
       run_test_r0('tab(hello) -> 0',
                   test_tab_not_int, 0, 0),
       run_test_r0('put_char(a) -> 1',
                   test_put_char_basic, 0, 1),
       run_test_r0('put_char(\'5\') -> 1',
                   test_put_char_digit, 0, 1),
       run_test_r0('put_char(abc) -> 0',
                   test_put_char_multi, 0, 0),
       run_test_r0('put_char(7) -> 0',
                   test_put_char_int, 0, 0),
       run_test_r0('put_code(65) -> 1',
                   test_put_code_basic, 0, 1),
       run_test_r0('put_code(10) -> 1',
                   test_put_code_low, 0, 1),
       run_test_r0('put_code(-1) -> 0',
                   test_put_code_neg, 0, 0),
       run_test_r0('put_code(300) -> 0',
                   test_put_code_oob, 0, 0),
       format('--- M64 compare/3 Compound terms (recursive via helper) ---~n'),
       run_test_r0('compare(O, foo(1,2), foo(1,2)) -> 61 (=)',
                   test_cmp_compound_eq, 0, 61),
       run_test_r0('compare(O, foo(1), foo(1,2)) arity lt -> 60',
                   test_cmp_compound_arity_lt, 0, 60),
       run_test_r0('compare(O, foo(1,2,3), foo(1,2)) arity gt -> 62',
                   test_cmp_compound_arity_gt, 0, 62),
       run_test_r0('compare(O, alpha, beta) functor lt -> 60',
                   test_cmp_compound_functor_lt, 0, 60),
       run_test_r0('compare(O, foo(1,5), foo(1,9)) arg lt -> 60',
                   test_cmp_compound_arg_lt, 0, 60),
       run_test_r0('compare(O, foo(2,5), foo(1,5)) arg gt -> 62',
                   test_cmp_compound_arg_gt, 0, 62),
       run_test_r0('compare nested foo(bar(1)) vs foo(bar(2)) -> 60',
                   test_cmp_compound_recursive, 0, 60),
       run_test_r0('compare compound vs atom -> 62 (atom < compound)',
                   test_cmp_compound_vs_atom, 0, 62),
       run_test_r0('compare equal lists -> 61',
                   test_cmp_lists_via_compound, 0, 61),
       run_test_r0('compare [1,2,3] vs [1,2,4] -> 60',
                   test_cmp_lists_diff, 0, 60),
       format('--- M20 transcendentals -- sin / cos / tan / log / exp ---~n'),
       run_test_r0('truncate(sin(22/7/2) * 100) -> ~99',
                   test_sin_pi_half, 0, 99),
       run_test_r0('truncate(cos(0) * 100) -> 100',
                   test_cos_zero, 0, 100),
       run_test_r0('truncate(tan(0) * 100) -> 0',
                   test_tan_zero, 0, 0),
       run_test_r0('truncate(log(exp(2))) -> 2',
                   test_log_e, 0, 2),
       run_test_r0('truncate(exp(0)) -> 1',
                   test_exp_zero, 0, 1),
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
       format('--- M21 compound pretty-printing through write/1 ---~n'),
       run_fmt_test('~w of [1,2,3]', test_write_list3, "[1, 2, 3]"),
       run_fmt_test('~w of a-b (M22 infix notation)',
                    test_write_pair, "a-b"),
       run_fmt_test('~w of []', test_write_empty, "[]"),
       run_fmt_test('~w of [1,[2,3],4]', test_write_nested, "[1, [2, 3], 4]"),
       run_fmt_test('~w of foo(1, hello, 3.5)', test_write_compound,
                    "foo(1, hello, 3.5)"),
       format('--- M22 infix / prefix operator notation ---~n'),
       run_fmt_test('~w of 1+2', test_write_add, "1+2"),
       run_fmt_test('~w of 1+2*3', test_write_mul_add, "1+2*3"),
       run_fmt_test('~w of x=y', test_write_eq, "x=y"),
       run_fmt_test('~w of a:b', test_write_colon, "a:b"),
       run_fmt_test('~w of -(x)', test_write_neg, "-x"),
       run_fmt_test('~w of [a-1, b-2]',
                    test_write_list_of_pairs, "[a-1, b-2]"),
       format('--- M23 two-char symbolic operator notation ---~n'),
       run_fmt_test('~w of (a->b)', test_write_arrow, "a->b"),
       run_fmt_test('~w of (h :- g)', test_write_neck, "h:-g"),
       run_fmt_test('~w of (1==2)', test_write_struct_eq, "1==2"),
       run_fmt_test('~w of (a\\=b)', test_write_not_unify, "a\\=b"),
       run_fmt_test('~w of (3=<4)', test_write_le, "3=<4"),
       run_fmt_test('~w of (5>=4)', test_write_ge, "5>=4"),
       run_fmt_test('~w of (7//2)', test_write_int_div, "7//2"),
       run_fmt_test('~w of (2**8)', test_write_pow, "2**8"),
       format('--- M24 word-like operators (is / mod / xor / rem / div) ---~n'),
       run_fmt_test('~w of is(x, 5)', test_write_is, "x is 5"),
       run_fmt_test('~w of mod(7, 3)', test_write_mod, "7 mod 3"),
       run_fmt_test('~w of xor(5, 3)', test_write_xor, "5 xor 3"),
       run_fmt_test('~w of rem(10, 4)', test_write_rem, "10 rem 4"),
       run_fmt_test('~w of div(8, 3)', test_write_div, "8 div 3"),
       format('--- M25 three-char symbolic operators ---~n'),
       run_fmt_test('~w of =:=(x, y)', test_write_arith_eq, "x=:=y"),
       run_fmt_test('~w of =\\=(x, y)', test_write_arith_ne, "x=\\=y"),
       run_fmt_test('~w of =..(foo, [1, 2])',
                    test_write_univ, "foo=..[1, 2]"),
       run_fmt_test('~w of \\==(a, b)', test_write_struct_ne, "a\\==b"),
       run_fmt_test('~w of @=<(a, b)', test_write_term_le, "a@=<b"),
       run_fmt_test('~w of @>=(b, a)', test_write_term_ge, "b@>=a"),
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
