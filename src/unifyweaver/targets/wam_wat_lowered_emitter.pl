:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_wat_lowered_emitter.pl — WAM-lowered WAT emission
%
% Emits per-predicate WAT fast-path functions for deterministic clause-1
% prefixes.  The shape mirrors the hybrid WAM targets (Rust, Scala,
% Haskell, F#, LLVM): try the lowered clause-1 path first, and if it fails
% the generated public entry point reinitialises state and falls back to
% the complete bytecode interpreter.

:- module(wam_wat_lowered_emitter, [
    wam_wat_lowerable/3,
    lower_predicate_to_wat/4,
    wat_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2]).

%% wam_wat_lowerable(+PI, +WamCode, -Reason) is semidet.
wam_wat_lowerable(_PI, WamCode, Reason) :-
    parse_wam_text_wat(WamCode, Instrs),
    clause1_instrs_wat(Instrs, C1, MultiClause),
    (   is_deterministic_pred_wat(C1),
        forall(member(I, C1), wat_supported(I))
    ->  ( MultiClause == true -> Reason = multi_clause_1 ; Reason = single_clause )
    ;   % Clause-1 has an inner choice point: lower only if it is a pure
        % (C -> T ; E) / \+ / once block (T2) whose pieces are all supported.
        % structure_ite folds the soft-cut block into ite(Cond,Then,Else) and
        % drops the structural markers; we then emit native WAT if/else with a
        % trail rollback before the else. structure_ite only succeeds for a
        % genuine single ITE block (its then-path must end in a jump; a real
        % multi-clause predicate ends each clause in proceed, so it cannot
        % fold) — so no extra multi-clause guard is needed here.
        wat_structured_clause1(WamCode, Structured),
        forall(member(I, Structured), wat_supported_structured(I)),
        Reason = ite_lowered
    ).

%% wat_structured_clause1(+WamCode, -Structured) is semidet.
%  Re-parse keeping labels (reinserted from the label map), take clause 1, and
%  fold its ITE block(s) into ite/3. Succeeds only when an ite is present and
%  every block is consumed (no leftover try_me_else/trust_me/retry_me_else).
wat_structured_clause1(WamCode, Structured) :-
    wat_labeled_instrs(WamCode, Labeled),
    take_to_proceed_labeled(Labeled, C1L),
    % structure_ite can offer more than one fold for clauses with sequential
    % or nested ITE blocks; commit to the first fully-structured one so the
    % gate and the emitter agree on a single deterministic result (otherwise
    % the project writer emits the lowered function once per solution).
    once(( structure_ite(C1L, Structured),
           member(ite(_, _, _), Structured),
           \+ member(try_me_else(_), Structured),
           \+ member(trust_me, Structured),
           \+ member(retry_me_else(_), Structured) )).

%% wat_labeled_instrs(+WamCode, -Labeled)
%  Parse like parse_wam_text_wat but reinsert label(Name) markers inline (at
%  the instruction index recorded in the label map), so structure_ite can see
%  the try_me_else/label/trust_me/jump block boundaries it folds on.
wat_labeled_instrs(WamCode, Labeled) :-
    (   string(WamCode) -> S = WamCode ; atom_string(WamCode, S) ),
    split_string(S, "\n", "", Lines),
    wam_wat_target:wam_lines_to_instrs(Lines, 0, Instrs, Labels),
    reinsert_labels(Instrs, 0, Labels, Labeled).

% Insert label(Name) before the instruction whose 0-based index matches the
% label's PC. Pred-name labels (containing '/') are skipped — they are not part
% of any ITE block and would only add noise.
reinsert_labels([], _, _, []).
reinsert_labels([I|Rest], Idx, Labels, Out) :-
    findall(label(Name),
            ( member(NameS-Idx, Labels), \+ sub_string(NameS, _, _, _, "/"),
              atom_string(Name, NameS) ),
            Here),
    append(Here, [I|More], Out),
    Idx1 is Idx + 1,
    reinsert_labels(Rest, Idx1, Labels, More).

take_to_proceed_labeled([], []).
take_to_proceed_labeled([proceed|_], [proceed]) :- !.
take_to_proceed_labeled([I|Rest], [I|More]) :- take_to_proceed_labeled(Rest, More).

%% wat_supported_structured(+StructuredInstr)
wat_supported_structured(ite(C, T, E)) :- !,
    forall(member(I, C), wat_supported_structured(I)),
    forall(member(I, T), wat_supported_structured(I)),
    forall(member(I, E), wat_supported_structured(I)).
wat_supported_structured(label(_)) :- !.   % residual head/cont label: skipped on emit
wat_supported_structured(I) :- wat_supported(I).

parse_wam_text_wat(WamCode, Instrs) :-
    (   string(WamCode) -> S = WamCode
    ;   atom_string(WamCode, S)
    ),
    split_string(S, "\n", "", Lines),
    wam_wat_target:wam_lines_to_instrs(Lines, 0, Instrs, _Labels).

clause1_instrs_wat([try_me_else(_)|Rest], C1, true) :- !,
    take_to_proceed_wat(Rest, C1).
clause1_instrs_wat(Instrs, Instrs, false).

take_to_proceed_wat([], []).
take_to_proceed_wat([proceed|_], [proceed]) :- !.
take_to_proceed_wat([I|Rest], [I|More]) :- take_to_proceed_wat(Rest, More).

is_deterministic_pred_wat(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

wat_supported(get_constant(_, _)).
wat_supported(get_variable(_, _)).
wat_supported(get_value(_, _)).
wat_supported(get_structure(_, _)).
wat_supported(get_list(_)).
wat_supported(unify_variable(_)).
wat_supported(unify_value(_)).
wat_supported(unify_constant(_)).
wat_supported(put_constant(_, _)).
wat_supported(put_variable(_, _)).
wat_supported(put_value(_, _)).
wat_supported(put_structure(_, _)).
wat_supported(put_list(_)).
wat_supported(set_variable(_)).
wat_supported(set_value(_)).
wat_supported(set_constant(_)).
wat_supported(allocate).
wat_supported(deallocate).
wat_supported(proceed).
wat_supported(builtin_call(Op, _)) :- deterministic_builtin_wat(Op).

% Keep calls/execute in the interpreter for now: a lowered WAT function has
% no cross-predicate continuation frame, so inlining them would not be sound.
deterministic_builtin_wat('=/2').
deterministic_builtin_wat('true/0').
deterministic_builtin_wat('fail/0').
deterministic_builtin_wat('!/0').
deterministic_builtin_wat('is/2').
deterministic_builtin_wat('=:=/2').
deterministic_builtin_wat('=\\=/2').
deterministic_builtin_wat('</2').
deterministic_builtin_wat('>/2').
deterministic_builtin_wat('=</2').
deterministic_builtin_wat('>=/2').
deterministic_builtin_wat('var/1').
deterministic_builtin_wat('nonvar/1').
deterministic_builtin_wat('atom/1').
deterministic_builtin_wat('number/1').
deterministic_builtin_wat('integer/1').
deterministic_builtin_wat('float/1').
deterministic_builtin_wat('atomic/1').
deterministic_builtin_wat('is_list/1').

wat_lowered_func_name(Pred/Arity, Name) :-
    atom_string(Pred, S),
    string_codes(S, Codes),
    maplist(wat_safe_code, Codes, SafeCodes),
    string_codes(Safe, SafeCodes),
    format(atom(Name), 'lowered_~w_~w', [Safe, Arity]).

wat_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ), !.
wat_safe_code(_, 0'_).

%% lower_predicate_to_wat(+PI, +WamCode, +Options, -lowered(PredName, FuncName, Code))
lower_predicate_to_wat(PI, WamCode, _Options, lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    wat_lowered_func_name(Pred/Arity, FuncName),
    wam_wat_lowerable(PI, WamCode, Reason),
    (   Reason == ite_lowered
    ->  wat_structured_clause1(WamCode, Structured),
        with_output_to(atom(Code), emit_lowered_ite_function_wat(FuncName, Structured))
    ;   parse_wam_text_wat(WamCode, Instrs),
        clause1_instrs_wat(Instrs, C1, _MultiClause),
        with_output_to(atom(Code), emit_lowered_function_wat(FuncName, C1))
    ).

emit_lowered_function_wat(FuncName, Instrs) :-
    format('(func $~w (result i32)~n', [FuncName]),
    format('  ;; WAM-WAT lowered clause-1 fast path. On failure the public entry replays via $run_loop.~n'),
    emit_lowered_instrs_wat(Instrs),
    format('  (i32.const 1)~n'),
    format(')~n').

emit_lowered_instrs_wat([]).
emit_lowered_instrs_wat([proceed|_]) :- !,
    format('  (return (i32.const 1))~n').
emit_lowered_instrs_wat([Instr|Rest]) :-
    wam_wat_target:wam_instruction_to_wat_operands(Instr, [], DoName, Op1, Op2),
    format('  (if (i32.eqz (call $do_~w (i64.const ~w) (i64.const ~w)))~n', [DoName, Op1, Op2]),
    format('    (then (return (i32.const 0))))~n'),
    emit_lowered_instrs_wat(Rest).

% =====================================================================
% T2: if-then-else / negation / once  (structured ite/3)
% =====================================================================
%
%  Emit native WAT if/else for each ite(Cond,Then,Else). The condition runs
%  inside a labeled `(block $condK (result i32) … (i32.const 1))`: each
%  condition instruction that fails branches out with `(br $condK (i32.const
%  0))`, otherwise the block yields 1. A trail mark saved before the block lets
%  the else branch undo any partial bindings the condition made
%  (`$unwind_trail`), mirroring the bytecode `try_me_else`+`cut_ite` semantics
%  where the failed condition's choice-point trail mark is unwound. The
%  then/else branches emit with the ENCLOSING failure continuation (a real
%  failure there fails the whole lowered clause → the public entry replays via
%  $run_loop), while the condition emits with the block-local one.
%
%  WAT requires all locals declared right after the header, so we count the
%  ite blocks first and declare one `$ite_markK` per block; the body emit uses
%  the same monotone K so marks never collide across nesting.
emit_lowered_ite_function_wat(FuncName, Structured) :-
    count_ites(Structured, NMarks),
    format('(func $~w (result i32)~n', [FuncName]),
    format('  ;; WAM-WAT lowered if-then-else (T2). On failure the public entry replays via $run_loop.~n'),
    forall(between(1, NMarks, K0),
           ( K is K0 - 1, format('  (local $ite_mark~w i32)~n', [K]) )),
    nb_setval(wat_ite_ctr, 0),
    emit_struct_wat(Structured, '(return (i32.const 0))'),
    format('  (i32.const 1)~n'),
    format(')~n').

count_ites(List, N) :- count_ites(List, 0, N).
count_ites([], N, N).
count_ites([ite(C,T,E)|Rest], N0, N) :- !,
    N1 is N0 + 1,
    count_ites(C, N1, N2), count_ites(T, N2, N3), count_ites(E, N3, N4),
    count_ites(Rest, N4, N).
count_ites([_|Rest], N0, N) :- count_ites(Rest, N0, N).

%% emit_struct_wat(+StructuredInstrs, +FailCont)
%  FailCont is the WAT snippet to run when an instruction fails (either
%  `(return (i32.const 0))` at clause level, or `(br $condK (i32.const 0))`
%  inside a condition block).
emit_struct_wat([], _).
emit_struct_wat([label(_)|Rest], FailCont) :- !,
    emit_struct_wat(Rest, FailCont).
emit_struct_wat([proceed|_], _) :- !,
    format('  (return (i32.const 1))~n').
emit_struct_wat([ite(Cond, Then, Else)|Rest], FailCont) :- !,
    nb_getval(wat_ite_ctr, K), K1 is K + 1, nb_setval(wat_ite_ctr, K1),
    format('  (local.set $ite_mark~w (call $get_trail_top))~n', [K]),
    format('  (if (block $ite_cond~w (result i32)~n', [K]),
    format(atom(CondFail), '(br $ite_cond~w (i32.const 0))', [K]),
    emit_struct_wat(Cond, CondFail),
    format('    (i32.const 1))~n'),
    format('    (then~n'),
    emit_struct_wat(Then, FailCont),
    format('    )~n'),
    format('    (else~n'),
    format('      (call $unwind_trail (local.get $ite_mark~w))~n', [K]),
    emit_struct_wat(Else, FailCont),
    format('    ))~n'),
    emit_struct_wat(Rest, FailCont).
emit_struct_wat([Instr|Rest], FailCont) :-
    wam_wat_target:wam_instruction_to_wat_operands(Instr, [], DoName, Op1, Op2),
    format('  (if (i32.eqz (call $do_~w (i64.const ~w) (i64.const ~w)))~n', [DoName, Op1, Op2]),
    format('    (then ~w))~n', [FailCont]),
    emit_struct_wat(Rest, FailCont).
