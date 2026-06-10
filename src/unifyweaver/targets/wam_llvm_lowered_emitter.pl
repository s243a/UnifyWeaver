:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_llvm_lowered_emitter.pl — WAM-lowered LLVM IR emission
%
% Emits one LLVM function per deterministic single-clause predicate,
% with the WAM instruction sequence inlined as straight-line basic
% blocks instead of going through the @step switch dispatcher.
%
% Modelled on wam_fsharp_lowered_emitter.pl (1157 lines) and
% wam_rust_lowered_emitter.pl (452 lines). The LLVM port is structurally
% the same: parse WAM text, check lowerability, emit a sequence of basic
% blocks each ending in a branch to the next instruction's block.
%
% Differences from those targets:
%
%   - The host language is LLVM IR, not F#/Rust. Each instruction
%     emits raw IR with unique SSA names (`%inst.<N>.<part>`) and a
%     basic block label (`pc_<N>`) so SSA dominance holds across the
%     chain.
%
%   - Where the @step dispatcher's case body does `ret i1 true`
%     (success → run_loop increments PC and re-enters), the lowered
%     emitter branches to the next instruction's block instead.
%     `ret i1 false` (failure) becomes `br label %lowered_fail`.
%
%   - `wam_inc_pc` calls are dropped — there is no PC to advance,
%     the control flow IS the sequencing.
%
%   - Each lowered function allocates its own %WamState in `entry`
%     and frees it before every `ret` (matches the existing
%     `emit_one_entry_func` calling convention in wam_llvm_target.pl,
%     so the lowered function is a drop-in replacement for the
%     WAM-fallback entry function).
%
% Lowerability gate (mirrors the Rust/F# emitters):
%
%   1. Single clause only (no try_me_else/retry_me_else/trust_me).
%      Multi-clause predicates fall back to the WAM bytecode path
%      where indexing + choice-points are handled.
%
%   2. Every instruction in the body must be in the `supported/1`
%      whitelist. Anything else (call/2, execute/1, get_structure/2,
%      get_list/1, jump/1, cut_ite/0, ...) fails the gate and the
%      predicate falls back to bytecode.
%
% Initial supported set:
%   - get_constant, get_variable, get_value
%   - put_constant, put_variable, put_value, put_structure
%   - set_constant, set_variable, set_value
%   - allocate, deallocate
%   - builtin_call (delegates to @execute_builtin)
%   - proceed, fail

:- module(wam_llvm_lowered_emitter, [
    wam_llvm_lowerable/3,            % +Pred/Arity, +WamCode, -Shape
    wam_llvm_lowerable_with_closure/4, % +Pred/Arity, +WamCode, +ClosureSet,
                                     %  -Shape (M4 — call/execute aware)
    lower_predicate_to_llvm/4,       % +Pred/Arity, +WamCode, +Options, -LLVMCode
    is_deterministic_pred_llvm/1,    % +Instrs
    llvm_lowered_func_name/2,        % +Pred/Arity, -LLVMFuncName
    clause1_instrs/2,                % +Instrs, -Clause1Body (exported for tests)
    call_execute_targets/2,          % +Instrs, -PredArities (M4)
    emit_native_wrapper/2,           % +Pred/Arity, -WrapperCode (M3)
    emit_hybrid_dispatcher/5         % +Pred/Arity, +StartPC, +InstrCount,
                                     % +LabelArraySize, -DispatcherCode (M3)
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../bindings/llvm_wam_bindings', [reg_name_to_index/2]).
:- use_module(wam_ite_structurer, [structure_ite/2]).
:- use_module(wam_clause_chain, [clause_chain/2]).

% wam_llvm_lowerable/3's clauses are interleaved with the T4 helper
% llvm_all_clauses_lowerable/1 (kept next to the multi_clause_n gate it
% serves); declare it discontiguous so SWI does not warn.
:- discontiguous wam_llvm_lowerable/3.

% NB: we DO NOT `:- use_module(wam_llvm_target, ...)` here, because
% wam_llvm_target.pl `:- use_module`s this file in turn. The two
% predicates we borrow from the target — wam_instruction_to_llvm_literal/2
% (atom interning passthrough) and builtin_op_to_id/2 — are called
% explicitly with the `wam_llvm_target:` qualifier so the dependency
% is resolved lazily at call time instead of import time.

% ============================================================================
% WAM-text parser (target-agnostic shape; copied from rust emitter)
% ============================================================================

parse_wam_text(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines(Lines, Instrs).

parse_lines([], []).
parse_lines([Line|Rest], Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_lines(Rest, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  parse_lines(Rest, Instrs)
        ;   instr_from_parts(CleanParts, Instr)
        ->  Instrs = [Instr|RestInstrs],
            parse_lines(Rest, RestInstrs)
        ;   parse_lines(Rest, Instrs)
        )
    ).

instr_from_parts(["get_constant", C, Ai], get_constant(C, Ai)).
instr_from_parts(["get_variable", Xn, Ai], get_variable(Xn, Ai)).
instr_from_parts(["get_value", Xn, Ai], get_value(Xn, Ai)).
instr_from_parts(["get_structure", F, Ai], get_structure(F, Ai)).
instr_from_parts(["get_list", Ai], get_list(Ai)).
instr_from_parts(["unify_variable", Xn], unify_variable(Xn)).
instr_from_parts(["unify_value", Xn], unify_value(Xn)).
instr_from_parts(["unify_constant", C], unify_constant(C)).
instr_from_parts(["put_variable", Xn, Ai], put_variable(Xn, Ai)).
instr_from_parts(["put_value", Xn, Ai], put_value(Xn, Ai)).
instr_from_parts(["put_constant", C, Ai], put_constant(C, Ai)).
instr_from_parts(["put_structure", F, Ai], put_structure(F, Ai)).
instr_from_parts(["put_list", Ai], put_list(Ai)).
instr_from_parts(["set_variable", Xn], set_variable(Xn)).
instr_from_parts(["set_value", Xn], set_value(Xn)).
instr_from_parts(["set_constant", C], set_constant(C)).
instr_from_parts(["call", P, N], call(P, N)).
instr_from_parts(["execute", P], execute(P)).
instr_from_parts(["proceed"], proceed).
instr_from_parts(["fail"], fail).
instr_from_parts(["allocate"], allocate).
instr_from_parts(["deallocate"], deallocate).
instr_from_parts(["builtin_call", Op, Ar], builtin_call(Op, Ar)).
instr_from_parts(["call_foreign", Pred, Ar], call_foreign(Pred, Ar)).
instr_from_parts(["try_me_else", L], try_me_else(L)).
instr_from_parts(["retry_me_else", L], retry_me_else(L)).
instr_from_parts(["trust_me"], trust_me).
instr_from_parts(["jump", L], jump(L)).
instr_from_parts(["cut_ite"], cut_ite).
% ite_use_y_level(true) soft-cut form (the LLVM pipeline's default): a
% get_level Yn snapshots the choice-point level into a permanent register
% in the clause prefix; cut Yn commits to it. In the lowered (single
% basic-block-graph) form there are no tracked choice points, so cut Yn is
% the structural ITE commit (dropped by wam_ite_structurer's is_commit/1)
% and get_level Yn is dead — emitted as a no-op (see emit_instr/4 below).
instr_from_parts(["get_level", Yn], get_level(Yn)).
instr_from_parts(["cut", Yn], cut(Yn)).

% ----------------------------------------------------------------------------
% Label-preserving parse (for if-then-else / negation / once lowering)
%
% The base parse_wam_text/2 above drops label lines (and the structural
% try_me_else / trust_me / jump / cut_ite become no-ops once labels are
% gone), which erases the LElse/LCont boundaries of an ITE block. The
% lowered ITE path needs them, so it parses through parse_wam_text_labeled/2
% which keeps label(Name) markers (as strings, so they unify with the
% string arguments of try_me_else/jump) and then folds the stream with the
% shared wam_ite_structurer. Mirrors the Rust/Clojure labeled parsers.
% ----------------------------------------------------------------------------

parse_wam_text_labeled(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines_labeled(Lines, Instrs).

parse_lines_labeled([], []).
parse_lines_labeled([Line|Rest], Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_lines_labeled(Rest, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelStr),   % strip trailing ':'
            Instrs = [label(LabelStr)|More],
            parse_lines_labeled(Rest, More)
        ;   instr_from_parts(CleanParts, Instr)
        ->  Instrs = [Instr|More],
            parse_lines_labeled(Rest, More)
        ;   parse_lines_labeled(Rest, Instrs)
        )
    ).

% ============================================================================
% Lowerability gate
% ============================================================================

%% wam_llvm_lowerable(+PI, +WamCode, -Shape) is semidet.
%
%  Succeeds iff the predicate's *clause-1 body* is safe to lower to a
%  standalone LLVM function. Shape distinguishes how the dispatcher
%  should wire it up:
%
%    single_clause    — the bytecode has no try_me_else / retry_me_else
%                       / trust_me anywhere; the lowered function is
%                       the whole predicate. No bytecode fallback
%                       needed; the caller's @<pred> wrapper just
%                       delegates to @lowered_<pred>_<arity>.
%
%    multi_clause_c1  — clause 1 is deterministic + supported, but the
%                       bytecode has try_me_else / retry_me_else /
%                       trust_me for additional clauses. The lowered
%                       function is the *fast path*; on failure the
%                       dispatcher falls back to running the full
%                       bytecode (all clauses) through @run_loop.
%                       Mirrors the F# emitter's multi-clause
%                       approach (see wam_fsharp_lowered_emitter.pl).
%
%    clause_chain    — the clauses discriminate on a DISTINCT
%                      first-argument constant (lowering type T5). ALL
%                      clauses are lowered into one function as a
%                      first-argument dispatch; an unbound first argument
%                      defers to the full bytecode (which enumerates), so
%                      this is wired up exactly like multi_clause_c1
%                      (hybrid: native fast path + bytecode fallback).
%                      Takes precedence over multi_clause_c1.
wam_llvm_lowerable(_PI, WamCode, clause_chain) :-
    (   is_list(WamCode) -> Instrs = WamCode
    ;   atom(WamCode) -> parse_wam_text(WamCode, Instrs)
    ;   parse_wam_text(WamCode, Instrs)
    ),
    llvm_clause_chain_lowerable(Instrs).
wam_llvm_lowerable(_PI, WamCode, Shape) :-
    (   is_list(WamCode) -> Instrs = WamCode
    ;   atom(WamCode) -> parse_wam_text(WamCode, Instrs)
    ;   parse_wam_text(WamCode, Instrs)
    ),
    clause1_instrs(Instrs, C1),
    is_deterministic_pred_llvm(C1),
    forall(member(I, C1), supported(I)),
    ( has_choice_point_instrs(Instrs)
    -> ( llvm_all_clauses_lowerable(Instrs)
       -> Shape = multi_clause_n   % T4: every clause lowered inline
       ;  Shape = multi_clause_c1  % T3: clause 1 inline, bytecode for 2+
       )
    ;  Shape = single_clause
    ).

%% llvm_all_clauses_lowerable(+Instrs) is semidet.
%  True when EVERY clause of a multi-clause predicate is a clean supported
%  deterministic body (no inner ITE / call / execute, ends in a terminal) —
%  the T4 (multi_clause_n) condition. Such a predicate lowers all clauses
%  inline (tried in order with a register/trail restore between attempts), so
%  the interpreter is never entered for the predicate. Clauses containing a
%  call / execute keep the multi_clause_c1 path (their callees may not be
%  lowered in this module).
llvm_all_clauses_lowerable(Instrs) :-
    llvm_split_clauses(Instrs, Clauses),
    Clauses = [_, _ | _],
    forall(member(Cl, Clauses),
           ( \+ member(try_me_else(_), Cl),
             \+ member(cut_ite, Cl),
             \+ member(jump(_), Cl),
             \+ member(call(_, _), Cl),
             \+ member(execute(_), Cl),
             forall(member(I, Cl), supported(I)),
             last(Cl, Last), ( Last == proceed ; Last == fail ) )).

% ITE / negation / once: clause 1 contains an internal choice point
% (try_me_else) that is NOT a multi-clause separator but a soft-cut block
% emitted for ( Cond -> Then ; Else ) / \+ Goal / once/1. The base path
% above rejects it (is_deterministic_pred_llvm fails on the try_me_else);
% here we accept it iff the labeled stream folds cleanly into structured
% ite(...) blocks whose every leaf instruction is supported. Such a
% predicate lowers as a single self-contained function (no bytecode
% fallback needed), so Shape = single_clause.
wam_llvm_lowerable(_PI, WamCode, single_clause) :-
    llvm_structured_clause1(WamCode, Structured),
    forall(member(I, Structured), llvm_supported_structured(I)).

%% llvm_clause_chain_lowerable(+Instrs) is semidet.
%  True when the predicate is a distinct-first-argument-constant clause chain
%  (T5) and every clause's remainder is a deterministic, supported body.
llvm_clause_chain_lowerable(Instrs) :-
    clause_chain(Instrs, chain(Guards)),
    forall(member(guard(_, Rem), Guards),
           ( is_deterministic_pred_llvm(Rem),
             forall(member(I, Rem), supported(I)) )).

%% llvm_structured_clause1(+WamCode, -Structured) is semidet.
%
%  Parse the label-preserving stream, take clause 1's body, and fold its
%  if-then-else blocks into ite(Cond,Then,Else) terms via the shared
%  structurer. Fails (so the caller declines) for genuine multi-clause
%  predicates — their try_me_else is the first instruction and/or the
%  structurer cannot match a clean jump-terminated then-path.
llvm_structured_clause1(WamCode, Structured) :-
    ( is_list(WamCode) -> LInstrs = WamCode
    ; parse_wam_text_labeled(WamCode, LInstrs)
    ),
    \+ ( LInstrs = [try_me_else(_)|_] ),    % not a predicate-level multi-clause
    take_to_proceed(LInstrs, C1L),
    structure_ite(C1L, Structured),
    member(ite(_,_,_), Structured),         % there is at least one ITE to lower
    \+ member(try_me_else(_), Structured),  % no choice point survived structuring
    \+ member(retry_me_else(_), Structured),
    \+ member(trust_me, Structured).

%% llvm_supported_structured(+StructuredInstr) is semidet.
%  Recurse through ite(Cond,Then,Else); plain instrs must be supported.
llvm_supported_structured(ite(C, T, E)) :- !,
    forall(member(I, C), llvm_supported_structured(I)),
    forall(member(I, T), llvm_supported_structured(I)),
    forall(member(I, E), llvm_supported_structured(I)).
llvm_supported_structured(I) :- supported(I).

is_deterministic_pred_llvm(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

%% has_choice_point_instrs(+Instrs) is semidet.
%
%  Predicate-level check: does the bytecode contain any choice-point
%  instructions across all clauses? Used to distinguish single-clause
%  from multi-clause predicates after clause-1 lowerability passes.
has_choice_point_instrs(Instrs) :-
    member(I, Instrs),
    ( I = try_me_else(_)
    ; I = retry_me_else(_)
    ; I = trust_me
    ), !.

%% clause1_instrs(+Instrs, -Clause1Body) is det.
%
%  Extract clause 1's instruction body from a full predicate's parsed
%  instruction list, mirroring wam_fsharp_lowered_emitter.pl /
%  wam_rust_lowered_emitter.pl. Strips any switch_on_constant prefix
%  (multi-clause indexing) and the `try_me_else` first-clause marker;
%  the body extends through the first `proceed` or `fail` terminator.
%
%  For single-clause predicates with no try_me_else, returns the full
%  instruction list as-is (it IS clause 1).
clause1_instrs(Instrs0, C1) :-
    % NB: switch_on_constant / switch_on_constant_a2 instructions are
    % silently ignored by parse_wam_text/2 (the parser doesn't have an
    % instr_from_parts/2 clause for them), so the head of Instrs0 here
    % is whatever follows the switch prefix. The strip_* clauses below
    % are belt-and-braces in case a future parser change emits these.
    strip_switch_prefix(Instrs0, Instrs),
    ( Instrs = [try_me_else(_)|Rest]
    -> take_to_proceed(Rest, C1)
    ;  C1 = Instrs
    ).

strip_switch_prefix([switch_on_constant(_,_,_)|Rest], Out) :- !,
    strip_switch_prefix(Rest, Out).
strip_switch_prefix([switch_on_constant_a2(_,_,_)|Rest], Out) :- !,
    strip_switch_prefix(Rest, Out).
strip_switch_prefix(L, L).

take_to_proceed([], []).
take_to_proceed([proceed|_], [proceed]) :- !.
take_to_proceed([fail|_], [fail]) :- !.
take_to_proceed([I|Rest], [I|Out]) :- take_to_proceed(Rest, Out).

%% wam_llvm_lowerable_with_closure(+PI, +WamCode, +ClosureSet, -Shape).
%
%  M4 lowerability check that accounts for call/execute dependencies.
%  ClosureSet is a list of `Pred/Arity` indicators that are known to
%  be lowered in the current module (the closure of lowerable preds).
%  Succeeds with Shape iff:
%
%    1. The standard wam_llvm_lowerable/3 check passes (clause-1 body
%       is deterministic + every instruction is in the supported set).
%    2. Every `call`/`execute` target in the clause-1 body is in
%       ClosureSet — so the emitter can confidently emit
%       `call i1 @lowered_<callee>_<arity>(%vm)` knowing the symbol
%       will be defined at link time.
%
%  Predicates with no call/execute (the M3 lowerable set) trivially
%  pass the closure check — their lowerable status is independent of
%  the closure. This predicate is therefore a strict refinement of
%  wam_llvm_lowerable/3.
wam_llvm_lowerable_with_closure(PI, WamCode, ClosureSet, Shape) :-
    wam_llvm_lowerable(PI, WamCode, Shape),
    % Extract the lowered body's call/execute targets and verify all are in
    % the closure set. clause_chain lowers ALL clauses, so its closure must
    % cover every clause's targets; the other shapes lower only clause 1.
    (   is_list(WamCode) -> Instrs = WamCode
    ;   atom(WamCode) -> parse_wam_text(WamCode, Instrs)
    ;   parse_wam_text(WamCode, Instrs)
    ),
    ( Shape == clause_chain -> Body = Instrs ; clause1_instrs(Instrs, Body) ),
    call_execute_targets(Body, Targets),
    forall(member(T, Targets), memberchk(T, ClosureSet)).

%% call_execute_targets(+Instrs, -Targets) is det.
%
%  Walks an instruction list and returns the de-duplicated set of
%  `Pred/Arity` indicators reached via `call`/`execute`. Used by both
%  the closure analysis and as a hook for future cross-predicate
%  passes.
call_execute_targets(Instrs, Targets) :-
    findall(T, instr_calls_target(Instrs, T), TargetList0),
    sort(TargetList0, Targets).

instr_calls_target(Instrs, Pred/Arity) :-
    member(I, Instrs),
    ( I = call(PredStr, _)
    ; I = execute(PredStr)
    ),
    % Use the target's split_functor_arity/3 — handles names with `/`
    % in them (e.g. integer-division `//2`).
    wam_llvm_target:split_functor_arity(PredStr, NameStr, Arity),
    atom_string(Pred, NameStr).

supported(get_constant(_, _)).
supported(get_variable(_, _)).
supported(get_value(_, _)).
supported(get_structure(_, _)).
supported(get_list(_)).
supported(unify_variable(_)).
supported(unify_value(_)).
supported(unify_constant(_)).
supported(put_constant(_, _)).
supported(put_variable(_, _)).
supported(put_value(_, _)).
supported(put_structure(_, _)).
supported(set_constant(_)).
supported(set_variable(_)).
supported(set_value(_)).
supported(allocate).
supported(deallocate).
supported(get_level(_)).   % no-op in lowered form (soft-cut is structural)
supported(builtin_call(_, _)).
supported(call(_, _)).         % M4 — direct call into another lowered kernel
supported(execute(_)).         % M4 — tail call into another lowered kernel
supported(proceed).
supported(fail).

% ============================================================================
% Function-name generation
% ============================================================================

%% llvm_lowered_func_name(+Pred/Arity, -LLVMName) is det.
%
%  Pure-LLVM identifier safe for `define @<name>`. Prefixed so it never
%  collides with the WAM-fallback entry function (`@<pred>`), which lets
%  external drivers link against either one explicitly.
llvm_lowered_func_name(Pred/Arity, Name) :-
    atom_string(Pred, PredStr),
    sanitize_llvm_ident(PredStr, SanePredStr),
    format(atom(Name), 'lowered_~w_~w', [SanePredStr, Arity]).

sanitize_llvm_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(llvm_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

llvm_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
llvm_safe_code(_, 0'_).

% ============================================================================
% Top-level: emit one LLVM function for the predicate
% ============================================================================

%% lower_predicate_to_llvm(+PI, +WamCode, +Options, -LLVMCode) is det.
%
%  Caller must have already verified wam_llvm_lowerable/3.
%  Emits a function for *clause 1 only* — for multi-clause predicates,
%  clauses 2+ are reached via dispatcher fallback to the bytecode path.
%
%  M4 signature change: the kernel now takes a caller-supplied
%  `%WamState*` and reads its argument registers (A1..AN) from there.
%  No state allocation or free inside the kernel — that's the
%  responsibility of the public entry `@<pred>` wrapper.
%
%  This makes lowered-to-lowered `call`/`execute` cheap: the caller
%  passes its shared state, the callee operates on it, bindings flow
%  through the trail naturally. Without this change, every nested
%  call would need to package args as `%Value` and allocate a fresh
%  state, defeating the perf benefit and breaking binding propagation.
% ITE / negation / once predicates take the structured-emit path (basic
% blocks with a soft-cut commit and trail-rollback else branch); everything
% else takes the straight-line clause-1 path below, which is byte-for-byte
% unchanged.
lower_predicate_to_llvm(PI, WamCode, _Options, LLVMCode) :-
    (   is_list(WamCode) -> CCInstrs = WamCode
    ;   parse_wam_text(WamCode, CCInstrs)
    ),
    llvm_clause_chain_lowerable(CCInstrs), !,
    lower_clause_chain_to_llvm(PI, CCInstrs, LLVMCode).
lower_predicate_to_llvm(PI, WamCode, _Options, LLVMCode) :-
    llvm_structured_clause1(WamCode, Structured), !,
    lower_ite_predicate_to_llvm(PI, Structured, LLVMCode).
lower_predicate_to_llvm(PI, WamCode, _Options, LLVMCode) :-
    ( is_list(WamCode) -> MCInstrs = WamCode ; parse_wam_text(WamCode, MCInstrs) ),
    llvm_all_clauses_lowerable(MCInstrs), !,
    lower_multi_clause_n_to_llvm(PI, MCInstrs, LLVMCode).
lower_predicate_to_llvm(PI, WamCode, _Options, LLVMCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    llvm_lowered_func_name(Pred/Arity, FuncName),
    (   is_list(WamCode) -> Instrs = WamCode
    ;   parse_wam_text(WamCode, Instrs)
    ),
    % Lower the clause-1 body only. For single-clause preds this is
    % the whole instruction list; for multi-clause preds it's the body
    % between try_me_else and the first proceed/fail.
    clause1_instrs(Instrs, C1Instrs),
    % Emit one basic block per instruction, then the shared
    % succeed/fail epilogue blocks.
    emit_all_instrs(C1Instrs, 0, BodyBlocks),
    atomic_list_concat(BodyBlocks, '\n', BodyStr),
    format(atom(LLVMCode),
'; === lowered kernel: ~w/~w ===
; M4: takes a caller-supplied %WamState* — does not allocate or free
; state. Args are read from the shared state\'s A1..A~w registers,
; which the caller must have populated via preceding put_* instrs
; (or, for the outermost call, the @<pred> public-entry wrapper).
; Each WAM instruction lives in its own basic block named pc_<N>;
; success branches to the next block, failure branches to
; %lowered_fail. The final proceed branches to %lowered_succeed.
define i1 @~w(%WamState* %vm) {
entry:
  br label %pc_0

~w

lowered_succeed:
  ret i1 true

lowered_fail:
  ret i1 false
}',
        [Pred, Arity, Arity, FuncName, BodyStr]).

% ============================================================================
% T5: multi-clause as a first-argument dispatch (wam_clause_chain)
%
%  The clauses discriminate on a DISTINCT first-argument constant, so at most
%  one matches a *bound* first argument. We deref A1 once: if it is unbound we
%  branch to %lowered_fail (the hybrid wrapper then re-runs the full bytecode,
%  which enumerates every clause, binding A1 in turn). If it is bound, control
%  flows through each clause's head get_constant — a mismatch branches to the
%  NEXT clause's first block, a match runs that clause's body. A leaf failure
%  or a fall-through past the last clause returns false (sound: distinct
%  discriminators mean no other clause could have matched anyway, and the
%  bytecode fallback then also fails).
%
%  Each clause is emitted in FULL (its leading get_constant is kept): on the
%  matching fast path it re-confirms the already-bound value, and its mismatch
%  edge is exactly what chains to the next clause. Block numbering is shared
%  and monotonic across clauses (pc_<N>), so no labels collide.
% ============================================================================

%% lower_clause_chain_to_llvm(+PI, +Instrs, -LLVMCode) is det.
lower_clause_chain_to_llvm(PI, Instrs, LLVMCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    llvm_lowered_func_name(Pred/Arity, FuncName),
    parse_reg('A1', A1Idx),
    llvm_split_clauses(Instrs, Clauses),
    emit_clause_chain_blocks(Clauses, 0, ClauseBlocksList),
    atomic_list_concat(ClauseBlocksList, '\n\n', BodyStr),
    format(atom(LLVMCode),
'; === lowered kernel (T5 first-argument dispatch): ~w/~w ===
; All clauses are lowered into one function. The dispatcher derefs A1 once;
; an unbound first argument branches to %lowered_fail so the hybrid wrapper
; re-runs the full bytecode (which enumerates every clause). A bound A1 flows
; through a chain of get_constant heads: a non-matching head branches to the
; next clause, the matching head runs its body.
define i1 @~w(%WamState* %vm) {
entry:
  br label %t5_dispatch

t5_dispatch:
  %t5.a1 = call %Value @wam_get_reg_deref(%WamState* %vm, i32 ~w)
  %t5.unb = call i1 @value_is_unbound(%Value %t5.a1)
  br i1 %t5.unb, label %lowered_fail, label %pc_0

~w

lowered_succeed:
  ret i1 true

lowered_fail:
  ret i1 false
}',
        [Pred, Arity, FuncName, A1Idx, BodyStr]).

%% emit_clause_chain_blocks(+Clauses, +StartN, -BlocksList) is det.
%  Emit each clause's basic blocks. The clause's leading instruction (its head
%  get_constant) is emitted via emit_instr_f so its failure edge points at the
%  next clause's first block (pc_<NextStartN>) instead of %lowered_fail; the
%  rest of the clause body emits normally (failures → %lowered_fail). Block
%  numbers run monotonically across clauses so nothing collides.
emit_clause_chain_blocks([], _, []).
emit_clause_chain_blocks([Clause|Rest], StartN, [ClauseStr|More]) :-
    Clause = [First|RestInstrs],
    length(Clause, Len),
    NextStartN is StartN + Len,
    ( Rest == []
    ->  FailLabel = lowered_fail
    ;   format(atom(FailLabel), 'pc_~w', [NextStartN])
    ),
    ( RestInstrs == []
    ->  NextWithin = lowered_succeed
    ;   N1 is StartN + 1, format(atom(NextWithin), 'pc_~w', [N1])
    ),
    emit_instr_f(First, StartN, NextWithin, FailLabel, FirstBlock),
    N2 is StartN + 1,
    emit_all_instrs(RestInstrs, N2, RestBlocks),
    atomic_list_concat([FirstBlock|RestBlocks], '\n', ClauseStr),
    emit_clause_chain_blocks(Rest, NextStartN, More).

% ============================================================================
% T4: multi-clause, all clauses inline (multi_clause_n)
%
%  Generalises the T5 clause-chain emit to predicates that do NOT discriminate
%  on a distinct first-argument constant: every clause is lowered inline and
%  tried in order, but — unlike T5, where only the head get_constant decides
%  the clause — ANY instruction failure rolls back to the next clause. Because
%  a partially-run clause may have clobbered the A-registers (put_* / get_value)
%  and bound variables (trailed), each retry first restores the entry register
%  snapshot (memcpy, exactly as a bytecode try_me_else choice point saves them)
%  and unwinds the trail to the entry mark. The first clause runs on the fresh
%  entry state; clause K (K>1) is preceded by a clause_K_restore block. The
%  last clause's failures go to %lowered_fail (the hybrid wrapper then re-runs
%  the bytecode, which also fails — sound). First-solution / deterministic-
%  prefix semantics, strictly more than multi_clause_c1.
% ============================================================================

%% lower_multi_clause_n_to_llvm(+PI, +Instrs, -LLVMCode) is det.
lower_multi_clause_n_to_llvm(PI, Instrs, LLVMCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    llvm_lowered_func_name(Pred/Arity, FuncName),
    llvm_split_clauses(Instrs, Clauses),
    emit_t4_clauses(Clauses, 0, 1, ClauseBlocks),
    atomic_list_concat(ClauseBlocks, '\n', ClauseStr),
    format(atom(LLVMCode),
'; === lowered kernel (T4 all-clauses inline): ~w/~w ===
; Every clause is lowered; any instruction failure restores the entry
; register snapshot + trail mark and falls through to the next clause.
define i1 @~w(%WamState* %vm) {
entry:
  %t4.regbuf = alloca [64 x %Value]
  %t4.regbuf_raw = bitcast [64 x %Value]* %t4.regbuf to i8*
  %t4.src_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %t4.src_raw = bitcast %Value* %t4.src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t4.regbuf_raw, i8* %t4.src_raw, i64 1024, i1 false)
  %t4.tm_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %t4.tmv = load i32, i32* %t4.tm_ptr
  br label %pc_0

~w

lowered_succeed:
  ret i1 true

lowered_fail:
  ret i1 false
}',
        [Pred, Arity, FuncName, ClauseStr]).

%% emit_t4_clauses(+Clauses, +StartN, +Idx, -Blocks) is det.
%  Emit each clause's blocks. Clause Idx (1-based) starts at block pc_<StartN>;
%  every instruction's failure edge points at the next clause's restore block
%  (clause_<Idx+1>_restore), or %lowered_fail for the last clause. Clauses 2+
%  are preceded by a restore block that re-snapshots the registers + unwinds
%  the trail before re-entering at pc_<StartN>.
emit_t4_clauses([], _, _, []).
emit_t4_clauses([Cl | Rest], StartN, Idx, [Block | More]) :-
    length(Cl, Len),
    NextStartN is StartN + Len,
    NextIdx is Idx + 1,
    ( Rest == []
    ->  FailLabel = lowered_fail
    ;   format(atom(FailLabel), 'clause_~w_restore', [NextIdx])
    ),
    emit_all_instrs_f(Cl, StartN, FailLabel, InstrBlocks),
    atomic_list_concat(InstrBlocks, '\n', InstrStr),
    ( Idx =:= 1
    ->  Block = InstrStr
    ;   t4_restore_block(Idx, StartN, RestoreStr),
        atomic_list_concat([RestoreStr, InstrStr], '\n', Block)
    ),
    emit_t4_clauses(Rest, NextStartN, NextIdx, More).

%% emit_all_instrs_f(+Instrs, +StartN, +FailLabel, -Blocks) is det.
%  Like emit_all_instrs, but every instruction's failure edge is redirected to
%  FailLabel (instead of the default %lowered_fail).
emit_all_instrs_f([], _, _, []).
emit_all_instrs_f([I | Rest], N, FailLabel, [Block | More]) :-
    next_label(Rest, N, NextLabel),
    emit_instr_f(I, N, NextLabel, FailLabel, Block),
    N1 is N + 1,
    emit_all_instrs_f(Rest, N1, FailLabel, More).

%% t4_restore_block(+Idx, +StartN, -Str) — restore the entry register snapshot
%  and trail mark, then branch into clause Idx at pc_<StartN>.
t4_restore_block(Idx, StartN, Str) :-
    format(atom(Str),
'clause_~w_restore:
  %t4.dst~w = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %t4.dst~w_raw = bitcast %Value* %t4.dst~w to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t4.dst~w_raw, i8* %t4.regbuf_raw, i64 1024, i1 false)
  call void @unwind_trail(%WamState* %vm, i32 %t4.tmv)
  br label %pc_~w',
        [Idx, Idx, Idx, Idx, Idx, StartN]).

%% llvm_split_clauses(+Instrs, -Clauses) is semidet.
%  Split the parsed instruction list (which opens with try_me_else) at the
%  choice-point separators into per-clause instruction lists, each trimmed to
%  its terminal proceed/fail. Mirrors wam_clause_chain's split_clauses.
llvm_split_clauses([try_me_else(_)|Rest], [Clause|More]) :-
    llvm_collect_clause(Rest, C, After),
    take_to_proceed(C, Clause),
    llvm_split_more(After, More).

llvm_split_more([], []).
llvm_split_more([retry_me_else(_)|Rest], [Clause|More]) :- !,
    llvm_collect_clause(Rest, C, After),
    take_to_proceed(C, Clause),
    llvm_split_more(After, More).
llvm_split_more([trust_me|Rest], [Clause|More]) :- !,
    llvm_collect_clause(Rest, C, After),
    take_to_proceed(C, Clause),
    llvm_split_more(After, More).

llvm_collect_clause([], [], []).
llvm_collect_clause([retry_me_else(L)|Rest], [], [retry_me_else(L)|Rest]) :- !.
llvm_collect_clause([trust_me|Rest], [], [trust_me|Rest]) :- !.
llvm_collect_clause([I|Rest], [I|More], After) :-
    llvm_collect_clause(Rest, More, After).

% ============================================================================
% Structured (if-then-else / negation / once) emission
%
% Where the straight-line path chains one basic block per instruction with
% every failure branching to %lowered_fail, the structured path emits an
% ite(Cond,Then,Else) block as:
%
%   ite_<K>:                              ; capture pre-condition trail mark
%     %ite_<K>.tmv = load <trail size>
%     br label %<first cond block>
%   <cond blocks: failure → ite_<K>_else, success → first then block>
%   <then blocks: failure → outer fail, success → continuation>
%   ite_<K>_else:                         ; condition failed: roll back, run else
%     call void @unwind_trail(%vm, i32 %ite_<K>.tmv)
%     br label %<first else block>
%   <else blocks: failure → outer fail, success → continuation>
%
% Because the WAM state is a single mutable %WamState* (registers, heap and
% trail are all mutated in place and every binding is trailed), unwinding
% the trail to the pre-condition mark fully restores any partial bindings a
% failed condition made — no phi nodes or register snapshots are needed
% (mirrors the Rust emitter's vm.unwind_trail_to and the Go register copy).
%
% The per-instruction emitters are reused verbatim through emit_instr_f/5,
% which simply redirects their hard-coded `%lowered_fail` branch to the
% supplied failure label. Non-ITE output is therefore byte-identical.
% ============================================================================

%% lower_ite_predicate_to_llvm(+PI, +Structured, -LLVMCode) is det.
lower_ite_predicate_to_llvm(PI, Structured, LLVMCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    llvm_lowered_func_name(Pred/Arity, FuncName),
    nb_setval(wam_llvm_lowered_uid, 0),
    emit_seq(Structured, 'lowered_fail', 'lowered_succeed', BodyStr, FirstLabel),
    format(atom(LLVMCode),
'; === lowered kernel (if-then-else / negation / once): ~w/~w ===
; M4: takes a caller-supplied %WamState*. Each plain instruction lives in
; its own basic block (pc_<N>); each ( Cond -> Then ; Else ) / \\+ / once
; block lives in ite_<K>* blocks with a trail-rollback else branch. The
; condition\'s commit (cut_ite / !/0) is structural and not emitted.
define i1 @~w(%WamState* %vm) {
entry:
  br label %~w

~w

lowered_succeed:
  ret i1 true

lowered_fail:
  ret i1 false
}',
        [Pred, Arity, FuncName, FirstLabel, BodyStr]).

%% fresh_uid(-Uid) — monotonic counter for unique block / SSA names.
fresh_uid(Uid) :-
    nb_getval(wam_llvm_lowered_uid, Uid),
    Uid1 is Uid + 1,
    nb_setval(wam_llvm_lowered_uid, Uid1).

%% item_entry_label(+Item, +Uid, -Label) — block a predecessor branches to.
item_entry_label(ite(_,_,_), Uid, Label) :- !,
    format(atom(Label), 'ite_~w', [Uid]).
item_entry_label(_Plain, Uid, Label) :-
    format(atom(Label), 'pc_~w', [Uid]).

%% emit_seq(+Items, +FailLabel, +ContLabel, -Code, -FirstLabel) is det.
%
%  Chain a list of structured items (plain instrs and ite(...) blocks) into
%  basic blocks. Each item's success flows to the next item's entry block,
%  the last to ContLabel; each item's failure flows to FailLabel. FirstLabel
%  is the entry block of the whole chain (ContLabel when the list is empty),
%  so the caller can branch into it.
emit_seq([], _Fail, ContLabel, '', ContLabel) :- !.
emit_seq([Item|Rest], Fail, ContLabel, Code, FirstLabel) :-
    fresh_uid(Uid),
    item_entry_label(Item, Uid, FirstLabel),
    emit_seq(Rest, Fail, ContLabel, RestCode, NextLabel),
    emit_item(Item, Uid, Fail, NextLabel, ItemCode),
    ( RestCode == ''
    -> Code = ItemCode
    ;  atomic_list_concat([ItemCode, RestCode], '\n', Code)
    ).

%% emit_item(+Item, +Uid, +FailLabel, +NextLabel, -Code) is det.
%
%  Plain instruction → one basic block (pc_<Uid>) via emit_instr_f/5.
%  ite(Cond,Then,Else) → the entry/cond/then/else block group described
%  in the section header above.
emit_item(ite(Cond, Then, Else), Uid, Fail, Cont, Code) :- !,
    format(atom(EntryLabel), 'ite_~w', [Uid]),
    format(atom(ElseLabel),  'ite_~w_else', [Uid]),
    % then/else flow to the shared continuation; their failures escape to
    % the enclosing failure label.
    emit_seq(Then, Fail, Cont, ThenCode, ThenFirst),
    emit_seq(Else, Fail, Cont, ElseCode, ElseFirst),
    % the condition's failure goes to the else block; success falls into
    % the first then block.
    emit_seq(Cond, ElseLabel, ThenFirst, CondCode, CondFirst),
    % entry block: snapshot the trail length (struct field 9) before the
    % condition runs, then enter the condition.
    format(atom(EntryBlock),
'~w:
  ; ( Cond -> Then ; Else ) / negation / once  [block ~w]
  %ite_~w.tm = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %ite_~w.tmv = load i32, i32* %ite_~w.tm
  br label %~w',
        [EntryLabel, Uid, Uid, Uid, Uid, CondFirst]),
    % else preamble: roll back any bindings the failed condition made, then
    % enter the else branch.
    format(atom(ElseBlock),
'~w:
  ; condition failed -- unwind to pre-condition trail mark, run else
  call void @unwind_trail(%WamState* %vm, i32 %ite_~w.tmv)
  br label %~w',
        [ElseLabel, Uid, ElseFirst]),
    atomic_list_concat(
        [EntryBlock, CondCode, ThenCode, ElseBlock, ElseCode], '\n', Code).
emit_item(Instr, Uid, Fail, Next, Code) :-
    emit_instr_f(Instr, Uid, Next, Fail, Code).

%% emit_instr_f(+Instr, +N, +Next, +FailLabel, -Code) is det.
%
%  Emit a single instruction's basic block, redirecting its failure branch
%  to FailLabel. Reuses the straight-line emit_instr/4 verbatim and rewrites
%  its hard-coded `%lowered_fail` target. When FailLabel is the default
%  'lowered_fail' this is byte-identical to emit_instr/4 (no rewrite).
emit_instr_f(Instr, N, Next, lowered_fail, Code) :- !,
    emit_instr(Instr, N, Next, Code).
emit_instr_f(Instr, N, Next, FailLabel, Code) :-
    emit_instr(Instr, N, Next, Code0),
    atom_concat('%', FailLabel, FailRef),
    atomic_list_concat(Parts, '%lowered_fail', Code0),
    atomic_list_concat(Parts, FailRef, Code).

build_param_list(0, "") :- !.
build_param_list(Arity, ParamList) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>format(atom(S), "%Value %a~w", [I]), Indices, Parts),
    atomic_list_concat(Parts, ', ', ParamList).

build_arg_setup(0, "") :- !.
build_arg_setup(Arity, Setup) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(
        RegIdx is I - 1,
        format(atom(S),
            '  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %a~w)',
            [RegIdx, I])
    ), Indices, Parts),
    atomic_list_concat(Parts, '\n', Setup).

% ============================================================================
% Per-instruction emission
%
% Each instruction is emitted as:
%
%   pc_<N>:
%     ; <opcode>
%     ...inline IR using @wam_* helpers (all alwaysinline'd as of
%        PR #2539 so this collapses to the same code @step would
%        produce for the same instruction)
%     br label %<next_label>
%
% where <next_label> is `pc_<N+1>` if there's a successor, or
% `lowered_succeed` for the final proceed instruction.
%
% Failure paths inside an instruction emit `br label %lowered_fail`.
% The `lowered_fail` block is defined by the wrapper above and frees
% %vm before returning false.
% ============================================================================

emit_all_instrs([], _, []).
emit_all_instrs([I|Rest], N, [Block|RestBlocks]) :-
    next_label(Rest, N, NextLabel),
    emit_instr(I, N, NextLabel, Block),
    N1 is N + 1,
    emit_all_instrs(Rest, N1, RestBlocks).

% Successor label for instruction N: the next instruction's block,
% or `lowered_succeed` if this is the last instruction. (Note:
% `proceed` overrides this and always jumps to lowered_succeed.)
next_label([], _, 'lowered_succeed') :- !.
next_label(_, N, Lbl) :-
    N1 is N + 1,
    format(atom(Lbl), 'pc_~w', [N1]).

% --- proceed: terminal success ---
emit_instr(proceed, N, _Next, Block) :- !,
    format(atom(Block),
'pc_~w:
  ; proceed
  br label %lowered_succeed', [N]).

% --- fail: terminal failure ---
emit_instr(fail, N, _Next, Block) :- !,
    format(atom(Block),
'pc_~w:
  ; fail
  br label %lowered_fail', [N]).

% --- get_level Yn: snapshot CP level (no-op in lowered form) ---
% The soft cut is realised structurally by the basic-block layout (a failed
% condition branches to the else block), so there is no choice-point level
% to capture. Emit an empty block that falls through to the next instr.
emit_instr(get_level(YnStr), N, Next, Block) :- !,
    format(atom(Block),
'pc_~w:
  ; get_level ~w (no-op: soft cut is structural in the lowered form)
  br label %~w',
        [N, YnStr, Next]).

% --- get_variable Xn, Ai: copy reg Ai → reg Xn ---
emit_instr(get_variable(XnStr, AiStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    parse_reg(AiStr, AiIdx),
    format(atom(Block),
'pc_~w:
  ; get_variable ~w, ~w
  %gv.~w.val = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %gv.~w.val)
  br label %~w',
        [N, XnStr, AiStr,
         N, AiIdx,
         XnIdx,
         XnIdx, N,
         Next]).

% --- get_value Xn, Ai: unify Xn ≡ Ai ---
emit_instr(get_value(XnStr, AiStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    parse_reg(AiStr, AiIdx),
    format(atom(Block),
'pc_~w:
  ; get_value ~w, ~w
  %gval.~w.va = call %Value @wam_get_reg_deref(%WamState* %vm, i32 ~w)
  %gval.~w.vx = call %Value @wam_get_reg_deref(%WamState* %vm, i32 ~w)
  %gval.~w.aub = call i1 @value_is_unbound(%Value %gval.~w.va)
  br i1 %gval.~w.aub, label %pc_~w_bind_a, label %pc_~w_check_x

pc_~w_bind_a:
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_bind_reg(%WamState* %vm, i32 ~w, %Value %gval.~w.vx)
  br label %~w

pc_~w_check_x:
  %gval.~w.xub = call i1 @value_is_unbound(%Value %gval.~w.vx)
  br i1 %gval.~w.xub, label %pc_~w_bind_x, label %pc_~w_check_eq

pc_~w_bind_x:
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_bind_reg(%WamState* %vm, i32 ~w, %Value %gval.~w.va)
  br label %~w

pc_~w_check_eq:
  %gval.~w.eq = call i1 @value_equals(%Value %gval.~w.va, %Value %gval.~w.vx)
  br i1 %gval.~w.eq, label %~w, label %lowered_fail',
        [N, XnStr, AiStr,
         N, AiIdx,
         N, XnIdx,
         N, N,
         N, N, N,
         N,
         AiIdx,
         AiIdx, N,
         Next,
         N,
         N, N,
         N, N, N,
         N,
         XnIdx,
         XnIdx, N,
         Next,
         N,
         N, N, N,
         N, Next]).

% --- get_constant Const, Ai: bind Ai if unbound; check equality otherwise ---
emit_instr(get_constant(CStr, AiStr), N, Next, Block) :- !,
    parse_reg(AiStr, AiIdx),
    parse_constant(CStr, Tag, Payload),
    format(atom(Block),
'pc_~w:
  ; get_constant ~w, ~w
  %gc.~w.cur = call %Value @wam_get_reg_deref(%WamState* %vm, i32 ~w)
  %gc.~w.unb = call i1 @value_is_unbound(%Value %gc.~w.cur)
  br i1 %gc.~w.unb, label %pc_~w_bind, label %pc_~w_check

pc_~w_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  %gc.~w.bv0 = insertvalue %Value undef, i32 ~w, 0
  %gc.~w.bv = insertvalue %Value %gc.~w.bv0, i64 ~w, 1
  call void @wam_bind_reg(%WamState* %vm, i32 ~w, %Value %gc.~w.bv)
  br label %~w

pc_~w_check:
  %gc.~w.ev0 = insertvalue %Value undef, i32 ~w, 0
  %gc.~w.ev = insertvalue %Value %gc.~w.ev0, i64 ~w, 1
  %gc.~w.eq = call i1 @value_equals(%Value %gc.~w.cur, %Value %gc.~w.ev)
  br i1 %gc.~w.eq, label %~w, label %lowered_fail',
        [N, CStr, AiStr,
         N, AiIdx,
         N, N,
         N, N, N,
         N,
         AiIdx,
         N, Tag,
         N, N, Payload,
         AiIdx, N,
         Next,
         N,
         N, Tag,
         N, N, Payload,
         N, N, N,
         N, Next]).

% --- put_constant Const, Ai: set Ai to literal value ---
emit_instr(put_constant(CStr, AiStr), N, Next, Block) :- !,
    parse_reg(AiStr, AiIdx),
    parse_constant(CStr, Tag, Payload),
    format(atom(Block),
'pc_~w:
  ; put_constant ~w, ~w
  %pcst.~w.v0 = insertvalue %Value undef, i32 ~w, 0
  %pcst.~w.v = insertvalue %Value %pcst.~w.v0, i64 ~w, 1
  %pcst.~w.old = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %pcst.~w.v)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pcst.~w.old, %Value %pcst.~w.v)
  br label %~w',
        [N, CStr, AiStr,
         N, Tag,
         N, N, Payload,
         N, AiIdx,
         AiIdx,
         AiIdx, N,
         N, N,
         Next]).

% --- put_variable Xn, Ai: fresh Unbound heap cell, Ref into both regs ---
emit_instr(put_variable(XnStr, AiStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    parse_reg(AiStr, AiIdx),
    format(atom(Block),
'pc_~w:
  ; put_variable ~w, ~w
  %pv.~w.unb = call %Value @value_unbound(i8* null)
  %pv.~w.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %pv.~w.unb)
  %pv.~w.ref = call %Value @value_ref(i32 %pv.~w.addr)
  %pv.~w.old = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %pv.~w.ref)
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %pv.~w.ref)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pv.~w.old, %Value %pv.~w.ref)
  br label %~w',
        [N, XnStr, AiStr,
         N,
         N, N,
         N, N,
         N, AiIdx,
         XnIdx,
         XnIdx, N,
         AiIdx,
         AiIdx, N,
         N, N,
         Next]).

% --- put_value Xn, Ai: copy reg Xn → reg Ai ---
emit_instr(put_value(XnStr, AiStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    parse_reg(AiStr, AiIdx),
    format(atom(Block),
'pc_~w:
  ; put_value ~w, ~w
  %pvl.~w.val = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)
  %pvl.~w.old = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %pvl.~w.val)
  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %pvl.~w.old, %Value %pvl.~w.val)
  br label %~w',
        [N, XnStr, AiStr,
         N, XnIdx,
         N, AiIdx,
         AiIdx,
         AiIdx, N,
         N, N,
         Next]).

% --- put_structure F/Arity, Ai: allocate %Compound on arena, push WriteCtx ---
%
% Mirrors the @step case at wam_llvm_target.pl 'put_structure', but
% with %op1 (functor string ptr) replaced by the module-level
% @.fn_<sanitized> global the WAM target emits for every functor.
% Built by concatenating per-line format calls so the (heavy)
% placeholder accounting stays local and verifiable.
emit_instr(put_structure(FStr, AiStr), N, Next, Block) :- !,
    parse_reg(AiStr, AiIdx),
    parse_functor(FStr, FunctorName, FArity),
    % Bijective hex-escape encoding shared with the bytecode path
    % (wam_llvm_target.pl:4052). `+` → `_2B`, `.` → `_2E`, etc.
    wam_llvm_target:sanitize_functor_for_llvm(FunctorName, SaneFN),
    % Register the functor so the module-assembly pass emits the
    % matching `@.fn_<sane>` private constant.
    wam_llvm_target:register_functor_string(FunctorName),
    string_length(FunctorName, FNLen0),
    FNLen is FNLen0 + 1,    % +1 for the NUL terminator
    ArgsBytes is FArity * 16,
    format(atom(L0), 'pc_~w:', [N]),
    format(atom(L1), '  ; put_structure ~w, ~w', [FStr, AiStr]),
    format(atom(L2),
'  %ps.~w.fn_ptr = getelementptr [~w x i8], [~w x i8]* @.fn_~w, i32 0, i32 0',
        [N, FNLen, FNLen, SaneFN]),
    L3 = '  call void @wam_arena_ensure()',
    format(atom(L4),
'  %ps.~w.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64',
        [N]),
    format(atom(L5),
'  %ps.~w.cp_mem = call i8* @wam_arena_alloc(i64 %ps.~w.cp_size)',
        [N, N]),
    format(atom(L6),
'  %ps.~w.cp = bitcast i8* %ps.~w.cp_mem to %Compound*',
        [N, N]),
    format(atom(L7),
'  %ps.~w.fn_slot = getelementptr %Compound, %Compound* %ps.~w.cp, i32 0, i32 0',
        [N, N]),
    format(atom(L8),
'  store i8* %ps.~w.fn_ptr, i8** %ps.~w.fn_slot',
        [N, N]),
    format(atom(L9),
'  %ps.~w.ar_slot = getelementptr %Compound, %Compound* %ps.~w.cp, i32 0, i32 1',
        [N, N]),
    format(atom(L10),
'  store i32 ~w, i32* %ps.~w.ar_slot',
        [FArity, N]),
    format(atom(L11),
'  %ps.~w.args_mem = call i8* @wam_arena_alloc(i64 ~w)',
        [N, ArgsBytes]),
    format(atom(L12),
'  %ps.~w.args = bitcast i8* %ps.~w.args_mem to %Value*',
        [N, N]),
    format(atom(L13),
'  %ps.~w.args_slot = getelementptr %Compound, %Compound* %ps.~w.cp, i32 0, i32 2',
        [N, N]),
    format(atom(L14),
'  store %Value* %ps.~w.args, %Value** %ps.~w.args_slot',
        [N, N]),
    format(atom(L15),
'  %ps.~w.cp_i64 = ptrtoint %Compound* %ps.~w.cp to i64',
        [N, N]),
    format(atom(L16),
'  %ps.~w.val0 = insertvalue %Value undef, i32 3, 0',
        [N]),
    format(atom(L17),
'  %ps.~w.val = insertvalue %Value %ps.~w.val0, i64 %ps.~w.cp_i64, 1',
        [N, N, N]),
    format(atom(L18),
'  %ps.~w.old = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)',
        [N, AiIdx]),
    format(atom(L19),
'  call void @wam_trail_binding(%WamState* %vm, i32 ~w)',
        [AiIdx]),
    format(atom(L20),
'  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %ps.~w.val)',
        [AiIdx, N]),
    format(atom(L21),
'  call void @wam_bind_through_if_unbound_ref(%WamState* %vm, %Value %ps.~w.old, %Value %ps.~w.val)',
        [N, N]),
    format(atom(L22),
'  call void @wam_push_write_ctx(%WamState* %vm, i32 ~w)',
        [FArity]),
    format(atom(L23), '  br label %~w', [Next]),
    atomic_list_concat(
        [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10,
         L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21, L22, L23],
        '\n', Block).

% --- set_value Xn: append reg Xn to current WriteCtx ---
emit_instr(set_value(XnStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    format(atom(Block),
'pc_~w:
  ; set_value ~w
  %sve.~w.val = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %sve.~w.val)
  br label %~w',
        [N, XnStr,
         N, XnIdx,
         N,
         Next]).

% --- set_constant Const: append literal Value to current WriteCtx ---
emit_instr(set_constant(CStr), N, Next, Block) :- !,
    parse_constant(CStr, Tag, Payload),
    format(atom(Block),
'pc_~w:
  ; set_constant ~w
  %sc.~w.v0 = insertvalue %Value undef, i32 ~w, 0
  %sc.~w.v = insertvalue %Value %sc.~w.v0, i64 ~w, 1
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %sc.~w.v)
  br label %~w',
        [N, CStr,
         N, Tag,
         N, N, Payload,
         N,
         Next]).

% --- set_variable Xn: append a fresh unbound Ref and store in reg Xn ---
emit_instr(set_variable(XnStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    format(atom(Block),
'pc_~w:
  ; set_variable ~w
  %sv.~w.unb = call %Value @value_unbound(i8* null)
  %sv.~w.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %sv.~w.unb)
  %sv.~w.ref = call %Value @value_ref(i32 %sv.~w.addr)
  call void @wam_trail_binding(%WamState* %vm, i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %sv.~w.ref)
  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %sv.~w.ref)
  br label %~w',
        [N, XnStr,
         N,
         N, N,
         N, N,
         XnIdx,
         XnIdx, N,
         N,
         Next]).

% --- allocate: push environment frame (snapshot Y-regs into y_save) ---
%
% Same per-line construction pattern as put_structure — each line's
% format/3 takes ≤ 4 args so the placeholder count is locally
% verifiable. Functionally equivalent to the @step 'allocate' case body.
emit_instr(allocate, N, Next, Block) :- !,
    format(atom(L0), 'pc_~w:', [N]),
    L1 = '  ; allocate (env frame push: type=0, save CP, snapshot regs[16..63])',
    L_ensure = '  call void @wam_stack_ensure_capacity(%WamState* %vm)',
    format(atom(L2),
'  %al.~w.ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3',
        [N]),
    format(atom(L3),
'  %al.~w.ss = load i32, i32* %al.~w.ss_ptr',
        [N, N]),
    format(atom(L4),
'  %al.~w.stack_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 2',
        [N]),
    format(atom(L5),
'  %al.~w.stack = load %StackEntry*, %StackEntry** %al.~w.stack_ptr',
        [N, N]),
    format(atom(L6),
'  %al.~w.entry = getelementptr %StackEntry, %StackEntry* %al.~w.stack, i32 %al.~w.ss',
        [N, N, N]),
    format(atom(L7),
'  %al.~w.type_ptr = getelementptr %StackEntry, %StackEntry* %al.~w.entry, i32 0, i32 0',
        [N, N]),
    format(atom(L8),
'  store i32 0, i32* %al.~w.type_ptr',
        [N]),
    format(atom(L9),
'  %al.~w.cp_save = call i32 @wam_get_cp(%WamState* %vm)',
        [N]),
    format(atom(L10),
'  %al.~w.cp_i64 = sext i32 %al.~w.cp_save to i64',
        [N, N]),
    format(atom(L11),
'  %al.~w.aux_ptr = getelementptr %StackEntry, %StackEntry* %al.~w.entry, i32 0, i32 1',
        [N, N]),
    format(atom(L12),
'  store i64 %al.~w.cp_i64, i64* %al.~w.aux_ptr',
        [N, N]),
    L13 = '  ; M10: snapshot regs[48..63] (Y window) into y_save (field 3).',
    format(atom(L14),
'  %al.~w.ys_dst = getelementptr %StackEntry, %StackEntry* %al.~w.entry, i32 0, i32 3, i32 0',
        [N, N]),
    format(atom(L15),
'  %al.~w.dst_i8 = bitcast %Value* %al.~w.ys_dst to i8*',
        [N, N]),
    format(atom(L16),
'  %al.~w.regs_src = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 48',
        [N]),
    format(atom(L17),
'  %al.~w.src_i8 = bitcast %Value* %al.~w.regs_src to i8*',
        [N, N]),
    format(atom(L18),
'  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %al.~w.dst_i8, i8* %al.~w.src_i8, i64 256, i1 false)',
        [N, N]),
    format(atom(L19),
'  %al.~w.new_ss = add i32 %al.~w.ss, 1',
        [N, N]),
    format(atom(L20),
'  store i32 %al.~w.new_ss, i32* %al.~w.ss_ptr',
        [N, N]),
    format(atom(L21), '  br label %~w', [Next]),
    atomic_list_concat(
        [L0, L1, L_ensure, L2, L3, L4, L5, L6, L7, L8, L9, L10,
         L11, L12, L13, L14, L15, L16, L17, L18, L19, L20, L21],
        '\n', Block).

% --- deallocate: pop most recent env frame, restore CP + Y-regs ---
%
% More involved control flow than allocate — we scan the stack back
% looking for the most recent EnvFrame (type==0), skipping any
% intermediate UnifyCtx / WriteCtx entries. Each block (entry, scan,
% loop, skip, pop) is emitted independently via per-line format
% concatenation so each placeholder list stays trivially countable.
emit_instr(deallocate, N, Next, Block) :- !,
    format(atom(LblScan), 'pc_~w_scan', [N]),
    format(atom(LblLoop), 'pc_~w_loop', [N]),
    format(atom(LblSkip), 'pc_~w_skip', [N]),
    format(atom(LblPop),  'pc_~w_pop',  [N]),
    % --- entry block ---
    format(atom(E0), 'pc_~w:', [N]),
    E1 = '  ; deallocate (scan back for EnvFrame, restore CP + y_save)',
    format(atom(E2),
'  %da.~w.ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3',
        [N]),
    format(atom(E3),
'  %da.~w.ss = load i32, i32* %da.~w.ss_ptr',
        [N, N]),
    format(atom(E4),
'  %da.~w.has = icmp sgt i32 %da.~w.ss, 0',
        [N, N]),
    format(atom(E5),
'  br i1 %da.~w.has, label %~w, label %~w',
        [N, LblScan, Next]),
    % --- scan block ---
    format(atom(S0), '~w:', [LblScan]),
    format(atom(S1),
'  %da.~w.stack_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 2',
        [N]),
    format(atom(S2),
'  %da.~w.stack = load %StackEntry*, %StackEntry** %da.~w.stack_ptr',
        [N, N]),
    format(atom(S3),
'  br label %~w', [LblLoop]),
    % --- loop block ---
    format(atom(LB0), '~w:', [LblLoop]),
    format(atom(LB1),
'  %da.~w.idx = phi i32 [%da.~w.ss, %~w], [%da.~w.prev, %~w]',
        [N, N, LblScan, N, LblSkip]),
    format(atom(LB2),
'  %da.~w.prev = sub i32 %da.~w.idx, 1',
        [N, N]),
    format(atom(LB3),
'  %da.~w.eslot = getelementptr %StackEntry, %StackEntry* %da.~w.stack, i32 %da.~w.prev',
        [N, N, N]),
    format(atom(LB4),
'  %da.~w.tp = getelementptr %StackEntry, %StackEntry* %da.~w.eslot, i32 0, i32 0',
        [N, N]),
    format(atom(LB5),
'  %da.~w.ty = load i32, i32* %da.~w.tp',
        [N, N]),
    format(atom(LB6),
'  %da.~w.is_env = icmp eq i32 %da.~w.ty, 0',
        [N, N]),
    format(atom(LB7),
'  br i1 %da.~w.is_env, label %~w, label %~w',
        [N, LblPop, LblSkip]),
    % --- skip block ---
    format(atom(SK0), '~w:', [LblSkip]),
    format(atom(SK1),
'  %da.~w.cont = icmp sgt i32 %da.~w.prev, 0',
        [N, N]),
    format(atom(SK2),
'  br i1 %da.~w.cont, label %~w, label %~w',
        [N, LblLoop, Next]),
    % --- pop block ---
    format(atom(P0), '~w:', [LblPop]),
    format(atom(P1),
'  %da.~w.aux_ptr = getelementptr %StackEntry, %StackEntry* %da.~w.eslot, i32 0, i32 1',
        [N, N]),
    format(atom(P2),
'  %da.~w.aux = load i64, i64* %da.~w.aux_ptr',
        [N, N]),
    format(atom(P3),
'  %da.~w.cp_i32 = trunc i64 %da.~w.aux to i32',
        [N, N]),
    format(atom(P4),
'  call void @wam_set_cp(%WamState* %vm, i32 %da.~w.cp_i32)',
        [N]),
    format(atom(P5),
'  %da.~w.ys_src = getelementptr %StackEntry, %StackEntry* %da.~w.eslot, i32 0, i32 3, i32 0',
        [N, N]),
    format(atom(P6),
'  %da.~w.src_i8 = bitcast %Value* %da.~w.ys_src to i8*',
        [N, N]),
    format(atom(P7),
'  %da.~w.regs_dst = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 48',
        [N]),
    format(atom(P8),
'  %da.~w.dst_i8 = bitcast %Value* %da.~w.regs_dst to i8*',
        [N, N]),
    format(atom(P9),
'  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %da.~w.dst_i8, i8* %da.~w.src_i8, i64 256, i1 false)',
        [N, N]),
    format(atom(P10),
'  store i32 %da.~w.prev, i32* %da.~w.ss_ptr',
        [N, N]),
    format(atom(P11), '  br label %~w', [Next]),
    atomic_list_concat(
        [E0, E1, E2, E3, E4, E5,
         '', S0, S1, S2, S3,
         '', LB0, LB1, LB2, LB3, LB4, LB5, LB6, LB7,
         '', SK0, SK1, SK2,
         '', P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11],
        '\n', Block).

% --- get_structure F/Arity, Ai: read mode if Ai is a compound,
%     write mode if Ai is unbound, fail otherwise ---
%
% Mirrors the @step 'get_structure' case. Read mode: extract the
% compound's args array, push a UnifyCtx; subsequent unify_* instrs
% consume args from the ctx. Write mode: allocate a fresh compound on
% the arena, push a WriteCtx; subsequent set_* and unify_* instrs (in
% write mode) populate args. Anything else (bound non-compound) fails.
emit_instr(get_structure(FStr, AiStr), N, Next, Block) :- !,
    parse_reg(AiStr, AiIdx),
    parse_functor(FStr, FunctorName, FArity),
    wam_llvm_target:sanitize_functor_for_llvm(FunctorName, SaneFN),
    wam_llvm_target:register_functor_string(FunctorName),
    string_length(FunctorName, FNLen0),
    FNLen is FNLen0 + 1,
    format(atom(LblWrite), 'pc_~w_gs_write', [N]),
    format(atom(LblCheck), 'pc_~w_gs_check', [N]),
    format(atom(LblRead),  'pc_~w_gs_read',  [N]),
    % --- entry block: deref Ai and branch on tag ---
    format(atom(E0), 'pc_~w:', [N]),
    format(atom(E1), '  ; get_structure ~w, ~w', [FStr, AiStr]),
    format(atom(E2),
'  %gs.~w.val = call %Value @wam_get_reg_deref(%WamState* %vm, i32 ~w)',
        [N, AiIdx]),
    format(atom(E3),
'  %gs.~w.tag = extractvalue %Value %gs.~w.val, 0',
        [N, N]),
    format(atom(E4),
'  %gs.~w.is_cp = icmp eq i32 %gs.~w.tag, 3',
        [N, N]),
    format(atom(E5),
'  br i1 %gs.~w.is_cp, label %~w, label %~w',
        [N, LblRead, LblCheck]),
    % --- check_unb block: bound non-compound → fail, unbound → write ---
    format(atom(C0), '~w:', [LblCheck]),
    format(atom(C1),
'  %gs.~w.unb = call i1 @value_is_unbound(%Value %gs.~w.val)',
        [N, N]),
    format(atom(C2),
'  br i1 %gs.~w.unb, label %~w, label %lowered_fail',
        [N, LblWrite]),
    % --- write block: allocate compound, push WriteCtx ---
    %
    % Same shape as put_structure (uses the @.fn_<sane> functor global
    % and registers it for emission), but binds Ai to a Ref into the
    % freshly-allocated compound's args region. Subsequent set_*/unify_*
    % instrs in write mode populate the args.
    format(atom(W0), '~w:', [LblWrite]),
    format(atom(W1),
'  %gs.~w.fn_ptr = getelementptr [~w x i8], [~w x i8]* @.fn_~w, i32 0, i32 0',
        [N, FNLen, FNLen, SaneFN]),
    W2 = '  call void @wam_arena_ensure()',
    format(atom(W3),
'  %gs.~w.cp_size = ptrtoint %Compound* getelementptr (%Compound, %Compound* null, i32 1) to i64',
        [N]),
    format(atom(W4),
'  %gs.~w.cp_mem = call i8* @wam_arena_alloc(i64 %gs.~w.cp_size)',
        [N, N]),
    format(atom(W5),
'  %gs.~w.cp_ptr = bitcast i8* %gs.~w.cp_mem to %Compound*',
        [N, N]),
    format(atom(W6),
'  %gs.~w.fn_slot = getelementptr %Compound, %Compound* %gs.~w.cp_ptr, i32 0, i32 0',
        [N, N]),
    format(atom(W7),
'  store i8* %gs.~w.fn_ptr, i8** %gs.~w.fn_slot',
        [N, N]),
    format(atom(W8),
'  %gs.~w.ar_slot = getelementptr %Compound, %Compound* %gs.~w.cp_ptr, i32 0, i32 1',
        [N, N]),
    format(atom(W9),
'  store i32 ~w, i32* %gs.~w.ar_slot',
        [FArity, N]),
    ArgsBytes is FArity * 16,
    format(atom(W10),
'  %gs.~w.args_mem = call i8* @wam_arena_alloc(i64 ~w)',
        [N, ArgsBytes]),
    format(atom(W11),
'  %gs.~w.args = bitcast i8* %gs.~w.args_mem to %Value*',
        [N, N]),
    format(atom(W12),
'  %gs.~w.args_slot = getelementptr %Compound, %Compound* %gs.~w.cp_ptr, i32 0, i32 2',
        [N, N]),
    format(atom(W13),
'  store %Value* %gs.~w.args, %Value** %gs.~w.args_slot',
        [N, N]),
    format(atom(W14),
'  %gs.~w.cp_i64 = ptrtoint %Compound* %gs.~w.cp_ptr to i64',
        [N, N]),
    format(atom(W15),
'  %gs.~w.cv0 = insertvalue %Value undef, i32 3, 0',
        [N]),
    format(atom(W16),
'  %gs.~w.cv = insertvalue %Value %gs.~w.cv0, i64 %gs.~w.cp_i64, 1',
        [N, N, N]),
    format(atom(W17),
'  call void @wam_trail_binding(%WamState* %vm, i32 ~w)',
        [AiIdx]),
    format(atom(W18),
'  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %gs.~w.cv)',
        [AiIdx, N]),
    format(atom(W19),
'  call void @wam_push_write_ctx(%WamState* %vm, i32 ~w)',
        [FArity]),
    format(atom(W20),
'  call void @wam_write_ctx_set_args(%WamState* %vm, %Value* %gs.~w.args)',
        [N]),
    format(atom(W21), '  br label %~w', [Next]),
    % --- read block: extract args from compound, push UnifyCtx ---
    format(atom(R0), '~w:', [LblRead]),
    format(atom(R1),
'  %gs.~w.rcp_bits = extractvalue %Value %gs.~w.val, 1',
        [N, N]),
    format(atom(R2),
'  %gs.~w.rcp_ptr = inttoptr i64 %gs.~w.rcp_bits to %Compound*',
        [N, N]),
    format(atom(R3),
'  %gs.~w.rargs_slot = getelementptr %Compound, %Compound* %gs.~w.rcp_ptr, i32 0, i32 2',
        [N, N]),
    format(atom(R4),
'  %gs.~w.rargs = load %Value*, %Value** %gs.~w.rargs_slot',
        [N, N]),
    format(atom(R5),
'  call void @wam_push_unify_ctx(%WamState* %vm, %Value* %gs.~w.rargs, i32 ~w)',
        [N, FArity]),
    format(atom(R6), '  br label %~w', [Next]),
    atomic_list_concat(
        [E0, E1, E2, E3, E4, E5,
         '', C0, C1, C2,
         '', W0, W1, W2, W3, W4, W5, W6, W7, W8, W9,
              W10, W11, W12, W13, W14, W15, W16, W17, W18, W19, W20, W21,
         '', R0, R1, R2, R3, R4, R5, R6],
        '\n', Block).

% --- get_list Ai: special case of get_structure for the `.`/2 cons cell ---
%
% Read mode: Ai is a list — push a 2-arg UnifyCtx onto its args. Write
% mode: Ai is unbound — push a fresh 2-cell heap region (functor marker,
% then 2 unbound slots), bind Ai to a Ref into it, push a WriteCtx.
emit_instr(get_list(AiStr), N, Next, Block) :- !,
    parse_reg(AiStr, AiIdx),
    format(atom(LblWrite), 'pc_~w_gl_write', [N]),
    format(atom(LblRead),  'pc_~w_gl_read',  [N]),
    % --- entry: deref Ai, branch on tag ---
    format(atom(E0), 'pc_~w:', [N]),
    format(atom(E1), '  ; get_list ~w', [AiStr]),
    format(atom(E2),
'  %gl.~w.val = call %Value @wam_get_reg_deref(%WamState* %vm, i32 ~w)',
        [N, AiIdx]),
    format(atom(E3),
'  %gl.~w.tag = extractvalue %Value %gl.~w.val, 0',
        [N, N]),
    % Tag uge 5 → Ref(5) or Unbound(6) → write; otherwise read or fail.
    format(atom(E4),
'  %gl.~w.is_ru = icmp uge i32 %gl.~w.tag, 5',
        [N, N]),
    format(atom(E5),
'  br i1 %gl.~w.is_ru, label %~w, label %~w',
        [N, LblWrite, LblRead]),
    % --- write block ---
    format(atom(W0), '~w:', [LblWrite]),
    format(atom(W1),
'  %gl.~w.marker = call %Value @value_atom(i8* null)',
        [N]),
    format(atom(W2),
'  %gl.~w.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %gl.~w.marker)',
        [N, N]),
    format(atom(W3),
'  %gl.~w.unb = call %Value @value_unbound(i8* null)',
        [N]),
    format(atom(W4),
'  call i32 @wam_heap_push(%WamState* %vm, %Value %gl.~w.unb)',
        [N]),
    format(atom(W5),
'  call i32 @wam_heap_push(%WamState* %vm, %Value %gl.~w.unb)',
        [N]),
    format(atom(W6),
'  %gl.~w.ref = call %Value @value_ref(i32 %gl.~w.addr)',
        [N, N]),
    format(atom(W7),
'  call void @wam_trail_binding(%WamState* %vm, i32 ~w)',
        [AiIdx]),
    format(atom(W8),
'  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %gl.~w.ref)',
        [AiIdx, N]),
    W9 = '  call void @wam_push_write_ctx(%WamState* %vm, i32 2)',
    format(atom(W10),
'  %gl.~w.hp_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 5',
        [N]),
    format(atom(W11),
'  %gl.~w.hp = load %Value*, %Value** %gl.~w.hp_ptr',
        [N, N]),
    format(atom(W12),
'  %gl.~w.h_addr = add i32 %gl.~w.addr, 1',
        [N, N]),
    format(atom(W13),
'  %gl.~w.args = getelementptr %Value, %Value* %gl.~w.hp, i32 %gl.~w.h_addr',
        [N, N, N]),
    format(atom(W14),
'  call void @wam_write_ctx_set_args(%WamState* %vm, %Value* %gl.~w.args)',
        [N]),
    format(atom(W15), '  br label %~w', [Next]),
    % --- read block: matches a list payload that's a %List* ptr ---
    %
    % The bytecode @step's get_list case has a `ret i1 false` at the
    % read label as a TODO sentinel (see wam_llvm_target.pl 'get_list'
    % case body) — that suggests "ground list reading" was not yet
    % implemented in the interpreter and is also out of scope here.
    % Mirror that: a bound non-unbound non-ref Ai means failure.
    %
    % Once the interpreter grows ground-list read support, this block
    % can be replaced with a UnifyCtx push pointing at the %List's
    % elements buffer.
    format(atom(R0), '~w:', [LblRead]),
    R1 = '  br label %lowered_fail',
    atomic_list_concat(
        [E0, E1, E2, E3, E4, E5,
         '', W0, W1, W2, W3, W4, W5, W6, W7, W8, W9,
              W10, W11, W12, W13, W14, W15,
         '', R0, R1],
        '\n', Block).

% --- unify_variable Xn ---
%
% Stack-context-sensitive: read mode pops the next arg from a UnifyCtx
% (which a preceding get_structure/get_list put there); write mode
% creates a fresh unbound heap cell and appends to a WriteCtx. The
% stack-type peek decides which.
emit_instr(unify_variable(XnStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    format(atom(LblRead),  'pc_~w_uv_read',  [N]),
    format(atom(LblWrite), 'pc_~w_uv_write', [N]),
    format(atom(E0), 'pc_~w:', [N]),
    format(atom(E1), '  ; unify_variable ~w', [XnStr]),
    format(atom(E2),
'  %uv.~w.stype = call i32 @wam_peek_stack_type(%WamState* %vm)',
        [N]),
    format(atom(E3),
'  %uv.~w.is_read = icmp eq i32 %uv.~w.stype, 1',
        [N, N]),
    format(atom(E4),
'  br i1 %uv.~w.is_read, label %~w, label %~w',
        [N, LblRead, LblWrite]),
    % --- read block ---
    format(atom(R0), '~w:', [LblRead]),
    format(atom(R1),
'  %uv.~w.arg = call %Value @wam_unify_ctx_next(%WamState* %vm)',
        [N]),
    format(atom(R2),
'  call void @wam_trail_binding(%WamState* %vm, i32 ~w)',
        [XnIdx]),
    format(atom(R3),
'  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %uv.~w.arg)',
        [XnIdx, N]),
    format(atom(R4), '  br label %~w', [Next]),
    % --- write block ---
    format(atom(W0), '~w:', [LblWrite]),
    format(atom(W1),
'  %uv.~w.unb = call %Value @value_unbound(i8* null)',
        [N]),
    format(atom(W2),
'  %uv.~w.addr = call i32 @wam_heap_push(%WamState* %vm, %Value %uv.~w.unb)',
        [N, N]),
    format(atom(W3),
'  %uv.~w.ref = call %Value @value_ref(i32 %uv.~w.addr)',
        [N, N]),
    format(atom(W4),
'  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %uv.~w.ref)',
        [N]),
    format(atom(W5),
'  call void @wam_trail_binding(%WamState* %vm, i32 ~w)',
        [XnIdx]),
    format(atom(W6),
'  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %uv.~w.ref)',
        [XnIdx, N]),
    format(atom(W7), '  br label %~w', [Next]),
    atomic_list_concat(
        [E0, E1, E2, E3, E4,
         '', R0, R1, R2, R3, R4,
         '', W0, W1, W2, W3, W4, W5, W6, W7],
        '\n', Block).

% --- unify_value Xn ---
%
% Read mode: unify reg Xn with the next arg from the UnifyCtx; if
% either side is unbound, bind it to the other. Write mode: append
% the reg value into the current WriteCtx.
emit_instr(unify_value(XnStr), N, Next, Block) :- !,
    parse_reg(XnStr, XnIdx),
    format(atom(LblRead),  'pc_~w_uval_read',  [N]),
    format(atom(LblWrite), 'pc_~w_uval_write', [N]),
    format(atom(LblBind),  'pc_~w_uval_bind',  [N]),
    format(atom(LblROk),   'pc_~w_uval_rok',   [N]),
    format(atom(E0), 'pc_~w:', [N]),
    format(atom(E1), '  ; unify_value ~w', [XnStr]),
    format(atom(E2),
'  %uvl.~w.stype = call i32 @wam_peek_stack_type(%WamState* %vm)',
        [N]),
    format(atom(E3),
'  %uvl.~w.is_read = icmp eq i32 %uvl.~w.stype, 1',
        [N, N]),
    format(atom(E4),
'  br i1 %uvl.~w.is_read, label %~w, label %~w',
        [N, LblRead, LblWrite]),
    % --- read block ---
    format(atom(R0), '~w:', [LblRead]),
    format(atom(R1),
'  %uvl.~w.exp = call %Value @wam_unify_ctx_next(%WamState* %vm)',
        [N]),
    format(atom(R2),
'  %uvl.~w.act = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)',
        [N, XnIdx]),
    format(atom(R3),
'  %uvl.~w.eq = call i1 @value_equals(%Value %uvl.~w.exp, %Value %uvl.~w.act)',
        [N, N, N]),
    format(atom(R4),
'  %uvl.~w.exp_unb = call i1 @value_is_unbound(%Value %uvl.~w.exp)',
        [N, N]),
    format(atom(R5),
'  %uvl.~w.act_unb = call i1 @value_is_unbound(%Value %uvl.~w.act)',
        [N, N]),
    format(atom(R6),
'  %uvl.~w.ok1 = or i1 %uvl.~w.eq, %uvl.~w.exp_unb',
        [N, N, N]),
    format(atom(R7),
'  %uvl.~w.ok = or i1 %uvl.~w.ok1, %uvl.~w.act_unb',
        [N, N, N]),
    format(atom(R8),
'  br i1 %uvl.~w.ok, label %~w, label %lowered_fail',
        [N, LblROk]),
    % rok block — if actual is unbound, bind it
    format(atom(RO0), '~w:', [LblROk]),
    format(atom(RO1),
'  br i1 %uvl.~w.act_unb, label %~w, label %~w',
        [N, LblBind, Next]),
    % bind block
    format(atom(B0), '~w:', [LblBind]),
    format(atom(B1),
'  call void @wam_trail_binding(%WamState* %vm, i32 ~w)',
        [XnIdx]),
    format(atom(B2),
'  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %uvl.~w.exp)',
        [XnIdx, N]),
    format(atom(B3), '  br label %~w', [Next]),
    % --- write block ---
    format(atom(W0), '~w:', [LblWrite]),
    format(atom(W1),
'  %uvl.~w.wv = call %Value @wam_get_reg(%WamState* %vm, i32 ~w)',
        [N, XnIdx]),
    format(atom(W2),
'  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %uvl.~w.wv)',
        [N]),
    format(atom(W3), '  br label %~w', [Next]),
    atomic_list_concat(
        [E0, E1, E2, E3, E4,
         '', R0, R1, R2, R3, R4, R5, R6, R7, R8,
         '', RO0, RO1,
         '', B0, B1, B2, B3,
         '', W0, W1, W2, W3],
        '\n', Block).

% --- unify_constant C ---
%
% Read mode: pop next arg from UnifyCtx, check equal to constant.
% Unbound expected → accept (no binding done — matches the bytecode
% @step shape). Write mode: append the constant to the WriteCtx.
emit_instr(unify_constant(CStr), N, Next, Block) :- !,
    parse_constant(CStr, Tag, Payload),
    format(atom(LblRead),  'pc_~w_uc_read',  [N]),
    format(atom(LblWrite), 'pc_~w_uc_write', [N]),
    format(atom(E0), 'pc_~w:', [N]),
    format(atom(E1), '  ; unify_constant ~w', [CStr]),
    format(atom(E2),
'  %uc.~w.stype = call i32 @wam_peek_stack_type(%WamState* %vm)',
        [N]),
    format(atom(E3),
'  %uc.~w.is_read = icmp eq i32 %uc.~w.stype, 1',
        [N, N]),
    format(atom(E4),
'  %uc.~w.val0 = insertvalue %Value undef, i32 ~w, 0',
        [N, Tag]),
    format(atom(E5),
'  %uc.~w.val = insertvalue %Value %uc.~w.val0, i64 ~w, 1',
        [N, N, Payload]),
    format(atom(E6),
'  br i1 %uc.~w.is_read, label %~w, label %~w',
        [N, LblRead, LblWrite]),
    % --- read block ---
    format(atom(R0), '~w:', [LblRead]),
    format(atom(R1),
'  %uc.~w.exp = call %Value @wam_unify_ctx_next(%WamState* %vm)',
        [N]),
    format(atom(R2),
'  %uc.~w.eq = call i1 @value_equals(%Value %uc.~w.exp, %Value %uc.~w.val)',
        [N, N, N]),
    format(atom(R3),
'  %uc.~w.exp_unb = call i1 @value_is_unbound(%Value %uc.~w.exp)',
        [N, N]),
    format(atom(R4),
'  %uc.~w.ok = or i1 %uc.~w.eq, %uc.~w.exp_unb',
        [N, N, N]),
    format(atom(R5),
'  br i1 %uc.~w.ok, label %~w, label %lowered_fail',
        [N, Next]),
    % --- write block ---
    format(atom(W0), '~w:', [LblWrite]),
    format(atom(W1),
'  call i32 @wam_heap_push(%WamState* %vm, %Value %uc.~w.val)',
        [N]),
    format(atom(W2),
'  call void @wam_write_ctx_set_arg(%WamState* %vm, %Value %uc.~w.val)',
        [N]),
    format(atom(W3), '  br label %~w', [Next]),
    atomic_list_concat(
        [E0, E1, E2, E3, E4, E5, E6,
         '', R0, R1, R2, R3, R4, R5,
         '', W0, W1, W2, W3],
        '\n', Block).

% --- call <pred>/<arity>: direct call into another lowered kernel (M4) ---
%
% The callee must also be lowered as a kernel with the
% `(%WamState*) → i1` signature. Caller has already populated A1..AN
% via preceding put_*/get_* instructions; the callee reads them from
% the shared state. Bindings the callee makes (via wam_bind_reg etc.)
% remain visible to the caller through the shared trail.
%
% The op-2 continuation PC in the original WAM call is irrelevant in
% the lowered form because there is no PC — control flow IS the
% sequencing. On success branch to the next basic block; on failure
% branch to %lowered_fail (the caller's epilogue).
%
% Closure constraint: this only emits correctly if the callee is in
% the module's lowered set. wam_llvm_target's closure analysis (M4)
% gates emission to ensure that.
emit_instr(call(PredStr, _ContPC), N, Next, Block) :- !,
    parse_call_target(PredStr, CalleeName, CalleeArity),
    llvm_lowered_func_name(CalleeName/CalleeArity, CalleeKernel),
    format(atom(Block),
'pc_~w:
  ; call ~w (lowered kernel, shared state)
  %call.~w.r = call i1 @~w(%WamState* %vm)
  br i1 %call.~w.r, label %~w, label %lowered_fail',
        [N, PredStr,
         N, CalleeKernel,
         N, Next]).

% --- execute <pred>: tail call into another lowered kernel (M4) ---
%
% Same as call but in tail position. Uses LLVM `musttail` so the
% caller's stack frame is reused — important for any predicate doing
% iterative recursion (transitive_closure, list traversal, etc.).
% musttail requires matching signatures, which all lowered kernels
% satisfy by construction.
emit_instr(execute(PredStr), N, _Next, Block) :- !,
    parse_call_target(PredStr, CalleeName, CalleeArity),
    llvm_lowered_func_name(CalleeName/CalleeArity, CalleeKernel),
    format(atom(Block),
'pc_~w:
  ; execute ~w (tail call into lowered kernel)
  %exec.~w.r = musttail call i1 @~w(%WamState* %vm)
  ret i1 %exec.~w.r',
        [N, PredStr,
         N, CalleeKernel,
         N]).

% --- builtin_call op/Arity, ArgCount: delegate to @execute_builtin ---
emit_instr(builtin_call(OpStr, ArityStr), N, Next, Block) :- !,
    ( atom(OpStr) -> OpAtom = OpStr ; atom_string(OpAtom, OpStr) ),
    wam_llvm_target:builtin_op_to_id(OpAtom, OpId),
    ( atom(ArityStr) -> atom_number(ArityStr, Arity)
    ; number_string(Arity, ArityStr)
    ),
    format(atom(Block),
'pc_~w:
  ; builtin_call ~w, ~w
  %bi.~w.r = call i1 @execute_builtin(%WamState* %vm, i32 ~w, i32 ~w)
  br i1 %bi.~w.r, label %~w, label %lowered_fail',
        [N, OpStr, ArityStr,
         N, OpId, Arity,
         N, Next]).

% --- Anything else slipped past the gate: fall back loudly ---
emit_instr(I, N, _Next, _Block) :-
    throw(error(wam_llvm_lowering_unsupported(I, N),
          'wam_llvm_lowered_emitter: instruction reached emitter but is not in the supported set')).

% ============================================================================
% Helpers: register / constant / functor parsing
% ============================================================================

%% parse_reg(+RegStr, -Index)
%  Accepts either a string ("A1", "X3", "Y2") or an atom; returns the
%  fixed-array index in [64 x %Value]. Delegates to the shared mapping
%  in bindings/llvm_wam_bindings.pl so any future ABI change applies
%  everywhere.
parse_reg(RegStr, Index) :-
    (   atom(RegStr) -> Atom = RegStr
    ;   atom_string(Atom, RegStr)
    ),
    reg_name_to_index(Atom, Index).

%% parse_constant(+CStr, -Tag, -Payload)
%
%  Maps a WAM-text constant token ("42", "foo", "'quoted atom'") to
%  the %Value tag and i64 payload used by @value_*/@wam_set_reg_*.
%
%    integer literal  → tag 1, payload = the integer
%    atom (or quoted) → tag 0, payload = interned atom ID
%
%  Atom interning goes through the same wam_instruction_to_llvm_literal
%  packing the rest of the LLVM target uses, so a single atom appears
%  with the same ID in both lowered and bytecode-compiled predicates.
parse_constant(CStr, Tag, Payload) :-
    (   string(CStr) -> S = CStr
    ;   atom_string(CStr, S)
    ),
    (   number_string(N, S), integer(N)
    ->  Tag = 1, Payload = N
    ;   % Strip the surrounding quotes from quoted atoms before interning.
        ( sub_string(S, 0, 1, _, "'"), sub_string(S, _, 1, 0, "'")
        ->  string_length(S, L), Inner is L - 2,
            sub_string(S, 1, Inner, _, BareStr),
            atom_string(BareAtom, BareStr)
        ;   atom_string(BareAtom, S)
        ),
        % Reuse the WAM target's interning by going through
        % wam_instruction_to_llvm_literal/2 on a synthetic get_constant
        % — that path runs the same llvm_pack_value/intern_atom logic
        % and produces a literal we can scrape the payload from.
        wam_llvm_target:wam_instruction_to_llvm_literal(get_constant(BareAtom, 'A1'), Lit),
        % Lit looks like "{ i32 0, i64 <payload>, i64 <op2> }".
        % Extract the first i64 with a regex that survives Prolog's
        % atom_string conversion of Lit.
        atom_string(Lit, LitStr),
        split_string(LitStr, " ,", " ,{}", Tokens),
        nth0(3, Tokens, PayloadStr),    % skip "i32" "0" "i64"
        number_string(Payload, PayloadStr),
        Tag = 0
    ).

%% parse_functor(+FStr, -Name, -Arity)
%
%  "foo/2" → Name="foo", Arity=2. Delegates to
%  wam_llvm_target:split_functor_arity/3 which handles names with `/`
%  in them (e.g. integer-division `//2`).
parse_functor(FStr, Name, Arity) :-
    wam_llvm_target:split_functor_arity(FStr, Name, Arity).

%% parse_call_target(+PredStr, -Name, -Arity)
%
%  "foo/2" → Name='foo', Arity=2. Used by the call/execute emitters
%  to resolve the callee's lowered-kernel symbol.
parse_call_target(PredStr, NameAtom, Arity) :-
    wam_llvm_target:split_functor_arity(PredStr, NameStr, Arity),
    atom_string(NameAtom, NameStr).

% ============================================================================
% M3: Public-entry wrappers + multi-clause dispatchers
% ============================================================================

%% emit_native_wrapper(+Pred/Arity, -WrapperCode) is det.
%
%  Emits `define i1 @<pred>(%Value %a1, ...)` as the public entry for
%  external callers. Allocates a fresh %WamState, copies the %Value
%  arguments into A1..AN, calls the kernel, frees the state.
%
%  Under M4, the kernel takes %WamState* (not Value params), so this
%  wrapper handles the state lifecycle on behalf of external callers.
%  Calls from one lowered kernel to another skip this wrapper and
%  share state directly via @lowered_<callee>_<arity>(%WamState*).
emit_native_wrapper(Pred/Arity, WrapperCode) :-
    atom_string(Pred, PredStr),
    llvm_lowered_func_name(Pred/Arity, LoweredName),
    build_param_list(Arity, ParamList),
    build_arg_setup(Arity, ArgSetup),
    format(atom(WrapperCode),
'; Public entry for ~w/~w (M4 native wrapper).
; Allocates state, copies args into A-registers, calls kernel, frees.
define i1 @~w(~w) {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* null, i32 0, i32* null, i32 0)
~w
  %r = call i1 @~w(%WamState* %vm)
  call void @wam_state_free(%WamState* %vm)
  ret i1 %r
}',
        [PredStr, Arity,
         PredStr, ParamList,
         ArgSetup,
         LoweredName]).

%% emit_hybrid_dispatcher(+Pred/Arity, +StartPC, +InstrCount,
%%                        +LabelArraySize, -DispatcherCode) is det.
%
%  Emits the multi-clause dispatcher entry `@<pred>` for predicates
%  whose clause 1 is lowerable but whose full body contains
%  try_me_else / retry_me_else / trust_me (additional clauses need the
%  bytecode interpreter for backtrack).
%
%  M4 redesign: both fast and slow paths share a SINGLE %WamState
%  allocated by the dispatcher. The fast path is the lowered kernel
%  invoked on that shared state. On fast-path failure the dispatcher
%  resets the bytecode entry point (PC = StartPC) and runs @run_loop
%  on the SAME state — bindings made by the fast path remain visible.
%  This is a step toward proper clause-1-with-rollback semantics; full
%  trail rollback between fast and slow is deferred (the bindings
%  difference matters only for predicates whose clause 1 partially
%  succeeds — first-arg-indexed predicates fail fast on the head match
%  before any binding work).
emit_hybrid_dispatcher(Pred/Arity, StartPC, InstrCount, LabelArraySize,
                       DispatcherCode) :-
    atom_string(Pred, PredStr),
    llvm_lowered_func_name(Pred/Arity, LoweredName),
    build_param_list(Arity, ParamList),
    build_arg_setup(Arity, ArgSetup),
    format(atom(DispatcherCode),
'; Public entry for ~w/~w (M4 hybrid dispatcher, shared %WamState).
; Fast path: lowered clause 1 (deterministic + supported instr subset).
; Slow path on fast-path failure: full bytecode via @run_loop, which
; handles try_me_else / retry_me_else / trust_me for clauses 2+.
define i1 @~w(~w) {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
    i32 ~w,
    i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
    i32 ~w)
~w
  %fast = call i1 @~w(%WamState* %vm)
  br i1 %fast, label %success, label %slow_path

success:
  call void @wam_state_free(%WamState* %vm)
  ret i1 true

slow_path:
  call void @wam_set_pc(%WamState* %vm, i32 ~w)
  %slow = call i1 @run_loop(%WamState* %vm)
  call void @wam_state_free(%WamState* %vm)
  ret i1 %slow
}',
        [PredStr, Arity,
         PredStr, ParamList,
         InstrCount, InstrCount,
         InstrCount,
         LabelArraySize, LabelArraySize,
         LabelArraySize,
         ArgSetup,
         LoweredName,
         StartPC]).

%% build_arg_list(+Arity, -ArgList)
%
%  Comma-separated list "%Value %a1, %Value %a2, ..." used at the
%  call sites in the native wrapper and hybrid dispatcher (where the
%  callee expects the same %Value parameters the entry received).
build_arg_list(0, "") :- !.
build_arg_list(Arity, ArgList) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>format(atom(S), "%Value %a~w", [I]), Indices, Parts),
    atomic_list_concat(Parts, ', ', ArgList).

%% build_arg_setup_slow(+Arity, -SetupIR)
%
%  Mirror of build_arg_setup/2 but emitted into the hybrid dispatcher's
%  slow_path block — copies each %Value %a<i> into the fresh
%  %WamState's argument register i-1 before calling @run_loop.
build_arg_setup_slow(0, "") :- !.
build_arg_setup_slow(Arity, Setup) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(
        RegIdx is I - 1,
        format(atom(S),
            '  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %a~w)',
            [RegIdx, I])
    ), Indices, Parts),
    atomic_list_concat(Parts, '\n', Setup).
