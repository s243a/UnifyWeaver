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
    wam_llvm_lowerable/3,            % +Pred/Arity, +WamCode, -Reason
    lower_predicate_to_llvm/4,       % +Pred/Arity, +WamCode, +Options, -LLVMCode
    is_deterministic_pred_llvm/1,    % +Instrs
    llvm_lowered_func_name/2         % +Pred/Arity, -LLVMFuncName
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../bindings/llvm_wam_bindings', [reg_name_to_index/2]).

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

% ============================================================================
% Lowerability gate
% ============================================================================

%% wam_llvm_lowerable(+PI, +WamCode, -Reason) is semidet.
%
%  Succeeds iff the predicate is safe to lower to a standalone LLVM
%  function. Reason is bound to a short tag for logging.
wam_llvm_lowerable(_PI, WamCode, Reason) :-
    (   is_list(WamCode) -> Instrs = WamCode
    ;   atom(WamCode) -> parse_wam_text(WamCode, Instrs)
    ;   parse_wam_text(WamCode, Instrs)
    ),
    is_deterministic_pred_llvm(Instrs),
    forall(member(I, Instrs), supported(I)),
    Reason = deterministic.

is_deterministic_pred_llvm(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

supported(get_constant(_, _)).
supported(get_variable(_, _)).
supported(get_value(_, _)).
supported(put_constant(_, _)).
supported(put_variable(_, _)).
supported(put_value(_, _)).
supported(put_structure(_, _)).
supported(set_constant(_)).
supported(set_variable(_)).
supported(set_value(_)).
supported(allocate).
supported(deallocate).
supported(builtin_call(_, _)).
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
lower_predicate_to_llvm(PI, WamCode, _Options, LLVMCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    llvm_lowered_func_name(Pred/Arity, FuncName),
    (   is_list(WamCode) -> Instrs = WamCode
    ;   parse_wam_text(WamCode, Instrs)
    ),
    % Build the parameter list and the entry-block "copy %a<i> into reg i-1".
    build_param_list(Arity, ParamList),
    build_arg_setup(Arity, ArgSetup),
    % Emit one basic block per instruction, then the shared
    % succeed/fail epilogue blocks.
    emit_all_instrs(Instrs, 0, BodyBlocks),
    atomic_list_concat(BodyBlocks, '\n', BodyStr),
    format(atom(LLVMCode),
'; === lowered predicate: ~w/~w ===
; Emitted by wam_llvm_lowered_emitter.pl. Each WAM instruction lives in
; its own basic block named pc_<N>. Success branches to the next block;
; failure branches to %lowered_fail. The final proceed branches to
; %lowered_succeed. Both epilogue blocks free the %WamState before
; returning so callers do not leak the ~~85 KB state allocation across
; repeated invocations.
define i1 @~w(~w) {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* null,
    i32 0,
    i32* null,
    i32 0)
~w
  br label %pc_0

~w

lowered_succeed:
  call void @wam_state_free(%WamState* %vm)
  ret i1 true

lowered_fail:
  call void @wam_state_free(%WamState* %vm)
  ret i1 false
}',
        [Pred, Arity, FuncName, ParamList, ArgSetup, BodyStr]).

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
    L13 = '  ; Snapshot regs[16..63] into y_save (field 3 = [48 x %Value]).',
    format(atom(L14),
'  %al.~w.ys_dst = getelementptr %StackEntry, %StackEntry* %al.~w.entry, i32 0, i32 3, i32 0',
        [N, N]),
    format(atom(L15),
'  %al.~w.dst_i8 = bitcast %Value* %al.~w.ys_dst to i8*',
        [N, N]),
    format(atom(L16),
'  %al.~w.regs_src = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 16',
        [N]),
    format(atom(L17),
'  %al.~w.src_i8 = bitcast %Value* %al.~w.regs_src to i8*',
        [N, N]),
    format(atom(L18),
'  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %al.~w.dst_i8, i8* %al.~w.src_i8, i64 768, i1 false)',
        [N, N]),
    format(atom(L19),
'  %al.~w.new_ss = add i32 %al.~w.ss, 1',
        [N, N]),
    format(atom(L20),
'  store i32 %al.~w.new_ss, i32* %al.~w.ss_ptr',
        [N, N]),
    format(atom(L21), '  br label %~w', [Next]),
    atomic_list_concat(
        [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10,
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
'  %da.~w.regs_dst = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 16',
        [N]),
    format(atom(P8),
'  %da.~w.dst_i8 = bitcast %Value* %da.~w.regs_dst to i8*',
        [N, N]),
    format(atom(P9),
'  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %da.~w.dst_i8, i8* %da.~w.src_i8, i64 768, i1 false)',
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
%  "foo/2" → Name="foo", Arity=2. Handles either a string or atom
%  representation.
parse_functor(FStr, Name, Arity) :-
    (   atom(FStr) -> atom_string(FStr, S)
    ;   string(FStr) -> S = FStr
    ),
    split_string(S, "/", "", [NameStr, ArityStr]),
    Name = NameStr,
    number_string(Arity, ArityStr).
