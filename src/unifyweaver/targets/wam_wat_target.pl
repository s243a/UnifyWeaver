:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_wat_target.pl - WAM-to-WAT (WebAssembly Text) Transpilation Target
%
% Transpiles WAM instructions to self-contained .wat modules.
% Uses linear memory with tagged values (12 bytes each),
% br_table for instruction dispatch, and bump allocation for the heap.
%
% Phase 0: Templates (value, state, runtime, module)
% Phase 1: WAM instructions -> data segment bytes
% Phase 2: step dispatch via br_table + runtime helpers
% Phase 3: Project assembly (write_wam_wat_project/3)

:- module(wam_wat_target, [
    compile_step_wam_to_wat/2,          % +Options, -WatCode
    compile_wam_helpers_to_wat/2,       % +Options, -WatCode
    compile_wam_runtime_to_wat/2,       % +Options, -WatCode
    compile_wam_predicate_to_wat/4,     % +Pred/Arity, +WamCode, +Options, -WatCode
    wam_instruction_to_wat_bytes/3,     % +WamInstr, +LabelMap, -ByteHex
    reg_name_to_index/2,                % +Name, -Index
    atom_hash_i64/2,                    % +Atom, -Hash
    write_wam_wat_project/3             % +Predicates, +Options, +OutputFile
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

:- discontiguous wam_wat_case/2.

% ============================================================================
% Constants
% ============================================================================

wam_state_base(65536).
wam_reg_base(65600).      % 65536 + 64
wam_trail_base(66560).    % 65536 + 1024
wam_stack_base(73728).    % 65536 + 8192
wam_heap_base(131072).    % page 2

% Register and choice point geometry (derived from register count)
wam_num_regs(64).
wam_val_size(12).         % bytes per tagged value
%% wam_cp_size = 24 (metadata: next_pc + trail_mark + saved_cp + saved_heap_top
%%               + saved_env_base + retry_n) + CP_SAVE_REGS A registers * val_size.
%% The retry_n slot at +20 carries builtin iteration state (currently used
%% by nondeterministic arg/3 to track the next argument index to try on
%% backtrack). A value of 0 means the CP was not pushed by arg/3.
%% Only the first CP_SAVE_REGS argument registers (A1..A_N) are saved in choice
%% points. Y registers live in environment frames and are NOT saved. X temporaries
%% above CP_SAVE_REGS are also NOT saved — they are dead across choice point
%% boundaries in well-compiled WAM code (only argument registers carry state
%% across alternatives). Reducing from 32 to 8 cuts CP push/pop memcpy from
%% 384 bytes to 96 bytes — a 4x reduction that directly impacts recursive
%% predicates like sum_ints which create/restore CPs on every clause entry.
wam_cp_save_regs(8).
wam_cp_size(Size) :- wam_cp_save_regs(N), wam_val_size(V), Size is 24 + N * V.

% Tag constants
tag_atom(0).
tag_integer(1).
tag_float(2).
tag_compound(3).
tag_list(4).
tag_ref(5).
tag_unbound(6).
tag_bool(7).

% Instruction tags
instr_tag(get_constant,    0).
instr_tag(get_variable,    1).
instr_tag(get_value,       2).
instr_tag(get_structure,   3).
instr_tag(get_list,        4).
instr_tag(unify_variable,  5).
instr_tag(unify_value,     6).
instr_tag(unify_constant,  7).
instr_tag(put_constant,    8).
instr_tag(put_variable,    9).
instr_tag(put_value,       10).
instr_tag(put_structure,   11).
instr_tag(put_list,        12).
instr_tag(set_variable,    13).
instr_tag(set_value,       14).
instr_tag(set_constant,    15).
instr_tag(allocate,        16).
instr_tag(deallocate,      17).
instr_tag(call,            18).
instr_tag(execute,         19).
instr_tag(proceed,         20).
instr_tag(builtin_call,    21).
instr_tag(try_me_else,     22).
instr_tag(retry_me_else,   23).
instr_tag(trust_me,        24).
instr_tag(neck_cut_test,   25).
instr_tag(cut_ite,         26).
instr_tag(jump,            27).
instr_tag(begin_aggregate, 28).
instr_tag(end_aggregate,   29).
instr_tag(nop,             30).
instr_tag(switch_on_const, 31).
instr_tag(switch_entry,    32).
instr_tag(switch_on_struct,     33).
instr_tag(switch_struct_entry,  34).
instr_tag(switch_on_term_hdr,   35).
instr_tag(fused_is_add,    36).
instr_tag(fused_is_sub,    37).
instr_tag(fused_is_mul,    38).
instr_tag(fused_is_add_const, 39).
instr_tag(fused_is_mul_const, 40).
instr_tag(arg_direct,      41).
instr_tag(functor_direct,  42).
instr_tag(copy_term_direct, 43).
instr_tag(univ_direct,      44).
instr_tag(is_list_direct,   45).
instr_tag(arg_reg_direct,   46).
instr_tag(arg_lit_direct,   47).
instr_tag(arg_to_a1_reg,    48).
instr_tag(arg_to_a1_lit,    49).
instr_tag(arg_call_reg_3,   50).
instr_tag(arg_call_lit_3,   51).
instr_tag(arg_call_reg_3_dead, 52).
instr_tag(arg_call_lit_3_dead, 53).
instr_tag(arg_call_reg_1,   54).
instr_tag(arg_call_lit_1,   55).
instr_tag(arg_call_reg_1_dead, 56).
instr_tag(arg_call_lit_1_dead, 57).
instr_tag(arg_call_reg_2,   58).
instr_tag(arg_call_lit_2,   59).
instr_tag(arg_call_reg_2_dead, 60).
instr_tag(arg_call_lit_2_dead, 61).
instr_tag(tail_call_5,      62).
instr_tag(deallocate_proceed, 63).
instr_tag(tail_call_5_c1_lit, 64).
instr_tag(deallocate_builtin_proceed, 65).
instr_tag(deallocate_arg_direct_proceed,       66).
instr_tag(deallocate_functor_direct_proceed,   67).
instr_tag(deallocate_copy_term_direct_proceed, 68).
instr_tag(deallocate_univ_direct_proceed,      69).
instr_tag(deallocate_is_list_direct_proceed,   70).
instr_tag(builtin_proceed,                     71).
instr_tag(type_dispatch_a1,                    72).

% Builtin operation IDs
builtin_id('write/1',  0).
builtin_id('nl/0',     1).
builtin_id('is/2',     2).
builtin_id('=:=/2',    3).
builtin_id('=\\=/2',   4).
builtin_id('</2',      5).
builtin_id('>/2',      6).
builtin_id('=</2',     7).
builtin_id('>=/2',     8).
builtin_id('var/1',    9).
builtin_id('nonvar/1', 10).
builtin_id('atom/1',   11).
builtin_id('integer/1', 12).
builtin_id('float/1',  13).
builtin_id('number/1', 14).
builtin_id('true/0',   15).
builtin_id('fail/0',   16).
builtin_id('!/0',      17).
%% Term inspection builtins (WAM_TERM_BUILTINS plan). IDs 18-21 are
%% reserved here in one block; individual backends may implement them
%% incrementally. An unimplemented ID returns fail from $execute_builtin.
builtin_id('functor/3',   18).
builtin_id('arg/3',       19).
builtin_id('=../2',       20).
builtin_id('copy_term/2', 21).
builtin_id('=/2',         22).
builtin_id('compound/1',  23).
builtin_id('is_list/1',   24).
builtin_id('==/2',        25).
builtin_id('\\==/2',      26).
builtin_id('@</2',        27).
builtin_id('@>/2',        28).
builtin_id('@=</2',       29).
builtin_id('@>=/2',       30).

% ============================================================================
% Register name -> index mapping
% ============================================================================

%% reg_name_to_index(+Name, -Index)
%  A1-A32 -> 0-31, X1-X32 -> 32-63, Y1-Y32 -> 32-63 (share X space)
%% Register index mapping:
%%   A1-A32 → 0-31   (argument registers, in register file)
%%   X1-X32 → 32-63  (temporaries, in register file)
%%   Y1-Y32 → 64-95  (permanent variables, in environment frame on stack)
%% Y indices are in a SEPARATE range (64+) from X (32+) so that
%% $reg_offset can route them to different storage: register file for
%% A/X, environment frame for Y. Prior to this fix, X and Y shared
%% the same index range (both N+31), causing runtime collisions when
%% a callee's Y registers stomped the caller's Y registers via the
%% global register file.
reg_name_to_index(Name, Index) :-
    atom_string(Name, Str),
    string_codes(Str, [Prefix|Rest]),
    number_codes(N, Rest),
    (   Prefix =:= 0'A -> Index is N - 1
    ;   Prefix =:= 0'X -> Index is N + 31
    ;   Prefix =:= 0'Y -> Index is N + 63
    ;   Index = 0
    ).

% ============================================================================
% Atom hashing (matches wat_target.pl hash)
% ============================================================================

%% atom_hash_i64(+Atom, -Hash)
%  DJB2-like hash: acc * 31 + char mod 2^31-1
atom_hash_i64(Atom, Hash) :-
    atom_codes(Atom, Codes),
    hash_codes(Codes, 0, Hash).

hash_codes([], Acc, Acc).
hash_codes([C|Cs], Acc, Hash) :-
    Acc1 is (Acc * 31 + C) mod 2147483647,
    hash_codes(Cs, Acc1, Hash).

% ============================================================================
% PHASE 1: WAM instruction encoding to data segment bytes
% ============================================================================

%% wam_instruction_to_wat_bytes(+WamInstr, +LabelMap, -HexString)
%  Encodes a WAM instruction as a 20-byte hex string for a data segment.
%  Format: [tag:i32-le][op1:i64-le][op2:i64-le]

wam_instruction_to_wat_bytes(get_constant(C, Ai), _Labels, Hex) :-
    instr_tag(get_constant, Tag),
    encode_constant_with_tag(C, ConstTag, Op1),
    reg_name_to_index(Ai, RegIdx),
    %% op2 layout: high 32 bits = value-cell tag hint, low 32 bits = reg idx.
    %% Runtime extracts RegIdx via i32.wrap_i64 and ConstTag via
    %% i64.shr_u 32 + i32.wrap_i64. See wam_wat_case(put_constant, ...)
    %% and friends for the decode side.
    Op2 is (ConstTag << 32) \/ RegIdx,
    encode_instr_hex(Tag, Op1, Op2, Hex).

wam_instruction_to_wat_bytes(get_variable(Xn, Ai), _Labels, Hex) :-
    instr_tag(get_variable, Tag),
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, XnIdx, AiIdx, Hex).

wam_instruction_to_wat_bytes(get_value(Xn, Ai), _Labels, Hex) :-
    instr_tag(get_value, Tag),
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, XnIdx, AiIdx, Hex).

wam_instruction_to_wat_bytes(get_structure(F, Ai), _Labels, Hex) :-
    instr_tag(get_structure, Tag),
    encode_structure_op1(F, Op1),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, Op1, AiIdx, Hex).

wam_instruction_to_wat_bytes(get_list(Ai), _Labels, Hex) :-
    instr_tag(get_list, Tag),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, AiIdx, 0, Hex).

wam_instruction_to_wat_bytes(unify_variable(Xn), _Labels, Hex) :-
    instr_tag(unify_variable, Tag),
    reg_name_to_index(Xn, XnIdx),
    encode_instr_hex(Tag, XnIdx, 0, Hex).

wam_instruction_to_wat_bytes(unify_value(Xn), _Labels, Hex) :-
    instr_tag(unify_value, Tag),
    reg_name_to_index(Xn, XnIdx),
    encode_instr_hex(Tag, XnIdx, 0, Hex).

wam_instruction_to_wat_bytes(unify_constant(C), _Labels, Hex) :-
    instr_tag(unify_constant, Tag),
    encode_constant_with_tag(C, ConstTag, Op1),
    %% op2 = value-cell tag hint (atom=0, integer=1, float=2).
    encode_instr_hex(Tag, Op1, ConstTag, Hex).

wam_instruction_to_wat_bytes(put_constant(C, Ai), _Labels, Hex) :-
    instr_tag(put_constant, Tag),
    encode_constant_with_tag(C, ConstTag, Op1),
    reg_name_to_index(Ai, RegIdx),
    %% op2: high 32 = tag hint, low 32 = reg idx. See note on get_constant.
    Op2 is (ConstTag << 32) \/ RegIdx,
    encode_instr_hex(Tag, Op1, Op2, Hex).

wam_instruction_to_wat_bytes(put_variable(Xn, Ai), _Labels, Hex) :-
    instr_tag(put_variable, Tag),
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, XnIdx, AiIdx, Hex).

wam_instruction_to_wat_bytes(put_value(Xn, Ai), _Labels, Hex) :-
    instr_tag(put_value, Tag),
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, XnIdx, AiIdx, Hex).

wam_instruction_to_wat_bytes(put_structure(F, Ai), _Labels, Hex) :-
    instr_tag(put_structure, Tag),
    encode_structure_op1(F, Op1),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, Op1, AiIdx, Hex).

wam_instruction_to_wat_bytes(put_list(Ai), _Labels, Hex) :-
    instr_tag(put_list, Tag),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, AiIdx, 0, Hex).

wam_instruction_to_wat_bytes(set_variable(Xn), _Labels, Hex) :-
    instr_tag(set_variable, Tag),
    reg_name_to_index(Xn, XnIdx),
    encode_instr_hex(Tag, XnIdx, 0, Hex).

wam_instruction_to_wat_bytes(set_value(Xn), _Labels, Hex) :-
    instr_tag(set_value, Tag),
    reg_name_to_index(Xn, XnIdx),
    encode_instr_hex(Tag, XnIdx, 0, Hex).

wam_instruction_to_wat_bytes(set_constant(C), _Labels, Hex) :-
    instr_tag(set_constant, Tag),
    encode_constant_with_tag(C, ConstTag, Op1),
    %% op2 = value-cell tag hint.
    encode_instr_hex(Tag, Op1, ConstTag, Hex).

wam_instruction_to_wat_bytes(allocate, _Labels, Hex) :-
    instr_tag(allocate, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

wam_instruction_to_wat_bytes(deallocate, _Labels, Hex) :-
    instr_tag(deallocate, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

wam_instruction_to_wat_bytes(call(P, N), Labels, Hex) :-
    instr_tag(call, Tag),
    resolve_label(P, Labels, PC),
    encode_instr_hex(Tag, PC, N, Hex).

wam_instruction_to_wat_bytes(execute(P), Labels, Hex) :-
    instr_tag(execute, Tag),
    resolve_label(P, Labels, PC),
    encode_instr_hex(Tag, PC, 0, Hex).

wam_instruction_to_wat_bytes(proceed, _Labels, Hex) :-
    instr_tag(proceed, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

wam_instruction_to_wat_bytes(builtin_call(Op, N), _Labels, Hex) :-
    instr_tag(builtin_call, Tag),
    (   builtin_id(Op, BId) -> true ; BId = 255 ),
    encode_instr_hex(Tag, BId, N, Hex).

wam_instruction_to_wat_bytes(try_me_else(Label), Labels, Hex) :-
    instr_tag(try_me_else, Tag),
    resolve_label(Label, Labels, PC),
    encode_instr_hex(Tag, PC, 0, Hex).

wam_instruction_to_wat_bytes(retry_me_else(Label), Labels, Hex) :-
    instr_tag(retry_me_else, Tag),
    resolve_label(Label, Labels, PC),
    encode_instr_hex(Tag, PC, 0, Hex).

wam_instruction_to_wat_bytes(trust_me, _Labels, Hex) :-
    instr_tag(trust_me, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

wam_instruction_to_wat_bytes(cut_ite, _Labels, Hex) :-
    instr_tag(cut_ite, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

wam_instruction_to_wat_bytes(jump(Label), Labels, Hex) :-
    instr_tag(jump, Tag),
    resolve_label(Label, Labels, PC),
    encode_instr_hex(Tag, PC, 0, Hex).

%% begin_aggregate Type, ValReg, ResReg
%% op1 = (agg_type << 32) | val_reg_index, op2 = res_reg_index
wam_instruction_to_wat_bytes(begin_aggregate(Type, ValReg, ResReg), _Labels, Hex) :-
    instr_tag(begin_aggregate, Tag),
    agg_type_id(Type, TypeId),
    reg_name_to_index(ValReg, ValIdx),
    reg_name_to_index(ResReg, ResIdx),
    Op1 is (TypeId << 32) \/ ValIdx,
    encode_instr_hex(Tag, Op1, ResIdx, Hex).

wam_instruction_to_wat_bytes(end_aggregate(ValReg), _Labels, Hex) :-
    instr_tag(end_aggregate, Tag),
    reg_name_to_index(ValReg, ValIdx),
    encode_instr_hex(Tag, ValIdx, 0, Hex).

wam_instruction_to_wat_bytes(nop, _Labels, Hex) :-
    instr_tag(nop, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

%% fused_is_add(Dest, Src1, Src2): Dest := deref(Src1) + deref(Src2).
%% Emitted by peephole_fused_is_add when the body is `Dest is Src1+Src2`
%% with all three being registers (matches the sum_ints-style leaf
%% accumulator pattern). Bypasses +/2 compound allocation and the
%% $eval_arith recursive walk.
%% op1 layout: Dest (low 8) | Src1 (bits 8-15) | Src2 (bits 16-23).
wam_instruction_to_wat_bytes(fused_is_add(DestReg, Src1Reg, Src2Reg),
                             _Labels, Hex) :-
    instr_tag(fused_is_add, Tag),
    reg_name_to_index(DestReg, DestIdx),
    reg_name_to_index(Src1Reg, Src1Idx),
    reg_name_to_index(Src2Reg, Src2Idx),
    Op1 is DestIdx \/ (Src1Idx << 8) \/ (Src2Idx << 16),
    encode_instr_hex(Tag, Op1, 0, Hex).

%% fused_is_sub / fused_is_mul: same op1 layout as fused_is_add.
wam_instruction_to_wat_bytes(fused_is_sub(DestReg, Src1Reg, Src2Reg),
                             _Labels, Hex) :-
    instr_tag(fused_is_sub, Tag),
    reg_name_to_index(DestReg, DestIdx),
    reg_name_to_index(Src1Reg, Src1Idx),
    reg_name_to_index(Src2Reg, Src2Idx),
    Op1 is DestIdx \/ (Src1Idx << 8) \/ (Src2Idx << 16),
    encode_instr_hex(Tag, Op1, 0, Hex).
wam_instruction_to_wat_bytes(fused_is_mul(DestReg, Src1Reg, Src2Reg),
                             _Labels, Hex) :-
    instr_tag(fused_is_mul, Tag),
    reg_name_to_index(DestReg, DestIdx),
    reg_name_to_index(Src1Reg, Src1Idx),
    reg_name_to_index(Src2Reg, Src2Idx),
    Op1 is DestIdx \/ (Src1Idx << 8) \/ (Src2Idx << 16),
    encode_instr_hex(Tag, Op1, 0, Hex).

%% fused_is_add_const(Dest, Src, Const) and fused_is_mul_const:
%% Dest := deref(Src) <op> Const, where Const is a signed integer
%% literal. op1 layout: Dest (low 8) | Src (bits 8-15). op2 = Const.
%% Used for sequences like `N1 is N - 1` which WAM compiles as
%% `put_structure +/2; set_value N; set_constant -1; is/2`.
wam_instruction_to_wat_bytes(fused_is_add_const(DestReg, SrcReg, Const),
                             _Labels, Hex) :-
    instr_tag(fused_is_add_const, Tag),
    reg_name_to_index(DestReg, DestIdx),
    reg_name_to_index(SrcReg, SrcIdx),
    Op1 is DestIdx \/ (SrcIdx << 8),
    encode_instr_hex(Tag, Op1, Const, Hex).
wam_instruction_to_wat_bytes(fused_is_mul_const(DestReg, SrcReg, Const),
                             _Labels, Hex) :-
    instr_tag(fused_is_mul_const, Tag),
    reg_name_to_index(DestReg, DestIdx),
    reg_name_to_index(SrcReg, SrcIdx),
    Op1 is DestIdx \/ (SrcIdx << 8),
    encode_instr_hex(Tag, Op1, Const, Hex).

%% arg_direct and functor_direct: zero-operand instructions that
%% bypass the $execute_builtin br_table by invoking $builtin_arg /
%% $builtin_functor directly. Fires on `builtin_call(arg/3, 3)` and
%% `builtin_call(functor/3, 3)` respectively.
wam_instruction_to_wat_bytes(arg_direct, _Labels, Hex) :-
    instr_tag(arg_direct, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(functor_direct, _Labels, Hex) :-
    instr_tag(functor_direct, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(copy_term_direct, _Labels, Hex) :-
    instr_tag(copy_term_direct, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(univ_direct, _Labels, Hex) :-
    instr_tag(univ_direct, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(is_list_direct, _Labels, Hex) :-
    instr_tag(is_list_direct, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

%% arg_reg_direct(NReg, TReg, DestReg): specialized arg/3 for the
%% pattern `put_value N, A1; put_value T, A2; put_variable Dest, A3;
%% arg_direct`. Reads N and T directly from their source registers
%% and writes the result straight to Dest, skipping the A1/A2/A3
%% intermediate bindings (and their trail pushes on backtrack-safe
%% write-once regs).
%% op1 layout: NIdx (low 8) | TIdx (bits 8-15) | DestIdx (bits 16-23).
wam_instruction_to_wat_bytes(arg_reg_direct(NReg, TReg, DestReg),
                             _Labels, Hex) :-
    instr_tag(arg_reg_direct, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(DestReg, DIdx),
    Op1 is NIdx \/ (TIdx << 8) \/ (DIdx << 16),
    encode_instr_hex(Tag, Op1, 0, Hex).

%% arg_lit_direct(NLiteral, TReg, DestReg): specialized arg/3 for the
%% pattern `put_constant N, A1; put_value T, A2; put_variable Dest, A3;
%% arg_direct`. Same as arg_reg_direct but N is a compile-time integer
%% literal (no deref needed for it, no tag check).
%% op1 layout: TIdx (low 8) | DestIdx (bits 8-15).  op2 = N.
wam_instruction_to_wat_bytes(arg_lit_direct(N, TReg, DestReg),
                             _Labels, Hex) :-
    instr_tag(arg_lit_direct, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(DestReg, DIdx),
    Op1 is TIdx \/ (DIdx << 8),
    encode_instr_hex(Tag, Op1, N, Hex).

%% arg_to_a1_reg / arg_to_a1_lit: fusion of arg_reg_direct +
%% put_value(Dest, A1). The next call's A1 always needs the arg
%% value, so fold the put_value into the arg instruction — saves
%% one instruction dispatch + one 12-byte reg-to-reg copy per call.
%% Dest is still written (the WAM reg may be read again later or
%% referenced via the env frame), but A1 is written in the same
%% pass using the cached arg tag/payload.
%% Same op1 layout as arg_reg_direct / arg_lit_direct.
wam_instruction_to_wat_bytes(arg_to_a1_reg(NReg, TReg, DestReg),
                             _Labels, Hex) :-
    instr_tag(arg_to_a1_reg, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(DestReg, DIdx),
    Op1 is NIdx \/ (TIdx << 8) \/ (DIdx << 16),
    encode_instr_hex(Tag, Op1, 0, Hex).
wam_instruction_to_wat_bytes(arg_to_a1_lit(N, TReg, DestReg),
                             _Labels, Hex) :-
    instr_tag(arg_to_a1_lit, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(DestReg, DIdx),
    Op1 is TIdx \/ (DIdx << 8),
    encode_instr_hex(Tag, Op1, N, Hex).

%% arg_call_reg_3 / arg_call_lit_3: fusion of arg_to_a1_{reg,lit} +
%% put_value(A2_src, A2) + put_variable(RetDest, A3) + call(Pred, 3).
%% Applies specifically to 3-arg calls (the common shape for
%% sum_ints_args, term_depth_args, and similar recursive walkers).
%% Saves three instruction dispatches per call.
%%
%% op1 layout:
%%   bits  0-7:  NIdx  (source reg for the arg index; unused in lit variant)
%%   bits  8-15: TIdx  (source reg for the compound)
%%   bits 16-23: ArgDestIdx (receives arg value + goes to A1)
%%   bits 24-31: A2SrcIdx (copied to A2)
%%   bits 32-39: RetDestIdx (fresh var for A3)
%% op2: target predicate PC (from resolve_label).
%% For arg_call_lit_3, NIdx is 0xFF (sentinel) and op1 also carries
%% the integer literal in bits 40-55 (16 bits — adequate for
%% single-digit arg indices which is the only realistic literal case).
wam_instruction_to_wat_bytes(
    arg_call_reg_3(NReg, TReg, ArgDestReg, A2SrcReg, RetDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_reg_3, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2SrcReg, A2Idx),
    reg_name_to_index(RetDestReg, RdIdx),
    Op1 is NIdx
        \/ (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (RdIdx << 32),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).
wam_instruction_to_wat_bytes(
    arg_call_lit_3(N, TReg, ArgDestReg, A2SrcReg, RetDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_lit_3, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2SrcReg, A2Idx),
    reg_name_to_index(RetDestReg, RdIdx),
    Op1 is (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (RdIdx << 32)
        \/ ((N /\ 0xFFFF) << 40),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% _dead variants: same as arg_call_{reg,lit}_3 but ArgDest is known
%% to be dead after the call, so the instruction only writes to A1
%% (skipping the 12-byte write to ArgDest's env-frame slot). Emitted
%% by peephole_arg_call_3 when reg_used_before_clause_end/2 says no.
%% Operand layout identical to the live variants (the ArgDestIdx field
%% is still encoded but ignored at runtime).
wam_instruction_to_wat_bytes(
    arg_call_reg_3_dead(NReg, TReg, ArgDestReg, A2SrcReg, RetDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_reg_3_dead, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2SrcReg, A2Idx),
    reg_name_to_index(RetDestReg, RdIdx),
    Op1 is NIdx
        \/ (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (RdIdx << 32),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).
wam_instruction_to_wat_bytes(
    arg_call_lit_3_dead(N, TReg, ArgDestReg, A2SrcReg, RetDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_lit_3_dead, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2SrcReg, A2Idx),
    reg_name_to_index(RetDestReg, RdIdx),
    Op1 is (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (RdIdx << 32)
        \/ ((N /\ 0xFFFF) << 40),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% arg_call_reg_1 / arg_call_lit_1: fusion of arg_to_a1_{reg,lit} +
%% call(Pred, 1). Matches the shape `pred(X) :- arg(I, T, X), pred2(X).`
%% After arg_to_a1 has already loaded A1, the call with arity 1 has
%% nothing else to set up — just save CP and jump. This collapses the
%% 2-instruction window to one dispatch.
%%
%% op1 layout:
%%   bits  0-7:  NIdx  (source reg for arg index; unused in lit variant)
%%   bits  8-15: TIdx  (source reg for the compound)
%%   bits 16-23: ArgDestIdx (receives arg value + goes to A1)
%%   bits 40-55: N (lit variant only, 16-bit literal index)
%% op2: target predicate PC.
wam_instruction_to_wat_bytes(
    arg_call_reg_1(NReg, TReg, ArgDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_reg_1, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    Op1 is NIdx \/ (TIdx << 8) \/ (AdIdx << 16),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).
wam_instruction_to_wat_bytes(
    arg_call_lit_1(N, TReg, ArgDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_lit_1, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    Op1 is (TIdx << 8) \/ (AdIdx << 16) \/ ((N /\ 0xFFFF) << 40),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% _dead variants: ArgDest is provably dead after the call, so only A1
%% is written. Encoded identically to the live variant; the ArgDestIdx
%% field is ignored at runtime.
wam_instruction_to_wat_bytes(
    arg_call_reg_1_dead(NReg, TReg, ArgDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_reg_1_dead, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    Op1 is NIdx \/ (TIdx << 8) \/ (AdIdx << 16),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).
wam_instruction_to_wat_bytes(
    arg_call_lit_1_dead(N, TReg, ArgDestReg, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_lit_1_dead, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    Op1 is (TIdx << 8) \/ (AdIdx << 16) \/ ((N /\ 0xFFFF) << 40),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% arg_call_reg_2 / arg_call_lit_2: fusion for arity-2 calls. Handles
%% two sub-patterns distinguished by an IsVar flag in op1 bit 32:
%%   IsVar=0: arg_to_a1_* + put_value(A2Reg, 'A2') + call(Pred, 2)
%%            → A2Reg is a source (copied to A2).
%%   IsVar=1: arg_to_a1_* + put_variable(A2Reg, 'A2') + call(Pred, 2)
%%            → A2Reg is a fresh-var destination (A2 and A2Reg both
%%              receive a new heap ref cell).
%%
%% op1 layout:
%%   bits  0-7:  NIdx (reg variant) / unused (lit variant)
%%   bits  8-15: TIdx
%%   bits 16-23: ArgDestIdx
%%   bits 24-31: A2Idx
%%   bit  32:    IsVar flag
%%   bits 40-55: N (lit variant only)
%% op2: target predicate PC.
wam_instruction_to_wat_bytes(
    arg_call_reg_2(NReg, TReg, ArgDestReg, A2Reg, IsVar, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_reg_2, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2Reg, A2Idx),
    IsVarBit is IsVar /\ 1,
    Op1 is NIdx
        \/ (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (IsVarBit << 32),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).
wam_instruction_to_wat_bytes(
    arg_call_lit_2(N, TReg, ArgDestReg, A2Reg, IsVar, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_lit_2, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2Reg, A2Idx),
    IsVarBit is IsVar /\ 1,
    Op1 is (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (IsVarBit << 32)
        \/ ((N /\ 0xFFFF) << 40),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% _dead variants: ArgDest write elided; same encoding otherwise.
wam_instruction_to_wat_bytes(
    arg_call_reg_2_dead(NReg, TReg, ArgDestReg, A2Reg, IsVar, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_reg_2_dead, Tag),
    reg_name_to_index(NReg, NIdx),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2Reg, A2Idx),
    IsVarBit is IsVar /\ 1,
    Op1 is NIdx
        \/ (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (IsVarBit << 32),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).
wam_instruction_to_wat_bytes(
    arg_call_lit_2_dead(N, TReg, ArgDestReg, A2Reg, IsVar, Pred),
    Labels, Hex) :-
    instr_tag(arg_call_lit_2_dead, Tag),
    reg_name_to_index(TReg, TIdx),
    reg_name_to_index(ArgDestReg, AdIdx),
    reg_name_to_index(A2Reg, A2Idx),
    IsVarBit is IsVar /\ 1,
    Op1 is (TIdx << 8)
        \/ (AdIdx << 16)
        \/ (A2Idx << 24)
        \/ (IsVarBit << 32)
        \/ ((N /\ 0xFFFF) << 40),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% deallocate_proceed: fusion of `deallocate + proceed`. Zero
%% operands. Restores CP+env_base from the frame header, pops the
%% frame, then jumps to CP (returning to caller) — same behavior as
%% deallocate followed by proceed, with one less dispatch.
wam_instruction_to_wat_bytes(deallocate_proceed, _Labels, Hex) :-
    instr_tag(deallocate_proceed, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

%% deallocate_builtin_proceed: fusion of
%%   `deallocate + builtin_call(Op, Arity) + proceed`
%% Runs deallocate, invokes the builtin (via $execute_builtin), then
%% on success jumps to CP. On failure returns 0 to trigger backtrack
%% — same semantics as the 3-instruction sequence. Fires on common
%% clause-end shapes with a final `=/2`, `is/2`, or `!/0`.
%% op1: builtin id (mirrors builtin_call's encoding).
%% op2: builtin arity.
wam_instruction_to_wat_bytes(deallocate_builtin_proceed(Op, N),
                             _Labels, Hex) :-
    instr_tag(deallocate_builtin_proceed, Tag),
    (   builtin_id(Op, OpId)
    ->  true
    ;   OpId = 0
    ),
    encode_instr_hex(Tag, OpId, N, Hex).

%% deallocate_<X>_direct_proceed: fusion of
%%   `deallocate + <X>_direct + proceed`
%% where <X>_direct is one of the zero-operand direct-dispatch
%% builtins (arg_direct, functor_direct, copy_term_direct,
%% univ_direct, is_list_direct). Each calls the specific $builtin_*
%% directly — no br_table dispatch on the direct-instr tag, no
%% $execute_builtin wrapper. Zero operands.
%% Fires in bench_arg_read, bench_functor_read, bench_copy_flat,
%% bench_copy_nested, bench_univ_decomp — the single-builtin
%% microbench workloads.
wam_instruction_to_wat_bytes(deallocate_arg_direct_proceed, _Labels, Hex) :-
    instr_tag(deallocate_arg_direct_proceed, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(deallocate_functor_direct_proceed, _Labels, Hex) :-
    instr_tag(deallocate_functor_direct_proceed, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(deallocate_copy_term_direct_proceed, _Labels, Hex) :-
    instr_tag(deallocate_copy_term_direct_proceed, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(deallocate_univ_direct_proceed, _Labels, Hex) :-
    instr_tag(deallocate_univ_direct_proceed, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).
wam_instruction_to_wat_bytes(deallocate_is_list_direct_proceed, _Labels, Hex) :-
    instr_tag(deallocate_is_list_direct_proceed, Tag),
    encode_instr_hex(Tag, 0, 0, Hex).

%% builtin_proceed: fusion of `builtin_call(Op, Arity) + proceed`
%% WITHOUT a preceding deallocate. Fires in clauses that don't use
%% Y-registers (no allocate/deallocate) and end with a terminal
%% builtin — e.g. `bench_is_arith/0` ends with
%%   builtin_call(is/2, 2) + proceed
%% and `fib/3` clause 2 ends with
%%   builtin_call(!/0, 0) + proceed
%% op1 = builtin id, op2 = arity.
wam_instruction_to_wat_bytes(builtin_proceed(Op, N), _Labels, Hex) :-
    instr_tag(builtin_proceed, Tag),
    (   builtin_id(Op, OpId)
    ->  true
    ;   OpId = 0
    ),
    encode_instr_hex(Tag, OpId, N, Hex).

%% type_dispatch_a1: first-argument indexing via A1's tag. Dispatches
%% directly to one of four clause bodies based on A1's runtime tag,
%% bypassing the try_me_else/retry_me_else/trust_me chain. Tags:
%%   0 (atom)     -> AtomLbl
%%   1 (integer)  -> IntLbl
%%   3 (compound) -> CmpdLbl
%%   anything else (ref/unbound/float/list/bool) -> DefaultLbl
%% A 0 target for any slot means "no dispatch for that tag" -> fall
%% through to the next instruction (typically try_me_else).
%%
%% op1 layout: atom_target (low 32) | int_target (high 32).
%% op2 layout: cmpd_target (low 32) | default_target (high 32).
%%
%% Semantically safe even without cuts: retry_me_else and trust_me
%% are guarded for cp_count=0 (they no-op when the flow arrived via
%% first-arg indexing without a preceding try_me_else CP push).
wam_instruction_to_wat_bytes(
    type_dispatch_a1(AtomLbl, IntLbl, CmpdLbl, DefaultLbl),
    Labels, Hex) :-
    instr_tag(type_dispatch_a1, Tag),
    resolve_opt_label(AtomLbl,    Labels, AtomPC),
    resolve_opt_label(IntLbl,     Labels, IntPC),
    resolve_opt_label(CmpdLbl,    Labels, CmpdPC),
    resolve_opt_label(DefaultLbl, Labels, DefaultPC),
    Op1 is (AtomPC /\ 0xFFFFFFFF) \/ ((IntPC /\ 0xFFFFFFFF) << 32),
    Op2 is (CmpdPC /\ 0xFFFFFFFF) \/ ((DefaultPC /\ 0xFFFFFFFF) << 32),
    encode_instr_hex(Tag, Op1, Op2, Hex).

%% resolve_opt_label(+Label, +Labels, -PC)
%  Like resolve_label/3 but allows atom 0 (or integer 0) as "no
%  dispatch" — returns PC = 0. Real labels resolve normally.
resolve_opt_label(0, _, 0) :- !.
resolve_opt_label(Lbl, Labels, PC) :- resolve_label(Lbl, Labels, PC).

%% tail_call_5: fusion of a 5-arg tail-call setup window:
%%   put_value(R1,A1) put_value(R2,A2) put_value(R3,A3)
%%   put_value(R4,A4) put_value(R5,A5) deallocate execute(Pred)
%% → tail_call_5(R1, R2, R3, R4, R5, Pred)
%% Saves six instruction dispatches per recursive tail on 5-arity
%% predicates (sum_ints_args/5 and term_depth_args/5 in the bench).
%%
%% op1 layout: R1Idx (0-7), R2Idx (8-15), R3Idx (16-23),
%%             R4Idx (24-31), R5Idx (32-39).
%% op2: target predicate PC.
wam_instruction_to_wat_bytes(
    tail_call_5(R1, R2, R3, R4, R5, Pred),
    Labels, Hex) :-
    instr_tag(tail_call_5, Tag),
    reg_name_to_index(R1, I1),
    reg_name_to_index(R2, I2),
    reg_name_to_index(R3, I3),
    reg_name_to_index(R4, I4),
    reg_name_to_index(R5, I5),
    Op1 is I1
        \/ (I2 << 8)
        \/ (I3 << 16)
        \/ (I4 << 24)
        \/ (I5 << 32),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% tail_call_5_c1_lit: K=5 tail-call fusion where the first argument
%% is an integer literal rather than a register copy. Matches
%%   put_constant(K, A1) put_value(R2, A2) put_value(R3, A3)
%%   put_value(R4, A4) put_value(R5, A5) deallocate execute(Pred)
%% Fires on sum_ints/3 clause 2 where the recursive tail into
%% sum_ints_args/5 passes literal index 1 as the first argument.
%%
%% op1 layout: R2Idx (8-15), R3Idx (16-23), R4Idx (24-31),
%%             R5Idx (32-39), C1 literal (40-55, 16-bit signed).
%% op2: target PC.
%% Falls through the peephole if C1 doesn't fit in 16 signed bits
%% (unusual — most put_constant first args are small indices).
wam_instruction_to_wat_bytes(
    tail_call_5_c1_lit(C1, R2, R3, R4, R5, Pred),
    Labels, Hex) :-
    instr_tag(tail_call_5_c1_lit, Tag),
    reg_name_to_index(R2, I2),
    reg_name_to_index(R3, I3),
    reg_name_to_index(R4, I4),
    reg_name_to_index(R5, I5),
    Op1 is (I2 << 8)
        \/ (I3 << 16)
        \/ (I4 << 24)
        \/ (I5 << 32)
        \/ ((C1 /\ 0xFFFF) << 40),
    resolve_label(Pred, Labels, PC),
    encode_instr_hex(Tag, Op1, PC, Hex).

%% switch_on_const header: op1 layout = (count << 32) | reg_idx (0 or 1).
wam_instruction_to_wat_bytes(switch_on_const(RegIdx, Count), _Labels, Hex) :-
    instr_tag(switch_on_const, Tag),
    Op1 is (Count << 32) \/ RegIdx,
    encode_instr_hex(Tag, Op1, 0, Hex).

%% switch_entry: op1 = constant payload (full i64, e.g. atom hash or
%% integer value), op2 = (const_tag << 32) | target_pc. Scanned by
%% switch_on_const at runtime; never executed directly.
wam_instruction_to_wat_bytes(switch_entry(CTag, CPayload, Label), Labels, Hex) :-
    instr_tag(switch_entry, Tag),
    resolve_label(Label, Labels, TargetPC),
    Op2 is (CTag << 32) \/ (TargetPC /\ 0xFFFFFFFF),
    encode_instr_hex(Tag, CPayload, Op2, Hex).

%% switch_on_struct header: op1 = (count << 32) | reg_idx.
wam_instruction_to_wat_bytes(switch_on_struct(RegIdx, Count), _Labels, Hex) :-
    instr_tag(switch_on_struct, Tag),
    Op1 is (Count << 32) \/ RegIdx,
    encode_instr_hex(Tag, Op1, 0, Hex).

%% switch_struct_entry: op1 = functor_hash (DJB2 of "Functor/Arity" atom),
%% op2 = (arity << 32) | target_pc. Scanned by switch_on_struct.
wam_instruction_to_wat_bytes(switch_struct_entry(FHash, Arity, Label), Labels, Hex) :-
    instr_tag(switch_struct_entry, Tag),
    resolve_label(Label, Labels, TargetPC),
    Op2 is (Arity << 32) \/ (TargetPC /\ 0xFFFFFFFF),
    encode_instr_hex(Tag, FHash, Op2, Hex).

%% switch_on_term_hdr: op1 = (const_count << 32) | reg_idx,
%% op2 = struct_count. The CC following switch_entry records are const
%% cases; the SC after those are switch_struct_entry records.
wam_instruction_to_wat_bytes(switch_on_term_hdr(RegIdx, CC, SC), _Labels, Hex) :-
    instr_tag(switch_on_term_hdr, Tag),
    Op1 is (CC << 32) \/ RegIdx,
    encode_instr_hex(Tag, Op1, SC, Hex).

%% parse_switch_entries(+Parts, -Entries)
%  Parts is a list of strings like ["10:default,", "20:L_my_fact_1_2,",
%  "30:L_my_fact_1_3"]. Each entry strips the trailing comma and splits
%  on ':'. The key is the constant (integer or atom), the value is a
%  label string.
parse_switch_entries([], []).
parse_switch_entries([Part|Rest], [Const-Label|Entries]) :-
    clean_comma(Part, Clean),
    split_string(Clean, ":", "", [KeyStr, LabelStr0]),
    string_to_atom(LabelStr0, Label),
    parse_switch_const(KeyStr, Const),
    parse_switch_entries(Rest, Entries).

parse_switch_const(Str, int(Val)) :-
    number_string(Val, Str),
    integer(Val), !.
parse_switch_const(Str, atom(Atom)) :-
    atom_string(Atom, Str).

%% build_switch_instrs(+RegIdx, +Entries, -MultiResult)
%  Produces a multi(...) pseudo-instruction that wam_lines_to_instrs
%  expands into (1 + N_indexed) physical WAM instructions, where
%  N_indexed is the number of non-default entries.
%
%  Entries tagged with the `default` label (emitted by the canonical
%  WAM indexer for the FIRST clause) are dropped from the indexed
%  table: the first clause is already the natural fall-through target
%  after the switch + entries + try_me_else chain, so an indexed
%  jump is unnecessary. Dropping it also avoids target=0 label
%  resolution (the `default` atom is not a real label in the global
%  label map).
build_switch_instrs(RegIdx, AllEntries, multi([Header|EntryInstrs])) :-
    exclude(is_default_entry, AllEntries, Entries),
    length(Entries, Count),
    Header = switch_on_const(RegIdx, Count),
    maplist(entry_to_instr, Entries, EntryInstrs).

is_default_entry(_-default).

entry_to_instr(int(V)-Label, switch_entry(1, V, Label)).
entry_to_instr(atom(A)-Label, switch_entry(0, H, Label)) :-
    atom_hash_i64(A, H),
    record_atom(A).

%% parse_struct_entries(+Parts, -Entries)
%  Parts is a list of strings like ["bar/1:default,", "baz/2:L_X,",
%  "qux/3:L_Y"]. Each entry splits on the LAST ':' (functor names may
%  themselves contain '/', but labels don''t contain ':'). The key is
%  an atom ''Functor/Arity'' (matching atom_hash_i64 output).
parse_struct_entries([], []).
parse_struct_entries([Part|Rest], [struct(FA, Arity)-Label|Entries]) :-
    clean_comma(Part, Clean),
    split_struct_entry(Clean, FAStr, LabelStr),
    string_to_atom(LabelStr, Label),
    atom_string(FA, FAStr),
    %% Extract arity from functor/arity string (last '/' token).
    split_string(FAStr, "/", "", SParts),
    last(SParts, ArityStr),
    number_string(Arity, ArityStr),
    parse_struct_entries(Rest, Entries).

%% split_struct_entry(+Str, -FunctorArity, -Label)
%  Splits "functor/arity:label" on the LAST ':' — functor names never
%  contain ':' but labels may contain '/', so we can't use split_string
%  naively.
split_struct_entry(Str, FA, Label) :-
    string_length(Str, Len),
    last_colon(Str, Len, Pos),
    sub_string(Str, 0, Pos, _, FA),
    Pos1 is Pos + 1,
    sub_string(Str, Pos1, _, 0, Label).

last_colon(Str, Len, Pos) :-
    Len > 0,
    Idx is Len - 1,
    sub_string(Str, Idx, 1, _, C),
    (   C == ":" -> Pos = Idx
    ;   last_colon(Str, Idx, Pos)
    ).

%% build_switch_struct_instrs(+RegIdx, +Entries, -MultiResult)
%  Same shape as build_switch_instrs but for structure entries.
%  Entries are struct(FunctorArityAtom, Arity)-Label pairs.
build_switch_struct_instrs(RegIdx, AllEntries, multi([Header|EntryInstrs])) :-
    exclude(is_default_entry, AllEntries, Entries),
    length(Entries, Count),
    Header = switch_on_struct(RegIdx, Count),
    maplist(struct_entry_to_instr, Entries, EntryInstrs).

struct_entry_to_instr(struct(FA, Arity)-Label,
                      switch_struct_entry(H, Arity, Label)) :-
    atom_hash_i64(FA, H),
    record_atom(FA).

%% parse_term_entries(+Parts, -ConstEntries, -StructEntries)
%  Parts is like ["constant:0:default,", "1:L_2,", "nil:L_3,",
%  "structure:leaf/1:L_4,", "tree/2:L_5"]. The "constant:" and
%  "structure:" prefixes switch sections. Section is "none" when the
%  canonical emitter has no entries of that type.
parse_term_entries(Parts, ConstEntries, StructEntries) :-
    parse_term_entries_(Parts, const, [], [], CRev, SRev),
    reverse(CRev, ConstEntries),
    reverse(SRev, StructEntries).

parse_term_entries_([], _, CAcc, SAcc, CAcc, SAcc).
parse_term_entries_([P|Rest], Section, CAcc, SAcc, COut, SOut) :-
    clean_comma(P, Clean),
    (   string_concat("constant:", Body, Clean)
    ->  NewSection = const,
        parse_term_section_entry(Body, NewSection, E),
        add_term_entry(NewSection, E, CAcc, SAcc, C1, S1),
        parse_term_entries_(Rest, NewSection, C1, S1, COut, SOut)
    ;   string_concat("structure:", Body, Clean)
    ->  NewSection = struct,
        parse_term_section_entry(Body, NewSection, E),
        add_term_entry(NewSection, E, CAcc, SAcc, C1, S1),
        parse_term_entries_(Rest, NewSection, C1, S1, COut, SOut)
    ;   Clean == "none"
    ->  parse_term_entries_(Rest, Section, CAcc, SAcc, COut, SOut)
    ;   parse_term_section_entry(Clean, Section, E),
        add_term_entry(Section, E, CAcc, SAcc, C1, S1),
        parse_term_entries_(Rest, Section, C1, S1, COut, SOut)
    ).

parse_term_section_entry(Body, const, Const-Label) :-
    split_string(Body, ":", "", [KeyStr, LabelStr]),
    string_to_atom(LabelStr, Label),
    parse_switch_const(KeyStr, Const).
parse_term_section_entry(Body, struct, struct(FA, Arity)-Label) :-
    split_struct_entry(Body, FAStr, LabelStr),
    string_to_atom(LabelStr, Label),
    atom_string(FA, FAStr),
    split_string(FAStr, "/", "", SParts),
    last(SParts, ArityStr),
    number_string(Arity, ArityStr).

add_term_entry(const, E, CAcc, SAcc, [E|CAcc], SAcc).
add_term_entry(struct, E, CAcc, SAcc, CAcc, [E|SAcc]).

%% build_switch_term_instrs(+RegIdx, +ConstEntries, +StructEntries, -Multi)
%  Encodes switch_on_term as:
%    [header, const_entries..., struct_entries...]
%  Header: switch_on_term_hdr(reg_idx, const_count, struct_count)
%  Const entries: switch_entry (tag 32, reused from switch_on_const)
%  Struct entries: switch_struct_entry (tag 34)
build_switch_term_instrs(RegIdx, AllConsts, AllStructs,
                         multi([Header|AllEntries])) :-
    exclude(is_default_entry, AllConsts, Consts),
    exclude(is_default_entry, AllStructs, Structs),
    length(Consts, CC),
    length(Structs, SC),
    Header = switch_on_term_hdr(RegIdx, CC, SC),
    maplist(entry_to_instr, Consts, CInstrs),
    maplist(struct_entry_to_instr, Structs, SInstrs),
    append(CInstrs, SInstrs, AllEntries).

agg_type_id(sum, 0).
agg_type_id(count, 1).
agg_type_id(max, 2).
agg_type_id(min, 3).
agg_type_id(collect, 4).
agg_type_id(_, 0).  % default to sum

wam_instruction_to_wat_bytes(neck_cut_test(GuardOp, GuardArity, ElseLabel), Labels, Hex) :-
    instr_tag(neck_cut_test, Tag),
    resolve_label(ElseLabel, Labels, ElsePC),
    (   builtin_id(GuardOp, GuardId) -> true ; GuardId = 255 ),
    %% op1 = else PC, op2 = (guard_builtin_id << 32) | guard_arity
    Op2 is (GuardId << 32) \/ GuardArity,
    encode_instr_hex(Tag, ElsePC, Op2, Hex).

% --- Encoding helpers ---

%% encode_constant(+C, -I64Val)
%  Encodes a Prolog constant as an i64 value.
%  For atoms: the hash. For integers: the raw value.
encode_constant(C, Val) :- encode_constant_with_tag(C, _Tag, Val).

%% Atom collection for the lexicographic atom name table. Atoms used
%% anywhere in the compiled module are accumulated here during
%% encoding; assemble_atom_table/3 then emits a sorted table with
%% (hash, name_offset, name_length) triples plus a concatenated string
%% pool. The runtime's $atom_compare uses the table so @</2 on atoms
%% orders by lexicographic name rather than DJB2 hash (the previous
%% behavior, which was deterministic but non-standard).
:- dynamic seen_atom/1.

record_atom(A) :-
    (   atom(A)
    ->  (seen_atom(A) -> true ; assertz(seen_atom(A)))
    ;   true
    ).

%% encode_constant_with_tag(+C, -Tag, -I64Val)
%  Encodes a Prolog constant as (Tag, Value) where Tag is the runtime
%  value-cell tag (0 = atom, 1 = integer, 2 = float) and I64Val is the
%  payload stored in the instruction's op1 field.
%
%  The canonical WAM layer hands constants down as strings parsed
%  from WAM assembly text, not as typed Prolog terms — so "42" arrives
%  as the string "42", not the integer 42. We parse strings back into
%  their original types here so the type/1 family of builtins
%  (integer/1, atom/1, etc.) can dispatch on a meaningful runtime
%  tag at the A_i register. Prior to this change, all constants
%  unconditionally stored tag=0 (atom), which made integer/1 always
%  fail and atom/1 always "succeed" — including on integer constants.
encode_constant_with_tag(atom(A), 0, Hash) :- !,
    atom_hash_i64(A, Hash),
    record_atom(A).
encode_constant_with_tag(integer(I), 1, I) :- !.
encode_constant_with_tag(I, 1, I) :- integer(I), !.
encode_constant_with_tag(F, 2, Bits) :- float(F), !, Bits is float_integer_part(F).
encode_constant_with_tag(A, 0, Hash) :- atom(A), !,
    atom_hash_i64(A, Hash),
    record_atom(A).
encode_constant_with_tag(S, Tag, Val) :-
    string(S), !,
    (   number_string(I, S), integer(I)
    ->  Tag = 1, Val = I
    ;   atom_string(A, S),
        atom_hash_i64(A, Hash),
        record_atom(A),
        Tag = 0, Val = Hash
    ).
encode_constant_with_tag(_, 0, 0).

%% encode_structure_op1(+FSlashArity, -I64)
%  Encodes a functor/arity descriptor (e.g. 'f/2') as an i64 payload.
%  Layout: high 32 bits = arity, low 32 bits = DJB2 hash of the full
%  'Functor/Arity' atom form. The low 32 bits retain the existing hash
%  behavior so backwards compatibility with earlier encoding is exact
%  when arity fits in 31 bits (it always does here — atom_hash_i64
%  already limits the hash to 2^31-1). The high 32 bits let the runtime
%  recover arity for functor/3 and arg/3 without a separate table.
encode_structure_op1(FSlashArity, Op1) :-
    atom_hash_i64(FSlashArity, Hash),
    functor_arity_of(FSlashArity, Arity),
    record_atom(FSlashArity),
    Op1 is (Arity << 32) \/ (Hash /\ 0xFFFFFFFF).

%% functor_arity_of(+FSlashArity, -Arity)
%  Extracts the integer arity from an atom of the form 'Functor/Arity'.
%  Falls back to 0 if the atom does not match this shape.
functor_arity_of(FSlashArity, Arity) :-
    atom_string(FSlashArity, Str),
    (   split_string(Str, "/", "", Parts),
        Parts = [_, AStr],
        number_string(A, AStr)
    ->  Arity = A
    ;   Arity = 0
    ).

%% resolve_label(+Label, +LabelMap, -PC)
resolve_label(Label, Labels, PC) :-
    (   member(Label-PC, Labels) -> true
    ;   atom_string(Label, LStr),
        member(LStr-PC, Labels) -> true
    ;   PC = 0
    ).

%% encode_instr_hex(+Tag, +Op1, +Op2, -HexString)
%  Produces a 20-byte little-endian hex escape string for a WAT data segment.
encode_instr_hex(Tag, Op1, Op2, Hex) :-
    i32_to_le_hex(Tag, TagHex),
    i64_to_le_hex(Op1, Op1Hex),
    i64_to_le_hex(Op2, Op2Hex),
    atomic_list_concat([TagHex, Op1Hex, Op2Hex], Hex).

%% i32_to_le_hex(+Val, -Hex)
%  Encodes a 32-bit integer as 4 little-endian hex escape bytes.
i32_to_le_hex(Val, Hex) :-
    V is Val /\ 0xFFFFFFFF,
    B0 is V /\ 0xFF,
    B1 is (V >> 8) /\ 0xFF,
    B2 is (V >> 16) /\ 0xFF,
    B3 is (V >> 24) /\ 0xFF,
    format(atom(Hex), "\\~|~`0t~16r~2+\\~|~`0t~16r~2+\\~|~`0t~16r~2+\\~|~`0t~16r~2+",
           [B0, B1, B2, B3]).

%% i64_to_le_hex(+Val, -Hex)
%  Encodes a 64-bit integer as 8 little-endian hex escape bytes.
i64_to_le_hex(Val, Hex) :-
    V is Val /\ 0xFFFFFFFFFFFFFFFF,
    Lo is V /\ 0xFFFFFFFF,
    Hi is (V >> 32) /\ 0xFFFFFFFF,
    i32_to_le_hex(Lo, LoHex),
    i32_to_le_hex(Hi, HiHex),
    atom_concat(LoHex, HiHex, Hex).

% ============================================================================
% PHASE 2: WAM line parser (same pattern as wam_go_target.pl)
% ============================================================================

%% wam_lines_to_instrs(+Lines, +PC, -Instrs, -Labels)
%  Parse WAM assembly text lines into instruction terms and label map.
%  A single line may yield multiple physical instructions via the
%  multi(Is) wrapper (used by switch_on_const to emit a header plus
%  one switch_entry per indexed constant).
wam_lines_to_instrs([], _, [], []).
wam_lines_to_instrs([Line|Rest], PC, Instrs, Labels) :-
    normalize_space(string(Trimmed), Line),
    (   Trimmed = ""
    ->  wam_lines_to_instrs(Rest, PC, Instrs, Labels)
    ;   sub_string(Trimmed, _, 1, 0, ":")
    ->  sub_string(Trimmed, 0, _, 1, LabelName),
        Labels = [LabelName-PC|RestLabels],
        wam_lines_to_instrs(Rest, PC, Instrs, RestLabels)
    ;   split_string(Trimmed, " \t", " \t", Parts),
        Parts \== []
    ->  wam_parts_to_instr(Parts, Instr),
        (   Instr = multi(Is)
        ->  length(Is, N),
            PC1 is PC + N,
            append(Is, RestInstrs, Instrs)
        ;   PC1 is PC + 1,
            Instrs = [Instr|RestInstrs]
        ),
        wam_lines_to_instrs(Rest, PC1, RestInstrs, Labels)
    ;   wam_lines_to_instrs(Rest, PC, Instrs, Labels)
    ).

%% wam_lines_to_instrs_with_labels(+Lines, +Counter, -InstrsWithLabels)
%  Like wam_lines_to_instrs but keeps label markers as label(Name)
%  pseudo-instructions in the list. This allows peephole passes to
%  transform the instruction list (adding/removing instructions)
%  without breaking label-to-PC mappings.
wam_lines_to_instrs_with_labels([], _, []).
wam_lines_to_instrs_with_labels([Line|Rest], C, Result) :-
    normalize_space(string(Trimmed), Line),
    (   Trimmed = ""
    ->  wam_lines_to_instrs_with_labels(Rest, C, Result)
    ;   sub_string(Trimmed, _, 1, 0, ":")
    ->  sub_string(Trimmed, 0, _, 1, LabelName),
        Result = [label(LabelName)|RestResult],
        wam_lines_to_instrs_with_labels(Rest, C, RestResult)
    ;   split_string(Trimmed, " \t", " \t", Parts),
        Parts \== []
    ->  wam_parts_to_instr(Parts, Instr),
        (   Instr = multi(Is)
        ->  length(Is, N),
            C1 is C + N,
            append(Is, RestResult, Result)
        ;   C1 is C + 1,
            Result = [Instr|RestResult]
        ),
        wam_lines_to_instrs_with_labels(Rest, C1, RestResult)
    ;   wam_lines_to_instrs_with_labels(Rest, C, Result)
    ).

%% extract_instrs_and_labels(+WithLabels, +PC, -Instrs, -Labels)
%  Strips label(Name) markers from the list, recording each label's
%  PC (the index of the next real instruction after the marker).
extract_instrs_and_labels([], _, [], []).
extract_instrs_and_labels([label(Name)|Rest], PC, Instrs, [Name-PC|Labels]) :-
    !,
    extract_instrs_and_labels(Rest, PC, Instrs, Labels).
extract_instrs_and_labels([Instr|Rest], PC, [Instr|Instrs], Labels) :-
    PC1 is PC + 1,
    extract_instrs_and_labels(Rest, PC1, Instrs, Labels).

%% wam_parts_to_instr(+Parts, -Instr)
%  Convert parsed WAM line parts to an instruction term.
wam_parts_to_instr(["get_constant", C, Ai], get_constant(CC, CAi)) :-
    clean_comma(C, CC), clean_comma(Ai, CAi0), atom_string(CAi, CAi0).
wam_parts_to_instr(["get_variable", Xn, Ai], get_variable(CXn, CAi)) :-
    clean_comma(Xn, CXn0), clean_comma(Ai, CAi0),
    atom_string(CXn, CXn0), atom_string(CAi, CAi0).
wam_parts_to_instr(["get_value", Xn, Ai], get_value(CXn, CAi)) :-
    clean_comma(Xn, CXn0), clean_comma(Ai, CAi0),
    atom_string(CXn, CXn0), atom_string(CAi, CAi0).
wam_parts_to_instr(["get_structure", F, Ai], get_structure(CF, CAi)) :-
    clean_comma(F, CF0), clean_comma(Ai, CAi0),
    atom_string(CF, CF0), atom_string(CAi, CAi0).
wam_parts_to_instr(["get_list", Ai], get_list(CAi)) :-
    clean_comma(Ai, CAi0), atom_string(CAi, CAi0).
wam_parts_to_instr(["unify_variable", Xn], unify_variable(CXn)) :-
    clean_comma(Xn, CXn0), atom_string(CXn, CXn0).
wam_parts_to_instr(["unify_value", Xn], unify_value(CXn)) :-
    clean_comma(Xn, CXn0), atom_string(CXn, CXn0).
wam_parts_to_instr(["unify_constant", C], unify_constant(CC)) :-
    clean_comma(C, CC).
wam_parts_to_instr(["put_constant", C, Ai], put_constant(CC, CAi)) :-
    clean_comma(C, CC), clean_comma(Ai, CAi0), atom_string(CAi, CAi0).
wam_parts_to_instr(["put_variable", Xn, Ai], put_variable(CXn, CAi)) :-
    clean_comma(Xn, CXn0), clean_comma(Ai, CAi0),
    atom_string(CXn, CXn0), atom_string(CAi, CAi0).
wam_parts_to_instr(["put_value", Xn, Ai], put_value(CXn, CAi)) :-
    clean_comma(Xn, CXn0), clean_comma(Ai, CAi0),
    atom_string(CXn, CXn0), atom_string(CAi, CAi0).
wam_parts_to_instr(["put_structure", F, Ai], put_structure(CF, CAi)) :-
    clean_comma(F, CF0), clean_comma(Ai, CAi0),
    atom_string(CF, CF0), atom_string(CAi, CAi0).
wam_parts_to_instr(["put_list", Ai], put_list(CAi)) :-
    clean_comma(Ai, CAi0), atom_string(CAi, CAi0).
wam_parts_to_instr(["set_variable", Xn], set_variable(CXn)) :-
    clean_comma(Xn, CXn0), atom_string(CXn, CXn0).
wam_parts_to_instr(["set_value", Xn], set_value(CXn)) :-
    clean_comma(Xn, CXn0), atom_string(CXn, CXn0).
wam_parts_to_instr(["set_constant", C], set_constant(CC)) :-
    clean_comma(C, CC).
wam_parts_to_instr(["allocate"], allocate).
wam_parts_to_instr(["deallocate"], deallocate).
wam_parts_to_instr(["call", P, N], call(CP, CN)) :-
    clean_comma(P, CP0), clean_comma(N, CN0),
    atom_string(CP, CP0),
    (number_string(CN, CN0) -> true ; CN = 0).
wam_parts_to_instr(["execute", P], execute(CP)) :-
    clean_comma(P, CP0), atom_string(CP, CP0).
wam_parts_to_instr(["proceed"], proceed).
wam_parts_to_instr(["builtin_call", Op, N], builtin_call(COp, CN)) :-
    clean_comma(Op, COp0), clean_comma(N, CN0),
    atom_string(COp, COp0),
    (number_string(CN, CN0) -> true ; CN = 0).
wam_parts_to_instr(["try_me_else", L], try_me_else(CL)) :-
    clean_comma(L, CL0), atom_string(CL, CL0).
wam_parts_to_instr(["retry_me_else", L], retry_me_else(CL)) :-
    clean_comma(L, CL0), atom_string(CL, CL0).
wam_parts_to_instr(["trust_me"], trust_me).
wam_parts_to_instr(["cut_ite"], cut_ite).
wam_parts_to_instr(["jump", L], jump(CL)) :-
    clean_comma(L, CL0), atom_string(CL, CL0).
%% First-argument indexing: switch_on_constant + switch_on_constant_a2
%% get real O(N) linear-scan dispatch that commits to the matching
%% clause, skipping the try_me_else/retry/trust chain on a hit. A miss
%% (bound non-matching, or unbound) falls through to the chain.
%% switch_on_structure/switch_on_term remain nop until structure-indexed
%% dispatch is implemented.
wam_parts_to_instr(["switch_on_constant"|Rest], Result) :- !,
    parse_switch_entries(Rest, Entries),
    build_switch_instrs(0, Entries, Result).
wam_parts_to_instr(["switch_on_constant_a2"|Rest], Result) :- !,
    parse_switch_entries(Rest, Entries),
    build_switch_instrs(1, Entries, Result).
wam_parts_to_instr(["switch_on_structure"|Rest], Result) :- !,
    parse_struct_entries(Rest, Entries),
    build_switch_struct_instrs(0, Entries, Result).
wam_parts_to_instr(["switch_on_term"|Rest], Result) :- !,
    parse_term_entries(Rest, ConstEntries, StructEntries),
    build_switch_term_instrs(0, ConstEntries, StructEntries, Result).

wam_parts_to_instr(["begin_aggregate", Type, ValReg, ResReg],
                   begin_aggregate(CType, CValReg, CResReg)) :-
    clean_comma(Type, CType0), atom_string(CType, CType0),
    clean_comma(ValReg, CValReg0), atom_string(CValReg, CValReg0),
    clean_comma(ResReg, CResReg0), atom_string(CResReg, CResReg0).
wam_parts_to_instr(["end_aggregate", ValReg], end_aggregate(CValReg)) :-
    clean_comma(ValReg, CValReg0), atom_string(CValReg, CValReg0).
% Fallback
wam_parts_to_instr(Parts, allocate) :-
    format(user_error, '  WAM-WAT: unrecognized instruction: ~w~n', [Parts]).

clean_comma(S, Clean) :-
    (   sub_string(S, Before, 1, 0, ",")
    ->  sub_string(S, 0, Before, 1, Clean)
    ;   Clean = S
    ).

% ============================================================================
% PHASE 2: compile_wam_predicate_to_wat/4
% ============================================================================

%% compile_wam_predicate_to_wat(+Pred/Arity, +WamCode, +Options, -WatResult)
%  WatResult = wat_pred(DataSeg, EntryFunc, CodeBase, NumInstrs)
compile_wam_predicate_to_wat(Pred/Arity, WamCode, Options, WatResult) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    %% First pass: collect labels and instructions
    wam_lines_to_instrs(Lines, 0, Instrs, Labels),
    length(Instrs, NumInstrs),
    %% Get code base offset from options or compute it
    option(code_base(CodeBase), Options, 131072),
    %% Encode instructions to hex bytes
    maplist(encode_instr_with_labels(Labels), Instrs, HexParts),
    atomic_list_concat(HexParts, DataBytes),
    %% Generate data segment
    format(atom(DataSeg),
        '(data (i32.const ~w) "~w")',
        [CodeBase, DataBytes]),
    %% Generate entry function
    wat_pred_name(Pred, Arity, FuncName),
    %% Heap must start AFTER the instruction data segment — otherwise
    %% heap_push_val on the first allocation overwrites instruction 0
    %% bytes, and any subsequent allocations walk forward through the
    %% code. We observed this concretely with $builtin_univ decompose
    %% of a compound: the decompose path builds (arity+1) cons cells
    %% (36 bytes each) plus any prior compound writes, and the write
    %% range crossed into proceed''s bytes, causing the next step to
    %% misinterpret proceed as garbage and fail. Atomic-only tests
    %% masked the bug because they allocate fewer cells and never
    %% reached the proceed bytes. Put the heap on page 3 (offset
    %% 196608) which is past any plausible code segment in the current
    %% 4-page module layout.
    option(wam_heap_start(HeapStart), Options, 196608),
    format(atom(EntryFunc),
'(func $~w (export "~w") (result i32)
  (call $wam_init (i32.const ~w))
  (call $run_loop (i32.const ~w) (i32.const ~w)))',
        [FuncName, FuncName, HeapStart, CodeBase, NumInstrs]),
    WatResult = wat_pred(DataSeg, EntryFunc, CodeBase, NumInstrs).

encode_instr_with_labels(Labels, Instr, Hex) :-
    wam_instruction_to_wat_bytes(Instr, Labels, Hex).

wat_pred_name(Pred, Arity, Name) :-
    format(atom(Name), '~w_~w', [Pred, Arity]).

% ============================================================================
% PHASE 2: Step dispatch (br_table)
% ============================================================================

%% compile_step_wam_to_wat(+Options, -WatCode)
compile_step_wam_to_wat(_Options, WatCode) :-
    %% Collect all instruction case bodies
    findall(Tag-Body, (wam_wat_case(InstrName, Body), instr_tag(InstrName, Tag)), Cases),
    sort(Cases, SortedCases),
    %% Generate the do_* helper functions
    maplist(gen_do_func, SortedCases, DoFuncs),
    atomic_list_concat(DoFuncs, '\n\n', DoFuncsCode),
    %% Generate the step function with if-else chain (simpler than br_table nesting)
    gen_step_function(SortedCases, StepFunc),
    format(atom(WatCode), '~w\n\n~w', [DoFuncsCode, StepFunc]).

gen_do_func(Tag-Body, Code) :-
    instr_tag(Name, Tag),
    format(atom(Code),
'(func $do_~w (param $op1 i64) (param $op2 i64) (result i32)
~w)', [Name, Body]).

gen_step_function(Cases, Code) :-
    %% Generate br_table dispatch with nested blocks.
    %%
    %% Block nesting: $default is outermost, $b0 is innermost. br_table
    %% dispatches tag N to $bN. Exiting $bN puts control just after
    %% $bN's close, where the case body for tag N runs — a return that
    %% exits the function with the $do_<instr> call result.
    %%
    %% There must be NO bare `)` between br_table and the first case
    %% body: if present, that `)` would close $b0 before its case body,
    %% making tag N run the body for tag N-1 (an off-by-one dispatch
    %% bug). The original WAM-WAT target PR #1224 had this bug; it went
    %% unnoticed because wat2wasm_validates used assertion/1 which
    %% warns rather than failing the test, so runtime behavior was
    %% never actually validated.
    length(Cases, N),
    MaxTag is N - 1,
    %% br_table labels: $b0 $b1 ... $bN $default
    numlist(0, MaxTag, TagNums),
    maplist(br_label, TagNums, BrLabels),
    atomic_list_concat(BrLabels, ' ', BrTableStr),
    %% Opening nested blocks (innermost = highest tag)
    maplist(open_block, TagNums, OpenBlocks),
    reverse(OpenBlocks, RevOpenBlocks),
    atomic_list_concat(RevOpenBlocks, '\n', OpenBlocksStr),
    %% Closing blocks + dispatch calls (innermost first = tag 0)
    maplist(close_and_call, Cases, CloseBlocks),
    atomic_list_concat(CloseBlocks, '\n', CloseBlocksStr),
    format(atom(Code),
'(func $step (param $code_base i32) (param $pc i32) (result i32)
  (local $tag i32)
  (local $op1 i64)
  (local $op2 i64)
  (local.set $tag (call $fetch_instr_tag (local.get $code_base) (local.get $pc)))
  (local.set $op1 (call $fetch_instr_op1 (local.get $code_base) (local.get $pc)))
  (local.set $op2 (call $fetch_instr_op2 (local.get $code_base) (local.get $pc)))
  (block $default
~w
    (br_table ~w $default (local.get $tag))
~w
  )
  (i32.const 0)
)', [OpenBlocksStr, BrTableStr, CloseBlocksStr]).

br_label(N, Label) :- format(atom(Label), '$b~w', [N]).
open_block(N, Block) :- format(atom(Block), '    (block $b~w', [N]).

close_and_call(Tag-_, Code) :-
    instr_tag(Name, Tag),
    format(atom(Code),
'  ) ;; $b~w (~w)
  (return (call $do_~w (local.get $op1) (local.get $op2)))',
        [Tag, Name, Name]).

% ============================================================================
% PHASE 2: Instruction case bodies (WAT S-expressions)
% ============================================================================

% --- Head unification ---

wam_wat_case(get_constant,
'  ;; op2 layout: high 32 = value-cell tag hint, low 32 = reg idx.
  ;; Deref through Ref chains so heap-aliased variables are handled.
  (local $reg_idx i32) (local $c_tag i32) (local $d_addr i32)
  (local.set $reg_idx (i32.wrap_i64 (local.get $op2)))
  (local.set $c_tag (i32.wrap_i64 (i64.shr_u (local.get $op2) (i64.const 32))))
  (local.set $d_addr (call $deref_reg_addr (local.get $reg_idx)))
  (if (result i32) (i32.eq (call $val_tag (local.get $d_addr)) (i32.const 6))
    (then
      (call $trail_binding_at (local.get $d_addr))
      (call $val_store (local.get $d_addr) (local.get $c_tag) (local.get $op1))
      (call $inc_pc)
      (i32.const 1))
    (else
      (if (result i32) (i32.and
            (i32.eq (call $val_tag (local.get $d_addr)) (local.get $c_tag))
            (i64.eq (call $val_payload (local.get $d_addr)) (local.get $op1)))
        (then (call $inc_pc) (i32.const 1))
        (else (i32.const 0)))))').

wam_wat_case(get_variable,
'  (local $xn i32) (local $ai i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (call $copy_to_reg (local.get $xn) (call $reg_offset (local.get $ai)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(get_value,
'  (local $xn i32) (local $ai i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (if (result i32) (call $unify_regs (local.get $xn) (local.get $ai))
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(get_structure,
'  ;; op1 = functor hash, op2 = reg index. Deref through Ref chains.
  (local $ai i32) (local $d_addr i32) (local $tag i32) (local $addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (local.set $d_addr (call $deref_reg_addr (local.get $ai)))
  (local.set $tag (call $val_tag (local.get $d_addr)))
  (if (result i32) (i32.eq (local.get $tag) (i32.const 6)) ;; unbound
    (then
      ;; Write mode: allocate compound on heap, bind the dereffed cell
      (local.set $addr (call $heap_push_val (i32.const 3) (local.get $op1)))
      (call $trail_binding_at (local.get $d_addr))
      (call $val_store (local.get $d_addr) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
      (call $set_mode (i32.const 1))
      (call $inc_pc)
      (i32.const 1))
    (else
      ;; Read mode: check functor match
      (if (result i32) (i32.and
            (i32.eq (local.get $tag) (i32.const 3))
            (i64.eq (call $val_payload (local.get $d_addr)) (local.get $op1)))
        (then
          (call $set_mode (i32.const 0))
          (call $inc_pc)
          (i32.const 1))
        (else (i32.const 0)))))').

wam_wat_case(get_list,
'  ;; Deref through Ref chains.
  (local $ai i32) (local $d_addr i32) (local $tag i32) (local $addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op1)))
  (local.set $d_addr (call $deref_reg_addr (local.get $ai)))
  (local.set $tag (call $val_tag (local.get $d_addr)))
  (if (result i32) (i32.eq (local.get $tag) (i32.const 6)) ;; unbound
    (then
      (local.set $addr (call $heap_push_val (i32.const 4) (i64.const 0)))
      (call $trail_binding_at (local.get $d_addr))
      (call $val_store (local.get $d_addr) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
      (call $set_mode (i32.const 1))
      (call $inc_pc)
      (i32.const 1))
    (else
      (if (result i32) (i32.eq (local.get $tag) (i32.const 4)) ;; list
        (then
          (call $set_mode (i32.const 0))
          (call $inc_pc)
          (i32.const 1))
        (else (i32.const 0)))))').

wam_wat_case(unify_variable,
'  (local $xn i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (if (call $get_mode) ;; write mode
    (then
      ;; Create new unbound on heap and bind to register
      (call $set_reg (local.get $xn) (i32.const 6) (i64.extend_i32_u (local.get $xn))))
    (else
      ;; Read mode: nothing to do for basic unify_variable
      (nop)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(unify_value,
'  (local $xn i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (if (result i32) (call $get_mode) ;; write mode
    (then
      ;; Push RAW register value to heap (no deref — same rationale
      ;; as set_value: preserve Ref indirection for nested compounds).
      (drop (call $heap_push_val
        (call $get_reg_tag (local.get $xn))
        (call $get_reg_payload (local.get $xn))))
      (call $inc_pc)
      (i32.const 1))
    (else
      ;; Read mode: unify with next structure arg
      (call $inc_pc)
      (i32.const 1)))').

wam_wat_case(unify_constant,
'  ;; op2 = value-cell tag hint (0 atom, 1 integer, 2 float).
  (local $c_tag i32)
  (local.set $c_tag (i32.wrap_i64 (local.get $op2)))
  (if (result i32) (call $get_mode) ;; write mode
    (then
      (drop (call $heap_push_val (local.get $c_tag) (local.get $op1)))
      (call $inc_pc)
      (i32.const 1))
    (else
      ;; Read mode: match constant
      (call $inc_pc)
      (i32.const 1)))').

% --- Body construction ---

wam_wat_case(put_constant,
'  ;; op2 layout: high 32 = value-cell tag hint (0 atom, 1 integer,
  ;; 2 float), low 32 = target reg index. Encoder packs this in
  ;; wam_instruction_to_wat_bytes/3 via encode_constant_with_tag/3.
  (local $ai i32) (local $c_tag i32)
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (local.set $c_tag (i32.wrap_i64 (i64.shr_u (local.get $op2) (i64.const 32))))
  (call $set_reg (local.get $ai) (local.get $c_tag) (local.get $op1))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(put_variable,
'  ;; Allocate ONE unbound cell on the heap and point both Xn and Ai
  ;; at it via Ref(H). This is the standard WAM variable aliasing
  ;; mechanism: any subsequent binding of Ai (e.g. by a callee) is
  ;; visible when Xn is later read, because both dereference to the
  ;; same heap cell. Prior to this fix, put_variable created two
  ;; independent Unbound cells with the same payload — bindings on
  ;; Ai did not propagate to Xn, silently breaking multi-goal
  ;; predicates.
  (local $xn i32) (local $ai i32) (local $addr i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (local.set $addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (local.get $xn) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
  (call $set_reg (local.get $ai) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(put_value,
'  (local $xn i32) (local $ai i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (call $copy_to_reg (local.get $ai) (call $reg_offset (local.get $xn)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(put_structure,
'  ;; Allocate compound header on heap, then BIND (not just overwrite)
  ;; the target register to Ref(compound_addr). Binding through the
  ;; deref chain is essential: if Ai holds Ref(unbound_cell) from a
  ;; prior set_variable, the unbound cell must be updated to
  ;; Ref(compound) so nested compounds are linked. Example: for
  ;; +(*(1000,3),7), set_variable X3 creates the unbound arg1 of +,
  ;; then put_structure */2, X3 must bind THAT cell to Ref(*).
  (local $ai i32) (local $addr i32) (local $d_addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (local.set $addr (call $heap_push_val (i32.const 3) (local.get $op1)))
  ;; If the target register derefs to an unbound cell (from a prior
  ;; set_variable), bind through the Ref chain so the heap cell is
  ;; updated — this links nested compounds correctly (e.g. +(*(1000,3),7)).
  ;; Otherwise, just overwrite the register directly.
  (local.set $d_addr (call $deref_reg_addr (local.get $ai)))
  (if (i32.eq (call $val_tag (local.get $d_addr)) (i32.const 6))
    (then
      (call $trail_binding_at (local.get $d_addr))
      (call $val_store (local.get $d_addr) (i32.const 5) (i64.extend_i32_u (local.get $addr))))
    (else
      (call $set_reg (local.get $ai) (i32.const 5) (i64.extend_i32_u (local.get $addr)))))
  (call $set_mode (i32.const 1))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(put_list,
'  (local $ai i32) (local $addr i32) (local $d_addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op1)))
  (local.set $addr (call $heap_push_val (i32.const 4) (i64.const 0)))
  (local.set $d_addr (call $deref_reg_addr (local.get $ai)))
  (if (i32.eq (call $val_tag (local.get $d_addr)) (i32.const 6))
    (then
      (call $trail_binding_at (local.get $d_addr))
      (call $val_store (local.get $d_addr) (i32.const 5) (i64.extend_i32_u (local.get $addr))))
    (else
      (call $set_reg (local.get $ai) (i32.const 5) (i64.extend_i32_u (local.get $addr)))))
  (call $set_mode (i32.const 1))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(set_variable,
'  ;; Allocate an unbound cell on the heap and set Xn to Ref(addr).
  ;; Used inside put_structure/put_list bodies to create a slot for
  ;; a variable argument.
  (local $xn i32) (local $addr i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (local.set $addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (local.get $xn) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(set_value,
'  ;; Push the RAW register cell onto the heap (no deref). When building
  ;; compound arguments, a register holding Ref(inner_compound) must be
  ;; pushed as a Ref — if we deref, the compound header gets flattened
  ;; into the arg cell and nested compound evaluation breaks (e.g.
  ;; is/2 on 1000*3+7 would fail because the * subexpression loses
  ;; its address indirection).
  (local $xn i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (drop (call $heap_push_val
    (call $get_reg_tag (local.get $xn))
    (call $get_reg_payload (local.get $xn))))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(set_constant,
'  ;; op2 = value-cell tag hint (0 atom, 1 integer, 2 float).
  (local $c_tag i32)
  (local.set $c_tag (i32.wrap_i64 (local.get $op2)))
  (drop (call $heap_push_val (local.get $c_tag) (local.get $op1)))
  (call $inc_pc)
  (i32.const 1)').

% --- Control flow ---

wam_wat_case(allocate,
'  ;; Push a new environment frame on the stack. Y registers live
  ;; inside this frame (accessed via $env_base in $reg_offset), so
  ;; each call level gets its own Y storage and there is no need to
  ;; save/restore Y explicitly — the frame IS the Y storage.
  ;; Frame: [prev_env_base:i32 +0][CP:i32 +4][32 Y slots: 384 bytes +8] = 392 bytes.
  (local $soff i32)
  (local.set $soff (call $get_stack_top))
  ;; Save previous env_base and CP
  (i32.store (local.get $soff) (global.get $env_base))
  (i32.store (i32.add (local.get $soff) (i32.const 4)) (call $get_cp))
  ;; This frame becomes the current environment
  (global.set $env_base (local.get $soff))
  (call $set_stack_top (i32.add (local.get $soff) (i32.const 392)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(deallocate,
'  ;; Pop the current environment frame. Restores $env_base to the
  ;; previous frame and CP from the frame header. The Y slots in the
  ;; popped frame become dead (stack_top moves back).
  (local $frame i32)
  (local.set $frame (global.get $env_base))
  ;; Restore CP from the frame
  (call $set_cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  ;; Restore previous env_base
  (global.set $env_base (i32.load (local.get $frame)))
  ;; Pop the frame
  (call $set_stack_top (local.get $frame))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(call,
'  ;; op1 = target PC, op2 = arity (unused here)
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op1)))
  (i32.const 1)').

wam_wat_case(execute,
'  ;; op1 = target PC (tail call, no CP save)
  (call $set_pc (i32.wrap_i64 (local.get $op1)))
  (i32.const 1)').

wam_wat_case(proceed,
'  ;; Return to continuation point
  (if (result i32) (i32.ge_s (call $get_cp) (i32.const 0))
    (then
      (call $set_pc (call $get_cp))
      (i32.const 1))
    (else
      ;; No CP means done
      (call $set_halted (i32.const 1))
      (i32.const 1)))').

wam_wat_case(builtin_call,
'  ;; op1 = builtin ID, op2 = arity
  (if (result i32) (call $execute_builtin
        (i32.wrap_i64 (local.get $op1))
        (i32.wrap_i64 (local.get $op2)))
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

% --- Choice points ---

wam_wat_case(try_me_else,
'  ;; op1 = alternative label PC
  ;; Save choice point: next_pc, CP, trail_mark, registers
  (call $push_choice_point (i32.wrap_i64 (local.get $op1)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(retry_me_else,
'  ;; op1 = next alternative label PC
  ;; Update choice point with new alternative. If cp_count is 0, the
  ;; control flow arrived here via first-arg indexing (switch_on_const
  ;; committed to this clause directly), bypassing try_me_else — so no
  ;; CP exists to update. Treat retry_me_else as a no-op in that case:
  ;; the caller has already committed, no alternatives need tracking.
  (if (i32.gt_s (call $get_cp_count) (i32.const 0))
    (then
      (call $update_choice_point (i32.wrap_i64 (local.get $op1)))))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(neck_cut_test,
'  ;; Combined guard test + conditional jump for cut-deterministic
  ;; 2-clause predicates. Emitted by the peephole optimizer when it
  ;; detects try_me_else + guard + cut. No choice point is created.
  ;; op1 = else clause PC, op2 = (guard_builtin_id << 32) | arity.
  ;; If guard succeeds: advance PC (continue with clause 1 body).
  ;; If guard fails: undo the current allocate frame and jump to
  ;; the else clause — no CP push/pop, no register save/restore.
  (local $else_pc i32) (local $guard_id i32) (local $guard_arity i32)
  (local.set $else_pc (i32.wrap_i64 (local.get $op1)))
  (local.set $guard_id (i32.wrap_i64 (i64.shr_u (local.get $op2) (i64.const 32))))
  (local.set $guard_arity (i32.wrap_i64 (local.get $op2)))
  (if (result i32) (call $execute_builtin (local.get $guard_id) (local.get $guard_arity))
    (then
      (call $inc_pc)
      (i32.const 1))
    (else
      ;; Guard failed: pop the env frame that allocate pushed for
      ;; clause 1 (the else clause will do its own allocate).
      (call $set_stack_top (global.get $env_base))
      (global.set $env_base (i32.load (global.get $env_base)))
      (call $set_pc (local.get $else_pc))
      (i32.const 1)))').

wam_wat_case(cut_ite,
'  ;; Soft cut for if-then-else: pop only the most recent CP (the
  ;; ITE try_me_else), preserving any outer choice points and
  ;; aggregate frames. Unlike !/0 which zeros cp_count, cut_ite
  ;; just decrements by 1.
  (if (i32.gt_s (call $get_cp_count) (i32.const 0))
    (then
      (call $set_cp_count (i32.sub (call $get_cp_count) (i32.const 1)))))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(jump,
'  ;; Unconditional jump to target PC (op1). Used after the then-
  ;; branch of if-then-else to skip over the else-branch.
  (call $set_pc (i32.wrap_i64 (local.get $op1)))
  (i32.const 1)').

wam_wat_case(begin_aggregate,
'  ;; Push an aggregate frame: a normal choice point PLUS set global
  ;; aggregate state. The inner body will run; each solution reaching
  ;; end_aggregate adds to the running accumulator and force-backtracks.
  ;; When backtrack exhausts all solutions, $backtrack detects
  ;; $agg_active and finalizes instead of failing.
  ;; op1 = (agg_type << 32) | val_reg_index, op2 = res_reg_index.
  (local $agg_type i32) (local $val_reg i32) (local $res_reg i32)
  (local.set $val_reg (i32.wrap_i64 (local.get $op1)))
  (local.set $agg_type (i32.wrap_i64 (i64.shr_u (local.get $op1) (i64.const 32))))
  (local.set $res_reg (i32.wrap_i64 (local.get $op2)))
  ;; Save aggregate metadata in globals. Do NOT push a CP — the inner
  ;; body goals will push their own try_me_else CPs for clause dispatch.
  ;; When all inner solutions are exhausted, cp_count reaches 0 and
  ;; $backtrack detects $agg_active for finalization.
  (global.set $agg_active (i32.const 1))
  (global.set $agg_type (local.get $agg_type))
  (global.set $agg_val_reg (local.get $val_reg))
  (global.set $agg_res_reg (local.get $res_reg))
  (global.set $agg_sum (i64.const 0))
  (global.set $agg_count (i32.const 0))
  (global.set $agg_return_pc (i32.const 0))
  ;; Save caller CP so finalization can restore it. The inner call
  ;; (e.g. my_fact) will overwrite CP; without saving, the proceed
  ;; at the end of the aggregate predicate would jump back into the
  ;; end_aggregate instruction rather than returning to the caller.
  (global.set $agg_saved_cp (call $get_cp))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(end_aggregate,
'  ;; Collect one solution value and force-backtrack to try the next.
  ;; op1 = val_reg_index. Read the value, accumulate, then return 0
  ;; (fail) so run_loop calls $backtrack for the next solution.
  ;; When $backtrack exhausts all inner solutions and $agg_active is 1,
  ;; it finalizes instead of returning failure.
  (local $val_reg i32) (local $val i64)
  (local.set $val_reg (i32.wrap_i64 (local.get $op1)))
  ;; Read and accumulate the value
  (local.set $val (call $deref_reg_payload (local.get $val_reg)))
  ;; Type-specific accumulation
  (if (i32.eqz (global.get $agg_type)) ;; sum
    (then (global.set $agg_sum (i64.add (global.get $agg_sum) (local.get $val)))))
  ;; Count aggregate: $agg_count is the result directly, incremented
  ;; once per solution at the bottom of this handler (it doubles as the
  ;; "seen count" for max/min, so the bump below is unconditional).
  ;; The previous explicit count-branch incremented twice per solution
  ;; and produced 2*N for aggregate_all(count, ...) queries.
  (if (i32.eq (global.get $agg_type) (i32.const 2)) ;; max
    (then (if (i32.or (i32.eqz (global.get $agg_count))
                      (i64.gt_s (local.get $val) (global.get $agg_sum)))
            (then (global.set $agg_sum (local.get $val))))))
  (if (i32.eq (global.get $agg_type) (i32.const 3)) ;; min
    (then (if (i32.or (i32.eqz (global.get $agg_count))
                      (i64.lt_s (local.get $val) (global.get $agg_sum)))
            (then (global.set $agg_sum (local.get $val))))))
  (global.set $agg_count (i32.add (global.get $agg_count) (i32.const 1)))
  ;; Store the return PC = current PC + 1 (the instruction after end_aggregate)
  (global.set $agg_return_pc (i32.add (call $get_pc) (i32.const 1)))
  ;; Force-backtrack to try next solution (return 0 = step failed)
  (i32.const 0)').

wam_wat_case(nop,
'  ;; No-op: placeholder for unimplemented indexing instructions.
  ;; Just advance PC and continue.
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(fused_is_add,
'  ;; Dest := deref(Src1) + deref(Src2), where all three are registers.
  ;; op1 layout: Dest (low 8), Src1 (bits 8-15), Src2 (bits 16-23).
  ;; Both source registers must deref to integers (tag=1); on any
  ;; other tag the instruction fails (returns 0) and the run_loop
  ;; triggers backtrack. This specialized form skips the +/2 compound
  ;; allocation + $eval_arith recursion that `is/2` would otherwise
  ;; perform — a meaningful win on the sum_ints-style recursive
  ;; accumulator pattern.
  (local $op1i i32) (local $dest i32) (local $src1 i32) (local $src2 i32)
  (local $a_addr i32) (local $b_addr i32)
  (local $a i64) (local $b i64)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $dest (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $src1
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $src2
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 16)) (i32.const 0xFF)))
  (local.set $a_addr (call $deref_reg_addr (local.get $src1)))
  (local.set $b_addr (call $deref_reg_addr (local.get $src2)))
  (if (i32.or
        (i32.ne (call $val_tag (local.get $a_addr)) (i32.const 1))
        (i32.ne (call $val_tag (local.get $b_addr)) (i32.const 1)))
    (then (return (i32.const 0))))
  (local.set $a (call $val_payload (local.get $a_addr)))
  (local.set $b (call $val_payload (local.get $b_addr)))
  (call $bind_reg_deref (local.get $dest) (i32.const 1)
    (i64.add (local.get $a) (local.get $b)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(fused_is_sub,
'  ;; Dest := deref(Src1) - deref(Src2). Same layout and semantics as
  ;; fused_is_add; see that case for details.
  (local $op1i i32) (local $dest i32) (local $src1 i32) (local $src2 i32)
  (local $a_addr i32) (local $b_addr i32)
  (local $a i64) (local $b i64)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $dest (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $src1
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $src2
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 16)) (i32.const 0xFF)))
  (local.set $a_addr (call $deref_reg_addr (local.get $src1)))
  (local.set $b_addr (call $deref_reg_addr (local.get $src2)))
  (if (i32.or
        (i32.ne (call $val_tag (local.get $a_addr)) (i32.const 1))
        (i32.ne (call $val_tag (local.get $b_addr)) (i32.const 1)))
    (then (return (i32.const 0))))
  (local.set $a (call $val_payload (local.get $a_addr)))
  (local.set $b (call $val_payload (local.get $b_addr)))
  (call $bind_reg_deref (local.get $dest) (i32.const 1)
    (i64.sub (local.get $a) (local.get $b)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(fused_is_mul,
'  ;; Dest := deref(Src1) * deref(Src2). Same layout and semantics as
  ;; fused_is_add; see that case for details.
  (local $op1i i32) (local $dest i32) (local $src1 i32) (local $src2 i32)
  (local $a_addr i32) (local $b_addr i32)
  (local $a i64) (local $b i64)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $dest (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $src1
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $src2
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 16)) (i32.const 0xFF)))
  (local.set $a_addr (call $deref_reg_addr (local.get $src1)))
  (local.set $b_addr (call $deref_reg_addr (local.get $src2)))
  (if (i32.or
        (i32.ne (call $val_tag (local.get $a_addr)) (i32.const 1))
        (i32.ne (call $val_tag (local.get $b_addr)) (i32.const 1)))
    (then (return (i32.const 0))))
  (local.set $a (call $val_payload (local.get $a_addr)))
  (local.set $b (call $val_payload (local.get $b_addr)))
  (call $bind_reg_deref (local.get $dest) (i32.const 1)
    (i64.mul (local.get $a) (local.get $b)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(fused_is_add_const,
'  ;; Dest := deref(Src) + Const. op1 layout: Dest (low 8), Src (bits
  ;; 8-15); op2 holds the signed constant as i64. Fires on the
  ;; `N1 is N - 1` pattern which the WAM layer lowers to +/2 with a
  ;; negated constant.
  (local $op1i i32) (local $dest i32) (local $src i32)
  (local $a_addr i32) (local $a i64)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $dest (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $src
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $a_addr (call $deref_reg_addr (local.get $src)))
  (if (i32.ne (call $val_tag (local.get $a_addr)) (i32.const 1))
    (then (return (i32.const 0))))
  (local.set $a (call $val_payload (local.get $a_addr)))
  (call $bind_reg_deref (local.get $dest) (i32.const 1)
    (i64.add (local.get $a) (local.get $op2)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(fused_is_mul_const,
'  ;; Dest := deref(Src) * Const. Same layout as fused_is_add_const.
  (local $op1i i32) (local $dest i32) (local $src i32)
  (local $a_addr i32) (local $a i64)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $dest (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $src
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $a_addr (call $deref_reg_addr (local.get $src)))
  (if (i32.ne (call $val_tag (local.get $a_addr)) (i32.const 1))
    (then (return (i32.const 0))))
  (local.set $a (call $val_payload (local.get $a_addr)))
  (call $bind_reg_deref (local.get $dest) (i32.const 1)
    (i64.mul (local.get $a) (local.get $op2)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(arg_direct,
'  ;; Direct arg/3 call, bypassing the $execute_builtin br_table.
  ;; Emitted by the peephole in place of `builtin_call(arg/3, 3)`.
  (if (result i32) (call $builtin_arg)
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(functor_direct,
'  ;; Direct functor/3 call, bypassing the $execute_builtin br_table.
  (if (result i32) (call $builtin_functor)
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(copy_term_direct,
'  ;; Direct copy_term/2 call, bypassing the $execute_builtin br_table.
  (if (result i32) (call $builtin_copy_term)
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(univ_direct,
'  ;; Direct =../2 call, bypassing the $execute_builtin br_table.
  (if (result i32) (call $builtin_univ)
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(is_list_direct,
'  ;; Direct is_list/1 call, bypassing the $execute_builtin br_table.
  (if (result i32) (call $builtin_is_list)
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(arg_reg_direct,
'  ;; arg/3 specialized for `put_value N, A1; put_value T, A2;
  ;; put_variable Dest, A3; builtin_call arg/3, 3`.
  ;;
  ;; Fast path (~98% of calls in term-walking code): N is a bound
  ;; integer, T is a bound compound, arg index in range. Reads the
  ;; arg cell directly from the compound and writes it into Dest
  ;; via set_reg (no deref, no trail — matches put_variable
  ;; semantics of writing a fresh env-frame slot; on backtrack the
  ;; env frame is abandoned via stack_top replay).
  ;;
  ;; Fallback: sets up A1/A2/A3 the way put_value/put_value/put_variable
  ;; would and invokes builtin_arg. Covers the nondet mode (A1
  ;; unbound, triggers arity-enumeration via the CP retry slot) and
  ;; any non-hot-path tag/bounds failure.
  ;;
  ;; op1: NIdx (low 8), TIdx (bits 8-15), DestIdx (bits 16-23).
  (local $op1i i32)
  (local $n_idx i32) (local $t_idx i32) (local $dest_idx i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $fresh_addr i32)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $n_idx (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $t_idx
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $dest_idx
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 16)) (i32.const 0xFF)))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u
          (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (call $set_reg (local.get $dest_idx)
            (call $val_tag (local.get $arg_off))
            (call $val_payload (local.get $arg_off)))
          (call $inc_pc)
          (return (i32.const 1))))))
  ;; Fallback: reconstruct A1/A2/A3 and call $builtin_arg.
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $dest_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (result i32) (call $builtin_arg)
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(arg_lit_direct,
'  ;; arg/3 specialized for `put_constant N, A1; put_value T, A2;
  ;; put_variable Dest, A3; builtin_call arg/3, 3`.
  ;; Same fast-path / fallback structure as arg_reg_direct, but
  ;; skips the N deref + integer-tag check entirely since N is an
  ;; i64 literal carried in op2.
  ;; op1: TIdx (low 8), DestIdx (bits 8-15).  op2: N (i64 signed).
  (local $op1i i32)
  (local $t_idx i32) (local $dest_idx i32)
  (local $t_addr i32) (local $n i32) (local $arity i32)
  (local $arg_off i32) (local $fresh_addr i32)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $t_idx (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $dest_idx
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $n (i32.wrap_i64 (local.get $op2)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u
          (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (call $set_reg (local.get $dest_idx)
            (call $val_tag (local.get $arg_off))
            (call $val_payload (local.get $arg_off)))
          (call $inc_pc)
          (return (i32.const 1))))))
  ;; Fallback: same reconstruction as arg_reg_direct.
  (call $set_reg (i32.const 0) (i32.const 1) (local.get $op2))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $dest_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (result i32) (call $builtin_arg)
    (then (call $inc_pc) (i32.const 1))
    (else (i32.const 0)))').

wam_wat_case(arg_to_a1_reg,
'  ;; arg_reg_direct fused with put_value(Dest, A1). Fast path reads
  ;; the arg cell once and writes it to BOTH Dest (env-frame slot,
  ;; for any subsequent reads) AND A1 (argument register, for the
  ;; next call). Saves one instruction dispatch + one 12-byte
  ;; reg-to-reg copy per arg/3-call-to-Prolog transition.
  ;; op1: NIdx (low 8), TIdx (bits 8-15), DestIdx (bits 16-23).
  (local $op1i i32)
  (local $n_idx i32) (local $t_idx i32) (local $dest_idx i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $n_idx (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $t_idx
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $dest_idx
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 16)) (i32.const 0xFF)))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u
          (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (local.get $dest_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $inc_pc)
          (return (i32.const 1))))))
  ;; Fallback: emulate the original 5-instr sequence (put_value N A1;
  ;; put_value T A2; put_variable Dest A3; arg_direct; put_value Dest A1).
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $dest_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  ;; After $builtin_arg binds A3 (= the fresh cell), also move the
  ;; bound value into A1 (the put_value A1 that would have followed).
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $dest_idx)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(arg_to_a1_lit,
'  ;; arg_lit_direct fused with put_value(Dest, A1). Same structure
  ;; as arg_to_a1_reg but N is an i64 literal in op2.
  ;; op1: TIdx (low 8), DestIdx (bits 8-15).  op2: N (i64 signed).
  (local $op1i i32)
  (local $t_idx i32) (local $dest_idx i32)
  (local $t_addr i32) (local $n i32) (local $arity i32)
  (local $arg_off i32) (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (i32.wrap_i64 (local.get $op1)))
  (local.set $t_idx (i32.and (local.get $op1i) (i32.const 0xFF)))
  (local.set $dest_idx
    (i32.and (i32.shr_u (local.get $op1i) (i32.const 8)) (i32.const 0xFF)))
  (local.set $n (i32.wrap_i64 (local.get $op2)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u
          (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (local.get $dest_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $inc_pc)
          (return (i32.const 1))))))
  ;; Fallback: same reconstruction as arg_lit_direct + post-call A1 copy.
  (call $set_reg (i32.const 0) (i32.const 1) (local.get $op2))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $dest_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $dest_idx)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(arg_call_reg_3,
'  ;; Fuses arg_to_a1_reg + put_value(A2_src, A2) + put_variable(RetDest, A3) +
  ;; call(Pred, 3) into one instruction. On the fast path, reads the
  ;; arg cell once, writes it to ArgDest and A1, copies A2_src to A2,
  ;; allocates a fresh unbound cell for A3/RetDest, saves CP (= next
  ;; PC after this instruction), and jumps to Pred.
  ;; op1 packing: N (0-7), T (8-15), ArgDest (16-23), A2Src (24-31),
  ;; RetDest (32-39).  op2 = target PC.
  (local $op1i i64)
  (local $n_idx i32) (local $t_idx i32)
  (local $ad_idx i32) (local $a2_idx i32) (local $rd_idx i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $n_idx (i32.wrap_i64 (i64.and (local.get $op1i) (i64.const 0xFF))))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $rd_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0xFF))))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          ;; ArgDest := arg cell; A1 := arg cell.
          (call $set_reg (local.get $ad_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          ;; A2 := A2Src.
          (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
          ;; A3 / RetDest := Ref(fresh unbound heap cell).
          (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
          (call $set_reg (i32.const 2) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          (call $set_reg (local.get $rd_idx) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          ;; Save CP, jump to target PC.
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback: full emulation via $builtin_arg, then set up A2/A3 and call.
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  ;; Post-arg: set up A2/A3 for the pending call.
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $rd_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_lit_3,
'  ;; Same fusion as arg_call_reg_3 but with literal N in bits 40-55 of op1.
  (local $op1i i64)
  (local $t_idx i32)
  (local $ad_idx i32) (local $a2_idx i32) (local $rd_idx i32)
  (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $rd_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0xFF))))
  (local.set $n (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 40)) (i64.const 0xFFFF))))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (local.get $ad_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
          (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
          (call $set_reg (i32.const 2) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          (call $set_reg (local.get $rd_idx) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback.
  (call $set_reg (i32.const 0) (i32.const 1) (i64.extend_i32_s (local.get $n)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $rd_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_reg_3_dead,
'  ;; Same as arg_call_reg_3 but the ArgDest write is elided —
  ;; liveness analysis in peephole_arg_call_3 proved ArgDest is
  ;; never read between here and the clause end. Saves one 12-byte
  ;; reg write per recursive call (sum_ints_args / term_depth_args
  ;; are the common hit sites since the call result comes back via
  ;; the A3/RetDest slot, not via ArgDest).
  (local $op1i i64)
  (local $n_idx i32) (local $t_idx i32)
  (local $a2_idx i32) (local $rd_idx i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $ad_idx i32)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $n_idx (i32.wrap_i64 (i64.and (local.get $op1i) (i64.const 0xFF))))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $rd_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0xFF))))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          ;; ElideD: only write A1 (ArgDest is dead).
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
          (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
          (call $set_reg (i32.const 2) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          (call $set_reg (local.get $rd_idx) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback: same as arg_call_reg_3 fallback (writes to ad_idx;
  ;; correctness-preserving since ArgDest just has a dead write).
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $rd_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_lit_3_dead,
'  ;; Dead-ArgDest variant of arg_call_lit_3. See arg_call_reg_3_dead.
  (local $op1i i64)
  (local $t_idx i32)
  (local $ad_idx i32) (local $a2_idx i32) (local $rd_idx i32)
  (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $rd_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0xFF))))
  (local.set $n (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 40)) (i64.const 0xFFFF))))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
          (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
          (call $set_reg (i32.const 2) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          (call $set_reg (local.get $rd_idx) (i32.const 5)
            (i64.extend_i32_u (local.get $fresh_addr)))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback.
  (call $set_reg (i32.const 0) (i32.const 1) (i64.extend_i32_s (local.get $n)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $a2_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $rd_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_reg_1,
'  ;; Fuses arg_to_a1_reg + call(Pred, 1) into one instruction. The
  ;; arg_to_a1 already loads A1 with the arg cell; a 1-arg call has
  ;; nothing else to set up, so we just save CP and jump.
  ;; op1 packing: N (0-7), T (8-15), ArgDest (16-23). op2 = target PC.
  (local $op1i i64)
  (local $n_idx i32) (local $t_idx i32) (local $ad_idx i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $n_idx (i32.wrap_i64 (i64.and (local.get $op1i) (i64.const 0xFF))))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (local.get $ad_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback: full emulation via $builtin_arg, then jump.
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_lit_1,
'  ;; Same fusion as arg_call_reg_1 but with literal N in bits 40-55 of op1.
  (local $op1i i64)
  (local $t_idx i32) (local $ad_idx i32)
  (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $n (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 40)) (i64.const 0xFFFF))))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (local.get $ad_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback.
  (call $set_reg (i32.const 0) (i32.const 1) (i64.extend_i32_s (local.get $n)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_reg_1_dead,
'  ;; Same as arg_call_reg_1 but ArgDest write is elided — liveness
  ;; analysis in peephole_arg_call_k proved ArgDest is never read
  ;; between here and the clause end.
  (local $op1i i64)
  (local $n_idx i32) (local $t_idx i32) (local $ad_idx i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $n_idx (i32.wrap_i64 (i64.and (local.get $op1i) (i64.const 0xFF))))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          ;; Elided: only write A1 (ArgDest is dead).
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback: same as live variant (writes ad_idx for correctness;
  ;; the dead write is harmless).
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_lit_1_dead,
'  ;; Dead-ArgDest variant of arg_call_lit_1. See arg_call_reg_1_dead.
  (local $op1i i64)
  (local $t_idx i32) (local $ad_idx i32)
  (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $n (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 40)) (i64.const 0xFFFF))))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback.
  (call $set_reg (i32.const 0) (i32.const 1) (i64.extend_i32_s (local.get $n)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_reg_2,
'  ;; Fuses arg_to_a1_reg + (put_value|put_variable)(A2Reg, A2) +
  ;; call(Pred, 2). IsVar bit in op1:32 selects between the two A2
  ;; setup modes: 0 = copy A2Reg to A2 (put_value); 1 = allocate a
  ;; fresh ref cell and bind both A2 and A2Reg to it (put_variable).
  ;; op1 packing: N (0-7), T (8-15), ArgDest (16-23), A2Reg (24-31),
  ;; IsVar (bit 32). op2 = target PC.
  (local $op1i i64)
  (local $n_idx i32) (local $t_idx i32)
  (local $ad_idx i32) (local $a2_idx i32)
  (local $is_var i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $n_idx (i32.wrap_i64 (i64.and (local.get $op1i) (i64.const 0xFF))))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $is_var (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0x1))))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (local.get $ad_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          ;; A2 setup: copy (IsVar=0) or fresh-bind (IsVar=1).
          (if (i32.eqz (local.get $is_var))
            (then
              (call $copy_to_reg (i32.const 1)
                (call $reg_offset (local.get $a2_idx))))
            (else
              (local.set $fresh_addr
                (call $heap_push_val (i32.const 6) (i64.const 0)))
              (call $set_reg (i32.const 1) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))
              (call $set_reg (local.get $a2_idx) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback: $builtin_arg then A2 setup.
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (if (i32.eqz (local.get $is_var))
    (then
      (call $copy_to_reg (i32.const 1)
        (call $reg_offset (local.get $a2_idx))))
    (else
      (local.set $fresh_addr
        (call $heap_push_val (i32.const 6) (i64.const 0)))
      (call $set_reg (i32.const 1) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))
      (call $set_reg (local.get $a2_idx) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_lit_2,
'  ;; Same as arg_call_reg_2 but with literal N in bits 40-55 of op1.
  (local $op1i i64)
  (local $t_idx i32)
  (local $ad_idx i32) (local $a2_idx i32)
  (local $is_var i32)
  (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $is_var (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0x1))))
  (local.set $n (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 40)) (i64.const 0xFFFF))))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (local.get $ad_idx)
            (local.get $arg_tag) (local.get $arg_payload))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (if (i32.eqz (local.get $is_var))
            (then
              (call $copy_to_reg (i32.const 1)
                (call $reg_offset (local.get $a2_idx))))
            (else
              (local.set $fresh_addr
                (call $heap_push_val (i32.const 6) (i64.const 0)))
              (call $set_reg (i32.const 1) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))
              (call $set_reg (local.get $a2_idx) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback.
  (call $set_reg (i32.const 0) (i32.const 1) (i64.extend_i32_s (local.get $n)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (if (i32.eqz (local.get $is_var))
    (then
      (call $copy_to_reg (i32.const 1)
        (call $reg_offset (local.get $a2_idx))))
    (else
      (local.set $fresh_addr
        (call $heap_push_val (i32.const 6) (i64.const 0)))
      (call $set_reg (i32.const 1) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))
      (call $set_reg (local.get $a2_idx) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_reg_2_dead,
'  ;; Dead-ArgDest variant of arg_call_reg_2 — only A1 is written in
  ;; the fast path.
  (local $op1i i64)
  (local $n_idx i32) (local $t_idx i32)
  (local $ad_idx i32) (local $a2_idx i32)
  (local $is_var i32)
  (local $n_addr i32) (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $n_idx (i32.wrap_i64 (i64.and (local.get $op1i) (i64.const 0xFF))))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $is_var (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0x1))))
  (local.set $n_addr (call $deref_reg_addr (local.get $n_idx)))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $n_addr)) (i32.const 1))
        (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3)))
    (then
      (local.set $n (i32.wrap_i64 (call $val_payload (local.get $n_addr))))
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (if (i32.eqz (local.get $is_var))
            (then
              (call $copy_to_reg (i32.const 1)
                (call $reg_offset (local.get $a2_idx))))
            (else
              (local.set $fresh_addr
                (call $heap_push_val (i32.const 6) (i64.const 0)))
              (call $set_reg (i32.const 1) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))
              (call $set_reg (local.get $a2_idx) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback.
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $n_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (if (i32.eqz (local.get $is_var))
    (then
      (call $copy_to_reg (i32.const 1)
        (call $reg_offset (local.get $a2_idx))))
    (else
      (local.set $fresh_addr
        (call $heap_push_val (i32.const 6) (i64.const 0)))
      (call $set_reg (i32.const 1) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))
      (call $set_reg (local.get $a2_idx) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(arg_call_lit_2_dead,
'  ;; Dead-ArgDest variant of arg_call_lit_2.
  (local $op1i i64)
  (local $t_idx i32)
  (local $ad_idx i32) (local $a2_idx i32)
  (local $is_var i32)
  (local $t_addr i32)
  (local $n i32) (local $arity i32) (local $arg_off i32)
  (local $arg_tag i32) (local $arg_payload i64)
  (local $fresh_addr i32)
  (local.set $op1i (local.get $op1))
  (local.set $t_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $ad_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $a2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $is_var (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0x1))))
  (local.set $n (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 40)) (i64.const 0xFFFF))))
  (local.set $t_addr (call $deref_reg_addr (local.get $t_idx)))
  (if (i32.eq (call $val_tag (local.get $t_addr)) (i32.const 3))
    (then
      (local.set $arity (i32.wrap_i64 (i64.shr_u
        (call $val_payload (local.get $t_addr)) (i64.const 32))))
      (if (i32.and
            (i32.ge_s (local.get $n) (i32.const 1))
            (i32.le_s (local.get $n) (local.get $arity)))
        (then
          (local.set $arg_off
            (i32.add (local.get $t_addr)
                     (i32.mul (local.get $n) (i32.const 12))))
          (local.set $arg_tag (call $val_tag (local.get $arg_off)))
          (local.set $arg_payload (call $val_payload (local.get $arg_off)))
          (call $set_reg (i32.const 0)
            (local.get $arg_tag) (local.get $arg_payload))
          (if (i32.eqz (local.get $is_var))
            (then
              (call $copy_to_reg (i32.const 1)
                (call $reg_offset (local.get $a2_idx))))
            (else
              (local.set $fresh_addr
                (call $heap_push_val (i32.const 6) (i64.const 0)))
              (call $set_reg (i32.const 1) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))
              (call $set_reg (local.get $a2_idx) (i32.const 5)
                (i64.extend_i32_u (local.get $fresh_addr)))))
          (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
          (call $set_pc (i32.wrap_i64 (local.get $op2)))
          (return (i32.const 1))))))
  ;; Fallback.
  (call $set_reg (i32.const 0) (i32.const 1) (i64.extend_i32_s (local.get $n)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $t_idx)))
  (local.set $fresh_addr (call $heap_push_val (i32.const 6) (i64.const 0)))
  (call $set_reg (i32.const 2) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (call $set_reg (local.get $ad_idx) (i32.const 5)
    (i64.extend_i32_u (local.get $fresh_addr)))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $ad_idx)))
  (if (i32.eqz (local.get $is_var))
    (then
      (call $copy_to_reg (i32.const 1)
        (call $reg_offset (local.get $a2_idx))))
    (else
      (local.set $fresh_addr
        (call $heap_push_val (i32.const 6) (i64.const 0)))
      (call $set_reg (i32.const 1) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))
      (call $set_reg (local.get $a2_idx) (i32.const 5)
        (i64.extend_i32_u (local.get $fresh_addr)))))
  (call $set_cp (i32.add (call $get_pc) (i32.const 1)))
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(tail_call_5,
'  ;; Fuses the 7-instruction 5-arg tail-call-setup window into one
  ;; dispatch: copy 5 source registers (typically Y slots from the
  ;; current env frame) into A1-A5, deallocate the env frame, jump
  ;; to target (tail call, no CP save).
  ;; op1 packing: R1Idx (0-7), R2Idx (8-15), R3Idx (16-23),
  ;;              R4Idx (24-31), R5Idx (32-39). op2 = target PC.
  ;; Order matters: copy source values to A-regs BEFORE deallocate,
  ;; since deallocate invalidates the Y-slot addresses.
  (local $op1i i64)
  (local $r1_idx i32) (local $r2_idx i32) (local $r3_idx i32)
  (local $r4_idx i32) (local $r5_idx i32)
  (local $frame i32)
  (local.set $op1i (local.get $op1))
  (local.set $r1_idx (i32.wrap_i64 (i64.and (local.get $op1i) (i64.const 0xFF))))
  (local.set $r2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $r3_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $r4_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $r5_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0xFF))))
  ;; Copy Y/X values into A1-A5.
  (call $copy_to_reg (i32.const 0) (call $reg_offset (local.get $r1_idx)))
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $r2_idx)))
  (call $copy_to_reg (i32.const 2) (call $reg_offset (local.get $r3_idx)))
  (call $copy_to_reg (i32.const 3) (call $reg_offset (local.get $r4_idx)))
  (call $copy_to_reg (i32.const 4) (call $reg_offset (local.get $r5_idx)))
  ;; Inline deallocate: restore CP + env_base from frame header,
  ;; pop the frame by resetting stack_top.
  (local.set $frame (global.get $env_base))
  (call $set_cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  ;; Tail-call jump (no CP save — deallocate already restored CP).
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(deallocate_proceed,
'  ;; Fuses `deallocate + proceed` — restore CP+env_base from the
  ;; current env frame header, pop the frame, then jump to CP.
  ;; Matches the base-case ending shape of clauses like
  ;; sum_ints/3''s integer-leaf clause and fib/3''s recursive case.
  ;; No operands.
  (local $frame i32)
  (local $cp i32)
  (local.set $frame (global.get $env_base))
  (local.set $cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  (call $set_cp (local.get $cp))
  ;; Proceed: jump to CP if valid, otherwise halt.
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then
      (call $set_pc (local.get $cp))
      (i32.const 1))
    (else
      (call $set_halted (i32.const 1))
      (i32.const 1)))').

wam_wat_case(tail_call_5_c1_lit,
'  ;; K=5 tail-call fusion with literal first argument. Writes the
  ;; literal (sign-extended from 16 bits) to A1, copies R2-R5 to
  ;; A2-A5, inlines deallocate, jumps to target.
  ;; op1 packing: R2 (8-15), R3 (16-23), R4 (24-31), R5 (32-39),
  ;;              C1 literal (40-55, 16-bit signed). op2 = target PC.
  (local $op1i i64)
  (local $r2_idx i32) (local $r3_idx i32)
  (local $r4_idx i32) (local $r5_idx i32)
  (local $c1 i64)
  (local $frame i32)
  (local.set $op1i (local.get $op1))
  (local.set $r2_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 8)) (i64.const 0xFF))))
  (local.set $r3_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 16)) (i64.const 0xFF))))
  (local.set $r4_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 24)) (i64.const 0xFF))))
  (local.set $r5_idx (i32.wrap_i64
    (i64.and (i64.shr_u (local.get $op1i) (i64.const 32)) (i64.const 0xFF))))
  ;; Sign-extend 16-bit literal: shift left then arithmetic right.
  (local.set $c1 (i64.shr_s
    (i64.shl (i64.shr_u (local.get $op1i) (i64.const 40)) (i64.const 48))
    (i64.const 48)))
  ;; A1 := integer literal.
  (call $set_reg (i32.const 0) (i32.const 1) (local.get $c1))
  ;; A2-A5 := R2-R5 values (read before deallocate).
  (call $copy_to_reg (i32.const 1) (call $reg_offset (local.get $r2_idx)))
  (call $copy_to_reg (i32.const 2) (call $reg_offset (local.get $r3_idx)))
  (call $copy_to_reg (i32.const 3) (call $reg_offset (local.get $r4_idx)))
  (call $copy_to_reg (i32.const 4) (call $reg_offset (local.get $r5_idx)))
  ;; Inline deallocate.
  (local.set $frame (global.get $env_base))
  (call $set_cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  ;; Tail-call jump.
  (call $set_pc (i32.wrap_i64 (local.get $op2)))
  (i32.const 1)').

wam_wat_case(deallocate_builtin_proceed,
'  ;; Fuses `deallocate + builtin_call(Op, Arity) + proceed`.
  ;; Order matches the original 3-instruction sequence:
  ;;   1. Pop the env frame (CP restored from frame header).
  ;;   2. Invoke the builtin via $execute_builtin.
  ;;   3. On success, jump to CP; on failure, return 0 to trigger
  ;;      backtrack (same semantics as builtin_call returning 0).
  ;; op1 = builtin id, op2 = arity.
  (local $frame i32)
  (local $cp i32)
  (local.set $frame (global.get $env_base))
  (local.set $cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  (call $set_cp (local.get $cp))
  ;; Execute builtin; on failure, bail with 0.
  (if (i32.eqz (call $execute_builtin
                  (i32.wrap_i64 (local.get $op1))
                  (i32.wrap_i64 (local.get $op2))))
    (then (return (i32.const 0))))
  ;; Proceed: jump to CP if valid, otherwise halt.
  ;; Note: $execute_builtin may have updated CP (e.g., `!/0` cuts
  ;; to a higher choice point but leaves CP untouched). Reread CP
  ;; from the register to be safe.
  (local.set $cp (call $get_cp))
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then
      (call $set_pc (local.get $cp))
      (i32.const 1))
    (else
      (call $set_halted (i32.const 1))
      (i32.const 1)))').

%% Helper macro for the direct-dispatch clause-end fusions below:
%% deallocate + $builtin_<X> + proceed, invoking the specific
%% $builtin_* directly (no dispatch overhead). Matches the
%% semantics of the 3-instruction sequence exactly.
wam_wat_case(deallocate_arg_direct_proceed,
'  ;; Fuses deallocate + arg_direct + proceed.
  (local $frame i32)
  (local $cp i32)
  (local.set $frame (global.get $env_base))
  (local.set $cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  (call $set_cp (local.get $cp))
  (if (i32.eqz (call $builtin_arg)) (then (return (i32.const 0))))
  (local.set $cp (call $get_cp))
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then (call $set_pc (local.get $cp)) (i32.const 1))
    (else (call $set_halted (i32.const 1)) (i32.const 1)))').

wam_wat_case(deallocate_functor_direct_proceed,
'  ;; Fuses deallocate + functor_direct + proceed.
  (local $frame i32)
  (local $cp i32)
  (local.set $frame (global.get $env_base))
  (local.set $cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  (call $set_cp (local.get $cp))
  (if (i32.eqz (call $builtin_functor)) (then (return (i32.const 0))))
  (local.set $cp (call $get_cp))
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then (call $set_pc (local.get $cp)) (i32.const 1))
    (else (call $set_halted (i32.const 1)) (i32.const 1)))').

wam_wat_case(deallocate_copy_term_direct_proceed,
'  ;; Fuses deallocate + copy_term_direct + proceed.
  (local $frame i32)
  (local $cp i32)
  (local.set $frame (global.get $env_base))
  (local.set $cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  (call $set_cp (local.get $cp))
  (if (i32.eqz (call $builtin_copy_term)) (then (return (i32.const 0))))
  (local.set $cp (call $get_cp))
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then (call $set_pc (local.get $cp)) (i32.const 1))
    (else (call $set_halted (i32.const 1)) (i32.const 1)))').

wam_wat_case(deallocate_univ_direct_proceed,
'  ;; Fuses deallocate + univ_direct + proceed.
  (local $frame i32)
  (local $cp i32)
  (local.set $frame (global.get $env_base))
  (local.set $cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  (call $set_cp (local.get $cp))
  (if (i32.eqz (call $builtin_univ)) (then (return (i32.const 0))))
  (local.set $cp (call $get_cp))
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then (call $set_pc (local.get $cp)) (i32.const 1))
    (else (call $set_halted (i32.const 1)) (i32.const 1)))').

wam_wat_case(deallocate_is_list_direct_proceed,
'  ;; Fuses deallocate + is_list_direct + proceed.
  (local $frame i32)
  (local $cp i32)
  (local.set $frame (global.get $env_base))
  (local.set $cp (i32.load (i32.add (local.get $frame) (i32.const 4))))
  (global.set $env_base (i32.load (local.get $frame)))
  (call $set_stack_top (local.get $frame))
  (call $set_cp (local.get $cp))
  (if (i32.eqz (call $builtin_is_list)) (then (return (i32.const 0))))
  (local.set $cp (call $get_cp))
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then (call $set_pc (local.get $cp)) (i32.const 1))
    (else (call $set_halted (i32.const 1)) (i32.const 1)))').

wam_wat_case(builtin_proceed,
'  ;; Fuses `builtin_call(Op, Arity) + proceed` without a preceding
  ;; deallocate. Fires in clauses that don''t use Y-registers. After
  ;; $execute_builtin succeeds, read CP and dispatch as proceed
  ;; would (some builtins — e.g. `!/0` — may update CP).
  ;; op1 = builtin id, op2 = arity.
  (local $cp i32)
  (if (i32.eqz (call $execute_builtin
                  (i32.wrap_i64 (local.get $op1))
                  (i32.wrap_i64 (local.get $op2))))
    (then (return (i32.const 0))))
  (local.set $cp (call $get_cp))
  (if (result i32) (i32.ge_s (local.get $cp) (i32.const 0))
    (then
      (call $set_pc (local.get $cp))
      (i32.const 1))
    (else
      (call $set_halted (i32.const 1))
      (i32.const 1)))').

wam_wat_case(type_dispatch_a1,
'  ;; First-argument indexing via A1 tag. Routes directly to one of
  ;; four clause body labels based on tag:
  ;;   tag 0 (atom)     -> atom_target     (op1 low 32)
  ;;   tag 1 (integer)  -> int_target      (op1 high 32)
  ;;   tag 3 (compound) -> cmpd_target     (op2 low 32)
  ;;   other tags       -> default_target  (op2 high 32)
  ;; A 0 target means "no dispatch for this tag" — fall through to
  ;; next instruction (typically try_me_else, which handles the case
  ;; via the original clause-selection chain).
  (local $d_addr i32) (local $d_tag i32)
  (local $atom_tgt i32) (local $int_tgt i32)
  (local $cmpd_tgt i32) (local $default_tgt i32)
  (local $matched_tgt i32)
  (local.set $atom_tgt
    (i32.wrap_i64 (i64.and (local.get $op1) (i64.const 0xFFFFFFFF))))
  (local.set $int_tgt
    (i32.wrap_i64 (i64.shr_u (local.get $op1) (i64.const 32))))
  (local.set $cmpd_tgt
    (i32.wrap_i64 (i64.and (local.get $op2) (i64.const 0xFFFFFFFF))))
  (local.set $default_tgt
    (i32.wrap_i64 (i64.shr_u (local.get $op2) (i64.const 32))))
  (local.set $d_addr (call $deref_reg_addr (i32.const 0)))
  (local.set $d_tag (call $val_tag (local.get $d_addr)))
  ;; Pick the tag-specific target, falling back to default_tgt.
  (if (i32.eq (local.get $d_tag) (i32.const 0))
    (then (local.set $matched_tgt (local.get $atom_tgt)))
    (else
      (if (i32.eq (local.get $d_tag) (i32.const 1))
        (then (local.set $matched_tgt (local.get $int_tgt)))
        (else
          (if (i32.eq (local.get $d_tag) (i32.const 3))
            (then (local.set $matched_tgt (local.get $cmpd_tgt)))
            (else (local.set $matched_tgt (local.get $default_tgt))))))))
  ;; If the chosen target is 0 and we had a tag-specific miss (atom/
  ;; int/cmpd slot was 0 but the default is available), use default.
  (if (i32.eqz (local.get $matched_tgt))
    (then (local.set $matched_tgt (local.get $default_tgt))))
  ;; Jump if we have a target, else fall through.
  (if (i32.ne (local.get $matched_tgt) (i32.const 0))
    (then
      (call $set_pc (local.get $matched_tgt))
      (return (i32.const 1))))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(switch_on_const,
'  ;; First-argument constant indexing header.
  ;; op1 layout: low 32 bits = reg_idx (0 for A1, 1 for A2);
  ;;             high 32 bits = entry_count.
  ;; The following `entry_count` instructions are switch_entry records:
  ;;   op1 = constant payload (i64),
  ;;   op2 = (const_tag << 32) | target_pc.
  ;; Semantics: deref target register. If bound to a constant whose
  ;; tag+payload matches an entry, jump directly to that entry target PC
  ;; (committing to the clause, skipping the try_me_else/retry/trust chain).
  ;; If no entry matches, or the register is unbound/ref/compound/list,
  ;; fall through past all entries into the linear dispatch chain.
  (local $reg_idx i32) (local $count i32) (local $cur_pc i32)
  (local $code_base i32) (local $d_addr i32) (local $d_tag i32)
  (local $d_payload i64)
  (local $i i32) (local $entry_pc i32)
  (local $entry_op1 i64) (local $entry_op2 i64)
  (local $entry_tag i32) (local $entry_target i32)
  (local.set $reg_idx (i32.wrap_i64 (local.get $op1)))
  (local.set $count (i32.wrap_i64 (i64.shr_u (local.get $op1) (i64.const 32))))
  (local.set $cur_pc (call $get_pc))
  (local.set $code_base (global.get $wam_code_base))
  (local.set $d_addr (call $deref_reg_addr (local.get $reg_idx)))
  (local.set $d_tag (call $val_tag (local.get $d_addr)))
  (local.set $d_payload (call $val_payload (local.get $d_addr)))
  ;; Only atom/integer/float are indexable. For any other tag, fall
  ;; through past entries.
  (if (i32.le_u (local.get $d_tag) (i32.const 2))
    (then
      (local.set $i (i32.const 0))
      (block $scan_done
        (loop $scan
          (br_if $scan_done (i32.ge_u (local.get $i) (local.get $count)))
          (local.set $entry_pc
            (i32.add (i32.add (local.get $cur_pc) (i32.const 1))
                     (local.get $i)))
          (local.set $entry_op1
            (call $fetch_instr_op1 (local.get $code_base) (local.get $entry_pc)))
          (local.set $entry_op2
            (call $fetch_instr_op2 (local.get $code_base) (local.get $entry_pc)))
          (local.set $entry_tag
            (i32.wrap_i64 (i64.shr_u (local.get $entry_op2) (i64.const 32))))
          (local.set $entry_target
            (i32.wrap_i64 (i64.and (local.get $entry_op2)
                                   (i64.const 0xFFFFFFFF))))
          (if (i32.and
                (i32.eq (local.get $entry_tag) (local.get $d_tag))
                (i64.eq (local.get $entry_op1) (local.get $d_payload)))
            (then
              (call $set_pc (local.get $entry_target))
              (return (i32.const 1))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $scan)))))
  ;; No match (or unindexable tag): skip past entries to the first real
  ;; instruction, which is the try_me_else of the fall-through chain.
  (call $set_pc
    (i32.add (i32.add (local.get $cur_pc) (i32.const 1))
             (local.get $count)))
  (i32.const 1)').

wam_wat_case(switch_entry,
'  ;; Record scanned by switch_on_const. Should never execute in normal
  ;; flow — switch_on_const always jumps over this slot via $set_pc.
  ;; If reached (e.g. via an unusual backtrack target), behave as nop.
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(switch_on_struct,
'  ;; First-argument structure indexing header.
  ;; op1 layout: low 32 bits = reg_idx, high 32 bits = entry_count.
  ;; The following `entry_count` instructions are switch_struct_entry
  ;; records:
  ;;   op1 = functor_hash (DJB2 of "Functor/Arity" atom),
  ;;   op2 = (arity << 32) | target_pc.
  ;; Semantics: deref target reg. If tag=3 (compound), read the header
  ;; payload for arity + functor_hash and scan entries for a match.
  ;; If tag=4 (cons list), synthesize (arity=2, functor_hash=hash("[|]/2"))
  ;; and scan. On match, jump to that clause. Else fall through past
  ;; entries into the linear dispatch chain.
  (local $reg_idx i32) (local $count i32) (local $cur_pc i32)
  (local $code_base i32) (local $d_addr i32) (local $d_tag i32)
  (local $d_payload i64)
  (local $d_arity i32) (local $d_fhash i32)
  (local $i i32) (local $entry_pc i32)
  (local $entry_op1 i64) (local $entry_op2 i64)
  (local $entry_fhash i32) (local $entry_arity i32) (local $entry_target i32)
  (local.set $reg_idx (i32.wrap_i64 (local.get $op1)))
  (local.set $count (i32.wrap_i64 (i64.shr_u (local.get $op1) (i64.const 32))))
  (local.set $cur_pc (call $get_pc))
  (local.set $code_base (global.get $wam_code_base))
  (local.set $d_addr (call $deref_reg_addr (local.get $reg_idx)))
  (local.set $d_tag (call $val_tag (local.get $d_addr)))
  (local.set $d_payload (call $val_payload (local.get $d_addr)))
  ;; Extract arity + functor_hash from the dereffed cell. Only tag=3
  ;; (compound) and tag=4 (cons) are indexable; other tags fall through.
  (if (i32.eq (local.get $d_tag) (i32.const 3))
    (then
      (local.set $d_arity
        (i32.wrap_i64 (i64.shr_u (local.get $d_payload) (i64.const 32))))
      (local.set $d_fhash
        (i32.wrap_i64 (i64.and (local.get $d_payload) (i64.const 0xFFFFFFFF))))
      (local.set $i (i32.const 0))
      (block $scan_done
        (loop $scan
          (br_if $scan_done (i32.ge_u (local.get $i) (local.get $count)))
          (local.set $entry_pc
            (i32.add (i32.add (local.get $cur_pc) (i32.const 1))
                     (local.get $i)))
          (local.set $entry_op1
            (call $fetch_instr_op1 (local.get $code_base) (local.get $entry_pc)))
          (local.set $entry_op2
            (call $fetch_instr_op2 (local.get $code_base) (local.get $entry_pc)))
          (local.set $entry_fhash (i32.wrap_i64 (local.get $entry_op1)))
          (local.set $entry_arity
            (i32.wrap_i64 (i64.shr_u (local.get $entry_op2) (i64.const 32))))
          (local.set $entry_target
            (i32.wrap_i64 (i64.and (local.get $entry_op2)
                                   (i64.const 0xFFFFFFFF))))
          (if (i32.and
                (i32.eq (local.get $entry_fhash) (local.get $d_fhash))
                (i32.eq (local.get $entry_arity) (local.get $d_arity)))
            (then
              (call $set_pc (local.get $entry_target))
              (return (i32.const 1))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $scan)))))
  ;; Cons list: synthesize (arity=2, fhash=hash("[|]/2")=87825375).
  (if (i32.eq (local.get $d_tag) (i32.const 4))
    (then
      (local.set $d_arity (i32.const 2))
      (local.set $d_fhash (i32.const 87825375))
      (local.set $i (i32.const 0))
      (block $scan_done2
        (loop $scan2
          (br_if $scan_done2 (i32.ge_u (local.get $i) (local.get $count)))
          (local.set $entry_pc
            (i32.add (i32.add (local.get $cur_pc) (i32.const 1))
                     (local.get $i)))
          (local.set $entry_op1
            (call $fetch_instr_op1 (local.get $code_base) (local.get $entry_pc)))
          (local.set $entry_op2
            (call $fetch_instr_op2 (local.get $code_base) (local.get $entry_pc)))
          (local.set $entry_fhash (i32.wrap_i64 (local.get $entry_op1)))
          (local.set $entry_arity
            (i32.wrap_i64 (i64.shr_u (local.get $entry_op2) (i64.const 32))))
          (local.set $entry_target
            (i32.wrap_i64 (i64.and (local.get $entry_op2)
                                   (i64.const 0xFFFFFFFF))))
          (if (i32.and
                (i32.eq (local.get $entry_fhash) (local.get $d_fhash))
                (i32.eq (local.get $entry_arity) (local.get $d_arity)))
            (then
              (call $set_pc (local.get $entry_target))
              (return (i32.const 1))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $scan2)))))
  ;; Fall through past entries.
  (call $set_pc
    (i32.add (i32.add (local.get $cur_pc) (i32.const 1))
             (local.get $count)))
  (i32.const 1)').

wam_wat_case(switch_struct_entry,
'  ;; Record scanned by switch_on_struct / switch_on_term_hdr. Never
  ;; executed directly; behave as nop if reached.
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(switch_on_term_hdr,
'  ;; Mixed constant + structure indexing header.
  ;; op1 layout: low 32 = reg_idx, high 32 = const_count.
  ;; op2 = struct_count.
  ;; Layout in the instruction stream: header, then const_count
  ;; switch_entry records, then struct_count switch_struct_entry records.
  ;; Dispatch: deref reg. If tag <= 2 (atom/int/float), scan the
  ;; constant section. If tag = 3 (compound) or 4 (cons), scan the
  ;; structure section. Else fall through past both sections.
  (local $reg_idx i32) (local $cc i32) (local $sc i32)
  (local $cur_pc i32) (local $code_base i32)
  (local $d_addr i32) (local $d_tag i32) (local $d_payload i64)
  (local $d_arity i32) (local $d_fhash i32)
  (local $i i32) (local $entry_pc i32)
  (local $e1 i64) (local $e2 i64)
  (local $et i32) (local $ea i32) (local $etarget i32)
  (local.set $reg_idx (i32.wrap_i64 (local.get $op1)))
  (local.set $cc (i32.wrap_i64 (i64.shr_u (local.get $op1) (i64.const 32))))
  (local.set $sc (i32.wrap_i64 (local.get $op2)))
  (local.set $cur_pc (call $get_pc))
  (local.set $code_base (global.get $wam_code_base))
  (local.set $d_addr (call $deref_reg_addr (local.get $reg_idx)))
  (local.set $d_tag (call $val_tag (local.get $d_addr)))
  (local.set $d_payload (call $val_payload (local.get $d_addr)))
  ;; --- Constant section (tags 0..2) ---
  (if (i32.le_u (local.get $d_tag) (i32.const 2))
    (then
      (local.set $i (i32.const 0))
      (block $cdone
        (loop $cscan
          (br_if $cdone (i32.ge_u (local.get $i) (local.get $cc)))
          (local.set $entry_pc
            (i32.add (i32.add (local.get $cur_pc) (i32.const 1))
                     (local.get $i)))
          (local.set $e1
            (call $fetch_instr_op1 (local.get $code_base) (local.get $entry_pc)))
          (local.set $e2
            (call $fetch_instr_op2 (local.get $code_base) (local.get $entry_pc)))
          (local.set $et
            (i32.wrap_i64 (i64.shr_u (local.get $e2) (i64.const 32))))
          (local.set $etarget
            (i32.wrap_i64 (i64.and (local.get $e2)
                                   (i64.const 0xFFFFFFFF))))
          (if (i32.and
                (i32.eq (local.get $et) (local.get $d_tag))
                (i64.eq (local.get $e1) (local.get $d_payload)))
            (then
              (call $set_pc (local.get $etarget))
              (return (i32.const 1))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $cscan)))))
  ;; --- Structure section (tags 3, 4) ---
  (if (i32.or (i32.eq (local.get $d_tag) (i32.const 3))
              (i32.eq (local.get $d_tag) (i32.const 4)))
    (then
      (if (i32.eq (local.get $d_tag) (i32.const 4))
        (then
          (local.set $d_arity (i32.const 2))
          (local.set $d_fhash (i32.const 87825375)))
        (else
          (local.set $d_arity
            (i32.wrap_i64 (i64.shr_u (local.get $d_payload) (i64.const 32))))
          (local.set $d_fhash
            (i32.wrap_i64 (i64.and (local.get $d_payload)
                                   (i64.const 0xFFFFFFFF))))))
      (local.set $i (i32.const 0))
      (block $sdone
        (loop $sscan
          (br_if $sdone (i32.ge_u (local.get $i) (local.get $sc)))
          (local.set $entry_pc
            (i32.add (i32.add
                      (i32.add (local.get $cur_pc) (i32.const 1))
                      (local.get $cc))
                     (local.get $i)))
          (local.set $e1
            (call $fetch_instr_op1 (local.get $code_base) (local.get $entry_pc)))
          (local.set $e2
            (call $fetch_instr_op2 (local.get $code_base) (local.get $entry_pc)))
          (local.set $ea
            (i32.wrap_i64 (i64.shr_u (local.get $e2) (i64.const 32))))
          (local.set $etarget
            (i32.wrap_i64 (i64.and (local.get $e2)
                                   (i64.const 0xFFFFFFFF))))
          (if (i32.and
                (i64.eq (local.get $e1)
                        (i64.extend_i32_u (local.get $d_fhash)))
                (i32.eq (local.get $ea) (local.get $d_arity)))
            (then
              (call $set_pc (local.get $etarget))
              (return (i32.const 1))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $sscan)))))
  ;; Fall through past all entries (const + struct).
  (call $set_pc
    (i32.add (i32.add
              (i32.add (local.get $cur_pc) (i32.const 1))
              (local.get $cc))
             (local.get $sc)))
  (i32.const 1)').

wam_wat_case(trust_me,
'  ;; Remove choice point (last alternative). Guarded for cp_count=0
  ;; to support first-arg indexing that commits directly to this clause
  ;; without a matching try_me_else (see retry_me_else note).
  (if (i32.gt_s (call $get_cp_count) (i32.const 0))
    (then
      (call $pop_choice_point_no_restore)))
  (call $inc_pc)
  (i32.const 1)').

% ============================================================================
% PHASE 2b: Runtime helpers
% ============================================================================

%% compile_wam_helpers_to_wat(+Options, -WatCode)
compile_wam_helpers_to_wat(_Options, WatCode) :-
    wam_cp_size(CPSize),
    wam_cp_save_regs(CPSaveRegs),
    wam_num_regs(_NumRegs),
    wam_val_size(_ValSize),
    format(atom(WatCode),
';; --- Run loop ---
(func $run_loop (param $code_base i32) (param $num_instrs i32) (result i32)
  (local $pc i32)
  ;; Expose code_base to do_* handlers that need to walk the instruction
  ;; stream (e.g. switch_on_const reading switch_entry records at PC+1..).
  (global.set $wam_code_base (local.get $code_base))
  (block $exit
    (loop $continue
      ;; Check halted
      (br_if $exit (call $is_halted))
      ;; Check PC bounds
      (local.set $pc (call $get_pc))
      (if (i32.or (i32.lt_s (local.get $pc) (i32.const 0))
                  (i32.ge_s (local.get $pc) (local.get $num_instrs)))
        (then (return (i32.const 0))))
      ;; Step
      (if (call $step (local.get $code_base) (local.get $pc))
        (then (br $continue)))
      ;; Step failed - try backtrack
      (if (call $backtrack)
        (then (br $continue)))
      ;; Backtrack failed
      (return (i32.const 0))
    )
  )
  (i32.const 1)
)

;; --- Backtrack ---
(func $backtrack (result i32)
  (local $cp_off i32)
  (local $n i32)
  (local $trail_mark i32)
  (local $saved_cp i32)
  (local $next_pc i32)
  (local $i i32)
  ;; Check if any choice points
  (local.set $n (call $get_cp_count))
  (if (i32.eqz (local.get $n))
    (then
      ;; No more choice points. If we are inside an aggregate
      ;; (begin_aggregate was active), finalize instead of failing.
      (if (global.get $agg_active)
        (then
          ;; Finalize: bind res_reg to the accumulated result
          (global.set $agg_active (i32.const 0))
          ;; sum/max/min → bind to $agg_sum; count → bind to $agg_count
          (if (i32.eq (global.get $agg_type) (i32.const 1))
            (then
              (call $bind_reg_deref (global.get $agg_res_reg)
                (i32.const 1) (i64.extend_i32_u (global.get $agg_count))))
            (else
              (call $bind_reg_deref (global.get $agg_res_reg)
                (i32.const 1) (global.get $agg_sum))))
          ;; Restore the CP that was current when begin_aggregate ran.
          ;; The inner call (e.g. my_fact) overwrote CP; the proceed
          ;; at the end of the aggregate predicate needs the original
          ;; caller CP so it returns properly.
          (call $set_cp (global.get $agg_saved_cp))
          ;; Jump to the return PC (instruction after end_aggregate)
          (call $set_pc (global.get $agg_return_pc))
          (return (i32.const 1))))
      (return (i32.const 0))))
  ;; Choice point layout in stack: [next_pc:i32][trail_mark:i32][saved_cp:i32][64 regs x 12 bytes]
  ;; Size = ~w bytes per choice point (12 metadata + ~w regs x 12)
  ;; Pop the latest choice point (stored before env frames in stack)
  ;; For simplicity, choice points stored at fixed offsets from cp_count
  ;; TODO: implement full choice point stack
  (local.set $next_pc (call $cp_get_next_pc))
  (local.set $trail_mark (call $cp_get_trail_mark))
  (local.set $saved_cp (call $cp_get_saved_cp))
  ;; Unwind trail (restores bound heap cells to their old values)
  (call $unwind_trail (local.get $trail_mark))
  ;; Restore heap top (reclaim heap cells allocated since the CP was
  ;; pushed — standard WAM behavior that prevents stale Ref cells in
  ;; restored registers from pointing to garbage on the heap).
  (call $set_heap_top (call $cp_get_saved_heap_top))
  ;; Restore env_base (so Y register access via $reg_offset goes to
  ;; the correct environment frame on the stack).
  (global.set $env_base (call $cp_get_saved_env_base))
  ;; Restore A/X registers from choice point (Y registers are in the
  ;; environment frame and do not need separate CP save/restore).
  (call $cp_restore_regs)
  ;; Restore state
  (call $set_pc (local.get $next_pc))
  (call $set_cp (local.get $saved_cp))
  (call $set_halted (i32.const 0))
  (call $set_mode (i32.const 0))
  (i32.const 1)
)

;; --- Trail unwinding ---
;; Trail entries now store a memory offset (register OR heap address),
;; not a register index. Unwind uses val_store to restore the cell
;; at that offset, which works uniformly for both register cells
;; and heap cells created by put_variable.
(func $unwind_trail (param $mark i32)
  (local $toff i32)
  (local $mem_off i32)
  (local $old_tag i32)
  (local $old_payload i64)
  (block $done
    (loop $unwind
      (local.set $toff (call $get_trail_top))
      (br_if $done (i32.le_u (local.get $toff) (local.get $mark)))
      ;; Back up one entry (16 bytes)
      (local.set $toff (i32.sub (local.get $toff) (i32.const 16)))
      (call $set_trail_top (local.get $toff))
      ;; Restore cell at the saved memory offset
      (local.set $mem_off (i32.load (local.get $toff)))
      (local.set $old_tag (i32.load (i32.add (local.get $toff) (i32.const 4))))
      (local.set $old_payload (i64.load (i32.add (local.get $toff) (i32.const 8))))
      (call $val_store (local.get $mem_off) (local.get $old_tag) (local.get $old_payload))
      (br $unwind)
    )
  )
)

;; --- Choice point management ---
;; Choice points stored in a dedicated area starting at offset 98304 (page 1.5)
;; Each CP: [next_pc:i32 +0][trail_mark:i32 +4][saved_cp:i32 +8]
;;         [heap_top:i32 +12][env_base:i32 +16][retry_n:i32 +20]
;;         [8 A regs: 96 bytes +24] = ~w bytes.
;; retry_n is nonzero only when the CP was pushed by nondeterministic
;; arg/3; it stores the N most recently returned by the iterator so the
;; next retry can increment it. All other push paths initialize it to 0.

(func $cp_base_offset (result i32) (i32.const 98304))

(func $cp_offset (param $idx i32) (result i32)
  (i32.add (call $cp_base_offset) (i32.mul (local.get $idx) (i32.const ~w))))

(func $push_choice_point (param $next_pc i32)
  (local $n i32) (local $off i32) (local $i i32)
  (local.set $n (call $get_cp_count))
  (local.set $off (call $cp_offset (local.get $n)))
  ;; Save next_pc, trail mark, CP, heap_top, env_base, retry_n=0.
  (i32.store (local.get $off) (local.get $next_pc))
  (i32.store (i32.add (local.get $off) (i32.const 4)) (call $get_trail_top))
  (i32.store (i32.add (local.get $off) (i32.const 8)) (call $get_cp))
  (i32.store (i32.add (local.get $off) (i32.const 12)) (call $get_heap_top))
  (i32.store (i32.add (local.get $off) (i32.const 16)) (global.get $env_base))
  (i32.store (i32.add (local.get $off) (i32.const 20)) (i32.const 0))
  ;; Save first ~w argument registers at +24 (metadata ends at +24).
  (local.set $i (i32.const 0))
  (block $done
    (loop $save
      (br_if $done (i32.ge_u (local.get $i) (i32.const ~w)))
      (call $copy_from_reg (local.get $i)
        (i32.add (i32.add (local.get $off) (i32.const 24))
                 (i32.mul (local.get $i) (i32.const 12))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $save)))
  (call $set_cp_count (i32.add (local.get $n) (i32.const 1))))

(func $update_choice_point (param $next_pc i32)
  (local $n i32) (local $off i32)
  (local.set $n (i32.sub (call $get_cp_count) (i32.const 1)))
  (local.set $off (call $cp_offset (local.get $n)))
  (i32.store (local.get $off) (local.get $next_pc)))

(func $pop_choice_point_no_restore
  (call $set_cp_count (i32.sub (call $get_cp_count) (i32.const 1))))

(func $cp_get_next_pc (result i32)
  (local $off i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (i32.load (local.get $off)))

(func $cp_get_trail_mark (result i32)
  (local $off i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (i32.load (i32.add (local.get $off) (i32.const 4))))

(func $cp_get_saved_cp (result i32)
  (local $off i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (i32.load (i32.add (local.get $off) (i32.const 8))))

(func $cp_get_saved_heap_top (result i32)
  (local $off i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (i32.load (i32.add (local.get $off) (i32.const 12))))

(func $cp_get_saved_env_base (result i32)
  (local $off i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (i32.load (i32.add (local.get $off) (i32.const 16))))

(func $cp_restore_regs
  (local $off i32) (local $i i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (local.set $i (i32.const 0))
  (block $done
    (loop $restore
      (br_if $done (i32.ge_u (local.get $i) (i32.const ~w)))
      (call $copy_to_reg (local.get $i)
        (i32.add (i32.add (local.get $off) (i32.const 24))
                 (i32.mul (local.get $i) (i32.const 12))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $restore))))

;; Read/write the retry_n slot (+20) on the top choice point. Used by
;; nondeterministic arg/3 to track iteration state per-CP so nested
;; unbound-N calls do not clobber each other''s state.
(func $cp_get_retry_n (result i32)
  (local $off i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (i32.load (i32.add (local.get $off) (i32.const 20))))

(func $cp_set_retry_n (param $n i32)
  (local $off i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (i32.store (i32.add (local.get $off) (i32.const 20)) (local.get $n)))

;; --- Unification ---
;; Unify two registers. Follows Ref chains via $deref_reg_addr so
;; heap-allocated variable cells created by put_variable are properly
;; reached and bound. No occurs check (standard Prolog semantics).
(func $unify_regs (param $r1 i32) (param $r2 i32) (result i32)
  (local $a1 i32) (local $a2 i32)
  (local $t1 i32) (local $t2 i32) (local $p1 i64) (local $p2 i64)
  ;; Deref both registers to the final cell address (may be on heap).
  (local.set $a1 (call $deref_reg_addr (local.get $r1)))
  (local.set $a2 (call $deref_reg_addr (local.get $r2)))
  (local.set $t1 (call $val_tag (local.get $a1)))
  (local.set $p1 (call $val_payload (local.get $a1)))
  (local.set $t2 (call $val_tag (local.get $a2)))
  (local.set $p2 (call $val_payload (local.get $a2)))
  ;; If cell 1 is unbound, bind it to cell 2 value
  (if (i32.eq (local.get $t1) (i32.const 6))
    (then
      (call $trail_binding_at (local.get $a1))
      (call $val_store (local.get $a1) (local.get $t2) (local.get $p2))
      (return (i32.const 1))))
  ;; If cell 2 is unbound, bind it to cell 1 value
  (if (i32.eq (local.get $t2) (i32.const 6))
    (then
      (call $trail_binding_at (local.get $a2))
      (call $val_store (local.get $a2) (local.get $t1) (local.get $p1))
      (return (i32.const 1))))
  ;; Both bound: shallow equality check
  (i32.and
    (i32.eq (local.get $t1) (local.get $t2))
    (i64.eq (local.get $p1) (local.get $p2))))

;; --- Builtin dispatch ---
;; O(1) br_table dispatch for ALL builtins. Earlier versions routed
;; term inspection (18-21) through a linear if-chain and left IDs
;; 9-14 (type checks) + ID 22 (=/2) unimplemented — IDs 9-14 mapped
;; to $default (returning 0), and =/2 was never even in the id table
;; so the canonical WAM layer emitted `execute =/2` which resolved
;; to PC=0 and corrupted execution. This version:
;;   * maps IDs 9-14 to real type-check handlers ($var, $nonvar,
;;     $atom, $integer, $float, $number)
;;   * adds ID 22 $eq for =/2 calling $unify_regs A1 A2
;; All builtins still dispatch in O(1) via one bounds check + one
;; indirect branch.
(func $execute_builtin (param $id i32) (param $arity i32) (result i32)
  (block $default
    (block $tge (block $tle (block $tgt (block $tlt
    (block $tne (block $teq
    (block $is_list_b
    (block $compound_b
    (block $eq
    (block $copy_term (block $univ (block $arg (block $functor
    (block $cut (block $fail (block $true_b
    (block $number (block $float (block $integer
    (block $atom (block $nonvar (block $var
    (block $arith_ge (block $arith_le (block $arith_gt (block $arith_lt
    (block $arith_ne (block $arith_eq (block $is
    (block $nl (block $write
      (br_table $write $nl $is $arith_eq $arith_ne $arith_lt $arith_gt $arith_le $arith_ge
                $var $nonvar $atom $integer $float $number
                $true_b $fail $cut
                $functor $arg $univ $copy_term
                $eq
                $compound_b $is_list_b
                $teq $tne $tlt $tgt $tle $tge
                $default (local.get $id))
    ) ;; write
    (call $print_i64 (call $deref_reg_payload (i32.const 0)))
    (return (i32.const 1))
    ) ;; nl
    (call $print_newline)
    (return (i32.const 1))
    ) ;; is
    (return (call $builtin_is))
    ) ;; =:=
    (return (call $builtin_arith_cmp (i32.const 0)))
    ) ;; =\\=
    (return (call $builtin_arith_cmp (i32.const 1)))
    ) ;; <
    (return (call $builtin_arith_cmp (i32.const 2)))
    ) ;; >
    (return (call $builtin_arith_cmp (i32.const 3)))
    ) ;; =<
    (return (call $builtin_arith_cmp (i32.const 4)))
    ) ;; >=
    (return (call $builtin_arith_cmp (i32.const 5)))
    ) ;; var/1 (ID 9): A1 tag == 6 (unbound)
    (return (i32.eq (call $deref_reg_tag (i32.const 0)) (i32.const 6)))
    ) ;; nonvar/1 (ID 10): A1 tag != 6
    (return (i32.ne (call $deref_reg_tag (i32.const 0)) (i32.const 6)))
    ) ;; atom/1 (ID 11): A1 tag == 0
    (return (i32.eq (call $deref_reg_tag (i32.const 0)) (i32.const 0)))
    ) ;; integer/1 (ID 12): A1 tag == 1
    (return (i32.eq (call $deref_reg_tag (i32.const 0)) (i32.const 1)))
    ) ;; float/1 (ID 13): A1 tag == 2
    (return (i32.eq (call $deref_reg_tag (i32.const 0)) (i32.const 2)))
    ) ;; number/1 (ID 14): A1 tag == 1 (integer) or 2 (float)
    (return (i32.or
              (i32.eq (call $deref_reg_tag (i32.const 0)) (i32.const 1))
              (i32.eq (call $deref_reg_tag (i32.const 0)) (i32.const 2))))
    ) ;; true (ID 15)
    (return (i32.const 1))
    ) ;; fail (ID 16)
    (return (i32.const 0))
    ) ;; cut (ID 17)
    (call $set_cp_count (i32.const 0))
    (return (i32.const 1))
    ) ;; functor/3 (ID 18)
    (return (call $builtin_functor))
    ) ;; arg/3 (ID 19)
    (return (call $builtin_arg))
    ) ;; =../2 (ID 20)
    (return (call $builtin_univ))
    ) ;; copy_term/2 (ID 21)
    (return (call $builtin_copy_term))
    ) ;; =/2 (ID 22): unify A1 and A2 via the shared $unify_regs helper
    (return (call $unify_regs (i32.const 0) (i32.const 1)))
    ) ;; compound/1 (ID 23): A1 tag is compound (3) or list cons (4)
    (return (call $builtin_compound))
    ) ;; is_list/1 (ID 24): A1 dereferences to a proper list (terminates at empty-list atom)
    (return (call $builtin_is_list))
    ) ;; ==/2 (ID 25): structural equality
    (return (call $term_equal
              (call $deref_reg_addr (i32.const 0))
              (call $deref_reg_addr (i32.const 1))))
    ) ;; \\==/2 (ID 26): structural non-equality
    (return (i32.eqz (call $term_equal
              (call $deref_reg_addr (i32.const 0))
              (call $deref_reg_addr (i32.const 1)))))
    ) ;; @</2 (ID 27): standard-order less-than
    (return (i32.lt_s (call $term_compare
              (call $deref_reg_addr (i32.const 0))
              (call $deref_reg_addr (i32.const 1))) (i32.const 0)))
    ) ;; @>/2 (ID 28): standard-order greater-than
    (return (i32.gt_s (call $term_compare
              (call $deref_reg_addr (i32.const 0))
              (call $deref_reg_addr (i32.const 1))) (i32.const 0)))
    ) ;; @=</2 (ID 29): standard-order less-or-equal
    (return (i32.le_s (call $term_compare
              (call $deref_reg_addr (i32.const 0))
              (call $deref_reg_addr (i32.const 1))) (i32.const 0)))
    ) ;; @>=/2 (ID 30): standard-order greater-or-equal
    (return (i32.ge_s (call $term_compare
              (call $deref_reg_addr (i32.const 0))
              (call $deref_reg_addr (i32.const 1))) (i32.const 0)))
  )
  ;; $default fall-through: unknown builtin ID
  (i32.const 0)
)

;; --- is/2: A1 = eval(A2) ---
;; Evaluates the arithmetic expression in A2, binds (or checks) A1 to
;; the result. Uses $eval_arith for recursive compound evaluation.
(func $builtin_is (result i32)
  (local $val i64) (local $a1_addr i32) (local $a2_addr i32)
  ;; Evaluate A2 as an arithmetic expression.
  (global.set $arith_ok (i32.const 1))
  (local.set $a2_addr (call $deref_reg_addr (i32.const 1)))
  (local.set $val (call $eval_arith (local.get $a2_addr)))
  (if (i32.eqz (global.get $arith_ok))
    (then (return (i32.const 0))))
  ;; Bind or check A1.
  (local.set $a1_addr (call $deref_reg_addr (i32.const 0)))
  (if (i32.eq (call $val_tag (local.get $a1_addr)) (i32.const 6))
    (then
      (call $trail_binding_at (local.get $a1_addr))
      (call $val_store (local.get $a1_addr) (i32.const 1) (local.get $val))
      (return (i32.const 1))))
  (if (i32.and
        (i32.eq (call $val_tag (local.get $a1_addr)) (i32.const 1))
        (i64.eq (call $val_payload (local.get $a1_addr)) (local.get $val)))
    (then (return (i32.const 1))))
  (i32.const 0)
)

;; --- Recursive arithmetic evaluator ---
;; Reads the cell at $addr and evaluates it as an integer arithmetic
;; expression. Handles:
;;   tag=1 (integer) → return payload directly
;;   tag=5 (ref)     → follow to target, retry
;;   tag=3 (compound) with arity 2 → dispatch on functor hash:
;;     42830 = hash("+/2")  → add
;;     44752 = hash("-/2")  → subtract
;;     41869 = hash("*/2")  → multiply
;;   tag=3 with arity 1 → dispatch:
;;     44751 = hash("-/1")  → unary negate
;;   anything else → set $arith_ok = 0, return 0
;; Functor hashes are pre-computed DJB2 (acc*31+char mod 2^31-1)
;; matching atom_hash_i64/2 in the Prolog compile layer.
(func $eval_arith (param $addr i32) (result i64)
  (local $tag i32) (local $payload i64) (local $target i32)
  (local $header i64) (local $functor_hash i32) (local $arity i32)
  (local $a i64) (local $b i64)
  (local.set $tag (call $val_tag (local.get $addr)))
  (local.set $payload (call $val_payload (local.get $addr)))
  ;; Integer leaf
  (if (i32.eq (local.get $tag) (i32.const 1))
    (then (return (local.get $payload))))
  ;; Ref: follow one level then retry
  (if (i32.eq (local.get $tag) (i32.const 5))
    (then
      (return (call $eval_arith
                (i32.wrap_i64 (local.get $payload))))))
  ;; Compound: read header, dispatch on functor hash
  (if (i32.eq (local.get $tag) (i32.const 3))
    (then
      (local.set $header (local.get $payload))
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u (local.get $header) (i64.const 32))))
      (local.set $functor_hash
        (i32.wrap_i64 (i64.and (local.get $header) (i64.const 0xFFFFFFFF))))
      ;; Binary operators (arity 2)
      (if (i32.eq (local.get $arity) (i32.const 2))
        (then
          (local.set $a (call $eval_arith
                          (i32.add (local.get $addr) (i32.const 12))))
          (if (i32.eqz (global.get $arith_ok)) (then (return (i64.const 0))))
          (local.set $b (call $eval_arith
                          (i32.add (local.get $addr) (i32.const 24))))
          (if (i32.eqz (global.get $arith_ok)) (then (return (i64.const 0))))
          ;; + (hash 42830)
          (if (i32.eq (local.get $functor_hash) (i32.const 42830))
            (then (return (i64.add (local.get $a) (local.get $b)))))
          ;; - (hash 44752)
          (if (i32.eq (local.get $functor_hash) (i32.const 44752))
            (then (return (i64.sub (local.get $a) (local.get $b)))))
          ;; * (hash 41869)
          (if (i32.eq (local.get $functor_hash) (i32.const 41869))
            (then (return (i64.mul (local.get $a) (local.get $b)))))
          ))
      ;; Unary operators (arity 1)
      (if (i32.eq (local.get $arity) (i32.const 1))
        (then
          (local.set $a (call $eval_arith
                          (i32.add (local.get $addr) (i32.const 12))))
          (if (i32.eqz (global.get $arith_ok)) (then (return (i64.const 0))))
          ;; unary - (hash 44751)
          (if (i32.eq (local.get $functor_hash) (i32.const 44751))
            (then (return (i64.sub (i64.const 0) (local.get $a)))))
          ))))
  ;; Failure: not evaluable
  (global.set $arith_ok (i32.const 0))
  (i64.const 0)
)

;; --- compound/1: A1 dereferences to a compound (tag=3) or cons (tag=4) ---
(func $builtin_compound (result i32)
  (local $tag i32)
  (local.set $tag (call $deref_reg_tag (i32.const 0)))
  (i32.or
    (i32.eq (local.get $tag) (i32.const 3))
    (i32.eq (local.get $tag) (i32.const 4))))

;; --- is_list/1: A1 dereferences to a proper list ---
;; A proper list is either the atom empty-list (tag=0, payload=2914, DJB2
;; hash of chars 91,93) or a cons cell whose tail, after walking Ref
;; chains, is itself a proper list. The canonical WAM compiler emits two
;; cons representations:
;;   * put_list: a tag=4 (list) cell with head at +12, tail at +24.
;;   * put_structure [|]/2: a tag=3 (compound) cell whose header payload
;;     encodes arity=2 and functor_hash=87825375 (DJB2 of "[|]/2"), with
;;     head at +12 and tail at +24.
;; This helper accepts either representation so that mixed-build lists
;; (a put_list outer cons wrapping put_structure inner conses) validate.
(func $builtin_is_list (result i32)
  (local $addr i32) (local $tag i32) (local $payload i64)
  (local $arity i32) (local $fhash i32)
  (local.set $addr (call $deref_reg_addr (i32.const 0)))
  (loop $walk
    (local.set $tag (call $val_tag (local.get $addr)))
    (local.set $payload (call $val_payload (local.get $addr)))
    ;; Empty-list atom terminates: success.
    (if (i32.and
          (i32.eq (local.get $tag) (i32.const 0))
          (i64.eq (local.get $payload) (i64.const 2914)))
      (then (return (i32.const 1))))
    ;; tag=4 cons (put_list form).
    (if (i32.eq (local.get $tag) (i32.const 4))
      (then
        (local.set $addr
          (i32.add (local.get $addr) (i32.const 24)))
        (block $tail_done
          (loop $tail_deref
            (br_if $tail_done
              (i32.ne (call $val_tag (local.get $addr)) (i32.const 5)))
            (local.set $addr
              (i32.wrap_i64 (call $val_payload (local.get $addr))))
            (br $tail_deref)))
        (br $walk)))
    ;; tag=3 compound with functor [|]/2 (put_structure form).
    (if (i32.eq (local.get $tag) (i32.const 3))
      (then
        (local.set $arity
          (i32.wrap_i64 (i64.shr_u (local.get $payload) (i64.const 32))))
        (local.set $fhash
          (i32.wrap_i64 (i64.and (local.get $payload) (i64.const 0xFFFFFFFF))))
        (if (i32.and
              (i32.eq (local.get $arity) (i32.const 2))
              (i32.eq (local.get $fhash) (i32.const 87825375)))
          (then
            (local.set $addr
              (i32.add (local.get $addr) (i32.const 24)))
            (block $tail_done2
              (loop $tail_deref2
                (br_if $tail_done2
                  (i32.ne (call $val_tag (local.get $addr)) (i32.const 5)))
                (local.set $addr
                  (i32.wrap_i64 (call $val_payload (local.get $addr))))
                (br $tail_deref2)))
            (br $walk)))))
    ;; Any other tag (integer, float, compound non-cons, unbound, ref to
    ;; unbound): not a proper list.
    (return (i32.const 0)))
  (unreachable))

;; --- Deref an arbitrary memory cell through Ref chains ---
;; Follows tag=5 (Ref) indirections to the final cell and returns its
;; memory address. Shares semantics with $deref_reg_addr but starts
;; from a cell offset rather than a register index. Used by the term
;; equality and term-order helpers to walk compound/cons argument
;; slots which may hold Ref(H) after put_variable aliasing.
(func $deref_cell (param $off i32) (result i32)
  (block $done
    (loop $d
      (br_if $done (i32.ne (call $val_tag (local.get $off)) (i32.const 5)))
      (local.set $off (i32.wrap_i64 (call $val_payload (local.get $off))))
      (br $d)))
  (local.get $off))

;; --- Structural equality (==/2, \\==/2) ---
;; Compares two terms for structural identity WITHOUT unification.
;; Takes cell memory addresses. Derefs Ref chains at each level.
;; Compound terms compare header (arity + functor_hash) then recurse
;; on each argument. Cons cells (tag=4) compare head and tail.
;; Mixed representations (tag=3 [|]/2 vs tag=4 cons) return NOT equal;
;; this matches what the WAM-WAT compiler emits for a given source
;; but is more restrictive than SWI Prolog''s == (which normalizes).
(func $term_equal (param $a i32) (param $b i32) (result i32)
  (local $aa i32) (local $bb i32)
  (local $ta i32) (local $tb i32)
  (local $pa i64) (local $pb i64)
  (local $arity i32) (local $i i32)
  (local.set $aa (call $deref_cell (local.get $a)))
  (local.set $bb (call $deref_cell (local.get $b)))
  (local.set $ta (call $val_tag (local.get $aa)))
  (local.set $tb (call $val_tag (local.get $bb)))
  (if (i32.ne (local.get $ta) (local.get $tb))
    (then (return (i32.const 0))))
  (local.set $pa (call $val_payload (local.get $aa)))
  (local.set $pb (call $val_payload (local.get $bb)))
  ;; Tag 0/1/2 (atom/int/float): payload comparison is sufficient.
  (if (i32.le_u (local.get $ta) (i32.const 2))
    (then (return (i64.eq (local.get $pa) (local.get $pb)))))
  ;; Tag 6 (unbound): identity comparison (same cell address).
  (if (i32.eq (local.get $ta) (i32.const 6))
    (then (return (i32.eq (local.get $aa) (local.get $bb)))))
  ;; Tag 3 (compound): header equal + each arg equal.
  (if (i32.eq (local.get $ta) (i32.const 3))
    (then
      (if (i64.ne (local.get $pa) (local.get $pb))
        (then (return (i32.const 0))))
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u (local.get $pa) (i64.const 32))))
      (local.set $i (i32.const 1))
      (block $cdone
        (loop $cloop
          (br_if $cdone (i32.gt_s (local.get $i) (local.get $arity)))
          (if (i32.eqz (call $term_equal
                (i32.add (local.get $aa)
                  (i32.mul (local.get $i) (i32.const 12)))
                (i32.add (local.get $bb)
                  (i32.mul (local.get $i) (i32.const 12)))))
            (then (return (i32.const 0))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $cloop)))
      (return (i32.const 1))))
  ;; Tag 4 (cons list): head (+12) and tail (+24).
  (if (i32.eq (local.get $ta) (i32.const 4))
    (then
      (if (i32.eqz (call $term_equal
            (i32.add (local.get $aa) (i32.const 12))
            (i32.add (local.get $bb) (i32.const 12))))
        (then (return (i32.const 0))))
      (return (call $term_equal
            (i32.add (local.get $aa) (i32.const 24))
            (i32.add (local.get $bb) (i32.const 24))))))
  ;; Tag 5 should not survive $deref_cell; reach here only for unknown tags.
  (i32.const 0))

;; --- Atom name table lookup + lexicographic compare ---
;; Table layout (at 327680):
;;   [count:i32][entries × 16 bytes: hash:i64, name_off:i32, name_len:i32]
;;   [string_pool: concatenated atom name bytes]
;; Entries are sorted by hash at compile time so the runtime could
;; binary-search; we linear-scan here for simplicity.

(func $atom_table_base (result i32) (i32.const 327680))

;; Find the entry for a given atom hash. Returns the absolute memory
;; offset of the entry (pointing at the hash field) or -1 if not found.
(func $atom_lookup (param $hash i64) (result i32)
  (local $base i32) (local $count i32) (local $i i32) (local $off i32)
  (local.set $base (call $atom_table_base))
  (local.set $count (i32.load (local.get $base)))
  (local.set $i (i32.const 0))
  (block $done
    (loop $scan
      (br_if $done (i32.ge_u (local.get $i) (local.get $count)))
      (local.set $off
        (i32.add (i32.add (local.get $base) (i32.const 4))
                 (i32.mul (local.get $i) (i32.const 16))))
      (if (i64.eq (i64.load (local.get $off)) (local.get $hash))
        (then (return (local.get $off))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $scan)))
  (i32.const -1))

;; Lexicographic byte comparison of two atom names, given their hashes.
;; Returns -1 / 0 / +1. Falls back to hash comparison if either atom
;; is missing from the table (defensive — should not happen for atoms
;; reached by the compiler). Identical hashes short-circuit to 0
;; without a lookup.
(func $atom_compare (param $ha i64) (param $hb i64) (result i32)
  (local $ea i32) (local $eb i32)
  (local $oa i32) (local $la i32) (local $ob i32) (local $lb i32)
  (local $i i32) (local $minlen i32)
  (local $ca i32) (local $cb i32)
  (if (i64.eq (local.get $ha) (local.get $hb))
    (then (return (i32.const 0))))
  (local.set $ea (call $atom_lookup (local.get $ha)))
  (local.set $eb (call $atom_lookup (local.get $hb)))
  (if (i32.or (i32.eq (local.get $ea) (i32.const -1))
              (i32.eq (local.get $eb) (i32.const -1)))
    (then
      ;; Fallback: hash order (deterministic if table incomplete).
      (if (i64.lt_u (local.get $ha) (local.get $hb))
        (then (return (i32.const -1))))
      (return (i32.const 1))))
  (local.set $oa (i32.load (i32.add (local.get $ea) (i32.const 8))))
  (local.set $la (i32.load (i32.add (local.get $ea) (i32.const 12))))
  (local.set $ob (i32.load (i32.add (local.get $eb) (i32.const 8))))
  (local.set $lb (i32.load (i32.add (local.get $eb) (i32.const 12))))
  (if (i32.lt_s (local.get $la) (local.get $lb))
    (then (local.set $minlen (local.get $la)))
    (else (local.set $minlen (local.get $lb))))
  (local.set $i (i32.const 0))
  (block $eq_prefix
    (loop $bytecmp
      (br_if $eq_prefix (i32.ge_u (local.get $i) (local.get $minlen)))
      (local.set $ca
        (i32.load8_u (i32.add (local.get $oa) (local.get $i))))
      (local.set $cb
        (i32.load8_u (i32.add (local.get $ob) (local.get $i))))
      (if (i32.lt_u (local.get $ca) (local.get $cb))
        (then (return (i32.const -1))))
      (if (i32.gt_u (local.get $ca) (local.get $cb))
        (then (return (i32.const 1))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $bytecmp)))
  ;; Prefix bytes equal; shorter atom is less.
  (if (i32.lt_s (local.get $la) (local.get $lb))
    (then (return (i32.const -1))))
  (if (i32.gt_s (local.get $la) (local.get $lb))
    (then (return (i32.const 1))))
  (i32.const 0))

;; --- Standard-order comparison (@</2, @>/2, @=</2, @>=/2) ---
;; Returns -1 if a < b, 0 if equal, 1 if a > b per Prolog standard
;; term order: Var < Number < Atom < Compound. Atom-vs-atom order is
;; lexicographic via $atom_compare (PR integrating a name table).
;; Numeric order within tag 1 (int) and tag 2 (float) is correct.
;; Compounds order by arity, then functor_hash, then args left-to-right.
(func $tag_order (param $tag i32) (result i32)
  ;; Map raw tag to category order: 6→0, 1/2→1, 0→2, 3/4→3, else→4.
  (if (i32.eq (local.get $tag) (i32.const 6))
    (then (return (i32.const 0))))
  (if (i32.or (i32.eq (local.get $tag) (i32.const 1))
              (i32.eq (local.get $tag) (i32.const 2)))
    (then (return (i32.const 1))))
  (if (i32.eq (local.get $tag) (i32.const 0))
    (then (return (i32.const 2))))
  (if (i32.or (i32.eq (local.get $tag) (i32.const 3))
              (i32.eq (local.get $tag) (i32.const 4)))
    (then (return (i32.const 3))))
  (i32.const 4))

(func $term_compare (param $a i32) (param $b i32) (result i32)
  (local $aa i32) (local $bb i32)
  (local $ta i32) (local $tb i32)
  (local $oa i32) (local $ob i32)
  (local $pa i64) (local $pb i64)
  (local $arity_a i32) (local $arity_b i32)
  (local $fa i64) (local $fb i64)
  (local $i i32) (local $c i32)
  (local.set $aa (call $deref_cell (local.get $a)))
  (local.set $bb (call $deref_cell (local.get $b)))
  (local.set $ta (call $val_tag (local.get $aa)))
  (local.set $tb (call $val_tag (local.get $bb)))
  (local.set $oa (call $tag_order (local.get $ta)))
  (local.set $ob (call $tag_order (local.get $tb)))
  (if (i32.ne (local.get $oa) (local.get $ob))
    (then
      (if (i32.lt_s (local.get $oa) (local.get $ob))
        (then (return (i32.const -1))))
      (return (i32.const 1))))
  (local.set $pa (call $val_payload (local.get $aa)))
  (local.set $pb (call $val_payload (local.get $bb)))
  ;; Same tag-order. Branch by tag.
  ;; Tag 6 (unbound): compare addresses.
  (if (i32.eq (local.get $ta) (i32.const 6))
    (then
      (if (i32.lt_s (local.get $aa) (local.get $bb))
        (then (return (i32.const -1))))
      (if (i32.gt_s (local.get $aa) (local.get $bb))
        (then (return (i32.const 1))))
      (return (i32.const 0))))
  ;; Tag 1 (int): signed i64 comparison.
  (if (i32.eq (local.get $ta) (i32.const 1))
    (then
      (if (i64.lt_s (local.get $pa) (local.get $pb))
        (then (return (i32.const -1))))
      (if (i64.gt_s (local.get $pa) (local.get $pb))
        (then (return (i32.const 1))))
      (return (i32.const 0))))
  ;; Tag 2 (float): reinterpret payload bits to f64, compare.
  (if (i32.eq (local.get $ta) (i32.const 2))
    (then
      (if (f64.lt (f64.reinterpret_i64 (local.get $pa))
                  (f64.reinterpret_i64 (local.get $pb)))
        (then (return (i32.const -1))))
      (if (f64.gt (f64.reinterpret_i64 (local.get $pa))
                  (f64.reinterpret_i64 (local.get $pb)))
        (then (return (i32.const 1))))
      (return (i32.const 0))))
  ;; Tag 0 (atom): look up the atom name table and do a lexicographic
  ;; byte comparison on the stored names. Falls back to hash
  ;; comparison if either atom is missing from the table (should not
  ;; happen for atoms seen by the compiler, but defensive).
  (if (i32.eq (local.get $ta) (i32.const 0))
    (then
      (return (call $atom_compare (local.get $pa) (local.get $pb)))))
  ;; Tag 3 (compound): arity, then functor_hash, then args.
  (if (i32.eq (local.get $ta) (i32.const 3))
    (then
      ;; If b is tag 4 (cons), treat as arity=2 functor=[|]/2.
      (local.set $arity_a
        (i32.wrap_i64 (i64.shr_u (local.get $pa) (i64.const 32))))
      (local.set $fa (i64.and (local.get $pa) (i64.const 0xFFFFFFFF)))
      (if (i32.eq (local.get $tb) (i32.const 4))
        (then
          (local.set $arity_b (i32.const 2))
          (local.set $fb (i64.const 87825375)))
        (else
          (local.set $arity_b
            (i32.wrap_i64 (i64.shr_u (local.get $pb) (i64.const 32))))
          (local.set $fb (i64.and (local.get $pb) (i64.const 0xFFFFFFFF)))))
      (if (i32.lt_s (local.get $arity_a) (local.get $arity_b))
        (then (return (i32.const -1))))
      (if (i32.gt_s (local.get $arity_a) (local.get $arity_b))
        (then (return (i32.const 1))))
      (if (i64.lt_u (local.get $fa) (local.get $fb))
        (then (return (i32.const -1))))
      (if (i64.gt_u (local.get $fa) (local.get $fb))
        (then (return (i32.const 1))))
      ;; Same header: recurse on args.
      (local.set $i (i32.const 1))
      (block $cdone
        (loop $cloop
          (br_if $cdone (i32.gt_s (local.get $i) (local.get $arity_a)))
          (local.set $c (call $term_compare
                (i32.add (local.get $aa)
                  (i32.mul (local.get $i) (i32.const 12)))
                (i32.add (local.get $bb)
                  (i32.mul (local.get $i) (i32.const 12)))))
          (if (i32.ne (local.get $c) (i32.const 0))
            (then (return (local.get $c))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $cloop)))
      (return (i32.const 0))))
  ;; Tag 4 (cons) at a; b might also be tag 4 or tag 3. If tag 3,
  ;; the symmetric case above handled it by the $oa == $ob check
  ;; (tag 3 and 4 both map to order 3). Here both are tag 4 or
  ;; a=tag4 b=tag3. Handle a=tag4 b=tag3 by swapping-and-negating.
  (if (i32.eq (local.get $ta) (i32.const 4))
    (then
      (if (i32.eq (local.get $tb) (i32.const 3))
        (then (return (i32.sub (i32.const 0)
                       (call $term_compare (local.get $bb) (local.get $aa))))))
      ;; Both cons. Arity 2, same functor — just recurse on head and tail.
      (local.set $c (call $term_compare
            (i32.add (local.get $aa) (i32.const 12))
            (i32.add (local.get $bb) (i32.const 12))))
      (if (i32.ne (local.get $c) (i32.const 0))
        (then (return (local.get $c))))
      (return (call $term_compare
            (i32.add (local.get $aa) (i32.const 24))
            (i32.add (local.get $bb) (i32.const 24))))))
  (i32.const 0))

;; --- Arithmetic comparison: 0==eq, 1==ne, 2==lt, 3==gt, 4==le, 5==ge ---
;; Nested-block br_table dispatch. Each case line has the form
;;   ) (return (i64.XX ...))
;; where the leading `)` closes the corresponding $XX block and the
;; return expression must balance its own parens — one too many closes
;; per line makes the function paren-imbalanced and wat2wasm rejects
;; the entire module with cascading "undefined function" errors on
;; downstream symbols. (The original WAM-WAT target PR #1224 had an
;; extra `)` per line; it was never caught because the wat2wasm_validates
;; test used assertion/1 which only warns rather than failing the test.)
(func $builtin_arith_cmp (param $op i32) (result i32)
  (local $a i64) (local $b i64)
  (local.set $a (call $deref_reg_payload (i32.const 0)))
  (local.set $b (call $deref_reg_payload (i32.const 1)))
  (block $default
    (block $ge (block $le (block $gt (block $lt (block $ne (block $eq
      (br_table $eq $ne $lt $gt $le $ge $default (local.get $op))
    ) (return (i64.eq (local.get $a) (local.get $b)))
    ) (return (i64.ne (local.get $a) (local.get $b)))
    ) (return (i64.lt_s (local.get $a) (local.get $b)))
    ) (return (i64.gt_s (local.get $a) (local.get $b)))
    ) (return (i64.le_s (local.get $a) (local.get $b)))
    ) (return (i64.ge_s (local.get $a) (local.get $b)))
  )
  (i32.const 0)
)

;; --- arg/3: A1 = N, A2 = T (ref->compound), A3 = result ---
;; Two modes based on A1:
;;   Ground N: deterministic — unify A3 with the Nth arg cell of T.
;;   Unbound N: nondeterministic — enumerate N = 1..arity on backtrack.
;;     Iteration state lives in the CP frame''s retry_n slot (+20) —
;;     NOT in globals — so nested unbound-N calls each carry their own
;;     state on their own CP and cannot clobber each other. Detection
;;     of retry vs fresh entry: if cp_count > 0 AND top CP''s next_pc
;;     equals the current builtin_call PC AND its retry_n > 0, it''s a
;;     retry from our own CP; otherwise start fresh. Fresh entry with
;;     arity >= 1 always pushes a CP so the retry path has somewhere
;;     to write state; arity=1 exhausts on the immediate next retry
;;     and pops the CP.
;; Argument cells are laid out contiguously after the compound header,
;; each cell 12 bytes. Arity is the high 32 bits of the compound header
;; payload (see encode_structure_op1).
(func $builtin_arg (result i32)
  (local $n i32) (local $a1_tag i32)
  (local $t_tag i32)
  (local $comp_addr i32)
  (local $comp_payload i64)
  (local $arity i32)
  (local $arg_off i32)
  (local $arg_tag i32)
  (local $arg_payload i64)
  (local $a3_tag i32)
  (local $cur_pc i32)
  (local $retry_n i32)
  ;; Dereference A2 first — both modes need T to be a compound.
  (local.set $t_tag (call $deref_reg_tag (i32.const 1)))
  (if (i32.ne (local.get $t_tag) (i32.const 3))
    (then (return (i32.const 0))))
  (local.set $comp_addr (call $deref_reg_addr (i32.const 1)))
  (local.set $comp_payload (call $val_payload (local.get $comp_addr)))
  (local.set $arity
    (i32.wrap_i64 (i64.shr_u (local.get $comp_payload) (i64.const 32))))
  ;; A1 tag drives mode selection.
  (local.set $a1_tag (call $deref_reg_tag (i32.const 0)))
  (if (i32.eq (local.get $a1_tag) (i32.const 6))
    (then
      (local.set $cur_pc (call $get_pc))
      ;; Retry detection: cp_count>0 AND top CP''s next_pc matches our
      ;; current PC AND its retry_n slot is populated. If so, advance
      ;; N from the slot. Otherwise push a fresh CP and start at N=1.
      (local.set $retry_n (i32.const 0))
      (if (i32.gt_s (call $get_cp_count) (i32.const 0))
        (then
          (if (i32.and
                (i32.eq (call $cp_get_next_pc) (local.get $cur_pc))
                (i32.gt_s (call $cp_get_retry_n) (i32.const 0)))
            (then
              (local.set $retry_n (call $cp_get_retry_n))))))
      (if (i32.eqz (local.get $retry_n))
        (then
          ;; Fresh entry: push CP (always, so arity=1 still has a slot
          ;; to exhaust against on retry), write retry_n=1.
          (if (i32.lt_s (local.get $arity) (i32.const 1))
            (then (return (i32.const 0))))
          (call $push_choice_point (local.get $cur_pc))
          (call $cp_set_retry_n (i32.const 1))
          (local.set $n (i32.const 1)))
        (else
          ;; Retry: N = retry_n + 1. If > arity, pop CP and fail.
          (local.set $n (i32.add (local.get $retry_n) (i32.const 1)))
          (if (i32.gt_s (local.get $n) (local.get $arity))
            (then
              (call $pop_choice_point_no_restore)
              (return (i32.const 0))))
          (call $cp_set_retry_n (local.get $n))))
      ;; Bind A1 to N (trailed).
      (call $bind_reg_deref (i32.const 0) (i32.const 1)
        (i64.extend_i32_u (local.get $n)))
      ;; Bind or unify A3 with T[N].
      (local.set $arg_off
        (i32.add (local.get $comp_addr)
                 (i32.mul (local.get $n) (i32.const 12))))
      (local.set $arg_tag (call $val_tag (local.get $arg_off)))
      (local.set $arg_payload (call $val_payload (local.get $arg_off)))
      (local.set $a3_tag (call $deref_reg_tag (i32.const 2)))
      (if (i32.eq (local.get $a3_tag) (i32.const 6))
        (then
          (call $bind_reg_deref (i32.const 2) (local.get $arg_tag)
            (local.get $arg_payload))
          (return (i32.const 1))))
      (return (if (result i32)
        (i32.and
          (i32.eq (local.get $a3_tag) (local.get $arg_tag))
          (i64.eq (call $deref_reg_payload (i32.const 2)) (local.get $arg_payload)))
        (then (i32.const 1))
        (else (i32.const 0))))))
  ;; --- Deterministic mode: A1 is ground ---
  (local.set $n (i32.wrap_i64 (call $deref_reg_payload (i32.const 0))))
  (if (i32.lt_s (local.get $n) (i32.const 1))
    (then (return (i32.const 0))))
  (if (i32.gt_s (local.get $n) (local.get $arity))
    (then (return (i32.const 0))))
  (local.set $arg_off
    (i32.add (local.get $comp_addr)
             (i32.mul (local.get $n) (i32.const 12))))
  (local.set $arg_tag (call $val_tag (local.get $arg_off)))
  (local.set $arg_payload (call $val_payload (local.get $arg_off)))
  (local.set $a3_tag (call $deref_reg_tag (i32.const 2)))
  (if (i32.eq (local.get $a3_tag) (i32.const 6))
    (then
      (call $bind_reg_deref (i32.const 2) (local.get $arg_tag) (local.get $arg_payload))
      (return (i32.const 1))))
  (if (result i32)
    (i32.and
      (i32.eq (local.get $a3_tag) (local.get $arg_tag))
      (i64.eq (call $deref_reg_payload (i32.const 2)) (local.get $arg_payload)))
    (then (i32.const 1))
    (else (i32.const 0)))
)

;; --- functor/3: A1 = T, A2 = N, A3 = A ---
;; Two modes determined by A1''s tag:
;;   - Unbound A1: construct mode. Read N and A from A2/A3, allocate a
;;     fresh compound with $arity unbound args, bind A1 to the result.
;;   - Instantiated A1: read mode. Extract functor/arity and bind A2/A3.
;; Atomic T (atom/integer/float): N = T, A = 0.
;; List case (tag=4) not handled in v1 — returns fail. Follow-up.
(func $builtin_functor (result i32)
  (local $t_tag i32)
  (local $t_payload i64)
  (local $n_tag i32)
  (local $n_payload i64)
  (local $arity i32)
  (local $comp_addr i32)
  (local $header_payload i64)
  (local $i i32)
  (local.set $t_tag (call $deref_reg_tag (i32.const 0)))
  ;; --- Construct mode ---
  (if (i32.eq (local.get $t_tag) (i32.const 6))
    (then
      (local.set $n_tag (call $deref_reg_tag (i32.const 1)))
      (local.set $n_payload (call $deref_reg_payload (i32.const 1)))
      (local.set $arity
        (i32.wrap_i64 (call $deref_reg_payload (i32.const 2))))
      ;; Arity must be >= 0.
      (if (i32.lt_s (local.get $arity) (i32.const 0))
        (then (return (i32.const 0))))
      ;; Arity 0: bind T to N verbatim (atom or integer).
      (if (i32.eq (local.get $arity) (i32.const 0))
        (then
          (call $bind_reg_deref (i32.const 0) (local.get $n_tag) (local.get $n_payload))
          (return (i32.const 1))))
      ;; Arity > 0: N must be an atom to form a valid compound.
      (if (i32.ne (local.get $n_tag) (i32.const 0))
        (then (return (i32.const 0))))
      ;; header payload = (arity << 32) | (hash & 0xFFFFFFFF)
      (local.set $header_payload
        (i64.or
          (i64.shl (i64.extend_i32_u (local.get $arity)) (i64.const 32))
          (i64.and (local.get $n_payload) (i64.const 0xFFFFFFFF))))
      (local.set $comp_addr
        (call $heap_push_val (i32.const 3) (local.get $header_payload)))
      ;; Push $arity fresh unbound argument cells.
      (local.set $i (i32.const 0))
      (block $done_args
        (loop $arg_loop
          (br_if $done_args (i32.ge_s (local.get $i) (local.get $arity)))
          (drop (call $heap_push_val (i32.const 6) (i64.const 0)))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $arg_loop)))
      ;; Bind T = ref(comp_addr).
      (call $bind_reg_deref (i32.const 0) (i32.const 5)
        (i64.extend_i32_u (local.get $comp_addr)))
      (return (i32.const 1))))
  ;; --- Read mode: compound (tag=3 after deref) ---
  (if (i32.eq (local.get $t_tag) (i32.const 3))
    (then
      (local.set $comp_addr (call $deref_reg_addr (i32.const 0)))
      (local.set $header_payload (call $val_payload (local.get $comp_addr)))
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u (local.get $header_payload) (i64.const 32))))
      ;; Bind A2 = atom(functor_hash)
      (call $bind_reg_deref (i32.const 1) (i32.const 0)
        (i64.and (local.get $header_payload) (i64.const 0xFFFFFFFF)))
      ;; Bind A3 = integer(arity)
      (call $bind_reg_deref (i32.const 2) (i32.const 1)
        (i64.extend_i32_u (local.get $arity)))
      (return (i32.const 1))))
  ;; --- Read mode: atomic (atom/integer/float) ---
  ;; For atomic T: N = T (same tag + payload), A = 0
  (if (i32.le_u (local.get $t_tag) (i32.const 2))
    (then
      (local.set $t_payload (call $deref_reg_payload (i32.const 0)))
      (call $bind_reg_deref (i32.const 1) (local.get $t_tag) (local.get $t_payload))
      (call $bind_reg_deref (i32.const 2) (i32.const 1) (i64.const 0))
      (return (i32.const 1))))
  (i32.const 0)
)

;; --- =../2 (univ): A1 = T, A2 = L ---
;; Two modes determined by A1''s tag:
;;   - Instantiated A1 (atomic or compound ref): decompose. Build a
;;     fresh cons list [functor | args] on the heap and bind A2 to it.
;;   - Unbound A1: compose. Walk the cons list in A2, take the head as
;;     the functor and the tail elements as args, build a fresh
;;     compound, bind A1.
;; List representation: cons cells are tag=4 ''list'' cells with payload
;; zero, whose head is stored at offset +12 and tail at offset +24. This
;; matches what the canonical WAM''s put_list + set_constant/set_value
;; instructions produce at runtime (see wam_wat_case(put_list, ...)).
;; The empty list is the atom ''[]'' — tag=0 with payload 2914 (DJB2-like
;; acc*31+char applied to chars 91, 93).
;;
;; Decompose mode writes freshly-allocated cells to an unbound A2 (v1
;; limitation: if A2 is already bound to a list, this helper does NOT
;; structurally unify — it blindly rebinds). Compose mode requires A2
;; to deref to a cons cell and produces a fresh compound for an unbound
;; A1.
(func $builtin_univ (result i32)
  (local $t_tag i32)
  (local $t_payload i64)
  (local $comp_addr i32)
  (local $header_payload i64)
  (local $arity i32)
  (local $functor_hash i64)
  (local $list_root i32)
  (local $i i32)
  (local $cons_addr i32)
  (local $arg_off i32)
  (local $next_cons i32)
  (local $cur i32)
  (local $head_tag i32)
  (local $head_payload i64)
  (local $tail_tag i32)
  (local $tail_payload i64)
  (local $nelts i32)
  (local $list_start i32)
  (local.set $t_tag (call $deref_reg_tag (i32.const 0)))
  ;; --- Decompose: atomic T (atom/integer/float, tag <= 2) ---
  (if (i32.le_u (local.get $t_tag) (i32.const 2))
    (then
      (local.set $t_payload (call $deref_reg_payload (i32.const 0)))
      ;; Cons cell header (tag=4, payload 0) — head and tail follow.
      (local.set $list_root
        (call $heap_push_val (i32.const 4) (i64.const 0)))
      ;; head = T (copied tag+payload as-is)
      (drop (call $heap_push_val (local.get $t_tag) (local.get $t_payload)))
      ;; tail = atom([])
      (drop (call $heap_push_val (i32.const 0) (i64.const 2914)))
      (call $bind_reg_deref (i32.const 1) (i32.const 5)
        (i64.extend_i32_u (local.get $list_root)))
      (return (i32.const 1))))
  ;; --- Decompose: compound T (tag=3 after deref) ---
  (if (i32.eq (local.get $t_tag) (i32.const 3))
    (then
      (local.set $comp_addr (call $deref_reg_addr (i32.const 0)))
      (local.set $header_payload (call $val_payload (local.get $comp_addr)))
      (local.set $arity
        (i32.wrap_i64 (i64.shr_u (local.get $header_payload) (i64.const 32))))
      (local.set $functor_hash
        (i64.and (local.get $header_payload) (i64.const 0xFFFFFFFF)))
      ;; Allocate (arity+1) cons cells in heap order. Each cons cell
      ;; takes 36 bytes: list header at base+i*36, head at +12,
      ;; tail at +24.
      (local.set $list_root (call $get_heap_top))
      (local.set $i (i32.const 0))
      (block $done_cells
        (loop $cell_loop
          (br_if $done_cells
            (i32.gt_s (local.get $i) (local.get $arity)))
          ;; List header
          (local.set $cons_addr
            (call $heap_push_val (i32.const 4) (i64.const 0)))
          ;; Head
          (if (i32.eqz (local.get $i))
            (then
              ;; element 0 = functor atom
              (drop (call $heap_push_val (i32.const 0)
                      (local.get $functor_hash))))
            (else
              ;; element i (i>=1) = compound''s i-th arg cell
              (local.set $arg_off
                (i32.add (local.get $comp_addr)
                         (i32.mul (local.get $i) (i32.const 12))))
              (drop (call $heap_push_val
                      (call $val_tag (local.get $arg_off))
                      (call $val_payload (local.get $arg_off))))))
          ;; Tail
          (if (i32.eq (local.get $i) (local.get $arity))
            (then
              (drop (call $heap_push_val (i32.const 0) (i64.const 2914))))
            (else
              (local.set $next_cons
                (i32.add (local.get $cons_addr) (i32.const 36)))
              (drop (call $heap_push_val (i32.const 5)
                      (i64.extend_i32_u (local.get $next_cons))))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $cell_loop)))
      (call $bind_reg_deref (i32.const 1) (i32.const 5)
        (i64.extend_i32_u (local.get $list_root)))
      (return (i32.const 1))))
  ;; --- Compose: A1 unbound, A2 = cons-list. ---
  (if (i32.eq (local.get $t_tag) (i32.const 6))
    (then
      ;; A2 must be a cons cell (tag=4 after deref).
      (if (i32.ne (call $deref_reg_tag (i32.const 1)) (i32.const 4))
        (then (return (i32.const 0))))
      (local.set $list_start (call $deref_reg_addr (i32.const 1)))
      ;; Pass 1: count elements and validate cons-list structure.
      (local.set $cur (local.get $list_start))
      (local.set $nelts (i32.const 0))
      (block $count_done
        (loop $count_loop
          (if (i32.ne (call $val_tag (local.get $cur)) (i32.const 4))
            (then (return (i32.const 0))))
          (local.set $nelts (i32.add (local.get $nelts) (i32.const 1)))
          (local.set $tail_tag
            (call $val_tag (i32.add (local.get $cur) (i32.const 24))))
          (local.set $tail_payload
            (call $val_payload (i32.add (local.get $cur) (i32.const 24))))
          (if (i32.eq (local.get $tail_tag) (i32.const 0))
            (then
              ;; tail must be the atom []
              (if (i64.eq (local.get $tail_payload) (i64.const 2914))
                (then (br $count_done))
                (else (return (i32.const 0))))))
          (if (i32.eq (local.get $tail_tag) (i32.const 5))
            (then
              (local.set $cur (i32.wrap_i64 (local.get $tail_payload)))
              (br $count_loop)))
          ;; Any other tail tag is malformed for our cons representation.
          (return (i32.const 0))))
      (if (i32.eqz (local.get $nelts))
        (then (return (i32.const 0))))
      ;; Head of first cons must be the functor.
      (local.set $head_tag
        (call $val_tag (i32.add (local.get $list_start) (i32.const 12))))
      (local.set $head_payload
        (call $val_payload (i32.add (local.get $list_start) (i32.const 12))))
      ;; Single-element list: bind T to that element verbatim.
      (if (i32.eq (local.get $nelts) (i32.const 1))
        (then
          (call $bind_reg_deref (i32.const 0) (local.get $head_tag) (local.get $head_payload))
          (return (i32.const 1))))
      ;; Multi-element list: head must be an atom (the functor).
      (if (i32.ne (local.get $head_tag) (i32.const 0))
        (then (return (i32.const 0))))
      ;; arity = nelts - 1. Allocate compound header + arity arg cells.
      (local.set $arity (i32.sub (local.get $nelts) (i32.const 1)))
      (local.set $header_payload
        (i64.or
          (i64.shl (i64.extend_i32_u (local.get $arity)) (i64.const 32))
          (i64.and (local.get $head_payload) (i64.const 0xFFFFFFFF))))
      (local.set $comp_addr
        (call $heap_push_val (i32.const 3) (local.get $header_payload)))
      ;; Pass 2: walk tail of list, copy each element into a fresh arg cell.
      (local.set $cur
        (i32.wrap_i64
          (call $val_payload (i32.add (local.get $list_start) (i32.const 24)))))
      (local.set $i (i32.const 0))
      (block $copy_done
        (loop $copy_loop
          (br_if $copy_done (i32.ge_s (local.get $i) (local.get $arity)))
          ;; head of cons at $cur
          (local.set $head_tag
            (call $val_tag (i32.add (local.get $cur) (i32.const 12))))
          (local.set $head_payload
            (call $val_payload (i32.add (local.get $cur) (i32.const 12))))
          (drop (call $heap_push_val (local.get $head_tag) (local.get $head_payload)))
          ;; advance to next cons via tail ref
          (local.set $tail_tag
            (call $val_tag (i32.add (local.get $cur) (i32.const 24))))
          (if (i32.eq (local.get $tail_tag) (i32.const 5))
            (then
              (local.set $cur
                (i32.wrap_i64
                  (call $val_payload (i32.add (local.get $cur) (i32.const 24)))))))
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br $copy_loop)))
      (call $bind_reg_deref (i32.const 0) (i32.const 5)
        (i64.extend_i32_u (local.get $comp_addr)))
      (return (i32.const 1))))
  (i32.const 0)
)

;; --- copy_term/2: A1 = T, A2 = Copy ---
;; Builds a structurally identical copy of T in which every unbound
;; variable has been replaced by a fresh unbound. Sharing is preserved
;; within the copy: two positions that share a source variable map to
;; the SAME fresh variable in the copy (the spec''s critical property
;; for copy_term/2).
;;
;; Scope (deep):
;;   - Atomic T (atom/integer/float): bind Copy = T verbatim.
;;   - Unbound T: allocate a fresh unbound and bind Copy to a ref.
;;   - Ref to compound or list T: iterative deep copy via a work stack,
;;     recursing through nested compounds and lists at arbitrary depth
;;     without using WAT''s call stack. Sharing is preserved across the
;;     entire traversal, not just the top level.
;;
;; Algorithm (iterative, work-stack driven):
;;
;;   Scratch region (page 4, fixed offset 262144):
;;     262144..262911 (768 B) — var map: 64 entries x 12 bytes
;;                              [src_payload:i64][dst_addr:i32]
;;                              src_payload is the source unbound''s
;;                              variable id; dst_addr is the heap address
;;                              of the fresh cell materialized the first
;;                              time we saw that source var.
;;     262912..263935 (1024 B) — work stack: 128 entries x 8 bytes
;;                              [src_addr:i32][dst_addr:i32]
;;                              Each entry asks "copy the cell at src_addr
;;                              into the existing cell at dst_addr".
;;
;;   Top-level setup: allocate one placeholder unbound cell on the heap.
;;   Push (root_src_addr, placeholder) onto the work stack. The worklist
;;   loop will rewrite the placeholder in-place with a direct value
;;   (atomic) or a ref to a freshly-built nested structure.
;;
;;   Worklist loop (LIFO, pop until empty):
;;     atomic (tag <= 2)  — val_store tag/payload at dst
;;     unbound (tag = 6)  — lookup src_payload in var map
;;                            hit:  val_store ref(existing) at dst
;;                            miss: heap_push_val a fresh unbound cell;
;;                                  record (src_payload, new_addr) in map;
;;                                  val_store ref(new_addr) at dst
;;     ref (tag = 5)      — repush (src''s target, dst) — follow once
;;     compound (tag = 3) — read arity-packed header; heap_push_val a new
;;                          header; heap_push_val arity placeholder cells
;;                          immediately after; val_store ref(new_comp) at
;;                          dst; push (arg_src, new_comp + i*12) for each
;;                          source arg so the next iteration fills the
;;                          reserved slot.
;;     list (tag = 4)     — heap_push_val a list header + head + tail
;;                          placeholders (3 cells); val_store ref(new_list)
;;                          at dst; push (tail_src, new_list+24) then
;;                          (head_src, new_list+12).
;;
;;   Each iteration ends with br $work. The loop exits via br_if $done
;;   when the work stack empties. After exit, read the placeholder''s
;;   tag/payload and bind A2 to that exact pair.
;;
;; The scratch is FIXED-OFFSET in page 4, separate from the WAM heap.
;; This means copy_term does NOT permanently bloat heap_top with
;; bookkeeping — successive calls simply overwrite the same 1792-byte
;; scratch. copy_term is not re-entrant (no nested copy_term calls
;; inside itself), so the single scratch region is sufficient.
;;
;; Hard limits (v1): 64 distinct source variables per call; 128 pending
;; work items at any time. Overflow triggers fail (return 0).
(func $builtin_copy_term (result i32)
  (local $t_tag i32)
  (local $t_payload i64)
  (local $src_addr i32)
  (local $top_placeholder i32)
  (local $result_tag i32)
  (local $result_payload i64)
  (local $map_base i32)
  (local $stack_base i32)
  (local $stack_end i32)
  (local $map_n i32)
  (local $map_limit i32)
  (local $stack_top i32)
  (local $s i32)
  (local $d i32)
  (local $s_tag i32)
  (local $s_payload i64)
  (local $header_payload i64)
  (local $arity i32)
  (local $new_comp i32)
  (local $new_cell i32)
  (local $i i32)
  (local $j i32)
  (local $entry_off i32)
  (local $found i32)
  (local $found_off i32)
  (local.set $t_tag (call $deref_reg_tag (i32.const 0)))
  (local.set $t_payload (call $deref_reg_payload (i32.const 0)))
  ;; --- Atomic T: direct bind ---
  (if (i32.le_u (local.get $t_tag) (i32.const 2))
    (then
      (call $bind_reg_deref (i32.const 1) (local.get $t_tag) (local.get $t_payload))
      (return (i32.const 1))))
  ;; --- Unbound T: fresh unbound, bind ref ---
  (if (i32.eq (local.get $t_tag) (i32.const 6))
    (then
      (local.set $new_cell
        (call $heap_push_val (i32.const 6) (i64.const 0)))
      (call $bind_reg_deref (i32.const 1) (i32.const 5)
        (i64.extend_i32_u (local.get $new_cell)))
      (return (i32.const 1))))
  ;; --- Compound T (tag=3 after deref): worklist-driven deep copy ---
  (if (i32.eq (local.get $t_tag) (i32.const 3))
    (then
      (local.set $src_addr (call $deref_reg_addr (i32.const 0)))
      ;; Scratch base at page 4 start (262144). Layout:
      ;;   [262144..262911] var map (768 B, 64 entries x 12)
      ;;   [262912..263935] work stack (1024 B, 128 entries x 8)
      (local.set $map_base (i32.const 262144))
      (local.set $stack_base (i32.const 262912))
      (local.set $stack_end (i32.const 263936))
      (local.set $map_n (i32.const 0))
      (local.set $map_limit (i32.const 64))
      (local.set $stack_top (local.get $stack_base))
      ;; Top-level placeholder on the heap — the worklist rewrites
      ;; this cell in-place with the root of the copy.
      (local.set $top_placeholder
        (call $heap_push_val (i32.const 6) (i64.const 0)))
      ;; Push initial work item (src, top_placeholder).
      (i32.store (local.get $stack_top) (local.get $src_addr))
      (i32.store (i32.add (local.get $stack_top) (i32.const 4))
                 (local.get $top_placeholder))
      (local.set $stack_top (i32.add (local.get $stack_top) (i32.const 8)))
      ;; Main worklist loop.
      (block $done
        (loop $work
          ;; Empty? done.
          (br_if $done
            (i32.le_u (local.get $stack_top) (local.get $stack_base)))
          ;; Pop.
          (local.set $stack_top
            (i32.sub (local.get $stack_top) (i32.const 8)))
          (local.set $s (i32.load (local.get $stack_top)))
          (local.set $d
            (i32.load (i32.add (local.get $stack_top) (i32.const 4))))
          (local.set $s_tag (call $val_tag (local.get $s)))
          (local.set $s_payload (call $val_payload (local.get $s)))
          ;; --- Atomic: write direct value ---
          (if (i32.le_u (local.get $s_tag) (i32.const 2))
            (then
              (call $val_store (local.get $d) (local.get $s_tag)
                                (local.get $s_payload))
              (br $work)))
          ;; --- Unbound: var-map lookup, materialize or share ---
          (if (i32.eq (local.get $s_tag) (i32.const 6))
            (then
              (local.set $found (i32.const 0))
              (local.set $found_off (i32.const 0))
              (local.set $j (i32.const 0))
              (block $search_done
                (loop $search
                  (br_if $search_done
                    (i32.ge_u (local.get $j) (local.get $map_n)))
                  (local.set $entry_off
                    (i32.add (local.get $map_base)
                             (i32.mul (local.get $j) (i32.const 12))))
                  (if (i64.eq (i64.load (local.get $entry_off))
                              (local.get $s_payload))
                    (then
                      (local.set $found_off
                        (i32.load
                          (i32.add (local.get $entry_off) (i32.const 8))))
                      (local.set $found (i32.const 1))
                      (br $search_done)))
                  (local.set $j (i32.add (local.get $j) (i32.const 1)))
                  (br $search)))
              (if (local.get $found)
                (then
                  ;; Shared: write ref to existing fresh cell.
                  (call $val_store (local.get $d) (i32.const 5)
                    (i64.extend_i32_u (local.get $found_off))))
                (else
                  ;; First occurrence: allocate fresh unbound, record.
                  (local.set $new_cell
                    (call $heap_push_val (i32.const 6) (i64.const 0)))
                  (if (i32.lt_u (local.get $map_n) (local.get $map_limit))
                    (then
                      (local.set $entry_off
                        (i32.add (local.get $map_base)
                                 (i32.mul (local.get $map_n) (i32.const 12))))
                      (i64.store (local.get $entry_off)
                                 (local.get $s_payload))
                      (i32.store
                        (i32.add (local.get $entry_off) (i32.const 8))
                        (local.get $new_cell))
                      (local.set $map_n
                        (i32.add (local.get $map_n) (i32.const 1))))
                    (else
                      ;; Map full — fail.
                      (return (i32.const 0))))
                  (call $val_store (local.get $d) (i32.const 5)
                    (i64.extend_i32_u (local.get $new_cell)))))
              (br $work)))
          ;; --- Ref: follow one level by re-pushing ---
          (if (i32.eq (local.get $s_tag) (i32.const 5))
            (then
              (if (i32.ge_u
                    (i32.add (local.get $stack_top) (i32.const 8))
                    (local.get $stack_end))
                (then (return (i32.const 0))))
              (i32.store (local.get $stack_top)
                         (i32.wrap_i64 (local.get $s_payload)))
              (i32.store
                (i32.add (local.get $stack_top) (i32.const 4))
                (local.get $d))
              (local.set $stack_top
                (i32.add (local.get $stack_top) (i32.const 8)))
              (br $work)))
          ;; --- Compound: allocate header + placeholder args, push work ---
          (if (i32.eq (local.get $s_tag) (i32.const 3))
            (then
              (local.set $header_payload (local.get $s_payload))
              (local.set $arity
                (i32.wrap_i64
                  (i64.shr_u (local.get $header_payload) (i64.const 32))))
              (local.set $new_comp
                (call $heap_push_val (i32.const 3)
                                     (local.get $header_payload)))
              ;; Reserve $arity placeholder arg slots.
              (local.set $i (i32.const 0))
              (block $rsv_done
                (loop $reserve
                  (br_if $rsv_done (i32.ge_s (local.get $i) (local.get $arity)))
                  (drop (call $heap_push_val (i32.const 6) (i64.const 0)))
                  (local.set $i (i32.add (local.get $i) (i32.const 1)))
                  (br $reserve)))
              ;; d := ref(new_comp)
              (call $val_store (local.get $d) (i32.const 5)
                (i64.extend_i32_u (local.get $new_comp)))
              ;; Push (arg_src, arg_dst) pairs for i=1..arity.
              (local.set $i (i32.const 1))
              (block $push_done
                (loop $push_args
                  (br_if $push_done
                    (i32.gt_s (local.get $i) (local.get $arity)))
                  (if (i32.ge_u
                        (i32.add (local.get $stack_top) (i32.const 8))
                        (local.get $stack_end))
                    (then (return (i32.const 0))))
                  (i32.store (local.get $stack_top)
                    (i32.add (local.get $s)
                             (i32.mul (local.get $i) (i32.const 12))))
                  (i32.store
                    (i32.add (local.get $stack_top) (i32.const 4))
                    (i32.add (local.get $new_comp)
                             (i32.mul (local.get $i) (i32.const 12))))
                  (local.set $stack_top
                    (i32.add (local.get $stack_top) (i32.const 8)))
                  (local.set $i (i32.add (local.get $i) (i32.const 1)))
                  (br $push_args)))
              (br $work)))
          ;; --- List: header + head + tail placeholders, push 2 work items ---
          (if (i32.eq (local.get $s_tag) (i32.const 4))
            (then
              (local.set $new_comp
                (call $heap_push_val (i32.const 4) (i64.const 0)))
              (drop (call $heap_push_val (i32.const 6) (i64.const 0)))
              (drop (call $heap_push_val (i32.const 6) (i64.const 0)))
              (call $val_store (local.get $d) (i32.const 5)
                (i64.extend_i32_u (local.get $new_comp)))
              ;; Capacity: need 2 slots.
              (if (i32.ge_u
                    (i32.add (local.get $stack_top) (i32.const 16))
                    (local.get $stack_end))
                (then (return (i32.const 0))))
              ;; Push tail first (LIFO pops it second).
              (i32.store (local.get $stack_top)
                (i32.add (local.get $s) (i32.const 24)))
              (i32.store
                (i32.add (local.get $stack_top) (i32.const 4))
                (i32.add (local.get $new_comp) (i32.const 24)))
              (local.set $stack_top
                (i32.add (local.get $stack_top) (i32.const 8)))
              ;; Push head.
              (i32.store (local.get $stack_top)
                (i32.add (local.get $s) (i32.const 12)))
              (i32.store
                (i32.add (local.get $stack_top) (i32.const 4))
                (i32.add (local.get $new_comp) (i32.const 12)))
              (local.set $stack_top
                (i32.add (local.get $stack_top) (i32.const 8)))
              (br $work)))
          ;; Unknown tag — fail.
          (return (i32.const 0))))
      ;; Worklist done. Read the placeholder''s final tag/payload and
      ;; bind A2 to that value.
      (local.set $result_tag (call $val_tag (local.get $top_placeholder)))
      (local.set $result_payload
        (call $val_payload (local.get $top_placeholder)))
      (call $bind_reg_deref (i32.const 1) (local.get $result_tag)
                                    (local.get $result_payload))
      (return (i32.const 1))))
  (i32.const 0)
)
', [CPSize, CPSaveRegs, CPSize, CPSize, CPSaveRegs, CPSaveRegs, CPSaveRegs]).

%% compile_wam_runtime_to_wat(+Options, -WatCode)
compile_wam_runtime_to_wat(Options, WatCode) :-
    compile_step_wam_to_wat(Options, StepCode),
    compile_wam_helpers_to_wat(Options, HelpersCode),
    format(atom(WatCode), '~w\n\n~w', [StepCode, HelpersCode]).

% ============================================================================
% PHASE 3: Project assembly
% ============================================================================

%% write_wam_wat_project(+Predicates, +Options, +OutputFile)
write_wam_wat_project(Predicates, Options, OutputFile) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

    %% Read and render value template
    read_template_file('templates/targets/wat_wam/value.wat.mustache', ValueTemplate),
    render_template(ValueTemplate, [], ValueCode),

    %% Read and render state template
    read_template_file('templates/targets/wat_wam/state.wat.mustache', StateTemplate),
    render_template(StateTemplate, [], StateCode),

    %% Generate runtime (step + helpers)
    compile_step_wam_to_wat(Options, StepBody),
    compile_wam_helpers_to_wat(Options, HelpersCode),
    read_template_file('templates/targets/wat_wam/runtime.wat.mustache', RuntimeTemplate),
    render_template(RuntimeTemplate, [
        step_body=StepBody,
        helper_functions=HelpersCode
    ], RuntimeCode),

    %% Compile predicates
    wam_heap_base(HeapBase),
    compile_wat_predicates(Predicates, Options, HeapBase, DataSegs, PredFuncs, Exports),

    %% Memory pages: 0=native WAT strings/memo; 1=WAM state (regs, trail,
    %% env/choice stack); 2=code (instruction data segments); 3=WAM heap;
    %% 4=copy_term scratch (var map + work stack, fixed offsets at 262144);
    %% 5=atom name table (count + fixed-size entries + variable-length
    %% string pool, fixed offset at 327680). The atom table is built at
    %% compile time from every atom reached by encode_*; the runtime
    %% $atom_compare uses it so @</2 on atoms orders lexicographically
    %% rather than by DJB2 hash.
    MemPages = 6,

    %% Assemble module
    read_template_file('templates/targets/wat_wam/module.wat.mustache', ModuleTemplate),
    render_template(ModuleTemplate, [
        date=Date,
        module_name=ModuleName,
        memory_pages=MemPages,
        data_segments=DataSegs,
        value_functions=ValueCode,
        state_functions=StateCode,
        runtime_functions=RuntimeCode,
        native_predicates='',
        wam_predicates=PredFuncs,
        exports=Exports
    ], FullModule),

    %% Write output file
    write_file(OutputFile, FullModule),
    format('WAM-WAT module written to: ~w~n', [OutputFile]).

%% compile_wat_predicates(+Preds, +Opts, +CodeBase, -DataSegs, -Funcs, -Exports)
%
%  Two-pass project-level compilation:
%
%    Pass 1: parse each predicate's WAM text into instructions + local
%            labels, compute each predicate's cumulative start PC.
%    Build:  global label table = union of every predicate's labels
%            with their local PCs shifted by the predicate's start PC.
%            Each predicate's entry label (e.g. 'sum_ints/3') lives at
%            its start PC, so cross-predicate call/execute instructions
%            resolve correctly.
%    Pass 2: re-encode each predicate's instructions using the global
%            label table so internal labels (try_me_else targets) and
%            external references (call sibling/N) both resolve to
%            absolute PCs within a single merged instruction array.
%    Assemble: concatenate all encoded bytes into ONE data segment at
%            CodeBase. Entry functions share the same CodeBase and
%            total instruction count, but each sets PC to its own
%            start offset before calling $run_loop.
%
%  Prior to this change, each predicate had its own per-predicate
%  label table and emitted its own data segment at a distinct base.
%  Cross-predicate Execute / Call instructions encoded the target
%  with local-label resolution, which defaults to PC=0 on miss —
%  the VM would then jump to PC=0 of the CALLING predicate's code,
%  silently re-entering the caller and creating an infinite loop
%  (or immediate failure if the re-entry path failed a guard). This
%  fix unifies the label namespace across all predicates in a
%  project, making multi-predicate WAM programs runnable end-to-end.
compile_wat_predicates(Predicates, Options, CodeBase, DataSegs, Funcs, Exports) :-
    %% Reset atom accumulator before compiling this project.
    retractall(seen_atom(_)),
    %% The empty-list atom [] is used by is_list/1 and list cons
    %% cells; seed it so it always has an entry in the atom table
    %% regardless of source-code occurrence.
    record_atom('[]'),
    %% Pass 1: parse + collect per-predicate data with cumulative PCs.
    pass1_parse_predicates(Predicates, Options, 0, PredData, TotalInstrs),
    %% Build global label table (shift each predicate's local labels).
    build_global_labels(PredData, [], GlobalLabels),
    %% Pass 2: re-encode each predicate's instructions against global
    %% labels and concatenate the byte sequences. Encoding paths also
    %% call record_atom/1, populating seen_atom/1 as a side effect.
    encode_all_predicates(PredData, GlobalLabels, AllHex),
    (   AllHex == ''
    ->  CodeSeg = ''
    ;   format(atom(CodeSeg),
            '(data (i32.const ~w) "~w")', [CodeBase, AllHex])
    ),
    %% Emit the atom name table as a second data segment.
    assemble_atom_table(AtomTableSeg),
    (   AtomTableSeg == ''
    ->  DataSegs = CodeSeg
    ;   atomic_list_concat([CodeSeg, '\n  ', AtomTableSeg], DataSegs)
    ),
    %% Entry functions: one per predicate, each setting its own
    %% start PC before invoking the shared run loop.
    option(wam_heap_start(HeapStart), Options, 196608),
    gen_all_entry_funcs(PredData, HeapStart, CodeBase, TotalInstrs,
                        Funcs, Exports).

%% wam_atom_table_base(-Offset)
%  Fixed start address of the atom name table. Lives on page 5
%  (offset 327680) — past copy_term scratch at page 4 (262144) and
%  the WAM heap which bump-allocates from page 3 (196608).
%  One memory page holds up to 4096 16-byte entries plus a string
%  pool; the current module bumps memory_pages from 5 to 6 to
%  cover this region.
wam_atom_table_base(327680).

%% assemble_atom_table(-DataSegHex)
%  After encoding has recorded every atom reached via encode_*
%  helpers, emit a data segment at wam_atom_table_base containing:
%    [count:i32][entries × 16 bytes][string_pool]
%  Each entry: (hash:i64, name_offset:i32, name_length:i32). Sorted
%  by hash so the runtime can binary-search (linear scan works too
%  for small counts). name_offset is absolute within linear memory
%  and points into the string_pool region immediately following the
%  entries array.
assemble_atom_table(Seg) :-
    findall(A, seen_atom(A), AtomsRaw),
    sort(AtomsRaw, Atoms),
    (   Atoms == []
    ->  Seg = ''
    ;   wam_atom_table_base(Base),
        length(Atoms, Count),
        StringBase is Base + 4 + Count * 16,
        build_atom_entries(Atoms, StringBase, HashOffsetPairs, PoolBytesList),
        %% Sort entries by hash for binary-searchable table.
        sort(1, @=<, HashOffsetPairs, SortedPairs),
        i32_to_le_hex(Count, CountHex),
        maplist(atom_entry_to_hex, SortedPairs, EntryHexes),
        atomic_list_concat(PoolBytesList, PoolHex),
        atomic_list_concat([CountHex | EntryHexes], EntriesHex),
        atomic_list_concat([EntriesHex, PoolHex], AllHex),
        format(atom(Seg),
            '(data (i32.const ~w) "~w")', [Base, AllHex])
    ).

%% build_atom_entries(+Atoms, +StringBase, -HashOffsetPairs, -PoolHexParts)
%  For each atom assign its name a region in the string pool; build
%  parallel lists of (Hash, Offset, Length) tuples and hex bytes.
build_atom_entries([], _, [], []).
build_atom_entries([A|Rest], Off0, [entry(Hash, Off0, Len, A)|RestPairs],
                   [NameHex|RestPool]) :-
    atom_hash_i64(A, Hash),
    atom_codes(A, Codes),
    length(Codes, Len),
    maplist(code_to_byte_hex, Codes, CodeHexes),
    atomic_list_concat(CodeHexes, NameHex),
    Off1 is Off0 + Len,
    build_atom_entries(Rest, Off1, RestPairs, RestPool).

%% atom_entry_to_hex(+entry(Hash, Off, Len, Name), -Hex)
%  Encode one 16-byte entry: hash (i64), offset (i32), length (i32).
atom_entry_to_hex(entry(Hash, Off, Len, _A), Hex) :-
    i64_to_le_hex(Hash, HashHex),
    i32_to_le_hex(Off, OffHex),
    i32_to_le_hex(Len, LenHex),
    atomic_list_concat([HashHex, OffHex, LenHex], Hex).

code_to_byte_hex(C, Hex) :-
    B is C /\ 0xFF,
    format(atom(Hex), "\\~|~`0t~16r~2+", [B]).

%% pass1_parse_predicates(+Preds, +Opts, +StartPC, -PredData, -TotalInstrs)
%
%  PredData is a list of pred_data(Pred/Arity, Instrs, LocalLabels,
%  StartPC, NumInstrs) entries in input order, where StartPC is the
%  cumulative instruction index at which this predicate's first
%  instruction lives in the merged instruction array.
pass1_parse_predicates([], _, _, [], 0).
pass1_parse_predicates([PredInd|Rest], Options, StartPC,
                       [pred_data(Pred/Arity, Instrs, LocalLabels,
                                  StartPC, NumInstrs) | RestData],
                       TotalInstrs) :-
    (   PredInd = _Module:Pred/Arity -> true
    ;   PredInd = Pred/Arity
    ),
    (   catch(
            wam_target:compile_predicate_to_wam(PredInd, Options, WamCode),
            _, fail)
    ->  atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        %% Parse with label markers interspersed in the instruction list
        wam_lines_to_instrs_with_labels(Lines, 0, InstrsWithLabels),
        %% Run peephole optimizer (may remove instructions, changing PCs)
        peephole_neck_cut(InstrsWithLabels, NeckCutOptimized),
        peephole_nested_arith(NeckCutOptimized, NestedArithOptimized),
        peephole_fused_arith(NestedArithOptimized, ArithOptimized),
        peephole_direct_builtins(ArithOptimized, DirectBuiltinsOptimized),
        peephole_arg_to_a1(DirectBuiltinsOptimized, ArgToA1Optimized),
        peephole_arg_call_k(ArgToA1Optimized, ArgCallKOptimized),
        peephole_tail_call_k(ArgCallKOptimized, TailCallKOptimized),
        peephole_type_dispatch(TailCallKOptimized, OptimizedWithLabels),
        %% Extract real instructions and recompute label PCs from the
        %% optimized list. Label markers (label(Name)) are stripped;
        %% their position becomes the label PC in the local table.
        extract_instrs_and_labels(OptimizedWithLabels, 0, Instrs, LocalLabels),
        length(Instrs, NumInstrs),
        format(user_error,
               '  ~w/~w: WAM compilation (~w instructions, start PC ~w)~n',
               [Pred, Arity, NumInstrs, StartPC])
    ;   Instrs = [], LocalLabels = [], NumInstrs = 0,
        format(user_error, '  ~w/~w: compilation failed~n', [Pred, Arity])
    ),
    NextPC is StartPC + NumInstrs,
    pass1_parse_predicates(Rest, Options, NextPC, RestData, RestTotal),
    TotalInstrs is NumInstrs + RestTotal.

%% build_global_labels(+PredData, +Acc, -GlobalLabels)
%  For each predicate, shift its local labels by its start PC and
%  accumulate into a single project-wide LabelName-PC list. Labels
%  are expected to be globally unique (internal WAM labels already
%  encode the predicate name; entry labels are the Pred/Arity form).
build_global_labels([], Acc, Acc).
build_global_labels([pred_data(_, _, LocalLabels, StartPC, _)|Rest],
                    Acc, GlobalLabels) :-
    shift_labels(LocalLabels, StartPC, Shifted),
    append(Shifted, Acc, NewAcc),
    build_global_labels(Rest, NewAcc, GlobalLabels).

shift_labels([], _, []).
shift_labels([Name-LocalPC|Rest], Shift, [Name-GlobalPC|RestShifted]) :-
    GlobalPC is LocalPC + Shift,
    shift_labels(Rest, Shift, RestShifted).

%% encode_all_predicates(+PredData, +GlobalLabels, -AllHex)
%  Re-encode every predicate's instructions against the merged label
%  table, concatenating the resulting hex byte strings in predicate
%  order. Before encoding, each predicate's instruction list is run
%  through a peephole optimizer that converts cut-deterministic
%  2-clause patterns into neck_cut_test instructions (no CP needed).
encode_all_predicates([], _, '').
encode_all_predicates([pred_data(_, Instrs, _, _, _)|Rest],
                      GlobalLabels, AllHex) :-
    maplist(encode_instr_with_labels(GlobalLabels), Instrs, HexParts),
    atomic_list_concat(HexParts, PredHex),
    encode_all_predicates(Rest, GlobalLabels, RestHex),
    atomic_list_concat([PredHex, RestHex], AllHex).

%% peephole_neck_cut(+Instrs, -Optimized)
%  Detects 2-clause cut-deterministic patterns and replaces
%  try_me_else + guard + cut with a single neck_cut_test instruction
%  that does an inline guard check + conditional jump, no CP.
%
%  Pattern: try_me_else(L), allocate, ...get/put...,
%           builtin_call(Guard, N), builtin_call('!/0', 0), ...body1...,
%           L:trust_me, allocate, ...body2...
%
%  Output:  allocate, ...get/put..., neck_cut_test(Guard, N, L),
%           ...body1 (without cut)...,
%           L:allocate (trust_me removed), ...body2...
%% peephole_fused_arith(+Instrs, -Optimized)
%  Detects `Dest is Src1 OP Src2` for OP in {+, -, *} with all three
%  being registers and rewrites to a single fused_is_<op> instruction.
%  The canonical WAM layer emits:
%    put_value(Dest, A1), put_structure('OP/2', A2),
%    set_value(Src1), set_value(Src2), [deallocate,] builtin_call(is/2, 2)
%  The put_value puts the output var slot in A1; put_structure +
%  set_value × 2 build a freshly-allocated OP/2 compound in A2; is/2
%  evaluates A2 and binds A1. The fused form skips the compound alloc
%  and the $eval_arith recursion, writing the result directly to Dest.
%  Matches both with and without the intervening deallocate; preserves
%  the deallocate if present so stack semantics don't change.
fused_arith_op('+/2', fused_is_add).
fused_arith_op('-/2', fused_is_sub).
fused_arith_op('*/2', fused_is_mul).

%% Const-variant names: +/2 with a constant operand fuses to
%% fused_is_add_const, */2 fuses to fused_is_mul_const. -/2 doesn't
%% appear with const-on-right in practice — WAM normalizes `N - K`
%% to `N + (-K)` using +/2, so the -/2 const variant would be dead.
fused_arith_const_op('+/2', fused_is_add_const).
fused_arith_const_op('*/2', fused_is_mul_const).

peephole_fused_arith([], []).
%% Reg-Reg variants (with and without deallocate).
peephole_fused_arith([put_value(Dest, 'A1'),
                      put_structure(OpFunctor, 'A2'),
                      set_value(Src1),
                      set_value(Src2),
                      deallocate,
                      builtin_call('is/2', 2) | Rest],
                     [FusedInstr, deallocate | Out]) :-
    fused_arith_op(OpFunctor, FusedName),
    !,
    FusedInstr =.. [FusedName, Dest, Src1, Src2],
    peephole_fused_arith(Rest, Out).
peephole_fused_arith([put_value(Dest, 'A1'),
                      put_structure(OpFunctor, 'A2'),
                      set_value(Src1),
                      set_value(Src2),
                      builtin_call('is/2', 2) | Rest],
                     [FusedInstr | Out]) :-
    fused_arith_op(OpFunctor, FusedName),
    !,
    FusedInstr =.. [FusedName, Dest, Src1, Src2],
    peephole_fused_arith(Rest, Out).
%% Reg-Const and Const-Reg variants. For +/2 and */2 the operation
%% is commutative, so either operand-order fuses to the same
%% _const instruction.
peephole_fused_arith([put_value(Dest, 'A1'),
                      put_structure(OpFunctor, 'A2'),
                      set_value(Src),
                      set_constant(K),
                      deallocate,
                      builtin_call('is/2', 2) | Rest],
                     [FusedInstr, deallocate | Out]) :-
    fused_arith_const_op(OpFunctor, FusedName),
    integer(K),
    !,
    FusedInstr =.. [FusedName, Dest, Src, K],
    peephole_fused_arith(Rest, Out).
peephole_fused_arith([put_value(Dest, 'A1'),
                      put_structure(OpFunctor, 'A2'),
                      set_value(Src),
                      set_constant(K),
                      builtin_call('is/2', 2) | Rest],
                     [FusedInstr | Out]) :-
    fused_arith_const_op(OpFunctor, FusedName),
    integer(K),
    !,
    FusedInstr =.. [FusedName, Dest, Src, K],
    peephole_fused_arith(Rest, Out).
peephole_fused_arith([put_value(Dest, 'A1'),
                      put_structure(OpFunctor, 'A2'),
                      set_constant(K),
                      set_value(Src),
                      deallocate,
                      builtin_call('is/2', 2) | Rest],
                     [FusedInstr, deallocate | Out]) :-
    fused_arith_const_op(OpFunctor, FusedName),
    integer(K),
    !,
    FusedInstr =.. [FusedName, Dest, Src, K],
    peephole_fused_arith(Rest, Out).
peephole_fused_arith([put_value(Dest, 'A1'),
                      put_structure(OpFunctor, 'A2'),
                      set_constant(K),
                      set_value(Src),
                      builtin_call('is/2', 2) | Rest],
                     [FusedInstr | Out]) :-
    fused_arith_const_op(OpFunctor, FusedName),
    integer(K),
    !,
    FusedInstr =.. [FusedName, Dest, Src, K],
    peephole_fused_arith(Rest, Out).
peephole_fused_arith([H|T], [H|Out]) :-
    peephole_fused_arith(T, Out).

%% peephole_nested_arith(+Instrs, -Optimized)
%  Handles one level of arithmetic nesting (3-operand expressions
%  like `X is A + B + C` or `X is (A - B) * C`). Deeper nesting is
%  left to future work.
%
%  Left-associative pattern (WAM default for `A OP1 B OP2 C`):
%
%    put_value Dest, A1
%    put_structure OuterFn, A2
%    set_variable Tmp           -- outer arg 1 is the inner compound
%    put_structure InnerFn, Tmp
%    <set_value|set_constant>    -- inner arg 1
%    <set_value|set_constant>    -- inner arg 2
%    <set_value|set_constant>    -- outer arg 2
%    [deallocate]
%    builtin_call is/2, 2
%
%  Right-associative pattern (WAM for `A OP1 (B OP2 C)`):
%
%    put_value Dest, A1
%    put_structure OuterFn, A2
%    <set_value|set_constant>    -- outer arg 1
%    set_variable Tmp            -- outer arg 2 is the inner compound
%    put_structure InnerFn, Tmp
%    <set_value|set_constant>    -- inner arg 1
%    <set_value|set_constant>    -- inner arg 2
%    [deallocate]
%    builtin_call is/2, 2
%
%  Both rewrite to two fused_is_* instructions via the scratch reg
%  Tmp, which after the first fused_is_* holds an integer
%  (integer-tagged via $bind_reg_deref) — exactly what the second
%  fused_is_* needs as an operand. No further uses of Tmp are
%  possible (the WAM compiler allocates Tmp fresh per `is/2` site).

peephole_nested_arith([], []).
%% Left-assoc with optional deallocate.
peephole_nested_arith([put_value(Dest, 'A1'),
                       put_structure(OuterFn, 'A2'),
                       set_variable(Tmp),
                       put_structure(InnerFn, Tmp),
                       InnerArg1, InnerArg2,
                       OuterArg2,
                       deallocate,
                       builtin_call('is/2', 2) | Rest],
                      Out) :-
    build_nested_left(OuterFn, InnerFn,
                      Dest, Tmp,
                      InnerArg1, InnerArg2, OuterArg2,
                      [deallocate], Fused),
    !,
    append(Fused, Tail, Out),
    peephole_nested_arith(Rest, Tail).
peephole_nested_arith([put_value(Dest, 'A1'),
                       put_structure(OuterFn, 'A2'),
                       set_variable(Tmp),
                       put_structure(InnerFn, Tmp),
                       InnerArg1, InnerArg2,
                       OuterArg2,
                       builtin_call('is/2', 2) | Rest],
                      Out) :-
    build_nested_left(OuterFn, InnerFn,
                      Dest, Tmp,
                      InnerArg1, InnerArg2, OuterArg2,
                      [], Fused),
    !,
    append(Fused, Tail, Out),
    peephole_nested_arith(Rest, Tail).
%% Right-assoc with optional deallocate.
peephole_nested_arith([put_value(Dest, 'A1'),
                       put_structure(OuterFn, 'A2'),
                       OuterArg1,
                       set_variable(Tmp),
                       put_structure(InnerFn, Tmp),
                       InnerArg1, InnerArg2,
                       deallocate,
                       builtin_call('is/2', 2) | Rest],
                      Out) :-
    build_nested_right(OuterFn, InnerFn,
                       Dest, Tmp,
                       OuterArg1, InnerArg1, InnerArg2,
                       [deallocate], Fused),
    !,
    append(Fused, Tail, Out),
    peephole_nested_arith(Rest, Tail).
peephole_nested_arith([put_value(Dest, 'A1'),
                       put_structure(OuterFn, 'A2'),
                       OuterArg1,
                       set_variable(Tmp),
                       put_structure(InnerFn, Tmp),
                       InnerArg1, InnerArg2,
                       builtin_call('is/2', 2) | Rest],
                      Out) :-
    build_nested_right(OuterFn, InnerFn,
                       Dest, Tmp,
                       OuterArg1, InnerArg1, InnerArg2,
                       [], Fused),
    !,
    append(Fused, Tail, Out),
    peephole_nested_arith(Rest, Tail).
peephole_nested_arith([H|T], [H|Out]) :-
    peephole_nested_arith(T, Out).

%% build_nested_left(+OuterFn, +InnerFn, +Dest, +Tmp,
%%                   +InnerArg1, +InnerArg2, +OuterArg2,
%%                   +Tail, -Instrs)
%% Builds [InnerFused, OuterFused | Tail] for left-associative case.
%% Inner arg2 and outer arg2 may each be set_value(R) or
%% set_constant(K). Fails if the operator+arg-shape combination
%% isn't supported (falls through to the general is/2 path).
build_nested_left(OuterFn, InnerFn, Dest, Tmp,
                  InnerArg1, InnerArg2, OuterArg2,
                  Tail, Instrs) :-
    fused_reg_reg(InnerFn, InnerArg1, InnerArg2, Tmp, InnerInstr),
    fused_with_outer2(OuterFn, Tmp, OuterArg2, Dest, OuterInstr),
    Instrs = [InnerInstr, OuterInstr | Tail].

build_nested_right(OuterFn, InnerFn, Dest, Tmp,
                   OuterArg1, InnerArg1, InnerArg2,
                   Tail, Instrs) :-
    fused_reg_reg(InnerFn, InnerArg1, InnerArg2, Tmp, InnerInstr),
    fused_with_outer1(OuterFn, OuterArg1, Tmp, Dest, OuterInstr),
    Instrs = [InnerInstr, OuterInstr | Tail].

%% fused_reg_reg(+Fn, +Arg1, +Arg2, +Dest, -Instr)
%% Arg1 and Arg2 are set_value(R) or set_constant(K). Builds the
%% appropriate fused_is_<op>[_const] instruction. Only the forms
%% the downstream fused instructions can represent are supported.
fused_reg_reg(Fn, set_value(R1), set_value(R2), Dest, Instr) :-
    fused_arith_op(Fn, Name),
    Instr =.. [Name, Dest, R1, R2].
fused_reg_reg(Fn, set_value(R), set_constant(K), Dest, Instr) :-
    integer(K),
    fused_arith_const_op(Fn, Name),
    Instr =.. [Name, Dest, R, K].
fused_reg_reg(Fn, set_constant(K), set_value(R), Dest, Instr) :-
    integer(K),
    fused_arith_const_op(Fn, Name),
    Instr =.. [Name, Dest, R, K].

%% fused_with_outer2(+Fn, +Tmp, +OuterArg2, +Dest, -Instr)
%% Outer slot 1 is the Tmp (reg), outer slot 2 is set_value/const.
fused_with_outer2(Fn, Tmp, set_value(R), Dest, Instr) :-
    fused_arith_op(Fn, Name),
    Instr =.. [Name, Dest, Tmp, R].
fused_with_outer2(Fn, Tmp, set_constant(K), Dest, Instr) :-
    integer(K),
    fused_arith_const_op(Fn, Name),
    Instr =.. [Name, Dest, Tmp, K].

%% fused_with_outer1(+Fn, +OuterArg1, +Tmp, +Dest, -Instr)
%% Outer slot 1 is set_value/const, outer slot 2 is the Tmp (reg).
fused_with_outer1(Fn, set_value(R), Tmp, Dest, Instr) :-
    fused_arith_op(Fn, Name),
    Instr =.. [Name, Dest, R, Tmp].
fused_with_outer1(Fn, set_constant(K), Tmp, Dest, Instr) :-
    integer(K),
    fused_arith_const_op(Fn, Name),
    Instr =.. [Name, Dest, Tmp, K].

%% peephole_direct_builtins(+Instrs, -Optimized)
%  Rewrites `builtin_call(arg/3, 3)` → arg_direct and
%  `builtin_call(functor/3, 3)` → functor_direct. Skips one br_table
%  dispatch and one function-call boundary per call; meaningful on
%  term-walking hot loops (bench_sum_*, bench_term_depth).
peephole_direct_builtins([], []).
%% arg/3 specialization — 4-instr pattern with register N.
peephole_direct_builtins([put_value(NReg, 'A1'),
                          put_value(TReg, 'A2'),
                          put_variable(Dest, 'A3'),
                          builtin_call('arg/3', 3) | Rest],
                         [arg_reg_direct(NReg, TReg, Dest) | Out]) :-
    !,
    peephole_direct_builtins(Rest, Out).
%% arg/3 specialization — 4-instr pattern with integer-literal N.
peephole_direct_builtins([put_constant(N, 'A1'),
                          put_value(TReg, 'A2'),
                          put_variable(Dest, 'A3'),
                          builtin_call('arg/3', 3) | Rest],
                         [arg_lit_direct(N, TReg, Dest) | Out]) :-
    integer(N),
    !,
    peephole_direct_builtins(Rest, Out).
peephole_direct_builtins([builtin_call(Op, Arity) | Rest],
                         [DirectInstr | Out]) :-
    direct_builtin_map(Op, Arity, DirectInstr),
    !,
    peephole_direct_builtins(Rest, Out).
peephole_direct_builtins([H|T], [H|Out]) :-
    peephole_direct_builtins(T, Out).

%% direct_builtin_map(+Op, +Arity, -DirectInstr)
%  Table of builtin_call forms that have a direct-dispatch
%  specialization. Each direct instruction is zero-operand and
%  invokes the underlying $builtin_* helper directly, skipping the
%  $execute_builtin br_table hop.
direct_builtin_map('arg/3',       3, arg_direct).
direct_builtin_map('functor/3',   3, functor_direct).
direct_builtin_map('copy_term/2', 2, copy_term_direct).
direct_builtin_map('=../2',       2, univ_direct).
direct_builtin_map('is_list/1',   1, is_list_direct).

%% peephole_arg_to_a1(+Instrs, -Optimized)
%  Fuses arg_reg_direct(N, T, Dest) + put_value(Dest, A1) into a
%  single arg_to_a1_reg(N, T, Dest) instruction (and analogously
%  for arg_lit_direct → arg_to_a1_lit). Runs AFTER
%  peephole_direct_builtins so it sees the direct-dispatch form
%  rather than the raw builtin_call sequence.
%
%  This fires in sum_ints_args / term_depth_args where arg/3's
%  result is immediately put_value'd into A1 for the next call.
%  Saves one instruction dispatch + one 12-byte reg-to-reg copy.
peephole_arg_to_a1([], []).
peephole_arg_to_a1([arg_reg_direct(N, T, Dest),
                    put_value(Dest2, 'A1') | Rest],
                   [arg_to_a1_reg(N, T, Dest) | Out]) :-
    Dest == Dest2,
    !,
    peephole_arg_to_a1(Rest, Out).
peephole_arg_to_a1([arg_lit_direct(N, T, Dest),
                    put_value(Dest2, 'A1') | Rest],
                   [arg_to_a1_lit(N, T, Dest) | Out]) :-
    Dest == Dest2,
    !,
    peephole_arg_to_a1(Rest, Out).
peephole_arg_to_a1([H|T], [H|Out]) :-
    peephole_arg_to_a1(T, Out).

%% peephole_arg_call_k(+Instrs, -Optimized)
%  Fuses the call setup that follows arg_to_a1_{reg,lit} for arity-1
%  and arity-3 calls:
%
%  Arity-3 (sum_ints_args, term_depth_args shape):
%    arg_to_a1_reg(N, T, ArgDest)
%    put_value(A2Src, 'A2')
%    put_variable(RetDest, 'A3')
%    call(Pred, 3)
%  → arg_call_reg_3(N, T, ArgDest, A2Src, RetDest, Pred)
%
%  Arity-1 (`pred(X) :- arg(I,T,X), pred2(X).` shape):
%    arg_to_a1_reg(N, T, ArgDest)
%    call(Pred, 1)
%  → arg_call_reg_1(N, T, ArgDest, Pred)
%
%  Arity-2 is a potential follow-up (put_value(A2Src,A2) or
%  put_variable(A2Dst,A2) varies). Not handled here.
%
%  Match the arity-3 patterns BEFORE arity-1 so the longer match wins
%  — otherwise the arity-1 clause would consume the arg_to_a1 and
%  leave the A2/A3 setup uncollapsed.
peephole_arg_call_k([], []).
peephole_arg_call_k([arg_to_a1_reg(N, T, ArgDest),
                     put_value(A2Src, 'A2'),
                     put_variable(RetDest, 'A3'),
                     call(Pred, 3) | Rest],
                    [Fused | Out]) :-
    !,
    ( reg_used_before_clause_end(ArgDest, Rest)
    ->  Fused = arg_call_reg_3(N, T, ArgDest, A2Src, RetDest, Pred)
    ;   Fused = arg_call_reg_3_dead(N, T, ArgDest, A2Src, RetDest, Pred)
    ),
    peephole_arg_call_k(Rest, Out).
peephole_arg_call_k([arg_to_a1_lit(N, T, ArgDest),
                     put_value(A2Src, 'A2'),
                     put_variable(RetDest, 'A3'),
                     call(Pred, 3) | Rest],
                    [Fused | Out]) :-
    !,
    ( reg_used_before_clause_end(ArgDest, Rest)
    ->  Fused = arg_call_lit_3(N, T, ArgDest, A2Src, RetDest, Pred)
    ;   Fused = arg_call_lit_3_dead(N, T, ArgDest, A2Src, RetDest, Pred)
    ),
    peephole_arg_call_k(Rest, Out).
peephole_arg_call_k([arg_to_a1_reg(N, T, ArgDest),
                     PutA2,
                     call(Pred, 2) | Rest],
                    [Fused | Out]) :-
    a2_setup(PutA2, A2Reg, IsVar),
    !,
    ( reg_used_before_clause_end(ArgDest, Rest)
    ->  Fused = arg_call_reg_2(N, T, ArgDest, A2Reg, IsVar, Pred)
    ;   Fused = arg_call_reg_2_dead(N, T, ArgDest, A2Reg, IsVar, Pred)
    ),
    peephole_arg_call_k(Rest, Out).
peephole_arg_call_k([arg_to_a1_lit(N, T, ArgDest),
                     PutA2,
                     call(Pred, 2) | Rest],
                    [Fused | Out]) :-
    a2_setup(PutA2, A2Reg, IsVar),
    !,
    ( reg_used_before_clause_end(ArgDest, Rest)
    ->  Fused = arg_call_lit_2(N, T, ArgDest, A2Reg, IsVar, Pred)
    ;   Fused = arg_call_lit_2_dead(N, T, ArgDest, A2Reg, IsVar, Pred)
    ),
    peephole_arg_call_k(Rest, Out).
peephole_arg_call_k([arg_to_a1_reg(N, T, ArgDest),
                     call(Pred, 1) | Rest],
                    [Fused | Out]) :-
    !,
    ( reg_used_before_clause_end(ArgDest, Rest)
    ->  Fused = arg_call_reg_1(N, T, ArgDest, Pred)
    ;   Fused = arg_call_reg_1_dead(N, T, ArgDest, Pred)
    ),
    peephole_arg_call_k(Rest, Out).
peephole_arg_call_k([arg_to_a1_lit(N, T, ArgDest),
                     call(Pred, 1) | Rest],
                    [Fused | Out]) :-
    !,
    ( reg_used_before_clause_end(ArgDest, Rest)
    ->  Fused = arg_call_lit_1(N, T, ArgDest, Pred)
    ;   Fused = arg_call_lit_1_dead(N, T, ArgDest, Pred)
    ),
    peephole_arg_call_k(Rest, Out).
peephole_arg_call_k([H|T], [H|Out]) :-
    peephole_arg_call_k(T, Out).

%% peephole_tail_call_k(+Instrs, -Optimized)
%  Fuses the 5-arg tail-call-setup window:
%     put_value(R1,A1) put_value(R2,A2) put_value(R3,A3)
%     put_value(R4,A4) put_value(R5,A5) deallocate execute(Pred)
%  → tail_call_5(R1, R2, R3, R4, R5, Pred)
%
%  Fires on sum_ints_args/5 and term_depth_args/5 in the bench —
%  the 7-instruction window collapses to one dispatch, saving six
%  $step iterations per recursion.
peephole_tail_call_k([], []).
peephole_tail_call_k([put_value(R1, 'A1'),
                      put_value(R2, 'A2'),
                      put_value(R3, 'A3'),
                      put_value(R4, 'A4'),
                      put_value(R5, 'A5'),
                      deallocate,
                      execute(Pred) | Rest],
                     [tail_call_5(R1, R2, R3, R4, R5, Pred) | Out]) :-
    !,
    peephole_tail_call_k(Rest, Out).
peephole_tail_call_k([put_constant(C1Raw, 'A1'),
                      put_value(R2, 'A2'),
                      put_value(R3, 'A3'),
                      put_value(R4, 'A4'),
                      put_value(R5, 'A5'),
                      deallocate,
                      execute(Pred) | Rest],
                     [tail_call_5_c1_lit(C1, R2, R3, R4, R5, Pred) | Out]) :-
    %% Constants arrive as strings from the WAM-text parser; accept
    %% either an already-parsed integer or a numeric string. Atom
    %% constants (non-numeric strings) don't match this fusion.
    ( integer(C1Raw) -> C1 = C1Raw
    ; string(C1Raw), number_string(C1, C1Raw), integer(C1)
    ),
    C1 >= -32768, C1 =< 32767,
    !,
    peephole_tail_call_k(Rest, Out).
peephole_tail_call_k([deallocate, builtin_call(Op, N), proceed | Rest],
                     [deallocate_builtin_proceed(Op, N) | Out]) :-
    !,
    peephole_tail_call_k(Rest, Out).
peephole_tail_call_k([deallocate, Direct, proceed | Rest],
                     [Fused | Out]) :-
    dealloc_direct_fused(Direct, Fused),
    !,
    peephole_tail_call_k(Rest, Out).
peephole_tail_call_k([deallocate, proceed | Rest],
                     [deallocate_proceed | Out]) :-
    !,
    peephole_tail_call_k(Rest, Out).
peephole_tail_call_k([builtin_call(Op, N), proceed | Rest],
                     [builtin_proceed(Op, N) | Out]) :-
    !,
    peephole_tail_call_k(Rest, Out).
peephole_tail_call_k([H|T], [H|Out]) :-
    peephole_tail_call_k(T, Out).

%% peephole_type_dispatch(+Instrs, -Optimized)
%  Detects 3-clause predicates whose clauses 1 and 2 begin with type
%  guards on A1 (integer/1 or atom/1), and whose clause 3 is the
%  untyped default (typical in compound-walking patterns like
%  term_depth/2 where clause 3 handles compounds). Rewrites the
%  predicate entry to:
%
%    label(Pred)
%    type_dispatch_a1(atom_tgt, int_tgt, cmpd_tgt)
%    try_me_else(L2)               ← fallback for unbound A1
%    label(<synthetic clause1_entry>)
%    allocate                      ← clause 1 body continues as before
%    ...
%
%  The type_dispatch_a1 instruction dispatches A1's runtime tag
%  directly to one of the clause bodies, bypassing the try_me_else
%  chain. retry_me_else and trust_me inside the chain are
%  cp_count-guarded, so they no-op when the flow arrives via direct
%  dispatch.
%
%  Only matches the specific shape seen in the bench — a 3-clause
%  predicate with integer/1 and atom/1 guards on the first two
%  clauses. Narrower patterns (2-clause, compound guards, etc.)
%  would need additional peephole clauses.
%% 3-clause variant: matches `try_me_else + clause(guard1) +
%% retry_me_else + clause(guard2) + trust_me + default_clause`.
%% Fires on term_depth/2.
%%
%% default_tgt is set to the last clause's label — the untyped
%% fallback handles every non-matched tag (ref/unbound/float/list),
%% routing directly to it instead of falling through to the
%% now-redundant try_me_else chain.
%%
%% Since default_tgt makes the try_me_else/retry_me_else/trust_me
%% chain unreachable on every path, we drop those three
%% instructions from the output entirely. The label markers (L2, L3,
%% SynthL1) stay — they're the dispatch targets. retry_me_else and
%% trust_me are already guarded for cp_count=0, so removing them
%% is equivalent to having them run as no-ops; dropping them saves
%% one instruction dispatch per clause 2/3 entry plus data-segment
%% space.
peephole_type_dispatch(Instrs, Optimized) :-
    Instrs = [label(Pred), try_me_else(L2) | After],
    atom_string(L2, L2S),
    split_at_label(After, L2S, Clause1Body,
                   [label(L2S), retry_me_else(L3) | After2]),
    atom_string(L3, L3S),
    split_at_label(After2, L3S, Clause2Body,
                   [label(L3S), trust_me | Clause3Body]),
    classify_guard_in_body(Clause1Body, Type1),
    classify_guard_in_body(Clause2Body, Type2),
    Type1 \== Type2,
    !,
    format(string(SynthL1), 'L_type_dispatch_clause1_~w', [Pred]),
    assign_target(Type1, SynthL1, A1, I1),
    assign_target(Type2, L2,      A2, I2),
    pick_bound(A1, A2, AtomTgt),
    pick_bound(I1, I2, IntTgt),
    CmpdTgt = L3,
    DefaultTgt = L3,
    append([label(Pred),
            type_dispatch_a1(AtomTgt, IntTgt, CmpdTgt, DefaultTgt),
            label(SynthL1) | Clause1Body],
           [label(L2S) | Clause2Body],
           Head1),
    append(Head1,
           [label(L3S) | Clause3Body],
           Optimized).
%% 2-clause variant: matches `try_me_else + clause(guard) + trust_me
%% + default_clause`. Fires on sum_ints/3 (integer leaf + compound
%% walk). Same chain-dropping optimization as the 3-clause variant.
peephole_type_dispatch(Instrs, Optimized) :-
    Instrs = [label(Pred), try_me_else(L2) | After],
    atom_string(L2, L2S),
    split_at_label(After, L2S, Clause1Body,
                   [label(L2S), trust_me | Clause2Body]),
    classify_guard_in_body(Clause1Body, Type1),
    !,
    format(string(SynthL1), 'L_type_dispatch_clause1_~w', [Pred]),
    assign_target(Type1, SynthL1, A1, I1),
    pick_bound(A1, _, AtomTgt),
    pick_bound(I1, _, IntTgt),
    CmpdTgt = L2,
    DefaultTgt = L2,
    append([label(Pred),
            type_dispatch_a1(AtomTgt, IntTgt, CmpdTgt, DefaultTgt),
            label(SynthL1) | Clause1Body],
           [label(L2S) | Clause2Body],
           Optimized).
peephole_type_dispatch(Instrs, Instrs).

%% split_at_label(+Instrs, +LabelString, -Before, -AtAndAfter)
split_at_label([label(L) | Rest], L, [], [label(L) | Rest]) :- !.
split_at_label([H|T], L, [H|Before], After) :-
    split_at_label(T, L, Before, After).

%% classify_guard_in_body(+Body, -Type)
%  Scans the body of one clause for the leading type-guard pattern
%  `put_value(_, A1), builtin_call(<Type>/1, 1)`. Succeeds with Type
%  bound to 'integer' or 'atom' if found; fails if no such guard.
classify_guard_in_body([put_value(_, 'A1'),
                        builtin_call('integer/1', 1) | _],
                       integer) :- !.
classify_guard_in_body([put_value(_, 'A1'),
                        builtin_call('atom/1', 1) | _],
                       atom) :- !.
classify_guard_in_body([_ | Rest], Type) :-
    classify_guard_in_body(Rest, Type).

%% assign_target(+Type, +Lbl, -AtomSlot, -IntSlot)
assign_target(integer, Lbl, _, Lbl).
assign_target(atom,    Lbl, Lbl, _).

%% pick_bound(+A, +B, -Result)
%  Pick whichever of A or B is bound; else 0.
pick_bound(A, _, A) :- nonvar(A), !.
pick_bound(_, B, B) :- nonvar(B), !.
pick_bound(_, _, 0).

%% a2_setup(+Instr, -A2Reg, -IsVar)
%  Match the instruction that sets A2 in the K=2 fusion and extract
%  the source/destination register and mode flag. Put_value is a
%  source copy (IsVar=0); put_variable is a fresh-ref bind (IsVar=1).
a2_setup(put_value(A2Src, 'A2'), A2Src, 0).
a2_setup(put_variable(A2Dst, 'A2'), A2Dst, 1).

%% dealloc_direct_fused(+DirectInstr, -FusedForm)
%  Maps each zero-operand direct-dispatch builtin to its fused
%  `deallocate + <X>_direct + proceed` form.
dealloc_direct_fused(arg_direct,        deallocate_arg_direct_proceed).
dealloc_direct_fused(functor_direct,    deallocate_functor_direct_proceed).
dealloc_direct_fused(copy_term_direct,  deallocate_copy_term_direct_proceed).
dealloc_direct_fused(univ_direct,       deallocate_univ_direct_proceed).
dealloc_direct_fused(is_list_direct,    deallocate_is_list_direct_proceed).

%% reg_used_before_clause_end(+Reg, +Instrs)
%  Liveness check: succeeds if Reg is referenced on any reachable
%  path from here to the end of the current clause body. Fails
%  (= Reg is dead) if no such reference exists.
%
%  The scan is a linear walk that tracks `try_me_else` / `trust_me`
%  nesting via a depth counter:
%    - Outside any nested try_me_else (depth 0): proceed/execute
%      marks the end of the current clause → stop.
%    - Inside a nested try_me_else…trust_me block (depth > 0): the
%      proceed in the then-branch of an in-clause disjunction or
%      if-then-else is NOT the clause end — the else-branch still
%      lies ahead in the linear stream → keep scanning.
%
%  This handles the two common in-clause control-flow shapes:
%    (guard -> then ; else)   — try_me_else ... trust_me
%    (a ; b ; c)              — try_me_else ... retry_me_else ... trust_me
%  and their nested variants. Forward jumps / cut_ite / switch_* /
%  label markers are transparent: the scan walks past them because
%  every reachable target within the clause is also in the linear
%  stream later.
%
%  A reference in an unreachable tail (e.g., after an unconditional
%  jump, in dead code) would be counted as live — harmless over-
%  approximation; the correctness direction is "never call dead on
%  something live".
reg_used_before_clause_end(Reg, Instrs) :-
    reg_used_scan(Reg, Instrs, 0).

reg_used_scan(_, [], _) :- !, fail.
reg_used_scan(Reg, [Instr|_], _) :-
    instr_references_reg(Instr, Reg), !.
reg_used_scan(_, [Instr|_], 0) :-
    clause_end_instr(Instr), !, fail.
reg_used_scan(Reg, [try_me_else(_)|Rest], Depth) :- !,
    D1 is Depth + 1,
    reg_used_scan(Reg, Rest, D1).
reg_used_scan(Reg, [trust_me|Rest], Depth) :-
    Depth > 0, !,
    D1 is Depth - 1,
    reg_used_scan(Reg, Rest, D1).
reg_used_scan(Reg, [_|Rest], Depth) :-
    reg_used_scan(Reg, Rest, Depth).

clause_end_instr(proceed).
clause_end_instr(execute(_)).

%% instr_references_reg(+Instr, +Reg)
%  True if Reg appears as any operand of Instr. Uniform term-walking
%  check — since all register-bearing WAM instructions carry their
%  regs as atomic arguments (e.g., put_value('Y1', 'A1')), a membership
%  test finds them regardless of position. Integer operands, functors
%  like '+/2', and label atoms don't collide because Reg is always a
%  canonical atom like 'Y1'/'X2'/'A3' and == is strict equality.
instr_references_reg(Instr, Reg) :-
    Instr =.. [_|Args],
    member(Arg, Args),
    Arg == Reg, !.

peephole_neck_cut([try_me_else(L), allocate | Rest], Optimized) :-
    find_guard_and_cut(Rest, BeforeGuard, GuardOp, GuardArity, AfterCut),
    remove_label_trust_me(AfterCut, L, Cleaned),
    !,
    append([allocate | BeforeGuard],
           [neck_cut_test(GuardOp, GuardArity, L) | Cleaned],
           Optimized).
peephole_neck_cut(Instrs, Instrs).

%% find_guard_and_cut(+Instrs, -Before, -GuardOp, -GuardArity, -After)
%  Scans for builtin_call(Guard, N), builtin_call('!/0', 0) and splits.
%  Only matches if the guard is within the first 15 elements (instructions
%  + label markers). Skips label markers in the "Before" accumulator.
find_guard_and_cut(Instrs, Before, GuardOp, GuardArity, After) :-
    find_guard_and_cut_(Instrs, 0, Before, GuardOp, GuardArity, After).

find_guard_and_cut_([builtin_call(G, N), builtin_call('!/0', 0) | After],
                    _, [], G, N, After) :- !.
find_guard_and_cut_([I | Rest], Depth, [I | Before], G, N, After) :-
    Depth < 15,
    D1 is Depth + 1,
    find_guard_and_cut_(Rest, D1, Before, G, N, After).

%% remove_label_trust_me(+Instrs, +Label, -Cleaned)
%  Finds label(L), trust_me in the instruction stream and removes the
%  trust_me (keeping the label — it is the jump target for neck_cut_test).
remove_label_trust_me([], _, []).
remove_label_trust_me([label(L), trust_me | Rest], L, [label(L) | Rest]) :- !.
remove_label_trust_me([I | Rest], L, [I | Cleaned]) :-
    remove_label_trust_me(Rest, L, Cleaned).

%% gen_all_entry_funcs(+PredData, +HeapStart, +CodeBase, +TotalInstrs,
%%                     -Funcs, -Exports)
%  Emit one exported entry function per predicate. All entry functions
%  share the same CodeBase and TotalInstrs; they differ only in their
%  StartPC, which each function writes into the VM's PC register
%  immediately after $wam_init (before invoking the run loop).
gen_all_entry_funcs([], _, _, _, '', '').
gen_all_entry_funcs([pred_data(Pred/Arity, _, _, StartPC, _)|Rest],
                    HeapStart, CodeBase, TotalInstrs,
                    EntryFuncs, Exports) :-
    wat_pred_name(Pred, Arity, FName),
    format(atom(EF),
'(func $~w (export "~w") (result i32)
  (call $wam_init (i32.const ~w))
  (call $set_pc (i32.const ~w))
  (call $run_loop (i32.const ~w) (i32.const ~w)))',
        [FName, FName, HeapStart, StartPC, CodeBase, TotalInstrs]),
    format(atom(Export), '  ;; exported: ~w', [FName]),
    gen_all_entry_funcs(Rest, HeapStart, CodeBase, TotalInstrs,
                        RestEF, RestExports),
    atomic_list_concat([EF, '\n', RestEF], EntryFuncs),
    atomic_list_concat([Export, '\n', RestExports], Exports).

%% read_template_file(+Path, -Content)
read_template_file(Path, Content) :-
    (   exists_file(Path)
    ->  read_file_to_string(Path, Content, [])
    ;   format(atom(Content), ";; Template not found: ~w", [Path])
    ).

%% write_file(+Path, +Content)
write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).
