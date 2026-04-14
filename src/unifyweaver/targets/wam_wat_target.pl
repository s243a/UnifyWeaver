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
%% wam_cp_size = 20 (metadata: next_pc + trail_mark + saved_cp + saved_heap_top
%%               + saved_env_base) + CP_SAVE_REGS A registers * val_size.
%% Only the first CP_SAVE_REGS argument registers (A1..A_N) are saved in choice
%% points. Y registers live in environment frames and are NOT saved. X temporaries
%% above CP_SAVE_REGS are also NOT saved — they are dead across choice point
%% boundaries in well-compiled WAM code (only argument registers carry state
%% across alternatives). Reducing from 32 to 8 cuts CP push/pop memcpy from
%% 384 bytes to 96 bytes — a 4x reduction that directly impacts recursive
%% predicates like sum_ints which create/restore CPs on every clause entry.
wam_cp_save_regs(8).
wam_cp_size(Size) :- wam_cp_save_regs(N), wam_val_size(V), Size is 20 + N * V.

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

% --- Encoding helpers ---

%% encode_constant(+C, -I64Val)
%  Encodes a Prolog constant as an i64 value.
%  For atoms: the hash. For integers: the raw value.
encode_constant(C, Val) :- encode_constant_with_tag(C, _Tag, Val).

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
encode_constant_with_tag(atom(A), 0, Hash) :- !, atom_hash_i64(A, Hash).
encode_constant_with_tag(integer(I), 1, I) :- !.
encode_constant_with_tag(I, 1, I) :- integer(I), !.
encode_constant_with_tag(F, 2, Bits) :- float(F), !, Bits is float_integer_part(F).
encode_constant_with_tag(A, 0, Hash) :- atom(A), !, atom_hash_i64(A, Hash).
encode_constant_with_tag(S, Tag, Val) :-
    string(S), !,
    (   number_string(I, S), integer(I)
    ->  Tag = 1, Val = I
    ;   atom_string(A, S),
        atom_hash_i64(A, Hash),
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
        PC1 is PC + 1,
        Instrs = [Instr|RestInstrs],
        wam_lines_to_instrs(Rest, PC1, RestInstrs, Labels)
    ;   wam_lines_to_instrs(Rest, PC, Instrs, Labels)
    ).

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
  ;; Update choice point with new alternative
  (call $update_choice_point (i32.wrap_i64 (local.get $op1)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(trust_me,
'  ;; Remove choice point (last alternative)
  (call $pop_choice_point_no_restore)
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
    (then (return (i32.const 0))))
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
;; Each CP: [next_pc:i32 +0][trail_mark:i32 +4][saved_cp:i32 +8][heap_top:i32 +12][env_base:i32 +16][32 A/X regs: 384 bytes +20] = ~w bytes

(func $cp_base_offset (result i32) (i32.const 98304))

(func $cp_offset (param $idx i32) (result i32)
  (i32.add (call $cp_base_offset) (i32.mul (local.get $idx) (i32.const ~w))))

(func $push_choice_point (param $next_pc i32)
  (local $n i32) (local $off i32) (local $i i32)
  (local.set $n (call $get_cp_count))
  (local.set $off (call $cp_offset (local.get $n)))
  ;; Save next_pc, trail mark, CP, heap_top, env_base
  (i32.store (local.get $off) (local.get $next_pc))
  (i32.store (i32.add (local.get $off) (i32.const 4)) (call $get_trail_top))
  (i32.store (i32.add (local.get $off) (i32.const 8)) (call $get_cp))
  (i32.store (i32.add (local.get $off) (i32.const 12)) (call $get_heap_top))
  (i32.store (i32.add (local.get $off) (i32.const 16)) (global.get $env_base))
  ;; Save first ~w argument registers at +20. Only A1..A_N carry
  ;; meaningful state across choice point alternatives (N = CP_SAVE_REGS,
  ;; currently 8, covering predicates up to arity 8). Saving fewer
  ;; registers reduces the per-CP memcpy from 384 bytes (32 regs) to
  ;; 96 bytes (8 regs) — a 4x reduction that directly impacts
  ;; recursive predicates like sum_ints.
  (local.set $i (i32.const 0))
  (block $done
    (loop $save
      (br_if $done (i32.ge_u (local.get $i) (i32.const ~w)))
      (call $copy_from_reg (local.get $i)
        (i32.add (i32.add (local.get $off) (i32.const 20))
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
        (i32.add (i32.add (local.get $off) (i32.const 20))
                 (i32.mul (local.get $i) (i32.const 12))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $restore))))

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

;; --- arg/3: A1 = N (1-based), A2 = T (ref->compound), A3 = result ---
;; Reads the Nth argument cell of the compound T references, unifies it
;; with A3. Argument cells are laid out contiguously after the compound
;; header, each cell 12 bytes. Arity is recovered from the high 32 bits
;; of the compound header payload (see encode_structure_op1 in
;; wam_wat_target.pl).
(func $builtin_arg (result i32)
  (local $n i32)
  (local $t_tag i32)
  (local $comp_addr i32)
  (local $comp_tag i32)
  (local $comp_payload i64)
  (local $arity i32)
  (local $arg_off i32)
  (local $arg_tag i32)
  (local $arg_payload i64)
  (local $a3_tag i32)
  ;; N from A1 (register index 0). Payload is the raw integer value.
  (local.set $n (i32.wrap_i64 (call $deref_reg_payload (i32.const 0))))
  (if (i32.lt_s (local.get $n) (i32.const 1))
    (then (return (i32.const 0))))
  ;; T from A2 (register index 1). Must be a compound (tag=3 after deref).
  (local.set $t_tag (call $deref_reg_tag (i32.const 1)))
  (if (i32.ne (local.get $t_tag) (i32.const 3))
    (then (return (i32.const 0))))
  (local.set $comp_addr (call $deref_reg_addr (i32.const 1)))
  (local.set $comp_payload (call $val_payload (local.get $comp_addr)))
  (local.set $arity
    (i32.wrap_i64 (i64.shr_u (local.get $comp_payload) (i64.const 32))))
  (if (i32.gt_s (local.get $n) (local.get $arity))
    (then (return (i32.const 0))))
  ;; Argument N at compound_addr + N*12 (header at +0, args start at +12).
  (local.set $arg_off
    (i32.add (local.get $comp_addr)
             (i32.mul (local.get $n) (i32.const 12))))
  (local.set $arg_tag (call $val_tag (local.get $arg_off)))
  (local.set $arg_payload (call $val_payload (local.get $arg_off)))
  ;; A3 (register index 2): bind if unbound, else shallow compare.
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
    %% 4=copy_term scratch (var map + work stack, fixed offsets at 262144).
    %% See $builtin_copy_term for the scratch layout. The scratch lives
    %% outside the heap so deep copy_term calls don''t permanently bloat
    %% heap_top with unreclaimable bookkeeping.
    MemPages = 5,

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
    %% Pass 1: parse + collect per-predicate data with cumulative PCs.
    pass1_parse_predicates(Predicates, Options, 0, PredData, TotalInstrs),
    %% Build global label table (shift each predicate's local labels).
    build_global_labels(PredData, [], GlobalLabels),
    %% Pass 2: re-encode each predicate's instructions against global
    %% labels and concatenate the byte sequences.
    encode_all_predicates(PredData, GlobalLabels, AllHex),
    (   AllHex == ''
    ->  DataSegs = ''
    ;   format(atom(DataSegs),
            '(data (i32.const ~w) "~w")', [CodeBase, AllHex])
    ),
    %% Entry functions: one per predicate, each setting its own
    %% start PC before invoking the shared run loop.
    option(wam_heap_start(HeapStart), Options, 196608),
    gen_all_entry_funcs(PredData, HeapStart, CodeBase, TotalInstrs,
                        Funcs, Exports).

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
        wam_lines_to_instrs(Lines, 0, Instrs, LocalLabels),
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
%  order. The order matches pass1 so instruction indices remain
%  consistent with start PCs.
encode_all_predicates([], _, '').
encode_all_predicates([pred_data(_, Instrs, _, _, _)|Rest],
                      GlobalLabels, AllHex) :-
    maplist(encode_instr_with_labels(GlobalLabels), Instrs, HexParts),
    atomic_list_concat(HexParts, PredHex),
    encode_all_predicates(Rest, GlobalLabels, RestHex),
    atomic_list_concat([PredHex, RestHex], AllHex).

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
