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
%% wam_cp_size = 12 (metadata: next_pc + trail_mark + saved_cp) + num_regs * val_size
wam_cp_size(Size) :- wam_num_regs(N), wam_val_size(V), Size is 12 + N * V.

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

% ============================================================================
% Register name -> index mapping
% ============================================================================

%% reg_name_to_index(+Name, -Index)
%  A1-A32 -> 0-31, X1-X32 -> 32-63, Y1-Y32 -> 32-63 (share X space)
reg_name_to_index(Name, Index) :-
    atom_string(Name, Str),
    string_codes(Str, [Prefix|Rest]),
    number_codes(N, Rest),
    (   Prefix =:= 0'A -> Index is N - 1
    ;   Prefix =:= 0'X -> Index is N + 31
    ;   Prefix =:= 0'Y -> Index is N + 31
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
    encode_constant(C, Op1),
    reg_name_to_index(Ai, RegIdx),
    encode_instr_hex(Tag, Op1, RegIdx, Hex).

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
    atom_hash_i64(F, FHash),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, FHash, AiIdx, Hex).

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
    encode_constant(C, Op1),
    encode_instr_hex(Tag, Op1, 0, Hex).

wam_instruction_to_wat_bytes(put_constant(C, Ai), _Labels, Hex) :-
    instr_tag(put_constant, Tag),
    encode_constant(C, Op1),
    reg_name_to_index(Ai, RegIdx),
    encode_instr_hex(Tag, Op1, RegIdx, Hex).

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
    atom_hash_i64(F, FHash),
    reg_name_to_index(Ai, AiIdx),
    encode_instr_hex(Tag, FHash, AiIdx, Hex).

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
    encode_constant(C, Op1),
    encode_instr_hex(Tag, Op1, 0, Hex).

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
encode_constant(atom(A), Hash) :- !, atom_hash_i64(A, Hash).
encode_constant(integer(I), I) :- !.
encode_constant(N, N) :- integer(N), !.
encode_constant(N, Bits) :- float(N), !, Bits is float_integer_part(N).
encode_constant(A, Hash) :- atom(A), !, atom_hash_i64(A, Hash).
encode_constant(_, 0).

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
    format(atom(EntryFunc),
'(func $~w (export "~w") (result i32)
  (call $wam_init (i32.const ~w))
  (call $run_loop (i32.const ~w) (i32.const ~w)))',
        [FuncName, FuncName, CodeBase, CodeBase, NumInstrs]),
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
    %% WAT br_table jumps to block N (counting from innermost).
    %% We nest blocks so that tag 0 breaks to the outermost (first to close),
    %% and after each block close we call the corresponding do_ function.
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
    %% Closing blocks + dispatch calls (outermost first = tag 0)
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
  )
~w
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
'  (local $reg_idx i32)
  (local.set $reg_idx (i32.wrap_i64 (local.get $op2)))
  (if (result i32) (call $val_is_unbound (call $reg_offset (local.get $reg_idx)))
    (then
      (call $trail_binding (local.get $reg_idx))
      (call $set_reg (local.get $reg_idx) (i32.const 0) (local.get $op1))
      (call $inc_pc)
      (i32.const 1))
    (else
      (if (result i32) (i32.and
            (i32.eq (call $get_reg_tag (local.get $reg_idx)) (i32.const 0))
            (i64.eq (call $get_reg_payload (local.get $reg_idx)) (local.get $op1)))
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
'  ;; op1 = functor hash, op2 = reg index
  (local $ai i32) (local $tag i32) (local $addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (local.set $tag (call $get_reg_tag (local.get $ai)))
  (if (result i32) (i32.eq (local.get $tag) (i32.const 6)) ;; unbound
    (then
      ;; Write mode: allocate compound on heap
      (local.set $addr (call $heap_push_val (i32.const 3) (local.get $op1)))
      (call $trail_binding (local.get $ai))
      (call $set_reg (local.get $ai) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
      (call $set_mode (i32.const 1))
      (call $inc_pc)
      (i32.const 1))
    (else
      ;; Read mode: check functor match
      (if (result i32) (i32.and
            (i32.eq (local.get $tag) (i32.const 3))
            (i64.eq (call $get_reg_payload (local.get $ai)) (local.get $op1)))
        (then
          (call $set_mode (i32.const 0))
          (call $inc_pc)
          (i32.const 1))
        (else (i32.const 0)))))').

wam_wat_case(get_list,
'  (local $ai i32) (local $tag i32) (local $addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op1)))
  (local.set $tag (call $get_reg_tag (local.get $ai)))
  (if (result i32) (i32.eq (local.get $tag) (i32.const 6)) ;; unbound
    (then
      (local.set $addr (call $heap_push_val (i32.const 4) (i64.const 0)))
      (call $trail_binding (local.get $ai))
      (call $set_reg (local.get $ai) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
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
      ;; Push register value to heap
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
'  (if (result i32) (call $get_mode) ;; write mode
    (then
      (drop (call $heap_push_val (i32.const 0) (local.get $op1)))
      (call $inc_pc)
      (i32.const 1))
    (else
      ;; Read mode: match constant
      (call $inc_pc)
      (i32.const 1)))').

% --- Body construction ---

wam_wat_case(put_constant,
'  (local $ai i32)
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  ;; Determine tag from constant type (op1 encodes either atom hash or integer)
  ;; For now, use atom tag=0 for hashed values, integer tag=1 for raw integers
  (call $set_reg (local.get $ai) (i32.const 0) (local.get $op1))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(put_variable,
'  (local $xn i32) (local $ai i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  ;; Create unbound variable in both Xn and Ai
  (call $set_reg (local.get $xn) (i32.const 6) (i64.extend_i32_u (local.get $xn)))
  (call $set_reg (local.get $ai) (i32.const 6) (i64.extend_i32_u (local.get $xn)))
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
'  (local $ai i32) (local $addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op2)))
  (local.set $addr (call $heap_push_val (i32.const 3) (local.get $op1)))
  (call $set_reg (local.get $ai) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
  (call $set_mode (i32.const 1))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(put_list,
'  (local $ai i32) (local $addr i32)
  (local.set $ai (i32.wrap_i64 (local.get $op1)))
  (local.set $addr (call $heap_push_val (i32.const 4) (i64.const 0)))
  (call $set_reg (local.get $ai) (i32.const 5) (i64.extend_i32_u (local.get $addr)))
  (call $set_mode (i32.const 1))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(set_variable,
'  (local $xn i32) (local $addr i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  ;; Create new unbound on heap
  (local.set $addr (call $heap_push_val (i32.const 6) (i64.extend_i32_u (local.get $xn))))
  (call $set_reg (local.get $xn) (i32.const 6) (i64.extend_i32_u (local.get $xn)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(set_value,
'  (local $xn i32)
  (local.set $xn (i32.wrap_i64 (local.get $op1)))
  (drop (call $heap_push_val
    (call $get_reg_tag (local.get $xn))
    (call $get_reg_payload (local.get $xn))))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(set_constant,
'  (drop (call $heap_push_val (i32.const 0) (local.get $op1)))
  (call $inc_pc)
  (i32.const 1)').

% --- Control flow ---

wam_wat_case(allocate,
'  ;; Push environment frame: save CP on stack
  (local $soff i32)
  (local.set $soff (call $get_stack_top))
  (i32.store (local.get $soff) (call $get_cp))
  (call $set_stack_top (i32.add (local.get $soff) (i32.const 4)))
  (call $inc_pc)
  (i32.const 1)').

wam_wat_case(deallocate,
'  ;; Pop environment frame: restore CP
  (local $soff i32)
  (local.set $soff (i32.sub (call $get_stack_top) (i32.const 4)))
  (call $set_cp (i32.load (local.get $soff)))
  (call $set_stack_top (local.get $soff))
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
    wam_num_regs(NumRegs),
    wam_val_size(ValSize),
    RegBytes is NumRegs * ValSize,
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
  ;; Unwind trail
  (call $unwind_trail (local.get $trail_mark))
  ;; Restore registers from choice point
  (call $cp_restore_regs)
  ;; Restore state
  (call $set_pc (local.get $next_pc))
  (call $set_cp (local.get $saved_cp))
  (call $set_halted (i32.const 0))
  (call $set_mode (i32.const 0))
  (i32.const 1)
)

;; --- Trail unwinding ---
(func $unwind_trail (param $mark i32)
  (local $toff i32)
  (local $reg_idx i32)
  (local $old_tag i32)
  (local $old_payload i64)
  (block $done
    (loop $unwind
      (local.set $toff (call $get_trail_top))
      (br_if $done (i32.le_u (local.get $toff) (local.get $mark)))
      ;; Back up one entry (16 bytes)
      (local.set $toff (i32.sub (local.get $toff) (i32.const 16)))
      (call $set_trail_top (local.get $toff))
      ;; Restore register
      (local.set $reg_idx (i32.load (local.get $toff)))
      (local.set $old_tag (i32.load (i32.add (local.get $toff) (i32.const 4))))
      (local.set $old_payload (i64.load (i32.add (local.get $toff) (i32.const 8))))
      (call $set_reg (local.get $reg_idx) (local.get $old_tag) (local.get $old_payload))
      (br $unwind)
    )
  )
)

;; --- Choice point management ---
;; Choice points stored in a dedicated area starting at offset 98304 (page 1.5)
;; Each CP: [next_pc:i32 +0][trail_mark:i32 +4][saved_cp:i32 +8][~w regs: ~w bytes +12] = ~w bytes

(func $cp_base_offset (result i32) (i32.const 98304))

(func $cp_offset (param $idx i32) (result i32)
  (i32.add (call $cp_base_offset) (i32.mul (local.get $idx) (i32.const ~w))))

(func $push_choice_point (param $next_pc i32)
  (local $n i32) (local $off i32) (local $i i32)
  (local.set $n (call $get_cp_count))
  (local.set $off (call $cp_offset (local.get $n)))
  ;; Save next_pc, trail mark, CP
  (i32.store (local.get $off) (local.get $next_pc))
  (i32.store (i32.add (local.get $off) (i32.const 4)) (call $get_trail_top))
  (i32.store (i32.add (local.get $off) (i32.const 8)) (call $get_cp))
  ;; Save all 64 registers (64 x 12 = 768 bytes)
  (local.set $i (i32.const 0))
  (block $done
    (loop $save
      (br_if $done (i32.ge_u (local.get $i) (i32.const ~w)))
      (call $copy_from_reg (local.get $i)
        (i32.add (i32.add (local.get $off) (i32.const 12))
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

(func $cp_restore_regs
  (local $off i32) (local $i i32)
  (local.set $off (call $cp_offset (i32.sub (call $get_cp_count) (i32.const 1))))
  (local.set $i (i32.const 0))
  (block $done
    (loop $restore
      (br_if $done (i32.ge_u (local.get $i) (i32.const ~w)))
      (call $copy_to_reg (local.get $i)
        (i32.add (i32.add (local.get $off) (i32.const 12))
                 (i32.mul (local.get $i) (i32.const 12))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $restore)))
  ;; Pop the choice point
  (call $pop_choice_point_no_restore))

;; --- Unification ---
;; Unify two registers. No occurs check (standard Prolog semantics).
(func $unify_regs (param $r1 i32) (param $r2 i32) (result i32)
  (local $t1 i32) (local $t2 i32) (local $p1 i64) (local $p2 i64)
  (local.set $t1 (call $get_reg_tag (local.get $r1)))
  (local.set $p1 (call $get_reg_payload (local.get $r1)))
  (local.set $t2 (call $get_reg_tag (local.get $r2)))
  (local.set $p2 (call $get_reg_payload (local.get $r2)))
  ;; If r1 is unbound, bind it to r2
  (if (i32.eq (local.get $t1) (i32.const 6))
    (then
      (call $trail_binding (local.get $r1))
      (call $set_reg (local.get $r1) (local.get $t2) (local.get $p2))
      (return (i32.const 1))))
  ;; If r2 is unbound, bind it to r1
  (if (i32.eq (local.get $t2) (i32.const 6))
    (then
      (call $trail_binding (local.get $r2))
      (call $set_reg (local.get $r2) (local.get $t1) (local.get $p1))
      (return (i32.const 1))))
  ;; Both bound: check equality
  (i32.and
    (i32.eq (local.get $t1) (local.get $t2))
    (i64.eq (local.get $p1) (local.get $p2))))

;; --- Builtin dispatch ---
(func $execute_builtin (param $id i32) (param $arity i32) (result i32)
  (block $default
    (block $cut (block $fail (block $true_b
    (block $arith_ge (block $arith_le (block $arith_gt (block $arith_lt
    (block $arith_ne (block $arith_eq (block $is
    (block $nl (block $write
      (br_table $write $nl $is $arith_eq $arith_ne $arith_lt $arith_gt $arith_le $arith_ge
                $default $default $default $default $default $default
                $true_b $fail $cut $default (local.get $id))
    ) ;; write
    (call $print_i64 (call $get_reg_payload (i32.const 0)))
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
    ) ;; true
    (return (i32.const 1))
    ) ;; fail
    (return (i32.const 0))
    ) ;; cut
    (call $set_cp_count (i32.const 0))
    (return (i32.const 1))
  )
  (i32.const 0)
)

;; --- is/2: A1 = eval(A2) ---
(func $builtin_is (result i32)
  (local $val i64)
  ;; For now, just copy A2 to A1 if A2 is an integer
  ;; TODO: full arithmetic expression evaluation
  (if (call $val_is_integer (call $reg_offset (i32.const 1)))
    (then
      (local.set $val (call $get_reg_payload (i32.const 1)))
      (call $trail_binding (i32.const 0))
      (call $set_reg (i32.const 0) (i32.const 1) (local.get $val))
      (return (i32.const 1))))
  (i32.const 0)
)

;; --- Arithmetic comparison: 0==eq, 1==ne, 2==lt, 3==gt, 4==le, 5==ge ---
(func $builtin_arith_cmp (param $op i32) (result i32)
  (local $a i64) (local $b i64)
  (local.set $a (call $get_reg_payload (i32.const 0)))
  (local.set $b (call $get_reg_payload (i32.const 1)))
  (block $default
    (block $ge (block $le (block $gt (block $lt (block $ne (block $eq
      (br_table $eq $ne $lt $gt $le $ge $default (local.get $op))
    ) (return (i64.eq (local.get $a) (local.get $b))))
    ) (return (i64.ne (local.get $a) (local.get $b))))
    ) (return (i64.lt_s (local.get $a) (local.get $b))))
    ) (return (i64.gt_s (local.get $a) (local.get $b))))
    ) (return (i64.le_s (local.get $a) (local.get $b))))
    ) (return (i64.ge_s (local.get $a) (local.get $b))))
  )
  (i32.const 0)
)
', [CPSize, NumRegs, NumRegs, RegBytes, CPSize, CPSize, NumRegs, NumRegs]).

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

    %% Calculate memory pages needed (3 minimum: native WAT + WAM state + heap)
    MemPages = 4,

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

%% compile_wat_predicates(+Preds, +Opts, +BaseOff, -DataSegs, -Funcs, -Exports)
compile_wat_predicates([], _, _, '', '', '').
compile_wat_predicates([PredInd|Rest], Options, BaseOff, DataSegs, Funcs, Exports) :-
    (   PredInd = _Module:Pred/Arity -> true
    ;   PredInd = Pred/Arity
    ),
    %% Try WAM compilation
    (   catch(
            wam_target:compile_predicate_to_wam(PredInd, Options, WamCode),
            _, fail)
    ->  PredOpts = [code_base(BaseOff)|Options],
        compile_wam_predicate_to_wat(Pred/Arity, WamCode, PredOpts, WatResult),
        WatResult = wat_pred(DS, EF, _, NumInstrs),
        NextBase is BaseOff + NumInstrs * 20,
        format(user_error, '  ~w/~w: WAM compilation (~w instructions)~n',
               [Pred, Arity, NumInstrs]),
        wat_pred_name(Pred, Arity, FName),
        format(atom(Export), '  ;; exported: ~w', [FName])
    ;   DS = '', EF = '', NextBase = BaseOff,
        Export = '',
        format(user_error, '  ~w/~w: compilation failed~n', [Pred, Arity])
    ),
    compile_wat_predicates(Rest, Options, NextBase, RestDS, RestFuncs, RestExports),
    atomic_list_concat([DS, '\n', RestDS], DataSegs),
    atomic_list_concat([EF, '\n', RestFuncs], Funcs),
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
