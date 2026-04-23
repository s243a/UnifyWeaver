:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_c_target.pl - WAM-to-C Transpilation Target
%
% Transpiles WAM runtime predicates to C code.
%
% Design goals:
% - C99 or C11 compatible.
% - Explicit memory and pointer handling.
% - WAM registers (A, S, H, HB, TR, P, CP, B) mapped to C struct fields.
% - Unification trail and heap modeled as C arrays with explicit bounds.

:- module(wam_c_target, [
    compile_step_wam_to_c/2,          % +Options, -CCode
    compile_wam_helpers_to_c/2,       % +Options, -CCode
    compile_wam_runtime_to_c/2,       % +Options, -CCode
    compile_wam_predicate_to_c/4,     % +Pred/Arity, +WamCode, +Options, -CCode
    wam_instruction_to_c_literal/2,   % +WamInstr, -CCode
    wam_instruction_to_c_literal/3,   % +WamInstr, +LabelMap, -CCode
    write_wam_c_project/3             % +Predicates, +Options, +ProjectDir
]).

:- use_module(library(lists)).
:- use_module(library(option)).

:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../bindings/c_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

% ============================================================================
% PHASE 4: Hybrid Module Assembly
% ============================================================================

%% write_wam_c_project(+Predicates, +Options, +ProjectDir)
%  Generates a full C project for the given predicates.
write_wam_c_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    % Generate runtime .c and .h files
    compile_wam_runtime_to_c(Options, RuntimeCode),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    write_file(RuntimePath, RuntimeCode),

    % Compile predicates and generate lib.c
    compile_predicates_for_project(Predicates, Options, PredicatesCode),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    write_file(LibPath, PredicatesCode),

    format('WAM C project created at: ~w~n', [ProjectDir]).

%% write_file(+Path, +Content)
write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

%% compile_predicates_for_project(+Predicates, +Options, -Code)
compile_predicates_for_project([], _, "").
compile_predicates_for_project([PredIndicator|Rest], Options, Code) :-
    predicate_indicator_parts(PredIndicator, Module, Pred, Arity),
    (   wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode)
    ->  compile_wam_predicate_to_c(Module:Pred/Arity, WamCode, Options, PredCode)
    ;   format(atom(PredCode), '// ~w/~w: compilation failed', [Pred, Arity])
    ),
    compile_predicates_for_project(Rest, Options, RestCode),
    format(atom(Code), '~w\n\n~w', [PredCode, RestCode]).

predicate_indicator_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
predicate_indicator_parts(Pred/Arity, user, Pred, Arity).

% ============================================================================
% PHASE 2: WAM instructions -> C Struct Literals
% ============================================================================

%% wam_instruction_to_c_literal(+WamInstr, -CCode)
wam_instruction_to_c_literal(get_constant(C, Ai), Code) :-
    c_value_literal(C, Val), c_reg_index(Ai, IsY_Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_GET_CONSTANT, .val = ~w, .reg = ~w, .is_y_reg = ~w }', [Val, Idx, IsY_Ai]).
wam_instruction_to_c_literal(get_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VARIABLE, .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(get_value(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VALUE, .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_constant(C, Ai), Code) :-
    c_value_literal(C, Val), c_reg_index(Ai, IsY_Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_PUT_CONSTANT, .val = ~w, .reg = ~w, .is_y_reg = ~w }', [Val, Idx, IsY_Ai]).
wam_instruction_to_c_literal(put_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VARIABLE, .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_value(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VALUE, .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(get_structure(F, Ai), Code) :-
    c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_STRUCTURE, .pred = "~w", .reg_ai = ~w, .is_y_ai = ~w }', [F, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_structure(F, Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_STRUCTURE, .pred = "~w", .reg_xn = ~w, .is_y_xn = ~w }', [F, XIdx, IsY_Xn]).
wam_instruction_to_c_literal(get_list(Ai), Code) :-
    c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_LIST, .reg_ai = ~w, .is_y_ai = ~w }', [AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_list(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_LIST, .reg_xn = ~w, .is_y_xn = ~w }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(unify_variable(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_UNIFY_VARIABLE, .reg_xn = ~w, .is_y_xn = ~w }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(unify_value(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_UNIFY_VALUE, .reg_xn = ~w, .is_y_xn = ~w }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(unify_constant(C), Code) :-
    c_value_literal(C, Val),
    format(atom(Code), '{ .tag = INSTR_UNIFY_CONSTANT, .val = ~w }', [Val]).
wam_instruction_to_c_literal(call(P, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_CALL, .pred = "~w", .arity = ~w }', [P, N]).
wam_instruction_to_c_literal(execute(P), Code) :-
    format(atom(Code), '{ .tag = INSTR_EXECUTE, .pred = "~w" }', [P]).
wam_instruction_to_c_literal(try_me_else(_Label), _) :-
    throw(error(context_error(missing_label_map, "try_me_else/1 requires LabelMap for target_pc resolution. Use wam_instruction_to_c_literal/3 instead."), _)).
wam_instruction_to_c_literal(retry_me_else(_Label), _) :-
    throw(error(context_error(missing_label_map, "retry_me_else/1 requires LabelMap for target_pc resolution. Use wam_instruction_to_c_literal/3 instead."), _)).


wam_instruction_to_c_literal(trust_me, '{ .tag = INSTR_TRUST_ME }').
wam_instruction_to_c_literal(proceed, '{ .tag = INSTR_PROCEED }').
wam_instruction_to_c_literal(allocate, '{ .tag = INSTR_ALLOCATE }').
wam_instruction_to_c_literal(deallocate, '{ .tag = INSTR_DEALLOCATE }').
wam_instruction_to_c_literal(Instr, Code) :-
    format(atom(Code), '// TODO: ~w', [Instr]).

wam_instruction_to_c_literal(try_me_else(Label), LabelMap, Code) :-
    ( member(Label-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Code), '{ .tag = INSTR_TRY_ME_ELSE, .target_pc = ~w }', [TargetPC]).
wam_instruction_to_c_literal(retry_me_else(Label), LabelMap, Code) :-
    ( member(Label-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Code), '{ .tag = INSTR_RETRY_ME_ELSE, .target_pc = ~w }', [TargetPC]).
wam_instruction_to_c_literal(Instr, _, Code) :- wam_instruction_to_c_literal(Instr, Code).


c_value_literal(Atom, Lit) :- atom(Atom), format(atom(Lit), 'val_atom("~w")', [Atom]).
c_value_literal(Int, Lit) :- integer(Int), format(atom(Lit), 'val_int(~w)', [Int]).

c_reg_index(RegAtom, IsY, Idx) :-
    atom_chars(RegAtom, Chars),
    (   Chars = [Prefix|NumChars],
        (Prefix == 'a'; Prefix == 'x'; Prefix == 'A'; Prefix == 'X')
    ->  IsY = 0, catch(number_chars(Idx, NumChars), _, fail)
    ;   Chars = [Prefix|NumChars],
        (Prefix == 'y'; Prefix == 'Y')
    ->  IsY = 1, catch(number_chars(Idx, NumChars), _, fail)
    ;   throw(error(wam_c_target_error(unknown_register(RegAtom)), _))
    ).

% ============================================================================
% PHASE 2b: wam_predicate -> C Array
% ============================================================================

wam_line_to_c_instr(["get_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    c_value_literal(CC, Val), c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_CONSTANT, .val = ~w, .reg = ~w, .is_y_reg = ~w }', [Val, Idx, IsY]).
wam_line_to_c_instr(["get_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_GET_VARIABLE, .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["put_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    c_value_literal(CC, Val), c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_CONSTANT, .val = ~w, .reg = ~w, .is_y_reg = ~w }', [Val, Idx, IsY]).
wam_line_to_c_instr(["get_structure", F, Ai], Instr) :-
    clean_comma(F, CF), clean_comma(Ai, CAi),
    c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_STRUCTURE, .pred = "~w", .reg_ai = ~w, .is_y_ai = ~w }', [CF, Idx, IsY]).
wam_line_to_c_instr(["put_structure", F, Xn], Instr) :-
    clean_comma(F, CF), clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_STRUCTURE, .pred = "~w", .reg_xn = ~w, .is_y_xn = ~w }', [CF, Idx, IsY]).
wam_line_to_c_instr(["get_list", Ai], Instr) :-
    clean_comma(Ai, CAi),
    c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_LIST, .reg_ai = ~w, .is_y_ai = ~w }', [Idx, IsY]).
wam_line_to_c_instr(["put_list", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_LIST, .reg_xn = ~w, .is_y_xn = ~w }', [Idx, IsY]).
wam_line_to_c_instr(["unify_variable", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_VARIABLE, .reg_xn = ~w, .is_y_xn = ~w }', [Idx, IsY]).
wam_line_to_c_instr(["unify_value", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_VALUE, .reg_xn = ~w, .is_y_xn = ~w }', [Idx, IsY]).
wam_line_to_c_instr(["unify_constant", C], Instr) :-
    clean_comma(C, CC),
    c_value_literal(CC, Val),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_CONSTANT, .val = ~w }', [Val]).
wam_line_to_c_instr(["call", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(atom(Instr), '{ .tag = INSTR_CALL, .pred = "~w", .arity = ~w }', [CP, CN]).
wam_line_to_c_instr(["execute", P], Instr) :-
    clean_comma(P, CP),
    format(atom(Instr), '{ .tag = INSTR_EXECUTE, .pred = "~w" }', [CP]).
wam_line_to_c_instr(["try_me_else", L], LabelMap, Arity, Instr) :-
    clean_comma(L, CL),
    ( member(CL-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Instr), '{ .tag = INSTR_TRY_ME_ELSE, .target_pc = ~w, .arity = ~w }', [TargetPC, Arity]).
wam_line_to_c_instr(["retry_me_else", L], LabelMap, Arity, Instr) :-
    clean_comma(L, CL),
    ( member(CL-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Instr), '{ .tag = INSTR_RETRY_ME_ELSE, .target_pc = ~w, .arity = ~w }', [TargetPC, Arity]).
wam_line_to_c_instr(["trust_me"], _, '{ .tag = INSTR_TRUST_ME }').
wam_line_to_c_instr(["proceed"], _, '{ .tag = INSTR_PROCEED }').
wam_line_to_c_instr(["allocate"], _, '{ .tag = INSTR_ALLOCATE }').
wam_line_to_c_instr(["deallocate"], _, '{ .tag = INSTR_DEALLOCATE }').
wam_line_to_c_instr(Parts, _, Instr) :-
    atomic_list_concat(Parts, ' ', Combined),
    format(atom(Instr), '/* TODO: ~w */ {0}', [Combined]).

clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).

wam_lines_to_c_pass1([], _, []).
wam_lines_to_c_pass1([Line|Rest], PC, LabelMap) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == [] -> wam_lines_to_c_pass1(Rest, PC, LabelMap)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            LabelMap = [LabelName-PC|RestMap],
            wam_lines_to_c_pass1(Rest, PC, RestMap)
        ;   NPC is PC + 1,
            wam_lines_to_c_pass1(Rest, NPC, LabelMap)
        )
    ).

wam_lines_to_c_pass2([], _, _, _, []).
wam_lines_to_c_pass2([Line|Rest], PC, LabelMap, Arity, Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == [] -> wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            (   sub_string(LabelName, 0, 2, _, "L_")
            ->  wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, Instrs)
            ;   format(atom(PredReg), '    wam_register_predicate(state, "~w", ~w);', [LabelName, PC]),
                Instrs = [PredReg|RestInstrs],
                wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, RestInstrs)
            )
        ;   (   wam_line_to_c_instr(CleanParts, LabelMap, Arity, CInstr)
            ->  true
            ;   wam_line_to_c_instr(CleanParts, LabelMap, CInstr_NoArity)
            ->  CInstr = CInstr_NoArity
            ;   wam_line_to_c_instr(CleanParts, CInstr_NoMap)
            ->  CInstr = CInstr_NoMap
            ;   CInstr = '{0}'
            ),
            format(atom(InstrEntry), '    state->code[~w] = (Instruction)~w;', [PC, CInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_c_pass2(Rest, NPC, LabelMap, Arity, RestInstrs)
        )
    ).

compile_wam_predicate_to_c(PredIndicator, WamCode, _Options, CCode) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    atom_string(Pred, PredStr),
    atom_string(WamCode, WamStr),
    % Note: WamCode is a string generated by wam_target:compile_predicate_to_wam/3
    % (e.g. "get_constant a, A1\ncall foo/2, 2\n"), NOT a list of terms.
    % We parse it line-by-line into structural C literals.
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_c_pass1(Lines, 0, LabelMap),
    wam_lines_to_c_pass2(Lines, 0, LabelMap, Arity, InstrParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    
    % Calculate required code_size dynamically
    length(InstrParts, CodeSize),
    
    format(atom(CCode), 
'/* WAM-compiled predicate: ~w/~w */
void setup_~w_~w(WamState* state) {
    if (!state->code || state->code_size < ~w) {
        state->code_size = ~w;
        state->code = realloc(state->code, sizeof(Instruction) * state->code_size);
    }
~w
}', [PredStr, Arity, PredStr, Arity, CodeSize, CodeSize, InstrLiterals]).

% ============================================================================
% PHASE 3: step_wam/3 -> C switch statement
% ============================================================================

compile_step_wam_to_c(_Options, CCode) :-
    format(string(CCode),
'    bool step_wam(WamState* state, Instruction* instr) {
        switch (instr->tag) {
            case INSTR_GET_CONSTANT: {
                WamValue *cell = resolve_reg(state, instr->reg, instr->is_y_reg);
                if (val_is_unbound(*cell)) {
                    trail_binding(state, cell);
                    *cell = instr->val;
                    state->P++;
                    return true;
                } else if (val_equal(*cell, instr->val)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_GET_VARIABLE: {
                // Per WAM spec: copy A[Ai] to X[Xn] without trailing.
                // Trailing is only for mutations of already-bound cells.
                WamValue *cell_xn = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->reg_ai, instr->is_y_ai);
                *cell_xn = *cell_ai;
                state->P++;
                return true;
            }
            case INSTR_GET_VALUE: {
                WamValue *cell_xn = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->reg_ai, instr->is_y_ai);
                if (!wam_unify(state, cell_xn, cell_ai)) return false;
                state->P++;
                return true;
            }
            case INSTR_PUT_CONSTANT: {
                WamValue *cell = resolve_reg(state, instr->reg, instr->is_y_reg);
                *cell = instr->val;
                state->P++;
                return true;
            }
            case INSTR_PUT_VARIABLE: {
                WamValue ref = wam_make_ref(state);
                WamValue *cell_xn = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->reg_ai, instr->is_y_ai);
                *cell_xn = ref;
                *cell_ai = ref;
                state->P++;
                return true;
            }
            case INSTR_PUT_VALUE: {
                WamValue *cell_xn = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->reg_ai, instr->is_y_ai);
                *cell_ai = *cell_xn;
                state->P++;
                return true;
            }
            case INSTR_ALLOCATE: {
                int new_e_idx = state->E + 1;
                if (new_e_idx >= state->E_cap) {
                    state->E_cap = state->E_cap ? state->E_cap * 2 : WAM_INITIAL_CAP;
                    state->E_array = realloc(state->E_array, sizeof(EnvFrame) * state->E_cap);
                }
                state->E_array[new_e_idx].cp = state->CP;
                state->E_array[new_e_idx].saved_e = state->E;
                state->E = new_e_idx;
                state->P++;
                return true;
            }
            case INSTR_DEALLOCATE: {
                if (state->E >= 0) {
                    state->CP = state->E_array[state->E].cp;
                    state->E = state->E_array[state->E].saved_e;
                }
                state->P++;
                return true;
            }
            case INSTR_PROCEED: {
                state->P = state->CP;
                return true;
            }
            case INSTR_CALL: {
                state->CP = state->P + 1;
                int target = resolve_predicate(state, instr->pred);
                if (target >= 0) { state->P = target; return true; }
                return false;
            }
            case INSTR_EXECUTE: {
                int target = resolve_predicate(state, instr->pred);
                if (target >= 0) { state->P = target; return true; }
                return false;
            }
            case INSTR_TRY_ME_ELSE: {
                int target = instr->target_pc;
                int arity = instr->arity ? instr->arity : 32;
                push_choice_point(state, target, arity);
                state->P++;
                return true;
            }
            case INSTR_RETRY_ME_ELSE: {
                int target = instr->target_pc;
                ChoicePoint *cp = &state->B_array[state->B - 1];
                cp->next_pc = target;
                state->P++;
                return true;
            }
            case INSTR_TRUST_ME: {
                pop_choice_point(state);
                state->P++;
                return true;
            }
            case INSTR_GET_STRUCTURE: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->reg_ai, instr->is_y_ai));
                if (cell->tag == VAL_UNBOUND) {
                    trail_binding(state, cell);
                    WamValue s; s.tag = VAL_STR; s.data.ref_addr = state->H;
                    *cell = s;
                    
                    // Note: instr->pred includes the arity suffix (e.g. "foo/2"), which is stored as the functor atom
                    const char *slash = strchr(instr->pred, '/');
                    assert(slash != NULL && "Functor missing arity suffix");
                    int arity = strtol(slash + 1, NULL, 10);
                    
                    // Invariant contract: We proactively pre-reserve capacity for the functor + all arity arguments.
                    // Subsequent UNIFY_* instructions in write mode will push values sequentially via state->H++.
                    // While UNIFY_* instructions have their own single-slot capacity guards, this pre-allocation 
                    // ensures contiguous allocation and avoids multiple reallocs during the structure building sequence.
                    int required = state->H + 1 + arity;
                    if (required >= state->H_cap) {
                        if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                        while (required >= state->H_cap) state->H_cap *= 2;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->H_array[state->H] = val_atom(instr->pred);
                    state->H++;
                    state->mode = MODE_WRITE;
                } else if (cell->tag == VAL_STR) {
                    WamValue *f = &state->H_array[cell->data.ref_addr];
                    if (f->tag == VAL_ATOM && strcmp(f->data.atom, instr->pred) == 0) {
                        state->S = cell->data.ref_addr + 1;
                        state->mode = MODE_READ;
                    } else { return false; }
                } else { return false; }
                state->P++;
                return true;
            }
            case INSTR_PUT_STRUCTURE: {
                WamValue s; s.tag = VAL_STR; s.data.ref_addr = state->H;
                WamValue *cell = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                *cell = s;
                
                const char *slash = strchr(instr->pred, '/');
                assert(slash != NULL && "Functor missing arity suffix");
                int arity = strtol(slash + 1, NULL, 10);
                
                // Invariant contract: Proactively pre-reserve capacity for functor + arguments.
                // UNIFY_* instructions will sequentially append to H.
                int required = state->H + 1 + arity;
                if (required >= state->H_cap) {
                    if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                    while (required >= state->H_cap) state->H_cap *= 2;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->H_array[state->H] = val_atom(instr->pred);
                state->H++;
                state->mode = MODE_WRITE;
                state->P++;
                return true;
            }
            case INSTR_GET_LIST: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->reg_ai, instr->is_y_ai));
                if (cell->tag == VAL_UNBOUND) {
                    trail_binding(state, cell);
                    WamValue l; l.tag = VAL_LIST; l.data.ref_addr = state->H;
                    *cell = l;
                    
                    // Invariant contract: Proactively pre-reserve capacity for [head|tail].
                    // UNIFY_* instructions will sequentially append to H.
                    int required = state->H + 2;
                    if (required >= state->H_cap) {
                        if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                        while (required >= state->H_cap) state->H_cap *= 2;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->mode = MODE_WRITE;
                } else if (cell->tag == VAL_LIST) {
                    state->S = cell->data.ref_addr;
                    state->mode = MODE_READ;
                } else { return false; }
                state->P++;
                return true;
            }
            case INSTR_PUT_LIST: {
                WamValue l; l.tag = VAL_LIST; l.data.ref_addr = state->H;
                WamValue *cell = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                *cell = l;
                
                // Invariant contract: Proactively pre-reserve capacity for [head|tail].
                // UNIFY_* instructions will sequentially append to H.
                int required = state->H + 2;
                if (required >= state->H_cap) {
                    if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
                    while (required >= state->H_cap) state->H_cap *= 2;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->mode = MODE_WRITE;
                state->P++;
                return true;
            }
            case INSTR_UNIFY_VARIABLE: {
                WamValue *cell = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                if (state->mode == MODE_READ) {
                    *cell = state->H_array[state->S];
                    state->S++;
                } else {
                    // Note: wam_make_ref allocates an unbound cell in H_array and increments H,
                    // satisfying the 1-slot heap pre-reservation invariant.
                    WamValue ref = wam_make_ref(state);
                    *cell = ref;
                }
                state->P++;
                return true;
            }
            case INSTR_UNIFY_VALUE: {
                WamValue *cell = resolve_reg(state, instr->reg_xn, instr->is_y_xn);
                if (state->mode == MODE_READ) {
                    if (!wam_unify(state, cell, &state->H_array[state->S])) return false;
                    state->S++;
                } else {
                    if (state->H >= state->H_cap) {
                        state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->H_array[state->H] = *cell;
                    state->H++;
                }
                state->P++;
                return true;
            }
            case INSTR_UNIFY_CONSTANT: {
                if (state->mode == MODE_READ) {
                    WamValue *cell = wam_deref_ptr(state, &state->H_array[state->S]);
                    if (cell->tag == VAL_UNBOUND) {
                        trail_binding(state, cell);
                        *cell = instr->val;
                    } else if (!val_equal(*cell, instr->val)) {
                        return false;
                    }
                    state->S++;
                } else {
                    if (state->H >= state->H_cap) {
                        state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->H_array[state->H] = instr->val;
                    state->H++;
                }
                state->P++;
                return true;
            }
            default: return false;
        }
    }

    int wam_run(WamState* state) {
        // Outer backtracking loop
        while (state->P >= 0 && state->P < state->code_size) {
            Instruction* instr = &state->code[state->P];
            if (!step_wam(state, instr)) {
                if (state->B == 0) {
                    return WAM_HALT; // Failure, no choice points left
                }
                ChoicePoint* cp = &state->B_array[state->B - 1];
                restore_choice_point(state, cp); // Restores H, E, CP, A, unwinds TR
                state->P = cp->next_pc; // Explicitly jump to alternative
            }
        }
        return (state->P == WAM_HALT) ? 0 : WAM_ERR_OOB; // 0 on success (HALT), else OOB error
    }').

compile_wam_helpers_to_c(_Options, CCode) :-
    CCode = '#include "wam_runtime.h"\n\n// TODO: More C Helpers'.

compile_wam_runtime_to_c(Options, CCode) :-
    compile_step_wam_to_c(Options, StepCode),
    compile_wam_helpers_to_c(Options, HelpersCode),
    format(atom(CCode), "~w\n\n~w", [HelpersCode, StepCode]).
