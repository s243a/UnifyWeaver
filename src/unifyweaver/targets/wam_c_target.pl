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
wam_instruction_to_c_literal(call(P, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_CALL, .pred = "~w", .arity = ~w }', [P, N]).
wam_instruction_to_c_literal(execute(P), Code) :-
    format(atom(Code), '{ .tag = INSTR_EXECUTE, .pred = "~w" }', [P]).
wam_instruction_to_c_literal(try_me_else(Label), Code) :-
    format(atom(Code), '{ .tag = INSTR_TRY_ME_ELSE, .label = "~w" }', [Label]).
wam_instruction_to_c_literal(retry_me_else(Label), Code) :-
    format(atom(Code), '{ .tag = INSTR_RETRY_ME_ELSE, .label = "~w" }', [Label]).
wam_instruction_to_c_literal(trust_me, '{ .tag = INSTR_TRUST_ME }').
wam_instruction_to_c_literal(proceed, '{ .tag = INSTR_PROCEED }').
wam_instruction_to_c_literal(allocate, '{ .tag = INSTR_ALLOCATE }').
wam_instruction_to_c_literal(deallocate, '{ .tag = INSTR_DEALLOCATE }').
wam_instruction_to_c_literal(Instr, Code) :-
    format(atom(Code), '// TODO: ~w', [Instr]).

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
wam_line_to_c_instr(["call", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(atom(Instr), '{ .tag = INSTR_CALL, .pred = "~w", .arity = ~w }', [CP, CN]).
wam_line_to_c_instr(["execute", P], Instr) :-
    clean_comma(P, CP),
    format(atom(Instr), '{ .tag = INSTR_EXECUTE, .pred = "~w" }', [CP]).
wam_line_to_c_instr(["try_me_else", L], Instr) :-
    clean_comma(L, CL),
    format(atom(Instr), '{ .tag = INSTR_TRY_ME_ELSE, .label = "~w" }', [CL]).
wam_line_to_c_instr(["retry_me_else", L], Instr) :-
    clean_comma(L, CL),
    format(atom(Instr), '{ .tag = INSTR_RETRY_ME_ELSE, .label = "~w" }', [CL]).
wam_line_to_c_instr(["trust_me"], '{ .tag = INSTR_TRUST_ME }').
wam_line_to_c_instr(["proceed"], '{ .tag = INSTR_PROCEED }').
wam_line_to_c_instr(["allocate"], '{ .tag = INSTR_ALLOCATE }').
wam_line_to_c_instr(["deallocate"], '{ .tag = INSTR_DEALLOCATE }').
wam_line_to_c_instr(Parts, Instr) :-
    atomic_list_concat(Parts, ' ', Combined),
    format(atom(Instr), '/* TODO: ~w */ {0}', [Combined]).

clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).

wam_lines_to_c([], _, [], []).
wam_lines_to_c([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_c(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(atom(LabelInsert), '    state->label_names[state->label_count] = "~w"; state->label_pcs[state->label_count++] = ~w;', [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_c(Rest, PC, Instrs, RestLabels)
        ;   wam_line_to_c_instr(CleanParts, CInstr),
            format(atom(InstrEntry), '    state->code[~w] = (Instruction)~w;', [PC, CInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_c(Rest, NPC, RestInstrs, Labels)
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
    wam_lines_to_c(Lines, 0, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    atomic_list_concat(LabelParts, '\n', LabelLiterals),
    format(atom(CCode), 
'/* WAM-compiled predicate: ~w/~w */
void setup_~w_~w(WamState* state) {
    if (!state->code) {
        state->code_size = 1000; // placeholder size
        state->code = malloc(sizeof(Instruction) * state->code_size);
        state->label_cap = 100;
        state->label_names = malloc(sizeof(char*) * state->label_cap);
        state->label_pcs = malloc(sizeof(int) * state->label_cap);
        state->label_count = 0;
    }
~w
~w
}', [PredStr, Arity, PredStr, Arity, InstrLiterals, LabelLiterals]).

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
                if (val_equal(*cell_xn, *cell_ai)) {
                    state->P++;
                    return true;
                }
                // TODO: full wam_unify() fallback
                return false;
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
                int target = resolve_label(state, instr->pred);
                if (target >= 0) { state->P = target; return true; }
                return false;
            }
            case INSTR_EXECUTE: {
                int target = resolve_label(state, instr->pred);
                if (target >= 0) { state->P = target; return true; }
                return false;
            }
            case INSTR_TRY_ME_ELSE: {
                int target = resolve_label(state, instr->label);
                // TODO: extract actual arity from context; defaulting to 32
                push_choice_point(state, target, 32);
                state->P++;
                return true;
            }
            case INSTR_RETRY_ME_ELSE: {
                int target = resolve_label(state, instr->label);
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
            case INSTR_GET_STRUCTURE:
            case INSTR_GET_LIST:
            case INSTR_PUT_STRUCTURE:
            case INSTR_PUT_LIST:
            case INSTR_UNIFY_VARIABLE:
            case INSTR_UNIFY_VALUE:
            case INSTR_UNIFY_CONSTANT: {
                // TODO: Full heap and structure allocation/unification
                return false;
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
        return (state->P == WAM_HALT) ? 0 : (WAM_HALT - 1); // 0 on success (HALT), else OOB error
    }').

compile_wam_helpers_to_c(_Options, CCode) :-
    CCode = '#include "wam_runtime.h"\n\n// TODO: More C Helpers'.

compile_wam_runtime_to_c(Options, CCode) :-
    compile_step_wam_to_c(Options, StepCode),
    compile_wam_helpers_to_c(Options, HelpersCode),
    format(atom(CCode), "~w\n\n~w", [HelpersCode, StepCode]).
