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
    c_value_literal(C, Val), c_reg_index(Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_GET_CONSTANT, .val = ~w, .reg = ~w }', [Val, Idx]).
wam_instruction_to_c_literal(get_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, XIdx), c_reg_index(Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VARIABLE, .reg_xn = ~w, .reg_ai = ~w }', [XIdx, AIdx]).
wam_instruction_to_c_literal(get_value(Xn, Ai), Code) :-
    c_reg_index(Xn, XIdx), c_reg_index(Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VALUE, .reg_xn = ~w, .reg_ai = ~w }', [XIdx, AIdx]).
wam_instruction_to_c_literal(put_constant(C, Ai), Code) :-
    c_value_literal(C, Val), c_reg_index(Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_PUT_CONSTANT, .val = ~w, .reg = ~w }', [Val, Idx]).
wam_instruction_to_c_literal(put_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, XIdx), c_reg_index(Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VARIABLE, .reg_xn = ~w, .reg_ai = ~w }', [XIdx, AIdx]).
wam_instruction_to_c_literal(put_value(Xn, Ai), Code) :-
    c_reg_index(Xn, XIdx), c_reg_index(Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VALUE, .reg_xn = ~w, .reg_ai = ~w }', [XIdx, AIdx]).
wam_instruction_to_c_literal(call(P, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_CALL, .pred = "~w", .arity = ~w }', [P, N]).
wam_instruction_to_c_literal(execute(P), Code) :-
    format(atom(Code), '{ .tag = INSTR_EXECUTE, .pred = "~w" }', [P]).
wam_instruction_to_c_literal(proceed, '{ .tag = INSTR_PROCEED }').
wam_instruction_to_c_literal(allocate, '{ .tag = INSTR_ALLOCATE }').
wam_instruction_to_c_literal(deallocate, '{ .tag = INSTR_DEALLOCATE }').
% ... add other instructions as needed ...
wam_instruction_to_c_literal(Instr, Code) :-
    format(atom(Code), '// TODO: ~w', [Instr]).

c_value_literal(Atom, Lit) :- atom(Atom), format(atom(Lit), 'val_atom("~w")', [Atom]).
c_value_literal(Int, Lit) :- integer(Int), format(atom(Lit), 'val_int(~w)', [Int]).
c_reg_index(RegAtom, Idx) :- 
    atom_chars(RegAtom, [_|NumChars]), 
    number_chars(Idx, NumChars).

% ============================================================================
% PHASE 2b: wam_predicate -> C Array
% ============================================================================

compile_wam_predicate_to_c(PredIndicator, WamCode, _Options, CCode) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    atom_string(Pred, PredStr),
    atom_string(WamCode, WamStr),
    % Parsing lines is a placeholder, actual impl should use WAM terms
    format(atom(CCode), 
'// WAM-compiled predicate: ~w/~w
void ~w_~w(WamState* state) {
    // TODO: Instruction array and label initialization
    /*
~w
    */
}', [PredStr, Arity, PredStr, Arity, WamStr]).

% ============================================================================
% PHASE 3: step_wam/3 -> C switch statement
% ============================================================================

compile_step_wam_to_c(_Options, CCode) :-
    format(string(CCode),
'    bool step(WamState* state, Instruction* instr) {
        switch (instr->tag) {
            case INSTR_GET_CONSTANT: {
                WamValue val = state->A[instr->reg];
                if (val_is_unbound(val)) {
                    trail_binding(state, &state->A[instr->reg]);
                    state->A[instr->reg] = instr->val;
                    state->P++;
                    return true;
                } else if (val_equal(val, instr->val)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_PROCEED: {
                state->P = state->CP;
                return true;
            }
            // TODO: generate other match arms
            default: return false;
        }
    }').

compile_wam_helpers_to_c(_Options, CCode) :-
    CCode = '// TODO: C Helpers'.

compile_wam_runtime_to_c(Options, CCode) :-
    compile_step_wam_to_c(Options, StepCode),
    compile_wam_helpers_to_c(Options, HelpersCode),
    format(atom(CCode), "~w\n\n~w", [HelpersCode, StepCode]).
