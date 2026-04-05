:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_c_target.pl - WAM-to-C Transpilation Target
%
% Transpiles WAM runtime to C source code.
% Phase 2: WAM instructions → switch cases
% Phase 3: Helper functions → C function bodies
%
% WAM state uses a struct with dynamic arrays for heap/trail/stack
% and a simple string-keyed register map. Values are tagged unions.

:- module(wam_c_target, [
    compile_step_wam_to_c/2,       % +Options, -Code
    compile_wam_helpers_to_c/2,     % +Options, -Code
    compile_wam_runtime_to_c/2,     % +Options, -Code
    compile_wam_predicate_to_c/4,   % +Pred/Arity, +WamCode, +Options, -Code
    write_wam_c_project/3,          % +Predicates, +Options, +ProjectDir
    wam_c_case/2                    % +InstrName, -CCode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../bindings/c_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

:- discontiguous wam_c_case/2.

% ============================================================================
% PHASE 2: WAM Instructions → C switch cases
% ============================================================================

%% compile_step_wam_to_c(+Options, -Code)
%  Generates the wam_step() function body with a switch on instruction tag.
compile_step_wam_to_c(_Options, Code) :-
    findall(Case, compile_c_step_case(Case), Cases),
    atomic_list_concat(Cases, '\n\n', CasesCode),
    format(string(Code),
'/* Execute a single WAM instruction. Returns 1 on success, 0 on failure. */
int wam_step(WamState *state, WamInstr *instr) {
    switch (instr->tag) {
~w

    default:
        return 0;
    }
}', [CasesCode]).

compile_c_step_case(CaseCode) :-
    wam_c_case(InstrName, BodyCode),
    upcase_atom(InstrName, Upper),
    format(string(CaseCode), '    case WAM_~w: {\n~w\n    }', [Upper, BodyCode]).

upcase_atom(Atom, Upper) :-
    atom_string(Atom, S),
    string_upper(S, U),
    atom_string(Upper, U).

% --- Head Unification Instructions ---

wam_c_case(get_constant,
'        WamValue *val = wam_reg_get(state, instr->arg1);
        if (val && wam_values_equal(val, instr->val)) {
            state->pc++;
            return 1;
        }
        if (val && val->tag == VAL_UNBOUND) {
            wam_trail_binding(state, instr->arg1);
            wam_reg_put(state, instr->arg1, instr->val);
            state->pc++;
            return 1;
        }
        return 0;').

wam_c_case(get_variable,
'        WamValue *val = wam_reg_get(state, instr->arg1);
        wam_trail_binding(state, instr->arg2);
        wam_reg_put(state, instr->arg2, val);
        state->pc++;
        return 1;').

wam_c_case(get_value,
'        WamValue *val_a = wam_reg_get(state, instr->arg1);
        WamValue *val_x = wam_reg_get(state, instr->arg2);
        if (wam_unify(state, val_a, val_x)) {
            state->pc++;
            return 1;
        }
        return 0;').

wam_c_case(get_structure,
'        WamValue *val = wam_reg_get(state, instr->arg2);
        if (val && val->tag == VAL_UNBOUND) {
            int addr = state->heap_size;
            wam_heap_push(state, wam_make_str(instr->arg1));
            wam_trail_binding(state, instr->arg2);
            wam_reg_put(state, instr->arg2, wam_make_ref(addr));
            int arity = wam_parse_functor_arity(instr->arg1);
            wam_push_write_ctx(state, arity);
            state->pc++;
            return 1;
        }
        if (val && val->tag == VAL_REF) {
            int addr = val->data.ref_addr;
            WamValue *entry = wam_heap_get(state, addr);
            if (entry && entry->tag == VAL_STR && strcmp(entry->data.str_functor, instr->arg1) == 0) {
                int arity = wam_parse_functor_arity(instr->arg1);
                wam_push_unify_ctx(state, state->heap, addr + 1, arity);
                state->pc++;
                return 1;
            }
        }
        return 0;').

wam_c_case(get_list,
'        WamValue *val = wam_reg_get(state, instr->arg1);
        if (val && val->tag == VAL_UNBOUND) {
            int addr = state->heap_size;
            wam_heap_push(state, wam_make_str("./2"));
            wam_trail_binding(state, instr->arg1);
            wam_reg_put(state, instr->arg1, wam_make_ref(addr));
            wam_push_write_ctx(state, 2);
            state->pc++;
            return 1;
        }
        if (val && val->tag == VAL_REF) {
            int addr = val->data.ref_addr;
            WamValue *entry = wam_heap_get(state, addr);
            if (entry && entry->tag == VAL_STR && strcmp(entry->data.str_functor, "./2") == 0) {
                wam_push_unify_ctx(state, state->heap, addr + 1, 2);
                state->pc++;
                return 1;
            }
        }
        return 0;').

% --- Body Construction Instructions ---

wam_c_case(put_constant,
'        wam_reg_put(state, instr->arg1, instr->val);
        state->pc++;
        return 1;').

wam_c_case(put_variable,
'        char name[32];
        snprintf(name, sizeof(name), "_V%d", state->pc);
        WamValue *fresh = wam_make_unbound(name);
        wam_trail_binding(state, instr->arg1);
        wam_reg_put(state, instr->arg1, fresh);
        wam_reg_put(state, instr->arg2, fresh);
        state->pc++;
        return 1;').

wam_c_case(put_value,
'        WamValue *val = wam_reg_get(state, instr->arg1);
        wam_trail_binding(state, instr->arg2);
        wam_reg_put(state, instr->arg2, val);
        state->pc++;
        return 1;').

wam_c_case(put_structure,
'        int addr = state->heap_size;
        wam_heap_push(state, wam_make_str(instr->arg1));
        wam_trail_binding(state, instr->arg2);
        wam_reg_put(state, instr->arg2, wam_make_ref(addr));
        int arity = wam_parse_functor_arity(instr->arg1);
        wam_push_write_ctx(state, arity);
        state->pc++;
        return 1;').

wam_c_case(put_list,
'        int addr = state->heap_size;
        wam_heap_push(state, wam_make_str("./2"));
        wam_trail_binding(state, instr->arg1);
        wam_reg_put(state, instr->arg1, wam_make_ref(addr));
        wam_push_write_ctx(state, 2);
        state->pc++;
        return 1;').

wam_c_case(set_variable,
'        int addr = state->heap_size;
        char name[32];
        snprintf(name, sizeof(name), "_H%d", addr);
        WamValue *fresh = wam_make_unbound(name);
        wam_heap_push(state, fresh);
        wam_reg_put(state, instr->arg1, fresh);
        state->pc++;
        return 1;').

wam_c_case(set_value,
'        WamValue *val = wam_reg_get(state, instr->arg1);
        wam_heap_push(state, val);
        state->pc++;
        return 1;').

wam_c_case(set_constant,
'        wam_heap_push(state, instr->val);
        state->pc++;
        return 1;').

% --- Unification Instructions ---

wam_c_case(unify_variable,
'        return wam_step_unify_variable(state, instr->arg1);').

wam_c_case(unify_value,
'        return wam_step_unify_value(state, instr->arg1);').

wam_c_case(unify_constant,
'        return wam_step_unify_constant(state, instr->val);').

% --- Control Flow Instructions ---

wam_c_case(call,
'        state->cp = state->pc + 1;
        int target = wam_resolve_label(state, instr->arg1);
        if (target < 0) return 0;
        state->pc = target;
        return 1;').

wam_c_case(execute,
'        int target = wam_resolve_label(state, instr->arg1);
        if (target < 0) return 0;
        state->pc = target;
        return 1;').

wam_c_case(proceed,
'        state->pc = state->cp;
        return 1;').

wam_c_case(allocate,
'        wam_allocate_env(state);
        state->pc++;
        return 1;').

wam_c_case(deallocate,
'        wam_deallocate_env(state);
        state->pc++;
        return 1;').

% --- Choice Point Instructions ---

wam_c_case(try_me_else,
'        int target = wam_resolve_label(state, instr->arg1);
        wam_push_choice_point(state, target);
        state->pc++;
        return 1;').

wam_c_case(retry_me_else,
'        int target = wam_resolve_label(state, instr->arg1);
        wam_update_choice_point(state, target);
        state->pc++;
        return 1;').

wam_c_case(trust_me,
'        wam_pop_choice_point(state);
        state->pc++;
        return 1;').

% --- Indexing Instructions ---

wam_c_case(switch_on_constant,
'        return wam_step_switch_on_constant(state, instr);').

% --- Builtin Instructions ---

wam_c_case(builtin_call,
'        return wam_execute_builtin(state, instr->arg1, instr->arity);').

% ============================================================================
% PHASE 3: Helper functions → C function bodies
% ============================================================================

%% compile_wam_helpers_to_c(+Options, -Code)
compile_wam_helpers_to_c(_Options, Code) :-
    compile_run_loop_to_c(RunCode),
    compile_backtrack_to_c(BTCode),
    compile_unwind_trail_to_c(UnwindCode),
    compile_unify_to_c(UnifyCode),
    compile_utility_helpers_to_c(UtilCode),
    atomic_list_concat([RunCode, '\n\n', BTCode, '\n\n', UnwindCode, '\n\n',
                        UnifyCode, '\n\n', UtilCode], Code).

compile_run_loop_to_c(Code) :-
    format(string(Code),
'/* Main fetch-step-backtrack loop. Returns 1 on success, 0 on failure. */
int wam_run(WamState *state) {
    while (state->pc != WAM_HALT) {
        if (state->pc < 1 || state->pc > state->code_size) return 0;
        WamInstr *instr = &state->code[state->pc - 1];
        if (wam_step(state, instr)) continue;
        if (wam_backtrack(state)) continue;
        return 0;
    }
    return 1;
}', []).

compile_backtrack_to_c(Code) :-
    format(string(Code),
'/* Restore from most recent choice point. Returns 1 on success, 0 on failure. */
int wam_backtrack(WamState *state) {
    if (state->cp_size <= 0) return 0;
    wam_restore_choice_point(state);
    return 1;
}', []).

compile_unwind_trail_to_c(Code) :-
    format(string(Code),
'/* Undo register bindings back to trail mark. */
void wam_unwind_trail(WamState *state, int mark) {
    while (state->trail_size > mark) {
        state->trail_size--;
        TrailEntry *entry = &state->trail[state->trail_size];
        if (entry->old_val == NULL) {
            wam_reg_delete(state, entry->key);
        } else {
            wam_reg_put(state, entry->key, entry->old_val);
        }
    }
}', []).

compile_unify_to_c(Code) :-
    format(string(Code),
'/* Unify two WAM values. Returns 1 on success, 0 on failure. */
int wam_unify(WamState *state, WamValue *v1, WamValue *v2) {
    if (v1 == v2) return 1;
    if (wam_values_equal(v1, v2)) return 1;
    if (v1 && v1->tag == VAL_UNBOUND) {
        wam_trail_binding(state, v1->data.unbound_name);
        wam_reg_put(state, v1->data.unbound_name, v2);
        return 1;
    }
    if (v2 && v2->tag == VAL_UNBOUND) {
        wam_trail_binding(state, v2->data.unbound_name);
        wam_reg_put(state, v2->data.unbound_name, v1);
        return 1;
    }
    return 0;
}', []).

compile_utility_helpers_to_c(Code) :-
    % Use char_code to avoid quoting issues with '/' in C char literal
    format(string(Part1),
'/* Trail a register binding before overwriting. */
void wam_trail_binding(WamState *state, const char *key) {
    if (state->trail_size >= state->trail_cap) {
        state->trail_cap = state->trail_cap ? state->trail_cap * 2 : 64;
        state->trail = realloc(state->trail, sizeof(TrailEntry) * state->trail_cap);
    }
    TrailEntry *entry = &state->trail[state->trail_size++];
    entry->key = strdup(key);
    entry->old_val = wam_reg_get(state, key);
}

/* Parse arity from functor name like "foo/2". Returns 0 if no slash. */
int wam_parse_functor_arity(const char *fn) {
    const char *slash = strrchr(fn, 47); /* 47 = ASCII slash */
    if (!slash) return 0;
    return atoi(slash + 1);
}

/* Resolve a label to a PC value. Returns -1 if not found. */
int wam_resolve_label(WamState *state, const char *label) {
    for (int i = 0; i < state->label_count; i++) {
        if (strcmp(state->label_names[i], label) == 0)
            return state->label_pcs[i];
    }
    return -1;
}', []),
    format(string(Part2),
'/* Execute a builtin predicate. */
int wam_execute_builtin(WamState *state, const char *op, int arity) {
    if (strcmp(op, "is/2") == 0) {
        WamValue *expr = wam_reg_get(state, "A2");
        int result = wam_eval_arith(state, expr);
        WamValue *lhs = wam_reg_get(state, "A1");
        if (lhs && lhs->tag == VAL_UNBOUND) {
            wam_trail_binding(state, "A1");
            wam_reg_put(state, "A1", wam_make_int(result));
            state->pc++;
            return 1;
        }
        if (lhs && lhs->tag == VAL_INT && lhs->data.integer == result) {
            state->pc++;
            return 1;
        }
        return 0;
    }
    return 0; /* unknown builtin */
}

/* Evaluate an arithmetic expression. */
int wam_eval_arith(WamState *state, WamValue *val) {
    if (!val) return 0;
    if (val->tag == VAL_INT) return val->data.integer;
    return 0; /* compound arithmetic not yet implemented */
}', []),
    atomic_list_concat([Part1, '\n\n', Part2], Code).

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3
% ============================================================================

%% compile_wam_runtime_to_c(+Options, -Code)
compile_wam_runtime_to_c(Options, Code) :-
    compile_step_wam_to_c(Options, StepCode),
    compile_wam_helpers_to_c(Options, HelpersCode),
    format(string(Code),
'/* WAM Runtime - Generated by UnifyWeaver */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wam_runtime.h"

~w

~w
', [HelpersCode, StepCode]).

% ============================================================================
% PREDICATE WRAPPER
% ============================================================================

%% compile_wam_predicate_to_c(+Pred/Arity, +WamCode, +Options, -Code)
compile_wam_predicate_to_c(Pred/Arity, WamCode, _Options, Code) :-
    atom_string(Pred, PredStr),
    wam_code_to_c_instructions(WamCode, InstrLiterals, LabelLiterals),
    format(string(Code),
'/* WAM-compiled predicate: ~w/~w */
void wam_setup_~w(WamState *state) {
    /* Instructions */
~w
    /* Labels */
~w
}', [PredStr, Arity, PredStr, InstrLiterals, LabelLiterals]).

wam_code_to_c_instructions(WamCode, InstrLiterals, LabelLiterals) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_c(Lines, 1, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    atomic_list_concat(LabelParts, '\n', LabelLiterals).

wam_lines_to_c([], _, [], []).
wam_lines_to_c([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_c(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelInsert),
                '    wam_add_label(state, "~w", ~w);', [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_c(Rest, PC, Instrs, RestLabels)
        ;   wam_line_to_c_instr(CleanParts, CInstr),
            format(string(InstrEntry), '    ~w', [CInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_c(Rest, NPC, RestInstrs, Labels)
        )
    ).

wam_line_to_c_instr(["get_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Instr), 'wam_emit_get_constant(state, "~w", "~w");', [CC, CAi]).
wam_line_to_c_instr(["get_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Instr), 'wam_emit_get_variable(state, "~w", "~w");', [CXn, CAi]).
wam_line_to_c_instr(["put_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Instr), 'wam_emit_put_constant(state, "~w", "~w");', [CC, CAi]).
wam_line_to_c_instr(["proceed"], 'wam_emit_proceed(state);').
wam_line_to_c_instr(["call", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(string(Instr), 'wam_emit_call(state, "~w", ~w);', [CP, CN]).
wam_line_to_c_instr(["execute", P], Instr) :-
    clean_comma(P, CP),
    format(string(Instr), 'wam_emit_execute(state, "~w");', [CP]).
wam_line_to_c_instr(["allocate", N], Instr) :-
    clean_comma(N, CN),
    format(string(Instr), 'wam_emit_allocate(state, ~w);', [CN]).
wam_line_to_c_instr(["deallocate"], 'wam_emit_deallocate(state);').
wam_line_to_c_instr(["builtin_call", Op, Ar], Instr) :-
    clean_comma(Op, COp), clean_comma(Ar, CAr),
    format(string(Instr), 'wam_emit_builtin_call(state, "~w", ~w);', [COp, CAr]).
wam_line_to_c_instr(Parts, Instr) :-
    atomic_list_concat(Parts, ' ', Combined),
    format(string(Instr), '/* ~w */', [Combined]).

clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).

% ============================================================================
% PROJECT GENERATION
% ============================================================================

%% write_wam_c_project(+Predicates, +Options, +ProjectDir)
write_wam_c_project(Predicates, Options, ProjectDir) :-
    option(program_name(ProgName), Options, 'wam_program'),
    make_directory_path(ProjectDir),
    % Generate runtime .c
    compile_wam_runtime_to_c(Options, RuntimeCode),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    open(RuntimePath, write, RS),
    write(RS, RuntimeCode),
    close(RS),
    % Generate predicate wrappers
    forall(
        member(Pred/Arity-WamCode, Predicates),
        (   compile_wam_predicate_to_c(Pred/Arity, WamCode, Options, PredCode),
            atom_string(Pred, PredStr),
            format(atom(PredFile), '~w.c', [PredStr]),
            directory_file_path(ProjectDir, PredFile, PredPath),
            open(PredPath, write, PS),
            write(PS, PredCode),
            close(PS)
        )
    ),
    % Generate Makefile
    format(string(MakeCode),
'CC = gcc
CFLAGS = -O2 -Wall -std=c99
TARGET = ~w
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
\t$(CC) $(CFLAGS) -o $@ $^

%%.o: %%.c wam_runtime.h
\t$(CC) $(CFLAGS) -c $<

clean:
\trm -f $(OBJS) $(TARGET)

.PHONY: clean
', [ProgName]),
    directory_file_path(ProjectDir, 'Makefile', MakePath),
    open(MakePath, write, MkS),
    write(MkS, MakeCode),
    close(MkS).
