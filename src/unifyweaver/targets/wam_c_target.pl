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
    detect_kernels/2,                 % +Predicates, -DetectedKernels
    generate_setup_detected_kernels_c/2, % +DetectedKernels, -CCode
    plan_wam_c_lowered_helpers/4,     % +Predicates, +Options, +DetectedKeys, -Plans
    write_wam_c_project/3             % +Predicates, +Options, +ProjectDir
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(pairs), [pairs_keys/2]).

:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../bindings/c_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/recursive_kernel_detection', [
    detect_recursive_kernel/4
]).

% ============================================================================
% PHASE 4: Hybrid Module Assembly
% ============================================================================

%% write_wam_c_project(+Predicates, +Options, +ProjectDir)
%  Generates a full C project for the given predicates.
write_wam_c_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    detect_kernels_for_options(Predicates, Options, DetectedKernels),
    % Generate runtime .c and .h files
    compile_wam_runtime_to_c(Options, RuntimeCode),
    directory_file_path(ProjectDir, 'wam_runtime.c', RuntimePath),
    write_file(RuntimePath, RuntimeCode),

    % Compile predicates and generate lib.c
    pairs_keys(DetectedKernels, DetectedKeys),
    plan_wam_c_lowered_helpers(Predicates, Options, DetectedKeys, LoweredPlans),
    maybe_report_wam_c_lowered_helper_plan(Options, LoweredPlans),
    compile_lowered_helpers_for_project(LoweredPlans, LoweredKeys, LoweredCode, SetupLoweredCode),
    render_wam_c_lowered_helper_plan(LoweredPlans, LoweredPlanCode),
    generate_setup_detected_kernels_c(DetectedKernels, SetupKernelCode),
    compile_predicates_for_project(Predicates,
                                   [detected_kernel_keys(DetectedKeys),
                                    lowered_helper_keys(LoweredKeys)|Options],
                                   PredicatesCode),
    format(atom(LibCode), '#include "wam_runtime.h"~n~n~w~n~n~w~n~n~w~n~n~w',
           [LoweredPlanCode, SetupKernelCode, LoweredCode, SetupLoweredCode]),
    format(atom(LibCodeWithPredicates), '~w~n~n~w', [LibCode, PredicatesCode]),
    directory_file_path(ProjectDir, 'lib.c', LibPath),
    write_file(LibPath, LibCodeWithPredicates),

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
    format(atom(Key), '~w/~w', [Pred, Arity]),
    option(detected_kernel_keys(DetectedKeys), Options, []),
    option(lowered_helper_keys(LoweredKeys), Options, []),
    (   memberchk(Key, DetectedKeys)
    ->  format(atom(WamCode), '~w/~w:\n    call_foreign ~w, ~w\n    proceed',
               [Pred, Arity, Key, Arity]),
        compile_wam_predicate_to_c(Module:Pred/Arity, WamCode, Options, PredCode)
    ;   memberchk(Key, LoweredKeys)
    ->  format(atom(WamCode), '~w/~w:\n    call_foreign ~w, ~w\n    proceed',
               [Pred, Arity, Key, Arity]),
        compile_wam_predicate_to_c(Module:Pred/Arity, WamCode, Options, PredCode)
    ;   wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode)
    ->  compile_wam_predicate_to_c(Module:Pred/Arity, WamCode, Options, PredCode)
    ;   format(atom(PredCode), '// ~w/~w: compilation failed', [Pred, Arity])
    ),
    compile_predicates_for_project(Rest, Options, RestCode),
    format(atom(Code), '~w\n\n~w', [PredCode, RestCode]).

predicate_indicator_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
predicate_indicator_parts(Pred/Arity, user, Pred, Arity).

detect_kernels_for_options(Predicates, Options, DetectedKernels) :-
    (   option(no_kernels(true), Options)
    ->  DetectedKernels = [],
        format(user_error, '[WAM-C] kernel detection suppressed~n', [])
    ;   detect_kernels(Predicates, DetectedKernels),
        (   DetectedKernels \= []
        ->  pairs_keys(DetectedKernels, DetectedKeys),
            format(user_error, '[WAM-C] detected kernels: ~w~n', [DetectedKeys])
        ;   true
        )
    ).

%% detect_kernels(+Predicates, -DetectedKernels)
%  Run the shared recursive-kernel detector over project predicates.
detect_kernels([], []).
detect_kernels([PI|Rest], Kernels) :-
    predicate_indicator_parts(PI, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses \= [],
        detect_recursive_kernel(Pred, Arity, Clauses, Kernel),
        wam_c_supported_kernel(Kernel)
    ->  format(atom(Key), '~w/~w', [Pred, Arity]),
        Kernels = [Key-Kernel|RestKernels]
    ;   Kernels = RestKernels
    ),
    detect_kernels(Rest, RestKernels).

wam_c_supported_kernel(recursive_kernel(category_ancestor, _Pred, _ConfigOps)).

%% generate_setup_detected_kernels_c(+DetectedKernels, -CCode)
%  Emit C startup wiring for detected kernels. This function registers only
%  native handlers; callers still decide when to load/register fact sources.
generate_setup_detected_kernels_c([], Code) :- !,
    Code = 'void setup_detected_wam_c_kernels(WamState* state) {\n    (void)state;\n}'.
generate_setup_detected_kernels_c(DetectedKernels, Code) :-
    maplist(wam_c_kernel_registration_line, DetectedKernels, Lines),
    atomic_list_concat(Lines, '\n', Body),
    format(atom(Code),
           'void setup_detected_wam_c_kernels(WamState* state) {\n~w\n}',
           [Body]).

wam_c_kernel_registration_line(Key-recursive_kernel(category_ancestor, _Pred, ConfigOps), Line) :-
    wam_c_kernel_max_depth(ConfigOps, MaxDepth),
    format(atom(Line),
           '    wam_register_category_ancestor_kernel(state, "~w", ~w);',
           [Key, MaxDepth]).

wam_c_kernel_max_depth(ConfigOps, MaxDepth) :-
    (   member(max_depth(MaxDepth0), ConfigOps),
        integer(MaxDepth0),
        MaxDepth0 > 0
    ->  MaxDepth = MaxDepth0
    ;   MaxDepth = 10
    ).

% ============================================================================
% Prototype native lowered helpers
% ============================================================================

plan_wam_c_lowered_helpers(Predicates, Options, DetectedKeys, Plans) :-
    (   option(lowered_helpers(true), Options)
    ->  maplist(wam_c_predicate_key, Predicates, AvailableKeys),
        maplist(plan_wam_c_lowered_helper(DetectedKeys, AvailableKeys), Predicates, Plans)
    ;   maplist(plan_wam_c_lowered_helper_disabled, Predicates, Plans)
    ).

wam_c_predicate_key(PredIndicator, Key) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]).

plan_wam_c_lowered_helper(DetectedKeys, AvailableKeys, PredIndicator, Plan) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    (   memberchk(Key, DetectedKeys)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, interpreted, detected_kernel)
    ;   lowered_fact_helper_rows(PredIndicator, Rows)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, fact_only(Rows))
    ;   lowered_alias_helper_call(PredIndicator, AvailableKeys, TargetKey, TargetArity)
    ->  Plan = wam_c_lowered_helper_plan(Key, PredIndicator, lowered, alias_call(TargetKey, TargetArity))
    ;   lowered_fact_helper_rejection_reason(PredIndicator, Reason),
        Plan = wam_c_lowered_helper_plan(Key, PredIndicator, rejected, Reason)
    ).

plan_wam_c_lowered_helper_disabled(PredIndicator, Plan) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    Plan = wam_c_lowered_helper_plan(Key, PredIndicator, interpreted, lowering_disabled).

lowered_fact_helper_rejection_reason(PredIndicator, non_fact_clause) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    user:clause(Head, Body),
    Body \== true,
    !.
lowered_fact_helper_rejection_reason(PredIndicator, unsupported_fact_arguments) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    user:clause(Head, true),
    !.
lowered_fact_helper_rejection_reason(_, no_clauses).

compile_lowered_helpers_for_project(Plans, LoweredKeys, Code, SetupCode) :-
    findall(Key-CodePart-SetupLine,
            (   member(wam_c_lowered_helper_plan(Key, PredIndicator, lowered, fact_only(Rows)), Plans),
                lowered_helper_for_predicate(fact_only(Rows), PredIndicator, Key, CodePart, SetupLine)
            ),
            FactEntries),
    findall(Key-CodePart-SetupLine,
            (   member(wam_c_lowered_helper_plan(Key, PredIndicator, lowered, alias_call(TargetKey, TargetArity)), Plans),
                lowered_helper_for_predicate(alias_call(TargetKey, TargetArity), PredIndicator, Key, CodePart, SetupLine)
            ),
            AliasEntries),
    append(FactEntries, AliasEntries, Entries),
    findall(K, member(K-_-_, Entries), LoweredKeys),
    findall(C, member(_-C-_, Entries), Codes),
    findall(S, member(_-_-S, Entries), SetupLines),
    atomic_list_concat(Codes, '\n\n', Code),
    (   SetupLines = []
    ->  SetupCode = 'void setup_lowered_wam_c_helpers(WamState* state) {\n    (void)state;\n}'
    ;   atomic_list_concat(SetupLines, '\n', SetupBody),
        format(atom(SetupCode),
               'void setup_lowered_wam_c_helpers(WamState* state) {\n~w\n}',
               [SetupBody])
    ).

render_wam_c_lowered_helper_plan([], '// WAM-C lowered helper plan: none').
render_wam_c_lowered_helper_plan(Plans, Code) :-
    maplist(render_wam_c_lowered_helper_plan_line, Plans, Lines),
    atomic_list_concat(['// WAM-C lowered helper plan'|Lines], '\n', Code).

render_wam_c_lowered_helper_plan_line(wam_c_lowered_helper_plan(Key, _PredIndicator, Action, Reason), Line) :-
    wam_c_lowered_plan_reason_label(Reason, ReasonLabel),
    format(atom(Line), '// - ~w ~w: ~w', [Action, Key, ReasonLabel]).

maybe_report_wam_c_lowered_helper_plan(Options, Plans) :-
    (   option(report_lowered_helpers(true), Options)
    ->  findall(Key-ReasonLabel, wam_c_lowered_helper_plan_by_action(Plans, lowered, Key, ReasonLabel), Lowered),
        findall(Key-ReasonLabel, wam_c_lowered_helper_plan_by_action(Plans, interpreted, Key, ReasonLabel), Interpreted),
        findall(Key-ReasonLabel, wam_c_lowered_helper_plan_by_action(Plans, rejected, Key, ReasonLabel), Rejected),
        format(user_error,
               '[WAM-C] lowered helper plan: lowered=~w interpreted=~w rejected=~w~n',
               [Lowered, Interpreted, Rejected])
    ;   true
    ).

wam_c_lowered_helper_plan_by_action(Plans, Action, Key, ReasonLabel) :-
    member(wam_c_lowered_helper_plan(Key, _, Action, Reason), Plans),
    wam_c_lowered_plan_reason_label(Reason, ReasonLabel).

wam_c_lowered_plan_reason_label(fact_only(_Rows), fact_only) :- !.
wam_c_lowered_plan_reason_label(alias_call(TargetKey, _Arity), Label) :- !,
    format(atom(Label), 'alias_call(~w)', [TargetKey]).
wam_c_lowered_plan_reason_label(Reason, Reason).

lowered_fact_helper_rows(PredIndicator, Rows) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Args,
            (   user:clause(Head, true),
                Head =.. [_|Args],
                maplist(wam_c_lowered_constant, Args)
            ),
            Rows),
    Rows \= [],
    \+ ( user:clause(Head, Body), Body \== true ).

lowered_alias_helper_call(PredIndicator, AvailableKeys, TargetKey, Arity) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), [Head-Body]),
    lowered_alias_body_goal(Body, TargetPred, BodyArgs),
    atom(TargetPred),
    TargetPred \== Pred,
    Head =.. [_|HeadArgs],
    same_variable_arguments(HeadArgs, BodyArgs),
    TargetIndicator = user:TargetPred/Arity,
    lowered_fact_helper_rows(TargetIndicator, _TargetRows),
    format(atom(TargetKey), '~w/~w', [TargetPred, Arity]),
    memberchk(TargetKey, AvailableKeys).

lowered_alias_body_goal(Module:Goal, TargetPred, BodyArgs) :-
    Module == user,
    !,
    Goal =.. [TargetPred|BodyArgs].
lowered_alias_body_goal(Goal, TargetPred, BodyArgs) :-
    Goal =.. [TargetPred|BodyArgs].

same_variable_arguments([], []).
same_variable_arguments([Left|LeftRest], [Right|RightRest]) :-
    var(Left),
    Left == Right,
    same_variable_arguments(LeftRest, RightRest).

lowered_helper_for_predicate(fact_only(Rows), PredIndicator, Key, Code, SetupLine) :-
    lowered_fact_helper_for_predicate(PredIndicator, Rows, Key, Code, SetupLine).
lowered_helper_for_predicate(alias_call(TargetKey, TargetArity), PredIndicator, Key, Code, SetupLine) :-
    lowered_alias_helper_for_predicate(PredIndicator, TargetKey, TargetArity, Key, Code, SetupLine).

lowered_fact_helper_for_predicate(PredIndicator, Rows, Key, Code, SetupLine) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_c_symbol_name(Pred, Arity, Symbol),
    maplist(wam_c_lowered_fact_row(Arity), Rows, RowCodes),
    atomic_list_concat(RowCodes, '\n', BodyCode),
    format(atom(Code),
'static bool ~w(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != ~w) return false;
~w
    return false;
}',
           [Symbol, Arity, BodyCode]),
    format(atom(SetupLine),
           '    wam_register_foreign_predicate(state, "~w", ~w, ~w);',
           [Key, Arity, Symbol]).

lowered_alias_helper_for_predicate(PredIndicator, TargetKey, TargetArity, Key, Code, SetupLine) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_c_symbol_name(Pred, Arity, Symbol),
    wam_c_symbol_for_key(TargetKey, TargetSymbol),
    format(atom(Code),
'static bool ~w(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != ~w) return false;
    return ~w(state, "~w", ~w);
}',
           [Symbol, Arity, TargetSymbol, TargetKey, TargetArity]),
    format(atom(SetupLine),
           '    wam_register_foreign_predicate(state, "~w", ~w, ~w);',
           [Key, Arity, Symbol]).

wam_c_symbol_for_key(Key, Symbol) :-
    atomic_list_concat([PredAtom, ArityAtom], '/', Key),
    atom_number(ArityAtom, Arity),
    wam_c_symbol_name(PredAtom, Arity, Symbol).

wam_c_lowered_constant(Arg) :- atom(Arg), !.
wam_c_lowered_constant(Arg) :- integer(Arg).

wam_c_symbol_name(Pred, Arity, Symbol) :-
    atom_chars(Pred, Chars),
    maplist(wam_c_symbol_char, Chars, SafeChars),
    atom_chars(SafePred, SafeChars),
    format(atom(Symbol), 'wam_c_lowered_~w_~w', [SafePred, Arity]).

wam_c_symbol_char(Char, Char) :-
    char_type(Char, alnum), !.
wam_c_symbol_char('_', '_') :- !.
wam_c_symbol_char(_, '_').

wam_c_lowered_fact_row(Arity, Args, Code) :-
    findall(MatchLine,
            (   nth0(I, Args, Arg),
                c_value_literal(Arg, ValLit),
                format(atom(MatchLine),
                       '    if (!val_is_unbound(*cells[~w]) && !val_equal(*cells[~w], ~w)) match = false;',
                       [I, I, ValLit])
            ),
            MatchLines),
    findall(BindLine,
            (   nth0(I, Args, Arg),
                c_value_literal(Arg, ValLit),
                format(atom(BindLine),
                       '        if (val_is_unbound(*cells[~w])) { trail_binding(state, cells[~w]); *cells[~w] = ~w; }',
                       [I, I, I, ValLit])
            ),
            BindLines),
    atomic_list_concat(MatchLines, '\n', MatchCode),
    atomic_list_concat(BindLines, '\n', BindCode),
    format(atom(Code),
'    {
        WamValue *cells[~w];
        for (int i = 0; i < ~w; i++) cells[i] = wam_deref_ptr(state, &state->A[i]);
        bool match = true;
~w
        if (match) {
~w
            return true;
        }
    }',
           [Arity, Arity, MatchCode, BindCode]).

% ============================================================================
% PHASE 2: WAM instructions -> C Struct Literals
% ============================================================================

%% wam_instruction_to_c_literal(+WamInstr, -CCode)
wam_instruction_to_c_literal(get_constant(C, Ai), Code) :-
    c_value_literal(C, Val), c_reg_index(Ai, IsY_Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_GET_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY_Ai]).
wam_instruction_to_c_literal(get_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(get_value(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_constant(C, Ai), Code) :-
    c_value_literal(C, Val), c_reg_index(Ai, IsY_Ai, Idx),
    format(atom(Code), '{ .tag = INSTR_PUT_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY_Ai]).
wam_instruction_to_c_literal(put_variable(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_value(Xn, Ai), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx), c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(get_structure(F, Ai), Code) :-
    c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [F, AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_structure(F, Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [F, XIdx, IsY_Xn]).
wam_instruction_to_c_literal(get_list(Ai), Code) :-
    c_reg_index(Ai, IsY_Ai, AIdx),
    format(atom(Code), '{ .tag = INSTR_GET_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [AIdx, IsY_Ai]).
wam_instruction_to_c_literal(put_list(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_PUT_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(set_variable(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_SET_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(set_value(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_SET_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(set_constant(C), Code) :-
    c_value_literal(C, Val),
    format(atom(Code), '{ .tag = INSTR_SET_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_instruction_to_c_literal(unify_variable(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_UNIFY_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(unify_value(Xn), Code) :-
    c_reg_index(Xn, IsY_Xn, XIdx),
    format(atom(Code), '{ .tag = INSTR_UNIFY_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [XIdx, IsY_Xn]).
wam_instruction_to_c_literal(unify_constant(C), Code) :-
    c_value_literal(C, Val),
    format(atom(Code), '{ .tag = INSTR_UNIFY_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_instruction_to_c_literal(call(P, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [P, N]).
wam_instruction_to_c_literal(execute(P), Code) :-
    format(atom(Code), '{ .tag = INSTR_EXECUTE, .as.pred = { .pred = "~w" } }', [P]).
wam_instruction_to_c_literal(builtin_call(Op, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_BUILTIN_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [Op, N]).
wam_instruction_to_c_literal(call_foreign(P, N), Code) :-
    format(atom(Code), '{ .tag = INSTR_CALL_FOREIGN, .as.pred = { .pred = "~w", .arity = ~w } }', [P, N]).
wam_instruction_to_c_literal(try_me_else(_Label), _) :-
    throw(error(context_error(missing_label_map, "try_me_else/1 requires LabelMap for target_pc resolution. Use wam_instruction_to_c_literal/3 instead."), _)).
wam_instruction_to_c_literal(retry_me_else(_Label), _) :-
    throw(error(context_error(missing_label_map, "retry_me_else/1 requires LabelMap for target_pc resolution. Use wam_instruction_to_c_literal/3 instead."), _)).


wam_instruction_to_c_literal(trust_me, '{ .tag = INSTR_TRUST_ME }').
wam_instruction_to_c_literal(proceed, '{ .tag = INSTR_PROCEED }').
wam_instruction_to_c_literal(allocate, '{ .tag = INSTR_ALLOCATE }').
wam_instruction_to_c_literal(deallocate, '{ .tag = INSTR_DEALLOCATE }').
wam_instruction_to_c_literal(Instr, _) :-
    throw(error(wam_c_target_error(unsupported_instruction(Instr)), _)).

wam_instruction_to_c_literal(try_me_else(Label), LabelMap, Code) :-
    ( member(Label-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Code), '{ .tag = INSTR_TRY_ME_ELSE, .as.choice = { .target_pc = ~w } }', [TargetPC]).
wam_instruction_to_c_literal(retry_me_else(Label), LabelMap, Code) :-
    ( member(Label-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Code), '{ .tag = INSTR_RETRY_ME_ELSE, .as.choice = { .target_pc = ~w } }', [TargetPC]).
wam_instruction_to_c_literal(Instr, _, Code) :- wam_instruction_to_c_literal(Instr, Code).


c_value_literal(Str, Lit) :-
    string(Str),
    (   number_string(Int, Str),
        integer(Int)
    ->  c_value_literal(Int, Lit)
    ;   atom_string(Atom, Str),
        c_value_literal(Atom, Lit)
    ).
c_value_literal(Atom, Lit) :- atom(Atom), format(atom(Lit), 'val_atom("~w")', [Atom]).
c_value_literal(Int, Lit) :- integer(Int), format(atom(Lit), 'val_int(~w)', [Int]).

c_reg_index(RegStr, IsY, Idx) :-
    string(RegStr),
    atom_string(RegAtom, RegStr),
    c_reg_index(RegAtom, IsY, Idx).
c_reg_index(RegAtom, IsY, Idx) :-
    atom_chars(RegAtom, Chars),
    (   Chars = [Prefix|NumChars],
        (Prefix == 'a'; Prefix == 'A')
    ->  IsY = 0,
        catch(number_chars(RegNo, NumChars), _, fail),
        Idx is RegNo - 1
    ;   Chars = [Prefix|NumChars],
        (Prefix == 'x'; Prefix == 'X')
    ->  IsY = 2,
        catch(number_chars(RegNo, NumChars), _, fail),
        Idx is RegNo - 1
    ;   Chars = [Prefix|NumChars],
        (Prefix == 'y'; Prefix == 'Y')
    ->  IsY = 1,
        catch(number_chars(RegNo, NumChars), _, fail),
        Idx is RegNo - 1
    ;   throw(error(wam_c_target_error(unknown_register(RegAtom)), _))
    ).

% ============================================================================
% PHASE 2b: wam_predicate -> C Array
% ============================================================================

% wam_line_to_c_instr/2, /3, /4
% Note: wam_line_to_c_instr has 2-arity, 3-arity, and 4-arity clauses.
% The 4-arity clauses are used for branch instructions (like try_me_else) that require the predicate's Arity.
% Non-branch instructions safely fall back to the 3-arity or 2-arity catch-alls during pass 2.
wam_line_to_c_instr(["get_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    c_value_literal(CC, Val), c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY]).
wam_line_to_c_instr(["get_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_GET_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["get_value", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_GET_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["put_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    c_value_literal(CC, Val), c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_CONSTANT, .as.constant = { .val = ~w, .reg = ~w, .is_y_reg = ~w } }', [Val, Idx, IsY]).
wam_line_to_c_instr(["put_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_PUT_VARIABLE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["put_value", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    c_reg_index(CXn, IsY_Xn, XIdx), c_reg_index(CAi, IsY_Ai, AIdx),
    format(atom(Instr), '{ .tag = INSTR_PUT_VALUE, .as.reg_pair = { .reg_xn = ~w, .is_y_xn = ~w, .reg_ai = ~w, .is_y_ai = ~w } }', [XIdx, IsY_Xn, AIdx, IsY_Ai]).
wam_line_to_c_instr(["get_structure", F, Ai], Instr) :-
    clean_comma(F, CF), clean_comma(Ai, CAi),
    c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [CF, Idx, IsY]).
wam_line_to_c_instr(["put_structure", F, Xn], Instr) :-
    clean_comma(F, CF), clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_STRUCTURE, .as.functor = { .pred = "~w", .reg = ~w, .is_y_reg = ~w } }', [CF, Idx, IsY]).
wam_line_to_c_instr(["get_list", Ai], Instr) :-
    clean_comma(Ai, CAi),
    c_reg_index(CAi, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_GET_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["put_list", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_PUT_LIST, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["set_variable", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_SET_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["set_value", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_SET_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["set_constant", C], Instr) :-
    clean_comma(C, CC),
    c_value_literal(CC, Val),
    format(atom(Instr), '{ .tag = INSTR_SET_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_line_to_c_instr(["unify_variable", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_VARIABLE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["unify_value", Xn], Instr) :-
    clean_comma(Xn, CXn),
    c_reg_index(CXn, IsY, Idx),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_VALUE, .as.reg = { .reg = ~w, .is_y_reg = ~w } }', [Idx, IsY]).
wam_line_to_c_instr(["unify_constant", C], Instr) :-
    clean_comma(C, CC),
    c_value_literal(CC, Val),
    format(atom(Instr), '{ .tag = INSTR_UNIFY_CONSTANT, .as.constant = { .val = ~w } }', [Val]).
wam_line_to_c_instr(["call", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(atom(Instr), '{ .tag = INSTR_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [CP, CN]).
wam_line_to_c_instr(["execute", P], Instr) :-
    clean_comma(P, CP),
    format(atom(Instr), '{ .tag = INSTR_EXECUTE, .as.pred = { .pred = "~w" } }', [CP]).
wam_line_to_c_instr(["builtin_call", Op, N], Instr) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    format(atom(Instr), '{ .tag = INSTR_BUILTIN_CALL, .as.pred = { .pred = "~w", .arity = ~w } }', [COp, CN]).
wam_line_to_c_instr(["call_foreign", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(atom(Instr), '{ .tag = INSTR_CALL_FOREIGN, .as.pred = { .pred = "~w", .arity = ~w } }', [CP, CN]).
wam_line_to_c_instr(["try_me_else", L], LabelMap, Arity, Instr) :-
    clean_comma(L, CL),
    ( member(CL-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Instr), '{ .tag = INSTR_TRY_ME_ELSE, .as.choice = { .target_pc = ~w, .arity = ~w } }', [TargetPC, Arity]).
wam_line_to_c_instr(["retry_me_else", L], LabelMap, Arity, Instr) :-
    clean_comma(L, CL),
    ( member(CL-TargetPC, LabelMap) -> true ; TargetPC = -1 ),
    format(atom(Instr), '{ .tag = INSTR_RETRY_ME_ELSE, .as.choice = { .target_pc = ~w, .arity = ~w } }', [TargetPC, Arity]).
wam_line_to_c_instr(["trust_me"], _, '{ .tag = INSTR_TRUST_ME }').
wam_line_to_c_instr(["proceed"], _, '{ .tag = INSTR_PROCEED }').
wam_line_to_c_instr(["allocate"], _, '{ .tag = INSTR_ALLOCATE }').
wam_line_to_c_instr(["deallocate"], _, '{ .tag = INSTR_DEALLOCATE }').
wam_line_to_c_instr(Parts, _, _) :-
    throw(error(wam_c_target_error(unsupported_instruction_tokens(Parts)), _)).

clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).

wam_lines_to_c_pass1([], _, []).
wam_lines_to_c_pass1([Line|Rest], PC, LabelMap) :-
    split_string(Line, " \t", " \t", Parts),  % comma intentionally excluded: entries are now space-separated
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

wam_lines_to_c_pass2([], PC, _, _, PC, []).
wam_lines_to_c_pass2([Line|Rest], PC, LabelMap, Arity, CodeSize, Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == [] -> wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, CodeSize, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            (   sub_string(LabelName, 0, 2, _, "L_")
            ->  wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, CodeSize, Instrs)
            ;   format(atom(PredReg), '    wam_register_predicate_hash(state, "~w", ~w);', [LabelName, PC]),
                Instrs = [PredReg|RestInstrs],
                wam_lines_to_c_pass2(Rest, PC, LabelMap, Arity, CodeSize, RestInstrs)
            )
        ;   wam_generate_c_instruction(PC, CleanParts, LabelMap, Arity, CodeLines),
            NPC is PC + 1,
            append(CodeLines, RestInstrs, Instrs),
            wam_lines_to_c_pass2(Rest, NPC, LabelMap, Arity, CodeSize, RestInstrs)
        )
    ).

wam_generate_c_instruction(PC, Parts, LabelMap, Arity, CodeLines) :-
    (   (   Parts = ["switch_on_constant" | Entries],
            SwitchReg = 0
        ;   Parts = ["switch_on_constant_a2" | Entries],
            SwitchReg = 1
        )
    ->  length(Entries, HashSize),
        format(atom(L0), '    state->code[~w] = (Instruction){ .tag = INSTR_SWITCH_ON_CONSTANT, .as.switch_index = { .reg = ~w, .hash_size = ~w } };', [PC, SwitchReg, HashSize]),
        format(atom(L1), '    state->code[~w].as.switch_index.hash_table = malloc(sizeof(HashEntry) * ~w);', [PC, HashSize]),
        generate_hash_table_entries(PC, "as.switch_index.hash_table", Entries, 0, LabelMap, HashLines),
        append([L0, L1], HashLines, CodeLines)
    ;   Parts = ["switch_on_structure" | Entries]
    ->  length(Entries, HashSize),
        format(atom(L0), '    state->code[~w] = (Instruction){ .tag = INSTR_SWITCH_ON_STRUCTURE, .as.switch_index = { .hash_size = ~w } };', [PC, HashSize]),
        format(atom(L1), '    state->code[~w].as.switch_index.hash_table = malloc(sizeof(HashEntry) * ~w);', [PC, HashSize]),
        generate_hash_table_entries(PC, "as.switch_index.hash_table", Entries, 0, LabelMap, HashLines),
        append([L0, L1], HashLines, CodeLines)
    ;   Parts = ["switch_on_term", CLenStr | Rest1]
    ->  number_string(CLen, CLenStr),
        length(CEntries, CLen),
        append(CEntries, [SLenStr | Rest2], Rest1),
        number_string(SLen, SLenStr),
        length(SEntries, SLen),
        append(SEntries, [ListLabelStr], Rest2),
        (   ListLabelStr == "none"
        ->  ListPC = -1
        ;   ListLabelStr == "default"
        ->  ListPC is PC + 1
        ;   member(ListLabelStr-ListPC, LabelMap)
        ->  true
        ;   ListPC = -1
        ),
        format(atom(L0), '    state->code[~w] = (Instruction){ .tag = INSTR_SWITCH_ON_TERM, .as.switch_index = { .hash_size = ~w, .s_hash_size = ~w, .list_target_pc = ~w } };', [PC, CLen, SLen, ListPC]),
        format(atom(L1), '    state->code[~w].as.switch_index.hash_table = malloc(sizeof(HashEntry) * ~w);', [PC, CLen]),
        format(atom(L2), '    state->code[~w].as.switch_index.s_hash_table = malloc(sizeof(HashEntry) * ~w);', [PC, SLen]),
        generate_hash_table_entries(PC, "as.switch_index.hash_table", CEntries, 0, LabelMap, CHashLines),
        generate_hash_table_entries(PC, "as.switch_index.s_hash_table", SEntries, 0, LabelMap, SHashLines),
        append([L0, L1, L2 | CHashLines], SHashLines, CodeLines)
    ;   (   wam_line_to_c_instr(Parts, LabelMap, Arity, CInstr)
        ->  true
        ;   wam_line_to_c_instr(Parts, CInstr_NoMap)
        ->  CInstr = CInstr_NoMap
        ;   wam_line_to_c_instr(Parts, LabelMap, CInstr_NoArity)
        ->  CInstr = CInstr_NoArity
        ),
        format(atom(L0), '    state->code[~w] = (Instruction)~w;', [PC, CInstr]),
        CodeLines = [L0]
    ).

generate_hash_table_entries(_, _, [], _, _, []).
generate_hash_table_entries(PC, TableName, [Entry|Rest], Idx, LabelMap, [Line|RestLines]) :-
    split_string(Entry, ":", "", Parts),
    (   Parts = [KeyStr, LabelStr]
    ->  (   LabelStr == "default"
        ->  TargetPC is PC + 1
        ;   member(LabelStr-TargetPC, LabelMap)
        ->  true
        ;   TargetPC = -1
        ),
        (   number_string(KeyNum, KeyStr), integer(KeyNum)
        ->  c_value_literal(KeyNum, ValLit)
        ;   atom_string(KeyAtom, KeyStr),
            c_value_literal(KeyAtom, ValLit)
        ),
        format(atom(Line), '    state->code[~w].~w[~w] = (HashEntry){ ~w, ~w };', [PC, TableName, Idx, ValLit, TargetPC]),
        NextIdx is Idx + 1,
        generate_hash_table_entries(PC, TableName, Rest, NextIdx, LabelMap, RestLines)
    ;   throw(error(wam_c_target_error(invalid_switch_entry(Entry)), _))
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
    wam_lines_to_c_pass2(Lines, 0, LabelMap, Arity, CodeSize, InstrParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    
    format(atom(CCode), 
'/* WAM-compiled predicate: ~w/~w */
void setup_~w_~w(WamState* state) {
    if (state->code) {
        for (int i = 0; i < state->code_size; i++) {
            if ((state->code[i].tag == INSTR_SWITCH_ON_CONSTANT || state->code[i].tag == INSTR_SWITCH_ON_STRUCTURE || state->code[i].tag == INSTR_SWITCH_ON_TERM) && state->code[i].as.switch_index.hash_table) {
                free(state->code[i].as.switch_index.hash_table);
                state->code[i].as.switch_index.hash_table = NULL;
            }
            if (state->code[i].tag == INSTR_SWITCH_ON_TERM && state->code[i].as.switch_index.s_hash_table) {
                free(state->code[i].as.switch_index.s_hash_table);
                state->code[i].as.switch_index.s_hash_table = NULL;
            }
        }
    }
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
    CCode =
'    bool step_wam(WamState* state, Instruction* instr) {
        switch (instr->tag) {
            case INSTR_GET_CONSTANT: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->as.constant.reg, instr->as.constant.is_y_reg));
                if (val_is_unbound(*cell)) {
                    trail_binding(state, cell);
                    *cell = instr->as.constant.val;
                    state->P++;
                    return true;
                } else if (val_equal(*cell, instr->as.constant.val)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_GET_VARIABLE: {
                // Per WAM spec: copy A[Ai] to X[Xn] without trailing.
                // Trailing is only for mutations of already-bound cells.
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
                *cell_xn = *cell_ai;
                state->P++;
                return true;
            }
            case INSTR_GET_VALUE: {
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
                if (!wam_unify(state, cell_xn, cell_ai)) return false;
                state->P++;
                return true;
            }
            case INSTR_PUT_CONSTANT: {
                WamValue *cell = resolve_reg(state, instr->as.constant.reg, instr->as.constant.is_y_reg);
                *cell = instr->as.constant.val;
                state->P++;
                return true;
            }
            case INSTR_PUT_VARIABLE: {
                WamValue ref = wam_make_ref(state);
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
                *cell_xn = ref;
                *cell_ai = ref;
                state->P++;
                return true;
            }
            case INSTR_PUT_VALUE: {
                WamValue *cell_xn = resolve_reg(state, instr->as.reg_pair.reg_xn, instr->as.reg_pair.is_y_xn);
                WamValue *cell_ai = resolve_reg(state, instr->as.reg_pair.reg_ai, instr->as.reg_pair.is_y_ai);
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
                int continuation = state->CP;
                if (continuation != WAM_HALT && state->call_base_top > 0) {
                    int target_b = state->call_bases[--state->call_base_top];
                    wam_prune_choice_points(state, target_b);
                }
                state->P = continuation;
                return true;
            }
            case INSTR_CALL: {
                if (state->call_base_top >= WAM_CALL_STACK_SIZE) return false;
                state->call_bases[state->call_base_top++] = state->B;
                state->CP = state->P + 1;
                int target = resolve_predicate_hash(state, instr->as.pred.pred);
                if (target >= 0) { state->P = target; return true; }
                state->call_base_top--;
                return false;
            }
            case INSTR_EXECUTE: {
                int target = resolve_predicate_hash(state, instr->as.pred.pred);
                if (target >= 0) { state->P = target; return true; }
                return false;
            }
            case INSTR_BUILTIN_CALL: {
                if (wam_execute_builtin(state, instr->as.pred.pred, instr->as.pred.arity)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_CALL_FOREIGN: {
                if (wam_execute_foreign_predicate(state, instr->as.pred.pred, instr->as.pred.arity)) {
                    state->P++;
                    return true;
                }
                return false;
            }
            case INSTR_TRY_ME_ELSE: {
                int target = instr->as.choice.target_pc;
                int arity = instr->as.choice.arity ? instr->as.choice.arity : 32;
                push_choice_point(state, target, arity);
                state->P++;
                return true;
            }
            case INSTR_RETRY_ME_ELSE: {
                int target = instr->as.choice.target_pc;
                if (state->B > 0) {
                    ChoicePoint *cp = &state->B_array[state->B - 1];
                    cp->next_pc = target;
                }
                state->P++;
                return true;
            }
            case INSTR_TRUST_ME: {
                pop_choice_point(state);
                state->P++;
                return true;
            }
            case INSTR_SWITCH_ON_CONSTANT: {
                WamValue *cell = wam_deref_ptr(state, &state->A[instr->as.switch_index.reg]);
                if (val_is_unbound(*cell)) {
                    state->P++;
                    return true; // Unbound variable falls through to the sequential try_me_else chain
                }
                if (cell->tag != VAL_ATOM && cell->tag != VAL_INT) {
                    return false; // Type mismatch, fail
                }
                for (int i = 0; i < instr->as.switch_index.hash_size; i++) {
                    if (val_equal(*cell, instr->as.switch_index.hash_table[i].key)) {
                        state->P = instr->as.switch_index.hash_table[i].target_pc;
                        return true;
                    }
                }
                return false; // Not found in index, fail
            }
            case INSTR_SWITCH_ON_STRUCTURE: {
                WamValue *cell = wam_deref_ptr(state, &state->A[0]);
                if (val_is_unbound(*cell)) {
                    state->P++;
                    return true; // Unbound variable falls through to the sequential try_me_else chain
                }
                if (cell->tag != VAL_STR) {
                    return false; // Type mismatch, fail
                }
                WamValue *f = &state->H_array[cell->data.ref_addr];
                for (int i = 0; i < instr->as.switch_index.hash_size; i++) {
                    if (val_equal(*f, instr->as.switch_index.hash_table[i].key)) {
                        state->P = instr->as.switch_index.hash_table[i].target_pc;
                        return true;
                    }
                }
                return false; // Not found in index, fail
            }
            case INSTR_SWITCH_ON_TERM: {
                WamValue *cell = wam_deref_ptr(state, &state->A[0]);
                if (val_is_unbound(*cell)) {
                    state->P++;
                    return true; // Unbound variable falls through to the sequential try_me_else chain
                }
                if (cell->tag == VAL_ATOM || cell->tag == VAL_INT) {
                    for (int i = 0; i < instr->as.switch_index.hash_size; i++) {
                        if (val_equal(*cell, instr->as.switch_index.hash_table[i].key)) {
                            state->P = instr->as.switch_index.hash_table[i].target_pc;
                            return true;
                        }
                    }
                } else if (cell->tag == VAL_STR) {
                    WamValue *f = &state->H_array[cell->data.ref_addr];
                    for (int i = 0; i < instr->as.switch_index.s_hash_size; i++) {
                        if (val_equal(*f, instr->as.switch_index.s_hash_table[i].key)) {
                            state->P = instr->as.switch_index.s_hash_table[i].target_pc;
                            return true;
                        }
                    }
                } else if (cell->tag == VAL_LIST) {
                    if (instr->as.switch_index.list_target_pc >= 0) {
                        state->P = instr->as.switch_index.list_target_pc;
                    } else {
                        state->P++;
                    }
                    return true;
                }
                return false; // Not found in either index — fail and backtrack
            }
            case INSTR_GET_STRUCTURE: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->as.functor.reg, instr->as.functor.is_y_reg));
                if (cell->tag == VAL_UNBOUND) {
                    trail_binding(state, cell);
                    WamValue s; s.tag = VAL_STR; s.data.ref_addr = state->H;
                    *cell = s;
                    
                    // Note: instr->as.functor.pred includes the arity suffix (e.g. "foo/2"), which is stored as the functor atom
                    const char *slash = strchr(instr->as.functor.pred, ''/'');
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
                    state->H_array[state->H] = val_atom(instr->as.functor.pred);
                    state->H++;
                    state->mode = MODE_WRITE;
                } else if (cell->tag == VAL_STR) {
                    WamValue *f = &state->H_array[cell->data.ref_addr];
                    if (f->tag == VAL_ATOM && strcmp(f->data.atom, instr->as.functor.pred) == 0) {
                        state->S = cell->data.ref_addr + 1;
                        state->mode = MODE_READ;
                    } else { return false; }
                } else { return false; }
                state->P++;
                return true;
            }
            case INSTR_PUT_STRUCTURE: {
                WamValue s; s.tag = VAL_STR; s.data.ref_addr = state->H;
                WamValue *cell = resolve_reg(state, instr->as.functor.reg, instr->as.functor.is_y_reg);
                *cell = s;
                
                const char *slash = strchr(instr->as.functor.pred, ''/'');
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
                state->H_array[state->H] = val_atom(instr->as.functor.pred);
                state->H++;
                state->mode = MODE_WRITE;
                state->P++;
                return true;
            }
            case INSTR_GET_LIST: {
                WamValue *cell = wam_deref_ptr(state, resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg));
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
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
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
            case INSTR_SET_VARIABLE: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                WamValue ref = wam_make_ref(state);
                *cell = ref;
                state->P++;
                return true;
            }
            case INSTR_SET_VALUE: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
                if (state->H >= state->H_cap) {
                    state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->H_array[state->H] = *cell;
                state->H++;
                state->P++;
                return true;
            }
            case INSTR_SET_CONSTANT: {
                if (state->H >= state->H_cap) {
                    state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                    state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                }
                state->H_array[state->H] = instr->as.constant.val;
                state->H++;
                state->P++;
                return true;
            }
            case INSTR_UNIFY_VARIABLE: {
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
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
                WamValue *cell = resolve_reg(state, instr->as.reg.reg, instr->as.reg.is_y_reg);
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
                        *cell = instr->as.constant.val;
                    } else if (!val_equal(*cell, instr->as.constant.val)) {
                        return false;
                    }
                    state->S++;
                } else {
                    if (state->H >= state->H_cap) {
                        state->H_cap = state->H_cap ? state->H_cap * 2 : WAM_INITIAL_CAP;
                        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
                    }
                    state->H_array[state->H] = instr->as.constant.val;
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
    }'.

compile_wam_helpers_to_c(_Options, CCode) :-
    CCode =
'#include "wam_runtime.h"

void wam_state_init(WamState *state) {
    memset(state, 0, sizeof(WamState));
    state->H_cap = WAM_INITIAL_CAP;
    state->TR_cap = WAM_INITIAL_CAP;
    state->B_cap = WAM_INITIAL_CAP;
    state->E_cap = WAM_INITIAL_CAP;
    state->E = -1;
    state->H_array = malloc(sizeof(WamValue) * state->H_cap);
    state->TR_array = malloc(sizeof(TrailEntry) * state->TR_cap);
    state->B_array = malloc(sizeof(ChoicePoint) * state->B_cap);
    state->E_array = malloc(sizeof(EnvFrame) * state->E_cap);
}

void wam_free_state(WamState *state) {
    for (int i = 0; i < WAM_ATOM_HASH_SIZE; i++) {
        AtomEntry *e = state->atom_table[i];
        while (e) {
            AtomEntry *next = e->next;
            free(e->str);
            free(e);
            e = next;
        }
    }
    for (int i = 0; i < state->code_size; i++) {
        if (state->code[i].tag == INSTR_SWITCH_ON_CONSTANT || state->code[i].tag == INSTR_SWITCH_ON_STRUCTURE || state->code[i].tag == INSTR_SWITCH_ON_TERM) {
            free(state->code[i].as.switch_index.hash_table);
        }
        if (state->code[i].tag == INSTR_SWITCH_ON_TERM) {
            free(state->code[i].as.switch_index.s_hash_table);
        }
    }
    free(state->code);
    free(state->H_array);
    free(state->TR_array);
    free(state->B_array);
    free(state->E_array);
    free(state->category_edges);
    memset(state, 0, sizeof(WamState));
}

int wam_run_predicate(WamState *state, const char *pred,
                      WamValue *args, int arity) {
    int entry = resolve_predicate_hash(state, pred);
    if (entry < 0) return WAM_ERR_OOB;
    int base_b = state->B;
    int base_call_base_top = state->call_base_top;
    for (int i = 0; i < arity; i++) state->A[i] = args[i];
    state->CP = WAM_HALT;
    state->P = entry;
    int rc = wam_run(state);
    wam_prune_choice_points(state, base_b);
    state->call_base_top = base_call_base_top;
    return rc;
}

static bool wam_eval_arith(WamState *state, WamValue value, int *out) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag == VAL_INT) {
        *out = cell->data.integer;
        return true;
    }
    if (cell->tag != VAL_STR) return false;

    int addr = cell->data.ref_addr;
    WamValue *functor = &state->H_array[addr];
    if (functor->tag != VAL_ATOM) return false;

    int lhs = 0;
    int rhs = 0;
    if (!wam_eval_arith(state, state->H_array[addr + 1], &lhs)) return false;
    if (!wam_eval_arith(state, state->H_array[addr + 2], &rhs)) return false;

    if (strcmp(functor->data.atom, "+/2") == 0) {
        *out = lhs + rhs;
        return true;
    }
    if (strcmp(functor->data.atom, "-/2") == 0) {
        *out = lhs - rhs;
        return true;
    }
    if (strcmp(functor->data.atom, "*/2") == 0) {
        *out = lhs * rhs;
        return true;
    }
    if (strcmp(functor->data.atom, "//2") == 0 || strcmp(functor->data.atom, "div/2") == 0) {
        if (rhs == 0) return false;
        *out = lhs / rhs;
        return true;
    }
    return false;
}

bool wam_execute_builtin(WamState *state, const char *op, int arity) {
    if (strcmp(op, "true/0") == 0 && arity == 0) return true;
    if ((strcmp(op, "fail/0") == 0 || strcmp(op, "false/0") == 0) && arity == 0) return false;
    if (strcmp(op, "!/0") == 0 && arity == 0) {
        state->B = 0;
        return true;
    }

    if (strcmp(op, "=/2") == 0 && arity == 2) {
        return wam_unify(state, &state->A[0], &state->A[1]);
    }

    if (arity == 1) {
        WamValue *a1 = wam_deref_ptr(state, &state->A[0]);
        if (strcmp(op, "atom/1") == 0) return a1->tag == VAL_ATOM;
        if (strcmp(op, "integer/1") == 0) return a1->tag == VAL_INT;
        if (strcmp(op, "number/1") == 0) return a1->tag == VAL_INT;
        if (strcmp(op, "float/1") == 0) return false;
        if (strcmp(op, "var/1") == 0) return val_is_unbound(*a1);
        if (strcmp(op, "nonvar/1") == 0) return !val_is_unbound(*a1);
        if (strcmp(op, "compound/1") == 0) return a1->tag == VAL_STR || a1->tag == VAL_LIST;
        if (strcmp(op, "is_list/1") == 0) return a1->tag == VAL_LIST;
    }

    if (strcmp(op, "is/2") == 0 && arity == 2) {
        int result = 0;
        if (!wam_eval_arith(state, state->A[1], &result)) return false;
        WamValue value = val_int(result);
        return wam_unify(state, &state->A[0], &value);
    }

    if (arity == 2) {
        int lhs = 0;
        int rhs = 0;
        if (!wam_eval_arith(state, state->A[0], &lhs)) return false;
        if (!wam_eval_arith(state, state->A[1], &rhs)) return false;

        if (strcmp(op, ">/2") == 0) return lhs > rhs;
        if (strcmp(op, "</2") == 0) return lhs < rhs;
        if (strcmp(op, ">=/2") == 0) return lhs >= rhs;
        if (strcmp(op, "=</2") == 0) return lhs <= rhs;
        if (strcmp(op, "=:=/2") == 0) return lhs == rhs;
        if (strcmp(op, "=\\\\=/2") == 0) return lhs != rhs;
    }

    return false;
}

bool wam_execute_foreign_predicate(WamState *state, const char *pred, int arity) {
    WamForeignHandler handler = resolve_foreign_predicate(state, pred, arity);
    if (!handler) return false;
    return handler(state, pred, arity);
}

void wam_register_category_parent(WamState *state, const char *child, const char *parent) {
    if (state->category_edge_count >= state->category_edge_cap) {
        state->category_edge_cap = state->category_edge_cap ? state->category_edge_cap * 2 : WAM_INITIAL_CAP;
        state->category_edges = realloc(state->category_edges, sizeof(CategoryEdge) * state->category_edge_cap);
    }
    state->category_edges[state->category_edge_count].child = wam_intern_atom(state, child);
    state->category_edges[state->category_edge_count].parent = wam_intern_atom(state, parent);
    state->category_edge_count++;
}

void wam_fact_source_init(WamFactSource *source) {
    memset(source, 0, sizeof(WamFactSource));
}

void wam_fact_source_close(WamFactSource *source) {
    free(source->edges);
    memset(source, 0, sizeof(WamFactSource));
}

static void wam_fact_source_add_edge(WamState *state,
                                     WamFactSource *source,
                                     const char *child,
                                     const char *parent) {
    if (source->edge_count >= source->edge_cap) {
        source->edge_cap = source->edge_cap ? source->edge_cap * 2 : WAM_INITIAL_CAP;
        source->edges = realloc(source->edges, sizeof(CategoryEdge) * source->edge_cap);
    }
    source->edges[source->edge_count].child = wam_intern_atom(state, child);
    source->edges[source->edge_count].parent = wam_intern_atom(state, parent);
    source->edge_count++;
}

bool wam_fact_source_load_tsv(WamState *state, WamFactSource *source, const char *path) {
    FILE *file = fopen(path, "r");
    if (!file) return false;

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        char *start = line;
        while (*start == 32 || *start == 9) start++;
        if (*start == 0 || *start == 10 || *start == 35) continue;

        char *sep = strchr(start, 9);
        if (!sep) sep = strchr(start, 32);
        if (!sep) {
            fclose(file);
            return false;
        }

        *sep = 0;
        char *parent = sep + 1;
        while (*parent == 32 || *parent == 9) parent++;

        char *end = parent + strlen(parent);
        while (end > parent && (end[-1] == 10 || end[-1] == 13 ||
                                end[-1] == 32 || end[-1] == 9)) {
            *--end = 0;
        }

        if (*start == 0 || *parent == 0) {
            fclose(file);
            return false;
        }
        wam_fact_source_add_edge(state, source, start, parent);
    }

    fclose(file);
    return true;
}

bool wam_fact_source_load_lmdb(WamState *state, WamFactSource *source,
                               const char *env_path, const char *db_name) {
#ifndef WAM_C_ENABLE_LMDB
    (void)state;
    (void)source;
    (void)env_path;
    (void)db_name;
    return false;
#else
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_cursor *cursor = NULL;
    MDB_dbi dbi = 0;
    int rc = 0;
    bool ok = false;

    rc = mdb_env_create(&env);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_env_set_maxdbs(env, 16);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_env_open(env, env_path, MDB_RDONLY, 0664);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_dbi_open(txn, (db_name && db_name[0]) ? db_name : NULL, 0, &dbi);
    if (rc != MDB_SUCCESS) goto done;
    rc = mdb_cursor_open(txn, dbi, &cursor);
    if (rc != MDB_SUCCESS) goto done;

    MDB_val key;
    MDB_val data;
    rc = mdb_cursor_get(cursor, &key, &data, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        char *child = malloc(key.mv_size + 1);
        char *parent = malloc(data.mv_size + 1);
        if (!child || !parent) {
            free(child);
            free(parent);
            goto done;
        }
        memcpy(child, key.mv_data, key.mv_size);
        child[key.mv_size] = 0;
        memcpy(parent, data.mv_data, data.mv_size);
        parent[data.mv_size] = 0;
        wam_fact_source_add_edge(state, source, child, parent);
        free(child);
        free(parent);
        rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT);
    }
    ok = (rc == MDB_NOTFOUND);

done:
    if (cursor) mdb_cursor_close(cursor);
    if (txn) mdb_txn_abort(txn);
    if (env) {
        if (dbi) mdb_dbi_close(env, dbi);
        mdb_env_close(env);
    }
    return ok;
#endif
}

int wam_fact_source_lookup_arg1(WamFactSource *source, const char *arg1,
                                CategoryEdge *out_edges, int max_edges) {
    int count = 0;
    for (int i = 0; i < source->edge_count; i++) {
        if (strcmp(source->edges[i].child, arg1) != 0) continue;
        if (count < max_edges) out_edges[count] = source->edges[i];
        count++;
    }
    return count;
}

bool wam_register_category_parent_fact_source(WamState *state, WamFactSource *source) {
    for (int i = 0; i < source->edge_count; i++) {
        wam_register_category_parent(state, source->edges[i].child, source->edges[i].parent);
    }
    return true;
}

void wam_int_results_init(WamIntResults *results) {
    memset(results, 0, sizeof(WamIntResults));
}

void wam_int_results_close(WamIntResults *results) {
    free(results->values);
    memset(results, 0, sizeof(WamIntResults));
}

bool wam_int_results_push(WamIntResults *results, int value) {
    if (results->count >= results->cap) {
        results->cap = results->cap ? results->cap * 2 : WAM_INITIAL_CAP;
        results->values = realloc(results->values, sizeof(int) * results->cap);
        if (!results->values) {
            results->count = 0;
            results->cap = 0;
            return false;
        }
    }
    results->values[results->count++] = value;
    return true;
}

void wam_register_category_ancestor_kernel(WamState *state, const char *pred, int max_depth) {
    state->category_max_depth = max_depth > 0 ? max_depth : 10;
    wam_register_foreign_predicate(state, pred, 4, wam_category_ancestor_handler);
}

static bool wam_value_as_atom(WamState *state, WamValue value, const char **out) {
    WamValue *cell = wam_deref_ptr(state, &value);
    if (cell->tag != VAL_ATOM) return false;
    *out = cell->data.atom;
    return true;
}

static bool wam_list_contains_atom(WamState *state, WamValue value, const char *atom) {
    WamValue *cell = wam_deref_ptr(state, &value);
    while (cell->tag == VAL_LIST) {
        WamValue *head = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr]);
        if (head->tag == VAL_ATOM && strcmp(head->data.atom, atom) == 0) return true;
        cell = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr + 1]);
    }
    return cell->tag == VAL_ATOM && strcmp(cell->data.atom, atom) == 0;
}

static bool wam_list_atoms_to_array(WamState *state,
                                    WamValue value,
                                    const char **out,
                                    int *out_len,
                                    int max_len) {
    WamValue *cell = wam_deref_ptr(state, &value);
    int count = 0;
    while (cell->tag == VAL_LIST) {
        if (count >= max_len) return false;
        WamValue *head = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr]);
        if (head->tag != VAL_ATOM) return false;
        out[count++] = head->data.atom;
        cell = wam_deref_ptr(state, &state->H_array[cell->data.ref_addr + 1]);
    }
    if (cell->tag == VAL_ATOM && strcmp(cell->data.atom, "[]") == 0) {
        *out_len = count;
        return true;
    }
    return false;
}

static bool wam_visited_array_contains(const char **visited, int visited_len, const char *atom) {
    for (int i = 0; i < visited_len; i++) {
        if (strcmp(visited[i], atom) == 0) return true;
    }
    return false;
}

static bool wam_category_ancestor_dfs(WamState *state,
                                      const char *cat,
                                      const char *root,
                                      int depth,
                                      int max_depth,
                                      const char **visited,
                                      int visited_len,
                                      WamIntResults *results) {
    bool found = false;
    for (int i = 0; i < state->category_edge_count; i++) {
        CategoryEdge *edge = &state->category_edges[i];
        if (strcmp(edge->child, cat) != 0) continue;
        if (wam_visited_array_contains(visited, visited_len, edge->parent)) continue;
        if (strcmp(edge->parent, root) == 0) {
            if (!wam_int_results_push(results, depth + 1)) return false;
            found = true;
        }
    }

    if (visited_len >= max_depth || visited_len >= 64) return found;

    for (int i = 0; i < state->category_edge_count; i++) {
        CategoryEdge *edge = &state->category_edges[i];
        if (strcmp(edge->child, cat) != 0) continue;
        if (wam_visited_array_contains(visited, visited_len, edge->parent)) continue;
        visited[visited_len] = edge->parent;
        if (wam_category_ancestor_dfs(state, edge->parent, root, depth + 1,
                                      max_depth, visited, visited_len + 1,
                                      results)) {
            found = true;
        }
    }
    return found;
}

static bool wam_category_ancestor_inputs(WamState *state,
                                         const char **cat_out,
                                         const char **root_out,
                                         const char **visited,
                                         int *visited_len_out) {
    const char *cat = NULL;
    const char *root = NULL;
    if (!wam_value_as_atom(state, state->A[0], &cat)) return false;
    if (!wam_value_as_atom(state, state->A[1], &root)) return false;
    int visited_len = 0;
    if (!wam_list_atoms_to_array(state, state->A[3], visited, &visited_len, 64)) return false;
    if (wam_list_contains_atom(state, state->A[3], root)) return false;
    if (visited_len == 0) {
        visited[visited_len++] = cat;
    }
    *cat_out = cat;
    *root_out = root;
    *visited_len_out = visited_len;
    return true;
}

bool wam_collect_category_ancestor_hops(WamState *state, WamIntResults *results) {
    const char *cat = NULL;
    const char *root = NULL;
    const char *visited[64];
    int visited_len = 0;
    if (!wam_category_ancestor_inputs(state, &cat, &root, visited, &visited_len)) return false;

    int max_depth = state->category_max_depth > 0 ? state->category_max_depth : 10;
    return wam_category_ancestor_dfs(state, cat, root, 0, max_depth,
                                     visited, visited_len, results);
}

bool wam_category_ancestor_handler(WamState *state, const char *pred, int arity) {
    (void)pred;
    if (arity != 4) return false;

    WamIntResults results;
    wam_int_results_init(&results);
    if (!wam_collect_category_ancestor_hops(state, &results) || results.count == 0) {
        wam_int_results_close(&results);
        return false;
    }

    WamValue result = val_int(results.values[0]);
    bool ok = wam_unify(state, &state->A[2], &result);
    wam_int_results_close(&results);
    return ok;
}
'.

compile_wam_runtime_to_c(Options, CCode) :-
    compile_step_wam_to_c(Options, StepCode),
    compile_wam_helpers_to_c(Options, HelpersCode),
    format(atom(CCode), "~w\n\n~w", [HelpersCode, StepCode]).
