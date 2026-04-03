:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_go_target.pl - WAM-to-Go Transpilation Target
%
% Transpiles WAM runtime predicates (wam_runtime.pl) to Go code.
% Phase 2: WAM instructions → Go struct literals
% Phase 3: step_wam/3 → type switch cases
%
% See: docs/design/WAM_GO_TRANSPILATION_IMPLEMENTATION_PLAN.md

:- module(wam_go_target, [
    compile_step_wam_to_go/2,          % +Options, -GoCode
    compile_wam_helpers_to_go/2,       % +Options, -GoCode
    compile_wam_runtime_to_go/2,       % +Options, -GoCode
    compile_wam_predicate_to_go/4,     % +Pred/Arity, +WamCode, +Options, -GoCode
    wam_instruction_to_go_literal/2,   % +WamInstr, -GoLiteral
    wam_line_to_go_literal/2,          % +Parts, -GoLit
    write_wam_go_project/3             % +Predicates, +Options, +ProjectDir
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../bindings/go_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../targets/go_target', [compile_predicate_to_go/3]).

:- discontiguous wam_go_case/2.

% ============================================================================
% PHASE 4: Hybrid Module Assembly
% ============================================================================

%% write_wam_go_project(+Predicates, +Options, +ProjectDir)
%  Generates a full Go project for the given predicates.
write_wam_go_project(Predicates, Options, ProjectDir) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    option(package_name(PackageName), Options, 'wam'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

    % Create directory structure
    make_directory_path(ProjectDir),

    % Generate go.mod
    read_template_file('templates/targets/go_wam/go.mod.mustache', GoModTemplate),
    render_template(GoModTemplate, [module_name=ModuleName], GoModContent),
    directory_file_path(ProjectDir, 'go.mod', GoModPath),
    write_file(GoModPath, GoModContent),

    % Write value.go from template
    read_template_file('templates/targets/go_wam/value.go.mustache', ValueTemplate),
    render_template(ValueTemplate, [package_name=PackageName, date=Date], ValueCode),
    directory_file_path(ProjectDir, 'value.go', ValuePath),
    write_file(ValuePath, ValueCode),

    % Write instructions.go from template
    read_template_file('templates/targets/go_wam/instructions.go.mustache', InstrTemplate),
    render_template(InstrTemplate, [package_name=PackageName, date=Date], InstrCode),
    directory_file_path(ProjectDir, 'instructions.go', InstrPath),
    write_file(InstrPath, InstrCode),

    % Write state.go from template
    read_template_file('templates/targets/go_wam/state.go.mustache', StateTemplate),
    render_template(StateTemplate, [package_name=PackageName, date=Date], StateCode),
    directory_file_path(ProjectDir, 'state.go', StatePath),
    write_file(StatePath, StateCode),

    % Generate runtime.go: template + transpiled runtime methods
    compile_step_wam_to_go(Options, StepMethod),
    compile_wam_helpers_to_go(Options, HelperMethods),
    read_template_file('templates/targets/go_wam/runtime.go.mustache', RuntimeTemplate),
    render_template(RuntimeTemplate, [
        package_name=PackageName,
        step_method=StepMethod,
        helper_methods=HelperMethods
    ], RuntimeCode),
    directory_file_path(ProjectDir, 'runtime.go', RuntimePath),
    write_file(RuntimePath, RuntimeCode),

    % Compile predicates and generate lib.go (predicates)
    compile_predicates_for_project(Predicates, Options, PredicatesCode),
    format(atom(LibContent),
'package ~w

~w
', [PackageName, PredicatesCode]),
    directory_file_path(ProjectDir, 'lib.go', LibPath),
    write_file(LibPath, LibContent),

    format('WAM Go project created at: ~w~n', [ProjectDir]).

%% read_template_file(+Path, -Content)
read_template_file(Path, Content) :-
    (   exists_file(Path)
    ->  read_file_to_string(Path, Content, [])
    ;   format(atom(Content), "// Template not found: ~w", [Path])
    ).

%% compile_predicates_for_project(+Predicates, +Options, -Code)
compile_predicates_for_project([], _, "").
compile_predicates_for_project([PredIndicator|Rest], Options, Code) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    (   % Try native Go lowering first
        catch(
            go_target:compile_predicate_to_go(Module:Pred/Arity,
                [include_main(false)|Options], PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        Strategy = native
    ;   % Fall back to WAM compilation
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode),
        compile_wam_predicate_to_go(Pred/Arity, WamCode, Options, PredCode)
    ->  format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
        Strategy = wam
    ;   % Neither worked
        format(atom(PredCode), '// ~w/~w: compilation failed', [Pred, Arity]),
        Strategy = failed
    ),
    compile_predicates_for_project(Rest, Options, RestCode),
    (   RestCode == ""
    ->  format(atom(Code), "// Strategy: ~w\n~w", [Strategy, PredCode])
    ;   format(atom(Code), "// Strategy: ~w\n~w\n\n~w", [Strategy, PredCode, RestCode])
    ).

%% compile_wam_runtime_to_go(+Options, -GoCode)
%  Placeholder for potentially transpiling larger parts of wam_runtime.pl
compile_wam_runtime_to_go(Options, GoCode) :-
    compile_step_wam_to_go(Options, StepCode),
    compile_wam_helpers_to_go(Options, HelpersCode),
    format(atom(GoCode), "~w\n\n~w", [StepCode, HelpersCode]).

%% write_file(+Path, +Content)
write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

% ============================================================================
% PHASE 2: WAM instructions → Go struct literals
% ============================================================================

%% wam_instruction_to_go_literal(+WamInstr, -GoLiteral)
%  Converts a WAM instruction term to a Go struct literal string.

wam_instruction_to_go_literal(get_constant(C, Ai), GoLiteral) :-
    go_value_literal(C, GoVal),
    format(atom(GoLiteral), '&GetConstant{C: ~w, Ai: "~w"}', [GoVal, Ai]).
wam_instruction_to_go_literal(get_variable(Xn, Ai), GoLiteral) :-
    format(atom(GoLiteral), '&GetVariable{Xn: "~w", Ai: "~w"}', [Xn, Ai]).
wam_instruction_to_go_literal(get_value(Xn, Ai), GoLiteral) :-
    format(atom(GoLiteral), '&GetValue{Xn: "~w", Ai: "~w"}', [Xn, Ai]).
wam_instruction_to_go_literal(get_structure(F, Ai), GoLiteral) :-
    format(atom(GoLiteral), '&GetStructure{Functor: "~w", Ai: "~w"}', [F, Ai]).
wam_instruction_to_go_literal(get_list(Ai), GoLiteral) :-
    format(atom(GoLiteral), '&GetList{Ai: "~w"}', [Ai]).
wam_instruction_to_go_literal(unify_variable(Xn), GoLiteral) :-
    format(atom(GoLiteral), '&UnifyVariable{Xn: "~w"}', [Xn]).
wam_instruction_to_go_literal(unify_value(Xn), GoLiteral) :-
    format(atom(GoLiteral), '&UnifyValue{Xn: "~w"}', [Xn]).
wam_instruction_to_go_literal(unify_constant(C), GoLiteral) :-
    go_value_literal(C, GoVal),
    format(atom(GoLiteral), '&UnifyConstant{C: ~w}', [GoVal]).

wam_instruction_to_go_literal(put_constant(C, Ai), GoLiteral) :-
    go_value_literal(C, GoVal),
    format(atom(GoLiteral), '&PutConstant{C: ~w, Ai: "~w"}', [GoVal, Ai]).
wam_instruction_to_go_literal(put_variable(Xn, Ai), GoLiteral) :-
    format(atom(GoLiteral), '&PutVariable{Xn: "~w", Ai: "~w"}', [Xn, Ai]).
wam_instruction_to_go_literal(put_value(Xn, Ai), GoLiteral) :-
    format(atom(GoLiteral), '&PutValue{Xn: "~w", Ai: "~w"}', [Xn, Ai]).
wam_instruction_to_go_literal(put_structure(F, Ai), GoLiteral) :-
    format(atom(GoLiteral), '&PutStructure{Functor: "~w", Ai: "~w"}', [F, Ai]).
wam_instruction_to_go_literal(put_list(Ai), GoLiteral) :-
    format(atom(GoLiteral), '&PutList{Ai: "~w"}', [Ai]).
wam_instruction_to_go_literal(set_variable(Xn), GoLiteral) :-
    format(atom(GoLiteral), '&SetVariable{Xn: "~w"}', [Xn]).
wam_instruction_to_go_literal(set_value(Xn), GoLiteral) :-
    format(atom(GoLiteral), '&SetValue{Xn: "~w"}', [Xn]).
wam_instruction_to_go_literal(set_constant(C), GoLiteral) :-
    go_value_literal(C, GoVal),
    format(atom(GoLiteral), '&SetConstant{C: ~w}', [GoVal]).

wam_instruction_to_go_literal(allocate, '&Allocate{}').
wam_instruction_to_go_literal(deallocate, '&Deallocate{}').
wam_instruction_to_go_literal(call(P, N), GoLiteral) :-
    format(atom(GoLiteral), '&Call{Pred: "~w", Arity: ~w}', [P, N]).
wam_instruction_to_go_literal(execute(P), GoLiteral) :-
    format(atom(GoLiteral), '&Execute{Pred: "~w"}', [P]).
wam_instruction_to_go_literal(proceed, '&Proceed{}').
wam_instruction_to_go_literal(builtin_call(Op, N), GoLiteral) :-
    format(atom(GoLiteral), '&BuiltinCall{Op: "~w", Arity: ~w}', [Op, N]).

wam_instruction_to_go_literal(try_me_else(Label), GoLiteral) :-
    format(atom(GoLiteral), '&TryMeElse{Label: "~w"}', [Label]).
wam_instruction_to_go_literal(retry_me_else(Label), GoLiteral) :-
    format(atom(GoLiteral), '&RetryMeElse{Label: "~w"}', [Label]).
wam_instruction_to_go_literal(trust_me, '&TrustMe{}').

wam_instruction_to_go_literal(switch_on_constant(Table), GoLiteral) :-
    maplist(go_const_case, Table, Cases),
    atomic_list_concat(Cases, ', ', CaseStr),
    format(atom(GoLiteral), '&SwitchOnConstant{Cases: []ConstCase{~w}}', [CaseStr]).
wam_instruction_to_go_literal(switch_on_structure(Table), GoLiteral) :-
    maplist(go_struct_case, Table, Cases),
    atomic_list_concat(Cases, ', ', CaseStr),
    format(atom(GoLiteral), '&SwitchOnStructure{Cases: []StructCase{~w}}', [CaseStr]).
wam_instruction_to_go_literal(switch_on_constant_a2(Table), GoLiteral) :-
    maplist(go_const_case, Table, Cases),
    atomic_list_concat(Cases, ', ', CaseStr),
    format(atom(GoLiteral), '&SwitchOnConstantA2{Cases: []ConstCase{~w}}', [CaseStr]).

%% Label instruction (WAM assembler pseudo-instruction)
wam_instruction_to_go_literal(label(L), GoLiteral) :-
    format(atom(GoLiteral), '// label: ~w', [L]).

%% Fallback
wam_instruction_to_go_literal(Instr, GoLiteral) :-
    format(atom(GoLiteral), '// TODO: ~w', [Instr]).

% --- Value literal helpers ---

go_value_literal(N, GoVal) :- integer(N), !, format(atom(GoVal), '&Integer{Val: ~w}', [N]).
go_value_literal(N, GoVal) :- float(N), !, format(atom(GoVal), '&Float{Val: ~w}', [N]).
go_value_literal(A, GoVal) :- atom(A), !, format(atom(GoVal), '&Atom{Name: "~w"}', [A]).
go_value_literal(T, GoVal) :- format(atom(GoVal), '&Atom{Name: "~w"}', [T]).

go_const_case(Val-Label, Case) :-
    go_value_literal(Val, GoVal),
    format(atom(Case), '{Val: ~w, Label: "~w"}', [GoVal, Label]).
go_struct_case(Functor-Label, Case) :-
    format(atom(Case), '{Functor: "~w", Label: "~w"}', [Functor, Label]).

% ============================================================================
% PHASE 2b: Compile WAM predicate → Go code array + labels
% ============================================================================

%% compile_wam_predicate_to_go(+Pred/Arity, +WamCode, +Options, -GoCode)
%  Takes WAM instruction output and produces Go code with instruction
%  slice and label map.
compile_wam_predicate_to_go(Pred/Arity, WamCode, _Options, GoCode) :-
    atom_string(Pred, PredStr),
    %% Parse WAM code lines into instruction terms
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_go(Lines, 0, GoLiterals, LabelEntries),
    %% Build Go slice
    maplist([Lit, Entry]>>(format(atom(Entry), '        ~w,', [Lit])), GoLiterals, Entries),
    atomic_list_concat(Entries, '\n', EntriesStr),
    %% Format labels
    maplist([Label-Idx, Entry]>>(format(atom(Entry), '        "~w": ~w,', [Label, Idx])), LabelEntries, LabelRows),
    atomic_list_concat(LabelRows, '\n', LabelsStr),
    format(atom(GoCode),
'// WAM-compiled predicate: ~w/~w
var ~wCode = []Instruction{
~w
}

var ~wLabels = map[string]int{
~w
}
', [PredStr, Arity, PredStr, EntriesStr, PredStr, LabelsStr]).

%% wam_lines_to_go(+Lines, +PC, -GoLits, -LabelEntries)
wam_lines_to_go([], _, [], []).
wam_lines_to_go([Line|Rest], PC, GoLits, Labels) :-
    split_string(Line, " \t", " \t", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_go(Rest, PC, GoLits, Labels)
    ;   CleanParts = [First|_],
        (   % Label line: "pred/2:" or "L_label:"
            sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            Labels = [LabelName-PC | RestLabels],
            wam_lines_to_go(Rest, PC, GoLits, RestLabels)
        ;   % Instruction line
            wam_line_to_go_literal(CleanParts, GoLit),
            GoLits = [GoLit | RestLits],
            NPC is PC + 1,
            wam_lines_to_go(Rest, NPC, RestLits, Labels)
        )
    ).

%% wam_line_to_go_literal(+Parts, -GoLit)
wam_line_to_go_literal(["get_constant", C, Ai], GoLit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    go_value_literal(CC, GoVal),
    format(atom(GoLit), '&GetConstant{C: ~w, Ai: "~w"}', [GoVal, CAi]).
wam_line_to_go_literal(["get_variable", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(atom(GoLit), '&GetVariable{Xn: "~w", Ai: "~w"}', [CXn, CAi]).
wam_line_to_go_literal(["get_value", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(atom(GoLit), '&GetValue{Xn: "~w", Ai: "~w"}', [CXn, CAi]).
wam_line_to_go_literal(["get_structure", FN, Ai], GoLit) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    format(atom(GoLit), '&GetStructure{Functor: "~w", Ai: "~w"}', [CFN, CAi]).
wam_line_to_go_literal(["get_list", Ai], GoLit) :-
    clean_comma(Ai, CAi),
    format(atom(GoLit), '&GetList{Ai: "~w"}', [CAi]).
wam_line_to_go_literal(["unify_variable", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    format(atom(GoLit), '&UnifyVariable{Xn: "~w"}', [CXn]).
wam_line_to_go_literal(["unify_value", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    format(atom(GoLit), '&UnifyValue{Xn: "~w"}', [CXn]).
wam_line_to_go_literal(["unify_constant", C], GoLit) :-
    clean_comma(C, CC),
    go_value_literal(CC, GoVal),
    format(atom(GoLit), '&UnifyConstant{C: ~w}', [GoVal]).

wam_line_to_go_literal(["put_constant", C, Ai], GoLit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    go_value_literal(CC, GoVal),
    format(atom(GoLit), '&PutConstant{C: ~w, Ai: "~w"}', [GoVal, CAi]).
wam_line_to_go_literal(["put_variable", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(atom(GoLit), '&PutVariable{Xn: "~w", Ai: "~w"}', [CXn, CAi]).
wam_line_to_go_literal(["put_value", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(atom(GoLit), '&PutValue{Xn: "~w", Ai: "~w"}', [CXn, CAi]).
wam_line_to_go_literal(["put_structure", FN, Ai], GoLit) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    format(atom(GoLit), '&PutStructure{Functor: "~w", Ai: "~w"}', [CFN, CAi]).
wam_line_to_go_literal(["put_list", Ai], GoLit) :-
    clean_comma(Ai, CAi),
    format(atom(GoLit), '&PutList{Ai: "~w"}', [CAi]).
wam_line_to_go_literal(["set_variable", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    format(atom(GoLit), '&SetVariable{Xn: "~w"}', [CXn]).
wam_line_to_go_literal(["set_value", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    format(atom(GoLit), '&SetValue{Xn: "~w"}', [CXn]).
wam_line_to_go_literal(["set_constant", C], GoLit) :-
    clean_comma(C, CC),
    go_value_literal(CC, GoVal),
    format(atom(GoLit), '&SetConstant{C: ~w}', [GoVal]).

wam_line_to_go_literal(["allocate"], '&Allocate{}').
wam_line_to_go_literal(["deallocate"], '&Deallocate{}').
wam_line_to_go_literal(["call", P, N], GoLit) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(atom(GoLit), '&Call{Pred: "~w", Arity: ~w}', [CP, CN]).
wam_line_to_go_literal(["execute", P], GoLit) :-
    clean_comma(P, CP),
    format(atom(GoLit), '&Execute{Pred: "~w"}', [CP]).
wam_line_to_go_literal(["proceed"], '&Proceed{}').
wam_line_to_go_literal(["builtin_call", Op, N], GoLit) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    format(atom(GoLit), '&BuiltinCall{Op: "~w", Arity: ~w}', [COp, CN]).

wam_line_to_go_literal(["try_me_else", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&TryMeElse{Label: "~w"}', [CL]).
wam_line_to_go_literal(["retry_me_else", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&RetryMeElse{Label: "~w"}', [CL]).
wam_line_to_go_literal(["trust_me"], '&TrustMe{}').

wam_line_to_go_literal(["switch_on_constant" | Table], GoLit) :-
    format_switch_table(Table, CaseStr),
    format(atom(GoLit), '&SwitchOnConstant{Cases: []ConstCase{~w}}', [CaseStr]).
wam_line_to_go_literal(["switch_on_structure" | Table], GoLit) :-
    format_switch_table(Table, CaseStr),
    format(atom(GoLit), '&SwitchOnStructure{Cases: []StructCase{~w}}', [CaseStr]).

wam_line_to_go_literal(Parts, GoLit) :-
    atomic_list_concat(Parts, " ", Line),
    format(atom(GoLit), '// TODO: ~w', [Line]).

clean_comma(S, Clean) :-
    (   sub_string(S, Before, 1, 0, ",")
    ->  sub_string(S, 0, Before, 1, Clean)
    ;   Clean = S
    ).

format_switch_table(Table, CaseStr) :-
    maplist(format_switch_case, Table, Cases),
    atomic_list_concat(Cases, ", ", CaseStr).

format_switch_case(Entry, Case) :-
    split_string(Entry, ":", " ", [ValStr, LabelStr]),
    clean_comma(LabelStr, Label),
    (   number_string(N, ValStr)
    ->  go_value_literal(N, GoVal)
    ;   go_value_literal(ValStr, GoVal)
    ),
    format(atom(Case), '{Val: ~w, Label: "~w"}', [GoVal, Label]).

% --- Value literal helpers ---

go_value_literal(atom(A), GoVal) :- !, format(atom(GoVal), '&Atom{Name: "~w"}', [A]).
go_value_literal(integer(I), GoVal) :- !, format(atom(GoVal), '&Integer{Val: ~w}', [I]).
go_value_literal(N, GoVal) :- integer(N), !, format(atom(GoVal), '&Integer{Val: ~w}', [N]).
go_value_literal(N, GoVal) :- float(N), !, format(atom(GoVal), '&Float{Val: ~w}', [N]).
go_value_literal(A, GoVal) :- atom(A), !, format(atom(GoVal), '&Atom{Name: "~w"}', [A]).
go_value_literal(T, GoVal) :- format(atom(GoVal), '&Atom{Name: "~w"}', [T]).

% ============================================================================
% PHASE 3: step_wam/3 → Go type switch cases
% ============================================================================

%% compile_step_wam_to_go(+Options, -GoCode)
%  Generates the Step() method body as a Go type switch.
compile_step_wam_to_go(_Options, GoCode) :-
    findall(Case, compile_go_step_case(Case), Cases),
    atomic_list_concat(Cases, '\n', CasesCode),
    format(atom(GoCode),
'// Step executes a single WAM instruction.
func (vm *WamState) Step(instr Instruction) bool {
    switch i := instr.(type) {
~w
    default:
        return false
    }
}', [CasesCode]).

compile_go_step_case(CaseCode) :-
    wam_go_case(GoType, BodyCode),
    format(atom(CaseCode),
        '    case *~w:\n~w', [GoType, BodyCode]).

% --- Head Unification Instructions ---

wam_go_case('GetConstant', '        val, ok := vm.Regs[i.Ai]
        if !ok { return false }
        if isUnbound(val) {
            vm.trailBinding(i.Ai)
            vm.Regs[i.Ai] = i.C
            vm.PC++
            return true
        }
        if valueEquals(val, i.C) {
            vm.PC++
            return true
        }
        return false').

wam_go_case('GetVariable', '        val, ok := vm.Regs[i.Ai]
        if !ok { return false }
        vm.trailBinding(i.Xn)
        vm.putReg(i.Xn, val)
        vm.PC++
        return true').

wam_go_case('GetValue', '        valA, okA := vm.Regs[i.Ai]
        valX := vm.getReg(i.Xn)
        if !okA { return false }
        if valueEquals(valA, valX) {
            vm.PC++
            return true
        }
        if isUnbound(valA) {
            vm.trailBinding(i.Ai)
            vm.Regs[i.Ai] = valX
            vm.PC++
            return true
        }
        if isUnbound(valX) {
            vm.trailBinding(i.Xn)
            vm.putReg(i.Xn, valA)
            vm.PC++
            return true
        }
        return false').

wam_go_case('GetStructure', '        val, ok := vm.Regs[i.Ai]
        if !ok { return false }
        if isUnbound(val) {
            addr := len(vm.Heap)
            vm.Heap = append(vm.Heap, &Atom{Name: "str(" + i.Functor + ")"})
            vm.trailBinding(i.Ai)
            vm.Regs[i.Ai] = &Ref{Addr: addr}
            arity := parseFunctorArity(i.Functor)
            vm.Stack = append(vm.Stack, &WriteCtx{N: arity})
            vm.PC++
            return true
        }
        if ref, ok := val.(*Ref); ok {
            if atom, ok := vm.Heap[ref.Addr].(*Atom); ok {
                if atom.Name == "str("+i.Functor+")" {
                    arity := parseFunctorArity(i.Functor)
                    args := vm.heapSubargs(ref.Addr+1, arity)
                    vm.Stack = append(vm.Stack, &UnifyCtx{Args: args})
                    vm.PC++
                    return true
                }
            }
        }
        if f, args := decompose(val); f != "" {
            check := fmt.Sprintf("%s/%d", f, len(args))
            if check == i.Functor {
                vm.Stack = append(vm.Stack, &UnifyCtx{Args: args})
                vm.PC++
                return true
            }
        }
        return false').

wam_go_case('GetList', '        val, ok := vm.Regs[i.Ai]
        if !ok { return false }
        if isUnbound(val) {
            addr := len(vm.Heap)
            vm.Heap = append(vm.Heap, &Atom{Name: "str(./2)"})
            vm.trailBinding(i.Ai)
            vm.Regs[i.Ai] = &Ref{Addr: addr}
            vm.Stack = append(vm.Stack, &WriteCtx{N: 2})
            vm.PC++
            return true
        }
        if list, ok := val.(*List); ok && len(list.Elements) > 0 {
            head := list.Elements[0]
            tail := &List{Elements: list.Elements[1:]}
            vm.Stack = append(vm.Stack, &UnifyCtx{Args: []Value{head, tail}})
            vm.PC++
            return true
        }
        return false').

wam_go_case('UnifyVariable', '        if ctx := vm.peekUnifyCtx(); ctx != nil && len(ctx.Args) > 0 {
            arg := ctx.Args[0]
            ctx.Args = ctx.Args[1:]
            if len(ctx.Args) == 0 { vm.popStack() }
            vm.trailBinding(i.Xn)
            vm.putReg(i.Xn, arg)
            vm.PC++
            return true
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            addr := len(vm.Heap)
            v := &Unbound{Name: fmt.Sprintf("_H%d", addr)}
            vm.Heap = append(vm.Heap, v)
            wctx.N--
            if wctx.N == 0 { vm.popStack() }
            vm.putReg(i.Xn, v)
            vm.PC++
            return true
        }
        return false').

wam_go_case('UnifyValue', '        if ctx := vm.peekUnifyCtx(); ctx != nil && len(ctx.Args) > 0 {
            expected := ctx.Args[0]
            ctx.Args = ctx.Args[1:]
            if len(ctx.Args) == 0 { vm.popStack() }
            actual := vm.getReg(i.Xn)
            if valueEquals(expected, actual) || isUnbound(expected) || isUnbound(actual) {
                if isUnbound(actual) { vm.putReg(i.Xn, expected) }
                vm.PC++
                return true
            }
            return false
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            val := vm.getReg(i.Xn)
            vm.Heap = append(vm.Heap, val)
            wctx.N--
            if wctx.N == 0 { vm.popStack() }
            vm.PC++
            return true
        }
        return false').

wam_go_case('UnifyConstant', '        if ctx := vm.peekUnifyCtx(); ctx != nil && len(ctx.Args) > 0 {
            expected := ctx.Args[0]
            ctx.Args = ctx.Args[1:]
            if len(ctx.Args) == 0 { vm.popStack() }
            if valueEquals(expected, i.C) || isUnbound(expected) {
                vm.PC++
                return true
            }
            return false
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            vm.Heap = append(vm.Heap, i.C)
            wctx.N--
            if wctx.N == 0 { vm.popStack() }
            vm.PC++
            return true
        }
        return false').

% --- Body Construction Instructions ---

wam_go_case('PutConstant', '        vm.Regs[i.Ai] = i.C
        vm.PC++
        return true').

wam_go_case('PutVariable', '        v := &Unbound{Name: i.Xn}
        vm.putReg(i.Xn, v)
        vm.Regs[i.Ai] = v
        vm.PC++
        return true').

wam_go_case('PutValue', '        val := vm.getReg(i.Xn)
        vm.Regs[i.Ai] = val
        vm.PC++
        return true').

wam_go_case('PutStructure', '        addr := len(vm.Heap)
        vm.Heap = append(vm.Heap, &Atom{Name: "str(" + i.Functor + ")"})
        vm.Regs[i.Ai] = &Ref{Addr: addr}
        arity := parseFunctorArity(i.Functor)
        vm.Stack = append(vm.Stack, &WriteCtx{N: arity})
        vm.PC++
        return true').

wam_go_case('PutList', '        addr := len(vm.Heap)
        vm.Heap = append(vm.Heap, &Atom{Name: "str(./2)"})
        vm.Regs[i.Ai] = &Ref{Addr: addr}
        vm.Stack = append(vm.Stack, &WriteCtx{N: 2})
        vm.PC++
        return true').

wam_go_case('SetVariable', '        addr := len(vm.Heap)
        v := &Unbound{Name: fmt.Sprintf("_H%d", addr)}
        vm.Heap = append(vm.Heap, v)
        vm.putReg(i.Xn, v)
        vm.PC++
        return true').

wam_go_case('SetValue', '        val := vm.getReg(i.Xn)
        vm.Heap = append(vm.Heap, val)
        vm.PC++
        return true').

wam_go_case('SetConstant', '        vm.Heap = append(vm.Heap, i.C)
        vm.PC++
        return true').

% --- Control Instructions ---

wam_go_case('Allocate', '        vm.Stack = append(vm.Stack, &EnvFrame{CP: vm.CP})
        vm.PC++
        return true').

wam_go_case('Deallocate', '        if env := vm.popEnvFrame(); env != nil {
            vm.CP = env.CP
        }
        vm.PC++
        return true').

wam_go_case('Call', '        vm.CP = vm.PC + 1
        if pc, ok := vm.Labels[i.Pred]; ok {
            vm.PC = pc
            return true
        }
        return false').

wam_go_case('Execute', '        if pc, ok := vm.Labels[i.Pred]; ok {
            vm.PC = pc
            return true
        }
        return false').

wam_go_case('Proceed', '        if vm.CP > 0 {
            vm.PC = vm.CP
        } else {
            vm.PC = 0 // halt
        }
        return true').

wam_go_case('BuiltinCall', '        result := vm.executeBuiltin(i.Op, i.Arity)
        if result {
            vm.PC++
        }
        return result').

% --- Choice Point Instructions ---

wam_go_case('TryMeElse', '        nextPC := 0
        if pc, ok := vm.Labels[i.Label]; ok {
            nextPC = pc
        }
        vm.pushChoicePoint(nextPC)
        vm.PC++
        return true').

wam_go_case('RetryMeElse', '        if pc, ok := vm.Labels[i.Label]; ok {
            if len(vm.ChoicePoints) > 0 {
                vm.ChoicePoints[len(vm.ChoicePoints)-1].NextPC = pc
            }
        }
        vm.PC++
        return true').

wam_go_case('TrustMe', '        if len(vm.ChoicePoints) > 0 {
            vm.ChoicePoints = vm.ChoicePoints[:len(vm.ChoicePoints)-1]
        }
        vm.PC++
        return true').

% --- Indexing Instructions ---

wam_go_case('SwitchOnConstant', '        if val, ok := vm.Regs["A1"]; ok && !isUnbound(val) {
            for _, c := range i.Cases {
                if valueEquals(c.Val, val) {
                    if pc, ok := vm.Labels[c.Label]; ok {
                        vm.PC = pc
                        return true
                    }
                }
            }
        }
        vm.PC++
        return true').

wam_go_case('SwitchOnStructure', '        if val, ok := vm.Regs["A1"]; ok {
            if f, args := decompose(val); f != "" {
                key := fmt.Sprintf("%s/%d", f, len(args))
                for _, c := range i.Cases {
                    if c.Functor == key {
                        if pc, ok := vm.Labels[c.Label]; ok {
                            vm.PC = pc
                            return true
                        }
                    }
                }
            }
        }
        vm.PC++
        return true').

wam_go_case('SwitchOnConstantA2', '        if val, ok := vm.Regs["A2"]; ok && !isUnbound(val) {
            for _, c := range i.Cases {
                if valueEquals(c.Val, val) {
                    if pc, ok := vm.Labels[c.Label]; ok {
                        vm.PC = pc
                        return true
                    }
                }
            }
        }
        vm.PC++
        return true').

% ============================================================================
% PHASE 3b: Helper functions → Go
% ============================================================================

%% compile_wam_helpers_to_go(+Options, -GoCode)
compile_wam_helpers_to_go(_Options, GoCode) :-
    format(atom(GoCode),
'// Run executes the WAM instruction loop until halt or failure.
func (vm *WamState) Run() bool {
    for {
        if vm.PC == 0 {
            return true
        }
        instr := vm.fetch()
        if instr == nil {
            return false
        }
        if !vm.Step(instr) {
            if !vm.backtrack() {
                return false
            }
        }
    }
}

// backtrack restores state from the most recent choice point.
func (vm *WamState) backtrack() bool {
    if len(vm.ChoicePoints) == 0 {
        return false
    }
    cp := vm.ChoicePoints[len(vm.ChoicePoints)-1]
    vm.unwindTrail(cp.TrailMark)
    vm.Regs = copyMap(cp.Regs)
    vm.PC = cp.NextPC
    vm.CP = cp.CP
    return true
}

// unwindTrail restores bindings back to the given trail mark.
func (vm *WamState) unwindTrail(mark int) {
    for len(vm.Trail) > mark {
        entry := vm.Trail[len(vm.Trail)-1]
        vm.Trail = vm.Trail[:len(vm.Trail)-1]
        delete(vm.Regs, entry.Var)
    }
}

// fetch retrieves the instruction at the current PC.
func (vm *WamState) fetch() Instruction {
    if vm.PC >= 0 && vm.PC < len(vm.Code) {
        return vm.Code[vm.PC]
    }
    return nil
}
', []).
