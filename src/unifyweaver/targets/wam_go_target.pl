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
:- use_module('../core/recursive_kernel_detection',
             [detect_recursive_kernel/4, kernel_metadata/4]).
:- use_module('../core/template_system').
:- use_module('../bindings/go_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../targets/go_target', [compile_predicate_to_go/3]).

:- discontiguous wam_go_case/2.
:- discontiguous wam_line_to_go_literal/4.

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
compile_predicates_for_project(Predicates, Options, Code) :-
    classify_predicates(Predicates, Options, Classified),
    collect_wam_entries(Classified, Options, 0, WamEntries, AllInstrParts, AllLabelEntries),
    compile_shared_foreign_setup(Classified, Options, SharedForeignSetup),
    (   WamEntries \== []
    ->  atomic_list_concat(AllInstrParts, '\n', AllInstrs),
        atomic_list_concat(AllLabelEntries, '\n', AllLabels),
        format(atom(SharedCode),
'~w

var sharedWamCodeRaw = []Instruction{
~w
}

var sharedWamLabels = map[string]int{
~w
}

var sharedWamCode = resolveInstructions(sharedWamCodeRaw, sharedWamLabels)
', [SharedForeignSetup, AllInstrs, AllLabels])
    ;   SharedCode = ""
    ),
    generate_predicate_codes(Classified, WamEntries, PredCodes),
    atomic_list_concat(PredCodes, '\n\n', PredicatesCode),
    (   SharedCode == ""
    ->  Code = PredicatesCode
    ;   format(atom(Code), '~w~n~w', [SharedCode, PredicatesCode])
    ).

classify_predicates([], _, []).
classify_predicates([PredIndicator|Rest], Options, [Entry|RestEntries]) :-
    predicate_indicator_parts(PredIndicator, Module, Pred, Arity),
    (   go_foreign_spec(Module:Pred/Arity, Options, _SetupOps, _RewriteCalls, _EntryPred/_EntryArity),
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode)
    ->  compile_wam_predicate_to_go(Module:Pred/Arity, WamCode, Options, PredCode),
        format(user_error, '  ~w/~w: WAM fallback (foreign)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, wam_foreign, PredCode)
    ;   catch(
            go_target:compile_predicate_to_go(Module:Pred/Arity,
                [include_package(false)|Options], PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, native, PredCode)
    ;   option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode)
    ->  (   go_foreign_spec(Module:Pred/Arity, Options, _SetupOps, _RewriteCalls, _EntryPred/_EntryArity)
        ->  compile_wam_predicate_to_go(Module:Pred/Arity, WamCode, Options, PredCode),
            format(user_error, '  ~w/~w: WAM fallback (foreign)~n', [Pred, Arity]),
            Entry = classified(Module, Pred, Arity, wam_foreign, PredCode)
        ;   format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
            Entry = classified(Module, Pred, Arity, wam, WamCode)
        )
    ;   format(atom(PredCode), '// ~w/~w: compilation failed', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, failed, PredCode)
    ),
    classify_predicates(Rest, Options, RestEntries).

collect_wam_entries([], _, _, [], [], []).
collect_wam_entries([classified(Module, Pred, Arity, wam, WamCode)|Rest], Options, PC,
                    [wam_entry(Pred, Arity, PC)|RestEntries],
                    AllInstrs, AllLabels) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_go(Lines, PC, Module:Pred/Arity, Options, GoLiterals, LabelEntries),
    maplist([Lit, Entry]>>(format(atom(Entry), '        ~w,', [Lit])), GoLiterals, InstrEntries),
    maplist([Label-Idx, Entry]>>(format(atom(Entry), '        "~w": ~w,', [Label, Idx])), LabelEntries, LabelRows),
    length(GoLiterals, InstrCount),
    NextPC is PC + InstrCount,
    collect_wam_entries(Rest, Options, NextPC, RestEntries, RestInstrs, RestLabels),
    append(InstrEntries, RestInstrs, AllInstrs),
    append(LabelRows, RestLabels, AllLabels).
collect_wam_entries([_|Rest], Options, PC, Entries, Instrs, Labels) :-
    collect_wam_entries(Rest, Options, PC, Entries, Instrs, Labels).

generate_predicate_codes([], _, []).
generate_predicate_codes([classified(_, _Pred, _Arity, native, PredCode)|Rest], WamEntries,
                         [Code|RestCodes]) :-
    format(atom(Code), '// Strategy: native~n~w', [PredCode]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, wam_foreign, PredCode)|Rest], WamEntries,
                         [Code|RestCodes]) :-
    format(atom(Code), '// Strategy: wam_foreign~n~w', [PredCode]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, Pred, Arity, wam, _WamCode)|Rest], WamEntries,
                         [Code|RestCodes]) :-
    member(wam_entry(Pred, Arity, StartPC), WamEntries),
    compile_wam_predicate_to_go_shared(Pred/Arity, StartPC, Code),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, failed, PredCode)|Rest], WamEntries,
                         [Code|RestCodes]) :-
    format(atom(Code), '// Strategy: failed~n~w', [PredCode]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).

compile_shared_foreign_setup(Classified, Options, Code) :-
    findall(Line,
        ( member(classified(Module, Pred, Arity, wam_foreign, _), Classified),
          go_foreign_spec(Module:Pred/Arity, Options, SetupOps, _RewriteCalls, _EntryPred/_EntryArity),
          maplist(go_foreign_setup_line, SetupOps, SetupLines),
          member(Line, SetupLines)
        ),
        RawLines),
    sort(RawLines, Lines),
    (   Lines == []
    ->  Body = ""
    ;   atomic_list_concat(Lines, '\n', Body)
    ),
    format(atom(Code),
'func setupSharedForeignPredicates(vm *WamState) {
~w
}', [Body]).

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
    escape_go_string(Op, EscapedOp),
    format(atom(GoLiteral), '&BuiltinCall{Op: "~w", Arity: ~w}', [EscapedOp, N]).

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
compile_wam_predicate_to_go(PredIndicator, WamCode, Options, GoCode) :-
    predicate_indicator_parts(PredIndicator, _Module, Pred, Arity),
    atom_string(Pred, PredStr),
    capitalize_atom(Pred, CapPred),
    build_go_wam_arg_list(Arity, ArgList),
    build_go_wam_arg_setup(Arity, ArgSetup),
    (   foreign_wrapper_setup(PredIndicator, WamCode, Options, InstrSetup, ForeignSetup, RunExpr)
    ->  format(atom(GoCode),
'// WAM-compiled predicate: ~w/~w
func ~w(~w) bool {
    vm := NewWamState(nil, nil)
~w
~w
~w
    return ~w
}
', [PredStr, Arity, CapPred, ArgList, InstrSetup, ForeignSetup, ArgSetup, RunExpr])
    ;   atom_string(WamCode, WamStr),
        split_string(WamStr, "\n", "", Lines),
        wam_lines_to_go(Lines, 0, Pred/Arity, Options, GoLiterals, LabelEntries),
        maplist([Lit, Entry]>>(format(atom(Entry), '        ~w,', [Lit])), GoLiterals, Entries),
        atomic_list_concat(Entries, '\n', EntriesStr),
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

var ~wResolvedCode = resolveInstructions(~wCode, ~wLabels)

func ~w(~w) bool {
    vm := NewWamState(~wResolvedCode, ~wLabels)
~w
    return vm.Run()
}
', [PredStr, Arity, CapPred, EntriesStr, CapPred, LabelsStr,
    CapPred, CapPred, CapPred, CapPred, ArgList, CapPred, CapPred, ArgSetup])
    ).

compile_wam_predicate_to_go_shared(Pred/Arity, StartPC, GoCode) :-
    atom_string(Pred, PredStr),
    capitalize_atom(Pred, CapPred),
    build_go_wam_arg_list(Arity, ArgList),
    build_go_wam_arg_setup(Arity, ArgSetup),
    format(atom(GoCode),
'// Strategy: wam
// WAM-compiled predicate: ~w/~w (shared table, pc=~w)
var ~wCode = sharedWamCode
var ~wLabels = sharedWamLabels
const ~wStartPC = ~w

func ~w(~w) bool {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = ~w
~w
    return vm.Run()
}
', [PredStr, Arity, StartPC,
    CapPred, CapPred, CapPred, StartPC,
    CapPred, ArgList, StartPC, ArgSetup]).

build_go_wam_arg_list(0, "") :- !.
build_go_wam_arg_list(Arity, ArgList) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>format(atom(S), 'a~w Value', [I]), Indices, Parts),
    atomic_list_concat(Parts, ', ', ArgList).

build_go_wam_arg_setup(0, "") :- !.
build_go_wam_arg_setup(Arity, Setup) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>format(atom(S), '    vm.Regs["A~w"] = a~w', [I, I]), Indices, Lines),
    atomic_list_concat(Lines, '\n', Setup).

foreign_wrapper_setup(PredIndicator, _WamCode, Options, InstrSetup, Setup, RunExpr) :-
    go_foreign_spec(PredIndicator, Options, SetupOps, _RewriteCalls, EntryPred/EntryArity),
    InstrSetup = '    vm.PC = 1',
    go_foreign_setup_code(SetupOps, Setup),
    format(atom(RunExpr), 'vm.executeForeignPredicate("~w", ~w)', [EntryPred, EntryArity]).

go_foreign_spec(_PredArity, Options, _SetupOps, _RewriteCalls, _EntryPredArity) :-
    option(no_kernels(true), Options),
    !,
    fail.
go_foreign_spec(PredArity, Options, SetupOps, RewriteCalls, EntryPredArity) :-
    option(foreign_lowering(ForeignSpec), Options),
    nonvar(ForeignSpec),
    ForeignSpec \== true,
    (   is_list(ForeignSpec)
    ->  member(Spec, ForeignSpec),
        go_foreign_spec_term(PredArity, Spec, SetupOps, RewriteCalls, EntryPredArity)
    ;   go_foreign_spec_term(PredArity, ForeignSpec, SetupOps, RewriteCalls, EntryPredArity)
    ),
    !.
go_foreign_spec(PredIndicator, Options, SetupOps, RewriteCalls, EntryPredArity) :-
    option(foreign_lowering(true), Options),
    go_foreign_lowering_spec(PredIndicator, SetupOps, RewriteCalls, EntryPredArity),
    !.

go_foreign_spec_term(Pred/Arity,
        foreign_predicate(Pred/Arity, SetupOps, RewriteCalls),
        SetupOps,
        RewriteCalls,
        Pred/Arity) :-
    is_list(SetupOps),
    is_list(RewriteCalls).

go_foreign_setup_code([], "").
go_foreign_setup_code(Ops, Setup) :-
    maplist(go_foreign_setup_line, Ops, Lines),
    atomic_list_concat(Lines, '\n', Setup).

go_foreign_setup_line(register_foreign_native_kind(Pred/Arity, Kind), Line) :-
    format(atom(Line), '    vm.registerForeignNativeKind("~w/~w", "~w")', [Pred, Arity, Kind]).
go_foreign_setup_line(register_foreign_result_layout(Pred/Arity, tuple(ResultArity)), Line) :-
    format(atom(Line), '    vm.registerForeignResultLayout("~w/~w", "tuple:~w")', [Pred, Arity, ResultArity]),
    !.
go_foreign_setup_line(register_foreign_result_layout(Pred/Arity, Layout), Line) :-
    format(atom(Line), '    vm.registerForeignResultLayout("~w/~w", "~w")', [Pred, Arity, Layout]).
go_foreign_setup_line(register_foreign_result_mode(Pred/Arity, Mode), Line) :-
    format(atom(Line), '    vm.registerForeignResultMode("~w/~w", "~w")', [Pred, Arity, Mode]).
go_foreign_setup_line(register_foreign_string_config(Pred/Arity, Key, ValuePred/ValueArity), Line) :-
    format(atom(Line), '    vm.registerForeignStringConfig("~w/~w", "~w", "~w/~w")',
        [Pred, Arity, Key, ValuePred, ValueArity]).
go_foreign_setup_line(register_foreign_string_config(Pred/Arity, Key, Value), Line) :-
    format(atom(Line), '    vm.registerForeignStringConfig("~w/~w", "~w", "~w")',
        [Pred, Arity, Key, Value]).
go_foreign_setup_line(register_foreign_usize_config(Pred/Arity, Key, Value), Line) :-
    format(atom(Line), '    vm.registerForeignUsizeConfig("~w/~w", "~w", ~w)', [Pred, Arity, Key, Value]).
go_foreign_setup_line(register_indexed_atom_fact2(Pred/Arity, Pairs), Line) :-
    go_fact_pairs_literal(Pairs, Literal),
    format(atom(Line), '    vm.registerIndexedAtomFact2Pairs("~w/~w", []AtomPair{~w})', [Pred, Arity, Literal]).
go_foreign_setup_line(register_indexed_weighted_edge(Pred/Arity, Triples), Line) :-
    go_fact_triples_literal(Triples, Literal),
    format(atom(Line), '    vm.registerIndexedWeightedEdgeTriples("~w/~w", []WeightedEdgeTriple{~w})', [Pred, Arity, Literal]).

go_fact_pairs_literal(Pairs, Literal) :-
    maplist(go_fact_pair_literal, Pairs, PairLiterals),
    atomic_list_concat(PairLiterals, ', ', Literal).

go_fact_pair_literal(Left-Right, Literal) :-
    format(atom(Literal), '{Left: "~w", Right: "~w"}', [Left, Right]).

go_fact_triples_literal(Triples, Literal) :-
    maplist(go_fact_triple_literal, Triples, TripleLiterals),
    atomic_list_concat(TripleLiterals, ', ', Literal).

go_fact_triple_literal(Left-Right-Weight, Literal) :-
    format(atom(Literal), '{Left: "~w", Right: "~w", Weight: ~15g}', [Left, Right, Weight]).

capitalize_atom(Atom, Cap) :-
    atom_codes(Atom, [First|Rest]),
    code_type(FirstUpper, to_upper(First)),
    atom_codes(Cap, [FirstUpper|Rest]).

predicate_indicator_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
predicate_indicator_parts(Pred/Arity, user, Pred, Arity).

go_foreign_lowering_spec(PredIndicator, SetupOps, RewriteCalls, EntryPred/EntryArity) :-
    predicate_indicator_parts(PredIndicator, Module, Pred, Arity),
    functor(Head, Pred, Arity),
    findall(Head-Body, Module:clause(Head, Body), Clauses),
    Clauses \= [],
    go_recursive_kernel(Module, Pred, Arity, Clauses, Kernel),
    go_recursive_kernel_spec(Kernel, SetupOps, RewriteCalls, EntryPred/EntryArity).

go_recursive_kernel(_Module, Pred, Arity, Clauses, recursive_kernel(countdown_sum2, Pred/Arity, [])) :-
    go_foreign_lowerable_countdown_sum(Pred, Arity, Clauses).
go_recursive_kernel(_Module, Pred, Arity, Clauses, recursive_kernel(list_suffix2, Pred/Arity, [])) :-
    go_foreign_lowerable_list_suffix(Pred, Arity, Clauses).
go_recursive_kernel(_Module, Pred, Arity, Clauses, recursive_kernel(list_suffixes2, Pred/Arity, [])) :-
    go_foreign_lowerable_list_suffixes(Pred, Arity, Clauses).
go_recursive_kernel(Module, Pred, Arity, Clauses,
        recursive_kernel(weighted_shortest_path3, Pred/Arity,
            [weight_pred(WeightPred/3), fact_triples(FactTriples)])) :-
    go_foreign_lowerable_weighted_shortest_path(Module, Pred, Arity, Clauses, WeightPred/3, FactTriples).
go_recursive_kernel(Module, Pred, Arity, Clauses, Kernel) :-
    go_foreign_lowerable_astar_shortest_path(Module, Pred, Arity, Clauses, Kernel).
go_recursive_kernel(Module, Pred, Arity, Clauses, Kernel) :-
    detect_recursive_kernel(Pred, Arity, Clauses, Kernel0),
    go_supported_shared_kernel(Kernel0),
    go_recursive_kernel_with_facts(Module, Kernel0, Kernel).

go_supported_shared_kernel(recursive_kernel(transitive_closure2, _, _)).
go_supported_shared_kernel(recursive_kernel(transitive_distance3, _, _)).
go_supported_shared_kernel(recursive_kernel(transitive_parent_distance4, _, _)).
go_supported_shared_kernel(recursive_kernel(transitive_step_parent_distance5, _, _)).
go_recursive_kernel_with_facts(Module,
        recursive_kernel(KernelKind, PredIndicator, KernelConfig0),
        recursive_kernel(KernelKind, PredIndicator,
            [edge_pred(EdgePred/2), fact_pairs(FactPairs)])) :-
    member(KernelKind, [transitive_closure2, transitive_distance3,
        transitive_parent_distance4, transitive_step_parent_distance5]),
    member(edge_pred(EdgePred/2), KernelConfig0),
    go_binary_edge_fact_pairs(Module, EdgePred/2, FactPairs),
    FactPairs \= [].
go_recursive_kernel_spec(recursive_kernel(KernelKind, PredIndicator, KernelConfig),
        SetupOps, RewriteCalls, PredIndicator) :-
    go_recursive_kernel_setup_ops(KernelKind, PredIndicator, KernelConfig, SetupOps),
    RewriteCalls = [PredIndicator].

go_recursive_kernel_setup_ops(KernelKind, PredIndicator, KernelConfig,
        [ register_foreign_native_kind(PredIndicator, NativeKind),
          register_foreign_result_layout(PredIndicator, ResultLayout),
          register_foreign_result_mode(PredIndicator, ResultMode)
        |ConfigOps]) :-
    go_recursive_kernel_metadata(KernelKind, KernelConfig, NativeKind, ResultLayout, ResultMode),
    go_recursive_kernel_config_ops(PredIndicator, KernelConfig, ConfigOps).

go_recursive_kernel_metadata(countdown_sum2, _KernelConfig, countdown_sum2, tuple(1), deterministic).
go_recursive_kernel_metadata(list_suffix2, _KernelConfig, list_suffix2, tuple(1), stream).
go_recursive_kernel_metadata(list_suffixes2, _KernelConfig, list_suffixes2, tuple(1), deterministic_collection).
go_recursive_kernel_metadata(astar_shortest_path4, _KernelConfig, astar_shortest_path4, tuple(1), stream).
go_recursive_kernel_metadata(KernelKind, KernelConfig, NativeKind, ResultLayout, ResultMode) :-
    kernel_metadata(recursive_kernel(KernelKind, _PredIndicator, KernelConfig),
        NativeKind, ResultLayout, ResultMode).

go_recursive_kernel_config_ops(_PredIndicator, [], []).
go_recursive_kernel_config_ops(PredIndicator, [edge_pred(EdgePred/2), fact_pairs(FactPairs)], [
        register_foreign_string_config(PredIndicator, edge_pred, EdgePred/2),
        register_indexed_atom_fact2(EdgePred/2, FactPairs)
    ]).
go_recursive_kernel_config_ops(PredIndicator, [weight_pred(WeightPred/3), fact_triples(FactTriples)], [
        register_foreign_string_config(PredIndicator, weight_pred, WeightPred/3),
        register_indexed_weighted_edge(WeightPred/3, FactTriples)
    ]).
go_recursive_kernel_config_ops(PredIndicator,
        [weight_pred(WeightPred/3), fact_triples(FactTriples),
         direct_dist_pred(DirectPred/3), direct_triples(DirectTriples),
         dimensionality(Dim)], [
        register_foreign_string_config(PredIndicator, weight_pred, WeightPred/3),
        register_indexed_weighted_edge(WeightPred/3, FactTriples),
        register_foreign_string_config(PredIndicator, direct_dist_pred, DirectPred/3),
        register_indexed_weighted_edge(DirectPred/3, DirectTriples),
        register_foreign_usize_config(PredIndicator, dimensionality, Dim)
    ]).
go_recursive_kernel_config_ops(PredIndicator,
        [weight_pred(WeightPred/3), fact_triples(FactTriples),
         dimensionality(Dim)], [
        register_foreign_string_config(PredIndicator, weight_pred, WeightPred/3),
        register_indexed_weighted_edge(WeightPred/3, FactTriples),
        register_foreign_usize_config(PredIndicator, dimensionality, Dim)
    ]).

go_binary_edge_fact_pairs(Module, EdgePred/2, FactPairs) :-
    findall(Left-Right,
        ( functor(EdgeHead, EdgePred, 2),
          Module:clause(EdgeHead, true),
          EdgeHead =.. [EdgePred, Left, Right],
          atom(Left),
          atom(Right)
        ),
        FactPairs).

go_weighted_edge_fact_triples(Module, WeightPred/3, FactTriples) :-
    findall(Left-Right-Weight,
        ( functor(WeightHead, WeightPred, 3),
          Module:clause(WeightHead, true),
          WeightHead =.. [WeightPred, Left, Right, Weight],
          atom(Left),
          atom(Right),
          number(Weight)
        ),
        FactTriples).

go_foreign_lowerable_countdown_sum(Pred, 2, Clauses) :-
    member(BaseHead-true, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, 0, 0],
    RecHead =.. [Pred, N, Sum],
    RecBody = (GtGoal, (StepGoal, (RecGoal, SumGoal))),
    GtGoal =.. [>, N, 0],
    StepGoal =.. [is, PrevN, StepExpr],
    (   StepExpr =.. [-, N, 1]
    ;   StepExpr =.. [+, N, -1]
    ),
    RecGoal =.. [Pred, PrevN, PrevSum],
    SumGoal =.. [is, Sum, SumExpr],
    SumExpr =.. [+, PrevSum, N].

go_foreign_lowerable_list_suffix(Pred, 2, Clauses) :-
    member(BaseHead-true, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, BaseList, BaseList],
    var(BaseList),
    RecHead =.. [Pred, InputList, Suffix],
    InputList = [_|Tail],
    RecBody =.. [Pred, Tail, Suffix].

go_foreign_lowerable_list_suffixes(Pred, 2, Clauses) :-
    member(BaseHead-true, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead =.. [Pred, [], [[]]],
    RecHead =.. [Pred, [Head|Tail], [[Head|Tail]|Rest]],
    RecBody =.. [Pred, Tail, Rest].

go_foreign_lowerable_weighted_shortest_path(Module, Pred, 3, Clauses, WeightPred/3, FactTriples) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead \== RecHead,
    BaseHead =.. [Pred, BaseStart, BaseTarget, BaseWeight],
    BaseBody =.. [WeightPred, BaseStart, BaseTarget, BaseWeight],
    RecHead =.. [Pred, RecStart, RecTarget, RecCost],
    go_extract_weighted_rec_body(Pred, WeightPred, RecStart, RecTarget, RecCost, RecBody),
    go_weighted_edge_fact_triples(Module, WeightPred/3, FactTriples),
    FactTriples \= [].

go_extract_weighted_rec_body(Pred, WeightPred, Start, Target, Cost, Body) :-
    Body = (WeightGoal, (RecGoal, IsGoal)),
    WeightGoal =.. [WeightPred, Start, Mid, W],
    RecGoal =.. [Pred, Mid, Target, RestCost],
    IsGoal =.. [is, Cost, PlusExpr],
    (   PlusExpr =.. [+, W, RestCost]
    ;   PlusExpr =.. [+, RestCost, W]
    ),
    !.
go_extract_weighted_rec_body(Pred, WeightPred, Start, Target, Cost, Body) :-
    Body = (WeightGoal, (NegGoal, (RecGoal, IsGoal))),
    WeightGoal =.. [WeightPred, Start, Mid, W],
    NegGoal = (\+ _),
    RecGoal =.. [Pred, Mid, Target, RestCost],
    IsGoal =.. [is, Cost, PlusExpr],
    (   PlusExpr =.. [+, W, RestCost]
    ;   PlusExpr =.. [+, RestCost, W]
    ),
    !.

go_foreign_lowerable_astar_shortest_path(Module, Pred, 4, Clauses,
        recursive_kernel(astar_shortest_path4, Pred/4, KernelConfig)) :-
    member(BaseHead-BaseBody, Clauses),
    member(RecHead-RecBody, Clauses),
    BaseHead \== RecHead,
    BaseHead =.. [Pred, BaseStart, BaseTarget, _BaseDim, BaseWeight],
    BaseBody =.. [WeightPred, BaseStart, BaseTarget, BaseWeight],
    RecHead =.. [Pred, RecStart, RecTarget, RecDim, RecCost],
    RecBody = (WeightGoal, (RecGoal, IsGoal)),
    WeightGoal =.. [WeightPred, RecStart, Mid, W],
    RecGoal =.. [Pred, Mid, RecTarget, RecDim, RestCost],
    IsGoal =.. [is, RecCost, PlusExpr],
    (   PlusExpr =.. [+, W, RestCost]
    ;   PlusExpr =.. [+, RestCost, W]
    ),
    go_weighted_edge_fact_triples(Module, WeightPred/3, FactTriples),
    FactTriples \= [],
    go_foreign_astar_direct_pred(Module, WeightPred/3, DirectPred/3, DirectTriples),
    go_foreign_astar_dimensionality(Module, Dim),
    KernelConfig = [weight_pred(WeightPred/3), fact_triples(FactTriples),
                    direct_dist_pred(DirectPred/3), direct_triples(DirectTriples),
                    dimensionality(Dim)].

go_foreign_astar_direct_pred(Module, _FallbackPred, DirectPred/3, DirectTriples) :-
    go_weighted_edge_fact_triples(Module, direct_semantic_dist/3, DirectTriples),
    DirectTriples \= [],
    DirectPred = direct_semantic_dist,
    !.
go_foreign_astar_direct_pred(_Module, FallbackPred, DirectPred, DirectTriples) :-
    FallbackPred = DirectPred,
    DirectTriples = [].

go_foreign_astar_dimensionality(Module, Dim) :-
    (   current_predicate(Module:dimensionality/1),
        Module:dimensionality(Dim0)
    ;   current_predicate(user:dimensionality/1),
        user:dimensionality(Dim0)
    ),
    integer(Dim0),
    !,
    Dim = Dim0.
go_foreign_astar_dimensionality(_Module, 5).

%% wam_lines_to_go(+Lines, +PC, -GoLits, -LabelEntries)
wam_lines_to_go([], _, _, _, [], []).
wam_lines_to_go([Line|Rest], PC, PredIndicator, Options, GoLits, Labels) :-
    split_string(Line, " \t", " \t", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_go(Rest, PC, PredIndicator, Options, GoLits, Labels)
    ;   CleanParts = [First|_],
        (   % Label line: "pred/2:" or "L_label:"
            sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            Labels = [LabelName-PC | RestLabels],
            wam_lines_to_go(Rest, PC, PredIndicator, Options, GoLits, RestLabels)
        ;   % Instruction line
            wam_line_to_go_literal(CleanParts, PredIndicator, Options, GoLit),
            GoLits = [GoLit | RestLits],
            NPC is PC + 1,
            wam_lines_to_go(Rest, NPC, PredIndicator, Options, RestLits, Labels)
        )
    ).

%% wam_line_to_go_literal(+Parts, -GoLit)
%% parse_string_to_go_val(+Str, -GoVal)
parse_string_to_go_val(Str, GoVal) :-
    (   number_string(N, Str)
    ->  go_value_literal(N, GoVal)
    ;   go_value_literal(Str, GoVal)
    ).

wam_line_to_go_literal(["call", P, N], PredIndicator, Options, GoLit) :-
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    (   go_foreign_rewrite_call(Options, PredIndicator, CP, Num, ForeignPred, ForeignArity)
    ->  format(atom(GoLit), '&CallForeign{Pred: "~w", Arity: ~w}', [ForeignPred, ForeignArity])
    ;   format(atom(GoLit), '&Call{Pred: "~w", Arity: ~w}', [CP, CN])
    ).
wam_line_to_go_literal(["execute", P], PredIndicator, Options, GoLit) :-
    clean_comma(P, CP),
    (   go_foreign_rewrite_execute(Options, PredIndicator, CP, ForeignPred, ForeignArity)
    ->  format(atom(GoLit), '&CallForeign{Pred: "~w", Arity: ~w}', [ForeignPred, ForeignArity])
    ;   format(atom(GoLit), '&Execute{Pred: "~w"}', [CP])
    ).

wam_line_to_go_literal(["get_constant", C, Ai], GoLit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    parse_string_to_go_val(CC, GoVal),
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
    parse_string_to_go_val(CC, GoVal),
    format(atom(GoLit), '&UnifyConstant{C: ~w}', [GoVal]).

wam_line_to_go_literal(["put_constant", C, Ai], GoLit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    parse_string_to_go_val(CC, GoVal),
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
    parse_string_to_go_val(CC, GoVal),
    format(atom(GoLit), '&SetConstant{C: ~w}', [GoVal]).

wam_line_to_go_literal(["allocate"], '&Allocate{}').
wam_line_to_go_literal(["deallocate"], '&Deallocate{}').
wam_line_to_go_literal(["proceed"], '&Proceed{}').
wam_line_to_go_literal(["builtin_call", Op, N], GoLit) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    escape_go_string(COp, EscapedOp),
    format(atom(GoLit), '&BuiltinCall{Op: "~w", Arity: ~w}', [EscapedOp, CN]).

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

go_foreign_rewrite_call(Options, CurrentPred, TargetPredArity, Num, ForeignPred, ForeignArity) :-
    go_foreign_spec(CurrentPred, Options, _SetupOps, RewriteCalls, ForeignPred/ForeignArity),
    member(TargetPred/TargetArity, RewriteCalls),
    format(string(ExpectedTarget), "~w/~w", [TargetPred, TargetArity]),
    TargetPredArity == ExpectedTarget,
    Num =:= ForeignArity.
go_foreign_rewrite_call(Options, CurrentPred, TargetPredArity, Num, ForeignPred, ForeignArity) :-
    predicate_indicator_parts(CurrentPred, Module, _CurrentPred, _CurrentArity),
    target_predicate_parts(TargetPredArity, TargetPred, TargetArity),
    go_foreign_spec(Module:TargetPred/TargetArity, Options, _SetupOps, _RewriteCalls, ForeignPred/ForeignArity),
    Num =:= ForeignArity.

go_foreign_rewrite_execute(Options, CurrentPred, TargetPredArity, ForeignPred, ForeignArity) :-
    go_foreign_spec(CurrentPred, Options, _SetupOps, RewriteCalls, ForeignPred/ForeignArity),
    member(TargetPred/TargetArity, RewriteCalls),
    format(string(ExpectedTarget), "~w/~w", [TargetPred, TargetArity]),
    TargetPredArity == ExpectedTarget.
go_foreign_rewrite_execute(Options, CurrentPred, TargetPredArity, ForeignPred, ForeignArity) :-
    predicate_indicator_parts(CurrentPred, Module, _CurrentPred, _CurrentArity),
    target_predicate_parts(TargetPredArity, TargetPred, TargetArity),
    go_foreign_spec(Module:TargetPred/TargetArity, Options, _SetupOps, _RewriteCalls, ForeignPred/ForeignArity).

target_predicate_parts(TargetPredArity, Pred, Arity) :-
    split_string(TargetPredArity, "/", "", [PredStr, ArityStr]),
    atom_string(Pred, PredStr),
    number_string(Arity, ArityStr).

wam_line_to_go_literal(Parts, _PredIndicator, _Options, GoLit) :-
    wam_line_to_go_literal(Parts, GoLit).

clean_comma(S, Clean) :-
    (   sub_string(S, Before, 1, 0, ",")
    ->  sub_string(S, 0, Before, 1, Clean)
    ;   Clean = S
    ).

format_switch_table(Table, CaseStr) :-
    maplist(format_switch_case, Table, Cases),
    atomic_list_concat(Cases, ", ", CaseStr).

format_switch_case(Entry, Case) :-
    (   split_string(Entry, ":", " ", [ValStr, LabelStr])
    ->  clean_comma(LabelStr, Label),
        (   number_string(N, ValStr)
        ->  go_value_literal(N, GoVal)
        ;   go_value_literal(ValStr, GoVal)
        ),
        format(atom(Case), '{Val: ~w, Label: "~w"}', [GoVal, Label])
    ;   % Robustness fallback for malformed entries
        format(atom(Case), '{Val: &Atom{Name: "malformed"}, Label: "~w"}', [Entry])
    ).

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
            u := val.(*Unbound)
            vm.trailBinding(u.Name)
            vm.Regs[u.Name] = i.C
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
        if vm.Unify(valA, valX) {
            vm.PC++
            return true
        }
        return false').

wam_go_case('GetStructure', '        val, ok := vm.Regs[i.Ai]
        if !ok { return false }
        if isUnbound(val) {
            addr := len(vm.Heap)
            arity := parseFunctorArity(i.Functor)
            s := &Structure{Functor: i.Functor, Arity: arity, Args: make([]Value, arity)}
            vm.Heap = append(vm.Heap, s)
            vm.CurrentStruct = s
            vm.CurrentList = nil
            vm.trailBinding(i.Ai)
            vm.Regs[i.Ai] = &Ref{Addr: addr}
            vm.Stack = append(vm.Stack, &WriteCtx{N: arity})
            vm.PC++
            return true
        }
        val = vm.deref(val)
        if s, ok := val.(*Structure); ok {
            if s.Functor == i.Functor {
                vm.Stack = append(vm.Stack, &UnifyCtx{Args: s.Args})
                vm.PC++
                return true
            }
        }
        return false').

wam_go_case('GetList', '        val, ok := vm.Regs[i.Ai]
        if !ok { return false }
        if isUnbound(val) {
            addr := len(vm.Heap)
            l := &List{Elements: make([]Value, 2)}
            vm.Heap = append(vm.Heap, l)
            vm.CurrentList = l
            vm.CurrentStruct = nil
            vm.trailBinding(i.Ai)
            vm.Regs[i.Ai] = &Ref{Addr: addr}
            vm.Stack = append(vm.Stack, &WriteCtx{N: 2})
            vm.PC++
            return true
        }
        val = vm.deref(val)
        if list, ok := val.(*List); ok && len(list.Elements) > 0 {
            vm.Stack = append(vm.Stack, &UnifyCtx{Args: list.Elements})
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
            if vm.CurrentStruct != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = v
            } else if vm.CurrentList != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = v
            }
            wctx.N--
            if wctx.N == 0 { 
                vm.popStack() 
                vm.CurrentStruct = nil
                vm.CurrentList = nil
            }
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
            if vm.Unify(expected, actual) {
                vm.PC++
                return true
            }
            return false
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            val := vm.getReg(i.Xn)
            vm.Heap = append(vm.Heap, val)
            if vm.CurrentStruct != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = val
            } else if vm.CurrentList != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = val
            }
            wctx.N--
            if wctx.N == 0 { 
                vm.popStack() 
                vm.CurrentStruct = nil
                vm.CurrentList = nil
            }
            vm.PC++
            return true
        }
        return false').

wam_go_case('UnifyConstant', '        if ctx := vm.peekUnifyCtx(); ctx != nil && len(ctx.Args) > 0 {
            expected := ctx.Args[0]
            ctx.Args = ctx.Args[1:]
            if len(ctx.Args) == 0 { vm.popStack() }
            if valueEquals(vm.deref(expected), i.C) {
                vm.PC++
                return true
            }
            return false
        }
        if wctx := vm.peekWriteCtx(); wctx != nil && wctx.N > 0 {
            vm.Heap = append(vm.Heap, i.C)
            if vm.CurrentStruct != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = i.C
            } else if vm.CurrentList != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = i.C
            }
            wctx.N--
            if wctx.N == 0 { 
                vm.popStack() 
                vm.CurrentStruct = nil
                vm.CurrentList = nil
            }
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
        arity := parseFunctorArity(i.Functor)
        s := &Structure{Functor: i.Functor, Arity: arity, Args: make([]Value, arity)}
        vm.Heap = append(vm.Heap, s)
        vm.CurrentStruct = s
        vm.CurrentList = nil
        vm.Regs[i.Ai] = &Ref{Addr: addr}
        vm.Stack = append(vm.Stack, &WriteCtx{N: arity})
        vm.PC++
        return true').

wam_go_case('PutList', '        addr := len(vm.Heap)
        l := &List{Elements: make([]Value, 2)}
        vm.Heap = append(vm.Heap, l)
        vm.CurrentList = l
        vm.CurrentStruct = nil
        vm.Regs[i.Ai] = &Ref{Addr: addr}
        vm.Stack = append(vm.Stack, &WriteCtx{N: 2})
        vm.PC++
        return true').

wam_go_case('SetVariable', '        addr := len(vm.Heap)
        v := &Unbound{Name: fmt.Sprintf("_H%d", addr)}
        vm.Heap = append(vm.Heap, v)
        if vm.CurrentStruct != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = v
                wctx.N--
                if wctx.N == 0 { 
                    vm.popStack() 
                    vm.CurrentStruct = nil
                }
            }
        } else if vm.CurrentList != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = v
                wctx.N--
                if wctx.N == 0 { 
                    vm.popStack() 
                    vm.CurrentList = nil
                }
            }
        }
        vm.putReg(i.Xn, v)
        vm.PC++
        return true').

wam_go_case('SetValue', '        val := vm.getReg(i.Xn)
        vm.Heap = append(vm.Heap, val)
        if vm.CurrentStruct != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = val
                wctx.N--
                if wctx.N == 0 { 
                    vm.popStack() 
                    vm.CurrentStruct = nil
                }
            }
        } else if vm.CurrentList != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = val
                wctx.N--
                if wctx.N == 0 { 
                    vm.popStack() 
                    vm.CurrentList = nil
                }
            }
        }
        vm.PC++
        return true').

wam_go_case('SetConstant', '        vm.Heap = append(vm.Heap, i.C)
        if vm.CurrentStruct != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := vm.CurrentStruct.Arity - wctx.N
                vm.CurrentStruct.Args[idx] = i.C
                wctx.N--
                if wctx.N == 0 { 
                    vm.popStack() 
                    vm.CurrentStruct = nil
                }
            }
        } else if vm.CurrentList != nil {
            if wctx := vm.peekWriteCtx(); wctx != nil {
                idx := 2 - wctx.N
                vm.CurrentList.Elements[idx] = i.C
                wctx.N--
                if wctx.N == 0 { 
                    vm.popStack() 
                    vm.CurrentList = nil
                }
            }
        }
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

wam_go_case('CallForeign', '        return vm.executeForeignPredicate(i.Pred, i.Arity)').

wam_go_case('CallPc', '        vm.CP = vm.PC + 1
        vm.PC = i.TargetPC
        return true').

wam_go_case('Execute', '        if pc, ok := vm.Labels[i.Pred]; ok {
            vm.PC = pc
            return true
        }
        return false').

wam_go_case('ExecutePc', '        vm.PC = i.TargetPC
        return true').

wam_go_case('Proceed', '        if vm.CP > 0 {
            vm.PC = vm.CP
        } else {
            vm.Halted = true
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

wam_go_case('TryMeElsePc', '        vm.pushChoicePoint(i.NextPC)
        vm.PC++
        return true').

wam_go_case('RetryMeElse', '        if pc, ok := vm.Labels[i.Label]; ok {
            if len(vm.ChoicePoints) > 0 {
                vm.ChoicePoints[len(vm.ChoicePoints)-1].NextPC = pc
            }
        }
        vm.PC++
        return true').

wam_go_case('RetryMeElsePc', '        if len(vm.ChoicePoints) > 0 {
            vm.ChoicePoints[len(vm.ChoicePoints)-1].NextPC = i.NextPC
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

wam_go_case('SwitchOnConstantPc', '        if val, ok := vm.Regs["A1"]; ok && !isUnbound(val) {
            for _, c := range i.Cases {
                if valueEquals(c.Val, val) {
                    vm.PC = c.TargetPC
                    return true
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

wam_go_case('SwitchOnStructurePc', '        if val, ok := vm.Regs["A1"]; ok {
            if f, args := decompose(val); f != "" {
                key := fmt.Sprintf("%s/%d", f, len(args))
                for _, c := range i.Cases {
                    if c.Functor == key {
                        vm.PC = c.TargetPC
                        return true
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

wam_go_case('SwitchOnConstantA2Pc', '        if val, ok := vm.Regs["A2"]; ok && !isUnbound(val) {
            for _, c := range i.Cases {
                if valueEquals(c.Val, val) {
                    vm.PC = c.TargetPC
                    return true
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
        if vm.Halted {
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

// RunParallel executes the WAM search in parallel using goroutines.
func (vm *WamState) RunParallel(maxWorkers int) <-chan []Value {
	results := make(chan []Value, 100)
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	var explore func(state *WamState, hasToken bool)
	explore = func(state *WamState, hasToken bool) {
		defer wg.Done()
		if !hasToken {
			sem <- struct{}{}
		}
		defer func() { <-sem }()

		for {
			if state.Halted {
				results <- state.CollectResults()
				return
			}
			instr := state.fetch()
			if instr == nil {
				return
			}

			if !state.Step(instr) {
				if !state.backtrack() {
					return
				}
			}

			// Fork choice points if we have workers available
			if len(state.ChoicePoints) > 0 {
				select {
				case sem <- struct{}{}:
					// We acquired a token, pass it to the child
					if alt := state.ForkAtChoicePoint(); alt != nil {
						wg.Add(1)
						go explore(alt, true)
					} else {
						// Failed to fork
						<-sem
					}
				default:
					// No worker available, continue sequentially
				}
			}
		}
	}

	wg.Add(1)
	go explore(vm, false)

	go func() {
		wg.Wait()
		close(results)
	}()

	return results
}

// CollectResults gathers values from A registers.
func (vm *WamState) CollectResults() []Value {
	results := make([]Value, 0)
	for i := 1; ; i++ {
		name := fmt.Sprintf("A%d", i)
		val, ok := vm.Regs[name]
		if !ok {
			break
		}
		results = append(results, vm.deref(val))
	}
	return results
}

// fetch retrieves the instruction at the current PC.
func (vm *WamState) fetch() Instruction {
    if vm.PC >= 0 && vm.PC < len(vm.Code) {
        return vm.Code[vm.PC]
    }
    return nil
}

func resolveInstructions(code []Instruction, labels map[string]int) []Instruction {
    resolved := make([]Instruction, 0, len(code))
    for _, instr := range code {
        switch i := instr.(type) {
        case *Call:
            if pc, ok := labels[i.Pred]; ok {
                resolved = append(resolved, &CallPc{TargetPC: pc, Arity: i.Arity})
            } else {
                resolved = append(resolved, instr)
            }
        case *CallForeign:
            resolved = append(resolved, instr)
        case *Execute:
            if pc, ok := labels[i.Pred]; ok {
                resolved = append(resolved, &ExecutePc{TargetPC: pc})
            } else {
                resolved = append(resolved, instr)
            }
        case *TryMeElse:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &TryMeElsePc{NextPC: pc})
            } else {
                resolved = append(resolved, instr)
            }
        case *RetryMeElse:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &RetryMeElsePc{NextPC: pc})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnConstant:
            cases := make([]ConstPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := labels[c.Label]
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, ConstPcCase{Val: c.Val, TargetPC: pc})
            }
            if complete {
                resolved = append(resolved, &SwitchOnConstantPc{Cases: cases})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnStructure:
            cases := make([]StructPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := labels[c.Label]
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, StructPcCase{Functor: c.Functor, TargetPC: pc})
            }
            if complete {
                resolved = append(resolved, &SwitchOnStructurePc{Cases: cases})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnConstantA2:
            cases := make([]ConstPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := labels[c.Label]
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, ConstPcCase{Val: c.Val, TargetPC: pc})
            }
            if complete {
                resolved = append(resolved, &SwitchOnConstantA2Pc{Cases: cases})
            } else {
                resolved = append(resolved, instr)
            }
        default:
            resolved = append(resolved, instr)
        }
    }
    return resolved
}

func (vm *WamState) registerForeignNativeKind(predKey string, kind string) {
    vm.ForeignNativeKinds[predKey] = kind
}

func (vm *WamState) registerForeignResultLayout(predKey string, layout string) {
    vm.ForeignResultLayouts[predKey] = layout
}

func (vm *WamState) registerForeignResultMode(predKey string, mode string) {
    vm.ForeignResultModes[predKey] = mode
}

func (vm *WamState) registerForeignStringConfig(predKey string, key string, value string) {
    cfg, ok := vm.ForeignStringConfigs[predKey]
    if !ok {
        cfg = make(map[string]string)
        vm.ForeignStringConfigs[predKey] = cfg
    }
    cfg[key] = value
}

func (vm *WamState) registerForeignUsizeConfig(predKey string, key string, value int) {
    cfg, ok := vm.ForeignUsizeConfigs[predKey]
    if !ok {
        cfg = make(map[string]int)
        vm.ForeignUsizeConfigs[predKey] = cfg
    }
    cfg[key] = value
}

func (vm *WamState) registerIndexedAtomFact2Pairs(predKey string, pairs []AtomPair) {
    vm.IndexedAtomFactPairs[predKey] = pairs
}

func (vm *WamState) registerIndexedWeightedEdgeTriples(predKey string, triples []WeightedEdgeTriple) {
    vm.IndexedWeightedEdgeTriples[predKey] = triples
}

func (vm *WamState) foreignResultLayout(predKey string) string {
    return vm.ForeignResultLayouts[predKey]
}

func (vm *WamState) foreignResultMode(predKey string) string {
    return vm.ForeignResultModes[predKey]
}

func (vm *WamState) foreignStringConfig(predKey string, key string) string {
    cfg, ok := vm.ForeignStringConfigs[predKey]
    if !ok {
        return ""
    }
    return cfg[key]
}

func (vm *WamState) foreignUsizeConfig(predKey string, key string) int {
    cfg, ok := vm.ForeignUsizeConfigs[predKey]
    if !ok {
        return 0
    }
    return cfg[key]
}

func parseForeignTupleLayout(layout string) int {
    var arity int
    if _, err := fmt.Sscanf(layout, "tuple:%d", &arity); err == nil {
        return arity
    }
    return 0
}

func (vm *WamState) applyForeignResult(predKey string, resultRegs []string, result Value) bool {
    tupleArity := parseForeignTupleLayout(vm.foreignResultLayout(predKey))
    if tupleArity <= 1 {
        if len(resultRegs) < 1 {
            return false
        }
        return vm.Unify(vm.getReg(resultRegs[0]), result)
    }
    tuple, ok := result.(*Compound)
    if !ok || tuple.Functor != "__tuple__" || len(tuple.Args) != tupleArity || len(resultRegs) < tupleArity {
        return false
    }
    for idx := 0; idx < tupleArity; idx++ {
        if !vm.Unify(vm.getReg(resultRegs[idx]), tuple.Args[idx]) {
            return false
        }
    }
    return true
}

func (vm *WamState) finishForeignResults(predKey string, resultRegs []string, results []Value) bool {
    if len(results) == 0 {
        return false
    }
    resumePC := vm.PC + 1
    mode := vm.foreignResultMode(predKey)
    switch mode {
    case "stream":
        baseRegs := copyMap(vm.Regs)
        baseStack := copyStack(vm.Stack)
        trailMark := len(vm.Trail)
        heapTop := len(vm.Heap)
        for idx, result := range results {
            vm.unwindTrail(trailMark)
            vm.Regs = copyMap(baseRegs)
            vm.Stack = copyStack(baseStack)
            if heapTop >= 0 && heapTop <= len(vm.Heap) {
                vm.Heap = vm.Heap[:heapTop]
            }
            vm.CP = vm.CP
            vm.Halted = false
            vm.CurrentStruct = nil
            vm.CurrentList = nil
            if !vm.applyForeignResult(predKey, resultRegs, result) {
                continue
            }
            if idx+1 < len(results) {
                remaining := append([]Value(nil), results[idx+1:]...)
                vm.ChoicePoints = append(vm.ChoicePoints, ChoicePoint{
                    NextPC: resumePC,
                    ResumePC: resumePC,
                    CP: vm.CP,
                    Regs: baseRegs,
                    Stack: baseStack,
                    HeapTop: heapTop,
                    TrailMark: trailMark,
                    ForeignPredKey: predKey,
                    ForeignResultRegs: append([]string(nil), resultRegs...),
                    ForeignResults: remaining,
                })
            }
            vm.PC = resumePC
            return true
        }
        return false
    default:
        if !vm.applyForeignResult(predKey, resultRegs, results[0]) {
            return false
        }
        vm.PC = resumePC
        return true
    }
}

func valueAsAtomString(vm *WamState, v Value) (string, bool) {
    val := vm.deref(v)
    atom, ok := val.(*Atom)
    if !ok {
        return "", false
    }
    return atom.Name, true
}

func valueAsInteger(vm *WamState, v Value) (int64, bool) {
    val := vm.deref(v)
    integer, ok := val.(*Integer)
    if !ok {
        return 0, false
    }
    return integer.Val, true
}

func valueAsFloat(vm *WamState, v Value) (float64, bool) {
    val := vm.deref(v)
    switch n := val.(type) {
    case *Integer:
        return float64(n.Val), true
    case *Float:
        return n.Val, true
    default:
        return 0, false
    }
}

func listAsSlice(vm *WamState, v Value) ([]Value, bool) {
    val := vm.deref(v)
    list, ok := val.(*List)
    if !ok {
        return nil, false
    }
    return list.Elements, true
}

func (vm *WamState) collectNativeListSuffixes(items []Value, out *[]Value) {
    for idx := 0; idx <= len(items); idx++ {
        suffix := append([]Value(nil), items[idx:]...)
        *out = append(*out, &List{Elements: suffix})
    }
}

func tupleValue(items ...Value) Value {
    return &Compound{Functor: "__tuple__", Args: items}
}

func atomAdjacency(pairs []AtomPair) map[string][]string {
    adjacency := make(map[string][]string)
    for _, pair := range pairs {
        adjacency[pair.Left] = append(adjacency[pair.Left], pair.Right)
    }
    return adjacency
}

func weightedAdjacency(triples []WeightedEdgeTriple) map[string][]WeightedEdgeTriple {
    adjacency := make(map[string][]WeightedEdgeTriple)
    for _, triple := range triples {
        adjacency[triple.Left] = append(adjacency[triple.Left], triple)
    }
    return adjacency
}

func (vm *WamState) collectNativeTransitiveClosureResults(source string, pairs []AtomPair) []Value {
    adjacency := atomAdjacency(pairs)
    visited := make(map[string]bool)
    queue := append([]string(nil), adjacency[source]...)
    results := make([]Value, 0)
    for len(queue) > 0 {
        node := queue[0]
        queue = queue[1:]
        if visited[node] {
            continue
        }
        visited[node] = true
        results = append(results, &Atom{Name: node})
        queue = append(queue, adjacency[node]...)
    }
    return results
}

func (vm *WamState) collectNativeTransitiveDistanceResults(source string, pairs []AtomPair) []Value {
    adjacency := atomAdjacency(pairs)
    visited := map[string]bool{source: true}
    dist := map[string]int{source: 0}
    queue := []string{source}
    results := make([]Value, 0)
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        for _, next := range adjacency[current] {
            if visited[next] {
                continue
            }
            visited[next] = true
            dist[next] = dist[current] + 1
            queue = append(queue, next)
            results = append(results, tupleValue(
                &Atom{Name: next},
                &Integer{Val: int64(dist[next])},
            ))
        }
    }
    return results
}

func (vm *WamState) collectNativeTransitiveParentDistanceResults(source string, pairs []AtomPair) []Value {
    adjacency := atomAdjacency(pairs)
    visited := map[string]bool{source: true}
    dist := map[string]int{source: 0}
    parent := make(map[string]string)
    queue := []string{source}
    results := make([]Value, 0)
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        for _, next := range adjacency[current] {
            if visited[next] {
                continue
            }
            visited[next] = true
            dist[next] = dist[current] + 1
            parent[next] = current
            queue = append(queue, next)
            results = append(results, tupleValue(
                &Atom{Name: next},
                &Atom{Name: parent[next]},
                &Integer{Val: int64(dist[next])},
            ))
        }
    }
    return results
}

func (vm *WamState) collectNativeTransitiveStepParentDistanceResults(source string, pairs []AtomPair) []Value {
    adjacency := atomAdjacency(pairs)
    visited := map[string]bool{source: true}
    dist := map[string]int{source: 0}
    parent := make(map[string]string)
    firstStep := make(map[string]string)
    queue := []string{source}
    results := make([]Value, 0)
    for len(queue) > 0 {
        current := queue[0]
        queue = queue[1:]
        for _, next := range adjacency[current] {
            if visited[next] {
                continue
            }
            visited[next] = true
            dist[next] = dist[current] + 1
            parent[next] = current
            if current == source {
                firstStep[next] = next
            } else {
                firstStep[next] = firstStep[current]
            }
            queue = append(queue, next)
            results = append(results, tupleValue(
                &Atom{Name: next},
                &Atom{Name: firstStep[next]},
                &Atom{Name: parent[next]},
                &Integer{Val: int64(dist[next])},
            ))
        }
    }
    return results
}

func pickShortestCandidate(dist map[string]float64, settled map[string]bool) (string, bool) {
    bestNode := ""
    bestDist := 0.0
    found := false
    for node, d := range dist {
        if settled[node] {
            continue
        }
        if !found || d < bestDist || (d == bestDist && node < bestNode) {
            bestNode = node
            bestDist = d
            found = true
        }
    }
    return bestNode, found
}

func heuristicLookup(triples []WeightedEdgeTriple, from string, target string) float64 {
    for _, triple := range triples {
        if triple.Left == from && triple.Right == target {
            return triple.Weight
        }
    }
    return 0
}

func (vm *WamState) collectNativeWeightedShortestPathResults(source string, triples []WeightedEdgeTriple) []Value {
    adjacency := weightedAdjacency(triples)
    dist := map[string]float64{source: 0}
    settled := make(map[string]bool)
    results := make([]Value, 0)
    for {
        current, ok := pickShortestCandidate(dist, settled)
        if !ok {
            break
        }
        settled[current] = true
        if current != source {
            results = append(results, tupleValue(
                &Atom{Name: current},
                &Float{Val: dist[current]},
            ))
        }
        for _, edge := range adjacency[current] {
            candidate := dist[current] + edge.Weight
            prev, exists := dist[edge.Right]
            if !exists || candidate < prev {
                dist[edge.Right] = candidate
            }
        }
    }
    return results
}

func (vm *WamState) collectNativeAstarShortestPathResult(source string, target string, weighted []WeightedEdgeTriple, direct []WeightedEdgeTriple) []Value {
    adjacency := weightedAdjacency(weighted)
    gScore := map[string]float64{source: 0}
    open := map[string]bool{source: true}
    closed := make(map[string]bool)
    for len(open) > 0 {
        current := ""
        bestScore := 0.0
        found := false
        for node := range open {
            score := gScore[node] + heuristicLookup(direct, node, target)
            if !found || score < bestScore || (score == bestScore && node < current) {
                current = node
                bestScore = score
                found = true
            }
        }
        if !found {
            break
        }
        delete(open, current)
        if current == target {
            return []Value{&Float{Val: gScore[current]}}
        }
        closed[current] = true
        for _, edge := range adjacency[current] {
            if closed[edge.Right] {
                continue
            }
            candidate := gScore[current] + edge.Weight
            prev, exists := gScore[edge.Right]
            if !exists || candidate < prev {
                gScore[edge.Right] = candidate
                open[edge.Right] = true
            }
        }
    }
    return nil
}

func (vm *WamState) executeForeignPredicate(pred string, arity int) bool {
    predKey := fmt.Sprintf("%s/%d", pred, arity)
    nativeKind, ok := vm.ForeignNativeKinds[predKey]
    if !ok {
        return false
    }
    switch nativeKind {
    case "countdown_sum2":
        n, ok := valueAsInteger(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        sum := n * (n + 1) / 2
        return vm.finishForeignResults(predKey, []string{"A2"}, []Value{&Integer{Val: sum}})
    case "list_suffix2":
        items, ok := listAsSlice(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        suffixes := make([]Value, 0, len(items)+1)
        vm.collectNativeListSuffixes(items, &suffixes)
        packed := make([]Value, 0, len(suffixes))
        for _, suffix := range suffixes {
            packed = append(packed, suffix)
        }
        return vm.finishForeignResults(predKey, []string{"A2"}, packed)
    case "list_suffixes2":
        items, ok := listAsSlice(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        suffixes := make([]Value, 0, len(items)+1)
        vm.collectNativeListSuffixes(items, &suffixes)
        return vm.finishForeignResults(predKey, []string{"A2"}, []Value{&List{Elements: suffixes}})
    case "transitive_closure2":
        source, ok := valueAsAtomString(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveClosureResults(source, pairs)
        return vm.finishForeignResults(predKey, []string{"A2"}, results)
    case "transitive_distance3":
        source, ok := valueAsAtomString(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []string{"A2", "A3"}, results)
    case "transitive_parent_distance4":
        source, ok := valueAsAtomString(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveParentDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []string{"A2", "A3", "A4"}, results)
    case "transitive_step_parent_distance5":
        source, ok := valueAsAtomString(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveStepParentDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []string{"A2", "A3", "A4", "A5"}, results)
    case "weighted_shortest_path3":
        source, ok := valueAsAtomString(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        weightPred := vm.foreignStringConfig(predKey, "weight_pred")
        triples := vm.IndexedWeightedEdgeTriples[weightPred]
        results := vm.collectNativeWeightedShortestPathResults(source, triples)
        return vm.finishForeignResults(predKey, []string{"A2", "A3"}, results)
    case "astar_shortest_path4":
        source, ok := valueAsAtomString(vm, vm.getReg("A1"))
        if !ok {
            return false
        }
        target, ok := valueAsAtomString(vm, vm.getReg("A2"))
        if !ok {
            return false
        }
        if _, ok := valueAsFloat(vm, vm.getReg("A3")); !ok {
            return false
        }
        weightPred := vm.foreignStringConfig(predKey, "weight_pred")
        directPred := vm.foreignStringConfig(predKey, "direct_dist_pred")
        weighted := vm.IndexedWeightedEdgeTriples[weightPred]
        direct := vm.IndexedWeightedEdgeTriples[directPred]
        results := vm.collectNativeAstarShortestPathResult(source, target, weighted, direct)
        return vm.finishForeignResults(predKey, []string{"A4"}, results)
    default:
        return false
    }
}
', []).

%% escape_go_string(+Atom, -Escaped)
%  Escapes backslashes for Go string literals.
escape_go_string(Atom, Escaped) :-
    atom_string(Atom, Str),
    split_string(Str, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", Escaped).
