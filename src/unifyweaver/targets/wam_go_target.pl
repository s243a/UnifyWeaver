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
    write_wam_go_project/3,            % +Predicates, +Options, +ProjectDir
    init_atom_intern_table_go/0,       % reinitialize atom intern table for lowered Go
    intern_atom_go/2,                  % +AtomStr, -GoVarName
    emit_atom_table_go/1,              % -GoCode (var (...) declaration block)
    resolve_dimension_n_go/2,          % +Options, -DimN
    escape_go_string/2                 % +Str, -EscStr (re-exported helper)
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(pairs), [pairs_values/2, map_list_to_pairs/3]).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/recursive_kernel_detection',
             [detect_recursive_kernel/4, kernel_metadata/4]).
:- use_module('../core/template_system').
:- use_module('../bindings/go_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../targets/go_target', [compile_predicate_to_go/3]).
:- use_module('../targets/wam_go_lowered_emitter',
             [wam_go_lowerable/3, lower_predicate_to_go/4, go_func_name/2]).

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

    % Compile predicates and generate lib.go (predicates).
    %
    % Reset the atom intern table first: from this point on, both the
    % shared WAM bytecode in lib.go and the lowered Go functions in
    % lowered.go route every atom literal through intern_atom_go/2,
    % so they share package-level *Atom variables. The shared table is
    % emitted into atoms.go further down, so both lib.go and lowered.go
    % can reference the same vars from anywhere in `package main`.
    init_atom_intern_table_go,
    compile_predicates_for_project(Predicates, Options, PredicatesCode),
    format(atom(LibContent),
'package ~w

~w
', [PackageName, PredicatesCode]),
    directory_file_path(ProjectDir, 'lib.go', LibPath),
    write_file(LibPath, LibContent),

    % Generate lowered.go — deterministic predicate-as-function lowering.
    % Continues using the same intern table, so lowered.go's atom
    % references and lib.go's atom references coalesce.
    compile_lowered_predicates(Predicates, Options, LoweredCode),
    emit_atom_table_go(AtomTableCode),

    % Always write atoms.go. When there are interned atoms, lib.go references
    % their package-level vars. When there are none, runtime helpers still need
    % the internAtom function for dynamically sourced facts.
    (   AtomTableCode \== ""
    ->  format(atom(AtomsContent),
'package ~w

// Auto-generated atom intern table.
// Shared by lib.go (WAM bytecode literals) and lowered.go (lowered
// predicate functions). Pointer-identity equality on these vars is
// O(1); duplicate inline `&Atom{Name:"..."}` literals fall back to
// string compare in Atom.Equals.

~w
', [PackageName, AtomTableCode]),
        directory_file_path(ProjectDir, 'atoms.go', AtomsPath),
        write_file(AtomsPath, AtomsContent)
    ;   format(atom(AtomsContent),
'package ~w

// Runtime-only atom intern table. This project has no compile-time atom
// literals, but fact-source helpers may still construct atoms dynamically.
var atomInternMap = make(map[string]*Atom)

func internAtom(name string) *Atom {
    if a, ok := atomInternMap[name]; ok {
        return a
    }
    a := &Atom{Name: name}
    atomInternMap[name] = a
    return a
}
', [PackageName]),
        directory_file_path(ProjectDir, 'atoms.go', AtomsPath),
        write_file(AtomsPath, AtomsContent)
    ),

    (   LoweredCode \== ""
    ->  format(atom(LoweredContent),
'package ~w

import "fmt"

var _ = fmt.Sprintf

// Lowered predicates: direct Go methods on *WamState
// Generated by wam_go_lowered_emitter.pl
// (Atom intern table lives in atoms.go.)

~w
', [PackageName, LoweredCode]),
        directory_file_path(ProjectDir, 'lowered.go', LoweredPath),
        write_file(LoweredPath, LoweredContent)
    ;   true
    ),

    % Generate main.go with RunParallel when parallel(true)
    (   option(parallel(true), Options)
    ->  format(atom(MainContent),
'package main

import (
	"fmt"
	"~w"
)

func main() {
	ctx := ~w.NewWamContext(~w.SharedWamCode, ~w.SharedWamLabels)
	seeds := [][]~w.Value{
		{&~w.Atom{Name: "query"}},
	}
	results := ~w.RunParallel(ctx, seeds, 0)
	for i, res := range results {
		if res != nil {
			fmt.Printf("Seed %%d: %%v\\n", i, res)
		}
	}
}
', [ModuleName, PackageName, PackageName, PackageName, PackageName, PackageName, PackageName]),
        directory_file_path(ProjectDir, 'main.go', MainPath),
        write_file(MainPath, MainContent)
    ;   true
    ),

    format('WAM Go project created at: ~w~n', [ProjectDir]).

%% compile_lowered_predicates(+Predicates, +Options, -Code)
%  Attempts lowered emission for each predicate. Lowerable deterministic
%  predicates are emitted as direct Go methods on *WamState.
%
%  The inner once/1 is load-bearing: wam_target:compile_predicate_to_wam/3
%  is benignly non-deterministic for multi-clause fact predicates (it
%  produced 32 essentially-equivalent WAM bodies for category_parent/2
%  on the scale-300 fixture, differing only by ~5 chars of label name).
%  Without once/1, findall collects every alternative solution and the
%  emitter writes one duplicate `func (vm *WamState) Pred<Name>() bool`
%  per alternative — Go then refuses to compile the file with
%  "method already declared". Pin to the first solution per predicate.
compile_lowered_predicates([], _, "").
compile_lowered_predicates(Predicates, Options, Code) :-
    findall(GoCode,
        ( member(PredIndicator, Predicates),
          predicate_indicator_parts(PredIndicator, Module, Pred, Arity),
          \+ is_ffi_owned_fact(Module, Pred, Arity, Options),
          catch(
              once(( wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode),
                     wam_go_lowerable(Pred/Arity, WamCode, _Reason),
                     lower_predicate_to_go(Pred/Arity, WamCode, Options, GoLines),
                     atomic_list_concat(GoLines, '\n', GoCode)
                   )),
              _, fail)
        ),
        LoweredCodes),
    (   LoweredCodes == []
    ->  Code = ""
    ;   atomic_list_concat(LoweredCodes, '\n\n', Code)
    ).

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

// Exported aliases for main.go / parallel runner
var SharedWamCode = sharedWamCode
var SharedWamLabels = sharedWamLabels
', [SharedForeignSetup, AllInstrs, AllLabels])
    ;   SharedCode = ""
    ),
    generate_predicate_codes(Classified, WamEntries, PredCodes),
    atomic_list_concat(PredCodes, '\n\n', PredicatesCode),
    (   SharedCode == ""
    ->  Code = PredicatesCode
    ;   format(atom(Code), '~w~n~w', [SharedCode, PredicatesCode])
    ).

%% is_ffi_owned_fact(+Module, +Pred, +Arity, +Options)
%  True if this predicate has a go_foreign_spec AND all its clauses are
%  pure facts (head-only, body = true).  Such predicates are handled
%  entirely by the FFI kernel path — compiling WAM code for them is
%  wasted work (this was the -70% total win in Haskell).
is_ffi_owned_fact(Module, Pred, Arity, Options) :-
    go_foreign_spec(Module:Pred/Arity, Options, _SetupOps, _RewriteCalls, _Entry/_EntryArity),
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(Module:Head, Body), Clauses),
    Clauses \= [],
    forall(member(_-Body, Clauses), Body == true).

classify_predicates([], _, []).
classify_predicates([PredIndicator|Rest], Options, [Entry|RestEntries]) :-
    predicate_indicator_parts(PredIndicator, Module, Pred, Arity),
    (   is_ffi_owned_fact(Module, Pred, Arity, Options)
    ->  format(user_error, '  ~w/~w: FFI-owned fact (skipping WAM compilation)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, ffi_fact, '')
    ;   option(prefer_wam(true), Options),
        option(wam_fallback(WamFB0), Options, true),
        WamFB0 \== false,
        go_foreign_spec(Module:Pred/Arity, Options, _SetupOps0, _RewriteCalls0, _EntryPred0/_EntryArity0),
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode0)
    ->  compile_wam_predicate_to_go(Module:Pred/Arity, WamCode0, Options, PredCode0),
        format(user_error, '  ~w/~w: WAM fallback (foreign, preferred)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, wam_foreign, PredCode0)
    ;   option(prefer_wam(true), Options),
        option(wam_fallback(WamFB1), Options, true),
        WamFB1 \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode1)
    ->  format(user_error, '  ~w/~w: WAM fallback (preferred)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, wam, WamCode1)
    ;   go_foreign_spec(Module:Pred/Arity, Options, _SetupOps, _RewriteCalls, _EntryPred/_EntryArity),
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
generate_predicate_codes([classified(_, Pred, Arity, ffi_fact, _)|Rest], WamEntries,
                         [Code|RestCodes]) :-
    format(atom(Code), '// ~w/~w: FFI-owned fact — WAM compilation skipped', [Pred, Arity]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
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
    go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&GetConstant{C: ~w, Ai: ~w}', [GoVal, AiIdx]).
wam_instruction_to_go_literal(get_variable(Xn, Ai), GoLiteral) :-
    go_reg_index(Xn, XnIdx), go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&GetVariable{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_instruction_to_go_literal(get_value(Xn, Ai), GoLiteral) :-
    go_reg_index(Xn, XnIdx), go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&GetValue{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_instruction_to_go_literal(get_structure(F, Ai), GoLiteral) :-
    go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&GetStructure{Functor: "~w", Ai: ~w}', [F, AiIdx]).
wam_instruction_to_go_literal(get_list(Ai), GoLiteral) :-
    go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&GetList{Ai: ~w}', [AiIdx]).
wam_instruction_to_go_literal(unify_variable(Xn), GoLiteral) :-
    go_reg_index(Xn, XnIdx),
    format(atom(GoLiteral), '&UnifyVariable{Xn: ~w}', [XnIdx]).
wam_instruction_to_go_literal(unify_value(Xn), GoLiteral) :-
    go_reg_index(Xn, XnIdx),
    format(atom(GoLiteral), '&UnifyValue{Xn: ~w}', [XnIdx]).
wam_instruction_to_go_literal(unify_constant(C), GoLiteral) :-
    go_value_literal(C, GoVal),
    format(atom(GoLiteral), '&UnifyConstant{C: ~w}', [GoVal]).

wam_instruction_to_go_literal(put_constant(C, Ai), GoLiteral) :-
    go_value_literal(C, GoVal),
    go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&PutConstant{C: ~w, Ai: ~w}', [GoVal, AiIdx]).
wam_instruction_to_go_literal(put_variable(Xn, Ai), GoLiteral) :-
    go_reg_index(Xn, XnIdx), go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&PutVariable{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_instruction_to_go_literal(put_value(Xn, Ai), GoLiteral) :-
    go_reg_index(Xn, XnIdx), go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&PutValue{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_instruction_to_go_literal(put_structure(F, Ai), GoLiteral) :-
    go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&PutStructure{Functor: "~w", Ai: ~w}', [F, AiIdx]).
wam_instruction_to_go_literal(put_list(Ai), GoLiteral) :-
    go_reg_index(Ai, AiIdx),
    format(atom(GoLiteral), '&PutList{Ai: ~w}', [AiIdx]).
wam_instruction_to_go_literal(set_variable(Xn), GoLiteral) :-
    go_reg_index(Xn, XnIdx),
    format(atom(GoLiteral), '&SetVariable{Xn: ~w}', [XnIdx]).
wam_instruction_to_go_literal(set_value(Xn), GoLiteral) :-
    go_reg_index(Xn, XnIdx),
    format(atom(GoLiteral), '&SetValue{Xn: ~w}', [XnIdx]).
wam_instruction_to_go_literal(set_constant(C), GoLiteral) :-
    go_value_literal(C, GoVal),
    format(atom(GoLiteral), '&SetConstant{C: ~w}', [GoVal]).

wam_instruction_to_go_literal(allocate, '&Allocate{}').
wam_instruction_to_go_literal(deallocate, '&Deallocate{}').
wam_instruction_to_go_literal(call(P, N), GoLiteral) :-
    format(atom(GoLiteral), '&Call{Pred: "~w", Arity: ~w}', [P, N]).
wam_instruction_to_go_literal(call_indexed_atom_fact2(Pred), GoLiteral) :-
    escape_go_string(Pred, EscapedPred),
    format(atom(GoLiteral), '&CallIndexedAtomFact2{Pred: "~w"}', [EscapedPred]).
wam_instruction_to_go_literal(execute(P), GoLiteral) :-
    format(atom(GoLiteral), '&Execute{Pred: "~w"}', [P]).
wam_instruction_to_go_literal(proceed, '&Proceed{}').
wam_instruction_to_go_literal(builtin_call(Op, N), GoLiteral) :-
    escape_go_string(Op, EscapedOp),
    format(atom(GoLiteral), '&BuiltinCall{Op: "~w", Arity: ~w}', [EscapedOp, N]).

wam_instruction_to_go_literal(try_me_else(Label, Arity), GoLiteral) :-
    format(atom(GoLiteral), '&TryMeElse{Label: "~w", Arity: ~w}', [Label, Arity]).
wam_instruction_to_go_literal(try_me_else(Label), GoLiteral) :-
    format(atom(GoLiteral), '&TryMeElse{Label: "~w", Arity: 100}', [Label]).
wam_instruction_to_go_literal(retry_me_else(Label, Arity), GoLiteral) :-
    format(atom(GoLiteral), '&RetryMeElse{Label: "~w", Arity: ~w}', [Label, Arity]).
wam_instruction_to_go_literal(retry_me_else(Label), GoLiteral) :-
    format(atom(GoLiteral), '&RetryMeElse{Label: "~w", Arity: 100}', [Label]).
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
    predicate_go_name(Pred, CapPred),
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
    predicate_go_name(Pred, CapPred),
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
    maplist([I, S]>>(Idx is I - 1, format(atom(S), '    vm.Regs[~w] = a~w', [Idx, I])), Indices, Lines),
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
go_foreign_setup_line(register_tsv_atom_fact2(Pred/Arity, Path), Line) :-
    escape_go_string(Path, EscapedPath),
    format(atom(Line), '    if err := vm.registerTsvAtomFact2("~w/~w", "~w"); err != nil { panic(err) }',
        [Pred, Arity, EscapedPath]).
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

predicate_go_name(Pred, Name) :-
    atom_codes(Pred, Codes),
    maplist(go_ident_code, Codes, SafeCodes0),
    (   SafeCodes0 = [First|Rest]
    ->  code_type(UpperFirst, to_upper(First)),
        SafeCodes = [UpperFirst|Rest]
    ;   SafeCodes = [0'P]
    ),
    atom_codes(Name, SafeCodes).

go_ident_code(Code, Safe) :-
    (   code_type(Code, alnum)
    ->  Safe = Code
    ;   Safe = 0'_
    ).

predicate_indicator_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
predicate_indicator_parts(Pred/Arity, user, Pred, Arity).

predicate_indicator_arity(_Module:_Pred/Arity, Arity) :- !.
predicate_indicator_arity(_Pred/Arity, Arity) :- !.
predicate_indicator_arity(_, 100).  % fallback

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
go_supported_shared_kernel(recursive_kernel(category_ancestor, _, _)).
go_recursive_kernel_with_facts(Module,
        recursive_kernel(KernelKind, PredIndicator, KernelConfig0),
        recursive_kernel(KernelKind, PredIndicator,
            [edge_pred(EdgePred/2), fact_pairs(FactPairs)])) :-
    member(KernelKind, [transitive_closure2, transitive_distance3,
        transitive_parent_distance4, transitive_step_parent_distance5]),
    member(edge_pred(EdgePred/2), KernelConfig0),
    go_binary_edge_fact_pairs(Module, EdgePred/2, FactPairs),
    FactPairs \= [].
% category_ancestor carries a max_depth bound in addition to the
% edge predicate; preserve it so the runtime can stop the recursion
% at the user-specified depth.
go_recursive_kernel_with_facts(Module,
        recursive_kernel(category_ancestor, PredIndicator, KernelConfig0),
        recursive_kernel(category_ancestor, PredIndicator,
            [max_depth(MaxDepth), edge_pred(EdgePred/2), fact_pairs(FactPairs)])) :-
    member(max_depth(MaxDepth), KernelConfig0),
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
% category_ancestor's depth-bounded edge walk needs both the edge-
% pred adjacency and the user-specified max_depth so the runtime can
% match the WAM-bytecode semantics exactly.
go_recursive_kernel_config_ops(PredIndicator,
        [max_depth(MaxDepth), edge_pred(EdgePred/2), fact_pairs(FactPairs)], [
        register_foreign_usize_config(PredIndicator, max_depth, MaxDepth),
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
%
% When the WAM compiler emits an atom that needs Prolog quoting (e.g.
% an apostrophe-bearing category like 'People\'s_Republic_of_China'),
% the WAM TEXT carries the quoted form. The split-on-whitespace parser
% above hands that whole token here including the surrounding ' and
% the backslash-escaped inner quote. If we just pass it to
% go_value_literal/2 unchanged, the resulting Go literal becomes
%     &Atom{Name: "'People\'s_Republic_of_China'"}
% which (a) keeps the spurious outer apostrophes inside the atom's Name
% and (b) trips Go with `unknown escape sequence \'`. Use term_to_atom/2
% to round-trip the token back to a Prolog atom — that strips the outer
% quotes and unescapes \' to '.
parse_string_to_go_val(Str, GoVal) :-
    (   number_string(N, Str)
    ->  go_value_literal(N, GoVal)
    ;   atom_string(QuotedTok, Str),
        catch(term_to_atom(ParsedTerm, QuotedTok), _, fail),
        atom(ParsedTerm)
    ->  go_value_literal(ParsedTerm, GoVal)
    ;   go_value_literal(Str, GoVal)
    ).

wam_line_to_go_literal(["call", P, N], PredIndicator, Options, GoLit) :-
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    (   go_foreign_rewrite_call(Options, PredIndicator, CP, Num, ForeignPred, ForeignArity)
    ->  format(atom(GoLit), '&CallForeign{Pred: "~w", Arity: ~w}', [ForeignPred, ForeignArity])
    ;   format(atom(GoLit), '&Call{Pred: "~w", Arity: ~w}', [CP, CN])
    ).
wam_line_to_go_literal(["call_indexed_atom_fact2", Pred], _PredIndicator, _Options, GoLit) :-
    clean_comma(Pred, CPred),
    wam_instruction_to_go_literal(call_indexed_atom_fact2(CPred), GoLit).
wam_line_to_go_literal(["execute", P], PredIndicator, Options, GoLit) :-
    clean_comma(P, CP),
    (   go_foreign_rewrite_execute(Options, PredIndicator, CP, ForeignPred, ForeignArity)
    ->  format(atom(GoLit), '&CallForeign{Pred: "~w", Arity: ~w}', [ForeignPred, ForeignArity])
    ;   format(atom(GoLit), '&Execute{Pred: "~w"}', [CP])
    ).
wam_line_to_go_literal(["jump", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&Jump{Label: "~w"}', [CL]).
wam_line_to_go_literal(["cut_ite"], '&CutIte{}').
wam_line_to_go_literal(["begin_aggregate", AggType, ValueReg, ResultReg], GoLit) :-
    clean_comma(AggType, CAggType),
    clean_comma(ValueReg, CValueReg),
    clean_comma(ResultReg, CResultReg),
    go_reg_index(CValueReg, ValueRegIdx),
    go_reg_index(CResultReg, ResultRegIdx),
    format(atom(GoLit), '&BeginAggregate{AggType: "~w", ValueReg: ~w, ResultReg: ~w}',
        [CAggType, ValueRegIdx, ResultRegIdx]).
wam_line_to_go_literal(["end_aggregate", ValueReg], GoLit) :-
    clean_comma(ValueReg, CValueReg),
    go_reg_index(CValueReg, ValueRegIdx),
    format(atom(GoLit), '&EndAggregate{ValueReg: ~w}', [ValueRegIdx]).

wam_line_to_go_literal(["get_constant", C, Ai], GoLit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    parse_string_to_go_val(CC, GoVal),
    go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&GetConstant{C: ~w, Ai: ~w}', [GoVal, AiIdx]).
wam_line_to_go_literal(["get_variable", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    go_reg_index(CXn, XnIdx), go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&GetVariable{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_line_to_go_literal(["get_value", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    go_reg_index(CXn, XnIdx), go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&GetValue{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_line_to_go_literal(["get_structure", FN, Ai], GoLit) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&GetStructure{Functor: "~w", Ai: ~w}', [CFN, AiIdx]).
wam_line_to_go_literal(["get_list", Ai], GoLit) :-
    clean_comma(Ai, CAi),
    go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&GetList{Ai: ~w}', [AiIdx]).
wam_line_to_go_literal(["unify_variable", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    go_reg_index(CXn, XnIdx),
    format(atom(GoLit), '&UnifyVariable{Xn: ~w}', [XnIdx]).
wam_line_to_go_literal(["unify_value", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    go_reg_index(CXn, XnIdx),
    format(atom(GoLit), '&UnifyValue{Xn: ~w}', [XnIdx]).
wam_line_to_go_literal(["unify_constant", C], GoLit) :-
    clean_comma(C, CC),
    parse_string_to_go_val(CC, GoVal),
    format(atom(GoLit), '&UnifyConstant{C: ~w}', [GoVal]).

wam_line_to_go_literal(["put_constant", C, Ai], GoLit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    parse_string_to_go_val(CC, GoVal),
    go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&PutConstant{C: ~w, Ai: ~w}', [GoVal, AiIdx]).
wam_line_to_go_literal(["put_variable", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    go_reg_index(CXn, XnIdx), go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&PutVariable{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_line_to_go_literal(["put_value", Xn, Ai], GoLit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    go_reg_index(CXn, XnIdx), go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&PutValue{Xn: ~w, Ai: ~w}', [XnIdx, AiIdx]).
wam_line_to_go_literal(["put_structure", FN, Ai], GoLit) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&PutStructure{Functor: "~w", Ai: ~w}', [CFN, AiIdx]).
wam_line_to_go_literal(["put_list", Ai], GoLit) :-
    clean_comma(Ai, CAi),
    go_reg_index(CAi, AiIdx),
    format(atom(GoLit), '&PutList{Ai: ~w}', [AiIdx]).
wam_line_to_go_literal(["set_variable", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    go_reg_index(CXn, XnIdx),
    format(atom(GoLit), '&SetVariable{Xn: ~w}', [XnIdx]).
wam_line_to_go_literal(["set_value", Xn], GoLit) :-
    clean_comma(Xn, CXn),
    go_reg_index(CXn, XnIdx),
    format(atom(GoLit), '&SetValue{Xn: ~w}', [XnIdx]).
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
    format(atom(GoLit), '&TryMeElse{Label: "~w", Arity: 100}', [CL]).
wam_line_to_go_literal(["retry_me_else", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&RetryMeElse{Label: "~w", Arity: 100}', [CL]).
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

wam_line_to_go_literal(["try_me_else", L], PredIndicator, _Options, GoLit) :-
    clean_comma(L, CL),
    predicate_indicator_arity(PredIndicator, Arity),
    format(atom(GoLit), '&TryMeElse{Label: "~w", Arity: ~w}', [CL, Arity]).
wam_line_to_go_literal(["retry_me_else", L], PredIndicator, _Options, GoLit) :-
    clean_comma(L, CL),
    predicate_indicator_arity(PredIndicator, Arity),
    format(atom(GoLit), '&RetryMeElse{Label: "~w", Arity: ~w}', [CL, Arity]).
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
        format(atom(Case), '{Val: internAtom("malformed"), Label: "~w"}', [Entry])
    ).

% --- Register index encoding ---
% A1→0, A2→1, ...; X1→100, X2→101, ...; Y1→200, Y2→201, ...
go_reg_index(a(N), Idx) :- !, Idx is N - 1.
go_reg_index(x(N), Idx) :- !, Idx is N + 99.
go_reg_index(y(N), Idx) :- !, Idx is N + 199.
go_reg_index(Atom, Idx) :-
    atom(Atom), !,
    atom_string(Atom, Str),
    go_reg_index_str(Str, Idx).
go_reg_index(Str, Idx) :-
    string(Str), !,
    go_reg_index_str(Str, Idx).

go_reg_index_str(Str, Idx) :-
    (   sub_string(Str, 0, 1, _, "A"),
        sub_string(Str, 1, _, 0, NumStr),
        number_string(N, NumStr)
    ->  Idx is N - 1
    ;   sub_string(Str, 0, 1, _, "X"),
        sub_string(Str, 1, _, 0, NumStr),
        number_string(N, NumStr)
    ->  Idx is N + 99
    ;   sub_string(Str, 0, 1, _, "Y"),
        sub_string(Str, 1, _, 0, NumStr),
        number_string(N, NumStr)
    ->  Idx is N + 199
    ;   Idx = 0
    ).

% --- Value literal helpers ---
%
% Atom values are routed through the intern_atom_go/2 table so every
% reference to a given atom name in the generated Go source resolves
% to a single shared *Atom variable. This is what makes
% Atom.Equals's pointer-identity short-circuit (in value.go.mustache)
% actually pay off in the WAM bytecode (lib.go) — pre-fix, every
% fact emitted its own `&Atom{Name:"..."}` literal, so
% SwitchOnConstant's `valueEquals(c.Val, val)` always fell through to
% string compare, and the bench paid string-compare cost for every
% ground unification. With interning, distinct bytecode references to
% the same atom share a pointer.

go_value_literal(atom(A), GoVal) :- !, go_atom_to_literal(A, GoVal).
go_value_literal(integer(I), GoVal) :- !, format(atom(GoVal), '&Integer{Val: ~w}', [I]).
go_value_literal(N, GoVal) :- integer(N), !, format(atom(GoVal), '&Integer{Val: ~w}', [N]).
go_value_literal(N, GoVal) :- float(N), !, format(atom(GoVal), '&Float{Val: ~w}', [N]).
go_value_literal(A, GoVal) :- atom(A), !, go_atom_to_literal(A, GoVal).
go_value_literal(T, GoVal) :- go_atom_to_literal(T, GoVal).

%% go_atom_to_literal(+A, -GoVal)
%  Returns a Go expression denoting the atom A. If the intern table is
%  initialized this is a reference to a shared `wamAtom_*` package
%  variable; otherwise it falls back to an inline `&Atom{Name:"..."}`
%  literal so callers that don't run through write_wam_go_project/3
%  (e.g. ad-hoc bytecode-to-Go conversion in tests) still produce
%  valid Go.
go_atom_to_literal(A, GoVal) :-
    (   catch(intern_atom_go(A, AtomVar), _, fail)
    ->  GoVal = AtomVar
    ;   escape_go_atom_for_double_quoted(A, EA),
        % Route through internAtom so the runtime intern table
        % is the single source of pointer identity for atoms.
        % `Atom.Equals` is now pointer-only (see value.go.mustache);
        % a raw `&Atom{Name: ...}` here would produce a fresh
        % pointer that compares unequal against the same-named
        % atom from atomInternMap.
        format(atom(GoVal), 'internAtom("~w")', [EA])
    ).

%% escape_go_atom_for_double_quoted(+In, -Out)
%  Escape `\` -> `\\` and `"` -> `\"` so the result can be embedded
%  unmodified inside a Go double-quoted string literal.
escape_go_atom_for_double_quoted(In, Out) :-
    atom_string(In, S0),
    split_string(S0, "\\", "", Parts0),
    atomic_list_concat(Parts0, "\\\\", S1),
    atom_string(S1, S2),
    split_string(S2, "\"", "", Parts1),
    atomic_list_concat(Parts1, "\\\"", Out).

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

wam_go_case('GetConstant', '        val := vm.Regs[i.Ai]
        if val == nil { return false }
        if isUnbound(val) {
            u := val.(*Unbound)
            vm.bindUnbound(u, i.C)
            vm.PC++
            return true
        }
        if valueEquals(val, i.C) {
            vm.PC++
            return true
        }
        return false').

wam_go_case('GetVariable', '        val := vm.Regs[i.Ai]
        if val == nil { return false }
        vm.trailBinding(i.Xn)
        vm.putReg(i.Xn, val)
        vm.PC++
        return true').

wam_go_case('GetValue', '        valA := vm.Regs[i.Ai]
        valX := vm.getReg(i.Xn)
        if valA == nil { return false }
        if vm.Unify(valA, valX) {
            vm.PC++
            return true
        }
        return false').

wam_go_case('GetStructure', '        val := vm.Regs[i.Ai]
        if val == nil { return false }
        if isUnbound(val) {
            addr := vm.heapPush(nil)
            arity := parseFunctorArity(i.Functor)
            s := &Structure{Functor: i.Functor, Arity: arity, Args: make([]Value, arity)}
            vm.Heap[addr] = s
            vm.CurrentStruct = s
            vm.CurrentList = nil
            vm.bindUnbound(val.(*Unbound), &Ref{Addr: addr})
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

wam_go_case('GetList', '        val := vm.Regs[i.Ai]
        if val == nil { return false }
        if isUnbound(val) {
            addr := vm.heapPush(nil)
            l := &List{Elements: make([]Value, 2)}
            vm.Heap[addr] = l
            vm.CurrentList = l
            vm.CurrentStruct = nil
            vm.bindUnbound(val.(*Unbound), &Ref{Addr: addr})
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
            addr := vm.HeapLen
            v := &Unbound{Name: fmt.Sprintf("_H%d", addr), Idx: vm.allocVarId()}
            vm.heapPush(v)
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
            vm.heapPush(val)
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
            vm.heapPush(i.C)
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

wam_go_case('PutVariable', '        // Allocate a globally-unique Idx for the new logical variable.
        // The earlier strategy reused i.Xn (the X-register slot index)
        // as the Idx, which collided across activations: an outer
        // activation''s X207 (Idx=207) and an inner activation''s X207
        // (also Idx=207) would share Bindings[207] and clobber each
        // other. The first attempt at a fix `delete(Bindings, i.Xn)`
        // worked for 2-hop recursion but broke at 3+ hops because the
        // delete could erase an outer activation''s still-live binding.
        // Allocating a fresh Idx via allocVarId/0 sidesteps the
        // collision entirely.
        v := &Unbound{Name: fmt.Sprintf("_R%d", i.Xn), Idx: vm.allocVarId()}
        vm.putReg(i.Xn, v)
        vm.Regs[i.Ai] = v
        vm.PC++
        return true').

wam_go_case('PutValue', '        val := vm.getReg(i.Xn)
        vm.Regs[i.Ai] = val
        vm.PC++
        return true').

wam_go_case('PutStructure', '        addr := vm.heapPush(nil)
        arity := parseFunctorArity(i.Functor)
        s := &Structure{Functor: i.Functor, Arity: arity, Args: make([]Value, arity)}
        vm.Heap[addr] = s
        vm.CurrentStruct = s
        vm.CurrentList = nil
        vm.Regs[i.Ai] = &Ref{Addr: addr}
        vm.Stack = append(vm.Stack, &WriteCtx{N: arity})
        vm.PC++
        return true').

wam_go_case('PutList', '        addr := vm.heapPush(nil)
        l := &List{Elements: make([]Value, 2)}
        vm.Heap[addr] = l
        vm.CurrentList = l
        vm.CurrentStruct = nil
        vm.Regs[i.Ai] = &Ref{Addr: addr}
        vm.Stack = append(vm.Stack, &WriteCtx{N: 2})
        vm.PC++
        return true').

wam_go_case('SetVariable', '        addr := vm.HeapLen
        v := &Unbound{Name: fmt.Sprintf("_H%d", addr), Idx: vm.allocVarId()}
        vm.heapPush(v)
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
        vm.heapPush(val)
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

wam_go_case('SetConstant', '        vm.heapPush(i.C)
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

wam_go_case('Allocate', '        // Env trimming: PrevE links the new frame back to the previously
        // active env frame. vm.E moves to the index of the just-pushed
        // frame so peekEnvFrame is O(1) and Deallocate can walk the
        // PrevE chain without scanning the stack.
        env := &EnvFrame{CP: vm.CP, B0: len(vm.ChoicePoints), PrevE: vm.E}
        // Snapshot the Y-reg range (200..299) into the env frame so a
        // nested predicate that uses the same slot numbers via
        // PutVariable doesn''t silently clobber the caller''s Y-regs.
        // Restored at Deallocate. Bindings on caller-passed Unbounds
        // still propagate via the global Bindings[Idx] table — deref
        // of the restored Y-reg follows that binding, so we don''t
        // lose genuine results, only spurious leftover state.
        copy(env.SavedYRegs[:], vm.Regs[200:300])
        vm.Stack = append(vm.Stack, env)
        vm.E = len(vm.Stack) - 1
        vm.PC++
        return true').

wam_go_case('Deallocate', '        if vm.E >= 0 && vm.E < len(vm.Stack) {
            if env, ok := vm.Stack[vm.E].(*EnvFrame); ok {
                vm.CP = env.CP
                copy(vm.Regs[200:300], env.SavedYRegs[:])
                prevE := env.PrevE
                // Physical-pop only if it''s safe: the frame must be at
                // the top of the stack AND no younger choicepoint
                // references it (env.B0 == current CP count means the
                // frame was Allocated AFTER the youngest live CP, so
                // nothing depends on it staying around). Otherwise the
                // frame stays on the stack — backtrack truncation will
                // sweep it away when the referencing CPs are
                // exhausted, and a future Allocate can push above it
                // because vm.E now points at prevE, not at the dead
                // frame''s slot.
                if env.B0 >= len(vm.ChoicePoints) && vm.E == len(vm.Stack)-1 {
                    vm.Stack = vm.Stack[:vm.E]
                }
                vm.E = prevE
            }
        }
        vm.PC++
        return true').

wam_go_case('Call', '        vm.CP = vm.PC + 1
        if pc, ok := vm.Ctx.Labels[i.Pred]; ok {
            vm.PC = pc
            return true
        }
        return false').

wam_go_case('CallForeign', '        return vm.executeForeignPredicate(i.Pred, i.Arity)').

wam_go_case('CallIndexedAtomFact2', '        return vm.executeIndexedAtomFact2(i.Pred)').

wam_go_case('CallPc', '        vm.CP = vm.PC + 1
        vm.PC = i.TargetPC
        return true').

wam_go_case('Execute', '        if pc, ok := vm.Ctx.Labels[i.Pred]; ok {
            vm.PC = pc
            return true
        }
        return false').

wam_go_case('ExecutePc', '        vm.PC = i.TargetPC
        return true').

wam_go_case('Jump', '        if pc, ok := vm.Ctx.Labels[i.Label]; ok {
            vm.PC = pc
            return true
        }
        return false').

wam_go_case('JumpPc', '        vm.PC = i.TargetPC
        return true').

wam_go_case('CutIte', '        if len(vm.ChoicePoints) > 0 {
            vm.ChoicePoints = vm.ChoicePoints[:len(vm.ChoicePoints)-1]
        }
        vm.PC++
        return true').

wam_go_case('BeginAggregate', '        return vm.executeAggregate(i.AggType, i.ValueReg, i.ResultReg)').

wam_go_case('EndAggregate', '        vm.PC++
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
        if pc, ok := vm.Ctx.Labels[i.Label]; ok {
            nextPC = pc
        }
        vm.pushChoicePoint(nextPC, i.Arity)
        vm.PC++
        return true').

wam_go_case('TryMeElsePc', '        vm.pushChoicePoint(i.NextPC, i.Arity)
        vm.PC++
        return true').

wam_go_case('RetryMeElse', '        if pc, ok := vm.Ctx.Labels[i.Label]; ok {
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

wam_go_case('SwitchOnConstant', '        if val := vm.Regs[0]; val != nil && !isUnbound(val) {
            targets := make([]int, 0)
            for _, c := range i.Cases {
                if !valueEquals(c.Val, val) {
                    continue
                }
                if c.Label == "default" {
                    targets = append(targets, vm.indexedClauseBodyStart(vm.PC+1))
                    continue
                }
                if pc, ok := vm.Ctx.Labels[c.Label]; ok {
                    targets = append(targets, vm.indexedClauseBodyStart(pc))
                }
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
            }
        }
        vm.PC++
        return true').

wam_go_case('SwitchOnConstantPc', '        if val := vm.Regs[0]; val != nil && !isUnbound(val) {
            n := len(i.Cases)
            idx := sort.Search(n, func(j int) bool {
                return compareValues(i.Cases[j].Val, val) >= 0
            })
            targets := make([]int, 0)
            for idx < n && valueEquals(i.Cases[idx].Val, val) {
                targets = append(targets, vm.indexedClauseBodyStart(i.Cases[idx].TargetPC))
                idx++
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
            }
        }
        vm.PC++
        return true').

wam_go_case('SwitchOnStructure', '        if val := vm.Regs[0]; val != nil {
            if f, args := decompose(val); f != "" {
                key := fmt.Sprintf("%s/%d", f, len(args))
                targets := make([]int, 0)
                for _, c := range i.Cases {
                    if c.Functor != key {
                        continue
                    }
                    if c.Label == "default" {
                        targets = append(targets, vm.indexedClauseBodyStart(vm.PC+1))
                        continue
                    }
                    if pc, ok := vm.Ctx.Labels[c.Label]; ok {
                        targets = append(targets, vm.indexedClauseBodyStart(pc))
                    }
                }
                if len(targets) > 0 {
                    return vm.enterIndexedAlternatives(targets)
                }
            }
        }
        vm.PC++
        return true').

wam_go_case('SwitchOnStructurePc', '        if val := vm.Regs[0]; val != nil {
            if f, args := decompose(val); f != "" {
                key := fmt.Sprintf("%s/%d", f, len(args))
                targets := make([]int, 0)
                for _, c := range i.Cases {
                    if c.Functor == key {
                        targets = append(targets, vm.indexedClauseBodyStart(c.TargetPC))
                    }
                }
                if len(targets) > 0 {
                    return vm.enterIndexedAlternatives(targets)
                }
            }
        }
        vm.PC++
        return true').

wam_go_case('SwitchOnConstantA2', '        if val := vm.Regs[1]; val != nil && !isUnbound(val) {
            targets := make([]int, 0)
            for _, c := range i.Cases {
                if !valueEquals(c.Val, val) {
                    continue
                }
                if c.Label == "default" {
                    targets = append(targets, vm.indexedClauseBodyStart(vm.PC+1))
                    continue
                }
                if pc, ok := vm.Ctx.Labels[c.Label]; ok {
                    targets = append(targets, vm.indexedClauseBodyStart(pc))
                }
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
            }
        }
        vm.PC++
        return true').

wam_go_case('SwitchOnConstantA2Pc', '        if val := vm.Regs[1]; val != nil && !isUnbound(val) {
            n := len(i.Cases)
            idx := sort.Search(n, func(j int) bool {
                return compareValues(i.Cases[j].Val, val) >= 0
            })
            targets := make([]int, 0)
            for idx < n && valueEquals(i.Cases[idx].Val, val) {
                targets = append(targets, vm.indexedClauseBodyStart(i.Cases[idx].TargetPC))
                idx++
            }
            if len(targets) > 0 {
                return vm.enterIndexedAlternatives(targets)
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

func (vm *WamState) indexedClauseBodyStart(targetPC int) int {
    if targetPC < 0 || targetPC >= len(vm.Ctx.Code) {
        return targetPC
    }
    switch vm.Ctx.Code[targetPC].(type) {
    case *TryMeElse, *TryMeElsePc, *RetryMeElse, *RetryMeElsePc, *TrustMe:
        return targetPC + 1
    default:
        return targetPC
    }
}

func (vm *WamState) enterIndexedAlternatives(targets []int) bool {
    if len(targets) == 0 {
        return false
    }
    if len(targets) > 1 {
        // count non-nil A-regs as arity approximation when not statically known
        arity := 0
        for i := 0; i < 32; i++ {
            if vm.Regs[i] != nil { arity = i + 1 }
        }
        if arity < 1 { arity = 1 }
        vm.pushIndexedChoicePoint(targets[1:], arity)
    }
    vm.PC = targets[0]
    return true
}

func (vm *WamState) enterIndexedClause(targetPC int) bool {
    return vm.enterIndexedAlternatives([]int{vm.indexedClauseBodyStart(targetPC)})
}

func (vm *WamState) runIsolatedGoal(targetPC int, args []Value) bool {
    sub := vm.Clone()
    for idx := range sub.Regs {
        sub.Regs[idx] = nil
    }
    for idx, arg := range args {
        if idx >= len(sub.Regs) {
            break
        }
        sub.Regs[idx] = arg
    }
    sub.PC = targetPC
    sub.CP = 0
    sub.E = -1
    sub.Stack = nil
    sub.Trail = nil
    sub.TrailLen = 0
    sub.ChoicePoints = nil
    sub.Halted = false
    sub.CurrentStruct = nil
    sub.CurrentList = nil
    return sub.Run()
}

// CollectResults gathers values from A registers (indices 0..N-1).
func (vm *WamState) CollectResults() []Value {
	results := make([]Value, 0)
	for i := 0; i < 100; i++ {
		val := vm.Regs[i]
		if val == nil {
			break
		}
		results = append(results, vm.deref(val))
	}
	return results
}

// fetch retrieves the instruction at the current PC.
func (vm *WamState) fetch() Instruction {
    if vm.PC >= 0 && vm.PC < len(vm.Ctx.Code) {
        return vm.Ctx.Code[vm.PC]
    }
    return nil
}

func resolveInstructions(code []Instruction, labels map[string]int) []Instruction {
    // resolveLabel handles the "default" sentinel that the constant-index
    // emitter (build_constant_index/5 in wam_target.pl) puts on the
    // first clause''s entry — at runtime that label means "fall through
    // to the next instruction" (vm.PC+1). resolveInstructions runs over
    // the code linearly, so the "next" PC is always idx+1 of the
    // current SwitchOnConstant. Without this resolution, every
    // SwitchOnConstant with a default entry stays unresolved (because
    // labels["default"] doesn''t exist), keeping the linear-scan
    // SwitchOnConstant runtime case alive — at scale-300 with
    // category_parent''s 6000-clause table the O(N) scan dominates
    // the per-call cost.
    resolveLabel := func(label string, idx int) (int, bool) {
        if label == "default" {
            return idx + 1, true
        }
        pc, ok := labels[label]
        return pc, ok
    }
    resolved := make([]Instruction, 0, len(code))
    for idx, instr := range code {
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
        case *Jump:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &JumpPc{TargetPC: pc})
            } else {
                resolved = append(resolved, instr)
            }
        case *TryMeElse:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &TryMeElsePc{NextPC: pc, Arity: i.Arity})
            } else {
                resolved = append(resolved, instr)
            }
        case *RetryMeElse:
            if pc, ok := labels[i.Label]; ok {
                resolved = append(resolved, &RetryMeElsePc{NextPC: pc, Arity: i.Arity})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnConstant:
            cases := make([]ConstPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := resolveLabel(c.Label, idx)
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, ConstPcCase{Val: c.Val, TargetPC: pc})
            }
            if complete {
                sort.Slice(cases, func(a, b int) bool {
                    return compareValues(cases[a].Val, cases[b].Val) < 0
                })
                resolved = append(resolved, &SwitchOnConstantPc{Cases: cases})
            } else {
                resolved = append(resolved, instr)
            }
        case *SwitchOnStructure:
            cases := make([]StructPcCase, 0, len(i.Cases))
            complete := true
            for _, c := range i.Cases {
                pc, ok := resolveLabel(c.Label, idx)
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
                pc, ok := resolveLabel(c.Label, idx)
                if !ok {
                    complete = false
                    break
                }
                cases = append(cases, ConstPcCase{Val: c.Val, TargetPC: pc})
            }
            if complete {
                sort.Slice(cases, func(a, b int) bool {
                    return compareValues(cases[a].Val, cases[b].Val) < 0
                })
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

func (vm *WamState) executeAggregate(aggType string, valueReg int, resultReg int) bool {
    endPC, ok := vm.findMatchingAggregateEnd(vm.PC)
    if !ok {
        return false
    }
    sub := vm.Clone()
    baseChoicePoints := len(sub.ChoicePoints)
    sub.PC = vm.PC + 1
    sub.Halted = false
    sub.CurrentStruct = nil
    sub.CurrentList = nil

    values := make([]Value, 0)
    count := 0
    for {
        if !sub.runUntilPC(endPC, baseChoicePoints) {
            break
        }
        count++
        if aggType != "count" {
            val := sub.Regs[valueReg]
            if val == nil {
                return false
            }
            values = append(values, sub.deref(val))
        }
        if !sub.backtrackAbove(baseChoicePoints) {
            break
        }
    }

    result, ok := aggregateResultValue(aggType, values, count)
    if !ok {
        return false
    }
    if !vm.bindOrUnifyReg(resultReg, result) {
        return false
    }
    vm.PC = endPC + 1
    return true
}

func (vm *WamState) findMatchingAggregateEnd(startPC int) (int, bool) {
    depth := 1
    for pc := startPC + 1; pc < len(vm.Ctx.Code); pc++ {
        switch vm.Ctx.Code[pc].(type) {
        case *BeginAggregate:
            depth++
        case *EndAggregate:
            depth--
            if depth == 0 {
                return pc, true
            }
        }
    }
    return 0, false
}

func (vm *WamState) runUntilPC(targetPC int, baseChoicePoints int) bool {
    for {
        if vm.Halted {
            return false
        }
        if vm.PC == targetPC {
            return true
        }
        instr := vm.fetch()
        if instr == nil {
            return false
        }
        if !vm.Step(instr) {
            if !vm.backtrackAbove(baseChoicePoints) {
                return false
            }
        }
    }
}

func (vm *WamState) backtrackAbove(limit int) bool {
    if len(vm.ChoicePoints) <= limit {
        return false
    }
    return vm.backtrack()
}

func (vm *WamState) bindOrUnifyReg(reg int, val Value) bool {
    existing := vm.Regs[reg]
    if existing == nil {
        vm.trailBinding(reg)
        vm.putReg(reg, val)
        return true
    }
    return vm.Unify(existing, val)
}

func aggregateResultValue(aggType string, values []Value, count int) (Value, bool) {
    switch aggType {
    case "count":
        return &Integer{Val: int64(count)}, true
    case "collect":
        return &List{Elements: append([]Value(nil), values...)}, true
    case "bag", "bagof":
        return &List{Elements: append([]Value(nil), values...)}, true
    case "set", "setof":
        return &List{Elements: uniqueAggregateValues(values)}, true
    case "sum":
        total := 0.0
        for _, value := range values {
            number, ok := aggregateNumericValue(value)
            if !ok {
                return nil, false
            }
            total += number
        }
        return &Float{Val: total}, true
    case "max":
        if len(values) == 0 {
            return nil, false
        }
        best, ok := aggregateNumericValue(values[0])
        if !ok {
            return nil, false
        }
        for _, value := range values[1:] {
            number, ok := aggregateNumericValue(value)
            if !ok {
                return nil, false
            }
            if number > best {
                best = number
            }
        }
        return &Float{Val: best}, true
    case "min":
        if len(values) == 0 {
            return nil, false
        }
        best, ok := aggregateNumericValue(values[0])
        if !ok {
            return nil, false
        }
        for _, value := range values[1:] {
            number, ok := aggregateNumericValue(value)
            if !ok {
                return nil, false
            }
            if number < best {
                best = number
            }
        }
        return &Float{Val: best}, true
    default:
        return nil, false
    }
}

func uniqueAggregateValues(values []Value) []Value {
    out := make([]Value, 0, len(values))
    for _, value := range values {
        found := false
        for _, existing := range out {
            if valueEquals(existing, value) {
                found = true
                break
            }
        }
        if !found {
            out = append(out, value)
        }
    }
    return out
}

func aggregateNumericValue(value Value) (float64, bool) {
    switch t := value.(type) {
    case *Integer:
        return float64(t.Val), true
    case *Float:
        return t.Val, true
    default:
        return 0, false
    }
}

func (vm *WamState) registerForeignNativeKind(predKey string, kind string) {
    vm.Ctx.ForeignNativeKinds[predKey] = kind
}

func (vm *WamState) registerForeignResultLayout(predKey string, layout string) {
    vm.Ctx.ForeignResultLayouts[predKey] = layout
}

func (vm *WamState) registerForeignResultMode(predKey string, mode string) {
    vm.Ctx.ForeignResultModes[predKey] = mode
}

func (vm *WamState) registerForeignStringConfig(predKey string, key string, value string) {
    cfg, ok := vm.Ctx.ForeignStringConfigs[predKey]
    if !ok {
        cfg = make(map[string]string)
        vm.Ctx.ForeignStringConfigs[predKey] = cfg
    }
    cfg[key] = value
}

func (vm *WamState) registerForeignUsizeConfig(predKey string, key string, value int) {
    cfg, ok := vm.Ctx.ForeignUsizeConfigs[predKey]
    if !ok {
        cfg = make(map[string]int)
        vm.Ctx.ForeignUsizeConfigs[predKey] = cfg
    }
    cfg[key] = value
}

func (vm *WamState) registerIndexedAtomFact2Pairs(predKey string, pairs []AtomPair) {
    vm.Ctx.IndexedAtomFactPairs[predKey] = pairs
    vm.registerAtomFact2Source(predKey, newStaticAtomFact2Source(pairs))
}

func (vm *WamState) registerAtomFact2Source(predKey string, source AtomFact2Source) {
    if source == nil {
        return
    }
    vm.Ctx.AtomFact2Sources[predKey] = source
    vm.Ctx.IndexedAtomFactPairs[predKey] = source.Scan()
}

func (vm *WamState) registerTsvAtomFact2(predKey string, path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }
    lines := strings.Split(string(data), "\\n")
    pairs := make([]AtomPair, 0, len(lines))
    for idx, line := range lines {
        line = strings.TrimSpace(line)
        if line == "" {
            continue
        }
        if idx == 0 {
            continue
        }
        cols := strings.Split(line, "\\t")
        if len(cols) < 2 {
            continue
        }
        left := strings.TrimSpace(cols[0])
        right := strings.TrimSpace(cols[1])
        if left == "" || right == "" {
            continue
        }
        pairs = append(pairs, AtomPair{Left: left, Right: right})
    }
    vm.registerAtomFact2Source(predKey, newStaticAtomFact2Source(pairs))
    return nil
}

func (vm *WamState) registerIndexedWeightedEdgeTriples(predKey string, triples []WeightedEdgeTriple) {
    vm.Ctx.IndexedWeightedEdgeTriples[predKey] = triples
}

func (vm *WamState) foreignResultLayout(predKey string) string {
    return vm.Ctx.ForeignResultLayouts[predKey]
}

func (vm *WamState) foreignResultMode(predKey string) string {
    return vm.Ctx.ForeignResultModes[predKey]
}

func (vm *WamState) foreignStringConfig(predKey string, key string) string {
    cfg, ok := vm.Ctx.ForeignStringConfigs[predKey]
    if !ok {
        return ""
    }
    return cfg[key]
}

func (vm *WamState) foreignUsizeConfig(predKey string, key string) int {
    cfg, ok := vm.Ctx.ForeignUsizeConfigs[predKey]
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

type staticAtomFact2Source struct {
    pairs []AtomPair
    byLeft map[string][]AtomPair
}

func newStaticAtomFact2Source(pairs []AtomPair) *staticAtomFact2Source {
    copied := append([]AtomPair(nil), pairs...)
    byLeft := make(map[string][]AtomPair)
    for _, pair := range copied {
        byLeft[pair.Left] = append(byLeft[pair.Left], pair)
    }
    return &staticAtomFact2Source{pairs: copied, byLeft: byLeft}
}

func (source *staticAtomFact2Source) Scan() []AtomPair {
    if source == nil {
        return nil
    }
    return append([]AtomPair(nil), source.pairs...)
}

func (source *staticAtomFact2Source) LookupArg1(left string) []AtomPair {
    if source == nil {
        return nil
    }
    return append([]AtomPair(nil), source.byLeft[left]...)
}

func (vm *WamState) applyForeignResult(predKey string, resultRegs []int, result Value) bool {
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

func (vm *WamState) finishForeignResults(predKey string, resultRegs []int, results []Value) bool {
    if len(results) == 0 {
        return false
    }
    resumePC := vm.PC + 1
    mode := vm.foreignResultMode(predKey)
    switch mode {
    case "stream":
        return vm.finishStreamResults(predKey, resultRegs, results)
    default:
        if !vm.applyForeignResult(predKey, resultRegs, results[0]) {
            return false
        }
        vm.PC = resumePC
        return true
    }
}

func (vm *WamState) finishStreamResults(predKey string, resultRegs []int, results []Value) bool {
    if len(results) == 0 {
        return false
    }
    resumePC := vm.PC + 1
    var baseRegs [320]Value
    baseRegs = vm.Regs
    // Capture stack length and the env-pointer instead of cloning
    // the stack — backtrack truncates to baseStackLen and restores
    // baseE the same way the regular CP path does.
    baseStackLen := len(vm.Stack)
    baseE := vm.E
    trailMark := vm.TrailLen
    heapTop := vm.HeapLen
    for idx, result := range results {
        vm.unwindTrailTo(trailMark)
        vm.Regs = baseRegs
        if baseStackLen <= len(vm.Stack) {
            vm.Stack = vm.Stack[:baseStackLen]
        }
        vm.E = baseE
        if heapTop >= 0 && heapTop <= vm.HeapLen {
            vm.heapTrimTo(heapTop)
        }
        vm.Halted = false
        vm.CurrentStruct = nil
        vm.CurrentList = nil
        if !vm.applyForeignResult(predKey, resultRegs, result) {
            continue
        }
        if idx+1 < len(results) {
            remaining := append([]Value(nil), results[idx+1:]...)
            ycount := vm.MaxYReg - 200
            if ycount < 0 {
                ycount = 0
            }
            savedRegs := make([]Value, 8+ycount)
            copy(savedRegs[:8], baseRegs[:8])
            if ycount > 0 {
                copy(savedRegs[8:], baseRegs[200:200+ycount])
            }
            vm.ChoicePoints = append(vm.ChoicePoints, ChoicePoint{
                NextPC: resumePC,
                ResumePC: resumePC,
                CP: vm.CP,
                E: baseE,
                StackLen: baseStackLen,
                SavedRegs: savedRegs,
                HeapTop: heapTop,
                TrailMark: trailMark,
                ForeignPredKey: predKey,
                ForeignResultRegs: append([]int(nil), resultRegs...),
                ForeignResults: remaining,
            })
        }
        vm.PC = resumePC
        return true
    }
    return false
}

func (vm *WamState) executeIndexedAtomFact2(predKey string) bool {
    key, ok := valueAsAtomString(vm, vm.getReg(0))
    if !ok {
        return false
    }
    pairs := vm.Ctx.IndexedAtomFactPairs[predKey]
    if source, ok := vm.Ctx.AtomFact2Sources[predKey]; ok {
        pairs = source.LookupArg1(key)
    }
    results := make([]Value, 0)
    for _, pair := range pairs {
        if pair.Left == key {
            results = append(results, internAtom(pair.Right))
        }
    }
    return vm.finishStreamResults(predKey, []int{1}, results)
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
        results = append(results, internAtom(node))
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
                internAtom(next),
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
                internAtom(next),
                internAtom(parent[next]),
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
                internAtom(next),
                internAtom(firstStep[next]),
                internAtom(parent[next]),
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
                internAtom(current),
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

// listAsAtomStrings walks a Prolog cons-list and pulls each element
// out as an atom string. Uses vm.listToSlice rather than listAsSlice
// because the latter returns the 2-element [head, tail] cons cell,
// not the full flattened list — `category_ancestor`''s visited list
// can be many cells deep, and reading just the first cons would
// silently truncate it (causing the kernel to walk paths through
// already-visited nodes and produce wrong hop counts).
func listAsAtomStrings(vm *WamState, v Value) ([]string, bool) {
    items, ok := vm.listToSlice(v)
    if !ok {
        return nil, false
    }
    out := make([]string, 0, len(items))
    for _, item := range items {
        s, ok := valueAsAtomString(vm, item)
        if !ok {
            return nil, false
        }
        out = append(out, s)
    }
    return out, true
}

// collectNativeCategoryAncestorHops walks the parent edges of `cat`
// looking for `root`, emitting one hop count per matching path. Mirrors
// the Rust implementation at wam_rust_target.pl:1797 — same DFS, same
// max-depth semantics, same visited-set skip. The adjacency map is
// built once at the top level and threaded into the recursive helper
// to avoid rebuilding it per call.
func (vm *WamState) collectNativeCategoryAncestorHops(cat string, root string, visited []string, maxDepth int, pairs []AtomPair) []int64 {
    adjacency := atomAdjacency(pairs)
    var out []int64
    vm.collectNativeCategoryAncestorHopsRec(cat, root, visited, maxDepth, adjacency, &out)
    return out
}

func (vm *WamState) collectNativeCategoryAncestorHopsRec(cat string, root string, visited []string, maxDepth int, adjacency map[string][]string, out *[]int64) {
    rootSeen := false
    for _, v := range visited {
        if v == root {
            rootSeen = true
            break
        }
    }
    parents := adjacency[cat]
    if !rootSeen {
        for _, p := range parents {
            if p == root {
                *out = append(*out, 1)
                break
            }
        }
    }
    if len(visited) >= maxDepth {
        return
    }
    for _, parent := range parents {
        skip := false
        for _, v := range visited {
            if v == parent {
                skip = true
                break
            }
        }
        if skip {
            continue
        }
        nextVisited := make([]string, 0, len(visited)+1)
        nextVisited = append(nextVisited, parent)
        nextVisited = append(nextVisited, visited...)
        before := len(*out)
        vm.collectNativeCategoryAncestorHopsRec(parent, root, nextVisited, maxDepth, adjacency, out)
        for i := before; i < len(*out); i++ {
            (*out)[i] += 1
        }
    }
}

func (vm *WamState) executeForeignPredicate(pred string, arity int) bool {
    predKey := fmt.Sprintf("%s/%d", pred, arity)
    nativeKind, ok := vm.Ctx.ForeignNativeKinds[predKey]
    if !ok {
        return false
    }
    switch nativeKind {
    case "countdown_sum2":
        n, ok := valueAsInteger(vm, vm.getReg(0))
        if !ok {
            return false
        }
        sum := n * (n + 1) / 2
        return vm.finishForeignResults(predKey, []int{1}, []Value{&Integer{Val: sum}})
    case "list_suffix2":
        items, ok := listAsSlice(vm, vm.getReg(0))
        if !ok {
            return false
        }
        suffixes := make([]Value, 0, len(items)+1)
        vm.collectNativeListSuffixes(items, &suffixes)
        packed := make([]Value, 0, len(suffixes))
        for _, suffix := range suffixes {
            packed = append(packed, suffix)
        }
        return vm.finishForeignResults(predKey, []int{1}, packed)
    case "list_suffixes2":
        items, ok := listAsSlice(vm, vm.getReg(0))
        if !ok {
            return false
        }
        suffixes := make([]Value, 0, len(items)+1)
        vm.collectNativeListSuffixes(items, &suffixes)
        return vm.finishForeignResults(predKey, []int{1}, []Value{&List{Elements: suffixes}})
    case "transitive_closure2":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveClosureResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1}, results)
    case "transitive_distance3":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1, 2}, results)
    case "transitive_parent_distance4":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveParentDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1, 2, 3}, results)
    case "transitive_step_parent_distance5":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        results := vm.collectNativeTransitiveStepParentDistanceResults(source, pairs)
        return vm.finishForeignResults(predKey, []int{1, 2, 3, 4}, results)
    case "category_ancestor":
        // category_ancestor(Cat, Root, Hops, Visited) — output is Hops
        // (A3, reg index 2). Cat=A1, Root=A2, Visited=A4. The WAM
        // semantics: walk parents of Cat up to max_depth hops; emit one
        // integer hop count per path that reaches Root, skipping any
        // node already in Visited. See
        // src/unifyweaver/core/recursive_kernel_detection.pl:135 for the
        // canonical register layout and call spec.
        cat, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        root, ok := valueAsAtomString(vm, vm.getReg(1))
        if !ok {
            return false
        }
        visited, ok := listAsAtomStrings(vm, vm.getReg(3))
        if !ok {
            return false
        }
        maxDepth := vm.foreignUsizeConfig(predKey, "max_depth")
        edgePred := vm.foreignStringConfig(predKey, "edge_pred")
        pairs := vm.Ctx.IndexedAtomFactPairs[edgePred]
        hops := vm.collectNativeCategoryAncestorHops(cat, root, visited, maxDepth, pairs)
        if len(hops) == 0 {
            return false
        }
        results := make([]Value, len(hops))
        for i, h := range hops {
            results[i] = &Integer{Val: h}
        }
        return vm.finishForeignResults(predKey, []int{2}, results)
    case "weighted_shortest_path3":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        weightPred := vm.foreignStringConfig(predKey, "weight_pred")
        triples := vm.Ctx.IndexedWeightedEdgeTriples[weightPred]
        results := vm.collectNativeWeightedShortestPathResults(source, triples)
        return vm.finishForeignResults(predKey, []int{1, 2}, results)
    case "astar_shortest_path4":
        source, ok := valueAsAtomString(vm, vm.getReg(0))
        if !ok {
            return false
        }
        target, ok := valueAsAtomString(vm, vm.getReg(1))
        if !ok {
            return false
        }
        if _, ok := valueAsFloat(vm, vm.getReg(2)); !ok {
            return false
        }
        weightPred := vm.foreignStringConfig(predKey, "weight_pred")
        directPred := vm.foreignStringConfig(predKey, "direct_dist_pred")
        weighted := vm.Ctx.IndexedWeightedEdgeTriples[weightPred]
        direct := vm.Ctx.IndexedWeightedEdgeTriples[directPred]
        results := vm.collectNativeAstarShortestPathResult(source, target, weighted, direct)
        return vm.finishForeignResults(predKey, []int{3}, results)
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

% ============================================================================
% Atom Interning (for lowered Go emission)
%
% Mirrors wam_haskell_target.pl's intern table. Lowered emission collects
% every atom literal, deduplicates them, and emits a single
% var (...) block of *Atom values referenced by name. This avoids
% per-call allocation of identical &Atom{Name: "..."} literals in tight
% lowered loops.
% ============================================================================

:- dynamic go_atom_intern_id/2.    % go_atom_intern_id(String, GoVarName)
:- dynamic go_atom_intern_next/1.  % go_atom_intern_next(NextSeq)

%% init_atom_intern_table_go is det.
%  Reset the Go atom intern table. Called at the start of each
%  write_wam_go_project/3 invocation so independent generations don't
%  share state.
init_atom_intern_table_go :-
    retractall(go_atom_intern_id(_, _)),
    retractall(go_atom_intern_next(_)),
    assertz(go_atom_intern_next(0)).

%% intern_atom_go(+AtomStr, -GoVarName) is det.
%  Returns the package-level Go variable name (an *Atom pointer) bound
%  to AtomStr. If the atom hasn't been seen, allocates a new variable
%  with a stable name. Names follow the pattern wamAtom_<sanitized>_<seq>
%  to avoid collisions on atoms whose sanitized form would collapse.
intern_atom_go(AtomStr, GoVarName) :-
    atom_string(AtomStr, Str),
    (   go_atom_intern_id(Str, GoVarName)
    ->  true
    ;   retract(go_atom_intern_next(Seq)),
        Seq1 is Seq + 1,
        assertz(go_atom_intern_next(Seq1)),
        sanitize_atom_for_go_var(Str, Sanitized),
        format(atom(GoVarName), 'wamAtom_~w_~w', [Sanitized, Seq]),
        assertz(go_atom_intern_id(Str, GoVarName))
    ).

sanitize_atom_for_go_var("", "empty") :- !.
sanitize_atom_for_go_var(In, Out) :-
    string_codes(In, Codes),
    maplist(go_var_safe_code, Codes, OutCodes0),
    % Truncate to a sane upper bound to avoid runaway names
    length(OutCodes0, Len),
    (   Len =< 32
    ->  OutCodes = OutCodes0
    ;   length(OutCodes, 32),
        append(OutCodes, _, OutCodes0)
    ),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

go_var_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
go_var_safe_code(_, 0'_).

%% emit_atom_table_go(-GoCode) is det.
%  Emit a Go declaration block with:
%    1. A `var (...)` block for every interned atom
%    2. A by-name lookup map populated at package init time
%    3. An `internAtom(name string) *Atom` helper
%
%  The runtime helper is what makes pointer-identity equality pay off
%  for atoms the bench's main.go constructs at runtime — without it,
%  bench-side `&Atom{Name: x}` would always be a fresh pointer and
%  Atom.Equals would fall through to string compare. With it, the
%  bench can call `internAtom(x)` and get the SAME pointer the WAM
%  bytecode in lib.go references, so SwitchOnConstant matches in O(1).
%
%  Empty if no atoms were interned.
emit_atom_table_go(GoCode) :-
    findall(VarName-Str, go_atom_intern_id(Str, VarName), Pairs0),
    sort_pairs_by_seq(Pairs0, Pairs),
    (   Pairs == []
    ->  GoCode = ""
    ;   maplist(format_atom_decl, Pairs, DeclLines),
        atomic_list_concat([
            "// Interned atom literals (compile-time deduplicated)",
            "var (",
            ""
        ], "\n", DeclHeader),
        atomic_list_concat(DeclLines, "\n", DeclBody),
        atomic_list_concat([DeclHeader, DeclBody, "\n)\n"], DeclBlock),
        maplist(format_atom_register_line, Pairs, RegisterLines),
        atomic_list_concat(RegisterLines, "\n", RegisterBody),
        format(atom(RuntimeHelper),
'
// atomInternMap is populated at package init time with every interned
// atom from the var block above. internAtom(name) returns the shared
// pointer for that name, allocating + caching one if the name is new.
// Bench drivers should construct atoms via internAtom rather than
// `&Atom{Name: x}` so SwitchOnConstant in the WAM bytecode matches in
// O(1) (Atom.Equals short-circuits on pointer identity).
var atomInternMap = make(map[string]*Atom)

func init() {
~w
}

func internAtom(name string) *Atom {
    if a, ok := atomInternMap[name]; ok {
        return a
    }
    a := &Atom{Name: name}
    atomInternMap[name] = a
    return a
}
', [RegisterBody]),
        atomic_list_concat([DeclBlock, RuntimeHelper], GoCode)
    ).

format_atom_register_line(VarName-_, Line) :-
    format(atom(Line), "    atomInternMap[~w.Name] = ~w", [VarName, VarName]).

% Sort by the trailing _<seq> on the variable name so emission order
% matches first-seen order rather than alphabetical.
sort_pairs_by_seq(Pairs, Sorted) :-
    map_list_to_pairs(pair_seq_key, Pairs, Keyed),
    keysort(Keyed, KeyedSorted),
    pairs_values(KeyedSorted, Sorted).

pair_seq_key(VarName-_, Seq) :-
    atom_string(VarName, S),
    split_string(S, "_", "", Parts),
    last(Parts, SeqStr),
    number_string(Seq, SeqStr).

format_atom_decl(VarName-Str, Line) :-
    escape_go_string(Str, Escaped),
    format(atom(Line), "    ~w = &Atom{Name: \"~w\"}", [VarName, Escaped]).

% ============================================================================
% dimension_n Resolution (for codegen-time substitution)
%
% Mirrors wam_haskell_target.pl's resolve_dimension_n/2. Used by the
% effective-distance benchmark generator (and any other workload that
% needs a dimensionality value) so that user:dimension_n/1 reaches the
% generated Go code without being silently lost to a hardcoded default.
% ============================================================================

%% resolve_dimension_n_go(+Options, -DimN) is det.
%  Determine the aggregation formula's dimensionality. Priority order:
%    1. Options list: dimension_n(N)
%    2. user:dimension_n/1 fact (asserted by the workload)
%    3. Default: 5
%  Always returns a positive integer.
resolve_dimension_n_go(Options, DimN) :-
    (   option(dimension_n(N), Options),
        integer(N),
        N >= 1
    ->  DimN = N
    ;   current_predicate(user:dimension_n/1),
        user:dimension_n(N),
        integer(N),
        N >= 1
    ->  DimN = N
    ;   DimN = 5
    ).
