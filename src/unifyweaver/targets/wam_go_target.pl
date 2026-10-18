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
    escape_go_string/2,                % +Str, -EscStr (re-exported helper)
    iso_errors_rewrite_text/4,         % +Config, +PI, +WamText0, -WamText
    wam_go_iso_audit/3,                % +Predicates, +Options, -Audit
    wam_go_iso_audit_report/1          % +Audit
]).

:- use_module(library(lists)).
:- use_module(library(yall)).
:- use_module(library(option)).
:- use_module(library(pairs), [pairs_values/2, map_list_to_pairs/3]).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/recursive_kernel_detection',
             [detect_recursive_kernel/4, kernel_metadata/4]).
:- use_module('../core/cost_model', [resolve_auto_lmdb_materialisation/2]).
:- use_module('../core/iso_errors',
              [ iso_errors_resolve_options/2,
                iso_errors_load_config/2,
                iso_errors_mode_for/3,
                iso_errors_warn_multi_module/2,
                iso_errors_rewrite/4,
                iso_errors_rewrite_item/3,
                iso_errors_audit_normalise_pi/2,
                iso_errors_audit_walk/5
              ]).
% wam_text_parser's tokenizer is called module-qualified below: this
% target already defines its own wam_tokenize_line/2 for the Go literal
% emitter, and the two must not collide.
:- use_module('../targets/wam_text_parser',
              [ wam_recognise_instruction/2 ]).
:- use_module('../core/template_system').
:- use_module('../bindings/go_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../targets/go_target', [compile_predicate_to_go/3]).
:- use_module('../targets/wam_go_lowered_emitter',
             [wam_go_lowerable/3, lower_predicate_to_go/4, go_func_name/2]).
:- use_module('../targets/wam_go_templates',
             [go_render_template/3, go_render_helper_methods/1,
              go_render_step_method/1, go_preflight/0,
              go_render_file/3,
              go_render_project_lib_header/4,
              go_render_project_atoms_with_table/3,
              go_render_project_atoms_runtime_only/2,
              go_render_project_lowered_header/3,
              go_render_project_main_parallel/3]).
:- use_module('../core/prolog_term_parser', []).
:- use_module('../core/cpp_runtime_parser_wrappers', []).
:- use_module(wam_runtime_parser_capability, [
    parser_dependent_body_goal/2,
    wam_target_runtime_parser/3
]).

:- discontiguous wam_line_to_go_literal/4.

% ============================================================================
% PHASE 4: Hybrid Module Assembly
% ============================================================================

%% write_wam_go_project(+Predicates, +Options, +ProjectDir)
%  Generates a full Go project for the given predicates.
write_wam_go_project(Predicates0, Options, ProjectDir) :-
    % Runtime-parser mode. compiled(prolog_term_parser) appends the
    % portable parser + target-agnostic wrappers to the predicate list
    % so a generated project can call read_term_from_atom/2 directly;
    % `none` rejects a predicate whose body visibly needs a parser
    % rather than emitting a runtime that cannot honour the call.
    wam_target_runtime_parser(wam_go, Options, RuntimeParserMode),
    validate_go_runtime_parser_mode(Predicates0, RuntimeParserMode),
    go_project_predicates(Predicates0, RuntimeParserMode, Predicates),
    option(module_name(ModuleName), Options, 'wam_generated'),
    option(package_name(PackageName), Options, 'wam'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

    % Preflight the runtime templates before any output exists. A missing,
    % empty, or malformed template throws here (fail-closed), so a broken
    % asset never leaves a half-written project directory behind.
    go_preflight,

    % Create directory structure
    make_directory_path(ProjectDir),

    % Generate go.mod (fail-closed read + render through the adapter).
    go_render_file(go_mod, [module_name=ModuleName], GoModContent),
    directory_file_path(ProjectDir, 'go.mod', GoModPath),
    write_file(GoModPath, GoModContent),

    % Write value.go from template
    go_render_file(value, [package_name=PackageName, date=Date], ValueCode),
    directory_file_path(ProjectDir, 'value.go', ValuePath),
    write_file(ValuePath, ValueCode),

    % Write instructions.go from template
    go_render_file(instructions, [package_name=PackageName, date=Date], InstrCode),
    directory_file_path(ProjectDir, 'instructions.go', InstrPath),
    write_file(InstrPath, InstrCode),

    % Write state.go from template
    go_render_file(state, [package_name=PackageName, date=Date], StateCode),
    directory_file_path(ProjectDir, 'state.go', StatePath),
    write_file(StatePath, StateCode),

    % Generate runtime.go: shell spliced through the template adapter. The
    % adapter renders the helper section itself; the Prolog-assembled Step()
    % method is passed as a variable slot. Inserted Go is never rescanned, so
    % composite literals such as []qd{{source, 0}} stay literal.
    compile_step_wam_to_go(Options, StepMethod),
    go_render_template(runtime_shell,
        [package_name=PackageName, step_method=StepMethod], RuntimeCode),
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
    % lib.go has no import block of its own, but the native-strategy
    % predicate bodies reference std packages (fmt/context/sync/bufio/os/…).
    % Emit exactly the imports the generated code actually uses — Go errors
    % on both missing and unused imports.
    go_lib_import_block(PredicatesCode, ImportBlock),
    go_render_project_lib_header(PackageName, ImportBlock, PredicatesCode,
                                LibContent),
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
    ->  go_render_project_atoms_with_table(PackageName, AtomTableCode,
                                          AtomsContent)
    ;   go_render_project_atoms_runtime_only(PackageName, AtomsContent)
    ),
    directory_file_path(ProjectDir, 'atoms.go', AtomsPath),
    write_file(AtomsPath, AtomsContent),

    (   LoweredCode \== ""
    ->  go_render_project_lowered_header(PackageName, LoweredCode, LoweredContent),
        directory_file_path(ProjectDir, 'lowered.go', LoweredPath),
        write_file(LoweredPath, LoweredContent)
    ;   true
    ),

    % Generate main.go benchmark harness from template (when package is main)
    (   PackageName == main
    ->  go_render_file(main_bench, [package_name=PackageName], MainContent),
        directory_file_path(ProjectDir, 'main.go', MainPath),
        write_file(MainPath, MainContent)
    ;   option(parallel(true), Options)
    ->  go_render_project_main_parallel(ModuleName, PackageName, MainContent),
        directory_file_path(ProjectDir, 'main.go', MainPath),
        write_file(MainPath, MainContent)
    ;   true
    ),

    format('WAM Go project created at: ~w~n', [ProjectDir]).

%% go_project_predicates(+UserPreds, +Mode, -ProjectPreds)
%  In compiled mode, append the portable parser and the wrapper
%  predicates. Matches python_project_predicates/3 and
%  fsharp_project_predicates/3.
go_project_predicates(Predicates, compiled(prolog_term_parser), ProjectPredicates) :-
    !,
    go_runtime_parser_predicates(ParserPreds),
    go_runtime_parser_wrapper_predicates(WrapperPreds),
    append([Predicates, ParserPreds, WrapperPreds], Combined),
    sort(Combined, ProjectPredicates).
go_project_predicates(Predicates, _RuntimeParserMode, Predicates).

%% go_runtime_parser_predicates(-Predicates)
%  Non-imported predicates of `prolog_term_parser`, as
%  `prolog_term_parser:Name/Arity` indicators.
go_runtime_parser_predicates(Predicates) :-
    findall(prolog_term_parser:Name/Arity,
            ( current_predicate(prolog_term_parser:Name/Arity),
              functor(Head, Name, Arity),
              once(clause(prolog_term_parser:Head, _)),
              \+ predicate_property(prolog_term_parser:Head, imported_from(_))
            ),
            Raw),
    sort(Raw, Predicates).

%% go_runtime_parser_wrapper_predicates(-Predicates)
%  Non-imported predicates of `cpp_runtime_parser_wrappers` (the module
%  name is C++-historical; the wrappers themselves are target-agnostic).
%  These are what give the generated project read_term_from_atom/2,3.
go_runtime_parser_wrapper_predicates(Predicates) :-
    findall(cpp_runtime_parser_wrappers:Name/Arity,
            ( current_predicate(cpp_runtime_parser_wrappers:Name/Arity),
              functor(Head, Name, Arity),
              once(clause(cpp_runtime_parser_wrappers:Head, _)),
              \+ predicate_property(cpp_runtime_parser_wrappers:Head,
                                    imported_from(_))
            ),
            Raw),
    sort(Raw, Predicates).

%% validate_go_runtime_parser_mode(+Predicates, +Mode)
%  With the parser disabled, reject a predicate whose body has a
%  statically visible parser-dependent call rather than emitting a
%  runtime that silently cannot serve it. Mirrors
%  validate_python_runtime_parser_mode/2.
validate_go_runtime_parser_mode(Predicates, none) :-
    !,
    (   go_predicates_parser_dependency(Predicates, Pred, Builtin)
    ->  throw(error(permission_error(use, runtime_parser, Builtin),
                    context(write_wam_go_project/3,
                            parser_disabled_for_predicate(Pred))))
    ;   true
    ).
validate_go_runtime_parser_mode(_Predicates, _Mode).

go_predicates_parser_dependency(Predicates, Pred, Builtin) :-
    member(Pred, Predicates),
    go_predicate_clause(Pred, _Head, Body),
    parser_dependent_body_goal(Body, Builtin),
    !.

go_predicate_clause(Module:Name/Arity, Head, Body) :-
    !,
    functor(Head, Name, Arity),
    catch(clause(Module:Head, Body), _, fail).
go_predicate_clause(Name/Arity, Head, Body) :-
    functor(Head, Name, Arity),
    catch(clause(user:Head, Body), _, fail).

%% go_lib_import_block(+Code, -ImportBlock)
%  Emit a Go `import (...)` block listing exactly the std packages the
%  generated native predicate bodies reference. Go rejects both missing
%  and unused imports, so we detect actual usage ("<pkg>.") in the code
%  with line comments stripped (to avoid false positives like
%  "// fmt.Sprintf style"). Empty block when nothing is used.
go_lib_import_block(Code, ImportBlock) :-
    go_strip_line_comments(Code, Stripped),
    findall(Path,
        ( go_std_import(Marker, Path), sub_string(Stripped, _, _, _, Marker) ),
        Paths0),
    sort(Paths0, Paths),
    (   Paths == []
    ->  ImportBlock = ""
    ;   maplist([P, L]>>format(string(L), '\t"~w"', [P]), Paths, Ls),
        atomic_list_concat(Ls, "\n", Inner),
        format(string(ImportBlock), 'import (\n~w\n)\n\n', [Inner])
    ).

go_strip_line_comments(Code, Stripped) :-
    split_string(Code, "\n", "", Lines),
    maplist(go_strip_line_comment, Lines, SLines),
    atomic_list_concat(SLines, "\n", Stripped).
go_strip_line_comment(Line, Out) :-
    ( sub_string(Line, B, _, _, "//") -> sub_string(Line, 0, B, _, Out) ; Out = Line ).

%% go_pred_has_control_constructs(+Module:Pred/Arity)
%  True if any clause body uses ->, ; (disjunction / if-then-else), \+, or
%  once. The native Go strategy mis-compiles these (interface{} comparisons
%  for ->, an unwrapped stdin-pipeline fragment for \+); the WAM-lowered
%  path handles them correctly, so such predicates are routed to WAM.
%  See docs/GO_TARGET.md "Native control-construct codegen (deferred)" for
%  what is blocked and what a proper native implementation could add.
go_pred_has_control_constructs(Module:Pred/Arity) :-
    functor(Head, Pred, Arity),
    clause(Module:Head, Body),
    go_body_has_control(Body),
    !.

go_body_has_control(G) :- var(G), !, fail.
go_body_has_control((_ -> _)) :- !.
go_body_has_control((_ ; _)) :- !.
go_body_has_control(\+ _) :- !.
go_body_has_control(not(_)) :- !.
go_body_has_control(once(_)) :- !.
go_body_has_control((A , B)) :- !, ( go_body_has_control(A) -> true ; go_body_has_control(B) ).
go_body_has_control(_) :- fail.

%% go_std_import(+UsageMarker, -ImportPath)
go_std_import("fmt.",     "fmt").
go_std_import("context.", "context").
go_std_import("sync.",    "sync").
go_std_import("bufio.",   "bufio").
go_std_import("os.",      "os").
go_std_import("strings.", "strings").
go_std_import("strconv.", "strconv").
go_std_import("sort.",    "sort").

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
              once(( go_compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode),
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

% (read_template_file/2 and resolve_template_path/2 were removed in Phase 4
% slice 3: every project template now loads through the fail-closed adapter
% wam_go_templates:go_render_file/3 / go_render_project_*/N, replacing the old
% cwd-then-module lookup and its silent "// Template not found" fallback.)

%% compile_predicates_for_project(+Predicates, +Options, -Code)
compile_predicates_for_project([], _, "").
compile_predicates_for_project(Predicates, Options0, Code) :-
    % Go has no overloading: resolve name/arity collisions once, up front,
    % and thread the result through every emitter that mints an identifier.
    go_overloaded_predicate_names(Predicates, Overloaded),
    Options = [go_overloaded_names(Overloaded)|Options0],
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
    generate_predicate_codes(Classified, WamEntries, Options, PredCodes),
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

%% go_compile_predicate_to_wam(+PI, +Options, -WamCode)
%  compile_predicate_to_wam/3 plus the ISO three-form key rewrite.
%
%  The rewrite only runs when the caller actually configured ISO error
%  handling -- iso_errors(true|false), a per-predicate iso_errors(PI,
%  Mode) override, or iso_errors_config(File). Without any of those the
%  WAM text passes through untouched and generated projects keep the
%  plain builtin keys they have always used.
%
%  (Python rewrites unconditionally, mapping every default key to its
%  _lax form when no config is present. That is behaviour-preserving --
%  the _lax keys delegate to the plain ones -- but it would rewrite the
%  keys in every existing Go project for no gain, so Go opts in
%  instead.)
go_compile_predicate_to_wam(PI, Options, WamCode) :-
    % inline_bagof_setof: bagof/setof otherwise emit `call bagof/3`
    % (not a Go builtin) and fail silently. Required for the cut-
    % semantics probes (p14/p15) and any program that uses bagof/setof.
    wam_target:compile_predicate_to_wam(PI,
        [inline_bagof_setof(true)|Options], WamCode0),
    (   go_iso_configured(Options)
    ->  iso_errors_resolve_options(Options, Config),
        iso_errors_audit_normalise_pi(PI, NormPI),
        iso_errors_rewrite_text(Config, NormPI, WamCode0, WamCode)
    ;   WamCode = WamCode0
    ).

%% go_iso_configured(+Options)
%  True when Options carry any ISO error configuration.
go_iso_configured(Options) :-
    (   member(iso_errors(_), Options)
    ->  true
    ;   member(iso_errors(_, _), Options)
    ->  true
    ;   member(iso_errors_config(_), Options)
    ).

%% go_wam_fact_source_spec(+P, +Arity, +Options, -Spec)
%  True when Options declare a store-backed source for P/Arity. Only P/2
%  is streamed (mirrors the wamjs javascript_wam_fact_sources contract).
%    Spec = indexed(Prefix)   % D43 UWFI/UWIX seek store: Prefix.data + .idx
%         | lmdb(Dir)         % opt-in; loud error if no reader is built in
go_wam_fact_source_spec(P, Arity, Options, Spec) :-
    Arity =:= 2,
    (   option(go_wam_fact_sources(Sources), Options)
    ->  true
    ;   option(js_fact_sources(Sources), Options)
    ->  true
    ;   Sources = []
    ),
    member(source(PI, Spec), Sources),
    go_wam_fact_source_pi_match(PI, P, Arity).

go_wam_fact_source_pi_match(_:Name/Ar, P, Arity) :- !,
    Name == P, Ar =:= Arity.
go_wam_fact_source_pi_match(Name/Ar, P, Arity) :-
    Name == P, Ar =:= Arity.

%% go_fact_stream_wam_text(+P, +Arity, -WamText)
%  The two-instruction predicate body streamed for a fact source.
go_fact_stream_wam_text(P, Arity, WamText) :-
    format(atom(WamText),
        '~w/~w:\n    call_fact_stream ~w/~w\n    proceed\n',
        [P, Arity, P, Arity]).

%% go_wam_fact_source_setup_lines(+Classified, +Options, -Lines)
%  One registration call per store-backed source (indexed seek or lmdb gate).
go_wam_fact_source_setup_lines(Classified, Options, Lines) :-
    findall(Line,
        ( member(classified(_M, Pred, Arity, _Kind, _), Classified),
          go_wam_fact_source_spec(Pred, Arity, Options, Spec),
          go_wam_fact_source_register_line(Pred/Arity, Spec, Line)
        ),
        Lines0),
    sort(Lines0, Lines).

go_wam_fact_source_register_line(Pred/Arity, indexed(Prefix), Line) :-
    go_store_abs_path(Prefix, AbsPrefix),
    escape_go_string(AbsPrefix, EscPrefix),
    format(atom(Line),
        '    vm.registerIndexedSeekFact2("~w/~w", "~w")',
        [Pred, Arity, EscPrefix]).
go_wam_fact_source_register_line(Pred/Arity, lmdb(Dir), Line) :-
    go_store_abs_path(Dir, AbsDir),
    escape_go_string(AbsDir, EscDir),
    format(atom(Line),
        '    vm.registerLmdbSeekFact2("~w/~w", "~w")',
        [Pred, Arity, EscDir]).

go_store_abs_path(Path, Abs) :-
    atom_string(Path, PathStr),
    working_directory(Cwd, Cwd),
    (   catch(absolute_file_name(PathStr, Abs0, [relative_to(Cwd)]), _, fail),
        Abs0 \== []
    ->  Abs = Abs0
    ;   Abs = PathStr
    ).

classify_predicates([], _, []).
classify_predicates([PredIndicator|Rest], Options, [Entry|RestEntries]) :-
    predicate_indicator_parts(PredIndicator, Module, Pred, Arity),
    (   go_wam_fact_source_spec(Pred, Arity, Options, _Spec)
        % Store-backed P/2 fact source (D43 indexed seek / lmdb gate).
        % Compile to a two-instruction body [call_fact_stream, proceed] so
        % callers reach it by label exactly like any other predicate. The
        % source itself is registered in setupSharedForeignPredicates.
    ->  go_fact_stream_wam_text(Pred, Arity, WamText),
        format(user_error, '  ~w/~w: store-backed fact source~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, wam, WamText)
    ;   is_ffi_owned_fact(Module, Pred, Arity, Options)
    ->  format(user_error, '  ~w/~w: FFI-owned fact (skipping WAM compilation)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, ffi_fact, '')
    ;   option(prefer_wam(true), Options),
        option(wam_fallback(WamFB0), Options, true),
        WamFB0 \== false,
        go_foreign_spec(Module:Pred/Arity, Options, _SetupOps0, _RewriteCalls0, _EntryPred0/_EntryArity0),
        go_compile_predicate_to_wam(Module:Pred/Arity, [ite_use_y_level(true)|Options], WamCode0)
    ->  compile_wam_predicate_to_go(Module:Pred/Arity, WamCode0, Options, PredCode0),
        format(user_error, '  ~w/~w: WAM fallback (foreign, preferred)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, wam_foreign, PredCode0)
    ;   option(prefer_wam(true), Options),
        option(wam_fallback(WamFB1), Options, true),
        WamFB1 \== false,
        go_compile_predicate_to_wam(Module:Pred/Arity, [ite_use_y_level(true)|Options], WamCode1)
    ->  format(user_error, '  ~w/~w: WAM fallback (preferred)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, wam, WamCode1)
    ;   go_foreign_spec(Module:Pred/Arity, Options, _SetupOps, _RewriteCalls, _EntryPred/_EntryArity),
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        go_compile_predicate_to_wam(Module:Pred/Arity, [ite_use_y_level(true)|Options], WamCode)
    ->  compile_wam_predicate_to_go(Module:Pred/Arity, WamCode, Options, PredCode),
        format(user_error, '  ~w/~w: WAM fallback (foreign)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, wam_foreign, PredCode)
    ;   % The native Go strategy mis-compiles control constructs: (C->T;E)
        % becomes an `interface{} > 0` comparison (a type error) and \+ an
        % unwrapped stdin-pipeline fragment. These are handled correctly by
        % the WAM-lowered path, so decline native here and fall through to
        % the WAM fallback branch below.
        \+ go_pred_has_control_constructs(Module:Pred/Arity),
        catch(
            go_target:compile_predicate_to_go(Module:Pred/Arity,
                [include_package(false)|Options], PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, native, PredCode)
    ;   option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        go_compile_predicate_to_wam(Module:Pred/Arity, [ite_use_y_level(true)|Options], WamCode)
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

generate_predicate_codes([], _, _, []).
generate_predicate_codes([classified(_, Pred, Arity, ffi_fact, _)|Rest], WamEntries, Options,
                         [Code|RestCodes]) :-
    format(atom(Code), '// ~w/~w: FFI-owned fact — WAM compilation skipped', [Pred, Arity]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, native, PredCode)|Rest], WamEntries, Options,
                         [Code|RestCodes]) :-
    format(atom(Code), '// Strategy: native~n~w', [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, wam_foreign, PredCode)|Rest], WamEntries, Options,
                         [Code|RestCodes]) :-
    format(atom(Code), '// Strategy: wam_foreign~n~w', [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, Pred, Arity, wam, _WamCode)|Rest], WamEntries, Options,
                         [Code|RestCodes]) :-
    member(wam_entry(Pred, Arity, StartPC), WamEntries),
    compile_wam_predicate_to_go_shared(Pred/Arity, StartPC, Options, Code),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, failed, PredCode)|Rest], WamEntries, Options,
                         [Code|RestCodes]) :-
    format(atom(Code), '// Strategy: failed~n~w', [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).

compile_shared_foreign_setup(Classified, Options, Code) :-
    findall(Line,
        ( member(classified(Module, Pred, Arity, wam_foreign, _), Classified),
          go_foreign_spec(Module:Pred/Arity, Options, SetupOps, _RewriteCalls, _EntryPred/_EntryArity),
          maplist([Op, L]>>go_foreign_setup_line(Op, Options, L), SetupOps, SetupLines),
          member(Line, SetupLines)
        ),
        RawLines),
    sort(RawLines, ForeignLines),
    go_wam_fact_source_setup_lines(Classified, Options, FactLines),
    append(ForeignLines, FactLines, AllLines),
    (   AllLines == []
    ->  Body = ""
    ;   atomic_list_concat(AllLines, '\n', Body)
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

wam_go_direct_builtin(Pred, Arity) :-
    wam_go_direct_builtin(Pred, Arity, _Op).

wam_go_direct_builtin(memberchk/2, 2, 'memberchk/2').
wam_go_direct_builtin('memberchk/2', 2, 'memberchk/2').
wam_go_direct_builtin("memberchk/2", 2, 'memberchk/2').
wam_go_direct_builtin(select/3, 3, 'select/3').
wam_go_direct_builtin('select/3', 3, 'select/3').
wam_go_direct_builtin("select/3", 3, 'select/3').
wam_go_direct_builtin(delete/3, 3, 'delete/3').
wam_go_direct_builtin('delete/3', 3, 'delete/3').
wam_go_direct_builtin("delete/3", 3, 'delete/3').
wam_go_direct_builtin(subtract/3, 3, 'subtract/3').
wam_go_direct_builtin('subtract/3', 3, 'subtract/3').
wam_go_direct_builtin("subtract/3", 3, 'subtract/3').
wam_go_direct_builtin(intersection/3, 3, 'intersection/3').
wam_go_direct_builtin('intersection/3', 3, 'intersection/3').
wam_go_direct_builtin("intersection/3", 3, 'intersection/3').
wam_go_direct_builtin(union/3, 3, 'union/3').
wam_go_direct_builtin('union/3', 3, 'union/3').
wam_go_direct_builtin("union/3", 3, 'union/3').
wam_go_direct_builtin(append/3, 3, 'append/3').
wam_go_direct_builtin('append/3', 3, 'append/3').
wam_go_direct_builtin("append/3", 3, 'append/3').
wam_go_direct_builtin(permutation/2, 2, 'permutation/2').
wam_go_direct_builtin('permutation/2', 2, 'permutation/2').
wam_go_direct_builtin("permutation/2", 2, 'permutation/2').
wam_go_direct_builtin(reverse/2, 2, 'reverse/2').
wam_go_direct_builtin('reverse/2', 2, 'reverse/2').
wam_go_direct_builtin("reverse/2", 2, 'reverse/2').
wam_go_direct_builtin(last/2, 2, 'last/2').
wam_go_direct_builtin('last/2', 2, 'last/2').
wam_go_direct_builtin("last/2", 2, 'last/2').
wam_go_direct_builtin(nth0/3, 3, 'nth0/3').
wam_go_direct_builtin('nth0/3', 3, 'nth0/3').
wam_go_direct_builtin("nth0/3", 3, 'nth0/3').
wam_go_direct_builtin(nth1/3, 3, 'nth1/3').
wam_go_direct_builtin('nth1/3', 3, 'nth1/3').
wam_go_direct_builtin("nth1/3", 3, 'nth1/3').
wam_go_direct_builtin(numlist/3, 3, 'numlist/3').
wam_go_direct_builtin('numlist/3', 3, 'numlist/3').
wam_go_direct_builtin("numlist/3", 3, 'numlist/3').
wam_go_direct_builtin(between/3, 3, 'between/3').
wam_go_direct_builtin('between/3', 3, 'between/3').
wam_go_direct_builtin("between/3", 3, 'between/3').
% call/1 is compiled as `call call/1` (not is_builtin_pred), so without
% this rewrite the Go Call misses the label and fails. Opaque: a cut
% inside the metacall prunes only choice points created during it.
wam_go_direct_builtin(call, 1, 'call/1').
wam_go_direct_builtin(call/1, 1, 'call/1').
wam_go_direct_builtin('call/1', 1, 'call/1').
wam_go_direct_builtin("call/1", 1, 'call/1').
% P3 pkg_resolver: maplist/N meta-calls user preds (is_v3/1); predsort/3
% drives cmp_ver/3 for deb/3 order. Without these, Call{"maplist/N"}
% misses Labels and fails silently (A3). functor/3 is implemented in
% executeBuiltin but was unclassified — D60's silent-fail class.
wam_go_direct_builtin(maplist/2, 2, 'maplist/2').
wam_go_direct_builtin('maplist/2', 2, 'maplist/2').
wam_go_direct_builtin("maplist/2", 2, 'maplist/2').
wam_go_direct_builtin(maplist/3, 3, 'maplist/3').
wam_go_direct_builtin('maplist/3', 3, 'maplist/3').
wam_go_direct_builtin("maplist/3", 3, 'maplist/3').
wam_go_direct_builtin(maplist/4, 4, 'maplist/4').
wam_go_direct_builtin('maplist/4', 4, 'maplist/4').
wam_go_direct_builtin("maplist/4", 4, 'maplist/4').
wam_go_direct_builtin(predsort/3, 3, 'predsort/3').
wam_go_direct_builtin('predsort/3', 3, 'predsort/3').
wam_go_direct_builtin("predsort/3", 3, 'predsort/3').
wam_go_direct_builtin(functor/3, 3, 'functor/3').
wam_go_direct_builtin('functor/3', 3, 'functor/3').
wam_go_direct_builtin("functor/3", 3, 'functor/3').
wam_go_direct_builtin(arg/3, 3, 'arg/3').
wam_go_direct_builtin('arg/3', 3, 'arg/3').
wam_go_direct_builtin("arg/3", 3, 'arg/3').
wam_go_direct_builtin('=..', 2, '=../2').
wam_go_direct_builtin('=..'/2, 2, '=../2').
wam_go_direct_builtin('=../2', 2, '=../2').
wam_go_direct_builtin("=../2", 2, '=../2').
wam_go_direct_builtin(writeln/1, 1, 'writeln/1').
wam_go_direct_builtin('writeln/1', 1, 'writeln/1').
wam_go_direct_builtin("writeln/1", 1, 'writeln/1').
wam_go_direct_builtin(print/1, 1, 'print/1').
wam_go_direct_builtin('print/1', 1, 'print/1').
wam_go_direct_builtin("print/1", 1, 'print/1').
wam_go_direct_builtin(write_canonical/1, 1, 'write_canonical/1').
wam_go_direct_builtin('write_canonical/1', 1, 'write_canonical/1').
wam_go_direct_builtin("write_canonical/1", 1, 'write_canonical/1').
wam_go_direct_builtin(put_char/1, 1, 'put_char/1').
wam_go_direct_builtin('put_char/1', 1, 'put_char/1').
wam_go_direct_builtin("put_char/1", 1, 'put_char/1').
wam_go_direct_builtin(put_code/1, 1, 'put_code/1').
wam_go_direct_builtin('put_code/1', 1, 'put_code/1').
wam_go_direct_builtin("put_code/1", 1, 'put_code/1').
wam_go_direct_builtin(put_char/2, 2, 'put_char/2').
wam_go_direct_builtin('put_char/2', 2, 'put_char/2').
wam_go_direct_builtin("put_char/2", 2, 'put_char/2').
wam_go_direct_builtin(put_code/2, 2, 'put_code/2').
wam_go_direct_builtin('put_code/2', 2, 'put_code/2').
wam_go_direct_builtin("put_code/2", 2, 'put_code/2').
wam_go_direct_builtin(get_char/1, 1, 'get_char/1').
wam_go_direct_builtin('get_char/1', 1, 'get_char/1').
wam_go_direct_builtin("get_char/1", 1, 'get_char/1').
wam_go_direct_builtin(get_char/2, 2, 'get_char/2').
wam_go_direct_builtin('get_char/2', 2, 'get_char/2').
wam_go_direct_builtin("get_char/2", 2, 'get_char/2').
wam_go_direct_builtin(peek_char/1, 1, 'peek_char/1').
wam_go_direct_builtin('peek_char/1', 1, 'peek_char/1').
wam_go_direct_builtin("peek_char/1", 1, 'peek_char/1').
wam_go_direct_builtin(peek_char/2, 2, 'peek_char/2').
wam_go_direct_builtin('peek_char/2', 2, 'peek_char/2').
wam_go_direct_builtin("peek_char/2", 2, 'peek_char/2').
wam_go_direct_builtin(get_code/1, 1, 'get_code/1').
wam_go_direct_builtin('get_code/1', 1, 'get_code/1').
wam_go_direct_builtin("get_code/1", 1, 'get_code/1').
wam_go_direct_builtin(get_code/2, 2, 'get_code/2').
wam_go_direct_builtin('get_code/2', 2, 'get_code/2').
wam_go_direct_builtin("get_code/2", 2, 'get_code/2').
wam_go_direct_builtin(open/3, 3, 'open/3').
wam_go_direct_builtin('open/3', 3, 'open/3').
wam_go_direct_builtin("open/3", 3, 'open/3').
wam_go_direct_builtin(close/1, 1, 'close/1').
wam_go_direct_builtin('close/1', 1, 'close/1').
wam_go_direct_builtin("close/1", 1, 'close/1').
wam_go_direct_builtin(read_line_to_string/2, 2, 'read_line_to_string/2').
wam_go_direct_builtin('read_line_to_string/2', 2, 'read_line_to_string/2').
wam_go_direct_builtin("read_line_to_string/2", 2, 'read_line_to_string/2').
wam_go_direct_builtin(read_string/5, 5, 'read_string/5').
wam_go_direct_builtin('read_string/5', 5, 'read_string/5').
wam_go_direct_builtin("read_string/5", 5, 'read_string/5').
wam_go_direct_builtin(at_end_of_stream/1, 1, 'at_end_of_stream/1').
wam_go_direct_builtin('at_end_of_stream/1', 1, 'at_end_of_stream/1').
wam_go_direct_builtin("at_end_of_stream/1", 1, 'at_end_of_stream/1').
wam_go_direct_builtin(write_to_stream/2, 2, 'write_to_stream/2').
wam_go_direct_builtin('write_to_stream/2', 2, 'write_to_stream/2').
wam_go_direct_builtin("write_to_stream/2", 2, 'write_to_stream/2').
wam_go_direct_builtin(nl_to_stream/1, 1, 'nl_to_stream/1').
wam_go_direct_builtin('nl_to_stream/1', 1, 'nl_to_stream/1').
wam_go_direct_builtin("nl_to_stream/1", 1, 'nl_to_stream/1').
wam_go_direct_builtin(format/1, 1, 'format/1').
wam_go_direct_builtin('format/1', 1, 'format/1').
wam_go_direct_builtin("format/1", 1, 'format/1').
wam_go_direct_builtin(format/2, 2, 'format/2').
wam_go_direct_builtin('format/2', 2, 'format/2').
wam_go_direct_builtin("format/2", 2, 'format/2').
wam_go_direct_builtin(sum_list/2, 2, 'sum_list/2').
wam_go_direct_builtin('sum_list/2', 2, 'sum_list/2').
wam_go_direct_builtin("sum_list/2", 2, 'sum_list/2').
wam_go_direct_builtin(min_list/2, 2, 'min_list/2').
wam_go_direct_builtin('min_list/2', 2, 'min_list/2').
wam_go_direct_builtin("min_list/2", 2, 'min_list/2').
wam_go_direct_builtin(max_list/2, 2, 'max_list/2').
wam_go_direct_builtin('max_list/2', 2, 'max_list/2').
wam_go_direct_builtin("max_list/2", 2, 'max_list/2').
wam_go_direct_builtin(list_to_set/2, 2, 'list_to_set/2').
wam_go_direct_builtin('list_to_set/2', 2, 'list_to_set/2').
wam_go_direct_builtin("list_to_set/2", 2, 'list_to_set/2').
wam_go_direct_builtin(sort/2, 2, 'sort/2').
wam_go_direct_builtin('sort/2', 2, 'sort/2').
wam_go_direct_builtin("sort/2", 2, 'sort/2').
wam_go_direct_builtin(msort/2, 2, 'msort/2').
wam_go_direct_builtin('msort/2', 2, 'msort/2').
wam_go_direct_builtin("msort/2", 2, 'msort/2').
wam_go_direct_builtin(keysort/2, 2, 'keysort/2').
wam_go_direct_builtin('keysort/2', 2, 'keysort/2').
wam_go_direct_builtin("keysort/2", 2, 'keysort/2').
wam_go_direct_builtin('@</2', 2, '@</2').
wam_go_direct_builtin("@</2", 2, '@</2').
wam_go_direct_builtin('@=</2', 2, '@=</2').
wam_go_direct_builtin("@=</2", 2, '@=</2').
wam_go_direct_builtin('@>/2', 2, '@>/2').
wam_go_direct_builtin("@>/2", 2, '@>/2').
wam_go_direct_builtin('@>=/2', 2, '@>=/2').
wam_go_direct_builtin("@>=/2", 2, '@>=/2').
wam_go_direct_builtin(compare/3, 3, 'compare/3').
wam_go_direct_builtin('compare/3', 3, 'compare/3').
wam_go_direct_builtin("compare/3", 3, 'compare/3').
wam_go_direct_builtin(ground/1, 1, 'ground/1').
wam_go_direct_builtin('ground/1', 1, 'ground/1').
wam_go_direct_builtin("ground/1", 1, 'ground/1').
wam_go_direct_builtin(sub_atom/5, 5, 'sub_atom/5').
wam_go_direct_builtin('sub_atom/5', 5, 'sub_atom/5').
wam_go_direct_builtin("sub_atom/5", 5, 'sub_atom/5').
wam_go_direct_builtin(char_type/2, 2, 'char_type/2').
wam_go_direct_builtin('char_type/2', 2, 'char_type/2').
wam_go_direct_builtin("char_type/2", 2, 'char_type/2').
wam_go_direct_builtin(string_code/3, 3, 'string_code/3').
wam_go_direct_builtin('string_code/3', 3, 'string_code/3').
wam_go_direct_builtin("string_code/3", 3, 'string_code/3').
wam_go_direct_builtin(split_string/4, 4, 'split_string/4').
wam_go_direct_builtin('split_string/4', 4, 'split_string/4').
wam_go_direct_builtin("split_string/4", 4, 'split_string/4').
wam_go_direct_builtin(tab/1, 1, 'tab/1').
wam_go_direct_builtin('tab/1', 1, 'tab/1').
wam_go_direct_builtin("tab/1", 1, 'tab/1').
wam_go_direct_builtin(getenv/2, 2, 'getenv/2').
wam_go_direct_builtin('getenv/2', 2, 'getenv/2').
wam_go_direct_builtin("getenv/2", 2, 'getenv/2').
wam_go_direct_builtin(setenv/2, 2, 'setenv/2').
wam_go_direct_builtin('setenv/2', 2, 'setenv/2').
wam_go_direct_builtin("setenv/2", 2, 'setenv/2').
wam_go_direct_builtin(succ/2, 2, 'succ/2').
wam_go_direct_builtin('succ/2', 2, 'succ/2').
wam_go_direct_builtin("succ/2", 2, 'succ/2').
wam_go_direct_builtin(atom_number/2, 2, 'atom_number/2').
wam_go_direct_builtin('atom_number/2', 2, 'atom_number/2').
wam_go_direct_builtin("atom_number/2", 2, 'atom_number/2').
wam_go_direct_builtin(upcase_atom/2, 2, 'upcase_atom/2').
wam_go_direct_builtin('upcase_atom/2', 2, 'upcase_atom/2').
wam_go_direct_builtin("upcase_atom/2", 2, 'upcase_atom/2').
wam_go_direct_builtin(downcase_atom/2, 2, 'downcase_atom/2').
wam_go_direct_builtin('downcase_atom/2', 2, 'downcase_atom/2').
wam_go_direct_builtin("downcase_atom/2", 2, 'downcase_atom/2').
wam_go_direct_builtin(atom_concat/3, 3, 'atom_concat/3').
wam_go_direct_builtin('atom_concat/3', 3, 'atom_concat/3').
wam_go_direct_builtin("atom_concat/3", 3, 'atom_concat/3').
wam_go_direct_builtin(atom_length/2, 2, 'atom_length/2').
wam_go_direct_builtin('atom_length/2', 2, 'atom_length/2').
wam_go_direct_builtin("atom_length/2", 2, 'atom_length/2').
wam_go_direct_builtin(string_length/2, 2, 'string_length/2').
wam_go_direct_builtin('string_length/2', 2, 'string_length/2').
wam_go_direct_builtin("string_length/2", 2, 'string_length/2').
wam_go_direct_builtin(char_code/2, 2, 'char_code/2').
wam_go_direct_builtin('char_code/2', 2, 'char_code/2').
wam_go_direct_builtin("char_code/2", 2, 'char_code/2').
wam_go_direct_builtin(atom_codes/2, 2, 'atom_codes/2').
wam_go_direct_builtin('atom_codes/2', 2, 'atom_codes/2').
wam_go_direct_builtin("atom_codes/2", 2, 'atom_codes/2').
wam_go_direct_builtin(atom_chars/2, 2, 'atom_chars/2').
wam_go_direct_builtin('atom_chars/2', 2, 'atom_chars/2').
wam_go_direct_builtin("atom_chars/2", 2, 'atom_chars/2').
wam_go_direct_builtin(string_codes/2, 2, 'string_codes/2').
wam_go_direct_builtin('string_codes/2', 2, 'string_codes/2').
wam_go_direct_builtin("string_codes/2", 2, 'string_codes/2').
wam_go_direct_builtin(string_chars/2, 2, 'string_chars/2').
wam_go_direct_builtin('string_chars/2', 2, 'string_chars/2').
wam_go_direct_builtin("string_chars/2", 2, 'string_chars/2').
wam_go_direct_builtin(number_codes/2, 2, 'number_codes/2').
wam_go_direct_builtin('number_codes/2', 2, 'number_codes/2').
wam_go_direct_builtin("number_codes/2", 2, 'number_codes/2').
wam_go_direct_builtin(number_chars/2, 2, 'number_chars/2').
wam_go_direct_builtin('number_chars/2', 2, 'number_chars/2').
wam_go_direct_builtin("number_chars/2", 2, 'number_chars/2').
wam_go_direct_builtin(number_string/2, 2, 'number_string/2').
wam_go_direct_builtin('number_string/2', 2, 'number_string/2').
wam_go_direct_builtin("number_string/2", 2, 'number_string/2').
wam_go_direct_builtin(atom_string/2, 2, 'atom_string/2').
wam_go_direct_builtin('atom_string/2', 2, 'atom_string/2').
wam_go_direct_builtin("atom_string/2", 2, 'atom_string/2').
wam_go_direct_builtin(string_to_atom/2, 2, 'string_to_atom/2').
wam_go_direct_builtin('string_to_atom/2', 2, 'string_to_atom/2').
wam_go_direct_builtin("string_to_atom/2", 2, 'string_to_atom/2').

% ISO three-form keys plus the catch/throw substrate. Registering
% these makes `builtin_call <key>` and `execute <key>` route to the
% runtime's ISO branches; without an entry the emitter falls through
% to Execute{Pred: ...}, which looks the key up as an indexed fact
% table and silently fails. See the ISO ERROR HANDLING section.
wam_go_direct_builtin(catch/3, 3, 'catch/3').
wam_go_direct_builtin('catch/3', 3, 'catch/3').
wam_go_direct_builtin("catch/3", 3, 'catch/3').
wam_go_direct_builtin(throw/1, 1, 'throw/1').
wam_go_direct_builtin('throw/1', 1, 'throw/1').
wam_go_direct_builtin("throw/1", 1, 'throw/1').
wam_go_direct_builtin(is_iso/2, 2, 'is_iso/2').
wam_go_direct_builtin('is_iso/2', 2, 'is_iso/2').
wam_go_direct_builtin("is_iso/2", 2, 'is_iso/2').
wam_go_direct_builtin(is_lax/2, 2, 'is_lax/2').
wam_go_direct_builtin('is_lax/2', 2, 'is_lax/2').
wam_go_direct_builtin("is_lax/2", 2, 'is_lax/2').
wam_go_direct_builtin(succ_iso/2, 2, 'succ_iso/2').
wam_go_direct_builtin('succ_iso/2', 2, 'succ_iso/2').
wam_go_direct_builtin("succ_iso/2", 2, 'succ_iso/2').
wam_go_direct_builtin(succ_lax/2, 2, 'succ_lax/2').
wam_go_direct_builtin('succ_lax/2', 2, 'succ_lax/2').
wam_go_direct_builtin("succ_lax/2", 2, 'succ_lax/2').
wam_go_direct_builtin('<_iso'/2, 2, '<_iso/2').
wam_go_direct_builtin('<_iso/2', 2, '<_iso/2').
wam_go_direct_builtin("<_iso/2", 2, '<_iso/2').
wam_go_direct_builtin('<_lax'/2, 2, '<_lax/2').
wam_go_direct_builtin('<_lax/2', 2, '<_lax/2').
wam_go_direct_builtin("<_lax/2", 2, '<_lax/2').
wam_go_direct_builtin('>_iso'/2, 2, '>_iso/2').
wam_go_direct_builtin('>_iso/2', 2, '>_iso/2').
wam_go_direct_builtin(">_iso/2", 2, '>_iso/2').
wam_go_direct_builtin('>_lax'/2, 2, '>_lax/2').
wam_go_direct_builtin('>_lax/2', 2, '>_lax/2').
wam_go_direct_builtin(">_lax/2", 2, '>_lax/2').
wam_go_direct_builtin('=<_iso'/2, 2, '=<_iso/2').
wam_go_direct_builtin('=<_iso/2', 2, '=<_iso/2').
wam_go_direct_builtin("=<_iso/2", 2, '=<_iso/2').
wam_go_direct_builtin('=<_lax'/2, 2, '=<_lax/2').
wam_go_direct_builtin('=<_lax/2', 2, '=<_lax/2').
wam_go_direct_builtin("=<_lax/2", 2, '=<_lax/2').
wam_go_direct_builtin('>=_iso'/2, 2, '>=_iso/2').
wam_go_direct_builtin('>=_iso/2', 2, '>=_iso/2').
wam_go_direct_builtin(">=_iso/2", 2, '>=_iso/2').
wam_go_direct_builtin('>=_lax'/2, 2, '>=_lax/2').
wam_go_direct_builtin('>=_lax/2', 2, '>=_lax/2').
wam_go_direct_builtin(">=_lax/2", 2, '>=_lax/2').
wam_go_direct_builtin('=:=_iso'/2, 2, '=:=_iso/2').
wam_go_direct_builtin('=:=_iso/2', 2, '=:=_iso/2').
wam_go_direct_builtin("=:=_iso/2", 2, '=:=_iso/2').
wam_go_direct_builtin('=:=_lax'/2, 2, '=:=_lax/2').
wam_go_direct_builtin('=:=_lax/2', 2, '=:=_lax/2').
wam_go_direct_builtin("=:=_lax/2", 2, '=:=_lax/2').
wam_go_direct_builtin('=\\=_iso'/2, 2, '=\\=_iso/2').
wam_go_direct_builtin('=\\=_iso/2', 2, '=\\=_iso/2').
wam_go_direct_builtin("=\\=_iso/2", 2, '=\\=_iso/2').
wam_go_direct_builtin('=\\=_lax'/2, 2, '=\\=_lax/2').
wam_go_direct_builtin('=\\=_lax/2', 2, '=\\=_lax/2').
wam_go_direct_builtin("=\\=_lax/2", 2, '=\\=_lax/2').

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
    escape_go_string(F, EscapedFunctor),
    format(atom(GoLiteral), '&GetStructure{Functor: "~w", Ai: ~w}', [EscapedFunctor, AiIdx]).
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
    escape_go_string(F, EscapedFunctor),
    format(atom(GoLiteral), '&PutStructure{Functor: "~w", Ai: ~w}', [EscapedFunctor, AiIdx]).
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
    wam_go_direct_builtin(P, N, Op),
    !,
    escape_go_string(Op, EscapedPred),
    format(atom(GoLiteral), '&BuiltinCall{Op: "~w", Arity: ~w}', [EscapedPred, N]).
wam_instruction_to_go_literal(call(P, N), GoLiteral) :-
    format(atom(GoLiteral), '&Call{Pred: "~w", Arity: ~w}', [P, N]).
wam_instruction_to_go_literal(call_indexed_atom_fact2(Pred), GoLiteral) :-
    escape_go_string(Pred, EscapedPred),
    format(atom(GoLiteral), '&CallIndexedAtomFact2{Pred: "~w"}', [EscapedPred]).
% `execute <builtin>` is a *last* call: the builtin must run and then
% return to the caller. Emitting a plain BuiltinCall here left the
% predicate falling off the end of its code (output arguments bound,
% predicate reported as failed) — see the BuiltinExecute doc comment in
% instructions.go.mustache.
wam_instruction_to_go_literal(execute(P), GoLiteral) :-
    wam_go_direct_builtin(P, N, Op),
    !,
    escape_go_string(Op, EscapedPred),
    format(atom(GoLiteral), '&BuiltinExecute{Op: "~w", Arity: ~w}', [EscapedPred, N]).
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
wam_instruction_to_go_literal(get_level(Yn), GoLiteral) :-
    go_reg_index(Yn, YnIdx),
    format(atom(GoLiteral), '&GetLevel{Reg: ~w}', [YnIdx]).
wam_instruction_to_go_literal(cut(Yn), GoLiteral) :-
    go_reg_index(Yn, YnIdx),
    format(atom(GoLiteral), '&Cut{Reg: ~w}', [YnIdx]).

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
    predicate_go_name(Pred, Arity, Options, CapPred),
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

compile_wam_predicate_to_go_shared(Pred/Arity, StartPC, Options, GoCode) :-
    atom_string(Pred, PredStr),
    predicate_go_name(Pred, Arity, Options, CapPred),
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
    go_foreign_setup_code(SetupOps, Options, Setup),
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

go_foreign_setup_code(Ops, Setup) :-
    go_foreign_setup_code(Ops, [], Setup).

go_foreign_setup_code([], _Options, "").
go_foreign_setup_code(Ops, Options, Setup) :-
    maplist([Op, L]>>go_foreign_setup_line(Op, Options, L), Ops, Lines),
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
go_foreign_setup_line(register_lmdb_atom_fact2(Pred/Arity, ArtifactDir), Line) :-
    go_lmdb_register_line(Pred/Arity, ArtifactDir, [], Line).
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

%% go_foreign_setup_line(+Op, +Options, -Line)
%  Options-aware form. Only the LMDB registration reads Options (for the
%  materialisation tier and cache size); every other op is
%  option-independent and falls through to go_foreign_setup_line/2.
go_foreign_setup_line(register_lmdb_atom_fact2(Pred/Arity, ArtifactDir), Options, Line) :-
    !,
    go_lmdb_register_line(Pred/Arity, ArtifactDir, Options, Line).
go_foreign_setup_line(Op, _Options, Line) :-
    go_foreign_setup_line(Op, Line).

%% go_lmdb_register_line(+Pred/Arity, +ArtifactDir, +Options, -Line)
%  Emit the registerLmdbAtomFact2 call with its resolved materialisation
%  tier and L2 capacity.
go_lmdb_register_line(Pred/Arity, ArtifactDir, Options, Line) :-
    escape_go_string(ArtifactDir, EscapedArtifactDir),
    go_lmdb_materialisation_mode(Options, Mode),
    go_lmdb_l2_capacity(Options, Mode, Capacity),
    format(atom(Line),
        '    if err := vm.registerLmdbAtomFact2("~w/~w", "~w", "~w", ~w); err != nil { panic(err) }',
        [Pred, Arity, EscapedArtifactDir, Mode, Capacity]).

%% go_lmdb_materialisation_mode(+Options, -Mode)
%  Resolve lmdb_materialisation(eager|lazy|cached|auto) to a concrete
%  tier. `auto` defers to the shared cost model in core/cost_model.pl —
%  the same rule F#, Haskell and R use — so a workload described once
%  resolves the same way across targets.
%
%  The default is `cached`: it matches the F# default and is a strict
%  improvement on the previous unconditional per-lookup helper spawn
%  (the artifact is read-only for the process lifetime, so memoising
%  lookups cannot change results). Pass lmdb_materialisation(lazy) to
%  get the old behaviour back.
go_lmdb_materialisation_mode(Options, Mode) :-
    option(lmdb_materialisation(Requested), Options, cached),
    (   Requested == auto
    ->  resolve_auto_lmdb_materialisation(Options, Mode)
    ;   memberchk(Requested, [eager, lazy, cached])
    ->  Mode = Requested
    ;   throw(error(domain_error(lmdb_materialisation, Requested), _))
    ).

%% go_lmdb_l2_capacity(+Options, +Mode, -Capacity)
%  Resolve lmdb_l2_capacity(N|auto). Only the cached tier has a cache to
%  size, so eager/lazy report 0. `auto` scales with the demand-set
%  estimate, clamped to [256, 65536] — same shape as
%  compute_l2_capacity_fs/2.
go_lmdb_l2_capacity(_Options, Mode, 0) :-
    Mode \== cached,
    !.
go_lmdb_l2_capacity(Options, _Mode, Capacity) :-
    option(lmdb_l2_capacity(Requested), Options, 4096),
    (   Requested == auto
    ->  option(fact_count(F), Options, 0),
        option(demand_set_estimate(D), Options, F),
        Capacity is max(256, min(65536, integer(D * 0.1)))
    ;   integer(Requested), Requested > 0
    ->  Capacity = Requested
    ;   throw(error(domain_error(lmdb_l2_capacity_positive_integer, Requested), _))
    ).

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

% ============================================================================
% ISO ERROR HANDLING (three-form dispatch)
% ============================================================================
%
% Go joins C++/Elixir/Python/F#/Haskell as an ISO three-form adopter.
% The shared config loading, mode resolution, PI matching and audit
% walkers live in src/unifyweaver/core/iso_errors.pl; what stays here is
% the Go key tables, the WAM-text rewrite, and the audit report.
%
% Contract: docs/design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md.
%
% Three forms per builtin:
%   is/2      default key; rewritten per predicate to _iso or _lax
%   is_iso/2  throws error(instantiation_error, _),
%             error(type_error(evaluable, F/N), _), or
%             error(evaluation_error(zero_divisor), _)
%   is_lax/2  the historical silent-failure behaviour
%
% Explicit _iso/_lax keys written in source survive a mode flip: the
% rewrite only ever touches the *default* key.
%
% Only keys with a matching branch in the emitted Go runtime appear
% here -- an entry without one would silently route calls to a dead key.

iso_errors:iso_errors_default_to_iso("is/2", "is_iso/2").
iso_errors:iso_errors_default_to_iso(">/2", ">_iso/2").
iso_errors:iso_errors_default_to_iso("</2", "<_iso/2").
iso_errors:iso_errors_default_to_iso(">=/2", ">=_iso/2").
iso_errors:iso_errors_default_to_iso("=</2", "=<_iso/2").
iso_errors:iso_errors_default_to_iso("=:=/2", "=:=_iso/2").
iso_errors:iso_errors_default_to_iso("=\\=/2", "=\\=_iso/2").
iso_errors:iso_errors_default_to_iso("succ/2", "succ_iso/2").

iso_errors:iso_errors_default_to_lax("is/2", "is_lax/2").
iso_errors:iso_errors_default_to_lax(">/2", ">_lax/2").
iso_errors:iso_errors_default_to_lax("</2", "<_lax/2").
iso_errors:iso_errors_default_to_lax(">=/2", ">=_lax/2").
iso_errors:iso_errors_default_to_lax("=</2", "=<_lax/2").
iso_errors:iso_errors_default_to_lax("=:=/2", "=:=_lax/2").
iso_errors:iso_errors_default_to_lax("=\\=/2", "=\\=_lax/2").
iso_errors:iso_errors_default_to_lax("succ/2", "succ_lax/2").

%% iso_errors_rewrite_text(+Config, +PI, +WamText, -RewrittenText)
%  Rewrite the default builtin keys in one predicate's WAM text to the
%  ISO or lax key selected for that predicate.
%
%  Same strategy as the Python target: tokenize each line with the
%  shared quote-aware tokenizer, recognise it to a structured item,
%  apply the shared item-level rewrite, then splice the new key back
%  into the original line so whitespace is preserved byte-for-byte.
iso_errors_rewrite_text(Config, PI, WamText, RewrittenText) :-
    iso_errors_mode_for(Config, PI, Mode),
    (   Mode == false,
        \+ go_iso_has_lax_entries
    ->  RewrittenText = WamText
    ;   atom_string(WamText, S),
        split_string(S, "\n", "", Lines),
        maplist(go_iso_rewrite_line(Mode), Lines, RewrittenLines),
        atomic_list_concat(RewrittenLines, '\n', RewrittenText)
    ).

go_iso_has_lax_entries :- iso_errors:iso_errors_default_to_lax(_, _), !.

go_iso_rewrite_line(Mode, Line, OutLine) :-
    (   wam_text_parser:wam_tokenize_line(Line, Tokens),
        wam_recognise_instruction(Tokens, Item0),
        iso_errors_rewrite_item(Mode, Item0, Item1),
        Item0 \== Item1,
        arg(1, Item0, OldKey),
        arg(1, Item1, NewKey),
        go_iso_splice_line(Line, OldKey, NewKey, OutLine)
    ->  true
    ;   OutLine = Line
    ).

go_iso_splice_line(Line, Key, NewKey, OutLine) :-
    sub_string(Line, Before, KLen, _After, Key),
    string_length(Key, KLen),
    sub_string(Line, 0, Before, _, Pre),
    PostStart is Before + KLen,
    sub_string(Line, PostStart, _, 0, Post),
    string_concat(Pre, NewKey, T1),
    string_concat(T1, Post, OutLine), !.

go_iso_clean_key_token(Token0, Token) :-
    (   string_concat(Token, ",", Token0)
    ->  true
    ;   Token = Token0
    ).

%% wam_go_iso_audit(+Predicates, +Options, -Audit)
%  Read-only pass reporting, per predicate, which mode applies and what
%  each builtin_call site resolves to. Mirrors wam_python_iso_audit/3.
wam_go_iso_audit(Predicates, Options, Audit) :-
    iso_errors_resolve_options(Options, Config),
    findall(audit(PI, Mode, Sites), (
        member(P, Predicates),
        iso_errors_audit_normalise_pi(P, PI),
        iso_errors_mode_for(Config, PI, Mode),
        go_iso_audit_predicate(PI, Mode, Sites)
    ), Audit).

go_iso_audit_predicate(PI, Mode, Sites) :-
    (   catch(
            ( go_iso_audit_wam_for_pi(PI, WamText),
              go_iso_audit_parse_lines(WamText, Items)
            ),
            _, fail)
    ->  iso_errors_audit_walk(Items, 0, Mode, [], SitesRev),
        reverse(SitesRev, Sites)
    ;   Sites = []
    ).

go_iso_audit_wam_for_pi(Module:Pred/Arity, WamText) :- !,
    wam_target:compile_predicate_to_wam(Module:Pred/Arity, [], WamText).
go_iso_audit_wam_for_pi(Pred/Arity, WamText) :-
    wam_target:compile_predicate_to_wam(Pred/Arity, [], WamText).

go_iso_audit_parse_lines(WamText, Items) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    maplist(go_iso_audit_parse_one, Lines, MaybeItems),
    exclude(==(skip), MaybeItems, Items).

go_iso_audit_parse_one(Line, Item) :-
    split_string(Line, " \t", " \t", Parts0),
    exclude(==(""), Parts0, Parts),
    go_iso_audit_classify_line(Parts, Item).

go_iso_audit_classify_line([], skip).
go_iso_audit_classify_line([Tok], label) :-
    string_concat(_, ":", Tok), !.
go_iso_audit_classify_line(["builtin_call", Key0 | _], builtin_call(Key, 0)) :- !,
    go_iso_clean_key_token(Key0, Key).
go_iso_audit_classify_line(_, other).

%% wam_go_iso_audit_report(+Audit)
wam_go_iso_audit_report([]).
wam_go_iso_audit_report([audit(PI, Mode, Sites)|Rest]) :-
    format('~w [~w]~n', [PI, Mode]),
    (   Sites == []
    ->  format('  (no builtin_call sites)~n', [])
    ;   forall(member(site(PC, Orig, Res, Src, Flip), Sites),
               format('  pc=~w  ~w -> ~w  (~w)  flip-changes=~w~n',
                      [PC, Orig, Res, Src, Flip]))
    ),
    wam_go_iso_audit_report(Rest).

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

%% predicate_go_name(+Pred, +Arity, +Options, -Name)
%  Arity-aware wrapper name. Go has no overloading, so a project that
%  contains the *same* predicate name at two arities (e.g. the portable
%  parser's `read_term_from_atom/2` and `/3`) would emit two `func
%  Read_term_from_atom` declarations and fail to compile with
%  "redeclared in this block".
%
%  Names are only suffixed with the arity when the project actually
%  carries that name at more than one arity — the overloaded set is
%  computed once per project in compile_predicates_for_project/3 and
%  threaded through as `go_overloaded_names(Names)`. Projects without
%  overloads keep the historical unsuffixed names, so existing
%  generated code and its call sites are unchanged.
predicate_go_name(Pred, Arity, Options, Name) :-
    predicate_go_name(Pred, Base),
    option(go_overloaded_names(Overloaded), Options, []),
    (   memberchk(Pred, Overloaded)
    ->  atom_concat(Base, Arity, Name)
    ;   Name = Base
    ).

%% go_overloaded_predicate_names(+Predicates, -Names)
%  Sorted set of predicate names that appear at two or more distinct
%  arities in this project. These are the names that need arity
%  suffixes in the emitted Go identifiers.
go_overloaded_predicate_names(Predicates, Names) :-
    findall(Pred-Arity,
            ( member(PredIndicator, Predicates),
              predicate_indicator_parts(PredIndicator, _M, Pred, Arity)
            ),
            Pairs0),
    sort(Pairs0, Pairs),
    findall(Pred,
            ( member(Pred-A1, Pairs),
              member(Pred-A2, Pairs),
              A1 \== A2
            ),
            Dups0),
    sort(Dups0, Names).

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
    findall(row(Left, Right, Weight),
        ( functor(WeightHead, WeightPred, 3),
          Module:clause(WeightHead, true),
          WeightHead =.. [WeightPred, Left, Right, Weight]
        ),
        Rows),
    go_validate_weighted_rows(WeightPred, Rows, FactTriples).

go_validate_weighted_rows(_, [], []).
go_validate_weighted_rows(WeightPred, [row(Left, Right, Weight0)|Rows], FactTriples) :-
    (   atom(Left)
    ->  (   atom(Right), number(Weight0)
        ->  Weight is float(Weight0),
            FactTriples = [Left-Right-Weight|Rest]
        ;   Fact =.. [WeightPred, Left, Right, Weight0],
            format(atom(Message),
                   'native weighted relation ~w requires atom/atom/number rows',
                   [WeightPred]),
            throw(error(domain_error(weighted_kernel_fact, Fact),
                        context(wam_go_target:go_weighted_edge_fact_triples/3,
                                Message)))
        )
    ;   % Non-atom sources cannot be reached from the atom-keyed native ABI.
        FactTriples = Rest
    ),
    go_validate_weighted_rows(WeightPred, Rows, Rest).

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
    go_configured_astar_direct_pred(Module, DirectPred),
    go_weighted_edge_fact_triples(Module, DirectPred/3, DirectTriples),
    !.
go_foreign_astar_direct_pred(Module, _FallbackPred, DirectPred/3, DirectTriples) :-
    % Backward compatibility for workloads that predate direct_dist_pred/1.
    go_weighted_edge_fact_triples(Module, direct_semantic_dist/3, DirectTriples),
    DirectTriples \= [],
    DirectPred = direct_semantic_dist,
    !.
go_foreign_astar_direct_pred(_Module, FallbackPred, DirectPred, DirectTriples) :-
    FallbackPred = DirectPred,
    DirectTriples = [].

go_configured_astar_direct_pred(Module, DirectPred) :-
    (   current_predicate(Module:direct_dist_pred/1),
        Module:direct_dist_pred(Spec)
    ;   Module \== user,
        current_predicate(user:direct_dist_pred/1),
        user:direct_dist_pred(Spec)
    ),
    (   Spec = DirectPred/3
    ;   atom(Spec), DirectPred = Spec
    ),
    atom(DirectPred),
    !.

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
    wam_tokenize_line(Line, CleanParts),
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
            (   sub_atom(GoLit, 0, 2, _, '//')
            ->  % Non-instruction placeholder: some unimplemented ops (e.g.
                % arg/3 in certain modes) emit a `// TODO ...` comment instead
                % of a real instruction. Go ignores comment lines, so they
                % occupy NO slot in the runtime instruction array. They must
                % therefore not advance PC or be counted in InstrCount —
                % otherwise every predicate emitted after them gets a StartPC
                % shifted by the number of comments, so it is entered past its
                % prologue (e.g. skipping the put_variable that allocates a
                % findall template), silently corrupting execution. Skip the
                % comment entirely so PC accounting matches the emitted array.
                wam_lines_to_go(Rest, PC, PredIndicator, Options, GoLits, Labels)
            ;   GoLits = [GoLit | RestLits],
                NPC is PC + 1,
                wam_lines_to_go(Rest, NPC, PredIndicator, Options, RestLits, Labels)
            )
        )
    ).

wam_tokenize_line(Line, Parts) :-
    string_chars(Line, Chars),
    wam_tokenize_chars(Chars, [], RevParts),
    reverse(RevParts, Parts).

wam_tokenize_chars([], Parts, Parts).
wam_tokenize_chars([C|Rest], Parts0, Parts) :-
    char_type(C, space),
    !,
    wam_tokenize_chars(Rest, Parts0, Parts).
wam_tokenize_chars(['\''|Rest], Parts0, Parts) :-
    !,
    wam_take_quoted(Rest, Token, Remaining),
    wam_tokenize_chars(Remaining, [Token|Parts0], Parts).
wam_tokenize_chars(Chars, Parts0, Parts) :-
    wam_take_bare(Chars, Token, Remaining),
    (   Token == ""
    ->  Parts1 = Parts0
    ;   Parts1 = [Token|Parts0]
    ),
    wam_tokenize_chars(Remaining, Parts1, Parts).

wam_take_quoted(Rest, Token, Remaining) :-
    wam_take_quoted_inner(Rest, ['\''], RevQuoted, AfterQuote),
    reverse(RevQuoted, QuotedChars),
    wam_take_bare_tail(AfterQuote, TailChars, Remaining),
    append(QuotedChars, TailChars, TokenChars),
    string_chars(Token, TokenChars).

wam_take_quoted_inner([], Acc, Acc, []).
wam_take_quoted_inner(['\\', C|Rest], Acc, RevQuoted, Remaining) :-
    !,
    wam_take_quoted_inner(Rest, [C, '\\'|Acc], RevQuoted, Remaining).
wam_take_quoted_inner(['\''|Rest], Acc, ['\''|Acc], Rest) :-
    !.
wam_take_quoted_inner([C|Rest], Acc, RevQuoted, Remaining) :-
    wam_take_quoted_inner(Rest, [C|Acc], RevQuoted, Remaining).

wam_take_bare([], "", []).
wam_take_bare(Chars, Token, Remaining) :-
    wam_take_bare_chars(Chars, TokenChars, Remaining),
    string_chars(Token, TokenChars).

wam_take_bare_chars([], [], []).
wam_take_bare_chars([C|Rest], [], [C|Rest]) :-
    char_type(C, space),
    !.
wam_take_bare_chars([C|Rest], [C|TokenRest], Remaining) :-
    wam_take_bare_chars(Rest, TokenRest, Remaining).

wam_take_bare_tail([], [], []).
wam_take_bare_tail([C|Rest], [], [C|Rest]) :-
    char_type(C, space),
    !.
wam_take_bare_tail([C|Rest], [C|Tail], Remaining) :-
    wam_take_bare_tail(Rest, Tail, Remaining).

%% wam_line_to_go_literal(+Parts, -GoLit)
%% parse_string_to_go_val(+Str, -GoVal)
%
% When the WAM compiler emits an atom that needs Prolog quoting (e.g.
% an apostrophe-bearing category like 'People\'s_Republic_of_China'),
% the WAM TEXT carries the quoted form. The split-on-whitespace parser
% used to break quoted constants containing whitespace into multiple tokens.
% `wam_tokenize_line/2` now keeps that whole token here including the
% surrounding ' and the backslash-escaped inner quote. If we just pass it to
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
    ;   % A double-quoted string constant round-trips to a Prolog string.
        % Go has no string type (D37: double-quoted literals intern as
        % atoms), so emit the CONTENT as an atom rather than a token that
        % keeps its surrounding quotes. Without this, resolver_store.pl's
        % `VVS == "-"` compiled to an atom named `"-"` (quotes included),
        % never matching the atom `-` that split_string/4 yields.
        atom_string(QuotedTok, Str),
        catch(term_to_atom(ParsedTerm, QuotedTok), _, fail),
        string(ParsedTerm)
    ->  atom_string(ContentAtom, ParsedTerm),
        go_value_literal(ContentAtom, GoVal)
    ;   go_value_literal(Str, GoVal)
    ).

wam_line_to_go_literal(["call", P, N], PredIndicator, Options, GoLit) :-
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    (   go_foreign_rewrite_call(Options, PredIndicator, CP, Num, ForeignPred, ForeignArity)
    ->  format(atom(GoLit), '&CallForeign{Pred: "~w", Arity: ~w}', [ForeignPred, ForeignArity])
    ;   wam_go_direct_builtin(CP, Num, Op)
    ->  escape_go_string(Op, EscapedPred),
        format(atom(GoLit), '&BuiltinCall{Op: "~w", Arity: ~w}', [EscapedPred, CN])
    ;   format(atom(GoLit), '&Call{Pred: "~w", Arity: ~w}', [CP, CN])
    ).
wam_line_to_go_literal(["call_indexed_atom_fact2", Pred], _PredIndicator, _Options, GoLit) :-
    clean_comma(Pred, CPred),
    wam_instruction_to_go_literal(call_indexed_atom_fact2(CPred), GoLit).
wam_line_to_go_literal(["call_fact_stream", Pred], _PredIndicator, _Options, GoLit) :-
    clean_comma(Pred, CPred),
    escape_go_string(CPred, EscPred),
    format(atom(GoLit), '&CallFactStream{Pred: "~w"}', [EscPred]).
wam_line_to_go_literal(["execute", P], PredIndicator, Options, GoLit) :-
    clean_comma(P, CP),
    (   go_foreign_rewrite_execute(Options, PredIndicator, CP, ForeignPred, ForeignArity)
    ->  format(atom(GoLit), '&CallForeign{Pred: "~w", Arity: ~w}', [ForeignPred, ForeignArity])
    ;   wam_go_direct_builtin(CP, Num, Op)
    ->  escape_go_string(Op, EscapedPred),
        format(atom(GoLit), '&BuiltinExecute{Op: "~w", Arity: ~w}', [EscapedPred, Num])
    ;   format(atom(GoLit), '&Execute{Pred: "~w"}', [CP])
    ).
wam_line_to_go_literal(["jump", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&Jump{Label: "~w"}', [CL]).
wam_line_to_go_literal(["cut_ite"], '&CutIte{}').
wam_line_to_go_literal(["get_level", Yn], GoLit) :-
    clean_comma(Yn, CYn),
    go_reg_index(CYn, YnIdx),
    format(atom(GoLit), '&GetLevel{Reg: ~w}', [YnIdx]).
wam_line_to_go_literal(["cut", Yn], GoLit) :-
    clean_comma(Yn, CYn),
    go_reg_index(CYn, YnIdx),
    format(atom(GoLit), '&Cut{Reg: ~w}', [YnIdx]).
wam_line_to_go_literal(["begin_aggregate", AggType, ValueReg, ResultReg], GoLit) :-
    clean_comma(AggType, CAggType),
    clean_comma(ValueReg, CValueReg),
    clean_comma(ResultReg, CResultReg),
    go_reg_index(CValueReg, ValueRegIdx),
    go_reg_index(CResultReg, ResultRegIdx),
    format(atom(GoLit), '&BeginAggregate{AggType: "~w", ValueReg: ~w, ResultReg: ~w}',
        [CAggType, ValueRegIdx, ResultRegIdx]).
% bagof/setof emit a 4th witness-register argument
% (`begin_aggregate bagof, Y1, X2, ''`). Dropping it as `// TODO`
% skipped the instruction, so EndAggregate was a no-op and L stayed
% unbound (`__R200`). Witness grouping is not implemented; the 3-arg
% runtime path is correct when the witness list is empty.
wam_line_to_go_literal(["begin_aggregate", AggType, ValueReg, ResultReg, _Witnesses], GoLit) :-
    wam_line_to_go_literal(["begin_aggregate", AggType, ValueReg, ResultReg], GoLit).
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
wam_line_to_go_literal(["arg", N, Src, Dest], GoLit) :-
    % arg N, Src, Dest — extract argument N (1-based) of the term in
    % register Src into register Dest. Previously unhandled: it fell
    % through to the `// TODO` catch-all (a silent no-op), which is what
    % left findall templates unbound after copy_term. Lower it to a real
    % GetArgInto instruction.
    clean_comma(N, CN), clean_comma(Src, CSrc), clean_comma(Dest, CDest),
    go_reg_index(CSrc, SrcIdx), go_reg_index(CDest, DestIdx),
    format(atom(GoLit), '&GetArgInto{Index: ~w, Src: ~w, Dest: ~w}', [CN, SrcIdx, DestIdx]).
wam_line_to_go_literal(["get_structure", FN, Ai], GoLit) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    go_reg_index(CAi, AiIdx),
    escape_go_string(CFN, EscapedFunctor),
    format(atom(GoLit), '&GetStructure{Functor: "~w", Ai: ~w}', [EscapedFunctor, AiIdx]).
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
    escape_go_string(CFN, EscapedFunctor),
    format(atom(GoLit), '&PutStructure{Functor: "~w", Ai: ~w}', [EscapedFunctor, AiIdx]).
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
wam_line_to_go_literal(["try", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&Try{Label: "~w"}', [CL]).
wam_line_to_go_literal(["retry", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&Retry{Label: "~w"}', [CL]).
wam_line_to_go_literal(["trust", L], GoLit) :-
    clean_comma(L, CL),
    format(atom(GoLit), '&Trust{Label: "~w"}', [CL]).

wam_line_to_go_literal(["switch_on_constant" | Table], GoLit) :-
    format_switch_table(Table, CaseStr),
    format(atom(GoLit), '&SwitchOnConstant{Cases: []ConstCase{~w}}', [CaseStr]).
wam_line_to_go_literal(["switch_on_structure" | Table], GoLit) :-
    format_struct_switch_table(Table, CaseStr),
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

% switch_on_structure table entries are Functor/Arity:Label. StructCase
% carries a functor STRING (the runtime compares fmt.Sprintf("%s/%d",
% f, len(args))), not a Value. Reusing format_switch_case interned the
% functor as an atom and emitted `{Val: wamAtom_...}`, which does not
% compile (StructCase has Functor, not Val). Forced by catalog/6 vs
% catalog/9 first-arg indexing in uw-resolve.
format_struct_switch_table(Table, CaseStr) :-
    maplist(format_struct_switch_case, Table, Cases),
    atomic_list_concat(Cases, ", ", CaseStr).

format_struct_switch_case(Entry, Case) :-
    (   split_string(Entry, ":", " ", [FunStr, LabelStr])
    ->  clean_comma(LabelStr, Label),
        clean_comma(FunStr, Fun),
        escape_go_string(Fun, EscFun),
        format(atom(Case), '{Functor: "~w", Label: "~w"}', [EscFun, Label])
    ;   format(atom(Case), '{Functor: "malformed", Label: "~w"}', [Entry])
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
%  Generates the Step() method body as a Go type switch. The 50 case bodies now
%  live in five {{match instr}} library files under
%  templates/targets/go_wam/step/ and the header/footer in step_shell.go.mustache;
%  wam_go_templates:go_render_step_method/1 owns the assembly (the switch order
%  via go_step_case_order/1, the `case *Type:` line, the shell splice). See
%  docs/proposals/wam_go_template_refactor_design.md, Phase 3.
compile_step_wam_to_go(_Options, GoCode) :-
    go_render_step_method(GoCode).

% ============================================================================
% PHASE 3b: Helper functions → Go
% ============================================================================

%% compile_wam_helpers_to_go(+Options, -GoCode)
%  The static WAM helper methods. The Go source now lives in
%  templates/targets/go_wam/runtime/helpers.go.mustache and is spliced by
%  wam_go_templates:go_render_helper_methods/1 (design
%  docs/proposals/wam_go_template_refactor_design.md, Phase 1). The section
%  loader returns the blob without its final LF; this wrapper re-appends it so
%  the predicate keeps its historical trailing-LF contract (design Q1(b)),
%  leaving content-grepping callers and compile_wam_runtime_to_go/2 unchanged.
compile_wam_helpers_to_go(_Options, GoCode) :-
    go_render_helper_methods(Helpers),
    atom_concat(Helpers, '\n', GoCode).

%% escape_go_string(+Atom, -Escaped)
%  Escapes an atom/string for a Go double-quoted string literal. Backslash
%  MUST be escaped first (so the backslashes added for the later cases are
%  not doubled). Double-quote handling is required now that resolver_store.pl
%  interns a `"`-bearing atom; newline/tab/CR are covered defensively since a
%  Go string literal cannot carry a raw control character.
escape_go_string(Atom, Escaped) :-
    atom_string(Atom, Str0),
    go_str_replace(Str0, "\\", "\\\\", S1),
    go_str_replace(S1, "\"", "\\\"", S2),
    go_str_replace(S2, "\n", "\\n", S3),
    go_str_replace(S3, "\t", "\\t", S4),
    go_str_replace(S4, "\r", "\\r", Escaped).

go_str_replace(In, From, To, Out) :-
    split_string(In, From, "", Parts),
    atomic_list_concat(Parts, To, Out).

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
        format(atom(RuntimeHelper),
'
// atomInternMap is the single source of pointer identity for atoms.
// internAtom(name) returns the shared pointer for that name, allocating
// and caching one if the name is new. The wamAtom_ vars above are
// themselves initialised through internAtom, so they populate this map
// as a side effect of package initialisation.
//
// Deliberately NOT an init() that assigns into the map: Go runs every
// package-level variable initialiser before any init(), so a var in
// another file of this package that calls internAtom during its own
// initialisation (state.go''s emptyListAtom) would win the map slot and
// then be overwritten here, leaving two atoms with the same name and
// different pointers. Atom.Equals is pointer-only, so that silently
// broke equality — `get_constant []` against a real empty list.
//
// Bench drivers should construct atoms via internAtom rather than
// `&Atom{Name: x}` for the same reason: SwitchOnConstant in the WAM
// bytecode matches in O(1) on pointer identity.
var atomInternMap = make(map[string]*Atom)

func internAtom(name string) *Atom {
    if a, ok := atomInternMap[name]; ok {
        return a
    }
    a := &Atom{Name: name}
    atomInternMap[name] = a
    return a
}

// InternAtom is the exported form. Drivers and embedders must build
// atoms through this rather than &Atom{Name: x}: Atom.Equals is pointer
// identity only, so a fresh literal with the same name compares unequal
// to the interned one the bytecode carries.
func InternAtom(name string) *Atom {
    return internAtom(name)
}
', []),
        atomic_list_concat([DeclBlock, RuntimeHelper], GoCode)
    ).

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

%% format_atom_decl(+VarName-Str, -Line)
%  Each interned atom is initialised *through* internAtom rather than as
%  a bare &Atom literal.
%
%  Atom.Equals is pointer-only, so every atom with a given name has to be
%  the one object in atomInternMap. Registering the vars from an init()
%  did not guarantee that: Go runs all package-level variable
%  initialisers before any init(), so a var elsewhere in the package that
%  calls internAtom during its own initialisation — state.go's
%  `var emptyListAtom = internAtom("[]")` — created and cached its own
%  "[]" atom first, and the init() then *overwrote* the map entry with
%  the codegen's wamAtom_ var. The result was two distinct "[]" atoms:
%  one reached through emptyListAtom (list tails, get_list on a
%  one-element list) and one baked into the bytecode as a get_constant
%  operand. `get_constant []` then failed against a genuinely empty
%  list, which is what stopped the compiled portable parser dead at
%  parse_op_loop/10's base clause.
%
%  Going through internAtom makes the var block order-independent:
%  Go's initialisation-dependency analysis runs atomInternMap first
%  (internAtom references it), and whichever caller asks for a name
%  first, everyone gets the same pointer.
format_atom_decl(VarName-Str, Line) :-
    escape_go_string(Str, Escaped),
    format(atom(Line), "    ~w = internAtom(\"~w\")", [VarName, Escaped]).

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
