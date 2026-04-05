:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% prolog_target.pl - Prolog as Compilation Target
% Transpiles user Prolog code to standalone executable Prolog scripts
%
% Philosophy:
% - This is a GENERAL transpilation target, not partitioning-specific
% - User writes Prolog predicates → UnifyWeaver generates executable .pl script
% - Generated script includes runtime dependencies (partitioner, data sources, etc.)
% - Script is standalone: #!/usr/bin/env swipl with initialization
%
% Use Cases:
% - Partitioning/parallel execution (leverages Prolog's data structure strengths)
% - Data transformation pipelines (complex logic hard to express in bash)
% - Prolog predicates that need runtime library support
% - Quick prototyping (faster than bash transpilation)

:- module(prolog_target, [
    generate_prolog_script/3,     % +UserPredicates, +Options, -ScriptCode
    analyze_dependencies/2,       % +UserPredicates, -RequiredModules
    write_prolog_script/2,        % +ScriptCode, +OutputPath
    write_prolog_script/3         % +ScriptCode, +OutputPath, +Options
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(option)).
:- use_module(library(debug)).
:- use_module(prolog_dialects).
:- use_module(prolog_constraints).
:- use_module('../core/constraint_analyzer', [get_constraints/2]).
:- use_module('../core/advanced/pattern_matchers', [is_per_path_visited_pattern/4]).

%% ============================================
%% MAIN ENTRY POINT
%% ============================================

%% generate_prolog_script(+UserPredicates, +Options, -ScriptCode)
%  Generate complete Prolog script from user predicates
%
%  @arg UserPredicates List of predicate indicators (Pred/Arity)
%  @arg Options Compilation options:
%       - dialect(Dialect) - Target Prolog dialect (swi, gnu) default: swi
%       - entry_point(Goal) - Main goal to execute (default: main)
%       - inline_runtime(true/false) - Inline runtime library vs reference
%       - arguments(ArgSpec) - Command-line argument parsing
%       - output_mode(Mode) - stdout, file, return
%       - compile(true/false) - Generate compilation command for dialect
%       - branch_pruning(auto/false) - Enable or disable SWI PPV pruning
%         helpers (default: auto). Use false when benchmarking the
%         non-pruned baseline or when comparing generated variants.
%       - min_closure(auto/false) - Emit a SWI mode-directed min helper
%         for canonical counted or positive-weighted PPV recursion when
%         available (default: auto). This preserves the original
%         predicate and adds a separate '$min' helper for bound
%         shortest-path style queries.
%       - seeded_accumulation(auto/false) - Emit SWI helper predicates
%         for canonical counted recursive closures that aggregate over
%         the derived metric (default: auto). This currently supports
%         `power_sum` and `sum_metric` helper formulas, emits a
%         query-aware `*_selected` helper surface, and may emit an
%         additional grouped `power_sum_grouped` helper when the
%         canonical `(seed, key, metric)` shape is recognized.
%       - effective_distance_accumulation(auto/false) - Legacy alias for
%         seeded_accumulation/1 retained for compatibility with the
%         effective-distance benchmark surface.
%  @arg ScriptCode Generated script as atom
%
%  @example Generate script for user predicates
%    ?- generate_prolog_script([process_data/2, helper/1], [dialect(gnu)], Code).
generate_prolog_script(UserPredicates, Options, ScriptCode) :-
    % 0. Determine target dialect
    option(dialect(Dialect), Options, swi),

    % Validate dialect support
    (   supported_dialect(Dialect)
    ->  true
    ;   format(atom(Error), 'Unsupported dialect: ~w', [Dialect]),
        throw(error(unsupported_dialect(Dialect), Error))
    ),

    % Validate predicates for dialect
    validate_for_dialect(Dialect, UserPredicates, Issues),
    (   Issues = []
    ->  true
    ;   format('[Warning] Compatibility issues for ~w: ~w~n', [Dialect, Issues])
    ),

    % 1. Analyze what user code needs
    EnhancedOptions = [predicates(UserPredicates)|Options],
    analyze_dependencies(UserPredicates, Dependencies0),
    augment_generated_dependencies(UserPredicates, EnhancedOptions, Dependencies0, Dependencies),

    % 2. Generate dialect-specific script components
    dialect_shebang(Dialect, ShebangCode),

    dialect_header(Dialect, EnhancedOptions, HeaderCode),

    dialect_imports(Dialect, Dependencies, ImportsCode),

    generate_user_code(UserPredicates, Options, UserCode),

    option(entry_point(EntryGoal), Options, main),
    dialect_initialization(Dialect, EntryGoal, Options, InitCode),

    generate_main_predicate(Options, MainCode),

    % 3. Assemble complete script
    atomic_list_concat([
        ShebangCode,
        HeaderCode,
        ImportsCode,
        UserCode,
        '\n% === Entry Point ===',
        MainCode,
        InitCode
    ], '\n\n', ScriptCode).

%% ============================================
%% SCRIPT COMPONENT GENERATION
%% ============================================

%% generate_shebang(-Code)
%  Generate shebang line for executable script
generate_shebang('#!/usr/bin/env swipl').

%% generate_header(+UserPredicates, +Options, -Code)
%  Generate script header with metadata
generate_header(UserPredicates, Options, Code) :-
    get_time(Timestamp),
    format_time(atom(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    % Extract source file if provided
    (   member(source_file(SourceFile), Options)
    ->  format(atom(SourceComment), '% Source: ~w', [SourceFile])
    ;   SourceComment = ''
    ),

    % Format predicate list
    length(UserPredicates, NumPreds),
    format(atom(PredsStr), '~w', [UserPredicates]),

    % Build lines
    format(atom(DateLine), '% Generated: ~w', [DateStr]),
    format(atom(NumPredsLine), '% Predicates transpiled: ~w', [NumPreds]),
    format(atom(PredsLine), '% ~w', [PredsStr]),

    % Combine
    (   SourceComment = ''
    ->  Lines = ['% Generated by UnifyWeaver v0.0.3',
                 '% Target: Prolog',
                 DateLine,
                 NumPredsLine,
                 PredsLine]
    ;   Lines = ['% Generated by UnifyWeaver v0.0.3',
                 '% Target: Prolog',
                 DateLine,
                 SourceComment,
                 NumPredsLine,
                 PredsLine]
    ),
    atomic_list_concat(Lines, '\n', Code).

%% generate_module_imports(+Dependencies, +Options, -Code)
%  Generate module import statements
%
%  @arg Dependencies List of required modules/features
%  @arg Options Compilation options (inline_runtime, etc.)
%  @arg Code Generated import code
generate_module_imports(Dependencies, Options, Code) :-
    % Check if runtime should be inlined or referenced
    (   member(inline_runtime(true), Options)
    ->  % Will inline later, no imports needed
        Code = '% Runtime library inlined below'
    ;   % Reference runtime library via imports
        generate_import_statements(Dependencies, ImportStatements),
        atomic_list_concat(ImportStatements, '\n', Code)
    ).

%% generate_import_statements(+Dependencies, -Statements)
%  Convert dependency list to import statements
generate_import_statements(Dependencies, Statements) :-
    % Add search path setup - use multifile to make it work before module loads
    SearchPathSetup = [
        '% Set up UnifyWeaver runtime library search path',
        ':- multifile user:file_search_path/2.',
        ':- dynamic user:file_search_path/2.',
        '',
        '% Check for UNIFYWEAVER_HOME environment variable',
        'setup_unifyweaver_path :- getenv(\'UNIFYWEAVER_HOME\', Home), !, asserta(user:file_search_path(unifyweaver, Home)).',
        'setup_unifyweaver_path.  % Assume installed as pack if UNIFYWEAVER_HOME not set',
        '',
        ':- setup_unifyweaver_path.'
    ],

    % Convert dependencies to use_module statements
    findall(Import, (
        member(Dep, Dependencies),
        dependency_to_import(Dep, Import)
    ), Imports),

    % Combine
    append(SearchPathSetup, [''] , HeaderWithSpace),
    append(HeaderWithSpace, Imports, Statements).

%% dependency_to_import(+Dependency, -ImportStatement)
%  Convert dependency term to use_module statement
dependency_to_import(module(ModulePath), Import) :-
    format(atom(Import), ':- use_module(~w).', [ModulePath]).

dependency_to_import(ensure_loaded(ModulePath), Import) :-
    format(atom(Import), ':- ensure_loaded(~w).', [ModulePath]).

dependency_to_import(plugin_registration(Type, Name, Module), Import) :-
    format(atom(Import), ':- register_~w(~w, ~w).', [Type, Name, Module]).

%% generate_user_code(+UserPredicates, +Options, -Code)
%  Generate user's predicates (copied or templated)
%
%  Strategy:
%  - Simple predicates: Copy verbatim using clause/2
%  - Complex predicates with config: Use templates
%  - Detect which strategy per predicate
generate_user_code(UserPredicates, Options, Code) :-
    format(atom(Header), '% === User Code (Transpiled) ===~n', []),

    findall(PredCode, (
        member(Pred/Arity, UserPredicates),
        generate_predicate_code(Pred/Arity, Options, PredCode)
    ), PredCodes),

    atomic_list_concat([Header|PredCodes], '\n\n', Code).

%% generate_predicate_code(+Pred/Arity, +Options, -Code)
%  Generate code for a single predicate with constraint handling
generate_predicate_code(Pred/Arity, Options, Code) :-
    % Get dialect for constraint handling
    option(dialect(Dialect), Options, swi),

    % Copy predicate clauses verbatim, or generate an optimized worker when
    % the source matches a parameterized per-path recursion shape.
    (   maybe_generate_branch_pruned_predicate(Pred/Arity, Options, OptimizedCode)
    ->  BaseCode0 = OptimizedCode
    ;   copy_predicate_clauses(Pred/Arity, BaseCode0)
    ),
    maybe_append_min_closure_helper(Pred/Arity, Options, BaseCode0, BaseCode1),
    maybe_append_seeded_accumulation_helper(Pred/Arity, Options, BaseCode1, BaseCode),

    % Apply constraints if specified
    (   option(constraints(Constraints), Options)
    ->  handle_constraints(Constraints, Dialect, Pred/Arity, BaseCode, Code)
    ;   Code = BaseCode  % No constraints, use verbatim copy
    ).

%% copy_predicate_clauses(+Pred/Arity, -Code)
%  Copy predicate clauses verbatim using introspection
copy_predicate_clauses(Pred/Arity, Code) :-
    functor(Head, Pred, Arity),

    % Find all clauses
    findall(ClauseStr, (
        user:clause(Head, Body0),
        strip_codegen_module_qualifiers(Body0, Body),
        clause_to_string((Head :- Body), ClauseStr)
    ), ClauseStrs),

    (   ClauseStrs = []
    ->  % No clauses found - might be undefined or built-in
        format(atom(Code), '% ~w/~w - Not found or built-in~n', [Pred, Arity])
    ;   atomic_list_concat(ClauseStrs, '\n', Code)
    ).

%% clause_to_string(+Clause, -String)
%  Convert clause term to string representation
clause_to_string(Clause, String) :-
    % Use with_output_to to capture portray_clause output
    with_output_to(atom(String),
        portray_clause(Clause)
    ).

strip_codegen_module_qualifiers(Goal0, Goal) :-
    nonvar(Goal0),
    Goal0 = Module:Inner0,
    !,
    strip_codegen_module_qualifiers(Inner0, Inner),
    (   (   Module == prolog_target
        ;   Module == user
        )
    ->  Goal = Inner
    ;   Goal = Module:Inner
    ).
strip_codegen_module_qualifiers(true, true) :- !.
strip_codegen_module_qualifiers((A0, B0), (A, B)) :-
    !,
    strip_codegen_module_qualifiers(A0, A),
    strip_codegen_module_qualifiers(B0, B).
strip_codegen_module_qualifiers((A0 ; B0), (A ; B)) :-
    !,
    strip_codegen_module_qualifiers(A0, A),
    strip_codegen_module_qualifiers(B0, B).
strip_codegen_module_qualifiers((A0 -> B0 ; C0), (A -> B ; C)) :-
    !,
    strip_codegen_module_qualifiers(A0, A),
    strip_codegen_module_qualifiers(B0, B),
    strip_codegen_module_qualifiers(C0, C).
strip_codegen_module_qualifiers((A0 -> B0), (A -> B)) :-
    !,
    strip_codegen_module_qualifiers(A0, A),
    strip_codegen_module_qualifiers(B0, B).
strip_codegen_module_qualifiers(Goal, Goal).

%% maybe_generate_branch_pruned_predicate(+Pred/Arity, +Options, -Code)
%  For recursive per-path predicates with a concrete mode/1 declaration,
%  generate a standard Prolog wrapper plus helper predicates that prune
%  impossible branches before entering the recursive worker.
maybe_generate_branch_pruned_predicate(Pred/Arity, Options, Code) :-
    option(branch_pruning(Setting), Options, auto),
    (   Setting \= false
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'disabled by option', []),
        fail
    ),
    option(dialect(Dialect), Options, swi),
    (   Dialect == swi
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'dialect ~w is not supported', [Dialect]),
        fail
    ),
    findall(Head-Body,
        (   functor(Head, Pred, Arity),
            user:clause(Head, Body0),
            strip_codegen_module_qualifiers(Body0, Body)
        ),
        Clauses),
    (   Clauses \= []
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'no clauses found', []),
        fail
    ),
    clauses_to_pattern_pairs(Clauses, ClausePairs),
    (   is_per_path_visited_pattern(Pred, Arity, ClausePairs, VisitedPos)
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'predicate does not match canonical per-path visited recursion', []),
        fail
    ),
    (   branch_pruning_mode_positions(Pred/Arity, VisitedPos, InputPositions)
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'missing concrete mode declaration for non-visited inputs', []),
        fail
    ),
    partition(is_recursive_clause_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    (   RecClauses \= []
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'no recursive clauses found after PPV matching', []),
        fail
    ),
    (   BaseClauses \= []
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'no base clauses found after PPV matching', []),
        fail
    ),
    (   branch_pruning_driver_position(Pred, Arity, RecClauses, DriverPos)
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'could not infer a unique driver position', []),
        fail
    ),
    delete(InputPositions, DriverPos, InvariantPositions),
    (   branch_pruning_invariants_static(Pred, Arity, RecClauses, VisitedPos, InvariantPositions)
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'input invariants do not stay fixed across recursive clauses', []),
        fail
    ),
    (   build_branch_pruning_code(Pred/Arity, Clauses, BaseClauses, RecClauses,
            VisitedPos, DriverPos, InputPositions, InvariantPositions, Code)
    ->  true
    ;   branch_pruning_debug(Pred/Arity, 'failed while emitting pruned helper predicates', []),
        fail
    ).

branch_pruning_debug(Pred/Arity, Message, Args) :-
    atom_concat('Skipping branch pruning for ~w/~w: ', Message, Format),
    debug(prolog_branch_pruning, Format, [Pred, Arity|Args]).

maybe_append_min_closure_helper(Pred/Arity, Options, BaseCode, Code) :-
    option(min_closure(Setting), Options, auto),
    (   Setting == false
    ->  Code = BaseCode
    ;   maybe_generate_min_closure_helper(Pred/Arity, Options, HelperCode)
    ->  atomic_list_concat([BaseCode, HelperCode], '\n\n', Code)
    ;   Code = BaseCode
    ).

maybe_append_seeded_accumulation_helper(Pred/Arity, Options, BaseCode, Code) :-
    seeded_accumulation_setting(Options, Setting),
    (   Setting == false
    ->  Code = BaseCode
    ;   maybe_generate_seeded_accumulation_helper(Pred/Arity, Options, HelperCode)
    ->  atomic_list_concat([BaseCode, HelperCode], '\n\n', Code)
    ;   Code = BaseCode
    ).

seeded_accumulation_setting(Options, Setting) :-
    (   option(seeded_accumulation(Setting0), Options)
    ->  Setting = Setting0
    ;   option(effective_distance_accumulation(Setting1), Options)
    ->  Setting = Setting1
    ;   Setting = auto
    ).

maybe_generate_min_closure_helper(Pred/Arity, Options, Code) :-
    option(dialect(Dialect), Options, swi),
    Dialect == swi,
    (   maybe_generate_counted_min_closure_helper(Pred/Arity, Code)
    ;   maybe_generate_weighted_min_closure_helper(Pred/Arity, Code)
    ),
    !.

maybe_generate_counted_min_closure_helper(Pred/Arity, Code) :-
    findall(Head-Body,
        (   functor(Head, Pred, Arity),
            user:clause(Head, Body0),
            strip_codegen_module_qualifiers(Body0, Body)
        ),
        Clauses),
    Clauses \= [],
    partition(is_recursive_clause_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    counted_ppv_visited_position(Pred, Arity, RecClauses, VisitedPos),
    counted_ppv_driver_position(Pred, Arity, RecClauses, VisitedPos, DriverPos),
    branch_pruning_mode_positions(Pred/Arity, VisitedPos, InvariantPositions),
    branch_pruning_signature_positions(DriverPos, InvariantPositions, SignaturePositions),
    min_closure_metric_position(Arity, VisitedPos, SignaturePositions, MetricPos),
    maplist(min_closure_base_depth(MetricPos, VisitedPos), BaseClauses, BaseDepths),
    sort(BaseDepths, [BaseDepth]),
    RecClauses = [FirstRec|RestRecClauses],
    min_closure_recursive_shape(Pred, Arity, VisitedPos, MetricPos, FirstRec, StepIncrement, BudgetMode),
    forall(member(RecClause, RestRecClauses),
        min_closure_recursive_shape_compatible(Pred, Arity, VisitedPos, MetricPos,
            StepIncrement, BudgetMode, RecClause)),
    build_min_closure_code(Pred/Arity, BaseClauses, RecClauses, SignaturePositions,
        MetricPos, VisitedPos, BaseDepth, StepIncrement, BudgetMode, Code).

maybe_generate_weighted_min_closure_helper(Pred/Arity, Code) :-
    findall(Head-Body,
        (   functor(Head, Pred, Arity),
            user:clause(Head, Body0),
            strip_codegen_module_qualifiers(Body0, Body)
        ),
        Clauses),
    Clauses \= [],
    partition(is_recursive_clause_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    counted_ppv_visited_position(Pred, Arity, RecClauses, VisitedPos),
    counted_ppv_driver_position(Pred, Arity, RecClauses, VisitedPos, DriverPos),
    branch_pruning_mode_positions(Pred/Arity, VisitedPos, InvariantPositions),
    branch_pruning_signature_positions(DriverPos, InvariantPositions, SignaturePositions),
    min_closure_metric_position(Arity, VisitedPos, SignaturePositions, MetricPos),
    findall(BasePositive,
        (   member(BaseClause, BaseClauses),
            weighted_min_base_clause_shape(MetricPos, VisitedPos, BaseClause, BasePositive)
        ),
        BasePositives),
    length(BasePositives, BaseCount),
    length(BaseClauses, BaseCount),
    RecClauses = [FirstRec|RestRecClauses],
    weighted_min_recursive_shape(Pred, Arity, VisitedPos, MetricPos, FirstRec,
        BudgetMode, FirstRecPositive),
    BudgetMode = budget(_),
    findall(RestPositive,
        (   member(RecClause, RestRecClauses),
            weighted_min_recursive_shape(Pred, Arity, VisitedPos, MetricPos, RecClause,
                ClauseBudgetMode, RestPositive),
            budget_mode_compatible(BudgetMode, ClauseBudgetMode)
        ),
        RestRecPositives),
    length(RestRecPositives, RestCount),
    length(RestRecClauses, RestCount),
    weighted_positive_step_proven(Pred/Arity, MetricPos, BasePositives,
        [FirstRecPositive|RestRecPositives]),
    build_min_closure_code(Pred/Arity, BaseClauses, RecClauses, SignaturePositions,
        MetricPos, VisitedPos, 1, 1, BudgetMode, Code).

maybe_generate_seeded_accumulation_helper(Pred/Arity, Options, Code) :-
    option(dialect(Dialect), Options, swi),
    Dialect == swi,
    seeded_accumulation_formula(Pred/Arity, Options, Formula),
    (   seeded_accumulation_ppv_shape(Pred/Arity, Shape)
    ;   seeded_accumulation_counted_shape(Pred/Arity, Shape)
    ),
    build_seeded_accumulation_code(Pred/Arity, Shape, Formula, BaseCode),
    (   build_bound_seeded_accumulation_code(Pred/Arity, Shape, Formula, BoundCode)
    ->  Bound = some(BoundCode)
    ;   Bound = none
    ),
    (   build_specialized_seeded_accumulation_code(Pred/Arity, Shape, Formula, GroupedCode)
    ->  Grouped = some(GroupedCode)
    ;   Grouped = none
    ),
    (   Bound = some(_),
        Grouped = some(_),
        build_effective_distance_consumer_code(Pred/Arity, Shape, Formula, ConsumerCode)
    ;   true
    ),
    build_seeded_accumulation_selected_code(Pred/Arity, Shape, Formula, Bound, Grouped,
        SelectedCode),
    findall(Part,
        (   member(Part0, [BaseCode, BoundCode, GroupedCode, ConsumerCode, SelectedCode]),
            nonvar(Part0),
            Part = Part0
        ),
        Parts),
    (   Parts = [Only]
    ->  Code = Only
    ;   atomic_list_concat(Parts, '\n\n', Code)
    ).

seeded_accumulation_formula(Pred/Arity, Options, Formula) :-
    (   current_predicate(user:accumulation_formula/2),
        user:accumulation_formula(Pred/Arity, Formula0)
    ->  normalize_seeded_accumulation_formula(Formula0, Formula),
        seeded_accumulation_formula_available(Options, Formula)
    ;   seeded_accumulation_default_formula(Options, Formula)
    ).

normalize_seeded_accumulation_formula(power_sum(ParamPred/1, Offset),
        accumulation_formula(power_sum, ParamPred, Offset, power_sum, [])) :-
    number(Offset).
normalize_seeded_accumulation_formula(power_sum(ParamPred/1, Offset, Suffix),
        accumulation_formula(power_sum, ParamPred, Offset, Suffix, [])) :-
    number(Offset),
    atom(Suffix).
normalize_seeded_accumulation_formula(power_sum(ParamPred/1, Offset, Suffix, AliasSuffixes),
        accumulation_formula(power_sum, ParamPred, Offset, Suffix, AliasSuffixes)) :-
    number(Offset),
    atom(Suffix),
    is_list(AliasSuffixes),
    forall(member(Alias, AliasSuffixes), atom(Alias)).
normalize_seeded_accumulation_formula(sum_metric(Offset),
        accumulation_formula(sum_metric, none, Offset, sum_metric, [])) :-
    number(Offset).
normalize_seeded_accumulation_formula(sum_metric(Offset, Suffix),
        accumulation_formula(sum_metric, none, Offset, Suffix, [])) :-
    number(Offset),
    atom(Suffix).
normalize_seeded_accumulation_formula(sum_metric(Offset, Suffix, AliasSuffixes),
        accumulation_formula(sum_metric, none, Offset, Suffix, AliasSuffixes)) :-
    number(Offset),
    atom(Suffix),
    is_list(AliasSuffixes),
    forall(member(Alias, AliasSuffixes), atom(Alias)).

seeded_accumulation_formula_available(_Options,
        accumulation_formula(sum_metric, none, _Offset, _Suffix, _Aliases)).
seeded_accumulation_formula_available(Options,
        accumulation_formula(power_sum, ParamPred, _Offset, _Suffix, _Aliases)) :-
    helper_predicate_available(Options, ParamPred/1).

seeded_accumulation_default_formula(Options, Formula) :-
    helper_predicate_available(Options, dimension_n/1),
    !,
    Formula = accumulation_formula(power_sum, dimension_n, 1, power_sum, [effective_distance_sum]).
seeded_accumulation_default_formula(Options, Formula) :-
    helper_predicate_available(Options, influence_dimension/1),
    Formula = accumulation_formula(power_sum, influence_dimension, 1, power_sum, [influence_sum]).

helper_predicate_available(Options, Pred/Arity) :-
    option(predicates(UserPredicates), Options, []),
    memberchk(Pred/Arity, UserPredicates),
    !.
helper_predicate_available(_Options, Pred/Arity) :-
    functor(Head, Pred, Arity),
    clause(user:Head, _).

seeded_accumulation_ppv_shape(Pred/Arity,
        accumulation_shape(SignaturePositions, DriverPos, MetricPos, visited(VisitedPos))) :-
    findall(Head-Body,
        (   functor(Head, Pred, Arity),
            user:clause(Head, Body0),
            strip_codegen_module_qualifiers(Body0, Body)
        ),
        Clauses),
    Clauses \= [],
    partition(is_recursive_clause_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    counted_ppv_visited_position(Pred, Arity, RecClauses, VisitedPos),
    counted_ppv_driver_position(Pred, Arity, RecClauses, VisitedPos, DriverPos),
    branch_pruning_mode_positions(Pred/Arity, VisitedPos, InvariantPositions),
    branch_pruning_signature_positions(DriverPos, InvariantPositions, SignaturePositions),
    min_closure_metric_position(Arity, VisitedPos, SignaturePositions, MetricPos),
    maplist(min_closure_base_depth(MetricPos, VisitedPos), BaseClauses, BaseDepths),
    sort(BaseDepths, [1]),
    RecClauses = [FirstRec|RestRecClauses],
    min_closure_recursive_shape(Pred, Arity, VisitedPos, MetricPos, FirstRec, 1, BudgetMode),
    forall(member(RecClause, RestRecClauses),
        min_closure_recursive_shape_compatible(Pred, Arity, VisitedPos, MetricPos,
            1, BudgetMode, RecClause)).

seeded_accumulation_counted_shape(Pred/Arity,
        accumulation_shape([1, 2], 1, 3, none)) :-
    Arity =:= 3,
    findall(Head-Body,
        (   functor(Head, Pred, Arity),
            user:clause(Head, Body0),
            strip_codegen_module_qualifiers(Body0, Body)
        ),
        Clauses),
    Clauses \= [],
    partition(is_recursive_clause_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],
    maplist(counted_accumulation_base_depth, BaseClauses, BaseDepths),
    sort(BaseDepths, [1]),
    forall(member(RecClause, RecClauses),
        counted_accumulation_recursive_shape(Pred, RecClause, 1)).

build_seeded_accumulation_code(Pred/Arity,
        accumulation_shape(SignaturePositions, DriverPos, MetricPos, VisitedInfo),
        accumulation_formula(FormulaKind, ParamPred, Offset, PrimarySuffix, AliasSuffixes),
        Code) :-
    helper_name_for_suffix(Pred, PrimarySuffix, HelperName),
    length(SignaturePositions, SignatureCount),
    build_min_signature_vars(SignatureCount, SignatureArgs),
    append(SignatureArgs, [WeightSum], HeadArgs),
    HeadTerm =.. [HelperName|HeadArgs],
    build_seeded_accumulation_call_args(Arity, SignaturePositions, DriverPos,
        MetricPos, VisitedInfo, SignatureArgs, MetricVar, CallArgs),
    PredCall =.. [Pred|CallArgs],
    build_seeded_accumulation_key_goals(SignatureArgs, MetricVar, PredCall, KeyGoals),
    build_seeded_accumulation_body_goals(FormulaKind, ParamPred, Offset,
        PredCall, MetricVar, WeightSum, KeyGoals, BodyGoals),
    goals_to_body(BodyGoals, BodyTerm),
    clause_term(HeadTerm, BodyTerm, ClauseTerm),
    build_seeded_accumulation_alias_clauses(Pred, HelperName, AliasSuffixes,
        SignatureCount, AliasClauses),
    append([ClauseTerm], AliasClauses, Clauses),
    clauses_to_code(Clauses, ClauseCode),
    atomic_list_concat([
        '% Mode-directed seeded accumulation helper for canonical counted recursion.',
        ClauseCode
    ], '\n\n', Code).

build_specialized_seeded_accumulation_code(Pred/Arity,
        accumulation_shape([1, 2], 1, 3, VisitedInfo),
        accumulation_formula(FormulaKind, ParamPred, Offset, PrimarySuffix, AliasSuffixes),
        Code) :-
    FormulaKind == power_sum,
    build_grouped_alias_suffixes([PrimarySuffix|AliasSuffixes],
        [GroupedPrimarySuffix|GroupedAliasSuffixes]),
    helper_name_for_suffix(Pred, GroupedPrimarySuffix, HelperName),
    atom_concat(HelperName, '$grouped', GroupedName),
    atom_concat(GroupedName, '$sum_pairs', SumPairsName),
    atom_concat(GroupedName, '$take_same_key', TakeSameKeyName),
    build_specialized_grouped_seeded_accumulation_clauses(Pred/Arity, VisitedInfo,
        accumulation_formula(FormulaKind, ParamPred, Offset,
            GroupedPrimarySuffix, GroupedAliasSuffixes),
        HelperName, GroupedName, SumPairsName, TakeSameKeyName, Clauses, TableDeclCode),
    clauses_to_code(Clauses, ClauseCode),
    atomic_list_concat([
        '% Specialized mode-directed seeded accumulation helper for grouped counted recursion.',
        TableDeclCode,
        ClauseCode
    ], '\n\n', Code).

build_grouped_alias_suffixes([], []).
build_grouped_alias_suffixes([Suffix|Rest], [GroupedSuffix|GroupedRest]) :-
    format(atom(GroupedSuffix), '~w_grouped', [Suffix]),
    build_grouped_alias_suffixes(Rest, GroupedRest).

build_bound_alias_suffixes([], []).
build_bound_alias_suffixes([Suffix|Rest], [BoundSuffix|BoundRest]) :-
    format(atom(BoundSuffix), '~w_bound', [Suffix]),
    build_bound_alias_suffixes(Rest, BoundRest).

selected_suffix_for_suffix(Suffix, SelectedSuffix) :-
    format(atom(SelectedSuffix), '~w_selected', [Suffix]).

selected_suffixes_for_suffixes([], []).
selected_suffixes_for_suffixes([Suffix|Rest], [SelectedSuffix|SelectedRest]) :-
    selected_suffix_for_suffix(Suffix, SelectedSuffix),
    selected_suffixes_for_suffixes(Rest, SelectedRest).

build_seeded_accumulation_selected_code(Pred/_Arity,
        accumulation_shape(SignaturePositions, _DriverPos, _MetricPos, _VisitedInfo),
        accumulation_formula(_FormulaKind, _ParamPred, _Offset, PrimarySuffix, AliasSuffixes),
        Bound, Grouped, Code) :-
    helper_name_for_suffix(Pred, PrimarySuffix, PrimaryName),
    build_bound_alias_suffixes([PrimarySuffix], [BoundPrimarySuffix]),
    helper_name_for_suffix(Pred, BoundPrimarySuffix, BoundPrimaryName),
    build_grouped_alias_suffixes([PrimarySuffix], [GroupedPrimarySuffix]),
    helper_name_for_suffix(Pred, GroupedPrimarySuffix, GroupedPrimaryName),
    selected_suffix_for_suffix(PrimarySuffix, SelectedPrimarySuffix),
    helper_name_for_suffix(Pred, SelectedPrimarySuffix, SelectedPrimaryName),
    length(SignaturePositions, SignatureCount),
    build_seeded_accumulation_selected_primary_clause(SignatureCount, SelectedPrimaryName,
        PrimaryName, BoundPrimaryName, Bound, GroupedPrimaryName, Grouped, ClauseTerm),
    selected_suffixes_for_suffixes(AliasSuffixes, SelectedAliasSuffixes),
    build_seeded_accumulation_alias_clauses(Pred, SelectedPrimaryName, SelectedAliasSuffixes,
        SignatureCount, AliasClauses),
    append([ClauseTerm], AliasClauses, Clauses),
    clauses_to_code(Clauses, ClauseCode),
    atomic_list_concat([
        '% Query-aware seeded accumulation selector helper.',
        ClauseCode
    ], '\n\n', Code).

build_seeded_accumulation_selected_primary_clause(2, SelectedPrimaryName, PrimaryName,
        BoundPrimaryName, Bound, GroupedPrimaryName, Grouped, ClauseTerm) :-
    Head =.. [SelectedPrimaryName, SeedArg, KeyArg, WeightSum],
    GenericCall =.. [PrimaryName, SeedArg, KeyArg, WeightSum],
    build_seeded_accumulation_selected_bound_call(BoundPrimaryName, Bound,
        SeedArg, KeyArg, WeightSum, GenericCall, BoundCall),
    build_seeded_accumulation_selected_grouped_call(GroupedPrimaryName, Grouped,
        SeedArg, KeyArg, WeightSum, GenericCall, GroupedCall),
    Body = (nonvar(KeyArg) -> BoundCall ; GroupedCall),
    clause_term(Head, Body, ClauseTerm).
build_seeded_accumulation_selected_primary_clause(SignatureCount, SelectedPrimaryName,
        PrimaryName, _BoundPrimaryName, _Bound, _GroupedPrimaryName, _Grouped, ClauseTerm) :-
    build_min_signature_vars(SignatureCount, SignatureArgs),
    append(SignatureArgs, [_WeightSum], HeadArgs),
    Head =.. [SelectedPrimaryName|HeadArgs],
    PrimaryCall =.. [PrimaryName|HeadArgs],
    clause_term(Head, PrimaryCall, ClauseTerm).

build_seeded_accumulation_selected_bound_call(BoundPrimaryName, some(_BoundCode),
        SeedArg, KeyArg, WeightSum, _GenericCall, BoundCall) :-
    BoundCall =.. [BoundPrimaryName, SeedArg, KeyArg, WeightSum].
build_seeded_accumulation_selected_bound_call(_BoundPrimaryName, none,
        _SeedArg, _KeyArg, _WeightSum, GenericCall, GenericCall).

build_seeded_accumulation_selected_grouped_call(GroupedPrimaryName, some(_GroupedCode),
        SeedArg, KeyArg, WeightSum, _GenericCall, GroupedCall) :-
    GroupedCall =.. [GroupedPrimaryName, SeedArg, KeyArg, WeightSum].
build_seeded_accumulation_selected_grouped_call(_GroupedPrimaryName, none,
        _SeedArg, _KeyArg, _WeightSum, GenericCall, GenericCall).

build_bound_seeded_accumulation_code(Pred/Arity,
        accumulation_shape([1, 2], 1, MetricPos, VisitedInfo),
        accumulation_formula(FormulaKind, ParamPred, Offset, PrimarySuffix, AliasSuffixes),
        Code) :-
    memberchk(FormulaKind, [power_sum, sum_metric]),
    build_bound_alias_suffixes([PrimarySuffix|AliasSuffixes],
        [BoundPrimarySuffix|BoundAliasSuffixes]),
    helper_name_for_suffix(Pred, BoundPrimarySuffix, HelperName),
    build_bound_seeded_accumulation_clauses(Pred/Arity, VisitedInfo,
        accumulation_formula(FormulaKind, ParamPred, Offset,
            BoundPrimarySuffix, BoundAliasSuffixes),
        HelperName, MetricPos, Clauses),
    clauses_to_code(Clauses, ClauseCode),
    atomic_list_concat([
        '% Bound-key seeded accumulation helper for canonical counted recursion.',
        ClauseCode
    ], '\n\n', Code).

build_bound_seeded_accumulation_clauses(Pred/Arity, VisitedInfo,
        accumulation_formula(FormulaKind, ParamPred, Offset, _PrimarySuffix, AliasSuffixes),
        HelperName, MetricPos, Clauses) :-
    Head =.. [HelperName, SeedArg, KeyArg, WeightSum],
    build_seeded_accumulation_call_args(Arity, [1, 2], 1, MetricPos, VisitedInfo,
        [SeedArg, KeyArg], MetricVar, CallArgs),
    PredCall =.. [Pred|CallArgs],
    build_bound_seeded_accumulation_body_goals(FormulaKind, ParamPred, Offset,
        PredCall, KeyArg, MetricVar, WeightSum, BodyGoals),
    goals_to_body(BodyGoals, BodyTerm),
    clause_term(Head, BodyTerm, ClauseTerm),
    build_seeded_accumulation_alias_clauses(Pred, HelperName, AliasSuffixes, 2, AliasClauses),
    append([ClauseTerm], AliasClauses, Clauses).

build_bound_seeded_accumulation_body_goals(power_sum, ParamPred, Offset,
        PredCall, KeyArg, MetricVar, WeightSum, BodyGoals) :-
    ParamCall =.. [ParamPred, N],
    accumulation_metric_with_offset_goal(MetricVar, Offset, WeightedMetric, OffsetGoals),
    append([PredCall], OffsetGoals, AggregateGoals0),
    append(AggregateGoals0, [(W is WeightedMetric ** NegN)], AggregateGoals),
    goals_to_body(AggregateGoals, AggregateBody),
    AggregateGoal = aggregate_all(sum(W), AggregateBody, WeightSum),
    BodyGoals = [
        nonvar(KeyArg),
        ParamCall,
        (NegN is -N),
        AggregateGoal,
        (WeightSum > 0)
    ].
build_bound_seeded_accumulation_body_goals(sum_metric, _ParamPred, Offset,
        PredCall, KeyArg, MetricVar, WeightSum, BodyGoals) :-
    accumulation_metric_with_offset_goal(MetricVar, Offset, WeightedMetric, OffsetGoals),
    append([PredCall], OffsetGoals, AggregateGoals0),
    append(AggregateGoals0, [(W is WeightedMetric)], AggregateGoals),
    goals_to_body(AggregateGoals, AggregateBody),
    AggregateGoal = aggregate_all(sum(W), AggregateBody, WeightSum),
    BodyGoals = [
        nonvar(KeyArg),
        AggregateGoal,
        (WeightSum > 0)
    ].

build_specialized_grouped_seeded_accumulation_clauses(Pred/Arity, VisitedInfo,
        accumulation_formula(FormulaKind, ParamPred, Offset, _PrimarySuffix, AliasSuffixes),
        HelperName, GroupedName, SumPairsName, TakeSameKeyName, Clauses, TableDeclCode) :-
    HelperHead =.. [HelperName, SeedArg, KeyArg, WeightSum],
    GroupedCall =.. [GroupedName, SeedArg, PairSums],
    HelperBody = (GroupedCall, member(KeyArg-WeightSum, PairSums)),
    clause_term(HelperHead, HelperBody, HelperClause),
    build_specialized_grouped_pairs_clause(Pred/Arity, VisitedInfo, FormulaKind, ParamPred,
        Offset, GroupedName, SumPairsName, GroupedClause),
    build_specialized_grouped_sum_pairs_clauses(SumPairsName, TakeSameKeyName, SumPairsClauses),
    build_seeded_accumulation_alias_clauses(Pred, HelperName, AliasSuffixes, 2, AliasClauses),
    append([HelperClause, GroupedClause], SumPairsClauses, Clauses0),
    append(Clauses0, AliasClauses, Clauses),
    TableSpec =.. [GroupedName, +, -],
    format(atom(TableDeclCode), ':- table ~q.', [TableSpec]).

build_specialized_grouped_pairs_clause(Pred/Arity, VisitedInfo, FormulaKind, ParamPred, Offset,
        GroupedName, SumPairsName, ClauseTerm) :-
    build_seeded_accumulation_call_args(Arity, [1, 2], 1, 3, VisitedInfo,
        [SeedArg, KeyArg], MetricVar, CallArgs),
    PredCall =.. [Pred|CallArgs],
    build_specialized_grouped_seeded_accumulation_body_goals(FormulaKind, ParamPred, Offset,
        PredCall, KeyArg, MetricVar, WeightVar, SetupGoals, EnumGoals),
    goals_to_body(EnumGoals, EnumBody),
    append(SetupGoals, [
        findall(KeyArg-WeightVar, EnumBody, RawPairs),
        (RawPairs \= []),
        keysort(RawPairs, SortedPairs),
        SumPairsCall
    ], BodyGoals),
    SumPairsCall =.. [SumPairsName, SortedPairs, PairSums],
    goals_to_body(BodyGoals, BodyTerm),
    GroupedHead =.. [GroupedName, SeedArg, PairSums],
    clause_term(GroupedHead, BodyTerm, ClauseTerm).

build_specialized_grouped_seeded_accumulation_body_goals(power_sum, ParamPred, Offset,
        PredCall, KeyArg, MetricVar, WeightVar, SetupGoals, EnumGoals) :-
    ParamCall =.. [ParamPred, N],
    accumulation_metric_with_offset_goal(MetricVar, Offset, WeightedMetric, OffsetGoals),
    SetupGoals = [
        ParamCall,
        (NegN is -N)
    ],
    append([PredCall], OffsetGoals, EnumGoals0),
    append(EnumGoals0, [
        nonvar(KeyArg),
        (WeightVar is WeightedMetric ** NegN)
    ], EnumGoals).
build_specialized_grouped_seeded_accumulation_body_goals(sum_metric, _ParamPred, Offset,
        PredCall, KeyArg, MetricVar, WeightVar, [], EnumGoals) :-
    accumulation_metric_with_offset_goal(MetricVar, Offset, WeightedMetric, OffsetGoals),
    append([PredCall], OffsetGoals, EnumGoals0),
    append(EnumGoals0, [
        nonvar(KeyArg),
        (WeightVar is WeightedMetric)
    ], EnumGoals).

build_specialized_grouped_sum_pairs_clauses(SumPairsName, TakeSameKeyName, Clauses) :-
    EmptyHead =.. [SumPairsName, [], []],
    clause_term(EmptyHead, true, EmptyClause),
    RecHead =.. [SumPairsName, [Key-Weight|RestPairs], [Key-WeightSum|GroupedPairs]],
    TakeSameKeyCall =.. [TakeSameKeyName, RestPairs, Key, Weight, WeightSum, RemainingPairs],
    NextSumPairsCall =.. [SumPairsName, RemainingPairs, GroupedPairs],
    RecBody = (TakeSameKeyCall, NextSumPairsCall),
    clause_term(RecHead, RecBody, RecClause),
    TakeEmptyHead =.. [TakeSameKeyName, [], _IgnoredKey0, Acc, Acc, []],
    clause_term(TakeEmptyHead, true, TakeEmptyClause),
    TakeSameHead =.. [TakeSameKeyName, [Key-Weight|RestPairs], Key, Acc0, WeightSum, RemainingPairs],
    TakeSameBody = (
        !,
        Acc1 is Acc0 + Weight,
        TakeSameKeyCall1
    ),
    TakeSameKeyCall1 =.. [TakeSameKeyName, RestPairs, Key, Acc1, WeightSum, RemainingPairs],
    clause_term(TakeSameHead, TakeSameBody, TakeSameClause),
    TakeDoneHead =.. [TakeSameKeyName, RemainingPairs, _IgnoredKey1, WeightSum, WeightSum, RemainingPairs],
    clause_term(TakeDoneHead, true, TakeDoneClause),
    Clauses = [EmptyClause, RecClause, TakeEmptyClause, TakeSameClause, TakeDoneClause].

build_effective_distance_consumer_code(Pred/_Arity,
        accumulation_shape([1, 2], 1, _MetricPos, _VisitedInfo),
        Formula, Code) :-
    effective_distance_suffix_from_formula(Formula, EffectiveSuffix),
    build_bound_alias_suffixes([EffectiveSuffix], [EffectiveBoundSuffix]),
    helper_name_for_suffix(Pred, EffectiveBoundSuffix, EffectiveBoundHelperName),
    build_grouped_alias_suffixes([EffectiveSuffix], [EffectiveGroupedSuffix]),
    helper_name_for_suffix(Pred, EffectiveGroupedSuffix, EffectiveGroupedHelperName),
    helper_name_for_suffix(Pred, effective_distance_article_sum, ArticleSumHelperName),
    helper_name_for_suffix(Pred, effective_distance_article_sum_by_root, ArticleSumByRootHelperName),
    helper_name_for_suffix(Pred, effective_distance_article_sum_pairs_by_root,
        ArticleSumPairsByRootHelperName),
    helper_name_for_suffix(Pred, effective_distance_article_sum_root_grouped, RootGroupedName),
    helper_name_for_suffix(Pred, effective_distance_article_sum_article_grouped, ArticleGroupedName),
    atom_concat(ArticleSumHelperName, '$sum_pairs', SumPairsName),
    atom_concat(ArticleSumHelperName, '$take_same_key', TakeSameKeyName),
    build_effective_distance_consumer_clauses(ArticleSumHelperName, ArticleSumByRootHelperName,
        ArticleSumPairsByRootHelperName, RootGroupedName, ArticleGroupedName,
        EffectiveBoundHelperName, EffectiveGroupedHelperName, SumPairsName, TakeSameKeyName,
        Clauses, TableDeclCode),
    clauses_to_code(Clauses, ClauseCode),
    atomic_list_concat([
        '% Effective-distance grouped consumer helper for compact (article, root, weight_sum) rows.',
        TableDeclCode,
        ClauseCode
    ], '\n\n', Code).

effective_distance_suffix_from_formula(
        accumulation_formula(power_sum, _ParamPred, _Offset, PrimarySuffix, AliasSuffixes),
        EffectiveSuffix) :-
    member(EffectiveSuffix, [PrimarySuffix|AliasSuffixes]),
    EffectiveSuffix == effective_distance_sum.

build_effective_distance_consumer_clauses(ArticleSumHelperName, ArticleSumByRootHelperName,
        ArticleSumPairsByRootHelperName, RootGroupedName, ArticleGroupedName,
        EffectiveBoundHelperName, EffectiveGroupedHelperName, SumPairsName, TakeSameKeyName,
        Clauses, TableDeclCode) :-
    build_effective_distance_consumer_entry_clauses(ArticleSumHelperName,
        ArticleSumByRootHelperName, ArticleSumPairsByRootHelperName, RootGroupedName,
        ArticleGroupedName, EntryClauses),
    build_effective_distance_consumer_root_grouped_clause(RootGroupedName,
        EffectiveBoundHelperName, SumPairsName, RootGroupedClause),
    build_effective_distance_consumer_article_grouped_clause(ArticleGroupedName,
        EffectiveGroupedHelperName, SumPairsName, ArticleGroupedClause),
    build_specialized_grouped_sum_pairs_clauses(SumPairsName, TakeSameKeyName, SumPairClauses),
    append(EntryClauses, [RootGroupedClause, ArticleGroupedClause], Clauses0),
    append(Clauses0, SumPairClauses, Clauses),
    RootTableSpec =.. [RootGroupedName, +, -],
    ArticleTableSpec =.. [ArticleGroupedName, +, -],
    format(atom(TableDeclCode), ':- table ~q.~n:- table ~q.',
        [RootTableSpec, ArticleTableSpec]).

build_effective_distance_consumer_entry_clauses(ArticleSumHelperName, ArticleSumByRootHelperName,
        ArticleSumPairsByRootHelperName, RootGroupedName, ArticleGroupedName, Clauses) :-
    RootPairsHead =.. [ArticleSumPairsByRootHelperName, Root, ArticlePairs],
    RootPairsCall =.. [RootGroupedName, Root, ArticlePairs],
    RootPairsBody = (
        nonvar(Root),
        RootPairsCall
    ),
    clause_term(RootPairsHead, RootPairsBody, RootPairsClause),
    ByRootHead =.. [ArticleSumByRootHelperName, Root, Article, WeightSum],
    ByRootPairsCall =.. [ArticleSumPairsByRootHelperName, Root, ArticlePairs],
    ByRootBody = (
        nonvar(Root),
        ByRootPairsCall,
        member(Article-WeightSum, ArticlePairs)
    ),
    clause_term(ByRootHead, ByRootBody, ByRootClause),
    RootHead =.. [ArticleSumHelperName, Article, Root, WeightSum],
    RootByArticleCall =.. [ArticleSumByRootHelperName, Root, Article, WeightSum],
    RootBody = (
        nonvar(Root),
        !,
        RootByArticleCall
    ),
    clause_term(RootHead, RootBody, RootClause),
    ArticleHead =.. [ArticleSumHelperName, Article, Root, WeightSum],
    ArticleGroupedCall =.. [ArticleGroupedName, Article, RootPairs],
    ArticleBody = (
        var(Root),
        nonvar(Article),
        !,
        ArticleGroupedCall,
        member(Root-WeightSum, RootPairs)
    ),
    clause_term(ArticleHead, ArticleBody, ArticleClause),
    EnumerateHead =.. [ArticleSumHelperName, Article, Root, WeightSum],
    EnumerateBody = (
        var(Root),
        var(Article),
        setof(Article0, Cat0^article_category(Article0, Cat0), Articles),
        member(Article, Articles),
        ArticleGroupedCall,
        member(Root-WeightSum, RootPairs)
    ),
    clause_term(EnumerateHead, EnumerateBody, EnumerateClause),
    Clauses = [RootPairsClause, ByRootClause, RootClause, ArticleClause, EnumerateClause].

build_effective_distance_consumer_root_grouped_clause(RootGroupedName,
        EffectiveBoundHelperName, SumPairsName, ClauseTerm) :-
    Head =.. [RootGroupedName, Root, ArticlePairs],
    SeedCatsGoal = setof(Cat, Article^article_category(Article, Cat), SeedCats),
    EffectiveBoundCall =.. [EffectiveBoundHelperName, Cat, Root, Weight],
    SeedPairsGoal = findall(Cat-Weight,
        (   member(Cat, SeedCats),
            Cat \= Root,
            EffectiveBoundCall
        ),
        SeedPairs),
    SumPairsCall =.. [SumPairsName, SortedPairs, ArticlePairs],
    Body = (
        SeedCatsGoal,
        SeedPairsGoal,
        findall(Article-Weight,
            (   article_category(Article, Root),
                Weight = 1.0
            ;   article_category(Article, Cat),
                member(Cat-Weight, SeedPairs)
            ),
            RawPairs),
        RawPairs \= [],
        keysort(RawPairs, SortedPairs),
        SumPairsCall
    ),
    clause_term(Head, Body, ClauseTerm).

build_effective_distance_consumer_article_grouped_clause(ArticleGroupedName,
        EffectiveGroupedHelperName, SumPairsName, ClauseTerm) :-
    Head =.. [ArticleGroupedName, Article, RootPairs],
    EffectiveGroupedCall =.. [EffectiveGroupedHelperName, Cat, Root, Weight],
    DirectRootWeight = 1.0,
    SumPairsCall =.. [SumPairsName, SortedPairs, RootPairs],
    Body = (
        findall(Root-Weight,
            (   article_category(Article, Root),
                Weight = DirectRootWeight
            ;   article_category(Article, Cat),
                EffectiveGroupedCall
            ),
            RawPairs),
        RawPairs \= [],
        keysort(RawPairs, SortedPairs),
        SumPairsCall
    ),
    clause_term(Head, Body, ClauseTerm).

helper_name_for_suffix(Pred, Suffix, HelperName) :-
    format(atom(HelperName), '~w$~w', [Pred, Suffix]).

build_seeded_accumulation_key_goals(SignatureArgs, MetricVar, PredCall, KeyGoals) :-
    build_seeded_accumulation_key_term(SignatureArgs, KeyTerm),
    KeyGoals = [
        setof(KeyTerm, MetricVar^PredCall, KeyTerms),
        member(KeyTerm, KeyTerms)
    ].

build_seeded_accumulation_key_term(SignatureArgs, KeyTerm) :-
    length(SignatureArgs, SignatureCount),
    format(atom(Functor), 'sig~w', [SignatureCount]),
    KeyTerm =.. [Functor|SignatureArgs].

build_seeded_accumulation_body_goals(power_sum, ParamPred, Offset,
        PredCall, MetricVar, WeightSum, KeyGoals, BodyGoals) :-
    ParamCall =.. [ParamPred, N],
    accumulation_metric_with_offset_goal(MetricVar, Offset, WeightedMetric, OffsetGoals),
    append([PredCall], OffsetGoals, AggregateGoals0),
    append(AggregateGoals0, [(W is WeightedMetric ** NegN)], AggregateGoals),
    goals_to_body(AggregateGoals, AggregateBody),
    AggregateGoal = aggregate_all(sum(W), AggregateBody, WeightSum),
    append([
        ParamCall
    ], KeyGoals, BodyGoals0),
    append(BodyGoals0, [
        (NegN is -N),
        AggregateGoal,
        (WeightSum > 0)
    ], BodyGoals).
build_seeded_accumulation_body_goals(sum_metric, _ParamPred, Offset,
        PredCall, MetricVar, WeightSum, KeyGoals, BodyGoals) :-
    accumulation_metric_with_offset_goal(MetricVar, Offset, WeightedMetric, OffsetGoals),
    append([PredCall], OffsetGoals, AggregateGoals0),
    append(AggregateGoals0, [(W is WeightedMetric)], AggregateGoals),
    goals_to_body(AggregateGoals, AggregateBody),
    AggregateGoal = aggregate_all(sum(W), AggregateBody, WeightSum),
    append(KeyGoals, [
        AggregateGoal,
        (WeightSum > 0)
    ], BodyGoals).

accumulation_metric_with_offset_goal(MetricVar, Offset, WeightedMetric, Goals) :-
    (   Offset =:= 0
    ->  WeightedMetric = MetricVar,
        Goals = []
    ;   Goals = [(WeightedMetric is MetricVar + Offset)]
    ).

build_seeded_accumulation_alias_clauses(_Pred, _PrimaryName, [], _SignatureCount, []).
build_seeded_accumulation_alias_clauses(Pred, PrimaryName, [Suffix|Rest],
        SignatureCount, [ClauseTerm|ClauseTerms]) :-
    helper_name_for_suffix(Pred, Suffix, AliasName),
    build_min_signature_vars(SignatureCount, SignatureArgs),
    append(SignatureArgs, [_WeightSum], HeadArgs),
    AliasHead =.. [AliasName|HeadArgs],
    PrimaryCall =.. [PrimaryName|HeadArgs],
    ClauseTerm = (AliasHead :- PrimaryCall),
    build_seeded_accumulation_alias_clauses(Pred, PrimaryName, Rest,
        SignatureCount, ClauseTerms).

build_seeded_accumulation_call_args(Arity, SignaturePositions, DriverPos, MetricPos,
        VisitedInfo, SignatureArgs, MetricVar, CallArgs) :-
    length(CallArgs, Arity),
    bind_positions_1based(SignaturePositions, SignatureArgs, CallArgs),
    nth1(MetricPos, CallArgs, MetricVar),
    (   VisitedInfo = visited(VisitedPos)
    ->  nth1(DriverPos, CallArgs, DriverArg),
        nth1(VisitedPos, CallArgs, [DriverArg])
    ;   true
    ).

counted_accumulation_base_depth(Head-Body, BaseDepth) :-
    Head =.. [_Pred, From, To, MetricArg],
    var(From),
    var(To),
    body_goals(Body, Goals0),
    (   number(MetricArg)
    ->  BaseDepth = MetricArg,
        Goals = Goals0
    ;   var(MetricArg),
        select(MetricGoal, Goals0, Goals),
        metric_constant_goal(MetricArg, MetricGoal, BaseDepth)
    ),
    Goals = [EdgeGoal],
    counted_accumulation_edge_goal(From, To, EdgeGoal),
    number(BaseDepth),
    BaseDepth > 0.

counted_accumulation_recursive_shape(Pred, Head-Body, StepIncrement) :-
    Head =.. [Pred, From, To, MetricVar],
    var(From),
    var(To),
    var(MetricVar),
    body_goals(Body, Goals0),
    strip_counted_accumulation_budget_goals(Goals0, MetricVar, Goals1),
    select(EdgeGoal, Goals1, Goals2),
    counted_accumulation_edge_goal(From, Mid, EdgeGoal),
    select(RecGoal, Goals2, Goals3),
    counted_accumulation_recursive_goal(Pred, Mid, To, PrevMetric, RecGoal),
    select(MetricGoal, Goals3, RemainingGoals),
    metric_increment_goal(MetricVar, PrevMetric, MetricGoal, StepIncrement),
    number(StepIncrement),
    StepIncrement > 0,
    RemainingGoals == [].

counted_accumulation_edge_goal(From, To, Goal0) :-
    strip_module(Goal0, _, Goal),
    functor(Goal, _Functor, 2),
    arg(1, Goal, From),
    arg(2, Goal, To).

counted_accumulation_recursive_goal(Pred, From, To, MetricVar, Goal0) :-
    strip_module(Goal0, _, Goal),
    Goal =.. [Pred, From, To, MetricVar].

strip_counted_accumulation_budget_goals(Goals0, MetricVar, Goals) :-
    exclude(is_cut_goal, Goals0, GoalsNoCuts),
    (   select(MaxGoal, GoalsNoCuts, Goals1),
        max_depth_budget_goal(MaxGoal, MaxVar),
        select(CompareGoal, Goals1, Goals2),
        counted_accumulation_budget_goal(CompareGoal, MetricVar, MaxVar)
    ->  Goals = Goals2
    ;   Goals = GoalsNoCuts
    ).

counted_accumulation_budget_goal(Goal0, MetricVar, MaxVar) :-
    strip_module(Goal0, _, Goal),
    (   Goal =.. [=<, Left, Right]
    ;   Goal =.. [<, Left, Right]
    ),
    Left == MetricVar,
    Right == MaxVar.

counted_ppv_visited_position(Pred, Arity, RecClauses, VisitedPos) :-
    member(Head-Body, RecClauses),
    Head =.. [Pred|HeadArgs],
    between(1, Arity, VisitedPos),
    nth1(VisitedPos, HeadArgs, VisitedVar),
    var(VisitedVar),
    counted_ppv_clause_parts(Pred, Arity, VisitedPos, Head-Body,
        _HeadArgs, _PreRecGoals, _RecCall, _PostGoals, _StepGoal, _NextVar, VisitedVar),
    !.

counted_ppv_driver_position(Pred, Arity, RecClauses, VisitedPos, DriverPos) :-
    maplist(counted_ppv_clause_driver_position(Pred, Arity, VisitedPos), RecClauses, DriverPositions),
    sort(DriverPositions, [DriverPos]).

counted_ppv_clause_driver_position(Pred, Arity, VisitedPos, Head-Body, DriverPos) :-
    counted_ppv_clause_parts(Pred, Arity, VisitedPos, Head-Body,
        HeadArgs, _PreRecGoals, _RecCall, _PostGoals, StepGoal, _NextVar, _VisitedVar),
    step_goal_input_var(StepGoal, StepInputVar),
    findall(Pos,
        (   between(1, Arity, Pos),
            nth1(Pos, HeadArgs, Arg),
            Arg == StepInputVar
        ),
        [DriverPos]).

counted_ppv_clause_parts(Pred, Arity, VisitedPos, Head-Body,
        HeadArgs, PreRecGoals, RecCall, PostGoals, StepGoal, NextVar, VisitedVar) :-
    Head =.. [Pred|HeadArgs],
    nth1(VisitedPos, HeadArgs, VisitedVar),
    body_goals(Body, Goals),
    append(PreRecGoals, [RecCall|PostGoals], Goals),
    branch_pruning_recursive_goal(Pred, Arity, RecCall),
    RecCall =.. [_|RecArgs],
    nth1(1, RecArgs, NextVar),
    nth1(VisitedPos, RecArgs, VisitedArg),
    nonvar(VisitedArg),
    VisitedArg = [NextVar|Tail],
    Tail == VisitedVar,
    counted_ppv_step_goal(HeadArgs, PreRecGoals, NextVar, StepGoal).

counted_ppv_step_goal(HeadArgs, Goals, NextVar, StepGoal) :-
    member(StepGoal, Goals),
    counted_ppv_step_goal_shape(StepGoal, HeadArgs, NextVar),
    !.

counted_ppv_step_goal_shape(Goal0, HeadArgs, NextVar) :-
    strip_module(Goal0, _, Goal),
    callable(Goal),
    Goal \= (\+ _),
    Goal \= not(_),
    Goal \= !,
    Goal =.. [Functor|Args],
    Functor \= member,
    Args = [DriverVar|RestArgs],
    var(DriverVar),
    member_samevar(DriverVar, HeadArgs),
    member_samevar(NextVar, RestArgs),
    var(NextVar),
    NextVar \== DriverVar.

member_samevar(Var, [Candidate|_]) :-
    Candidate == Var,
    !.
member_samevar(Var, [_|Rest]) :-
    member_samevar(Var, Rest).

min_closure_metric_position(Arity, VisitedPos, SignaturePositions, MetricPos) :-
    findall(Pos,
        (   between(1, Arity, Pos),
            Pos =\= VisitedPos,
            \+ memberchk(Pos, SignaturePositions)
        ),
        [MetricPos]).

min_closure_base_depth(MetricPos, VisitedPos, Head-Body, BaseDepth) :-
    Head =.. [_|HeadArgs],
    nth1(VisitedPos, HeadArgs, VisitedVar),
    body_goals(Body, Goals0),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), Goals0, Goals1),
    strip_counted_ppv_setup_goals(Goals1, VisitedVar, _BudgetMode, Goals),
    min_closure_metric_constant(HeadArgs, MetricPos, Goals, BaseDepth),
    number(BaseDepth),
    BaseDepth > 0.

min_closure_metric_constant(HeadArgs, MetricPos, _Goals, BaseDepth) :-
    nth1(MetricPos, HeadArgs, BaseDepth),
    number(BaseDepth),
    !.
min_closure_metric_constant(HeadArgs, MetricPos, Goals, BaseDepth) :-
    nth1(MetricPos, HeadArgs, MetricVar),
    member(Goal, Goals),
    metric_constant_goal(MetricVar, Goal, BaseDepth),
    !.

metric_constant_goal(HeadVar, Goal0, Value) :-
    strip_module(Goal0, _, Goal),
    (   Goal =.. [is, HeadVar, Expr]
    ;   Goal =.. ['=', HeadVar, Expr]
    ),
    number(Expr),
    Value = Expr.

min_closure_recursive_shape(Pred, Arity, VisitedPos, MetricPos, Head-Body, StepIncrement, BudgetMode) :-
    counted_ppv_clause_parts(Pred, Arity, VisitedPos, Head-Body,
        HeadArgs, PreRecGoals0, RecCall, PostGoals, _StepGoal, _NextVar, VisitedVar),
    strip_counted_ppv_setup_goals(PreRecGoals0, VisitedVar, BudgetMode, PreRecGoals),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), PreRecGoals, _UsefulPreGoals),
    RecCall =.. [_|RecArgs],
    nth1(MetricPos, HeadArgs, HeadMetric),
    nth1(MetricPos, RecArgs, RecMetric),
    member(MetricGoal, PostGoals),
    metric_increment_goal(HeadMetric, RecMetric, MetricGoal, StepIncrement),
    number(StepIncrement),
    StepIncrement > 0.

min_closure_recursive_shape_compatible(Pred, Arity, VisitedPos, MetricPos,
        StepIncrement, BudgetMode, Clause) :-
    min_closure_recursive_shape(Pred, Arity, VisitedPos, MetricPos, Clause,
        StepIncrement, ClauseBudgetMode),
    budget_mode_compatible(BudgetMode, ClauseBudgetMode).

budget_mode_compatible(unbounded, unbounded).
budget_mode_compatible(budget(GoalA), budget(GoalB)) :-
    functor(GoalA, Functor, Arity),
    functor(GoalB, Functor, Arity).

metric_increment_goal(HeadVar, RecVar, Goal0, Increment) :-
    strip_module(Goal0, _, Goal),
    (   Goal =.. [is, HeadVar, Expr]
    ;   Goal =.. ['=', HeadVar, Expr]
    ),
    additive_metric_expr(Expr, RecVar, Increment).

additive_metric_expr(Expr, RecVar, Increment) :-
    Expr =.. [+, Left, Right],
    (   Left == RecVar,
        number(Right),
        Increment = Right
    ;   Right == RecVar,
        number(Left),
        Increment = Left
    ).

strip_counted_ppv_setup_goals(Goals, VisitedVar, BudgetMode, CleanGoals) :-
    exclude(is_cut_goal, Goals, GoalsNoCuts),
    (   select(MaxGoal, GoalsNoCuts, Goals1),
        max_depth_budget_goal(MaxGoal, MaxVar),
        select(LengthGoal, Goals1, Goals2),
        length_visited_goal(LengthGoal, VisitedVar, DepthVar),
        select(CompareGoal, Goals2, Goals3),
        depth_less_goal(CompareGoal, DepthVar, MaxVar)
    ->  BudgetMode = budget(MaxGoal),
        CleanGoals = Goals3
    ;   BudgetMode = unbounded,
        CleanGoals = GoalsNoCuts
    ).

weighted_positive_step_proven(Pred/Arity, MetricPos, _BasePositives, _RecPositives) :-
    weighted_positive_step_metadata(Pred/Arity, MetricPos),
    !.
weighted_positive_step_proven(_PredArity, _MetricPos, BasePositives, RecPositives) :-
    forall(member(Positive, BasePositives), Positive == true),
    forall(member(Positive, RecPositives), Positive == true).

weighted_positive_step_metadata(Pred/Arity, Position) :-
    get_constraints(Pred/Arity, Constraints),
    member(positive_step(Position), Constraints).

weighted_min_base_clause_shape(MetricPos, VisitedPos, Head-Body, PositiveStepProven) :-
    Head =.. [_|HeadArgs],
    nth1(VisitedPos, HeadArgs, VisitedVar),
    nth1(MetricPos, HeadArgs, MetricArg),
    body_goals(Body, Goals0),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), Goals0, Goals1),
    strip_counted_ppv_setup_goals(Goals1, VisitedVar, _IgnoredBudgetMode, Goals2),
    select(MetricGoal, Goals2, Goals3),
    weighted_metric_base_goal(MetricArg, MetricGoal, StepVar),
    weighted_extract_positive_step_guard(Goals3, StepVar, Goals, PositiveStepProven),
    \+ goals_reference_samevar(Goals, VisitedVar),
    weighted_step_auxiliary_goal(HeadArgs, StepVar, Goals).

weighted_metric_base_goal(MetricArg, Goal0, StepVar) :-
    strip_module(Goal0, _, Goal),
    (   Goal =.. [is, MetricArg, StepVar]
    ;   Goal =.. ['=', MetricArg, StepVar]
    ),
    var(StepVar),
    StepVar \== MetricArg.

weighted_min_recursive_shape(Pred, Arity, VisitedPos, MetricPos, Head-Body,
        BudgetMode, PositiveStepProven) :-
    counted_ppv_clause_parts(Pred, Arity, VisitedPos, Head-Body,
        HeadArgs, PreRecGoals0, RecCall, PostGoals0, _StepGoal, _NextVar, VisitedVar),
    strip_counted_ppv_setup_goals(PreRecGoals0, VisitedVar, BudgetMode, PreRecGoals1),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), PreRecGoals1, PreRecGoals2),
    RecCall =.. [_|RecArgs],
    nth1(MetricPos, HeadArgs, HeadMetric),
    nth1(MetricPos, RecArgs, RecMetric),
    select(MetricGoal, PostGoals0, PostGoals),
    weighted_metric_recursive_goal(HeadMetric, RecMetric, MetricGoal, StepVar),
    weighted_extract_positive_step_guard(PreRecGoals2, StepVar, PreRecGoals, PositiveStepProven),
    weighted_step_auxiliary_goal(HeadArgs, StepVar, PreRecGoals),
    \+ goals_reference_samevar(PreRecGoals, VisitedVar),
    \+ goals_reference_samevar(PostGoals, VisitedVar).

weighted_metric_recursive_goal(HeadMetric, RecMetric, Goal0, StepVar) :-
    strip_module(Goal0, _, Goal),
    (   Goal =.. [is, HeadMetric, Expr]
    ;   Goal =.. ['=', HeadMetric, Expr]
    ),
    Expr =.. [+, Left, Right],
    (   Left == RecMetric,
        var(Right),
        StepVar = Right
    ;   Right == RecMetric,
        var(Left),
        StepVar = Left
    ),
    StepVar \== RecMetric.

weighted_extract_positive_step_guard(Terms, StepVar, RemainingTerms, true) :-
    select(Guard0, Terms, RemainingTerms),
    weighted_positive_step_guard(Guard0, StepVar),
    !.
weighted_extract_positive_step_guard(Terms, _StepVar, Terms, false).

weighted_positive_step_guard(Goal0, StepVar) :-
    strip_module(Goal0, _, Goal),
    (   Goal =.. [>, StepVar, Value],
        number(Value),
        Value >= 0
    ;   Goal =.. [<, Value, StepVar],
        number(Value),
        Value >= 0
    ;   Goal =.. [>=, StepVar, Value],
        number(Value),
        Value > 0
    ;   Goal =.. [=<, Value, StepVar],
        number(Value),
        Value > 0
    ).

weighted_step_auxiliary_goal(HeadArgs, StepVar, Goals) :-
    member(Goal, Goals),
    weighted_step_auxiliary_goal_shape(HeadArgs, StepVar, Goal),
    !.

weighted_step_auxiliary_goal_shape(HeadArgs, StepVar, Goal0) :-
    strip_module(Goal0, _, Goal),
    callable(Goal),
    Goal \= (\+ _),
    Goal \= not(_),
    Goal \= !,
    Goal =.. [Functor, DriverVar, AuxValue],
    Functor \= member,
    var(DriverVar),
    member_samevar(DriverVar, HeadArgs),
    AuxValue == StepVar,
    var(StepVar),
    StepVar \== DriverVar.

is_cut_goal(!).

max_depth_budget_goal(Goal0, MaxVar) :-
    strip_module(Goal0, _, Goal),
    compound(Goal),
    Goal =.. [max_depth, MaxVar],
    var(MaxVar).

length_visited_goal(Goal0, VisitedVar, DepthVar) :-
    strip_module(Goal0, _, Goal),
    Goal =.. [length, ListVar, DepthVar],
    ListVar == VisitedVar,
    var(DepthVar).

depth_less_goal(Goal0, DepthVar, MaxVar) :-
    strip_module(Goal0, _, Goal),
    Goal =.. [<, Left, Right],
    Left == DepthVar,
    Right == MaxVar.

build_min_closure_code(Pred/Arity, BaseClauses, RecClauses, SignaturePositions,
        MetricPos, VisitedPos, BaseDepth, StepIncrement, BudgetMode, Code) :-
    atom_concat(Pred, '$min', MinName),
    (   BudgetMode = unbounded
    ->  InnerName = MinName,
        OuterClauseTerms = []
    ;   atom_concat(Pred, '$min_budget', InnerName),
        OuterClauseTerms = [OuterClause],
        build_min_outer_clause(MinName, InnerName, SignaturePositions, BudgetMode, OuterClause)
    ),
    maplist(build_min_base_clause(InnerName, SignaturePositions, MetricPos, VisitedPos,
            BaseDepth, BudgetMode), BaseClauses, MinBaseClauses),
    maplist(build_min_recursive_clause(Pred, Arity, InnerName, SignaturePositions, MetricPos,
            VisitedPos, BaseDepth, StepIncrement, BudgetMode), RecClauses, MinRecClauses),
    append(MinBaseClauses, MinRecClauses, InnerClauses),
    build_min_table_decl(InnerName, SignaturePositions, BudgetMode, TableDeclCode),
    clauses_to_code(InnerClauses, InnerCode),
    clauses_to_code(OuterClauseTerms, OuterCode),
    atomic_list_concat([
        '% Mode-directed min helper for canonical shortest-path style per-path recursion.',
        TableDeclCode,
        InnerCode,
        OuterCode
    ], '\n\n', Code).

build_min_outer_clause(MinName, InnerName, SignaturePositions, budget(MaxGoalTemplate), ClauseTerm) :-
    length(SignaturePositions, SigCount),
    build_min_signature_vars(SigCount, SignatureArgs),
    append(SignatureArgs, [MetricArg], MinHeadArgs),
    MinHead =.. [MinName|MinHeadArgs],
    copy_term(MaxGoalTemplate, MaxGoal),
    MaxGoal =.. [_Functor, BudgetArg],
    append(SignatureArgs, [BudgetArg, MetricArg], InnerCallArgs),
    InnerCall =.. [InnerName|InnerCallArgs],
    Body = (MaxGoal, InnerCall),
    ClauseTerm = (MinHead :- Body).

build_min_signature_vars(0, []).
build_min_signature_vars(Count, [_|Rest]) :-
    Count > 0,
    NextCount is Count - 1,
    build_min_signature_vars(NextCount, Rest).

build_min_table_decl(InnerName, SignaturePositions, BudgetMode, Code) :-
    length(SignaturePositions, SignatureCount),
    (   BudgetMode = unbounded
    ->  KeyCount = SignatureCount
    ;   KeyCount is SignatureCount + 1
    ),
    build_plus_modes(KeyCount, KeyModes),
    append(KeyModes, [min], Modes),
    TableSpec =.. [InnerName|Modes],
    format(atom(Code), ':- table ~q.', [TableSpec]).

build_plus_modes(0, []) :- !.
build_plus_modes(Count, [+|Rest]) :-
    Count > 0,
    NextCount is Count - 1,
    build_plus_modes(NextCount, Rest).

build_min_base_clause(InnerName, SignaturePositions, MetricPos, VisitedPos, BaseDepth,
        BudgetMode, Head-Body, ClauseTerm) :-
    Head =.. [_|HeadArgs],
    nth1(VisitedPos, HeadArgs, VisitedVar),
    body_goals(Body, Goals0),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), Goals0, Goals1),
    strip_counted_ppv_setup_goals(Goals1, VisitedVar, _IgnoredBudgetMode, Goals),
    \+ goals_reference_samevar(Goals, VisitedVar),
    select_positions_1based(SignaturePositions, HeadArgs, SignatureArgs),
    nth1(MetricPos, HeadArgs, MetricArg),
    build_min_base_head(InnerName, SignatureArgs, MetricArg, BudgetMode, HeadTerm, BudgetArg),
    (   BudgetMode = unbounded
    ->  BodyGoals = Goals
    ;   BudgetGuard = (BudgetArg >= BaseDepth),
        BodyGoals = [BudgetGuard|Goals]
    ),
    goals_to_body(BodyGoals, BodyTerm),
    clause_term(HeadTerm, BodyTerm, ClauseTerm).

build_min_base_head(InnerName, SignatureArgs, MetricArg, unbounded, HeadTerm, _BudgetArg) :-
    append(SignatureArgs, [MetricArg], Args),
    HeadTerm =.. [InnerName|Args].
build_min_base_head(InnerName, SignatureArgs, MetricArg, budget(_), HeadTerm, BudgetArg) :-
    append(SignatureArgs, [BudgetArg, MetricArg], Args),
    HeadTerm =.. [InnerName|Args].

build_min_recursive_clause(Pred, Arity, InnerName, SignaturePositions, MetricPos, VisitedPos,
        BaseDepth, StepIncrement, BudgetMode, Head-Body, ClauseTerm) :-
    counted_ppv_clause_parts(Pred, Arity, VisitedPos, Head-Body,
        HeadArgs, PreRecGoals0, RecCall0, PostGoals, _StepGoal, _NextVar, VisitedVar),
    strip_counted_ppv_setup_goals(PreRecGoals0, VisitedVar, _IgnoredBudgetMode, PreRecGoals1),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), PreRecGoals1, PreRecGoals),
    \+ goals_reference_samevar(PreRecGoals, VisitedVar),
    \+ goals_reference_samevar(PostGoals, VisitedVar),
    RecCall0 =.. [_|RecArgs],
    select_positions_1based(SignaturePositions, HeadArgs, SignatureArgs),
    select_positions_1based(SignaturePositions, RecArgs, NextSignatureArgs),
    nth1(MetricPos, HeadArgs, MetricArg),
    nth1(MetricPos, RecArgs, RecMetricArg),
    build_min_recursive_head(InnerName, SignatureArgs, MetricArg, BudgetMode, HeadTerm, BudgetArg),
    build_min_recursive_call(InnerName, NextSignatureArgs, RecMetricArg, BudgetMode,
        StepIncrement, BaseDepth, RecursiveGoals, BudgetArg),
    append(PreRecGoals, RecursiveGoals, BodyGoals0),
    append(BodyGoals0, PostGoals, BodyGoals),
    goals_to_body(BodyGoals, BodyTerm),
    clause_term(HeadTerm, BodyTerm, ClauseTerm).

build_min_recursive_head(InnerName, SignatureArgs, MetricArg, unbounded, HeadTerm, _BudgetArg) :-
    append(SignatureArgs, [MetricArg], Args),
    HeadTerm =.. [InnerName|Args].
build_min_recursive_head(InnerName, SignatureArgs, MetricArg, budget(_), HeadTerm, BudgetArg) :-
    append(SignatureArgs, [BudgetArg, MetricArg], Args),
    HeadTerm =.. [InnerName|Args].

build_min_recursive_call(InnerName, NextSignatureArgs, RecMetricArg, unbounded,
        _StepIncrement, _BaseDepth, [RecursiveCall], _BudgetArg) :-
    append(NextSignatureArgs, [RecMetricArg], Args),
    RecursiveCall =.. [InnerName|Args].
build_min_recursive_call(InnerName, NextSignatureArgs, RecMetricArg, budget(_),
        StepIncrement, BaseDepth, [BudgetCalc, BudgetGuard, RecursiveCall], BudgetArg) :-
    BudgetCalc = (NextBudget is BudgetArg - StepIncrement),
    BudgetGuard = (NextBudget >= BaseDepth),
    append(NextSignatureArgs, [NextBudget, RecMetricArg], Args),
    RecursiveCall =.. [InnerName|Args].

goals_reference_samevar([Goal|_], Var) :-
    term_contains_samevar(Goal, Var),
    !.
goals_reference_samevar([_|Rest], Var) :-
    goals_reference_samevar(Rest, Var).

term_contains_samevar(Term, Var) :-
    term_variables(Term, Vars),
    member(Candidate, Vars),
    Candidate == Var,
    !.

clauses_to_pattern_pairs([], []).
clauses_to_pattern_pairs([Head-Body|Rest], [(Head, Body)|Pairs]) :-
    clauses_to_pattern_pairs(Rest, Pairs).

branch_pruning_mode_positions(Pred/Arity, VisitedPos, InputPositions) :-
    current_predicate(user:mode/1),
    user:mode(ModeSpec),
    mode_term_signature(ModeSpec, Pred/Arity),
    parse_mode_spec(ModeSpec, Modes),
    \+ member(any, Modes),
    nth1(VisitedPos, Modes, input),
    findall(Pos,
        (   nth1(Pos, Modes, input),
            Pos =\= VisitedPos
        ),
        InputPositions),
    InputPositions \= [],
    !.

mode_term_signature(Term, Pred/Arity) :-
    compound(Term),
    Term =.. [Pred|Args],
    length(Args, Arity).

parse_mode_spec(Term, Modes) :-
    Term =.. [_|Args],
    maplist(parse_mode_symbol, Args, Modes).

parse_mode_symbol(+, input) :- !.
parse_mode_symbol(-, output) :- !.
parse_mode_symbol(?, any) :- !.

is_recursive_clause_for_pred(Pred, _Head-Body) :-
    contains_predicate_call(Body, Pred).

contains_predicate_call(Goal0, Pred) :-
    strip_module(Goal0, _, Goal),
    (   compound(Goal),
        Goal =.. [Pred|_]
    ->  true
    ;   Goal = (A, B)
    ->  (contains_predicate_call(A, Pred) ; contains_predicate_call(B, Pred))
    ;   Goal = (A ; B)
    ->  (contains_predicate_call(A, Pred) ; contains_predicate_call(B, Pred))
    ;   Goal = (A -> B)
    ->  (contains_predicate_call(A, Pred) ; contains_predicate_call(B, Pred))
    ;   Goal = (A -> B ; C)
    ->  ( contains_predicate_call(A, Pred)
        ; contains_predicate_call(B, Pred)
        ; contains_predicate_call(C, Pred)
        )
    ;   false
    ).

branch_pruning_driver_position(Pred, Arity, RecClauses, DriverPos) :-
    maplist(branch_pruning_clause_driver_position(Pred, Arity), RecClauses, DriverPositions),
    sort(DriverPositions, [DriverPos]).

branch_pruning_clause_driver_position(Pred, Arity, Head-Body, DriverPos) :-
    Head =.. [Pred|HeadArgs],
    body_goals(Body, [StepGoal|_]),
    step_goal_input_var(StepGoal, StepInputVar),
    findall(Pos,
        (   between(1, Arity, Pos),
            nth1(Pos, HeadArgs, Arg),
            Arg == StepInputVar
        ),
        [DriverPos]).

step_goal_input_var(Goal0, InputVar) :-
    strip_module(Goal0, _, Goal),
    compound(Goal),
    \+ branch_pruning_negated_member_goal(Goal, _),
    arg(1, Goal, InputVar),
    var(InputVar).

branch_pruning_invariants_static(_, _, [], _, _).
branch_pruning_invariants_static(Pred, Arity, [Head-Body|Rest], VisitedPos, InvariantPositions) :-
    Head =.. [Pred|HeadArgs],
    nth1(VisitedPos, HeadArgs, VisitedVar),
    body_goals(Body, Goals),
    recursive_prefix_goals(Pred, Arity, Goals, VisitedVar, _PrefixGoals, RecCall),
    RecCall =.. [_|RecArgs],
    forall(member(Pos, InvariantPositions),
        (   nth1(Pos, HeadArgs, HeadArg),
            nth1(Pos, RecArgs, RecArg),
            HeadArg == RecArg
        )),
    branch_pruning_invariants_static(Pred, Arity, Rest, VisitedPos, InvariantPositions).

build_branch_pruning_code(Pred/Arity, Clauses, BaseClauses, RecClauses,
        VisitedPos, DriverPos, InputPositions, InvariantPositions, Code) :-
    atom_concat(Pred, '$worker', WorkerName),
    atom_concat(Pred, '$pruned', PrunedName),
    atom_concat(Pred, '$prune', PruneName),
    atom_concat(Pred, '$prune_guard', PruneGuardName),
    atom_concat(Pred, '$prune_cache', PruneCacheName),
    branch_pruning_signature_positions(DriverPos, InvariantPositions, SignaturePositions),
    length(SignaturePositions, PruneArity),
    build_branch_pruning_wrapper(Pred/Arity, WorkerName, PrunedName, InputPositions, WrapperClause),
    build_branch_pruning_guard(PruneGuardName, PruneName, PruneCacheName, PruneArity, GuardClauses),
    maplist(build_prune_base_clause(Pred, PruneName, SignaturePositions, VisitedPos), BaseClauses, PruneBaseClauses),
    maplist(build_prune_recursive_clause(Pred, Arity, PruneName, PruneGuardName,
            SignaturePositions, VisitedPos), RecClauses, PruneRecClauses),
    append(PruneBaseClauses, PruneRecClauses, PruneClauses),
    maplist(rename_clause_for_worker(Pred, Arity, WorkerName, none), Clauses, WorkerClauses),
    guard_spec(SignaturePositions, PruneGuardName, GuardSpec),
    maplist(rename_clause_for_worker(Pred, Arity, PrunedName, GuardSpec), Clauses, PrunedClauses),
    format(atom(TableDeclCode), ':- table ~q/~d.', [PruneName, PruneArity]),
    format(atom(CacheDeclCode), ':- dynamic ~q/~d.', [PruneCacheName, PruneArity]),
    clauses_to_code(GuardClauses, GuardCode),
    clauses_to_code(PruneClauses, PruneCode),
    clauses_to_code(WorkerClauses, WorkerCode),
    clauses_to_code(PrunedClauses, PrunedCode),
    clause_to_string(WrapperClause, WrapperCode),
    atomic_list_concat([
        '% Branch-pruned Prolog generated from a parameterized per-path recursion.',
        TableDeclCode,
        CacheDeclCode,
        GuardCode,
        PruneCode,
        WorkerCode,
        PrunedCode,
        WrapperCode
    ], '\n\n', Code).

branch_pruning_signature_positions(DriverPos, InvariantPositions, [DriverPos|InvariantPositions]).

build_branch_pruning_wrapper(Pred/Arity, WorkerName, PrunedName, InputPositions, ClauseTerm) :-
    functor(Head, Pred, Arity),
    Head =.. [Pred|Args],
    build_nonvar_guard(InputPositions, Args, BoundGoal),
    WorkerCall =.. [WorkerName|Args],
    PrunedCall =.. [PrunedName|Args],
    ClauseTerm = (Head :- (BoundGoal -> PrunedCall ; WorkerCall)).

build_nonvar_guard([Pos], Args, nonvar(Arg)) :-
    !,
    nth1(Pos, Args, Arg).
build_nonvar_guard([Pos|Rest], Args, (nonvar(Arg), Tail)) :-
    nth1(Pos, Args, Arg),
    build_nonvar_guard(Rest, Args, Tail).

build_branch_pruning_guard(GuardName, PruneName, CacheName, Arity, [ClauseTerm]) :-
    functor(Head, GuardName, Arity),
    Head =.. [GuardName|Args],
    CacheCall =.. [CacheName|Args],
    RawCall =.. [PruneName|Args],
    GroundCheck = ground(Args),
    Body =
        (   GroundCheck
        ->  (   CacheCall
            ->  true
            ;   once(RawCall),
                (   CacheCall
                ->  true
                ;   assertz(CacheCall)
                )
            )
        ;   RawCall
        ),
    ClauseTerm = (Head :- Body).

build_prune_base_clause(Pred, PruneName, SignaturePositions, VisitedPos, Head-Body, ClauseTerm) :-
    Head =.. [Pred|HeadArgs],
    nth1(VisitedPos, HeadArgs, VisitedVar),
    body_goals(Body, Goals),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), Goals, BodyGoals),
    BodyGoals \= [],
    select_positions_1based(SignaturePositions, HeadArgs, PruneArgs),
    PruneHead =.. [PruneName|PruneArgs],
    goals_to_body(BodyGoals, PruneBody),
    clause_term(PruneHead, PruneBody, ClauseTerm).

build_prune_recursive_clause(Pred, Arity, PruneName, PruneGuardName,
        SignaturePositions, VisitedPos, Head-Body, ClauseTerm) :-
    Head =.. [Pred|HeadArgs],
    nth1(VisitedPos, HeadArgs, VisitedVar),
    body_goals(Body, Goals),
    recursive_prefix_goals(Pred, Arity, Goals, VisitedVar, PrefixGoals, RecCall),
    PrefixGoals \= [],
    RecCall =.. [_|RecArgs],
    select_positions_1based(SignaturePositions, HeadArgs, PruneHeadArgs),
    select_positions_1based(SignaturePositions, RecArgs, NextPruneArgs),
    PruneGuardCall =.. [PruneGuardName|NextPruneArgs],
    append(PrefixGoals, [PruneGuardCall], PruneGoals),
    PruneHead =.. [PruneName|PruneHeadArgs],
    goals_to_body(PruneGoals, PruneBody),
    clause_term(PruneHead, PruneBody, ClauseTerm).

recursive_prefix_goals(Pred, Arity, Goals, VisitedVar, PrefixGoals, RecCall) :-
    append(Prefix0, [RecCall|_], Goals),
    branch_pruning_recursive_goal(Pred, Arity, RecCall),
    exclude(branch_pruning_negated_member_goal_for(VisitedVar), Prefix0, PrefixGoals).

branch_pruning_recursive_goal(Pred, Arity, Goal0) :-
    strip_module(Goal0, _, Goal),
    compound(Goal),
    functor(Goal, Pred, GoalArity),
    GoalArity =:= Arity.

branch_pruning_negated_member_goal_for(VisitedVar, Goal) :-
    branch_pruning_negated_member_goal(Goal, VisitedVar).

branch_pruning_negated_member_goal(Goal0, VisitedVar) :-
    strip_module(Goal0, _, Goal),
    (   Goal =.. ['\\+', Inner],
        Inner =.. [member, _, Var]
    ;   Goal =.. [not, Inner],
        Inner =.. [member, _, Var]
    ),
    Var == VisitedVar.

rename_clause_for_worker(Pred, Arity, NewName, GuardSpec, Head-Body0, ClauseTerm) :-
    Head =.. [_|HeadArgs],
    NewHead =.. [NewName|HeadArgs],
    rename_recursive_calls(Body0, Pred, Arity, NewName, Body1),
    (   GuardSpec = none
    ->  FinalBody = Body1
    ;   GuardSpec = guard(SignaturePositions, GuardName),
        select_positions_1based(SignaturePositions, HeadArgs, GuardArgs),
        GuardCall =.. [GuardName|GuardArgs],
        prepend_goal(GuardCall, Body1, FinalBody)
    ),
    clause_term(NewHead, FinalBody, ClauseTerm).

guard_spec(SignaturePositions, GuardName, guard(SignaturePositions, GuardName)).

rename_recursive_calls(Goal0, Pred, Arity, NewName, Goal) :-
    nonvar(Goal0),
    Goal0 = Module:Inner0,
    !,
    rename_recursive_calls(Inner0, Pred, Arity, NewName, Renamed),
    (   (   Module == user
        ;   Module == prolog_target
        )
    ->  Goal = Renamed
    ;   Goal = Module:Renamed
    ).
rename_recursive_calls(Goal0, Pred, Arity, NewName, Goal) :-
    rename_recursive_calls_plain(Goal0, Pred, Arity, NewName, Goal).

rename_recursive_calls_plain(true, _Pred, _Arity, _NewName, true) :- !.
rename_recursive_calls_plain((A0, B0), Pred, Arity, NewName, (A, B)) :-
    !,
    rename_recursive_calls(A0, Pred, Arity, NewName, A),
    rename_recursive_calls(B0, Pred, Arity, NewName, B).
rename_recursive_calls_plain((A0 ; B0), Pred, Arity, NewName, (A ; B)) :-
    !,
    rename_recursive_calls(A0, Pred, Arity, NewName, A),
    rename_recursive_calls(B0, Pred, Arity, NewName, B).
rename_recursive_calls_plain((A0 -> B0), Pred, Arity, NewName, (A -> B)) :-
    !,
    rename_recursive_calls(A0, Pred, Arity, NewName, A),
    rename_recursive_calls(B0, Pred, Arity, NewName, B).
rename_recursive_calls_plain(Goal0, Pred, Arity, NewName, Goal) :-
    compound(Goal0),
    functor(Goal0, Pred, GoalArity),
    GoalArity =:= Arity,
    !,
    Goal0 =.. [_|Args],
    Goal =.. [NewName|Args].
rename_recursive_calls_plain(Goal, _Pred, _Arity, _NewName, Goal).

prepend_goal(Goal, true, Goal) :- !.
prepend_goal(Goal, Body, (Goal, Body)).

body_goals(true, []) :- !.
body_goals(Body0, Goals) :-
    strip_module(Body0, _, Body),
    body_goals_plain(Body, Goals).

body_goals_plain(true, []) :- !.
body_goals_plain((A, B), Goals) :-
    !,
    body_goals_plain(A, GoalsA),
    body_goals_plain(B, GoalsB),
    append(GoalsA, GoalsB, Goals).
body_goals_plain(Goal, [Goal]).

goals_to_body([], true).
goals_to_body([Goal], Goal) :- !.
goals_to_body([Goal|Rest], (Goal, Tail)) :-
    goals_to_body(Rest, Tail).

clause_term(Head, true, Head) :- !.
clause_term(Head, Body, (Head :- Body)).

clauses_to_code(Clauses, Code) :-
    maplist(clause_to_string, Clauses, ClauseStrings),
    atomic_list_concat(ClauseStrings, '\n', Code).

select_positions_1based([], _List, []).
select_positions_1based([Pos|Rest], List, [Elem|Elems]) :-
    nth1(Pos, List, Elem),
    select_positions_1based(Rest, List, Elems).

bind_positions_1based([], [], _List).
bind_positions_1based([Pos|RestPos], [Elem|RestElems], List) :-
    nth1(Pos, List, Elem),
    bind_positions_1based(RestPos, RestElems, List).

%% generate_entry_point(+Options, -Code)
%  Generate main entry point and initialization
generate_entry_point(Options, Code) :-
    % Determine entry point goal
    (   member(entry_point(EntryGoal), Options)
    ->  format(atom(EntryGoalStr), '~w', [EntryGoal])
    ;   EntryGoalStr = 'main'
    ),

    % Determine argument handling
    (   member(arguments(ArgSpec), Options)
    ->  generate_argument_parsing(ArgSpec, ArgParseCode)
    ;   ArgParseCode = '    % No argument parsing'
    ),

    % Build entry goal line
    format(atom(EntryGoalLine), '    ~w,', [EntryGoalStr]),

    % Build lines
    atomic_list_concat([
        '% === Entry Point ===',
        'main :-',
        ArgParseCode,
        EntryGoalLine,
        '    halt(0).',
        '',
        'main :-',
        '    % If main goal fails, exit with error',
        '    format(user_error, \'Error: Execution failed~n\', []),',
        '    halt(1).',
        '',
        ':- initialization(main, main).'
    ], '\n', Code).

%% generate_argument_parsing(+ArgSpec, -Code)
%  Generate code to parse command-line arguments
generate_argument_parsing(args(VarList), Code) :-
    % VarList = [file, option1, option2, ...]
    length(VarList, NumArgs),
    format(atom(Code),
           'current_prolog_flag(argv, Argv),~n~
                (   length(Argv, ~w)~n~
                ->  Argv = ~w~n~
                ;   format(user_error, \'Usage: ~w~n\', []),~n~
                    halt(1)~n~
                )',
           [NumArgs, VarList, VarList]).

generate_argument_parsing(optional_args(VarList, Defaults), Code) :-
    % With defaults for optional arguments
    format(atom(Code),
           'current_prolog_flag(argv, Argv),~n~
                parse_arguments(Argv, ~w, ~w)',
           [VarList, Defaults]).

%% ============================================
%% DEPENDENCY ANALYSIS
%% ============================================

%% analyze_dependencies(+UserPredicates, -Dependencies)
%  Analyze user code to determine runtime dependencies
%
%  @arg UserPredicates List of Pred/Arity
%  @arg Dependencies List of dependency terms:
%       - module(ModulePath) - Module to import
%       - ensure_loaded(Path) - File to ensure_loaded
%       - plugin_registration(Type, Name, Module) - Plugin to register
%
%  @example Detect partitioner usage
%    ?- analyze_dependencies([process/2], Deps).
%    Deps = [module(unifyweaver(core/partitioner)), ...].
analyze_dependencies(UserPredicates, Dependencies) :-
    % Analyze each predicate's body for feature usage
    findall(Dep, (
        member(Pred/Arity, UserPredicates),
        functor(Head, Pred, Arity),
        clause(Head, Body),
        extract_dependencies_from_body(Body, Dep)
    ), AllDeps),

    % Remove duplicates
    sort(AllDeps, Dependencies).

augment_generated_dependencies(UserPredicates, Options, Dependencies0, Dependencies) :-
    (   generated_helper_requires_aggregate(UserPredicates, Options)
    ->  sort([module(library(aggregate))|Dependencies0], Dependencies)
    ;   Dependencies = Dependencies0
    ).

generated_helper_requires_aggregate(UserPredicates, Options) :-
    member(Pred/Arity, UserPredicates),
    maybe_generate_seeded_accumulation_helper(Pred/Arity, Options, _),
    !.

%% extract_dependencies_from_body(+Body, -Dependency)
%  Extract dependencies from clause body
extract_dependencies_from_body(Body, Dependency) :-
    % Check for known patterns
    contains_goal(Body, Goal),
    goal_requires_dependency(Goal, Dependency).

%% contains_goal(+Body, -Goal)
%  Recursively extract goals from clause body
contains_goal(Goal, Goal) :-
    callable(Goal).

contains_goal((A, B), Goal) :-
    !,
    (   contains_goal(A, Goal)
    ;   contains_goal(B, Goal)
    ).

contains_goal((A ; B), Goal) :-
    !,
    (   contains_goal(A, Goal)
    ;   contains_goal(B, Goal)
    ).

contains_goal((A -> B ; C), Goal) :-
    !,
    (   contains_goal(A, Goal)
    ;   contains_goal(B, Goal)
    ;   contains_goal(C, Goal)
    ).

%% goal_requires_dependency(+Goal, -Dependency)
%  Map goal patterns to required dependencies
%
%  This is where we detect feature usage and map to runtime modules

% Partitioning features
goal_requires_dependency(partitioner_init(_, _, _), module(unifyweaver(core/partitioner))).
goal_requires_dependency(partitioner_partition(_, _, _), module(unifyweaver(core/partitioner))).
goal_requires_dependency(partition_data(_, _, _), module(unifyweaver(core/partitioner))).

% Parallel backend features
goal_requires_dependency(backend_init(_, _), module(unifyweaver(core/parallel_backend))).
goal_requires_dependency(backend_execute(_, _, _, _), module(unifyweaver(core/parallel_backend))).
goal_requires_dependency(parallel_map(_, _, _), module(unifyweaver(core/parallel_backend))).

% Data source features
goal_requires_dependency(read_csv(_, _), module(unifyweaver(sources/csv))).
goal_requires_dependency(read_json(_, _), module(unifyweaver(sources/json))).
goal_requires_dependency(http_get(_, _), module(unifyweaver(sources/http))).

% Detect strategy/backend registration needs
goal_requires_dependency(partitioner_init(Strategy, _, _), StrategyDep) :-
    Strategy =.. [StrategyName|_],
    strategy_requires_module(StrategyName, StrategyDep).

goal_requires_dependency(backend_init(Backend, _), BackendDep) :-
    Backend =.. [BackendName|_],
    backend_requires_module(BackendName, BackendDep).

%% strategy_requires_module(+StrategyName, -Dependency)
%  Map strategy name to required module
strategy_requires_module(fixed_size, ensure_loaded(unifyweaver(core/partitioners/fixed_size))).
strategy_requires_module(fixed_size, plugin_registration(partitioner, fixed_size, fixed_size_partitioner)).

strategy_requires_module(hash_based, ensure_loaded(unifyweaver(core/partitioners/hash_based))).
strategy_requires_module(hash_based, plugin_registration(partitioner, hash_based, hash_based_partitioner)).

strategy_requires_module(key_based, ensure_loaded(unifyweaver(core/partitioners/key_based))).
strategy_requires_module(key_based, plugin_registration(partitioner, key_based, key_based_partitioner)).

%% backend_requires_module(+BackendName, -Dependency)
%  Map backend name to required module
backend_requires_module(gnu_parallel, ensure_loaded(unifyweaver(core/backends/gnu_parallel))).
backend_requires_module(gnu_parallel, plugin_registration(backend, gnu_parallel, gnu_parallel_backend)).

%% ============================================
%% OUTPUT GENERATION
%% ============================================

%%generate_main_predicate(+Options, -MainPred)
%  Generate main/0 predicate from options
generate_main_predicate(Options, MainPred) :-
    option(entry_point(EntryGoal), Options, main),
    option(arguments(ArgSpec), Options, none),

    (   ArgSpec = none
    ->  ArgParseCode = ''
    ;   generate_argument_parsing(ArgSpec, ArgParseCode)
    ),

    format(atom(EntryGoalStr), '~w', [EntryGoal]),
    format(atom(EntryGoalLine), '    ~w,', [EntryGoalStr]),

    atomic_list_concat([
        'main :-',
        ArgParseCode,
        EntryGoalLine,
        '    halt(0).',
        '',
        'main :-',
        '    format(user_error, \'Error: Execution failed~n\', []),',
        '    halt(1).'
    ], '\n', MainPred).

%% write_prolog_script(+ScriptCode, +OutputPath)
%  Write generated script to file and make executable
write_prolog_script(ScriptCode, OutputPath) :-
    write_prolog_script(ScriptCode, OutputPath, []).

%% write_prolog_script(+ScriptCode, +OutputPath, +Options)
%  Write script with compilation support
write_prolog_script(ScriptCode, OutputPath, Options) :-
    % Write script
    open(OutputPath, write, Stream, [encoding(utf8)]),
    write(Stream, ScriptCode),
    nl(Stream),
    close(Stream),

    % Make executable (chmod +x)
    format(atom(ChmodCmd), 'chmod +x ~w', [OutputPath]),
    shell(ChmodCmd),

    format('[PrologTarget] Generated script: ~w~n', [OutputPath]),

    % Optionally compile for dialects that support it
    (   option(compile(true), Options),
        option(dialect(Dialect), Options)
    ->  compile_script_safe(Dialect, OutputPath, Options)
    ;   true
    ).

%% compile_script(+Dialect, +ScriptPath)
%  Compile script using dialect-specific compiler
compile_script(Dialect, ScriptPath) :-
    (   dialect_compile_command(Dialect, ScriptPath, CompileCmd)
    ->  format('[PrologTarget] Compiling with ~w: ~w~n', [Dialect, CompileCmd]),
        shell(CompileCmd, ExitCode),
        (   ExitCode = 0
        ->  format('[PrologTarget] Compilation complete~n')
        ;   format('[PrologTarget] ERROR: Compilation failed with exit code ~w~n', [ExitCode]),
            throw(error(compilation_failed(Dialect, ExitCode),
                       context(compile_script/2, 'Compiler returned non-zero exit code')))
        )
    ;   format('[PrologTarget] Dialect ~w does not support compilation~n', [Dialect])
    ).

%% compile_script_safe(+Dialect, +ScriptPath, +Options)
%  Compile script with graceful failure handling
%
%  If compilation fails, checks if fallback is enabled and attempts
%  alternative dialects. For v0.1, logs warnings but continues.
%
%  TODO (v0.2): Implement full multi-dialect fallback that regenerates
%  code for alternative dialects when compilation fails.
compile_script_safe(Dialect, ScriptPath, Options) :-
    catch(
        compile_script(Dialect, ScriptPath),
        error(compilation_failed(FailedDialect, ExitCode), Context),
        (   % Compilation failed - handle gracefully
            format('[PrologTarget] WARNING: ~w compilation failed (exit ~w)~n',
                   [FailedDialect, ExitCode]),

            % Check if we should fail or continue
            (   option(fail_on_compile_error(true), Options)
            ->  % Propagate error if strict mode
                throw(error(compilation_failed(FailedDialect, ExitCode), Context))
            ;   % Otherwise continue with interpreted script
                format('[PrologTarget] Continuing with interpreted script: ~w~n', [ScriptPath]),
                format('[PrologTarget] NOTE: Full multi-dialect fallback planned for v0.2~n')
            )
        )
    ).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% is_user_predicate(+Pred/Arity)
%  Check if predicate is user-defined (not built-in or library)
is_user_predicate(Pred/Arity) :-
    functor(Head, Pred, Arity),
    % Has at least one clause
    clause(Head, _),
    % Not from system modules
    \+ predicate_property(Head, built_in),
    \+ predicate_property(Head, imported_from(_)).
