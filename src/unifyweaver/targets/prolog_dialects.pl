:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% prolog_dialects.pl - Prolog Dialect Support for Target Generation
%
% Supports multiple Prolog implementations with different capabilities:
% - SWI-Prolog: Full-featured, interpreted, extensive libraries
% - GNU Prolog: Compiled via gplc, constraint solving, stack-optimized
%
% Philosophy:
% - Each dialect has different strengths and limitations
% - Generated code must be dialect-compatible
% - Graceful degradation when features unavailable
% - Clear error messages for unsupported features

:- module(prolog_dialects, [
    supported_dialect/1,           % +Dialect - Check if dialect supported
    dialect_capabilities/2,        % +Dialect, -Capabilities
    dialect_shebang/2,             % +Dialect, -ShebangLine
    dialect_header/3,              % +Dialect, +Options, -HeaderCode
    dialect_imports/3,             % +Dialect, +Dependencies, -ImportCode
    dialect_initialization/4,      % +Dialect, +EntryGoal, +Options, -InitCode
    dialect_compile_command/3,     % +Dialect, +ScriptPath, -CompileCmd
    validate_for_dialect/3,        % +Dialect, +Predicates, -Issues
    % Dialect alias configuration
    set_prolog_default/1,          % +DialectOrStrategy - Set default for 'prolog'
    get_prolog_default/1,          % -DialectOrStrategy - Get current default
    expand_prolog_alias/2          % +Alias, -ConcreteDialects
]).

:- use_module(library(lists)).
:- use_module(library(option)).

%% ============================================
%% DIALECT REGISTRY
%% ============================================

%% supported_dialect(+Dialect)
%  Check if a Prolog dialect is supported
%
%  @arg Dialect One of: swi, gnu, scryer, trealla
supported_dialect(swi).
supported_dialect(gnu).

%% dialect_info(?Dialect, ?Property, ?Value)
%  Dialect metadata and capabilities
%
%  Properties:
%  - name(FullName) - Human-readable name
%  - version_flag(Flag) - How to get version
%  - compilation(Method) - interpreted, compiled, both
%  - constraint_solver(Type) - clpfd, fd, none
%  - module_system(Type) - full, basic, none
dialect_info(swi, name, 'SWI-Prolog').
dialect_info(swi, version_flag, 'swipl --version').
dialect_info(swi, compilation, interpreted).
dialect_info(swi, constraint_solver, clpfd).
dialect_info(swi, module_system, full).
dialect_info(swi, io_capabilities, comprehensive).
dialect_info(swi, library_support, extensive).

dialect_info(gnu, name, 'GNU Prolog').
dialect_info(gnu, version_flag, 'gprolog --version').
dialect_info(gnu, compilation, compiled).
dialect_info(gnu, constraint_solver, fd).
dialect_info(gnu, module_system, basic).
dialect_info(gnu, io_capabilities, basic).
dialect_info(gnu, library_support, limited).
dialect_info(gnu, compiler, gplc).

%% ============================================
%% CAPABILITIES
%% ============================================

%% dialect_capabilities(+Dialect, -Capabilities)
%  Get list of capability terms for a dialect
%
%  @example
%    ?- dialect_capabilities(gnu, Caps).
%    Caps = [compiled, constraint_solver(fd), ...].
dialect_capabilities(Dialect, Capabilities) :-
    supported_dialect(Dialect),
    findall(Cap, dialect_capability(Dialect, Cap), Capabilities).

dialect_capability(Dialect, name(Name)) :-
    dialect_info(Dialect, name, Name).

dialect_capability(Dialect, compilation(Method)) :-
    dialect_info(Dialect, compilation, Method).

dialect_capability(Dialect, constraint_solver(Type)) :-
    dialect_info(Dialect, constraint_solver, Type).

dialect_capability(Dialect, module_system(Type)) :-
    dialect_info(Dialect, module_system, Type).

%% ============================================
%% SHEBANG GENERATION
%% ============================================

%% dialect_shebang(+Dialect, -ShebangLine)
%  Generate appropriate shebang line for dialect
%
%  SWI-Prolog: #!/usr/bin/env swipl
%  GNU Prolog: #!/usr/bin/env gprolog --consult-file (for interpreted)
%              No shebang for compiled binaries
dialect_shebang(swi, '#!/usr/bin/env swipl').

dialect_shebang(gnu, ShebangLine) :-
    % GNU Prolog can run interpreted or compiled
    % For interpreted scripts, use --consult-file
    ShebangLine = '#!/usr/bin/env gprolog --consult-file'.

%% ============================================
%% HEADER GENERATION
%% ============================================

%% dialect_header(+Dialect, +Options, -HeaderCode)
%  Generate dialect-specific header comments
dialect_header(Dialect, Options, HeaderCode) :-
    dialect_info(Dialect, name, DialectName),
    get_time(Timestamp),
    format_time(atom(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),

    option(predicates(UserPredicates), Options, []),
    length(UserPredicates, NumPreds),

    option(compilation_mode(Mode), Options, interpreted),

    format(atom(Line1), '% Target: Prolog (~w)', [DialectName]),
    format(atom(Line2), '% Compilation: ~w', [Mode]),
    format(atom(Line3), '% Generated: ~w', [DateStr]),
    format(atom(Line4), '% Predicates: ~w', [NumPreds]),

    format(atom(HeaderCode), '~w~n~w~n~w~n~w~n~w~n~w',
        ['% Generated by UnifyWeaver v0.1',
         Line1,
         Line2,
         Line3,
         Line4,
         '% ']).

%% ============================================
%% IMPORT GENERATION
%% ============================================

%% dialect_imports(+Dialect, +Dependencies, -ImportCode)
%  Generate dialect-specific import statements
%
%  SWI-Prolog: Uses :- use_module(library(...))
%  GNU Prolog: Uses include directive, limited module support
dialect_imports(swi, Dependencies, ImportCode) :-
    % SWI-Prolog uses full module system
    generate_swi_imports(Dependencies, ImportCode).

dialect_imports(gnu, Dependencies, ImportCode) :-
    % GNU Prolog has basic module system
    % Use include for files, built-in predicates for standard library
    generate_gnu_imports(Dependencies, ImportCode).

%% generate_swi_imports(+Dependencies, -Code)
%  Generate SWI-Prolog import statements
generate_swi_imports(Dependencies, Code) :-
    % Setup search path
    SearchPathSetup = [
        '% Set up UnifyWeaver runtime library search path',
        ':- multifile user:file_search_path/2.',
        ':- dynamic user:file_search_path/2.',
        '',
        'setup_unifyweaver_path :-',
        '    getenv(\'UNIFYWEAVER_HOME\', Home), !,',
        '    asserta(user:file_search_path(unifyweaver, Home)).',
        'setup_unifyweaver_path.',
        '',
        ':- setup_unifyweaver_path.',
        ''
    ],

    % Convert dependencies to use_module statements
    findall(Import, (
        member(Dep, Dependencies),
        swi_dependency_import(Dep, Import)
    ), Imports),

    append(SearchPathSetup, Imports, Lines),
    atomic_list_concat(Lines, '\n', Code).

swi_dependency_import(module(ModulePath), Import) :-
    format(atom(Import), ':- use_module(~w).', [ModulePath]).

swi_dependency_import(ensure_loaded(FilePath), Import) :-
    format(atom(Import), ':- ensure_loaded(~w).', [FilePath]).

%% generate_gnu_imports(+Dependencies, -Code)
%  Generate GNU Prolog import statements
generate_gnu_imports(Dependencies, Code) :-
    % GNU Prolog uses include/1 for file loading
    % Standard library predicates are built-in
    findall(Import, (
        member(Dep, Dependencies),
        gnu_dependency_import(Dep, Import)
    ), Imports),

    (   Imports = []
    ->  Code = '% No external dependencies'
    ;   atomic_list_concat(Imports, '\n', Code)
    ).

gnu_dependency_import(module(ModulePath), Import) :-
    % Convert module path to file path
    module_to_file_path(ModulePath, FilePath),
    format(atom(Import), ':- include(~w).', [FilePath]).

gnu_dependency_import(ensure_loaded(FilePath), Import) :-
    format(atom(Import), ':- include(~w).', [FilePath]).

%% module_to_file_path(+ModulePath, -FilePath)
%  Convert SWI-style module path to file path
module_to_file_path(library(Path), FilePath) :-
    !,
    format(atom(FilePath), '\'~w.pl\'', [Path]).

module_to_file_path(Path, FilePath) :-
    format(atom(FilePath), '\'~w.pl\'', [Path]).

%% ============================================
%% INITIALIZATION
%% ============================================

%% dialect_initialization(+Dialect, +EntryGoal, +Options, -InitCode)
%  Generate dialect-specific initialization code
%
%  For GNU Prolog:
%  - Compiled mode (--no-top-level): use :- initialization(Goal)
%  - Interpreted mode: use :- Goal
dialect_initialization(swi, EntryGoal, _Options, InitCode) :-
    % SWI-Prolog uses initialization/2 directive
    format(atom(GoalStr), '~w', [EntryGoal]),
    format(atom(InitCode), ':- initialization(~w, main).', [GoalStr]).

dialect_initialization(gnu, EntryGoal, Options, InitCode) :-
    format(atom(GoalStr), '~w', [EntryGoal]),
    (   option(compile(true), Options)
    ->  % Compiled mode: use initialization/1 for --no-top-level compatibility
        format(atom(InitLine), ':- initialization(~w).', [GoalStr]),
        Comment = '% Entry point (for compiled binary)'
    ;   % Interpreted mode: execute on load
        format(atom(InitLine), ':- ~w.', [GoalStr]),
        Comment = '% Entry point (called on load)'
    ),
    atomic_list_concat([Comment, InitLine], '\n', InitCode).

%% ============================================
%% COMPILATION SUPPORT
%% ============================================

%% dialect_compile_command(+Dialect, +ScriptPath, -CompileCmd)
%  Generate command to compile script for dialect
%
%  SWI-Prolog: Interpreted, no compilation (return fail)
%  GNU Prolog: Use gplc to compile to native binary
dialect_compile_command(swi, _ScriptPath, _CompileCmd) :-
    % SWI-Prolog is interpreted
    fail.

dialect_compile_command(gnu, ScriptPath, CompileCmd) :-
    % GNU Prolog compilation with gplc
    % Output binary has same name without .pl extension
    atom_concat(BaseName, '.pl', ScriptPath),
    !,
    format(atom(CompileCmd), 'gplc --no-top-level ~w -o ~w', [ScriptPath, BaseName]).

dialect_compile_command(gnu, ScriptPath, CompileCmd) :-
    % Fallback if no .pl extension
    format(atom(CompileCmd), 'gplc --no-top-level ~w', [ScriptPath]).

%% ============================================
%% VALIDATION
%% ============================================

%% validate_for_dialect(+Dialect, +Predicates, -Issues)
%  Check if predicates are compatible with dialect
%
%  Returns list of issue terms:
%  - unsupported_feature(Feature, Predicate)
%  - limited_support(Feature, Predicate, Workaround)
%  - warning(Message, Predicate)
validate_for_dialect(swi, _Predicates, []) :-
    % SWI-Prolog supports everything
    !.

validate_for_dialect(gnu, Predicates, Issues) :-
    % Check for GNU Prolog limitations
    findall(Issue, (
        member(Pred/Arity, Predicates),
        check_gnu_compatibility(Pred/Arity, Issue)
    ), Issues).

%% check_gnu_compatibility(+Pred/Arity, -Issue)
%  Check predicate compatibility with GNU Prolog
check_gnu_compatibility(Pred/Arity, Issue) :-
    functor(Head, Pred, Arity),
    clause(Head, Body),
    contains_incompatible_goal(Body, Goal),
    format(atom(Issue), 'unsupported_feature(~w, ~w/~w)', [Goal, Pred, Arity]).

%% contains_incompatible_goal(+Body, -Goal)
%  Find goals incompatible with GNU Prolog
contains_incompatible_goal(Goal, Goal) :-
    is_gnu_incompatible(Goal).

contains_incompatible_goal((A, B), Goal) :-
    !,
    (   contains_incompatible_goal(A, Goal)
    ;   contains_incompatible_goal(B, Goal)
    ).

contains_incompatible_goal((A ; B), Goal) :-
    !,
    (   contains_incompatible_goal(A, Goal)
    ;   contains_incompatible_goal(B, Goal)
    ).

%% is_gnu_incompatible(+Goal)
%  Goals that don't work in GNU Prolog
is_gnu_incompatible(setup_call_cleanup(_, _, _)).
is_gnu_incompatible(with_output_to(_, _)).
is_gnu_incompatible(thread_create(_, _, _)).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% dialect_available(+Dialect)
%  Check if dialect executable is available on system
dialect_available(swi) :-
    catch(
        shell('which swipl > /dev/null 2>&1'),
        _,
        fail
    ).

dialect_available(gnu) :-
    catch(
        shell('which gprolog > /dev/null 2>&1'),
        _,
        fail
    ).

%% recommend_dialect(+Predicates, -Dialect, -Reason)
%  Recommend best dialect for given predicates
recommend_dialect(Predicates, gnu, 'Stack-based algorithm suitable for compilation') :-
    % If predicates are tail-recursive and use arithmetic
    % GNU Prolog compilation is beneficial
    member(Pred/Arity, Predicates),
    functor(Head, Pred, Arity),
    clause(Head, Body),
    is_tail_recursive(Body),
    contains_arithmetic(Body),
    !.

recommend_dialect(_Predicates, swi, 'Default: Full feature support').

%% Helper predicates for recommendation
is_tail_recursive(Body) :-
    % Simplified check - look for recursive call in tail position
    Body = (_, _),
    !,
    Body = (_, RecCall),
    callable(RecCall).

is_tail_recursive(Body) :-
    callable(Body).

contains_arithmetic(Body) :-
    contains_goal(Body, (_ is _)).

contains_goal(Goal, Goal).
contains_goal((A, B), Goal) :-
    !,
    (   contains_goal(A, Goal)
    ;   contains_goal(B, Goal)
    ).

%% ============================================
%% DIALECT ALIAS CONFIGURATION
%% ============================================

%% prolog_default_strategy(?Strategy)
%  Configurable default for the 'prolog' alias
%
%  Strategies:
%  - swi                  % Use SWI-Prolog only (default)
%  - gnu                  % Use GNU Prolog only
%  - gnu_fallback_swi     % Try GNU, fall back to SWI
%  - swi_fallback_gnu     % Try SWI, fall back to GNU
%  - [Dialect1, Dialect2, ...] % Custom order
:- dynamic prolog_default_strategy/1.

% Default: Use SWI-Prolog (most compatible)
prolog_default_strategy(swi).

%% set_prolog_default(+DialectOrStrategy)
%  Configure what 'prolog' means
%
%  @arg DialectOrStrategy One of:
%       - swi, gnu (single dialect)
%       - gnu_fallback_swi, swi_fallback_gnu (strategy)
%       - [Dialect1, Dialect2, ...] (custom list)
%
%  @example Set to try GNU first, fall back to SWI
%    ?- set_prolog_default(gnu_fallback_swi).
%
%  @example Set to SWI only
%    ?- set_prolog_default(swi).
%
%  @example Custom order
%    ?- set_prolog_default([gnu, swi]).
set_prolog_default(Strategy) :-
    retractall(prolog_default_strategy(_)),
    assertz(prolog_default_strategy(Strategy)),
    format('[PrologDialects] Set default strategy: ~w~n', [Strategy]).

%% get_prolog_default(-DialectOrStrategy)
%  Get current 'prolog' alias configuration
%
%  @example
%    ?- get_prolog_default(Strategy).
%    Strategy = swi.
get_prolog_default(Strategy) :-
    (   prolog_default_strategy(Strategy)
    ->  true
    ;   Strategy = swi  % Fallback to default
    ).

%% expand_prolog_alias(+Alias, -ConcreteDialects)
%  Expand a dialect alias to concrete dialect list
%
%  If Alias is already a concrete dialect (swi/gnu), return as singleton.
%  If Alias is 'prolog', expand based on current strategy.
%  If Alias is a strategy name, expand to dialect list.
%
%  @arg Alias The target name (prolog, swi, gnu, etc.)
%  @arg ConcreteDialects List of concrete dialects in order to try
%
%  @example Default expansion
%    ?- expand_prolog_alias(prolog, Dialects).
%    Dialects = [swi].
%
%  @example After setting gnu_fallback_swi
%    ?- set_prolog_default(gnu_fallback_swi),
%       expand_prolog_alias(prolog, Dialects).
%    Dialects = [gnu, swi].
%
%  @example Concrete dialect passes through
%    ?- expand_prolog_alias(swi, Dialects).
%    Dialects = [swi].
expand_prolog_alias(prolog, Dialects) :-
    !,
    % Expand based on current strategy
    get_prolog_default(Strategy),
    expand_strategy(Strategy, Dialects).

expand_prolog_alias(Alias, [Alias]) :-
    % Already a concrete dialect
    supported_dialect(Alias),
    !.

expand_prolog_alias(Strategy, Dialects) :-
    % Try to expand as a strategy
    expand_strategy(Strategy, Dialects),
    !.

expand_prolog_alias(Unknown, [swi]) :-
    % Unknown alias, fall back to SWI with warning
    format('[Warning] Unknown Prolog dialect alias: ~w, using swi~n', [Unknown]).

%% expand_strategy(+Strategy, -Dialects)
%  Convert strategy name to dialect list
expand_strategy(swi, [swi]) :- !.
expand_strategy(gnu, [gnu]) :- !.

expand_strategy(gnu_fallback_swi, [gnu, swi]) :- !.
expand_strategy(swi_fallback_gnu, [swi, gnu]) :- !.

% Custom list
expand_strategy(List, List) :-
    is_list(List),
    !.

% Unknown strategy, use default
expand_strategy(Unknown, [swi]) :-
    format('[Warning] Unknown Prolog strategy: ~w, using swi~n', [Unknown]).
