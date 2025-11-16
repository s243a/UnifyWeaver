:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% prolog_constraints.pl - Constraint Handling for Prolog Target
%
% Handles compilation constraints (unique, unordered, etc.) when generating
% Prolog code. Supports different strategies and failure modes.

:- module(prolog_constraints, [
    constraint_handler/2,          % +Constraint, -Handler
    set_constraint_mode/2,         % +Constraint, +Mode
    get_constraint_mode/2,         % +Constraint, -Mode
    handle_constraint/5,           % +Constraint, +Dialect, +Pred, +Code, -TransformedCode
    handle_constraints/5,          % +Constraints, +Dialect, +Pred, +Code, -TransformedCode
    constraint_satisfied/3,        % +Constraint, +Dialect, -CanSatisfy
    set_constraint_failure_mode/1, % +Mode (fail, warn, ignore, error)
    get_constraint_failure_mode/1  % -Mode
]).

:- use_module(library(option)).
:- use_module(prolog_dialects).

%% ============================================
%% CONFIGURATION
%% ============================================

%% constraint_mode(?Constraint, ?Mode)
%  How to handle each type of constraint
%
%  Modes:
%  - native    : Use dialect-specific features (e.g., tabling for unique)
%  - wrapper   : Generate wrapper code to enforce
%  - ignore    : Don't enforce (document only)
%  - fail      : Fail compilation if can't satisfy
:- dynamic constraint_mode/2.

%% Default modes for each constraint
constraint_mode(unique, native).      % Use tabling when available
constraint_mode(unordered, ignore).   % Prolog naturally unordered

%% constraint_failure_mode(?Mode)
%  What to do when constraint can't be satisfied
%
%  Modes:
%  - fail  : Fail compilation (default, safest)
%  - warn  : Warn user but continue
%  - error : Throw error
%  - ignore: Silently ignore
:- dynamic constraint_failure_mode/1.

constraint_failure_mode(fail).  % Default: fail compilation

%% ============================================
%% CONFIGURATION API
%% ============================================

%% set_constraint_mode(+Constraint, +Mode)
%  Configure how to handle a specific constraint
set_constraint_mode(Constraint, Mode) :-
    retractall(constraint_mode(Constraint, _)),
    assertz(constraint_mode(Constraint, Mode)),
    format('[PrologConstraints] Set ~w handling to: ~w~n', [Constraint, Mode]).

%% get_constraint_mode(+Constraint, -Mode)
%  Get current handling mode for constraint
get_constraint_mode(Constraint, Mode) :-
    % Extract constraint name from compound term (e.g., unique(true) -> unique)
    (   Constraint =.. [ConstraintName|_]
    ->  true
    ;   ConstraintName = Constraint
    ),
    % Look up mode for constraint name
    (   constraint_mode(ConstraintName, Mode)
    ->  true
    ;   Mode = ignore  % Default: ignore unknown constraints
    ).

%% set_constraint_failure_mode(+Mode)
%  Set global failure handling mode
set_constraint_failure_mode(Mode) :-
    (   member(Mode, [fail, warn, error, ignore])
    ->  retractall(constraint_failure_mode(_)),
        assertz(constraint_failure_mode(Mode)),
        format('[PrologConstraints] Constraint failure mode: ~w~n', [Mode])
    ;   throw(error(invalid_mode(Mode), 'Must be: fail, warn, error, or ignore'))
    ).

%% get_constraint_failure_mode(-Mode)
%  Get current failure handling mode
get_constraint_failure_mode(Mode) :-
    (   constraint_failure_mode(Mode)
    ->  true
    ;   Mode = fail  % Default
    ).

%% ============================================
%% CONSTRAINT SATISFACTION CHECKING
%% ============================================

%% constraint_satisfied(+Constraint, +Dialect, -CanSatisfy)
%  Check if constraint can be satisfied for dialect
%
%  @arg CanSatisfy One of: yes, no, Method
%       - yes: Can satisfy (returns method)
%       - no: Cannot satisfy
constraint_satisfied(unique(true), swi, tabling) :-
    % SWI-Prolog supports tabling
    !.

constraint_satisfied(unique(true), gnu, wrapper) :-
    % GNU Prolog needs wrapper (no tabling)
    !.

constraint_satisfied(unique(false), _, yes) :-
    % No uniqueness needed - always satisfiable
    !.

constraint_satisfied(unordered(true), _, yes) :-
    % Prolog is naturally unordered (backtracking)
    !.

constraint_satisfied(unordered(false), _, yes) :-
    % Order preservation is natural in Prolog
    !.

constraint_satisfied(optimization(_), _, yes) :-
    % Optimization is a hint, not a hard constraint
    !.

constraint_satisfied(Constraint, Dialect, no) :-
    % Unknown constraint or unsupported by dialect
    format('[Warning] Constraint ~w not supported for ~w~n', [Constraint, Dialect]).

%% ============================================
%% CONSTRAINT HANDLING
%% ============================================

%% handle_constraint(+Constraint, +Dialect, +Pred, +Code, -TransformedCode)
%  Apply constraint transformation to generated code
%
%  @arg Constraint The constraint to handle (e.g., unique(true))
%  @arg Dialect The target Prolog dialect (swi, gnu)
%  @arg Pred The predicate indicator (Name/Arity)
%  @arg Code The original generated code
%  @arg TransformedCode Code with constraint enforcement added
handle_constraint(Constraint, Dialect, Pred, Code, TransformedCode) :-
    get_constraint_mode(Constraint, Mode),
    handle_constraint_with_mode(Mode, Constraint, Dialect, Pred, Code, TransformedCode).

%% handle_constraint_with_mode(+Mode, +Constraint, +Dialect, +Pred, +Code, -TransformedCode)
%  Handle constraint based on configured mode

% Native mode: Use dialect-specific features
handle_constraint_with_mode(native, Constraint, Dialect, Pred, Code, TransformedCode) :-
    constraint_satisfied(Constraint, Dialect, Method),
    Method \= no,
    !,
    apply_native_constraint(Method, Constraint, Dialect, Pred, Code, TransformedCode).

% Wrapper mode: Generate wrapper code
handle_constraint_with_mode(wrapper, Constraint, Dialect, Pred, Code, TransformedCode) :-
    !,
    apply_wrapper_constraint(Constraint, Dialect, Pred, Code, TransformedCode).

% Ignore mode: No transformation
handle_constraint_with_mode(ignore, _Constraint, _Dialect, _Pred, Code, Code) :- !.

% Fail mode: Check if satisfiable, fail if not
handle_constraint_with_mode(fail, Constraint, Dialect, Pred, Code, TransformedCode) :-
    (   constraint_satisfied(Constraint, Dialect, Method),
        Method \= no
    ->  apply_native_constraint(Method, Constraint, Dialect, Pred, Code, TransformedCode)
    ;   % Cannot satisfy - handle failure
        get_constraint_failure_mode(FailMode),
        handle_unsatisfiable_constraint(FailMode, Constraint, Dialect, Pred)
    ).

%% handle_unsatisfiable_constraint(+FailMode, +Constraint, +Dialect, +Pred)
%  Handle case where constraint cannot be satisfied
handle_unsatisfiable_constraint(fail, Constraint, Dialect, Pred) :-
    !,
    format('[Error] Cannot satisfy ~w for ~w on dialect ~w~n', [Constraint, Pred, Dialect]),
    format('  Compilation failed. Use set_constraint_failure_mode(warn) to continue.~n'),
    fail.

handle_unsatisfiable_constraint(error, Constraint, Dialect, Pred) :-
    !,
    format(atom(Msg), 'Cannot satisfy ~w for ~w on dialect ~w', [Constraint, Pred, Dialect]),
    throw(error(constraint_unsatisfiable(Constraint, Dialect, Pred), Msg)).

handle_unsatisfiable_constraint(warn, Constraint, Dialect, Pred) :-
    !,
    format('[Warning] Cannot satisfy ~w for ~w on dialect ~w~n', [Constraint, Pred, Dialect]),
    format('  Continuing without constraint enforcement.~n').

handle_unsatisfiable_constraint(ignore, _Constraint, _Dialect, _Pred) :-
    !.

%% ============================================
%% NATIVE CONSTRAINT IMPLEMENTATION
%% ============================================

%% apply_native_constraint(+Method, +Constraint, +Dialect, +Pred, +Code, -TransformedCode)
%  Apply constraint using native Prolog features

% SWI-Prolog tabling for unique constraint
apply_native_constraint(tabling, unique(true), swi, Pred/Arity, Code, TransformedCode) :-
    !,
    format(atom(TableDirective), ':- table ~w/~w.~n~n', [Pred, Arity]),
    atomic_list_concat([TableDirective, Code], TransformedCode).

% Wrapper fallback for GNU Prolog unique constraint
apply_native_constraint(wrapper, unique(true), gnu, Pred, Code, TransformedCode) :-
    !,
    apply_wrapper_constraint(unique(true), gnu, Pred, Code, TransformedCode).

% No transformation needed
apply_native_constraint(yes, _Constraint, _Dialect, _Pred, Code, Code) :- !.

%% ============================================
%% WRAPPER CONSTRAINT IMPLEMENTATION
%% ============================================

%% apply_wrapper_constraint(+Constraint, +Dialect, +Pred, +Code, -TransformedCode)
%  Apply constraint using wrapper code generation

% Unique constraint via setof wrapper
apply_wrapper_constraint(unique(true), _Dialect, Pred/Arity, Code, TransformedCode) :-
    !,
    % Generate argument list
    functor(Args, Pred, Arity),
    Args =.. [Pred|ArgList],
    length(ArgList, Arity),

    % Create impl predicate name
    atom_concat(Pred, '_impl', ImplPred),

    % Replace predicate name in code
    atomic_list_concat(Lines, '\n', Code),
    maplist(replace_predicate_name(Pred, ImplPred), Lines, ImplLines),
    atomic_list_concat(ImplLines, '\n', ImplCode),

    % Generate wrapper
    format(atom(ArgListStr), '~w', [ArgList]),
    format(atom(Wrapper),
           '~w~w :- setof(~w, ~w~w, Solutions), member(~w, Solutions).',
           [Pred, Args, ArgListStr, ImplPred, Args, ArgListStr]),

    atomic_list_concat([
        '% Implementation (for deduplication)',
        ImplCode,
        '',
        '% Wrapper (enforces uniqueness)',
        Wrapper
    ], '\n', TransformedCode).

apply_wrapper_constraint(_Constraint, _Dialect, _Pred, Code, Code).

%% replace_predicate_name(+OldName, +NewName, +Line, -NewLine)
%  Replace predicate name in a clause line
replace_predicate_name(OldName, NewName, Line, NewLine) :-
    atomic_list_concat(Parts, OldName, Line),
    length(Parts, NumParts),
    (   NumParts > 1
    ->  atomic_list_concat(Parts, NewName, NewLine)
    ;   NewLine = Line
    ).

%% ============================================
%% BATCH CONSTRAINT HANDLING
%% ============================================

%% handle_constraints(+Constraints, +Dialect, +Pred, +Code, -FinalCode)
%  Apply multiple constraints in sequence
handle_constraints([], _Dialect, _Pred, Code, Code).

handle_constraints([Constraint|Rest], Dialect, Pred, Code, FinalCode) :-
    (   catch(
            handle_constraint(Constraint, Dialect, Pred, Code, Code1),
            Error,
            (format('[Error] Constraint ~w failed: ~w~n', [Constraint, Error]), Code1 = Code)
        )
    ->  handle_constraints(Rest, Dialect, Pred, Code1, FinalCode)
    ;   % Constraint handling failed, check failure mode
        get_constraint_failure_mode(FailMode),
        (   FailMode = fail
        ->  fail  % Propagate failure
        ;   % Continue with remaining constraints
            format('[Warning] Skipping constraint ~w~n', [Constraint]),
            handle_constraints(Rest, Dialect, Pred, Code, FinalCode)
        )
    ).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% constraint_handler(+Constraint, -Handler)
%  Get the handler predicate for a constraint type
constraint_handler(unique(_), handle_unique_constraint).
constraint_handler(unordered(_), handle_unordered_constraint).
constraint_handler(optimization(_), handle_optimization_constraint).
