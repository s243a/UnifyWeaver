% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% r_target.pl - R Target for UnifyWeaver
% Generates standalone R scripts for record/field processing.
% Supports vectorized operations, data frames, and native pipes.

:- module(r_target, [
    compile_predicate/3,            % +Predicate, +Options, -RCode
    compile_predicate_to_r/3,       % +Predicate, +Options, -RCode
    infer_clauses_return_type/2,    % +Clauses, -Type
    compile_facts_to_r/3,           % +Pred, +Arity, -RCode
    init_r_target/0,                % Initialize R target with bindings
    compile_r_pipeline/3,           % +Predicates, +Options, -RCode
    test_r_pipeline/0               % Test pipeline generation
]).

:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module(library(option)).
:- use_module(common_generator).
:- use_module('../core/binding_registry').
:- use_module('../core/component_registry').
:- use_module('../bindings/r_bindings').
:- use_module('../core/template_system').
:- use_module('type_declarations').
:- multifile prolog:message//1.

% Track required packages/libraries
:- dynamic required_r_package/1.
:- dynamic collected_component/2.

%% init_r_target
%  Initialize the R target by loading bindings and clearing state.
init_r_target :-
    retractall(required_r_package(_)),
    retractall(collected_component(_, _)),
    init_r_bindings.

%% compile_predicate_to_r(+PredIndicator, +Options, -RCode)
%  Compile a Prolog predicate indicator into an R function.
%  Handles: no clauses, single clause, multiple clauses (if/else if chain).
compile_predicate(PredIndicator, Options, RCode) :-
    compile_predicate_to_r(PredIndicator, Options, RCode).

compile_predicate_to_r(PredIndicator, Options, RCode) :-
    (   PredIndicator = Module:Pred/Arity
    ->  true
    ;   PredIndicator = Pred/Arity,
        Module = user
    ),
    effective_r_return_type(Pred/Arity, Options, ReturnType),
    effective_type_diagnostics(Options, DiagnosticsMode),
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(Module:Head, Body), Clauses),
    (   Clauses = []
    ->  format(string(RCode), '# No clauses found for ~w/~w', [Pred, Arity])
    ;   Clauses = [SingleHead-SingleBody]
    ->  compile_rule(SingleHead, SingleBody, ReturnType, DiagnosticsMode, RCode)
    ;   compile_multi_clause_r(Pred, Arity, Clauses, ReturnType, DiagnosticsMode, RCode)
    ).

%% compile_multi_clause_r(+Pred, +Arity, +Clauses, -RCode)
%  Compile multiple clauses into an R function with if/else if chain.
compile_multi_clause_r(Pred, Arity, Clauses, ReturnType, DiagnosticsMode, RCode) :-
    % Generate standard arg names
    numlist(1, Arity, Indices),
    findall(ArgName, (
        member(I, Indices),
        format(atom(ArgName), 'arg~w', [I])
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList),

    % Compile each clause into an if/else if branch
    compile_clause_branches(Clauses, ArgNames, ReturnType, DiagnosticsMode, Branches),
    r_fallback_expression(ReturnType, Pred, FallbackExpr),
    (   Branches = []
    ->  format(string(RCode),
'~w <- function(~w) {
    ~w
}
', [Pred, ArgList, FallbackExpr])
    ;   atomic_list_concat(Branches, ' else ', BranchCode),
        format(string(RCode),
'~w <- function(~w) {
    ~w else {
        ~w
    }
}
', [Pred, ArgList, BranchCode, FallbackExpr])
    ).

%% compile_clause_branches(+Clauses, +ArgNames, -Branches)
compile_clause_branches([], _, _, _, []).
compile_clause_branches([Head-Body|Rest], ArgNames, ReturnType, DiagnosticsMode, Branches) :-
    compile_clause_branches(Rest, ArgNames, ReturnType, DiagnosticsMode, RestBranches),
    (   compile_clause_branch(Head, Body, ArgNames, ReturnType, DiagnosticsMode, Branch)
    ->  Branches = [Branch|RestBranches]
    ;   Branches = RestBranches
    ).

%% compile_clause_branch(+Head, +Body, +ArgNames, -Branch)
%  Compile one clause into an if(...) { ... } branch.
compile_clause_branch(Head, Body, ArgNames, ReturnType, DiagnosticsMode, Branch) :-
    head_pred_spec(Head, PredSpec),
    body_return_type_allowed(PredSpec, clause_filter, ReturnType, Body, DiagnosticsMode),
    Head =.. [_|HeadArgs],
    % Build condition from head argument patterns
    generate_head_condition(HeadArgs, ArgNames, Condition),
    % Build VarMap mapping head variables to arg names
    build_head_varmap(HeadArgs, ArgNames, VarMap),
    % Compile body with the VarMap
    compile_body(Body, VarMap, _, BodyCode),
    format(string(Branch), 'if (~w) {\n~s\n    }', [Condition, BodyCode]).

%% generate_head_condition(+HeadArgs, +ArgNames, -Condition)
%  Build R condition string from head argument patterns.
generate_head_condition(HeadArgs, ArgNames, Condition) :-
    maplist(generate_arg_condition, HeadArgs, ArgNames, Conditions),
    exclude(==('TRUE'), Conditions, NonTrivial),
    (   NonTrivial = []
    ->  Condition = 'TRUE'
    ;   atomic_list_concat(NonTrivial, ' && ', Condition)
    ).

%% generate_arg_condition(+HeadArg, +ArgName, -Condition)
%  Single argument condition: ground → equality check, var → TRUE.
generate_arg_condition(Arg, _ArgName, 'TRUE') :-
    var(Arg), !.
generate_arg_condition(Arg, ArgName, Condition) :-
    number(Arg), !,
    format(string(Condition), '~w == ~w', [ArgName, Arg]).
generate_arg_condition(Arg, ArgName, Condition) :-
    atom(Arg), !,
    format(string(Condition), '~w == "~w"', [ArgName, Arg]).
generate_arg_condition(Arg, ArgName, Condition) :-
    format(string(Condition), '~w == ~w', [ArgName, Arg]).

%% build_head_varmap(+HeadArgs, +ArgNames, -VarMap)
%  Map head variables to their corresponding arg names.
build_head_varmap([], [], []).
build_head_varmap([Arg|Args], [Name|Names], [Arg-Name|Rest]) :-
    var(Arg), !,
    build_head_varmap(Args, Names, Rest).
build_head_varmap([_|Args], [_|Names], Rest) :-
    build_head_varmap(Args, Names, Rest).

%% collect_declared_component(+Category, +Name)
%  Track a component that was referenced during compilation.
collect_declared_component(Category, Name) :-
    (   collected_component(Category, Name)
    ->  true
    ;   assertz(collected_component(Category, Name))
    ).

%% compile_collected_components(-Code)
%  Compile all collected components into R code.
compile_collected_components(Code) :-
    findall(CompCode, (
        collected_component(Category, Name),
        component_registry:compile_component(Category, Name, [], CompCode)
    ), CompCodes),
    (   CompCodes = []
    ->  Code = ''
    ;   atomic_list_concat(CompCodes, '\n\n', Code)
    ).

compile_rule(Head, Body, ReturnType, DiagnosticsMode, Code) :-
    Head =.. [Pred|Args],
    head_pred_spec(Head, PredSpec),
    
    % Generate variable mapping for arguments
    map_args(Args, 1, VarMap, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList),
    
    % Compile body
    (   body_return_type_allowed(PredSpec, single_clause_fallback, ReturnType, Body, DiagnosticsMode)
    ->  compile_body(Body, VarMap, _, BodyCode)
    ;   r_fallback_expression(ReturnType, Pred, FallbackExpr),
        format(string(BodyCode), '    ~w', [FallbackExpr])
    ),
    
    format(string(Code),
'~w <- function(~w) {
~s
}
', [Pred, ArgList, BodyCode]).

map_args([], _, [], []).
map_args([Arg|Rest], Idx, [Arg-VarName|MapRest], [VarName|NamesRest]) :-
    format(atom(VarName), 'arg~w', [Idx]),
    NextIdx is Idx + 1,
    map_args(Rest, NextIdx, MapRest, NamesRest).

compile_body(true, V, V, "") :- !.
compile_body((A, B), V0, V2, Code) :- !,
    compile_body(A, V0, V1, C1),
    compile_body(B, V1, V2, C2),
    format(string(Code), '~s~n~s', [C1, C2]).
compile_body(_M:Goal, V0, V1, Code) :- !,
    compile_body(Goal, V0, V1, Code).

% Specialized data frame streaming operations
compile_body(filter(DF, Expr, Out), V0, V1, Code) :- !,
    ensure_var(V0, Out, ROut, V1),
    resolve_val(V0, DF, RDF),
    translate_r_expr(Expr, V0, RExpr),
    format(string(Code), '    ~w <- subset(~w, ~w)', [ROut, RDF, RExpr]).

compile_body(group_by(DF, Col, Out), V0, V1, Code) :- !,
    ensure_var(V0, Out, ROut, V1),
    resolve_val(V0, DF, RDF),
    resolve_val(V0, Col, RCol),
    format(string(Code), '    ~w <- aggregate(. ~~ ~w, data=~w, FUN=list)', [ROut, RCol, RDF]).

compile_body(sort_by(DF, Col, Out), V0, V1, Code) :- !,
    ensure_var(V0, Out, ROut, V1),
    resolve_val(V0, DF, RDF),
    resolve_val(V0, Col, RCol),
    format(string(Code), '    ~w <- ~w[order(~w[[~w]]), ]', [ROut, RDF, RDF, RCol]).

compile_body(Goal, V0, V1, Code) :-
    functor(Goal, Pred, Arity),
    (   binding(r, Pred/Arity, TargetName, Inputs, Outputs, _Options)
    ->  Goal =.. [_|Args],
        length(Inputs, InCount),
        length(InArgs, InCount),
        append(InArgs, OutArgs, Args),
        
        maplist(resolve_val(V0), InArgs, RInArgs),
        atomic_list_concat(RInArgs, ', ', RArgsStr),
        format(string(Expr), '~w(~w)', [TargetName, RArgsStr]),
        
        (   Outputs = []
        ->  V1 = V0,
            format(string(Code), '    ~w', [Expr])
        ;   OutArgs = [OutVar],
            ensure_var(V0, OutVar, ROutVar, V1),
            format(string(Code), '    ~w <- ~w', [ROutVar, Expr])
        )
    ;   V1 = V0,
        format(string(Code), '    # Unknown predicate: ~w', [Goal])
    ).

resolve_val(VarMap, Var, Val) :-
    var(Var), lookup_var(Var, VarMap, Name), !,
    format(string(Val), '~w', [Name]).
resolve_val(_, Val, StrVal) :-
    atom(Val), !,
    (   Val == true
    ->  StrVal = 'TRUE'
    ;   Val == false
    ->  StrVal = 'FALSE'
    ;   format(string(StrVal), '"~w"', [Val])
    ).
resolve_val(_, Val, StrVal) :-
    format(string(StrVal), '~w', [Val]).

effective_r_return_type(_PredSpec, Options, ReturnType) :-
    option(type_constraints(false), Options),
    !,
    ReturnType = none.
effective_r_return_type(PredSpec, _Options, ReturnType) :-
    predicate_return_type(PredSpec, ReturnType),
    !.
effective_r_return_type(_PredSpec, _Options, none).

effective_type_diagnostics(Options, DiagnosticsMode) :-
    option(type_diagnostics(DiagnosticsMode), Options, off).

body_return_type_allowed(_PredSpec, _Action, none, _Body, _DiagnosticsMode) :-
    !.
body_return_type_allowed(PredSpec, Action, ReturnType, Body, DiagnosticsMode) :-
    infer_body_return_type(Body, InferredType),
    !,
    (   return_types_compatible(ReturnType, InferredType)
    ->  true
    ;   handle_type_diagnostic(
            DiagnosticsMode,
            r_type_constraint_violation(PredSpec, Action, ReturnType, InferredType, Body)
        ),
        fail
    ).
body_return_type_allowed(_PredSpec, _Action, _ReturnType, _Body, _DiagnosticsMode).

return_types_compatible(any, _InferredType) :-
    !.
return_types_compatible(_ReturnType, any) :-
    !.
return_types_compatible(ReturnType, InferredType) :-
    resolve_type(ReturnType, r, ReturnConcrete),
    resolve_type(InferredType, r, InferredConcrete),
    ReturnConcrete == InferredConcrete.

infer_body_return_type(true, boolean) :-
    !.
infer_body_return_type((Left, true), Type) :-
    !,
    infer_body_return_type(Left, Type).
infer_body_return_type((_, Right), Type) :-
    !,
    (   infer_body_return_type(Right, Type)
    ->  true
    ;   Right = (LeftRight, _),
        infer_body_return_type(LeftRight, Type)
    ).
infer_body_return_type(_Module:Goal, Type) :-
    !,
    infer_body_return_type(Goal, Type).
infer_body_return_type(filter(_, _, _), any) :-
    !.
infer_body_return_type(group_by(_, _, _), any) :-
    !.
infer_body_return_type(sort_by(_, _, _), any) :-
    !.
infer_body_return_type(Goal, Type) :-
    compound(Goal),
    functor(Goal, Pred, Arity),
    binding(r, Pred/Arity, _TargetName, _Inputs, Outputs, _Options),
    infer_binding_return_type(Outputs, Type).

infer_clauses_return_type(Clauses, Type) :-
    findall(ClauseType, (
        member(_Head-Body, Clauses),
        infer_body_return_type(Body, ClauseType)
    ), [FirstType|RestTypes]),
    all_same_return_type(RestTypes, FirstType),
    Type = FirstType.

all_same_return_type([], _).
all_same_return_type([Type|Rest], ReferenceType) :-
    return_types_compatible(ReferenceType, Type),
    return_types_compatible(Type, ReferenceType),
    all_same_return_type(Rest, ReferenceType).

infer_binding_return_type([], boolean).
infer_binding_return_type([BindingType|_], Type) :-
    normalize_binding_type(BindingType, Type).

normalize_binding_type(int, integer).
normalize_binding_type(integer, integer).
normalize_binding_type(float, float).
normalize_binding_type(number, number).
normalize_binding_type(logical, boolean).
normalize_binding_type(boolean, boolean).
normalize_binding_type(string, string).
normalize_binding_type(atom, atom).
normalize_binding_type(any, any).
normalize_binding_type(list(Type), list(Normalized)) :-
    normalize_binding_type(Type, Normalized).
normalize_binding_type(maybe(Type), maybe(Normalized)) :-
    normalize_binding_type(Type, Normalized).
normalize_binding_type(map(KeyType, ValueType), map(NormalizedKey, NormalizedValue)) :-
    normalize_binding_type(KeyType, NormalizedKey),
    normalize_binding_type(ValueType, NormalizedValue).
normalize_binding_type(set(Type), set(Normalized)) :-
    normalize_binding_type(Type, Normalized).
normalize_binding_type(pair(LeftType, RightType), pair(NormalizedLeft, NormalizedRight)) :-
    normalize_binding_type(LeftType, NormalizedLeft),
    normalize_binding_type(RightType, NormalizedRight).
normalize_binding_type(record(Name, Fields), record(Name, Fields)).

r_fallback_expression(none, Pred, Expr) :-
    format(string(Expr), 'stop("No matching clause for ~w")', [Pred]).
r_fallback_expression(boolean, _Pred, 'FALSE') :-
    !.
r_fallback_expression(any, _Pred, 'NULL') :-
    !.
r_fallback_expression(atom, _Pred, 'character()') :-
    !.
r_fallback_expression(string, _Pred, 'character()') :-
    !.
r_fallback_expression(integer, _Pred, 'integer()') :-
    !.
r_fallback_expression(float, _Pred, 'numeric()') :-
    !.
r_fallback_expression(number, _Pred, 'numeric()') :-
    !.
r_fallback_expression(maybe(_), _Pred, 'NA') :-
    !.
r_fallback_expression(list(_), _Pred, 'list()') :-
    !.
r_fallback_expression(set(_), _Pred, 'list()') :-
    !.
r_fallback_expression(pair(_, _), _Pred, 'list()') :-
    !.
r_fallback_expression(map(_, _), _Pred, 'new.env(parent = emptyenv())') :-
    !.
r_fallback_expression(record(_, _), _Pred, 'list()') :-
    !.
r_fallback_expression(ReturnType, Pred, Expr) :-
    resolve_type(ReturnType, r, ResolvedType),
    (   ResolvedType == 'character'
    ->  Expr = 'character()'
    ;   ResolvedType == 'integer'
    ->  Expr = 'integer()'
    ;   ResolvedType == 'numeric'
    ->  Expr = 'numeric()'
    ;   ResolvedType == 'logical'
    ->  Expr = 'FALSE'
    ;   ResolvedType == 'list'
    ->  Expr = 'list()'
    ;   ResolvedType == 'environment'
    ->  Expr = 'new.env(parent = emptyenv())'
    ;   format(string(Expr), 'stop("No matching clause for ~w")', [Pred])
    ).

head_pred_spec(Head, Pred/Arity) :-
    functor(Head, Pred, Arity).

handle_type_diagnostic(off, _Diagnostic) :-
    true.
handle_type_diagnostic(warn, Diagnostic) :-
    print_message(warning, Diagnostic).
handle_type_diagnostic(error, Diagnostic) :-
    throw(error(Diagnostic, context(r_target, 'Return type constraint violated during R compilation'))).

prolog:message(r_type_constraint_violation(PredSpec, Action, ExpectedType, InferredType, Body)) -->
    [ 'R target type constraint violation for ~p (~w): expected ~p, inferred ~p from body ~p'-
      [PredSpec, Action, ExpectedType, InferredType, Body]
    ].

ensure_var(VarMap, Var, Name, VarMap) :-
    lookup_var(Var, VarMap, Name), !.
ensure_var(VarMap, Var, Name, [Var-Name|VarMap]) :-
    gensym(v, Name).

lookup_var(Var, [V-Name|_], Name) :- Var == V, !.
lookup_var(Var, [_|Rest], Name) :- lookup_var(Var, Rest, Name).

% Expression translation for filter and arithmetic
translate_r_expr(Var, V, Res) :- var(Var), lookup_var(Var, V, Name), !, format(string(Res), '~w', [Name]).
translate_r_expr(Atom, _, Res) :- atom(Atom), !, format(string(Res), '~w', [Atom]). % Column names unquoted in subset
translate_r_expr(Num, _, Res) :- number(Num), !, format(string(Res), '~w', [Num]).
translate_r_expr(Expr, V, Res) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    translate_r_expr(Left, V, LRes),
    translate_r_expr(Right, V, RRes),
    r_op_map(Op, ROp),
    format(string(Res), '(~w ~w ~w)', [LRes, ROp, RRes]).

r_op_map(>, '>').
r_op_map(<, '<').
r_op_map(>=, '>=').
r_op_map(=<, '<=').
r_op_map(==, '==').
r_op_map(\=, '!=').
r_op_map(and, '&').
r_op_map(or, '|').

% ============================================================================
% FACT EXPORT - compile_facts_to_r/3
% ============================================================================

%% compile_facts_to_r(+Pred, +Arity, -RCode)
%  Export Prolog facts as an R script with list data and accessor functions.
%  Generates: <pred>_facts list, get_all_<pred>(), stream_<pred>(fn),
%  contains_<pred>(...) functions, and a main block.
compile_facts_to_r(Pred, Arity, RCode) :-
    atom_string(Pred, PredStr),

    % Collect all facts
    functor(Head, Pred, Arity),
    findall(Args, (user:clause(Head, true), Head =.. [_|Args]), AllFacts),

    % Format facts as R list of c() vectors
    findall(Entry, (
        member(Args, AllFacts),
        maplist(format_r_fact_arg, Args, FormattedArgs),
        atomic_list_concat(FormattedArgs, ', ', ArgsStr),
        format(string(Entry), '    c(~w)', [ArgsStr])
    ), Entries),
    atomic_list_concat(Entries, ',\n', EntriesCode),

    % Generate arg names for contains function
    numlist(1, Arity, Indices),
    findall(ArgName, (
        member(I, Indices),
        format(atom(ArgName), 'arg~w', [I])
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ContainsArgStr),

    % Generate contains comparison
    findall(Check, (
        member(I, Indices),
        format(string(Check), 'fact[~w] == arg~w', [I, I])
    ), Checks),
    atomic_list_concat(Checks, ' && ', CheckStr),

    format(string(RCode),
'#!/usr/bin/env Rscript
# Generated by UnifyWeaver R Target - Fact Export
# Predicate: ~w/~w

# Fact data
~w_facts <- list(
~w
)

# Return all facts
get_all_~w <- function() {
    return(~w_facts)
}

# Apply function to each fact
stream_~w <- function(fn) {
    lapply(~w_facts, fn)
}

# Check if a fact exists
contains_~w <- function(~w) {
    for (fact in ~w_facts) {
        if (~w) return(TRUE)
    }
    return(FALSE)
}

# Main
if (!interactive()) {
    for (fact in ~w_facts) {
        cat(paste(fact, collapse=":"), "\\n")
    }
}
', [PredStr, Arity,
    PredStr, EntriesCode,
    PredStr, PredStr,
    PredStr, PredStr,
    PredStr, ContainsArgStr, PredStr, CheckStr,
    PredStr]).

%% format_r_fact_arg(+Arg, -Formatted)
%  Format a single fact argument for R output.
format_r_fact_arg(Arg, Formatted) :-
    (   number(Arg)
    ->  format(string(Formatted), '~w', [Arg])
    ;   format(string(Formatted), '"~w"', [Arg])
    ).

% ============================================================================
% PIPELINE MODE
% ============================================================================

%% compile_r_pipeline(+Predicates, +Options, -RCode)
%  Generates an R pipeline (e.g. using native |> or dplyr %>%).
compile_r_pipeline(Predicates, Options, RCode) :-
    (member(pipeline_name(PipelineName), Options) -> true ; PipelineName = 'pipeline'),
    (member(pipeline_mode(Mode), Options) -> true ; Mode = sequential),
    
    r_pipeline_header(HeaderCode),
    generate_r_stage_functions(Predicates, StageFunctions),
    generate_r_pipeline_connector(Predicates, PipelineName, Mode, ConnectorCode),
    
    format(string(RCode), '~s~n~n~s~n~n~s',
           [HeaderCode, StageFunctions, ConnectorCode]).

r_pipeline_header(Code) :-
    format(string(Code),
'# Auto-generated R Pipeline
suppressPackageStartupMessages({
    library(jsonlite)
    # library(dplyr)  # Uncomment if using dplyr pipes
})

# Helper for reading JSONL
read_jsonl <- function(file_path) {
    con <- file(file_path, "r")
    lines <- readLines(con)
    close(con)
    df <- do.call(rbind, lapply(lines, fromJSON))
    return(as.data.frame(df))
}
', []).

generate_r_stage_functions([], "").
generate_r_stage_functions([Pred|Rest], Code) :-
    compile_predicate_to_r(Pred, [], StageCode),
    generate_r_stage_functions(Rest, RestCode),
    (RestCode = "" -> Code = StageCode ; format(string(Code), '~s~n~n~s', [StageCode, RestCode])).

generate_r_pipeline_connector(Predicates, PipelineName, sequential, Code) :-
    maplist(get_pred_name, Predicates, StageNames),
    atomic_list_concat(StageNames, ' |>\n    ', PipelineStr),
    format(string(Code),
'# Sequential pipeline connector: ~w
~w <- function(data) {
    data |>
    ~w
}
', [PipelineName, PipelineName, PipelineStr]).

generate_r_pipeline_connector(Predicates, PipelineName, generator, Code) :-
    maplist(get_pred_name, Predicates, StageNames),
    atomic_list_concat(StageNames, ' |>\n            ', PipelineStr),
    format(string(Code),
'# Fixpoint pipeline connector: ~w
# Iterates until no new rows are produced.
~w <- function(data) {
    # Initialize seen set with hash representation of rows
    seen <- new.env(hash=TRUE, parent=emptyenv())
    
    # helper to hash rows
    hash_row <- function(row) {
        digest::digest(row) # requires digest package
    }
    
    # Initial setup
    records <- data
    output_records <- data
    for (i in seq_len(nrow(records))) {
        seen[[hash_row(records[i,])]] <- TRUE
    }
    
    changed <- TRUE
    while (changed) {
        changed <- FALSE
        current_records <- records
        
        for (i in seq_len(nrow(current_records))) {
            row <- current_records[i, , drop=FALSE]
            # Run pipeline stages on the current row
            new_row <- row |>
            ~w
            
            key <- hash_row(new_row)
            if (is.null(seen[[key]])) {
                seen[[key]] <- TRUE
                records <- rbind(records, new_row)
                output_records <- rbind(output_records, new_row)
                changed <- TRUE
            }
        }
    }
    return(output_records)
}
', [PipelineName, PipelineName, PipelineStr]).

get_pred_name(Pred/_Arity, Pred).

% ============================================================================
% TREE RECURSION - R target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tree_recursion').
:- multifile tree_recursion:compile_tree_pattern/6.

%% R fibonacci-like tree recursion
tree_recursion:compile_tree_pattern(r, fibonacci, Pred, _Arity, UseMemo, RCode) :-
    atom_string(Pred, PredStr),
    (   UseMemo = true ->
        MemoDecl = '~w_memo <- new.env(hash=TRUE, parent=emptyenv())'
    ;   MemoDecl = '# Memoization disabled'
    ),
    (   UseMemo = true ->
        format(string(MemoDeclFormatted), MemoDecl, [PredStr])
    ;   MemoDeclFormatted = MemoDecl
    ),

    TemplateLines = [
        "# {{pred}}/2 - tree recursive pattern (Fibonacci-like in R)",
        "{{memo_decl}}",
        "",
        "{{pred}} <- function(n, expected=NULL) {",
        "    if (n == 0) return(0)",
        "    if (n == 1) return(1)",
        "",
        "    # Check memo",
        "    key <- as.character(n)",
        "    if (!is.null({{{pred}}_memo[[key]]})) {",
        "        result <- {{{pred}}_memo[[key]]}",
        "    } else {",
        "        # Recursive calls",
        "        result <- {{pred}}(n - 1) + {{pred}}(n - 2)",
        "        {{{pred}}_memo[[key]]} <- result",
        "    }",
        "",
        "    if (!is.null(expected)) {",
        "        return(result == expected)",
        "    }",
        "    return(result)",
        "}"
    ],
    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, memo_decl=MemoDeclFormatted], RCode).

%% R binary tree recursion
tree_recursion:compile_tree_pattern(r, binary_tree, Pred, _Arity, UseMemo, RCode) :-
    atom_string(Pred, PredStr),

    % Detect operation type from predicate name
    (   sub_atom(Pred, _, _, _, sum) ->
        BaseValue = "0",
        CombineExpr = "value + left_result + right_result"
    ;   sub_atom(Pred, _, _, _, height) ->
        BaseValue = "0",
        CombineExpr = "1 + max(left_result, right_result)"
    ;   sub_atom(Pred, _, _, _, count) ->
        BaseValue = "0",
        CombineExpr = "1 + left_result + right_result"
    ;   % Generic: sum-like default
        BaseValue = "0",
        CombineExpr = "value + left_result + right_result"
    ),

    % Memo declaration
    (   UseMemo = true ->
        format(string(MemoDecl), '~w_memo <- new.env(hash=TRUE, parent=emptyenv())', [PredStr])
    ;   MemoDecl = '# Memoization disabled'
    ),

    % Memo check code
    (   UseMemo = true ->
        format(string(MemoCheck),
'    # Check memo
    key <- digest::digest(tree)
    if (!is.null(~w_memo[[key]])) return(~w_memo[[key]])', [PredStr, PredStr])
    ;   MemoCheck = ''
    ),

    % Memo store code
    (   UseMemo = true ->
        format(string(MemoStore), '\n    ~w_memo[[key]] <<- result', [PredStr])
    ;   MemoStore = ''
    ),

    format(string(RCode),
'#!/usr/bin/env Rscript
# ~w - tree recursion (binary tree pattern, R)
# Trees: list(value=V, left=L, right=R) or NULL for empty

~w

~w <- function(tree) {
    # Base case: empty tree
    if (is.null(tree)) return(~w)

~w

    # Decompose tree
    value <- tree$value
    left <- tree$left
    right <- tree$right

    # Recursive calls
    left_result <- ~w(left)
    right_result <- ~w(right)

    # Combine results
    result <- ~w~w
    return(result)
}

# Example usage:
# tree <- list(value=5, left=list(value=3, left=NULL, right=NULL),
#              right=list(value=7, left=NULL, right=NULL))
# cat(~w(tree), "\\n")

# Run when script executed directly
if (!interactive()) {
    # Demo with a simple tree
    tree <- list(value=1,
        left=list(value=2, left=NULL, right=NULL),
        right=list(value=3, left=NULL, right=NULL))
    cat(~w(tree), "\\n")
}
', [PredStr, MemoDecl, PredStr, BaseValue, MemoCheck,
    PredStr, PredStr, CombineExpr, MemoStore,
    PredStr, PredStr]).

%% R generic tree recursion (template with commented example)
tree_recursion:compile_tree_pattern(r, generic, Pred, Arity, _UseMemo, RCode) :-
    atom_string(Pred, PredStr),
    format(string(RCode),
'#!/usr/bin/env Rscript
# ~w/~w - generic tree recursive pattern (R)
# Auto-generated template. Customize the tree structure and combine logic below.

~w_memo <- new.env(hash=TRUE, parent=emptyenv())

~w <- function(tree) {
    # Base case: empty/leaf node
    if (is.null(tree)) return(0)

    # Memo check
    key <- digest::digest(tree)
    if (!is.null(~w_memo[[key]])) return(~w_memo[[key]])

    # --- Customize below ---
    # Decompose tree into value and children
    # Example for binary tree: list(value=V, left=L, right=R)
    value <- tree$value
    children <- list(tree$left, tree$right)

    # Recurse on each child
    child_results <- sapply(children, function(child) {
        if (is.null(child)) 0 else ~w(child)
    })

    # Combine: sum, max, count, etc.
    result <- value + sum(child_results)
    # --- End customization ---

    # Store in memo
    ~w_memo[[key]] <<- result
    return(result)
}
', [PredStr, Arity, PredStr, PredStr, PredStr, PredStr, PredStr, PredStr]).

% ============================================================================
% TAIL RECURSION - R target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/tail_recursion').
:- multifile tail_recursion:compile_tail_pattern/9.

tail_recursion:compile_tail_pattern(r, PredStr, Arity, _BaseClauses, _RecClauses, AccPos, StepOp, _ExitAfterResult, RCode) :-
    (   Arity =:= 3 ->
        tail_ternary_loop_r(PredStr, AccPos, StepOp, RCode)
    ;   Arity =:= 2 ->
        tail_binary_loop_r(PredStr, RCode)
    ;   format('Warning: tail recursion in R with arity ~w not yet supported~n', [Arity]),
        fail
    ).

%% step_op_to_r(+StepOp, -RCode)
step_op_to_r(arithmetic(Expr), RCode) :-
    expr_to_r(Expr, RExpr),
    format(atom(RCode), 'current_acc <- ~w', [RExpr]).
step_op_to_r(unknown, 'current_acc <- current_acc + 1').

%% expr_to_r(+PrologExpr, -RExpr)
expr_to_r(_ + Const, RExpr) :- integer(Const), !, format(atom(RExpr), 'current_acc + ~w', [Const]).
expr_to_r(_ + _, 'current_acc + item') :- !.
expr_to_r(_ - _, 'current_acc - item') :- !.
expr_to_r(_ * _, 'current_acc * item') :- !.
expr_to_r(_, 'current_acc + 1').

%% tail_ternary_loop_r(+PredStr, +AccPos, +StepOp, -RCode)
tail_ternary_loop_r(PredStr, _AccPos, StepOp, RCode) :-
    step_op_to_r(StepOp, RStepOp),
    TemplateLines = [
        "# {{pred}} - tail recursive accumulator pattern (R)",
        "{{pred}} <- function(input, acc) {",
        "    items <- input",
        "    if (is.character(input)) {",
        "       # parse string input if necessary",
        "       items <- unlist(strsplit(gsub(\"\\\\[|\\\\]\", \"\", input), \",\"))",
        "       nums <- suppressWarnings(as.numeric(items))",
        "       if (all(!is.na(nums))) items <- nums",
        "    }",
        "    current_acc <- acc",
        "    for (item in items) {",
        "        {{step_op}}",
        "    }",
        "    return(current_acc)",
        "}",
        "{{pred}}_eval <- function(input) {",
        "    return({{pred}}(input, 0))",
        "}",
        "",
        "# Run when script executed directly",
        "if (!interactive()) {",
        "    args <- commandArgs(TRUE)",
        "    if (length(args) >= 1) {",
        "        items <- as.numeric(unlist(strsplit(args[1], \",\")))",
        "        cat({{pred}}_eval(items), \"\\n\")",
        "    }",
        "}"
    ],
    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, step_op=RStepOp], RCode).

%% tail_binary_loop_r(+PredStr, -RCode)
tail_binary_loop_r(PredStr, RCode) :-
    TemplateLines = [
        "# {{pred}} - tail recursive binary pattern (R)",
        "{{pred}} <- function(input, expected=NULL) {",
        "    count <- 0",
        "    items <- input",
        "    while (length(items) > 0) {",
        "        count <- count + 1",
        "        items <- items[-1]",
        "    }",
        "    if (!is.null(expected)) {",
        "        return(count == expected)",
        "    }",
        "    return(count)",
        "}",
        "",
        "# Run when script executed directly",
        "if (!interactive()) {",
        "    args <- commandArgs(TRUE)",
        "    if (length(args) >= 1) {",
        "        items <- as.numeric(unlist(strsplit(args[1], \",\")))",
        "        cat({{pred}}(items), \"\\n\")",
        "    }",
        "}"
    ],
    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr], RCode).

% ============================================================================
% MULTICALL LINEAR RECURSION - R target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/multicall_linear_recursion').
:- multifile multicall_linear_recursion:compile_multicall_pattern/6.

multicall_linear_recursion:compile_multicall_pattern(r, PredStr, BaseClauses, RecClauses, MemoEnabled, RCode) :-
    % Generate memo declaration
    (   MemoEnabled = true ->
        format(string(MemoDecl), '~w_memo <- new.env(hash=TRUE, parent=emptyenv())', [PredStr])
    ;   MemoDecl = '# Memoization disabled'
    ),

    % Generate base case if-blocks
    findall(BaseCaseCode, (
        member(clause(BHead, _), BaseClauses),
        BHead =.. [_P, BInput, BOutput],
        format(string(BaseCaseCode), '    if (n == ~w) return(~w)', [BInput, BOutput])
    ), BaseCaseCodes),
    atomic_list_concat(BaseCaseCodes, '\n', BaseCaseStr),

    % Generate memo check (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoCheck),
'    key <- as.character(n)
    if (!is.null(~w_memo[[key]])) return(~w_memo[[key]])', [PredStr, PredStr])
    ;   MemoCheck = ''
    ),

    % Extract recursive calls count and aggregation from RecClauses
    RecClauses = [clause(RecHead, RecBody)|_],
    RecHead =.. [Pred, _InputVar, _OutputVar],
    findall(Call, (extract_goal_from_body(RecBody, Call), functor(Call, Pred, 2)), RecCalls),
    find_aggregation_expr(RecBody, AggExpr),
    length(RecCalls, NumCalls),

    % Generate recursive call lines
    multicall_recursive_calls_r(PredStr, NumCalls, RecCallsCode),

    % Generate aggregation
    multicall_aggregation_r(AggExpr, NumCalls, AggCode),

    % Generate memo store (if enabled)
    (   MemoEnabled = true ->
        format(string(MemoStore), '    ~w_memo[[key]] <<- result', [PredStr])
    ;   MemoStore = ''
    ),

    % Assemble
    format(string(RCode),
'#!/usr/bin/env Rscript
# ~w - multi-call linear recursion with memoization (R)
# Pattern: Multiple independent recursive calls (e.g., fibonacci)

~w

~w <- function(n) {
~w

~w

    # Recursive calls
~w

    # Aggregate results
~w

~w
    return(result)
}

# Run when script executed directly
if (!interactive()) {
    args <- commandArgs(TRUE)
    if (length(args) >= 1) cat(~w(as.integer(args[1])), "\\n")
}
', [PredStr, MemoDecl, PredStr, BaseCaseStr, MemoCheck,
    RecCallsCode, AggCode, MemoStore, PredStr]).

%% multicall_recursive_calls_r(+PredStr, +NumCalls, -Code)
multicall_recursive_calls_r(PredStr, 2, Code) :-
    format(string(Code), '    result1 <- ~w(n - 1)\n    result2 <- ~w(n - 2)', [PredStr, PredStr]).
multicall_recursive_calls_r(PredStr, 3, Code) :-
    format(string(Code), '    result1 <- ~w(n - 1)\n    result2 <- ~w(n - 2)\n    result3 <- ~w(n - 3)', [PredStr, PredStr, PredStr]).

%% multicall_aggregation_r(+Expr, +NumCalls, -Code)
multicall_aggregation_r(_ + _, 2, Code) :-
    format(string(Code), '    result <- result1 + result2', []).
multicall_aggregation_r(_ * _, 2, Code) :-
    format(string(Code), '    result <- result1 * result2', []).
multicall_aggregation_r(_ + _ + _, 3, Code) :-
    format(string(Code), '    result <- result1 + result2 + result3', []).

% ============================================================================
% DIRECT MULTI-CALL RECURSION - R target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/direct_multi_call_recursion').
:- multifile direct_multi_call_recursion:compile_direct_multicall_pattern/5.

direct_multi_call_recursion:compile_direct_multicall_pattern(r, PredStr, BaseClauses, RecClause, RCode) :-
    % Extract base cases for R
    direct_extract_base_cases_r(BaseClauses, PredStr, BaseCasesCode),

    % Extract recursive case structure
    RecClause = clause(RecHead, RecBody),
    RecHead =.. [Pred, _InputVar, _OutputVar],
    extract_body_components(RecBody, Pred, Computations, RecCalls, Aggregation),

    % Generate R code for each part
    direct_computations_r(Computations, ComputationsCode),
    direct_recursive_calls_r(RecCalls, PredStr, RecCallsCode),
    direct_aggregation_r(Aggregation, AggregationCode),

    % Assemble
    format(string(RCode),
'#!/usr/bin/env Rscript
# ~w - direct recursive with memoization (multi-call pattern, R)
# Generated by direct_multi_call_recursion compiler

~w_memo <- new.env(hash=TRUE, parent=emptyenv())

~w <- function(input) {
    # Check memo
    key <- as.character(input)
    if (!is.null(~w_memo[[key]])) return(~w_memo[[key]])

~w

    # Computations
~w

    # Recursive calls
~w

    # Aggregate
~w

    # Memoize
    ~w_memo[[key]] <<- result
    return(result)
}

# Run when script executed directly
if (!interactive()) {
    args <- commandArgs(TRUE)
    if (length(args) >= 1) cat(~w(as.integer(args[1])), "\\n")
}
', [PredStr, PredStr, PredStr, PredStr, PredStr,
    BaseCasesCode, ComputationsCode, RecCallsCode, AggregationCode,
    PredStr, PredStr]).

%% direct_extract_base_cases_r(+BaseClauses, +PredStr, -RCode)
direct_extract_base_cases_r(BaseClauses, PredStr, RCode) :-
    findall(BaseCode, (
        member(clause(Head, _Body), BaseClauses),
        Head =.. [_Pred, BaseInput, BaseOutput],
        format(string(BaseCode),
'    # Base case: ~w -> ~w
    if (input == ~w) { ~w_memo[[key]] <<- ~w; return(~w) }',
            [BaseInput, BaseOutput, BaseInput, PredStr, BaseOutput, BaseOutput])
    ), BaseCodes),
    atomic_list_concat(BaseCodes, '\n', RCode).

%% direct_computations_r(+Computations, -RCode)
direct_computations_r(Computations, RCode) :-
    findall(Code, (
        member(Var is Expr, Computations),
        direct_var_to_r_name(Var, VarName),
        direct_translate_expr_to_r(Expr, RExpr),
        format(string(Code), '    ~w <- ~w', [VarName, RExpr])
    ), Codes),
    atomic_list_concat(Codes, '\n', RCode).

%% direct_recursive_calls_r(+RecCalls, +PredStr, -RCode)
direct_recursive_calls_r(RecCalls, PredStr, RCode) :-
    findall(Code, (
        member(RecCall, RecCalls),
        RecCall =.. [_Pred, ArgVar, ResultVar],
        direct_var_to_r_name(ArgVar, ArgName),
        direct_var_to_r_name(ResultVar, ResName),
        format(string(Code), '    ~w <- ~w(~w)', [ResName, PredStr, ArgName])
    ), Codes),
    atomic_list_concat(Codes, '\n', RCode).

%% direct_aggregation_r(+Aggregation, -RCode)
direct_aggregation_r(_Result is Expr, RCode) :-
    direct_translate_aggregation_expr_r(Expr, RExpr),
    format(string(RCode), '    result <- ~w', [RExpr]).

direct_translate_aggregation_expr_r(A + B, RExpr) :-
    direct_var_to_r_name(A, AName),
    direct_var_to_r_name(B, BName),
    format(string(RExpr), '~w + ~w', [AName, BName]).
direct_translate_aggregation_expr_r(A + B + C, RExpr) :-
    direct_var_to_r_name(A, AName),
    direct_var_to_r_name(B, BName),
    direct_var_to_r_name(C, CName),
    format(string(RExpr), '~w + ~w + ~w', [AName, BName, CName]).
direct_translate_aggregation_expr_r(A * B, RExpr) :-
    direct_var_to_r_name(A, AName),
    direct_var_to_r_name(B, BName),
    format(string(RExpr), '~w * ~w', [AName, BName]).

%% direct_var_to_r_name(+Var, -RName)
%  Convert Prolog variable to valid R identifier.
%  R identifiers must start with a letter or dot-not-followed-by-digit.
%  Prolog internal var names like _G12345 become v12345 in R.
direct_var_to_r_name(Var, RName) :-
    (   var(Var) ->
        term_string(Var, VarStr),
        atom_string(VarAtom, VarStr),
        downcase_atom(VarAtom, LowerName),
        ensure_r_identifier(LowerName, RName)
    ;   atom(Var) ->
        downcase_atom(Var, LowerName),
        ensure_r_identifier(LowerName, RName)
    ;   term_string(Var, VarStr),
        atom_string(VarAtom, VarStr),
        downcase_atom(VarAtom, LowerName),
        ensure_r_identifier(LowerName, RName)
    ).

%% ensure_r_identifier(+Name, -ValidName)
%  Prefix with 'v' if name starts with underscore or digit (invalid in R)
ensure_r_identifier(Name, ValidName) :-
    atom_chars(Name, [First|Rest]),
    (   (First = '_' ; char_type(First, digit)) ->
        atom_chars(ValidName, [v|Rest])
    ;   ValidName = Name
    ).

%% direct_translate_expr_to_r(+Expr, -RExpr)
direct_translate_expr_to_r(N - K, RExpr) :- var(N), integer(K), !,
    format(string(RExpr), 'input - ~w', [K]).
% SWI-Prolog stores N-K as N+(-K) in clause database
direct_translate_expr_to_r(N + K, RExpr) :- var(N), integer(K), K < 0, !,
    AbsK is abs(K),
    format(string(RExpr), 'input - ~w', [AbsK]).
direct_translate_expr_to_r(N + K, RExpr) :- var(N), integer(K), !,
    format(string(RExpr), 'input + ~w', [K]).
direct_translate_expr_to_r(Expr, 'input') :- var(Expr), !.

% ============================================================================
% LINEAR RECURSION - R target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/linear_recursion').
:- multifile linear_recursion:compile_linear_pattern/8.

%% generate_r_seq(+BaseInput, +Step, +Direction, -SeqExpr)
%  Generate the R seq() expression for a loop range based on extracted step info.
%  The loop iterates over values the input variable takes, excluding the base case.
%  Direction=down: recursion counts down from n toward base, loop mirrors that.
%  Direction=up:   recursion counts up from base toward n, loop mirrors that.
generate_r_seq(BaseInput, Step, down, SeqExpr) :- !,
    LoopEnd is BaseInput + Step,
    (   Step =:= 1 ->
        format(string(SeqExpr), 'seq(n, ~w)', [LoopEnd])
    ;   NegStep is -Step,
        format(string(SeqExpr), 'seq(n, ~w, by = ~w)', [LoopEnd, NegStep])
    ).
generate_r_seq(BaseInput, Step, up, SeqExpr) :-
    LoopStart is BaseInput + Step,
    (   Step =:= 1 ->
        format(string(SeqExpr), 'seq(~w, n)', [LoopStart])
    ;   format(string(SeqExpr), 'seq(~w, n, by = ~w)', [LoopStart, Step])
    ).

linear_recursion:compile_linear_pattern(r, PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, RCode) :-
    (   Arity =:= 2 ->
        linear_fold_based_r(PredStr, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, RCode)
    ;   linear_generic_r(PredStr, Arity, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, RCode)
    ).

%% linear_fold_based_r(+PredStr, +BaseClauses, +RecClauses, +MemoEnabled, +MemoStrategy, -RCode)
linear_fold_based_r(PredStr, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, RCode) :-
    extract_base_case_info(BaseClauses, BaseInput, BaseOutput),
    detect_input_type(BaseInput, InputType),
    extract_fold_operation(RecClauses, FoldExpr),
    (   InputType = numeric ->
        linear_numeric_fold_r(PredStr, BaseInput, BaseOutput, FoldExpr, MemoEnabled, MemoStrategy, RCode)
    ;   InputType = list ->
        linear_list_fold_r(PredStr, BaseInput, BaseOutput, FoldExpr, MemoEnabled, MemoStrategy, RCode)
    ;   linear_generic_r(PredStr, 2, BaseClauses, RecClauses, MemoEnabled, MemoStrategy, RCode)
    ).

%% linear_numeric_fold_r(+PredStr, +BaseInput, +BaseOutput, +_FoldExpr, +MemoEnabled, +MemoStrategy, -RCode)
%  When memo disabled, generate a simple for-loop instead of Reduce + memo
linear_numeric_fold_r(PredStr, BaseInput, BaseOutput, _FoldExpr, false, _MemoStrategy, RCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(linear_recursion:is_recursive_clause(Pred), Clauses, RecClauses, _BaseClauses),

    RecClauses = [clause(RHead, RBody)|_],
    RHead =.. [_Pred, InputVar, _OutputVar],
    find_recursive_call(RBody, RecCall),
    RecCall =.. [_RecPred, _RecInput, AccVar],
    find_last_is_expression(RBody, _ is ActualFoldExpr),

    translate_fold_expr(ActualFoldExpr, InputVar, AccVar, RFoldOp),
    % In the loop, i is 'current' and result is 'acc'
    % RFoldOp uses 'current' and 'acc' — replace with 'i' and 'result'
    atomic_list_concat(Parts1, 'current', RFoldOp),
    atomic_list_concat(Parts1, 'i', RLoopOp1),
    atomic_list_concat(Parts2, 'acc', RLoopOp1),
    atomic_list_concat(Parts2, 'result', RLoopOp),

    % Extract loop range from the recursion step — fail if not derivable
    extract_step_info(RecClauses, _, Step, Direction),
    generate_r_seq(BaseInput, Step, Direction, SeqExpr),

    format(string(RCode), '# ~w - linear recursion (numeric, loop-based, R)

~w <- function(n) {
    if (n == ~w) return(~w)
    result <- ~w
    for (i in ~w) {
        result <- ~w
    }
    return(result)
}

# Run when script executed directly
if (!interactive()) {
    args <- commandArgs(TRUE)
    if (length(args) >= 1) cat(~w(as.integer(args[1])), "\\n")
}
', [PredStr, PredStr, BaseInput, BaseOutput, BaseOutput, SeqExpr, RLoopOp, PredStr]).

%  When memo enabled, use Reduce + memoization
linear_numeric_fold_r(PredStr, BaseInput, BaseOutput, _FoldExpr, MemoEnabled, MemoStrategy, RCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(linear_recursion:is_recursive_clause(Pred), Clauses, RecClauses, _BaseClauses),

    RecClauses = [clause(RHead, RBody)|_],
    RHead =.. [_Pred, InputVar, _OutputVar],
    find_recursive_call(RBody, RecCall),
    RecCall =.. [_RecPred, _RecInput, AccVar],
    find_last_is_expression(RBody, _ is ActualFoldExpr),

    translate_fold_expr(ActualFoldExpr, InputVar, AccVar, RFoldOp),

    % Extract loop range from the recursion step — fail if not derivable
    extract_step_info(RecClauses, _, Step, Direction),
    generate_r_seq(BaseInput, Step, Direction, SeqExpr),

    (   MemoEnabled = true ->
        format(string(MemoDecl), '# Memoization table (~w strategy)~n~w_memo <- new.env(hash=TRUE, parent=emptyenv())~n', [MemoStrategy, PredStr]),
        format(string(MemoCheckCode), '    # Check memo~n    key <- as.character(n)~n    if (!is.null(~w_memo[[key]])) {~n        cached <- ~w_memo[[key]]~n        if (!is.null(expected)) {~n            if (cached == expected) return(TRUE) else return(FALSE)~n        } else {~n            return(cached)~n        }~n    }~n', [PredStr, PredStr]),
        format(string(MemoStoreCode), '    # Memoize~n    ~w_memo[[key]] <- result~n', [PredStr])
    ;   MemoDecl = '# Memoization disabled\n',
        MemoCheckCode = '',
        MemoStoreCode = ''
    ),

    format(string(RCode), '# ~w - fold-based linear recursion (numeric, R)

~w

~w_op <- function(current, acc) {
    return(~w)
}

~w <- function(n, expected=NULL) {
~w
    if (n == ~w) {
        result <- ~w
~w
        if (!is.null(expected)) {
            if (result == expected) return(TRUE) else return(FALSE)
        } else {
            return(result)
        }
    }

    # Recursive case using Reduce
    range_vals <- ~w
    result <- Reduce(~w_op, range_vals, init=~w)

~w
    if (!is.null(expected)) {
        if (result == expected) return(TRUE) else return(FALSE)
    } else {
        return(result)
    }
}

# Run when script executed directly
if (!interactive()) {
    args <- commandArgs(TRUE)
    if (length(args) >= 1) cat(~w(as.integer(args[1])), "\\n")
}
', [PredStr, MemoDecl, PredStr, RFoldOp, PredStr, MemoCheckCode, BaseInput, BaseOutput, MemoStoreCode, SeqExpr, PredStr, BaseOutput, MemoStoreCode, PredStr]).

%% linear_list_fold_r(+PredStr, +BaseInput, +BaseOutput, +_FoldExpr, +MemoEnabled, +MemoStrategy, -RCode)
linear_list_fold_r(PredStr, BaseInput, BaseOutput, _FoldExpr, MemoEnabled, MemoStrategy, RCode) :-
    atom_string(Pred, PredStr),
    functor(Head, Pred, 2),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),
    partition(linear_recursion:is_recursive_clause(Pred), Clauses, RecClauses, _BaseClauses),

    RecClauses = [clause(RHead, RBody)|_],
    RHead =.. [_Pred, InputVar, _OutputVar],
    find_recursive_call(RBody, RecCall),
    RecCall =.. [_RecPred, _RecInput, AccVar],
    find_last_is_expression(RBody, _ is ActualFoldExpr),

    % For list patterns, InputVar is [H|T] — extract the head element variable
    (   InputVar = [HeadVar|_] -> true ; HeadVar = InputVar ),
    translate_fold_expr(ActualFoldExpr, HeadVar, AccVar, RFoldOp),

    (   MemoEnabled = true ->
        format(string(MemoDecl), '# Memoization table (~w strategy)~n~w_memo <- new.env(hash=TRUE, parent=emptyenv())~n', [MemoStrategy, PredStr]),
        format(string(MemoCheckCode), '    # Check memo~n    key <- if (length(lst) == 0) "__empty__" else paste(lst, collapse=",")~n    if (!is.null(~w_memo[[key]])) {~n        cached <- ~w_memo[[key]]~n        if (!is.null(expected)) {~n            if (cached == expected) return(TRUE) else return(FALSE)~n        } else {~n            return(cached)~n        }~n    }~n', [PredStr, PredStr]),
        format(string(MemoStoreCode), '    # Memoize~n    ~w_memo[[key]] <- result~n', [PredStr])
    ;   MemoDecl = '# Memoization disabled\n',
        MemoCheckCode = '',
        MemoStoreCode = ''
    ),

    format(string(RCode), '# ~w - fold-based linear recursion (list, R)

~w

~w_op <- function(acc, current) {
    return(~w)
}

~w <- function(lst_input, expected=NULL) {
    lst <- lst_input
    if (is.character(lst)) {
        lst <- unlist(strsplit(gsub("\\\\[|\\\\]", "", lst), ","))
        if(all(!is.na(suppressWarnings(as.numeric(lst))))) lst <- as.numeric(lst)
    }

~w
    if (length(lst) == 0 || (length(lst) == 1 && lst[1] == "~w")) {
        result <- ~w
~w
        if (!is.null(expected)) {
            if (result == expected) return(TRUE) else return(FALSE)
        } else {
            return(result)
        }
    }

    # Recursive case using Reduce
    result <- Reduce(~w_op, lst, accumulate=FALSE, init=~w)

~w
    if (!is.null(expected)) {
        if (result == expected) return(TRUE) else return(FALSE)
    } else {
        return(result)
    }
}

# Run when script executed directly
if (!interactive()) {
    args <- commandArgs(TRUE)
    if (length(args) >= 1) {
        items <- as.numeric(unlist(strsplit(args[1], ",")))
        cat(~w(items), "\\n")
    }
}
', [PredStr, MemoDecl, PredStr, RFoldOp, PredStr, MemoCheckCode, BaseInput, BaseOutput, MemoStoreCode, PredStr, BaseOutput, MemoStoreCode, PredStr]).

%% linear_generic_r(+PredStr, +Arity, +BaseClauses, +RecClauses, +MemoEnabled, +MemoStrategy, -RCode)
linear_generic_r(PredStr, Arity, _BaseClauses, _RecClauses, MemoEnabled, _MemoStrategy, RCode) :-
    (   MemoEnabled = true ->
        MemoDecl = '~w_memo <- new.env(hash=TRUE, parent=emptyenv())',
        format(string(MemoDeclFormatted), MemoDecl, [PredStr])
    ;   MemoDeclFormatted = '# Memoization disabled'
    ),
    TemplateLines = [
        "# {{pred}}/{{arity}} - linear recursive pattern (generic R)",
        "{{memo_decl}}",
        "{{pred}} <- function(...) {",
        "    args <- list(...)",
        "    key <- paste(args, collapse=\"-\")",
        "    if (!is.null({{{pred}}_memo[[key]]})) {",
        "        return({{{pred}}_memo[[key]]})",
        "    }",
        "    warning(\"Generic linear recursion - not yet implemented in R\")",
        "    return(NULL)",
        "}"
    ],
    atomic_list_concat(TemplateLines, '\n', Template),
    render_template(Template, [pred=PredStr, arity=Arity, memo_decl=MemoDeclFormatted], RCode).

% ============================================================================
% MUTUAL RECURSION - R target delegation (multifile)
% ============================================================================

:- use_module('../core/advanced/mutual_recursion').
:- multifile mutual_recursion:compile_mutual_pattern/5.

mutual_recursion:compile_mutual_pattern(r, Predicates, MemoEnabled, MemoStrategy, RCode) :-
    findall(PredStr,
        (   member(Pred/_Arity, Predicates),
            atom_string(Pred, PredStr)
        ),
        PredStrs),
    atomic_list_concat(PredStrs, '_', GroupName),

    % Generate R header
    (   MemoEnabled = true ->
        format(string(MemoDecl), '~w_memo <- new.env(hash=TRUE, parent=emptyenv())', [GroupName])
    ;   MemoDecl = '# Shared memoization disabled'
    ),
    format(string(HeaderCode),
'#!/usr/bin/env Rscript
# Mutually recursive group: ~w
# Constraints: memo=~w, strategy=~w

~w
', [GroupName, MemoEnabled, MemoStrategy, MemoDecl]),

    % Generate function for each predicate
    findall(FuncCode,
        (   member(Pred/Arity, Predicates),
            mutual_function_r(Pred, Arity, GroupName, Predicates, MemoEnabled, FuncCode)
        ),
        FuncCodes),
    atomic_list_concat(FuncCodes, '\n\n', FunctionsCode),

    % Generate main dispatch
    mutual_main_dispatch_r(Predicates, DispatchCode),

    % Combine
    atomic_list_concat([HeaderCode, FunctionsCode, '\n\n', DispatchCode], RCode).

%% mutual_function_r(+Pred, +Arity, +GroupName, +AllPredicates, +MemoEnabled, -FuncCode)
mutual_function_r(Pred, Arity, GroupName, AllPredicates, MemoEnabled, FuncCode) :-
    atom_string(Pred, PredStr),

    % Get clauses
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), Clauses),

    % Separate base and recursive cases
    partition(mutual_recursion:is_mutual_recursive_clause(AllPredicates), Clauses, RecClauses, BaseClauses),

    % Generate base case code
    mutual_base_cases_r(GroupName, BaseClauses, MemoEnabled, BaseCode),

    % Generate recursive case code
    mutual_recursive_cases_r(GroupName, RecClauses, MemoEnabled, RecCode),

    % Memo check
    (   MemoEnabled = true ->
        format(string(MemoCheck),
'    key <- paste0("~w:", arg1)
    if (!is.null(~w_memo[[key]])) return(~w_memo[[key]])', [PredStr, GroupName, GroupName])
    ;   format(string(MemoCheck), '    # Memoization disabled', [])
    ),

    format(string(FuncCode),
'# ~w/~w - part of mutual recursion group
~w <- function(arg1) {
~w

~w

~w

    # No match found
    return(FALSE)
}
', [PredStr, Arity, PredStr, MemoCheck, BaseCode, RecCode]).

%% mutual_base_cases_r(+GroupName, +BaseClauses, +MemoEnabled, -BaseCode)
mutual_base_cases_r(_GroupName, [], _MemoEnabled, '    # 0 base case(s)') :- !.
mutual_base_cases_r(GroupName, BaseClauses, MemoEnabled, BaseCode) :-
    length(BaseClauses, NumBase),
    format(string(Header), '    # ~w base case(s)', [NumBase]),
    findall(Code,
        (   member(Clause, BaseClauses),
            mutual_base_case_r(GroupName, Clause, MemoEnabled, Code)
        ),
        BaseCodes),
    atomic_list_concat([Header|BaseCodes], '\n', BaseCode).

%% mutual_base_case_r(+GroupName, +Clause, +MemoEnabled, -Code)
mutual_base_case_r(GroupName, clause(Head, true), MemoEnabled, Code) :-
    Head =.. [_Pred, Value],
    (   MemoEnabled = true ->
        format(string(MemoStore), ' ~w_memo[[key]] <<- TRUE;', [GroupName])
    ;   MemoStore = ''
    ),
    format(string(Code),
'    if (arg1 == ~w) {~w return(TRUE) }',
        [Value, MemoStore]).

%% mutual_recursive_cases_r(+GroupName, +RecClauses, +MemoEnabled, -RecCode)
mutual_recursive_cases_r(_GroupName, [], _MemoEnabled, '    # 0 recursive case(s)') :- !.
mutual_recursive_cases_r(GroupName, RecClauses, MemoEnabled, RecCode) :-
    length(RecClauses, NumRec),
    format(string(Header), '    # ~w recursive case(s)', [NumRec]),
    findall(Code,
        (   member(Clause, RecClauses),
            mutual_recursive_case_r(GroupName, Clause, MemoEnabled, Code)
        ),
        RecCodes),
    atomic_list_concat([Header|RecCodes], '\n', RecCode).

%% mutual_recursive_case_r(+GroupName, +Clause, +MemoEnabled, -Code)
mutual_recursive_case_r(GroupName, clause(Head, Body), MemoEnabled, Code) :-
    Head =.. [_Pred, HeadVar],
    mutual_recursion:parse_recursive_body(Body, HeadVar, Conditions, Computations, RecCall),

    % Generate condition
    (   Conditions = [] ->
        CondStr = 'TRUE'
    ;   maplist(mutual_condition_r(HeadVar), Conditions, CondCodes),
        atomic_list_concat(CondCodes, ' && ', CondStr)
    ),

    % Generate computations
    maplist(mutual_computation_r(HeadVar), Computations, CompCodes),
    (   CompCodes = [] -> CompCode = ''
    ;   atomic_list_concat(CompCodes, '\n        ', CompCode)
    ),

    % Generate recursive call
    (   RecCall = none ->
        RecCode = ''
    ;   mutual_rec_call_r(HeadVar, RecCall, GroupName, MemoEnabled, RecCode)
    ),

    format(string(Code),
'    if (~w) {
        ~w
        ~w
    }', [CondStr, CompCode, RecCode]).

%% mutual_condition_r(+HeadVar, +Goal, -Code)
mutual_condition_r(HeadVar, Goal, Code) :-
    Goal =.. [Op, A, B],
    mutual_term_r(HeadVar, A, TermA),
    mutual_term_r(HeadVar, B, TermB),
    mutual_comparison_op(Op, ROp),
    format(string(Code), '~w ~w ~w', [TermA, ROp, TermB]).

mutual_comparison_op(>, '>').
mutual_comparison_op(<, '<').
mutual_comparison_op(>=, '>=').
mutual_comparison_op(=<, '<=').
mutual_comparison_op(=:=, '==').
mutual_comparison_op(==, '==').

%% mutual_computation_r(+HeadVar, +Goal, -Code)
mutual_computation_r(HeadVar, VarOut is Expr, Code) :-
    mutual_expr_r(HeadVar, Expr, RExpr),
    mutual_var_to_r_safe(VarOut, VarOutR),
    format(string(Code), '~w <- ~w', [VarOutR, RExpr]).

%% mutual_rec_call_r(+HeadVar, +RecCall, +GroupName, +MemoEnabled, -Code)
mutual_rec_call_r(HeadVar, RecCall, GroupName, MemoEnabled, Code) :-
    RecCall =.. [Pred|Args],
    atom_string(Pred, PredStr),
    maplist(mutual_call_arg_r(HeadVar), Args, RArgs),
    atomic_list_concat(RArgs, ', ', ArgsStr),
    (   MemoEnabled = true ->
        format(string(MemoStore), ' ~w_memo[[key]] <<- TRUE;', [GroupName])
    ;   MemoStore = ''
    ),
    format(string(Code),
'rec_result <- ~w(~w)
        if (isTRUE(rec_result)) {~w return(TRUE) } else { return(FALSE) }',
        [PredStr, ArgsStr, MemoStore]).

%% mutual_term_r(+HeadVar, +Term, -RTerm)
mutual_term_r(HeadVar, Term, 'arg1') :-
    Term == HeadVar, !.
mutual_term_r(_HeadVar, Term, RTerm) :-
    var(Term), !,
    mutual_var_to_r_safe(Term, RTerm).
mutual_term_r(_HeadVar, Term, RTerm) :-
    atomic(Term),
    format(string(RTerm), '~w', [Term]).

%% mutual_expr_r(+HeadVar, +Expr, -RExpr)
mutual_expr_r(HeadVar, Var, 'arg1') :-
    Var == HeadVar, !.
mutual_expr_r(_HeadVar, Var, RExpr) :-
    var(Var), !,
    mutual_var_to_r_safe(Var, RExpr).
mutual_expr_r(_HeadVar, Atom, RExpr) :-
    atomic(Atom), !,
    format(string(RExpr), '~w', [Atom]).
mutual_expr_r(HeadVar, A + B, RExpr) :-
    !,
    (   (number(B), B < 0) ->
        mutual_expr_r(HeadVar, A, RA),
        Pos is -B,
        format(string(RExpr), '~w - ~w', [RA, Pos])
    ;   mutual_expr_r(HeadVar, A, RA),
        mutual_expr_r(HeadVar, B, RB),
        format(string(RExpr), '~w + ~w', [RA, RB])
    ).
mutual_expr_r(HeadVar, A - B, RExpr) :-
    !,
    mutual_expr_r(HeadVar, A, RA),
    mutual_expr_r(HeadVar, B, RB),
    format(string(RExpr), '~w - ~w', [RA, RB]).
mutual_expr_r(HeadVar, A * B, RExpr) :-
    !,
    mutual_expr_r(HeadVar, A, RA),
    mutual_expr_r(HeadVar, B, RB),
    format(string(RExpr), '~w * ~w', [RA, RB]).

%% mutual_call_arg_r(+HeadVar, +Arg, -RArg)
mutual_call_arg_r(HeadVar, Var, 'arg1') :-
    Var == HeadVar, !.
mutual_call_arg_r(_HeadVar, Var, RArg) :-
    var(Var), !,
    mutual_var_to_r_safe(Var, RArg).
mutual_call_arg_r(_HeadVar, Atom, RArg) :-
    atomic(Atom), !,
    format(string(RArg), '~w', [Atom]).

%% mutual_var_to_r_safe(+Var, -RName)
%  Convert a Prolog variable to a valid R identifier.
%  R doesn't allow identifiers like _21562 (underscore + digits).
%  Prefix with 'v' to make them safe.
mutual_var_to_r_safe(Var, RName) :-
    format(atom(VarAtom), '~w', [Var]),
    atom_string(VarAtom, VarStr),
    string_lower(VarStr, VarLower),
    atom_string(VarLowerAtom, VarLower),
    (   sub_atom(VarLowerAtom, 0, 1, _, '_') ->
        atom_concat(v, VarLowerAtom, SafeAtom),
        atom_string(SafeAtom, RName)
    ;   RName = VarLower
    ).

%% mutual_main_dispatch_r(+Predicates, -DispatchCode)
mutual_main_dispatch_r(Predicates, DispatchCode) :-
    findall(CaseCode,
        (   member(Pred/Arity, Predicates),
            atom_string(Pred, PredStr),
            (   Arity > 0 ->
                format(string(CaseCode), '        "~w" = cat(~w(as.numeric(args[2])), "\\n")', [PredStr, PredStr])
            ;   format(string(CaseCode), '        "~w" = cat(~w(), "\\n")', [PredStr, PredStr])
            )
        ),
        CaseCodes),
    atomic_list_concat(CaseCodes, ',\n', CasesStr),

    format(string(DispatchCode),
'# Main dispatch
if (!interactive()) {
    args <- commandArgs(TRUE)
    if (length(args) < 1) {
        cat("Usage: Rscript <script> <function_name> [args...]\\n", file=stderr())
        quit(status=1)
    }
    switch(args[1],
~w,
        stop(paste("Unknown function:", args[1]))
    )
}
', [CasesStr]).

% ============================================================================
% TESTING
% ============================================================================

test_r_pipeline :-
    init_r_target,
    asserta((user:filter_stage(In, Out) :- filter(In, >(age, 18), Out))),
    asserta((user:sort_stage(In, Out) :- sort_by(In, age, Out))),
    asserta((user:group_stage(In, Out) :- group_by(In, role, Out))),
    
    format('~n=== R Sequential Pipeline Test ===~n'),
    compile_r_pipeline([filter_stage/2, sort_stage/2, group_stage/2], [pipeline_name(data_processor), pipeline_mode(sequential)], CodeSeq),
    format('~s~n', [CodeSeq]),
    
    format('~n=== R Fixpoint Generator Pipeline Test ===~n'),
    compile_r_pipeline([filter_stage/2, sort_stage/2, group_stage/2], [pipeline_name(data_processor_gen), pipeline_mode(generator)], CodeGen),
    format('~s~n', [CodeGen]),
    
    retractall(user:filter_stage(_, _)),
    retractall(user:sort_stage(_, _)),
    retractall(user:group_stage(_, _)).
