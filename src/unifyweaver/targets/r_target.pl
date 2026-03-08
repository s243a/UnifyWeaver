% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% r_target.pl - R Target for UnifyWeaver
% Generates standalone R scripts for record/field processing.
% Supports vectorized operations, data frames, and native pipes.

:- module(r_target, [
    compile_predicate_to_r/3,       % +Predicate, +Options, -RCode
    init_r_target/0,                % Initialize R target with bindings
    compile_r_pipeline/3,           % +Predicates, +Options, -RCode
    test_r_pipeline/0               % Test pipeline generation
]).

:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module(common_generator).
:- use_module('../core/binding_registry').
:- use_module('../bindings/r_bindings').

% Track required packages/libraries
:- dynamic required_r_package/1.

%% init_r_target
%  Initialize the R target by loading bindings and clearing state.
init_r_target :-
    retractall(required_r_package(_)),
    init_r_bindings.

%% compile_predicate_to_r(+PredIndicator, +Options, -RCode)
%  Compile a Prolog predicate indicator into an R function.
compile_predicate_to_r(PredIndicator, _Options, RCode) :-
    PredIndicator = Pred/Arity,
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(user:Head, Body), Clauses),
    (   Clauses = [Head-Body]
    ->  compile_rule(Head, Body, RCode)
    ;   format(string(RCode), '# Multiple clauses not supported yet for ~w/~w', [Pred, Arity])
    ).

compile_rule(Head, Body, Code) :-
    Head =.. [Pred|Args],
    
    % Generate variable mapping for arguments
    map_args(Args, 1, VarMap, ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList),
    
    % Compile body
    compile_body(Body, VarMap, _, BodyCode),
    
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
    format(string(StrVal), '~w', [Val]).

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
