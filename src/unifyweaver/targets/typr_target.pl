:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% typr_target.pl - TypR code generation target

:- module(typr_target, [
    target_info/1,
    compile_predicate/3,
    compile_predicate_to_typr/3,
    generated_typr_is_valid/2
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(readutil)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../core/template_system').
:- use_module('r_target', [compile_predicate_to_r/3, init_r_target/0, infer_clauses_return_type/2]).
:- use_module('type_declarations').

target_info(info{
    name: "TypR",
    family: r,
    file_extension: ".typr",
    runtime: typr,
    features: [types, gradual_typing, s3, transpiles_to_r],
    recursion_patterns: [transitive_closure],
    compile_command: "typr"
}).

compile_predicate(PredArity, Options, Code) :-
    compile_predicate_to_typr(PredArity, Options, Code).

compile_predicate_to_typr(PredIndicator, Options, Code) :-
    pred_indicator_parts(PredIndicator, Module, Pred, Arity),
    option(global_typed_mode(GlobalMode), Options, infer),
    resolve_typed_mode(Pred/Arity, Options, GlobalMode, TypedMode),
    (   option(base_pred(BasePredOption), Options)
    ->  normalize_base_pred(BasePredOption, BasePred),
        compile_typr_transitive_closure(Module, Pred/Arity, BasePred, TypedMode, Code)
    ;   detect_transitive_closure(Module, Pred, Arity, BasePred)
    ->  compile_typr_transitive_closure(Module, Pred/Arity, BasePred, TypedMode, Code)
    ;   compile_generic_typr(Module, Pred/Arity, Options, TypedMode, Code)
    ).

compile_typr_transitive_closure(Module, Pred/Arity, BasePred, TypedMode, Code) :-
    resolve_node_type(Pred/Arity, BasePred/2, NodeTypeTerm),
    read_file_to_string('templates/targets/typr/transitive_closure.mustache', Template, []),
    atom_string(Pred, PredStr),
    atom_string(BasePred, BaseStr),
    annotation_suffix(TypedMode, NodeTypeTerm, NodeAnnotation),
    annotation_suffix(TypedMode, NodeTypeTerm, AddFromAnnotation),
    annotation_suffix(TypedMode, NodeTypeTerm, AddToAnnotation),
    annotation_suffix(TypedMode, list(NodeTypeTerm), AllReturnAnnotation),
    annotation_suffix(TypedMode, boolean, CheckReturnAnnotation),
    empty_collection_expr(NodeTypeTerm, EmptyNodesExpr),
    base_seed_code(Module, BasePred, SeedCode),
    render_template(Template, [
        pred=PredStr,
        base=BaseStr,
        add_from_annotation=AddFromAnnotation,
        add_to_annotation=AddToAnnotation,
        node_annotation=NodeAnnotation,
        all_return_annotation=AllReturnAnnotation,
        check_return_annotation=CheckReturnAnnotation,
        empty_nodes_expr=EmptyNodesExpr,
        seed_code=SeedCode
    ], Code).

compile_generic_typr(Module, Pred/Arity, Options, TypedMode, Code) :-
    findall(Head-Body, predicate_clause(Module, Pred, Arity, Head, Body), Clauses),
    build_typr_function(Module:Pred/Arity, Options, TypedMode, Clauses, Code).

annotation_suffix(off, _TypeTerm, "") :- !.
annotation_suffix(Mode, TypeTerm, Annotation) :-
    should_emit_annotation(Mode, TypeTerm),
    !,
    resolve_type(TypeTerm, typr, ConcreteType),
    format(string(Annotation), ": ~w", [ConcreteType]).
annotation_suffix(_, _, "").

should_emit_annotation(explicit, TypeTerm) :-
    nonvar(TypeTerm).
should_emit_annotation(infer, any).
should_emit_annotation(infer, TypeTerm) :-
    compound(TypeTerm),
    TypeTerm \= boolean.
should_emit_annotation(infer, TypeTerm) :-
    atomic(TypeTerm),
    TypeTerm \= atom,
    TypeTerm \= string,
    TypeTerm \= integer,
    TypeTerm \= float,
    TypeTerm \= number,
    TypeTerm \= boolean.

resolve_node_type(PredSpec, _BasePredSpec, TypeTerm) :-
    resolved_type_term(PredSpec, none, 1, TypeTerm),
    !.
resolve_node_type(_PredSpec, BasePredSpec, TypeTerm) :-
    resolved_type_term(BasePredSpec, none, 1, TypeTerm),
    !.
resolve_node_type(_PredSpec, _BasePredSpec, atom).

resolved_type_term(PredSpec, _FallbackPredSpec, ArgIndex, TypeTerm) :-
    predicate_arg_type(PredSpec, ArgIndex, TypeTerm),
    !.
resolved_type_term(_PredSpec, FallbackPredSpec, ArgIndex, TypeTerm) :-
    FallbackPredSpec \== none,
    predicate_arg_type(FallbackPredSpec, ArgIndex, TypeTerm).

build_typed_arg_list(PredSpec, FallbackPredSpec, Arity, TypedMode, ArgList) :-
    findall(ArgName, (
        between(1, Arity, Index),
        typed_argument_name(PredSpec, FallbackPredSpec, Index, TypedMode, ArgName)
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList).

typed_argument_name(PredSpec, FallbackPredSpec, Index, TypedMode, ArgName) :-
    format(string(BaseName), 'arg~w', [Index]),
    (   resolved_type_term(PredSpec, FallbackPredSpec, Index, TypeTerm)
    ->  annotation_suffix(TypedMode, TypeTerm, Annotation),
        string_concat(BaseName, Annotation, ArgName)
    ;   ArgName = BaseName
    ).

normalize_base_pred(BasePred/_, BasePred) :- !.
normalize_base_pred(BasePred, BasePred).

pred_indicator_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
pred_indicator_parts(Pred/Arity, user, Pred, Arity).

detect_transitive_closure(Module, Pred, 2, BasePred) :-
    functor(Head, Pred, 2),
    findall(Body, clause(Module:Head, Body), Bodies),
    Bodies \= [],
    partition(calls_predicate(Pred, 2), Bodies, RecursiveBodies, BaseBodies),
    BaseBodies \= [],
    RecursiveBodies \= [],
    transitive_closure_pattern(Pred, BaseBodies, RecursiveBodies, BasePred).

calls_predicate(Pred, Arity, Goal) :-
    contains_goal(Goal, SubGoal),
    functor(SubGoal, Pred, Arity).

contains_goal((Left, _Right), Goal) :-
    contains_goal(Left, Goal).
contains_goal((_Left, Right), Goal) :-
    contains_goal(Right, Goal).
contains_goal(Goal, Goal) :-
    compound(Goal),
    Goal \= (_,_).

transitive_closure_pattern(Pred, BaseBodies, RecursiveBodies, BasePred) :-
    member(BaseBody, BaseBodies),
    BaseBody \= true,
    functor(BaseBody, BasePred, 2),
    BasePred \= Pred,
    member(RecBody, RecursiveBodies),
    RecBody = (BaseCall, RecCall),
    functor(BaseCall, BasePred, 2),
    functor(RecCall, Pred, 2),
    (   BaseCall =.. [BasePred, _X, Y],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ;   BaseCall =.. [BasePred, Y, _X],
        RecCall =.. [Pred, Y2, _Z],
        Y == Y2
    ).

predicate_clause(Module, Pred, Arity, Head, Body) :-
    functor(Head, Pred, Arity),
    clause(Module:Head, Body).

build_typr_function(Module:Pred/Arity, Options, TypedMode, Clauses, Code) :-
    atom_string(Pred, PredStr),
    build_typed_arg_list(Pred/Arity, none, Arity, TypedMode, TypedArgList),
    typr_function_body(Module:Pred/Arity, Options, TypedMode, Clauses, ReturnType, Body),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
	~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, Body]).

typr_function_body(_PredSpec, _Options, _TypedMode, [], "bool", "result <- false;\n\tresult").
typr_function_body(PredSpec, _Options, _TypedMode, Clauses, "bool", Body) :-
    all_fact_clauses(Clauses),
    fact_match_expression(PredSpec, Clauses, MatchExpr),
    !,
    format(string(Body), 'result <- @{ local({ ~w }) }@;\n\tresult', [MatchExpr]).
typr_function_body(PredSpec, Options, _TypedMode, _Clauses, ReturnType, Body) :-
    typr_declared_return_type(PredSpec, ReturnType),
    !,
    wrapped_r_body_expression(PredSpec, Options, WrappedExpr),
    format(string(Body), 'result <- @{ ~w }@;\n\tresult', [WrappedExpr]).
typr_function_body(PredSpec, Options, _TypedMode, Clauses, ReturnType, Body) :-
    inferred_typr_return_type(Clauses, ReturnType),
    !,
    wrapped_r_body_expression(PredSpec, Options, WrappedExpr),
    format(string(Body), 'result <- @{ ~w }@;\n\tresult', [WrappedExpr]).
typr_function_body(PredSpec, Options, _TypedMode, _Clauses, "Any", Body) :-
    wrapped_r_body_expression(PredSpec, Options, WrappedExpr),
    format(string(Body), 'result <- @{ ~w }@;\n\tresult', [WrappedExpr]).

typr_declared_return_type(PredSpec, ReturnType) :-
    resolved_return_type(PredSpec, typr, ReturnType).

inferred_typr_return_type(Clauses, ReturnType) :-
    infer_clauses_return_type(Clauses, AbstractType),
    resolve_type(AbstractType, typr, ReturnType).

all_fact_clauses([]).
all_fact_clauses([_-true|Rest]) :-
    all_fact_clauses(Rest).

fact_match_expression(PredSpec, Clauses, Expr) :-
    findall(ClauseExpr, (
        member(Head-true, Clauses),
        fact_clause_expression(PredSpec, Head, ClauseExpr)
    ), ClauseExprs),
    ClauseExprs \= [],
    atomic_list_concat(ClauseExprs, ' || ', Expr).

fact_clause_expression(_PredSpec, Head, Expr) :-
    Head =.. [_|HeadArgs],
    findall(Condition, (
        nth1(Index, HeadArgs, HeadArg),
        fact_arg_condition(Index, HeadArg, Condition)
    ), Conditions0),
    exclude(=(true), Conditions0, Conditions),
    (   Conditions = []
    ->  Expr = 'TRUE'
    ;   atomic_list_concat(Conditions, ' && ', Expr)
    ).

fact_arg_condition(_Index, HeadArg, true) :-
    var(HeadArg),
    !.
fact_arg_condition(Index, HeadArg, Condition) :-
    format(string(ArgName), 'arg~w', [Index]),
    r_literal(HeadArg, Literal),
    format(string(Condition), 'identical(~w, ~w)', [ArgName, Literal]).

base_seed_code(Module, BasePred, SeedCode) :-
    functor(BaseHead, BasePred, 2),
    findall(From-To, (
        clause(Module:BaseHead, true),
        BaseHead =.. [BasePred, From, To],
        nonvar(From),
        nonvar(To)
    ), Pairs),
    findall(Statement, (
        nth1(Index, Pairs, From-To),
        seed_statement(Index, BasePred, From, To, Statement)
    ), Statements),
    (   Statements = []
    ->  SeedCode = ""
    ;   atomic_list_concat(Statements, '\n', SeedCode)
    ).

seed_statement(Index, BasePred, From, To, Statement) :-
    r_literal(From, FromLiteral),
    r_literal(To, ToLiteral),
    format(string(Statement), '_seed_~w_~w <- add_~w(~w, ~w);', [BasePred, Index, BasePred, FromLiteral, ToLiteral]).

empty_collection_expr(atom, 'character()').
empty_collection_expr(string, 'character()').
empty_collection_expr(integer, 'integer()').
empty_collection_expr(float, 'numeric()').
empty_collection_expr(number, 'numeric()').
empty_collection_expr(boolean, 'logical()').
empty_collection_expr(any, 'c()').
empty_collection_expr(_, 'c()').

r_literal(Value, Literal) :-
    var(Value),
    !,
    Literal = 'NULL'.
r_literal(Value, Literal) :-
    string(Value),
    !,
    format(string(Literal), '"~s"', [Value]).
r_literal(Value, Literal) :-
    atom(Value),
    !,
    (   Value == true
    ->  Literal = 'TRUE'
    ;   Value == false
    ->  Literal = 'FALSE'
    ;   format(string(Literal), '"~w"', [Value])
    ).
r_literal(Value, Literal) :-
    number(Value),
    !,
    format(string(Literal), '~w', [Value]).
r_literal(Value, Literal) :-
    term_string(Value, Literal).

wrapped_r_body_expression(Module:Pred/Arity, Options, WrappedExpr) :-
    !,
    init_r_target,
    compile_predicate_to_r(Module:Pred/Arity, Options, RCode),
    r_function_body(RCode, Body),
    raw_arg_list(Arity, ArgList),
    format(string(WrappedExpr), '(function(~w) {\n~w\n})(~w)', [ArgList, Body, ArgList]).
wrapped_r_body_expression(Pred/Arity, Options, WrappedExpr) :-
    wrapped_r_body_expression(user:Pred/Arity, Options, WrappedExpr).

r_function_body(RCode, Body) :-
    split_string(RCode, "\n", "\n", Lines),
    Lines = [_Header|Rest],
    append(BodyLines, ["}"], Rest),
    atomic_list_concat(BodyLines, '\n', Body).

raw_arg_list(Arity, ArgList) :-
    findall(ArgName, (
        between(1, Arity, Index),
        format(string(ArgName), 'arg~w', [Index])
    ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgList).

generated_typr_is_valid(Code, Result) :-
    tmp_file(typr_validation, RootDir),
    make_directory(RootDir),
    setup_call_cleanup(
        true,
        (
            run_typr_command(RootDir, ['new', 'validation_project'], exit(0)),
            directory_file_path(RootDir, 'validation_project', ProjectDir),
            directory_file_path(ProjectDir, 'TypR/main.ty', MainFile),
            setup_call_cleanup(
                open(MainFile, write, Stream),
                write(Stream, Code),
                close(Stream)
            ),
            run_typr_command(ProjectDir, ['check'], Status),
            Result = Status
        ),
        delete_directory_and_contents(RootDir)
    ).

run_typr_command(ProjectDir, Args, Status) :-
    process_create(
        path(typr),
        Args,
        [ cwd(ProjectDir),
          stdout(pipe(Stdout)),
          stderr(pipe(Stderr)),
          process(Pid)
        ]
    ),
    read_string(Stdout, _, _),
    read_string(Stderr, _, _),
    close(Stdout),
    close(Stderr),
    process_wait(Pid, Status).
