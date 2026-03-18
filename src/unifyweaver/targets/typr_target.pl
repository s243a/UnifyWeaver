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
:- use_module('../core/binding_registry', [binding/6]).
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
    ),
    finalize_type_diagnostics_report(Options).

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
    init_r_target,
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
    typr_function_body(Module:Pred/Arity, Options, TypedMode, Clauses, ReturnType, RawBody),
    indent_text(RawBody, "\t", Body),
    format(string(Code),
'# Generated by UnifyWeaver TypR Target
# Predicate: ~w/~w

let ~w <- fn(~w): ~w {
~w
};
', [PredStr, Arity, PredStr, TypedArgList, ReturnType, Body]).

typr_function_body(_PredSpec, _Options, _TypedMode, [], "bool", "result <- false;\nresult").
typr_function_body(PredSpec, _Options, _TypedMode, Clauses, "bool", Body) :-
    all_fact_clauses(Clauses),
    fact_match_expression(PredSpec, Clauses, MatchExpr),
    !,
    format(string(Body), 'result <- @{ local({ ~w }) }@;\nresult', [MatchExpr]).
typr_function_body(PredSpec, Options, _TypedMode, Clauses, ReturnType, Body) :-
    generic_typr_return_type(PredSpec, Clauses, ReturnType),
    (   native_typr_clause_body(PredSpec, Clauses, Body)
    ->  true
    ;   wrapped_r_body_expression(PredSpec, Options, WrappedExpr),
        format(string(Body), 'result <- @{ ~w }@;\nresult', [WrappedExpr])
    ).

typr_declared_return_type(PredSpec, ReturnType) :-
    resolved_return_type(PredSpec, typr, ReturnType).

inferred_typr_return_type(Clauses, ReturnType) :-
    infer_clauses_return_type(Clauses, AbstractType),
    resolve_type(AbstractType, typr, ReturnType).

generic_typr_return_type(PredSpec, Clauses, ReturnType) :-
    (   typr_declared_return_type(PredSpec, ReturnType)
    ->  true
    ;   inferred_typr_return_type(Clauses, ReturnType)
    ->  true
    ;   ReturnType = "Any"
    ).

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

native_typr_clause_body(PredSpec, [Head-Body], Code) :-
    native_typr_clause(Head, Body, Condition, ClauseCode),
    !,
    (   Condition == 'TRUE'
    ->  Code = ClauseCode
    ;   pred_spec_name(PredSpec, PredName),
        branch_safe_typr_code(ClauseCode, BranchClauseCode),
        indent_text(BranchClauseCode, "\t", BranchCode),
        format(string(Code),
'if (~w) {
~w
} else {
	stop("No matching clause for ~w")
}', [Condition, BranchCode, PredName])
    ).
native_typr_clause_body(PredSpec, Clauses, Code) :-
    maplist(native_typr_clause_pair, Clauses, Branches),
    Branches \= [],
    pred_spec_name(PredSpec, PredName),
    branches_to_typr_if_chain(Branches, IfChain),
    format(string(Code), '~w else {\n\tstop("No matching clause for ~w")\n}', [IfChain, PredName]).

native_typr_clause_pair(Head-Body, branch(Condition, ClauseCode)) :-
    native_typr_clause(Head, Body, Condition, ClauseCode),
    !.

native_typr_clause(Head, Body, Condition, ClauseCode) :-
    Head =.. [Pred|HeadArgs],
    build_head_varmap(HeadArgs, 1, VarMap),
    typr_head_condition(HeadArgs, 1, HeadConditions),
    atom_string(Pred, PredName),
    native_typr_goal_sequence(Body, VarMap, PredName, GoalConditions, ClauseCode),
    append(HeadConditions, GoalConditions, Conditions),
    (   Conditions = []
    ->  Condition = 'TRUE'
    ;   atomic_list_concat(Conditions, ' && ', Condition)
    ).

native_typr_goal_sequence(Body, VarMap, PredName, Conditions, Code) :-
    normalize_typr_goals(Body, Goals),
    Goals \= [],
    native_typr_prefix_goals(
        Goals,
        VarMap,
        PredName,
        false,
        none,
        Conditions,
        PrefixLines,
        TailCode,
        _VarMapOut
    ),
    append(PrefixLines, [TailCode], BodyLines),
    atomic_list_concat(BodyLines, '\n', Code).

normalize_typr_goals(true, []) :- !.
normalize_typr_goals((Left, Right), Goals) :-
    !,
    normalize_typr_goals(Left, LeftGoals),
    normalize_typr_goals(Right, RightGoals),
    append(LeftGoals, RightGoals, Goals).
normalize_typr_goals(_Module:Goal, Goals) :-
    !,
    normalize_typr_goals(Goal, Goals).
normalize_typr_goals(Goal, [Goal]).

native_typr_prefix_goals([], VarMap, _PredName, _SeenOutput, none, [], [], 'true', VarMap).
native_typr_prefix_goals([], VarMap, _PredName, _SeenOutput, LastExpr, [], [], LastExpr, VarMap) :-
    LastExpr \== none.
native_typr_prefix_goals([Goal|Rest], VarMap0, PredName, _SeenOutput, _LastExpr0, Conditions, [Line|RestLines], TailCode, VarMapOut) :-
    native_typr_output_goal(Goal, VarMap0, PredName, VarMap1, Line, OutExpr),
    native_typr_prefix_goals(Rest, VarMap1, PredName, true, OutExpr, Conditions, RestLines, TailCode, VarMapOut).
native_typr_prefix_goals([Goal|Rest], VarMap0, PredName, false, LastExpr, [GuardCondition|RestConditions], RestLines, TailCode, VarMapOut) :-
    native_typr_guard_goal(Goal, VarMap0, GuardCondition),
    native_typr_prefix_goals(Rest, VarMap0, PredName, false, LastExpr, RestConditions, RestLines, TailCode, VarMapOut).
native_typr_prefix_goals([Goal|Rest], VarMap0, PredName, true, LastExpr, [], [], TailCode, VarMapOut) :-
    native_typr_guarded_tail_sequence([Goal|Rest], VarMap0, PredName, LastExpr, TailCode, VarMapOut).

native_typr_guarded_tail_sequence(Goals, VarMap0, PredName, LastExpr, Code, VarMapOut) :-
    native_typr_guarded_tail_sequence(Goals, VarMap0, PredName, LastExpr, [], Lines, FinalExpr, VarMapOut),
    append(Lines, [FinalExpr], BodyLines),
    atomic_list_concat(BodyLines, '\n', Code).

native_typr_guarded_tail_sequence([], VarMap, PredName, LastExpr, PendingGuards, [], FinalExpr, VarMap) :-
    guard_tail_final_expression(PendingGuards, PredName, LastExpr, FinalExpr).
native_typr_guarded_tail_sequence([Goal|Rest], VarMap0, PredName, _LastExpr0, PendingGuards, [Line|RestLines], FinalExpr, VarMapOut) :-
    native_typr_output_expr(Goal, VarMap0, PredName, VarMap1, OutVar, OutputExpr, IntroKind),
    guard_condition_expression(PendingGuards, GuardExpr),
    conditional_output_line(GuardExpr, PredName, IntroKind, OutVar, OutputExpr, Line),
    native_typr_guarded_tail_sequence(Rest, VarMap1, PredName, OutVar, [], RestLines, FinalExpr, VarMapOut).
native_typr_guarded_tail_sequence([Goal|Rest], VarMap0, PredName, LastExpr, PendingGuards0, Lines, FinalExpr, VarMapOut) :-
    native_typr_guard_goal(Goal, VarMap0, GuardCondition),
    append(PendingGuards0, [GuardCondition], PendingGuards),
    native_typr_guarded_tail_sequence(Rest, VarMap0, PredName, LastExpr, PendingGuards, Lines, FinalExpr, VarMapOut).

native_typr_output_goal(_Module:Goal, VarMap0, PredName, VarMap, Line, FinalExpr) :-
    !,
    native_typr_output_goal(Goal, VarMap0, PredName, VarMap, Line, FinalExpr).
native_typr_output_goal(Goal, VarMap0, PredName, VarMap, Line, FinalExpr) :-
    native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind),
    typr_assignment_line(IntroKind, FinalExpr, OutputExpr, Line).

native_typr_output_expr(_Module:Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind).
native_typr_output_expr(Goal, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    typr_disjunction_alternatives(Goal, Alternatives),
    native_typr_disjunction_output_expr(Alternatives, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind),
    !.
native_typr_output_expr(filter(DF, Expr, Out), VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    ensure_typr_var(VarMap0, Out, FinalExpr, VarMap, IntroKind),
    typr_resolve_value(VarMap0, DF, RDF),
    typr_translate_r_expr(Expr, VarMap0, RExpr),
    format(string(OutputExpr), '@{ subset(~w, ~w) }@', [RDF, RExpr]).
native_typr_output_expr(sort_by(DF, Col, Out), VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    ensure_typr_var(VarMap0, Out, FinalExpr, VarMap, IntroKind),
    typr_resolve_value(VarMap0, DF, RDF),
    typr_resolve_value(VarMap0, Col, RCol),
    format(string(OutputExpr), '@{ ~w[order(~w[[~w]]), ] }@', [RDF, RDF, RCol]).
native_typr_output_expr(group_by(DF, Col, Out), VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    !,
    ensure_typr_var(VarMap0, Out, FinalExpr, VarMap, IntroKind),
    typr_resolve_value(VarMap0, DF, RDF),
    typr_resolve_value(VarMap0, Col, RCol),
    format(string(OutputExpr), '@{ aggregate(. ~~ ~w, data=~w, FUN=list) }@', [RCol, RDF]).
native_typr_output_expr(Goal, VarMap0, _PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    functor(Goal, Pred, Arity),
    binding(r, Pred/Arity, TargetName, Inputs, Outputs, _Options),
    Outputs = [_],
    simple_r_binding_target(TargetName),
    Goal =.. [_|Args],
    length(Inputs, InCount),
    length(InArgs, InCount),
    append(InArgs, [OutArg], Args),
    maplist(typr_resolve_value(VarMap0), InArgs, ResolvedInArgs),
    atomic_list_concat(ResolvedInArgs, ', ', RArgsStr),
    ensure_typr_var(VarMap0, OutArg, FinalExpr, VarMap, IntroKind),
    format(string(OutputExpr), '@{ ~w(~w) }@', [TargetName, RArgsStr]).

typr_disjunction_alternatives((Left ; Right), Alternatives) :-
    !,
    typr_disjunction_alternatives(Left, LeftAlternatives),
    typr_disjunction_alternatives(Right, RightAlternatives),
    append(LeftAlternatives, RightAlternatives, Alternatives).
typr_disjunction_alternatives(Goal, [Goal]).

native_typr_disjunction_output_expr(Alternatives, VarMap0, PredName, VarMap, FinalExpr, OutputExpr, IntroKind) :-
    Alternatives = [_|[_|_]],
    typr_disjunction_shared_output_var(Alternatives, VarMap0, SharedVar),
    ensure_typr_var(VarMap0, SharedVar, FinalExpr, VarMap, IntroKind),
    maplist(native_typr_alternative_branch(VarMap0, PredName, SharedVar), Alternatives, Branches),
    branches_to_typr_output_if_chain(Branches, PredName, OutputExpr).

typr_disjunction_shared_output_var([Alternative|Rest], VarMap, SharedVar) :-
    typr_alternative_output_var(Alternative, SharedVar),
    var(SharedVar),
    typr_disjunction_output_var_allowed(VarMap, SharedVar),
    maplist(typr_alternative_output_var_matches(SharedVar), Rest).

typr_disjunction_output_var_allowed(VarMap, SharedVar) :-
    \+ varmap_contains_var(VarMap, SharedVar),
    !.
typr_disjunction_output_var_allowed(VarMap, SharedVar) :-
    lookup_typr_var(SharedVar, VarMap, ArgName),
    sub_string(ArgName, 0, 3, _, "arg").

typr_alternative_output_var_matches(ExpectedVar, Alternative) :-
    typr_alternative_output_var(Alternative, ActualVar),
    ActualVar == ExpectedVar.

typr_alternative_output_var(Alternative, OutputVar) :-
    normalize_typr_goals(Alternative, Goals),
    reverse(Goals, ReversedGoals),
    member(Goal, ReversedGoals),
    typr_goal_output_var(Goal, OutputVar),
    !.

typr_goal_output_var(_Module:Goal, OutputVar) :-
    !,
    typr_goal_output_var(Goal, OutputVar).
typr_goal_output_var(filter(_, _, OutputVar), OutputVar).
typr_goal_output_var(sort_by(_, _, OutputVar), OutputVar).
typr_goal_output_var(group_by(_, _, OutputVar), OutputVar).
typr_goal_output_var(Goal, OutputVar) :-
    functor(Goal, Pred, Arity),
    binding(r, Pred/Arity, TargetName, Inputs, Outputs, _Options),
    Outputs = [_],
    simple_r_binding_target(TargetName),
    Goal =.. [_|Args],
    length(Inputs, InCount),
    length(Outputs, OutCount),
    length(InArgs, InCount),
    length(OutArgs, OutCount),
    append(InArgs, OutArgs, Args),
    OutArgs = [OutputVar].

native_typr_alternative_branch(VarMap0, PredName, SharedVar, Alternative, branch(Condition, BranchCode)) :-
    typr_alternative_output_var_matches(SharedVar, Alternative),
    native_typr_goal_sequence(Alternative, VarMap0, PredName, Conditions, BranchCode),
    (   Conditions = []
    ->  Condition = 'TRUE'
    ;   atomic_list_concat(Conditions, ' && ', Condition)
    ).

varmap_contains_var([StoredVar-_|_], Var) :-
    Var == StoredVar,
    !.
varmap_contains_var([_|Rest], Var) :-
    varmap_contains_var(Rest, Var).

native_typr_guard_goal(_Module:Goal, VarMap, GuardCondition) :-
    !,
    native_typr_guard_goal(Goal, VarMap, GuardCondition).
native_typr_guard_goal(Goal, VarMap, GuardCondition) :-
    typr_native_guard_expr(Goal, VarMap, GuardCondition),
    !.
native_typr_guard_goal(Goal, VarMap, GuardCondition) :-
    functor(Goal, Pred, Arity),
    binding(r, Pred/Arity, TargetName, Inputs, [], Options),
    member(pattern(command), Options),
    simple_r_binding_target(TargetName),
    Goal =.. [_|Args],
    length(Inputs, InCount),
    length(InArgs, InCount),
    append(InArgs, [], Args),
    maplist(typr_resolve_value(VarMap), InArgs, ResolvedInArgs),
    typr_guard_expression(TargetName, ResolvedInArgs, GuardCondition).

typr_native_guard_expr(Expr, VarMap, GuardCondition) :-
    compound(Expr),
    Expr =.. [Op, _Left, _Right],
    r_expr_op_map(Op, _),
    typr_translate_r_expr(Expr, VarMap, ResolvedExpr0),
    typr_top_level_guard_expr(ResolvedExpr0, ResolvedExpr),
    format(string(GuardCondition), '@{ ~w }@', [ResolvedExpr]).

simple_r_binding_target(TargetName) :-
    atom(TargetName),
    \+ sub_atom(TargetName, _, _, _, '~').
simple_r_binding_target(TargetName) :-
    string(TargetName),
    \+ sub_string(TargetName, _, _, _, "~").

typr_resolve_value(VarMap, Var, Value) :-
    var(Var),
    lookup_typr_var(Var, VarMap, Value),
    !.
typr_resolve_value(_VarMap, Value, Literal) :-
    r_literal(Value, Literal).

typr_guard_expression(TargetName, [], GuardCondition) :-
    format(string(GuardCondition), '@{ ~w }@', [TargetName]).
typr_guard_expression(TargetName, ResolvedInArgs, GuardCondition) :-
    atomic_list_concat(ResolvedInArgs, ', ', RArgsStr),
    format(string(GuardCondition), '@{ ~w(~w) }@', [TargetName, RArgsStr]).

guard_condition_expression([], none).
guard_condition_expression(Conditions, GuardExpr) :-
    Conditions \= [],
    atomic_list_concat(Conditions, ' && ', GuardExpr).

typr_binding_prefix(existing, '').
typr_binding_prefix(new, 'let ').

typr_assignment_line(IntroKind, OutVar, OutputExpr, Line) :-
    typr_binding_prefix(IntroKind, Prefix),
    format(string(Line), '~w~w <- ~w;', [Prefix, OutVar, OutputExpr]).

conditional_output_line(none, _PredName, IntroKind, OutVar, OutputExpr, Line) :-
    typr_assignment_line(IntroKind, OutVar, OutputExpr, Line).
conditional_output_line(GuardExpr, PredName, IntroKind, OutVar, OutputExpr, Line) :-
    typr_binding_prefix(IntroKind, Prefix),
    format(string(Line),
'~w~w <- if (~w) {
	~w
} else {
	stop("No matching clause for ~w")
};', [Prefix, OutVar, GuardExpr, OutputExpr, PredName]).

guard_tail_final_expression([], _PredName, none, 'true').
guard_tail_final_expression([], _PredName, LastExpr, LastExpr) :-
    LastExpr \== none.
guard_tail_final_expression(PendingGuards, PredName, LastExpr, FinalExpr) :-
    PendingGuards \= [],
    guard_condition_expression(PendingGuards, GuardExpr),
    (   LastExpr == none
    ->  BaseExpr = 'true'
    ;   BaseExpr = LastExpr
    ),
    format(string(FinalExpr),
'if (~w) {
	~w
} else {
	stop("No matching clause for ~w")
}', [GuardExpr, BaseExpr, PredName]).

typr_translate_r_expr(Var, VarMap, Resolved) :-
    var(Var),
    lookup_typr_var(Var, VarMap, Resolved),
    !.
typr_translate_r_expr(Atom, _VarMap, Resolved) :-
    atom(Atom),
    !,
    format(string(Resolved), '~w', [Atom]).
typr_translate_r_expr(Text, _VarMap, Resolved) :-
    string(Text),
    !,
    format(string(Resolved), '"~s"', [Text]).
typr_translate_r_expr(Number, _VarMap, Resolved) :-
    number(Number),
    !,
    format(string(Resolved), '~w', [Number]).
typr_translate_r_expr(Expr, VarMap, Resolved) :-
    compound(Expr),
    Expr =.. [Op, Left, Right],
    typr_translate_r_expr(Left, VarMap, LeftResolved),
    typr_translate_r_expr(Right, VarMap, RightResolved),
    r_expr_op_map(Op, ROp),
    format(string(Resolved), '(~w ~w ~w)', [LeftResolved, ROp, RightResolved]).

typr_top_level_guard_expr(ResolvedExpr0, ResolvedExpr) :-
    (   sub_string(ResolvedExpr0, 0, 1, _, "("),
        sub_string(ResolvedExpr0, _, 1, 0, ")")
    ->  sub_string(ResolvedExpr0, 1, _, 1, ResolvedExpr)
    ;   ResolvedExpr = ResolvedExpr0
    ).

r_expr_op_map(>, '>').
r_expr_op_map(<, '<').
r_expr_op_map(>=, '>=').
r_expr_op_map(=<, '<=').
r_expr_op_map(=:=, '==').
r_expr_op_map(==, '==').
r_expr_op_map(\=, '!=').
r_expr_op_map(and, '&').
r_expr_op_map(or, '|').

ensure_typr_var(VarMap, Var, Name, VarMap, existing) :-
    lookup_typr_var(Var, VarMap, Name),
    !.
ensure_typr_var(VarMap, Var, Name, [Var-Name|VarMap], new) :-
    length(VarMap, ExistingCount),
    NextIndex is ExistingCount + 1,
    format(string(Name), 'v~w', [NextIndex]).
ensure_typr_var(VarMap, Var, Name, VarMapOut) :-
    ensure_typr_var(VarMap, Var, Name, VarMapOut, _).

lookup_typr_var(Var, [StoredVar-Name|_], Name) :-
    Var == StoredVar,
    !.
lookup_typr_var(Var, [_|Rest], Name) :-
    lookup_typr_var(Var, Rest, Name).

build_head_varmap([], _, []).
build_head_varmap([Arg|Args], Index, [Arg-ArgName|Rest]) :-
    var(Arg),
    !,
    format(string(ArgName), 'arg~w', [Index]),
    NextIndex is Index + 1,
    build_head_varmap(Args, NextIndex, Rest).
build_head_varmap([_|Args], Index, Rest) :-
    NextIndex is Index + 1,
    build_head_varmap(Args, NextIndex, Rest).

typr_head_condition([], _, []).
typr_head_condition([HeadArg|Rest], Index, Conditions) :-
    (   fact_arg_condition(Index, HeadArg, Condition),
        Condition \== true
    ->  Conditions = [Condition|RestConditions]
    ;   Conditions = RestConditions
    ),
    NextIndex is Index + 1,
    typr_head_condition(Rest, NextIndex, RestConditions).

branches_to_typr_if_chain([branch(Condition, Code)], IfChain) :-
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    format(string(IfChain), 'if (~w) {\n~w\n}', [Condition, IndentedCode]).
branches_to_typr_if_chain([branch(Condition, Code)|Rest], IfChain) :-
    branches_to_typr_if_chain(Rest, RestChain),
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    format(string(IfChain), 'if (~w) {\n~w\n} else ~w', [Condition, IndentedCode, RestChain]).

branches_to_typr_output_if_chain([branch(Condition, Code)], PredName, OutputExpr) :-
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    format(string(OutputExpr),
'if (~w) {
~w
} else {
	stop("No matching clause for ~w")
}', [Condition, IndentedCode, PredName]).
branches_to_typr_output_if_chain([branch(Condition, Code)|Rest], PredName, OutputExpr) :-
    Rest \= [],
    branch_safe_typr_code(Code, BranchCode),
    indent_text(BranchCode, "\t", IndentedCode),
    branches_to_typr_output_if_chain(Rest, PredName, RestChain),
    format(string(OutputExpr), 'if (~w) {\n~w\n} else ~w', [Condition, IndentedCode, RestChain]).

pred_spec_name(_Module:Pred/_, PredName) :-
    !,
    atom_string(Pred, PredName).
pred_spec_name(Pred/_, PredName) :-
    atom_string(Pred, PredName).

indent_text(Text, Prefix, Indented) :-
    split_string(Text, "\n", "", Lines),
    findall(IndentedLine, (
        member(Line, Lines),
        string_concat(Prefix, Line, IndentedLine)
    ), IndentedLines),
    atomic_list_concat(IndentedLines, '\n', Indented).

branch_safe_typr_code(Code, SafeCode) :-
    split_string(Code, "\n", "", Lines),
    maplist(branch_safe_typr_line, Lines, SafeLines),
    atomic_list_concat(SafeLines, '\n', SafeCode).

branch_safe_typr_line(Line, SafeLine) :-
    leading_layout(Line, Layout, Content),
    (   sub_string(Content, 0, 4, _, "let ")
    ->  SafeLine = Line
    ;   sub_string(Content, _, _, _, " <- "),
        sub_string(Content, _, 1, 0, ";")
    ->  format(string(SafeLine), '~wlet ~w', [Layout, Content])
    ;   SafeLine = Line
    ).

leading_layout(Line, Layout, Content) :-
    string_codes(Line, Codes),
    take_layout_codes(Codes, LayoutCodes, ContentCodes),
    string_codes(Layout, LayoutCodes),
    string_codes(Content, ContentCodes).

take_layout_codes([Code|Rest], [Code|LayoutCodes], ContentCodes) :-
    code_type(Code, space),
    !,
    take_layout_codes(Rest, LayoutCodes, ContentCodes).
take_layout_codes(ContentCodes, [], ContentCodes).

finalize_type_diagnostics_report(Options) :-
    (   memberchk(type_diagnostics_report(Report), Options),
        var(Report)
    ->  Report = []
    ;   true
    ).

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
