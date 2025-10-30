
:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% csharp_query_target.pl - Query IR synthesis for the C# runtime.
% Supports fact-only predicates and initial non-recursive rule bodies composed
% of relational joins and unions. The resulting plan dicts are consumed by the
% managed QueryRuntime (C#) infrastructure.

:- module(csharp_query_target, [
    build_query_plan/3,     % +PredIndicator, +Options, -PlanDict
    render_plan_to_csharp/2,% +PlanDict, -CSharpSource
    plan_module_name/2      % +PlanDict, -ModuleName
]).

:- use_module(library(apply)).
:- use_module(library(error)).
:- use_module(library(lists)).

%% build_query_plan(+PredIndicator, +Options, -PlanDict) is semidet.
%  Produce a declarative plan describing how to evaluate the requested
%  predicate. Plans are represented as dicts containing the head descriptor,
%  root operator, materialised fact tables, and metadata.
build_query_plan(Pred/Arity, Options, Plan) :-
    must_be(atom, Pred),
    must_be(integer, Arity),
    Arity >= 0,

    functor(HeadPattern, Pred, Arity),
    findall(HeadPattern-Body, clause(user:HeadPattern, Body), Clauses),
    partition_recursive_clauses(Pred, Arity, Clauses, BaseClauses, RecClauses),
    (   RecClauses == []
    ->  classify_clauses(BaseClauses, Classification),
        build_plan_by_class(Classification, Pred, Arity, BaseClauses, Options, Plan)
    ;   build_recursive_plan(Pred, Arity, BaseClauses, RecClauses, Options, Plan)
    ).

partition_recursive_clauses(Pred, Arity, Clauses, BaseClauses, RecClauses) :-
    partition(clause_is_recursive(Pred, Arity), Clauses, RecClauses, BaseClauses).

clause_is_recursive(Pred, Arity, _Head-Body) :-
    body_to_list(Body, Terms),
    member(Term, Terms),
    functor(Term, Pred, TermArity),
    TermArity =:= Arity.

%% classify_clauses(+Clauses, -Classification) is det.
classify_clauses([], none).
classify_clauses(Clauses, facts) :-
    Clauses \= [],
    forall(member(_-Body, Clauses), Body == true), !.
classify_clauses([_-Body], single_rule) :-
    Body \= true, !.
classify_clauses(Clauses, multiple_rules) :-
    Clauses \= [],
    Clauses \= [_-true], !.
classify_clauses(_Clauses, unsupported).

%% build_plan_by_class(+Class, +Pred, +Arity, +Clauses, +Options, -Plan) is semidet.
build_plan_by_class(none, Pred, Arity, _Clauses, _Options, _) :-
    format(user_error, 'C# query target: no clauses found for ~w/~w.~n', [Pred, Arity]),
    fail.
build_plan_by_class(unsupported, Pred, Arity, _Clauses, _Options, _) :-
    format(user_error,
           'C# query target: predicate shape not yet supported (~w/~w).~n',
           [Pred, Arity]),
    fail.
build_plan_by_class(facts, Pred, Arity, _Clauses, Options, Plan) :-
    ensure_relation(Pred, Arity, [], Relations),
    Relations = [relation{predicate:HeadSpec, facts:_}|_],
    Root = relation_scan{type:relation_scan, predicate:HeadSpec, width:Arity},
    Plan = plan{
        head:HeadSpec,
        root:Root,
        relations:Relations,
        metadata:_{classification:facts, options:Options},
        is_recursive:false
    }.
build_plan_by_class(single_rule, Pred, Arity, [Head-Body], Options, Plan) :-
    Head =.. [Pred|HeadArgs],
    length(HeadArgs, Arity),
    HeadSpec = predicate{name:Pred, arity:Arity},
    build_rule_clause(HeadSpec, HeadArgs, Body, Node, Relations),
    dedup_relations(Relations, UniqueRelations),
    Plan = plan{
        head:HeadSpec,
        root:Node,
        relations:UniqueRelations,
        metadata:_{classification:single_rule, options:Options},
        is_recursive:false
    }.
build_plan_by_class(multiple_rules, Pred, Arity, Clauses, Options, Plan) :-
    maplist(build_clause_node(Pred/Arity), Clauses, Nodes, RelationLists),
    append(RelationLists, RelationsFlat),
    dedup_relations(RelationsFlat, Relations),
    HeadSpec = predicate{name:Pred, arity:Arity},
    Root = union{type:union, sources:Nodes, width:Arity},
    Plan = plan{
        head:HeadSpec,
        root:Root,
        relations:Relations,
        metadata:_{classification:multiple_rules, options:Options},
        is_recursive:false
    }.
build_plan_by_class(single_rule, Pred, Arity, [_-true], _Options, _) :-
    format(user_error, 'C# query target: unexpected fact classified as rule (~w/~w).~n', [Pred, Arity]),
    fail.

%% Recursive plan construction ----------------------------------------------

build_recursive_plan(Pred, Arity, BaseClauses, RecClauses, Options, Plan) :-
    HeadSpec = predicate{name:Pred, arity:Arity},
    build_base_root(Pred, Arity, BaseClauses, Options, BaseRoot, BaseRelations),
    build_recursive_variants(HeadSpec, RecClauses, RecursiveNodes, RecursiveRelations),
    append(BaseRelations, RecursiveRelations, CombinedRelations0),
    dedup_relations(CombinedRelations0, CombinedRelations),
    Root = fixpoint{
        type:fixpoint,
        head:HeadSpec,
        base:BaseRoot,
        recursive:RecursiveNodes,
        width:Arity
    },
    Plan = plan{
        head:HeadSpec,
        root:Root,
        relations:CombinedRelations,
        metadata:_{classification:recursive, options:Options},
        is_recursive:true
    }.

build_base_root(_Pred, Arity, [], _Options, empty{type:empty, width:Arity}, []).
build_base_root(Pred, Arity, Clauses, Options, BaseRoot, Relations) :-
    classify_clauses(Clauses, Class),
    build_plan_by_class(Class, Pred, Arity, Clauses, Options, BasePlan),
    get_dict(root, BasePlan, BaseRoot),
    get_dict(relations, BasePlan, Relations).

build_recursive_variants(HeadSpec, Clauses, Variants, Relations) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    findall(variant(Node, RelList),
        (   member(Head-Body, Clauses),
            build_recursive_clause_variants(HeadSpec, Head-Body, VariantStructs),
            member(variant(Node, RelList), VariantStructs)
        ),
        VariantPairs),
    (   VariantPairs == []
    ->  format(user_error, 'C# query target: no recursive variants generated for ~w/~w.~n', [Pred, Arity]),
        fail
    ;   findall(Node, member(variant(Node, _), VariantPairs), Variants),
        findall(RelList, member(variant(_, RelList), VariantPairs), RelationLists),
        append(RelationLists, RelationsFlat),
        dedup_relations(RelationsFlat, Relations)
    ).

build_recursive_clause_variants(HeadSpec, Head-Body, Variants) :-
    Head =.. [_|HeadArgs],
    body_to_list(Body, Terms),
    recursive_occurrence_indices(HeadSpec, Terms, Occurrences),
    Occurrences \= [],
    findall(variant(Node, RelList),
        (   member(DeltaIndex, Occurrences),
            assign_roles_for_variant(HeadSpec, Terms, DeltaIndex, Roles),
            build_pipeline(HeadSpec, Terms, Roles, PipelineNode, RelList, VarMap, _Width),
            project_to_head(HeadArgs, PipelineNode, VarMap, Node)
        ),
        Variants).

recursive_occurrence_indices(HeadSpec, Terms, Indices) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    findall(Index,
        (   nth0(Index, Terms, Term),
            functor(Term, Pred, TermArity),
            TermArity =:= Arity
        ),
        Indices).

assign_roles_for_variant(HeadSpec, Terms, DeltaIndex, Roles) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    same_length(Terms, Roles),
    assign_roles_for_variant_(Terms, Pred, Arity, DeltaIndex, 0, Roles).

assign_roles_for_variant_([], _Pred, _Arity, _DeltaIndex, _Pos, []).
assign_roles_for_variant_([Term|Rest], Pred, Arity, DeltaIndex, Pos, [Role|Roles]) :-
    (   constraint_goal(Term)
    ->  Role = constraint
    ;   functor(Term, Pred, TermArity),
        TermArity =:= Arity
    ->  (   Pos =:= DeltaIndex -> Role = recursive(delta)
        ;   Role = recursive(total)
        )
    ;   Role = relation
    ),
    Pos1 is Pos + 1,
    assign_roles_for_variant_(Rest, Pred, Arity, DeltaIndex, Pos1, Roles).

roles_for_nonrecursive_terms(HeadSpec, Terms, Roles) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    same_length(Terms, Roles),
    roles_for_nonrecursive_terms_(Terms, Pred, Arity, Roles).

roles_for_nonrecursive_terms_([], _Pred, _Arity, []).
roles_for_nonrecursive_terms_([Term|Rest], Pred, Arity, [Role|Roles]) :-
    (   constraint_goal(Term)
    ->  Role = constraint
    ;   functor(Term, Pred, TermArity),
        TermArity =:= Arity
    ->  format(user_error, 'C# query target: recursive literal encountered in non-recursive clause (~w/~w).~n', [Pred, Arity]),
        fail
    ;   Role = relation
    ),
    roles_for_nonrecursive_terms_(Rest, Pred, Arity, Roles).

constraint_goal(Goal) :-
    functor(Goal, Functor, Arity),
    Arity =:= 2,
    (   Functor == '='
    ;   Functor == '=='
    ;   Functor == dif
    ;   atom_codes(Functor, [92, 61])
    ).

%% Clause helpers -----------------------------------------------------------

build_clause_node(_HeadSig, Head-true, Node, Relations) :-
    Head =.. [Pred|Args],
    length(Args, Arity),
    ensure_relation(Pred, Arity, [], Relations),
    HeadSpec = predicate{name:Pred, arity:Arity},
    Node = relation_scan{type:relation_scan, predicate:HeadSpec, width:Arity}.
build_clause_node(HeadSig, Head-Body, Node, Relations) :-
    HeadSig = Pred/Arity,
    Head =.. [Pred|HeadArgs],
    HeadSpec = predicate{name:Pred, arity:Arity},
    build_rule_clause(HeadSpec, HeadArgs, Body, Node, ClauseRelations),
    Relations = ClauseRelations.

build_rule_clause(HeadSpec, HeadArgs, Body, Root, Relations) :-
    Body \= true,
    body_to_list(Body, Terms),
    Terms \= [],
    roles_for_nonrecursive_terms(HeadSpec, Terms, Roles),
    build_pipeline(HeadSpec, Terms, Roles, PipelineNode, Relations, VarMap, Width),
    project_to_head(HeadArgs, PipelineNode, VarMap, Root),
    length(HeadArgs, WidthHead),
    Width >= WidthHead.

body_to_list(true, []) :- !.
body_to_list((A, B), Terms) :- !,
    body_to_list(A, TA),
    body_to_list(B, TB),
    append(TA, TB, Terms).
body_to_list(Goal, [Goal]).

build_pipeline(HeadSpec, [Term|Rest], [Role|Roles], FinalNode, FinalRelations, FinalVarMap, FinalWidth) :-
    build_initial_node(HeadSpec, Term, Role, Node0, Relations0, VarMap0, Width0),
    fold_terms(Rest, Roles, HeadSpec, Node0, VarMap0, Width0, Relations0,
               FinalNode, FinalVarMap, FinalWidth, FinalRelations).

build_initial_node(HeadSpec, _Term, constraint, _Node, _Relations, _VarMap, _Width) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    format(user_error, 'C# query target: clause for ~w/~w begins with a constraint; reorder body literals.~n', [Pred, Arity]),
    fail.
build_initial_node(_HeadSpec, Term, relation, Node, Relations, VarMap, Width) :-
    Term =.. [Pred|Args],
    length(Args, Arity),
    ensure_relation(Pred, Arity, [], Relations),
    relation_scan_node(Pred, Arity, Node),
    init_var_map(Args, 0, VarMap),
    Width = Arity.
build_initial_node(HeadSpec, Term, recursive(RoleKind), Node, Relations, VarMap, Width) :-
    get_dict(arity, HeadSpec, Arity),
    Term =.. [_|Args],
    length(Args, TermArity),
    (   TermArity =:= Arity
    ->  true
    ;   get_dict(name, HeadSpec, Pred),
        format(user_error, 'C# query target: recursive call arity mismatch in ~w/~w.~n', [Pred, Arity]),
        fail
    ),
    recursive_ref_node(HeadSpec, RoleKind, Node),
    init_var_map(Args, 0, VarMap),
    Width = Arity,
    Relations = [].

fold_terms([], [], _HeadSpec, Node, VarMap, Width, Relations, Node, VarMap, Width, Relations).
fold_terms([Term|Rest], [Role|Roles], HeadSpec, AccNode, VarMapIn, WidthIn, RelationsIn,
           NodeOut, VarMapOut, WidthOut, RelationsOut) :-
    (   Role = constraint
    ->  build_constraint_node(Term, AccNode, VarMapIn, WidthIn, ConstraintNode),
        fold_terms(Rest, Roles, HeadSpec, ConstraintNode, VarMapIn, WidthIn, RelationsIn,
                   NodeOut, VarMapOut, WidthOut, RelationsOut)
    ;   Role = relation
    ->  Term =.. [Pred|Args],
        length(Args, Arity),
        ensure_relation(Pred, Arity, RelationsIn, RelationsNext),
        relation_scan_node(Pred, Arity, RightNode),
        build_join_node(AccNode, RightNode, Args, VarMapIn, WidthIn, Arity, VarMapMid, WidthMid, JoinNode),
        fold_terms(Rest, Roles, HeadSpec, JoinNode, VarMapMid, WidthMid, RelationsNext,
                   NodeOut, VarMapOut, WidthOut, RelationsOut)
    ;   Role = recursive(RoleKind)
    ->  get_dict(arity, HeadSpec, Arity),
        Term =.. [_|Args],
        recursive_ref_node(HeadSpec, RoleKind, RightNode),
        build_join_node(AccNode, RightNode, Args, VarMapIn, WidthIn, Arity, VarMapMid, WidthMid, JoinNode),
        fold_terms(Rest, Roles, HeadSpec, JoinNode, VarMapMid, WidthMid, RelationsIn,
                   NodeOut, VarMapOut, WidthOut, RelationsOut)
    ).

relation_scan_node(Pred, Arity, relation_scan{type:relation_scan, predicate:predicate{name:Pred, arity:Arity}, width:Arity}).

recursive_ref_node(HeadSpec, RoleKind, recursive_ref{type:recursive_ref, predicate:HeadSpec, role:RoleKind, width:Arity}) :-
    get_dict(arity, HeadSpec, Arity).

build_join_node(LeftNode, RightNode, Args, VarMapIn, WidthIn, Arity, VarMapOut, WidthOut, JoinNode) :-
    shared_variable_keys(Args, VarMapIn, 0, LeftKeys, RightKeys),
    update_var_map(Args, VarMapIn, WidthIn, VarMapOut),
    WidthOut is WidthIn + Arity,
    JoinNode = join{
        type:join,
        left:LeftNode,
        right:RightNode,
        left_keys:LeftKeys,
        right_keys:RightKeys,
        left_width:WidthIn,
        right_width:Arity,
        width:WidthOut
    }.

build_constraint_node(Term, InputNode, VarMap, Width, selection{type:selection, input:InputNode, predicate:Condition, width:Width}) :-
    constraint_condition(Term, VarMap, Condition).

constraint_condition(Goal, VarMap, Condition) :-
    functor(Goal, Functor, 2),
    arg(1, Goal, Left),
    arg(2, Goal, Right),
    (   Functor == '='
    ->  build_condition(eq, Left, Right, VarMap, Condition)
    ;   Functor == '=='
    ->  build_condition(eq, Left, Right, VarMap, Condition)
    ;   Functor == dif
    ->  build_condition(neq, Left, Right, VarMap, Condition)
    ;   atom_codes(Functor, [92, 61])
    ->  build_condition(neq, Left, Right, VarMap, Condition)
    ;   format(user_error, 'C# query target: unsupported constraint goal ~q.~n', [Goal]),
        fail
    ).

build_condition(Type, Left, Right, VarMap, condition{type:Type, left:LeftOperand, right:RightOperand}) :-
    constraint_operand(VarMap, Left, LeftOperand),
    constraint_operand(VarMap, Right, RightOperand).

constraint_operand(VarMap, Term, operand{kind:column, index:Index}) :-
    var(Term),
    (   lookup_var_index(VarMap, Term, Index)
    ->  true
    ;   format(user_error, 'C# query target: variable ~w not bound before constraint evaluation.~n', [Term]),
        fail
    ).
constraint_operand(_VarMap, Term, operand{kind:value, value:Term}) :-
    atomic(Term), !.
constraint_operand(_VarMap, Term, operand{kind:value, value:Term}) :-
    string(Term), !.
constraint_operand(_VarMap, Term, _Operand) :-
    format(user_error, 'C# query target: unsupported constraint operand ~q.~n', [Term]),
    fail.

init_var_map(Args, Offset, VarMapOut) :-
    init_var_map_(Args, Offset, [], VarMapOut).

init_var_map_([], _, VarMap, VarMap).
init_var_map_([Arg|Rest], Offset, Acc, VarMapOut) :-
    (   var(Arg)
    ->  Acc1 = [Arg-Offset|Acc],
        Offset1 is Offset + 1
    ;   domain_error(variable, Arg)
    ),
    init_var_map_(Rest, Offset1, Acc1, VarMapOut).

shared_variable_keys(Args, VarMap, Pos, LeftKeys, RightKeys) :-
    shared_variable_keys_(Args, VarMap, Pos, [], LeftRev, [], RightRev),
    reverse(LeftRev, LeftKeys),
    reverse(RightRev, RightKeys).

shared_variable_keys_([], _VarMap, _Pos, Left, Left, Right, Right).
shared_variable_keys_([Arg|Rest], VarMap, Pos, LeftAcc, LeftOut, RightAcc, RightOut) :-
    (   var(Arg), lookup_var_index(VarMap, Arg, LeftIdx)
    ->  LeftAcc1 = [LeftIdx|LeftAcc],
        RightAcc1 = [Pos|RightAcc]
    ;   LeftAcc1 = LeftAcc,
        RightAcc1 = RightAcc
    ),
    Pos1 is Pos + 1,
    shared_variable_keys_(Rest, VarMap, Pos1, LeftAcc1, LeftOut, RightAcc1, RightOut).

update_var_map([], VarMap, _Offset, VarMap).
update_var_map([Arg|Rest], VarMapIn, Offset, VarMapOut) :-
    (   var(Arg)
    ->  (   lookup_var_index(VarMapIn, Arg, _)
        ->  VarMapMid = VarMapIn
        ;   VarMapMid = [Arg-Offset|VarMapIn]
        )
    ;   domain_error(variable, Arg)
    ),
    Offset1 is Offset + 1,
    update_var_map(Rest, VarMapMid, Offset1, VarMapOut).

lookup_var_index([Var0-Index|_], Var, Index) :-
    Var == Var0, !.
lookup_var_index([_|Rest], Var, Index) :-
    lookup_var_index(Rest, Var, Index).

projection_indices(Args, VarMap, Indices) :-
    maplist(variable_index(VarMap), Args, Indices).

variable_index(VarMap, Var, Index) :-
    (   lookup_var_index(VarMap, Var, Index)
    ->  true
    ;   format(user_error, 'C# query target: variable ~w not bound in body.~n', [Var]),
        fail
    ).

project_to_head(HeadArgs, PipelineNode, VarMap, projection{type:projection, input:PipelineNode, columns:Indices, width:Width}) :-
    projection_indices(HeadArgs, VarMap, Indices),
    length(HeadArgs, Width).

dedup_relations(Relations, Unique) :-
    dedup_relations(Relations, [], [], Rev),
    reverse(Rev, Unique).

dedup_relations([], _Seen, Acc, Acc).
dedup_relations([Relation|Rest], Seen, Acc, Unique) :-
    Relation = relation{predicate:predicate{name:Pred, arity:Arity}, facts:_},
    Key = Pred/Arity,
    (   memberchk(Key, Seen)
    ->  dedup_relations(Rest, Seen, Acc, Unique)
    ;   dedup_relations(Rest, [Key|Seen], [Relation|Acc], Unique)
    ).

ensure_relation(Pred, Arity, RelationsIn, RelationsOut) :-
    (   relation_present(RelationsIn, Pred, Arity)
    ->  RelationsOut = RelationsIn
    ;   gather_fact_rows(Pred, Arity, Rows),
        (   Rows == []
        ->  format(user_error, 'C# query target: no facts available for ~w/~w.~n', [Pred, Arity]),
            fail
        ;   Relation = relation{predicate:predicate{name:Pred, arity:Arity}, facts:Rows},
            RelationsOut = [Relation|RelationsIn]
        )
    ).

relation_present([relation{predicate:predicate{name:Pred, arity:Arity}, facts:_}|_], Pred, Arity) :- !.
relation_present([_|Rest], Pred, Arity) :-
    relation_present(Rest, Pred, Arity).

gather_fact_rows(Pred, Arity, Rows) :-
    findall(Row,
        (   functor(Head, Pred, Arity),
            clause(user:Head, true),
            Head =.. [_|Args],
            maplist(copy_term_value, Args, Row)
        ),
        Rows).

copy_term_value(Term, Copy) :-
    (   atomic(Term) -> Copy = Term
    ;   Term =.. [Functor|Args],
        maplist(copy_term_value, Args, CopyArgs),
        Copy =.. [Functor|CopyArgs]
    ).

%% Rendering ----------------------------------------------------------------

render_plan_to_csharp(Plan, Code) :-
    get_dict(head, Plan, predicate{name:Pred, arity:Arity}),
    get_dict(root, Plan, Root),
    get_dict(relations, Plan, Relations),
    get_dict(is_recursive, Plan, IsRecursive),
    relation_blocks(Relations, ProviderBody),
    (   ProviderBody == ''
    ->  ProviderSection = ''
    ;   format(atom(ProviderSection), '~w~n', [ProviderBody])
    ),
    emit_plan_expression(Root, PlanExpr),
    plan_module_name(Plan, ModuleClass),
    atom_string(Pred, PredStr),
    (   IsRecursive == true
    ->  RecLiteral = 'true'
    ;   RecLiteral = 'false'
    ),
    format(atom(Code),
'// Auto-generated by UnifyWeaver
using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;

namespace UnifyWeaver.Generated
{
    public static class ~w
    {
        public static (InMemoryRelationProvider Provider, QueryPlan Plan) Build()
        {
            var provider = new InMemoryRelationProvider();
~w            var plan = new QueryPlan(
                new PredicateId("~w", ~w),
                ~w,
                ~w
            );
            return (provider, plan);
        }
    }
}
', [ModuleClass, ProviderSection, PredStr, Arity, PlanExpr, RecLiteral]).

emit_plan_expression(Node, Expr) :-
    is_dict(Node, relation_scan), !,
    get_dict(predicate, Node, predicate{name:Name, arity:Arity}),
    atom_string(Name, NameStr),
    format(atom(Expr), 'new RelationScanNode(new PredicateId("~w", ~w))', [NameStr, Arity]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, projection), !,
    get_dict(input, Node, Input),
    get_dict(columns, Node, Columns),
    emit_plan_expression(Input, InputExpr),
    maplist(column_expression(tuple), Columns, ColumnExprs),
    atomic_list_concat(ColumnExprs, ', ', ColumnList),
    format(atom(Expr), 'new ProjectionNode(~w, tuple => new object[]{ ~w })', [InputExpr, ColumnList]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, selection), !,
    get_dict(input, Node, Input),
    get_dict(predicate, Node, Condition),
    emit_plan_expression(Input, InputExpr),
    selection_condition_expression(Condition, tuple, ConditionExpr),
    format(atom(Expr), 'new SelectionNode(~w, tuple => ~w)', [InputExpr, ConditionExpr]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, join), !,
    get_dict(left, Node, Left),
    get_dict(right, Node, Right),
    get_dict(left_keys, Node, LeftKeys),
    get_dict(right_keys, Node, RightKeys),
    get_dict(left_width, Node, LeftWidth),
    get_dict(right_width, Node, RightWidth),
    emit_plan_expression(Left, LeftExpr),
    emit_plan_expression(Right, RightExpr),
    join_predicate_expression(LeftKeys, RightKeys, PredicateExpr),
    join_projection_expression(LeftWidth, RightWidth, ProjectExpr),
    format(atom(Expr), 'new JoinNode(~w, ~w, (left, right) => ~w, (left, right) => new object[]{ ~w })',
        [LeftExpr, RightExpr, PredicateExpr, ProjectExpr]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, union), !,
    get_dict(sources, Node, Sources),
    maplist(emit_plan_expression, Sources, SourceExprs),
    atomic_list_concat(SourceExprs, ', ', SourceList),
    format(atom(Expr), 'new UnionNode(new PlanNode[]{ ~w })', [SourceList]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, fixpoint), !,
    get_dict(base, Node, BaseNode),
    get_dict(recursive, Node, RecursiveNodes),
    get_dict(head, Node, predicate{name:Name, arity:Arity}),
    emit_plan_expression(BaseNode, BaseExpr),
    maplist(emit_plan_expression, RecursiveNodes, RecursiveExprs),
    atomic_list_concat(RecursiveExprs, ', ', RecursiveList),
    atom_string(Name, NameStr),
    format(atom(Expr), 'new FixpointNode(~w, new PlanNode[]{ ~w }, new PredicateId("~w", ~w))',
        [BaseExpr, RecursiveList, NameStr, Arity]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, recursive_ref), !,
    get_dict(predicate, Node, predicate{name:Name, arity:Arity}),
    get_dict(role, Node, Role),
    recursive_role_atom(Role, RoleAtom),
    atom_string(Name, NameStr),
    format(atom(Expr), 'new RecursiveRefNode(new PredicateId("~w", ~w), RecursiveRefKind.~w)',
        [NameStr, Arity, RoleAtom]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, empty), !,
    get_dict(width, Node, Width),
    format(atom(Expr), 'new EmptyNode(~w)', [Width]).
emit_plan_expression(Node, _Expr) :-
    format(user_error, 'C# query target: cannot render plan node ~q.~n', [Node]),
    fail.

recursive_role_atom(delta, 'Delta').
recursive_role_atom(total, 'Total').

selection_condition_expression(condition{type:eq, left:Left, right:Right}, TupleVar, Expr) :-
    equality_condition_expression(Left, Right, TupleVar, Expr).
selection_condition_expression(condition{type:neq, left:Left, right:Right}, TupleVar, Expr) :-
    equality_condition_expression(Left, Right, TupleVar, EqExpr),
    format(atom(Expr), '!(~w)', [EqExpr]).

equality_condition_expression(Left, Right, TupleVar, Expr) :-
    operand_expression(Left, TupleVar, LeftExpr),
    operand_expression(Right, TupleVar, RightExpr),
    format(atom(Expr), 'Equals(~w, ~w)', [LeftExpr, RightExpr]).

operand_expression(operand{kind:column, index:Index}, TupleVar, Expr) :-
    format(atom(Expr), '~w[~w]', [TupleVar, Index]).
operand_expression(operand{kind:value, value:Value}, _TupleVar, Expr) :-
    csharp_literal(Value, Expr).

column_expression(Prefix, Index, Expr) :-
    format(atom(Expr), '~w[~w]', [Prefix, Index]).

join_predicate_expression([], [], 'true').
join_predicate_expression(LeftKeys, RightKeys, Expr) :-
    findall(Cond,
        (   nth0(I, LeftKeys, LIdx),
            nth0(I, RightKeys, RIdx),
            format(atom(Cond), 'Equals(left[~w], right[~w])', [LIdx, RIdx])
        ),
        Conds),
    (   Conds = []
    ->  Expr = 'true'
    ;   atomic_list_concat(Conds, ' && ', Expr)
    ).

join_projection_expression(LeftWidth, RightWidth, Expr) :-
    findall(Item,
        (   between(0, LeftWidth-1, LIdx),
            format(atom(Item), 'left[~w]', [LIdx])
        ),
        LeftItems),
    findall(Item,
        (   between(0, RightWidth-1, RIdx),
            format(atom(Item), 'right[~w]', [RIdx])
        ),
        RightItems),
    append(LeftItems, RightItems, Items),
    atomic_list_concat(Items, ', ', Expr).

relation_blocks(Relations, ProviderStatements) :-
    findall(Line,
        (   member(relation{predicate:predicate{name:Name, arity:Arity}, facts:Rows}, Relations),
            member(Row, Rows),
            atom_string(Name, NameStr),
            maplist(csharp_literal, Row, ArgLiterals),
            atomic_list_concat(ArgLiterals, ', ', ArgList),
            format(atom(Line), '            provider.AddFact(new PredicateId("~w", ~w), ~w);', [NameStr, Arity, ArgList])
        ),
        Lines),
    (   Lines == []
    ->  ProviderStatements = ''
    ;   atomic_list_concat(Lines, '\n', ProviderStatements)
    ).

csharp_literal(Value, Literal) :-
    (   number(Value)
    ->  format(atom(Literal), '~w', [Value])
    ;   atom(Value)
    ->  atom_string(Value, String),
        escape_csharp_string(String, Escaped),
        format(atom(Literal), '"~w"', [Escaped])
    ;   string(Value)
    ->  escape_csharp_string(Value, Escaped),
        format(atom(Literal), '"~w"', [Escaped])
    ;   term_string(Value, String),
        escape_csharp_string(String, Escaped),
        format(atom(Literal), '"~w"', [Escaped])
    ).

escape_csharp_string(Input, Escaped) :-
    string_codes(Input, Codes),
    maplist(escape_code, Codes, Parts),
    atomic_list_concat(Parts, '', Escaped).

escape_code(92, Atom) :- atom_codes(Atom, [92, 92]).
escape_code(34, Atom) :- atom_codes(Atom, [92, 34]).
escape_code(10, Atom) :- atom_codes(Atom, [92, 110]).
escape_code(13, Atom) :- atom_codes(Atom, [92, 114]).
escape_code(9, Atom)  :- atom_codes(Atom, [92, 116]).
escape_code(Code, Atom) :- atom_codes(Atom, [Code]).

plan_module_name(Plan, ModuleName) :-
    get_dict(head, Plan, predicate{name:Pred, arity:_}),
    predicate_pascal(Pred, Pascal),
    atom_concat(Pascal, 'QueryModule', ModuleName).

predicate_pascal(Atom, Pascal) :-
    atom_string(Atom, Text),
    split_string(Text, '_', '_', Parts),
    maplist(capitalise_string, Parts, Caps),
    atomic_list_concat(Caps, '', PascalString),
    atom_string(Pascal, PascalString).

capitalise_string(Input, Output) :-
    (   Input = ''
    ->  Output = ''
    ;   string_lower(Input, Lower),
        sub_string(Lower, 0, 1, _, First),
        sub_string(Lower, 1, _, 0, Rest),
        string_upper(First, UpperFirst),
        string_concat(UpperFirst, Rest, Output)
    ).
