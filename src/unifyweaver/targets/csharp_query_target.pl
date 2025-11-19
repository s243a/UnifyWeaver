
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
:- use_module(library(ugraphs), [vertices_edges_to_ugraph/3, transpose_ugraph/2, reachable/3]).
:- use_module('../core/dynamic_source_compiler', [is_dynamic_source/1, dynamic_source_metadata/2]).

spec_signature(predicate{name:Name, arity:Arity}, Name/Arity).

term_signature(Term0, Name/Arity) :-
    strip_module(Term0, _, Term),
    nonvar(Term),
    Term =.. [Name|Args],
    length(Args, Arity).

gather_predicate_clauses(predicate{name:Pred, arity:Arity}, Clauses) :-
    findall(Head-Body,
        (   functor(Head, Pred, Arity),
            clause(user:Head, Body)
        ),
        ClausePairs),
    (   ClausePairs == []
    ->  (   dynamic_source_metadata(Pred/Arity, _)
        ->  Clauses = []
        ;   format(user_error, 'C# query target: no clauses defined for ~w/~w.~n', [Pred, Arity]),
            fail
        )
    ;   Clauses = ClausePairs
    ).

predicate_defined(Name/Arity) :-
    current_predicate(user:Name/Arity).

predicate_dependencies(Name/Arity, Dependencies) :-
    Spec = predicate{name:Name, arity:Arity},
    (   gather_predicate_clauses(Spec, Clauses)
    ->  findall(Dep,
            (   member(_-Body, Clauses),
                body_to_list(Body, Terms),
                member(Term, Terms),
                \+ constraint_goal(Term),
                term_signature(Term, Dep),
                (   predicate_defined(Dep)
                ;   dynamic_source_metadata(Dep, _)
                )
            ),
            RawDeps),
        sort(RawDeps, Dependencies)
    ;   Dependencies = []
    ).

compute_dependency_group(Pred/Arity, GroupSpecs) :-
    build_dependency_graph([Pred/Arity], [], [], Vertices, [], Edges),
    sort(Vertices, SortedVertices),
    vertices_edges_to_ugraph(SortedVertices, Edges, Graph),
    reachable(Pred/Arity, Graph, Forward),
    transpose_ugraph(Graph, Transposed),
    reachable(Pred/Arity, Transposed, Backward),
    intersection(Forward, Backward, Component),
    Component \= [],
    maplist(vertex_to_spec, Component, Specs0),
    sort_components(Pred/Arity, Specs0, GroupSpecs),
    !.
compute_dependency_group(Pred/Arity, [predicate{name:Pred, arity:Arity}]).

build_dependency_graph([], _Visited, Vertices, Vertices, Edges, Edges).
build_dependency_graph([Vertex|Queue], Visited, VertAccIn, VerticesOut, EdgeAccIn, EdgesOut) :-
    (   memberchk(Vertex, Visited)
    ->  build_dependency_graph(Queue, Visited, VertAccIn, VerticesOut, EdgeAccIn, EdgesOut)
    ;   predicate_dependencies(Vertex, Deps),
        append(Deps, Queue, QueueNext),
        findall(Vertex-Dep, member(Dep, Deps), NewEdges),
        append(NewEdges, EdgeAccIn, EdgeAccMid),
        append([Vertex|Deps], VertAccIn, VertAccMid),
        build_dependency_graph(QueueNext, [Vertex|Visited], VertAccMid, VerticesOut, EdgeAccMid, EdgesOut)
    ).

vertex_to_spec(Name/Arity, predicate{name:Name, arity:Arity}).

sort_components(Pred/Arity, Specs, [HeadSpec|RestSpecs]) :-
    HeadSpec = predicate{name:Pred, arity:Arity},
    include(\=(HeadSpec), Specs, Others),
    maplist(spec_signature, Others, Signatures),
    sort(Signatures, SortedSigs),
    maplist(signature_to_spec, SortedSigs, RestSpecs).

signature_to_spec(Name/Arity, predicate{name:Name, arity:Arity}).

%% build_query_plan(+PredIndicator, +Options, -PlanDict) is semidet.
%  Produce a declarative plan describing how to evaluate the requested
%  predicate. Plans are represented as dicts containing the head descriptor,
%  root operator, materialised fact tables, and metadata.
build_query_plan(Pred/Arity, Options, Plan) :-
    must_be(atom, Pred),
    must_be(integer, Arity),
    Arity >= 0,
    HeadSpec = predicate{name:Pred, arity:Arity},
    compute_dependency_group(Pred/Arity, GroupSpecs),
    (   GroupSpecs = [HeadSpec]
    ->  gather_predicate_clauses(HeadSpec, Clauses),
        partition_recursive_clauses(Pred, Arity, Clauses, BaseClauses, RecClauses),
        (   RecClauses == []
        ->  classify_clauses(BaseClauses, Classification),
            build_plan_by_class(Classification, Pred, Arity, BaseClauses, Options, Plan)
        ;   build_recursive_plan(HeadSpec, [HeadSpec], BaseClauses, RecClauses, Options, Plan)
        )
    ;   build_mutual_recursive_plan(GroupSpecs, HeadSpec, Options, Plan)
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
    build_rule_clause([HeadSpec], HeadSpec, HeadArgs, Body, Node, Relations),
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

build_recursive_plan(HeadSpec, GroupSpecs, BaseClauses, RecClauses, Options, Plan) :-
    get_dict(arity, HeadSpec, Arity),
    build_base_root(HeadSpec, BaseClauses, Options, BaseRoot, BaseRelations),
    build_recursive_variants(HeadSpec, GroupSpecs, RecClauses, RecursiveNodes, RecursiveRelations),
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

build_mutual_recursive_plan(GroupSpecs, HeadSpec, Options, Plan) :-
    maplist(build_member_plan(GroupSpecs, Options), GroupSpecs, MemberStructs, RelationLists),
    append(RelationLists, RelationsFlat),
    dedup_relations(RelationsFlat, CombinedRelations),
    Root = mutual_fixpoint{
        type:mutual_fixpoint,
        head:HeadSpec,
        members:MemberStructs
    },
    Plan = plan{
        head:HeadSpec,
        root:Root,
        relations:CombinedRelations,
        metadata:_{classification:mutual_recursive, options:Options},
        is_recursive:true
    }.

build_member_plan(GroupSpecs, Options, PredSpec, member{
        type:member,
        predicate:PredSpec,
        base:BaseRoot,
        recursive:RecursiveNodes,
        width:Arity
    }, Relations) :-
    get_dict(arity, PredSpec, Arity),
    gather_predicate_clauses(PredSpec, Clauses),
    partition_mutual_clauses(GroupSpecs, Clauses, BaseClauses, RecClauses),
    build_base_root(PredSpec, BaseClauses, Options, BaseRoot, BaseRelations),
    build_recursive_variants(PredSpec, GroupSpecs, RecClauses, RecursiveNodes, RecursiveRelations),
    append(BaseRelations, RecursiveRelations, Relations).

partition_mutual_clauses(GroupSpecs, Clauses, BaseClauses, RecClauses) :-
    partition(clause_has_group_literal(GroupSpecs), Clauses, RecClauses, BaseClauses).

clause_has_group_literal(GroupSpecs, _Head-Body) :-
    body_to_list(Body, Terms),
    member(Term, Terms),
    group_literal_spec(Term, GroupSpecs, _).

build_base_root(_HeadSpec, [], _Options, empty{type:empty, width:Arity}, []) :-
    Arity = 0.
build_base_root(HeadSpec, [], _Options, empty{type:empty, width:Arity}, []) :-
    get_dict(arity, HeadSpec, Arity).
build_base_root(HeadSpec, Clauses, Options, BaseRoot, Relations) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    classify_clauses(Clauses, Class),
    build_plan_by_class(Class, Pred, Arity, Clauses, Options, BasePlan),
    get_dict(root, BasePlan, BaseRoot),
    get_dict(relations, BasePlan, Relations).

build_recursive_variants(_HeadSpec, _GroupSpecs, [], [], []) :- !.
build_recursive_variants(HeadSpec, GroupSpecs, Clauses, Variants, Relations) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    findall(variant(Node, RelList),
        (   member(Head-Body, Clauses),
            build_recursive_clause_variants(HeadSpec, GroupSpecs, Head-Body, VariantStructs),
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

build_recursive_clause_variants(HeadSpec, GroupSpecs, Head-Body, Variants) :-
    Head =.. [_|HeadArgs],
    body_to_list(Body, Terms),
    group_occurrence_positions(GroupSpecs, Terms, Occurrences),
    (   Occurrences = []
    ->  assign_roles_for_variant(HeadSpec, GroupSpecs, Terms, none, Roles),
        build_pipeline(HeadSpec, GroupSpecs, Terms, Roles, PipelineNode, RelList, VarMap, _Width),
        project_to_head(HeadArgs, PipelineNode, VarMap, Node),
        Variants = [variant(Node, RelList)]
    ;   findall(variant(Node, RelList),
        (   member(occurrence(Index, PredSpec), Occurrences),
            assign_roles_for_variant(HeadSpec, GroupSpecs, Terms, delta(Index, PredSpec), Roles),
            build_pipeline(HeadSpec, GroupSpecs, Terms, Roles, PipelineNode, RelList, VarMap, _Width),
            project_to_head(HeadArgs, PipelineNode, VarMap, Node)
        ),
        Variants)
    ).

group_occurrence_positions(GroupSpecs, Terms, Occurrences) :-
    findall(occurrence(Index, PredSpec),
        (   nth0(Index, Terms, Term),
            group_literal_spec(Term, GroupSpecs, PredSpec)
        ),
        Occurrences).

assign_roles_for_variant(HeadSpec, GroupSpecs, Terms, DeltaSpec, Roles) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    same_length(Terms, Roles),
    assign_roles_for_variant_(Terms, GroupSpecs, Pred, Arity, DeltaSpec, 0, Roles).

assign_roles_for_variant_([], _GroupSpecs, _Pred, _Arity, _DeltaSpec, _Pos, []).
assign_roles_for_variant_([Term|Rest], GroupSpecs, Pred, Arity, DeltaSpec, Pos, [Role|Roles]) :-
    (   constraint_goal(Term)
    ->  Role = constraint
    ;   term_signature(Term, Name/TermArity),
        TermArity =:= Arity,
        Pred == Name
    ->  (   DeltaSpec = delta(Pos, _)
        ->  Role = recursive(delta)
        ;   Role = recursive(total)
        )
    ;   group_literal_spec(Term, GroupSpecs, PredSpec)
    ->  (   DeltaSpec = delta(Pos, PredSpec)
        ->  RoleKind = delta
        ;   RoleKind = total
        ),
        Role = mutual(PredSpec, RoleKind)
    ;   Role = relation
    ),
    Pos1 is Pos + 1,
    assign_roles_for_variant_(Rest, GroupSpecs, Pred, Arity, DeltaSpec, Pos1, Roles).

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

group_literal_spec(Term, GroupSpecs, PredSpec) :-
    term_signature(Term, Name/Arity),
    member(PredSpec, GroupSpecs),
    spec_signature(PredSpec, Name/Arity).

constraint_goal(Goal0) :-
    strip_module(Goal0, _, Goal),
    (   arithmetic_goal(Goal)
    ->  true
    ;   functor(Goal, Functor, Arity),
        Arity =:= 2,
        (   Functor == '='
        ;   Functor == '=='
        ;   Functor == dif
        ;   Functor == >
        ;   Functor == <
        ;   Functor == >=
        ;   Functor == =<
        ;   Functor == =:=
        ;   atom_codes(Functor, [92, 61])
        )
    ).

arithmetic_goal(Goal0) :-
    strip_module(Goal0, _, Goal),
    functor(Goal, is, 2).

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
    build_rule_clause([HeadSpec], HeadSpec, HeadArgs, Body, Node, ClauseRelations),
    Relations = ClauseRelations.

build_rule_clause(GroupSpecs, HeadSpec, HeadArgs, Body, Root, Relations) :-
    Body \= true,
    body_to_list(Body, Terms),
    Terms \= [],
    roles_for_nonrecursive_terms(HeadSpec, Terms, Roles),
    build_pipeline(HeadSpec, GroupSpecs, Terms, Roles, PipelineNode, Relations, VarMap, Width),
    project_to_head(HeadArgs, PipelineNode, VarMap, Root),
    length(HeadArgs, WidthHead),
    Width >= WidthHead.

body_to_list(true, []) :- !.
body_to_list((A, B), Terms) :- !,
    body_to_list(A, TA),
    body_to_list(B, TB),
    append(TA, TB, Terms).
body_to_list(Goal, [Goal]).

build_pipeline(HeadSpec, GroupSpecs, [Term|Rest], [Role|Roles], FinalNode, FinalRelations, FinalVarMap, FinalWidth) :-
    build_initial_node(HeadSpec, GroupSpecs, Term, Role, Node0, Relations0, VarMap0, Width0),
    fold_terms(GroupSpecs, Rest, Roles, HeadSpec, Node0, VarMap0, Width0, Relations0,
               FinalNode, FinalVarMap, FinalWidth, FinalRelations).

build_initial_node(HeadSpec, _GroupSpecs, _Term, constraint, _Node, _Relations, _VarMap, _Width) :-
    get_dict(name, HeadSpec, Pred),
    get_dict(arity, HeadSpec, Arity),
    format(user_error, 'C# query target: clause for ~w/~w begins with a constraint; reorder body literals.~n', [Pred, Arity]),
    fail.
build_initial_node(_HeadSpec, _GroupSpecs, Term, relation, Node, Relations, VarMap, Width) :-
    Term =.. [Pred|Args],
    length(Args, Arity),
    ensure_relation(Pred, Arity, [], Relations),
    relation_scan_node(Pred, Arity, Node),
    init_var_map(Args, 0, VarMap),
    Width = Arity.
build_initial_node(HeadSpec, _GroupSpecs, Term, recursive(RoleKind), Node, Relations, VarMap, Width) :-
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
build_initial_node(_HeadSpec, _GroupSpecs, Term, mutual(PredSpec, RoleKind), Node, Relations, VarMap, Width) :-
    spec_signature(PredSpec, _Name/Arity),
    Term =.. [_|Args],
    cross_ref_node(PredSpec, RoleKind, Node),
    init_var_map(Args, 0, VarMap),
    Width = Arity,
    Relations = [].

fold_terms(_GroupSpecs, [], [], _HeadSpec, Node, VarMap, Width, Relations, Node, VarMap, Width, Relations).
fold_terms(GroupSpecs, [Term|Rest], [Role|Roles], HeadSpec, AccNode, VarMapIn, WidthIn, RelationsIn,
           NodeOut, VarMapOut, WidthOut, RelationsOut) :-
    (   Role = constraint
    ->  build_constraint_node(Term, AccNode, VarMapIn, WidthIn,
                               ConstraintNode, VarMapMid, WidthMid),
        fold_terms(GroupSpecs, Rest, Roles, HeadSpec, ConstraintNode, VarMapMid, WidthMid, RelationsIn,
                   NodeOut, VarMapOut, WidthOut, RelationsOut)
    ;   Role = relation
    ->  Term =.. [Pred|Args],
        length(Args, Arity),
        ensure_relation(Pred, Arity, RelationsIn, RelationsNext),
        relation_scan_node(Pred, Arity, RightNode),
        build_join_node(AccNode, RightNode, Args, VarMapIn, WidthIn, Arity, VarMapMid, WidthMid, JoinNode),
        fold_terms(GroupSpecs, Rest, Roles, HeadSpec, JoinNode, VarMapMid, WidthMid, RelationsNext,
                   NodeOut, VarMapOut, WidthOut, RelationsOut)
    ;   Role = recursive(RoleKind)
    ->  get_dict(arity, HeadSpec, Arity),
        Term =.. [_|Args],
        recursive_ref_node(HeadSpec, RoleKind, RightNode),
        build_join_node(AccNode, RightNode, Args, VarMapIn, WidthIn, Arity, VarMapMid, WidthMid, JoinNode),
        fold_terms(GroupSpecs, Rest, Roles, HeadSpec, JoinNode, VarMapMid, WidthMid, RelationsIn,
                   NodeOut, VarMapOut, WidthOut, RelationsOut)
    ;   Role = mutual(PredSpec, RoleKind)
    ->  spec_signature(PredSpec, _Name/Arity),
        Term =.. [_|Args],
        cross_ref_node(PredSpec, RoleKind, RightNode),
        build_join_node(AccNode, RightNode, Args, VarMapIn, WidthIn, Arity, VarMapMid, WidthMid, JoinNode),
        fold_terms(GroupSpecs, Rest, Roles, HeadSpec, JoinNode, VarMapMid, WidthMid, RelationsIn,
                   NodeOut, VarMapOut, WidthOut, RelationsOut)
    ).

relation_scan_node(Pred, Arity, relation_scan{type:relation_scan, predicate:predicate{name:Pred, arity:Arity}, width:Arity}).

recursive_ref_node(HeadSpec, RoleKind, recursive_ref{type:recursive_ref, predicate:HeadSpec, role:RoleKind, width:Arity}) :-
    get_dict(arity, HeadSpec, Arity).

cross_ref_node(PredSpec, RoleKind, cross_ref{type:cross_ref, predicate:PredSpec, role:RoleKind, width:Arity}) :-
    get_dict(arity, PredSpec, Arity).

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

build_constraint_node(Term, InputNode, VarMapIn, WidthIn,
        NodeOut, VarMapOut, WidthOut) :-
    (   arithmetic_goal(Term)
    ->  build_arithmetic_node(Term, InputNode, VarMapIn, WidthIn,
            NodeOut, VarMapOut, WidthOut)
    ;   constraint_condition(Term, VarMapIn, Condition),
        NodeOut = selection{
            type:selection,
            input:InputNode,
            predicate:Condition,
            width:WidthIn
        },
        VarMapOut = VarMapIn,
        WidthOut = WidthIn
    ).

build_arithmetic_node(Goal, InputNode, VarMapIn, WidthIn,
        arithmetic{
            type:arithmetic,
            input:InputNode,
            expression:Expression,
            result_index:ResultIndex,
            width:WidthOut
        },
        VarMapOut, WidthOut) :-
    Goal =.. [is, Left, ExprTerm],
    (   var(Left)
    ->  true
    ;   format(user_error, 'C# query target: left operand of is/2 must be an unbound variable (~q).~n', [Goal]),
        fail
    ),
    (   lookup_var_index(VarMapIn, Left, _Existing)
    ->  format(user_error, 'C# query target: variable ~w already bound before is/2 evaluation.~n', [Left]),
        fail
    ;   true
    ),
    arithmetic_expression(ExprTerm, VarMapIn, Expression),
    ResultIndex is WidthIn,
    WidthOut is WidthIn + 1,
    replace_var_index(VarMapIn, Left, ResultIndex, VarMapOut).

replace_var_index(VarMap, Var, Index, [Var-Index|Trimmed]) :-
    remove_var_mapping(VarMap, Var, Trimmed).

remove_var_mapping([], _Var, []).
remove_var_mapping([Var0-Idx|Rest], Var, Result) :-
    (   Var0 == Var
    ->  remove_var_mapping(Rest, Var, Result)
    ;   remove_var_mapping(Rest, Var, Tail),
        Result = [Var0-Idx|Tail]
    ).

arithmetic_expression(Term, _VarMap, expr{type:value, value:Term}) :-
    number(Term), !.
arithmetic_expression(Term, VarMap, expr{type:column, index:Index}) :-
    var(Term), !,
    (   lookup_var_index(VarMap, Term, Index)
    ->  true
    ;   format(user_error, 'C# query target: variable ~w not bound before arithmetic evaluation.~n', [Term]),
        fail
    ).
arithmetic_expression(Term, VarMap, Expr) :-
    Term =.. [Op, Arg],
    arithmetic_unary_operator(Op, Operator), !,
    arithmetic_expression(Arg, VarMap, SubExpr),
    (   Operator == identity
    ->  Expr = SubExpr
    ;   Expr = expr{type:unary, op:Operator, expr:SubExpr}
    ).
arithmetic_expression(Term, VarMap, expr{type:binary, op:Operator, left:LeftExpr, right:RightExpr}) :-
    Term =.. [Op, Left, Right],
    arithmetic_binary_operator(Op, Operator), !,
    arithmetic_expression(Left, VarMap, LeftExpr),
    arithmetic_expression(Right, VarMap, RightExpr).
arithmetic_expression(Term, _VarMap, _Expr) :-
    format(user_error, 'C# query target: unsupported arithmetic expression ~q.~n', [Term]),
    fail.

arithmetic_unary_operator(+, identity).
arithmetic_unary_operator(-, negate).

arithmetic_binary_operator(+, add).
arithmetic_binary_operator(-, subtract).
arithmetic_binary_operator(*, multiply).
arithmetic_binary_operator(/, divide).
arithmetic_binary_operator('//', int_divide).
arithmetic_binary_operator(mod, modulo).

constraint_condition(Goal0, VarMap, Condition) :-
    strip_module(Goal0, _, Goal),
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
    ;   Functor == >
    ->  build_condition(gt, Left, Right, VarMap, Condition)
    ;   Functor == <
    ->  build_condition(lt, Left, Right, VarMap, Condition)
    ;   Functor == >=
    ->  build_condition(ge, Left, Right, VarMap, Condition)
    ;   Functor == =<
    ->  build_condition(le, Left, Right, VarMap, Condition)
    ;   Functor == =:=
    ->  build_condition(eq, Left, Right, VarMap, Condition)
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
constraint_operand(_VarMap, _Term, _Operand) :-
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
    ;   dynamic_source_metadata(Pred/Arity, Metadata)
    ->  Relation = relation{predicate:predicate{name:Pred, arity:Arity}, facts:dynamic(Metadata)},
        RelationsOut = [Relation|RelationsIn]
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
    (   var(Term) -> Copy = Term
    ;   atomic(Term) -> Copy = Term
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
    relation_blocks(Relations, ProviderBody, UsesDynamic),
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
    (   UsesDynamic == true
    ->  DynamicUsing = 'using UnifyWeaver.QueryRuntime.Dynamic;
'
    ;   DynamicUsing = ''
    ),
    format(atom(Code),
'// Auto-generated by UnifyWeaver
using System;
using System.Linq;
using UnifyWeaver.QueryRuntime;
~w

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
', [DynamicUsing, ModuleClass, ProviderSection, PredStr, Arity, PlanExpr, RecLiteral]).

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
    is_dict(Node, arithmetic), !,
    get_dict(input, Node, Input),
    get_dict(expression, Node, Expression),
    get_dict(result_index, Node, ResultIndex),
    get_dict(width, Node, Width),
    emit_plan_expression(Input, InputExpr),
    emit_arithmetic_expression(Expression, ExpressionExpr),
    format(atom(Expr), 'new ArithmeticNode(~w, ~w, ~w, ~w)',
        [InputExpr, ExpressionExpr, ResultIndex, Width]).
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
    is_dict(Node, cross_ref), !,
    get_dict(predicate, Node, predicate{name:Name, arity:Arity}),
    get_dict(role, Node, Role),
    recursive_role_atom(Role, RoleAtom),
    atom_string(Name, NameStr),
    format(atom(Expr), 'new CrossRefNode(new PredicateId("~w", ~w), RecursiveRefKind.~w)',
        [NameStr, Arity, RoleAtom]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, mutual_fixpoint), !,
    get_dict(head, Node, predicate{name:Name, arity:Arity}),
    get_dict(members, Node, Members),
    maplist(emit_mutual_member_expression, Members, MemberExprs),
    atomic_list_concat(MemberExprs, ', ', MemberList),
    atom_string(Name, NameStr),
    format(atom(Expr), 'new MutualFixpointNode(new MutualMember[]{ ~w }, new PredicateId("~w", ~w))',
        [MemberList, NameStr, Arity]).
emit_plan_expression(Node, Expr) :-
    is_dict(Node, empty), !,
    get_dict(width, Node, Width),
    format(atom(Expr), 'new EmptyNode(~w)', [Width]).
emit_plan_expression(Node, _Expr) :-
    format(user_error, 'C# query target: cannot render plan node ~q.~n', [Node]),
    fail.

recursive_role_atom(delta, 'Delta').
recursive_role_atom(total, 'Total').

emit_mutual_member_expression(Member, Expr) :-
    get_dict(predicate, Member, predicate{name:Name, arity:Arity}),
    get_dict(base, Member, BasePlan),
    get_dict(recursive, Member, RecursivePlans),
    emit_plan_expression(BasePlan, BaseExpr),
    maplist(emit_plan_expression, RecursivePlans, RecursiveExprs),
    atomic_list_concat(RecursiveExprs, ', ', RecursiveList),
    atom_string(Name, NameStr),
    format(atom(Expr), 'new MutualMember(new PredicateId("~w", ~w), ~w, new PlanNode[]{ ~w })',
        [NameStr, Arity, BaseExpr, RecursiveList]).

selection_condition_expression(condition{type:eq, left:Left, right:Right}, TupleVar, Expr) :-
    operand_expression(Left, TupleVar, LeftExpr),
    operand_expression(Right, TupleVar, RightExpr),
    format(atom(Expr), 'Equals(~w, ~w)', [LeftExpr, RightExpr]).
selection_condition_expression(condition{type:neq, left:Left, right:Right}, TupleVar, Expr) :-
    operand_expression(Left, TupleVar, LeftExpr),
    operand_expression(Right, TupleVar, RightExpr),
    format(atom(Expr), '!(Equals(~w, ~w))', [LeftExpr, RightExpr]).
selection_condition_expression(condition{type:gt, left:Left, right:Right}, TupleVar, Expr) :-
    comparison_condition_expression(Left, Right, TupleVar, ' > 0', Expr).
selection_condition_expression(condition{type:ge, left:Left, right:Right}, TupleVar, Expr) :-
    comparison_condition_expression(Left, Right, TupleVar, ' >= 0', Expr).
selection_condition_expression(condition{type:lt, left:Left, right:Right}, TupleVar, Expr) :-
    comparison_condition_expression(Left, Right, TupleVar, ' < 0', Expr).
selection_condition_expression(condition{type:le, left:Left, right:Right}, TupleVar, Expr) :-
    comparison_condition_expression(Left, Right, TupleVar, ' <= 0', Expr).

comparison_condition_expression(Left, Right, TupleVar, Suffix, Expr) :-
    operand_expression(Left, TupleVar, LeftExpr),
    operand_expression(Right, TupleVar, RightExpr),
    format(atom(Expr), 'QueryExecutor.CompareValues(~w, ~w)~w', [LeftExpr, RightExpr, Suffix]).

emit_arithmetic_expression(expr{type:column, index:Index}, Expr) :-
    format(atom(Expr), 'new ColumnExpression(~w)', [Index]).
emit_arithmetic_expression(expr{type:value, value:Value}, Expr) :-
    csharp_literal(Value, Literal),
    format(atom(Expr), 'new ConstantExpression(~w)', [Literal]).
emit_arithmetic_expression(expr{type:unary, op:Operator, expr:SubExpr}, Expr) :-
    emit_arithmetic_expression(SubExpr, SubCode),
    arithmetic_unary_operator_atom(Operator, OperatorAtom),
    format(atom(Expr), 'new UnaryArithmeticExpression(ArithmeticUnaryOperator.~w, ~w)',
        [OperatorAtom, SubCode]).
emit_arithmetic_expression(expr{type:binary, op:Operator, left:Left, right:Right}, Expr) :-
    emit_arithmetic_expression(Left, LeftCode),
    emit_arithmetic_expression(Right, RightCode),
    arithmetic_binary_operator_atom(Operator, OperatorAtom),
    format(atom(Expr), 'new BinaryArithmeticExpression(ArithmeticBinaryOperator.~w, ~w, ~w)',
        [OperatorAtom, LeftCode, RightCode]).

arithmetic_unary_operator_atom(negate, 'Negate').

arithmetic_binary_operator_atom(add, 'Add').
arithmetic_binary_operator_atom(subtract, 'Subtract').
arithmetic_binary_operator_atom(multiply, 'Multiply').
arithmetic_binary_operator_atom(divide, 'Divide').
arithmetic_binary_operator_atom(int_divide, 'IntegerDivide').
arithmetic_binary_operator_atom(modulo, 'Modulo').

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
    LeftMax is LeftWidth - 1,
    findall(Item,
        (   between(0, LeftMax, LIdx),
            format(atom(Item), 'left[~w]', [LIdx])
        ),
        LeftItems),
    RightMax is RightWidth - 1,
    findall(Item,
        (   between(0, RightMax, RIdx),
            format(atom(Item), 'right[~w]', [RIdx])
        ),
        RightItems),
    append(LeftItems, RightItems, Items),
    atomic_list_concat(Items, ', ', Expr).

relation_blocks(Relations, ProviderStatements, UsesDynamic) :-
    findall(Line,
        (   member(relation{predicate:predicate{name:Name, arity:Arity}, facts:Rows}, Relations),
            Rows \= dynamic(_),
            member(Row, Rows),
            atom_string(Name, NameStr),
            maplist(csharp_literal, Row, ArgLiterals),
            atomic_list_concat(ArgLiterals, ', ', ArgList),
            format(atom(Line), '            provider.AddFact(new PredicateId("~w", ~w), ~w);', [NameStr, Arity, ArgList])
        ),
        FactLines),
    findall(Block,
        (   member(relation{predicate:predicate{name:Name, arity:Arity}, facts:dynamic(Metadata)}, Relations),
            dynamic_relation_block(Name, Arity, Metadata, Block)
        ),
        DynamicBlocks),
    append(FactLines, DynamicBlocks, AllLines),
    (   AllLines == []
    ->  ProviderStatements = '',
        UsesDynamic = false
    ;   atomic_list_concat(AllLines, '\n', ProviderStatements),
        (   DynamicBlocks == []
        ->  UsesDynamic = false
        ;   UsesDynamic = true
        )
    ).

dynamic_relation_block(Name, Arity, Metadata, Block) :-
    atom_string(Name, NameStr),
    dynamic_reader_literal(Metadata, Arity, ReaderLiteral),
    format(atom(Block),
'            foreach (var row in ~w.Read())
            {
                provider.AddFact(new PredicateId(\"~w\", ~w), row);
            }',
        [ReaderLiteral, NameStr, Arity]).

dynamic_reader_literal(Metadata, Arity, Literal) :-
    (   get_dict(record_format, Metadata, Format)
    ->  true
    ;   Format = text_line
    ),
    (   Format == json
    ->  json_reader_literal(Metadata, Arity, Literal)
    ;   delimited_reader_literal(Metadata, Arity, Literal)
    ).

delimited_reader_literal(Metadata, Arity, Literal) :-
    get_dict(field_separator, Metadata, FieldSep0),
    field_separator_literal(FieldSep0, FieldSepLiteral),
    get_dict(record_separator, Metadata, RecSep0),
    record_separator_literal(RecSep0, RecSepLiteral),
    get_dict(quote_style, Metadata, QuoteStyle0),
    quote_style_literal(QuoteStyle0, QuoteLiteral),
    metadata_skip_rows(Metadata, SkipRows),
    metadata_input_literal(Metadata, InputLiteral),
    format(atom(Literal),
'new DelimitedTextReader(new DynamicSourceConfig
            {
                InputPath = ~w,
                FieldSeparator = ~w,
                RecordSeparator = RecordSeparatorKind.~w,
                QuoteStyle = QuoteStyle.~w,
                SkipRows = ~w,
                ExpectedWidth = ~w
            })',
        [InputLiteral, FieldSepLiteral, RecSepLiteral, QuoteLiteral, SkipRows, Arity]).

json_reader_literal(Metadata, Arity, Literal) :-
    metadata_input_literal(Metadata, InputLiteral),
    metadata_skip_rows(Metadata, SkipRows),
    (   get_dict(record_separator, Metadata, RecSep0)
    ->  true
    ;   RecSep0 = line_feed
    ),
    record_separator_literal(RecSep0, RecSepLiteral),
    metadata_columns_literal(Metadata, Arity, ColumnLiteral),
    metadata_type_literal(Metadata, TypeLiteral),
    (   get_dict(return_object, Metadata, ReturnObject),
        ReturnObject == true
    ->  ReturnLiteral = 'true'
    ;   ReturnLiteral = 'false'
    ),
    format(atom(Literal),
'new JsonStreamReader(new JsonSourceConfig
            {
                InputPath = ~w,
                Columns = ~w,
                RecordSeparator = RecordSeparatorKind.~w,
                SkipRows = ~w,
                ExpectedWidth = ~w,
                TargetTypeName = ~w,
                ReturnObject = ~w
            })',
        [InputLiteral, ColumnLiteral, RecSepLiteral, SkipRows, Arity, TypeLiteral, ReturnLiteral]).

metadata_columns_literal(Metadata, Arity, Literal) :-
    (   get_dict(columns, Metadata, Columns),
        Columns \= []
    ->  maplist(atom_string, Columns, ColumnStrings)
    ;   numlist(1, Arity, Indexes),
        maplist(default_column_name, Indexes, ColumnStrings)
    ),
    string_array_literal(ColumnStrings, Literal).

metadata_type_literal(Metadata, Literal) :-
    (   get_dict(type_hint, Metadata, TypeHint),
        TypeHint \= none
    ->  csharp_literal(TypeHint, Literal)
    ;   Literal = 'null'
    ).

default_column_name(Index, Name) :-
    format(string(Name), 'col~w', [Index]).

metadata_skip_rows(Metadata, SkipRows) :-
    (   get_dict(skip_rows, Metadata, SkipRows0)
    ->  SkipRows = SkipRows0
    ;   SkipRows = 0
    ).

metadata_input_literal(Metadata, Literal) :-
    get_dict(input, Metadata, InputSpec),
    input_literal(InputSpec, Literal).

field_separator_literal(Value, Literal) :-
    literal_string(Value, Str),
    escape_verbatim_string(Str, Escaped),
    format(atom(Literal), '@"~w"', [Escaped]).

record_separator_literal(line_feed, 'LineFeed') :- !.
record_separator_literal(nul, 'Null') :- !.
record_separator_literal(json, 'Json') :- !.
record_separator_literal(Value, Name) :-
    literal_pascal_name(Value, Name).

quote_style_literal(none, 'None') :- !.
quote_style_literal(double_quote, 'DoubleQuote') :- !.
quote_style_literal(single_quote, 'SingleQuote') :- !.
quote_style_literal(json_escape, 'Json') :- !.
quote_style_literal(Value, Name) :-
    literal_pascal_name(Value, Name).

input_literal(file(Path), Literal) :-
    csharp_literal(Path, Literal).
input_literal(stdin, Literal) :-
    csharp_literal('stdin', Literal).
input_literal(pipe(Command), Literal) :-
    term_string(Command, String),
    csharp_literal(String, Literal).

csharp_literal(Value, Literal) :-
    number(Value),
    !,
    format(atom(Literal), '~w', [Value]).
csharp_literal(Value, Literal) :-
    atom(Value),
    !,
    atom_string(Value, String),
    escape_csharp_string(String, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).
csharp_literal(Value, Literal) :-
    string(Value),
    !,
    escape_csharp_string(Value, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).
csharp_literal(Value, Literal) :-
    term_string(Value, String),
    escape_csharp_string(String, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).

literal_string(Value, String) :-
    atom(Value),
    !,
    atom_string(Value, String).
literal_string(Value, String) :-
    string(Value),
    !,
    String = Value.
literal_string(Value, String) :-
    term_string(Value, String).

escape_csharp_string(Input, Escaped) :-
    string_codes(Input, Codes),
    maplist(escape_code, Codes, Parts),
    atomic_list_concat(Parts, '', Escaped).

escape_code(92, Atom) :- !, atom_codes(Atom, [92, 92]).
escape_code(34, Atom) :- !, atom_codes(Atom, [92, 34]).
escape_code(10, Atom) :- !, atom_codes(Atom, [92, 110]).
escape_code(13, Atom) :- !, atom_codes(Atom, [92, 114]).
escape_code(9, Atom)  :- !, atom_codes(Atom, [92, 116]).
escape_code(Code, Atom) :- atom_codes(Atom, [Code]).

escape_verbatim_string(Input, Escaped) :-
    string_codes(Input, Codes),
    maplist(escape_verbatim_code, Codes, Parts),
    atomic_list_concat(Parts, '', Escaped).

escape_verbatim_code(34, '""') :- !.
escape_verbatim_code(Code, Atom) :-
    atom_codes(Atom, [Code]).

string_array_literal([], 'Array.Empty<string>()') :- !.
string_array_literal(Items, Literal) :-
    findall(Quoted,
        (   member(Item, Items),
            escape_csharp_string(Item, Escaped),
            format(atom(Quoted), '"~w"', [Escaped])
        ),
        QuotedItems),
    atomic_list_concat(QuotedItems, ', ', Inner),
    format(atom(Literal), 'new[]{ ~w }', [Inner]).

plan_module_name(Plan, ModuleName) :-
    get_dict(head, Plan, predicate{name:Pred, arity:_}),
    predicate_pascal(Pred, Pascal),
    atom_concat(Pascal, 'QueryModule', ModuleName).

predicate_pascal(Atom, Pascal) :-
    atom_string(Atom, Text),
    snake_case_to_pascal(Text, PascalString),
    atom_string(Pascal, PascalString).

literal_pascal_name(Value, Name) :-
    (   atom(Value) -> atom_string(Value, String)
    ;   string(Value) -> String = Value
    ;   term_string(Value, String)
    ),
    snake_case_to_pascal(String, Name).

snake_case_to_pascal(Text, Pascal) :-
    split_string(Text, '_', '_', Parts),
    maplist(capitalise_string, Parts, Caps),
    atomic_list_concat(Caps, '', Pascal).

capitalise_string(Input, Output) :-
    (   Input = ''
    ->  Output = ''
    ;   string_lower(Input, Lower),
        sub_string(Lower, 0, 1, _, First),
        sub_string(Lower, 1, _, 0, Rest),
        string_upper(First, UpperFirst),
        string_concat(UpperFirst, Rest, Output)
    ).
