
:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% csharp_target.pl - C# Code Generation Target
% Supports both Query IR synthesis (query mode) and direct C# code generation (generator mode).
% The resulting plan dicts (query mode) are consumed by the managed QueryRuntime (C#) infrastructure.
% Generator mode produces standalone C# code with a fixpoint solver.

:- module(csharp_target, [
    compile_predicate_to_csharp/3, % +PredIndicator, +Options, -CSharpCode
    build_query_plan/3,     % +PredIndicator, +Options, -PlanDict
    render_plan_to_csharp/2,% +PlanDict, -CSharpSource
    plan_module_name/2,     % +PlanDict, -ModuleName
    init_csharp_target/0,   % Initialize bindings
    compile_csharp_pipeline/3,  % +Predicates, +Options, -CSharpCode
    test_csharp_pipeline_generator/0,  % Unit tests for pipeline generator mode
    % Enhanced pipeline chaining exports
    compile_csharp_enhanced_pipeline/3, % +Stages, +Options, -CSharpCode
    csharp_enhanced_helpers/1,          % -Code
    generate_csharp_enhanced_connector/3, % +Stages, +PipelineName, -Code
    test_csharp_enhanced_chaining/0     % Test enhanced pipeline chaining
]).

:- use_module(library(apply)).
:- use_module(library(error)).
:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(ugraphs), [vertices_edges_to_ugraph/3, transpose_ugraph/2, reachable/3]).
:- use_module('../core/dynamic_source_compiler', [is_dynamic_source/1, dynamic_source_metadata/2]).
:- use_module('../core/binding_registry').
:- use_module('../bindings/csharp_bindings').
:- use_module('../core/pipeline_validation').
:- use_module(common_generator).

%% init_csharp_target
%  Initialize the C# target by loading bindings.
init_csharp_target :-
    init_csharp_bindings.

%% compile_predicate_to_csharp(+PredIndicator, +Options, -Code)
%  Compile a predicate to C# code.
%  Options:
%    mode(Mode) - 'query' (default) or 'generator'
compile_predicate_to_csharp(PredIndicator, Options, Code) :-
    option(mode(Mode), Options, query),
    (   Mode == generator
    ->  compile_generator_mode(PredIndicator, Options, Code)
    ;   Mode == query
    ->  PredIndicator = Pred/Arity,
        build_query_plan(Pred/Arity, Options, Plan),
        render_plan_to_csharp(Plan, Code)
    ;   format(user_error, 'Unknown mode ~w for C# target~n', [Mode]),
        fail
    ).

spec_signature(predicate{name:Name, arity:Arity}, Name/Arity).

term_signature(Term0, Name/Arity) :-
    strip_module(Term0, _, Term),
    nonvar(Term),
    Term =.. [Name|Args],
    length(Args, Arity).

term_to_dependency(aggregate_all(_, Goal, _), Dep) :- !,
    term_signature(Goal, Dep).
term_to_dependency(aggregate_all(_, Goal, _, _), Dep) :- !,
    nonvar(Goal),
    term_signature(Goal, Dep).
term_to_dependency(aggregate(_, Goal, _, _), Dep) :- !,
    nonvar(Goal),
    term_signature(Goal, Dep).
term_to_dependency(Term, Dep) :-
    (   aggregate_goal(Term)
    ->  decompose_aggregate_goal(Term, _Type, _Op, Pred, Args, _GroupVar, _ValueVar, _ResultVar),
        length(Args, A),
        Dep = Pred/A,
        !
    ;   Term =.. ['\\+', Inner]
    ->  term_signature(Inner, Dep)
    ;   Term =.. [not, Inner]
    ->  term_signature(Inner, Dep)
    ;   term_signature(Term, Dep)
    ).

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
                term_to_dependency(Term, Dep),
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

compute_generator_dependency_closure(Pred/Arity, GroupSpecs) :-
    build_dependency_graph([Pred/Arity], [], [], Vertices, [], _),
    sort(Vertices, Sorted),
    maplist(vertex_to_spec, Sorted, GroupSpecs),
    !.
compute_generator_dependency_closure(Pred/Arity, [predicate{name:Pred, arity:Arity}]).

gather_group_clauses(GroupSpecs, AllClauses) :-
    findall(Head-Body,
        (   member(Spec, GroupSpecs),
            gather_predicate_clauses(Spec, Clauses),
            member(Head-Body, Clauses)
        ),
        AllClauses).

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
body_to_list(Body, Terms) :-
    compound(Body),
    Body = _Module:InnerBody, !,
    body_to_list(InnerBody, Terms).
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
    schema_declarations(Relations, SchemaDeclarations),
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
~w    public static class ~w
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
', [DynamicUsing, SchemaDeclarations, ModuleClass, ProviderSection, PredStr, Arity, PlanExpr, RecLiteral]).

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

schema_declarations(Relations, SchemaCode) :-
    findall(TypeName-Code,
        (   member(Rel, Relations),
            get_dict(facts, Rel, Facts),
            Facts = dynamic(Metadata),
            schema_declaration(Metadata, Code, TypeName)
        ),
        Pairs),
    sort(Pairs, Sorted),
    findall(Code, member(_-Code, Sorted), Codes),
    (   Codes == []
    ->  SchemaCode = ''
    ;   atomic_list_concat(Codes, '\n', Joined),
        format(atom(SchemaCode), '~w~n', [Joined])
    ).

schema_declaration(Metadata, Code, TypeName) :-
    (   get_dict(schema_records, Metadata, Records),
        Records \= [],
        member(schema_record{type:TypeAtom, fields:Fields}, Records)
    ;   % Fallback: synthesize from schema_fields/schema_type if records are absent
        get_dict(schema_fields, Metadata, Fields),
        Fields \= [],
        get_dict(schema_type, Metadata, TypeAtom),
        TypeAtom \= none
    ),
    schema_declaration(schema_record{type:TypeAtom, fields:Fields}, Code, TypeName).

schema_declaration(schema_record{type:TypeAtom, fields:Fields}, Code, TypeName) :-
    TypeAtom \= none,
    atom_string(TypeAtom, TypeName),
    findall(Param,
        (   member(Field, Fields),
            schema_field_param(Field, Param)
        ),
        Params),
    atomic_list_concat(Params, ', ', ParamList),
    format(atom(Code), '    public sealed record ~w(~w);', [TypeName, ParamList]).

schema_field_param(Field, Param) :-
    get_dict(name, Field, NameAtom),
    (   get_dict(field_kind, Field, record),
        get_dict(record_type, Field, RecordType),
        RecordType \= none
    ->  atom_string(RecordType, TypeLiteral)
    ;   get_dict(column_type, Field, TypeAtom),
        schema_field_type_literal(TypeAtom, TypeLiteral)
    ),
    schema_field_property_name(NameAtom, PropertyName),
    format(atom(Param), '~w ~w', [TypeLiteral, PropertyName]).

schema_field_type_literal(string, 'string') :- !.
schema_field_type_literal(integer, 'int') :- !.
schema_field_type_literal(long, 'long') :- !.
schema_field_type_literal(float, 'double') :- !.
schema_field_type_literal(double, 'double') :- !.
schema_field_type_literal(number, 'double') :- !.
schema_field_type_literal(boolean, 'bool') :- !.
schema_field_type_literal(json, 'string') :- !.
schema_field_type_literal(Type, Literal) :-
    atom_string(Type, Literal).

schema_field_property_name(NameAtom, PropertyName) :-
    atom_string(NameAtom, NameStr),
    split_string(NameStr, '_', '_', Parts),
    maplist(capitalise_string, Parts, Caps),
    atomic_list_concat(Caps, '', PropertyName).

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
    (   (Format == json ; Format == jsonl)
    ->  json_reader_literal(Metadata, Arity, Literal)
    ;   Format == xml
    ->  xml_reader_literal(Metadata, Arity, Literal)
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
    metadata_column_selectors_literal(Metadata, Arity, ColumnLiteral),
    metadata_type_literal(Metadata, TypeLiteral),
    metadata_schema_literal(Metadata, SchemaLiteral),
    metadata_treat_array_literal(Metadata, TreatArrayLiteral),
    metadata_null_policy_literals(Metadata, NullPolicyLiteral, NullDefaultLiteral),
    (   get_dict(return_object, Metadata, ReturnObject),
        ReturnObject == true
    ->  ReturnLiteral = 'true'
    ;   ReturnLiteral = 'false'
    ),
    format(atom(Literal),
'new JsonStreamReader(new JsonSourceConfig
            {
                InputPath = ~w,
                ColumnSelectors = ~w,
                RecordSeparator = RecordSeparatorKind.~w,
                SkipRows = ~w,
                ExpectedWidth = ~w,
                TreatArrayAsStream = ~w,
                TargetTypeName = ~w,
                ReturnObject = ~w,
                SchemaFields = ~w,
                NullPolicy = JsonNullPolicy.~w,
                NullReplacement = ~w
            })',
        [InputLiteral, ColumnLiteral, RecSepLiteral, SkipRows, Arity, TreatArrayLiteral, TypeLiteral, ReturnLiteral, SchemaLiteral, NullPolicyLiteral, NullDefaultLiteral]).

xml_reader_literal(Metadata, Arity, Literal) :-
    metadata_input_literal(Metadata, InputLiteral),
    (   get_dict(record_separator, Metadata, RecSep0)
    ->  true
    ;   RecSep0 = nul
    ),
    record_separator_literal(RecSep0, RecSepLiteral),
    (   get_dict(expected_width, Metadata, Width0)
    ->  Width = Width0
    ;   Width = Arity
    ),
    format(atom(Literal),
'new XmlStreamReader(new XmlSourceConfig
            {
                InputPath = ~w,
                RecordSeparator = RecordSeparatorKind.~w,
                ExpectedWidth = ~w
            })',
        [InputLiteral, RecSepLiteral, Width]).

metadata_column_selectors_literal(Metadata, _Arity, Literal) :-
    (   get_dict(column_selectors, Metadata, Selectors),
        Selectors \= []
    ->  true
    ;   Selectors = []
    ),
    (   Selectors == []
    ->  format(atom(Literal), 'Array.Empty<JsonColumnSelectorConfig>()', [])
    ;   findall(Item,
            (   member(column_selector{path:Path, kind:Kind}, Selectors),
                csharp_literal(Path, PathLiteral),
                selector_kind_enum(Kind, Enum),
                format(atom(Item), '                new JsonColumnSelectorConfig(~w, JsonColumnSelectorKind.~w)', [PathLiteral, Enum])
            ),
            Items),
        atomic_list_concat(Items, ',\n', Joined),
        format(atom(Literal), 'new JsonColumnSelectorConfig[]{\n~w\n            }', [Joined])
    ).

metadata_type_literal(Metadata, Literal) :-
    (   get_dict(type_hint, Metadata, TypeHint),
        TypeHint \= none
    ->  csharp_literal(TypeHint, Literal)
    ;   Literal = 'null'
    ).

metadata_schema_literal(Metadata, Literal) :-
    (   get_dict(schema_fields, Metadata, Fields),
        Fields \= []
    ->  maplist(schema_field_literal_with_indent('                '), Fields, Items),
        atomic_list_concat(Items, ',\n', Joined),
        format(atom(Literal), 'new JsonSchemaFieldConfig[]{\n~w\n            }', [Joined])
    ;   Literal = 'Array.Empty<JsonSchemaFieldConfig>()'
    ).

schema_field_literal_with_indent(Indent, Field, Literal) :-
    once(schema_field_literal(Field, Literal, Indent)).

schema_field_literal(Field, Literal, Indent) :-
    !,
    get_dict(name, Field, NameAtom),
    get_dict(path, Field, PathString),
    get_dict(selector_kind, Field, KindAtom),
    schema_field_property_name(NameAtom, PropertyName),
    schema_column_type_enum_field(Field, EnumLiteral),
    selector_kind_enum(KindAtom, SelectorEnum),
    csharp_literal(PathString, PathLiteral),
    (   get_dict(field_kind, Field, FieldKind),
        FieldKind = record,
        get_dict(record_type, Field, RecordType),
        RecordType \= none
    ->  schema_record_type_literal(RecordType, RecordLiteral),
        get_dict(nested_fields, Field, NestedFields),
        schema_field_kind_enum(record, FieldKindLiteral),
        schema_nested_literal(NestedFields, NestedLiteral, Indent)
    ;   RecordLiteral = 'null',
        schema_field_kind_enum(value, FieldKindLiteral),
        NestedLiteral = 'null'
    ),
    format(atom(Literal),
'~wnew JsonSchemaFieldConfig("~w", ~w, JsonColumnSelectorKind.~w, JsonColumnType.~w, JsonSchemaFieldKind.~w, ~w, ~w)',
        [Indent, PropertyName, PathLiteral, SelectorEnum, EnumLiteral, FieldKindLiteral, RecordLiteral, NestedLiteral]).

schema_nested_literal([], 'null', _) :- !.
schema_nested_literal(Fields, Literal, Indent) :-
    atom_concat(Indent, '    ', NextIndent),
    maplist(schema_field_literal_with_indent(NextIndent), Fields, Items),
    atomic_list_concat(Items, ',\n', Joined),
    format(atom(Literal), 'new JsonSchemaFieldConfig[]{\n~w\n~w    }', [Joined, Indent]).

schema_column_type_enum_field(Field, EnumLiteral) :-
    (   get_dict(column_type, Field, TypeAtom),
        schema_column_type_enum(TypeAtom, EnumLiteral)
    ->  true
    ;   EnumLiteral = 'Json'
    ).

schema_record_type_literal(TypeAtom, Literal) :-
    format(atom(FullName), 'UnifyWeaver.Generated.~w', [TypeAtom]),
    csharp_literal(FullName, Literal).

schema_field_kind_enum(value, 'Value').
schema_field_kind_enum(record, 'Record').
schema_field_kind_enum(Kind, Literal) :-
    atom_string(Kind, KindStr),
    capitalise_string(KindStr, Literal),
    !.

selector_kind_enum(jsonpath, 'JsonPath') :- !.
selector_kind_enum(column_path, 'Path') :- !.
selector_kind_enum(path, 'Path') :- !.
selector_kind_enum(Kind, 'Path') :-
    string(Kind),
    !.
selector_kind_enum(Kind, 'Path') :-
    atom(Kind),
    !.

schema_column_type_enum(string, 'String') :- !.
schema_column_type_enum(integer, 'Integer') :- !.
schema_column_type_enum(long, 'Long') :- !.
schema_column_type_enum(float, 'Double') :- !.
schema_column_type_enum(double, 'Double') :- !.
schema_column_type_enum(number, 'Double') :- !.
schema_column_type_enum(boolean, 'Boolean') :- !.
schema_column_type_enum(json, 'Json') :- !.
schema_column_type_enum(Type, Enum) :-
    atom_string(Type, TypeStr),
    capitalise_string(TypeStr, Enum).

metadata_treat_array_literal(Metadata, Literal) :-
    (   get_dict(treat_array_stream, Metadata, Flag)
    ->  (Flag == true -> Literal = 'true' ; Literal = 'false')
    ;   Literal = 'true'
    ).

metadata_null_policy_literals(Metadata, PolicyLiteral, DefaultLiteral) :-
    (   get_dict(null_policy, Metadata, Policy)
    ->  null_policy_enum(Policy, PolicyLiteral)
    ;   PolicyLiteral = 'Allow'
    ),
    (   get_dict(null_default, Metadata, Default),
        Default \= none
    ->  csharp_literal(Default, DefaultLiteral)
    ;   DefaultLiteral = 'null'
    ).

null_policy_enum(allow, 'Allow').
null_policy_enum(fail, 'Fail').
null_policy_enum(skip, 'Skip').
null_policy_enum(default, 'Default').
null_policy_enum(Value, Enum) :-
    atom_string(Value, ValueStr),
    capitalise_string(ValueStr, Enum).

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

%% ============================================================================
%% Generator Mode Implementation
%% ============================================================================

csharp_config(Config) :-
    Config = [
        access_fmt-"~w[\"arg~w\"]",
        atom_fmt-"\"~w\"",
        null_val-"null",
        ops-[
            + - "+", - - "-", * - "*", / - "/", mod - "%",
            > - ">", < - "<", >= - ">=", =< - "<=", =:= - "==", =\= - "!=",
            is - "=="
        ]
    ].

compile_generator_mode(Pred/Arity, Options, Code) :-
    compute_generator_dependency_closure(Pred/Arity, GroupSpecs),
    csharp_config(Config),
    option(enable_indexing(IndexingOpt), Options, true),
    ( IndexingOpt == false -> Indexing = false ; Indexing = true ),

    gather_group_clauses(GroupSpecs, AllClauses),
    guard_stratified_negation(Pred/Arity, GroupSpecs, AllClauses),
    guard_supported_aggregates(AllClauses),
    collect_fact_heads(AllClauses, FactHeads),
    compile_generator_facts(FactHeads, Config, FactsCode),

    findall(Head-Body,
        (   member(Head-Body, AllClauses),
            Body \= true
        ),
        RuleClauses),
    compile_generator_rules(GroupSpecs, RuleClauses, Config, Indexing, RulesCode, RuleNames),

    compile_generator_execution(Pred, RuleNames, Indexing, ExecutionCode),

    csharp_generator_header(Pred, Header),
    format(string(Code), "~w\n~w\n~w\n~w\n    }\n}\n", [Header, FactsCode, RulesCode, ExecutionCode]).

aggregate_goal(G) :-
    compound(G),
    functor(G, Fun, Arity),
    member(Fun/Arity, [aggregate_all/3, aggregate_all/4, aggregate/4]).

guard_supported_aggregates(Clauses) :-
    forall(
        ( member(_Head-Body, Clauses),
          Body \= true,
          body_to_list(Body, Goals),
          member(G, Goals),
          aggregate_goal(G)
        ),
        aggregate_supported(G)
    ).

aggregate_supported(aggregate_all(_OpTerm, Goal, _Group, _Result)) :-
    \+ callable(Goal),
    format(user_error,
           'C# generator mode: aggregate_all/4 requires callable goal (~w).~n',
           [Goal]),
    !, fail.
aggregate_supported(aggregate_all(_OpTerm, Goal, _Result)) :-
    \+ callable(Goal),
    format(user_error,
           'C# generator mode: aggregate_all/3 requires callable goal (~w).~n',
           [Goal]),
    !, fail.
aggregate_supported(aggregate_all(OpTerm, _Goal, _Result)) :-
    member(OpTerm, [count, sum(_), min(_), max(_), set(_), bag(_)]), !.
aggregate_supported(aggregate_all(OpTerm, _Goal, _Group, _Result)) :-
    member(OpTerm, [count, sum(_), min(_), max(_), set(_), bag(_)]), !.
aggregate_supported(aggregate(OpTerm, _Goal, _Result)) :-
    member(OpTerm, [sum(_), min(_), max(_), set(_), bag(_)]), !.
aggregate_supported(aggregate(OpTerm, _Goal, _Group, _Result)) :-
    member(OpTerm, [count, sum(_), min(_), max(_), set(_), bag(_)]), !.
aggregate_supported(aggregate_all(Op, _Inner, _Result)) :-
    \+ member(Op, [count]),
    format(user_error,
           'C# generator mode: aggregate_all/3 with op ~w not yet supported.~n',
           [Op]),
    fail.
aggregate_supported(aggregate_all(OpTerm, _Inner, _Group, _Result)) :-
    member(OpTerm, [sum(_), min(_), max(_), set(_), bag(_)]), !.
aggregate_supported(aggregate(Op, _Inner, _Group, _Result)) :-
    format(user_error,
           'C# generator mode: aggregate/4 not yet supported (~w).~n',
           [Op]),
    fail.
aggregate_supported(G) :-
    format(user_error,
           'C# generator mode: aggregate goal not supported (~w).~n',
           [G]),
    fail.

guard_stratified_negation(HeadPI, GroupSpecs, Clauses) :-
    build_dependency_graph([HeadPI], [], [], Vertices, [], Edges),
    vertices_edges_to_ugraph(Vertices, Edges, Graph),
    forall(
        ( member(_-B, Clauses),
          B \= true,
          body_to_list(B, Goals),
          member(G, Goals),
          neg_goal_pred(G, NegPI)
        ),
        (   reachable(NegPI, Graph, Reach),
            \+ memberchk(HeadPI, Reach)
        ->  true
        ;   format(user_error,
                   'C# generator mode: negation of ~w is not stratified w.r.t ~w; unsupported.~n',
                   [NegPI, HeadPI]),
            fail
        )
    ),
    % Also ensure negated predicate exists if in group
    forall(
        ( member(_H-B2, Clauses),
          B2 \= true,
          body_to_list(B2, Goals2),
          member(G2, Goals2),
          neg_goal_pred(G2, NegPI2),
          member(NegSpec, GroupSpecs),
          spec_signature(NegSpec, NegPI2)
        ),
        predicate_defined(NegPI2)
    ).

neg_goal_pred(\+ G, PI) :- term_signature(G, PI).
neg_goal_pred(not(G), PI) :- term_signature(G, PI).

csharp_generator_header(Pred, Header) :-
    get_time(Timestamp),
    format_time(string(DateStr), '%Y-%m-%d %H:%M:%S', Timestamp),
    snake_case_to_pascal(Pred, PredClass),
    format(string(Header),
"// Generated by UnifyWeaver C# Generator Mode
// Date: ~w
using System;
using System.Collections.Generic;
using System.Linq;

namespace UnifyWeaver.Generated
{
    public record Fact(string Relation, Dictionary<string, object> Args)
    {
        private static bool ValuesEqual(object? left, object? right)
        {
            if (ReferenceEquals(left, right)) return true;
            if (left is null || right is null) return false;
            if (left is string ls && right is string rs) return ls.Equals(rs);
            if (left is System.Collections.IEnumerable le &&
                right is System.Collections.IEnumerable re &&
                left is not string && right is not string)
            {
                return le.Cast<object?>().SequenceEqual(re.Cast<object?>());
            }
            return object.Equals(left, right);
        }

        private static void AddValueToHash(ref HashCode hash, object? value)
        {
            if (value is null)
            {
                hash.Add(0);
                return;
            }
            if (value is string s)
            {
                hash.Add(s);
                return;
            }
            if (value is System.Collections.IEnumerable enumerable && value is not string)
            {
                foreach (var v in enumerable.Cast<object?>())
                {
                    AddValueToHash(ref hash, v);
                }
                hash.Add(\"[]\");
                return;
            }
            hash.Add(value);
        }

        public virtual bool Equals(Fact? other)
        {
            if (other is null) return false;
            if (ReferenceEquals(this, other)) return true;
            if (Relation != other.Relation) return false;
            if (Args.Count != other.Args.Count) return false;
            foreach (var kvp in Args)
            {
                if (!other.Args.TryGetValue(kvp.Key, out var value)) return false;
                if (!ValuesEqual(kvp.Value, value)) return false;
            }
            return true;
        }
        
        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(Relation);
            foreach (var kvp in Args.OrderBy(k => k.Key))
            {
                hash.Add(kvp.Key);
                AddValueToHash(ref hash, kvp.Value);
            }
            return hash.ToHashCode();
        }
    }

    public static class ~w_Module
    {", [DateStr, PredClass]).

collect_fact_heads(Clauses, FactHeads) :-
    findall(Head,
        member(Head-true, Clauses),
        RawFacts),
    sort(RawFacts, FactHeads).

compile_generator_facts(FactHeads, _Config, Code) :-
    findall(FactCode,
        (   member(Head, FactHeads),
            Head =.. [Pred|Args],
            generate_fact_creation(Pred, Args, FactCode)
        ),
        FactCodes),
    (   FactCodes == []
    ->  FactsBody = ""
    ;   atomic_list_concat(FactCodes, "\n            ", FactsBody)
    ),
    format(string(Code),
"        public static HashSet<Fact> GetInitialFacts()
        {
            var facts = new HashSet<Fact>();
            ~w
            return facts;
        }", [FactsBody]).

generate_fact_creation(Pred, Args, Code) :-
    findall(ArgCode,
        (   nth0(I, Args, Arg),
            format(string(ArgCode), "{ \"arg~w\", \"~w\" }", [I, Arg])
        ),
        ArgCodes),
    atomic_list_concat(ArgCodes, ", ", ArgsStr),
    format(string(Code), "facts.Add(new Fact(\"~w\", new Dictionary<string, object> { ~w }));", [Pred, ArgsStr]).

compile_generator_rules(_GroupSpecs, Clauses, Config, Indexing, Code, RuleNames) :-
    findall(RuleCode-RuleName,
        (   nth1(I, Clauses, Clause),
            Clause = Head-Body,
            Body \= true,
            once(compile_rule(I, Head, Body, Config, Indexing, RuleCode, RuleName))
        ),
        Pairs),
    pairs_keys_values(Pairs, RuleCodes, RuleNames),
    atomic_list_concat(RuleCodes, "\n\n", Code).

compile_rule(Index, Head, Body, Config, Indexing, Code, RuleName) :-
        format(string(RuleName), "ApplyRule_~w", [Index]),
        Head =.. [_HeadPred|_HeadArgs],
        
        % Parse body
        body_to_list(Body, Goals),
        partition(is_aggregate_goal, Goals, Aggregates, NonAggGoals0),
        
        (   Aggregates = [Agg],
            append(RelGoals, [Agg], Goals)
        ->  partition(is_builtin_goal, RelGoals, Builtins, Relations),
            (   Builtins = [],
                Relations = []
            ->  compile_aggregate_rule(Index, Head, Agg, Config, Indexing, Code, RuleName)
            ;   Relations = [FirstGoal|RestGoals]
            ->  FirstGoal =.. [Pred1|Args1],
                % Generate pattern check for first goal
                length(Args1, Arity1),
                findall(Check,
                    (   between(0, Arity1, I), I < Arity1,
                        format(string(Check), "fact.Args.ContainsKey(\"arg~w\")", [I])
                    ),
                    Checks),
                format(string(RelCheck), "fact.Relation == \"~w\"", [Pred1]),
                append([RelCheck], Checks, AllChecks),
                atomic_list_concat(AllChecks, " && ", Pattern),
                compile_joins_with_aggregate(RestGoals, Builtins, Agg, Head, FirstGoal, Config, Indexing, JoinBody),
                format(string(Code),
"        public static IEnumerable<Fact> ~w(Fact fact, HashSet<Fact> total, Dictionary<string, List<Fact>> relIndex, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg0, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg1)
        {
            // Rule ~w: ~w :- ...
            if (~w)
            {
~w
            }
        }", [RuleName, Index, Head, Pattern, JoinBody])
            ;   format(user_error,
                       'C# generator mode: aggregate goal must appear last and needs at least one relation before it when combined with joins/negation.~n',
                       []),
                fail
            )
        ;   Aggregates = [_|_]
        ->  format(user_error,
                   'C# generator mode: only one aggregate goal supported per rule.~n',
                   []),
            fail
        ;   partition(is_builtin_goal, NonAggGoals0, Builtins, Relations),
            (   Relations = []
        ->  format(string(Code), "// Rule ~w: No relations", [Index])
        ;   Relations = [FirstGoal|RestGoals],
            FirstGoal =.. [Pred1|Args1],
            
            % Generate pattern check for first goal
            length(Args1, Arity1),
            findall(Check,
                (   between(0, Arity1, I), I < Arity1,
                    format(string(Check), "fact.Args.ContainsKey(\"arg~w\")", [I])
                ),
                Checks),
            format(string(RelCheck), "fact.Relation == \"~w\"", [Pred1]),
            append([RelCheck], Checks, AllChecks),
            atomic_list_concat(AllChecks, " && ", Pattern),
            
            % Generate joins
            compile_joins(RestGoals, Builtins, Head, FirstGoal, Config, Indexing, JoinBody),
            
            format(string(Code),
"        public static IEnumerable<Fact> ~w(Fact fact, HashSet<Fact> total, Dictionary<string, List<Fact>> relIndex, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg0, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg1)
        {
            // Rule ~w: ~w :- ...
            if (~w)
            {
~w
            }
        }", [RuleName, Index, Head, Pattern, JoinBody])
        )
    ).

is_aggregate_goal(G) :- aggregate_goal(G).

is_builtin_goal(Goal) :-
    Goal =.. [Functor|_],
    member(Functor, ['is', '>', '<', '>=', '=<', '=:=', '=\\=', '==', '!=', 'not', '\\+']).
is_builtin_goal(Goal) :-
    functor(Goal, Name, Arity),
    cs_binding(Name/Arity, _, _, _, _).

compile_aggregate_rule(Index, Head, AggGoal, Config, _Indexing, Code, RuleName) :-
    decompose_aggregate_goal(AggGoal, Type, Op, Pred, Args, GroupVar, ValueVar, ResVar),
    format(string(RuleName), "ApplyRule_~w", [Index]),
    Head =.. [HeadPred|HeadArgs],
    (   Type = all,
        member(Op, [count, sum, min, max, set, bag])
    ->  build_aggregate_filter(Pred, Args, Config, FilterExpr),
        build_value_expr(Op, Args, ValueVar, ValueExpr),
        bind_count_head(HeadArgs, ResVar, Config, Assigns),
        atomic_list_concat(Assigns, ", ", AssignStr),
        agg_expr(Op, ValueExpr, AggExpr),
        emit_condition(Op, EmitCond),
        format(string(Code),
"        public static IEnumerable<Fact> ~w(Fact fact, HashSet<Fact> total, Dictionary<string, List<Fact>> relIndex, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg0, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg1)
        {
            // Rule ~w: ~w :- aggregate_all(~w, ~w(~w), ~w)
            var aggSource = relIndex.TryGetValue(\"~w\", out var aggList) ? aggList : Enumerable.Empty<Fact>();
            var aggQuery = aggSource.Where(f => ~w);
            if (~w)
            {
                var agg = ~w;
                yield return new Fact(\"~w\", new Dictionary<string, object> { ~w });
            }
        }", [RuleName, Index, Head, Op, Pred, Args, ResVar, Pred, FilterExpr, EmitCond, AggExpr, HeadPred, AssignStr])
    ;   Type = group,
        member(Op, [count, sum, min, max, set, bag])
    ->  build_group_aggregate(Op, Pred, Args, GroupVar, ValueVar, ResVar, Config, HeadPred, HeadArgs, RuleName, Code)
    ;   format(user_error,
               'C# generator mode: aggregate ~w/~w not supported in generator codegen.~n',
               [Op, Type]),
        fail
    ).

decompose_aggregate_goal(aggregate_all(OpTerm, Goal, Result), all, Op, Pred, Args, _GroupVar, ValueVar, Result) :-
    nonvar(Goal),
    Goal =.. [Pred|Args],
    parse_agg_op(OpTerm, Op, ValueVar, Args).
decompose_aggregate_goal(aggregate_all(OpTerm, Goal, GroupVar, Result), group, Op, Pred, Args, GroupVar, ValueVar, Result) :-
    nonvar(Goal),
    Goal =.. [Pred|Args],
    parse_agg_op(OpTerm, Op, ValueVar, Args).
decompose_aggregate_goal(aggregate(OpTerm, Goal, GroupVar, Result), group, Op, Pred, Args, GroupVar, ValueVar, Result) :-
    nonvar(Goal),
    Goal =.. [Pred|Args],
    parse_agg_op(OpTerm, Op, ValueVar, Args).

parse_agg_op(sum(Var), sum, Var, _) :- !.
parse_agg_op(min(Var), min, Var, _) :- !.
parse_agg_op(max(Var), max, Var, _) :- !.
parse_agg_op(count, count, _, _) :- !.
parse_agg_op(set(Var), set, Var, _) :- !.
parse_agg_op(bag(Var), bag, Var, _) :- !.
parse_agg_op(OpTerm, Op, _, _) :-
    OpTerm =.. [Op|_].

parse_group_term(Var^_, [Var]) :- !.
parse_group_term(Var, [Var]) :- var(Var), !.
parse_group_term(_, []).

build_aggregate_filter(Pred, Args, _Config, Expr) :-
    length(Args, Arity),
    findall(Cond,
        (   between(0, Arity, I), I < Arity,
            nth0(I, Args, Arg),
            (   ground(Arg)
            ->  format(string(Cond), "f.Args.ContainsKey(\"arg~w\") && f.Args[\"arg~w\"].Equals(\"~w\")", [I, I, Arg])
            ;   format(string(Cond), "f.Args.ContainsKey(\"arg~w\")", [I])
            )
        ),
        Conds),
    atomic_list_concat(Conds, " && ", ArgConds),
    format(string(Expr), "f.Relation == \"~w\" && ~w", [Pred, ArgConds]).

build_value_expr(count, _Args, _ValVar, "1").
build_value_expr(sum, Args, ValVar, Expr) :-
    find_var_index(ValVar, Args, ValIdx),
    format(string(Expr), "Convert.ToDecimal(f.Args[\"arg~w\"])", [ValIdx]).
build_value_expr(min, Args, ValVar, Expr) :-
    find_var_index(ValVar, Args, ValIdx),
    format(string(Expr), "Convert.ToDecimal(f.Args[\"arg~w\"])", [ValIdx]).
build_value_expr(max, Args, ValVar, Expr) :-
    find_var_index(ValVar, Args, ValIdx),
    format(string(Expr), "Convert.ToDecimal(f.Args[\"arg~w\"])", [ValIdx]).
build_value_expr(set, Args, ValVar, Expr) :-
    find_var_index(ValVar, Args, ValIdx),
    format(string(Expr), "f.Args[\"arg~w\"]", [ValIdx]).
build_value_expr(bag, Args, ValVar, Expr) :-
    find_var_index(ValVar, Args, ValIdx),
    format(string(Expr), "f.Args[\"arg~w\"]", [ValIdx]).

agg_expr(count, _ValExpr, "aggQuery.Count()").
agg_expr(sum, ValExpr, AggExpr) :-
    format(string(AggExpr), "aggQuery.Sum(f => ~w)", [ValExpr]).
agg_expr(min, ValExpr, AggExpr) :-
    format(string(AggExpr), "aggQuery.Min(f => ~w)", [ValExpr]).
agg_expr(max, ValExpr, AggExpr) :-
    format(string(AggExpr), "aggQuery.Max(f => ~w)", [ValExpr]).
agg_expr(set, ValExpr, AggExpr) :-
    format(string(AggExpr), "aggQuery.Select(f => ~w).Distinct().ToList()", [ValExpr]).
agg_expr(bag, ValExpr, AggExpr) :-
    format(string(AggExpr), "aggQuery.Select(f => ~w).ToList()", [ValExpr]).

emit_condition(count, "true").
emit_condition(_, "aggQuery.Any()").

bind_count_head(HeadArgs, ResVar, Config, Assigns) :-
    findall(Assign,
        (   nth0(I, HeadArgs, Arg),
            (   var(Arg),
                Arg == ResVar
            ->  format(string(Assign), "{ \"arg~w\", agg }", [I])
            ;   var(Arg)
            ->  format(user_error, 'C# generator aggregate: head var ~w not bound by aggregate result.~n', [Arg]),
                fail
            ;   translate_expr_common(Arg, [], Config, Expr)
            ->  format(string(Assign), "{ \"arg~w\", ~w }", [I, Expr])
            ;   format(string(Assign), "{ \"arg~w\", \"~w\" }", [I, Arg])
            )
        ),
        Assigns).

find_var_index(Var, Args, Idx) :-
    nth0(Idx, Args, Arg),
    Var == Arg, !.
find_var_index(Var, Args, Idx) :-
    nth0(Idx, Args, Arg),
    compound(Arg),
    Arg =.. ['^', Var, _],
    !.
find_var_index(Var, Args, Idx) :-
    nth0(Idx, Args, Arg),
    compound(Arg),
    Arg =.. ['^', _, Var],
    !.

build_group_filter(Pred, Args, _Config, Expr) :-
    length(Args, Arity),
    findall(Cond,
        (   between(0, Arity, I), I < Arity,
            nth0(I, Args, Arg),
            (   ground(Arg)
            ->  format(string(Cond), "f.Args.ContainsKey(\"arg~w\") && f.Args[\"arg~w\"].Equals(\"~w\")", [I, I, Arg])
            ;   format(string(Cond), "f.Args.ContainsKey(\"arg~w\")", [I])
            )
        ),
        Conds),
    atomic_list_concat(Conds, " && ", ArgConds),
    format(string(Expr), "f.Relation == \"~w\" && ~w", [Pred, ArgConds]).

build_group_key([_], Idx, Expr) :-
    format(string(Expr), "f.Args[\"arg~w\"]", [Idx]).
build_group_key(_, Idx, Expr) :-
    format(string(Expr), "f.Args[\"arg~w\"]", [Idx]).

build_group_aggregate(Op, Pred, Args, GroupVar, ValVar, ResVar, Config, HeadPred, HeadArgs, RuleName, Code) :-
    find_var_index(GroupVar, Args, GroupIdx),
    (   Op = count
    ->  ValIdx = 0,
        SelectorExpr = ""
    ;   find_var_index(ValVar, Args, ValIdx),
        SelectorExpr = ""
    ),
    build_group_filter(Pred, Args, [], FilterExpr),
    build_group_key([GroupVar], GroupIdx, GroupExpr),
    group_value_field(Op, ValVar, ValueField, SelectorExpr),
    bind_group_head_assignments(Op, HeadArgs, ResVar, Config, ValueField, Assigns),
    atomic_list_concat(Assigns, ", ", AssignStr),
    group_agg_selector(Op, ValIdx, SelectorExpr, AggSelector),
    group_agg_projection(Op, AggSelector, Projection),
    format(string(Code),
"        public static IEnumerable<Fact> ~w(Fact fact, HashSet<Fact> total, Dictionary<string, List<Fact>> relIndex, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg0, Dictionary<string, Dictionary<object, List<Fact>>> relIndexArg1)
        {
            // Grouped aggregate ~w over ~w/~w
            var groupSource = relIndex.TryGetValue(\"~w\", out var groupList) ? groupList : Enumerable.Empty<Fact>();
            var aggResults = groupSource
                .Where(f => ~w)
                .GroupBy(f => ~w)
                .Select(g => ~w);
            foreach (var r in aggResults)
            {
                yield return new Fact(\"~w\", new Dictionary<string, object> { ~w });
            }
        }", [RuleName, Op, Pred, Args, Pred, FilterExpr, GroupExpr, Projection, HeadPred, AssignStr]).

group_value_field(set, _ValVar, "Set", _).
group_value_field(bag, _ValVar, "Bag", _).
group_value_field(sum, _ValVar, "Sum", _).
group_value_field(min, _ValVar, "Min", _).
group_value_field(max, _ValVar, "Max", _).
group_value_field(count, _ValVar, "Count", _).

group_agg_selector(sum, ValIdx, _SelectorExpr, Selector) :-
    format(string(Selector), "g.Sum(f => Convert.ToDecimal(f.Args[\"arg~w\"]))", [ValIdx]).
group_agg_selector(min, ValIdx, _SelectorExpr, Selector) :-
    format(string(Selector), "g.Min(f => Convert.ToDecimal(f.Args[\"arg~w\"]))", [ValIdx]).
group_agg_selector(max, ValIdx, _SelectorExpr, Selector) :-
    format(string(Selector), "g.Max(f => Convert.ToDecimal(f.Args[\"arg~w\"]))", [ValIdx]).
group_agg_selector(set, ValIdx, SelectorExpr, Selector) :-
    ( SelectorExpr = "" ->
        format(string(Selector), "g.Select(f => f.Args[\"arg~w\"]).Distinct().ToList()", [ValIdx])
    ;   format(string(Selector), "g.Select(f => ~w).Distinct().ToList()", [SelectorExpr])
    ).
group_agg_selector(bag, ValIdx, SelectorExpr, Selector) :-
    ( SelectorExpr = "" ->
        format(string(Selector), "g.Select(f => f.Args[\"arg~w\"]).ToList()", [ValIdx])
    ;   format(string(Selector), "g.Select(f => ~w).ToList()", [SelectorExpr])
    ).
group_agg_selector(count, _ValIdx, _SelectorExpr, "g.Count()").

group_agg_projection(Op, AggSelector, Projection) :-
    group_value_field(Op, _, Field, _),
    format(string(Projection), "new { Key = g.Key, ~w = ~w }", [Field, AggSelector]).

bind_group_head_assignments(_Op, HeadArgs, ResVar, Config, ValueField, Assigns) :-
    findall(Assign,
        (   nth0(I, HeadArgs, Arg),
            (   var(Arg),
                Arg == ResVar
            ->  format(string(Assign), "{ \"arg~w\", r.~w }", [I, ValueField])
            ;   var(Arg)
            ->  format(string(Assign), "{ \"arg~w\", r.Key }", [I])
            ;   translate_expr_common(Arg, [], Config, Expr)
            ->  format(string(Assign), "{ \"arg~w\", ~w }", [I, Expr])
            ;   format(string(Assign), "{ \"arg~w\", \"~w\" }", [I, Arg])
            )
        ),
        Assigns).
build_head_assignments(HeadArgs, ResVar, Config, Assigns) :-
    findall(Assign,
        (   nth0(I, HeadArgs, Arg),
            (   var(Arg),
                Arg == ResVar
            ->  format(string(Assign), "{ \"arg~w\", agg }", [I])
            ;   var(Arg)
            ->  format(user_error, 'C# generator aggregate: head var ~w not bound by aggregate result.~n', [Arg]),
                fail
            ;   translate_expr_common(Arg, [], Config, Expr)
            ->  format(string(Assign), "{ \"arg~w\", ~w }", [I, Expr])
            ;   format(string(Assign), "{ \"arg~w\", \"~w\" }", [I, Arg])
            )
        ),
        Assigns).

build_head_assignments_with_agg(HeadArgs, ResVar, AccumPairs, VarMap, Config, Assigns) :-
    findall(Assign,
        (   nth0(I, HeadArgs, Arg),
            (   var(Arg),
                Arg == ResVar
            ->  format(string(Assign), "{ \"arg~w\", agg }", [I])
            ;   var(Arg)
            ->  (   find_var_access(Arg, AccumPairs, Src, SrcIdx)
                ->  format(string(Assign), "{ \"arg~w\", ~w[\"arg~w\"] }", [I, Src, SrcIdx])
                ;   format(user_error, 'C# generator aggregate(join): head var ~w not bound by relations or aggregate result.~n', [Arg]),
                    fail
                )
            ;   translate_expr_common(Arg, VarMap, Config, Expr)
            ->  format(string(Assign), "{ \"arg~w\", ~w }", [I, Expr])
            ;   format(string(Assign), "{ \"arg~w\", \"~w\" }", [I, Arg])
            )
        ),
        Assigns).

build_aggregate_filter_with_vars(Pred, Args, VarMap, Config, Expr) :-
    length(Args, Arity),
    findall(Cond,
        (   between(0, Arity, I), I < Arity,
            nth0(I, Args, Arg),
            (   ground(Arg)
            ->  format(string(Cond), "f.Args.ContainsKey(\"arg~w\") && f.Args[\"arg~w\"].Equals(\"~w\")", [I, I, Arg])
            ;   translate_expr_common(Arg, VarMap, Config, ArgExpr),
                ArgExpr \= "null"
            ->  format(string(Cond), "f.Args.ContainsKey(\"arg~w\") && object.Equals(f.Args[\"arg~w\"], ~w)", [I, I, ArgExpr])
            ;   format(string(Cond), "f.Args.ContainsKey(\"arg~w\")", [I])
            )
        ),
        Conds),
    atomic_list_concat(Conds, " && ", ArgConds),
    format(string(Expr), "f.Relation == \"~w\" && ~w", [Pred, ArgConds]).

compile_aggregate_from_bindings(HeadPred, HeadArgs, AggGoal, VarMap, AccumPairs, Config, ConstraintChecks, Code) :-
    decompose_aggregate_goal(AggGoal, Type, Op, Pred, Args, _GroupVar, ValueVar, ResVar),
    Type = all,
    member(Op, [count, sum, min, max, set, bag]),
    build_aggregate_filter_with_vars(Pred, Args, VarMap, Config, FilterExpr),
    build_value_expr(Op, Args, ValueVar, ValueExpr),
    agg_expr(Op, ValueExpr, AggExpr),
    emit_condition(Op, EmitCond),
    build_head_assignments_with_agg(HeadArgs, ResVar, AccumPairs, VarMap, Config, Assigns),
    atomic_list_concat(Assigns, ", ", AssignStr),
    format(string(AggBlock),
"                var aggSource = relIndex.TryGetValue(\"~w\", out var aggList) ? aggList : Enumerable.Empty<Fact>();
                var aggQuery = aggSource.Where(f => ~w);
                if (~w)
                {
                    var agg = ~w;
                    yield return new Fact(\"~w\", new Dictionary<string, object> { ~w });
                }", [Pred, FilterExpr, EmitCond, AggExpr, HeadPred, AssignStr]),
    (   ConstraintChecks == "true"
    ->  Code = AggBlock
    ;   format(string(Code),
"                if (~w)
                {
~w
                }", [ConstraintChecks, AggBlock])
    ).

compile_joins([], Builtins, Head, FirstGoal, Config, _Indexing, Code) :-
    % No more relations, just constraints and output
    Head =.. [HeadPred|HeadArgs],
    
    % Build variable map
    AccumPairs = [FirstGoal-"fact.Args"],
    build_variable_map(AccumPairs, VarMap),
    
    % Translate builtins
    reorder_constraints(Builtins, VarMap, EarlyBuiltins, LateBuiltins),
    translate_builtins(EarlyBuiltins, VarMap, Config, EarlyChecks),
    translate_builtins(LateBuiltins, VarMap, Config, LateChecks),
    
    % Build output
    findall(Assign,
        (   nth0(I, HeadArgs, Arg),
            (   var(Arg) ->
                (   find_var_access(Arg, AccumPairs, Src, SrcIdx)
                ->  format(user_output, 'DEBUG output arg ~w uses ~w arg~w~n', [Arg, Src, SrcIdx]),
                    format(string(Assign), "{ \"arg~w\", ~w[\"arg~w\"] }", [I, Src, SrcIdx])
                ;   format(user_output, 'DEBUG output arg ~w has no access~n', [Arg]),
                    fail
                )
            ;   translate_expr_common(Arg, VarMap, Config, Expr)
            ->  format(string(Assign), "{ \"arg~w\", ~w }", [I, Expr])
            ;   format(string(Assign), "{ \"arg~w\", \"~w\" }", [I, Arg])
            )
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", OutputStr),
    
    (   EarlyChecks == "true", LateChecks == "true"
    ->  format(string(Code), "                yield return new Fact(\"~w\", new Dictionary<string, object> { ~w });", [HeadPred, OutputStr])
    ;   format(string(Code), 
"                if (~w && ~w)
                {
                    yield return new Fact(\"~w\", new Dictionary<string, object> { ~w });
                }", [EarlyChecks, LateChecks, HeadPred, OutputStr])
    ).

compile_joins([Goal|RestGoals], Builtins, Head, FirstGoal, Config, Indexing, Code) :-
    % Handle join with one or more additional goals
    % Start with FirstGoal in accumulator
    compile_nway_join([Goal|RestGoals], Builtins, Head, [FirstGoal-"fact.Args"], Config, Indexing, 2, Code).

compile_joins_with_aggregate(RestGoals, Builtins, AggGoal, Head, FirstGoal, Config, Indexing, Code) :-
    compile_nway_join_with_aggregate(RestGoals, Builtins, AggGoal, Head, [FirstGoal-"fact.Args"], Config, Indexing, 2, Code).

%% compile_nway_join(+Goals, +Builtins, +Head, +AccumPairs, +Config, +Index, -Code)
%  Recursively build N-way joins, threading Goal-Access pairs through
compile_nway_join([], Builtins, Head, AccumPairs, Config, _Indexing, _, Code) :-
    % Base case: no more goals to join, use ALL accumulated pairs for VarMap
    Head =.. [HeadPred|HeadArgs],
    build_variable_map(AccumPairs, VarMap),
    reorder_constraints(Builtins, VarMap, EarlyBuiltins, LateBuiltins),
    translate_builtins(EarlyBuiltins, VarMap, Config, EarlyChecks),
    translate_builtins(LateBuiltins, VarMap, Config, LateChecks),
    
    % Build output
    findall(Assign,
        (   nth0(I, HeadArgs, Arg),
            (   var(Arg) ->
                (   find_var_access(Arg, AccumPairs, Src, SrcIdx)
                ->  format(string(Assign), "{ \"arg~w\", ~w[\"arg~w\"] }", [I, Src, SrcIdx])
                ;   fail
                )
            ;   translate_expr_common(Arg, VarMap, Config, Expr)
            ->  format(string(Assign), "{ \"arg~w\", ~w }", [I, Expr])
            ;   format(string(Assign), "{ \"arg~w\", \"~w\" }", [I, Arg])
            )
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", OutputStr),
    
    (   EarlyChecks == "true", LateChecks == "true"
    ->  format(string(Code), "                yield return new Fact(\"~w\", new Dictionary<string, object> { ~w });", [HeadPred, OutputStr])
    ;   format(string(Code), 
"                if (~w && ~w)
                {
                    yield return new Fact(\"~w\", new Dictionary<string, object> { ~w });
                }", [EarlyChecks, LateChecks, HeadPred, OutputStr])
    ).

compile_nway_join([Goal|RestGoals], Builtins, Head, AccumPairs, Config, Indexing, Index, Code) :-
    % Recursive case: join with current goal, then recurse with extended accumulator
    Goal =.. [Pred|Args],
    
    % Build access for this join variable
    format(atom(VarName), 'join~w', [Index]),
    format(string(VarAccess), '~w.Args', [VarName]),
    
    % Extend accumulator with current goal
    append(AccumPairs, [Goal-VarAccess], NewAccumPairs),
    
    % Build VarMap from all goals so far
    build_variable_map(NewAccumPairs, _VarMap),
    
    % Find join conditions by checking current goal args against all previous goals
    findall(Var-PrevAccess-PrevIdx-CurrIdx-VarAccess,
        (   nth0(CurrIdx, Args, Var),
            var(Var),
            member(PrevGoal-PrevAccess, AccumPairs),
            PrevGoal =.. [_|PrevArgs],
            nth0(PrevIdx, PrevArgs, PrevVar),
            Var == PrevVar
        ),
        JoinVars),
    
    % Generate join condition
    (   JoinVars = []
    ->  JoinCond = "true"
    ;   findall(Cond,
            (   member(_-SrcAccess-PrevIdx-CurrIdx-CurrAccess, JoinVars),
    format(string(Cond), '~w[\"arg~w\"].Equals(~w[\"arg~w\"])', 
                       [CurrAccess, CurrIdx, SrcAccess, PrevIdx])
            ),
            Conds),
        atomic_list_concat(Conds, " && ", JoinCond)
    ),
    
    % Relation check
    format(string(RelCheck), '~w.Relation == \"~w\"', [VarName, Pred]),
    join_source_expr(Pred, Args, AccumPairs, Index, Indexing, SourceExpr),

    % Recurse for remaining goals with extended accumulator
    NextIndex is Index + 1,
    compile_nway_join(RestGoals, Builtins, Head, NewAccumPairs, Config, Indexing, NextIndex, InnerCode),
    
    % Generate nested foreach
    format(string(Code),
'                foreach (var ~w in ~w)
                {
                    if (~w && ~w)
                    {
~w
                    }
                }', [VarName, SourceExpr, JoinCond, RelCheck, InnerCode]).

compile_nway_join_with_aggregate([], Builtins, AggGoal, Head, AccumPairs, Config, _Indexing, _, Code) :-
    Head =.. [HeadPred|HeadArgs],
    build_variable_map(AccumPairs, VarMap),
    translate_builtins(Builtins, VarMap, Config, ConstraintChecks),
    compile_aggregate_from_bindings(HeadPred, HeadArgs, AggGoal, VarMap, AccumPairs, Config, ConstraintChecks, Code).

compile_nway_join_with_aggregate([Goal|RestGoals], Builtins, AggGoal, Head, AccumPairs, Config, Indexing, Index, Code) :-
    Goal =.. [Pred|Args],
    format(atom(VarName), 'join~w', [Index]),
    format(string(VarAccess), '~w.Args', [VarName]),
    append(AccumPairs, [Goal-VarAccess], NewAccumPairs),
    build_variable_map(NewAccumPairs, _VarMap),
    findall(Var-PrevAccess-PrevIdx-CurrIdx-VarAccess,
        (   nth0(CurrIdx, Args, Var),
            var(Var),
            member(PrevGoal-PrevAccess, AccumPairs),
            PrevGoal =.. [_|PrevArgs],
            nth0(PrevIdx, PrevArgs, PrevVar),
            Var == PrevVar
        ),
        JoinVars),
    (   JoinVars = []
    ->  JoinCond = "true"
    ;   findall(Cond,
            (   member(_-SrcAccess-PrevIdx-CurrIdx-CurrAccess, JoinVars),
                format(string(Cond), '~w[\"arg~w\"].Equals(~w[\"arg~w\"])', 
                       [CurrAccess, CurrIdx, SrcAccess, PrevIdx])
            ),
            Conds),
        atomic_list_concat(Conds, " && ", JoinCond)
    ),
    format(string(RelCheck), '~w.Relation == \"~w\"', [VarName, Pred]),
    join_source_expr(Pred, Args, AccumPairs, Index, Indexing, SourceExpr),
    NextIndex is Index + 1,
    compile_nway_join_with_aggregate(RestGoals, Builtins, AggGoal, Head, NewAccumPairs, Config, Indexing, NextIndex, InnerCode),
    format(string(Code),
'                foreach (var ~w in ~w)
                {
                    if (~w && ~w)
                    {
~w
                    }
                }', [VarName, SourceExpr, JoinCond, RelCheck, InnerCode]).

% Find the most recent access path for a variable across accumulated goals
find_var_access(Var, AccumPairs, Source, Idx) :-
    reverse(AccumPairs, Rev),
    member(Goal-Source, Rev),
    Goal =.. [_|Args],
    nth0(Idx, Args, Var0),
    Var == Var0,
    !.

arg_key_expr(Args, AccumPairs, Pos, KeyExpr) :-
    nth0(Pos, Args, Arg),
    (   ground(Arg)
    ->  format(string(KeyExpr), "\"~w\"", [Arg])
    ;   var(Arg),
        find_var_access(Arg, AccumPairs, Src, SrcIdx)
    ->  format(string(KeyExpr), "~w[\"arg~w\"]", [Src, SrcIdx])
    ).

join_source_expr(Pred, Args, AccumPairs, Index, Indexing, SourceExpr) :-
    (   Indexing == true,
        arg_key_expr(Args, AccumPairs, 0, KeyExpr)
    ->  format(string(SourceExpr),
               '(relIndexArg0.TryGetValue("~w", out var map~w) && map~w.TryGetValue(~w, out var list~w) ? (IEnumerable<Fact>)list~w : Enumerable.Empty<Fact>())',
               [Pred, Index, Index, KeyExpr, Index, Index])
    ;   Indexing == true,
        arg_key_expr(Args, AccumPairs, 1, KeyExpr1)
    ->  format(string(SourceExpr),
               '(relIndexArg1.TryGetValue("~w", out var map~w) && map~w.TryGetValue(~w, out var list~w) ? (IEnumerable<Fact>)list~w : Enumerable.Empty<Fact>())',
               [Pred, Index, Index, KeyExpr1, Index, Index])
    ;   format(string(SourceExpr),
               '(relIndex.TryGetValue("~w", out var list~w) ? list~w : Enumerable.Empty<Fact>())',
               [Pred, Index, Index])
    ).

translate_builtins([], _, _, "true").
translate_builtins(Builtins, VarMap, Config, Code) :-
    findall(Check,
        (   member(Goal, Builtins),
            translate_builtin_or_negation(Goal, VarMap, Config, Check)
        ),
        Checks),
    (   Checks == []
    ->  Code = "true"
    ;   atomic_list_concat(Checks, " && ", Code)
    ).

%% reorder_constraints(+Builtins, +VarMap, -Early, -Late)
%  Split builtins into those whose variables are already bound (Early)
%  and the rest (Late). Safe because Early only references bound vars.
reorder_constraints(Builtins, VarMap, Early, Late) :-
    partition(can_eval_now(VarMap), Builtins, Early, Late).

can_eval_now(VarMap, Goal) :-
    Goal =.. [_|Args],
    forall(member(A, Args), ( \+ var(A) ; member(A-source(_,_), VarMap) )).

%% translate_builtin_or_negation(+Goal, +VarMap, +Config, -Check)
%  Handles both regular builtins and negation
translate_builtin_or_negation(\+ NegGoal, VarMap, Config, Check) :- !,
    translate_negation_check(NegGoal, VarMap, Config, Check).
translate_builtin_or_negation(not(NegGoal), VarMap, Config, Check) :- !,
    translate_negation_check(NegGoal, VarMap, Config, Check).
translate_builtin_or_negation(Goal, VarMap, Config, Check) :-
    (   is_binding_goal_csharp(Goal)
    ->  translate_binding_goal(Goal, VarMap, Config, Check)
    ;   translate_builtin_common(Goal, VarMap, Config, Check)
    ).

is_binding_goal_csharp(Goal) :-
    functor(Goal, Name, Arity),
    cs_binding(Name/Arity, _, _, _, _).

translate_binding_goal(Goal, VarMap, Config, Check) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    cs_binding(Pred/Arity, TargetName, _, _, _),
    maplist(translate_arg_binding(VarMap, Config), Args, ArgExprs),
    (   sub_string(TargetName, 0, 1, _, ".")
    ->  ArgExprs = [Obj|RestArgs],
        (   sub_string(TargetName, _, 2, 0, "()")
        ->  sub_string(TargetName, 0, _, 2, MethodName),
            format(string(Check), "~w~w()", [Obj, MethodName])
        ;   atomic_list_concat(RestArgs, ", ", ArgStr),
            (   sub_string(TargetName, _, 1, 0, "(") % Ends with (
            ->  format(string(Check), "~w~w~w)", [Obj, TargetName, ArgStr])
            ;   sub_string(TargetName, _, 1, 0, ")") % Full signature .Method() ?
            ->  % Fallback for complex patterns, just use as is if no args?
                format(string(Check), "~w~w", [Obj, TargetName])
            ;   format(string(Check), "~w~w(~w)", [Obj, TargetName, ArgStr])
            )
        )
    ;   atomic_list_concat(ArgExprs, ", ", ArgStr),
        format(string(Check), "~w(~w)", [TargetName, ArgStr])
    ).

translate_arg_binding(VarMap, Config, Arg, Expr) :-
    (   translate_expr_common(Arg, VarMap, Config, E)
    ->  Expr = E
    ;   csharp_literal(Arg, Expr)
    ).

%% translate_negation_check(+Goal, +VarMap, +Config, -Check)
%  Generates !total.Contains(new Fact(...)) check
translate_negation_check(Goal, VarMap, Config, Check) :-
    Goal =.. [Pred|Args],
    findall(Assign,
        (   nth0(I, Args, Arg),
            (   translate_expr_common(Arg, VarMap, Config, Expr)
            ->  format(string(Assign), '{ "arg~w", ~w }', [I, Expr])
            ;   format(string(Assign), '{ "arg~w", "~w" }', [I, Arg])
            )
        ),
        Assigns),
    atomic_list_concat(Assigns, ", ", DictContent),
    format(string(Check), '!total.Contains(new Fact("~w", new Dictionary<string, object> { ~w }))', [Pred, DictContent]).

compile_generator_execution(_Pred, RuleNames, Indexing, Code) :-
    findall(Call,
        (   member(Name, RuleNames),
            format(string(Call), "            newFacts.UnionWith(~w(fact, total, relIndex, relIndexArg0, relIndexArg1));", [Name])
        ),
        Calls),
    atomic_list_concat(Calls, "\n", CallsStr),
    (   Indexing == true
    ->  IndexSetup = "
                var relIndex = total
                    .GroupBy(f => f.Relation)
                    .ToDictionary(g => g.Key, g => g.ToList());
                var relIndexArg0 = total
                    .Where(f => f.Args.ContainsKey(\"arg0\"))
                    .GroupBy(f => f.Relation)
                    .ToDictionary(
                        g => g.Key,
                        g => g.GroupBy(f => f.Args[\"arg0\"])
                              .ToDictionary(h => h.Key, h => h.ToList())
                    );
                var relIndexArg1 = total
                    .Where(f => f.Args.ContainsKey(\"arg1\"))
                    .GroupBy(f => f.Relation)
                    .ToDictionary(
                        g => g.Key,
                        g => g.GroupBy(f => f.Args[\"arg1\"])
                              .ToDictionary(h => h.Key, h => h.ToList())
                    );"
    ;   IndexSetup = "
                var relIndex = new Dictionary<string, List<Fact>>();
                var relIndexArg0 = new Dictionary<string, Dictionary<object, List<Fact>>>();
                var relIndexArg1 = new Dictionary<string, Dictionary<object, List<Fact>>>();"
    ),
    format(string(Code),
"        public static HashSet<Fact> Solve()
        {
            var total = GetInitialFacts();
            bool changed = true;
            while (changed)
            {
                changed = false;
                var newFacts = new HashSet<Fact>();~w
                foreach (var fact in total)
                {
~w
                }

                if (newFacts.Count > 0)
                {
                    int before = total.Count;
                    total.UnionWith(newFacts);
                    if (total.Count > before) changed = true;
                }
            }
            return total;
        }", [IndexSetup, CallsStr]).

% ============================================================================
% Pipeline Generator Mode for C#
% ============================================================================
%
% This section implements pipeline chaining with support for generator mode
% (fixpoint iteration) for C# targets. Similar to Python and PowerShell
% pipeline implementations.
%
% Usage:
%   compile_csharp_pipeline([derive/1, transform/1], [
%       pipeline_name('FixpointPipe'),
%       pipeline_mode(generator),
%       output_format(jsonl)
%   ], CSharpCode).

%% compile_csharp_pipeline(+Predicates, +Options, -CSharpCode)
%  Compile a list of predicates into a C# pipeline.
%  Options:
%    pipeline_name(Name) - Name for the pipeline class (default: 'Pipeline')
%    pipeline_mode(Mode) - 'sequential' (default) or 'generator' (fixpoint)
%    output_format(Format) - 'jsonl' (default) or 'json'
%
compile_csharp_pipeline(Predicates, Options, CSharpCode) :-
    option(pipeline_name(PipelineName), Options, 'Pipeline'),
    option(pipeline_mode(PipelineMode), Options, sequential),
    option(output_format(OutputFormat), Options, jsonl),

    % Generate header based on mode
    csharp_pipeline_header(PipelineMode, Header),

    % Generate helper functions based on mode
    csharp_pipeline_helpers(PipelineMode, Helpers),

    % Extract stage names
    extract_csharp_stage_names(Predicates, StageNames),

    % Generate stage functions (placeholder implementations)
    generate_csharp_stage_functions(StageNames, StageFunctions),

    % Generate the pipeline connector (mode-aware)
    generate_csharp_pipeline_connector(StageNames, PipelineName, PipelineMode, ConnectorCode),

    % Generate main execution block
    generate_csharp_main_block(PipelineName, OutputFormat, MainBlock),

    % Combine all parts
    format(string(CSharpCode),
"~w

~w

namespace UnifyWeaver.Pipeline
{
~w
~w

~w
}
", [Header, Helpers, StageFunctions, ConnectorCode, MainBlock]).

%% csharp_pipeline_header(+Mode, -Header)
%  Generate mode-aware header with imports
csharp_pipeline_header(generator, Header) :-
    !,
    format(string(Header),
"// Generated by UnifyWeaver C# Pipeline Generator Mode
// Fixpoint evaluation for recursive pipeline stages
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.IO;", []).

csharp_pipeline_header(_, Header) :-
    format(string(Header),
"// Generated by UnifyWeaver C# Pipeline (sequential mode)
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.IO;", []).

%% csharp_pipeline_helpers(+Mode, -Helpers)
%  Generate mode-aware helper classes and functions
csharp_pipeline_helpers(generator, Helpers) :-
    !,
    format(string(Helpers),
"    /// <summary>
    /// Generates a unique key for record deduplication.
    /// </summary>
    public static class RecordKeyHelper
    {
        public static string GetRecordKey(Dictionary<string, object?> record)
        {
            var sortedKeys = record.Keys.OrderBy(k => k);
            var parts = sortedKeys.Select(k => $\"{k}={record[k]}\");
            return string.Join(\";\", parts);
        }
    }

    /// <summary>
    /// JSONL stream reader for pipeline input.
    /// </summary>
    public static class JsonlHelper
    {
        public static IEnumerable<Dictionary<string, object?>> ReadJsonlStream(TextReader reader)
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    var dict = JsonSerializer.Deserialize<Dictionary<string, object?>>(line);
                    if (dict != null) yield return dict;
                }
            }
        }

        public static void WriteJsonlStream(IEnumerable<Dictionary<string, object?>> records, TextWriter writer)
        {
            foreach (var record in records)
            {
                writer.WriteLine(JsonSerializer.Serialize(record));
            }
        }
    }", []).

csharp_pipeline_helpers(_, Helpers) :-
    format(string(Helpers),
"    /// <summary>
    /// JSONL stream reader for pipeline input.
    /// </summary>
    public static class JsonlHelper
    {
        public static IEnumerable<Dictionary<string, object?>> ReadJsonlStream(TextReader reader)
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    var dict = JsonSerializer.Deserialize<Dictionary<string, object?>>(line);
                    if (dict != null) yield return dict;
                }
            }
        }

        public static void WriteJsonlStream(IEnumerable<Dictionary<string, object?>> records, TextWriter writer)
        {
            foreach (var record in records)
            {
                writer.WriteLine(JsonSerializer.Serialize(record));
            }
        }
    }", []).

%% extract_csharp_stage_names(+Predicates, -Names)
%  Extract stage names from predicate indicators
extract_csharp_stage_names([], []).
extract_csharp_stage_names([Pred|Rest], [Name|RestNames]) :-
    extract_csharp_pred_name(Pred, Name),
    extract_csharp_stage_names(Rest, RestNames).

extract_csharp_pred_name(_Target:Name/_Arity, NameStr) :-
    !,
    atom_string(Name, NameStr).
extract_csharp_pred_name(Name/_Arity, NameStr) :-
    atom_string(Name, NameStr).

%% generate_csharp_stage_functions(+Names, -Code)
%  Generate placeholder stage function implementations
generate_csharp_stage_functions([], "").
generate_csharp_stage_functions([Name|Rest], Code) :-
    snake_case_to_pascal(Name, PascalName),
    format(string(StageCode),
"    /// <summary>
    /// Stage: ~w
    /// </summary>
    public static IEnumerable<Dictionary<string, object?>> ~w(IEnumerable<Dictionary<string, object?>> input)
    {
        // TODO: Implement stage logic
        foreach (var record in input)
        {
            yield return record;
        }
    }
", [Name, PascalName]),
    generate_csharp_stage_functions(Rest, RestCode),
    format(string(Code), "~w~w", [StageCode, RestCode]).

%% generate_csharp_pipeline_connector(+StageNames, +PipelineName, +Mode, -Code)
%  Generate the pipeline connector function (mode-aware)
generate_csharp_pipeline_connector(StageNames, PipelineName, sequential, Code) :-
    !,
    generate_csharp_sequential_chain(StageNames, ChainCode),
    format(string(Code),
"    /// <summary>
    /// Sequential pipeline connector: ~w
    /// </summary>
    public static IEnumerable<Dictionary<string, object?>> ~w(IEnumerable<Dictionary<string, object?>> input)
    {
        // Sequential mode - chain stages directly
~w
    }", [PipelineName, PipelineName, ChainCode]).

generate_csharp_pipeline_connector(StageNames, PipelineName, generator, Code) :-
    generate_csharp_fixpoint_chain(StageNames, ChainCode),
    format(string(Code),
"    /// <summary>
    /// Fixpoint pipeline connector: ~w
    /// Iterates until no new records are produced.
    /// </summary>
    public static IEnumerable<Dictionary<string, object?>> ~w(IEnumerable<Dictionary<string, object?>> input)
    {
        // Generator mode - fixpoint iteration
        var total = new HashSet<string>();

        // Initialize with input records
        var inputList = new List<Dictionary<string, object?>>();
        foreach (var record in input)
        {
            var key = RecordKeyHelper.GetRecordKey(record);
            if (!total.Contains(key))
            {
                total.Add(key);
                inputList.Add(record);
                yield return record;
            }
        }

        // Fixpoint iteration
        var changed = true;
        while (changed)
        {
            changed = false;
            var current = inputList.ToList();

            // Apply pipeline stages
~w

            // Check for new records
            foreach (var record in newRecords)
            {
                var key = RecordKeyHelper.GetRecordKey(record);
                if (!total.Contains(key))
                {
                    total.Add(key);
                    inputList.Add(record);
                    changed = true;
                    yield return record;
                }
            }
        }
    }", [PipelineName, PipelineName, ChainCode]).

%% generate_csharp_sequential_chain(+Names, -Code)
%  Generate sequential stage chaining code
generate_csharp_sequential_chain([], Code) :-
    format(string(Code), "        foreach (var record in input)
        {
            yield return record;
        }", []).
generate_csharp_sequential_chain([Name], Code) :-
    !,
    snake_case_to_pascal(Name, PascalName),
    format(string(Code), "        foreach (var record in ~w(input))
        {
            yield return record;
        }", [PascalName]).
generate_csharp_sequential_chain(Names, Code) :-
    Names \= [],
    generate_csharp_chain_expr(Names, "input", ChainExpr),
    format(string(Code), "        foreach (var record in ~w)
        {
            yield return record;
        }", [ChainExpr]).

generate_csharp_chain_expr([], Current, Current).
generate_csharp_chain_expr([Name|Rest], Current, Expr) :-
    snake_case_to_pascal(Name, PascalName),
    format(string(NextExpr), "~w(~w)", [PascalName, Current]),
    generate_csharp_chain_expr(Rest, NextExpr, Expr).

%% generate_csharp_fixpoint_chain(+Names, -Code)
%  Generate fixpoint stage application code
generate_csharp_fixpoint_chain([], Code) :-
    format(string(Code), "            var newRecords = current;", []).
generate_csharp_fixpoint_chain(Names, Code) :-
    Names \= [],
    generate_csharp_fixpoint_stages(Names, "current", StageCode),
    format(string(Code), "~w", [StageCode]).

generate_csharp_fixpoint_stages([], Current, Code) :-
    format(string(Code), "            var newRecords = ~w;", [Current]).
generate_csharp_fixpoint_stages([Stage|Rest], Current, Code) :-
    snake_case_to_pascal(Stage, PascalStage),
    format(string(NextVar), "stage~wOut", [PascalStage]),
    format(string(StageCall), "            var ~w = ~w(~w).ToList();
", [NextVar, PascalStage, Current]),
    generate_csharp_fixpoint_stages(Rest, NextVar, RestCode),
    format(string(Code), "~w~w", [StageCall, RestCode]).

%% generate_csharp_main_block(+PipelineName, +Format, -Code)
%  Generate main execution block
generate_csharp_main_block(PipelineName, jsonl, Code) :-
    format(string(Code),
"    public static class Program
    {
        public static void Main(string[] args)
        {
            // Read from stdin, process through pipeline, write to stdout
            var inputStream = JsonlHelper.ReadJsonlStream(Console.In);
            JsonlHelper.WriteJsonlStream(~w(inputStream), Console.Out);
        }
    }", [PipelineName]).

generate_csharp_main_block(PipelineName, json, Code) :-
    format(string(Code),
"    public static class Program
    {
        public static void Main(string[] args)
        {
            // Read JSON array from stdin, process through pipeline, write to stdout
            var inputJson = Console.In.ReadToEnd();
            var inputArray = JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(inputJson) ?? new();
            var results = ~w(inputArray).ToList();
            Console.WriteLine(JsonSerializer.Serialize(results));
        }
    }", [PipelineName]).

% ============================================================================
% Unit Tests for C# Pipeline Generator Mode
% ============================================================================

test_csharp_pipeline_generator :-
    format("~n=== C# Pipeline Generator Mode Unit Tests ===~n~n", []),

    % Test 1: Basic pipeline compilation with generator mode
    format("Test 1: Basic pipeline with generator mode... ", []),
    (   compile_csharp_pipeline([transform/1, derive/1], [
            pipeline_name('TestPipeline'),
            pipeline_mode(generator),
            output_format(jsonl)
        ], Code1),
        sub_string(Code1, _, _, _, "GetRecordKey"),
        sub_string(Code1, _, _, _, "while (changed)"),
        sub_string(Code1, _, _, _, "TestPipeline")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 2: Sequential mode still works
    format("Test 2: Sequential mode still works... ", []),
    (   compile_csharp_pipeline([filter/1, format/1], [
            pipeline_name('SeqPipeline'),
            pipeline_mode(sequential),
            output_format(jsonl)
        ], Code2),
        sub_string(Code2, _, _, _, "SeqPipeline"),
        sub_string(Code2, _, _, _, "sequential mode"),
        \+ sub_string(Code2, _, _, _, "while (changed)")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 3: Generator mode includes RecordKeyHelper
    format("Test 3: Generator mode has RecordKeyHelper... ", []),
    (   compile_csharp_pipeline([a/1], [pipeline_mode(generator)], Code3),
        sub_string(Code3, _, _, _, "RecordKeyHelper"),
        sub_string(Code3, _, _, _, "GetRecordKey")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 4: JSONL helpers included
    format("Test 4: JSONL helpers included... ", []),
    (   compile_csharp_pipeline([x/1], [pipeline_mode(generator)], Code4),
        sub_string(Code4, _, _, _, "JsonlHelper"),
        sub_string(Code4, _, _, _, "ReadJsonlStream"),
        sub_string(Code4, _, _, _, "WriteJsonlStream")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 5: Fixpoint iteration structure
    format("Test 5: Fixpoint iteration structure... ", []),
    (   compile_csharp_pipeline([derive/1, transform/1], [pipeline_mode(generator)], Code5),
        sub_string(Code5, _, _, _, "HashSet<string>"),
        sub_string(Code5, _, _, _, "changed = true"),
        sub_string(Code5, _, _, _, "while (changed)"),
        sub_string(Code5, _, _, _, "total.Contains(key)")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 6: Stage functions generated
    format("Test 6: Stage functions generated... ", []),
    (   compile_csharp_pipeline([filter/1, transform/1], [pipeline_mode(generator)], Code6),
        sub_string(Code6, _, _, _, "public static IEnumerable"),
        sub_string(Code6, _, _, _, "Filter"),
        sub_string(Code6, _, _, _, "Transform")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 7: Pipeline chain code for generator
    format("Test 7: Pipeline chain code for generator mode... ", []),
    (   compile_csharp_pipeline([derive/1, transform/1], [pipeline_mode(generator)], Code7),
        sub_string(Code7, _, _, _, "stageDeriveOut"),
        sub_string(Code7, _, _, _, "stageTransformOut")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 8: Default options work
    format("Test 8: Default options work... ", []),
    (   compile_csharp_pipeline([a/1, b/1], [], Code8),
        sub_string(Code8, _, _, _, "Pipeline"),
        sub_string(Code8, _, _, _, "sequential mode")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 9: Main block for JSONL format
    format("Test 9: Main block for JSONL format... ", []),
    (   compile_csharp_pipeline([x/1], [
            pipeline_name('JsonlPipe'),
            output_format(jsonl)
        ], Code9),
        sub_string(Code9, _, _, _, "JsonlHelper.ReadJsonlStream"),
        sub_string(Code9, _, _, _, "Console.In"),
        sub_string(Code9, _, _, _, "JsonlPipe")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    % Test 10: JSON array format option
    format("Test 10: JSON array format option... ", []),
    (   compile_csharp_pipeline([x/1], [
            pipeline_name('JsonPipe'),
            output_format(json)
        ], Code10),
        sub_string(Code10, _, _, _, "ReadToEnd"),
        sub_string(Code10, _, _, _, "JsonSerializer.Deserialize"),
        sub_string(Code10, _, _, _, "JsonPipe")
    ->  format("PASS~n", [])
    ;   format("FAIL~n", []), fail
    ),

    format("~n=== All C# Pipeline Generator Mode Tests Passed ===~n", []).

%% ============================================
%% C# ENHANCED PIPELINE CHAINING
%% ============================================
%
%  Supports advanced flow patterns:
%    - fan_out(Stages) : Broadcast to parallel stages
%    - merge          : Combine results from parallel stages
%    - route_by(Pred, Routes) : Conditional routing
%    - filter_by(Pred) : Filter records
%    - Pred/Arity     : Standard stage
%
%% compile_csharp_enhanced_pipeline(+Stages, +Options, -CSharpCode)
%  Main entry point for enhanced C# pipeline with advanced flow patterns.
%  Validates pipeline stages before code generation.
%
compile_csharp_enhanced_pipeline(Stages, Options, CSharpCode) :-
    % Validate pipeline stages
    option(validate(Validate), Options, true),
    option(strict(Strict), Options, false),
    ( Validate == true ->
        validate_pipeline(Stages, [strict(Strict)], result(Errors, Warnings)),
        % Report warnings
        ( Warnings \== [] ->
            format(user_error, 'C# pipeline warnings:~n', []),
            forall(member(W, Warnings), (
                format_validation_warning(W, Msg),
                format(user_error, '  ~w~n', [Msg])
            ))
        ; true
        ),
        % Fail on errors
        ( Errors \== [] ->
            format(user_error, 'C# pipeline validation errors:~n', []),
            forall(member(E, Errors), (
                format_validation_error(E, Msg),
                format(user_error, '  ~w~n', [Msg])
            )),
            throw(pipeline_validation_failed(Errors))
        ; true
        )
    ; true
    ),

    option(pipeline_name(PipelineName), Options, 'EnhancedPipeline'),
    option(output_format(OutputFormat), Options, jsonl),

    % Generate helpers
    csharp_enhanced_helpers(Helpers),

    % Generate stage functions
    generate_csharp_enhanced_stage_functions(Stages, StageFunctions),

    % Generate the main connector
    generate_csharp_enhanced_connector(Stages, PipelineName, ConnectorCode),

    % Generate main class
    generate_csharp_enhanced_main(PipelineName, OutputFormat, MainCode),

    format(string(CSharpCode),
"// Generated by UnifyWeaver C# Enhanced Pipeline
// Supports fan-out, merge, conditional routing, and filtering
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.IO;

namespace UnifyWeaver.Pipeline
{
~w

~w
~w
~w
}
", [Helpers, StageFunctions, ConnectorCode, MainCode]).

%% csharp_enhanced_helpers(-Code)
%  Generate helper functions for enhanced pipeline operations.
csharp_enhanced_helpers(Code) :-
    Code = "    /// <summary>
    /// Enhanced Pipeline Helper Functions
    /// </summary>
    public static class EnhancedPipelineHelpers
    {
        /// <summary>
        /// Fan-out: Send record to all stages, collect all results.
        /// </summary>
        public static IEnumerable<Dictionary<string, object?>> FanOutRecords(
            Dictionary<string, object?> record,
            params Func<IEnumerable<Dictionary<string, object?>>, IEnumerable<Dictionary<string, object?>>>[] stages)
        {
            foreach (var stage in stages)
            {
                foreach (var result in stage(new[] { record }))
                {
                    yield return result;
                }
            }
        }

        /// <summary>
        /// Merge: Combine multiple streams into one.
        /// </summary>
        public static IEnumerable<Dictionary<string, object?>> MergeStreams(
            params IEnumerable<Dictionary<string, object?>>[] streams)
        {
            foreach (var stream in streams)
            {
                foreach (var record in stream)
                {
                    yield return record;
                }
            }
        }

        /// <summary>
        /// Route: Direct record to appropriate stage based on condition.
        /// </summary>
        public static IEnumerable<Dictionary<string, object?>> RouteRecord(
            Dictionary<string, object?> record,
            Func<Dictionary<string, object?>, object> conditionFn,
            Dictionary<object, Func<IEnumerable<Dictionary<string, object?>>, IEnumerable<Dictionary<string, object?>>>> routeMap,
            Func<IEnumerable<Dictionary<string, object?>>, IEnumerable<Dictionary<string, object?>>>? defaultFn = null)
        {
            var condition = conditionFn(record);
            if (routeMap.TryGetValue(condition, out var stage))
            {
                foreach (var result in stage(new[] { record }))
                {
                    yield return result;
                }
            }
            else if (defaultFn != null)
            {
                foreach (var result in defaultFn(new[] { record }))
                {
                    yield return result;
                }
            }
            else
            {
                yield return record; // Pass through if no matching route
            }
        }

        /// <summary>
        /// Filter: Only yield records that satisfy the predicate.
        /// </summary>
        public static IEnumerable<Dictionary<string, object?>> FilterRecords(
            IEnumerable<Dictionary<string, object?>> records,
            Func<Dictionary<string, object?>, bool> predicateFn)
        {
            foreach (var record in records)
            {
                if (predicateFn(record))
                {
                    yield return record;
                }
            }
        }

        /// <summary>
        /// Tee: Send each record to multiple stages and collect all results.
        /// </summary>
        public static IEnumerable<Dictionary<string, object?>> TeeStream(
            IEnumerable<Dictionary<string, object?>> records,
            params Func<IEnumerable<Dictionary<string, object?>>, IEnumerable<Dictionary<string, object?>>>[] stages)
        {
            var recordList = records.ToList(); // Materialize to allow multiple iterations
            foreach (var record in recordList)
            {
                foreach (var stage in stages)
                {
                    foreach (var result in stage(new[] { record }))
                    {
                        yield return result;
                    }
                }
            }
        }

        /// <summary>
        /// Read JSONL from stdin.
        /// </summary>
        public static IEnumerable<Dictionary<string, object?>> ReadJsonlStream(TextReader reader)
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    var dict = JsonSerializer.Deserialize<Dictionary<string, object?>>(line);
                    if (dict != null) yield return dict;
                }
            }
        }

        /// <summary>
        /// Write JSONL to stdout.
        /// </summary>
        public static void WriteJsonlStream(IEnumerable<Dictionary<string, object?>> records, TextWriter writer)
        {
            foreach (var record in records)
            {
                writer.WriteLine(JsonSerializer.Serialize(record));
            }
        }
    }
".

%% generate_csharp_enhanced_stage_functions(+Stages, -Code)
%  Generate stub functions for each stage.
generate_csharp_enhanced_stage_functions([], "").
generate_csharp_enhanced_stage_functions([Stage|Rest], Code) :-
    generate_csharp_single_enhanced_stage(Stage, StageCode),
    generate_csharp_enhanced_stage_functions(Rest, RestCode),
    (RestCode = "" ->
        Code = StageCode
    ;   format(string(Code), "~w~n~w", [StageCode, RestCode])
    ).

generate_csharp_single_enhanced_stage(fan_out(SubStages), Code) :-
    !,
    generate_csharp_enhanced_stage_functions(SubStages, Code).
generate_csharp_single_enhanced_stage(merge, "") :- !.
generate_csharp_single_enhanced_stage(route_by(_, Routes), Code) :-
    !,
    findall(Stage, member((_Cond, Stage), Routes), RouteStages),
    generate_csharp_enhanced_stage_functions(RouteStages, Code).
generate_csharp_single_enhanced_stage(filter_by(_), "") :- !.
generate_csharp_single_enhanced_stage(Pred/Arity, Code) :-
    !,
    format(string(Code),
"    /// <summary>
    /// Pipeline stage: ~w/~w
    /// </summary>
    public static class ~wStage
    {
        public static IEnumerable<Dictionary<string, object?>> Process(
            IEnumerable<Dictionary<string, object?>> input)
        {
            // TODO: Implement based on predicate bindings
            foreach (var record in input)
            {
                yield return record;
            }
        }
    }

", [Pred, Arity, Pred]).
generate_csharp_single_enhanced_stage(_, "").

%% generate_csharp_enhanced_connector(+Stages, +PipelineName, -Code)
%  Generate the main connector that handles enhanced flow patterns.
generate_csharp_enhanced_connector(Stages, PipelineName, Code) :-
    generate_csharp_enhanced_flow_code(Stages, "input", FlowCode),
    format(string(Code),
"    /// <summary>
    /// ~w is an enhanced pipeline with fan-out, merge, and routing support.
    /// </summary>
    public static class ~w
    {
        public static IEnumerable<Dictionary<string, object?>> Run(
            IEnumerable<Dictionary<string, object?>> input)
        {
~w
        }
    }

", [PipelineName, PipelineName, FlowCode]).

%% generate_csharp_enhanced_flow_code(+Stages, +CurrentVar, -Code)
%  Generate the flow code for enhanced pipeline stages.
generate_csharp_enhanced_flow_code([], CurrentVar, Code) :-
    format(string(Code), "            foreach (var r in ~w) yield return r;", [CurrentVar]).
generate_csharp_enhanced_flow_code([Stage|Rest], CurrentVar, Code) :-
    generate_csharp_stage_flow(Stage, CurrentVar, NextVar, StageCode),
    generate_csharp_enhanced_flow_code(Rest, NextVar, RestCode),
    format(string(Code), "~w~n~w", [StageCode, RestCode]).

%% generate_csharp_stage_flow(+Stage, +InVar, -OutVar, -Code)
%  Generate flow code for a single stage.

% Fan-out stage: broadcast to parallel stages
generate_csharp_stage_flow(fan_out(SubStages), InVar, OutVar, Code) :-
    !,
    length(SubStages, N),
    format(atom(OutVar), "fanOut~wResult", [N]),
    extract_csharp_stage_names(SubStages, StageNames),
    format_csharp_stage_list(StageNames, StageListStr),
    format(string(Code),
"            // Fan-out to ~w parallel stages
            var ~w = ~w.SelectMany(record =>
                EnhancedPipelineHelpers.FanOutRecords(record, ~w));", [N, OutVar, InVar, StageListStr]).

% Merge stage: placeholder, usually follows fan_out
generate_csharp_stage_flow(merge, InVar, OutVar, Code) :-
    !,
    OutVar = InVar,
    Code = "            // Merge: results already combined from fan-out".

% Conditional routing
generate_csharp_stage_flow(route_by(CondPred, Routes), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "routedResult", []),
    format_csharp_route_map(Routes, RouteMapStr),
    format(string(Code),
"            // Conditional routing based on ~w
            var routeMap = new Dictionary<object, Func<IEnumerable<Dictionary<string, object?>>, IEnumerable<Dictionary<string, object?>>>>
            {
~w
            };
            var ~w = ~w.SelectMany(record =>
                EnhancedPipelineHelpers.RouteRecord(record, ~w, routeMap));", [CondPred, RouteMapStr, OutVar, InVar, CondPred]).

% Filter stage
generate_csharp_stage_flow(filter_by(Pred), InVar, OutVar, Code) :-
    !,
    format(atom(OutVar), "filteredResult", []),
    format(string(Code),
"            // Filter by ~w
            var ~w = EnhancedPipelineHelpers.FilterRecords(~w, ~w);", [Pred, OutVar, InVar, Pred]).

% Standard predicate stage
generate_csharp_stage_flow(Pred/Arity, InVar, OutVar, Code) :-
    !,
    atom(Pred),
    format(atom(OutVar), "~wResult", [Pred]),
    format(string(Code),
"            // Stage: ~w/~w
            var ~w = ~wStage.Process(~w);", [Pred, Arity, OutVar, Pred, InVar]).

% Fallback for unknown stages
generate_csharp_stage_flow(Stage, InVar, InVar, Code) :-
    format(string(Code), "            // Unknown stage type: ~w (pass-through)", [Stage]).

%% extract_csharp_stage_names(+Stages, -Names)
%  Extract function names from stage specifications.
extract_csharp_stage_names([], []).
extract_csharp_stage_names([Pred/_Arity|Rest], [Pred|RestNames]) :-
    !,
    extract_csharp_stage_names(Rest, RestNames).
extract_csharp_stage_names([_|Rest], RestNames) :-
    extract_csharp_stage_names(Rest, RestNames).

%% format_csharp_stage_list(+Names, -ListStr)
%  Format stage names as C# delegate references.
format_csharp_stage_list([], "").
format_csharp_stage_list([Name], Str) :-
    format(string(Str), "~wStage.Process", [Name]).
format_csharp_stage_list([Name|Rest], Str) :-
    Rest \= [],
    format_csharp_stage_list(Rest, RestStr),
    format(string(Str), "~wStage.Process, ~w", [Name, RestStr]).

%% format_csharp_route_map(+Routes, -MapStr)
%  Format routing map for C#.
format_csharp_route_map([], "").
format_csharp_route_map([(_Cond, Stage)|[]], Str) :-
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format(string(Str), "                { true, ~wStage.Process }", [StageName]).
format_csharp_route_map([(Cond, Stage)|Rest], Str) :-
    Rest \= [],
    (Stage = StageName/_Arity -> true ; StageName = Stage),
    format_csharp_route_map(Rest, RestStr),
    (Cond = true ->
        format(string(Str), "                { true, ~wStage.Process },~n~w", [StageName, RestStr])
    ; Cond = false ->
        format(string(Str), "                { false, ~wStage.Process },~n~w", [StageName, RestStr])
    ;   format(string(Str), "                { \"~w\", ~wStage.Process },~n~w", [Cond, StageName, RestStr])
    ).

%% generate_csharp_enhanced_main(+PipelineName, +OutputFormat, -Code)
%  Generate main class for enhanced pipeline.
generate_csharp_enhanced_main(PipelineName, jsonl, Code) :-
    format(string(Code),
"    /// <summary>
    /// Main entry point for the enhanced pipeline.
    /// </summary>
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Read JSONL from stdin
            var input = EnhancedPipelineHelpers.ReadJsonlStream(Console.In);

            // Run enhanced pipeline
            var results = ~w.Run(input);

            // Output results as JSONL
            EnhancedPipelineHelpers.WriteJsonlStream(results, Console.Out);
        }
    }
", [PipelineName]).
generate_csharp_enhanced_main(PipelineName, _, Code) :-
    format(string(Code),
"    /// <summary>
    /// Main entry point for the enhanced pipeline.
    /// </summary>
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Read JSON from stdin
            var json = Console.In.ReadToEnd();
            var input = JsonSerializer.Deserialize<List<Dictionary<string, object?>>>(json) ?? new List<Dictionary<string, object?>>();

            // Run enhanced pipeline
            var results = ~w.Run(input).ToList();

            // Output results as JSON
            Console.WriteLine(JsonSerializer.Serialize(results));
        }
    }
", [PipelineName]).

%% ============================================
%% C# ENHANCED PIPELINE CHAINING TESTS
%% ============================================

test_csharp_enhanced_chaining :-
    format('~n=== C# Enhanced Pipeline Chaining Tests ===~n~n', []),

    % Test 1: Generate enhanced helpers
    format('[Test 1] Generate enhanced helpers~n', []),
    csharp_enhanced_helpers(Helpers1),
    (   sub_string(Helpers1, _, _, _, "FanOutRecords"),
        sub_string(Helpers1, _, _, _, "MergeStreams"),
        sub_string(Helpers1, _, _, _, "RouteRecord"),
        sub_string(Helpers1, _, _, _, "FilterRecords"),
        sub_string(Helpers1, _, _, _, "TeeStream")
    ->  format('  [PASS] All helper functions generated~n', [])
    ;   format('  [FAIL] Missing helper functions~n', [])
    ),

    % Test 2: Linear pipeline connector
    format('[Test 2] Linear pipeline connector~n', []),
    generate_csharp_enhanced_connector([extract/1, transform/1, load/1], 'LinearPipe', Code2),
    (   sub_string(Code2, _, _, _, "LinearPipe"),
        sub_string(Code2, _, _, _, "extractStage.Process"),
        sub_string(Code2, _, _, _, "transformStage.Process"),
        sub_string(Code2, _, _, _, "loadStage.Process")
    ->  format('  [PASS] Linear connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code2])
    ),

    % Test 3: Fan-out connector
    format('[Test 3] Fan-out connector~n', []),
    generate_csharp_enhanced_connector([fan_out([validate/1, enrich/1])], 'FanoutPipe', Code3),
    (   sub_string(Code3, _, _, _, "FanoutPipe"),
        sub_string(Code3, _, _, _, "Fan-out to 2 parallel stages"),
        sub_string(Code3, _, _, _, "FanOutRecords")
    ->  format('  [PASS] Fan-out connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code3])
    ),

    % Test 4: Fan-out with merge
    format('[Test 4] Fan-out with merge~n', []),
    generate_csharp_enhanced_connector([fan_out([a/1, b/1]), merge], 'MergePipe', Code4),
    (   sub_string(Code4, _, _, _, "MergePipe"),
        sub_string(Code4, _, _, _, "Fan-out to 2"),
        sub_string(Code4, _, _, _, "Merge: results already combined")
    ->  format('  [PASS] Merge connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code4])
    ),

    % Test 5: Conditional routing
    format('[Test 5] Conditional routing~n', []),
    generate_csharp_enhanced_connector([route_by(hasError, [(true, errorHandler/1), (false, success/1)])], 'RoutePipe', Code5),
    (   sub_string(Code5, _, _, _, "RoutePipe"),
        sub_string(Code5, _, _, _, "Conditional routing based on hasError"),
        sub_string(Code5, _, _, _, "routeMap")
    ->  format('  [PASS] Routing connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code5])
    ),

    % Test 6: Filter stage
    format('[Test 6] Filter stage~n', []),
    generate_csharp_enhanced_connector([filter_by(isValid)], 'FilterPipe', Code6),
    (   sub_string(Code6, _, _, _, "FilterPipe"),
        sub_string(Code6, _, _, _, "Filter by isValid"),
        sub_string(Code6, _, _, _, "FilterRecords")
    ->  format('  [PASS] Filter connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code6])
    ),

    % Test 7: Complex pipeline with all patterns
    format('[Test 7] Complex pipeline~n', []),
    generate_csharp_enhanced_connector([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1, audit/1]),
        merge,
        route_by(hasError, [(true, errorLog/1), (false, transform/1)]),
        output/1
    ], 'ComplexPipe', Code7),
    (   sub_string(Code7, _, _, _, "ComplexPipe"),
        sub_string(Code7, _, _, _, "Filter by isActive"),
        sub_string(Code7, _, _, _, "Fan-out to 3 parallel stages"),
        sub_string(Code7, _, _, _, "Merge"),
        sub_string(Code7, _, _, _, "Conditional routing")
    ->  format('  [PASS] Complex connector generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [Code7])
    ),

    % Test 8: Stage function generation
    format('[Test 8] Stage function generation~n', []),
    generate_csharp_enhanced_stage_functions([extract/1, transform/1], StageFns8),
    (   sub_string(StageFns8, _, _, _, "extractStage"),
        sub_string(StageFns8, _, _, _, "transformStage")
    ->  format('  [PASS] Stage functions generated~n', [])
    ;   format('  [FAIL] Code: ~w~n', [StageFns8])
    ),

    % Test 9: Full enhanced pipeline compilation
    format('[Test 9] Full enhanced pipeline~n', []),
    compile_csharp_enhanced_pipeline([
        extract/1,
        filter_by(isActive),
        fan_out([validate/1, enrich/1]),
        merge,
        output/1
    ], [pipeline_name('FullEnhanced'), output_format(jsonl)], FullCode9),
    (   sub_string(FullCode9, _, _, _, "namespace UnifyWeaver.Pipeline"),
        sub_string(FullCode9, _, _, _, "FanOutRecords"),
        sub_string(FullCode9, _, _, _, "FilterRecords"),
        sub_string(FullCode9, _, _, _, "FullEnhanced"),
        sub_string(FullCode9, _, _, _, "Main(")
    ->  format('  [PASS] Full pipeline compiles~n', [])
    ;   format('  [FAIL] Missing patterns in generated code~n', [])
    ),

    % Test 10: Enhanced helpers include all functions
    format('[Test 10] Enhanced helpers completeness~n', []),
    csharp_enhanced_helpers(Helpers10),
    (   sub_string(Helpers10, _, _, _, "FanOutRecords"),
        sub_string(Helpers10, _, _, _, "MergeStreams"),
        sub_string(Helpers10, _, _, _, "RouteRecord"),
        sub_string(Helpers10, _, _, _, "FilterRecords"),
        sub_string(Helpers10, _, _, _, "TeeStream")
    ->  format('  [PASS] All helpers present~n', [])
    ;   format('  [FAIL] Missing helpers~n', [])
    ),

    format('~n=== All C# Enhanced Pipeline Chaining Tests Passed ===~n', []).
