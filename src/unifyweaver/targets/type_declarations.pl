% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% type_declarations.pl - Shared type metadata and resolution helpers

:- module(type_declarations, [
    uw_type/3,
    uw_return_type/2,
    uw_typed_mode/2,
    uw_domain_type/2,
    resolve_type/3,
    resolve_typed_mode/4,
    build_type_context/3,
    predicate_arg_type/3,
    predicate_return_type/2,
    resolved_arg_type/5,
    resolved_return_type/3,
    has_explicit_any/2,
    uw_typed/2,
    clear_type_declarations/0
]).

:- use_module(library(lists)).
:- use_module(library(option)).

:- dynamic uw_type/3.
:- dynamic uw_return_type/2.
:- dynamic uw_typed_mode/2.
:- dynamic uw_domain_type/2.

clear_type_declarations :-
    retractall(uw_type(_, _, _)),
    retractall(uw_return_type(_, _)),
    retractall(uw_typed_mode(_, _)),
    retractall(uw_domain_type(_, _)).

resolve_typed_mode(PredSpec, Options, GlobalMode, Mode) :-
    normalize_pred_spec(PredSpec, NormalizedPredSpec),
    (   declared_uw_typed_mode(NormalizedPredSpec, Mode0)
    ->  Mode = Mode0
    ;   option(typed_mode(Mode0), Options)
    ->  Mode = Mode0
    ;   Mode = GlobalMode
    ).

predicate_arg_type(PredSpec, ArgIndex, TypeTerm) :-
    normalize_pred_spec(PredSpec, NormalizedPredSpec),
    declared_uw_type(NormalizedPredSpec, ArgIndex, Type0),
    resolve_domain_type(Type0, TypeTerm).

predicate_return_type(PredSpec, TypeTerm) :-
    normalize_pred_spec(PredSpec, NormalizedPredSpec),
    declared_uw_return_type(NormalizedPredSpec, Type0),
    resolve_domain_type(Type0, TypeTerm).

resolved_arg_type(PredSpec, FallbackPredSpec, ArgIndex, TargetLang, ConcreteType) :-
    (   predicate_arg_type(PredSpec, ArgIndex, AbstractType)
    ->  true
    ;   FallbackPredSpec \== none,
        predicate_arg_type(FallbackPredSpec, ArgIndex, AbstractType)
    ),
    resolve_type(AbstractType, TargetLang, ConcreteType).

resolved_return_type(PredSpec, TargetLang, ConcreteType) :-
    predicate_return_type(PredSpec, AbstractType),
    resolve_type(AbstractType, TargetLang, ConcreteType).

has_explicit_any(PredSpec, ArgIndex) :-
    normalize_pred_spec(PredSpec, NormalizedPredSpec),
    declared_uw_type(NormalizedPredSpec, ArgIndex, any).

uw_typed(PredSpec, true) :-
    normalize_pred_spec(PredSpec, NormalizedPredSpec),
    (   declared_uw_type(NormalizedPredSpec, _, _)
    ;   declared_uw_return_type(NormalizedPredSpec, _)
    ),
    !.
uw_typed(_, false).

build_type_context(PredSpec, TargetLang, Context) :-
    PredSpec = _/Arity,
    findall(Key=Value, (
        between(1, Arity, Index),
        predicate_arg_type(PredSpec, Index, AbstractType),
        resolve_type(AbstractType, TargetLang, ConcreteType),
        format(atom(Key), 'arg~w_type', [Index]),
        Value = ConcreteType
    ), ArgPairs),
    (   predicate_arg_type(PredSpec, 1, NodeAbstract),
        resolve_type(NodeAbstract, TargetLang, NodeType)
    ->  NodePairs = [node_type=NodeType]
    ;   NodePairs = []
    ),
    (   PredSpec = _/Arity,
        Arity >= 3,
        predicate_arg_type(PredSpec, 3, WeightAbstract),
        resolve_type(WeightAbstract, TargetLang, WeightType)
    ->  WeightPairs = [weight_type=WeightType]
    ;   WeightPairs = []
    ),
    (   predicate_arg_type(PredSpec, 1, LeftAbstract),
        predicate_arg_type(PredSpec, 2, RightAbstract),
        resolve_type(LeftAbstract, TargetLang, LeftType),
        resolve_type(RightAbstract, TargetLang, RightType)
    ->  edge_type_string(TargetLang, LeftType, RightType, EdgeType),
        EdgePairs = [edge_type=EdgeType]
    ;   EdgePairs = []
    ),
    (   predicate_return_type(PredSpec, ReturnAbstract),
        resolve_type(ReturnAbstract, TargetLang, ReturnType)
    ->  ReturnPairs = [return_type=ReturnType]
    ;   ReturnPairs = []
    ),
    uw_typed(PredSpec, Typed),
    append([[typed=Typed], NodePairs, WeightPairs, EdgePairs, ReturnPairs, ArgPairs], Context).

resolve_type(atom, haskell, "String").
resolve_type(string, haskell, "String").
resolve_type(integer, haskell, "Int").
resolve_type(float, haskell, "Double").
resolve_type(number, haskell, "Double").
resolve_type(boolean, haskell, "Bool").
resolve_type(any, haskell, "a").
resolve_type(list(Type), haskell, Concrete) :-
    resolve_type(Type, haskell, Inner),
    format(string(Concrete), "[~w]", [Inner]).
resolve_type(maybe(Type), haskell, Concrete) :-
    resolve_type(Type, haskell, Inner),
    format(string(Concrete), "Maybe ~w", [Inner]).
resolve_type(map(KeyType, ValueType), haskell, Concrete) :-
    resolve_type(KeyType, haskell, Key),
    resolve_type(ValueType, haskell, Value),
    format(string(Concrete), "Map.Map ~w ~w", [Key, Value]).
resolve_type(set(Type), haskell, Concrete) :-
    resolve_type(Type, haskell, Inner),
    format(string(Concrete), "Set.Set ~w", [Inner]).
resolve_type(pair(LeftType, RightType), haskell, Concrete) :-
    resolve_type(LeftType, haskell, Left),
    resolve_type(RightType, haskell, Right),
    format(string(Concrete), "(~w, ~w)", [Left, Right]).
resolve_type(record(Name, _Fields), haskell, Concrete) :-
    atom_string(Name, Concrete).

resolve_type(atom, java, "String").
resolve_type(string, java, "String").
resolve_type(integer, java, "Integer").
resolve_type(float, java, "Double").
resolve_type(number, java, "Double").
resolve_type(boolean, java, "Boolean").
resolve_type(any, java, "Object").
resolve_type(list(Type), java, Concrete) :-
    resolve_type(Type, java, Inner),
    format(string(Concrete), "List<~w>", [Inner]).
resolve_type(maybe(Type), java, Concrete) :-
    resolve_type(Type, java, Inner),
    format(string(Concrete), "Optional<~w>", [Inner]).
resolve_type(map(KeyType, ValueType), java, Concrete) :-
    resolve_type(KeyType, java, Key),
    resolve_type(ValueType, java, Value),
    format(string(Concrete), "Map<~w, ~w>", [Key, Value]).
resolve_type(set(Type), java, Concrete) :-
    resolve_type(Type, java, Inner),
    format(string(Concrete), "Set<~w>", [Inner]).
resolve_type(pair(LeftType, RightType), java, Concrete) :-
    resolve_type(LeftType, java, Left),
    resolve_type(RightType, java, Right),
    format(string(Concrete), "Map.Entry<~w, ~w>", [Left, Right]).
resolve_type(record(Name, _Fields), java, Concrete) :-
    atom_string(Name, Concrete).

resolve_type(atom, rust, "String").
resolve_type(string, rust, "String").
resolve_type(integer, rust, "i64").
resolve_type(float, rust, "f64").
resolve_type(number, rust, "f64").
resolve_type(boolean, rust, "bool").
resolve_type(any, rust, "serde_json::Value").
resolve_type(list(Type), rust, Concrete) :-
    resolve_type(Type, rust, Inner),
    format(string(Concrete), "Vec<~w>", [Inner]).
resolve_type(maybe(Type), rust, Concrete) :-
    resolve_type(Type, rust, Inner),
    format(string(Concrete), "Option<~w>", [Inner]).
resolve_type(map(KeyType, ValueType), rust, Concrete) :-
    resolve_type(KeyType, rust, Key),
    resolve_type(ValueType, rust, Value),
    format(string(Concrete), "HashMap<~w, ~w>", [Key, Value]).
resolve_type(set(Type), rust, Concrete) :-
    resolve_type(Type, rust, Inner),
    format(string(Concrete), "HashSet<~w>", [Inner]).
resolve_type(pair(LeftType, RightType), rust, Concrete) :-
    resolve_type(LeftType, rust, Left),
    resolve_type(RightType, rust, Right),
    format(string(Concrete), "(~w, ~w)", [Left, Right]).
resolve_type(record(Name, _Fields), rust, Concrete) :-
    atom_string(Name, Concrete).

resolve_type(atom, typr, "char").
resolve_type(string, typr, "char").
resolve_type(integer, typr, "int").
resolve_type(float, typr, "num").
resolve_type(number, typr, "num").
resolve_type(boolean, typr, "bool").
resolve_type(any, typr, "Any").
resolve_type(list(Type), typr, Concrete) :-
    resolve_type(Type, typr, Inner),
    format(string(Concrete), "[#N, ~w]", [Inner]).
resolve_type(maybe(Type), typr, Concrete) :-
    resolve_type(Type, typr, Inner),
    format(string(Concrete), "Option<~w>", [Inner]).
resolve_type(map(KeyType, ValueType), typr, Concrete) :-
    resolve_type(KeyType, typr, Key),
    resolve_type(ValueType, typr, Value),
    format(string(Concrete), "{ key: ~w, value: ~w}", [Key, Value]).
resolve_type(set(Type), typr, Concrete) :-
    resolve_type(Type, typr, Inner),
    format(string(Concrete), "[#N, ~w]", [Inner]).
resolve_type(pair(LeftType, RightType), typr, Concrete) :-
    resolve_type(LeftType, typr, Left),
    resolve_type(RightType, typr, Right),
    format(string(Concrete), "{ first: ~w, second: ~w}", [Left, Right]).
resolve_type(record(Name, _Fields), typr, Concrete) :-
    atom_string(Name, Concrete).

resolve_type(atom, r, "character").
resolve_type(string, r, "character").
resolve_type(integer, r, "integer").
resolve_type(float, r, "numeric").
resolve_type(number, r, "numeric").
resolve_type(boolean, r, "logical").
resolve_type(any, r, "ANY").
resolve_type(list(_), r, "list").
resolve_type(maybe(Type), r, Concrete) :-
    resolve_type(Type, r, Concrete).
resolve_type(map(_, _), r, "environment").
resolve_type(set(_), r, "list").
resolve_type(pair(_, _), r, "list").
resolve_type(record(Name, _Fields), r, Concrete) :-
    atom_string(Name, Concrete).

resolve_type(Type, TargetLang, Concrete) :-
    atom(Type),
    resolve_domain_type(Type, Resolved),
    Resolved \== Type,
    resolve_type(Resolved, TargetLang, Concrete).

edge_type_string(haskell, Left, Right, EdgeType) :-
    format(string(EdgeType), "(~w, ~w)", [Left, Right]).
edge_type_string(java, Left, Right, EdgeType) :-
    format(string(EdgeType), "Map.Entry<~w,~w>", [Left, Right]).
edge_type_string(rust, Left, Right, EdgeType) :-
    format(string(EdgeType), "(~w, ~w)", [Left, Right]).
edge_type_string(typr, Left, Right, EdgeType) :-
    format(string(EdgeType), "Tuple<~w, ~w>", [Left, Right]).
edge_type_string(r, _Left, _Right, "list").
edge_type_string(_, Left, Right, EdgeType) :-
    format(string(EdgeType), "(~w, ~w)", [Left, Right]).

resolve_domain_type(Type, Resolved) :-
    atom(Type),
    declared_uw_domain_type(Type, DomainType),
    !,
    resolve_domain_type(DomainType, Resolved).
resolve_domain_type(Type, Type).

normalize_pred_spec(_Module:PredSpec, PredSpec) :-
    !.
normalize_pred_spec(PredSpec, PredSpec).

declared_uw_type(PredSpec, ArgIndex, Type) :-
    declaration_uw_type_module(Module),
    clause(Module:uw_type(PredSpec, ArgIndex, Type), true).

declared_uw_return_type(PredSpec, Type) :-
    declaration_uw_return_type_module(Module),
    clause(Module:uw_return_type(PredSpec, Type), true).

declared_uw_typed_mode(PredSpec, Mode) :-
    declaration_uw_typed_mode_module(Module),
    clause(Module:uw_typed_mode(PredSpec, Mode), true).

declared_uw_domain_type(TypeName, DomainType) :-
    declaration_uw_domain_type_module(Module),
    clause(Module:uw_domain_type(TypeName, DomainType), true).

declaration_uw_type_module(type_declarations) :-
    module_has_local_uw_type_clauses(type_declarations).
declaration_uw_type_module(user) :-
    module_has_local_uw_type_clauses(user).
declaration_uw_type_module(Module) :-
    current_module(Module),
    Module \= type_declarations,
    Module \= user,
    module_has_local_uw_type_clauses(Module).

declaration_uw_typed_mode_module(type_declarations) :-
    module_has_local_uw_typed_mode_clauses(type_declarations).
declaration_uw_typed_mode_module(user) :-
    module_has_local_uw_typed_mode_clauses(user).
declaration_uw_typed_mode_module(Module) :-
    current_module(Module),
    Module \= type_declarations,
    Module \= user,
    module_has_local_uw_typed_mode_clauses(Module).

declaration_uw_return_type_module(type_declarations) :-
    module_has_local_uw_return_type_clauses(type_declarations).
declaration_uw_return_type_module(user) :-
    module_has_local_uw_return_type_clauses(user).
declaration_uw_return_type_module(Module) :-
    current_module(Module),
    Module \= type_declarations,
    Module \= user,
    module_has_local_uw_return_type_clauses(Module).

declaration_uw_domain_type_module(type_declarations) :-
    module_has_local_uw_domain_type_clauses(type_declarations).
declaration_uw_domain_type_module(user) :-
    module_has_local_uw_domain_type_clauses(user).
declaration_uw_domain_type_module(Module) :-
    current_module(Module),
    Module \= type_declarations,
    Module \= user,
    module_has_local_uw_domain_type_clauses(Module).

module_has_local_uw_type_clauses(Module) :-
    predicate_property(Module:uw_type(_, _, _), number_of_clauses(Count)),
    Count > 0,
    \+ predicate_property(Module:uw_type(_, _, _), imported_from(_)).

module_has_local_uw_typed_mode_clauses(Module) :-
    predicate_property(Module:uw_typed_mode(_, _), number_of_clauses(Count)),
    Count > 0,
    \+ predicate_property(Module:uw_typed_mode(_, _), imported_from(_)).

module_has_local_uw_return_type_clauses(Module) :-
    predicate_property(Module:uw_return_type(_, _), number_of_clauses(Count)),
    Count > 0,
    \+ predicate_property(Module:uw_return_type(_, _), imported_from(_)).

module_has_local_uw_domain_type_clauses(Module) :-
    predicate_property(Module:uw_domain_type(_, _), number_of_clauses(Count)),
    Count > 0,
    \+ predicate_property(Module:uw_domain_type(_, _), imported_from(_)).
