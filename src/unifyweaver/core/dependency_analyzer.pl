:- module(dependency_analyzer, [
    find_dependencies/2
]).

:- use_module(library(lists)).

%% find_dependencies(+PredicateIndicator, -Dependencies)
%  Find all predicate dependencies for a given predicate.
%  PredicateIndicator is in the form Functor/Arity.
find_dependencies(Functor/Arity, Dependencies) :-
    functor(Head, Functor, Arity),
    findall(Body, clause(Head, Body), Bodies),
    find_all_body_dependencies(Bodies, Functor, DepList),
    list_to_set(DepList, Dependencies).

%% find_all_body_dependencies(+Bodies, +OriginalFunctor, -Dependencies)
%  Find all dependencies in a list of bodies, excluding the original predicate itself.
find_all_body_dependencies([], _, []).
find_all_body_dependencies([Body|Rest], OriginalFunctor, Dependencies) :-
    body_dependencies(Body, BodyDeps),
    find_all_body_dependencies(Rest, OriginalFunctor, RestDeps),
    append(BodyDeps, RestDeps, AllDeps),
    exclude_original(AllDeps, OriginalFunctor, Dependencies).

%% body_dependencies(+Term, -Dependencies)
%  Recursively walk a term and find all predicate calls.
body_dependencies(Var, []) :- var(Var), !.
body_dependencies(Atom, []) :- atom(Atom), !.
body_dependencies(Number, []) :- number(Number), !.
body_dependencies((A, B), Deps) :-
    !, % Conjunction
    body_dependencies(A, ADeps),
    body_dependencies(B, BDeps),
    append(ADeps, BDeps, Deps).
body_dependencies((A; B), Deps) :-
    !, % Disjunction
    body_dependencies(A, ADeps),
    body_dependencies(B, BDeps),
    append(ADeps, BDeps, Deps).
body_dependencies((A -> B), Deps) :-
    !, % If-then
    body_dependencies(A, ADeps),
    body_dependencies(B, BDeps),
    append(ADeps, BDeps, Deps).
body_dependencies(\+ A, Deps) :-
    !, % Negation
    body_dependencies(A, Deps).
body_dependencies(Goal, [Functor/Arity]) :-
    compound(Goal),
    functor(Goal, Functor, Arity).

%% exclude_original(+AllDeps, +OriginalFunctor, -FilteredDeps)
%  Remove the original predicate from the list of dependencies to avoid self-recursion.
exclude_original([], _, []).
exclude_original([Functor/_|Rest], OriginalFunctor, Filtered) :-
    Functor == OriginalFunctor, !,
    exclude_original(Rest, OriginalFunctor, Filtered).
exclude_original([Dep|Rest], OriginalFunctor, [Dep|Filtered]) :-
    exclude_original(Rest, OriginalFunctor, Filtered).
