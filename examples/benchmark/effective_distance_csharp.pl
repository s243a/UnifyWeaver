%% ==========================================================================
%% Cross-Target Effective Distance Benchmark — C# Query Engine version
%%
%% The C# parameterized query engine handles is/2 arithmetic and cycle
%% detection (via semi-naive FixpointNode) natively. Two adjustments
%% from the generic Prolog version:
%%
%% 1. No explicit Visited list — FixpointNode handles cycle detection
%%    automatically via delta-set convergence with HashSet deduplication.
%%
%% 2. Base case constant moved from head to body — the query plan
%%    compiler currently requires variables in head positions, with
%%    binding done via is/2 in the body. So instead of:
%%      category_ancestor(Cat, Parent, 1) :- category_parent(Cat, Parent).
%%    we write:
%%      category_ancestor(Cat, Parent, Hops) :- category_parent(Cat, Parent), Hops is 1.
%%
%%    POTENTIAL BUG / FUTURE WORK: The plan compiler should ideally handle
%%    constants in head positions of recursive predicates by implicitly
%%    generating a SelectionNode (filtering on the constant column) or
%%    inlining the value into the projection. This would allow the natural
%%    Prolog idiom without the workaround. Filed as future enhancement for
%%    the parameterized query engine's head-pattern compilation.
%%
%% 3. Mode declarations required — the parameterized query engine needs
%%    user:mode/1 declarations to identify input (+) vs output (-) args.
%%
%% Usage:
%%   swipl -l examples/benchmark/effective_distance_csharp.pl \
%%         -l data/benchmark/dev/facts.pl \
%%         -g "use_module('src/unifyweaver/targets/csharp_target'),
%%             compile_predicate_to_csharp(category_ancestor/3,
%%                 [target(csharp_query)], Code),
%%             write(Code)" -t halt
%% ==========================================================================

:- discontiguous article_category/2.
:- discontiguous category_parent/2.
:- discontiguous root_category/1.

%% Mode declarations for parameterized query engine
:- dynamic user:mode/1.
:- assert(user:mode(category_ancestor(+, -, -))).

%% category_ancestor(+Cat, -Ancestor, -Hops)
%% Transitive closure over category_parent/2 with hop counter.
%% FixpointNode handles cycle detection automatically.
%%
%% Note: base case uses "Hops is 1" instead of head constant "1"
%% because the C# query plan compiler requires variables in head args.
category_ancestor(Cat, Parent, Hops) :-
    category_parent(Cat, Parent),
    Hops is 1.

category_ancestor(Cat, Ancestor, Hops) :-
    category_parent(Cat, Mid),
    category_ancestor(Mid, Ancestor, H1),
    Hops is H1 + 1.

%% path_to_root(+Article, -Root, -Hops)
path_to_root(Article, Root, Hops) :-
    article_category(Article, Cat),
    root_category(Cat),
    Root = Cat,
    Hops is 1.

path_to_root(Article, Root, Hops) :-
    article_category(Article, Cat),
    category_ancestor(Cat, AncestorCat, CatHops),
    root_category(AncestorCat),
    Root = AncestorCat,
    Hops is CatHops + 1.

%% effective_distance(+Article, -Root, -Deff)
effective_distance(Article, Root, Deff) :-
    setof(A, C^article_category(A, C), Articles),
    member(Article, Articles),
    root_category(Root),
    aggregate_all(sum(W),
        (path_to_root(Article, Root, Hops),
         W is Hops ** (-5)),
        WeightSum),
    WeightSum > 0,
    Deff is WeightSum ** (-0.2).
