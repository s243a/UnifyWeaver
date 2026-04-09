:- encoding(utf8).
% generate_prolog_effective_semantic_distance_benchmark.pl
%
% Generates a self-contained Prolog benchmark that computes effective
% semantic distance using the power-mean formula with weighted edges.
%
% Usage:
%   swipl -q -s generate_prolog_effective_semantic_distance_benchmark.pl -- \
%       <facts.pl> <edge_weights.pl> <output.pl> [N]
%
% N defaults to 5 (graph dimensionality).

:- use_module(library(lists)).
:- initialization(main, main).

main :-
    current_prolog_flag(argv, Args),
    (   Args = [FactsFile, WeightsFile, OutputFile, NStr]
    ->  atom_number(NStr, N)
    ;   Args = [FactsFile, WeightsFile, OutputFile]
    ->  N = 5
    ;   format(user_error, 'Usage: swipl ... -- <facts.pl> <edge_weights.pl> <output.pl> [N]~n', []),
        halt(1)
    ),
    generate_benchmark(FactsFile, WeightsFile, OutputFile, N).

generate_benchmark(FactsFile, WeightsFile, OutputFile, N) :-
    format(user_error, 'Generating effective semantic distance benchmark (N=~w)...~n', [N]),
    consult(FactsFile),
    consult(WeightsFile),

    findall(A-C, user:article_category(A, C), ArticleCats),
    findall(R, user:root_category(R), Roots),
    findall(edge_weight(C, P, W), user:edge_weight(C, P, W), WeightFacts),
    length(ArticleCats, NAC), length(Roots, NR), length(WeightFacts, NW),
    format(user_error, '  ~w article-category, ~w roots, ~w weighted edges~n', [NAC, NR, NW]),

    open(OutputFile, write, S),
    format(S, '%% Auto-generated effective semantic distance benchmark (N=~w)~n', [N]),
    format(S, '%% d_eff = (Sum d_i^(-N))^(-1/N) with semantic edge weights~n~n', []),
    format(S, ':- use_module(library(lists)).~n', []),
    format(S, ':- use_module(library(aggregate)).~n~n', []),

    % Write facts
    forall(member(A-C, ArticleCats),
        format(S, 'article_category(~q, ~q).~n', [A, C])),
    format(S, '~n', []),
    forall(member(R, Roots),
        format(S, 'root_category(~q).~n', [R])),
    format(S, '~n', []),
    forall(member(edge_weight(C, P, W), WeightFacts),
        format(S, 'edge_weight(~q, ~q, ~w).~n', [C, P, W])),
    format(S, '~n', []),

    % Write solver
    format(S, 'semantic_path_cost(X, Y, _V, W) :-~n', []),
    format(S, '    edge_weight(X, Y, W).~n', []),
    format(S, 'semantic_path_cost(X, Y, V, Cost) :-~n', []),
    format(S, '    edge_weight(X, Z, W),~n', []),
    format(S, '    \\+ member(Z, V),~n', []),
    format(S, '    semantic_path_cost(Z, Y, [Z|V], RC),~n', []),
    format(S, '    Cost is W + RC.~n~n', []),

    format(S, 'effective_semantic_dist(Start, Target, Deff) :-~n', []),
    format(S, '    NegN is -~w,~n', [N]),
    format(S, '    aggregate_all(sum(W),~n', []),
    format(S, '        ( semantic_path_cost(Start, Target, [Start], Cost),~n', []),
    format(S, '          W is Cost ** NegN ),~n', []),
    format(S, '        WeightSum),~n', []),
    format(S, '    WeightSum > 0,~n', []),
    format(S, '    InvN is -1 / ~w,~n', [N]),
    format(S, '    Deff is WeightSum ** InvN.~n~n', []),

    % Write main
    write(S, ':- initialization((\n'),
    write(S, '    forall(\n'),
    write(S, '        ( article_category(Article, Cat),\n'),
    write(S, '          root_category(Root),\n'),
    write(S, '          ( effective_semantic_dist(Cat, Root, Dist)\n'),
    write(S, '          -> true\n'),
    write(S, '          ;  Dist = inf\n'),
    write(S, '          )\n'),
    write(S, '        ),\n'),
    write(S, '        format("~w\\t~w\\t~w\\n", [Article, Root, Dist])\n'),
    write(S, '    ),\n'),
    write(S, '    halt\n'),
    write(S, '), main).\n'),

    close(S),
    format(user_error, '  Wrote: ~w~n', [OutputFile]).
