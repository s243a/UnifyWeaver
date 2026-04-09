:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% generate_prolog_min_semantic_distance_benchmark.pl
%
% Generates a self-contained Prolog benchmark that computes minimum
% semantic distance from articles to root categories using precomputed
% edge weights.
%
% Usage:
%   swipl -q -s generate_prolog_min_semantic_distance_benchmark.pl -- \
%       <facts.pl> <edge_weights.pl> <output.pl>

:- use_module(library(lists)).
:- use_module(library(option)).

:- initialization(main, main).

main :-
    current_prolog_flag(argv, Args),
    (   Args = [FactsFile, WeightsFile, OutputFile]
    ->  generate_benchmark(FactsFile, WeightsFile, OutputFile)
    ;   format(user_error, 'Usage: swipl ... -- <facts.pl> <edge_weights.pl> <output.pl>~n', []),
        halt(1)
    ).

generate_benchmark(FactsFile, WeightsFile, OutputFile) :-
    format(user_error, 'Generating min semantic distance benchmark...~n', []),

    % Read facts
    load_facts(FactsFile),
    format(user_error, '  Facts loaded~n', []),

    % Read edge weights
    load_weights(WeightsFile),
    format(user_error, '  Edge weights loaded~n', []),

    % Collect unique data
    findall(A-C, user:article_category(A, C), ArticleCats),
    findall(R, user:root_category(R), Roots),
    findall(edge_weight(C, P, W), user:edge_weight(C, P, W), WeightFacts),
    length(ArticleCats, NAC),
    length(Roots, NR),
    length(WeightFacts, NW),
    format(user_error, '  ~w article-category pairs, ~w roots, ~w weighted edges~n',
           [NAC, NR, NW]),

    % Write output
    open(OutputFile, write, S),
    write_header(S),
    write_facts(S, ArticleCats, Roots, WeightFacts),
    write_solver(S),
    write_main(S),
    close(S),
    format(user_error, '  Wrote: ~w~n', [OutputFile]).

load_facts(File) :-
    (   exists_file(File)
    ->  consult(File)
    ;   format(user_error, 'ERROR: ~w not found~n', [File]),
        halt(1)
    ).

load_weights(File) :-
    (   exists_file(File)
    ->  consult(File)
    ;   format(user_error, 'ERROR: ~w not found~n', [File]),
        halt(1)
    ).

write_header(S) :-
    format(S, '%% Auto-generated min semantic distance benchmark~n', []),
    format(S, '%% Computes shortest weighted path from article categories to roots~n', []),
    format(S, '%% Edge weights = 1 - cosine_similarity(embedding(from), embedding(to))~n~n', []),
    format(S, ':- use_module(library(lists)).~n~n', []).

write_facts(S, ArticleCats, Roots, WeightFacts) :-
    format(S, '%% Article-category assignments~n', []),
    forall(member(A-C, ArticleCats),
        format(S, 'article_category(~q, ~q).~n', [A, C])),
    format(S, '~n%% Root categories~n', []),
    forall(member(R, Roots),
        format(S, 'root_category(~q).~n', [R])),
    format(S, '~n%% Semantic edge weights~n', []),
    forall(member(edge_weight(C, P, W), WeightFacts),
        format(S, 'edge_weight(~q, ~q, ~w).~n', [C, P, W])),
    format(S, '~n', []).

write_solver(S) :-
    format(S, '%% Min semantic distance solver~n', []),
    format(S, '%% Uses Dijkstra-like exploration via aggregate_all(min)~n~n', []),
    format(S, 'semantic_path_cost(X, Y, _Visited, W) :-~n', []),
    format(S, '    edge_weight(X, Y, W).~n~n', []),
    format(S, 'semantic_path_cost(X, Y, Visited, Cost) :-~n', []),
    format(S, '    edge_weight(X, Z, W),~n', []),
    format(S, '    \\+ member(Z, Visited),~n', []),
    format(S, '    semantic_path_cost(Z, Y, [Z|Visited], RestCost),~n', []),
    format(S, '    Cost is W + RestCost.~n~n', []),
    format(S, 'min_semantic_dist(Start, Target, MinDist) :-~n', []),
    format(S, '    aggregate_all(min(Cost),~n', []),
    format(S, '        semantic_path_cost(Start, Target, [Start], Cost),~n', []),
    format(S, '        MinDist).~n~n', []).

write_main(S) :-
    format(S, '%% Benchmark entry point~n', []),
    format(S, ':- initialization((\n', []),
    format(S, '    forall(\n', []),
    format(S, '        ( article_category(Article, Cat),\n', []),
    format(S, '          root_category(Root),\n', []),
    format(S, '          ( min_semantic_dist(Cat, Root, Dist)\n', []),
    format(S, '          -> true\n', []),
    format(S, '          ;  Dist = inf\n', []),
    format(S, '          )\n', []),
    format(S, '        ),\n', []),
    write(S, '        format("~w\\t~w\\t~w\\n", [Article, Root, Dist])\n'),
    format(S, '    ),\n', []),
    format(S, '    halt\n', []),
    format(S, '), main).\n', []).
