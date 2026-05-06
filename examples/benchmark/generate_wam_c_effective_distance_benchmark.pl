:- module(generate_wam_c_effective_distance_benchmark,
          [ main/0,
            generate/3,
            generate/4
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_c_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module(library(lists)).
:- use_module(library(pairs), [pairs_keys/2, pairs_values/2]).

%% generate_wam_c_effective_distance_benchmark.pl
%%
%% Generates a small WAM-C effective-distance benchmark project. The generated
%% C runner loads category_parent/2 facts from TSV, executes the native
%% category_ancestor/4 all-hop collector when kernels are enabled, and includes
%% a C reference DFS path for kernels_off comparisons.
%%
%% Usage:
%%   swipl -q -s examples/benchmark/generate_wam_c_effective_distance_benchmark.pl -- \
%%       <facts.pl> <output-dir> [kernels_on|kernels_off]

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [FactsPath, OutputDir, KernelModeAtom]
    ->  generate(FactsPath, OutputDir, KernelModeAtom),
        halt(0)
    ;   Argv = [FactsPath, OutputDir]
    ->  generate(FactsPath, OutputDir, kernels_on),
        halt(0)
    ;   format(user_error,
               'Usage: ... -- <facts.pl> <output-dir> [kernels_on|kernels_off]~n',
               []),
        halt(1)
    ).

main :-
    format(user_error, 'Error: WAM-C effective-distance generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, KernelMode) :-
    generate(FactsPath, OutputDir, KernelMode, []).

generate(FactsPath, OutputDir, KernelMode, _Options) :-
    (   exists_file(FactsPath)
    ->  true
    ;   throw(error(existence_error(source_sink, FactsPath), _))
    ),
    parse_kernel_mode(KernelMode, ParsedMode),
    reset_benchmark_predicates,
    benchmark_workload_path(WorkloadPath),
    load_files(user:WorkloadPath, [silent(true), if(true)]),
    load_files(user:FactsPath, [silent(true), if(true)]),
    collect_benchmark_facts(CategoryParents, ArticleCategories, RootCategories),
    CategoryParents \= [],
    ArticleCategories \= [],
    RootCategories \= [],
    benchmark_dimension(Dimension),
    benchmark_max_depth(MaxDepth),
    make_directory_path(OutputDir),
    compile_wam_runtime_to_c([], RuntimeCode),
    directory_file_path(OutputDir, 'wam_runtime.c', RuntimePath),
    write_text_file(RuntimePath, RuntimeCode),
    WamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, WamCode, [], PredCode),
    format(atom(LibCode), '#include "wam_runtime.h"~n~n~w~n', [PredCode]),
    directory_file_path(OutputDir, 'lib.c', LibPath),
    write_text_file(LibPath, LibCode),
    category_parent_tsv(CategoryParents, CategoryTsv),
    directory_file_path(OutputDir, 'category_parent.tsv', CategoryPath),
    write_text_file(CategoryPath, CategoryTsv),
    effective_distance_main_code(ParsedMode, Dimension, MaxDepth,
                                 ArticleCategories, RootCategories, MainCode),
    directory_file_path(OutputDir, 'main.c', MainPath),
    write_text_file(MainPath, MainCode),
    directory_file_path(OutputDir, 'README.md', ReadmePath),
    effective_distance_readme(ParsedMode, Readme),
    write_text_file(ReadmePath, Readme),
    format(user_error,
           '[WAM-C-EffectiveDistance] mode=~w output=~w~n',
           [ParsedMode, OutputDir]).

parse_kernel_mode(kernels_on, kernels_on).
parse_kernel_mode(kernels_off, kernels_off).
parse_kernel_mode("kernels_on", kernels_on).
parse_kernel_mode("kernels_off", kernels_off).
parse_kernel_mode(Atom, Mode) :-
    atom(Atom),
    atom_string(Atom, String),
    parse_kernel_mode(String, Mode), !.
parse_kernel_mode(Atom, _) :-
    throw(error(domain_error(wam_c_kernel_mode, Atom), _)).

reset_benchmark_predicates :-
    forall(member(Name/Arity,
                  [ article_category/2,
                    category_parent/2,
                    root_category/1,
                    dimension_n/1,
                    max_depth/1
                  ]),
           maybe_abolish_user_predicate(Name/Arity)).

maybe_abolish_user_predicate(Name/Arity) :-
    (   current_predicate(user:Name/Arity)
    ->  abolish(user:Name/Arity)
    ;   true
    ).

collect_benchmark_facts(CategoryParents, ArticleCategories, RootCategories) :-
    findall(Child-Parent, user:category_parent(Child, Parent), CategoryParents0),
    sort(CategoryParents0, CategoryParents),
    findall(Article-Category, user:article_category(Article, Category), ArticleCategories0),
    sort(ArticleCategories0, ArticleCategories),
    findall(Root, user:root_category(Root), RootCategories0),
    sort(RootCategories0, RootCategories).

benchmark_dimension(Dimension) :-
    (   current_predicate(user:dimension_n/1),
        once(user:dimension_n(D0)),
        integer(D0),
        D0 > 0
    ->  Dimension = D0
    ;   Dimension = 5
    ).

benchmark_max_depth(MaxDepth) :-
    (   current_predicate(user:max_depth/1),
        once(user:max_depth(M0)),
        integer(M0),
        M0 > 0
    ->  MaxDepth = M0
    ;   MaxDepth = 10
    ).

category_parent_tsv(Pairs, Tsv) :-
    maplist(category_parent_tsv_line, Pairs, Lines),
    atomic_list_concat(Lines, '', Tsv).

category_parent_tsv_line(Child-Parent, Line) :-
    format(atom(Line), '~w\t~w\n', [Child, Parent]).

effective_distance_main_code(KernelMode, Dimension, MaxDepth,
                             ArticleCategories, RootCategories, Code) :-
    c_pair_arrays('ARTICLE_IDS', 'ARTICLE_CATS', ArticleCategories, ArticleArrays),
    pairs_keys(ArticleCategories, ArticleIds0),
    sort(ArticleIds0, ArticleIds),
    c_string_array('ARTICLE_COUNT', 'ARTICLES', ArticleIds, ArticlesArray),
    c_string_array('ROOT_COUNT', 'ROOTS', RootCategories, RootArray),
    kernel_mode_flag(KernelMode, KernelFlag),
    format(atom(Code),
'#include <math.h>
#include <stdio.h>
#include <string.h>
#include "wam_runtime.h"

void setup_category_ancestor_4(WamState* state);

~w
~w
~w

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    WamValue list;
    list.tag = VAL_LIST;
    list.data.ref_addr = state->H;
    state->H_array[state->H++] = val_atom(atom);
    state->H_array[state->H++] = val_atom("[]");
    return list;
}

static int visited_contains(const char **visited, int visited_len, const char *atom) {
    for (int i = 0; i < visited_len; i++) {
        if (strcmp(visited[i], atom) == 0) return 1;
    }
    return 0;
}

static int collect_reference_hops(WamFactSource *source,
                                  const char *cat,
                                  const char *root,
                                  int depth,
                                  int max_depth,
                                  const char **visited,
                                  int visited_len,
                                  WamIntResults *results) {
    int found = 0;
    for (int i = 0; i < source->edge_count; i++) {
        CategoryEdge *edge = &source->edges[i];
        if (strcmp(edge->child, cat) != 0) continue;
        if (visited_contains(visited, visited_len, edge->parent)) continue;
        if (strcmp(edge->parent, root) == 0) {
            if (!wam_int_results_push(results, depth + 1)) return 0;
            found = 1;
        }
    }
    if (visited_len >= max_depth || visited_len >= 64) return found;
    for (int i = 0; i < source->edge_count; i++) {
        CategoryEdge *edge = &source->edges[i];
        if (strcmp(edge->child, cat) != 0) continue;
        if (visited_contains(visited, visited_len, edge->parent)) continue;
        visited[visited_len] = edge->parent;
        if (collect_reference_hops(source, edge->parent, root, depth + 1,
                                   max_depth, visited, visited_len + 1,
                                   results)) {
            found = 1;
        }
    }
    return found;
}

static void collect_hops(WamState *state,
                         WamFactSource *source,
                         const char *cat,
                         const char *root,
                         int kernels_on,
                         WamIntResults *results) {
    if (kernels_on) {
        state->A[0] = val_atom(cat);
        state->A[1] = val_atom(root);
        state->A[2] = val_unbound("Hops");
        state->A[3] = make_visited_singleton(state, cat);
        (void)wam_collect_category_ancestor_hops(state, results);
    } else {
        const char *visited[64];
        visited[0] = cat;
        (void)collect_reference_hops(source, cat, root, 0, ~w, visited, 1, results);
    }
}

int main(void) {
    WamState state;
    WamFactSource source;
    wam_state_init(&state);
    wam_fact_source_init(&source);
    setup_category_ancestor_4(&state);

    if (!wam_fact_source_load_tsv(&state, &source, "category_parent.tsv")) {
        fprintf(stderr, "failed to load category_parent.tsv\\n");
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 1;
    }
    wam_register_category_parent_fact_source(&state, &source);
    wam_register_category_ancestor_kernel(&state, "category_ancestor/4", ~w);

    printf("article\\troot_category\\teffective_distance\\n");
    for (int ai = 0; ai < ARTICLE_COUNT; ai++) {
        for (int ri = 0; ri < ROOT_COUNT; ri++) {
            double weight_sum = 0.0;
            for (int ci = 0; ci < ARTICLE_CATEGORY_COUNT; ci++) {
                if (strcmp(ARTICLE_IDS[ci], ARTICLES[ai]) != 0) continue;
                WamIntResults hops;
                wam_int_results_init(&hops);
                if (strcmp(ARTICLE_CATS[ci], ROOTS[ri]) == 0) {
                    wam_int_results_push(&hops, 1);
                } else {
                    collect_hops(&state, &source, ARTICLE_CATS[ci], ROOTS[ri], ~w, &hops);
                    for (int hi = 0; hi < hops.count; hi++) {
                        hops.values[hi] += 1;
                    }
                }
                for (int hi = 0; hi < hops.count; hi++) {
                    weight_sum += pow((double)hops.values[hi], -~w.0);
                }
                wam_int_results_close(&hops);
            }
            if (weight_sum > 0.0) {
                double deff = pow(weight_sum, -1.0 / ~w.0);
                printf("%s\\t%s\\t%.6f\\n", ARTICLES[ai], ROOTS[ri], deff);
            }
        }
    }

    wam_fact_source_close(&source);
    wam_free_state(&state);
    return 0;
}
', [ArticleArrays, ArticlesArray, RootArray, MaxDepth, MaxDepth, KernelFlag, Dimension, Dimension]).

kernel_mode_flag(kernels_on, 1).
kernel_mode_flag(kernels_off, 0).

c_pair_arrays(NamesName, ValuesName, Pairs, Code) :-
    pairs_keys(Pairs, Keys),
    pairs_values(Pairs, Values),
    c_string_initializer(Keys, KeyInit),
    c_string_initializer(Values, ValueInit),
    length(Pairs, Count),
    format(atom(Code),
           'static const int ARTICLE_CATEGORY_COUNT = ~w;\nstatic const char *~w[] = { ~w };\nstatic const char *~w[] = { ~w };',
           [Count, NamesName, KeyInit, ValuesName, ValueInit]).

c_string_array(CountName, Name, Values, Code) :-
    c_string_initializer(Values, Init),
    length(Values, Count),
    format(atom(Code),
           'static const int ~w = ~w;\nstatic const char *~w[] = { ~w };',
           [CountName, Count, Name, Init]).

c_string_initializer(Values, Init) :-
    maplist(c_string_literal, Values, Literals),
    atomic_list_concat(Literals, ', ', Init).

c_string_literal(Value, Literal) :-
    format(atom(Raw), '~w', [Value]),
    atom_chars(Raw, Chars),
    phrase(c_escaped_chars(Chars), Escaped),
    atom_chars(EscapedAtom, Escaped),
    format(atom(Literal), '"~w"', [EscapedAtom]).

c_escaped_chars([]) --> [].
c_escaped_chars(['\\'|Rest]) --> ['\\', '\\'], c_escaped_chars(Rest).
c_escaped_chars(['"'|Rest]) --> ['\\', '"'], c_escaped_chars(Rest).
c_escaped_chars([C|Rest]) --> [C], c_escaped_chars(Rest).

effective_distance_readme(KernelMode, Readme) :-
    format(atom(Readme),
'# WAM-C Effective Distance Benchmark

Generated by `generate_wam_c_effective_distance_benchmark.pl`.

Build:

```sh
gcc -std=c11 -Wall -Wextra -I ../../src/unifyweaver/targets/wam_c_runtime wam_runtime.c lib.c main.c -lm -o wam_c_effective_distance
```

Run:

```sh
./wam_c_effective_distance
```

Kernel mode: `~w`.
', [KernelMode]).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).
