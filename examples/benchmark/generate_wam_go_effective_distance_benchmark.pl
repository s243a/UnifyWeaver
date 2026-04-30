:- module(generate_wam_go_effective_distance_benchmark, [main/0, generate/4]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_go_target').
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).

%% generate_wam_go_effective_distance_benchmark.pl
%%
%% Generates a Go hybrid-WAM benchmark for the effective-distance workload.
%%
%% Pipeline:
%%   1. Load the effective-distance workload and the benchmark facts.
%%   2. Generate optimized predicates via prolog_target (accumulated variant).
%%   3. Force the selected predicates through the shared Go WAM path.
%%   4. Emit a Go main package with a benchmark driver that queries the VM.
%%
%% Usage:
%%   swipl -q -s generate_wam_go_effective_distance_benchmark.pl -- \
%%       <facts.pl> <output-dir> [accumulated] [kernels_on|kernels_off]

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputDir, VariantAtom, KernelModeAtom]
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom]
    ->  KernelModeAtom = kernels_on
    ;   Argv = [FactsPath, OutputDir]
    ->  VariantAtom = accumulated,
        KernelModeAtom = kernels_on
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> [accumulated] [kernels_on|kernels_off]~n',
            []),
        halt(1)
    ),
    generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    parse_variant(VariantAtom, OptimizationOptions),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode0),
    replace_all_atoms(ScriptCode0, 'main :-', 'generated_main :-', ScriptCode),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),
    load_files(FactsPath, [silent(true)]),
    collect_wam_predicates(user, KernelModeAtom, Predicates),
    collect_article_categories(user, ArticleCategories),
    collect_roots(user, Roots),
    append([
        [module_name('wam-go-effective-distance-bench'),
         package_name(main),
         prefer_wam(true),
         wam_fallback(true),
         foreign_lowering(true),
         emit_mode(functions),
         parallel(true)],
        KernelOptions
    ], Options),
    write_wam_go_project(Predicates, Options, OutputDir),
    extract_shared_start_pc(OutputDir,
        "category_ancestor$effective_distance_sum_selected/3",
        WeightPC),
    % Resolve dimension_n at codegen time so user:dimension_n/1 reaches
    % the generated Go code (was previously hardcoded to 5.0 in main.go).
    wam_go_target:resolve_dimension_n_go(Options, DimN),
    write_main_go(OutputDir, WeightPC, KernelModeAtom, ArticleCategories, Roots, DimN),
    format(user_error,
           '[WAM-Go-EffectiveDistance] variant=~w kernels=~w output=~w~n',
           [VariantAtom, KernelModeAtom, OutputDir]).

parse_variant(accumulated, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false),
    seeded_accumulation(auto)
]).

parse_kernel_mode(kernels_on, []).
parse_kernel_mode(kernels_off, [no_kernels(true)]).

collect_wam_predicates(Module, kernels_on, [
    Module:dimension_n/1,
    Module:max_depth/1,
    Module:category_ancestor/4,
    Module:'category_ancestor$power_sum_bound'/3,
    Module:'category_ancestor$power_sum_selected'/3,
    Module:'category_ancestor$effective_distance_sum_selected'/3,
    Module:'category_ancestor$effective_distance_sum_bound'/3
]).

collect_wam_predicates(Module, kernels_off, [
    Module:dimension_n/1,
    Module:max_depth/1,
    Module:category_parent/2,
    Module:category_ancestor/4,
    Module:'category_ancestor$power_sum_bound'/3,
    Module:'category_ancestor$power_sum_selected'/3,
    Module:'category_ancestor$effective_distance_sum_selected'/3,
    Module:'category_ancestor$effective_distance_sum_bound'/3
]).

collect_article_categories(Module, Pairs) :-
    findall(Article-Category,
        Module:article_category(Article, Category),
        RawPairs),
    sort(RawPairs, Pairs).

collect_roots(Module, Roots) :-
    findall(Root, Module:root_category(Root), RawRoots),
    sort(RawRoots, Roots).

extract_shared_start_pc(OutputDir, Label, PC) :-
    directory_file_path(OutputDir, 'lib.go', LibPath),
    read_file_to_string(LibPath, Content, []),
    format(string(Needle), '"~w":', [Label]),
    sub_string(Content, Start, _, _, Needle),
    After is Start + string_length(Needle),
    sub_string(Content, After, _, _, Rest0),
    string_trim_left(Rest0, Rest),
    string_digits_prefix(Rest, Digits),
    number_string(PC, Digits).

string_trim_left(In, Out) :-
    string_codes(In, Codes),
    drop_ws(Codes, Trimmed),
    string_codes(Out, Trimmed).

drop_ws([C|Rest], Out) :-
    code_type(C, space),
    !,
    drop_ws(Rest, Out).
drop_ws(Codes, Codes).

string_digits_prefix(In, Digits) :-
    string_codes(In, Codes),
    take_digits(Codes, DigitCodes),
    DigitCodes \= [],
    string_codes(Digits, DigitCodes).

take_digits([C|Rest], [C|Digits]) :-
    code_type(C, digit),
    !,
    take_digits(Rest, Digits).
take_digits(_, []).

write_main_go(OutputDir, WeightPC, KernelModeAtom, ArticleCategories, Roots, DimN) :-
    go_article_categories_literal(ArticleCategories, ArticleCategoriesLiteral),
    go_string_slice_literal(Roots, RootsLiteral),
    make_directory_path(OutputDir),
    directory_file_path(OutputDir, 'main.go', MainPath),
    format(string(Code),
'package main

import (
    "fmt"
    "math"
    "os"
    "sort"
    "time"
)

const (
    effectiveDistanceWeightStartPC = ~w
)

var benchmarkArticleCategories = []articleCategoryPair{
~w
}

var benchmarkRoots = []string{~w}

type articleCategoryPair struct {
    Article string
    Category string
}

type resultRow struct {
    Article string
    Root string
    Distance float64
}

func newBenchmarkVM(startPC int, args ...Value) *WamState {
    vm := NewWamState(sharedWamCode, sharedWamLabels)
    setupSharedForeignPredicates(vm)
    vm.PC = startPC
    for i, arg := range args {
        vm.Regs[fmt.Sprintf("A%%d", i+1)] = arg
    }
    return vm
}

func collectSolutions(startPC int, args ...Value) [][]Value {
    vm := newBenchmarkVM(startPC, args...)
    solutions := make([][]Value, 0)
    if !vm.Run() {
        return solutions
    }
    solutions = append(solutions, append([]Value(nil), vm.CollectResults()...))
    for vm.backtrack() {
        if !vm.Run() {
            break
        }
        solutions = append(solutions, append([]Value(nil), vm.CollectResults()...))
    }
    return solutions
}

func atomName(v Value) string {
    if a, ok := v.(*Atom); ok {
        return a.Name
    }
    return vmString(v)
}

func vmString(v Value) string {
    if v == nil {
        return ""
    }
    return v.String()
}

func floatValue(v Value) (float64, bool) {
    switch t := v.(type) {
    case *Float:
        return t.Val, true
    case *Integer:
        return float64(t.Val), true
    default:
        return 0, false
    }
}

func weightSumForCategoryRoot(category string, root string) (float64, bool) {
    rows := collectSolutions(
        effectiveDistanceWeightStartPC,
        &Atom{Name: category},
        &Atom{Name: root},
        &Unbound{Name: "weight"},
    )
    if len(rows) == 0 || len(rows[0]) < 3 {
        return 0, false
    }
    return floatValue(rows[0][2])
}

func computeResults(root string, articleCats []articleCategoryPair) ([]resultRow, int, int) {
    uniqueSeeds := make(map[string]struct{})
    for _, pair := range articleCats {
        uniqueSeeds[pair.Category] = struct{}{}
    }
    seedWeights := make(map[string]float64)
    tupleCount := 0
    for seed := range uniqueSeeds {
        if seed == root {
            continue
        }
        if weight, ok := weightSumForCategoryRoot(seed, root); ok && weight > 0 {
            seedWeights[seed] = weight
            tupleCount++
        }
    }

    articleWeightSums := make(map[string]float64)
    for _, pair := range articleCats {
        if pair.Category == root {
            articleWeightSums[pair.Article] += 1.0
            continue
        }
        if weight, ok := seedWeights[pair.Category]; ok {
            articleWeightSums[pair.Article] += weight
        }
    }

    n := float64(~w)
    invN := -1.0 / n
    rows := make([]resultRow, 0, len(articleWeightSums))
    for article, weightSum := range articleWeightSums {
        if weightSum <= 0 {
            continue
        }
        rows = append(rows, resultRow{
            Article: article,
            Root: root,
            Distance: math.Pow(weightSum, invN),
        })
    }
    sort.Slice(rows, func(i, j int) bool {
        if rows[i].Distance != rows[j].Distance {
            return rows[i].Distance < rows[j].Distance
        }
        return rows[i].Article < rows[j].Article
    })
    return rows, len(uniqueSeeds), tupleCount
}

func main() {
    started := time.Now()
    roots := append([]string(nil), benchmarkRoots...)
    articleCats := append([]articleCategoryPair(nil), benchmarkArticleCategories...)
    loadMs := time.Since(started).Milliseconds()
    if len(roots) == 0 {
        fmt.Fprintln(os.Stderr, "no root categories")
        os.Exit(1)
    }
    root := roots[0]

    queryStart := time.Now()
    rows, seedCount, tupleCount := computeResults(root, articleCats)
    queryMs := time.Since(queryStart).Milliseconds()

    aggregationStart := time.Now()
    sort.Slice(rows, func(i, j int) bool {
        if rows[i].Distance != rows[j].Distance {
            return rows[i].Distance < rows[j].Distance
        }
        return rows[i].Article < rows[j].Article
    })
    aggregationMs := time.Since(aggregationStart).Milliseconds()
    totalMs := time.Since(started).Milliseconds()

    fmt.Println("article\\troot_category\\teffective_distance")
    for _, row := range rows {
        fmt.Printf("%%s\\t%%s\\t%%.6f\\n", row.Article, row.Root, row.Distance)
    }

    fmt.Fprintf(os.Stderr, "mode=accumulated_go_wam\\n")
    fmt.Fprintf(os.Stderr, "kernel_mode=~w\\n")
    fmt.Fprintf(os.Stderr, "load_ms=%%d\\n", loadMs)
    fmt.Fprintf(os.Stderr, "query_ms=%%d\\n", queryMs)
    fmt.Fprintf(os.Stderr, "aggregation_ms=%%d\\n", aggregationMs)
    fmt.Fprintf(os.Stderr, "total_ms=%%d\\n", totalMs)
    fmt.Fprintf(os.Stderr, "seed_count=%%d\\n", seedCount)
    fmt.Fprintf(os.Stderr, "tuple_count=%%d\\n", tupleCount)
    fmt.Fprintf(os.Stderr, "article_count=%%d\\n", len(rows))
}
', [WeightPC, ArticleCategoriesLiteral, RootsLiteral, DimN, KernelModeAtom]),
    setup_call_cleanup(
        open(MainPath, write, Stream),
        format(Stream, '~w', [Code]),
        close(Stream)
    ).

go_article_categories_literal([], '').
go_article_categories_literal(Pairs, Literal) :-
    maplist(go_article_category_entry, Pairs, Entries),
    atomic_list_concat(Entries, ",\n", Literal).

go_article_category_entry(Article-Category, Entry) :-
    escape_go_string(Article, EscArticle),
    escape_go_string(Category, EscCategory),
    format(atom(Entry), '    {Article: "~w", Category: "~w"}', [EscArticle, EscCategory]).

go_string_slice_literal([], '').
go_string_slice_literal(Strings, Literal) :-
    maplist(go_string_literal, Strings, Quoted),
    atomic_list_concat(Quoted, ', ', Literal).

go_string_literal(String, Literal) :-
    escape_go_string(String, Escaped),
    format(atom(Literal), '"~w"', [Escaped]).

replace_all_atoms(Input, Search, Replace, Output) :-
    (   sub_atom(Input, Before, _, After, Search)
    ->  sub_atom(Input, 0, Before, _, Prefix),
        sub_atom(Input, _, After, 0, Suffix),
        atom_concat(Prefix, Replace, Tmp),
        atom_concat(Tmp, Suffix, Next),
        replace_all_atoms(Next, Search, Replace, Output)
    ;   Output = Input
    ).

escape_go_string(Input, Escaped) :-
    atom_chars(Input, Chars),
    phrase(go_escaped_chars(Chars), EscChars),
    atom_chars(Escaped, EscChars).

go_escaped_chars([]) --> [].
go_escaped_chars(['\\'|Rest]) --> ['\\','\\'], go_escaped_chars(Rest).
go_escaped_chars(['"'|Rest]) --> ['\\','"'], go_escaped_chars(Rest).
go_escaped_chars(['\n'|Rest]) --> ['\\','n'], go_escaped_chars(Rest).
go_escaped_chars(['\t'|Rest]) --> ['\\','t'], go_escaped_chars(Rest).
go_escaped_chars([Char|Rest]) --> [Char], go_escaped_chars(Rest).
