:- module(generate_wam_r_effective_distance_benchmark,
          [ main/0,
            generate/3,
            generate/4,
            generate/5
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module('../../src/unifyweaver/targets/wam_r_target').
:- use_module('../../src/unifyweaver/core/template_system', [render_template/3]).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(option)).
:- use_module(library(lists)).
:- use_module(library(pairs), [group_pairs_by_key/2]).
:- use_module(library(readutil), [read_file_to_string/3]).

%% generate_wam_r_effective_distance_benchmark.pl
%%
%% Result-producing hybrid WAM R effective-distance benchmark.
%%
%% Pipeline:
%%   1. Load effective_distance.pl + scale facts.
%%   2. Generate accumulated helpers via prolog_target.
%%   3. Emit write_wam_r_project/3 with emit_mode(functions) (primary),
%%      auto-detected category_ancestor/4 kernel, and category_parent/2
%%      FactSource (grouped_by_first TSV).
%%   4. Append a Mustache companion runner that enumerates article x root
%%      and prints TSV + query_ms/total_ms.
%%
%% Usage:
%%   swipl -q -s generate_wam_r_effective_distance_benchmark.pl -- \
%%       <facts.pl> <output-dir> [kernels_on|kernels_off] [functions|interpreter]

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [FactsPath, OutputDir, KernelModeAtom, EmitModeAtom]
    ->  true
    ;   Argv = [FactsPath, OutputDir, KernelModeAtom]
    ->  EmitModeAtom = functions
    ;   Argv = [FactsPath, OutputDir]
    ->  KernelModeAtom = kernels_on,
        EmitModeAtom = functions
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> [kernels_on|kernels_off] [functions|interpreter]~n',
            []),
        halt(1)
    ),
    (   Argv == []
    ->  true
    ;   generate(FactsPath, OutputDir, KernelModeAtom, EmitModeAtom),
        halt(0)
    ).

main :-
    format(user_error, 'Error: R WAM effective-distance generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, KernelModeAtom) :-
    generate(FactsPath, OutputDir, KernelModeAtom, functions).

generate(FactsPath, OutputDir, KernelModeAtom, EmitModeAtom) :-
    generate(FactsPath, OutputDir, KernelModeAtom, EmitModeAtom, accumulated).

generate(FactsPath, OutputDir, KernelModeAtom, EmitModeAtom, VariantAtom) :-
    must_exist_file(FactsPath),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    parse_emit_mode(EmitModeAtom, EmitMode),
    parse_variant(VariantAtom, OptimizationOptions),
    benchmark_workload_path(WorkloadPath),
    reset_benchmark_predicates,
    user:load_files(WorkloadPath, [silent(true), if(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode0),
    replace_all_atoms(ScriptCode0, 'main :-', 'generated_main :-', ScriptCode),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    user:load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),
    cleanup_generated_script_entrypoint,
    user:load_files(FactsPath, [silent(true), if(true)]),
    collect_wam_predicates(KernelModeAtom, Predicates),
    write_category_parent_grouped_tsv(OutputDir, GroupedPath),
    benchmark_atoms(Atoms),
    resolve_dimension_n(DimN),
    append([
        [ module_name('wam-r-effective-distance-bench'),
          emit_mode(EmitMode),
          intern_atoms(Atoms),
          r_fact_sources([source(category_parent/2, grouped_by_first(GroupedPath))])
        ],
        KernelOptions
    ], Options),
    write_wam_r_project(Predicates, Options, OutputDir),
    write_effective_distance_runner(OutputDir, KernelModeAtom, EmitMode, DimN),
    format(user_error,
           '[WAM-R-EffectiveDistance] kernels=~w emit=~w output=~w~n',
           [KernelModeAtom, EmitMode, OutputDir]).

parse_variant(accumulated, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false),
    seeded_accumulation(auto)
]).

parse_kernel_mode(kernels_on, []).
parse_kernel_mode(kernels_off, [kernel_layout(off)]).

parse_emit_mode(functions, functions).
parse_emit_mode(interpreter, interpreter).

collect_wam_predicates(kernels_on, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_parent/2,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

collect_wam_predicates(kernels_off, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_parent/2,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

reset_benchmark_predicates :-
    forall(member(Name/Arity,
                  [ article_category/2,
                    category_parent/2,
                    root_category/1,
                    dimension_n/1,
                    max_depth/1,
                    category_ancestor/4,
                    'category_ancestor$power_sum_bound'/3,
                    'category_ancestor$power_sum_selected'/3,
                    'category_ancestor$effective_distance_sum_selected'/3,
                    'category_ancestor$effective_distance_sum_bound'/3,
                    power_sum_bound/4,
                    main/0,
                    generated_main/0
                  ]),
           (   current_predicate(user:Name/Arity)
           ->  abolish(user:Name/Arity)
           ;   true
           )).

cleanup_generated_script_entrypoint :-
    (   current_predicate(user:main/0)
    ->  abolish(user:main/0)
    ;   true
    ),
    (   current_predicate(user:generated_main/0)
    ->  abolish(user:generated_main/0)
    ;   true
    ).

%% replace_all_atoms(+Input, +Search, +Replace, -Output)
%  Recurses on the suffix only so Replace containing Search terminates.
replace_all_atoms(Input, Search, Replace, Output) :-
    (   sub_atom(Input, Before, _, After, Search)
    ->  sub_atom(Input, 0, Before, _, Prefix),
        sub_atom(Input, _, After, 0, Suffix),
        replace_all_atoms(Suffix, Search, Replace, NewSuffix),
        atom_concat(Prefix, Replace, Tmp),
        atom_concat(Tmp, NewSuffix, Output)
    ;   Output = Input
    ).

resolve_dimension_n(N) :-
    (   current_predicate(user:dimension_n/1),
        user:dimension_n(N0),
        number(N0)
    ->  N = N0
    ;   N = 5
    ).

write_category_parent_grouped_tsv(OutputDir, Path) :-
    directory_file_path(OutputDir, 'data', DataDir),
    make_directory_path(DataDir),
    directory_file_path(DataDir, 'category_parent_by_child.tsv', Path),
    findall(Child-Parent, current_category_parent_fact(Child, Parent), Pairs0),
    sort(Pairs0, Pairs),
    group_pairs_by_key(Pairs, Grouped),
    setup_call_cleanup(
        open(Path, write, Stream),
        forall(member(Child-Parents, Grouped),
               (   format(Stream, '~w', [Child]),
                   forall(member(Parent, Parents),
                          format(Stream, '\t~w', [Parent])),
                   nl(Stream)
               )),
        close(Stream)).

current_category_parent_fact(Child, Parent) :-
    current_predicate(user:category_parent/2),
    call(user:category_parent(Child, Parent)).

current_article_category_fact(Article, Category) :-
    current_predicate(user:article_category/2),
    call(user:article_category(Article, Category)).

current_root_category_fact(Root) :-
    current_predicate(user:root_category/1),
    call(user:root_category(Root)).

benchmark_atoms(Atoms) :-
    findall(Atom, benchmark_atom(Atom), Bag),
    sort(Bag, Atoms).

benchmark_atom(Atom) :-
    current_category_parent_fact(Child, Parent),
    member(Atom, [Child, Parent]).
benchmark_atom(Atom) :-
    current_article_category_fact(Article, Category),
    member(Atom, [Article, Category]).
benchmark_atom(Atom) :-
    current_root_category_fact(Atom).

write_effective_distance_runner(OutputDir, KernelModeAtom, EmitMode, DimN) :-
    find_r_ed_template(Template),
    get_time(T), format_time(string(DateStr), "%Y-%m-%d", T),
    atom_string(KernelModeAtom, KernelModeStr),
    atom_string(EmitMode, EmitModeStr),
    render_template(Template,
        [ 'date'=DateStr,
          'kernel_mode'=KernelModeStr,
          'emit_mode'=EmitModeStr,
          'dimension_n'=DimN
        ], Content),
    directory_file_path(OutputDir, 'R', RDir),
    make_directory_path(RDir),
    directory_file_path(RDir, 'run_effective_distance.R', Path),
    setup_call_cleanup(
        open(Path, write, Stream),
        write(Stream, Content),
        close(Stream)).

find_r_ed_template(Template) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, BenchDir),
    file_directory_name(BenchDir, ExamplesDir),
    file_directory_name(ExamplesDir, RepoRoot),
    directory_file_path(RepoRoot,
        'templates/targets/r_wam/run_effective_distance.R.mustache', AbsPath),
    read_file_to_string(AbsPath, Template, []).

must_exist_file(Path) :-
    (   exists_file(Path)
    ->  true
    ;   throw(error(existence_error(file, Path), _))
    ).
