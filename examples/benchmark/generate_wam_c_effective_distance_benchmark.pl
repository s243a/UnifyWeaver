:- module(generate_wam_c_effective_distance_benchmark,
          [ main/0,
            generate/3,
            generate/4
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_c_target').
:- use_module('../../src/unifyweaver/core/cost_model', [resolve_csr_io_policy/2]).
:- use_module(library(assoc), [get_assoc/3, list_to_assoc/2]).
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).
:- use_module(library(lists)).
:- use_module(library(pairs), [pairs_keys/2, pairs_values/2]).

%% generate_wam_c_effective_distance_benchmark.pl
%%
%% Generates a small WAM-C effective-distance benchmark project. The generated
%% C runner loads category_parent/2 facts from TSV or LMDB, executes the native
%% category_ancestor/4 all-hop collector when kernels are enabled, and includes
%% a C reference DFS path for kernels_off comparisons.
%%
%% Usage:
%%   swipl -q -s examples/benchmark/generate_wam_c_effective_distance_benchmark.pl -- \
%%       <facts.pl> <output-dir> [kernels_on|kernels_off] [facts_tsv|facts_lmdb] [layout_profile]
%%
%% Layout profiles:
%%   parent_only                - parent-path query only (default)
%%   child_scan                 - bounded child search over the loaded parent facts
%%   child_csr_sorted           - bounded child search using sorted-array reverse CSR
%%   child_csr_buffered_drop    - sorted-array reverse CSR with buffered pread/drop
%%   child_csr_lmdb_offset      - reverse CSR values plus LMDB row-offset lookup

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [FactsPath, OutputDir, KernelModeAtom, FactStorageAtom, LayoutProfileAtom]
    ->  parse_layout_profile(LayoutProfileAtom, LayoutProfileOptions),
        generate(FactsPath, OutputDir, KernelModeAtom,
                 [fact_storage(FactStorageAtom)|LayoutProfileOptions]),
        halt(0)
    ;   Argv = [FactsPath, OutputDir, KernelModeAtom, FactStorageAtom]
    ->  generate(FactsPath, OutputDir, KernelModeAtom, FactStorageAtom),
        halt(0)
    ;   Argv = [FactsPath, OutputDir, KernelModeAtom]
    ->  generate(FactsPath, OutputDir, KernelModeAtom, facts_tsv),
        halt(0)
    ;   Argv = [FactsPath, OutputDir]
    ->  generate(FactsPath, OutputDir, kernels_on),
        halt(0)
    ;   format(user_error,
               'Usage: ... -- <facts.pl> <output-dir> [kernels_on|kernels_off] [facts_tsv|facts_lmdb] [layout_profile]~n',
               []),
        halt(1)
    ).

main :-
    format(user_error, 'Error: WAM-C effective-distance generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, KernelMode) :-
    generate(FactsPath, OutputDir, KernelMode, []).

generate(FactsPath, OutputDir, KernelMode, OptionsOrFactStorage) :-
    (   exists_file(FactsPath)
    ->  true
    ;   throw(error(existence_error(source_sink, FactsPath), _))
    ),
    parse_kernel_mode(KernelMode, ParsedMode),
    parse_fact_storage_option(OptionsOrFactStorage, FactStorage),
    parse_child_search_options(OptionsOrFactStorage, ChildSearch),
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
    category_parent_tsv(CategoryParents, CategoryTsv),
    directory_file_path(OutputDir, 'category_parent.tsv', CategoryPath),
    write_text_file(CategoryPath, CategoryTsv),
    maybe_write_lmdb_seeder(FactStorage, OutputDir, CategoryParents),
    effective_distance_reverse_index_options(OptionsOrFactStorage, OutputDir,
                                             CategoryParents, ArticleCategories,
                                             RootCategories, ReverseIndexOptions),
    effective_distance_predicate_lib_code(ChildSearch, ReverseIndexOptions, LibCode),
    directory_file_path(OutputDir, 'lib.c', LibPath),
    write_text_file(LibPath, LibCode),
    effective_distance_main_code(ParsedMode, FactStorage, ChildSearch,
                                 ReverseIndexOptions, Dimension, MaxDepth,
                                 ArticleCategories, RootCategories, MainCode),
    directory_file_path(OutputDir, 'main.c', MainPath),
    write_text_file(MainPath, MainCode),
    directory_file_path(OutputDir, 'README.md', ReadmePath),
    effective_distance_readme(ParsedMode, FactStorage, ReverseIndexOptions, Readme),
    write_text_file(ReadmePath, Readme),
    format(user_error,
           '[WAM-C-EffectiveDistance] mode=~w fact_storage=~w output=~w~n',
           [ParsedMode, FactStorage, OutputDir]).

effective_distance_predicate_lib_code(ChildSearch, ReverseIndexOptions, LibCode) :-
    CategoryWamCode = 'category_ancestor/4:\n    call_foreign category_ancestor/4, 4\n    proceed',
    compile_wam_predicate_to_c(user:category_ancestor/4, CategoryWamCode, [], CategoryPredCode),
    (   child_search_enabled(ChildSearch)
    ->  BidirWamCode = 'bidirectional_ancestor/5:\n    call_foreign bidirectional_ancestor/5, 5\n    proceed',
        compile_wam_predicate_to_c(user:bidirectional_ancestor/5, BidirWamCode, [], BidirPredCode),
        format(atom(PredCode), '~w~n~n~w', [CategoryPredCode, BidirPredCode])
    ;   PredCode = CategoryPredCode
    ),
    generate_setup_reverse_index_c(ReverseIndexOptions, ReverseIndexCode),
    format(atom(LibCode), '#include "wam_runtime.h"~n~n~w~n~n~w~n',
           [ReverseIndexCode, PredCode]).

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

parse_fact_storage_option([], facts_tsv).
parse_fact_storage_option(Options, FactStorage) :-
    is_list(Options),
    !,
    (   memberchk(fact_storage(Raw), Options)
    ->  parse_fact_storage(Raw, FactStorage)
    ;   memberchk(Raw, Options),
        parse_fact_storage(Raw, FactStorage)
    ->  true
    ;   FactStorage = facts_tsv
    ).
parse_fact_storage_option(Raw, FactStorage) :-
    parse_fact_storage(Raw, FactStorage).

parse_fact_storage(facts_tsv, facts_tsv).
parse_fact_storage(facts_lmdb, facts_lmdb).
parse_fact_storage("facts_tsv", facts_tsv).
parse_fact_storage("facts_lmdb", facts_lmdb).
parse_fact_storage(Atom, Mode) :-
    atom(Atom),
    atom_string(Atom, String),
    parse_fact_storage(String, Mode), !.
parse_fact_storage(Atom, _) :-
    throw(error(domain_error(wam_c_fact_storage, Atom), _)).

parse_layout_profile(parent_only, []).
parse_layout_profile(child_scan, Options) :-
    child_search_layout_options(Options).
parse_layout_profile(child_csr_sorted, Options) :-
    child_search_layout_options(ChildOptions),
    append(ChildOptions,
           [reverse_index(csr([index_backend(sorted_array),
                               io_policy(buffered_pread)]))],
           Options).
parse_layout_profile(child_csr_buffered_drop, Options) :-
    child_search_layout_options(ChildOptions),
    append(ChildOptions,
           [reverse_index(csr([index_backend(sorted_array),
                               io_policy(buffered_pread_drop)]))],
           Options).
parse_layout_profile(child_csr_lmdb_offset, Options) :-
    child_search_layout_options(ChildOptions),
    append(ChildOptions,
           [reverse_index(csr([index_backend(lmdb_offset),
                               io_policy(buffered_pread)]))],
           Options).
parse_layout_profile("parent_only", Options) :-
    parse_layout_profile(parent_only, Options).
parse_layout_profile("child_scan", Options) :-
    parse_layout_profile(child_scan, Options).
parse_layout_profile("child_csr_sorted", Options) :-
    parse_layout_profile(child_csr_sorted, Options).
parse_layout_profile("child_csr_buffered_drop", Options) :-
    parse_layout_profile(child_csr_buffered_drop, Options).
parse_layout_profile("child_csr_lmdb_offset", Options) :-
    parse_layout_profile(child_csr_lmdb_offset, Options).
parse_layout_profile(Atom, Options) :-
    atom(Atom),
    atom_string(Atom, String),
    parse_layout_profile(String, Options), !.
parse_layout_profile(Atom, _) :-
    throw(error(domain_error(wam_c_layout_profile, Atom), _)).

child_search_layout_options([
    child_search(bounded),
    max_child_expansions(8),
    child_search_depth(1),
    parent_step_cost(1.0),
    child_step_cost(3.0),
    child_search_budget(10.0)
]).

parse_child_search_options(Options, ChildSearch) :-
    is_list(Options),
    !,
    (   memberchk(child_search(Raw), Options)
    ->  parse_child_search(Raw, Mode)
    ;   Mode = parent_only
    ),
    option_or_default(max_child_expansions, Options, 0, MaxChildren),
    option_or_default(child_search_depth, Options, 1, ChildDepth),
    option_or_default(parent_step_cost, Options, 1.0, ParentCost),
    option_or_default(child_step_cost, Options, 3.0, ChildCost),
    option_or_default(child_search_budget, Options, 10.0, Budget),
    validate_nonnegative_int(max_child_expansions, MaxChildren),
    validate_nonnegative_int(child_search_depth, ChildDepth),
    validate_positive_number(parent_step_cost, ParentCost),
    validate_positive_number(child_step_cost, ChildCost),
    validate_positive_number(child_search_budget, Budget),
    ChildSearch = child_search(Mode, MaxChildren, ChildDepth,
                               ParentCost, ChildCost, Budget).
parse_child_search_options(_, child_search(parent_only, 0, 1, 1.0, 1.0, 1.0e100)).

parse_child_search(parent_only, parent_only).
parse_child_search(bounded, bounded).
parse_child_search("parent_only", parent_only).
parse_child_search("bounded", bounded).
parse_child_search(Atom, Mode) :-
    atom(Atom),
    atom_string(Atom, String),
    parse_child_search(String, Mode), !.
parse_child_search(Atom, _) :-
    throw(error(domain_error(wam_c_child_search, Atom), _)).

effective_distance_reverse_index_options(Options, OutputDir, CategoryParents,
                                         ArticleCategories, RootCategories,
                                         ReverseIndexOptions) :-
    is_list(Options),
    memberchk(reverse_index(csr(CsrOptions0)), Options),
    !,
    effective_distance_csr_index_backend(CsrOptions0, IndexBackend),
    effective_distance_csr_io_policy(CsrOptions0, IoPolicy),
    effective_distance_csr_block_size_options(CsrOptions0, BlockSizeOptions),
    write_effective_distance_reverse_csr(OutputDir, CategoryParents,
                                         ArticleCategories, RootCategories,
                                         IndexBackend, IoPolicy, CategoryIdMap),
    append([
        storage_kind(csr_pread_artifact),
        phase(runtime_available),
        index_backend(IndexBackend),
        io_policy(IoPolicy)
    ], BlockSizeOptions, ArtifactOptions),
    ReverseIndexOptions = [
        reverse_index(artifact(ArtifactOptions)),
        reverse_csr_values_path('category_child.csr.val'),
        category_id_map(CategoryIdMap)
    | ReverseIndexPathOptions],
    effective_distance_reverse_index_path_options(IndexBackend, ReverseIndexPathOptions).
effective_distance_reverse_index_options(Options, _OutputDir, _CategoryParents,
                                         _ArticleCategories, _RootCategories,
                                         ReverseIndexOptions) :-
    is_list(Options),
    memberchk(reverse_index(artifact(ArtifactOptions)), Options),
    !,
    findall(Opt, effective_distance_reverse_artifact_option(Options, Opt), ExtraOptions),
    ReverseIndexOptions = [reverse_index(artifact(ArtifactOptions))|ExtraOptions].
effective_distance_reverse_index_options(_, _, _, _, _, []).

effective_distance_csr_index_backend(Options, Backend) :-
    must_be(list, Options),
    (   memberchk(index_backend(RawBackend), Options)
    ->  parse_effective_distance_csr_index_backend(RawBackend, Backend)
    ;   Backend = sorted_array
    ).

parse_effective_distance_csr_index_backend(sorted_array, sorted_array).
parse_effective_distance_csr_index_backend(lmdb_offset, lmdb_offset).
parse_effective_distance_csr_index_backend(Backend, _) :-
    throw(error(domain_error(wam_c_effective_distance_csr_index_backend, Backend), _)).

effective_distance_csr_io_policy(Options, IoPolicy) :-
    must_be(list, Options),
    resolve_csr_io_policy(Options, IoPolicy).

effective_distance_csr_block_size_options(Options, [block_size_edges(BlockSizeEdges)]) :-
    memberchk(block_size_edges(BlockSizeEdges), Options),
    !.
effective_distance_csr_block_size_options(_, []).

effective_distance_reverse_index_path_options(sorted_array,
                                             [reverse_csr_index_path('category_child.csr.idx')]).
effective_distance_reverse_index_path_options(lmdb_offset,
                                             [ reverse_csr_offset_lmdb_path('category_child.csr.offsets.lmdb'),
                                               reverse_csr_offset_lmdb_dbi(offsets)
                                             ]).

effective_distance_reverse_artifact_option(Options, reverse_csr_index_path(Path)) :-
    memberchk(reverse_csr_index_path(Path), Options).
effective_distance_reverse_artifact_option(Options, reverse_csr_values_path(Path)) :-
    memberchk(reverse_csr_values_path(Path), Options).
effective_distance_reverse_artifact_option(Options, reverse_csr_offset_lmdb_path(Path)) :-
    memberchk(reverse_csr_offset_lmdb_path(Path), Options).
effective_distance_reverse_artifact_option(Options, reverse_csr_offset_lmdb_dbi(Dbi)) :-
    memberchk(reverse_csr_offset_lmdb_dbi(Dbi), Options).
effective_distance_reverse_artifact_option(Options, category_id_map(Map)) :-
    memberchk(category_id_map(Map), Options).

option_or_default(Key, Options, Default, Value) :-
    Term =.. [Key, Value0],
    (   memberchk(Term, Options)
    ->  Value = Value0
    ;   Value = Default
    ).

validate_nonnegative_int(_Key, Value) :-
    integer(Value),
    Value >= 0,
    !.
validate_nonnegative_int(Key, Value) :-
    throw(error(domain_error(nonnegative_integer_option(Key), Value), _)).

validate_positive_number(_Key, Value) :-
    number(Value),
    Value > 0,
    !.
validate_positive_number(Key, Value) :-
    throw(error(domain_error(positive_number_option(Key), Value), _)).

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

write_effective_distance_reverse_csr(OutputDir, CategoryParents,
                                     ArticleCategories, RootCategories,
                                     IndexBackend, IoPolicy, CategoryIdMap) :-
    effective_distance_category_id_map(CategoryParents, ArticleCategories,
                                       RootCategories, CategoryIdMap),
    list_to_assoc(CategoryIdMap, CategoryIdsByAtom),
    directory_file_path(OutputDir, 'category_child.csr.idx', IndexPath),
    directory_file_path(OutputDir, 'category_child.csr.val', ValuesPath),
    maplist(category_parent_child_id_pair(CategoryIdsByAtom),
            CategoryParents,
            ChildPairs0),
    sort(ChildPairs0, ChildPairs),
    group_children_by_parent(ChildPairs, Rows),
    write_reverse_csr_files(IndexPath, ValuesPath, Rows),
    maybe_pad_reverse_csr_values(IoPolicy, ValuesPath),
    maybe_write_reverse_csr_offset_lmdb_seeder(IndexBackend, OutputDir, Rows).

category_parent_child_id_pair(CategoryIdsByAtom, Child-Parent, ParentId-ChildId) :-
    get_assoc(Child, CategoryIdsByAtom, ChildId),
    get_assoc(Parent, CategoryIdsByAtom, ParentId).

effective_distance_category_id_map(CategoryParents, ArticleCategories,
                                   RootCategories, CategoryIdMap) :-
    findall(Category,
            (   member(Child-Parent, CategoryParents),
                (Category = Child ; Category = Parent)
            ;   member(_Article-ArticleCategory, ArticleCategories),
                Category = ArticleCategory
            ;   member(Root, RootCategories),
                Category = Root
            ),
            Categories0),
    sort(Categories0, Categories),
    assign_category_ids(Categories, 1, CategoryIdMap).

assign_category_ids([], _NextId, []).
assign_category_ids([Category|Rest], Id, [Category-Id|MappedRest]) :-
    NextId is Id + 1,
    assign_category_ids(Rest, NextId, MappedRest).

group_children_by_parent([], []).
group_children_by_parent([Parent-Child|Rest], [csr_row(Parent, Children)|Rows]) :-
    take_parent_children(Rest, Parent, [Child], Remaining, Children0),
    sort(Children0, Children),
    group_children_by_parent(Remaining, Rows).

take_parent_children([Parent-Child|Rest], Parent, Acc, Remaining, Children) :-
    !,
    take_parent_children(Rest, Parent, [Child|Acc], Remaining, Children).
take_parent_children(Remaining, _Parent, Children, Remaining, Children).

write_reverse_csr_files(IndexPath, ValuesPath, Rows) :-
    setup_call_cleanup(
        open(IndexPath, write, IndexStream, [type(binary)]),
        setup_call_cleanup(
            open(ValuesPath, write, ValuesStream, [type(binary)]),
            write_reverse_csr_rows(Rows, 0, IndexStream, ValuesStream),
            close(ValuesStream)
        ),
        close(IndexStream)
    ).

write_reverse_csr_rows([], _Offset, _IndexStream, _ValuesStream).
write_reverse_csr_rows([csr_row(ParentId, Children)|Rows], Offset,
                       IndexStream, ValuesStream) :-
    length(Children, Count),
    put_i32_le(IndexStream, ParentId),
    put_u64_le(IndexStream, Offset),
    put_u32_le(IndexStream, Count),
    forall(member(ChildId, Children), put_i32_le(ValuesStream, ChildId)),
    NextOffset is Offset + Count,
    write_reverse_csr_rows(Rows, NextOffset, IndexStream, ValuesStream).

maybe_pad_reverse_csr_values(direct_io, ValuesPath) :-
    !,
    pad_file_to_alignment(ValuesPath, 4096).
maybe_pad_reverse_csr_values(_, _).

pad_file_to_alignment(Path, Alignment) :-
    size_file(Path, Size),
    Pad is (Alignment - (Size mod Alignment)) mod Alignment,
    (   Pad =:= 0
    ->  true
    ;   setup_call_cleanup(
            open(Path, append, Stream, [type(binary)]),
            forall(between(1, Pad, _), put_byte(Stream, 0)),
            close(Stream)
        )
    ).

maybe_write_reverse_csr_offset_lmdb_seeder(sorted_array, _OutputDir, _Rows).
maybe_write_reverse_csr_offset_lmdb_seeder(lmdb_offset, OutputDir, Rows) :-
    length(Rows, RowCount),
    reverse_csr_offset_lmdb_seeder_code(RowCount, SeederCode),
    directory_file_path(OutputDir, 'seed_category_child_csr_offsets_lmdb.c', SeederPath),
    write_text_file(SeederPath, SeederCode).

reverse_csr_offset_lmdb_seeder_code(ExpectedRows, Code) :-
    MapSize is max(ExpectedRows * 128, 1048576),
    format(atom(Code),
'#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "lmdb.h"

static int mkdir_if_needed(const char *path) {
    if (mkdir(path, 0775) == 0 || errno == EEXIST) return 1;
    return 0;
}

static int32_t read_i32_le(const unsigned char *p) {
    uint32_t v = ((uint32_t)p[0]) |
                 ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) |
                 ((uint32_t)p[3] << 24);
    return (int32_t)v;
}

static uint64_t read_u64_le(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)p[i]) << (8 * i);
    return v;
}

static uint32_t read_u32_le(const unsigned char *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void write_i32_le(unsigned char *p, int32_t value) {
    uint32_t v = (uint32_t)value;
    p[0] = (unsigned char)(v & 0xffu);
    p[1] = (unsigned char)((v >> 8) & 0xffu);
    p[2] = (unsigned char)((v >> 16) & 0xffu);
    p[3] = (unsigned char)((v >> 24) & 0xffu);
}

static void write_offset_record(unsigned char *p, uint64_t offset_edges, uint32_t child_count) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)((offset_edges >> (8 * i)) & 0xffu);
    p[8] = (unsigned char)(child_count & 0xffu);
    p[9] = (unsigned char)((child_count >> 8) & 0xffu);
    p[10] = (unsigned char)((child_count >> 16) & 0xffu);
    p[11] = (unsigned char)((child_count >> 24) & 0xffu);
}

static int seed_offsets(MDB_txn *txn, MDB_dbi dbi, const char *idx_path, int *row_count) {
    FILE *idx = fopen(idx_path, "rb");
    if (!idx) return 0;
    unsigned char record[16];
    int ok = 1;
    *row_count = 0;
    while (fread(record, 1, sizeof(record), idx) == sizeof(record)) {
        int32_t parent = read_i32_le(record);
        uint64_t offset_edges = read_u64_le(record + 4);
        uint32_t child_count = read_u32_le(record + 12);
        unsigned char key_bytes[4];
        unsigned char data_bytes[12];
        MDB_val key;
        MDB_val data;
        write_i32_le(key_bytes, parent);
        write_offset_record(data_bytes, offset_edges, child_count);
        key.mv_size = sizeof(key_bytes);
        key.mv_data = key_bytes;
        data.mv_size = sizeof(data_bytes);
        data.mv_data = data_bytes;
        if (mdb_put(txn, dbi, &key, &data, 0) != MDB_SUCCESS) {
            ok = 0;
            break;
        }
        (*row_count)++;
    }
    if (ferror(idx)) ok = 0;
    fclose(idx);
    return ok;
}

int main(void) {
    const char *env_path = "category_child.csr.offsets.lmdb";
    const char *idx_path = "category_child.csr.idx";
    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int row_count = 0;
    int rc = 0;

    if (!mkdir_if_needed(env_path)) return 1;
    rc = mdb_env_create(&env);
    if (rc != MDB_SUCCESS) return 2;
    rc = mdb_env_set_maxdbs(env, 2);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 3; }
    rc = mdb_env_set_mapsize(env, ~w);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 4; }
    rc = mdb_env_open(env, env_path, 0, 0664);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 5; }
    rc = mdb_txn_begin(env, NULL, 0, &txn);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 6; }
    rc = mdb_dbi_open(txn, "offsets", MDB_CREATE, &dbi);
    if (rc != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        mdb_env_close(env);
        return 7;
    }
    if (!seed_offsets(txn, dbi, idx_path, &row_count)) {
        mdb_txn_abort(txn);
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
        return 8;
    }
    rc = mdb_txn_commit(txn);
    if (rc != MDB_SUCCESS) {
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
        return 9;
    }
    if (row_count != ~w) {
        fprintf(stderr, "expected ~w offset rows, wrote %%d\\n", row_count);
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
        return 10;
    }
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
    return 0;
}
', [MapSize, ExpectedRows, ExpectedRows]).

put_i32_le(Stream, Value) :-
    Value >= -2147483648,
    Value =< 2147483647,
    Unsigned is Value /\ 0xffffffff,
    put_u32_le(Stream, Unsigned).

put_u32_le(Stream, Value) :-
    Value >= 0,
    Value =< 0xffffffff,
    Byte0 is Value /\ 0xff,
    Byte1 is (Value >> 8) /\ 0xff,
    Byte2 is (Value >> 16) /\ 0xff,
    Byte3 is (Value >> 24) /\ 0xff,
    put_byte(Stream, Byte0),
    put_byte(Stream, Byte1),
    put_byte(Stream, Byte2),
    put_byte(Stream, Byte3).

put_u64_le(Stream, Value) :-
    Value >= 0,
    Value =< 0xffffffffffffffff,
    Byte0 is Value /\ 0xff,
    Byte1 is (Value >> 8) /\ 0xff,
    Byte2 is (Value >> 16) /\ 0xff,
    Byte3 is (Value >> 24) /\ 0xff,
    Byte4 is (Value >> 32) /\ 0xff,
    Byte5 is (Value >> 40) /\ 0xff,
    Byte6 is (Value >> 48) /\ 0xff,
    Byte7 is (Value >> 56) /\ 0xff,
    put_byte(Stream, Byte0),
    put_byte(Stream, Byte1),
    put_byte(Stream, Byte2),
    put_byte(Stream, Byte3),
    put_byte(Stream, Byte4),
    put_byte(Stream, Byte5),
    put_byte(Stream, Byte6),
    put_byte(Stream, Byte7).

maybe_write_lmdb_seeder(facts_tsv, _OutputDir, _CategoryParents).
maybe_write_lmdb_seeder(facts_lmdb, OutputDir, CategoryParents) :-
    length(CategoryParents, RowCount),
    has_duplicate_child(CategoryParents, HasDuplicate),
    lmdb_seeder_code(RowCount, HasDuplicate, SeederCode),
    directory_file_path(OutputDir, 'seed_category_parent_lmdb.c', SeederPath),
    write_text_file(SeederPath, SeederCode).

has_duplicate_child(Pairs, HasDuplicate) :-
    pairs_keys(Pairs, Children),
    msort(Children, Sorted),
    (   append(_, [Child, Child|_], Sorted)
    ->  HasDuplicate = 1
    ;   HasDuplicate = 0
    ).

lmdb_seeder_code(ExpectedRows, ExpectedDuplicate, Code) :-
    format(atom(Code),
'#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "lmdb.h"

static int mkdir_if_needed(const char *path) {
    if (mkdir(path, 0775) == 0 || errno == EEXIST) return 1;
    return 0;
}

static char *trim_right(char *s) {
    char *end = s + strlen(s);
    while (end > s && (end[-1] == 10 || end[-1] == 13 ||
                       end[-1] == 32 || end[-1] == 9)) {
        *--end = 0;
    }
    return s;
}

static int put_edge(MDB_txn *txn, MDB_dbi dbi, const char *child, const char *parent) {
    MDB_val key;
    MDB_val data;
    key.mv_size = strlen(child);
    key.mv_data = (void *)child;
    data.mv_size = strlen(parent);
    data.mv_data = (void *)parent;
    return mdb_put(txn, dbi, &key, &data, 0);
}

static int seed_from_tsv(MDB_env *env, MDB_dbi dbi, const char *tsv_path) {
    FILE *file = fopen(tsv_path, "r");
    if (!file) return 0;
    MDB_txn *txn = NULL;
    if (mdb_txn_begin(env, NULL, 0, &txn) != MDB_SUCCESS) {
        fclose(file);
        return 0;
    }
    char line[1024];
    int ok = 1;
    while (fgets(line, sizeof(line), file)) {
        char *child = line;
        while (*child == 32 || *child == 9) child++;
        if (*child == 0 || *child == 10 || *child == 35) continue;
        char *sep = strchr(child, 9);
        if (!sep) sep = strchr(child, 32);
        if (!sep) { ok = 0; break; }
        *sep = 0;
        char *parent = sep + 1;
        while (*parent == 32 || *parent == 9) parent++;
        trim_right(parent);
        if (*child == 0 || *parent == 0) { ok = 0; break; }
        if (put_edge(txn, dbi, child, parent) != MDB_SUCCESS) {
            ok = 0;
            break;
        }
    }
    fclose(file);
    if (!ok) {
        mdb_txn_abort(txn);
        return 0;
    }
    return mdb_txn_commit(txn) == MDB_SUCCESS;
}

static int validate_lmdb(MDB_env *env, MDB_dbi dbi) {
    MDB_txn *txn = NULL;
    MDB_cursor *cursor = NULL;
    MDB_val key;
    MDB_val data;
    int count = 0;
    int has_duplicate_key = 0;
    char previous_key[1024];
    size_t previous_size = 0;
    previous_key[0] = 0;

    if (mdb_txn_begin(env, NULL, MDB_RDONLY, &txn) != MDB_SUCCESS) return 0;
    if (mdb_cursor_open(txn, dbi, &cursor) != MDB_SUCCESS) {
        mdb_txn_abort(txn);
        return 0;
    }
    int rc = mdb_cursor_get(cursor, &key, &data, MDB_FIRST);
    while (rc == MDB_SUCCESS) {
        if (previous_size == key.mv_size &&
            key.mv_size < sizeof(previous_key) &&
            memcmp(previous_key, key.mv_data, key.mv_size) == 0) {
            has_duplicate_key = 1;
        }
        if (key.mv_size < sizeof(previous_key)) {
            memcpy(previous_key, key.mv_data, key.mv_size);
            previous_key[key.mv_size] = 0;
            previous_size = key.mv_size;
        } else {
            previous_size = 0;
        }
        count++;
        rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT);
    }
    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    return rc == MDB_NOTFOUND && count == ~w && (~w == 0 || has_duplicate_key);
}

int main(void) {
    const char *env_path = "category_parent.lmdb";
    if (!mkdir_if_needed(env_path)) {
        fprintf(stderr, "failed to create %%s\\n", env_path);
        return 1;
    }

    MDB_env *env = NULL;
    MDB_txn *txn = NULL;
    MDB_dbi dbi = 0;
    int rc = mdb_env_create(&env);
    if (rc != MDB_SUCCESS) return 2;
    rc = mdb_env_set_mapsize(env, 10485760);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 3; }
    rc = mdb_env_open(env, env_path, 0, 0664);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 4; }
    rc = mdb_txn_begin(env, NULL, 0, &txn);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 5; }
    rc = mdb_dbi_open(txn, NULL, MDB_CREATE | MDB_DUPSORT, &dbi);
    if (rc == MDB_SUCCESS) rc = mdb_txn_commit(txn);
    else mdb_txn_abort(txn);
    if (rc != MDB_SUCCESS) { mdb_env_close(env); return 6; }

    if (!seed_from_tsv(env, dbi, "category_parent.tsv")) {
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
        return 7;
    }
    if (!validate_lmdb(env, dbi)) {
        fprintf(stderr, "LMDB category_parent artifact validation failed\\n");
        mdb_dbi_close(env, dbi);
        mdb_env_close(env);
        return 8;
    }
    mdb_dbi_close(env, dbi);
    mdb_env_close(env);
    return 0;
}
', [ExpectedRows, ExpectedDuplicate]).

effective_distance_main_code(KernelMode, FactStorage, ChildSearch, ReverseIndexOptions,
                             Dimension, MaxDepth,
                             ArticleCategories, RootCategories, Code) :-
    c_article_category_arrays(ArticleCategories, ArticleIds, ArticleArrays),
    c_string_array('ARTICLE_COUNT', 'ARTICLES', ArticleIds, ArticlesArray),
    c_string_array('ROOT_COUNT', 'ROOTS', RootCategories, RootArray),
    kernel_mode_flag(KernelMode, KernelFlag),
    child_search_flag(ChildSearch, ChildSearchFlag),
    child_search_max_children(ChildSearch, MaxChildExpansions),
    child_search_depth(ChildSearch, ChildSearchDepth),
    child_search_parent_cost(ChildSearch, ParentStepCost),
    child_search_child_cost(ChildSearch, ChildStepCost),
    child_search_budget(ChildSearch, ChildSearchBudget),
    fact_source_load_code(FactStorage, LoadCode),
    bidirectional_setup_declaration(ChildSearch, BidirSetupDeclaration),
    bidirectional_setup_call(ChildSearch, BidirSetupCall),
    bidirectional_register_call(ChildSearch, MaxDepth, ParentStepCost, ChildStepCost,
                                ChildSearchBudget, BidirRegisterCall),
    reverse_index_declaration(ReverseIndexOptions, ReverseIndexDeclaration),
    reverse_index_local(ReverseIndexOptions, ReverseIndexLocal),
    reverse_index_setup_call(ReverseIndexOptions, ReverseIndexSetupCall),
    reverse_index_teardown_call(ReverseIndexOptions, ReverseIndexTeardownCall),
    format(atom(Code),
'#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "wam_runtime.h"

void setup_category_ancestor_4(WamState* state);
~w
~w

~w
~w
~w

static double wam_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_sec * 1000.0) + ((double)tv.tv_usec / 1000.0);
}

static long long wam_read_env_limit(const char *name) {
    const char *raw = getenv(name);
    if (raw == NULL || raw[0] == 0) return 0;
    char *end = NULL;
    long long value = strtoll(raw, &end, 10);
    if (end == raw || *end != 0 || value < 0) {
        fprintf(stderr, "wam_c_effective_warning invalid_%s=%s\\n", name, raw);
        fflush(stderr);
        return 0;
    }
    return value;
}

static int wam_read_count_limit(const char *name, int total) {
    long long limit = wam_read_env_limit(name);
    if (limit <= 0 || limit > total) return total;
    return (int)limit;
}

static int wam_read_stride(const char *name) {
    long long value = wam_read_env_limit(name);
    if (value <= 0) return 1;
    return (int)value;
}

static int wam_read_offset(const char *name) {
    long long value = wam_read_env_limit(name);
    if (value <= 0) return 0;
    return (int)value;
}

static int wam_csv_contains(const char *csv, const char *value) {
    if (csv == NULL || csv[0] == 0) return 1;
    const char *cursor = csv;
    size_t value_len = strlen(value);
    while (*cursor != 0) {
        while (*cursor == 32 || *cursor == 9) cursor++;
        const char *start = cursor;
        while (*cursor != 0 && *cursor != 44) cursor++;
        const char *end = cursor;
        while (end > start && (end[-1] == 32 || end[-1] == 9)) end--;
        size_t token_len = (size_t)(end - start);
        if (token_len == value_len && strncmp(start, value, token_len) == 0) {
            return 1;
        }
        if (*cursor == 44) cursor++;
    }
    return 0;
}

static int wam_selected_index(int index,
                              const char *name,
                              const char *names,
                              int stride,
                              int offset) {
    if (!wam_csv_contains(names, name)) return 0;
    if (index < offset) return 0;
    return ((index - offset) % stride) == 0;
}

static int wam_count_selected(const char **values,
                              int count,
                              int limit,
                              const char *names,
                              int stride,
                              int offset) {
    int selected = 0;
    for (int i = 0; i < limit && i < count; i++) {
        if (wam_selected_index(i, values[i], names, stride, offset)) selected++;
    }
    return selected;
}

static int *wam_selected_prefix(const char **values,
                                int count,
                                int limit,
                                const char *names,
                                int stride,
                                int offset) {
    int prefix_count = limit < count ? limit : count;
    int *prefix = calloc((size_t)prefix_count + 1, sizeof(int));
    if (!prefix) return NULL;
    for (int i = 0; i < prefix_count; i++) {
        prefix[i + 1] = prefix[i] +
            (wam_selected_index(i, values[i], names, stride, offset) ? 1 : 0);
    }
    return prefix;
}

static int wam_selected_between(const int *prefix,
                                int start,
                                int end,
                                int limit) {
    if (!prefix) return 0;
    if (start < 0) start = 0;
    if (end < start) end = start;
    if (end > limit) end = limit;
    return prefix[end] - prefix[start];
}

static int visited_contains(const char **visited, int visited_len, const char *atom);

typedef struct {
    unsigned char *bits;
    int *indices;
    int count;
    int cap;
} WamCandidateRootSet;

static void wam_candidate_root_set_init(WamCandidateRootSet *set, int root_limit) {
    memset(set, 0, sizeof(WamCandidateRootSet));
    if (root_limit > 0) {
        set->bits = calloc((size_t)root_limit, sizeof(unsigned char));
        set->indices = malloc(sizeof(int) * (size_t)root_limit);
        if (set->indices) set->cap = root_limit;
    }
}

static void wam_candidate_root_set_close(WamCandidateRootSet *set) {
    free(set->bits);
    free(set->indices);
    memset(set, 0, sizeof(WamCandidateRootSet));
}

static void wam_candidate_root_set_reset(WamCandidateRootSet *set,
                                         int root_limit) {
    if (set->bits) memset(set->bits, 0, (size_t)root_limit);
    set->count = 0;
}

static int wam_compare_ints(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static void wam_candidate_root_set_sort(WamCandidateRootSet *set) {
    if (set->count > 1) {
        qsort(set->indices, (size_t)set->count, sizeof(int), wam_compare_ints);
    }
}

static int wam_candidate_root_set_push(WamCandidateRootSet *set, int root_index) {
    if (set->count >= set->cap) return 0;
    set->indices[set->count++] = root_index;
    return 1;
}

static int wam_root_index_of(const char *root, int root_limit) {
    int lo = 0;
    int hi = root_limit;
    while (lo < hi) {
        int mid = lo + ((hi - lo) / 2);
        int cmp = strcmp(ROOTS[mid], root);
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo < root_limit && strcmp(ROOTS[lo], root) == 0) return lo;
    return -1;
}

static int wam_mark_candidate_root(WamCandidateRootSet *candidate_roots,
                                   int root_limit,
                                   const char *category) {
    int root_index = wam_root_index_of(category, root_limit);
    if (root_index < 0) return 0;
    if (candidate_roots->bits[root_index]) return 0;
    if (!wam_candidate_root_set_push(candidate_roots, root_index)) return 0;
    candidate_roots->bits[root_index] = 1;
    return 1;
}

static int wam_mark_parent_candidate_roots(WamFactSource *source,
                                           const char *category,
                                           int parent_hops,
                                           int max_parent_hops,
                                           const char **visited,
                                           int visited_len,
                                           WamCandidateRootSet *candidate_roots,
                                           int root_limit) {
    int marks = wam_mark_candidate_root(candidate_roots, root_limit, category);
    if (parent_hops >= max_parent_hops || visited_len >= 64) return marks;

    CategoryEdge *edges = NULL;
    int edge_count = 0;
    if (!wam_fact_source_child_range(source, category, &edges, &edge_count)) {
        return marks;
    }
    for (int i = 0; i < edge_count; i++) {
        CategoryEdge *edge = &edges[i];
        if (visited_contains(visited, visited_len, edge->parent)) continue;
        visited[visited_len] = edge->parent;
        marks += wam_mark_parent_candidate_roots(source, edge->parent,
                                                 parent_hops + 1,
                                                 max_parent_hops,
                                                 visited, visited_len + 1,
                                                 candidate_roots, root_limit);
    }
    return marks;
}

static int wam_mark_child_candidate_roots(WamState *state,
                                          WamFactSource *source,
                                          const char *category,
                                          int max_child_expansions,
                                          int child_depth,
                                          double parent_step_cost,
                                          double child_step_cost,
                                          double budget,
                                          WamCandidateRootSet *candidate_roots,
                                          int root_limit) {
    if (max_child_expansions <= 0 || child_depth <= 0) return 0;
    if (child_depth != 1) return 0;
    if (parent_step_cost <= 0.0 || child_step_cost > budget) return 0;
    int max_parent_hops =
        (int)(((budget - child_step_cost) / parent_step_cost) + 1.0e-9);
    if (max_parent_hops < 0) return 0;

    int marks = 0;
    if (state->bidirectional_child_csr) {
        int parent_id = 0;
        if (!wam_category_atom_to_id(state, category, &parent_id)) return 0;
        int child_ids[256];
        int child_count = wam_reverse_csr_lookup_children(
            state->bidirectional_child_csr, parent_id, child_ids, 256);
        if (child_count < 0) return 0;
        int limit = child_count < 256 ? child_count : 256;
        for (int i = 0; i < limit; i++) {
            const char *child = NULL;
            const char *visited[64];
            if (!wam_category_id_to_atom(state, child_ids[i], &child)) continue;
            visited[0] = child;
            marks += wam_mark_parent_candidate_roots(source, child, 0,
                                                     max_parent_hops,
                                                     visited, 1,
                                                     candidate_roots,
                                                     root_limit);
        }
        return marks;
    }

    for (int i = 0; i < source->edge_count; i++) {
        CategoryEdge *edge = &source->edges[i];
        const char *visited[64];
        if (strcmp(edge->parent, category) != 0) continue;
        visited[0] = edge->child;
        marks += wam_mark_parent_candidate_roots(source, edge->child, 0,
                                                 max_parent_hops,
                                                 visited, 1,
                                                 candidate_roots,
                                                 root_limit);
    }
    return marks;
}

static int wam_mark_article_candidate_roots(WamState *state,
                                            WamFactSource *source,
                                            int category_start,
                                            int category_end,
                                            int root_limit,
                                            int max_depth,
                                            int child_search_enabled,
                                            int max_child_expansions,
                                            int child_depth,
                                            double parent_step_cost,
                                            double child_step_cost,
                                            double budget,
                                            WamCandidateRootSet *candidate_roots) {
    int marks = 0;
    for (int ci = category_start; ci < category_end; ci++) {
        const char *visited[64];
        visited[0] = ARTICLE_CATS[ci];
        marks += wam_mark_parent_candidate_roots(source, ARTICLE_CATS[ci], 0,
                                                 max_depth, visited, 1,
                                                 candidate_roots, root_limit);
        if (child_search_enabled) {
            marks += wam_mark_child_candidate_roots(state, source,
                                                    ARTICLE_CATS[ci],
                                                    max_child_expansions,
                                                    child_depth,
                                                    parent_step_cost,
                                                    child_step_cost,
                                                    budget,
                                                    candidate_roots,
                                                    root_limit);
        }
    }
    return marks;
}

static WamValue make_visited_singleton(WamState *state, const char *atom) {
    WamValue list;
    int required = state->H + 2;
    if (required > state->H_cap) {
        if (state->H_cap == 0) state->H_cap = WAM_INITIAL_CAP;
        while (required > state->H_cap) state->H_cap *= 2;
        state->H_array = realloc(state->H_array, sizeof(WamValue) * state->H_cap);
    }
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

typedef struct {
    double *values;
    int count;
    int cap;
} WamDoubleResults;

static void double_results_init(WamDoubleResults *results) {
    memset(results, 0, sizeof(WamDoubleResults));
}

static void double_results_close(WamDoubleResults *results) {
    free(results->values);
    memset(results, 0, sizeof(WamDoubleResults));
}

static int double_results_push(WamDoubleResults *results, double value) {
    if (results->count >= results->cap) {
        results->cap = results->cap ? results->cap * 2 : WAM_INITIAL_CAP;
        results->values = realloc(results->values, sizeof(double) * results->cap);
        if (!results->values) {
            results->count = 0;
            results->cap = 0;
            return 0;
        }
    }
    results->values[results->count++] = value;
    return 1;
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
    CategoryEdge *edges = NULL;
    int edge_count = 0;
    if (!wam_fact_source_child_range(source, cat, &edges, &edge_count)) return 0;
    for (int i = 0; i < edge_count; i++) {
        CategoryEdge *edge = &edges[i];
        if (visited_contains(visited, visited_len, edge->parent)) continue;
        if (strcmp(edge->parent, root) == 0) {
            if (!wam_int_results_push(results, depth + 1)) return 0;
            found = 1;
        }
    }
    if (visited_len >= max_depth || visited_len >= 64) return found;
    for (int i = 0; i < edge_count; i++) {
        CategoryEdge *edge = &edges[i];
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
                         WamIntResults *results);

static int collect_reference_child_costs(WamState *state,
                                         WamFactSource *source,
                                         const char *cat,
                                         const char *root,
                                         int max_child_expansions,
                                         int child_depth,
                                         double parent_step_cost,
                                         double child_step_cost,
                                         double budget,
                                         WamDoubleResults *results) {
    if (max_child_expansions <= 0 || child_depth <= 0) return 0;
    int found = 0;
    int expanded = 0;
    for (int i = 0; i < source->edge_count; i++) {
        CategoryEdge *edge = &source->edges[i];
        if (strcmp(edge->parent, cat) != 0) continue;
        if (expanded >= max_child_expansions) break;
        expanded++;

        WamIntResults child_hops;
        wam_int_results_init(&child_hops);
        collect_hops(state, source, edge->child, root, 0, &child_hops);
        if (child_hops.count == 0 && child_depth > 1 &&
            child_step_cost < budget) {
            WamDoubleResults child_costs;
            double_results_init(&child_costs);
            (void)collect_reference_child_costs(state, source, edge->child, root,
                                                max_child_expansions,
                                                child_depth - 1,
                                                parent_step_cost, child_step_cost,
                                                budget - child_step_cost,
                                                &child_costs);
            for (int ci = 0; ci < child_costs.count; ci++) {
                double path_cost = child_step_cost + child_costs.values[ci];
                if (path_cost > budget) continue;
                if (!double_results_push(results, path_cost)) {
                    double_results_close(&child_costs);
                    wam_int_results_close(&child_hops);
                    return 0;
                }
                found = 1;
            }
            double_results_close(&child_costs);
        }
        for (int hi = 0; hi < child_hops.count; hi++) {
            double path_cost = child_step_cost +
                               ((double)child_hops.values[hi] * parent_step_cost);
            if (path_cost > budget) continue;
            if (!double_results_push(results, path_cost)) {
                wam_int_results_close(&child_hops);
                return 0;
            }
            found = 1;
        }
        wam_int_results_close(&child_hops);
    }
    return found;
}

static int collect_bidirectional_child_costs(WamState *state,
                                             WamFactSource *source,
                                             const char *cat,
                                             const char *root,
                                             int kernels_on,
                                             int max_child_expansions,
                                             int child_depth,
                                             double parent_step_cost,
                                             double child_step_cost,
                                             double budget,
                                             WamDoubleResults *results) {
    if (!kernels_on) {
        return collect_reference_child_costs(state, source, cat, root,
                                             max_child_expansions, child_depth,
                                             parent_step_cost, child_step_cost,
                                             budget, results);
    }
    if (max_child_expansions <= 0 || child_depth <= 0) return 0;
    int found = 0;
    int accepted = 0;
    int heap_mark = state->H;
    WamBidirectionalAncestorResults bidir;
    wam_bidirectional_ancestor_results_init(&bidir);
    state->A[0] = val_atom(cat);
    state->A[1] = val_atom(root);
    state->A[2] = val_unbound("Total");
    state->A[3] = val_unbound("Parents");
    state->A[4] = val_unbound("Children");
    if (!wam_collect_bidirectional_ancestor_hops(state, &bidir)) {
        wam_bidirectional_ancestor_results_close(&bidir);
        state->H = heap_mark;
        return 0;
    }
    for (int i = 0; i < bidir.count; i++) {
        WamBidirectionalAncestorResult *path = &bidir.values[i];
        if (path->child_hops <= 0) continue;
        if (path->child_hops > child_depth) continue;
        if (accepted >= max_child_expansions) break;
        double path_cost =
            ((double)path->parent_hops * parent_step_cost) +
            ((double)path->child_hops * child_step_cost);
        if (path_cost > budget) continue;
        if (!double_results_push(results, path_cost)) {
            wam_bidirectional_ancestor_results_close(&bidir);
            state->H = heap_mark;
            return 0;
        }
        accepted++;
        found = 1;
    }
    wam_bidirectional_ancestor_results_close(&bidir);
    state->H = heap_mark;
    return found;
}

static void collect_hops(WamState *state,
                         WamFactSource *source,
                         const char *cat,
                         const char *root,
                         int kernels_on,
                         WamIntResults *results) {
    if (kernels_on) {
        int heap_mark = state->H;
        state->A[0] = val_atom(cat);
        state->A[1] = val_atom(root);
        state->A[2] = val_unbound("Hops");
        state->A[3] = make_visited_singleton(state, cat);
        (void)wam_collect_category_ancestor_hops(state, results);
        state->H = heap_mark;
    } else {
        const char *visited[64];
        visited[0] = cat;
        (void)collect_reference_hops(source, cat, root, 0, ~w, visited, 1, results);
    }
}

int main(void) {
    double setup_start_ms = wam_now_ms();
    WamState state;
    WamFactSource source;
~w
    wam_state_init(&state);
    wam_fact_source_init(&source);
    setup_category_ancestor_4(&state);
~w
~w

    if (!(~w)) {
        fprintf(stderr, "failed to load category_parent facts\\n");
~w
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 1;
    }
    wam_register_category_parent_fact_source(&state, &source);
    wam_register_category_ancestor_kernel(&state, "category_ancestor/4", ~w);
~w

    int article_limit = wam_read_count_limit("UW_WAM_C_EFFECTIVE_MAX_ARTICLES",
                                             ARTICLE_COUNT);
    int root_limit = wam_read_count_limit("UW_WAM_C_EFFECTIVE_MAX_ROOTS",
                                          ROOT_COUNT);
    long long query_limit = wam_read_env_limit("UW_WAM_C_EFFECTIVE_MAX_QUERIES");
    long long result_limit = wam_read_env_limit("UW_WAM_C_EFFECTIVE_MAX_RESULTS");
    long long progress_queries =
        wam_read_env_limit("UW_WAM_C_EFFECTIVE_PROGRESS_QUERIES");
    const char *article_names = getenv("UW_WAM_C_EFFECTIVE_ARTICLE_NAMES");
    const char *root_names = getenv("UW_WAM_C_EFFECTIVE_ROOT_NAMES");
    int article_stride = wam_read_stride("UW_WAM_C_EFFECTIVE_ARTICLE_STRIDE");
    int root_stride = wam_read_stride("UW_WAM_C_EFFECTIVE_ROOT_STRIDE");
    int article_offset = wam_read_offset("UW_WAM_C_EFFECTIVE_ARTICLE_OFFSET");
    int root_offset = wam_read_offset("UW_WAM_C_EFFECTIVE_ROOT_OFFSET");
    int selected_article_count = wam_count_selected(ARTICLES, ARTICLE_COUNT,
                                                    article_limit, article_names,
                                                    article_stride, article_offset);
    int selected_root_count = wam_count_selected(ROOTS, ROOT_COUNT, root_limit,
                                                 root_names, root_stride,
                                                 root_offset);
    int *selected_root_prefix = wam_selected_prefix(ROOTS, ROOT_COUNT, root_limit,
                                                    root_names, root_stride,
                                                    root_offset);
    long long candidate_filter_min_roots =
        wam_read_env_limit("UW_WAM_C_EFFECTIVE_CANDIDATE_FILTER_MIN_ROOTS");
    if (candidate_filter_min_roots <= 0) candidate_filter_min_roots = 256;
    WamCandidateRootSet candidate_roots;
    wam_candidate_root_set_init(&candidate_roots, root_limit);
    int candidate_filter_available =
        candidate_roots.bits != NULL &&
        candidate_roots.indices != NULL &&
        selected_root_prefix != NULL &&
        selected_root_count >= candidate_filter_min_roots &&
        !(~w && ~w != 1);
    if (root_limit > 0 && candidate_roots.bits == NULL) {
        fprintf(stderr, "wam_c_effective_warning candidate_filter_alloc_failed\\n");
        fflush(stderr);
    }
    if (root_limit > 0 && candidate_roots.indices == NULL) {
        fprintf(stderr, "wam_c_effective_warning candidate_schedule_alloc_failed\\n");
        fflush(stderr);
    }
    if (root_limit > 0 && selected_root_prefix == NULL) {
        fprintf(stderr, "wam_c_effective_warning selected_root_prefix_alloc_failed\\n");
        fflush(stderr);
    }
    double setup_ms = wam_now_ms() - setup_start_ms;
    fprintf(stderr,
            "wam_c_effective_setup article_count=%d root_count=%d "
            "article_category_count=%d article_limit=%d root_limit=%d "
            "selected_articles=%d selected_roots=%d query_limit=%lld "
            "result_limit=%lld article_stride=%d root_stride=%d "
            "article_offset=%d root_offset=%d "
            "candidate_filter_min_roots=%lld setup_ms=%.3f\\n",
            ARTICLE_COUNT, ROOT_COUNT, ARTICLE_CATEGORY_COUNT,
            article_limit, root_limit, selected_article_count,
            selected_root_count, query_limit, result_limit, article_stride,
            root_stride, article_offset, root_offset,
            candidate_filter_min_roots, setup_ms);
    fflush(stderr);

    printf("article\\troot_category\\teffective_distance\\n");
    double query_start_ms = wam_now_ms();
    double first_result_ms = -1.0;
    long long queries = 0;
    long long results = 0;
    long long category_visits = 0;
    long long direct_hits = 0;
    long long candidate_filter_articles = 0;
    long long candidate_filter_marks = 0;
    long long candidate_filter_skips = 0;
    long long candidate_schedule_articles = 0;
    long long candidate_schedule_roots = 0;
    long long parent_reachability_checks = 0;
    long long parent_reachability_prunes = 0;
    long long parent_collect_calls = 0;
    long long parent_path_results = 0;
    long long child_prefilter_checks = 0;
    long long child_prefilter_prunes = 0;
    long long child_prefilter_candidates = 0;
    long long child_collect_calls = 0;
    long long child_path_results = 0;
    double candidate_filter_ms = 0.0;
    double parent_reachability_ms = 0.0;
    double parent_collect_ms = 0.0;
    double child_prefilter_ms = 0.0;
    double child_collect_ms = 0.0;
    int stopped_by_cap = 0;
    for (int ai = 0; ai < article_limit && !stopped_by_cap; ai++) {
        if (!wam_selected_index(ai, ARTICLES[ai], article_names,
                                article_stride, article_offset)) {
            continue;
        }
        int category_start = ARTICLE_CATEGORY_STARTS[ai];
        int category_end = ARTICLE_CATEGORY_ENDS[ai];
        int candidate_filter_enabled = 0;
        if (candidate_filter_available) {
            wam_candidate_root_set_reset(&candidate_roots, root_limit);
            double candidate_filter_start_ms = wam_now_ms();
            candidate_filter_articles++;
            int article_marks = wam_mark_article_candidate_roots(
                &state, &source, category_start, category_end, root_limit,
                ~w, ~w, ~w, ~w, ~w, ~w, ~w, &candidate_roots);
            candidate_filter_ms += wam_now_ms() - candidate_filter_start_ms;
            candidate_filter_marks += article_marks;
            candidate_filter_enabled = 1;
            wam_candidate_root_set_sort(&candidate_roots);
            if (query_limit <= 0 && selected_root_prefix != NULL) {
                candidate_schedule_articles++;
                candidate_schedule_roots += candidate_roots.count;
            }
        }
        int candidate_schedule_enabled =
            candidate_filter_enabled && query_limit <= 0 &&
            selected_root_prefix != NULL;
        int sparse_pos = 0;
        int sparse_next_root = 0;
        for (int ri_scan = 0; ; ri_scan++) {
            int ri = 0;
            if (candidate_schedule_enabled) {
                if (sparse_pos >= candidate_roots.count) {
                    int tail_skips = wam_selected_between(selected_root_prefix,
                                                          sparse_next_root,
                                                          root_limit,
                                                          root_limit);
                    if (tail_skips > 0) {
                        queries += tail_skips;
                        candidate_filter_skips += tail_skips;
                    }
                    break;
                }
                ri = candidate_roots.indices[sparse_pos++];
                if (!wam_selected_index(ri, ROOTS[ri], root_names,
                                        root_stride, root_offset)) {
                    continue;
                }
                int logical_delta = wam_selected_between(selected_root_prefix,
                                                         sparse_next_root,
                                                         ri + 1,
                                                         root_limit);
                if (logical_delta <= 0) continue;
                queries += logical_delta;
                if (logical_delta > 1) {
                    candidate_filter_skips += logical_delta - 1;
                }
                sparse_next_root = ri + 1;
            } else {
                if (ri_scan >= root_limit) break;
                ri = ri_scan;
                if (!wam_selected_index(ri, ROOTS[ri], root_names,
                                        root_stride, root_offset)) {
                    continue;
                }
                if (query_limit > 0 && queries >= query_limit) {
                    stopped_by_cap = 1;
                    break;
                }
                queries++;
                if (candidate_filter_enabled && !candidate_roots.bits[ri]) {
                    candidate_filter_skips++;
                    if (progress_queries > 0 && queries % progress_queries == 0) {
                        fprintf(stderr,
                                "wam_c_effective_progress queries=%lld results=%lld "
                                "elapsed_ms=%.3f\\n",
                                queries, results, wam_now_ms() - query_start_ms);
                        fflush(stderr);
                    }
                    continue;
                }
            }
            double weight_sum = 0.0;
            for (int ci = category_start; ci < category_end; ci++) {
                category_visits++;
                WamIntResults hops;
                WamDoubleResults path_costs;
                wam_int_results_init(&hops);
                double_results_init(&path_costs);
                if (strcmp(ARTICLE_CATS[ci], ROOTS[ri]) == 0) {
                    double_results_push(&path_costs, ~w);
                    direct_hits++;
                } else {
                    int min_parent_hops = 0;
                    double reachability_start_ms = wam_now_ms();
                    parent_reachability_checks++;
                    int parent_reachable = wam_category_min_parent_hops(
                        &state, ARTICLE_CATS[ci], ROOTS[ri], &min_parent_hops);
                    parent_reachability_ms += wam_now_ms() - reachability_start_ms;
                    if (parent_reachable && min_parent_hops <= ~w) {
                        double parent_start_ms = wam_now_ms();
                        parent_collect_calls++;
                        collect_hops(&state, &source, ARTICLE_CATS[ci], ROOTS[ri], ~w, &hops);
                        parent_collect_ms += wam_now_ms() - parent_start_ms;
                        parent_path_results += hops.count;
                        for (int hi = 0; hi < hops.count; hi++) {
                            double_results_push(&path_costs,
                                                ((double)hops.values[hi] + 1.0) * ~w);
                        }
                    } else {
                        parent_reachability_prunes++;
                    }
                    if (path_costs.count == 0 && ~w) {
                        int child_candidate_count = 0;
                        double child_prefilter_start_ms = wam_now_ms();
                        child_prefilter_checks++;
                        int child_may_reach =
                            wam_category_child_may_reach_root_within_budget(
                                &state, ARTICLE_CATS[ci], ROOTS[ri], ~w, ~w,
                                ~w, ~w, ~w, &child_candidate_count);
                        child_prefilter_ms += wam_now_ms() - child_prefilter_start_ms;
                        child_prefilter_candidates += child_candidate_count;
                        if (child_may_reach) {
                            double child_start_ms = wam_now_ms();
                            child_collect_calls++;
                            (void)collect_bidirectional_child_costs(&state, &source,
                                                                    ARTICLE_CATS[ci], ROOTS[ri],
                                                                    ~w, ~w,
                                                                    ~w, ~w, ~w, ~w,
                                                                    &path_costs);
                            child_collect_ms += wam_now_ms() - child_start_ms;
                            child_path_results += path_costs.count;
                            for (int pi = 0; pi < path_costs.count; pi++) {
                                path_costs.values[pi] += ~w;
                            }
                        } else {
                            child_prefilter_prunes++;
                        }
                    }
                }
                for (int pi = 0; pi < path_costs.count; pi++) {
                    weight_sum += pow(path_costs.values[pi], -~w.0);
                }
                double_results_close(&path_costs);
                wam_int_results_close(&hops);
            }
            if (weight_sum > 0.0) {
                double deff = pow(weight_sum, -1.0 / ~w.0);
                if (first_result_ms < 0.0) {
                    first_result_ms = wam_now_ms() - query_start_ms;
                }
                printf("%s\\t%s\\t%.6f\\n", ARTICLES[ai], ROOTS[ri], deff);
                results++;
                if (result_limit > 0 && results >= result_limit) {
                    stopped_by_cap = 1;
                    break;
                }
            }
            if (progress_queries > 0 && queries % progress_queries == 0) {
                fprintf(stderr,
                        "wam_c_effective_progress queries=%lld results=%lld "
                        "elapsed_ms=%.3f\\n",
                        queries, results, wam_now_ms() - query_start_ms);
                fflush(stderr);
            }
        }
    }
    fprintf(stderr,
            "wam_c_effective_runtime queries=%lld results=%lld "
            "category_visits=%lld direct_hits=%lld "
            "candidate_filter_articles=%lld candidate_filter_marks=%lld "
            "candidate_filter_skips=%lld candidate_filter_ms=%.3f "
            "candidate_schedule_articles=%lld candidate_schedule_roots=%lld "
            "parent_reachability_checks=%lld parent_reachability_prunes=%lld "
            "parent_reachability_ms=%.3f parent_collect_calls=%lld "
            "parent_path_results=%lld parent_collect_ms=%.3f child_collect_calls=%lld "
            "child_prefilter_checks=%lld child_prefilter_prunes=%lld "
            "child_prefilter_candidates=%lld child_prefilter_ms=%.3f "
            "child_path_results=%lld child_collect_ms=%.3f "
            "first_result_ms=%.3f query_ms=%.3f capped=%d\\n",
            queries, results, category_visits, direct_hits,
            candidate_filter_articles, candidate_filter_marks,
            candidate_filter_skips, candidate_filter_ms,
            candidate_schedule_articles, candidate_schedule_roots,
            parent_reachability_checks, parent_reachability_prunes,
            parent_reachability_ms, parent_collect_calls, parent_path_results, parent_collect_ms,
            child_collect_calls, child_prefilter_checks, child_prefilter_prunes,
            child_prefilter_candidates, child_prefilter_ms,
            child_path_results, child_collect_ms,
            first_result_ms, wam_now_ms() - query_start_ms, stopped_by_cap);
    fflush(stderr);

~w
    wam_candidate_root_set_close(&candidate_roots);
    free(selected_root_prefix);
    wam_fact_source_close(&source);
    wam_free_state(&state);
    return 0;
}
', [BidirSetupDeclaration, ReverseIndexDeclaration,
    ArticleArrays, ArticlesArray, RootArray, MaxDepth,
    ReverseIndexLocal, BidirSetupCall, ReverseIndexSetupCall,
    LoadCode, ReverseIndexTeardownCall, MaxDepth, BidirRegisterCall,
    ChildSearchFlag, ChildSearchDepth,
    MaxDepth, ChildSearchFlag, MaxChildExpansions, ChildSearchDepth,
    ParentStepCost, ChildStepCost, ChildSearchBudget,
    ParentStepCost, MaxDepth, KernelFlag, ParentStepCost,
    ChildSearchFlag, MaxChildExpansions, ChildSearchDepth,
    ParentStepCost, ChildStepCost, ChildSearchBudget,
    KernelFlag, MaxChildExpansions, ChildSearchDepth,
    ParentStepCost, ChildStepCost, ChildSearchBudget, ParentStepCost,
    Dimension, Dimension, ReverseIndexTeardownCall]).

fact_source_load_code(facts_tsv,
                      'wam_fact_source_load_tsv(&state, &source, "category_parent.tsv")').
fact_source_load_code(facts_lmdb,
                      'wam_fact_source_load_lmdb(&state, &source, "category_parent.lmdb", NULL)').

kernel_mode_flag(kernels_on, 1).
kernel_mode_flag(kernels_off, 0).

child_search_flag(child_search(bounded, MaxChildren, ChildDepth, _, _, _), 1) :-
    MaxChildren > 0,
    ChildDepth > 0,
    !.
child_search_flag(_, 0).

child_search_enabled(ChildSearch) :-
    child_search_flag(ChildSearch, 1).

child_search_max_children(child_search(_, MaxChildren, _, _, _, _), MaxChildren).
child_search_depth(child_search(_, _, ChildDepth, _, _, _), ChildDepth).
child_search_parent_cost(child_search(_, _, _, ParentCost, _, _), ParentCost).
child_search_child_cost(child_search(_, _, _, _, ChildCost, _), ChildCost).
child_search_budget(child_search(_, _, _, _, _, Budget), Budget).

bidirectional_setup_declaration(ChildSearch,
                                'void setup_bidirectional_ancestor_5(WamState* state);') :-
    child_search_enabled(ChildSearch),
    !.
bidirectional_setup_declaration(_, '').

bidirectional_setup_call(ChildSearch,
                         '    setup_bidirectional_ancestor_5(&state);') :-
    child_search_enabled(ChildSearch),
    !.
bidirectional_setup_call(_, '').

bidirectional_register_call(ChildSearch, MaxDepth, ParentStepCost, ChildStepCost,
                            Budget, Call) :-
    child_search_enabled(ChildSearch),
    !,
    format(atom(Call),
           '    wam_register_bidirectional_ancestor_kernel(&state, "bidirectional_ancestor/5", ~w, ~w, ~w, ~w);',
           [MaxDepth, ParentStepCost, ChildStepCost, Budget]).
bidirectional_register_call(_, _, _, _, _, '').

reverse_index_enabled(ReverseIndexOptions) :-
    ReverseIndexOptions \= [].

reverse_index_declaration(ReverseIndexOptions,
'bool setup_wam_c_reverse_index_artifacts(WamState* state,
                                         WamReverseCsrArtifact* bidirectional_child_csr);
void teardown_wam_c_reverse_index_artifacts(WamState* state,
                                            WamReverseCsrArtifact* bidirectional_child_csr);') :-
    reverse_index_enabled(ReverseIndexOptions),
    !.
reverse_index_declaration(_, '').

reverse_index_local(ReverseIndexOptions,
'    WamReverseCsrArtifact bidirectional_child_csr;') :-
    reverse_index_enabled(ReverseIndexOptions),
    !.
reverse_index_local(_, '').

reverse_index_setup_call(ReverseIndexOptions,
'    if (!setup_wam_c_reverse_index_artifacts(&state, &bidirectional_child_csr)) {
        fprintf(stderr, "failed to load reverse CSR artifact\\n");
        wam_fact_source_close(&source);
        wam_free_state(&state);
        return 2;
    }') :-
    reverse_index_enabled(ReverseIndexOptions),
    !.
reverse_index_setup_call(_, '').

reverse_index_teardown_call(ReverseIndexOptions,
'    teardown_wam_c_reverse_index_artifacts(&state, &bidirectional_child_csr);') :-
    reverse_index_enabled(ReverseIndexOptions),
    !.
reverse_index_teardown_call(_, '').

c_article_category_arrays(Pairs, ArticleIds, Code) :-
    msort(Pairs, SortedPairs),
    pairs_keys(SortedPairs, Keys),
    pairs_values(SortedPairs, Values),
    sort(Keys, ArticleIds),
    article_category_ranges(ArticleIds, SortedPairs, Starts, Ends),
    c_string_initializer(Keys, KeyInit),
    c_string_initializer(Values, ValueInit),
    c_int_initializer(Starts, StartInit),
    c_int_initializer(Ends, EndInit),
    length(SortedPairs, Count),
    format(atom(Code),
           'static const int ARTICLE_CATEGORY_COUNT = ~w;\nstatic const char *ARTICLE_IDS[] = { ~w };\nstatic const char *ARTICLE_CATS[] = { ~w };\nstatic const int ARTICLE_CATEGORY_STARTS[] = { ~w };\nstatic const int ARTICLE_CATEGORY_ENDS[] = { ~w };',
           [Count, KeyInit, ValueInit, StartInit, EndInit]).

article_category_ranges(ArticleIds, SortedPairs, Starts, Ends) :-
    article_category_ranges(ArticleIds, SortedPairs, 0, Starts, Ends).

article_category_ranges([], [], _Offset, [], []).
article_category_ranges([Article|Articles], Pairs0, Offset,
                        [Offset|Starts], [End|Ends]) :-
    take_article_category_rows(Pairs0, Article, Count, Pairs),
    End is Offset + Count,
    article_category_ranges(Articles, Pairs, End, Starts, Ends).

take_article_category_rows([Article-_|Rest], Article, Count, Remaining) :-
    !,
    take_article_category_rows(Rest, Article, Count0, Remaining),
    Count is Count0 + 1.
take_article_category_rows(Remaining, _Article, 0, Remaining).

c_string_array(CountName, Name, Values, Code) :-
    c_string_initializer(Values, Init),
    length(Values, Count),
    format(atom(Code),
           'static const int ~w = ~w;\nstatic const char *~w[] = { ~w };',
           [CountName, Count, Name, Init]).

c_string_initializer(Values, Init) :-
    maplist(c_string_literal, Values, Literals),
    atomic_list_concat(Literals, ', ', Init).

c_int_initializer(Values, Init) :-
    maplist(format_int_atom, Values, Literals),
    atomic_list_concat(Literals, ', ', Init).

format_int_atom(Value, Atom) :-
    format(atom(Atom), '~w', [Value]).

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

effective_distance_readme(KernelMode, FactStorage, ReverseIndexOptions, Readme) :-
    build_notes(FactStorage, ReverseIndexOptions, BuildNotes),
    format(atom(Readme),
'# WAM-C Effective Distance Benchmark

Generated by `generate_wam_c_effective_distance_benchmark.pl`.

Build:

```sh
~w
```

Run:

```sh
./wam_c_effective_distance
```

Kernel mode: `~w`.
Fact storage: `~w`.
', [BuildNotes, KernelMode, FactStorage]).

build_notes(facts_tsv, ReverseIndexOptions,
            'gcc -std=c11 -Wall -Wextra seed_category_child_csr_offsets_lmdb.c -llmdb -o seed_category_child_csr_offsets_lmdb\n./seed_category_child_csr_offsets_lmdb\ngcc -std=c11 -Wall -Wextra -DWAM_C_ENABLE_LMDB -I ../../src/unifyweaver/targets/wam_c_runtime wam_runtime.c lib.c main.c -lm -llmdb -o wam_c_effective_distance') :-
    reverse_index_uses_lmdb_offset(ReverseIndexOptions),
    !.
build_notes(facts_tsv, _ReverseIndexOptions,
            'gcc -std=c11 -Wall -Wextra -I ../../src/unifyweaver/targets/wam_c_runtime wam_runtime.c lib.c main.c -lm -o wam_c_effective_distance').
build_notes(facts_lmdb, ReverseIndexOptions,
            'gcc -std=c11 -Wall -Wextra seed_category_parent_lmdb.c -llmdb -o seed_category_parent_lmdb\n./seed_category_parent_lmdb\ngcc -std=c11 -Wall -Wextra seed_category_child_csr_offsets_lmdb.c -llmdb -o seed_category_child_csr_offsets_lmdb\n./seed_category_child_csr_offsets_lmdb\ngcc -std=c11 -Wall -Wextra -DWAM_C_ENABLE_LMDB -I ../../src/unifyweaver/targets/wam_c_runtime wam_runtime.c lib.c main.c -lm -llmdb -o wam_c_effective_distance') :-
    reverse_index_uses_lmdb_offset(ReverseIndexOptions),
    !.
build_notes(facts_lmdb, _ReverseIndexOptions,
            'gcc -std=c11 -Wall -Wextra seed_category_parent_lmdb.c -llmdb -o seed_category_parent_lmdb\n./seed_category_parent_lmdb\ngcc -std=c11 -Wall -Wextra -DWAM_C_ENABLE_LMDB -I ../../src/unifyweaver/targets/wam_c_runtime wam_runtime.c lib.c main.c -lm -llmdb -o wam_c_effective_distance').

reverse_index_uses_lmdb_offset(ReverseIndexOptions) :-
    memberchk(reverse_index(artifact(ArtifactOptions)), ReverseIndexOptions),
    memberchk(index_backend(lmdb_offset), ArtifactOptions).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).
