:- module(generate_wam_scala_effective_distance_benchmark,
          [ main/0,
            generate/4,
            generate/5,
            parse_benchmark_data_mode/2,
            benchmark_effective_data_mode/3,
            collect_wam_predicates/2,
            scala_effective_distance_runner_code/2
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module('../../src/unifyweaver/targets/wam_scala_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).

%% generate_wam_scala_effective_distance_benchmark.pl
%%
%% Generates a result-producing Scala WAM effective-distance project. This is
%% intentionally separate from the generic Scala target tests: it adds a
%% benchmark companion object that emits the standard TSV result table while
%% leaving GeneratedProgram's predicate CLI unchanged.
%%
%% Usage:
%%   swipl -q -s generate_wam_scala_effective_distance_benchmark.pl -- \
%%       <facts.pl> <output-dir> [accumulated|seeded] [kernels_on|kernels_off] [sidecar|inline|artifact|auto]

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom, KernelModeAtom, DataModeAtom]
    ->  true
    ;   Argv = [FactsPath, OutputDir, VariantAtom, KernelModeAtom]
    ->  DataModeAtom = auto
    ;   Argv = [FactsPath, OutputDir, VariantAtom]
    ->  KernelModeAtom = kernels_on,
        DataModeAtom = auto
    ;   Argv = [FactsPath, OutputDir]
    ->  VariantAtom = accumulated,
        KernelModeAtom = kernels_on,
        DataModeAtom = auto
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> [accumulated|seeded] [kernels_on|kernels_off] [sidecar|inline|artifact|auto]~n',
            []),
        halt(1)
    ),
    (   Argv == []
    ->  true
    ;   generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom, DataModeAtom),
        halt(0)
    ).

main :-
    format(user_error, 'Error: Scala WAM effective-distance generation failed~n', []),
    halt(1).

generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom) :-
    generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom, auto).

generate(FactsPath, OutputDir, VariantAtom, KernelModeAtom, DataModeAtom) :-
    must_exist_file(FactsPath),
    benchmark_workload_path(WorkloadPath),
    reset_benchmark_predicates,
    load_files(user:WorkloadPath, [silent(true), if(true)]),
    load_files(user:FactsPath, [silent(true), if(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    parse_variant(VariantAtom, OptimizationOptions),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),
    maybe_load_optimized_predicates(VariantAtom, ScriptCode),
    maybe_assert_seeded_helper(VariantAtom),
    collect_wam_predicates(KernelModeAtom, Predicates),
    parse_benchmark_data_mode(DataModeAtom, DataMode),
    scala_options(OutputDir, KernelModeAtom, DataMode, Options),
    write_wam_scala_project(Predicates, Options, OutputDir),
    append_effective_distance_runner(OutputDir),
    format(user_error,
           '[WAM-Scala-EffectiveDistance] facts=~w variant=~w kernels=~w data_mode=~w output=~w~n',
           [FactsPath, VariantAtom, KernelModeAtom, DataMode, OutputDir]).

reset_benchmark_predicates :-
    forall(member(Name/Arity,
                  [ article_category/2,
                    category_parent/2,
                    root_category/1,
                    dimension_n/1,
                    max_depth/1,
                    category_ancestor/4,
                    power_sum_bound/4,
                    'category_ancestor$power_sum_bound'/3,
                    'category_ancestor$power_sum_selected'/3,
                    'category_ancestor$effective_distance_sum_selected'/3,
                    'category_ancestor$effective_distance_sum_bound'/3
                  ]),
           maybe_abolish_user_predicate(Name/Arity)).

maybe_abolish_user_predicate(Name/Arity) :-
    (   current_predicate(user:Name/Arity)
    ->  abolish(user:Name/Arity)
    ;   true
    ).

parse_variant(seeded, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false)
]).
parse_variant(accumulated, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false),
    seeded_accumulation(auto)
]).

parse_benchmark_data_mode(sidecar, sidecar).
parse_benchmark_data_mode(inline, inline).
parse_benchmark_data_mode(artifact, artifact).
parse_benchmark_data_mode(auto, Mode) :-
    (   benchmark_declared_data_mode(DeclaredMode),
        DeclaredMode \== auto
    ->  parse_benchmark_data_mode(DeclaredMode, Mode)
    ;   benchmark_fact_volume(RowCount),
        (   RowCount >= 128
        ->  Mode = sidecar
        ;   Mode = inline
        )
    ).

benchmark_declared_data_mode(Mode) :-
    member(Name,
           [ wam_scala_benchmark_data_mode,
             benchmark_data_mode
           ]),
    current_predicate(user:Name/1),
    Goal =.. [Name, DeclaredMode],
    once(call(user:Goal)),
    memberchk(DeclaredMode, [auto, sidecar, inline, artifact]),
    Mode = DeclaredMode.

benchmark_relation_declared_data_mode(Relation, Mode) :-
    member(Name,
           [ wam_scala_benchmark_relation_data_mode,
             benchmark_relation_data_mode
           ]),
    current_predicate(user:Name/2),
    Goal =.. [Name, Relation, DeclaredMode],
    once(call(user:Goal)),
    memberchk(DeclaredMode, [auto, sidecar, inline, artifact]),
    Mode = DeclaredMode.

benchmark_effective_data_mode(Relation, DataMode, RelationMode) :-
    (   benchmark_relation_declared_data_mode(Relation, DeclaredMode),
        DeclaredMode \== auto
    ->  parse_benchmark_data_mode(DeclaredMode, RelationMode)
    ;   RelationMode = DataMode
    ).

benchmark_fact_volume(RowCount) :-
    benchmark_fact_count(user:category_parent/2, ParentCount),
    benchmark_fact_count(user:article_category/2, ArticleCount),
    benchmark_fact_count(user:root_category/1, RootCount),
    RowCount is ParentCount + ArticleCount + RootCount.

benchmark_fact_count(Module:Name/Arity, Count) :-
    functor(Goal, Name, Arity),
    findall(Goal, Module:Goal, Goals),
    length(Goals, Count).

scala_options(OutputDir, kernels_on, DataMode, Options) :-
    benchmark_effective_data_mode(category_parent, DataMode, ParentMode),
    benchmark_atoms(Atoms),
    scala_fact_source_for_category_parent(OutputDir, ParentMode, FactSource),
    Options = [
        package('generated.wam_scala_effective_distance.core'),
        runtime_package('generated.wam_scala_effective_distance.core'),
        module_name('wam-scala-effective-distance'),
        intern_atoms(Atoms),
        scala_fact_sources([FactSource])
    ].
scala_options(_OutputDir, kernels_off, _DataMode, Options) :-
    benchmark_atoms(Atoms),
    Options = [
        package('generated.wam_scala_effective_distance.core'),
        runtime_package('generated.wam_scala_effective_distance.core'),
        module_name('wam-scala-effective-distance'),
        intern_atoms(Atoms)
    ].

scala_fact_source_for_category_parent(_OutputDir, inline, source(category_parent/2, Tuples)) :-
    findall([Child, Parent], current_category_parent_fact(Child, Parent), Tuples0),
    sort(Tuples0, Tuples).
scala_fact_source_for_category_parent(OutputDir, sidecar, source(category_parent/2, file(CsvPath))) :-
    category_parent_csv_path(OutputDir, sidecar, CsvPath),
    write_category_parent_csv(CsvPath).
scala_fact_source_for_category_parent(OutputDir, artifact, source(category_parent/2, file(CsvPath))) :-
    category_parent_csv_path(OutputDir, artifact, CsvPath),
    write_category_parent_csv(CsvPath).

collect_wam_predicates(kernels_on, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4
]).
collect_wam_predicates(kernels_off, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_parent/2,
    user:category_ancestor/4
]).

category_parent_csv_path(OutputDir, sidecar, CsvPath) :-
    directory_file_path(OutputDir, 'data', DataDir),
    make_directory_path(DataDir),
    directory_file_path(DataDir, 'category_parent.csv', CsvPath).
category_parent_csv_path(OutputDir, artifact, CsvPath) :-
    directory_file_path(OutputDir, 'data', DataDir),
    make_directory_path(DataDir),
    directory_file_path(DataDir, 'category_parent_artifact.csv', CsvPath).

write_category_parent_csv(CsvPath) :-
    findall(Child-Parent, current_category_parent_fact(Child, Parent), Pairs0),
    sort(Pairs0, Pairs),
    setup_call_cleanup(
        open(CsvPath, write, Stream),
        forall(member(Child-Parent, Pairs),
               format(Stream, '~w,~w~n', [Child, Parent])),
        close(Stream)).

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

current_category_parent_fact(Child, Parent) :-
    current_predicate(user:category_parent/2),
    call(user:category_parent(Child, Parent)).

current_article_category_fact(Article, Category) :-
    current_predicate(user:article_category/2),
    call(user:article_category(Article, Category)).

current_root_category_fact(Root) :-
    current_predicate(user:root_category/1),
    call(user:root_category(Root)).

maybe_assert_seeded_helper(seeded) :- !,
    retractall(user:power_sum_bound(_, _, _, _)),
    assertz((user:power_sum_bound(Cat, Root, NegN, WeightSum) :-
        aggregate_all(sum(W),
            (user:category_ancestor(Cat, Root, Hops, [Cat]),
             H is Hops + 1,
             W is H ** NegN),
            WeightSum))).
maybe_assert_seeded_helper(_).

maybe_load_optimized_predicates(seeded, _) :- !.
maybe_load_optimized_predicates(accumulated, ScriptCode) :-
    strip_generated_initialization(ScriptCode, LoadableScript),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, LoadableScript),
    close(TmpStream),
    load_files(user:TmpPath, [silent(true)]),
    delete_file(TmpPath),
    cleanup_generated_script_entrypoint.

strip_generated_initialization(ScriptCode, LoadableScript) :-
    split_string(ScriptCode, "\n", "", Lines0),
    exclude(generated_initialization_line, Lines0, Lines),
    atomic_list_concat(Lines, '\n', LoadableScript).

generated_initialization_line(Line) :-
    sub_string(Line, _, _, _, "initialization(main").

cleanup_generated_script_entrypoint :-
    (   current_predicate(user:main/0)
    ->  abolish(user:main/0)
    ;   true
    ).

append_effective_distance_runner(OutputDir) :-
    scala_effective_distance_runner_code(
        'generated.wam_scala_effective_distance.core',
        Code),
    directory_file_path(OutputDir, 'src/main/scala/generated/wam_scala_effective_distance/core', SrcDir),
    make_directory_path(SrcDir),
    directory_file_path(SrcDir, 'EffectiveDistanceRunner.scala', Path),
    setup_call_cleanup(
        open(Path, write, Stream),
        write(Stream, Code),
        close(Stream)).

scala_effective_distance_runner_code(Package, Code) :-
    format(string(Code), 'package ~w

import scala.collection.mutable
import java.nio.file.{Path, Paths}
import scala.io.Source

object EffectiveDistanceRunner {
  private val dimensionN = 5.0
  private val inverseDimensionN = -1.0 / dimensionN

  final case class ResultRow(article: String, root: String, distance: Double)

  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: EffectiveDistanceRunner <category_parent.tsv> <article_category.tsv>")
      sys.exit(1)
    }

    val edgePath = Paths.get(args(0))
    val articlePath = Paths.get(args(1))
    val rootPath = edgePath.getParent.resolve("root_categories.tsv")
    val articleCategories = loadPairs(articlePath)
    val roots = loadSingleColumn(rootPath)
    val categoriesByArticle = articleCategories.groupMap(_._1)(_._2)

    val rows = (for {
      root <- roots
      article <- categoriesByArticle.keys.toSeq.sorted
      weightSum = categoriesByArticle(article).flatMap(categoryWeights(_, root)).sum
      if weightSum > 0.0
    } yield ResultRow(article, root, Math.pow(weightSum, inverseDimensionN)))
      .sortBy(row => (row.distance, row.article, row.root))

    Console.err.println("mode=scala_wam_effective_distance")
    Console.err.println(s"article_count=${rows.size}")
    println("article\\troot_category\\teffective_distance")
    rows.foreach { row =>
      println(f"${row.article}\\t${row.root}\\t${row.distance}%.6f")
    }
  }

  private def categoryWeights(category: String, root: String): Seq[Double] =
    if (category == root) Seq(1.0)
    else categoryRootHops(category, root).map(hops => Math.pow(hops + 1.0, -dimensionN))

  private def categoryRootHops(category: String, root: String): Seq[Int] = {
    val startPc = GeneratedProgram.sharedProgram.dispatch("category_ancestor/4")
    val categoryTerm = atom(category)
    val rootTerm = atom(root)
    val hopsRef = Ref(0)
    val visited = listOf(categoryTerm)
    val state = WamRuntime.newState(startPc, Array(categoryTerm, rootTerm, hopsRef, visited))
    val results = mutable.ListBuffer.empty[Int]
    if (WamRuntime.run(state, GeneratedProgram.sharedProgram)) {
      var keepGoing = true
      while (keepGoing) {
        valueToInt(WamRuntime.deref(state.bindings, state.regs(3))).foreach(results += _)
        WamRuntime.backtrack(state)
        keepGoing = WamRuntime.run(state, GeneratedProgram.sharedProgram)
      }
    }
    results.toSeq
  }

  private def atom(raw: String): WamTerm =
    Atom(GeneratedProgram.internTable.intern(raw))

  private def listOf(items: WamTerm*): WamTerm = {
    val cons = GeneratedProgram.internTable.lookupId("[|]")
    val empty = GeneratedProgram.internTable.lookupId("[]")
    items.foldRight[WamTerm](Atom(empty)) { (head, tail) =>
      Struct(cons, 2, Array(head, tail))
    }
  }

  private def valueToInt(value: WamTerm): Option[Int] = value match {
    case IntTerm(value) => Some(value)
    case FloatTerm(value) => Some(value.toInt)
    case _ => None
  }

  private def loadPairs(path: Path): Vector[(String, String)] =
    withSource(path) { source =>
      source.getLines().drop(1).toVector.flatMap { line =>
        val trimmed = line.trim
        if (trimmed.isEmpty || trimmed.startsWith("#")) None
        else {
          val parts = trimmed.split("\\t")
          if (parts.length >= 2) Some(parts(0) -> parts(1)) else None
        }
      }
    }

  private def loadSingleColumn(path: Path): Vector[String] =
    withSource(path) { source =>
      source.getLines().drop(1).toVector
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
    }

  private def withSource[A](path: Path)(f: Source => A): A = {
    val source = Source.fromFile(path.toFile)
    try f(source)
    finally source.close()
  }
}
', [Package]).

must_exist_file(Path) :-
    (   exists_file(Path)
    ->  true
    ;   throw(error(existence_error(file, Path), _))
    ).
