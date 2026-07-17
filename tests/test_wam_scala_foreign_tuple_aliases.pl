:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0

% Executable regression for transactional, correlated foreign-result maps in
% the generated Scala WAM runtime.

:- use_module(library(filesex)).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').

scala_available :-
    catch(
        ( process_create(path(scalac), ['-version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _,
        fail).

scala_211_toolchain :-
    process_create(path(scalac), ['-version'],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid)]),
    read_string(Out, _, OutText),
    read_string(Err, _, ErrText),
    close(Out),
    close(Err),
    process_wait(Pid, exit(0)),
    string_concat(OutText, ErrText, VersionText),
    sub_string(VersionText, _, _, _, "version 2.11").

:- begin_tests(wam_scala_foreign_tuple_aliases,
               [condition(scala_available)]).

test(correlated_alias_candidates_are_transactional) :-
    Dir = 'output/test_wam_scala_foreign_tuple_aliases',
    setup_call_cleanup(
        prepare_scala_alias_project(Dir),
        run_scala_alias_test(Dir),
        cleanup_test_dir(Dir)).

:- end_tests(wam_scala_foreign_tuple_aliases).

prepare_scala_alias_project(Dir) :-
    cleanup_test_dir(Dir),
    write_wam_scala_project(
        [],
        [ package('generated.alias'),
          runtime_package('generated.alias'),
          module_name('aliasruntime')
        ],
        Dir),
    directory_file_path(Dir, 'src/main/scala/generated/alias', SourceDir),
    directory_file_path(SourceDir, 'WamRuntime.scala', RuntimePath),
    (   scala_211_toolchain
    ->  make_runtime_scalac_211_compatible(RuntimePath)
    ;   true
    ),
    directory_file_path(SourceDir, 'ForeignTupleAliasesTest.scala', TestPath),
    scala_alias_test_source(Source),
    setup_call_cleanup(
        open(TestPath, write, Stream, [encoding(utf8)]),
        format(Stream, '~s', [Source]),
        close(Stream)).

run_scala_alias_test(Dir) :-
    absolute_file_name(Dir, AbsDir),
    get_time(Now),
    Stamp is floor(Now * 1000000),
    format(atom(ClassesDir), '/tmp/uw_scala_alias_classes_~d', [Stamp]),
    setup_call_cleanup(
        make_directory_path(ClassesDir),
        compile_and_run_scala_alias_test(AbsDir, ClassesDir),
        cleanup_test_dir(ClassesDir)).

compile_and_run_scala_alias_test(AbsDir, ClassesDir) :-
    directory_file_path(AbsDir, src, SourceRoot),
    findall(Source,
            ( directory_member(SourceRoot, Relative,
                               [extensions([scala]), recursive(true)]),
              directory_file_path(SourceRoot, Relative, Source)
            ),
            Sources),
    (   scala_211_toolchain
    ->  CompilePrefix = ['-usejavacp']
    ;   CompilePrefix = []
    ),
    append(CompilePrefix, ['-d', ClassesDir | Sources], CompileArgs),
    process_create(path(scalac), CompileArgs,
                   [cwd(AbsDir), stdout(pipe(CompileOut)), stderr(pipe(CompileErr)),
                    process(CompilePid)]),
    read_string(CompileOut, _, CompileOutText),
    read_string(CompileErr, _, CompileErrText),
    close(CompileOut),
    close(CompileErr),
    process_wait(CompilePid, CompileStatus),
    (   CompileStatus == exit(0)
    ->  true
    ;   throw(error(scala_alias_compile_failed(
                        CompileStatus, CompileOutText, CompileErrText), _))
    ),
    (   scala_211_toolchain
    ->  RunPrefix = ['-usejavacp']
    ;   RunPrefix = []
    ),
    append(RunPrefix,
           ['-classpath', ClassesDir,
            'generated.alias.ForeignTupleAliasesTest'],
           RunArgs),
    process_create(path(scala), RunArgs,
                   [cwd(AbsDir), stdout(pipe(RunOut)), stderr(pipe(RunErr)),
                    process(RunPid)]),
    read_string(RunOut, _, RunOutText),
    read_string(RunErr, _, RunErrText),
    close(RunOut),
    close(RunErr),
    process_wait(RunPid, RunStatus),
    (   RunStatus == exit(0)
    ->  true
    ;   throw(error(scala_alias_run_failed(
                        RunStatus, RunOutText, RunErrText), _))
    ).

cleanup_test_dir(Dir) :-
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

% Debian's locally installable compiler is Scala 2.11, while generated
% projects target Scala 3.  The runtime is otherwise source-compatible; this
% generated-file-only rewrite replaces the one 2.13+ String convenience used
% outside the code under test so the executable regression can run locally.
make_runtime_scalac_211_compatible(RuntimePath) :-
    read_file_to_string(RuntimePath, Source0, []),
    atom_string(SourceAtom0, Source0),
    Old = 'name.tail.toIntOption.getOrElse(0)',
    New = 'scala.util.Try(name.tail.toInt).getOrElse(0)',
    atomic_list_concat(Parts, Old, SourceAtom0),
    atomic_list_concat(Parts, New, SourceAtom),
    atom_string(SourceAtom, Source),
    setup_call_cleanup(
        open(RuntimePath, write, Stream, [encoding(utf8)]),
        format(Stream, '~s', [Source]),
        close(Stream)).

scala_alias_test_source(
"package generated.alias

import scala.collection.immutable.ListMap

object ForeignTupleAliasesTest {
  private val shared = Ref(1000)
  private val distance = Ref(1001)

  private def solution(left: Int, right: Int, d: Int): Map[Int, WamTerm] =
    ListMap(1 -> Atom(left), 2 -> Atom(right), 3 -> IntTerm(d))

  private def machine(solutions: Seq[Map[Int, WamTerm]]): (WamState, WamProgram) = {
    val state = WamRuntime.newState(0, Array.empty[WamTerm])
    state.regs(1) = shared
    state.regs(2) = shared
    state.regs(3) = distance
    val handler = new ForeignHandler {
      def apply(args: Array[WamTerm]): ForeignResult = ForeignMulti(solutions)
    }
    val program = WamProgram(
      instructions = Array.empty[Instruction],
      labels = Map.empty,
      foreignHandlers = Map(\"alias/3\" -> handler),
      dispatch = Map.empty,
      internTable = InternTable(Array(\"zero\", \"one\", \"two\", \"three\", \"four\"))
    )
    (state, program)
  }

  private def requireAtom(state: WamState, value: WamTerm, expected: Int): Unit =
    assert(WamRuntime.deref(state.bindings, value) == Atom(expected))

  private def requireInteger(state: WamState, value: WamTerm, expected: Int): Unit =
    assert(WamRuntime.deref(state.bindings, value) == IntTerm(expected))

  private def requireClean(state: WamState): Unit = {
    assert(WamRuntime.deref(state.bindings, shared) == shared)
    assert(WamRuntime.deref(state.bindings, distance) == distance)
    assert(state.bindings.isEmpty)
    assert(state.trail.isEmpty)
    assert(state.choicePoints.isEmpty)
  }

  private def incompatibleFirstThenCompatible(): Unit = {
    val (state, program) = machine(Seq(
      solution(1, 2, 1),
      solution(3, 3, 2)
    ))
    WamRuntime.executeForeign(state, program, \"alias\", 3)
    requireAtom(state, shared, 3)
    requireInteger(state, distance, 2)
    assert(state.choicePoints.length == 1)
    WamRuntime.backtrack(state)
    requireClean(state)
  }

  private def retrySkipsIncompatibleThenFindsCompatible(): Unit = {
    val (state, program) = machine(Seq(
      solution(1, 1, 1),
      solution(2, 3, 2),
      solution(4, 4, 3)
    ))
    WamRuntime.executeForeign(state, program, \"alias\", 3)
    requireAtom(state, shared, 1)
    assert(state.choicePoints.length == 1)
    WamRuntime.backtrack(state)
    requireAtom(state, shared, 4)
    requireInteger(state, distance, 3)
    assert(state.choicePoints.length == 1)
    WamRuntime.backtrack(state)
    requireClean(state)
  }

  private def allIncompatibleLeavesNoState(): Unit = {
    val (state, program) = machine(Seq(
      solution(1, 2, 1),
      solution(3, 4, 2)
    ))
    WamRuntime.executeForeign(state, program, \"alias\", 3)
    requireClean(state)
  }

  def main(args: Array[String]): Unit = {
    incompatibleFirstThenCompatible()
    retrySkipsIncompatibleThenFindsCompatible()
    allIncompatibleLeavesNoState()
  }
}
").
