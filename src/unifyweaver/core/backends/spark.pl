:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% spark.pl - Apache Spark backend for distributed parallel execution
% Supports both PySpark (Python) and native Scala modes
% Best for in-memory distributed computing with large datasets

:- module(spark_backend, [
    backend_init_impl/2,
    backend_execute_impl/4,
    backend_cleanup_impl/1,

    % Code generation exports
    generate_pyspark_job/4,
    generate_scala_spark_job/4
]).

:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module(library(lists)).

%% ============================================
%% BACKEND IMPLEMENTATION
%% ============================================

%% backend_init_impl(+Config, -State)
%  Initialize Spark backend
%
%  Configuration options:
%  - backend_args([mode(M)]) - pyspark | scala | auto (default: auto)
%  - backend_args([master(M)]) - local[*] | yarn | spark://host:port | k8s://...
%  - backend_args([app_name(N)]) - Spark application name
%  - backend_args([executor_memory(M)]) - Executor memory (e.g., '4g')
%  - backend_args([driver_memory(M)]) - Driver memory (e.g., '2g')
%  - backend_args([num_executors(N)]) - Number of executors (YARN/K8s)
%  - backend_args([executor_cores(N)]) - Cores per executor
%  - backend_args([partitions(N)]) - Default parallelism
%  - backend_args([checkpoint_dir(P)]) - Checkpoint directory for fault tolerance
%  - backend_args([packages(L)]) - Additional Maven packages
%  - backend_args([py_files(L)]) - Additional Python files to distribute
%
%  @example Initialize PySpark with YARN
%    ?- backend_init_impl([backend_args([mode(pyspark), master(yarn), executor_memory('4g')])], State).
%
%  @example Initialize Scala Spark with local cluster
%    ?- backend_init_impl([backend_args([mode(scala), master('local[4]')])], State).
backend_init_impl(Config, State) :-
    % Check Spark is available
    check_spark_available(SparkHome, SparkVersion),

    % Extract configuration
    (   member(backend_args(Args), Config)
    ->  true
    ;   Args = []
    ),

    % Parse mode (pyspark or scala)
    (   member(mode(Mode), Args)
    ->  true
    ;   % Auto-detect: prefer pyspark if available
        (   check_pyspark_available
        ->  Mode = pyspark
        ;   Mode = scala
        )
    ),
    validate_spark_mode(Mode),

    % Parse master URL
    (   member(master(Master), Args)
    ->  true
    ;   Master = 'local[*]'
    ),

    % Parse application name
    (   member(app_name(AppName), Args)
    ->  true
    ;   AppName = 'UnifyWeaver'
    ),

    % Parse memory settings
    (   member(executor_memory(ExecMem), Args)
    ->  true
    ;   ExecMem = '1g'
    ),
    (   member(driver_memory(DriverMem), Args)
    ->  true
    ;   DriverMem = '1g'
    ),

    % Parse executor settings
    (   member(num_executors(NumExecutors), Args)
    ->  true
    ;   NumExecutors = 2
    ),
    (   member(executor_cores(ExecCores), Args)
    ->  true
    ;   ExecCores = 1
    ),

    % Parse partitions
    (   member(partitions(Partitions), Args)
    ->  true
    ;   Partitions = 4
    ),

    % Parse checkpoint directory
    (   member(checkpoint_dir(CheckpointDir), Args)
    ->  true
    ;   CheckpointDir = none
    ),

    % Parse additional packages/files
    (   member(packages(Packages), Args)
    ->  true
    ;   Packages = []
    ),
    (   member(py_files(PyFiles), Args)
    ->  true
    ;   PyFiles = []
    ),

    % Create temporary directory
    create_temp_directory(TempDir),

    % Initialize state
    State = state(
        spark_home(SparkHome),
        spark_version(SparkVersion),
        mode(Mode),
        master(Master),
        app_name(AppName),
        executor_memory(ExecMem),
        driver_memory(DriverMem),
        num_executors(NumExecutors),
        executor_cores(ExecCores),
        partitions(Partitions),
        checkpoint_dir(CheckpointDir),
        packages(Packages),
        py_files(PyFiles),
        temp_dir(TempDir)
    ),

    format('[Spark] Initialized: mode=~w, master=~w, version=~w~n',
           [Mode, Master, SparkVersion]).

%% validate_spark_mode(+Mode)
validate_spark_mode(pyspark) :- !.
validate_spark_mode(scala) :- !.
validate_spark_mode(auto) :- !.
validate_spark_mode(Mode) :-
    throw(error(domain_error(spark_mode, Mode),
                context(validate_spark_mode/1,
                       'Mode must be: pyspark, scala, or auto'))).

%% backend_execute_impl(+State, +Partitions, +ScriptPath, -Results)
%  Execute using Spark (PySpark or Scala)
backend_execute_impl(State, Partitions, ScriptPath, Results) :-
    State = state(
        spark_home(SparkHome),
        spark_version(_),
        mode(Mode),
        master(Master),
        app_name(AppName),
        executor_memory(ExecMem),
        driver_memory(DriverMem),
        num_executors(NumExecutors),
        executor_cores(ExecCores),
        partitions(DefaultPartitions),
        checkpoint_dir(CheckpointDir),
        packages(Packages),
        py_files(PyFiles),
        temp_dir(TempDir)
    ),

    length(Partitions, NumPartitions),
    format('[Spark] Executing ~w partitions in ~w mode~n', [NumPartitions, Mode]),

    % Write input data
    write_input_data(Partitions, TempDir, InputPath),
    format(atom(OutputPath), '~w/output', [TempDir]),

    % Generate and execute Spark job based on mode
    (   Mode = pyspark
    ->  generate_pyspark_job(ScriptPath, InputPath, OutputPath,
                              [partitions(DefaultPartitions), checkpoint(CheckpointDir)],
                              PySparkCode),
        format(atom(JobFile), '~w/spark_job.py', [TempDir]),
        write_file(JobFile, PySparkCode),
        build_pyspark_command(SparkHome, Master, AppName, ExecMem, DriverMem,
                              NumExecutors, ExecCores, PyFiles, JobFile, Command)
    ;   % Scala mode
        generate_scala_spark_job(ScriptPath, InputPath, OutputPath,
                                  [partitions(DefaultPartitions), checkpoint(CheckpointDir)],
                                  ScalaCode),
        format(atom(JobFile), '~w/SparkJob.scala', [TempDir]),
        write_file(JobFile, ScalaCode),
        compile_scala_spark_job(SparkHome, TempDir, JobFile, JarPath),
        build_spark_submit_command(SparkHome, Master, AppName, ExecMem, DriverMem,
                                   NumExecutors, ExecCores, Packages, JarPath, Command)
    ),

    format('[Spark] Executing: ~w~n', [Command]),
    execute_spark_command(Command, ExitCode),

    (   ExitCode =:= 0
    ->  format('[Spark] Job completed successfully~n', [])
    ;   format('[Spark] Warning: Exit code ~w~n', [ExitCode])
    ),

    % Collect results
    collect_results(OutputPath, Partitions, Results),
    format('[Spark] Collected results~n', []).

%% backend_cleanup_impl(+State)
%  Clean up Spark backend resources
backend_cleanup_impl(state(_, _, _, _, _, _, _, _, _, _, _, _, _, temp_dir(TempDir))) :-
    format('[Spark] Cleaning up...~n', []),
    (   exists_directory(TempDir)
    ->  delete_directory_and_contents(TempDir)
    ;   true
    ),
    format('[Spark] Cleanup complete~n', []).

%% ============================================
%% PYSPARK CODE GENERATION
%% ============================================

%% generate_pyspark_job(+ScriptPath, +InputPath, +OutputPath, +Options, -Code)
%  Generate PySpark job Python code
generate_pyspark_job(ScriptPath, InputPath, OutputPath, Options, Code) :-
    (   member(partitions(NumPartitions), Options)
    ->  true
    ;   NumPartitions = 4
    ),
    (   member(checkpoint(CheckpointDir), Options), CheckpointDir \= none
    ->  format(string(CheckpointCode),
               '    sc.setCheckpointDir("~w")', [CheckpointDir])
    ;   CheckpointCode = '    # No checkpoint configured'
    ),

    format(string(Code),
'#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UnifyWeaver Spark Job - Generated PySpark Code
Script reference: ~w
"""

import sys
import subprocess
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

def process_partition(partition):
    """Process a partition of records using the bash script."""
    results = []
    for record in partition:
        try:
            # Execute script with record as input
            result = subprocess.run(
                ["bash", "~w"],
                input=str(record),
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0 and result.stdout.strip():
                results.append(result.stdout.strip())
        except Exception as e:
            print(f"[Spark] Error processing record: {e}", file=sys.stderr)
    return results

def main():
    """Main Spark job execution."""
    # Initialize Spark
    spark = SparkSession.builder \\
        .appName("UnifyWeaver") \\
        .getOrCreate()

    sc = spark.sparkContext
    print(f"[Spark] Spark version: {spark.version}")
    print(f"[Spark] Application ID: {sc.applicationId}")

~w

    try:
        # Read input data
        input_rdd = sc.textFile("~w")
        print(f"[Spark] Loaded input data")

        # Repartition for parallelism
        input_rdd = input_rdd.repartition(~w)

        # Process partitions
        result_rdd = input_rdd.mapPartitions(process_partition)

        # Collect and save results
        result_rdd.saveAsTextFile("~w")

        print(f"[Spark] Results saved to: ~w")
        print(f"[Spark] Job completed successfully")

    except Exception as e:
        print(f"[Spark] Job failed: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
', [ScriptPath, ScriptPath, CheckpointCode, InputPath, NumPartitions, OutputPath, OutputPath]).

%% ============================================
%% SCALA SPARK CODE GENERATION
%% ============================================

%% generate_scala_spark_job(+ScriptPath, +InputPath, +OutputPath, +Options, -Code)
%  Generate Scala Spark job code
generate_scala_spark_job(ScriptPath, InputPath, OutputPath, Options, Code) :-
    (   member(partitions(NumPartitions), Options)
    ->  true
    ;   NumPartitions = 4
    ),
    (   member(checkpoint(CheckpointDir), Options), CheckpointDir \= none
    ->  format(string(CheckpointCode),
               '    sc.setCheckpointDir("~w")', [CheckpointDir])
    ;   CheckpointCode = '    // No checkpoint configured'
    ),

    format(string(Code),
'// UnifyWeaver Spark Job - Generated Scala Code
// Script reference: ~w

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import scala.sys.process._
import scala.collection.JavaConverters._

/**
 * UnifyWeaver Spark Job
 */
object SparkJob {

  def processPartition(iterator: Iterator[String]): Iterator[String] = {
    iterator.flatMap { record =>
      try {
        val result = Process(Seq("bash", "~w"))
          .#<(new java.io.ByteArrayInputStream(record.getBytes))
          .!!
        if (result.trim.nonEmpty) Some(result.trim) else None
      } catch {
        case e: Exception =>
          System.err.println(s"[Spark] Error processing record: ${e.getMessage}")
          None
      }
    }
  }

  def main(args: Array[String]): Unit = {
    // Initialize Spark
    val spark = SparkSession.builder()
      .appName("UnifyWeaver")
      .getOrCreate()

    val sc = spark.sparkContext
    println(s"[Spark] Spark version: ${spark.version}")
    println(s"[Spark] Application ID: ${sc.applicationId}")

~w

    try {
      // Read input data
      val inputRDD = sc.textFile("~w")
      println("[Spark] Loaded input data")

      // Repartition for parallelism
      val repartitioned = inputRDD.repartition(~w)

      // Process partitions
      val resultRDD = repartitioned.mapPartitions(processPartition)

      // Save results
      resultRDD.saveAsTextFile("~w")

      println(s"[Spark] Results saved to: ~w")
      println("[Spark] Job completed successfully")

    } catch {
      case e: Exception =>
        System.err.println(s"[Spark] Job failed: ${e.getMessage}")
        sys.exit(1)
    } finally {
      spark.stop()
    }
  }
}
', [ScriptPath, ScriptPath, CheckpointCode, InputPath, NumPartitions, OutputPath, OutputPath]).

%% ============================================
%% SPARK COMMAND BUILDING
%% ============================================

%% build_pyspark_command(+SparkHome, +Master, +AppName, +ExecMem, +DriverMem,
%%                       +NumExecutors, +ExecCores, +PyFiles, +JobFile, -Command)
build_pyspark_command(SparkHome, Master, AppName, ExecMem, DriverMem,
                      NumExecutors, ExecCores, PyFiles, JobFile, Command) :-
    (   SparkHome = ''
    ->  SparkSubmit = 'spark-submit'
    ;   format(atom(SparkSubmit), '~w/bin/spark-submit', [SparkHome])
    ),

    % Build py-files argument if any
    (   PyFiles = []
    ->  PyFilesArg = ''
    ;   atomic_list_concat(PyFiles, ',', PyFilesList),
        format(atom(PyFilesArg), '--py-files "~w"', [PyFilesList])
    ),

    format(atom(Command),
           '~w --master ~w --name "~w" --executor-memory ~w --driver-memory ~w --num-executors ~w --executor-cores ~w ~w "~w"',
           [SparkSubmit, Master, AppName, ExecMem, DriverMem,
            NumExecutors, ExecCores, PyFilesArg, JobFile]).

%% build_spark_submit_command(+SparkHome, +Master, +AppName, +ExecMem, +DriverMem,
%%                            +NumExecutors, +ExecCores, +Packages, +JarPath, -Command)
build_spark_submit_command(SparkHome, Master, AppName, ExecMem, DriverMem,
                           NumExecutors, ExecCores, Packages, JarPath, Command) :-
    (   SparkHome = ''
    ->  SparkSubmit = 'spark-submit'
    ;   format(atom(SparkSubmit), '~w/bin/spark-submit', [SparkHome])
    ),

    % Build packages argument if any
    (   Packages = []
    ->  PackagesArg = ''
    ;   atomic_list_concat(Packages, ',', PackagesList),
        format(atom(PackagesArg), '--packages "~w"', [PackagesList])
    ),

    format(atom(Command),
           '~w --master ~w --name "~w" --executor-memory ~w --driver-memory ~w --num-executors ~w --executor-cores ~w ~w --class SparkJob "~w"',
           [SparkSubmit, Master, AppName, ExecMem, DriverMem,
            NumExecutors, ExecCores, PackagesArg, JarPath]).

%% compile_scala_spark_job(+SparkHome, +TempDir, +ScalaFile, -JarPath)
compile_scala_spark_job(SparkHome, TempDir, ScalaFile, JarPath) :-
    % Get Spark JARs for classpath
    (   SparkHome = ''
    ->  % Try to get from spark-shell
        process_create(path('spark-shell'), ['--version'],
                       [stdout(null), stderr(null), process(_)])
    ;   true
    ),

    format(atom(JarPath), '~w/spark_job.jar', [TempDir]),
    format(atom(ClassDir), '~w/classes', [TempDir]),
    make_directory(ClassDir),

    % Get Spark classpath
    (   SparkHome = ''
    ->  SparkJars = '*'
    ;   format(atom(SparkJars), '~w/jars/*', [SparkHome])
    ),

    % Compile
    format(atom(CompileCmd),
           'scalac -cp "~w" -d "~w" "~w" 2>&1',
           [SparkJars, ClassDir, ScalaFile]),
    execute_shell(CompileCmd, CompileExit),
    (   CompileExit =:= 0
    ->  true
    ;   format('[Spark] Warning: Compilation may have issues~n', [])
    ),

    % Package JAR
    format(atom(JarCmd), 'jar -cvf "~w" -C "~w" . 2>&1', [JarPath, ClassDir]),
    execute_shell(JarCmd, _).

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% check_spark_available(-SparkHome, -SparkVersion)
check_spark_available(SparkHome, SparkVersion) :-
    % Try SPARK_HOME first
    (   getenv('SPARK_HOME', SparkHome)
    ->  true
    ;   % Try spark-submit in PATH
        catch(
            (   process_create(path('spark-submit'), ['--version'],
                              [stdout(pipe(S)), stderr(pipe(E)), process(PID)]),
                read_string(S, _, _),
                read_string(E, _, VersionStr),
                close(S),
                close(E),
                process_wait(PID, exit(_)),
                SparkHome = ''
            ),
            _,
            (   format('[Spark] ERROR: Spark not found~n', []),
                format('[Spark] Set SPARK_HOME or add spark-submit to PATH~n', []),
                throw(error(existence_error(runtime, spark),
                           context(check_spark_available/2,
                                  'Spark not found')))
            )
        )
    ),

    % Get version
    (   SparkHome = ''
    ->  VersionCmd = 'spark-submit --version 2>&1 | head -n 1'
    ;   format(atom(VersionCmd), '~w/bin/spark-submit --version 2>&1 | head -n 1', [SparkHome])
    ),
    process_create('/bin/bash', ['-c', VersionCmd],
                   [stdout(pipe(VS)), stderr(null), process(VP)]),
    read_string(VS, _, VersionOutput),
    close(VS),
    process_wait(VP, exit(_)),
    (   sub_string(VersionOutput, _, _, _, 'version')
    ->  normalize_space(atom(SparkVersion), VersionOutput)
    ;   SparkVersion = unknown
    ).

%% check_pyspark_available
check_pyspark_available :-
    catch(
        (   process_create(path(python3), ['-c', 'import pyspark'],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(0))
        ),
        _,
        fail
    ).

%% create_temp_directory(-TempDir)
create_temp_directory(TempDir) :-
    get_time(Timestamp),
    format(atom(TempDir), '/tmp/unifyweaver_spark_~w', [Timestamp]),
    make_directory(TempDir).

%% write_input_data(+Partitions, +TempDir, -InputPath)
write_input_data(Partitions, TempDir, InputPath) :-
    format(atom(InputPath), '~w/input', [TempDir]),
    make_directory(InputPath),
    forall(member(partition(ID, Data), Partitions),
           (   format(atom(FilePath), '~w/part-~5|~`0t~d~5|.txt', [InputPath, ID]),
               open(FilePath, write, Stream),
               forall(member(Item, Data),
                      (   (atom(Item) ; string(Item))
                      ->  writeln(Stream, Item)
                      ;   write_term(Stream, Item, [quoted(true), nl(true)])
                      )),
               close(Stream)
           )).

%% write_file(+Path, +Content)
write_file(Path, Content) :-
    open(Path, write, Stream, [encoding(utf8)]),
    write(Stream, Content),
    close(Stream).

%% execute_spark_command(+Command, -ExitCode)
execute_spark_command(Command, ExitCode) :-
    process_create('/bin/bash', ['-c', Command],
                   [stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
    read_string(Out, _, StdOut),
    read_string(Err, _, StdErr),
    close(Out),
    close(Err),
    process_wait(PID, exit(ExitCode)),
    (   StdOut \= "" -> format('[Spark] ~w~n', [StdOut]) ; true ),
    (   StdErr \= "" -> format('[Spark] ~w~n', [StdErr]) ; true ).

%% execute_shell(+Command, -ExitCode)
execute_shell(Command, ExitCode) :-
    process_create('/bin/bash', ['-c', Command],
                   [stdout(null), stderr(null), process(PID)]),
    process_wait(PID, exit(ExitCode)).

%% collect_results(+OutputPath, +Partitions, -Results)
collect_results(OutputPath, Partitions, Results) :-
    % Find and merge all part files
    format(atom(Pattern), '~w/part-*', [OutputPath]),
    expand_file_name(Pattern, PartFiles),
    (   PartFiles = []
    ->  AllOutput = ""
    ;   maplist(read_file_safe, PartFiles, PartOutputs),
        atomic_list_concat(PartOutputs, AllOutput)
    ),
    % Create result for each partition
    maplist(create_partition_result(AllOutput), Partitions, Results).

read_file_safe(File, Content) :-
    (   exists_file(File)
    ->  read_file_to_string(File, Content, [])
    ;   Content = ""
    ).

create_partition_result(Output, partition(ID, _), result(ID, Output)).
