:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% demo_distributed_backends.pl - Demonstration of distributed parallel backends
%
% This example demonstrates using Dask and Hadoop Streaming backends
% for parallel batch processing.
%
% Prerequisites:
%   - For Dask: pip install dask distributed
%   - For Hadoop: HADOOP_HOME set or hadoop in PATH
%
% Usage:
%   ?- [examples/demo_distributed_backends].
%   ?- demo_dask_basic.
%   ?- demo_dask_distributed.
%   ?- demo_hadoop_basic.

:- use_module(library(lists)).

% Load backend system
:- use_module(unifyweaver(core/parallel_backend)).
:- use_module(unifyweaver(core/partitioner)).
:- use_module(unifyweaver(core/backend_loader)).

%% ============================================
%% SETUP
%% ============================================

%% setup_backends
%  Load all available backends
setup_backends :-
    format('~n=== Loading Distributed Backends ===~n', []),
    load_all_backends,
    format('~n', []).

%% create_test_script(-ScriptPath)
%  Create a simple test script for processing
create_test_script(ScriptPath) :-
    ScriptPath = '/tmp/test_processor.sh',
    Script = '#!/bin/bash
# Simple test processor - doubles each number
while IFS= read -r line; do
    if [[ "$line" =~ ^[0-9]+$ ]]; then
        echo $((line * 2))
    else
        echo "$line"
    fi
done
',
    open(ScriptPath, write, Stream),
    write(Stream, Script),
    close(Stream),
    process_create(path(chmod), ['+x', ScriptPath], []).

%% create_test_data(-Data)
%  Generate test data (numbers 1-100)
create_test_data(Data) :-
    numlist(1, 100, Data).

%% ============================================
%% DASK DEMONSTRATIONS
%% ============================================

%% demo_dask_basic
%  Basic Dask demonstration using threaded scheduler
demo_dask_basic :-
    format('~n=== Dask Basic Demo (Threaded) ===~n', []),

    % Check if Dask is available
    (   check_dask_available
    ->  true
    ;   format('Dask not available. Install with: pip install dask distributed~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    create_test_data(Data),
    format('Test data: ~w items~n', [100]),

    % Partition data
    partitioner_init(fixed_size(25), [], PartHandle),
    partitioner_partition(PartHandle, Data, Partitions),
    length(Partitions, NumPartitions),
    format('Created ~w partitions~n', [NumPartitions]),
    partitioner_cleanup(PartHandle),

    % Initialize Dask backend with threads
    format('~nInitializing Dask with threaded scheduler...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/dask_distributed)),
            register_backend(dask_distributed, dask_distributed_backend)
        ),
        _,
        true
    ),
    backend_init(dask_distributed(workers(4), scheduler(threads)), Handle),

    % Execute
    format('~nExecuting parallel processing...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_dask_distributed
%  Dask demonstration using local distributed cluster
demo_dask_distributed :-
    format('~n=== Dask Distributed Demo (Local Cluster) ===~n', []),

    % Check if Dask is available
    (   check_dask_available
    ->  true
    ;   format('Dask not available. Install with: pip install dask distributed~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    create_test_data(Data),

    % Partition data
    partitioner_init(fixed_size(20), [], PartHandle),
    partitioner_partition(PartHandle, Data, Partitions),
    length(Partitions, NumPartitions),
    format('Created ~w partitions~n', [NumPartitions]),
    partitioner_cleanup(PartHandle),

    % Initialize Dask backend with distributed scheduler (local cluster)
    format('~nInitializing Dask distributed cluster...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/dask_distributed)),
            register_backend(dask_distributed, dask_distributed_backend)
        ),
        _,
        true
    ),
    backend_init(dask_distributed(
        workers(4),
        scheduler(distributed),
        memory_limit('1GB')
    ), Handle),

    % Execute
    format('~nExecuting parallel processing on cluster...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_dask_synchronous
%  Dask demonstration using synchronous scheduler (for debugging)
demo_dask_synchronous :-
    format('~n=== Dask Synchronous Demo (Single-threaded) ===~n', []),

    % Check if Dask is available
    (   check_dask_available
    ->  true
    ;   format('Dask not available~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5],

    % Simple partition
    Partitions = [partition(0, Data)],

    % Initialize with synchronous scheduler
    catch(
        (   use_module(unifyweaver(core/backends/dask_distributed)),
            register_backend(dask_distributed, dask_distributed_backend)
        ),
        _,
        true
    ),
    backend_init(dask_distributed(scheduler(synchronous)), Handle),

    % Execute
    format('Executing synchronously...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('Results: ~w~n', [Results]),

    % Cleanup
    backend_cleanup(Handle),
    format('=== Demo Complete ===~n', []).

%% ============================================
%% HADOOP DEMONSTRATIONS
%% ============================================

%% demo_hadoop_basic
%  Basic Hadoop Streaming demonstration
demo_hadoop_basic :-
    format('~n=== Hadoop Streaming Basic Demo ===~n', []),

    % Check if Hadoop is available
    (   check_hadoop_available
    ->  true
    ;   format('Hadoop not available. Set HADOOP_HOME or install Hadoop~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    create_test_data(Data),
    format('Test data: ~w items~n', [100]),

    % Partition data
    partitioner_init(fixed_size(50), [], PartHandle),
    partitioner_partition(PartHandle, Data, Partitions),
    length(Partitions, NumPartitions),
    format('Created ~w partitions~n', [NumPartitions]),
    partitioner_cleanup(PartHandle),

    % Initialize Hadoop Streaming backend
    format('~nInitializing Hadoop Streaming...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/hadoop_streaming)),
            register_backend(hadoop_streaming, hadoop_streaming_backend)
        ),
        _,
        true
    ),
    backend_init(hadoop_streaming(reducers(1)), Handle),

    % Execute
    format('~nExecuting MapReduce job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_hadoop_wordcount
%  Classic word count example with Hadoop
demo_hadoop_wordcount :-
    format('~n=== Hadoop Word Count Demo ===~n', []),

    % Check if Hadoop is available
    (   check_hadoop_available
    ->  true
    ;   format('Hadoop not available~n', []),
        fail
    ),

    % Create mapper script
    MapperPath = '/tmp/wordcount_mapper.sh',
    MapperScript = '#!/bin/bash
# Word count mapper - emit word<TAB>1 for each word
while IFS= read -r line; do
    for word in $line; do
        # Normalize: lowercase, remove punctuation
        clean=$(echo "$word" | tr "[:upper:]" "[:lower:]" | tr -d "[:punct:]")
        if [ -n "$clean" ]; then
            printf "%s\\t1\\n" "$clean"
        fi
    done
done
',
    open(MapperPath, write, MapperStream),
    write(MapperStream, MapperScript),
    close(MapperStream),
    process_create(path(chmod), ['+x', MapperPath], []),

    % Create reducer script
    ReducerPath = '/tmp/wordcount_reducer.sh',
    ReducerScript = '#!/bin/bash
# Word count reducer - sum counts for each word
declare -A counts
while IFS=$\'\\t\' read -r word count; do
    if [ -n "$word" ]; then
        ((counts[$word] += count))
    fi
done

# Output sorted by count (descending)
for word in "${!counts[@]}"; do
    printf "%s\\t%d\\n" "$word" "${counts[$word]}"
done | sort -t$\'\\t\' -k2 -nr
',
    open(ReducerPath, write, ReducerStream),
    write(ReducerStream, ReducerScript),
    close(ReducerStream),
    process_create(path(chmod), ['+x', ReducerPath], []),

    % Sample text data
    TextData = [
        'The quick brown fox jumps over the lazy dog',
        'The dog barks at the fox',
        'Quick brown dogs are lazy',
        'The fox is quick and brown',
        'Lazy dogs sleep all day'
    ],

    % Create partitions
    Partitions = [partition(0, TextData)],

    % Initialize Hadoop with custom mapper/reducer
    catch(
        (   use_module(unifyweaver(core/backends/hadoop_streaming)),
            register_backend(hadoop_streaming, hadoop_streaming_backend)
        ),
        _,
        true
    ),
    backend_init(hadoop_streaming(
        mapper(MapperPath),
        reducer(ReducerPath),
        reducers(1)
    ), Handle),

    % Execute
    format('Executing word count MapReduce...~n', []),
    backend_execute(Handle, Partitions, MapperPath, Results),

    % Show results
    format('~nWord counts:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% ============================================
%% SPARK DEMONSTRATIONS
%% ============================================

%% demo_spark_pyspark
%  Demonstration using PySpark
demo_spark_pyspark :-
    format('~n=== Spark PySpark Demo ===~n', []),

    % Check if Spark is available
    (   check_spark_available
    ->  true
    ;   format('Spark not available. Set SPARK_HOME or add spark-submit to PATH~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    create_test_data(Data),
    format('Test data: ~w items~n', [100]),

    % Partition data
    partitioner_init(fixed_size(25), [], PartHandle),
    partitioner_partition(PartHandle, Data, Partitions),
    partitioner_cleanup(PartHandle),

    % Initialize Spark backend in PySpark mode
    format('~nInitializing PySpark...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/spark)),
            register_backend(spark, spark_backend)
        ),
        _,
        true
    ),
    backend_init(spark(
        mode(pyspark),
        master('local[4]'),
        executor_memory('1g')
    ), Handle),

    % Execute
    format('~nExecuting Spark job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_spark_scala
%  Demonstration using native Scala Spark
demo_spark_scala :-
    format('~n=== Spark Scala Demo ===~n', []),

    % Check if Spark and Scala are available
    (   check_spark_available, check_scala_available
    ->  true
    ;   format('Spark or Scala not available~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    % Simple partition
    Partitions = [partition(0, Data)],

    % Initialize Spark backend in Scala mode
    format('~nInitializing Scala Spark...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/spark)),
            register_backend(spark, spark_backend)
        ),
        _,
        true
    ),
    backend_init(spark(
        mode(scala),
        master('local[2]'),
        executor_memory('512m')
    ), Handle),

    % Execute
    format('~nExecuting Spark job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_hadoop_native_java
%  Demonstration using native Hadoop Java API
demo_hadoop_native_java :-
    format('~n=== Hadoop Native Java Demo ===~n', []),

    % Check availability
    (   check_hadoop_available, check_java_available
    ->  true
    ;   format('Hadoop or Java not available~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    Partitions = [partition(0, Data)],

    % Initialize Hadoop Native backend
    format('~nInitializing Hadoop Native (Java)...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/hadoop_native)),
            register_backend(hadoop_native, hadoop_native_backend)
        ),
        _,
        true
    ),
    backend_init(hadoop_native(
        target_language(java),
        reducers(1)
    ), Handle),

    % Execute
    format('~nExecuting MapReduce job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_hadoop_native_clojure
%  Demonstration using native Hadoop Clojure API
demo_hadoop_native_clojure :-
    format('~n=== Hadoop Native Clojure Demo ===~n', []),

    % Check availability
    (   check_hadoop_available, check_clojure_available
    ->  true
    ;   format('Hadoop or Clojure not available~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    Partitions = [partition(0, Data)],

    % Initialize Hadoop Native backend with Clojure
    format('~nInitializing Hadoop Native (Clojure)...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/hadoop_native)),
            register_backend(hadoop_native, hadoop_native_backend)
        ),
        _,
        true
    ),
    backend_init(hadoop_native(
        target_language(clojure),
        reducers(1),
        combiner(true)
    ), Handle),

    % Execute
    format('~nExecuting MapReduce job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_spark_java
%  Demonstration using Java Spark
demo_spark_java :-
    format('~n=== Spark Java Demo ===~n', []),

    % Check if Spark and Java are available
    (   check_spark_available, check_java_available
    ->  true
    ;   format('Spark or Java not available~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    Partitions = [partition(0, Data)],

    % Initialize Spark backend in Java mode
    format('~nInitializing Java Spark...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/spark)),
            register_backend(spark, spark_backend)
        ),
        _,
        true
    ),
    backend_init(spark(
        mode(java),
        master('local[2]'),
        executor_memory('512m')
    ), Handle),

    % Execute
    format('~nExecuting Spark job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_spark_kotlin
%  Demonstration using Kotlin Spark
demo_spark_kotlin :-
    format('~n=== Spark Kotlin Demo ===~n', []),

    % Check if Spark and Kotlin are available
    (   check_spark_available, check_kotlin_available
    ->  true
    ;   format('Spark or Kotlin not available~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    Partitions = [partition(0, Data)],

    % Initialize Spark backend in Kotlin mode
    format('~nInitializing Kotlin Spark...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/spark)),
            register_backend(spark, spark_backend)
        ),
        _,
        true
    ),
    backend_init(spark(
        mode(kotlin),
        master('local[2]'),
        executor_memory('512m')
    ), Handle),

    % Execute
    format('~nExecuting Spark job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% demo_spark_clojure
%  Demonstration using Clojure Spark
demo_spark_clojure :-
    format('~n=== Spark Clojure Demo ===~n', []),

    % Check if Spark and Clojure are available
    (   check_spark_available, check_clojure_available
    ->  true
    ;   format('Spark or Clojure not available~n', []),
        fail
    ),

    % Setup
    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    Partitions = [partition(0, Data)],

    % Initialize Spark backend in Clojure mode
    format('~nInitializing Clojure Spark...~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/spark)),
            register_backend(spark, spark_backend)
        ),
        _,
        true
    ),
    backend_init(spark(
        mode(clojure),
        master('local[2]'),
        executor_memory('512m')
    ), Handle),

    % Execute
    format('~nExecuting Spark job...~n', []),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    % Show results
    format('~nResults:~n', []),
    show_results(Results),

    % Cleanup
    backend_cleanup(Handle),
    format('~n=== Demo Complete ===~n', []).

%% ============================================
%% COMPARISON DEMO
%% ============================================

%% demo_compare_backends
%  Compare performance of different backends
demo_compare_backends :-
    format('~n=== Backend Comparison Demo ===~n', []),

    % Setup
    create_test_script(ScriptPath),
    numlist(1, 1000, Data),
    format('Test data: 1000 items~n', []),

    % Partition data
    partitioner_init(fixed_size(100), [], PartHandle),
    partitioner_partition(PartHandle, Data, Partitions),
    partitioner_cleanup(PartHandle),

    format('~nTesting backends...~n~n', []),

    % Test Bash Fork
    format('--- Bash Fork ---~n', []),
    catch(
        (   use_module(unifyweaver(core/backends/bash_fork)),
            register_backend(bash_fork, bash_fork_backend),
            get_time(T1Start),
            backend_init(bash_fork(workers(4)), BashHandle),
            backend_execute(BashHandle, Partitions, ScriptPath, _BashResults),
            backend_cleanup(BashHandle),
            get_time(T1End),
            T1 is T1End - T1Start,
            format('Bash Fork time: ~3f seconds~n~n', [T1])
        ),
        E1,
        format('Bash Fork error: ~w~n~n', [E1])
    ),

    % Test Dask (if available)
    format('--- Dask Distributed ---~n', []),
    (   check_dask_available
    ->  catch(
            (   use_module(unifyweaver(core/backends/dask_distributed)),
                register_backend(dask_distributed, dask_distributed_backend),
                get_time(T2Start),
                backend_init(dask_distributed(workers(4), scheduler(threads)), DaskHandle),
                backend_execute(DaskHandle, Partitions, ScriptPath, _DaskResults),
                backend_cleanup(DaskHandle),
                get_time(T2End),
                T2 is T2End - T2Start,
                format('Dask time: ~3f seconds~n~n', [T2])
            ),
            E2,
            format('Dask error: ~w~n~n', [E2])
        )
    ;   format('Dask not available~n~n', [])
    ),

    % Test GNU Parallel (if available)
    format('--- GNU Parallel ---~n', []),
    (   check_gnu_parallel_available
    ->  catch(
            (   use_module(unifyweaver(core/backends/gnu_parallel)),
                register_backend(gnu_parallel, gnu_parallel_backend),
                get_time(T3Start),
                backend_init(gnu_parallel(workers(4)), GnuHandle),
                backend_execute(GnuHandle, Partitions, ScriptPath, _GnuResults),
                backend_cleanup(GnuHandle),
                get_time(T3End),
                T3 is T3End - T3Start,
                format('GNU Parallel time: ~3f seconds~n~n', [T3])
            ),
            E3,
            format('GNU Parallel error: ~w~n~n', [E3])
        )
    ;   format('GNU Parallel not available~n~n', [])
    ),

    format('=== Comparison Complete ===~n', []).

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% check_dask_available
%  Check if Python Dask is available
check_dask_available :-
    catch(
        (   process_create(path(python3), ['-c', 'import dask'],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(0))
        ),
        _,
        fail
    ).

%% check_hadoop_available
%  Check if Hadoop is available
check_hadoop_available :-
    (   getenv('HADOOP_HOME', _)
    ->  true
    ;   catch(
            (   process_create(path(which), [hadoop],
                              [stdout(null), stderr(null), process(PID)]),
                process_wait(PID, exit(0))
            ),
            _,
            fail
        )
    ).

%% check_gnu_parallel_available
%  Check if GNU Parallel is available
check_gnu_parallel_available :-
    catch(
        (   process_create(path(which), [parallel],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(0))
        ),
        _,
        fail
    ).

%% check_spark_available
%  Check if Apache Spark is available
check_spark_available :-
    (   getenv('SPARK_HOME', _)
    ->  true
    ;   catch(
            (   process_create(path(which), ['spark-submit'],
                              [stdout(null), stderr(null), process(PID)]),
                process_wait(PID, exit(0))
            ),
            _,
            fail
        )
    ).

%% check_java_available
%  Check if Java is available
check_java_available :-
    catch(
        (   process_create(path(java), ['--version'],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(0))
        ),
        _,
        fail
    ).

%% check_scala_available
%  Check if Scala is available
check_scala_available :-
    catch(
        (   process_create(path(which), [scalac],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(0))
        ),
        _,
        fail
    ).

%% check_kotlin_available
%  Check if Kotlin is available
check_kotlin_available :-
    catch(
        (   process_create(path(which), [kotlinc],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(0))
        ),
        _,
        fail
    ).

%% check_clojure_available
%  Check if Clojure is available
check_clojure_available :-
    catch(
        (   process_create(path(which), [clojure],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(0))
        ),
        _,
        % Try with clj
        catch(
            (   process_create(path(which), [clj],
                              [stdout(null), stderr(null), process(PID2)]),
                process_wait(PID2, exit(0))
            ),
            _,
            fail
        )
    ).

%% show_results(+Results)
%  Display results in a formatted way
show_results([]) :- !.
show_results([result(ID, Output)|Rest]) :-
    format('  Partition ~w: ~w~n', [ID, Output]),
    show_results(Rest).

%% ============================================
%% MAIN DEMO
%% ============================================

%% demo_all
%  Run all available demos
demo_all :-
    setup_backends,

    format('~n========================================~n', []),
    format('Running all available backend demos...~n', []),
    format('========================================~n', []),

    % Always available
    format('~n[1/6] Testing Bash Fork...~n', []),
    catch(demo_bash_fork_simple, E1, format('Error: ~w~n', [E1])),

    % Dask demos
    format('~n[2/6] Testing Dask...~n', []),
    (   check_dask_available
    ->  catch(demo_dask_basic, E2, format('Error: ~w~n', [E2]))
    ;   format('Skipped (Dask not available)~n', [])
    ),

    % Hadoop Streaming demos
    format('~n[3/6] Testing Hadoop Streaming...~n', []),
    (   check_hadoop_available
    ->  catch(demo_hadoop_basic, E3, format('Error: ~w~n', [E3]))
    ;   format('Skipped (Hadoop not available)~n', [])
    ),

    % Spark demos
    format('~n[4/6] Testing Spark...~n', []),
    (   check_spark_available
    ->  catch(demo_spark_pyspark, E4, format('Error: ~w~n', [E4]))
    ;   format('Skipped (Spark not available)~n', [])
    ),

    % Hadoop Native demos
    format('~n[5/6] Testing Hadoop Native (JVM)...~n', []),
    (   check_hadoop_available, check_java_available
    ->  catch(demo_hadoop_native_java, E5, format('Error: ~w~n', [E5]))
    ;   format('Skipped (Hadoop or Java not available)~n', [])
    ),

    % Comparison
    format('~n[6/6] Backend comparison...~n', []),
    catch(demo_compare_backends, E6, format('Error: ~w~n', [E6])),

    format('~n========================================~n', []),
    format('All demos complete!~n', []),
    format('========================================~n', []).

%% demo_bash_fork_simple
%  Simple bash fork demo for comparison
demo_bash_fork_simple :-
    format('~n=== Bash Fork Simple Demo ===~n', []),

    create_test_script(ScriptPath),
    Data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    partitioner_init(fixed_size(3), [], PartHandle),
    partitioner_partition(PartHandle, Data, Partitions),
    partitioner_cleanup(PartHandle),

    catch(
        (   use_module(unifyweaver(core/backends/bash_fork)),
            register_backend(bash_fork, bash_fork_backend)
        ),
        _,
        true
    ),
    backend_init(bash_fork(workers(2)), Handle),
    backend_execute(Handle, Partitions, ScriptPath, Results),

    format('Results:~n', []),
    show_results(Results),

    backend_cleanup(Handle),
    format('=== Demo Complete ===~n', []).

%% ============================================
%% USAGE INFORMATION
%% ============================================

:- format('~n==============================================~n', []).
:- format('Distributed Backends Demo loaded.~n', []).
:- format('~nAvailable demos:~n', []).
:- format('  ?- demo_all.                  % Run all demos~n', []).
:- format('~nDask:~n', []).
:- format('  ?- demo_dask_basic.           % Dask (threads)~n', []).
:- format('  ?- demo_dask_distributed.     % Dask local cluster~n', []).
:- format('~nHadoop Streaming:~n', []).
:- format('  ?- demo_hadoop_basic.         % Hadoop Streaming~n', []).
:- format('  ?- demo_hadoop_wordcount.     % Word count example~n', []).
:- format('~nHadoop Native (JVM):~n', []).
:- format('  ?- demo_hadoop_native_java.   % Hadoop Native (Java)~n', []).
:- format('  ?- demo_hadoop_native_clojure.% Hadoop Native (Clojure)~n', []).
:- format('~nSpark (All JVM Languages):~n', []).
:- format('  ?- demo_spark_pyspark.        % PySpark~n', []).
:- format('  ?- demo_spark_scala.          % Scala Spark~n', []).
:- format('  ?- demo_spark_java.           % Java Spark~n', []).
:- format('  ?- demo_spark_kotlin.         % Kotlin Spark~n', []).
:- format('  ?- demo_spark_clojure.        % Clojure Spark~n', []).
:- format('~nUtilities:~n', []).
:- format('  ?- demo_compare_backends.     % Compare all backends~n', []).
:- format('==============================================~n~n', []).
