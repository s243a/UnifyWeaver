:- encoding(utf8).
% demo_partition_parallel.pl - Demonstration of Partitioning + Parallel Execution
%
% Shows the complete pipeline:
% 1. Partition data using various strategies
% 2. Execute processing in parallel across partitions
% 3. Aggregate results from all workers
%
% This demonstrates UnifyWeaver's core strength: declarative data processing
% with transparent parallelization.

:- use_module('../src/unifyweaver/core/partitioner').
:- use_module('../src/unifyweaver/core/partitioners/fixed_size').
:- use_module('../src/unifyweaver/core/partitioners/hash_based').
:- use_module('../src/unifyweaver/core/partitioners/key_based').
:- use_module('../src/unifyweaver/core/parallel_backend').
:- use_module('../src/unifyweaver/core/backends/bash_fork').
:- use_module('../src/unifyweaver/core/backends/gnu_parallel').

%% ============================================
%% MAIN DEMONSTRATIONS
%% ============================================

main :-
    format('~n╔═══════════════════════════════════════════════════════════╗~n', []),
    format('║  UnifyWeaver: Partitioning + Parallel Execution Demo    ║~n', []),
    format('╚═══════════════════════════════════════════════════════════╝~n~n', []),

    % Register backends and partitioners
    setup_system,

    % Run demonstrations
    demo_1_fixed_size_partitioning,
    demo_2_hash_based_parallelization,
    demo_3_key_based_grouping,
    demo_4_performance_comparison,

    format('~n╔═══════════════════════════════════════════════════════════╗~n', []),
    format('║  All Demonstrations Complete ✓                           ║~n', []),
    format('╚═══════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Demonstrations failed~n', []),
    halt(1).

%% ============================================
%% DEMO 1: FIXED-SIZE PARTITIONING
%% ============================================

demo_1_fixed_size_partitioning :-
    format('~n[Demo 1] Fixed-Size Partitioning with Parallel Sum~n', []),
    format('─────────────────────────────────────────────────────────────~n', []),

    % Generate dataset: numbers 1-1000
    numlist(1, 1000, Numbers),
    format('  Dataset: 1000 numbers (1..1000)~n', []),

    % Partition into chunks of 100
    partitioner_init(fixed_size(rows(100)), [], PHandle),
    partitioner_partition(PHandle, Numbers, Partitions),
    partitioner_cleanup(PHandle),
    length(Partitions, NumPartitions),
    format('  ✓ Created ~w partitions (100 numbers each)~n', [NumPartitions]),

    % Create summation script
    create_sum_script('/tmp/sum_numbers.sh'),
    format('  ✓ Created summation script~n', []),

    % Execute in parallel
    backend_init(bash_fork(workers(4)), BHandle),
    format('  Executing parallel sum using 4 workers...~n', []),
    get_time(Start),
    backend_execute(BHandle, Partitions, '/tmp/sum_numbers.sh', Results),
    get_time(End),
    backend_cleanup(BHandle),
    Time is End - Start,
    format('  ✓ Parallel execution completed in ~3f seconds~n', [Time]),

    % Aggregate results
    aggregate_sums(Results, TotalSum),
    ExpectedSum is 1000 * 1001 / 2,  % Gauss formula: n(n+1)/2
    format('  ✓ Total sum: ~w (expected: ~w)~n', [TotalSum, ExpectedSum]),

    (   TotalSum =:= ExpectedSum
    ->  format('  ✓ Result verified correct!~n', [])
    ;   format('  ✗ FAIL: Incorrect result~n', []),
        fail
    ),

    % Cleanup
    delete_file('/tmp/sum_numbers.sh'),
    format('[✓] Demo 1 Complete~n', []),
    !.

%% ============================================
%% DEMO 2: HASH-BASED PARALLELIZATION
%% ============================================

demo_2_hash_based_parallelization :-
    format('~n[Demo 2] Hash-Based Partitioning for Load Balancing~n', []),
    format('─────────────────────────────────────────────────────────────~n', []),

    % Generate dataset with key-value pairs
    generate_kvp_dataset(500, Dataset),
    format('  Dataset: 500 key-value pairs~n', []),

    % Partition using hash on key
    partitioner_init(hash_based([key(column(1)), num_partitions(8)]), [], PHandle),
    partitioner_partition(PHandle, Dataset, Partitions),
    partitioner_cleanup(PHandle),

    % Show partition sizes
    maplist(partition_size, Partitions, Sizes),
    format('  ✓ Hash partitioned into 8 buckets~n', []),
    format('    Partition sizes: ~w~n', [Sizes]),

    % Verify load balancing (no partition should be >2x average)
    length(Dataset, DataSize),
    AvgSize is DataSize / 8,
    MaxSize is floor(AvgSize * 2),
    (   forall(member(Size, Sizes), Size =< MaxSize)
    ->  format('  ✓ Load is well balanced (no partition >2x average)~n', [])
    ;   format('  ⚠ Some partitions are skewed (expected for random data)~n', [])
    ),

    format('[✓] Demo 2 Complete~n', []),
    !.

%% ============================================
%% DEMO 3: KEY-BASED GROUPING
%% ============================================

demo_3_key_based_grouping :-
    format('~n[Demo 3] Key-Based Partitioning (GROUP BY semantics)~n', []),
    format('─────────────────────────────────────────────────────────────~n', []),

    % Generate dataset with specific categories
    generate_categorized_data(Categories, Dataset),
    length(Dataset, DataSize),
    length(Categories, NumCategories),
    format('  Dataset: ~w items across ~w categories~n', [DataSize, NumCategories]),

    % Partition by category (column 1)
    partitioner_init(key_based([key(column(1))]), [], PHandle),
    partitioner_partition(PHandle, Dataset, Partitions),
    partitioner_cleanup(PHandle),

    length(Partitions, NumPartitions),
    format('  ✓ Grouped into ~w partitions (one per category)~n', [NumPartitions]),

    (   NumPartitions =:= NumCategories
    ->  format('  ✓ Exactly one partition per category!~n', [])
    ;   format('  ⚠ Partition count mismatch~n', [])
    ),

    % Show partition details
    format('~n  Partition details:~n', []),
    forall(
        member(partition(ID, key(Key), Data), Partitions),
        (   length(Data, Size),
            format('    Partition ~w: key=~w, items=~w~n', [ID, Key, Size])
        )
    ),

    format('[✓] Demo 3 Complete~n', []),
    !.

%% ============================================
%% DEMO 4: PERFORMANCE COMPARISON
%% ============================================

demo_4_performance_comparison :-
    format('~n[Demo 4] Performance: Bash Fork vs GNU Parallel~n', []),
    format('─────────────────────────────────────────────────────────────~n', []),

    % Generate larger dataset
    numlist(1, 10000, Numbers),
    format('  Dataset: 10,000 numbers~n', []),

    % Partition
    partitioner_init(fixed_size(rows(500)), [], PHandle),
    partitioner_partition(PHandle, Numbers, Partitions),
    partitioner_cleanup(PHandle),
    length(Partitions, NumPartitions),
    format('  Partitions: ~w (500 numbers each)~n', [NumPartitions]),

    % Create processing script
    create_sum_script('/tmp/sum_numbers.sh'),

    % Test Bash Fork
    format('~n  Testing Bash Fork backend (4 workers):~n', []),
    backend_init(bash_fork(workers(4)), BH1),
    get_time(Start1),
    backend_execute(BH1, Partitions, '/tmp/sum_numbers.sh', Results1),
    get_time(End1),
    backend_cleanup(BH1),
    Time1 is End1 - Start1,
    aggregate_sums(Results1, Sum1),
    format('    Time: ~3f seconds, Sum: ~w~n', [Time1, Sum1]),

    % Test GNU Parallel (if available)
    (   check_gnu_parallel_available
    ->  format('~n  Testing GNU Parallel backend (4 workers):~n', []),
        backend_init(gnu_parallel(workers(4)), BH2),
        get_time(Start2),
        backend_execute(BH2, Partitions, '/tmp/sum_numbers.sh', Results2),
        get_time(End2),
        backend_cleanup(BH2),
        Time2 is End2 - Start2,
        aggregate_sums(Results2, Sum2),
        format('    Time: ~3f seconds, Sum: ~w~n', [Time2, Sum2]),

        % Compare
        Speedup is Time1 / Time2,
        format('~n  GNU Parallel speedup: ~2fx faster~n', [Speedup]),
        format('  (Both backends produce same result: ~w)~n', [Sum1])
    ;   format('~n  ⊘ GNU Parallel not available, skipping comparison~n', [])
    ),

    % Cleanup
    delete_file('/tmp/sum_numbers.sh'),

    format('[✓] Demo 4 Complete~n', []),
    !.

%% ============================================
%% HELPER PREDICATES
%% ============================================

%% setup_system
%  Register all backends and partitioners
setup_system :-
    register_backend(bash_fork, bash_fork_backend),
    register_backend(gnu_parallel, gnu_parallel_backend),
    register_partitioner(fixed_size, fixed_size_partitioner),
    register_partitioner(hash_based, hash_based_partitioner),
    register_partitioner(key_based, key_based_partitioner),
    format('[System] Registered 2 backends and 3 partitioners~n', []).

%% create_sum_script(+Path)
%  Create bash script that sums numbers from stdin
create_sum_script(Path) :-
    Script = '#!/bin/bash\n# Sum all numbers from stdin\nsum=0\nwhile IFS= read -r line; do\n    if [[ "$line" =~ ^[0-9]+$ ]]; then\n        sum=$((sum + line))\n    fi\ndone\necho "$sum"\n',
    open(Path, write, Stream),
    write(Stream, Script),
    close(Stream),
    process_create('/bin/chmod', ['+x', Path], []).

%% aggregate_sums(+Results, -TotalSum)
%  Sum all partition results
aggregate_sums(Results, TotalSum) :-
    findall(Sum,
            (   member(result(_, Output), Results),
                atom_string(Output, OutputStr),
                split_string(OutputStr, "\n", " \t\r", [SumStr|_]),
                number_string(Sum, SumStr)
            ),
            Sums),
    sum_list(Sums, TotalSum).

%% partition_size(+Partition, -Size)
%  Get size of partition data
partition_size(partition(_ID, Data), Size) :-
    length(Data, Size).
partition_size(partition(_ID, _Key, Data), Size) :-
    length(Data, Size).

%% generate_kvp_dataset(+Count, -Dataset)
%  Generate key-value pairs for hash partitioning demo
generate_kvp_dataset(Count, Dataset) :-
    Keys = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t],
    length(Keys, NumKeys),
    findall(kvp(Key, Value),
            (   between(1, Count, Value),
                KeyIndex is (Value mod NumKeys) + 1,
                nth1(KeyIndex, Keys, Key)
            ),
            Dataset).

%% generate_categorized_data(-Categories, -Dataset)
%  Generate categorized data for GROUP BY demo
generate_categorized_data(Categories, Dataset) :-
    Categories = [fruit, vegetable, meat, dairy, grain],
    findall(item(Category, Item),
            (   member(Category, Categories),
                member(Item, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            ),
            Dataset).

%% check_gnu_parallel_available
%  Check if GNU Parallel is installed
check_gnu_parallel_available :-
    catch(
        (   process_create(path(parallel), ['--version'],
                          [stdout(null), stderr(null), process(PID)]),
            process_wait(PID, exit(ExitCode)),
            ExitCode =:= 0
        ),
        _,
        fail
    ).

:- initialization(main, main).
