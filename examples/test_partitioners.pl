:- encoding(utf8).
% test_partitioners.pl - Unit tests for partitioning strategies
%
% Tests all partitioning strategies:
% - fixed_size
% - hash_based
% - key_based

% Only import the core partitioner module
% Strategy modules are loaded via the plugin registry
:- use_module('../src/unifyweaver/core/partitioner').

% Ensure strategy modules are loaded (but not imported)
:- ensure_loaded('../src/unifyweaver/core/partitioners/fixed_size').
:- ensure_loaded('../src/unifyweaver/core/partitioners/hash_based').
:- ensure_loaded('../src/unifyweaver/core/partitioners/key_based').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Partitioner Strategy Tests                           ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Register partitioning strategies
    register_partitioner(fixed_size, fixed_size_partitioner),
    register_partitioner(hash_based, hash_based_partitioner),
    register_partitioner(key_based, key_based_partitioner),

    % Run tests
    test_fixed_size_rows,
    test_fixed_size_bytes,
    test_hash_based_partition,
    test_hash_based_streaming,
    test_key_based_grouping,
    test_key_based_compound_terms,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Partitioner Tests Passed ✓                       ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Partitioner tests failed~n', []),
    halt(1).

%% ============================================
%% FIXED SIZE PARTITIONING TESTS
%% ============================================

test_fixed_size_rows :-
    format('~n[Test 1] Fixed-size partitioning by rows~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data
    TestData = [1,2,3,4,5,6,7,8,9,10,11],

    % Initialize partitioner with 4 rows per partition
    partitioner_init(fixed_size(rows(4)), [], Handle),

    % Partition data
    partitioner_partition(Handle, TestData, Partitions),

    % Verify partitions
    (   Partitions = [partition(0, P0), partition(1, P1), partition(2, P2)]
    ->  format('  ✓ Created 3 partitions~n', [])
    ;   format('  ✗ FAIL: Wrong number of partitions: ~w~n', [Partitions]),
        fail
    ),

    % Verify partition sizes
    (   length(P0, 4), length(P1, 4), length(P2, 3)
    ->  format('  ✓ Partition sizes correct: [4, 4, 3]~n', [])
    ;   format('  ✗ FAIL: Wrong partition sizes~n', []),
        fail
    ),

    % Verify partition contents
    (   P0 = [1,2,3,4], P1 = [5,6,7,8], P2 = [9,10,11]
    ->  format('  ✓ Partition contents correct~n', [])
    ;   format('  ✗ FAIL: Wrong partition contents~n', []),
        fail
    ),

    partitioner_cleanup(Handle),
    format('[✓] Test 1 Passed~n', []),
    !.

test_fixed_size_bytes :-
    format('~n[Test 2] Fixed-size partitioning by bytes~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data (atoms of various sizes)
    TestData = [a, bb, ccc, dddd, eeeee, ffffff],

    % Initialize partitioner with 10 bytes per partition
    partitioner_init(fixed_size(bytes(10)), [], Handle),

    % Partition data
    partitioner_partition(Handle, TestData, Partitions),

    % Verify partitions created (should be 2-3 partitions)
    length(Partitions, NumPartitions),
    (   NumPartitions >= 2, NumPartitions =< 3
    ->  format('  ✓ Created ~w partitions (expected 2-3)~n', [NumPartitions])
    ;   format('  ✗ FAIL: Wrong number of partitions: ~w~n', [NumPartitions]),
        fail
    ),

    % Verify all data preserved
    findall(Item, (member(partition(_, Items), Partitions), member(Item, Items)), AllItems),
    sort(TestData, SortedTest),
    sort(AllItems, SortedAll),
    (   SortedTest = SortedAll
    ->  format('  ✓ All data preserved across partitions~n', [])
    ;   format('  ✗ FAIL: Data lost or duplicated~n', []),
        fail
    ),

    partitioner_cleanup(Handle),
    format('[✓] Test 2 Passed~n', []),
    !.

%% ============================================
%% HASH-BASED PARTITIONING TESTS
%% ============================================

test_hash_based_partition :-
    format('~n[Test 3] Hash-based partitioning (batch)~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data with clear keys
    TestData = [
        row(1, alice, 25),
        row(2, bob, 30),
        row(3, charlie, 35),
        row(4, alice, 26),    % Same key as row 1
        row(5, bob, 31),      % Same key as row 2
        row(6, diana, 40)
    ],

    % Initialize partitioner: hash by second column (name), 3 partitions
    partitioner_init(hash_based(key(column(2)), num_partitions(3)), [], Handle),

    % Partition data
    partitioner_partition(Handle, TestData, Partitions),

    % Verify number of partitions (should be <= 3)
    length(Partitions, NumPartitions),
    (   NumPartitions =< 3
    ->  format('  ✓ Created ~w partitions (max 3)~n', [NumPartitions])
    ;   format('  ✗ FAIL: Too many partitions: ~w~n', [NumPartitions]),
        fail
    ),

    % Verify all data preserved
    findall(Item, (member(partition(_, Items), Partitions), member(Item, Items)), AllItems),
    length(TestData, TestLen),
    length(AllItems, AllLen),
    (   TestLen =:= AllLen
    ->  format('  ✓ All data preserved (6 items)~n', [])
    ;   format('  ✗ FAIL: Data lost or duplicated: ~w vs ~w~n', [TestLen, AllLen]),
        fail
    ),

    % Verify deterministic hashing: items with same key go to same partition
    verify_same_key_same_partition(Partitions, alice),
    verify_same_key_same_partition(Partitions, bob),

    partitioner_cleanup(Handle),
    format('[✓] Test 3 Passed~n', []),
    !.

verify_same_key_same_partition(Partitions, Key) :-
    findall(ID, (
        member(partition(ID, Items), Partitions),
        member(row(_, Key, _), Items)
    ), PartitionIDs),
    (   PartitionIDs = []
    ->  true  % Key not found, ok
    ;   sort(PartitionIDs, [_SingleID])
    ->  format('  ✓ All "~w" items in same partition~n', [Key])
    ;   format('  ✗ FAIL: "~w" items split across partitions: ~w~n', [Key, PartitionIDs]),
        fail
    ).

test_hash_based_streaming :-
    format('~n[Test 4] Hash-based partitioning (streaming)~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Initialize partitioner
    partitioner_init(hash_based(key(column(1)), num_partitions(4)), [], Handle),

    % Test streaming assignment
    TestItems = [user(alice), user(bob), user(alice), user(charlie)],

    % Assign each item
    maplist(test_stream_assign(Handle), TestItems, PartitionIDs),

    format('  ✓ Assigned items: ~w~n', [PartitionIDs]),

    % Verify alice items go to same partition
    nth0(0, TestItems, user(alice)),
    nth0(2, TestItems, user(alice)),
    nth0(0, PartitionIDs, AliceID1),
    nth0(2, PartitionIDs, AliceID2),
    (   AliceID1 =:= AliceID2
    ->  format('  ✓ Same key (alice) goes to same partition~n', [])
    ;   format('  ✗ FAIL: Same key goes to different partitions~n', []),
        fail
    ),

    partitioner_cleanup(Handle),
    format('[✓] Test 4 Passed~n', []),
    !.

test_stream_assign(Handle, Item, PartitionID) :-
    partitioner_assign(Handle, Item, PartitionID).

%% ============================================
%% KEY-BASED PARTITIONING TESTS
%% ============================================

test_key_based_grouping :-
    format('~n[Test 5] Key-based partitioning (grouping)~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data with categories
    TestData = [
        item(fruit, apple),
        item(vegetable, carrot),
        item(fruit, banana),
        item(vegetable, broccoli),
        item(fruit, cherry),
        item(meat, chicken)
    ],

    % Initialize partitioner: group by first column (category)
    partitioner_init(key_based(key(column(1))), [], Handle),

    % Partition data
    partitioner_partition(Handle, TestData, Partitions),

    % Verify 3 groups (fruit, vegetable, meat)
    length(Partitions, NumPartitions),
    (   NumPartitions =:= 3
    ->  format('  ✓ Created 3 partitions (one per category)~n', [])
    ;   format('  ✗ FAIL: Wrong number of partitions: ~w~n', [NumPartitions]),
        fail
    ),

    % Verify fruit group has 3 items
    member(partition(_, key(fruit), FruitItems), Partitions),
    length(FruitItems, FruitCount),
    (   FruitCount =:= 3
    ->  format('  ✓ Fruit partition has 3 items~n', [])
    ;   format('  ✗ FAIL: Fruit partition has ~w items~n', [FruitCount]),
        fail
    ),

    % Verify vegetable group has 2 items
    member(partition(_, key(vegetable), VegItems), Partitions),
    length(VegItems, VegCount),
    (   VegCount =:= 2
    ->  format('  ✓ Vegetable partition has 2 items~n', [])
    ;   format('  ✗ FAIL: Vegetable partition has ~w items~n', [VegCount]),
        fail
    ),

    % Verify meat group has 1 item
    member(partition(_, key(meat), MeatItems), Partitions),
    length(MeatItems, MeatCount),
    (   MeatCount =:= 1
    ->  format('  ✓ Meat partition has 1 item~n', [])
    ;   format('  ✗ FAIL: Meat partition has ~w items~n', [MeatCount]),
        fail
    ),

    partitioner_cleanup(Handle),
    format('[✓] Test 5 Passed~n', []),
    !.

test_key_based_compound_terms :-
    format('~n[Test 6] Key-based partitioning (compound terms)~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Create test data with compound terms
    TestData = [
        log(error, "Connection failed"),
        log(info, "Server started"),
        log(error, "Timeout"),
        log(warning, "High memory"),
        log(info, "Request processed")
    ],

    % Initialize partitioner: group by first argument (log level)
    partitioner_init(key_based(key(column(1))), [], Handle),

    % Partition data
    partitioner_partition(Handle, TestData, Partitions),

    % Verify 3 groups (error, info, warning)
    length(Partitions, NumPartitions),
    (   NumPartitions =:= 3
    ->  format('  ✓ Created 3 partitions (error, info, warning)~n', [])
    ;   format('  ✗ FAIL: Wrong number of partitions: ~w~n', [NumPartitions]),
        fail
    ),

    % Verify error partition
    member(partition(_, key(error), ErrorItems), Partitions),
    length(ErrorItems, ErrorCount),
    (   ErrorCount =:= 2
    ->  format('  ✓ Error partition has 2 items~n', [])
    ;   format('  ✗ FAIL: Error partition has ~w items~n', [ErrorCount]),
        fail
    ),

    % Verify all data preserved
    findall(Item, (member(partition(_, key(_), Items), Partitions), member(Item, Items)), AllItems),
    length(TestData, TestLen),
    length(AllItems, AllLen),
    (   TestLen =:= AllLen
    ->  format('  ✓ All data preserved~n', [])
    ;   format('  ✗ FAIL: Data lost or duplicated~n', []),
        fail
    ),

    partitioner_cleanup(Handle),
    format('[✓] Test 6 Passed~n', []),
    !.

:- initialization(main, main).
