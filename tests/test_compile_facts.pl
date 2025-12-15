:- encoding(utf8).
% Test suite for compile_facts_to_go/3 and compile_facts_to_rust/3 APIs
% Usage: swipl -g run_tests -t halt tests/test_compile_facts.pl

:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/go_target').
:- use_module('../src/unifyweaver/targets/rust_target').
:- use_module('../src/unifyweaver/targets/python_target').
:- use_module('../src/unifyweaver/targets/sql_target').
:- use_module('../src/unifyweaver/targets/powershell_target').

%% Test facts
test_person(alice, 25).
test_person(bob, 30).
test_person(charlie, 35).

test_color(red).
test_color(green).
test_color(blue).

test_triple(a, b, c).
test_triple(x, y, z).

%% Helper predicates
ensure_output_dir :-
    make_directory_path('output/compile_facts_test').

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% ============================================
%% Go Target Tests
%% ============================================

test_go_facts_binary :-
    Test = 'Go: compile_facts_to_go binary predicate',
    (   go_target:compile_facts_to_go(test_person, 2, Code),
        sub_atom(Code, _, _, _, 'type TEST_PERSON struct'),
        sub_atom(Code, _, _, _, 'GetAllTEST_PERSON'),
        sub_atom(Code, _, _, _, 'alice'),
        sub_atom(Code, _, _, _, 'bob')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Go struct or functions')
    ).

test_go_facts_unary :-
    Test = 'Go: compile_facts_to_go unary predicate',
    (   go_target:compile_facts_to_go(test_color, 1, Code),
        sub_atom(Code, _, _, _, 'type TEST_COLOR struct'),
        sub_atom(Code, _, _, _, 'StreamTEST_COLOR'),
        sub_atom(Code, _, _, _, 'red')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Go struct for unary')
    ).

test_go_facts_ternary :-
    Test = 'Go: compile_facts_to_go ternary predicate',
    (   go_target:compile_facts_to_go(test_triple, 3, Code),
        sub_atom(Code, _, _, _, 'Arg3 string'),
        sub_atom(Code, _, _, _, 'ContainsTEST_TRIPLE')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Arg3 or Contains for ternary')
    ).

test_go_facts_write :-
    Test = 'Go: Write compile_facts output to file',
    ensure_output_dir,
    (   go_target:compile_facts_to_go(test_person, 2, Code),
        open('output/compile_facts_test/test_person.go', write, S),
        write(S, Code),
        close(S),
        exists_file('output/compile_facts_test/test_person.go')
    ->  pass(Test)
    ;   fail_test(Test, 'Failed to write Go file')
    ).

%% ============================================
%% Rust Target Tests
%% ============================================

test_rust_facts_binary :-
    Test = 'Rust: compile_facts_to_rust binary predicate',
    (   rust_target:compile_facts_to_rust(test_person, 2, Code),
        sub_atom(Code, _, _, _, 'struct TEST_PERSON'),
        sub_atom(Code, _, _, _, 'get_all_test_person'),
        sub_atom(Code, _, _, _, 'alice')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Rust struct or functions')
    ).

test_rust_facts_derives :-
    Test = 'Rust: Struct has correct derives',
    (   rust_target:compile_facts_to_rust(test_color, 1, Code),
        sub_atom(Code, _, _, _, '#[derive(Debug, Clone, PartialEq, Eq)]')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing derive attributes')
    ).

test_rust_facts_contains :-
    Test = 'Rust: contains function generated',
    (   rust_target:compile_facts_to_rust(test_triple, 3, Code),
        sub_atom(Code, _, _, _, 'contains_test_triple')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing contains function')
    ).

test_rust_facts_write :-
    Test = 'Rust: Write compile_facts output to file',
    ensure_output_dir,
    (   rust_target:compile_facts_to_rust(test_person, 2, Code),
        open('output/compile_facts_test/test_person.rs', write, S),
        write(S, Code),
        close(S),
        exists_file('output/compile_facts_test/test_person.rs')
    ->  pass(Test)
    ;   fail_test(Test, 'Failed to write Rust file')
    ).

%% ============================================
%% Python Target Tests
%% ============================================

test_python_facts_binary :-
    Test = 'Python: compile_facts_to_python binary predicate',
    (   python_target:compile_facts_to_python(test_person, 2, Code),
        sub_atom(Code, _, _, _, 'class TEST_PERSON'),
        sub_atom(Code, _, _, _, 'get_all'),
        sub_atom(Code, _, _, _, 'alice')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected Python class or methods')
    ).

test_python_facts_stream :-
    Test = 'Python: stream method generated',
    (   python_target:compile_facts_to_python(test_color, 1, Code),
        sub_atom(Code, _, _, _, 'def stream')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing stream method')
    ).

%% ============================================
%% SQL Target Tests
%% ============================================

test_sql_facts_binary :-
    Test = 'SQL: compile_facts_to_sql binary predicate',
    (   sql_target:compile_facts_to_sql(test_person, 2, Code),
        sub_atom(Code, _, _, _, 'CREATE TABLE'),
        sub_atom(Code, _, _, _, 'INSERT INTO'),
        sub_atom(Code, _, _, _, 'alice')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected SQL statements')
    ).

test_sql_facts_select :-
    Test = 'SQL: SELECT statement generated',
    (   sql_target:compile_facts_to_sql(test_color, 1, Code),
        sub_atom(Code, _, _, _, 'SELECT * FROM test_color')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing SELECT statement')
    ).

%% ============================================
%% PowerShell Target Tests
%% ============================================

test_powershell_facts_binary :-
    Test = 'PowerShell: compile_facts_to_powershell binary predicate',
    (   powershell_target:compile_facts_to_powershell(test_person, 2, Code),
        sub_atom(Code, _, _, _, 'class TEST_PERSON'),
        sub_atom(Code, _, _, _, 'GetAll'),
        sub_atom(Code, _, _, _, 'alice')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing expected PowerShell class or methods')
    ).

test_powershell_facts_stream :-
    Test = 'PowerShell: Stream method generated',
    (   powershell_target:compile_facts_to_powershell(test_color, 1, Code),
        sub_atom(Code, _, _, _, 'Stream()')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Stream method')
    ).

%% ============================================
%% Run All Tests
%% ============================================

run_tests :-
    format('~n========================================~n'),
    format('UnifyWeaver compile_facts/3 Test Suite~n'),
    format('========================================~n~n'),
    
    format('--- Go Target Tests ---~n'),
    test_go_facts_binary,
    test_go_facts_unary,
    test_go_facts_ternary,
    test_go_facts_write,
    
    format('~n--- Rust Target Tests ---~n'),
    test_rust_facts_binary,
    test_rust_facts_derives,
    test_rust_facts_contains,
    test_rust_facts_write,
    
    format('~n--- Python Target Tests ---~n'),
    test_python_facts_binary,
    test_python_facts_stream,
    
    format('~n--- SQL Target Tests ---~n'),
    test_sql_facts_binary,
    test_sql_facts_select,
    
    format('~n--- PowerShell Target Tests ---~n'),
    test_powershell_facts_binary,
    test_powershell_facts_stream,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

%% Auto-run when loaded with swipl -g run_tests
:- initialization(run_tests, main).
