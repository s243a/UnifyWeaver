:- encoding(utf8).
% Test suite for Haskell target
% Usage: swipl -g run_tests -t halt tests/test_haskell_target.pl

:- use_module('../src/unifyweaver/targets/haskell_target').

%% Test data (facts)
test_person(tom, 25).
test_person(bob, 30).

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]).

%% Tests
test_haskell_recursion :-
    Test = 'Haskell: compile_recursion',
    (   haskell_target:compile_recursion_to_haskell(sum/2, [], Code),
        sub_atom(Code, _, _, _, 'BangPatterns'),
        sub_atom(Code, _, _, _, 'sum 0 !acc = acc'),
        sub_atom(Code, _, _, _, 'sum n !acc')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing BangPatterns or pattern matching')
    ).

test_haskell_rules :-
    Test = 'Haskell: compile_rules (ancestor)',
    (   haskell_target:compile_rules_to_haskell(ancestor/2, [base_pred(parent)], Code),
        sub_atom(Code, _, _, _, 'data Entity'),
        sub_atom(Code, _, _, _, 'ancestor :: Entity -> Entity -> Bool'),
        sub_atom(Code, _, _, _, 'parent x y')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Entity type or ancestor function')
    ).

test_haskell_module :-
    Test = 'Haskell: compile_module',
    (   haskell_target:compile_module_to_haskell(
            [pred(sum, 2, tail_recursion), pred(factorial, 1, factorial)],
            [module_name('PrologMath')],
            Code),
        sub_atom(Code, _, _, _, 'module PrologMath'),
        sub_atom(Code, _, _, _, 'sum :: Int -> Int -> Int'),
        sub_atom(Code, _, _, _, 'factorial :: Int -> Int')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing module or functions')
    ).

test_haskell_factorial :-
    Test = 'Haskell: factorial pattern',
    (   haskell_target:compile_module_to_haskell(
            [pred(factorial, 1, factorial)],
            [module_name('Factorial')],
            Code),
        sub_atom(Code, _, _, _, 'factorial 0 = 1'),
        sub_atom(Code, _, _, _, 'factorial 1 = 1'),
        sub_atom(Code, _, _, _, 'n * factorial')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing factorial patterns')
    ).

test_haskell_list_fold :-
    Test = 'Haskell: list_fold pattern',
    (   haskell_target:compile_module_to_haskell(
            [pred(listSum, 2, list_fold)],
            [module_name('ListSum')],
            Code),
        sub_atom(Code, _, _, _, 'listSum :: [Int] -> Int'),
        sub_atom(Code, _, _, _, 'foldr (+) 0')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing list_fold pattern')
    ).

test_haskell_list_tail_recursion :-
    Test = 'Haskell: list_tail_recursion pattern',
    (   haskell_target:compile_module_to_haskell(
            [pred(sumAcc, 3, list_tail_recursion)],
            [module_name('SumAcc')],
            Code),
        sub_atom(Code, _, _, _, 'sumAcc :: [Int] -> Int -> Int'),
        sub_atom(Code, _, _, _, 'sumAcc [] !acc = acc'),
        sub_atom(Code, _, _, _, 'sumAcc (h:t) !acc')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing list_tail_recursion pattern')
    ).

%% New tests for standard interface and Parsec

test_haskell_target_info :-
    Test = 'Haskell: target_info (standard interface)',
    (   haskell_target:target_info(Info),
        Info.name == "Haskell",
        Info.family == functional,
        member(pattern_matching, Info.features),
        Info.parser_support == parsec
    ->  pass(Test)
    ;   fail_test(Test, 'Missing target_info fields')
    ).

test_haskell_compile_predicate :-
    Test = 'Haskell: compile_predicate (standard interface)',
    (   haskell_target:compile_predicate(sum/2, [type(recursion)], Code),
        sub_atom(Code, _, _, _, 'sum :: Int')
    ->  pass(Test)
    ;   fail_test(Test, 'compile_predicate failed')
    ).

test_haskell_dcg_to_parsec :-
    Test = 'Haskell: DCG to Parsec (simple)',
    (   haskell_target:compile_dcg_to_parsec(
            (digit --> [d]),
            [module_name('DigitParser')],
            Code),
        sub_atom(Code, _, _, _, 'import Text.Parsec'),
        sub_atom(Code, _, _, _, 'digit :: Parser String'),
        sub_atom(Code, _, _, _, 'string "d"')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing Parsec imports or parser definition')
    ).

test_haskell_dcg_sequence :-
    Test = 'Haskell: DCG to Parsec (sequence)',
    (   haskell_target:compile_dcg_to_parsec(
            (ab --> [a], [b]),
            [module_name('ABParser')],
            Code),
        sub_atom(Code, _, _, _, '*>')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing sequence operator')
    ).

test_haskell_grammar_to_parsec :-
    Test = 'Haskell: Grammar to Parsec (multiple rules)',
    (   haskell_target:compile_grammar_to_parsec(
            [(expr --> term),
             (term --> digit),
             (digit --> [x])],
            [module_name('ExprParser'), start_symbol(expr)],
            Code),
        sub_atom(Code, _, _, _, 'parseexpr'),
        sub_atom(Code, _, _, _, 'expr :: Parser String'),
        sub_atom(Code, _, _, _, 'term :: Parser String')
    ->  pass(Test)
    ;   fail_test(Test, 'Missing grammar parsers')
    ).

%% Run all tests
run_tests :-
    format('~n========================================~n'),
    format('Haskell Target Test Suite~n'),
    format('========================================~n~n'),
    
    test_haskell_recursion,
    test_haskell_rules,
    test_haskell_module,
    test_haskell_factorial,
    test_haskell_list_fold,
    test_haskell_list_tail_recursion,
    test_haskell_target_info,
    test_haskell_compile_predicate,
    test_haskell_dcg_to_parsec,
    test_haskell_dcg_sequence,
    test_haskell_grammar_to_parsec,
    
    format('~n========================================~n'),
    format('All tests completed~n'),
    format('========================================~n').

:- initialization(run_tests, main).
