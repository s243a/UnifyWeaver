% test_left_join_proto.pl - Test LEFT JOIN pattern detection

:- use_module('src/unifyweaver/targets/sql_left_join_proto').

% Test pattern detection
test_pattern_detection :-
    write('Testing LEFT JOIN pattern detection...'), nl, nl,

    % Test 1: Basic LEFT JOIN pattern
    Pattern1 = (customers(CId, Name, _), (orders(_, CId, Product, _) ; Product = null)),
    (   detect_left_join_pattern(Pattern1, Left1, Right1, Fall1)
    ->  format('✓ Test 1: Detected pattern~n', []),
        format('  Left goals: ~w~n', [Left1]),
        format('  Right goal: ~w~n', [Right1]),
        format('  Fallback: ~w~n~n', [Fall1])
    ;   write('✗ Test 1: Failed to detect pattern'), nl
    ),

    % Test 2: Multi-column fallback
    Pattern2 = (customers(CId, Name, _),
                (orders(_, CId, Product, Amount) ; Product = null, Amount = null)),
    (   detect_left_join_pattern(Pattern2, Left2, Right2, Fall2)
    ->  format('✓ Test 2: Multi-column pattern~n', []),
        extract_null_bindings(Fall2, NullVars),
        format('  NULL bindings: ~w~n~n', [NullVars])
    ;   write('✗ Test 2: Failed to detect pattern'), nl
    ),

    % Test 3: Not a LEFT JOIN pattern (no null binding)
    Pattern3 = (customers(CId, Name, _), (orders(_, CId, Product, _) ; Product = 'N/A')),
    (   detect_left_join_pattern(Pattern3, _,_, _)
    ->  write('✗ Test 3: False positive - detected non-LEFT JOIN pattern'), nl
    ;   write('✓ Test 3: Correctly rejected non-LEFT JOIN pattern'), nl
    ),

    nl,
    write('Pattern detection tests complete!'), nl.

:- initialization(test_pattern_detection, main).
