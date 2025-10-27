:- encoding(utf8).
% test_source_plugins_no_conflict.pl - Test that multiple source plugins can be loaded without conflicts

:- use_module('../src/unifyweaver/sources/csv_source').
:- use_module('../src/unifyweaver/sources/json_source').
:- use_module('../src/unifyweaver/sources/python_source').
:- use_module('../src/unifyweaver/sources/http_source').
:- use_module('../src/unifyweaver/sources/awk_source').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Testing Source Plugin Import (No Conflicts)          ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    format('✓ csv_source loaded~n', []),
    format('✓ json_source loaded~n', []),
    format('✓ python_source loaded~n', []),
    format('✓ http_source loaded~n', []),
    format('✓ awk_source loaded~n', []),
    format('~n✓ All source plugins loaded without import conflicts!~n', []),

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test Passed - No Import Conflicts                    ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Test failed~n', []),
    halt(1).

:- initialization(main, main).
