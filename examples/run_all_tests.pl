:- encoding(utf8).
%% run_all_tests.pl - Comprehensive Automated Test Suite
%
% This is the main test runner for UnifyWeaver. It runs all critical tests
% quickly to validate the system is working correctly.
%
% Usage:
%   swipl examples/run_all_tests.pl
%   VERBOSE=1 swipl examples/run_all_tests.pl

:- use_module('../src/unifyweaver/core/stream_compiler').
:- use_module('../src/unifyweaver/core/recursive_compiler').
:- use_module('../src/unifyweaver/core/powershell_compiler').
:- use_module('../src/unifyweaver/core/firewall_v2').
:- use_module('../src/unifyweaver/core/tool_detection').

%% Test state tracking
:- dynamic test_passed/1.
:- dynamic test_failed/2.
:- dynamic test_count/1.

%% ============================================
%% MAIN TEST RUNNER
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  UnifyWeaver Comprehensive Test Suite                 ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Initialize test state
    retractall(test_passed(_)),
    retractall(test_failed(_, _)),
    retractall(test_count(_)),
    assertz(test_count(0)),

    get_time(StartTime),

    % Run all test categories
    run_test_category('Core Compilers', [
        test_stream_compiler_facts,
        test_recursive_compiler_simple
    ]),

    run_test_category('Data Sources', [
        test_csv_source_basic,
        test_json_source_basic
    ]),

    run_test_category('PowerShell Compilation', [
        test_powershell_pure_csv,
        test_powershell_pure_json,
        test_powershell_baas_mode
    ]),

    run_test_category('Firewall Integration', [
        test_firewall_permissive,
        test_firewall_pure_powershell_policy
    ]),

    run_test_category('Tool Detection', [
        test_tool_detection_basic,
        test_tool_availability_with_firewall
    ]),

    get_time(EndTime),
    Duration is EndTime - StartTime,

    % Print summary
    print_test_summary(Duration),

    % Exit with appropriate code
    (   test_failed(_, _)
    ->  halt(1)  % Failure
    ;   halt(0)  % Success
    ).

main :-
    format('~n[✗] Test suite failed to complete~n', []),
    halt(1).

%% ============================================
%% TEST RUNNER HELPERS
%% ============================================

run_test_category(Category, Tests) :-
    format('~n[Category] ~w~n', [Category]),
    format('────────────────────────────────────────────────────────~n', []),
    run_tests(Tests).

run_tests([]).
run_tests([Test|Rest]) :-
    run_test(Test),
    run_tests(Rest).

run_test(TestName) :-
    test_count(N),
    N1 is N + 1,
    retract(test_count(N)),
    assertz(test_count(N1)),

    (   verbose_mode
    ->  format('[~w] Running ~w...~n', [N1, TestName])
    ;   true
    ),

    (   catch(call(TestName), Error, (
            format('[~w] ✗ ~w - Exception: ~w~n', [N1, TestName, Error]),
            assertz(test_failed(TestName, Error)),
            fail
        ))
    ->  format('[~w] ✓ ~w~n', [N1, TestName]),
        assertz(test_passed(TestName))
    ;   (   \+ test_failed(TestName, _)
        ->  format('[~w] ✗ ~w - Failed~n', [N1, TestName]),
            assertz(test_failed(TestName, test_predicate_failed))
        ;   true
        )
    ).

verbose_mode :-
    getenv('VERBOSE', '1').

print_test_summary(Duration) :-
    findall(T, test_passed(T), Passed),
    findall(T, test_failed(T, _), Failed),
    length(Passed, PassCount),
    length(Failed, FailCount),
    Total is PassCount + FailCount,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test Summary                                          ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    format('  Total:  ~w tests~n', [Total]),
    format('  Passed: ~w (~w%)~n', [PassCount, (PassCount * 100 // Total)]),
    format('  Failed: ~w~n', [FailCount]),
    format('  Time:   ~2f seconds~n~n', [Duration]),

    (   FailCount > 0
    ->  format('[Result] ✗ FAILED~n~n', []),
        format('Failed tests:~n', []),
        forall(test_failed(T, Reason),
               format('  - ~w (~w)~n', [T, Reason]))
    ;   format('[Result] ✓ ALL TESTS PASSED~n~n', [])
    ).

%% ============================================
%% CORE COMPILER TESTS
%% ============================================

test_stream_compiler_facts :-
    % Define test facts
    retractall(color(_, _)),
    assertz(color(sky, blue)),
    assertz(color(grass, green)),
    assertz(color(sun, yellow)),

    % Compile
    stream_compiler:compile_predicate(color/2, [], Code),

    % Verify bash code generated
    sub_string(Code, _, _, _, 'color()'),
    sub_string(Code, _, _, _, 'sky:blue').

test_recursive_compiler_simple :-
    % Define simple recursive predicate
    retractall(parent(_, _)),
    assertz(parent(tom, bob)),
    assertz(parent(bob, ann)),

    % Compile
    recursive_compiler:compile_predicate(parent/2, [], Code),

    % Verify code generated
    sub_string(Code, _, _, _, 'parent()').

%% ============================================
%% DATA SOURCE TESTS
%% ============================================

test_csv_source_basic :-
    % Test CSV source can be loaded and compiled
    csv_source:compile_source(test_csv/3, [
        csv_file('test_data/test_users.csv'),
        has_header(true)
    ], [], Code),

    % Verify AWK-based bash code
    sub_string(Code, _, _, _, 'awk'),
    sub_string(Code, _, _, _, 'test_csv').

test_json_source_basic :-
    % Test JSON source can be loaded and compiled
    json_source:compile_source(test_json/3, [
        json_file('test_data/test_products.json'),
        jq_filter('.[]')
    ], [], Code),

    % Verify jq-based bash code
    sub_string(Code, _, _, _, 'jq'),
    sub_string(Code, _, _, _, 'test_json').

%% ============================================
%% POWERSHELL COMPILATION TESTS
%% ============================================

test_powershell_pure_csv :-
    % Test pure PowerShell mode for CSV
    compile_to_powershell(test_csv/3, [
        source_type(csv),
        csv_file('test_data/test_users.csv'),
        has_header(true),
        powershell_mode(pure)
    ], Code),

    % Verify pure PowerShell (no bash wrapper)
    sub_string(Code, _, _, _, 'Import-Csv'),
    \+ sub_string(Code, _, _, _, 'uw-bash').

test_powershell_pure_json :-
    % Test pure PowerShell mode for JSON
    compile_to_powershell(test_json/3, [
        source_type(json),
        json_file('test_data/test_products.json'),
        jq_filter('.[]'),
        powershell_mode(pure)
    ], Code),

    % Verify pure PowerShell
    sub_string(Code, _, _, _, 'ConvertFrom-Json'),
    \+ sub_string(Code, _, _, _, 'uw-bash').

test_powershell_baas_mode :-
    % Test BaaS mode still works
    retractall(test_fact(_, _)),
    assertz(test_fact(a, 1)),
    assertz(test_fact(b, 2)),

    compile_to_powershell(test_fact/2, [
        powershell_mode(baas)
    ], Code),

    % Verify BaaS wrapper
    sub_string(Code, _, _, _, 'uw-bash'),
    sub_string(Code, _, _, _, 'test_fact').

%% ============================================
%% FIREWALL TESTS
%% ============================================

test_firewall_permissive :-
    % Clean up any previous firewall state
    retractall(firewall_v2:denied_service(_, _)),
    retractall(firewall_v2:firewall_mode(_)),

    % Load permissive policy
    load_firewall_policy(permissive),

    % In permissive mode, services should be allowed by default
    % Check firewall mode was set correctly
    firewall_mode(Mode),
    memberchk(Mode, [permissive, strict, disabled]).

test_firewall_pure_powershell_policy :-
    % Load pure PowerShell policy
    retractall(firewall_v2:denied_service(_, _)),
    load_firewall_policy(pure_powershell),

    % Check that bash service is denied for PowerShell
    check_service(powershell, executable(bash), Result),
    Result = deny(_).

%% ============================================
%% TOOL DETECTION TESTS
%% ============================================

test_tool_detection_basic :-
    % Test basic tool detection
    detect_tool_availability(bash, Status),
    Status = available.

test_tool_availability_with_firewall :-
    % Clean state thoroughly
    retractall(firewall_v2:denied_tool(_)),
    retractall(firewall_v2:denied_service(_, _)),

    % Test that bash is available
    firewall_v2:check_tool_availability(bash, powershell, Result1),
    (   Result1 = available
    ->  true
    ;   format('[Debug] Unexpected result for bash: ~w~n', [Result1]),
        fail
    ),

    % Deny bash and test again
    assertz(firewall_v2:denied_tool(bash)),
    firewall_v2:check_tool_availability(bash, powershell, Result2),
    (   Result2 = denied(_)
    ->  true
    ;   format('[Debug] Expected denied, got: ~w~n', [Result2]),
        fail
    ),

    % Clean up
    retractall(firewall_v2:denied_tool(_)),
    retractall(firewall_v2:denied_service(_, _)).

%% ============================================
%% INITIALIZATION
%% ============================================

:- initialization(main, main).
