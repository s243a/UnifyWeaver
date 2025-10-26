:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_pure_powershell.pl - Test pure PowerShell compilation
% Tests CSV/JSON/HTTP sources in pure PowerShell mode vs BaaS mode

:- use_module('../src/unifyweaver/core/powershell_compiler').

%% ============================================
%% TEST CASES
%% ============================================

test_all :-
    format('~n╔════════════════════════════════════════════════════╗~n', []),
    format('║  Pure PowerShell Compilation Tests                ║~n', []),
    format('╚════════════════════════════════════════════════════╝~n~n', []),

    test_csv_pure,
    test_csv_baas,
    test_csv_auto,
    test_json_pure,
    test_json_baas,
    test_http_pure,
    test_mode_detection,

    format('~n╔════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                               ║~n', []),
    format('╚════════════════════════════════════════════════════╝~n', []).

%% Test 1: CSV in Pure PowerShell mode
test_csv_pure :-
    format('~n[Test 1] CSV source - Pure PowerShell mode~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    compile_to_powershell(test_user/2, [
        source_type(csv),
        csv_file('input/sample_users.csv'),
        has_header(true),
        powershell_mode(pure)
    ], Code),

    % Verify it's pure PowerShell (no bash/uw-bash)
    \+ sub_string(Code, _, _, _, 'uw-bash'),
    \+ sub_string(Code, _, _, _, '$bashScript'),
    sub_string(Code, _, _, _, 'Import-Csv'),

    format('[✓] Generated pure PowerShell code~n', []),
    format('[✓] Uses Import-Csv cmdlet~n', []),
    format('[✓] No bash dependency~n', []),
    !.

%% Test 2: CSV in BaaS mode
test_csv_baas :-
    format('~n[Test 2] CSV source - Bash-as-a-Service mode~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    compile_to_powershell(test_user/2, [
        source_type(csv),
        csv_file('input/sample_users.csv'),
        has_header(true),
        powershell_mode(baas)
    ], Code),

    % Verify it uses bash wrapper
    sub_string(Code, _, _, _, 'uw-bash'),
    sub_string(Code, _, _, _, '$bashScript'),
    sub_string(Code, _, _, _, 'awk'),

    format('[✓] Generated BaaS wrapper~n', []),
    format('[✓] Uses uw-bash~n', []),
    format('[✓] Contains AWK code~n', []),
    !.

%% Test 3: CSV in Auto mode (should choose pure)
test_csv_auto :-
    format('~n[Test 3] CSV source - Auto mode (should choose pure)~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    compile_to_powershell(test_user/2, [
        source_type(csv),
        csv_file('input/sample_users.csv'),
        has_header(true),
        powershell_mode(auto)
    ], Code),

    % Auto mode should choose pure for CSV
    \+ sub_string(Code, _, _, _, 'uw-bash'),
    sub_string(Code, _, _, _, 'Import-Csv'),

    format('[✓] Auto mode selected pure PowerShell~n', []),
    format('[✓] Used native Import-Csv~n', []),
    !.

%% Test 4: JSON in Pure mode
test_json_pure :-
    format('~n[Test 4] JSON source - Pure PowerShell mode~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    compile_to_powershell(test_product/2, [
        source_type(json),
        json_file('input/sample_products.json'),
        jq_filter('.[]'),
        powershell_mode(pure)
    ], Code),

    % Verify pure PowerShell
    \+ sub_string(Code, _, _, _, 'uw-bash'),
    \+ sub_string(Code, _, _, _, 'jq'),
    sub_string(Code, _, _, _, 'ConvertFrom-Json'),

    format('[✓] Generated pure PowerShell code~n', []),
    format('[✓] Uses ConvertFrom-Json cmdlet~n', []),
    format('[✓] No jq dependency~n', []),
    !.

%% Test 5: JSON in BaaS mode
test_json_baas :-
    format('~n[Test 5] JSON source - Bash-as-a-Service mode~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    compile_to_powershell(test_product/2, [
        source_type(json),
        json_file('input/sample_products.json'),
        jq_filter('.[]'),
        powershell_mode(baas)
    ], Code),

    % Verify BaaS wrapper
    sub_string(Code, _, _, _, 'uw-bash'),
    sub_string(Code, _, _, _, 'jq'),

    format('[✓] Generated BaaS wrapper~n', []),
    format('[✓] Uses jq~n', []),
    !.

%% Test 6: HTTP in Pure mode
test_http_pure :-
    format('~n[Test 6] HTTP source - Pure PowerShell mode~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    compile_to_powershell(test_api/2, [
        source_type(http),
        url('https://jsonplaceholder.typicode.com/users'),
        http_method(get),
        powershell_mode(pure)
    ], Code),

    % Verify pure PowerShell
    \+ sub_string(Code, _, _, _, 'uw-bash'),
    \+ sub_string(Code, _, _, _, 'curl'),
    sub_string(Code, _, _, _, 'Invoke-RestMethod'),

    format('[✓] Generated pure PowerShell code~n', []),
    format('[✓] Uses Invoke-RestMethod cmdlet~n', []),
    format('[✓] No curl dependency~n', []),
    !.

%% Test 7: Mode detection logic
test_mode_detection :-
    format('~n[Test 7] Mode detection logic~n', []),
    format('─────────────────────────────────────────────────────~n', []),

    % CSV should be detected as pure-capable
    supports_pure_powershell(test_user/2, [source_type(csv)]),
    format('[✓] CSV detected as pure-capable~n', []),

    % JSON should be detected as pure-capable
    supports_pure_powershell(test_product/2, [source_type(json)]),
    format('[✓] JSON detected as pure-capable~n', []),

    % HTTP should be detected as pure-capable
    supports_pure_powershell(test_api/2, [source_type(http)]),
    format('[✓] HTTP detected as pure-capable~n', []),

    % AWK should NOT be pure-capable
    \+ supports_pure_powershell(test_log/3, [source_type(awk)]),
    format('[✓] AWK correctly not pure-capable~n', []),

    !.

%% ============================================
%% OUTPUT GENERATION TESTS
%% ============================================

test_generate_all :-
    format('~n╔════════════════════════════════════════════════════╗~n', []),
    format('║  Generating Test Scripts                          ║~n', []),
    format('╚════════════════════════════════════════════════════╝~n~n', []),

    generate_csv_pure,
    generate_csv_baas,
    generate_json_pure,
    generate_json_baas,

    format('~n[✓] All test scripts generated~n', []),
    format('    Check test_output/ directory~n', []).

generate_csv_pure :-
    format('[Generating] CSV pure PowerShell -> test_output/csv_pure.ps1~n', []),
    compile_to_powershell(test_user/2, [
        source_type(csv),
        csv_file('input/sample_users.csv'),
        has_header(true),
        powershell_mode(pure),
        output_file('test_output/csv_pure.ps1')
    ], _).

generate_csv_baas :-
    format('[Generating] CSV BaaS -> test_output/csv_baas.ps1~n', []),
    compile_to_powershell(test_user/2, [
        source_type(csv),
        csv_file('input/sample_users.csv'),
        has_header(true),
        powershell_mode(baas),
        output_file('test_output/csv_baas.ps1')
    ], _).

generate_json_pure :-
    format('[Generating] JSON pure PowerShell -> test_output/json_pure.ps1~n', []),
    compile_to_powershell(test_product/2, [
        source_type(json),
        json_file('input/sample_products.json'),
        jq_filter('.[]'),
        powershell_mode(pure),
        output_file('test_output/json_pure.ps1')
    ], _).

generate_json_baas :-
    format('[Generating] JSON BaaS -> test_output/json_baas.ps1~n', []),
    compile_to_powershell(test_product/2, [
        source_type(json),
        json_file('input/sample_products.json'),
        jq_filter('.[]'),
        powershell_mode(baas),
        output_file('test_output/json_baas.ps1')
    ], _).

%% ============================================
%% MAIN ENTRY POINTS
%% ============================================

main :-
    test_all,
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

% For interactive testing
:- initialization(main, main).
