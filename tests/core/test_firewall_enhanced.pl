:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_firewall_enhanced.pl - Tests for enhanced firewall system

:- module(test_firewall_enhanced, [
    test_firewall_enhanced/0,
    test_service_validation/0,
    test_network_validation/0,
    test_python_import_validation/0,
    test_file_access_validation/0,
    test_cache_validation/0
]).

:- use_module('../../src/unifyweaver/core/firewall').
:- use_module(library(lists)).

%% Main test predicate
test_firewall_enhanced :-
    format('Testing enhanced firewall system...~n', []),
    test_service_validation,
    test_network_validation,
    test_python_import_validation,
    test_file_access_validation,
    test_cache_validation,
    format('✅ All enhanced firewall tests passed~n', []).

%% Test service validation
test_service_validation :-
    format('  Testing service validation...~n', []),
    
    % Test allowed service
    catch(
        (   validate_service(python3, [services([python3, curl, jq])]),
            format('    ✅ Allowed service validation works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Test denied service (should fail validation)
    % Note: Current firewall prints message but doesn't throw exception
    % See context/firewall_behavior_design.md for future enhancement
    (   \+ validate_service(dangerous_tool, [services([python3, curl]), denied([dangerous_tool])])
    ->  format('    ✅ Denied service properly blocked~n', [])
    ;   format('    ❌ Denied service validation failed~n', []),
        fail
    ),
    
    % Test service not in allowlist (should fail silently in this implementation)
    catch(
        (   \+ validate_service(unlisted_service, [services([python3, curl])]),
            format('    ✅ Unlisted service properly blocked~n', [])
        ),
        Error2,
        (   format('    ❌ Error: ~w~n', [Error2]),
            fail
        )
    ).

%% Test network access validation
test_network_validation :-
    format('  Testing network validation...~n', []),
    
    % Test allowed network access
    catch(
        (   validate_network_access('https://api.github.com/users', [
                network_access(allowed),
                network_hosts(['*.github.com', 'api.example.com'])
            ]),
            format('    ✅ Allowed network access works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Test denied network access (should fail)
    (   \+ validate_network_access('https://malicious.com', [network_access(denied)])
    ->  format('    ✅ Network access properly denied~n', [])
    ;   format('    ❌ Network access should have been denied~n', []),
        fail
    ).

%% Test Python import validation
test_python_import_validation :-
    format('  Testing Python import validation...~n', []),
    
    % Test allowed imports
    TestCode1 = 'import sys\nimport json\nfrom sqlite3 import connect',
    catch(
        (   validate_python_imports(TestCode1, [python_modules([sys, json, sqlite3])]),
            format('    ✅ Allowed Python imports work~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Test blocked import (should fail)  
    TestCode2 = 'import os\nimport sys',
    (   \+ validate_python_imports(TestCode2, [python_modules([sys, json])])
    ->  format('    ✅ Blocked Python import properly denied~n', [])
    ;   format('    ❌ Blocked import should have failed~n', []),
        fail
    ).

%% Test file access validation
test_file_access_validation :-
    format('  Testing file access validation...~n', []),
    
    % Test allowed file read
    catch(
        (   validate_file_access('data/users.csv', read, [
                file_read_patterns(['data/*', 'config/*.json'])
            ]),
            format('    ✅ Allowed file read works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Test blocked file access (should fail)
    (   \+ validate_file_access('/etc/passwd', read, [
                file_read_patterns(['data/*', 'config/*'])
            ])
    ->  format('    ✅ Blocked file access properly denied~n', [])
    ;   format('    ❌ Blocked file access should have failed~n', []),
        fail
    ).

%% Test cache directory validation
test_cache_validation :-
    format('  Testing cache validation...~n', []),
    
    % Test allowed cache directory
    catch(
        (   validate_cache_directory('/tmp/cache/api_data', [
                cache_dirs(['/tmp/*', '/var/cache/unifyweaver/*'])
            ]),
            format('    ✅ Allowed cache directory works~n', [])
        ),
        Error,
        (   format('    ❌ Error: ~w~n', [Error]),
            fail
        )
    ),
    
    % Test blocked cache directory (should fail)
    (   \+ validate_cache_directory('/root/secret_cache', [
                cache_dirs(['/tmp/*', '/var/cache/*'])
            ])
    ->  format('    ✅ Blocked cache access properly denied~n', [])
    ;   format('    ❌ Blocked cache access should have failed~n', []),
        fail
    ).
