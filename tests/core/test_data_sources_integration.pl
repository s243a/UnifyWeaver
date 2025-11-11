:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_data_sources_integration.pl - Integration tests for all data source plugins

:- module(test_data_sources_integration, [
    test_data_sources_integration/0,
    test_csv_to_python_pipeline/0,
    test_http_to_json_pipeline/0,
    test_multi_source_firewall/0,
    test_data_source_firewall_enforcement/0,
    test_data_source_firewall_permissive_mode/0,
    test_real_world_scenario/0
]).

% Load source modules without importing their predicates (to avoid conflicts)
:- use_module('../../src/unifyweaver/sources/csv_source', []).
:- use_module('../../src/unifyweaver/sources/python_source', []).
:- use_module('../../src/unifyweaver/sources/http_source', []).
:- use_module('../../src/unifyweaver/sources/json_source', []).
:- use_module('../../src/unifyweaver/core/firewall').
:- use_module('../../src/unifyweaver/sources').
:- use_module(library(lists)).

%% Main integration test predicate
test_data_sources_integration :-
    format('Testing data source integration...~n', []),
    test_csv_to_python_pipeline,
    test_http_to_json_pipeline,
    test_data_source_firewall_enforcement,
    test_data_source_firewall_permissive_mode,
    test_multi_source_firewall,
    test_real_world_scenario,
    format('✅ All integration tests passed~n', []).

%% Test CSV → Python pipeline integration
test_csv_to_python_pipeline :-
    format('  Testing CSV → Python pipeline...~n', []),
    
    % Create test CSV data
    TestCsvData = 'name,age,city\nalice,25,nyc\nbob,30,sf\ncharlie,35,la\n',
    write_test_file('test_users.csv', TestCsvData),
    
    % Compile CSV source
    catch(
        (   csv_source:compile_source(users/3, [
                csv_file('test_users.csv'),
                has_header(true)
            ], [], CsvCode),
            
            % Verify CSV code generation
            sub_atom(CsvCode, _, _, _, 'users()'),
            sub_atom(CsvCode, _, _, _, 'name, age, city'),
            
            format('    ✅ CSV source compiles correctly~n', [])
        ),
        Error1,
        (   format('    ❌ CSV Error: ~w~n', [Error1]),
            fail
        )
    ),
    
    % Compile Python source to process CSV data
    catch(
        (   python_source:compile_source(process_users/2, [
                python_inline('
import sys
for line in sys.stdin:
    name, age, city = line.strip().split(":")
    if int(age) >= 30:
        print(f"{name}:{city}")
'),
                input_mode(stdin)
            ], [], PythonCode),
            
            % Verify Python code with heredoc pattern
            sub_atom(PythonCode, _, _, _, '/dev/fd/3'),
            sub_atom(PythonCode, _, _, _, 'PYTHON'),
            sub_atom(PythonCode, _, _, _, 'process_users()'),
            
            format('    ✅ Python source compiles with heredoc pattern~n', [])
        ),
        Error2,
        (   format('    ❌ Python Error: ~w~n', [Error2]),
            fail
        )
    ),
    
    % Clean up
    delete_test_file('test_users.csv').

%% Test HTTP → JSON pipeline integration  
test_http_to_json_pipeline :-
    format('  Testing HTTP → JSON pipeline...~n', []),
    
    % Compile HTTP source (mock API endpoint)
    catch(
        (   http_source:compile_source(api_users/1, [
                url('https://jsonplaceholder.typicode.com/users'),
                method(get),
                cache_duration(300)
            ], [], HttpCode),
            
            % Verify HTTP code with caching
            sub_atom(HttpCode, _, _, _, 'api_users()'),
            sub_atom(HttpCode, _, _, _, 'cache_file'),
            sub_atom(HttpCode, _, _, _, 'cache_duration'),
            sub_atom(HttpCode, _, _, _, 'curl'),
            
            format('    ✅ HTTP source compiles with caching~n', [])
        ),
        Error1,
        (   format('    ❌ HTTP Error: ~w~n', [Error1]),
            fail
        )
    ),
    
    % Compile JSON source to parse HTTP response
    catch(
        (   json_source:compile_source(parse_api_users/2, [
                jq_filter('.[] | {name: .name, email: .email}'),
                json_stdin(true),
                output_format(tsv)
            ], [], JsonCode),
            
            % Verify JSON code with jq
            sub_atom(JsonCode, _, _, _, 'parse_api_users()'),
            sub_atom(JsonCode, _, _, _, 'jq'),
            sub_atom(JsonCode, _, _, _, '.[] | {name: .name, email: .email}'),
            
            format('    ✅ JSON source compiles with jq integration~n', [])
        ),
        Error2,
        (   format('    ❌ JSON Error: ~w~n', [Error2]),
            fail
        )
    ).

%% Test permissive/warn firewall modes
test_data_source_firewall_permissive_mode :-
    format('  Testing data source firewall permissive mode...~n', []),
    setup_call_cleanup(
        (   firewall:set_firewall_mode(permissive),
            assertz(firewall:rule_firewall(permissive_api/2, [network_hosts(['*.example.com'])]))
        ),
        (
            retractall(dynamic_source_compiler:dynamic_source_def(permissive_api/2, _, _)),
            (   sources:source(http, permissive_api, [url('https://malicious.com')])
            ->  format('    ✅ Permissive mode allowed source (warning expected)~n', [])
            ;   format('    ❌ Permissive mode should not block source~n'),
                fail
            )
        ),
        (
            firewall:set_firewall_mode(strict),
            retractall(firewall:rule_firewall(permissive_api/2, _)),
            retractall(dynamic_source_compiler:dynamic_source_def(permissive_api/2, _, _))
        )
    ),
    format('  Testing data source firewall warn mode...~n', []),
    setup_call_cleanup(
        (   firewall:set_firewall_mode(warn),
            assertz(firewall:rule_firewall(warn_api/2, [network_hosts(['*.example.com'])]))
        ),
        (
            retractall(dynamic_source_compiler:dynamic_source_def(warn_api/2, _, _)),
            (   sources:source(http, warn_api, [url('https://malicious.com')])
            ->  format('    ✅ Warn mode allowed source with warning~n', [])
            ;   format('    ❌ Warn mode should not block source~n'),
                fail
            )
        ),
        (
            firewall:set_firewall_mode(strict),
            retractall(firewall:rule_firewall(warn_api/2, _)),
            retractall(dynamic_source_compiler:dynamic_source_def(warn_api/2, _, _))
        )
    ).

%% Test multi-source firewall integration
test_multi_source_firewall :-
    format('  Testing multi-source firewall...~n', []),
    
    % Test firewall allows required services
    TestFirewall = [
        services([awk, python3, curl, jq]),
        network_access(allowed),
        network_hosts(['*.typicode.com', 'api.github.com']),
        python_modules([sys, json, sqlite3]),
        file_read_patterns(['data/*', 'test_*.csv']),
        cache_dirs(['/tmp/*'])
    ],
    
    % Test CSV with firewall
    catch(
        (   validate_against_firewall(bash, [csv_file('test_data.csv')], TestFirewall),
            format('    ✅ CSV firewall validation works~n', [])
        ),
        Error1,
        (   format('    ❌ CSV Firewall Error: ~w~n', [Error1]),
            fail
        )
    ),
    
    % Test Python with firewall  
    catch(
        (   validate_against_firewall(bash, [
                python_inline('import sys\nprint("hello")')
            ], TestFirewall),
            format('    ✅ Python firewall validation works~n', [])
        ),
        Error2,
        (   format('    ❌ Python Firewall Error: ~w~n', [Error2]),
            fail
        )
    ),
    
    % Test HTTP with firewall
    catch(
        (   validate_against_firewall(bash, [
                url('https://jsonplaceholder.typicode.com/users')
                % Removed cache_file test - cache validation handled separately
            ], TestFirewall),
            format('    ✅ HTTP firewall validation works~n', [])
        ),
        Error3,
        (   format('    ❌ HTTP Firewall Error: ~w~n', [Error3]),
            fail
        )
    ).

%% Test firewall enforcement during source registration
test_data_source_firewall_enforcement :-
    format('  Testing data source firewall enforcement...~n', []),
    setup_call_cleanup(
        (   assertz(firewall:rule_firewall(blocked_api/2, [network_hosts(['*.typicode.com'])])),
            assertz(firewall:rule_firewall(allowed_api/2, [network_hosts(['*.typicode.com', 'jsonplaceholder.typicode.com'])]))
        ),
        (
            retractall(dynamic_source_compiler:dynamic_source_def(blocked_api/2, _, _)),
            retractall(dynamic_source_compiler:dynamic_source_def(allowed_api/2, _, _)),
            (   \+ sources:source(http, blocked_api, [url('https://malicious.com')])
            ->  format('    ✅ Blocked host rejected at registration~n', [])
            ;   format('    ❌ Blocked host should not register~n'),
                fail
            ),
            (   sources:source(http, allowed_api, [url('https://jsonplaceholder.typicode.com/posts')])
            ->  format('    ✅ Allowed host registered successfully~n', [])
            ;   format('    ❌ Allowed host should register~n'),
                fail
            )
        ),
        (
            retractall(firewall:rule_firewall(blocked_api/2, _)),
            retractall(firewall:rule_firewall(allowed_api/2, _)),
            retractall(dynamic_source_compiler:dynamic_source_def(blocked_api/2, _, _)),
            retractall(dynamic_source_compiler:dynamic_source_def(allowed_api/2, _, _))
        )
    ).

%% Test real-world scenario: GitHub API → SQLite ETL
test_real_world_scenario :-
    format('  Testing real-world ETL scenario...~n', []),
    
    % 1. HTTP Source: Fetch GitHub API data
    catch(
        (   http_source:compile_source(github_repos/1, [
                url('https://api.github.com/users/octocat/repos'),
                headers(['User-Agent: UnifyWeaver/0.0.2']),
                cache_duration(3600)
            ], [], HttpCode),
            
            sub_atom(HttpCode, _, _, _, 'github_repos()'),
            sub_atom(HttpCode, _, _, _, 'User-Agent: UnifyWeaver/0.0.2'),
            
            format('    ✅ Step 1: HTTP source configured~n', [])
        ),
        Error1,
        (   format('    ❌ HTTP Setup Error: ~w~n', [Error1]),
            fail
        )
    ),
    
    % 2. JSON Source: Parse repository data
    catch(
        (   json_source:compile_source(extract_repo_info/3, [
                jq_filter('.[] | [.name, .description, .stargazers_count] | @tsv'),
                json_stdin(true),
                raw_output(true)
            ], [], JsonCode),
            
            sub_atom(JsonCode, _, _, _, 'extract_repo_info()'),
            sub_atom(JsonCode, _, _, _, '@tsv'),
            
            format('    ✅ Step 2: JSON parser configured~n', [])
        ),
        Error2,
        (   format('    ❌ JSON Setup Error: ~w~n', [Error2]),
            fail
        )
    ),
    
    % 3. Python Source: Store in SQLite database
    catch(
        (   python_source:compile_source(store_repos/3, [
                sqlite_query('INSERT INTO repos (name, description, stars) VALUES (?, ?, ?)'),
                database('github_repos.db')
            ], [], SqliteCode),
            
            sub_atom(SqliteCode, _, _, _, 'store_repos()'),
            sub_atom(SqliteCode, _, _, _, 'sqlite3'),
            sub_atom(SqliteCode, _, _, _, 'github_repos.db'),
            
            format('    ✅ Step 3: SQLite storage configured~n', [])
        ),
        Error3,
        (   format('    ❌ SQLite Setup Error: ~w~n', [Error3]),
            fail
        )
    ),
    
    format('    ✅ Real-world ETL pipeline: HTTP → JSON → SQLite ready~n', []).

%% Helper predicates for test file management
write_test_file(File, Data) :-
    open(File, write, Stream),
    write(Stream, Data),
    close(Stream).

delete_test_file(File) :-
    (   exists_file(File)
    ->  delete_file(File)
    ;   true
    ).
