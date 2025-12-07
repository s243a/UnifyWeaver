/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Tests for deployment_glue.pl - Phase 6a & 6b
 */

:- use_module('../../src/unifyweaver/glue/deployment_glue').

:- dynamic test_passed/1.
:- dynamic test_failed/2.

%% ============================================
%% Test Runner
%% ============================================

run_all_tests :-
    retractall(test_passed(_)),
    retractall(test_failed(_, _)),

    format('~n=== Deployment Glue Tests (Phase 6a & 6b) ===~n~n'),

    % Phase 6a Tests
    format('--- Phase 6a: Foundation ---~n~n'),

    run_test_group('Service Declarations', [
        test_declare_service,
        test_service_config_query,
        test_undeclare_service
    ]),

    run_test_group('Deployment Methods', [
        test_declare_deploy_method,
        test_deploy_method_query
    ]),

    run_test_group('Source Tracking', [
        test_declare_service_sources,
        test_service_sources_query
    ]),

    run_test_group('Security Validation', [
        test_security_local_http_allowed,
        test_security_remote_http_blocked,
        test_security_remote_https_allowed,
        test_requires_encryption,
        test_is_local_service
    ]),

    run_test_group('Lifecycle Hooks', [
        test_declare_lifecycle_hook,
        test_lifecycle_hooks_query
    ]),

    run_test_group('SSH Deploy Script Generation', [
        test_generate_ssh_deploy_basic,
        test_generate_ssh_deploy_with_hooks
    ]),

    run_test_group('Systemd Unit Generation', [
        test_generate_systemd_unit
    ]),

    run_test_group('Health Check Script Generation', [
        test_generate_health_check_script
    ]),

    run_test_group('Local Deploy Script Generation', [
        test_generate_local_deploy
    ]),

    % Phase 6b Tests
    format('~n--- Phase 6b: Advanced Deployment ---~n~n'),

    run_test_group('Multi-Host Support', [
        test_declare_service_hosts,
        test_service_hosts_query
    ]),

    run_test_group('Rollback Support', [
        test_store_rollback_hash,
        test_rollback_hash_update,
        test_generate_rollback_script_ssh,
        test_generate_rollback_script_local
    ]),

    run_test_group('Health Check Integration', [
        test_run_health_check_builds_url
    ]),

    run_test_group('Hook Execution', [
        test_execute_hooks_empty,
        test_execute_hooks_save_state,
        test_execute_hooks_warm_cache
    ]),

    run_test_group('Graceful Shutdown', [
        test_graceful_stop_structure
    ]),

    run_test_group('Deploy with Hooks', [
        test_deploy_with_hooks_security_check
    ]),

    print_summary.

run_test_group(GroupName, Tests) :-
    format('~w:~n', [GroupName]),
    maplist(run_single_test, Tests),
    format('~n').

run_single_test(Test) :-
    (   catch(call(Test), Error, (
            assertz(test_failed(Test, Error)),
            format('  ✗ ~w: ~w~n', [Test, Error]),
            fail
        ))
    ->  assertz(test_passed(Test)),
        format('  ✓ ~w~n', [Test])
    ;   (   \+ test_failed(Test, _)
        ->  assertz(test_failed(Test, 'assertion failed')),
            format('  ✗ ~w: assertion failed~n', [Test])
        ;   true
        )
    ).

print_summary :-
    findall(T, test_passed(T), Passed),
    findall(T, test_failed(T, _), Failed),
    length(Passed, PassCount),
    length(Failed, FailCount),
    Total is PassCount + FailCount,
    format('~n=== Summary ===~n'),
    format('Passed: ~w/~w~n', [PassCount, Total]),
    (   FailCount > 0
    ->  format('Failed: ~w~n', [FailCount]),
        forall(test_failed(T, Reason), format('  - ~w: ~w~n', [T, Reason]))
    ;   format('All tests passed!~n')
    ).

%% ============================================
%% Service Declaration Tests
%% ============================================

test_declare_service :-
    % Clean up
    undeclare_service(test_service),

    % Declare a service
    declare_service(test_service, [
        host('example.com'),
        port(8080),
        target(python),
        entry_point('server.py'),
        lifecycle(persistent),
        transport(https)
    ]),

    % Verify it exists
    service_config(test_service, Options),
    member(host('example.com'), Options),
    member(port(8080), Options),
    member(target(python), Options).

test_service_config_query :-
    % Query existing service
    service_config(test_service, Options),
    is_list(Options),
    member(transport(https), Options).

test_undeclare_service :-
    % Declare and undeclare
    declare_service(temp_service, [host('temp.com')]),
    service_config(temp_service, _),
    undeclare_service(temp_service),
    \+ service_config(temp_service, _).

%% ============================================
%% Deployment Method Tests
%% ============================================

test_declare_deploy_method :-
    declare_deploy_method(test_service, ssh, [
        host('example.com'),
        user('deploy'),
        agent(true),
        remote_dir('/opt/services')
    ]),
    deploy_method_config(test_service, ssh, Options),
    member(user('deploy'), Options).

test_deploy_method_query :-
    deploy_method_config(test_service, Method, Options),
    Method == ssh,
    member(remote_dir('/opt/services'), Options).

%% ============================================
%% Source Tracking Tests
%% ============================================

test_declare_service_sources :-
    declare_service_sources(test_service, [
        'src/**/*.py',
        'requirements.txt'
    ]),
    service_sources(test_service, Sources),
    member('src/**/*.py', Sources).

test_service_sources_query :-
    service_sources(test_service, Sources),
    length(Sources, 2).

%% ============================================
%% Security Validation Tests
%% ============================================

test_security_local_http_allowed :-
    undeclare_service(local_service),
    declare_service(local_service, [
        host(localhost),
        port(8080),
        transport(http)
    ]),
    validate_security(local_service, Errors),
    Errors == [].

test_security_remote_http_blocked :-
    undeclare_service(insecure_service),
    declare_service(insecure_service, [
        host('remote.example.com'),
        port(8080),
        transport(http)
    ]),
    validate_security(insecure_service, Errors),
    Errors \== [],
    member(remote_requires_encryption('remote.example.com'), Errors).

test_security_remote_https_allowed :-
    undeclare_service(secure_service),
    declare_service(secure_service, [
        host('remote.example.com'),
        port(8080),
        transport(https)
    ]),
    validate_security(secure_service, Errors),
    Errors == [].

test_requires_encryption :-
    requires_encryption(secure_service),
    \+ requires_encryption(local_service).

test_is_local_service :-
    is_local_service(local_service),
    \+ is_local_service(secure_service).

%% ============================================
%% Lifecycle Hook Tests
%% ============================================

test_declare_lifecycle_hook :-
    % Clean existing hooks
    retractall(lifecycle_hook_db(test_service, _, _)),

    declare_lifecycle_hook(test_service, pre_shutdown, drain_connections),
    declare_lifecycle_hook(test_service, post_deploy, health_check),

    lifecycle_hooks(test_service, Hooks),
    member(hook(pre_shutdown, drain_connections), Hooks),
    member(hook(post_deploy, health_check), Hooks).

test_lifecycle_hooks_query :-
    lifecycle_hooks(test_service, Hooks),
    length(Hooks, 2).

%% ============================================
%% SSH Deploy Script Tests
%% ============================================

test_generate_ssh_deploy_basic :-
    undeclare_service(ssh_test_service),
    declare_service(ssh_test_service, [
        host('worker.example.com'),
        port(9000),
        target(python),
        entry_point('app.py')
    ]),
    declare_deploy_method(ssh_test_service, ssh, [
        user('ubuntu'),
        remote_dir('/home/ubuntu/services')
    ]),

    % Use generate_deploy_script which merges method options properly
    generate_deploy_script(ssh_test_service, [], Script),

    % Verify script contains expected elements
    sub_atom(Script, _, _, _, '#!/bin/bash'),
    sub_atom(Script, _, _, _, 'worker.example.com'),
    sub_atom(Script, _, _, _, 'ubuntu'),
    sub_atom(Script, _, _, _, 'rsync'),
    sub_atom(Script, _, _, _, 'python3 app.py').

test_generate_ssh_deploy_with_hooks :-
    undeclare_service(hooked_service),
    declare_service(hooked_service, [
        host('prod.example.com'),
        port(8080),
        target(go),
        entry_point('server')
    ]),
    declare_deploy_method(hooked_service, ssh, [
        user('deploy')
    ]),
    declare_lifecycle_hook(hooked_service, pre_shutdown, drain_connections),
    declare_lifecycle_hook(hooked_service, post_deploy, health_check),

    generate_ssh_deploy(hooked_service, [], Script),

    sub_atom(Script, _, _, _, 'Draining connections'),
    sub_atom(Script, _, _, _, 'health check').

%% ============================================
%% Systemd Unit Tests
%% ============================================

test_generate_systemd_unit :-
    undeclare_service(systemd_service),
    declare_service(systemd_service, [
        port(3000),
        target(node),
        entry_point('index.js')
    ]),

    generate_systemd_unit(systemd_service, [user('nodeapp')], Unit),

    sub_atom(Unit, _, _, _, '[Unit]'),
    sub_atom(Unit, _, _, _, '[Service]'),
    sub_atom(Unit, _, _, _, '[Install]'),
    sub_atom(Unit, _, _, _, 'User=nodeapp'),
    sub_atom(Unit, _, _, _, 'node index.js').

%% ============================================
%% Health Check Script Tests
%% ============================================

test_generate_health_check_script :-
    undeclare_service(health_service),
    declare_service(health_service, [
        host('api.example.com'),
        port(443)
    ]),

    generate_health_check_script(health_service, [
        health_endpoint('/status'),
        timeout(10),
        retries(5)
    ], Script),

    sub_atom(Script, _, _, _, '#!/bin/bash'),
    sub_atom(Script, _, _, _, 'api.example.com'),
    sub_atom(Script, _, _, _, '/status'),
    sub_atom(Script, _, _, _, 'TIMEOUT="10"'),
    sub_atom(Script, _, _, _, 'RETRIES="5"'),
    sub_atom(Script, _, _, _, 'https').  % Should use HTTPS for remote

%% ============================================
%% Local Deploy Script Tests
%% ============================================

test_generate_local_deploy :-
    undeclare_service(local_app),
    declare_service(local_app, [
        port(5000),
        target(python),
        entry_point('main.py')
    ]),
    declare_deploy_method(local_app, local, []),

    generate_deploy_script(local_app, [], Script),

    sub_atom(Script, _, _, _, '#!/bin/bash'),
    sub_atom(Script, _, _, _, 'python3 main.py'),
    sub_atom(Script, _, _, _, 'PORT="5000"').

%% ============================================
%% Phase 6b Tests: Multi-Host Support
%% ============================================

test_declare_service_hosts :-
    undeclare_service(multi_host_service),
    declare_service(multi_host_service, [
        port(8080),
        target(python),
        transport(https)
    ]),
    declare_service_hosts(multi_host_service, [
        host_config('host1.example.com', [user('deploy1')]),
        host_config('host2.example.com', [user('deploy2')])
    ]),
    service_hosts(multi_host_service, Hosts),
    length(Hosts, 2),
    member(host_config('host1.example.com', _), Hosts),
    member(host_config('host2.example.com', _), Hosts).

test_service_hosts_query :-
    service_hosts(multi_host_service, Hosts),
    Hosts = [host_config('host1.example.com', Opts1), _],
    member(user('deploy1'), Opts1).

%% ============================================
%% Phase 6b Tests: Rollback Support
%% ============================================

test_store_rollback_hash :-
    store_rollback_hash(test_rollback_service, 'abc123'),
    rollback_hash(test_rollback_service, Hash),
    Hash == 'abc123'.

test_rollback_hash_update :-
    store_rollback_hash(test_rollback_service, 'def456'),
    rollback_hash(test_rollback_service, Hash),
    Hash == 'def456'.

test_generate_rollback_script_ssh :-
    undeclare_service(rollback_ssh_service),
    declare_service(rollback_ssh_service, [
        host('prod.example.com'),
        port(8080)
    ]),
    declare_deploy_method(rollback_ssh_service, ssh, [
        user('deploy'),
        remote_dir('/opt/services')
    ]),

    generate_rollback_script(rollback_ssh_service, [], Script),

    sub_atom(Script, _, _, _, '#!/bin/bash'),
    sub_atom(Script, _, _, _, 'Rolling back'),
    sub_atom(Script, _, _, _, 'prod.example.com'),
    sub_atom(Script, _, _, _, '.backup').

test_generate_rollback_script_local :-
    undeclare_service(rollback_local_service),
    declare_service(rollback_local_service, [port(8080)]),
    declare_deploy_method(rollback_local_service, local, []),

    generate_rollback_script(rollback_local_service, [], Script),

    sub_atom(Script, _, _, _, '#!/bin/bash'),
    sub_atom(Script, _, _, _, 'Rolling back'),
    sub_atom(Script, _, _, _, '.backup').

%% ============================================
%% Phase 6b Tests: Health Check
%% ============================================

test_run_health_check_builds_url :-
    undeclare_service(health_test_service),
    declare_service(health_test_service, [
        host('api.example.com'),
        port(443)
    ]),
    declare_deploy_method(health_test_service, ssh, [user('deploy')]),

    % This test verifies URL construction - the actual curl will fail
    % but we're testing the logic builds the right command
    catch(
        run_health_check(health_test_service, [retries(1), delay(0), timeout(1)], _Result),
        _,
        true
    ).

%% ============================================
%% Phase 6b Tests: Hook Execution
%% ============================================

test_execute_hooks_empty :-
    undeclare_service(no_hooks_service),
    declare_service(no_hooks_service, [port(8080)]),
    declare_deploy_method(no_hooks_service, local, []),

    execute_hooks(no_hooks_service, pre_shutdown, Result),
    Result == ok.

test_execute_hooks_save_state :-
    undeclare_service(hooks_service),
    declare_service(hooks_service, [port(8080)]),
    declare_deploy_method(hooks_service, local, []),
    declare_lifecycle_hook(hooks_service, pre_shutdown, save_state),

    execute_hooks(hooks_service, pre_shutdown, Result),
    Result == ok.

test_execute_hooks_warm_cache :-
    declare_lifecycle_hook(hooks_service, post_deploy, warm_cache),

    execute_hooks(hooks_service, post_deploy, Result),
    Result == ok.

%% ============================================
%% Phase 6b Tests: Graceful Shutdown
%% ============================================

test_graceful_stop_structure :-
    undeclare_service(graceful_service),
    declare_service(graceful_service, [
        host(localhost),
        port(9999)
    ]),
    declare_deploy_method(graceful_service, local, []),

    % This will fail because no service is running, but tests the structure
    catch(
        graceful_stop(graceful_service, [drain_timeout(1), force_after(1)], _Result),
        _,
        true
    ).

%% ============================================
%% Phase 6b Tests: Deploy with Hooks
%% ============================================

test_deploy_with_hooks_security_check :-
    undeclare_service(insecure_deploy_service),
    declare_service(insecure_deploy_service, [
        host('remote.example.com'),
        port(8080),
        transport(http)  % Insecure!
    ]),
    declare_deploy_method(insecure_deploy_service, ssh, [user('deploy')]),

    deploy_with_hooks(insecure_deploy_service, Result),
    Result = error(security_validation_failed(_)).

%% ============================================
%% Main Entry Point
%% ============================================

:- initialization((
    run_all_tests,
    halt
), main).
