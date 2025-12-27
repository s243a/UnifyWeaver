% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% rpyc_demo.pl - RPyC Network-Based Python Integration Demo
%
% This example demonstrates using RPyC for network-based RPC
% communication between Prolog and Python, integrated with
% UnifyWeaver's glue system.
%
% Prerequisites:
%   1. Start the RPyC server first:
%      python rpyc_server.py
%
%   2. Then run this demo:
%      ?- [rpyc_demo].
%      ?- run_demo.

:- use_module(library(janus)).
:- use_module('../../src/unifyweaver/glue/rpyc_glue').

%% ============================================
%% Demo 1: Code Generation (No Server Needed)
%% ============================================

demo_code_generation :-
    format('~n=== Demo 1: Code Generation ===~n'),
    format('(Does not require running server)~n~n'),

    % Generate client wrapper
    format('1.1 Generating client wrapper:~n'),
    generate_rpyc_client([
        exposed(sqrt_wrapper/2, [module(math), function(sqrt)])
    ], [host('localhost'), security(unsecured)], ClientCode),
    format('~w~n', [ClientCode]),

    % Generate service
    format('~n1.2 Generating RPyC service:~n'),
    generate_rpyc_service([
        exposed(transform_data/2, [input(list), output(list)]),
        exposed(predict/2, [input(dict), output(float)])
    ], [service_name('MLService'), imports([numpy])], ServiceCode),
    format('~w~n', [ServiceCode]).

%% ============================================
%% Demo 2: Connection (Requires Server)
%% ============================================

demo_connection :-
    format('~n=== Demo 2: Connection Test ===~n'),

    format('2.1 Connecting to localhost:18812 (unsecured)...~n'),
    catch(
        (   rpyc_connect('localhost', [
                security(unsecured),
                acknowledge_risk(true),
                remote_port(18812)
            ], Proxy),
            format('    Connected successfully!~n'),

            % Get modules access
            format('2.2 Testing module access:~n'),
            rpyc_import(Proxy, sys, Sys),
            py_call(Sys:version, PyVersion),
            format('    Remote Python version: ~w~n', [PyVersion]),

            % Disconnect
            format('2.3 Disconnecting...~n'),
            rpyc_disconnect(Proxy),
            format('    Disconnected.~n')
        ),
        Error,
        (   format('~n    Connection failed: ~w~n', [Error]),
            format('~n    Make sure the RPyC server is running:~n'),
            format('      python rpyc_server.py~n')
        )
    ).

%% ============================================
%% Demo 3: Remote Computation
%% ============================================

demo_remote_computation :-
    format('~n=== Demo 3: Remote Computation ===~n'),

    catch(
        (   rpyc_connect('localhost', [
                security(unsecured),
                acknowledge_risk(true)
            ], Proxy),

            % Math operations
            format('3.1 Remote math operations:~n'),
            rpyc_import(Proxy, math, Math),
            py_call(Math:sqrt(16), Sqrt16),
            format('    math.sqrt(16) = ~w~n', [Sqrt16]),

            py_call(Math:factorial(10), Fact10),
            format('    math.factorial(10) = ~w~n', [Fact10]),

            % Remote code execution
            format('~n3.2 Remote code execution:~n'),
            rpyc_exec(Proxy, "
result = sum(range(1, 101))
squares = [x**2 for x in range(1, 11)]
", NS),
            py_call(NS:get('result'), SumResult),
            format('    sum(1..100) = ~w~n', [SumResult]),

            rpyc_disconnect(Proxy)
        ),
        Error,
        (   format('    Error: ~w~n', [Error]),
            format('    Start server with: python rpyc_server.py~n')
        )
    ).

%% ============================================
%% Demo 4: Proxy Layers
%% ============================================

demo_proxy_layers :-
    format('~n=== Demo 4: Proxy Layers ===~n'),

    catch(
        (   rpyc_connect('localhost', [
                security(unsecured),
                acknowledge_risk(true)
            ], Proxy),

            % Layer 1: Root
            format('4.1 Layer 1 (root) - Direct method access:~n'),
            rpyc_root(Proxy, Root),
            format('    Root proxy obtained~n'),

            % Layer 2: Wrapped Root
            format('4.2 Layer 2 (wrapped_root) - Safe attribute access:~n'),
            rpyc_wrapped_root(Proxy, WrappedRoot),
            format('    WrappedRoot proxy obtained~n'),

            % Layer 3: Auto Root
            format('4.3 Layer 3 (auto_root) - Automatic wrapping:~n'),
            rpyc_auto_root(Proxy, AutoRoot),
            format('    AutoRoot proxy obtained~n'),

            % Layer 4: Smart Root
            format('4.4 Layer 4 (smart_root) - Local-class-aware:~n'),
            rpyc_smart_root(Proxy, SmartRoot),
            format('    SmartRoot proxy obtained~n'),

            rpyc_disconnect(Proxy)
        ),
        Error,
        format('    Skipped (server not available): ~w~n', [Error])
    ).

%% ============================================
%% Demo 5: Transport Comparison
%% ============================================

demo_transport_comparison :-
    format('~n=== Demo 5: Transport Comparison ===~n'),
    format('~n'),
    format('Transport     | Location   | Objects      | Use Case~n'),
    format('--------------|------------|--------------|------------------------~n'),
    format('pipe          | Same host  | Serialized   | Process isolation~n'),
    format('http          | Network    | Serialized   | REST APIs~n'),
    format('janus         | In-process | Live (share) | NumPy, ML, tight integ~n'),
    format('rpyc          | Network    | Live (proxy) | Remote compute, distrib~n'),
    format('~n'),
    format('RPyC fills the gap of "network + live objects".~n'),
    format('Use RPyC when you need remote machine access with~n'),
    format('the convenience of live object proxies.~n').

%% ============================================
%% Main Demo Runner
%% ============================================

run_demo :-
    format('~n========================================~n'),
    format('RPyC Network-Based Python Integration Demo~n'),
    format('========================================~n'),

    % Check Janus availability
    (   current_module(janus)
    ->  format('Janus available: yes~n'),
        demo_code_generation,
        demo_connection,
        demo_remote_computation,
        demo_proxy_layers,
        demo_transport_comparison
    ;   format('Janus not available.~n'),
        format('Only code generation demos will run.~n'),
        demo_code_generation,
        demo_transport_comparison
    ),

    format('~n========================================~n'),
    format('Demo complete!~n'),
    format('========================================~n').

%% ============================================
%% Server Setup Helper
%% ============================================

%% generate_server
%  Generate a server script file in the current directory.
generate_server :-
    generate_rpyc_server([
        output_file('rpyc_server.py'),
        service_name('DemoService')
    ], Script),
    format('Generated: rpyc_server.py~n'),
    format('Run it with: python rpyc_server.py~n').

% Auto-run message on load
:- initialization(format('~nRPyC demo loaded.~nRun ?- run_demo. to start.~nRun ?- generate_server. to create a server script.~n')).
