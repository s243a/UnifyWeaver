/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Integration tests for network_glue module
 */

:- use_module('../../../src/unifyweaver/glue/network_glue').

%% ============================================
%% Test Helpers
%% ============================================

run_tests :-
    format('~n=== Network Glue Integration Tests ===~n~n'),
    run_all_tests,
    format('~nAll tests passed!~n').

run_all_tests :-
    test_service_registry,
    test_go_http_server,
    test_python_http_server,
    test_rust_http_server,
    test_go_http_client,
    test_python_http_client,
    test_bash_http_client,
    test_socket_server,
    test_socket_client,
    test_network_pipeline.

assert_contains(String, Substring, TestName) :-
    (   sub_atom(String, _, _, _, Substring)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED: "~w" not found~n', [TestName, Substring]),
        fail
    ).

assert_true(Goal, TestName) :-
    (   call(Goal)
    ->  format('  ✓ ~w~n', [TestName])
    ;   format('  ✗ ~w FAILED~n', [TestName]),
        fail
    ).

%% ============================================
%% Test: Service Registry
%% ============================================

test_service_registry :-
    format('Test: Service registry~n'),

    % Register a service
    register_service(ml_api, 'http://ml-service:8080', [timeout(60), format(json)]),
    assert_true(service(ml_api, 'http://ml-service:8080'), 'Service registered'),
    assert_true(service_options(ml_api, [timeout(60), format(json)]), 'Service options stored'),

    % Endpoint URL construction
    endpoint_url(ml_api, '/predict', URL),
    assert_true(URL == 'http://ml-service:8080/predict', 'Endpoint URL constructed'),

    % Multiple services
    register_service(db_api, 'http://db-service:5000', []),
    assert_true(service(db_api, 'http://db-service:5000'), 'Second service registered'),

    % Unregister
    unregister_service(ml_api),
    assert_true(\+ service(ml_api, _), 'Service unregistered'),

    % Cleanup
    unregister_service(db_api),

    format('~n').

%% ============================================
%% Test: Go HTTP Server
%% ============================================

test_go_http_server :-
    format('Test: Go HTTP server generation~n'),

    generate_go_http_server(
        [
            endpoint('/api/process', process, []),
            endpoint('/api/analyze', analyze, [])
        ],
        [port(8080), cors(true)],
        GoServer
    ),

    assert_contains(GoServer, 'package main', 'Go has package main'),
    assert_contains(GoServer, 'net/http', 'Go imports net/http'),
    assert_contains(GoServer, 'encoding/json', 'Go imports json'),
    assert_contains(GoServer, 'type Request struct', 'Go has Request type'),
    assert_contains(GoServer, 'type Response struct', 'Go has Response type'),
    assert_contains(GoServer, 'corsMiddleware', 'Go has CORS middleware'),
    assert_contains(GoServer, '_api_processHandler', 'Go has process handler'),
    assert_contains(GoServer, '_api_analyzeHandler', 'Go has analyze handler'),
    assert_contains(GoServer, 'http.HandleFunc', 'Go registers handlers'),
    assert_contains(GoServer, '"8080"', 'Go uses port 8080'),

    format('~n').

%% ============================================
%% Test: Python HTTP Server
%% ============================================

test_python_http_server :-
    format('Test: Python HTTP server generation~n'),

    generate_python_http_server(
        [
            endpoint('/api/transform', transform, [methods(['POST', 'GET'])])
        ],
        [port(5000), cors(true)],
        PyServer
    ),

    assert_contains(PyServer, '#!/usr/bin/env python3', 'Python has shebang'),
    assert_contains(PyServer, 'from flask import Flask', 'Python imports Flask'),
    assert_contains(PyServer, 'from flask_cors import CORS', 'Python imports CORS'),
    assert_contains(PyServer, 'app = Flask', 'Python creates app'),
    assert_contains(PyServer, '@app.route("/api/transform"', 'Python has route decorator'),
    assert_contains(PyServer, 'methods=["POST", "GET"]', 'Python has methods'),
    assert_contains(PyServer, 'request.get_json()', 'Python reads JSON'),
    assert_contains(PyServer, 'jsonify({"success"', 'Python returns JSON'),
    assert_contains(PyServer, '5000', 'Python uses port 5000'),

    format('~n').

%% ============================================
%% Test: Rust HTTP Server
%% ============================================

test_rust_http_server :-
    format('Test: Rust HTTP server generation~n'),

    generate_rust_http_server(
        [endpoint('/api/compute', compute, [])],
        [port(9000)],
        RustServer
    ),

    assert_contains(RustServer, 'use actix_web', 'Rust uses actix_web'),
    assert_contains(RustServer, 'use actix_cors::Cors', 'Rust uses actix_cors'),
    assert_contains(RustServer, 'use serde::{Deserialize, Serialize}', 'Rust uses serde'),
    assert_contains(RustServer, 'struct Request', 'Rust has Request struct'),
    assert_contains(RustServer, 'struct Response', 'Rust has Response struct'),
    assert_contains(RustServer, '_api_compute_handler', 'Rust has handler function'),
    assert_contains(RustServer, 'HttpServer::new', 'Rust creates server'),
    assert_contains(RustServer, 'web::post()', 'Rust uses POST method'),
    assert_contains(RustServer, '"9000"', 'Rust uses port 9000'),

    format('~n').

%% ============================================
%% Test: Go HTTP Client
%% ============================================

test_go_http_client :-
    format('Test: Go HTTP client generation~n'),

    generate_go_http_client(
        [
            service_def(ml, 'http://ml:8080', ['/predict', '/classify'])
        ],
        [timeout(60)],
        GoClient
    ),

    assert_contains(GoClient, 'package main', 'Go has package main'),
    assert_contains(GoClient, 'net/http', 'Go imports http'),
    assert_contains(GoClient, 'Timeout: 60 * time.Second', 'Go has timeout'),
    assert_contains(GoClient, 'func callService', 'Go has callService'),
    assert_contains(GoClient, 'client.Post', 'Go uses POST'),
    assert_contains(GoClient, 'func ml__predict', 'Go has predict function'),
    assert_contains(GoClient, 'func ml__classify', 'Go has classify function'),
    assert_contains(GoClient, 'http://ml:8080/predict', 'Go has full URL'),

    format('~n').

%% ============================================
%% Test: Python HTTP Client
%% ============================================

test_python_http_client :-
    format('Test: Python HTTP client generation~n'),

    generate_python_http_client(
        [
            service_def(api, 'http://api:3000', ['/users', '/orders'])
        ],
        [timeout(30)],
        PyClient
    ),

    assert_contains(PyClient, '#!/usr/bin/env python3', 'Python has shebang'),
    assert_contains(PyClient, 'import requests', 'Python imports requests'),
    assert_contains(PyClient, 'TIMEOUT = 30', 'Python has timeout'),
    assert_contains(PyClient, 'class ServiceError', 'Python has error class'),
    assert_contains(PyClient, 'def call_service', 'Python has call_service'),
    assert_contains(PyClient, 'requests.post', 'Python uses POST'),
    assert_contains(PyClient, 'def api__users', 'Python has users function'),
    assert_contains(PyClient, 'def api__orders', 'Python has orders function'),
    assert_contains(PyClient, 'http://api:3000/users', 'Python has full URL'),

    format('~n').

%% ============================================
%% Test: Bash HTTP Client
%% ============================================

test_bash_http_client :-
    format('Test: Bash HTTP client generation~n'),

    generate_bash_http_client(
        [
            service_def(svc, 'http://localhost:8000', ['/health', '/data'])
        ],
        [timeout(15)],
        BashClient
    ),

    assert_contains(BashClient, '#!/bin/bash', 'Bash has shebang'),
    assert_contains(BashClient, 'TIMEOUT=15', 'Bash has timeout'),
    assert_contains(BashClient, 'call_service()', 'Bash has call_service'),
    assert_contains(BashClient, 'curl -s -X POST', 'Bash uses curl'),
    assert_contains(BashClient, '--max-time "$TIMEOUT"', 'Bash uses timeout'),
    assert_contains(BashClient, 'jq -r ".data"', 'Bash uses jq'),
    assert_contains(BashClient, 'svc__health()', 'Bash has health function'),
    assert_contains(BashClient, 'svc__data()', 'Bash has data function'),
    assert_contains(BashClient, 'http://localhost:8000/health', 'Bash has full URL'),

    format('~n').

%% ============================================
%% Test: Socket Server
%% ============================================

test_socket_server :-
    format('Test: Socket server generation~n'),

    % Go socket server
    generate_socket_server(go, 9000, [buffer_size(131072)], GoSocket),
    assert_contains(GoSocket, 'package main', 'Go has package main'),
    assert_contains(GoSocket, 'net.Listen("tcp"', 'Go listens on TCP'),
    assert_contains(GoSocket, '131072', 'Go has custom buffer'),
    assert_contains(GoSocket, 'handleConnection', 'Go has handler'),
    assert_contains(GoSocket, 'go handleConnection', 'Go uses goroutines'),

    % Python socket server
    generate_socket_server(python, 9000, [], PySocket),
    assert_contains(PySocket, '#!/usr/bin/env python3', 'Python has shebang'),
    assert_contains(PySocket, 'import socket', 'Python imports socket'),
    assert_contains(PySocket, 'import threading', 'Python imports threading'),
    assert_contains(PySocket, 'server.listen', 'Python listens'),
    assert_contains(PySocket, 'thread.daemon', 'Python uses daemon threads'),

    format('~n').

%% ============================================
%% Test: Socket Client
%% ============================================

test_socket_client :-
    format('Test: Socket client generation~n'),

    % Go socket client
    generate_socket_client(go, 'localhost', [port(9000)], GoClient),
    assert_contains(GoClient, 'type SocketClient struct', 'Go has client struct'),
    assert_contains(GoClient, 'net.Dial("tcp"', 'Go dials TCP'),
    assert_contains(GoClient, 'func (c *SocketClient) Send', 'Go has Send method'),
    assert_contains(GoClient, 'func (c *SocketClient) Close', 'Go has Close method'),

    % Python socket client
    generate_socket_client(python, '127.0.0.1', [port(8000)], PyClient),
    assert_contains(PyClient, 'class SocketClient', 'Python has client class'),
    assert_contains(PyClient, 'self.sock.connect', 'Python connects'),
    assert_contains(PyClient, 'def send(self', 'Python has send method'),
    assert_contains(PyClient, '__enter__', 'Python supports context manager'),

    format('~n').

%% ============================================
%% Test: Network Pipeline
%% ============================================

test_network_pipeline :-
    format('Test: Network pipeline generation~n'),

    % Python network pipeline
    generate_network_pipeline(
        [
            step(fetch, remote, 'http://api:8080/fetch', [timeout(30)]),
            step(process, local, '    result = {"processed": data}', []),
            step(store, remote, 'http://db:5000/store', [])
        ],
        [language(python)],
        PyPipeline
    ),

    assert_contains(PyPipeline, '#!/usr/bin/env python3', 'Python has shebang'),
    assert_contains(PyPipeline, 'import requests', 'Python imports requests'),
    assert_contains(PyPipeline, 'def step_fetch(data)', 'Python has fetch step'),
    assert_contains(PyPipeline, 'def step_process(data)', 'Python has process step'),
    assert_contains(PyPipeline, 'def step_store(data)', 'Python has store step'),
    assert_contains(PyPipeline, 'Remote step: fetch', 'Python documents remote'),
    assert_contains(PyPipeline, 'Local step: process', 'Python documents local'),
    assert_contains(PyPipeline, 'requests.post', 'Python calls remote'),
    assert_contains(PyPipeline, 'run_pipeline', 'Python has run function'),

    % Bash network pipeline
    generate_network_pipeline(
        [
            step(get, remote, 'http://api/get', []),
            step(transform, local, 'echo "$data" | jq ".transformed = true"', [])
        ],
        [language(bash)],
        BashPipeline
    ),

    assert_contains(BashPipeline, '#!/bin/bash', 'Bash has shebang'),
    assert_contains(BashPipeline, 'set -euo pipefail', 'Bash has strict mode'),
    assert_contains(BashPipeline, 'step_get()', 'Bash has get step'),
    assert_contains(BashPipeline, 'step_transform()', 'Bash has transform step'),
    assert_contains(BashPipeline, 'curl -s -X POST', 'Bash uses curl'),
    assert_contains(BashPipeline, 'run_pipeline()', 'Bash has run function'),

    format('~n').

%% ============================================
%% Main
%% ============================================

:- initialization(run_tests, main).
