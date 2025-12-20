/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Network Glue - Remote target communication via HTTP and sockets
 *
 * This module generates:
 * - HTTP server wrappers for exposing predicates as REST endpoints
 * - HTTP client wrappers for calling remote predicates
 * - Socket-based communication for low-latency streaming
 * - Service registry for discovery and routing
 */

:- module(network_glue, [
    % Service registry
    register_service/3,             % register_service(+Name, +URL, +Options)
    service/2,                      % service(?Name, ?URL)
    service_options/2,              % service_options(?Name, ?Options)
    unregister_service/1,           % unregister_service(+Name)

    % HTTP server generation
    generate_http_server/4,         % generate_http_server(+Target, +Endpoints, +Options, -Code)
    generate_go_http_server/3,      % generate_go_http_server(+Endpoints, +Options, -Code)
    generate_python_http_server/3,  % generate_python_http_server(+Endpoints, +Options, -Code)
    generate_rust_http_server/3,    % generate_rust_http_server(+Endpoints, +Options, -Code)

    % HTTP client generation
    generate_http_client/4,         % generate_http_client(+Target, +Services, +Options, -Code)
    generate_go_http_client/3,      % generate_go_http_client(+Services, +Options, -Code)
    generate_python_http_client/3,  % generate_python_http_client(+Services, +Options, -Code)
    generate_bash_http_client/3,    % generate_bash_http_client(+Services, +Options, -Code)

    % Socket communication
    generate_socket_server/4,       % generate_socket_server(+Target, +Port, +Options, -Code)
    generate_socket_client/4,       % generate_socket_client(+Target, +Host, +Options, -Code)

    % Network pipeline
    generate_network_pipeline/3,    % generate_network_pipeline(+Steps, +Options, -Code)

    % Utilities
    endpoint_url/3,                 % endpoint_url(+Service, +Endpoint, -URL)

    % KG Topology Phase 3: Distributed routing endpoints
    generate_kg_query_endpoint/3,   % generate_kg_query_endpoint(+Target, +Options, -Code)
    generate_kg_routes/3,           % generate_kg_routes(+Target, +Options, -Code)
    % KG Topology Phase 4: Federated query endpoints
    generate_federation_endpoint/3  % generate_federation_endpoint(+Target, +Options, -Code)
]).

:- use_module(library(lists)).

%% ============================================
%% Service Registry
%% ============================================

:- dynamic service_db/3.

%% register_service(+Name, +URL, +Options)
%  Register a remote service.
%
%  Options:
%    - timeout(Seconds)
%    - retries(N)
%    - format(json|tsv)
%    - auth(bearer(Token)|basic(User,Pass))
%
register_service(Name, URL, Options) :-
    retractall(service_db(Name, _, _)),
    assertz(service_db(Name, URL, Options)).

%% service(?Name, ?URL)
%  Query registered services.
%
service(Name, URL) :-
    service_db(Name, URL, _).

%% service_options(?Name, ?Options)
%  Get service options.
%
service_options(Name, Options) :-
    service_db(Name, _, Options).

%% unregister_service(+Name)
%  Remove a service from registry.
%
unregister_service(Name) :-
    retractall(service_db(Name, _, _)).

%% endpoint_url(+Service, +Endpoint, -URL)
%  Construct full URL for an endpoint.
%
endpoint_url(Service, Endpoint, URL) :-
    service(Service, BaseURL),
    atom_concat(BaseURL, Endpoint, URL).

%% ============================================
%% HTTP Server Generation
%% ============================================

%% generate_http_server(+Target, +Endpoints, +Options, -Code)
%  Generate HTTP server code for the specified target.
%
%  Endpoints: List of endpoint(Path, Handler, EndpointOptions)
%
generate_http_server(go, Endpoints, Options, Code) :-
    generate_go_http_server(Endpoints, Options, Code).
generate_http_server(python, Endpoints, Options, Code) :-
    generate_python_http_server(Endpoints, Options, Code).
generate_http_server(rust, Endpoints, Options, Code) :-
    generate_rust_http_server(Endpoints, Options, Code).

%% generate_go_http_server(+Endpoints, +Options, -Code)
%  Generate a Go HTTP server using net/http.
%
generate_go_http_server(Endpoints, Options, Code) :-
    option_or_default(port(Port), Options, 8080),
    option_or_default(cors(CORS), Options, true),

    maplist(go_endpoint_handler, Endpoints, Handlers),
    atomic_list_concat(Handlers, '\n', HandlersCode),

    maplist(go_endpoint_route, Endpoints, Routes),
    atomic_list_concat(Routes, '\n    ', RoutesCode),

    (CORS == true -> go_cors_middleware(CORSMiddleware) ; CORSMiddleware = ''),

    format(atom(Code), '
package main

import (
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net/http"
    "os"
)

// Request/Response types
type Request struct {
    Data interface{} `json:"data"`
}

type Response struct {
    Success bool        `json:"success"`
    Data    interface{} `json:"data,omitempty"`
    Error   string      `json:"error,omitempty"`
}

~w

~w

func main() {
    port := "~w"
    if p := os.Getenv("PORT"); p != "" {
        port = p
    }

    // Register handlers
    ~w

    log.Printf("Server starting on port %s", port)
    log.Fatal(http.ListenAndServe(":"+port, nil))
}
', [CORSMiddleware, HandlersCode, Port, RoutesCode]).

go_cors_middleware('
// CORS middleware
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

        if r.Method == "OPTIONS" {
            w.WriteHeader(http.StatusOK)
            return
        }

        next(w, r)
    }
}
').

go_endpoint_handler(endpoint(Path, Handler, _Opts), Code) :-
    path_to_func_name(Path, FuncName),
    format(atom(Code), '
// Handler for ~w
func ~wHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")

    // Read request body
    body, err := io.ReadAll(r.Body)
    if err != nil {
        json.NewEncoder(w).Encode(Response{Success: false, Error: "Failed to read request"})
        return
    }

    var req Request
    if len(body) > 0 {
        if err := json.Unmarshal(body, &req); err != nil {
            json.NewEncoder(w).Encode(Response{Success: false, Error: "Invalid JSON"})
            return
        }
    }

    // Process
    result, err := ~w(req.Data)
    if err != nil {
        json.NewEncoder(w).Encode(Response{Success: false, Error: err.Error()})
        return
    }

    json.NewEncoder(w).Encode(Response{Success: true, Data: result})
}
', [Path, FuncName, Handler]).

go_endpoint_route(endpoint(Path, _Handler, _Opts), Code) :-
    path_to_func_name(Path, FuncName),
    format(atom(Code), 'http.HandleFunc("~w", corsMiddleware(~wHandler))', [Path, FuncName]).

%% generate_python_http_server(+Endpoints, +Options, -Code)
%  Generate a Python HTTP server using Flask.
%
generate_python_http_server(Endpoints, Options, Code) :-
    option_or_default(port(Port), Options, 8080),
    option_or_default(cors(CORS), Options, true),

    maplist(python_endpoint_route, Endpoints, Routes),
    atomic_list_concat(Routes, '\n', RoutesCode),

    (CORS == true ->
        CORSImport = 'from flask_cors import CORS',
        CORSInit = 'CORS(app)'
    ;
        CORSImport = '',
        CORSInit = ''
    ),

    format(atom(Code), '
#!/usr/bin/env python3
"""
Generated HTTP Server
"""

from flask import Flask, request, jsonify
import os
~w

app = Flask(__name__)
~w

~w

@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({"success": False, "error": str(error)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", ~w))
    app.run(host="0.0.0.0", port=port, debug=False)
', [CORSImport, CORSInit, RoutesCode, Port]).

python_endpoint_route(endpoint(Path, Handler, Opts), Code) :-
    option_or_default(methods(Methods), Opts, ['POST']),
    methods_to_python_list(Methods, MethodsList),
    path_to_func_name(Path, RouteName),
    format(atom(Code), '
@app.route("~w", methods=~w)
def ~w_route():
    try:
        data = request.get_json() if request.is_json else {}
        result = ~w(data.get("data"))
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
', [Path, MethodsList, RouteName, Handler]).

methods_to_python_list(Methods, List) :-
    maplist(method_to_string, Methods, Strings),
    atomic_list_concat(Strings, ', ', Inner),
    format(atom(List), '[~w]', [Inner]).

method_to_string(Method, String) :-
    format(atom(String), '"~w"', [Method]).

%% generate_rust_http_server(+Endpoints, +Options, -Code)
%  Generate a Rust HTTP server using Actix-web.
%
generate_rust_http_server(Endpoints, Options, Code) :-
    option_or_default(port(Port), Options, 8080),

    maplist(rust_endpoint_handler, Endpoints, Handlers),
    atomic_list_concat(Handlers, '\n', HandlersCode),

    maplist(rust_endpoint_route, Endpoints, Routes),
    atomic_list_concat(Routes, '\n            ', RoutesCode),

    format(atom(Code), '
use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Deserialize)]
struct Request {
    data: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct Response {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

~w

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let port: u16 = env::var("PORT")
        .unwrap_or_else(|_| "~w".to_string())
        .parse()
        .unwrap();

    println!("Server starting on port {}", port);

    HttpServer::new(|| {
        let cors = Cors::permissive();

        App::new()
            .wrap(cors)
            ~w
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
', [HandlersCode, Port, RoutesCode]).

rust_endpoint_handler(endpoint(Path, Handler, _Opts), Code) :-
    path_to_func_name(Path, FuncName),
    format(atom(Code), '
async fn ~w_handler(req: web::Json<Request>) -> impl Responder {
    match ~w(req.data.clone()) {
        Ok(result) => HttpResponse::Ok().json(Response {
            success: true,
            data: Some(result),
            error: None,
        }),
        Err(e) => HttpResponse::BadRequest().json(Response {
            success: false,
            data: None,
            error: Some(e.to_string()),
        }),
    }
}
', [FuncName, Handler]).

rust_endpoint_route(endpoint(Path, _Handler, _Opts), Code) :-
    path_to_func_name(Path, FuncName),
    format(atom(Code), '.route("~w", web::post().to(~w_handler))', [Path, FuncName]).

%% ============================================
%% HTTP Client Generation
%% ============================================

%% generate_http_client(+Target, +Services, +Options, -Code)
%  Generate HTTP client code for calling remote services.
%
generate_http_client(go, Services, Options, Code) :-
    generate_go_http_client(Services, Options, Code).
generate_http_client(python, Services, Options, Code) :-
    generate_python_http_client(Services, Options, Code).
generate_http_client(bash, Services, Options, Code) :-
    generate_bash_http_client(Services, Options, Code).

%% generate_go_http_client(+Services, +Options, -Code)
%  Generate Go HTTP client.
%
generate_go_http_client(Services, Options, Code) :-
    option_or_default(timeout(Timeout), Options, 30),

    maplist(go_service_client, Services, Clients),
    atomic_list_concat(Clients, '\n', ClientsCode),

    format(atom(Code), '
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

var client = &http.Client{
    Timeout: ~w * time.Second,
}

type Request struct {
    Data interface{} `json:"data"`
}

type Response struct {
    Success bool            `json:"success"`
    Data    json.RawMessage `json:"data"`
    Error   string          `json:"error"`
}

func callService(url string, data interface{}) (json.RawMessage, error) {
    reqBody, err := json.Marshal(Request{Data: data})
    if err != nil {
        return nil, fmt.Errorf("marshal error: %w", err)
    }

    resp, err := client.Post(url, "application/json", bytes.NewReader(reqBody))
    if err != nil {
        return nil, fmt.Errorf("request error: %w", err)
    }
    defer resp.Body.Close()

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("read error: %w", err)
    }

    var result Response
    if err := json.Unmarshal(body, &result); err != nil {
        return nil, fmt.Errorf("unmarshal error: %w", err)
    }

    if !result.Success {
        return nil, fmt.Errorf("service error: %s", result.Error)
    }

    return result.Data, nil
}

~w
', [Timeout, ClientsCode]).

go_service_client(service_def(Name, URL, Endpoints), Code) :-
    maplist(go_service_endpoint(Name, URL), Endpoints, EndpointCodes),
    atomic_list_concat(EndpointCodes, '\n', Code).

go_service_endpoint(Name, BaseURL, Endpoint, Code) :-
    format(atom(FullURL), '~w~w', [BaseURL, Endpoint]),
    path_to_func_name(Endpoint, EndpointFunc),
    format(atom(FuncName), '~w_~w', [Name, EndpointFunc]),
    format(atom(Code), '
// Call ~w~w
func ~w(data interface{}) (json.RawMessage, error) {
    return callService("~w", data)
}
', [Name, Endpoint, FuncName, FullURL]).

%% generate_python_http_client(+Services, +Options, -Code)
%  Generate Python HTTP client using requests.
%
generate_python_http_client(Services, Options, Code) :-
    option_or_default(timeout(Timeout), Options, 30),

    maplist(python_service_client, Services, Clients),
    atomic_list_concat(Clients, '\n', ClientsCode),

    format(atom(Code), '
#!/usr/bin/env python3
"""
Generated HTTP Client
"""

import requests
from typing import Any, Optional

TIMEOUT = ~w

class ServiceError(Exception):
    pass

def call_service(url: str, data: Any = None) -> Any:
    """Call a remote service endpoint."""
    try:
        response = requests.post(
            url,
            json={"data": data},
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            raise ServiceError(result.get("error", "Unknown error"))

        return result.get("data")
    except requests.RequestException as e:
        raise ServiceError(f"Request failed: {e}")

~w
', [Timeout, ClientsCode]).

python_service_client(service_def(Name, URL, Endpoints), Code) :-
    maplist(python_service_endpoint(Name, URL), Endpoints, EndpointCodes),
    atomic_list_concat(EndpointCodes, '\n', Code).

python_service_endpoint(Name, BaseURL, Endpoint, Code) :-
    format(atom(FullURL), '~w~w', [BaseURL, Endpoint]),
    path_to_func_name(Endpoint, EndpointFunc),
    format(atom(FuncName), '~w_~w', [Name, EndpointFunc]),
    format(atom(Code), '
def ~w(data: Any = None) -> Any:
    """Call ~w~w"""
    return call_service("~w", data)
', [FuncName, Name, Endpoint, FullURL]).

%% generate_bash_http_client(+Services, +Options, -Code)
%  Generate Bash HTTP client using curl.
%
generate_bash_http_client(Services, Options, Code) :-
    option_or_default(timeout(Timeout), Options, 30),

    maplist(bash_service_client, Services, Clients),
    atomic_list_concat(Clients, '\n', ClientsCode),

    format(atom(Code), '#!/bin/bash
# Generated HTTP Client

TIMEOUT=~w

call_service() {
    local url="$1"
    local data="$2"

    response=$(curl -s -X POST \\
        -H "Content-Type: application/json" \\
        -d "{\\\"data\\\": $data}" \\
        --max-time "$TIMEOUT" \\
        "$url")

    success=$(echo "$response" | jq -r ".success")
    if [ "$success" != "true" ]; then
        error=$(echo "$response" | jq -r ".error")
        echo "Error: $error" >&2
        return 1
    fi

    echo "$response" | jq -r ".data"
}

~w
', [Timeout, ClientsCode]).

bash_service_client(service_def(Name, URL, Endpoints), Code) :-
    maplist(bash_service_endpoint(Name, URL), Endpoints, EndpointCodes),
    atomic_list_concat(EndpointCodes, '\n', Code).

bash_service_endpoint(Name, BaseURL, Endpoint, Code) :-
    format(atom(FullURL), '~w~w', [BaseURL, Endpoint]),
    path_to_func_name(Endpoint, EndpointFunc),
    format(atom(FuncName), '~w_~w', [Name, EndpointFunc]),
    format(atom(Code), '
# Call ~w~w
~w() {
    local data="${1:-null}"
    call_service "~w" "$data"
}
', [Name, Endpoint, FuncName, FullURL]).

%% ============================================
%% Socket Communication
%% ============================================

%% generate_socket_server(+Target, +Port, +Options, -Code)
%  Generate a TCP socket server for low-latency streaming.
%
generate_socket_server(go, Port, Options, Code) :-
    option_or_default(format(_Format), Options, json),
    option_or_default(buffer_size(BufSize), Options, 65536),

    format(atom(Code), '
package main

import (
    "bufio"
    "encoding/json"
    "fmt"
    "io"
    "log"
    "net"
    "os"
)

const bufferSize = ~w

func handleConnection(conn net.Conn) {
    defer conn.Close()
    reader := bufio.NewReaderSize(conn, bufferSize)
    writer := bufio.NewWriterSize(conn, bufferSize)

    for {
        // Read line
        line, err := reader.ReadBytes(\'\\n\')
        if err != nil {
            if err != io.EOF {
                log.Printf("Read error: %v", err)
            }
            return
        }

        // Process
        result, err := process(line)
        if err != nil {
            log.Printf("Process error: %v", err)
            continue
        }

        // Write response
        writer.Write(result)
        writer.WriteByte(\'\\n\')
        writer.Flush()
    }
}

func process(data []byte) ([]byte, error) {
    // TODO: Implement processing logic
    return data, nil
}

func main() {
    port := "~w"
    if p := os.Getenv("PORT"); p != "" {
        port = p
    }

    listener, err := net.Listen("tcp", ":"+port)
    if err != nil {
        log.Fatalf("Failed to listen: %v", err)
    }
    defer listener.Close()

    log.Printf("Socket server listening on port %s", port)

    for {
        conn, err := listener.Accept()
        if err != nil {
            log.Printf("Accept error: %v", err)
            continue
        }
        go handleConnection(conn)
    }
}
', [BufSize, Port]).

generate_socket_server(python, Port, Options, Code) :-
    option_or_default(buffer_size(BufSize), Options, 65536),

    format(atom(Code), '
#!/usr/bin/env python3
"""
Generated Socket Server
"""

import socket
import threading
import json
import os

BUFFER_SIZE = ~w

def handle_connection(conn, addr):
    print(f"Connection from {addr}")
    try:
        while True:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                break

            # Process
            result = process(data)

            # Send response
            conn.sendall(result + b"\\n")
    except Exception as e:
        print(f"Error handling {addr}: {e}")
    finally:
        conn.close()

def process(data: bytes) -> bytes:
    # TODO: Implement processing logic
    return data

def main():
    port = int(os.environ.get("PORT", ~w))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(5)

    print(f"Socket server listening on port {port}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_connection, args=(conn, addr))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    main()
', [BufSize, Port]).

%% generate_socket_client(+Target, +Host, +Options, -Code)
%  Generate a TCP socket client.
%
generate_socket_client(go, Host, Options, Code) :-
    option_or_default(port(Port), Options, 9000),
    option_or_default(buffer_size(BufSize), Options, 65536),

    format(atom(Code), '
package main

import (
    "bufio"
    "fmt"
    "net"
)

const bufferSize = ~w

type SocketClient struct {
    conn   net.Conn
    reader *bufio.Reader
    writer *bufio.Writer
}

func NewSocketClient(host string, port int) (*SocketClient, error) {
    addr := fmt.Sprintf("%s:%d", host, port)
    conn, err := net.Dial("tcp", addr)
    if err != nil {
        return nil, err
    }

    return &SocketClient{
        conn:   conn,
        reader: bufio.NewReaderSize(conn, bufferSize),
        writer: bufio.NewWriterSize(conn, bufferSize),
    }, nil
}

func (c *SocketClient) Send(data []byte) ([]byte, error) {
    // Write
    c.writer.Write(data)
    c.writer.WriteByte(\'\\n\')
    if err := c.writer.Flush(); err != nil {
        return nil, err
    }

    // Read response
    return c.reader.ReadBytes(\'\\n\')
}

func (c *SocketClient) Close() {
    c.conn.Close()
}

// Example usage
func main() {
    client, err := NewSocketClient("~w", ~w)
    if err != nil {
        panic(err)
    }
    defer client.Close()

    response, err := client.Send([]byte("hello"))
    if err != nil {
        panic(err)
    }
    fmt.Printf("Response: %s", response)
}
', [BufSize, Host, Port]).

generate_socket_client(python, Host, Options, Code) :-
    option_or_default(port(Port), Options, 9000),
    option_or_default(buffer_size(BufSize), Options, 65536),

    format(atom(Code), '
#!/usr/bin/env python3
"""
Generated Socket Client
"""

import socket

BUFFER_SIZE = ~w

class SocketClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send(self, data: bytes) -> bytes:
        self.sock.sendall(data + b"\\n")
        return self.sock.recv(BUFFER_SIZE)

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

# Example usage
if __name__ == "__main__":
    with SocketClient("~w", ~w) as client:
        response = client.send(b"hello")
        print(f"Response: {response}")
', [BufSize, Host, Port]).

%% ============================================
%% Network Pipeline
%% ============================================

%% generate_network_pipeline(+Steps, +Options, -Code)
%  Generate a pipeline that includes network calls.
%
%  Steps: List of step(Name, Type, Config, StepOptions)
%         Type = local | remote
%
generate_network_pipeline(Steps, Options, Code) :-
    option_or_default(language(Lang), Options, python),

    (Lang == python ->
        generate_python_network_pipeline(Steps, Options, Code)
    ; Lang == go ->
        generate_go_network_pipeline(Steps, Options, Code)
    ;
        generate_bash_network_pipeline(Steps, Options, Code)
    ).

generate_python_network_pipeline(Steps, _Options, Code) :-
    maplist(python_pipeline_step, Steps, StepCodes),
    atomic_list_concat(StepCodes, '\n', StepsCode),

    length(Steps, N),
    numlist(1, N, Nums),
    maplist(python_step_call, Nums, Calls),
    atomic_list_concat(Calls, '\n        ', CallsCode),

    format(atom(Code), '
#!/usr/bin/env python3
"""
Generated Network Pipeline
"""

import requests
import sys
import json

~w

def run_pipeline(input_data):
    data = input_data
    ~w
    return data

if __name__ == "__main__":
    # Read from stdin or use argument
    if len(sys.argv) > 1:
        input_data = json.loads(sys.argv[1])
    else:
        input_data = json.load(sys.stdin)

    result = run_pipeline(input_data)
    print(json.dumps(result, indent=2))
', [StepsCode, CallsCode]).

python_pipeline_step(step(Name, local, Script, _Opts), Code) :-
    format(atom(Code), '
def step_~w(data):
    """Local step: ~w"""
~w
    return result
', [Name, Name, Script]).

python_pipeline_step(step(Name, remote, URL, Opts), Code) :-
    option_or_default(timeout(Timeout), Opts, 30),
    format(atom(Code), '
def step_~w(data):
    """Remote step: ~w"""
    response = requests.post(
        "~w",
        json={"data": data},
        timeout=~w
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("success"):
        raise Exception(result.get("error"))
    return result.get("data")
', [Name, Name, URL, Timeout]).

python_step_call(N, Call) :-
    format(atom(Call), 'data = step_~w(data)  # Step ~w', [N, N]).

generate_bash_network_pipeline(Steps, _Options, Code) :-
    maplist(bash_pipeline_step, Steps, StepCodes),
    atomic_list_concat(StepCodes, '\n', StepsCode),

    length(Steps, N),
    numlist(1, N, Nums),
    maplist(bash_step_call, Nums, Calls),
    atomic_list_concat(Calls, '\n    ', CallsCode),

    format(atom(Code), '#!/bin/bash
# Generated Network Pipeline

set -euo pipefail

~w

run_pipeline() {
    local data="$1"
    ~w
    echo "$data"
}

# Run
input="${1:-$(cat)}"
run_pipeline "$input"
', [StepsCode, CallsCode]).

bash_pipeline_step(step(Name, local, Script, _Opts), Code) :-
    format(atom(Code), '
step_~w() {
    local data="$1"
    ~w
}
', [Name, Script]).

bash_pipeline_step(step(Name, remote, URL, Opts), Code) :-
    option_or_default(timeout(Timeout), Opts, 30),
    format(atom(Code), '
step_~w() {
    local data="$1"
    curl -s -X POST \\
        -H "Content-Type: application/json" \\
        -d "{\\\"data\\\": $data}" \\
        --max-time ~w \\
        "~w" | jq -r ".data"
}
', [Name, Timeout, URL]).

bash_step_call(N, Call) :-
    format(atom(Call), 'data=$(step_~w "$data")  # Step ~w', [N, N]).

generate_go_network_pipeline(Steps, _Options, Code) :-
    maplist(go_pipeline_step, Steps, StepCodes),
    atomic_list_concat(StepCodes, '\n', StepsCode),

    length(Steps, N),
    numlist(1, N, Nums),
    maplist(go_step_call, Nums, Steps, Calls),
    atomic_list_concat(Calls, '\n    ', CallsCode),

    format(atom(Code), '
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "time"
)

var client = &http.Client{Timeout: 30 * time.Second}

~w

func runPipeline(input interface{}) (interface{}, error) {
    data := input
    var err error
    ~w
    return data, nil
}

func main() {
    var input interface{}
    if err := json.NewDecoder(os.Stdin).Decode(&input); err != nil {
        fmt.Fprintf(os.Stderr, "Error reading input: %v\\n", err)
        os.Exit(1)
    }

    result, err := runPipeline(input)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Pipeline error: %v\\n", err)
        os.Exit(1)
    }

    json.NewEncoder(os.Stdout).Encode(result)
}
', [StepsCode, CallsCode]).

go_pipeline_step(step(Name, local, Logic, _Opts), Code) :-
    format(atom(Code), '
func step_~w(data interface{}) (interface{}, error) {
    // Local step: ~w
~w
}
', [Name, Name, Logic]).

go_pipeline_step(step(Name, remote, URL, _Opts), Code) :-
    format(atom(Code), '
func step_~w(data interface{}) (interface{}, error) {
    // Remote step: ~w
    reqBody, _ := json.Marshal(map[string]interface{}{"data": data})
    resp, err := client.Post("~w", "application/json", bytes.NewReader(reqBody))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    var result struct {
        Success bool        `json:"success"`
        Data    interface{} `json:"data"`
        Error   string      `json:"error"`
    }
    json.Unmarshal(body, &result)
    if !result.Success {
        return nil, fmt.Errorf(result.Error)
    }
    return result.Data, nil
}
', [Name, Name, URL]).

go_step_call(N, step(Name, _, _, _), Call) :-
    format(atom(Call), 'data, err = step_~w(data)  // Step ~w: ~w
    if err != nil { return nil, err }', [Name, N, Name]).

%% ============================================
%% Utility Predicates
%% ============================================

option_or_default(Option, Options, _Default) :-
    member(Option, Options),
    !.
option_or_default(Option, _Options, Default) :-
    Option =.. [_, Default].

path_to_func_name(Path, FuncName) :-
    atom_string(Path, PathStr),
    string_chars(PathStr, Chars),
    maplist(sanitize_char, Chars, SanitizedChars),
    string_chars(SanitizedStr, SanitizedChars),
    atom_string(FuncName, SanitizedStr).

sanitize_char('/', '_') :- !.
sanitize_char('-', '_') :- !.
sanitize_char('.', '_') :- !.
sanitize_char(C, C).

%% ============================================
%% KG Topology Phase 3: Distributed Query Endpoints
%% ============================================

%% generate_kg_query_endpoint(+Target, +Options, -Code)
%  Generate HTTP endpoint handlers for distributed KG queries.
%  Endpoints:
%    - POST /kg/query   - Handle distributed KG query
%    - POST /kg/register - Register node with discovery
%    - GET  /kg/health  - Health check for KG node

generate_kg_query_endpoint(python, Options, Code) :-
    ( member(api_instance(APIVar), Options) -> true ; APIVar = 'kg_api' ),
    format(string(Code), "
# KG Topology Phase 3: Distributed Query Endpoints

@app.route('/kg/query', methods=['POST'])
def handle_kg_query():
    '''Handle distributed KG query from another node.'''
    try:
        request_data = request.json

        if request_data.get('__type') != 'kg_query':
            return jsonify({'error': 'Invalid request type'}), 400

        result = ~w.handle_remote_query(request_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            '__type': 'kg_response',
            'error': str(e),
            'source_node': ~w.node_id
        }), 500


@app.route('/kg/register', methods=['POST'])
def handle_kg_register():
    '''Register this node with discovery service.'''
    try:
        data = request.json or {}
        host = data.get('host', 'localhost')
        port = data.get('port', 8080)
        tags = data.get('tags', ['kg_node'])

        success = ~w.register_node(host=host, port=port, tags=tags)

        return jsonify({
            'status': 'registered' if success else 'failed',
            'node_id': ~w.node_id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/kg/health', methods=['GET'])
def handle_kg_health():
    '''Health check for KG node.'''
    try:
        interfaces = ~w.list_interfaces(active_only=True)
        stats = ~w.get_query_stats()

        return jsonify({
            'status': 'healthy',
            'node_id': ~w.node_id,
            'interfaces': len(interfaces),
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
", [APIVar, APIVar, APIVar, APIVar, APIVar, APIVar, APIVar]).


generate_kg_query_endpoint(go, Options, Code) :-
    ( member(api_instance(APIVar), Options) -> true ; APIVar = 'kgAPI' ),
    format(string(Code), '
// KG Topology Phase 3: Distributed Query Endpoints

// handleKGQuery handles distributed KG queries from other nodes
func handleKGQuery(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")

    var request map[string]interface{}
    if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
        json.NewEncoder(w).Encode(map[string]interface{}{
            "__type": "kg_response",
            "error":  "Invalid JSON",
        })
        return
    }

    reqType, _ := request["__type"].(string)
    if reqType != "kg_query" {
        json.NewEncoder(w).Encode(map[string]interface{}{
            "__type": "kg_response",
            "error":  "Invalid request type",
        })
        return
    }

    result, err := ~w.HandleRemoteQuery(request)
    if err != nil {
        json.NewEncoder(w).Encode(map[string]interface{}{
            "__type":      "kg_response",
            "error":       err.Error(),
            "source_node": ~w.NodeID,
        })
        return
    }

    json.NewEncoder(w).Encode(result)
}

// handleKGRegister registers this node with discovery service
func handleKGRegister(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")

    var data struct {
        Host string   `json:"host"`
        Port int      `json:"port"`
        Tags []string `json:"tags"`
    }

    if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
        data.Host = "localhost"
        data.Port = 8080
        data.Tags = []string{"kg_node"}
    }

    success := ~w.RegisterNode(data.Host, data.Port, data.Tags)

    json.NewEncoder(w).Encode(map[string]interface{}{
        "status":  map[bool]string{true: "registered", false: "failed"}[success],
        "node_id": ~w.NodeID,
    })
}

// handleKGHealth returns health check for KG node
func handleKGHealth(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")

    interfaces := ~w.ListInterfaces(true)
    stats := ~w.GetQueryStats()

    json.NewEncoder(w).Encode(map[string]interface{}{
        "status":     "healthy",
        "node_id":    ~w.NodeID,
        "interfaces": len(interfaces),
        "stats":      stats,
    })
}

// RegisterKGRoutes adds KG topology routes to the given mux
func RegisterKGRoutes(mux *http.ServeMux) {
    mux.HandleFunc("POST /kg/query", handleKGQuery)
    mux.HandleFunc("POST /kg/register", handleKGRegister)
    mux.HandleFunc("GET /kg/health", handleKGHealth)
}
', [APIVar, APIVar, APIVar, APIVar, APIVar, APIVar, APIVar]).


generate_kg_query_endpoint(rust, Options, Code) :-
    ( member(api_instance(APIVar), Options) -> true ; APIVar = 'kg_api' ),
    format(string(Code), '
// KG Topology Phase 3: Distributed Query Endpoints

use axum::{
    extract::State,
    http::StatusCode,
    Json,
    routing::{get, post},
    Router,
};
use serde_json::{json, Value};
use std::sync::Arc;

/// Handle distributed KG query from another node
async fn handle_kg_query(
    State(~w): State<Arc<DistributedKGTopologyAPI>>,
    Json(request): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let req_type = request.get("__type").and_then(|v| v.as_str());

    if req_type != Some("kg_query") {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "__type": "kg_response",
                "error": "Invalid request type"
            })),
        ));
    }

    match ~w.handle_remote_query(&request) {
        Ok(result) => Ok(Json(result)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "__type": "kg_response",
                "error": e.to_string(),
                "source_node": ~w.node_id()
            })),
        )),
    }
}

/// Register this node with discovery service
async fn handle_kg_register(
    State(~w): State<Arc<DistributedKGTopologyAPI>>,
    Json(data): Json<Value>,
) -> Json<Value> {
    let host = data.get("host")
        .and_then(|v| v.as_str())
        .unwrap_or("localhost");
    let port = data.get("port")
        .and_then(|v| v.as_u64())
        .unwrap_or(8080) as u16;
    let tags: Vec<String> = data.get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_else(|| vec!["kg_node".to_string()]);

    let success = ~w.register_node(host, port, &tags);

    Json(json!({
        "status": if success { "registered" } else { "failed" },
        "node_id": ~w.node_id()
    }))
}

/// Health check for KG node
async fn handle_kg_health(
    State(~w): State<Arc<DistributedKGTopologyAPI>>,
) -> Json<Value> {
    let interfaces = ~w.list_interfaces(true);
    let stats = ~w.get_query_stats();

    Json(json!({
        "status": "healthy",
        "node_id": ~w.node_id(),
        "interfaces": interfaces.len(),
        "stats": stats
    }))
}

/// Create router with KG topology routes
pub fn kg_routes(api: Arc<DistributedKGTopologyAPI>) -> Router {
    Router::new()
        .route("/kg/query", post(handle_kg_query))
        .route("/kg/register", post(handle_kg_register))
        .route("/kg/health", get(handle_kg_health))
        .with_state(api)
}
', [APIVar, APIVar, APIVar, APIVar, APIVar, APIVar, APIVar, APIVar]).


%% generate_kg_routes(+Target, +Options, -Code)
%  Generate route registration code.

generate_kg_routes(python, Options, Code) :-
    format(string(Code), "
# Add KG routes to Flask app
# Note: Use generate_kg_query_endpoint to get the endpoint handlers
", []).

generate_kg_routes(go, _Options, Code) :-
    format(string(Code), '
// Add KG routes to HTTP mux:
//   RegisterKGRoutes(mux)
', []).

generate_kg_routes(rust, _Options, Code) :-
    format(string(Code), '
// Add KG routes to Axum router:
//   let app = Router::new().merge(kg_routes(api));
', []).


% =============================================================================
% KG TOPOLOGY PHASE 4: FEDERATED QUERY ENDPOINTS
% =============================================================================

%% generate_federation_endpoint(+Target, +Options, -Code)
%  Generate federation-specific HTTP endpoints for distributed queries.
%  These endpoints handle incoming federated queries and initiate outgoing federation.

generate_federation_endpoint(python, Options, Code) :-
    % Extract federation options
    ( member(federation_k(K), Options) -> true ; K = 3 ),
    ( member(aggregation(Strategy), Options) -> true ; Strategy = sum ),
    ( member(timeout_ms(Timeout), Options) -> true ; Timeout = 5000 ),

    format(string(Code), '
# KG Topology Phase 4: Federated Query Endpoints
# Generated from Prolog service definition

from flask import Flask, request, jsonify
from federated_query import (
    FederatedQueryEngine,
    AggregationStrategy,
    AggregationConfig,
)

# Federation configuration
FEDERATION_K = ~w
AGGREGATION_STRATEGY = "~w"
TIMEOUT_MS = ~w

@app.route("/kg/federated", methods=["POST"])
def handle_federated_query():
    """
    Handle incoming federated query from another node.

    This endpoint is called by other nodes during federation.
    It executes a local search and returns exp_scores + partition_sum
    for distributed softmax aggregation.
    """
    data = request.get_json()

    if data.get("__type") != "kg_federated_query":
        return jsonify({"error": "Invalid request type"}), 400

    try:
        response = kg_api.handle_federated_query(data)
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "__type": "kg_federated_response",
            "__id": data.get("__id"),
            "source_node": kg_api.node_id,
            "error": str(e),
            "results": [],
            "partition_sum": 0.0
        }), 500


@app.route("/kg/federate", methods=["POST"])
def initiate_federation():
    """
    Initiate a federated query across the KG network.

    This endpoint is called by clients to query across all nodes.
    It coordinates with other nodes and aggregates results.
    """
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = data.get("top_k", 5)

    if not query_text:
        return jsonify({"error": "query_text is required"}), 400

    try:
        response = federation_engine.federated_query(
            query_text=query_text,
            top_k=top_k
        )
        return jsonify(response.to_dict())
    except Exception as e:
        return jsonify({
            "error": str(e),
            "results": []
        }), 500


@app.route("/kg/federation/stats", methods=["GET"])
def federation_stats():
    """Return federation statistics."""
    stats = federation_engine.get_stats()
    corpus_info = kg_api.get_corpus_info()

    return jsonify({
        "federation": stats,
        "corpus": corpus_info,
        "node_id": kg_api.node_id
    })
', [K, Strategy, Timeout]).


generate_federation_endpoint(go, Options, Code) :-
    % Extract federation options
    ( member(federation_k(K), Options) -> true ; K = 3 ),
    ( member(aggregation(Strategy), Options) -> true ; Strategy = sum ),
    ( member(timeout_ms(Timeout), Options) -> true ; Timeout = 5000 ),

    format(string(Code), '
// KG Topology Phase 4: Federated Query Endpoints
// Generated from Prolog service definition

package main

import (
    "context"
    "encoding/json"
    "net/http"
    "time"
)

// Federation configuration
const (
    FederationK       = ~w
    AggregationMethod = "~w"
    FederationTimeout = ~w * time.Millisecond
)

// HandleFederatedQuery handles incoming federated queries from other nodes
func (s *Server) HandleFederatedQuery(w http.ResponseWriter, r *http.Request) {
    var req map[string]interface{}
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    reqType, _ := req["__type"].(string)
    if reqType != "kg_federated_query" {
        http.Error(w, "Invalid request type", http.StatusBadRequest)
        return
    }

    response, err := s.kgAPI.HandleFederatedQuery(req)
    if err != nil {
        errorResp := map[string]interface{}{
            "__type":        "kg_federated_response",
            "__id":          req["__id"],
            "source_node":   s.kgAPI.NodeID,
            "error":         err.Error(),
            "results":       []interface{}{},
            "partition_sum": 0.0,
        }
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(errorResp)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// InitiateFederation starts a federated query across the network
func (s *Server) InitiateFederation(w http.ResponseWriter, r *http.Request) {
    var req struct {
        QueryText string `json:"query_text"`
        TopK      int    `json:"top_k"`
    }
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    if req.QueryText == "" {
        http.Error(w, "query_text is required", http.StatusBadRequest)
        return
    }

    if req.TopK == 0 {
        req.TopK = 5
    }

    ctx, cancel := context.WithTimeout(r.Context(), FederationTimeout)
    defer cancel()

    response, err := s.federationEngine.FederatedQuery(
        ctx,
        req.QueryText,
        nil,  // embedding computed by engine
        req.TopK,
    )
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response.ToDict())
}

// FederationStats returns federation statistics
func (s *Server) FederationStats(w http.ResponseWriter, r *http.Request) {
    stats := s.federationEngine.GetStats()
    corpusInfo := s.kgAPI.GetCorpusInfo()

    response := map[string]interface{}{
        "federation": stats,
        "corpus":     corpusInfo,
        "node_id":    s.kgAPI.NodeID,
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// RegisterFederationRoutes registers federation endpoints
func RegisterFederationRoutes(mux *http.ServeMux, server *Server) {
    mux.HandleFunc("/kg/federated", server.HandleFederatedQuery)
    mux.HandleFunc("/kg/federate", server.InitiateFederation)
    mux.HandleFunc("/kg/federation/stats", server.FederationStats)
}
', [K, Strategy, Timeout]).


generate_federation_endpoint(rust, Options, Code) :-
    % Extract federation options
    ( member(federation_k(K), Options) -> true ; K = 3 ),
    ( member(aggregation(Strategy), Options) -> true ; Strategy = sum ),
    ( member(timeout_ms(Timeout), Options) -> true ; Timeout = 5000 ),

    format(string(Code), '
// KG Topology Phase 4: Federated Query Endpoints
// Generated from Prolog service definition

use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::{timeout, Duration};

// Federation configuration
pub const FEDERATION_K: usize = ~w;
pub const AGGREGATION_METHOD: &str = "~w";
pub const FEDERATION_TIMEOUT_MS: u64 = ~w;

#[derive(Deserialize)]
pub struct FederatedQueryRequest {
    #[serde(rename = "__type")]
    pub request_type: String,
    #[serde(rename = "__id")]
    pub request_id: Option<String>,
    pub payload: Option<QueryPayload>,
}

#[derive(Deserialize)]
pub struct QueryPayload {
    pub query_text: String,
    pub top_k: Option<usize>,
}

#[derive(Deserialize)]
pub struct FederateRequest {
    pub query_text: String,
    pub top_k: Option<usize>,
}

#[derive(Serialize)]
pub struct FederationStatsResponse {
    pub federation: serde_json::Value,
    pub corpus: serde_json::Value,
    pub node_id: String,
}

/// Handle incoming federated query from another node
pub async fn handle_federated_query(
    State(api): State<Arc<DistributedKGTopologyAPI>>,
    Json(req): Json<FederatedQueryRequest>,
) -> impl IntoResponse {
    if req.request_type != "kg_federated_query" {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "Invalid request type"})),
        );
    }

    match api.handle_federated_query(&req).await {
        Ok(response) => (StatusCode::OK, Json(response)),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "__type": "kg_federated_response",
                "__id": req.request_id,
                "source_node": api.node_id(),
                "error": e.to_string(),
                "results": [],
                "partition_sum": 0.0
            })),
        ),
    }
}

/// Initiate a federated query across the network
pub async fn initiate_federation(
    State(engine): State<Arc<FederatedQueryEngine>>,
    Json(req): Json<FederateRequest>,
) -> impl IntoResponse {
    if req.query_text.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "query_text is required"})),
        );
    }

    let top_k = req.top_k.unwrap_or(5);
    let timeout_duration = Duration::from_millis(FEDERATION_TIMEOUT_MS);

    match timeout(
        timeout_duration,
        engine.federated_query(&req.query_text, top_k),
    )
    .await
    {
        Ok(Ok(response)) => (StatusCode::OK, Json(response.to_dict())),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string(), "results": []})),
        ),
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(serde_json::json!({"error": "Federation timeout", "results": []})),
        ),
    }
}

/// Return federation statistics
pub async fn federation_stats(
    State(api): State<Arc<DistributedKGTopologyAPI>>,
    State(engine): State<Arc<FederatedQueryEngine>>,
) -> impl IntoResponse {
    Json(FederationStatsResponse {
        federation: engine.get_stats(),
        corpus: api.get_corpus_info(),
        node_id: api.node_id().to_string(),
    })
}

/// Create router with federation routes
pub fn federation_routes(
    api: Arc<DistributedKGTopologyAPI>,
    engine: Arc<FederatedQueryEngine>,
) -> Router {
    Router::new()
        .route("/kg/federated", post(handle_federated_query))
        .route("/kg/federate", post(initiate_federation))
        .route("/kg/federation/stats", get(federation_stats))
        .with_state(api)
        .with_state(engine)
}
', [K, Strategy, Timeout]).


% =============================================================================
% PHASE 6e: CROSS-MODEL FEDERATION ENDPOINTS
% =============================================================================

%% generate_cross_model_endpoint(+Target, +Options, -Code)
%  Generate HTTP endpoints for cross-model federation.

generate_cross_model_endpoint(python, Options, Code) :-
    % Extract options with defaults
    (member(fusion_method(FusionMethod), Options) -> true ; FusionMethod = weighted_sum),
    (member(rrf_k(RRFk), Options) -> true ; RRFk = 60),
    (member(consensus_threshold(ConsThresh), Options) -> true ; ConsThresh = 0.1),
    (member(consensus_boost(ConsBoost), Options) -> true ; ConsBoost = 1.5),
    format(atom(Code), '
from flask import request, jsonify
from cross_model_federation import (
    CrossModelFederatedEngine,
    CrossModelConfig,
    ModelPoolConfig,
    FusionMethod,
    AdaptiveModelWeights
)
import json
import os

# Cross-model state (initialized at startup)
cross_model_engine = None
adaptive_weights = None
WEIGHTS_FILE = os.environ.get("WEIGHTS_FILE", "model_weights.json")

def init_cross_model(router, pool_configs):
    """Initialize cross-model engine."""
    global cross_model_engine, adaptive_weights

    pools = [
        ModelPoolConfig(
            model_name=p["model"],
            weight=p.get("weight", 1.0),
            federation_k=p.get("federation_k", 5)
        )
        for p in pool_configs
    ]

    config = CrossModelConfig(
        pools=pools,
        fusion_method=FusionMethod.~w,
        rrf_k=~w,
        consensus_threshold=~w,
        consensus_boost_factor=~w
    )

    cross_model_engine = CrossModelFederatedEngine(router, config)
    models = [p.model_name for p in pools]
    adaptive_weights = AdaptiveModelWeights(models)

    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "r") as f:
            adaptive_weights = AdaptiveModelWeights.from_dict(json.load(f))

@app.route("/kg/cross-model", methods=["POST"])
def cross_model_query():
    if not cross_model_engine:
        return jsonify({"error": "Not initialized"}), 503
    data = request.json
    response = cross_model_engine.federated_query(
        query_text=data.get("query_text", ""),
        top_k=data.get("top_k", 10)
    )
    return jsonify(response.to_dict())

@app.route("/kg/cross-model/pools", methods=["GET"])
def cross_model_pools():
    if not cross_model_engine:
        return jsonify({"error": "Not initialized"}), 503
    pools = cross_model_engine.discover_pools()
    return jsonify({"pools": {m: len(n) for m, n in pools.items()}})

@app.route("/kg/cross-model/weights", methods=["GET"])
def get_weights():
    if not adaptive_weights:
        return jsonify({"error": "Not initialized"}), 503
    return jsonify(adaptive_weights.to_dict())

@app.route("/kg/cross-model/weights", methods=["PUT"])
def set_weights():
    if not adaptive_weights:
        return jsonify({"error": "Not initialized"}), 503
    for m, w in request.json.get("weights", {}).items():
        if m in adaptive_weights.weights:
            adaptive_weights.weights[m] = w
    total = sum(adaptive_weights.weights.values())
    if total > 0:
        adaptive_weights.weights = {m: w/total for m, w in adaptive_weights.weights.items()}
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(adaptive_weights.to_dict(), f)
    return jsonify(adaptive_weights.to_dict())

@app.route("/kg/cross-model/feedback", methods=["POST"])
def submit_feedback():
    if not adaptive_weights:
        return jsonify({"error": "Not initialized"}), 503
    data = request.json
    adaptive_weights.update(data["chosen_answer"], data.get("pool_rankings", {}))
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(adaptive_weights.to_dict(), f)
    return jsonify({"status": "updated", "weights": adaptive_weights.get_weights()})

@app.route("/kg/cross-model/stats", methods=["GET"])
def cross_model_stats():
    if not cross_model_engine:
        return jsonify({"error": "Not initialized"}), 503
    return jsonify(cross_model_engine.get_stats())
', [FusionMethod, RRFk, ConsThresh, ConsBoost]).

generate_cross_model_endpoint(go, Options, Code) :-
    (member(fusion_method(FusionMethod), Options) -> true ; FusionMethod = weighted_sum),
    format(atom(Code), '
// Cross-model federation handlers for Go
// See cross_model_federation.go for full implementation

type CrossModelEngine struct {
    pools   map[string]*PoolRouter
    weights map[string]float64
    mu      sync.RWMutex
}

func (e *CrossModelEngine) HandleQuery(w http.ResponseWriter, r *http.Request) {
    var req struct {
        QueryText string `json:"query_text"`
        TopK      int    `json:"top_k"`
    }
    json.NewDecoder(r.Body).Decode(&req)
    response := e.FederatedQuery(req.QueryText, req.TopK)
    json.NewEncoder(w).Encode(response)
}

func (e *CrossModelEngine) HandlePools(w http.ResponseWriter, r *http.Request) {
    pools := make(map[string]int)
    for m, router := range e.pools {
        pools[m] = len(router.DiscoverNodes())
    }
    json.NewEncoder(w).Encode(map[string]interface{}{"pools": pools})
}

func (e *CrossModelEngine) HandleWeights(w http.ResponseWriter, r *http.Request) {
    e.mu.RLock()
    defer e.mu.RUnlock()
    json.NewEncoder(w).Encode(map[string]interface{}{"weights": e.weights})
}

func (e *CrossModelEngine) HandleFeedback(w http.ResponseWriter, r *http.Request) {
    var req struct {
        ChosenAnswer string              `json:"chosen_answer"`
        PoolRankings map[string][]string `json:"pool_rankings"`
    }
    json.NewDecoder(r.Body).Decode(&req)
    e.UpdateWeights(req.ChosenAnswer, req.PoolRankings)
    json.NewEncoder(w).Encode(map[string]string{"status": "updated"})
}

func RegisterCrossModelRoutes(mux *http.ServeMux, e *CrossModelEngine) {
    mux.HandleFunc("/kg/cross-model", e.HandleQuery)
    mux.HandleFunc("/kg/cross-model/pools", e.HandlePools)
    mux.HandleFunc("/kg/cross-model/weights", e.HandleWeights)
    mux.HandleFunc("/kg/cross-model/feedback", e.HandleFeedback)
}
', [FusionMethod]).

generate_cross_model_endpoint(rust, Options, Code) :-
    (member(fusion_method(FusionMethod), Options) -> true ; FusionMethod = weighted_sum),
    format(atom(Code), '
// Cross-model federation routes for Rust/Axum
use axum::{extract::{State, Json}, routing::{get, post, put}, Router};
use std::sync::Arc;

pub async fn handle_cross_model_query(
    State(engine): State<Arc<CrossModelEngine>>,
    Json(req): Json<QueryRequest>,
) -> Json<CrossModelResponse> {
    Json(engine.federated_query(&req.query_text, req.top_k).await)
}

pub async fn handle_pools(State(engine): State<Arc<CrossModelEngine>>) -> Json<PoolsResponse> {
    Json(engine.get_pools().await)
}

pub async fn handle_weights(State(engine): State<Arc<CrossModelEngine>>) -> Json<WeightsResponse> {
    Json(engine.get_weights().await)
}

pub async fn handle_feedback(
    State(engine): State<Arc<CrossModelEngine>>,
    Json(req): Json<FeedbackRequest>,
) -> Json<StatusResponse> {
    engine.update_weights(&req.chosen_answer, &req.pool_rankings).await;
    Json(StatusResponse { status: "updated".to_string() })
}

pub fn cross_model_routes(engine: Arc<CrossModelEngine>) -> Router {
    Router::new()
        .route("/kg/cross-model", post(handle_cross_model_query))
        .route("/kg/cross-model/pools", get(handle_pools))
        .route("/kg/cross-model/weights", get(handle_weights))
        .route("/kg/cross-model/weights", put(handle_weights))
        .route("/kg/cross-model/feedback", post(handle_feedback))
        .with_state(engine)
}
', [FusionMethod]).
