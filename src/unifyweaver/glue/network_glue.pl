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
    endpoint_url/3                  % endpoint_url(+Service, +Endpoint, -URL)
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
    format(atom(Code), '
@app.route("~w", methods=~w)
def ~w():
    try:
        data = request.get_json() if request.is_json else {}
        result = ~w(data.get("data"))
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400
', [Path, MethodsList, Handler, Handler]).

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
