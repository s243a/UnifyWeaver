<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 11: HTTP Services

## Overview

For distributed systems, HTTP provides a standard communication protocol. This chapter covers:

- Service registry
- Go HTTP servers (net/http)
- Python Flask servers
- Rust Actix-web servers
- Consistent API format

## The Network Glue Module

Located at `src/unifyweaver/glue/network_glue.pl`:

```prolog
:- module(network_glue, [
    % Service registry
    register_service/3,
    service/2,
    service_options/2,
    unregister_service/1,
    endpoint_url/3,

    % HTTP server generation
    generate_http_server/4,
    generate_go_http_server/3,
    generate_python_http_server/3,
    generate_rust_http_server/3,

    % HTTP client generation
    generate_http_client/4,
    generate_go_http_client/3,
    generate_python_http_client/3,
    generate_bash_http_client/3
]).
```

## Service Registry

### Registering Services

```prolog
register_service(Name, URL, Options).
```

Example:
```prolog
% Register an ML service
register_service(ml_api, 'http://ml.example.com:8080', [
    timeout(60),
    retries(3),
    format(json)
]).

% Register a local service
register_service(transform, 'http://localhost:5000', [
    timeout(30)
]).
```

### Querying Services

```prolog
?- service(ml_api, URL).
URL = 'http://ml.example.com:8080'.

?- service_options(ml_api, Opts).
Opts = [timeout(60), retries(3), format(json)].
```

### Building Endpoint URLs

```prolog
?- endpoint_url(ml_api, '/predict', URL).
URL = 'http://ml.example.com:8080/predict'.
```

## Consistent API Format

All generated HTTP services use the same JSON schema:

### Request Format
```json
{
    "data": <any>
}
```

### Success Response
```json
{
    "success": true,
    "data": <result>
}
```

### Error Response
```json
{
    "success": false,
    "error": "<message>"
}
```

This consistency enables:
- Generic client code
- Easy debugging
- Interchangeable implementations

## Go HTTP Server Generation

```prolog
generate_go_http_server(
    [
        endpoint('/process', process_handler, []),
        endpoint('/analyze', analyze_handler, [methods(['POST'])])
    ],
    [port(8080), cors(true)],
    Code
).
```

Generated:

```go
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

// Handler for /process
func _processHandler(w http.ResponseWriter, r *http.Request) {
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
    result, err := process_handler(req.Data)
    if err != nil {
        json.NewEncoder(w).Encode(Response{Success: false, Error: err.Error()})
        return
    }

    json.NewEncoder(w).Encode(Response{Success: true, Data: result})
}

// Handler for /analyze
func _analyzeHandler(w http.ResponseWriter, r *http.Request) {
    // Similar structure...
}

func main() {
    port := "8080"
    if p := os.Getenv("PORT"); p != "" {
        port = p
    }

    // Register handlers
    http.HandleFunc("/process", corsMiddleware(_processHandler))
    http.HandleFunc("/analyze", corsMiddleware(_analyzeHandler))

    log.Printf("Server starting on port %s", port)
    log.Fatal(http.ListenAndServe(":"+port, nil))
}
```

## Python Flask Server Generation

```prolog
generate_python_http_server(
    [
        endpoint('/api/transform', transform, [methods(['POST', 'GET'])])
    ],
    [port(5000), cors(true)],
    Code
).
```

Generated:

```python
#!/usr/bin/env python3
"""
Generated HTTP Server
"""

from flask import Flask, request, jsonify
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/api/transform", methods=["POST", "GET"])
def _api_transform_route():
    try:
        data = request.get_json() if request.is_json else {}
        result = transform(data.get("data"))
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({"success": False, "error": str(error)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
```

## Rust Actix-web Server Generation

```prolog
generate_rust_http_server(
    [
        endpoint('/api/compute', compute, [])
    ],
    [port(9000)],
    Code
).
```

Generated:

```rust
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

async fn _api_compute_handler(req: web::Json<Request>) -> impl Responder {
    match compute(req.data.clone()) {
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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let port: u16 = env::var("PORT")
        .unwrap_or_else(|_| "9000".to_string())
        .parse()
        .unwrap();

    println!("Server starting on port {}", port);

    HttpServer::new(|| {
        let cors = Cors::permissive();

        App::new()
            .wrap(cors)
            .route("/api/compute", web::post().to(_api_compute_handler))
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
```

## HTTP Client Generation

### Go HTTP Client

```prolog
generate_go_http_client(
    [
        service_def(ml_api, 'http://ml:8080', ['/predict', '/classify'])
    ],
    [timeout(60)],
    Code
).
```

Generated:

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type Request struct {
    Data interface{} `json:"data"`
}

type Response struct {
    Success bool            `json:"success"`
    Data    json.RawMessage `json:"data"`
    Error   string          `json:"error"`
}

var client = &http.Client{
    Timeout: 60 * time.Second,
}

func callService(url string, data interface{}) (json.RawMessage, error) {
    reqBody, _ := json.Marshal(Request{Data: data})

    resp, err := client.Post(url, "application/json", bytes.NewReader(reqBody))
    if err != nil {
        return nil, fmt.Errorf("request error: %w", err)
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)

    var result Response
    json.Unmarshal(body, &result)

    if !result.Success {
        return nil, fmt.Errorf("service error: %s", result.Error)
    }

    return result.Data, nil
}

// Call ml_api/predict
func ml_api__predict(data interface{}) (json.RawMessage, error) {
    return callService("http://ml:8080/predict", data)
}

// Call ml_api/classify
func ml_api__classify(data interface{}) (json.RawMessage, error) {
    return callService("http://ml:8080/classify", data)
}
```

### Python HTTP Client

```prolog
generate_python_http_client(
    [
        service_def(api, 'http://localhost:5000', ['/users', '/orders'])
    ],
    [timeout(30)],
    Code
).
```

Generated:

```python
#!/usr/bin/env python3
"""
Generated HTTP Client
"""

import requests
from typing import Any

TIMEOUT = 30

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

def api__users(data: Any = None) -> Any:
    """Call api/users"""
    return call_service("http://localhost:5000/users", data)

def api__orders(data: Any = None) -> Any:
    """Call api/orders"""
    return call_service("http://localhost:5000/orders", data)
```

### Bash HTTP Client

```prolog
generate_bash_http_client(
    [
        service_def(service, 'http://localhost:8080', ['/health', '/data'])
    ],
    [timeout(30)],
    Code
).
```

Generated:

```bash
#!/bin/bash
# Generated HTTP Client

TIMEOUT=30

call_service() {
    local url="$1"
    local data="$2"

    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        --max-time "$TIMEOUT" \
        -d "{\"data\": $data}" \
        "$url")

    success=$(echo "$response" | jq -r '.success')

    if [ "$success" != "true" ]; then
        error=$(echo "$response" | jq -r '.error')
        echo "Error: $error" >&2
        return 1
    fi

    echo "$response" | jq '.data'
}

# Call service/health
service__health() {
    call_service "http://localhost:8080/health" "${1:-null}"
}

# Call service/data
service__data() {
    call_service "http://localhost:8080/data" "${1:-null}"
}
```

## Server Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `port(P)` | Integer | 8080 | Listen port |
| `cors(B)` | true/false | true | Enable CORS |
| `host(H)` | String | "0.0.0.0" | Bind address |

## Client Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `timeout(S)` | Integer | 30 | Request timeout |
| `retries(N)` | Integer | 0 | Retry count |
| `format(F)` | json | json | Request format |

## Chapter Summary

- **Service registry** tracks remote services
- **Go servers** use net/http with CORS
- **Python servers** use Flask
- **Rust servers** use Actix-web
- **Consistent API** enables interoperability
- **Client generation** for Go, Python, Bash

## Next Steps

In Chapter 12, we'll explore distributed pipelines:
- Socket communication
- Network pipeline orchestration
- Mixed local/remote steps

## Exercises

1. **Go server**: Generate a Go server with /login and /logout endpoints.

2. **Python client**: Generate a Python client for an ML service at http://ml:8080.

3. **Full stack**: Generate both server (Go) and client (Python) for a prediction service.

4. **Testing**: Write curl commands to test the generated API format.

## Code Examples

See `examples/05-distributed/` for:
- `server.go` - Go HTTP server
- `server.py` - Python Flask server
- `client.py` - Python client
- `client.sh` - Bash client

---

## Navigation

**←** [Previous: Chapter 10: Native Binary Orchestration](10_native_orchestration) | [📖 Book 7: Cross-Target Glue](./) | [Next: Chapter 12: Distributed Pipelines →](12_distributed_pipelines)
