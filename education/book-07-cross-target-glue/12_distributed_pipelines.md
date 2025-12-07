<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 12: Distributed Pipelines

## Overview

Real-world systems often span multiple machines. This chapter covers:

- Socket communication
- Network pipeline orchestration
- Mixed local/remote steps
- Error handling in distributed systems

## Socket Communication

For low-latency, high-throughput communication.

### Go Socket Server

```prolog
generate_socket_server(go, 9000, [buffer_size(65536)], Code).
```

Generated:

```go
package main

import (
    "bufio"
    "fmt"
    "net"
    "os"
)

const bufferSize = 65536

func handleConnection(conn net.Conn) {
    defer conn.Close()

    reader := bufio.NewReaderSize(conn, bufferSize)
    writer := bufio.NewWriterSize(conn, bufferSize)

    for {
        line, err := reader.ReadString('\n')
        if err != nil {
            return
        }

        // Process
        result := process(line)

        writer.WriteString(result)
        writer.Flush()
    }
}

func main() {
    listener, err := net.Listen("tcp", ":9000")
    if err != nil {
        fmt.Fprintln(os.Stderr, "Failed to listen:", err)
        os.Exit(1)
    }
    defer listener.Close()

    fmt.Println("Server listening on :9000")

    for {
        conn, err := listener.Accept()
        if err != nil {
            fmt.Fprintln(os.Stderr, "Accept error:", err)
            continue
        }

        go handleConnection(conn)
    }
}
```

### Go Socket Client

```prolog
generate_socket_client(go, 'localhost:9000', [timeout(10)], Code).
```

Generated:

```go
package main

import (
    "bufio"
    "net"
    "time"
)

type SocketClient struct {
    conn   net.Conn
    reader *bufio.Reader
    writer *bufio.Writer
}

func NewClient(addr string) (*SocketClient, error) {
    conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
    if err != nil {
        return nil, err
    }

    return &SocketClient{
        conn:   conn,
        reader: bufio.NewReader(conn),
        writer: bufio.NewWriter(conn),
    }, nil
}

func (c *SocketClient) Send(data string) (string, error) {
    c.writer.WriteString(data + "\n")
    c.writer.Flush()

    response, err := c.reader.ReadString('\n')
    return response, err
}

func (c *SocketClient) Close() {
    c.conn.Close()
}
```

### Python Socket Server

```prolog
generate_socket_server(python, 9000, [buffer_size(65536)], Code).
```

Generated:

```python
#!/usr/bin/env python3
import socket
import threading

BUFFER_SIZE = 65536

def handle_client(conn, addr):
    print(f"Connection from {addr}")
    try:
        while True:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                break

            # Process
            result = process(data.decode())

            conn.sendall(result.encode())
    finally:
        conn.close()

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 9000))
    server.listen(5)

    print("Server listening on :9000")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    main()
```

### Python Socket Client

```prolog
generate_socket_client(python, 'localhost:9000', [timeout(10)], Code).
```

Generated:

```python
#!/usr/bin/env python3
import socket

class SocketClient:
    def __init__(self, host, port, timeout=10):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((host, port))

    def send(self, data):
        self.sock.sendall((data + '\n').encode())
        response = self.sock.recv(65536)
        return response.decode()

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

## Network Pipeline Generation

### Mixed Local/Remote Pipeline

```prolog
generate_network_pipeline(
    [
        step(fetch, remote, 'http://data-api:8080/fetch', []),
        step(process, local, './process', []),
        step(store, remote, 'http://db-api:8080/store', [])
    ],
    [],
    Code
).
```

### Python Network Pipeline

Generated:

```python
#!/usr/bin/env python3
"""
Generated Network Pipeline
"""

import requests
from typing import Any

TIMEOUT = 30

def call_remote(url: str, data: Any) -> Any:
    """Call a remote service."""
    response = requests.post(
        url,
        json={"data": data},
        timeout=TIMEOUT
    )
    result = response.json()
    if not result.get("success"):
        raise Exception(result.get("error"))
    return result.get("data")

def step_fetch(data):
    """Remote step: fetch"""
    return call_remote("http://data-api:8080/fetch", data)

def step_process(data):
    """Local step: process"""
    # Your local processing logic here
    return process(data)

def step_store(data):
    """Remote step: store"""
    return call_remote("http://db-api:8080/store", data)

def run_pipeline(initial_data):
    """Execute the complete pipeline."""
    data = initial_data

    # Step 1: fetch (remote)
    data = step_fetch(data)

    # Step 2: process (local)
    data = step_process(data)

    # Step 3: store (remote)
    data = step_store(data)

    return data
```

### Bash Network Pipeline

```prolog
generate_network_pipeline(
    [
        step(get, remote, 'http://api:8080/get', []),
        step(transform, local, 'jq .data', []),
        step(post, remote, 'http://api:8080/post', [])
    ],
    [target(bash)],
    Code
).
```

Generated:

```bash
#!/bin/bash
set -euo pipefail

TIMEOUT=30

call_remote() {
    local url="$1"
    local data="$2"

    curl -s -X POST \
        -H "Content-Type: application/json" \
        --max-time "$TIMEOUT" \
        -d "{\"data\": $data}" \
        "$url" | jq -r '.data'
}

run_pipeline() {
    local data="$1"

    # Step 1: get (remote)
    data=$(call_remote "http://api:8080/get" "$data")

    # Step 2: transform (local)
    data=$(echo "$data" | jq .data)

    # Step 3: post (remote)
    data=$(call_remote "http://api:8080/post" "$data")

    echo "$data"
}

run_pipeline "$1"
```

### Go Network Pipeline

```prolog
generate_network_pipeline(
    Steps,
    [target(go), async(true)],
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

var client = &http.Client{Timeout: 30 * time.Second}

func callRemote(url string, data interface{}) (json.RawMessage, error) {
    body, _ := json.Marshal(map[string]interface{}{"data": data})
    resp, err := client.Post(url, "application/json", bytes.NewReader(body))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    respBody, _ := io.ReadAll(resp.Body)
    var result struct {
        Success bool            `json:"success"`
        Data    json.RawMessage `json:"data"`
        Error   string          `json:"error"`
    }
    json.Unmarshal(respBody, &result)

    if !result.Success {
        return nil, fmt.Errorf(result.Error)
    }
    return result.Data, nil
}

func RunPipeline(initialData interface{}) (interface{}, error) {
    var data interface{} = initialData

    // Step 1: fetch (remote)
    fetchResult, err := callRemote("http://data-api:8080/fetch", data)
    if err != nil {
        return nil, fmt.Errorf("fetch failed: %w", err)
    }
    data = fetchResult

    // Step 2: process (local)
    data = process(data)

    // Step 3: store (remote)
    storeResult, err := callRemote("http://db-api:8080/store", data)
    if err != nil {
        return nil, fmt.Errorf("store failed: %w", err)
    }

    return storeResult, nil
}
```

## Error Handling Strategies

### Retry with Backoff

```prolog
generate_network_pipeline(
    Steps,
    [retry(3), backoff(exponential)],
    Code
).
```

Generated (Python):

```python
import time
from functools import wraps

def with_retry(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = backoff_factor ** attempt
                        print(f"Retry {attempt + 1} after {sleep_time}s...")
                        time.sleep(sleep_time)
            raise last_exception
        return wrapper
    return decorator

@with_retry(max_retries=3)
def step_fetch(data):
    return call_remote("http://data-api:8080/fetch", data)
```

### Circuit Breaker

```prolog
generate_network_pipeline(
    Steps,
    [circuit_breaker(failures(5), reset_after(60))],
    Code
).
```

Generated (Python):

```python
import time
from threading import Lock

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
        self.lock = Lock()

    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == 'open':
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = 'half-open'
                else:
                    raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            with self.lock:
                self.failure_count = 0
                self.state = 'closed'
            return result
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
            raise

circuit = CircuitBreaker()

def step_fetch_safe(data):
    return circuit.call(step_fetch, data)
```

### Fallback

```prolog
generate_network_pipeline(
    Steps,
    [fallback(step(fetch), cache_fetch)],
    Code
).
```

Generated:

```python
def step_fetch_with_fallback(data):
    try:
        return step_fetch(data)
    except Exception as e:
        print(f"Fetch failed, using cache: {e}")
        return cache_fetch(data)
```

## Monitoring Distributed Pipelines

### Logging

```prolog
generate_network_pipeline(
    Steps,
    [logging(verbose)],
    Code
).
```

Generated:

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def step_fetch(data):
    start = time.time()
    logger.info(f"fetch: starting with {len(str(data))} bytes")

    try:
        result = call_remote("http://data-api:8080/fetch", data)
        duration = time.time() - start
        logger.info(f"fetch: completed in {duration:.3f}s")
        return result
    except Exception as e:
        duration = time.time() - start
        logger.error(f"fetch: failed after {duration:.3f}s: {e}")
        raise
```

### Metrics

```prolog
generate_network_pipeline(
    Steps,
    [metrics(prometheus)],
    Code
).
```

Generated:

```python
from prometheus_client import Counter, Histogram

step_requests = Counter(
    'pipeline_step_requests_total',
    'Total requests per step',
    ['step']
)

step_latency = Histogram(
    'pipeline_step_latency_seconds',
    'Latency per step',
    ['step']
)

def step_fetch(data):
    step_requests.labels(step='fetch').inc()

    with step_latency.labels(step='fetch').time():
        return call_remote("http://data-api:8080/fetch", data)
```

## Chapter Summary

- **Socket communication** for low-latency streaming
- **Network pipelines** mix local and remote steps
- **Error handling** with retry, circuit breaker, fallback
- **Monitoring** with logging and metrics
- **Multiple targets** (Python, Bash, Go)

## Next Steps

In Chapter 13, we'll provide a complete API reference:
- All predicates documented
- All options explained
- Common patterns

## Exercises

1. **Socket server**: Generate a Go socket server that echoes messages.

2. **Distributed pipeline**: Create a pipeline with 2 remote and 1 local step.

3. **Error handling**: Add retry with exponential backoff to a network pipeline.

4. **Monitoring**: Add Prometheus metrics to a pipeline and visualize in Grafana.

## Code Examples

See `examples/05-distributed/` for:
- `socket_server.go` - Go socket server
- `socket_client.py` - Python client
- `network_pipeline.py` - Mixed pipeline
- `resilient_pipeline.py` - With error handling

---

## Navigation

**←** [Previous: Chapter 11: HTTP Services](11_http_services) | [📖 Book 7: Cross-Target Glue](./) | [Next: Chapter 13: API Reference →](13_api_reference)
