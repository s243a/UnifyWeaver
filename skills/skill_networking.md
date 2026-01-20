# Skill: Networking

Generate HTTP servers, HTTP clients, and socket-based communication code for multiple languages.

## When to Use

- User asks "how do I create an HTTP server?"
- User needs to call remote services from generated code
- User wants low-latency socket communication
- User needs network pipelines with local and remote steps

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/network_glue').

% Generate HTTP server
generate_http_server(python, Endpoints, [port(8080), cors(true)], Code).

% Generate HTTP client
generate_http_client(go, Services, [timeout(30)], Code).

% Generate socket server
generate_socket_server(go, 9000, [], Code).
```

## Service Registry

### Register Service

```prolog
register_service(Name, URL, Options).
```

**Options:**
- `timeout(Seconds)` - Request timeout
- `retries(N)` - Retry count
- `format(json|tsv)` - Data format
- `auth(bearer(Token)|basic(User,Pass))` - Authentication

**Example:**
```prolog
register_service(ml_service, 'http://ml.example.com:8080', [
    timeout(60),
    retries(3),
    auth(bearer('my-token'))
]).
```

### Query Services

```prolog
service(Name, URL).
service_options(Name, Options).
endpoint_url(Service, Endpoint, FullURL).
```

### Unregister

```prolog
unregister_service(Name).
```

## HTTP Server Generation

### Supported Targets

| Target | Framework | Features |
|--------|-----------|----------|
| `python` | Flask | CORS, JSON, error handling |
| `go` | net/http | CORS middleware, JSON |
| `rust` | Actix-web | Async, CORS, typed responses |

### Generate Server

```prolog
generate_http_server(Target, Endpoints, Options, Code).
```

**Endpoints Format:**
```prolog
endpoint(Path, Handler, EndpointOptions)
```

**Options:**
- `port(Port)` - Server port (default: 8080)
- `cors(Bool)` - Enable CORS (default: true)

### Python Server Example

```prolog
generate_python_http_server([
    endpoint('/api/process', process_data, [methods(['POST'])])
], [port(8080)], Code).
```

**Output:**
```python
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/api/process", methods=["POST"])
def process_data_route():
    try:
        data = request.get_json() if request.is_json else {}
        result = process_data(data.get("data"))
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
```

### Go Server Example

```prolog
generate_go_http_server(Endpoints, [port(8080)], Code).
```

Generates:
- Request/Response types
- CORS middleware
- Handler functions
- Route registration

### Rust Server Example

```prolog
generate_rust_http_server(Endpoints, [port(8080)], Code).
```

Generates Actix-web server with:
- Serde types
- CORS middleware
- Async handlers

## HTTP Client Generation

### Supported Targets

| Target | Library | Features |
|--------|---------|----------|
| `python` | requests | Timeout, error handling |
| `go` | net/http | Typed, configurable |
| `bash` | curl | Simple scripting |

### Generate Client

```prolog
generate_http_client(Target, Services, Options, Code).
```

**Services Format:**
```prolog
service_def(Name, BaseURL, Endpoints)
```

**Options:**
- `timeout(Seconds)` - Request timeout (default: 30)

### Python Client Example

```prolog
generate_python_http_client([
    service_def(ml, 'http://ml.example.com', ['/predict', '/train'])
], [timeout(30)], Code).
```

**Output:**
```python
import requests
from typing import Any

TIMEOUT = 30

def call_service(url: str, data: Any = None) -> Any:
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

def ml_predict(data: Any = None) -> Any:
    """Call ml/predict"""
    return call_service("http://ml.example.com/predict", data)

def ml_train(data: Any = None) -> Any:
    """Call ml/train"""
    return call_service("http://ml.example.com/train", data)
```

### Bash Client Example

```prolog
generate_bash_http_client(Services, [timeout(30)], Code).
```

Generates shell functions using curl and jq.

## Socket Communication

### Socket Server

```prolog
generate_socket_server(Target, Port, Options, Code).
```

**Options:**
- `format(json|tsv)` - Data format
- `buffer_size(Bytes)` - Buffer size (default: 65536)

**Go Example:**
```prolog
generate_socket_server(go, 9000, [buffer_size(65536)], Code).
```

Generates TCP server with:
- Connection handling
- Line-based protocol
- Buffered I/O

### Socket Client

```prolog
generate_socket_client(Target, Host, Options, Code).
```

**Options:**
- `port(Port)` - Server port
- `buffer_size(Bytes)` - Buffer size

**Python Example:**
```prolog
generate_socket_client(python, 'localhost', [port(9000)], Code).
```

**Output:**
```python
class SocketClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def send(self, data: bytes) -> bytes:
        self.sock.sendall(data + b"\n")
        return self.sock.recv(BUFFER_SIZE)

    def close(self):
        self.sock.close()
```

## Network Pipeline

### Generate Pipeline

```prolog
generate_network_pipeline(Steps, Options, Code).
```

**Steps Format:**
```prolog
step(Name, local, Script)    % Local processing
step(Name, remote, URL)      % Remote HTTP call
```

**Options:**
- `language(Lang)` - Output language (python, go, bash)

**Example:**
```prolog
generate_network_pipeline([
    step(preprocess, local, 'clean_data(data)'),
    step(predict, remote, 'http://ml.example.com/predict'),
    step(postprocess, local, 'format_result(result)')
], [language(python)], Code).
```

## KG Topology Endpoints

### Distributed Query Endpoints

```prolog
generate_kg_query_endpoint(Target, Options, Code).
```

Generates endpoints for distributed knowledge graph queries:
- `POST /kg/query` - Handle distributed query
- `POST /kg/register` - Register node with discovery
- `GET /kg/health` - Health check

### Federation Endpoints

```prolog
generate_federation_endpoint(Target, Options, Code).
```

Generates endpoints for federated queries:
- `POST /kg/federated` - Handle incoming federated query
- `POST /kg/federate` - Initiate federation
- `GET /kg/federation/stats` - Federation statistics

### Cross-Model Endpoints

```prolog
generate_cross_model_endpoint(Target, Options, Code).
```

## Related

**Parent Skill:**
- `skill_infrastructure.md` - Infrastructure sub-master

**Sibling Skills:**
- `skill_deployment.md` - Service deployment
- `skill_authentication.md` - Auth backends
- `skill_web_frameworks.md` - REST API frameworks

**Code:**
- `src/unifyweaver/glue/network_glue.pl`
