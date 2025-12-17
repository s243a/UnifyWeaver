# Client-Server Architecture: Technical Specification

## Overview

This document specifies the syntax, semantics, and protocols for UnifyWeaver's client-server architecture.

## 1. Core Concepts

### 1.1 Service

A **service** is a named, reusable unit that:
- Receives requests
- Performs computation (possibly stateful)
- Returns responses

```prolog
service(ServiceName, HandlerSpec).
service(ServiceName, Options, HandlerSpec).
```

### 1.2 Client

A **client** is code that:
- Sends requests to services
- Receives responses
- Handles errors/timeouts

Clients can be:
- Pipeline stages (`call_service/4`)
- Standalone programs
- Other services

### 1.3 Channel

A **channel** is the communication path between client and server:
- In-process: Direct function calls
- Cross-process: Pipes, sockets
- Network: HTTP, WebSocket, TCP

## 2. Service Definition Syntax

### 2.1 Basic Service

```prolog
% Minimal service definition
service(echo, [
    receive(Message),
    respond(Message)
]).

% Service with handler predicate
service(user_lookup, [
    receive(user_id),
    lookup_user/1,
    respond(user_record)
]).

% Service with inline transformation
service(double, [
    receive(Number),
    transform(N, N * 2),
    respond(Result)
]).
```

### 2.2 Service Options

```prolog
service(ServiceName, Options, HandlerSpec).

% Available options:
Options = [
    % Transport configuration
    transport(in_process),          % Default: direct function call
    transport(unix_socket(Path)),   % Unix domain socket
    transport(tcp(Host, Port)),     % TCP socket
    transport(http(Endpoint)),      % HTTP endpoint

    % Protocol configuration
    protocol(jsonl),                % Default: JSON Lines
    protocol(messagepack),          % Binary MessagePack
    protocol(protobuf(Schema)),     % Protocol Buffers

    % Behavior configuration
    stateful(true),                 % Service maintains state between requests
    timeout(Ms),                    % Default timeout for requests
    max_concurrent(N),              % Max concurrent requests

    % Error handling
    on_error(respond_error),        % Default error behavior
    on_error(log_and_continue),
    on_error(crash)
].
```

### 2.3 Handler Specification

The handler spec is a list of operations:

```prolog
HandlerSpec = [Operation1, Operation2, ...].

% Operations:
receive(Variable)              % Bind request to Variable
respond(Value)                 % Send response
respond_error(Error)           % Send error response

Predicate/Arity                % Execute predicate (pipeline stage)
transform(In, Out)             % Inline transformation
state_get(Key, Value)          % Read from service state
state_put(Key, Value)          % Write to service state
state_modify(Key, Func)        % Modify service state

call_service(Name, Req, Resp)  % Call another service
```

### 2.4 Stateful Services

```prolog
% Counter service with state
service(counter, [stateful(true)], [
    receive(Operation),
    route_by(operation, [
        (increment, [
            state_modify(count, succ),
            state_get(count, Value),
            respond(Value)
        ]),
        (decrement, [
            state_modify(count, pred),
            state_get(count, Value),
            respond(Value)
        ]),
        (get, [
            state_get(count, Value),
            respond(Value)
        ]),
        (reset, [
            state_put(count, 0),
            respond(ok)
        ])
    ])
]).

% Cache service with state
service(cache, [stateful(true)], [
    receive(Request),
    route_by(op, [
        (get, [
            state_get(Request.key, Value),
            respond(Value)
        ]),
        (set, [
            state_put(Request.key, Request.value),
            respond(ok)
        ]),
        (delete, [
            state_delete(Request.key),
            respond(ok)
        ])
    ])
]).
```

## 3. Client Syntax

### 3.1 Pipeline Integration

```prolog
% call_service(ServiceName, RequestExpr, ResponseVar)
pipeline([
    parse/1,
    call_service(user_lookup, record.user_id, user_data),
    merge_user_data/1,
    output/1
]).

% With options
pipeline([
    parse/1,
    call_service(slow_service, request, response, [
        timeout(5000),
        retry(3),
        fallback(default_value)
    ]),
    output/1
]).
```

### 3.2 Standalone Client

```prolog
% Define a client for a service
client(user_client, user_lookup, [
    transport(http('http://localhost:8080/users'))
]).

% Use client in code
query_user(UserId, UserData) :-
    client_call(user_client, UserId, UserData).
```

### 3.3 Async Client

```prolog
% Non-blocking service call
pipeline([
    parse/1,
    call_service_async(slow_service, request, FutureRef),
    % ... do other work ...
    await_service(FutureRef, response),
    output/1
]).
```

## 4. Protocol Specification

### 4.1 Request Format (JSONL)

```json
{"__type": "request", "__id": "uuid-123", "__service": "user_lookup", "payload": {"user_id": 42}}
```

Fields:
- `__type`: Always "request"
- `__id`: Unique request identifier (for correlation)
- `__service`: Target service name
- `payload`: Request data

### 4.2 Response Format (JSONL)

Success:
```json
{"__type": "response", "__id": "uuid-123", "__status": "ok", "payload": {"user_id": 42, "name": "Alice"}}
```

Error:
```json
{"__type": "response", "__id": "uuid-123", "__status": "error", "error": {"code": "NOT_FOUND", "message": "User not found"}}
```

Fields:
- `__type`: Always "response"
- `__id`: Matching request ID
- `__status`: "ok" or "error"
- `payload`: Response data (on success)
- `error`: Error details (on failure)

### 4.3 Streaming Protocol

For services that return multiple responses:

```json
{"__type": "response", "__id": "uuid-123", "__status": "streaming", "__seq": 1, "payload": {...}}
{"__type": "response", "__id": "uuid-123", "__status": "streaming", "__seq": 2, "payload": {...}}
{"__type": "response", "__id": "uuid-123", "__status": "end", "__seq": 3}
```

## 5. Transport Specifications

### 5.1 In-Process Transport

```python
# Python implementation
class InProcessService:
    def __init__(self, handler_fn, stateful=False):
        self.handler = handler_fn
        self.state = {} if stateful else None

    def call(self, request):
        return self.handler(request, self.state)

# Generated code for call_service
def call_user_lookup(request):
    return _services['user_lookup'].call(request)
```

```go
// Go implementation
type InProcessService struct {
    handler func(request Record, state *State) Record
    state   *State
}

func (s *InProcessService) Call(request Record) Record {
    return s.handler(request, s.state)
}
```

### 5.2 Unix Socket Transport

```python
# Server side
import socket

def run_service(service_name, handler, socket_path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen(1)

    while True:
        conn, _ = sock.accept()
        for line in conn.makefile():
            request = json.loads(line)
            response = handler(request['payload'])
            conn.send(json.dumps({
                '__type': 'response',
                '__id': request['__id'],
                '__status': 'ok',
                'payload': response
            }).encode() + b'\n')
```

### 5.3 HTTP Transport

```python
# Using Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/service/<name>', methods=['POST'])
def handle_service(name):
    handler = _services.get(name)
    if not handler:
        return jsonify({'error': 'Unknown service'}), 404

    req = request.json
    response = handler(req['payload'])
    return jsonify({
        '__type': 'response',
        '__id': req['__id'],
        '__status': 'ok',
        'payload': response
    })
```

## 6. Validation

### 6.1 Service Validation

```prolog
% In pipeline_validation.pl

% Service definition validation
is_valid_service(service(Name, HandlerSpec)) :-
    atom(Name),
    is_valid_handler_spec(HandlerSpec).

is_valid_service(service(Name, Options, HandlerSpec)) :-
    atom(Name),
    is_list(Options),
    validate_service_options(Options),
    is_valid_handler_spec(HandlerSpec).

% Handler spec validation
is_valid_handler_spec([]).
is_valid_handler_spec([Op|Rest]) :-
    is_valid_service_operation(Op),
    is_valid_handler_spec(Rest).

% Operation validation
is_valid_service_operation(receive(Var)) :- var(Var).
is_valid_service_operation(respond(_)).
is_valid_service_operation(respond_error(_)).
is_valid_service_operation(Pred/Arity) :- atom(Pred), integer(Arity).
is_valid_service_operation(transform(_, _)).
is_valid_service_operation(state_get(Key, _)) :- atom(Key).
is_valid_service_operation(state_put(Key, _)) :- atom(Key).
is_valid_service_operation(call_service(Name, _, _)) :- atom(Name).
```

### 6.2 Client Validation

```prolog
% call_service as pipeline stage
is_valid_stage(call_service(Name, _, _)) :-
    atom(Name).

is_valid_stage(call_service(Name, _, _, Options)) :-
    atom(Name),
    is_list(Options),
    validate_client_options(Options).

% Client options
validate_client_options([]).
validate_client_options([Opt|Rest]) :-
    is_valid_client_option(Opt),
    validate_client_options(Rest).

is_valid_client_option(timeout(Ms)) :- integer(Ms), Ms > 0.
is_valid_client_option(retry(N)) :- integer(N), N >= 0.
is_valid_client_option(fallback(_)).
is_valid_client_option(transport(_)).
```

## 7. Error Handling

### 7.1 Error Types

```prolog
% Service-side errors
service_error(not_found, "Resource not found").
service_error(invalid_request, "Invalid request format").
service_error(internal_error, "Internal service error").
service_error(unauthorized, "Authentication required").

% Client-side errors
client_error(timeout, "Request timed out").
client_error(connection_failed, "Could not connect to service").
client_error(service_unavailable, "Service is unavailable").
```

### 7.2 Error Handling in Services

```prolog
service(safe_lookup, [
    receive(Request),
    try_catch(
        [
            validate_request/1,
            lookup_data/1,
            respond(Result)
        ],
        respond_error(Error)
    )
]).
```

### 7.3 Error Handling in Clients

```prolog
pipeline([
    parse/1,
    try_catch(
        call_service(risky_service, request, response),
        [
            on_error(timeout, use_cached/1),
            on_error(not_found, create_default/1),
            on_error(_, log_and_skip/1)
        ]
    ),
    output/1
]).
```

## 8. Service Discovery (Future)

### 8.1 Local Registry

```prolog
% Register service at compile time
:- register_service(user_lookup, [
    transport(in_process)
]).

% Lookup service
get_service(Name, ServiceRef) :-
    service_registry(Name, ServiceRef).
```

### 8.2 External Registry (Future)

```prolog
% Connect to service registry
:- service_registry(consul, 'http://consul:8500').

% Services registered externally
service(user_lookup, [
    transport(discover(consul, 'user-service'))
]).
```

## 9. Metrics and Observability

### 9.1 Built-in Metrics

Services automatically track:
- Request count
- Response time (p50, p95, p99)
- Error rate
- Active connections

### 9.2 Integration with Audit

```prolog
service(audited_lookup, [
    audit([include_timing(true)]),  % Track in audit log
    receive(Request),
    lookup/1,
    respond(Result)
]).
```

## 10. Type Definitions

### 10.1 Request/Response Types

```prolog
% Optional type annotations for documentation and validation
service(typed_lookup, [
    types([
        request: user_id(integer),
        response: user_record([
            id: integer,
            name: string,
            email: string
        ])
    ]),
    receive(UserId),
    lookup_user/1,
    respond(UserRecord)
]).
```

### 10.2 Schema Validation

```prolog
% Validate requests against schema
service(validated_service, [
    validate_schema(true),
    types([...]),
    receive(Request),
    process/1,
    respond(Response)
]).
```

---

*This document specifies the technical details. See `CLIENT_SERVER_IMPLEMENTATION.md` for the implementation roadmap.*
