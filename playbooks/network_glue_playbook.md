# Playbook: Network Communication (HTTP/Sockets)

## Audience
This playbook demonstrates network_glue for exposing predicates as REST endpoints and calling remote services.

## Overview
The `network_glue` module generates:
- HTTP servers (Go, Python, Rust) exposing predicates as REST APIs
- HTTP clients (Go, Python, Bash) calling remote predicates
- Socket-based communication for streaming
- Service registry for discovery

## When to Use

✅ **Use network_glue when:**
- Exposing Prolog predicates as REST APIs
- Calling remote UnifyWeaver services
- Need low-latency socket communication
- Building microservices architecture

## Agent Inputs

1. **Glue Module** – `src/unifyweaver/glue/network_glue.pl`

## Example Usage

### HTTP Server Generation

```prolog
:- use_module('src/unifyweaver/glue/network_glue').

% Define endpoints
Endpoints = [
    endpoint('/api/users', get_users/1, [method(get)]),
    endpoint('/api/user/:id', get_user/2, [method(get)])
].

% Generate Go HTTP server
?- generate_go_http_server(Endpoints, [port(8080)], Code).
```

### HTTP Client Generation

```prolog
% Register remote service
:- register_service(users_api, 'http://localhost:8080', []).

% Generate Python client
Services = [service(users_api, [get_users/1, get_user/2])].
?- generate_python_http_client(Services, [], Code).
```

### Socket Communication

```prolog
% Generate socket server (port 9000)
?- generate_socket_server(python, 9000, [], ServerCode).

% Generate socket client
?- generate_socket_client(python, 'localhost:9000', [], ClientCode).
```

## Key Features

- **Multi-language support**: Go, Python, Rust, Bash
- **RESTful API generation**: Automatic endpoint creation
- **Service discovery**: Registry for remote services
- **Socket streaming**: Low-latency communication
- **Network pipelines**: Chain remote services

## See Also

- `playbooks/http_source_playbook.md` - HTTP data sources
- `playbooks/deployment_glue_playbook.md` - Service deployment

## Summary

**Key Concepts:**
- ✅ Expose predicates as REST APIs
- ✅ Call remote UnifyWeaver services
- ✅ Socket-based streaming
- ✅ Service registry and discovery
