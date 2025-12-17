# Playbook: Network Communication (HTTP/Sockets)

## Audience
This playbook is a high-level guide for coding agents. It demonstrates network_glue for exposing predicates as REST endpoints and calling remote services.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "network_glue" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use network glue"


## Workflow Overview
Use network_glue for network communication:
1. Register remote services in the service registry
2. Generate HTTP servers exposing predicates as REST APIs
3. Generate HTTP clients for calling remote predicates
4. Generate socket servers/clients for low-latency streaming

## Agent Inputs
Reference the following artifacts:
1. **Glue Module** – `src/unifyweaver/glue/network_glue.pl` contains network communication predicates
2. **Module Documentation** – See module header for API details

## Key Features

- HTTP server generation (Go, Python, Rust)
- HTTP client generation (Go, Python, Bash)
- Socket-based communication for streaming
- Service registry for discovery and routing

## Execution Guidance

Consult the module for predicate usage:

```prolog
:- use_module('src/unifyweaver/glue/network_glue').

% Register remote service
:- register_service(users_api, 'http://localhost:8080', []).

% Generate HTTP server
Endpoints = [endpoint('/api/users', get_users/1, [method(get)])].
?- generate_go_http_server(Endpoints, [port(8080)], Code).

% Generate HTTP client
Services = [service(users_api, [get_users/1])].
?- generate_python_http_client(Services, [], Code).

% Generate socket server
?- generate_socket_server(python, 9000, [], ServerCode).
```

## Expected Outcome
- HTTP servers expose predicates as REST APIs
- HTTP clients successfully call remote services
- Socket communication established for streaming
- Service registry populated correctly

## Citations
[1] src/unifyweaver/glue/network_glue.pl
