# Skill: Infrastructure (Sub-Master)

Deployment, authentication, and networking capabilities for UnifyWeaver services.

## When to Use

- User asks "how do I deploy my service?"
- User needs Docker, Kubernetes, or cloud function deployment
- User wants authentication backends (JWT, sessions, OAuth)
- User needs HTTP/socket server or client generation

## Skill Hierarchy

```
skill_server_tools.md (parent)
└── skill_infrastructure.md (this file)
    ├── skill_deployment.md - Docker, K8s, cloud functions, secrets
    ├── skill_authentication.md - Auth backends, JWT, sessions
    └── skill_networking.md - HTTP/socket servers and clients
```

## Quick Start

### Deployment

```prolog
:- use_module('src/unifyweaver/glue/deployment_glue').

% Declare service
declare_service(my_api, [
    host('server.example.com'),
    port(8080),
    target(python),
    lifecycle(persistent)
]).

% Generate Dockerfile
generate_dockerfile(my_api, [], Dockerfile).

% Generate Kubernetes deployment
generate_k8s_deployment(my_api, [], Manifest).

% Generate AWS Lambda function
generate_lambda_function(my_func, [], Package).
```

### Authentication

```prolog
:- use_module('src/unifyweaver/glue/auth_backends').

% Configure auth for app
app(my_app, [
    auth([
        backend(text_file),
        password_hash(bcrypt),
        token_type(jwt),
        session_duration(86400)
    ])
]).

% Generate auth server
generate_auth_server(AppSpec, node, Files).
```

### Networking

```prolog
:- use_module('src/unifyweaver/glue/network_glue').

% Generate HTTP server
generate_http_server(python, Endpoints, [port(8080), cors(true)], Code).

% Generate HTTP client
generate_http_client(go, Services, [timeout(30)], Code).

% Generate socket server
generate_socket_server(go, 9000, [buffer_size(65536)], Code).
```

## Capabilities Overview

### Deployment

| Feature | Description |
|---------|-------------|
| SSH Deployment | Agent forwarding, service lifecycle |
| Docker | Dockerfile, docker-compose generation |
| Kubernetes | Deployments, services, ingress, ConfigMaps |
| Cloud Functions | AWS Lambda, Google Cloud Functions, Azure Functions |
| Secrets | Vault, AWS Secrets Manager, Azure Key Vault, GCP |
| Multi-Region | Geographic failover, traffic management |

### Authentication

| Backend | Description | Status |
|---------|-------------|--------|
| `mock` | Client-side for development | Implemented |
| `text_file` | Simple file storage | Implemented |
| `sqlite` | SQLite database | Proposed |
| `postgresql` | PostgreSQL database | Proposed |
| `mongodb` | MongoDB document store | Proposed |
| `ldap` | LDAP/Active Directory | Proposed |
| `oauth2` | OAuth2 providers | Proposed |

### Networking

| Target | HTTP Server | HTTP Client | Socket Server | Socket Client |
|--------|-------------|-------------|---------------|---------------|
| Python | Flask | requests | threading | socket |
| Go | net/http | net/http | net | net |
| Rust | Actix-web | reqwest | tokio | tokio |
| Bash | - | curl | - | - |

## Child Skills

- `skill_deployment.md` - Docker, Kubernetes, cloud functions, secrets management
- `skill_authentication.md` - Auth backends, password hashing, token types
- `skill_networking.md` - HTTP/socket server and client generation

## Related

**Parent Skill:**
- `skill_server_tools.md` - Backend services master

**Sibling Skills:**
- `skill_web_frameworks.md` - REST API frameworks
- `skill_ipc.md` - Inter-process communication

**Code:**
- `src/unifyweaver/glue/deployment_glue.pl`
- `src/unifyweaver/glue/auth_backends.pl`
- `src/unifyweaver/glue/network_glue.pl`
