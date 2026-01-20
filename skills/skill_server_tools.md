# Skill: Server Tools (Master)

Backend services, APIs, inter-process communication, and infrastructure management for UnifyWeaver applications.

## When to Use

- User asks "how do I create a backend API?"
- User needs to set up Flask, FastAPI, or Express server
- User wants to connect Prolog to Python services
- User needs deployment, authentication, or networking
- User asks about inter-process communication patterns

## Skill Hierarchy

```
skill_server_tools.md (this file)
├── skill_web_frameworks.md (sub-master) - Web API frameworks
│   ├── skill_flask_api.md - Flask REST APIs
│   ├── skill_fastapi.md - FastAPI with Pydantic
│   └── skill_express_api.md - Express.js routers
├── skill_ipc.md (sub-master) - Inter-Process Communication
│   ├── skill_pipe_communication.md - TSV/JSON pipe protocols
│   ├── skill_rpyc.md - Remote Python Call (network RPC)
│   └── skill_python_bridges.md - Cross-runtime Python embedding
└── skill_infrastructure.md (sub-master) - Deployment & Ops
    ├── skill_deployment.md - Docker, K8s, cloud functions
    ├── skill_authentication.md - Auth backends, JWT, sessions
    └── skill_networking.md - HTTP/socket servers & clients
```

## Quick Start

### Web API (Flask)

```prolog
:- use_module('src/unifyweaver/glue/flask_generator').

% Generate Flask app with handlers
generate_flask_app([fetch_tasks, create_task], [app_name('TaskAPI')], Code).
```

### Web API (FastAPI)

```prolog
:- use_module('src/unifyweaver/glue/fastapi_generator').

% Generate FastAPI with Pydantic models
generate_fastapi_app([list_items, create_item], [app_name('ItemAPI')], Code).
generate_pydantic_model(product, [field(name, string), field(price, number)], Model).
```

### Web API (Express)

```prolog
:- use_module('src/unifyweaver/glue/express_generator').

% Generate Express router with security
generate_express_router(python_api, [endpoints([math_endpoints])], Code).
generate_secure_router(my_api, SecureCode).
```

### Inter-Process Communication (Pipes)

```prolog
:- use_module('src/unifyweaver/glue/pipe_glue').

% Generate TSV pipe reader/writer
generate_pipe_writer(python, [name, age, score], [], WriterCode).
generate_pipe_reader(awk, [name, age, score], [], ReaderCode).
```

### Remote Python Call (RPyC)

```prolog
:- use_module('src/unifyweaver/glue/rpyc_glue').

% Connect to Python service
rpyc_connect('localhost', [port(18812), security(ssh)], Proxy).
rpyc_call(Proxy, numpy, mean, [[1,2,3,4,5]], Result).
```

### Python Bridges

```prolog
:- use_module('src/unifyweaver/glue/python_bridges_glue').

% Auto-select best bridge for platform
auto_select_bridge(RuntimeEnv, Preferences, Bridge).
generate_pythonnet_rpyc_client(Predicates, Code).
```

### Deployment

```prolog
:- use_module('src/unifyweaver/glue/deployment_glue').

% Declare and deploy service
declare_service(my_api, [host('server.example.com'), port(8080), target(python)]).
generate_docker_compose(my_project, [], ComposeYaml).
generate_k8s_deployment(my_api, [], Manifest).
```

### Authentication

```prolog
:- use_module('src/unifyweaver/glue/auth_backends').

% Configure auth for app
app(my_app, [
    auth([
        backend(text_file),
        password_hash(bcrypt),
        token_type(jwt)
    ])
]).
generate_auth_server(AppSpec, node, Files).
```

### Networking

```prolog
:- use_module('src/unifyweaver/glue/network_glue').

% Generate HTTP server/client
generate_http_server(go, Endpoints, [port(8080), cors(true)], ServerCode).
generate_http_client(python, Services, [timeout(30)], ClientCode).
```

## Capabilities Overview

### Web Frameworks

| Framework | Language | Features |
|-----------|----------|----------|
| Flask | Python | Routes, blueprints, CORS, auth handlers |
| FastAPI | Python | Pydantic models, async, OpenAPI, pagination |
| Express | Node.js | Routers, middleware, security integration |

### IPC Patterns

| Pattern | Use Case | Targets |
|---------|----------|---------|
| Pipe (TSV/JSON) | Unix pipeline integration | awk, python, bash, go, rust |
| RPyC | Network-based Python RPC | Python with SSH/SSL/unsecured |
| Python Bridges | Cross-runtime embedding | .NET, JVM, Rust, Ruby, FFI |

### Infrastructure

| Component | Capabilities |
|-----------|--------------|
| Deployment | SSH, Docker, K8s, AWS Lambda, GCF, Azure Functions |
| Secrets | Vault, AWS Secrets Manager, Azure Key Vault, GCP |
| Monitoring | Health checks, metrics, logging, alerting |
| Auth | Mock, text file, SQLite, PostgreSQL, MongoDB, LDAP, OAuth2 |

## Child Skills

For detailed usage of specific capabilities, see the child skills:

### Web Frameworks
- `skill_web_frameworks.md` - Overview of all web framework options
- `skill_flask_api.md` - Flask route and handler generation
- `skill_fastapi.md` - FastAPI with Pydantic models
- `skill_express_api.md` - Express.js router generation

### Inter-Process Communication
- `skill_ipc.md` - Overview of IPC patterns
- `skill_pipe_communication.md` - TSV/JSON pipe protocols
- `skill_rpyc.md` - Remote Python Call (RPyC)
- `skill_python_bridges.md` - Cross-runtime Python embedding

### Infrastructure
- `skill_infrastructure.md` - Overview of deployment and ops
- `skill_deployment.md` - Docker, Kubernetes, cloud functions
- `skill_authentication.md` - Auth backends and token management
- `skill_networking.md` - HTTP/socket server and client generation

## Related

**Sibling Master Skills:**
- `skill_gui_tools.md` - Frontend/GUI generation
- `skill_mindmap_tools.md` - Mindmap organization

**Code:**
- `src/unifyweaver/glue/flask_generator.pl`
- `src/unifyweaver/glue/fastapi_generator.pl`
- `src/unifyweaver/glue/express_generator.pl`
- `src/unifyweaver/glue/pipe_glue.pl`
- `src/unifyweaver/glue/rpyc_glue.pl`
- `src/unifyweaver/glue/python_bridges_glue.pl`
- `src/unifyweaver/glue/deployment_glue.pl`
- `src/unifyweaver/glue/auth_backends.pl`
- `src/unifyweaver/glue/network_glue.pl`
