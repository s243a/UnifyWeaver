# Proposal: UnifyWeaver HTTP CLI Server Generation

## Summary

Extend UnifyWeaver's source/binding/component architecture to declaratively generate HTTP CLI servers with authentication, WebSocket shell, and HTTPS support - replacing the current manually-written TypeScript implementation.

## Current State

The `http-server.ts` (1900+ lines) was manually written and includes:
- JWT authentication with role-based access control
- WebSocket shell with PTY support
- HTML interface with login and text mode
- Command execution with sandboxing
- HTTPS support (newly added)

This should be generated from a declarative Prolog specification.

## Proposed Architecture

### 1. Service Source Type

Extend the `source/3` directive to support service definitions:

```prolog
:- source(service, http_cli_server, [
    name('UnifyWeaver HTTP CLI Server'),
    port(3001),

    % Protocol
    https([
        cert_env('SSL_CERT'),
        key_env('SSL_KEY'),
        auto_redirect(true)
    ]),

    % Endpoints
    endpoints([
        endpoint(health, get, '/health', [public(true)]),
        endpoint(commands, get, '/commands', [roles([user, admin, shell])]),
        endpoint(exec, post, '/exec', [roles([admin, shell])]),
        endpoint(grep, post, '/grep', [roles([user, admin, shell])]),
        endpoint(find, post, '/find', [roles([user, admin, shell])]),
        endpoint(cat, post, '/cat', [roles([user, admin, shell])]),
        endpoint(browse, post, '/browse', [roles([user, admin, shell])])
    ]),

    % WebSocket
    websocket([
        path('/shell'),
        roles([shell]),
        handler(shell_pty)
    ]),

    % HTML interface
    serve_html(true)
]).
```

### 2. Auth Source Type

Define authentication as a separate composable source:

```prolog
:- source(auth, jwt_auth, [
    backend(text_file),           % or: sqlite, postgres, memory
    users_file('users.txt'),
    password_hash(bcrypt),        % or: sha256, argon2
    token_type(jwt),
    token_expiry(86400),          % 24 hours

    % Roles and permissions
    roles([
        role(shell, [description('Full shell access'), permissions([all])]),
        role(admin, [description('Admin access'), permissions([read, write, exec])]),
        role(user, [description('Basic access'), permissions([read])])
    ]),

    % Default users (created on first run)
    default_users([
        user('shell@local', 'shell', [shell, admin, user]),
        user('admin@local', 'admin', [admin, user]),
        user('user@local', 'user', [user])
    ])
]).
```

### 3. Shell Component

Define shell as a component with sandbox configuration:

```prolog
:- component(shell, websocket_shell, [
    sandbox([
        backend(proot),           % or: app_filter, none
        root_dir(env('SANDBOX_ROOT')),
        allowed_commands([ls, cat, grep, find, head, tail, wc, pwd]),
        blocked_patterns([
            'rm -rf',
            'sudo',
            '> /dev/'
        ]),
        timeout(300),
        max_output(50000)
    ]),

    % UI options
    ui([
        text_mode(true),          % Enable text mode toggle
        capture_mode(true),       % Mobile keyboard capture
        theme(catppuccin)
    ]),

    % Access levels by role
    access_levels([
        shell -> level(superadmin, [pty(true), sandbox(none)]),
        admin -> level(full, [pty(true), sandbox(app_filter)]),
        user -> level(trusted, [pty(false), sandbox(proot)])
    ])
]).
```

### 4. Binding: Connect Auth to Service

```prolog
:- binding(http_cli_auth, jwt_auth, http_cli_server, [
    protect_endpoints(true),
    public_endpoints([health]),
    token_header('Authorization'),
    token_prefix('Bearer ')
]).

:- binding(shell_auth, jwt_auth, websocket_shell, [
    require_role(shell),
    token_query_param('token')
]).
```

### 5. Complete App Spec

```prolog
:- app(http_cli, [
    sources([http_cli_server, jwt_auth]),
    components([websocket_shell]),
    bindings([http_cli_auth, shell_auth]),

    target(typescript),
    output_dir('src/unifyweaver/shell'),

    files([
        file('http-server.ts', http_server_template),
        file('auth.ts', auth_template),
        file('shell-handler.ts', shell_template)
    ])
]).
```

## Code Generation

### New Generator Module

Create `src/unifyweaver/glue/http_server_generator.pl`:

```prolog
:- module(http_server_generator, [
    generate_http_server/3,       % +ServiceSpec, +Target, -Code
    generate_auth_module/3,       % +AuthSpec, +Target, -Code
    generate_shell_handler/3,     % +ShellSpec, +Target, -Code
    generate_html_interface/3     % +AppSpec, +Target, -Code
]).

%! generate_http_server(+ServiceSpec, +Target, -Code)
%
%  Generate HTTP/HTTPS server code for the given target.
%
%  Targets: typescript, python (fastapi), go
%
generate_http_server(ServiceSpec, typescript, Code) :-
    % Extract config from spec
    service_port(ServiceSpec, Port),
    service_endpoints(ServiceSpec, Endpoints),
    service_https(ServiceSpec, HTTPSConfig),

    % Generate imports
    generate_ts_imports(Endpoints, Imports),

    % Generate endpoint handlers
    maplist(generate_ts_endpoint, Endpoints, HandlerCodes),

    % Generate server setup with HTTPS
    generate_ts_server_setup(Port, HTTPSConfig, ServerCode),

    % Combine
    atomic_list_concat([Imports, HandlerCodes, ServerCode], '\n\n', Code).
```

### Template System Integration

Use the existing `template_library.pl` for code templates:

```prolog
% Register HTTP server templates
:- register_template(http_server_ts, '
import * as http from "http";
import * as https from "https";
import * as fs from "fs";

{{#if https}}
const sslOptions = {
  cert: fs.readFileSync(process.env.{{https.cert_env}} || "cert.pem"),
  key: fs.readFileSync(process.env.{{https.key_env}} || "key.pem")
};
const server = https.createServer(sslOptions, handleRequest);
{{else}}
const server = http.createServer(handleRequest);
{{/if}}

{{#each endpoints}}
// {{method}} {{path}}
async function handle_{{name}}(req, res) {
  {{#if roles}}
  requireRoles(req, [{{roles}}]);
  {{/if}}
  // ... endpoint logic
}
{{/each}}
').
```

## Migration Path

### Phase 1: Generate alongside manual code
- Create generators that produce equivalent code
- Compare output with manual implementation
- Fix any discrepancies

### Phase 2: Replace manual code
- Switch to generated code
- Keep manual code as reference/backup
- Update docs

### Phase 3: Extend with new features
- Add more backends (FastAPI, Go)
- Add more auth providers (OAuth, OIDC)
- Add more shell sandboxing options

## Benefits

1. **Declarative** - Server config is data, not code
2. **Consistent** - Same patterns across all generated servers
3. **Maintainable** - Change spec, regenerate code
4. **Multi-target** - Generate TypeScript, Python, Go from same spec
5. **Testable** - Validate specs before generation
6. **Composable** - Mix and match auth, shell, endpoints

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/unifyweaver/glue/http_server_generator.pl` | Create | Main server generator |
| `src/unifyweaver/glue/auth_generator.pl` | Create | Auth module generator |
| `src/unifyweaver/glue/shell_generator.pl` | Create | Shell handler generator |
| `src/unifyweaver/glue/app_generator.pl` | Modify | Add service/component support |
| `src/unifyweaver/sources/service_source.pl` | Create | Service source type |
| `skills/skill_http_server.md` | Create | Skill documentation |

## Example Usage

```prolog
% Load the generators
:- use_module('src/unifyweaver/glue/http_server_generator').
:- use_module('src/unifyweaver/glue/auth_generator').

% Define the spec (or load from file)
:- consult('specs/http_cli_spec.pl').

% Generate
?- generate_complete_http_server(http_cli, typescript, 'src/unifyweaver/shell', Files).
Files = [
    file('http-server.ts', '...'),
    file('auth.ts', '...'),
    file('shell-handler.ts', '...')
].
```

## Timeline Estimate

- Phase 1 (Generators): Create basic generators
- Phase 2 (Templates): Integrate with template system
- Phase 3 (Migration): Replace manual code
- Phase 4 (Docs): Update skills and documentation

## Related

- `skill_data_sources.md` - Source architecture
- `skill_data_binding.md` - Binding patterns
- `skill_component_library.md` - Component system
- `skill_authentication.md` - Auth patterns
- `context/vue_guard_app/` - Previous app recovery (reference implementation)
