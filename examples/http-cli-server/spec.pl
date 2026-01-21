% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% spec.pl - HTTP CLI Server Specification
%
% Declarative specification for generating the HTTP CLI server
% using UnifyWeaver's service source and generators.
%
% Usage:
%   swipl -g "consult('spec.pl'), generate_all, halt" -t halt
%
% Or from project root:
%   ./examples/http-cli-server/generate.sh

:- use_module('../../src/unifyweaver/sources/service_source').
:- use_module('../../src/unifyweaver/glue/http_server_generator').
:- use_module('../../src/unifyweaver/glue/auth_generator').

%% ============================================================================
%% SERVICE SPECIFICATION
%% ============================================================================

%! http_cli_service(-Spec) is det
%
%  Define the HTTP CLI server specification.
%
http_cli_service(Spec) :-
    Spec = service(http_cli_server, [
        name('UnifyWeaver HTTP CLI Server'),
        port(3001),

        % HTTPS configuration
        https([
            cert_env('SSL_CERT'),
            key_env('SSL_KEY')
        ]),

        % Endpoints
        endpoints([
            endpoint(health, get, '/health', [public(true)]),
            endpoint(auth_status, get, '/auth/status', [public(true)]),
            endpoint(auth_login, post, '/auth/login', [public(true)]),
            endpoint(auth_me, get, '/auth/me', [roles([user, admin, shell])]),
            endpoint(commands, get, '/commands', [roles([user, admin, shell])]),
            endpoint(browse, post, '/browse', [roles([user, admin, shell])]),
            endpoint(grep, post, '/grep', [roles([user, admin, shell])]),
            endpoint(find, post, '/find', [roles([user, admin, shell])]),
            endpoint(cat, post, '/cat', [roles([user, admin, shell])]),
            endpoint(exec, post, '/exec', [roles([admin, shell])]),
            endpoint(feedback, post, '/feedback', [roles([user, admin, shell])])
        ]),

        % WebSocket shell
        websocket([
            path('/shell'),
            roles([shell]),
            handler(shell_pty)
        ]),

        % CORS
        cors([
            origins(['*'])
        ]),

        % Serve HTML interface
        serve_html(true)
    ]).

%% ============================================================================
%% AUTH SPECIFICATION
%% ============================================================================

%! http_cli_auth(-Spec) is det
%
%  Define the authentication specification.
%
http_cli_auth(Spec) :-
    Spec = auth(http_cli_auth, [
        backend(text_file),
        users_file('users.txt'),
        password_hash(bcrypt),
        token_type(jwt),
        token_expiry(86400),  % 24 hours

        % Role definitions
        roles([
            role(shell, [
                description('Full shell access including WebSocket PTY'),
                permissions([all])
            ]),
            role(admin, [
                description('Admin access - can execute commands'),
                permissions([read, write, exec])
            ]),
            role(user, [
                description('Basic user - browse and search only'),
                permissions([read])
            ])
        ]),

        % Default users created on first run
        default_users([
            user('shell@local', 'shell', [shell, admin, user]),
            user('admin@local', 'admin', [admin, user]),
            user('user@local', 'user', [user])
        ])
    ]).

%% ============================================================================
%% GENERATION
%% ============================================================================

%! generate_all is det
%
%  Generate all files for the HTTP CLI server.
%
generate_all :-
    http_cli_service(ServiceSpec),
    http_cli_auth(AuthSpec),

    % Generate server code
    generate_http_server(ServiceSpec, typescript, ServerCode),

    % Generate auth module
    generate_auth_module(AuthSpec, typescript, AuthCode),

    % Generate auth config JSON
    generate_auth_config(AuthSpec, AuthConfigJSON),

    % Write files
    write_file('generated/server.ts', ServerCode),
    write_file('generated/auth.ts', AuthCode),
    write_file('generated/auth-config.json', AuthConfigJSON),

    format('Generated files:~n'),
    format('  - generated/server.ts~n'),
    format('  - generated/auth.ts~n'),
    format('  - generated/auth-config.json~n').

%! write_file(+Path, +Content) is det
%
%  Write content to file.
%
write_file(Path, Content) :-
    open(Path, write, Stream),
    write(Stream, Content),
    close(Stream).

%% ============================================================================
%% CUSTOMIZATION HOOKS
%% ============================================================================

%! customize_endpoint(+Name, -CustomCode) is nondet
%
%  Hook for customizing endpoint implementations.
%  Override in your own spec file to add custom logic.
%
customize_endpoint(browse, Code) :-
    Code = '  // Custom browse implementation
  const dirPath = body.path || SANDBOX_ROOT;
  // ... directory listing logic'.

customize_endpoint(grep, Code) :-
    Code = '  // Custom grep implementation
  const pattern = body.pattern;
  const searchPath = body.path || SANDBOX_ROOT;
  // ... grep logic'.
