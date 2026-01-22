:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% service_source.pl - Service source plugin for HTTP/HTTPS servers
%
% Defines declarative specifications for HTTP servers with endpoints,
% authentication, WebSocket support, and HTTPS configuration.
%
% Usage:
%   :- source(service, my_server, [
%       port(3001),
%       https([cert_env('SSL_CERT'), key_env('SSL_KEY')]),
%       endpoints([...]),
%       websocket([path('/ws'), roles([admin])])
%   ]).

:- module(service_source, [
    % Plugin interface
    source_info/1,
    validate_config/1,

    % Service spec accessors
    service_name/2,
    service_port/2,
    service_https/2,
    service_endpoints/2,
    service_websocket/2,
    service_cors/2,
    service_serve_html/2,

    % Endpoint accessors
    endpoint_name/2,
    endpoint_method/2,
    endpoint_path/2,
    endpoint_roles/2,
    endpoint_public/2,

    % HTTPS accessors
    https_enabled/1,
    https_cert_env/2,
    https_key_env/2,
    https_cert_path/2,
    https_key_path/2,

    % WebSocket accessors
    websocket_path/2,
    websocket_roles/2,
    websocket_handler/2
]).

:- use_module(library(lists)).

%% ============================================================================
%% PLUGIN INTERFACE
%% ============================================================================

%! source_info(-Info) is det
%
%  Provide information about this source plugin.
%
source_info(info(
    name('Service Source'),
    version('1.0.0'),
    description('Define HTTP/HTTPS servers with endpoints, auth, and WebSocket'),
    supported_arities([1])
)).

%! validate_config(+Config) is semidet
%
%  Validate configuration for service source.
%
validate_config(Config) :-
    % Port is optional but must be valid if present
    (   member(port(Port), Config)
    ->  (   integer(Port), Port > 0, Port < 65536
        ->  true
        ;   format('Error: port must be integer 1-65535, got ~w~n', [Port]),
            fail
        )
    ;   true
    ),

    % Validate endpoints if present
    (   member(endpoints(Endpoints), Config)
    ->  validate_endpoints(Endpoints)
    ;   true
    ),

    % Validate HTTPS config if present
    (   member(https(HTTPSConfig), Config)
    ->  validate_https_config(HTTPSConfig)
    ;   true
    ),

    % Validate WebSocket config if present
    (   member(websocket(WSConfig), Config)
    ->  validate_websocket_config(WSConfig)
    ;   true
    ).

%! validate_endpoints(+Endpoints) is semidet
%
%  Validate list of endpoint definitions.
%
validate_endpoints([]).
validate_endpoints([E|Es]) :-
    validate_endpoint(E),
    validate_endpoints(Es).

%! validate_endpoint(+Endpoint) is semidet
%
%  Validate a single endpoint definition.
%
validate_endpoint(endpoint(Name, Method, Path, _Options)) :-
    atom(Name),
    member(Method, [get, post, put, patch, delete, head, options]),
    atom(Path),
    !.
validate_endpoint(E) :-
    format('Error: invalid endpoint format ~w~n', [E]),
    format('Expected: endpoint(name, method, path, options)~n', []),
    fail.

%! validate_https_config(+Config) is semidet
%
%  Validate HTTPS configuration.
%
validate_https_config(Config) :-
    (   is_list(Config)
    ->  true
    ;   Config == true
    ->  true
    ;   Config == false
    ->  true
    ;   format('Error: https config must be list or boolean~n', []),
        fail
    ).

%! validate_websocket_config(+Config) is semidet
%
%  Validate WebSocket configuration.
%
validate_websocket_config(Config) :-
    (   is_list(Config)
    ->  (   member(path(Path), Config)
        ->  atom(Path)
        ;   format('Error: websocket config requires path~n', []),
            fail
        )
    ;   format('Error: websocket config must be list~n', []),
        fail
    ).

%% ============================================================================
%% SERVICE SPEC ACCESSORS
%% ============================================================================

%! service_name(+Spec, -Name) is det
%
%  Extract service name from spec.
%
service_name(service(Name, _Config), Name).

%! service_port(+Spec, -Port) is det
%
%  Extract port from service spec, default 3001.
%
service_port(service(_Name, Config), Port) :-
    (   member(port(Port), Config)
    ->  true
    ;   Port = 3001
    ).

%! service_https(+Spec, -HTTPSConfig) is det
%
%  Extract HTTPS configuration from service spec.
%
service_https(service(_Name, Config), HTTPSConfig) :-
    (   member(https(HTTPSConfig), Config)
    ->  true
    ;   HTTPSConfig = false
    ).

%! service_endpoints(+Spec, -Endpoints) is det
%
%  Extract endpoints from service spec.
%
service_endpoints(service(_Name, Config), Endpoints) :-
    (   member(endpoints(Endpoints), Config)
    ->  true
    ;   Endpoints = []
    ).

%! service_websocket(+Spec, -WSConfig) is det
%
%  Extract WebSocket configuration from service spec.
%
service_websocket(service(_Name, Config), WSConfig) :-
    (   member(websocket(WSConfig), Config)
    ->  true
    ;   WSConfig = none
    ).

%! service_cors(+Spec, -CORSConfig) is det
%
%  Extract CORS configuration from service spec.
%
service_cors(service(_Name, Config), CORSConfig) :-
    (   member(cors(CORSConfig), Config)
    ->  true
    ;   CORSConfig = [origins(['*'])]
    ).

%! service_serve_html(+Spec, -ServeHTML) is det
%
%  Check if service should serve HTML interface.
%
service_serve_html(service(_Name, Config), ServeHTML) :-
    (   member(serve_html(ServeHTML), Config)
    ->  true
    ;   ServeHTML = false
    ).

%% ============================================================================
%% ENDPOINT ACCESSORS
%% ============================================================================

%! endpoint_name(+Endpoint, -Name) is det
endpoint_name(endpoint(Name, _Method, _Path, _Options), Name).

%! endpoint_method(+Endpoint, -Method) is det
endpoint_method(endpoint(_Name, Method, _Path, _Options), Method).

%! endpoint_path(+Endpoint, -Path) is det
endpoint_path(endpoint(_Name, _Method, Path, _Options), Path).

%! endpoint_roles(+Endpoint, -Roles) is det
%
%  Extract required roles from endpoint options.
%
endpoint_roles(endpoint(_Name, _Method, _Path, Options), Roles) :-
    (   member(roles(Roles), Options)
    ->  true
    ;   Roles = []
    ).

%! endpoint_public(+Endpoint, -IsPublic) is det
%
%  Check if endpoint is public (no auth required).
%
endpoint_public(endpoint(_Name, _Method, _Path, Options), IsPublic) :-
    (   member(public(IsPublic), Options)
    ->  true
    ;   IsPublic = false
    ).

%% ============================================================================
%% HTTPS ACCESSORS
%% ============================================================================

%! https_enabled(+HTTPSConfig) is semidet
%
%  Check if HTTPS is enabled.
%
https_enabled(true).
https_enabled(Config) :-
    is_list(Config),
    Config \= [].

%! https_cert_env(+HTTPSConfig, -EnvVar) is det
%
%  Get environment variable name for SSL certificate.
%
https_cert_env(Config, EnvVar) :-
    is_list(Config),
    member(cert_env(EnvVar), Config).

%! https_key_env(+HTTPSConfig, -EnvVar) is det
%
%  Get environment variable name for SSL key.
%
https_key_env(Config, EnvVar) :-
    is_list(Config),
    member(key_env(EnvVar), Config).

%! https_cert_path(+HTTPSConfig, -Path) is det
%
%  Get path to SSL certificate file.
%
https_cert_path(Config, Path) :-
    is_list(Config),
    member(cert(Path), Config).

%! https_key_path(+HTTPSConfig, -Path) is det
%
%  Get path to SSL key file.
%
https_key_path(Config, Path) :-
    is_list(Config),
    member(key(Path), Config).

%% ============================================================================
%% WEBSOCKET ACCESSORS
%% ============================================================================

%! websocket_path(+WSConfig, -Path) is det
%
%  Get WebSocket endpoint path.
%
websocket_path(Config, Path) :-
    is_list(Config),
    member(path(Path), Config).

%! websocket_roles(+WSConfig, -Roles) is det
%
%  Get required roles for WebSocket access.
%
websocket_roles(Config, Roles) :-
    is_list(Config),
    (   member(roles(Roles), Config)
    ->  true
    ;   Roles = []
    ).

%! websocket_handler(+WSConfig, -Handler) is det
%
%  Get WebSocket handler type.
%
websocket_handler(Config, Handler) :-
    is_list(Config),
    (   member(handler(Handler), Config)
    ->  true
    ;   Handler = default
    ).

%% ============================================================================
%% SERVICE SPEC BUILDER
%% ============================================================================

%! build_service_spec(+Name, +Options, -Spec) is det
%
%  Build a service specification from options.
%
%  Example:
%    build_service_spec(my_server, [
%        port(3001),
%        https([cert_env('SSL_CERT'), key_env('SSL_KEY')]),
%        endpoints([
%            endpoint(health, get, '/health', [public(true)]),
%            endpoint(exec, post, '/exec', [roles([admin])])
%        ])
%    ], Spec).
%
build_service_spec(Name, Options, service(Name, Options)) :-
    validate_config(Options).

%% ============================================================================
%% PREDEFINED SERVICE TEMPLATES
%% ============================================================================

%! service_template(+TemplateName, -Spec) is det
%
%  Get predefined service templates.
%
service_template(http_cli, service(http_cli, [
    port(3001),
    https([cert_env('SSL_CERT'), key_env('SSL_KEY')]),
    endpoints([
        endpoint(health, get, '/health', [public(true)]),
        endpoint(commands, get, '/commands', [roles([user, admin, shell])]),
        endpoint(exec, post, '/exec', [roles([admin, shell])]),
        endpoint(grep, post, '/grep', [roles([user, admin, shell])]),
        endpoint(find, post, '/find', [roles([user, admin, shell])]),
        endpoint(cat, post, '/cat', [roles([user, admin, shell])]),
        endpoint(browse, post, '/browse', [roles([user, admin, shell])])
    ]),
    websocket([
        path('/shell'),
        roles([shell]),
        handler(shell_pty)
    ]),
    serve_html(true)
])).

service_template(api_server, service(api_server, [
    port(8080),
    endpoints([
        endpoint(health, get, '/health', [public(true)]),
        endpoint(api, post, '/api', [roles([user])])
    ]),
    cors([origins(['*'])]),
    serve_html(false)
])).
